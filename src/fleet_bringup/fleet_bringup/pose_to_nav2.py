#!/usr/bin/env python3

from __future__ import annotations

from copy import deepcopy
from functools import partial
from typing import Optional

import rclpy
from action_msgs.msg import GoalStatus
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.exceptions import ParameterAlreadyDeclaredException
from geometry_msgs.msg import PoseStamped
from lifecycle_msgs.msg import State
from lifecycle_msgs.srv import GetState
from nav2_msgs.action import NavigateToPose
from nav2_msgs.srv import ManageLifecycleNodes
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import Bool


def _safe_declare(node: Node, name: str, default):
    try:
        node.declare_parameter(name, default)
    except ParameterAlreadyDeclaredException:
        pass
    return node.get_parameter(name).value


class PoseToNav2Action(Node):
    """Convert RViz PoseStamped goal topics to Nav2 NavigateToPose actions.

    RViz's default GoalTool publishes geometry_msgs/PoseStamped on a configurable topic.
    Nav2 ultimately consumes NavigateToPose actions. This bridge lets us debug each
    domain with plain RViz goal topics and also send Burger goals across domains by
    bridging only a PoseStamped topic, not a full action.
    """

    def __init__(self) -> None:
        super().__init__('pose_to_nav2_action')
        _safe_declare(self, 'use_sim_time', True)
        self.goal_pose_topic = self._abs(str(
            _safe_declare(self, 'goal_pose_topic', '/goal_pose')
        ))
        self.navigate_action = self._strip_action(str(
            _safe_declare(self, 'navigate_action', '/navigate_to_pose')
        ))
        self.default_frame_id = str(
            _safe_declare(self, 'default_frame_id', 'map')
        )
        self.stamp_with_now = bool(
            _safe_declare(self, 'stamp_with_now', True)
        )
        self.cancel_previous_goal = bool(
            _safe_declare(self, 'cancel_previous_goal', True)
        )
        self.log_feedback = bool(
            _safe_declare(self, 'log_feedback', False)
        )
        self.wait_for_lifecycle_active = bool(
            _safe_declare(self, 'wait_for_lifecycle_active', True)
        )
        self.bt_state_service_name = self._abs(str(
            _safe_declare(
                self,
                'bt_navigator_state_service',
                '/bt_navigator/get_state',
            )
        ))
        self.navigation_lifecycle_service_name = self._abs(str(
            _safe_declare(
                self,
                'navigation_lifecycle_service',
                '/lifecycle_manager_navigation/manage_nodes',
            )
        ))
        self.auto_start_navigation_lifecycle = bool(
            _safe_declare(self, 'auto_start_navigation_lifecycle', True)
        )
        self.lifecycle_retry_sec = max(
            0.5, float(_safe_declare(self, 'lifecycle_retry_sec', 2.0))
        )
        self.retry_failed_results = bool(
            _safe_declare(self, 'retry_failed_results', True)
        )
        self.max_result_retries = max(
            0, int(_safe_declare(self, 'max_result_retries', 2))
        )
        self.result_retry_sec = max(
            0.5, float(_safe_declare(self, 'result_retry_sec', 2.0))
        )
        self.max_send_retries = max(
            0, int(_safe_declare(self, 'max_send_retries', 2))
        )
        self.send_exception_retry_sec = max(
            0.1,
            float(_safe_declare(self, 'send_exception_retry_sec', 1.0)),
        )
        self.rejected_retry_sec = max(
            0.1, float(_safe_declare(self, 'rejected_retry_sec', 2.0))
        )
        self.max_retry_backoff_sec = max(
            0.1, float(_safe_declare(self, 'max_retry_backoff_sec', 15.0))
        )
        self.require_localization_ready = bool(
            _safe_declare(self, 'require_localization_ready', False)
        )
        self.localization_ready_topic = self._abs(str(
            _safe_declare(self, 'localization_ready_topic', '/localization_ready')
        ))
        cancel_topic = str(_safe_declare(self, 'cancel_topic', '')).strip()
        self.cancel_topic = self._abs(cancel_topic) if cancel_topic else ''

        self.client: ActionClient = ActionClient(
            self, NavigateToPose, self.navigate_action
        )
        self.state_client = self.create_client(
            GetState, self.bt_state_service_name
        )
        self.lifecycle_client = self.create_client(
            ManageLifecycleNodes, self.navigation_lifecycle_service_name
        )
        self.sub = self.create_subscription(
            PoseStamped,
            self.goal_pose_topic,
            self._on_goal_pose,
            10,
        )
        self.localization_ready = False
        if self.require_localization_ready:
            latched_qos = QoSProfile(
                depth=1,
                reliability=ReliabilityPolicy.RELIABLE,
                durability=DurabilityPolicy.TRANSIENT_LOCAL,
                history=HistoryPolicy.KEEP_LAST,
            )
            self.create_subscription(
                Bool,
                self.localization_ready_topic,
                self._on_localization_ready,
                latched_qos,
            )
        if self.cancel_topic:
            cancel_qos = QoSProfile(
                depth=1,
                reliability=ReliabilityPolicy.RELIABLE,
                durability=DurabilityPolicy.TRANSIENT_LOCAL,
                history=HistoryPolicy.KEEP_LAST,
            )
            self.create_subscription(
                Bool,
                self.cancel_topic,
                self._on_cancel,
                cancel_qos,
            )
        self.current_goal_handle = None
        self.current_goal_id = 0
        self.pending_goal: Optional[PoseStamped] = None
        self.inflight_goal_ids = set()
        self.goal_count = 0
        self.last_wait_log_time = -1.0e9
        self.retry_not_before = -1.0e9
        self.navigation_active: Optional[bool] = None
        self.lifecycle_state_known = False
        self.state_future = None
        self.startup_future = None
        self.last_lifecycle_start_time = -1.0e9
        self.failed_result_retries = 0
        self.retry_attempts = {
            'send_exception': 0,
            'rejected': 0,
            'result': 0,
        }
        self.retry_timer = self.create_timer(0.25, self._try_send_pending)

        self.get_logger().info(
            'POSE_TO_NAV2_ACTION_READY | '
            f'in={self.goal_pose_topic} action={self.navigate_action} '
            f'frame={self.default_frame_id} '
            f'cancel_previous={self.cancel_previous_goal} '
            f'wait_lifecycle={self.wait_for_lifecycle_active} '
            f'require_localization_ready={self.require_localization_ready} '
            f'localization_ready_topic={self.localization_ready_topic} '
            f'cancel_topic={self.cancel_topic or "disabled"}'
        )

    @staticmethod
    def _abs(topic: str) -> str:
        topic = topic.strip()
        return topic if topic.startswith('/') else '/' + topic

    @staticmethod
    def _strip_action(action_name: str) -> str:
        action_name = action_name.strip()
        return action_name if action_name.startswith('/') else '/' + action_name

    def _on_goal_pose(self, msg: PoseStamped) -> None:
        # Keep only the newest command while Nav2 is starting. Dropping a goal
        # here makes an RViz goal appear to vanish with no way to recover it.
        self.pending_goal = deepcopy(msg)
        self.failed_result_retries = 0
        self.retry_attempts = {
            'send_exception': 0,
            'rejected': 0,
            'result': 0,
        }
        self.retry_not_before = -1.0e9
        self._try_send_pending()

    def _on_localization_ready(self, msg: Bool) -> None:
        previous = self.localization_ready
        self.localization_ready = bool(msg.data)
        if self.localization_ready and not previous:
            self.get_logger().warn(
                f'LOCALIZATION_READY_FOR_NAV2_GOALS | topic={self.localization_ready_topic}'
            )
            self.retry_not_before = -1.0e9
            self._try_send_pending()
        elif previous and not self.localization_ready:
            # A latched false commonly means the localization bootstrap
            # restarted.  Invalidate every action request already handed to
            # Nav2 and cancel the accepted one, while retaining at most the
            # one goal that has not yet been sent.
            self._invalidate_inflight(
                cause='localization_not_ready',
                clear_pending=False,
            )
            self.get_logger().warn(
                'LOCALIZATION_NOT_READY_NAV2_CANCELLED | '
                f'topic={self.localization_ready_topic}'
            )

    def _on_cancel(self, msg: Bool) -> None:
        if not msg.data:
            return
        had_work = self._invalidate_inflight(
            cause='explicit_cancel',
            clear_pending=True,
        )
        if not had_work:
            self.get_logger().info(
                f'NAV2_GOAL_CANCEL_REQUEST_EMPTY | topic={self.cancel_topic}'
            )
            return
        self.get_logger().warn(
            f'NAV2_GOAL_CANCEL_REQUESTED | topic={self.cancel_topic}'
        )

    def _invalidate_inflight(self, *, cause: str, clear_pending: bool) -> bool:
        """Make every already-sent callback stale and cancel an accepted goal.

        ``goal_count`` is both a monotonic send id and the latest valid
        generation.  Advancing it here means a goal accepted after the cancel
        race is immediately canceled by ``_goal_response_cb`` instead of being
        installed as the current handle.
        """
        had_work = bool(
            self.pending_goal is not None
            or self.current_goal_handle is not None
            or self.inflight_goal_ids
        )
        self.goal_count += 1
        self.retry_not_before = -1.0e9
        if clear_pending:
            self.pending_goal = None
        handle = self.current_goal_handle
        handle_id = self.current_goal_id
        self.current_goal_handle = None
        self.current_goal_id = 0
        if handle is not None:
            self._request_cancel(handle, goal_id=handle_id, cause=cause)
        return had_work

    def _request_cancel(self, goal_handle, *, goal_id: int, cause: str) -> None:
        try:
            future = goal_handle.cancel_goal_async()
        except Exception as exc:  # noqa: BLE001
            self.get_logger().warn(
                f'NAV2_GOAL_CANCEL_FAILED | cause={cause} goal_id={goal_id} {exc}'
            )
            return
        if future is not None:
            future.add_done_callback(
                partial(self._cancel_response_cb, goal_id=goal_id, cause=cause)
            )

    def _cancel_response_cb(self, future, *, goal_id: int, cause: str) -> None:
        try:
            response = future.result()
        except Exception as exc:  # noqa: BLE001
            self.get_logger().warn(
                f'NAV2_GOAL_CANCEL_ACK_FAILED | cause={cause} '
                f'goal_id={goal_id} {exc}'
            )
            return
        return_code = getattr(response, 'return_code', None)
        goals_canceling = getattr(response, 'goals_canceling', None)
        count = len(goals_canceling) if goals_canceling is not None else None
        self.get_logger().info(
            'NAV2_GOAL_CANCEL_ACK | '
            f'cause={cause} goal_id={goal_id} '
            f'return_code={return_code} goals_canceling={count}'
        )

    def _try_send_pending(self) -> None:
        if self.pending_goal is None:
            return
        now = self.get_clock().now().nanoseconds * 1.0e-9
        if now < self.retry_not_before:
            return
        if self._nav2_lifecycle_blocks_send(now):
            return
        if self._localization_blocks_send(now):
            return
        if not self.client.server_is_ready():
            if now - self.last_wait_log_time >= 5.0:
                self.get_logger().warn(
                    'NAV2_ACTION_SERVER_NOT_READY | '
                    f'action={self.navigate_action} '
                    f'goal_topic={self.goal_pose_topic} | latest goal retained'
                )
                self.last_wait_log_time = now
            return

        msg = self.pending_goal
        self.pending_goal = None
        if self.cancel_previous_goal and self.current_goal_handle is not None:
            previous_handle = self.current_goal_handle
            previous_id = self.current_goal_id
            self.current_goal_handle = None
            self.current_goal_id = 0
            self._request_cancel(
                previous_handle,
                goal_id=previous_id,
                cause='newer_goal',
            )

        goal = NavigateToPose.Goal()
        goal.pose = msg
        if not goal.pose.header.frame_id:
            goal.pose.header.frame_id = self.default_frame_id
        if self.stamp_with_now:
            goal.pose.header.stamp = self.get_clock().now().to_msg()

        self.goal_count += 1
        request_id = self.goal_count
        x = goal.pose.pose.position.x
        y = goal.pose.pose.position.y
        self.get_logger().info(
            f'SEND_NAV2_GOAL | n={self.goal_count} '
            f'topic={self.goal_pose_topic} action={self.navigate_action} '
            f'frame={goal.pose.header.frame_id} xy=({x:.3f},{y:.3f})'
        )

        feedback_callback = self._feedback_cb if self.log_feedback else None
        try:
            future = self.client.send_goal_async(
                goal,
                feedback_callback=feedback_callback,
            )
        except Exception as exc:  # noqa: BLE001
            self.get_logger().error(f'NAV2_GOAL_SEND_EXCEPTION | {exc}')
            self._schedule_retry(
                sent_goal=goal.pose,
                goal_id=request_id,
                cause='send_exception',
                base_delay=self.send_exception_retry_sec,
                limit=self.max_send_retries,
            )
            return
        self.inflight_goal_ids.add(request_id)
        future.add_done_callback(
            partial(
                self._goal_response_cb,
                goal_id=request_id,
                sent_goal=deepcopy(goal.pose),
            )
        )

    def _localization_blocks_send(self, now: float) -> bool:
        if not self.require_localization_ready or self.localization_ready:
            return False
        if now - self.last_wait_log_time >= 5.0:
            self.get_logger().warn(
                'NAV2_GOAL_HELD_FOR_LOCALIZATION | '
                f'waiting for {self.localization_ready_topic}=true | latest goal retained'
            )
            self.last_wait_log_time = now
        return True

    def _nav2_lifecycle_blocks_send(self, now: float) -> bool:
        if not self.wait_for_lifecycle_active:
            return False

        self._poll_navigation_state()
        if self.navigation_active:
            return False

        # Unit tests and non-Nav2 action bridges may not expose lifecycle
        # services. In that case, keep the old action-server-only behavior.
        state_ready = self.state_client.service_is_ready()
        lifecycle_ready = self.lifecycle_client.service_is_ready()
        if not state_ready and not lifecycle_ready and not self.lifecycle_state_known:
            return False

        if self.auto_start_navigation_lifecycle:
            self._request_navigation_startup(now)

        if now - self.last_wait_log_time >= 5.0:
            state = 'unknown'
            if self.navigation_active is False:
                state = 'inactive'
            self.get_logger().warn(
                'NAV2_LIFECYCLE_NOT_ACTIVE | '
                f'bt_state={state} '
                f'state_service={self.bt_state_service_name} '
                f'lifecycle_service={self.navigation_lifecycle_service_name} '
                '| latest goal retained'
            )
            self.last_wait_log_time = now
        return True

    def _poll_navigation_state(self) -> None:
        if not self.state_client.service_is_ready():
            return
        if self.state_future is not None and not self.state_future.done():
            return
        future = self.state_client.call_async(GetState.Request())
        self.state_future = future
        future.add_done_callback(self._on_navigation_state)

    def _on_navigation_state(self, future) -> None:
        try:
            response = future.result()
        except Exception as exc:  # noqa: BLE001
            self.get_logger().warn(f'NAV2_STATE_QUERY_FAILED | {exc}')
            return
        state_id = int(response.current_state.id)
        state_label = str(response.current_state.label)
        self.lifecycle_state_known = True
        self.navigation_active = state_id == State.PRIMARY_STATE_ACTIVE
        if self.navigation_active:
            self.get_logger().info(
                f'NAV2_LIFECYCLE_ACTIVE | bt_navigator={state_label}'
            )
            self.retry_not_before = -1.0e9

    def _request_navigation_startup(self, now: float) -> None:
        if not self.lifecycle_client.service_is_ready():
            return
        if self.startup_future is not None and not self.startup_future.done():
            return
        if now - self.last_lifecycle_start_time < self.lifecycle_retry_sec:
            return
        request = ManageLifecycleNodes.Request()
        request.command = ManageLifecycleNodes.Request.STARTUP
        self.startup_future = self.lifecycle_client.call_async(request)
        self.startup_future.add_done_callback(self._on_navigation_startup)
        self.last_lifecycle_start_time = now
        self.get_logger().warn(
            'NAV2_LIFECYCLE_STARTUP_REQUESTED | '
            f'service={self.navigation_lifecycle_service_name}'
        )

    def _on_navigation_startup(self, future) -> None:
        try:
            response = future.result()
        except Exception as exc:  # noqa: BLE001
            self.get_logger().warn(f'NAV2_LIFECYCLE_STARTUP_FAILED | {exc}')
            return
        if getattr(response, 'success', False):
            self.get_logger().info('NAV2_LIFECYCLE_STARTUP_OK')
        else:
            self.get_logger().warn('NAV2_LIFECYCLE_STARTUP_NOT_READY')

    def _goal_response_cb(
        self,
        future,
        goal_id: int,
        sent_goal: PoseStamped,
    ) -> None:
        self.inflight_goal_ids.discard(goal_id)
        try:
            goal_handle = future.result()
        except Exception as exc:  # noqa: BLE001
            self.get_logger().error(f'NAV2_GOAL_SEND_EXCEPTION | {exc}')
            self._schedule_retry(
                sent_goal=sent_goal,
                goal_id=goal_id,
                cause='send_exception',
                base_delay=self.send_exception_retry_sec,
                limit=self.max_send_retries,
            )
            return
        if not goal_handle.accepted:
            self.get_logger().warn('NAV2_GOAL_REJECTED')
            if goal_id == self.goal_count:
                self.navigation_active = False
                if self.auto_start_navigation_lifecycle:
                    self._request_navigation_startup(
                        self.get_clock().now().nanoseconds * 1.0e-9
                    )
            self._schedule_retry(
                sent_goal=sent_goal,
                goal_id=goal_id,
                cause='rejected',
                base_delay=self.rejected_retry_sec,
                limit=self.max_send_retries,
            )
            return

        # Action responses may arrive out of order. Never let an older response
        # replace the handle for a newer user or safety goal.
        if goal_id != self.goal_count:
            self._request_cancel(
                goal_handle,
                goal_id=goal_id,
                cause='stale_goal_response',
            )
            self.get_logger().info(
                f'STALE_GOAL_CANCELLED | n={goal_id} '
                f'latest={self.goal_count}'
            )
            return

        self.current_goal_handle = goal_handle
        self.current_goal_id = goal_id
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(
            partial(
                self._result_cb,
                goal_id=goal_id,
                sent_goal=deepcopy(sent_goal),
            )
        )
        self.get_logger().info('NAV2_GOAL_ACCEPTED')

    def _result_cb(
        self,
        future,
        goal_id: int,
        sent_goal: PoseStamped,
    ) -> None:
        try:
            result = future.result()
        except Exception as exc:  # noqa: BLE001
            self.get_logger().error(f'NAV2_GOAL_RESULT_EXCEPTION | {exc}')
            return
        if goal_id == self.current_goal_id:
            self.current_goal_handle = None
            self.current_goal_id = 0
        self.get_logger().info(f'NAV2_GOAL_RESULT | status={result.status}')
        if (
            goal_id == self.goal_count
            and result.status not in (
                GoalStatus.STATUS_SUCCEEDED,
                GoalStatus.STATUS_CANCELED,
            )
            and self.retry_failed_results
        ):
            scheduled = self._schedule_retry(
                sent_goal=sent_goal,
                goal_id=goal_id,
                cause='result',
                base_delay=self.result_retry_sec,
                limit=self.max_result_retries,
            )
            if scheduled:
                self.navigation_active = None

    def _schedule_retry(
        self,
        *,
        sent_goal: PoseStamped,
        goal_id: int,
        cause: str,
        base_delay: float,
        limit: int,
    ) -> bool:
        """Retain only the latest goal and bound retries per failure cause."""
        if goal_id != self.goal_count or self.pending_goal is not None:
            return False
        attempt = int(self.retry_attempts.get(cause, 0))
        if attempt >= limit:
            self.get_logger().warning(
                'NAV2_GOAL_RETRY_EXHAUSTED | '
                f'cause={cause} retries={attempt}/{limit}'
            )
            return False
        attempt += 1
        self.retry_attempts[cause] = attempt
        if cause == 'result':
            self.failed_result_retries = attempt
        delay = min(
            self.max_retry_backoff_sec,
            max(0.1, float(base_delay)) * (2 ** (attempt - 1)),
        )
        self.pending_goal = deepcopy(sent_goal)
        self.retry_not_before = (
            self.get_clock().now().nanoseconds * 1.0e-9 + delay
        )
        self.get_logger().warning(
            'NAV2_GOAL_RETRY_SCHEDULED | '
            f'cause={cause} retry={attempt}/{limit} in={delay:.1f}s'
        )
        return True

    def _feedback_cb(self, feedback_msg) -> None:
        fb = feedback_msg.feedback
        try:
            remaining = fb.distance_remaining
        except AttributeError:
            remaining = float('nan')
        self.get_logger().info(
            f'NAV2_GOAL_FEEDBACK | distance_remaining={remaining:.3f}'
        )


def main() -> None:
    rclpy.init()
    node = PoseToNav2Action()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.destroy_node()
        except KeyboardInterrupt:
            pass
        if rclpy.ok():
            try:
                rclpy.shutdown()
            except KeyboardInterrupt:
                pass


if __name__ == '__main__':
    main()
