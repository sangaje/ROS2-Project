#!/usr/bin/env python3

from __future__ import annotations

from copy import deepcopy
from functools import partial
from typing import Optional

import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.exceptions import ParameterAlreadyDeclaredException
from geometry_msgs.msg import PoseStamped
from nav2_msgs.action import NavigateToPose


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

        self.client: ActionClient = ActionClient(
            self, NavigateToPose, self.navigate_action
        )
        self.sub = self.create_subscription(
            PoseStamped,
            self.goal_pose_topic,
            self._on_goal_pose,
            10,
        )
        self.current_goal_handle = None
        self.current_goal_id = 0
        self.pending_goal: Optional[PoseStamped] = None
        self.goal_count = 0
        self.last_wait_log_time = -1.0e9
        self.retry_not_before = -1.0e9
        self.retry_timer = self.create_timer(0.25, self._try_send_pending)

        self.get_logger().info(
            'POSE_TO_NAV2_ACTION_READY | '
            f'in={self.goal_pose_topic} action={self.navigate_action} '
            f'frame={self.default_frame_id} '
            f'cancel_previous={self.cancel_previous_goal}'
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
        self.retry_not_before = -1.0e9
        self._try_send_pending()

    def _try_send_pending(self) -> None:
        if self.pending_goal is None:
            return
        now = self.get_clock().now().nanoseconds * 1.0e-9
        if now < self.retry_not_before:
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
            try:
                self.current_goal_handle.cancel_goal_async()
            except Exception as exc:  # noqa: BLE001
                self.get_logger().warn(
                    f'CANCEL_PREVIOUS_GOAL_FAILED | {exc}'
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
        future = self.client.send_goal_async(
            goal,
            feedback_callback=feedback_callback,
        )
        future.add_done_callback(
            partial(
                self._goal_response_cb,
                goal_id=request_id,
                sent_goal=deepcopy(goal.pose),
            )
        )

    def _goal_response_cb(
        self,
        future,
        goal_id: int,
        sent_goal: PoseStamped,
    ) -> None:
        try:
            goal_handle = future.result()
        except Exception as exc:  # noqa: BLE001
            self.get_logger().error(f'NAV2_GOAL_SEND_EXCEPTION | {exc}')
            if goal_id == self.goal_count and self.pending_goal is None:
                self.pending_goal = sent_goal
                self.retry_not_before = (
                    self.get_clock().now().nanoseconds * 1.0e-9 + 1.0
                )
            return
        if not goal_handle.accepted:
            self.get_logger().warn('NAV2_GOAL_REJECTED')
            if goal_id == self.goal_count and self.pending_goal is None:
                self.pending_goal = sent_goal
                self.retry_not_before = (
                    self.get_clock().now().nanoseconds * 1.0e-9 + 1.0
                )
            return

        # Action responses may arrive out of order. Never let an older response
        # replace the handle for a newer user or safety goal.
        if goal_id != self.goal_count:
            goal_handle.cancel_goal_async()
            self.get_logger().info(
                f'STALE_GOAL_CANCELLED | n={goal_id} '
                f'latest={self.goal_count}'
            )
            return

        self.current_goal_handle = goal_handle
        self.current_goal_id = goal_id
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(
            partial(self._result_cb, goal_id=goal_id)
        )
        self.get_logger().info('NAV2_GOAL_ACCEPTED')

    def _result_cb(self, future, goal_id: int) -> None:
        try:
            result = future.result()
        except Exception as exc:  # noqa: BLE001
            self.get_logger().error(f'NAV2_GOAL_RESULT_EXCEPTION | {exc}')
            return
        if goal_id == self.current_goal_id:
            self.current_goal_handle = None
            self.current_goal_id = 0
        self.get_logger().info(f'NAV2_GOAL_RESULT | status={result.status}')

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
