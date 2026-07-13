"""Nav2 action lifecycle owned by a field robot.

This class deliberately owns only action-client bookkeeping.  Role changes,
arrival validation, and retry policy remain orchestration decisions in
``UnifiedFieldRobot`` so their existing timing and semantics stay intact.
"""

from __future__ import annotations

from copy import deepcopy
from functools import partial
from typing import Callable, Optional

from geometry_msgs.msg import PoseStamped
from nav2_msgs.action import NavigateToPose

from .motion_authority import MotionAuthority, NAV_AUTHORITIES, nav_motion_is_quiescent


class NavGoalManager:
    def __init__(
        self,
        client,
        *,
        now_stamp: Callable[[], object],
        copy_pose: Callable[[PoseStamped], PoseStamped],
        log,
        set_authority: Callable[[MotionAuthority, str], None],
        current_authority: Callable[[], MotionAuthority],
        on_goal_sent: Callable[[str], None],
        on_failure: Callable[[str, str], None],
        on_result: Callable[[str, object, str], None],
    ) -> None:
        self.client = client
        self._now_stamp = now_stamp
        self._copy_pose = copy_pose
        self._log = log
        self._set_authority = set_authority
        self._current_authority = current_authority
        self._on_goal_sent = on_goal_sent
        self._on_failure = on_failure
        self._on_result = on_result

        self.goal_epoch = 0
        self.active_goal_handle = None
        self.active_goal_source: Optional[str] = None
        self.inflight_goal_ids: set[int] = set()
        self.cancel_requests = 0
        self.pending_goal: Optional[PoseStamped] = None
        self.pending_source: Optional[str] = None

    @property
    def active_goal_count(self) -> int:
        return (1 if self.active_goal_handle is not None else 0) + len(self.inflight_goal_ids)

    @property
    def is_idle(self) -> bool:
        return nav_motion_is_quiescent(self.active_goal_count, self.cancel_requests)

    @property
    def has_pending_goal(self) -> bool:
        return self.pending_goal is not None

    def request_goal(self, pose: PoseStamped, source: str) -> None:
        """Keep only the latest goal, canceling the prior accepted one."""
        self.pending_goal = self._copy_pose(pose)
        self.pending_source = source
        if self.active_goal_handle is not None or self.inflight_goal_ids:
            self.invalidate(f'new_{source}_goal', clear_pending=False)

    def dispatch(self, *, source_allowed: Callable[[str], bool], can_send: Callable[[], bool], action_name: str) -> None:
        if self.pending_goal is None or self.pending_source is None:
            return
        source = self.pending_source
        if not source_allowed(source):
            self.pending_goal = None
            self.pending_source = None
            return
        if not can_send():
            return
        if not self.client.server_is_ready():
            self._log.warning(
                f'FIELD_NAV2_WAIT | source={source} action={action_name}',
                throttle_duration_sec=5.0,
            )
            return

        pose = self.pending_goal
        self.pending_goal = None
        self.pending_source = None
        goal = NavigateToPose.Goal()
        goal.pose = deepcopy(pose)
        goal.pose.header.frame_id = goal.pose.header.frame_id or 'map'
        goal.pose.header.stamp = self._now_stamp()
        self.goal_epoch += 1
        goal_id = self.goal_epoch
        try:
            future = self.client.send_goal_async(goal)
        except Exception as exc:  # noqa: BLE001
            self._log.error(f'FIELD_NAV_GOAL_SEND_ERROR | source={source} {exc}')
            self._on_failure(source, 'send_exception')
            return
        self.inflight_goal_ids.add(goal_id)
        authority = (
            MotionAuthority.NORMAL_FOLLOW
            if source == 'FOLLOW' else MotionAuthority.FAILOVER_RECOVERY_NAV
        )
        self._on_goal_sent(source)
        self._set_authority(authority, f'{source.lower()}_goal_sent')
        future.add_done_callback(partial(self._goal_response_cb, goal_id=goal_id, source=source))
        self._log.warning(
            f'FIELD_NAV_GOAL_SENT | source={source} '
            f'x={goal.pose.pose.position.x:.3f} y={goal.pose.pose.position.y:.3f}'
        )

    def invalidate(self, reason: str, *, clear_pending: bool) -> None:
        self.goal_epoch += 1
        if clear_pending:
            self.pending_goal = None
            self.pending_source = None
        handle = self.active_goal_handle
        self.active_goal_handle = None
        self.active_goal_source = None
        if self._current_authority() in NAV_AUTHORITIES:
            self._set_authority(MotionAuthority.NONE, reason)
        if handle is not None:
            self._request_cancel(handle, self.goal_epoch - 1, reason)

    def _goal_response_cb(self, future, *, goal_id: int, source: str) -> None:
        self.inflight_goal_ids.discard(goal_id)
        try:
            handle = future.result()
        except Exception as exc:  # noqa: BLE001
            self._log.warning(f'FIELD_NAV_GOAL_ERROR | source={source} {exc}')
            if goal_id == self.goal_epoch:
                self._set_authority(MotionAuthority.NONE, 'goal_response_error')
                self._on_failure(source, 'response_exception')
            return
        if goal_id != self.goal_epoch:
            if handle.accepted:
                self._request_cancel(handle, goal_id, 'stale_goal_response')
            return
        if not handle.accepted:
            self._log.warning(f'FIELD_NAV_GOAL_REJECTED | source={source}')
            self._set_authority(MotionAuthority.NONE, 'goal_rejected')
            self._on_failure(source, 'rejected')
            return
        self._log.warning(
            f'FIELD_NAV_GOAL_ACCEPTED | source={source} goal_id={goal_id}'
        )
        self.active_goal_handle = handle
        self.active_goal_source = source
        handle.get_result_async().add_done_callback(
            partial(self._goal_result_cb, goal_id=goal_id, source=source)
        )

    def _goal_result_cb(self, future, *, goal_id: int, source: str) -> None:
        try:
            result = future.result()
            status = result.status
        except Exception as exc:  # noqa: BLE001
            status, error = None, str(exc)
        else:
            error = ''
        if goal_id != self.goal_epoch:
            self._log.info(
                f'STALE_FIELD_NAV_RESULT_IGNORED | source={source} '
                f'goal_id={goal_id} current={self.goal_epoch}'
            )
            return
        self.active_goal_handle = None
        self.active_goal_source = None
        self._set_authority(MotionAuthority.NONE, 'goal_result')
        self._log.warning(f'FIELD_NAV_RESULT | source={source} status={status}')
        self._on_result(source, status, error)

    def _request_cancel(self, handle, goal_id: int, reason: str) -> None:
        try:
            future = handle.cancel_goal_async()
            self._log.warning(f'FIELD_NAV_CANCEL | reason={reason}')
        except Exception as exc:  # noqa: BLE001
            self._log.warning(f'FIELD_NAV_CANCEL_ERROR | reason={reason} {exc}')
            return
        self.cancel_requests += 1
        future.add_done_callback(partial(self._cancel_response_cb, goal_id=goal_id, reason=reason))

    def _cancel_response_cb(self, future, *, goal_id: int, reason: str) -> None:
        self.cancel_requests = max(0, self.cancel_requests - 1)
        try:
            response = future.result()
            count = len(getattr(response, 'goals_canceling', []) or [])
        except Exception as exc:  # noqa: BLE001
            self._log.warning(
                f'FIELD_NAV_CANCEL_ACK_ERROR | goal_id={goal_id} reason={reason} {exc}'
            )
            return
        self._log.info(
            f'FIELD_NAV_CANCEL_ACK | goal_id={goal_id} '
            f'reason={reason} goals_canceling={count}'
        )
