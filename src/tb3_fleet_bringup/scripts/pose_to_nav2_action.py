#!/usr/bin/env python3

from __future__ import annotations

import math
from typing import Optional

import rclpy
from geometry_msgs.msg import PoseStamped
from nav2_msgs.action import NavigateToPose
from rclpy.action import ActionClient
from rclpy.exceptions import ParameterAlreadyDeclaredException
from rclpy.node import Node


def _abs_name(name: str) -> str:
    name = str(name).strip()
    return name if name.startswith('/') else '/' + name


def _declare_if_needed(node: Node, name: str, default):
    try:
        node.declare_parameter(name, default)
    except ParameterAlreadyDeclaredException:
        pass
    return node.get_parameter(name).value


def _yaw_from_pose(msg: PoseStamped) -> float:
    q = msg.pose.orientation
    return math.atan2(
        2.0 * (q.w * q.z + q.x * q.y),
        1.0 - 2.0 * (q.y * q.y + q.z * q.z),
    )


def _yaw_delta(a: float, b: float) -> float:
    return abs(math.atan2(math.sin(a - b), math.cos(a - b)))


def _pose_delta(a: PoseStamped, b: PoseStamped) -> tuple[float, float]:
    dx = float(a.pose.position.x) - float(b.pose.position.x)
    dy = float(a.pose.position.y) - float(b.pose.position.y)
    return math.hypot(dx, dy), _yaw_delta(_yaw_from_pose(a), _yaw_from_pose(b))


class PoseToNav2Action(Node):
    """Convert one PoseStamped goal topic into one Nav2 NavigateToPose action."""

    def __init__(self) -> None:
        super().__init__('pose_to_nav2_action')

        goal_topic = _declare_if_needed(self, 'goal_topic', '/waffle_goal_pose')
        compat_goal_topic = str(_declare_if_needed(self, 'goal_pose_topic', '')).strip()
        action_name = _declare_if_needed(self, 'action_name', '/navigate_to_pose')
        compat_action = str(_declare_if_needed(self, 'navigate_action', '')).strip()
        self.position_epsilon_m = float(_declare_if_needed(self, 'position_epsilon_m', 0.20))
        self.yaw_epsilon_rad = float(_declare_if_needed(self, 'yaw_epsilon_rad', 0.35))
        self.retry_cooldown_sec = float(_declare_if_needed(self, 'retry_cooldown_sec', 5.0))
        self.wait_for_server_timeout_sec = float(_declare_if_needed(self, 'wait_for_server_timeout_sec', 30.0))
        self.cancel_active_on_new_goal = bool(_declare_if_needed(self, 'cancel_active_on_new_goal', False))
        self.default_frame_id = str(_declare_if_needed(self, 'default_frame_id', 'map')).strip() or 'map'
        _declare_if_needed(self, 'use_sim_time', True)

        self.goal_topic = _abs_name(compat_goal_topic or goal_topic)
        self.action_name = _abs_name(compat_action or action_name)

        self.client = ActionClient(self, NavigateToPose, self.action_name)
        self.create_subscription(PoseStamped, self.goal_topic, self._on_goal, 10)
        self.create_timer(0.5, self._tick)

        self.pending_goal: Optional[PoseStamped] = None
        self.active_goal: Optional[PoseStamped] = None
        self.active_goal_handle = None
        self.cancel_in_progress = False
        self.last_finished_goal: Optional[PoseStamped] = None
        self.last_failure_time_sec = -1.0
        self.goal_seq = 0
        self.active_seq = 0
        self.wait_logged = False

        self.get_logger().info(
            'POSE_TO_NAV2_READY | '
            f'goal_topic={self.goal_topic} action_name={self.action_name} '
            f'position_epsilon_m={self.position_epsilon_m:.2f} yaw_epsilon_rad={self.yaw_epsilon_rad:.2f} '
            f'retry_cooldown_sec={self.retry_cooldown_sec:.1f} cancel_active_on_new_goal={self.cancel_active_on_new_goal}'
        )

    def _now_sec(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def _copy_goal(self, msg: PoseStamped) -> PoseStamped:
        out = PoseStamped()
        out.header = msg.header
        out.header.frame_id = out.header.frame_id or self.default_frame_id
        out.header.stamp = self.get_clock().now().to_msg()
        out.pose = msg.pose
        return out

    def _is_same_goal(self, a: Optional[PoseStamped], b: PoseStamped) -> bool:
        if a is None:
            return False
        dist, dyaw = _pose_delta(a, b)
        return dist <= self.position_epsilon_m and dyaw <= self.yaw_epsilon_rad

    def _on_goal(self, msg: PoseStamped) -> None:
        goal = self._copy_goal(msg)
        if self._is_same_goal(self.active_goal, goal):
            self.get_logger().info('IGNORE_DUPLICATE_GOAL | same_as=active')
            return
        if self._is_same_goal(self.pending_goal, goal):
            self.get_logger().info('IGNORE_DUPLICATE_GOAL | same_as=pending')
            return
        if self.active_goal_handle is None and self._is_same_goal(self.last_finished_goal, goal):
            since_failure = self._now_sec() - self.last_failure_time_sec
            if self.last_failure_time_sec > 0.0 and since_failure < self.retry_cooldown_sec:
                self.get_logger().info(
                    f'IGNORE_DUPLICATE_GOAL | retry_cooldown_remaining={self.retry_cooldown_sec - since_failure:.1f}s'
                )
                return

        self.pending_goal = goal
        self.get_logger().info(
            f'QUEUE_NAV2_GOAL | topic={self.goal_topic} xy=({goal.pose.position.x:.3f},{goal.pose.position.y:.3f})'
        )
        if self.cancel_active_on_new_goal and self.active_goal_handle is not None and not self.cancel_in_progress:
            self.cancel_in_progress = True
            self.get_logger().info('CANCEL_ACTIVE_NAV2_GOAL | new_dynamic_goal_pending')
            future = self.active_goal_handle.cancel_goal_async()
            future.add_done_callback(self._on_cancel_done)
            return
        self._tick()

    def _tick(self) -> None:
        if self.pending_goal is None or self.active_goal_handle is not None or self.cancel_in_progress:
            return
        if not self.client.wait_for_server(timeout_sec=0.01):
            if not self.wait_logged:
                self.get_logger().warn(
                    f'NAV2_ACTION_SERVER_NOT_READY | action={self.action_name} waiting_up_to={self.wait_for_server_timeout_sec:.1f}s'
                )
                self.wait_logged = True
            return
        self.wait_logged = False
        self._send_pending_goal()

    def _send_pending_goal(self) -> None:
        assert self.pending_goal is not None
        goal_pose = self.pending_goal
        self.pending_goal = None

        self.goal_seq += 1
        seq = self.goal_seq
        action_goal = NavigateToPose.Goal()
        action_goal.pose = goal_pose

        self.active_goal = goal_pose
        self.active_seq = seq
        self.get_logger().info(
            f'SEND_NAV2_GOAL | seq={seq} action={self.action_name} '
            f'xy=({goal_pose.pose.position.x:.3f},{goal_pose.pose.position.y:.3f})'
        )
        future = self.client.send_goal_async(action_goal)
        future.add_done_callback(lambda fut: self._on_goal_response(seq, fut))

    def _on_cancel_done(self, future) -> None:
        try:
            future.result()
            self.get_logger().info('CANCEL_ACTIVE_NAV2_GOAL_DONE')
        except Exception as exc:
            self.get_logger().warn(f'CANCEL_ACTIVE_NAV2_GOAL_FAILED | error={exc}')
        self.cancel_in_progress = False
        self.active_goal = None
        self.active_goal_handle = None
        self._tick()

    def _on_goal_response(self, seq: int, future) -> None:
        try:
            handle = future.result()
        except Exception as exc:
            self.get_logger().error(f'NAV2_GOAL_REJECTED | seq={seq} exception={exc}')
            self._mark_finished(failed=True)
            return

        if not handle.accepted:
            self.get_logger().warn(f'NAV2_GOAL_REJECTED | seq={seq}')
            self._mark_finished(failed=True)
            return

        self.active_goal_handle = handle
        self.get_logger().info(f'NAV2_GOAL_ACCEPTED | seq={seq}')
        result_future = handle.get_result_async()
        result_future.add_done_callback(lambda fut: self._on_result(seq, fut))

    def _on_result(self, seq: int, future) -> None:
        if seq != self.active_seq:
            self.get_logger().info(f'IGNORE_STALE_NAV2_RESULT | seq={seq} active_seq={self.active_seq}')
            return
        if self.active_goal_handle is None and self.cancel_in_progress:
            return
        failed = True
        try:
            result = future.result()
            status = int(result.status)
            failed = status != 4
            self.get_logger().info(f'NAV2_GOAL_RESULT status={status} | seq={seq}')
        except Exception as exc:
            self.get_logger().error(f'NAV2_GOAL_RESULT status=exception | seq={seq} error={exc}')
        self._mark_finished(failed=failed)
        if self.pending_goal is not None and not self._is_same_goal(self.last_finished_goal, self.pending_goal):
            self.get_logger().info('RETRY_AFTER_FAILURE | pending_new_goal_ready')
            self._tick()

    def _mark_finished(self, failed: bool) -> None:
        self.last_finished_goal = self.active_goal
        self.active_goal = None
        self.active_goal_handle = None
        if failed:
            self.last_failure_time_sec = self._now_sec()


def main() -> None:
    rclpy.init()
    node = PoseToNav2Action()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
