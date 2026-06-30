#!/usr/bin/env python3

from __future__ import annotations

from typing import Optional

import math

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
        self.goal_pose_topic = self._abs(str(_safe_declare(self, 'goal_pose_topic', '/goal_pose')))
        self.navigate_action = self._strip_action(str(_safe_declare(self, 'navigate_action', '/navigate_to_pose')))
        self.default_frame_id = str(_safe_declare(self, 'default_frame_id', 'map'))
        self.stamp_with_now = bool(_safe_declare(self, 'stamp_with_now', True))
        self.wait_for_server_sec = float(_safe_declare(self, 'wait_for_server_sec', 2.0))
        self.cancel_previous_goal = bool(_safe_declare(self, 'cancel_previous_goal', True))
        self.log_feedback = bool(_safe_declare(self, 'log_feedback', False))
        self.ignore_duplicate_goals = bool(_safe_declare(self, 'ignore_duplicate_goals', True))
        self.same_goal_xy_tolerance_m = float(_safe_declare(self, 'same_goal_xy_tolerance_m', 0.12))
        self.same_goal_yaw_tolerance_rad = float(_safe_declare(self, 'same_goal_yaw_tolerance_rad', 0.35))
        self.min_resend_period_sec = float(_safe_declare(self, 'min_resend_period_sec', 4.0))

        self.client: ActionClient = ActionClient(self, NavigateToPose, self.navigate_action)
        self.sub = self.create_subscription(PoseStamped, self.goal_pose_topic, self._on_goal_pose, 10)
        self.current_goal_handle = None
        self.current_goal_active = False
        self.last_sent_goal: Optional[PoseStamped] = None
        self.last_sent_time = None
        self.goal_count = 0

        self.get_logger().info(
            'V72_POSE_TO_NAV2_ACTION_READY | '
            f'in={self.goal_pose_topic} action={self.navigate_action} frame={self.default_frame_id} '
            f'cancel_previous={self.cancel_previous_goal} dedupe={self.ignore_duplicate_goals} '
            f'xy_tol={self.same_goal_xy_tolerance_m:.2f} resend_min={self.min_resend_period_sec:.1f}s'
        )

    @staticmethod
    def _abs(topic: str) -> str:
        topic = topic.strip()
        return topic if topic.startswith('/') else '/' + topic

    @staticmethod
    def _strip_action(action_name: str) -> str:
        action_name = action_name.strip()
        return action_name if action_name.startswith('/') else '/' + action_name

    @staticmethod
    def _yaw_from_pose(msg: PoseStamped) -> float:
        q = msg.pose.orientation
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        return math.atan2(siny_cosp, cosy_cosp)

    @staticmethod
    def _angle_diff(a: float, b: float) -> float:
        return math.atan2(math.sin(a - b), math.cos(a - b))

    def _is_duplicate_goal(self, msg: PoseStamped) -> bool:
        if not self.ignore_duplicate_goals or self.last_sent_goal is None:
            return False
        dx = float(msg.pose.position.x) - float(self.last_sent_goal.pose.position.x)
        dy = float(msg.pose.position.y) - float(self.last_sent_goal.pose.position.y)
        dist = math.hypot(dx, dy)
        dyaw = abs(self._angle_diff(self._yaw_from_pose(msg), self._yaw_from_pose(self.last_sent_goal)))
        if dist > self.same_goal_xy_tolerance_m or dyaw > self.same_goal_yaw_tolerance_rad:
            return False
        now = self.get_clock().now()
        if self.last_sent_time is not None:
            try:
                dt = (now - self.last_sent_time).nanoseconds * 1e-9
            except Exception:
                dt = self.min_resend_period_sec
            if dt >= self.min_resend_period_sec and not self.current_goal_active:
                return False
        return True

    def _on_goal_pose(self, msg: PoseStamped) -> None:
        if self._is_duplicate_goal(msg):
            x = float(msg.pose.position.x)
            y = float(msg.pose.position.y)
            self.get_logger().info(
                f'V72_IGNORE_DUPLICATE_GOAL | topic={self.goal_pose_topic} xy=({x:.3f},{y:.3f}) active={self.current_goal_active}'
            )
            return

        if not self.client.wait_for_server(timeout_sec=self.wait_for_server_sec):
            self.get_logger().warn(
                f'V72_NAV2_ACTION_SERVER_NOT_READY | action={self.navigate_action} '
                f'goal_topic={self.goal_pose_topic}'
            )
            return

        if self.cancel_previous_goal and self.current_goal_handle is not None:
            try:
                self.current_goal_handle.cancel_goal_async()
            except Exception as exc:  # noqa: BLE001
                self.get_logger().warn(f'V72_CANCEL_PREVIOUS_GOAL_FAILED | {exc}')

        goal = NavigateToPose.Goal()
        goal.pose = msg
        if not goal.pose.header.frame_id:
            goal.pose.header.frame_id = self.default_frame_id
        if self.stamp_with_now:
            goal.pose.header.stamp = self.get_clock().now().to_msg()

        self.goal_count += 1
        x = goal.pose.pose.position.x
        y = goal.pose.pose.position.y
        self.get_logger().info(
            f'V72_SEND_NAV2_GOAL | n={self.goal_count} topic={self.goal_pose_topic} '
            f'action={self.navigate_action} frame={goal.pose.header.frame_id} xy=({x:.3f},{y:.3f})'
        )

        self.last_sent_goal = goal.pose
        self.last_sent_time = self.get_clock().now()
        self.current_goal_active = True
        future = self.client.send_goal_async(goal, feedback_callback=self._feedback_cb if self.log_feedback else None)
        future.add_done_callback(self._goal_response_cb)

    def _goal_response_cb(self, future) -> None:
        try:
            goal_handle = future.result()
        except Exception as exc:  # noqa: BLE001
            self.get_logger().error(f'V72_NAV2_GOAL_SEND_EXCEPTION | {exc}')
            return
        if not goal_handle.accepted:
            self.current_goal_active = False
            self.get_logger().warn('V72_NAV2_GOAL_REJECTED')
            return
        self.current_goal_handle = goal_handle
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self._result_cb)
        self.get_logger().info('V72_NAV2_GOAL_ACCEPTED')

    def _result_cb(self, future) -> None:
        try:
            result = future.result()
        except Exception as exc:  # noqa: BLE001
            self.get_logger().error(f'V72_NAV2_GOAL_RESULT_EXCEPTION | {exc}')
            return
        self.current_goal_active = False
        self.current_goal_handle = None
        self.get_logger().info(f'V72_NAV2_GOAL_RESULT | status={result.status}')

    def _feedback_cb(self, feedback_msg) -> None:
        fb = feedback_msg.feedback
        try:
            remaining = fb.distance_remaining
        except AttributeError:
            remaining = float('nan')
        self.get_logger().info(f'V72_NAV2_GOAL_FEEDBACK | distance_remaining={remaining:.3f}')


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
