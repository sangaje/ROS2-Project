#!/usr/bin/env python3

from __future__ import annotations

import math
from typing import Optional, Tuple

import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from nav2_msgs.action import NavigateToPose


def _yaw_from_quat(q) -> float:
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


def _quat_from_yaw(yaw: float):
    half = yaw * 0.5
    return (0.0, 0.0, math.sin(half), math.cos(half))


class DomainBridgeNav2Follower(Node):
    """Burger follower in Domain 26.

    Input from domain_bridge:
      /leader_pose, frame=map

    Output in Domain 26:
      /navigate_to_pose action goal

    v40 keeps real Nav2. It does not publish /cmd_vel directly.
    """

    def __init__(self) -> None:
        super().__init__('domain_bridge_nav2_follower')
        self.declare_parameter('leader_pose_topic', '/leader_pose')
        self.declare_parameter('navigate_action', '/navigate_to_pose')
        self.declare_parameter('follow_distance', 1.05)
        self.declare_parameter('goal_period_sec', 1.5)
        self.declare_parameter('goal_update_distance', 0.25)
        self.declare_parameter('wait_for_action_timeout_sec', 1.0)
        self.declare_parameter('cancel_previous_goal', False)
        self.declare_parameter('log_wait_every_n', 10)

        self.leader_pose_topic = self._abs(str(self.get_parameter('leader_pose_topic').value))
        self.navigate_action = self._abs(str(self.get_parameter('navigate_action').value))
        self.follow_distance = float(self.get_parameter('follow_distance').value)
        self.goal_period_sec = float(self.get_parameter('goal_period_sec').value)
        self.goal_update_distance = float(self.get_parameter('goal_update_distance').value)
        self.wait_timeout = float(self.get_parameter('wait_for_action_timeout_sec').value)
        self.cancel_previous_goal = bool(self.get_parameter('cancel_previous_goal').value)
        self.log_wait_every_n = max(1, int(self.get_parameter('log_wait_every_n').value))

        self.leader_pose: Optional[PoseStamped] = None
        self.last_goal_xy: Optional[Tuple[float, float]] = None
        self.active_goal_handle = None
        self.goal_count = 0
        self.wait_count = 0
        self._action_ready = False

        self.sub = self.create_subscription(PoseStamped, self.leader_pose_topic, self._on_leader_pose, 20)
        self.client = ActionClient(self, NavigateToPose, self.navigate_action)
        self.create_timer(self.goal_period_sec, self._tick)
        self.get_logger().info(
            'V40_DOMAIN_BRIDGE_NAV2_FOLLOWER_READY | '
            f'in={self.leader_pose_topic} action={self.navigate_action} follow_distance={self.follow_distance:.2f} '
            f'period={self.goal_period_sec:.2f}s update_dist={self.goal_update_distance:.2f}'
        )

    @staticmethod
    def _abs(topic: str) -> str:
        return topic if topic.startswith('/') else '/' + topic

    def _on_leader_pose(self, msg: PoseStamped) -> None:
        self.leader_pose = msg

    def _target_from_leader(self) -> PoseStamped:
        assert self.leader_pose is not None
        p = self.leader_pose.pose.position
        yaw = _yaw_from_quat(self.leader_pose.pose.orientation)
        tx = p.x - self.follow_distance * math.cos(yaw)
        ty = p.y - self.follow_distance * math.sin(yaw)
        qx, qy, qz, qw = _quat_from_yaw(yaw)

        goal = PoseStamped()
        goal.header.stamp = self.get_clock().now().to_msg()
        goal.header.frame_id = 'map'
        goal.pose.position.x = tx
        goal.pose.position.y = ty
        goal.pose.position.z = 0.0
        goal.pose.orientation.x = qx
        goal.pose.orientation.y = qy
        goal.pose.orientation.z = qz
        goal.pose.orientation.w = qw
        return goal

    def _tick(self) -> None:
        if self.leader_pose is None:
            self.wait_count += 1
            if self.wait_count % self.log_wait_every_n == 1:
                self.get_logger().warn('V40_FOLLOW_WAIT | no /leader_pose yet')
            return

        if not self._action_ready:
            self._action_ready = self.client.server_is_ready()
            if not self._action_ready:
                self.wait_count += 1
                if self.wait_count % self.log_wait_every_n == 1:
                    self.get_logger().warn(f'V40_FOLLOW_WAIT | action server not ready: {self.navigate_action}')
                return

        goal_pose = self._target_from_leader()
        gx = goal_pose.pose.position.x
        gy = goal_pose.pose.position.y
        if self.last_goal_xy is not None:
            dx = gx - self.last_goal_xy[0]
            dy = gy - self.last_goal_xy[1]
            if math.hypot(dx, dy) < self.goal_update_distance:
                return

        if self.cancel_previous_goal and self.active_goal_handle is not None:
            try:
                self.active_goal_handle.cancel_goal_async()
            except Exception:
                pass

        goal_msg = NavigateToPose.Goal()
        goal_msg.pose = goal_pose
        self.last_goal_xy = (gx, gy)
        self.goal_count += 1
        self.get_logger().info(f'V40_FOLLOW_GOAL_SENT | n={self.goal_count} target=({gx:.2f},{gy:.2f})')
        future = self.client.send_goal_async(goal_msg)
        future.add_done_callback(self._on_goal_response)

    def _on_goal_response(self, future) -> None:
        try:
            handle = future.result()
        except Exception as e:
            self.get_logger().error(f'V40_FOLLOW_GOAL_ERROR | {e}')
            return
        if not handle.accepted:
            self.get_logger().warn('V40_FOLLOW_GOAL_REJECTED')
            return
        self.active_goal_handle = handle
        self.get_logger().info('V40_FOLLOW_GOAL_ACCEPTED')


def main() -> None:
    rclpy.init()
    node = DomainBridgeNav2Follower()
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
