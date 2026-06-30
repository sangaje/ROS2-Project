#!/usr/bin/env python3

from __future__ import annotations

import math
import sys

import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from nav2_msgs.action import NavigateToPose


def _quat_from_yaw(yaw: float):
    half = yaw * 0.5
    return (0.0, 0.0, math.sin(half), math.cos(half))


class SendNavGoal(Node):
    def __init__(self, x: float, y: float, yaw: float) -> None:
        super().__init__('send_nav_goal')
        self.client = ActionClient(self, NavigateToPose, '/navigate_to_pose')
        self.x = x
        self.y = y
        self.yaw = yaw

    def send(self) -> None:
        self.get_logger().info(f'Waiting for /navigate_to_pose ...')
        self.client.wait_for_server()
        qx, qy, qz, qw = _quat_from_yaw(self.yaw)
        goal = NavigateToPose.Goal()
        goal.pose.header.stamp = self.get_clock().now().to_msg()
        goal.pose.header.frame_id = 'map'
        goal.pose.pose.position.x = self.x
        goal.pose.pose.position.y = self.y
        goal.pose.pose.position.z = 0.0
        goal.pose.pose.orientation.x = qx
        goal.pose.pose.orientation.y = qy
        goal.pose.pose.orientation.z = qz
        goal.pose.pose.orientation.w = qw
        self.get_logger().info(f'SEND_NAV_GOAL | target=({self.x:.2f},{self.y:.2f},{self.yaw:.2f}) action=/navigate_to_pose')
        future = self.client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, future)
        handle = future.result()
        if handle is None or not handle.accepted:
            self.get_logger().error('SEND_NAV_GOAL_REJECTED')
            return
        self.get_logger().info('SEND_NAV_GOAL_ACCEPTED')


def main() -> None:
    if len(sys.argv) < 4:
        print('Usage: ros2 run tb3_fleet_bringup send_nav_goal -- X Y YAW')
        raise SystemExit(2)
    x, y, yaw = float(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3])
    rclpy.init()
    node = SendNavGoal(x, y, yaw)
    try:
        node.send()
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
