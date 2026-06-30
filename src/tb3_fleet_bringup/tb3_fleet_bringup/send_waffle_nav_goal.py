#!/usr/bin/env python3
from __future__ import annotations

import math
import sys
import rclpy
from rclpy.action import ActionClient
from nav2_msgs.action import NavigateToPose
from geometry_msgs.msg import PoseStamped


def main() -> None:
    if len(sys.argv) < 3:
        print('usage: ROS_DOMAIN_ID=25 ros2 run tb3_fleet_bringup send_waffle_nav_goal -- X Y [YAW_RAD]')
        sys.exit(2)
    x = float(sys.argv[1])
    y = float(sys.argv[2])
    yaw = float(sys.argv[3]) if len(sys.argv) >= 4 else 0.0

    rclpy.init()
    node = rclpy.create_node('send_waffle_nav_goal')
    client = ActionClient(node, NavigateToPose, '/navigate_to_pose')
    node.get_logger().info('WAIT_ACTION | /navigate_to_pose')
    if not client.wait_for_server(timeout_sec=30.0):
        node.get_logger().error('ACTION_NOT_AVAILABLE | /navigate_to_pose')
        node.destroy_node()
        rclpy.shutdown()
        sys.exit(1)

    goal = NavigateToPose.Goal()
    goal.pose = PoseStamped()
    goal.pose.header.stamp = node.get_clock().now().to_msg()
    goal.pose.header.frame_id = 'map'
    goal.pose.pose.position.x = x
    goal.pose.pose.position.y = y
    goal.pose.pose.orientation.z = math.sin(yaw * 0.5)
    goal.pose.pose.orientation.w = math.cos(yaw * 0.5)

    node.get_logger().info(f'SEND_WAFFLE_NAV_GOAL | x={x:.2f} y={y:.2f} yaw={yaw:.2f}')
    future = client.send_goal_async(goal)
    rclpy.spin_until_future_complete(node, future)
    handle = future.result()
    if handle is None or not handle.accepted:
        node.get_logger().error('WAFFLE_NAV_GOAL_REJECTED')
        node.destroy_node()
        rclpy.shutdown()
        sys.exit(1)
    node.get_logger().info('WAFFLE_NAV_GOAL_ACCEPTED')
    result_future = handle.get_result_async()
    rclpy.spin_until_future_complete(node, result_future)
    node.get_logger().info(f'WAFFLE_NAV_GOAL_DONE | result={result_future.result().result}')
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
