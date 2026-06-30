#!/usr/bin/env python3
"""One-shot test publisher for /fleet/group_goal."""

import math
import sys
import time

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped


def yaw_to_quat(yaw: float):
    qz = math.sin(yaw * 0.5)
    qw = math.cos(yaw * 0.5)
    return 0.0, 0.0, qz, qw


class GroupGoalSender(Node):
    def __init__(self):
        super().__init__('group_goal_sender')
        self.pub = self.create_publisher(PoseStamped, '/fleet/group_goal', 10)

    def send(self, x: float, y: float, yaw: float, frame_id: str = 'map'):
        msg = PoseStamped()
        msg.header.frame_id = frame_id
        msg.pose.position.x = float(x)
        msg.pose.position.y = float(y)
        msg.pose.position.z = 0.0
        qx, qy, qz, qw = yaw_to_quat(float(yaw))
        msg.pose.orientation.x = qx
        msg.pose.orientation.y = qy
        msg.pose.orientation.z = qz
        msg.pose.orientation.w = qw

        time.sleep(0.5)
        for i in range(5):
            msg.header.stamp = self.get_clock().now().to_msg()
            self.pub.publish(msg)
            rclpy.spin_once(self, timeout_sec=0.05)

        self.get_logger().info(f'GROUP_GOAL_SENT | x={x:.3f} y={y:.3f} yaw={yaw:.3f} frame={frame_id}')


def main():
    if len(sys.argv) < 4:
        print('Usage: ros2 run tb3_fleet_master send_group_goal -- <x> <y> <yaw_rad> [frame_id]')
        print('Example: ros2 run tb3_fleet_master send_group_goal -- 2.0 -1.0 0.0')
        sys.exit(1)

    x = float(sys.argv[1])
    y = float(sys.argv[2])
    yaw = float(sys.argv[3])
    frame_id = sys.argv[4] if len(sys.argv) >= 5 else 'map'

    rclpy.init()
    node = GroupGoalSender()
    node.send(x, y, yaw, frame_id)
    node.destroy_node()
    try:
        if rclpy.ok():
            rclpy.shutdown()
    except Exception:
        pass


if __name__ == '__main__':
    main()
