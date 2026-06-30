#!/usr/bin/env python3
from __future__ import annotations

import math
import rclpy
from rclpy.node import Node
from rclpy.exceptions import ParameterAlreadyDeclaredException
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry


def safe_declare(node: Node, name: str, default):
    try:
        node.declare_parameter(name, default)
    except ParameterAlreadyDeclaredException:
        pass
    return node.get_parameter(name).value


def yaw_from_quat(q) -> float:
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


def quat_from_yaw(yaw: float):
    class Q:
        pass
    q = Q()
    q.x = 0.0
    q.y = 0.0
    q.z = math.sin(yaw * 0.5)
    q.w = math.cos(yaw * 0.5)
    return q


class OdomPosePublisher(Node):
    """Publish an odom-integrated PoseStamped in the chosen global frame.

    This does not publish TF. It is only a fallback pose topic for the A* controller
    while Cartographer has not yet produced map->odom and /leader_pose is unavailable.
    With frame_tools_v40, /odom_nav starts at the robot start pose as (0,0,0), so the
    default initial offset is also (0,0,0), matching Cartographer's early SLAM frame.
    """

    def __init__(self):
        super().__init__('odom_pose_publisher_v59')
        safe_declare(self, 'use_sim_time', True)
        self.odom_topic = str(safe_declare(self, 'odom_topic', '/odom_nav'))
        self.output_topic = str(safe_declare(self, 'output_topic', '/leader_pose_odom_fallback'))
        self.frame_id = str(safe_declare(self, 'frame_id', 'map'))
        self.initial_x = float(safe_declare(self, 'initial_x', 0.0))
        self.initial_y = float(safe_declare(self, 'initial_y', 0.0))
        self.initial_yaw = float(safe_declare(self, 'initial_yaw', 0.0))
        self.log_every_n = int(safe_declare(self, 'log_every_n', 100))
        self.count = 0
        self.pub = self.create_publisher(PoseStamped, self.output_topic, 10)
        self.sub = self.create_subscription(Odometry, self.odom_topic, self.on_odom, 30)
        self.get_logger().info(
            f'V59_ODOM_POSE_FALLBACK_READY | odom={self.odom_topic} out={self.output_topic} '
            f'frame={self.frame_id} initial=({self.initial_x:.2f},{self.initial_y:.2f},{self.initial_yaw:.2f})'
        )

    def on_odom(self, msg: Odometry):
        ox = float(msg.pose.pose.position.x)
        oy = float(msg.pose.pose.position.y)
        oyaw = yaw_from_quat(msg.pose.pose.orientation)
        c = math.cos(self.initial_yaw)
        s = math.sin(self.initial_yaw)
        x = self.initial_x + c * ox - s * oy
        y = self.initial_y + s * ox + c * oy
        yaw = self.initial_yaw + oyaw
        q = quat_from_yaw(yaw)

        out = PoseStamped()
        out.header.stamp = msg.header.stamp
        if out.header.stamp.sec == 0 and out.header.stamp.nanosec == 0:
            out.header.stamp = self.get_clock().now().to_msg()
        out.header.frame_id = self.frame_id
        out.pose.position.x = x
        out.pose.position.y = y
        out.pose.position.z = float(msg.pose.pose.position.z)
        out.pose.orientation.x = q.x
        out.pose.orientation.y = q.y
        out.pose.orientation.z = q.z
        out.pose.orientation.w = q.w
        self.pub.publish(out)
        self.count += 1
        if self.log_every_n > 0 and self.count % self.log_every_n == 0:
            self.get_logger().info(f'V59_ODOM_POSE_FALLBACK | out={self.output_topic} n={self.count} xy=({x:.2f},{y:.2f}) yaw={yaw:.2f}')


def main():
    rclpy.init()
    node = OdomPosePublisher()
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
