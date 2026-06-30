#!/usr/bin/env python3

from __future__ import annotations

import math

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped


def _yaw_from_quat(q) -> float:
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


def _yaw_to_quat(yaw: float):
    half = yaw * 0.5
    return (0.0, 0.0, math.sin(half), math.cos(half))


class LeaderPosePublisher(Node):
    """Publish leader pose in the shared map frame.

    Gazebo model odometry is relative to the model spawn pose and often starts near
    (0, 0). Earlier versions simply relabeled that odometry as map, which made
    /leader_pose wrong. v29 adds the known spawn/initial pose offset so the bridge
    sends a map-frame leader pose to the follower domain.
    """

    def __init__(self) -> None:
        super().__init__('leader_pose_publisher')
        self.declare_parameter('odom_topic', '/odom')
        self.declare_parameter('leader_pose_topic', '/leader_pose')
        self.declare_parameter('output_frame_id', 'map')
        self.declare_parameter('initial_x', -1.8)
        self.declare_parameter('initial_y', 0.5)
        self.declare_parameter('initial_yaw', 0.0)
        self.declare_parameter('apply_initial_offset', True)
        self.declare_parameter('log_every_n', 50)

        self.odom_topic = self._abs(str(self.get_parameter('odom_topic').value))
        self.leader_pose_topic = self._abs(str(self.get_parameter('leader_pose_topic').value))
        self.output_frame_id = str(self.get_parameter('output_frame_id').value)
        self.initial_x = float(self.get_parameter('initial_x').value)
        self.initial_y = float(self.get_parameter('initial_y').value)
        self.initial_yaw = float(self.get_parameter('initial_yaw').value)
        self.apply_initial_offset = bool(self.get_parameter('apply_initial_offset').value)
        self.log_every_n = max(0, int(self.get_parameter('log_every_n').value))
        self.count = 0

        self.pub = self.create_publisher(PoseStamped, self.leader_pose_topic, 10)
        self.sub = self.create_subscription(Odometry, self.odom_topic, self._on_odom, 20)
        self.get_logger().info(
            'V29_LEADER_POSE_PUBLISHER_READY | '
            f'in={self.odom_topic} out={self.leader_pose_topic} frame={self.output_frame_id} '
            f'initial=({self.initial_x:.2f},{self.initial_y:.2f},{self.initial_yaw:.2f}) '
            f'apply_initial_offset={self.apply_initial_offset}'
        )

    @staticmethod
    def _abs(topic: str) -> str:
        return topic if topic.startswith('/') else '/' + topic

    def _on_odom(self, msg: Odometry) -> None:
        out = PoseStamped()
        out.header.stamp = msg.header.stamp
        if out.header.stamp.sec == 0 and out.header.stamp.nanosec == 0:
            out.header.stamp = self.get_clock().now().to_msg()
        out.header.frame_id = self.output_frame_id

        odom_x = msg.pose.pose.position.x
        odom_y = msg.pose.pose.position.y
        odom_yaw = _yaw_from_quat(msg.pose.pose.orientation)

        if self.apply_initial_offset:
            c = math.cos(self.initial_yaw)
            s = math.sin(self.initial_yaw)
            map_x = self.initial_x + c * odom_x - s * odom_y
            map_y = self.initial_y + s * odom_x + c * odom_y
            map_yaw = self.initial_yaw + odom_yaw
        else:
            map_x = odom_x
            map_y = odom_y
            map_yaw = odom_yaw

        qx, qy, qz, qw = _yaw_to_quat(map_yaw)
        out.pose.position.x = map_x
        out.pose.position.y = map_y
        out.pose.position.z = msg.pose.pose.position.z
        out.pose.orientation.x = qx
        out.pose.orientation.y = qy
        out.pose.orientation.z = qz
        out.pose.orientation.w = qw
        self.pub.publish(out)

        self.count += 1
        if self.log_every_n and self.count % self.log_every_n == 0:
            self.get_logger().info(
                f'V29_LEADER_POSE_PUB | n={self.count} map_xy=({map_x:.2f},{map_y:.2f}) '
                f'odom_xy=({odom_x:.2f},{odom_y:.2f}) frame={out.header.frame_id}'
            )


def main() -> None:
    rclpy.init()
    node = LeaderPosePublisher()
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
