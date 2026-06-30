#!/usr/bin/env python3

from __future__ import annotations

import math
from typing import Optional

import rclpy
from rclpy.node import Node
from rclpy.exceptions import ParameterAlreadyDeclaredException
from geometry_msgs.msg import PoseStamped
from tf2_ros import Buffer, TransformListener, TransformException


def _safe_declare(node: Node, name: str, default):
    try:
        node.declare_parameter(name, default)
    except ParameterAlreadyDeclaredException:
        pass
    return node.get_parameter(name).value


class TfPosePublisher(Node):
    """Publish PoseStamped from a TF transform.

    v44 purpose:
      - Waffle leader pose under Cartographer: publish map->base_footprint, not dead-reckoned odom.
      - Burger debug pose: publish map->base_footprint using its local map->odom publisher.
    """

    def __init__(self) -> None:
        super().__init__('tf_pose_publisher_v44')
        _safe_declare(self, 'use_sim_time', True)
        self.output_topic = self._abs(str(_safe_declare(self, 'output_topic', '/leader_pose')))
        self.target_frame = str(_safe_declare(self, 'target_frame', 'map'))
        self.source_frame = str(_safe_declare(self, 'source_frame', 'base_footprint'))
        self.publish_rate_hz = max(1.0, float(_safe_declare(self, 'publish_rate_hz', 10.0)))
        self.timeout_sec = max(0.01, float(_safe_declare(self, 'timeout_sec', 0.05)))
        self.log_every_n = max(0, int(_safe_declare(self, 'log_every_n', 200)))

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.pub = self.create_publisher(PoseStamped, self.output_topic, 10)
        self.count = 0
        self.miss_count = 0
        self.create_timer(1.0 / self.publish_rate_hz, self._tick)
        self.get_logger().info(
            f'V44_TF_POSE_PUBLISHER_READY | {self.target_frame}->{self.source_frame} -> {self.output_topic} '
            f'rate={self.publish_rate_hz:.1f}Hz'
        )

    @staticmethod
    def _abs(topic: str) -> str:
        topic = topic.strip()
        return topic if topic.startswith('/') else '/' + topic

    def _tick(self) -> None:
        try:
            tf = self.tf_buffer.lookup_transform(
                self.target_frame,
                self.source_frame,
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=self.timeout_sec),
            )
        except TransformException as exc:
            self.miss_count += 1
            if self.miss_count <= 5 or (self.log_every_n and self.miss_count % self.log_every_n == 0):
                self.get_logger().warn(
                    f'V44_TF_POSE_WAIT | target={self.target_frame} source={self.source_frame} miss={self.miss_count} reason={exc}'
                )
            return

        msg = PoseStamped()
        msg.header = tf.header
        msg.header.frame_id = self.target_frame
        msg.pose.position.x = tf.transform.translation.x
        msg.pose.position.y = tf.transform.translation.y
        msg.pose.position.z = tf.transform.translation.z
        msg.pose.orientation = tf.transform.rotation
        self.pub.publish(msg)

        self.count += 1
        if self.log_every_n and self.count % self.log_every_n == 0:
            q = msg.pose.orientation
            yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))
            self.get_logger().info(
                f'V44_TF_POSE | topic={self.output_topic} n={self.count} xy=({msg.pose.position.x:.2f},{msg.pose.position.y:.2f}) yaw={yaw:.2f}'
            )


def main() -> None:
    rclpy.init()
    node = TfPosePublisher()
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
