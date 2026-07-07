#!/usr/bin/env python3

from __future__ import annotations

import math
from typing import Iterable, Optional

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


def _frame_candidates(raw: object, primary: str) -> list[str]:
    if isinstance(raw, str):
        values: Iterable[object] = raw.split(',')
    else:
        values = raw if isinstance(raw, Iterable) else []
    frames = []
    for value in values:
        frame = str(value).strip().strip('/')
        if frame and frame not in frames:
            frames.append(frame)
    primary = primary.strip().strip('/')
    if primary and primary not in frames:
        frames.insert(0, primary)
    return frames or ['base_footprint']


class TfPosePublisher(Node):
    """Publish PoseStamped from a TF transform.

    v44 purpose:
      - Leader pose under Cartographer: publish map->base_footprint, not dead-reckoned odom.
      - Burger debug pose: publish map->base_footprint using its local map->odom publisher.
    """

    def __init__(self) -> None:
        super().__init__('tf_pose_publisher_v44')
        _safe_declare(self, 'use_sim_time', True)
        self.output_topic = self._abs(str(_safe_declare(self, 'output_topic', '/leader_pose')))
        self.target_frame = str(_safe_declare(self, 'target_frame', 'map'))
        self.source_frame = str(_safe_declare(self, 'source_frame', 'base_footprint'))
        self.source_frame_candidates = _frame_candidates(
            _safe_declare(
                self,
                'source_frame_candidates',
                [self.source_frame, 'base_link'],
            ),
            self.source_frame,
        )
        self.publish_rate_hz = max(1.0, float(_safe_declare(self, 'publish_rate_hz', 10.0)))
        self.timeout_sec = max(0.01, float(_safe_declare(self, 'timeout_sec', 0.05)))
        self.log_every_n = max(0, int(_safe_declare(self, 'log_every_n', 200)))

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.pub = self.create_publisher(PoseStamped, self.output_topic, 10)
        self.count = 0
        self.miss_count = 0
        self.first_pose_logged = False
        self.create_timer(1.0 / self.publish_rate_hz, self._tick)
        self.get_logger().info(
            f'V44_TF_POSE_PUBLISHER_READY | target={self.target_frame} '
            f'sources={self.source_frame_candidates} -> {self.output_topic} '
            f'rate={self.publish_rate_hz:.1f}Hz'
        )

    @staticmethod
    def _abs(topic: str) -> str:
        topic = topic.strip()
        return topic if topic.startswith('/') else '/' + topic

    def _tick(self) -> None:
        last_error = None
        selected_source = None
        tf = None
        for source_frame in self.source_frame_candidates:
            try:
                tf = self.tf_buffer.lookup_transform(
                    self.target_frame,
                    source_frame,
                    rclpy.time.Time(),
                    timeout=rclpy.duration.Duration(seconds=self.timeout_sec),
                )
                selected_source = source_frame
                break
            except TransformException as exc:
                last_error = exc

        if tf is None or selected_source is None:
            self.miss_count += 1
            if self.miss_count <= 5 or (self.log_every_n and self.miss_count % self.log_every_n == 0):
                self.get_logger().warn(
                    f'V44_TF_POSE_WAIT | target={self.target_frame} '
                    f'sources={self.source_frame_candidates} '
                    f'miss={self.miss_count} reason={last_error}'
                )
            return

        self._publish_pose(tf, selected_source)

    def _publish_pose(self, tf, selected_source: str) -> None:
        if self.miss_count:
            self.get_logger().info(
                f'V44_TF_POSE_RECOVERED | target={self.target_frame} '
                f'source={selected_source} topic={self.output_topic}'
            )
            self.miss_count = 0

        msg = PoseStamped()
        msg.header = tf.header
        msg.header.frame_id = self.target_frame
        msg.pose.position.x = tf.transform.translation.x
        msg.pose.position.y = tf.transform.translation.y
        msg.pose.position.z = tf.transform.translation.z
        msg.pose.orientation = tf.transform.rotation
        self.pub.publish(msg)

        self.count += 1
        if not self.first_pose_logged:
            self.first_pose_logged = True
            tag = (
                'LEADER_LOCAL_POSE_FIRST_RX'
                if self.output_topic == '/leader_pose'
                else 'TF_POSE_FIRST_RX'
            )
            self.get_logger().info(
                f'{tag} | topic={self.output_topic} target={self.target_frame} '
                f'source={selected_source} frame_id={msg.header.frame_id} '
                f'xy=({msg.pose.position.x:.2f},{msg.pose.position.y:.2f})'
            )
        if self.log_every_n and self.count % self.log_every_n == 0:
            q = msg.pose.orientation
            yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))
            self.get_logger().info(
                f'V44_TF_POSE | topic={self.output_topic} source={selected_source} '
                f'n={self.count} xy=({msg.pose.position.x:.2f},{msg.pose.position.y:.2f}) yaw={yaw:.2f}'
            )

def main() -> None:
    rclpy.init()
    node = TfPosePublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.destroy_node()
        except KeyboardInterrupt:
            pass
        if rclpy.ok():
            try:
                rclpy.shutdown()
            except KeyboardInterrupt:
                pass


if __name__ == '__main__':
    main()
