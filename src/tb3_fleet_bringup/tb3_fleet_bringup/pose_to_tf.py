#!/usr/bin/env python3

from __future__ import annotations

import rclpy
from geometry_msgs.msg import PoseStamped, TransformStamped
from rclpy.node import Node
from tf2_ros import TransformBroadcaster


class PoseToTfBroadcaster(Node):
    def __init__(self) -> None:
        super().__init__('pose_to_tf_broadcaster')
        self.declare_parameter('input_topic', '/burger_pose')
        self.declare_parameter('parent_frame', 'map')
        self.declare_parameter('child_frame', 'burger/base_footprint')
        self.declare_parameter('republish_hz', 10.0)
        self.declare_parameter('stale_timeout_sec', 2.0)

        self.input_topic = str(self.get_parameter('input_topic').value)
        self.parent_frame = str(self.get_parameter('parent_frame').value)
        self.child_frame = str(self.get_parameter('child_frame').value)
        republish_hz = max(1.0, float(self.get_parameter('republish_hz').value))
        self.stale_timeout = max(
            0.5,
            float(self.get_parameter('stale_timeout_sec').value),
        )

        self._last_pose: PoseStamped | None = None
        self._last_pose_time = -1.0e9
        self._last_wait_log_time = -1.0e9
        self._tf = TransformBroadcaster(self)
        self._sub = self.create_subscription(PoseStamped, self.input_topic, self._pose_cb, 10)
        self.create_timer(1.0 / republish_hz, self._republish)
        self.get_logger().info(
            f'pose_to_tf ready: {self.input_topic} -> {self.parent_frame}->{self.child_frame}'
        )

    def _pose_cb(self, msg: PoseStamped) -> None:
        now = self._now()
        recovered = (
            self._last_pose is None
            or now - self._last_pose_time > self.stale_timeout
        )
        self._last_pose = msg
        self._last_pose_time = now
        self._broadcast(msg)
        if recovered:
            self.get_logger().info(
                f'POSE_TO_TF_RECOVERED | input={self.input_topic} '
                f'{self.parent_frame}->{self.child_frame}'
            )

    def _republish(self) -> None:
        now = self._now()
        if self._last_pose is None:
            self._log_wait(now, f'no messages on {self.input_topic}')
            return
        age = now - self._last_pose_time
        if age > self.stale_timeout:
            self._log_wait(
                now,
                f'{self.input_topic} stale for {age:.1f}s',
            )
            return
        self._broadcast(self._last_pose)

    def _now(self) -> float:
        return self.get_clock().now().nanoseconds * 1.0e-9

    def _log_wait(self, now: float, reason: str) -> None:
        if now - self._last_wait_log_time < 5.0:
            return
        self.get_logger().warning(
            f'POSE_TO_TF_WAIT | {reason} | expected '
            f'{self.parent_frame}->{self.child_frame}'
        )
        self._last_wait_log_time = now

    def _broadcast(self, msg: PoseStamped) -> None:
        tf = TransformStamped()
        tf.header.stamp = self.get_clock().now().to_msg()
        tf.header.frame_id = self.parent_frame
        tf.child_frame_id = self.child_frame
        tf.transform.translation.x = msg.pose.position.x
        tf.transform.translation.y = msg.pose.position.y
        tf.transform.translation.z = msg.pose.position.z
        tf.transform.rotation = msg.pose.orientation
        self._tf.sendTransform(tf)


def main() -> None:
    rclpy.init()
    node = PoseToTfBroadcaster()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, RuntimeError):
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
