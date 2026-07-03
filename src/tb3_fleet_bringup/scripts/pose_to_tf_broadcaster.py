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

        self.input_topic = str(self.get_parameter('input_topic').value)
        self.parent_frame = str(self.get_parameter('parent_frame').value)
        self.child_frame = str(self.get_parameter('child_frame').value)
        republish_hz = max(1.0, float(self.get_parameter('republish_hz').value))

        self._last_pose: PoseStamped | None = None
        self._tf = TransformBroadcaster(self)
        self._sub = self.create_subscription(PoseStamped, self.input_topic, self._pose_cb, 10)
        self.create_timer(1.0 / republish_hz, self._republish)
        self.get_logger().info(
            f'pose_to_tf ready: {self.input_topic} -> {self.parent_frame}->{self.child_frame}'
        )

    def _pose_cb(self, msg: PoseStamped) -> None:
        self._last_pose = msg
        self._broadcast(msg)

    def _republish(self) -> None:
        if self._last_pose is not None:
            self._broadcast(self._last_pose)

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
