#!/usr/bin/env python3

from copy import deepcopy

import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import LaserScan


class ScanFrameRelay(Node):
    def __init__(self):
        super().__init__('scan_frame_relay')
        self.declare_parameter('input_topic', '/scan')
        self.declare_parameter('output_topic', '/burger_scan_relay')
        self.declare_parameter('output_frame', 'burger/base_scan')

        input_topic = str(self.get_parameter('input_topic').value)
        output_topic = str(self.get_parameter('output_topic').value)
        self.output_frame = str(self.get_parameter('output_frame').value)

        qos = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )
        self.pub = self.create_publisher(LaserScan, output_topic, qos)
        self.sub = self.create_subscription(LaserScan, input_topic, self._scan_cb, qos)
        self.get_logger().info(
            f'scan_frame_relay ready: {input_topic} -> {output_topic} frame={self.output_frame}'
        )

    def _scan_cb(self, msg: LaserScan):
        out = deepcopy(msg)
        out.header.frame_id = self.output_frame
        self.pub.publish(out)


def main():
    rclpy.init()
    node = ScanFrameRelay()
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
