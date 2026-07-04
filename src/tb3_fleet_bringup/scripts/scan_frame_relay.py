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
        self.declare_parameter('input_reliability', 'best_effort')
        self.declare_parameter('output_reliability', 'best_effort')

        input_topic = str(self.get_parameter('input_topic').value)
        output_topic = str(self.get_parameter('output_topic').value)
        self.output_frame = str(self.get_parameter('output_frame').value)
        input_reliability = str(self.get_parameter('input_reliability').value)
        output_reliability = str(self.get_parameter('output_reliability').value)

        input_qos = QoSProfile(
            depth=10,
            reliability=self._reliability(input_reliability),
            durability=DurabilityPolicy.VOLATILE,
        )
        output_qos = QoSProfile(
            depth=10,
            reliability=self._reliability(output_reliability),
            durability=DurabilityPolicy.VOLATILE,
        )
        self.pub = self.create_publisher(LaserScan, output_topic, output_qos)
        self.sub = self.create_subscription(LaserScan, input_topic, self._scan_cb, input_qos)
        self.get_logger().info(
            f'scan_frame_relay ready: {input_topic}({input_reliability}) -> '
            f'{output_topic}({output_reliability}) frame={self.output_frame}'
        )

    @staticmethod
    def _reliability(value: str):
        if value.strip().lower() in ('reliable', 'r'):
            return ReliabilityPolicy.RELIABLE
        return ReliabilityPolicy.BEST_EFFORT

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
