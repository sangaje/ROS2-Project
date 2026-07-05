#!/usr/bin/env python3
"""
Follower sensor relay for Gazebo.

domain_bridge can remap topic names but not message frame IDs. Gazebo publishes
the follower as burger/* frames, so after bridging into the follower domain this
node strips burger/ from scan and odom frames for AMCL/Nav2, then republishes a
burger/base_scan copy for RViz on the leader domain.
"""

import rclpy
from copy import deepcopy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan


class SimBurgerScanRelay(Node):
    def __init__(self):
        super().__init__('sim_burger_scan_relay')
        self.declare_parameter('scan_input_topic', '/scan_bridge')
        self.declare_parameter('scan_output_topic', '/scan')
        self.declare_parameter('burger_scan_output_topic', '/burger_scan_relay')
        self.declare_parameter('odom_input_topic', '/odom_bridge')
        self.declare_parameter('odom_output_topic', '/odom')

        scan_input_topic = str(self.get_parameter('scan_input_topic').value)
        scan_output_topic = str(self.get_parameter('scan_output_topic').value)
        burger_scan_output_topic = str(self.get_parameter('burger_scan_output_topic').value)
        odom_input_topic = str(self.get_parameter('odom_input_topic').value)
        odom_output_topic = str(self.get_parameter('odom_output_topic').value)

        qos = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )
        odom_qos = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )
        self._scan_pub = self.create_publisher(LaserScan, scan_output_topic, qos)
        self._burger_scan_pub = self.create_publisher(LaserScan, burger_scan_output_topic, qos)
        self._odom_pub = self.create_publisher(Odometry, odom_output_topic, odom_qos)
        self._scan_sub = self.create_subscription(LaserScan, scan_input_topic, self._scan_cb, qos)
        self._odom_sub = self.create_subscription(Odometry, odom_input_topic, self._odom_cb, odom_qos)
        self.get_logger().info(
            f'sim_burger_sensor_relay ready: {scan_input_topic}->'
            f'{scan_output_topic}/{burger_scan_output_topic}, {odom_input_topic}->{odom_output_topic}'
        )

    @staticmethod
    def _strip_burger(frame_id: str) -> str:
        return frame_id.replace('burger/', '', 1)

    def _scan_cb(self, msg: LaserScan):
        follower_scan = deepcopy(msg)
        follower_scan.header.frame_id = self._strip_burger(msg.header.frame_id)
        self._scan_pub.publish(follower_scan)

        leader_scan = deepcopy(msg)
        leader_scan.header.frame_id = 'burger/' + self._strip_burger(msg.header.frame_id)
        self._burger_scan_pub.publish(leader_scan)

    def _odom_cb(self, msg: Odometry):
        msg = deepcopy(msg)
        msg.header.frame_id = self._strip_burger(msg.header.frame_id)
        msg.child_frame_id = self._strip_burger(msg.child_frame_id)
        self._odom_pub.publish(msg)


def main():
    rclpy.init()
    node = SimBurgerScanRelay()
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
