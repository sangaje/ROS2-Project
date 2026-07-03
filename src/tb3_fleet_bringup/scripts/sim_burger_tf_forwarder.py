#!/usr/bin/env python3
"""
Forwards /burger/tf, published by the Gazebo follower bridge on the leader
domain, to /tf while keeping all burger/-prefixed frame IDs intact.

This makes burger/odom -> burger/base_footprint visible in the leader-domain TF
tree so the static TF map -> burger/odom and this dynamic TF give RViz a
complete map -> burger/odom -> burger/base_footprint -> burger/base_scan chain.
"""
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from tf2_msgs.msg import TFMessage


class SimBurgerTfForwarder(Node):
    def __init__(self):
        super().__init__('sim_burger_tf_forwarder')
        qos = QoSProfile(
            depth=100,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )
        self._pub = self.create_publisher(TFMessage, '/tf', qos)
        self._sub = self.create_subscription(TFMessage, '/burger/tf', self._cb, qos)
        self.get_logger().info('sim_burger_tf_forwarder ready: /burger/tf -> /tf')

    def _cb(self, msg: TFMessage):
        if msg.transforms:
            self._pub.publish(msg)


def main():
    rclpy.init()
    node = SimBurgerTfForwarder()
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
