#!/usr/bin/env python3
"""
Subscribe to /burger/tf on the follower domain and republish it as /tf with the
burger/ prefix stripped so AMCL/Nav2 can use standard frame names.
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from tf2_msgs.msg import TFMessage


class SimBurgerTFRelay(Node):
    def __init__(self):
        super().__init__('sim_burger_tf_relay')
        qos = QoSProfile(
            depth=100,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )
        self._pub = self.create_publisher(TFMessage, '/tf', qos)
        self._sub = self.create_subscription(TFMessage, '/burger/tf', self._cb, qos)

    def _cb(self, msg: TFMessage):
        out = TFMessage()
        for t in msg.transforms:
            t.header.frame_id = t.header.frame_id.replace('burger/', '', 1)
            t.child_frame_id  = t.child_frame_id.replace('burger/', '', 1)
            out.transforms.append(t)
        if out.transforms:
            self._pub.publish(out)


def main():
    rclpy.init()
    node = SimBurgerTFRelay()
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
