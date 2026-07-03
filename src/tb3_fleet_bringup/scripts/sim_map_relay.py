#!/usr/bin/env python3
"""
Receives /map_bridge (volatile, from domain_bridge) and republishes as /map
with transient_local durability so AMCL and costmaps can latch it.

domain_bridge has a known issue where transient_local topics are not reliably
re-latched on the target domain. This relay acts as a proper transient_local
publisher so late subscribers (AMCL, costmap) always get the latest map.
"""
import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from nav_msgs.msg import OccupancyGrid


class SimMapRelay(Node):
    def __init__(self):
        super().__init__('sim_map_relay')
        self.declare_parameter('input_topic', '/map_bridge')
        self.declare_parameter('output_topic', '/map')

        input_topic = str(self.get_parameter('input_topic').value)
        output_topic = str(self.get_parameter('output_topic').value)

        pub_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
        )
        sub_qos = QoSProfile(
            depth=5,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
        )
        self._pub = self.create_publisher(OccupancyGrid, output_topic, pub_qos)
        self._sub = self.create_subscription(OccupancyGrid, input_topic, self._cb, sub_qos)
        self.get_logger().info(
            f'map relay ready: {input_topic} (volatile) -> {output_topic} (transient_local)'
        )

    def _cb(self, msg: OccupancyGrid):
        self._pub.publish(msg)
        self.get_logger().info(
            f'Map relayed: {msg.info.width}x{msg.info.height} @ {msg.info.resolution:.3f}m/cell',
            throttle_duration_sec=10.0,
        )


def main():
    rclpy.init()
    node = SimMapRelay()
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
