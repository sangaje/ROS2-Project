#!/usr/bin/env python3
"""Map Relay - /scout/map -> /map.

Burger 가 SLAM 한 맵을 Nav2 가 기대하는 /map 토픽으로 그대로 forward.
QoS 는 transient_local (latched) 으로 설정해서 late joiner 도 받게.

토픽:
    Subscribe:  /scout/map  nav_msgs/OccupancyGrid
    Publish:    /map        nav_msgs/OccupancyGrid

파라미터:
    input_topic:  default '/scout/map'
    output_topic: default '/map'
"""

from __future__ import annotations

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy

from nav_msgs.msg import OccupancyGrid


class MapRelay(Node):
    def __init__(self):
        super().__init__('map_relay')

        self.declare_parameter('input_topic', '/scout/map')
        self.declare_parameter('output_topic', '/map')

        in_topic = self.get_parameter('input_topic').value
        out_topic = self.get_parameter('output_topic').value

        # transient_local QoS (map 은 latched 토픽 표준)
        latched_qos = QoSProfile(
            depth=1,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
        )

        self.pub = self.create_publisher(OccupancyGrid, out_topic, latched_qos)
        self.create_subscription(
            OccupancyGrid, in_topic, self.on_map, latched_qos)

        self.relay_count = 0
        self.last_log_t = 0.0

        self.get_logger().info("=" * 50)
        self.get_logger().info(f"Map Relay: {in_topic}  ->  {out_topic}")
        self.get_logger().info("QoS: transient_local (latched)")
        self.get_logger().info("=" * 50)

    def on_map(self, msg: OccupancyGrid):
        self.pub.publish(msg)
        self.relay_count += 1

        # 로그 5초 간격
        now = self.get_clock().now().nanoseconds * 1e-9
        if now - self.last_log_t > 5.0:
            info = msg.info
            self.get_logger().info(
                f"Map relayed #{self.relay_count}: "
                f"{info.width}x{info.height} @ {info.resolution:.3f}m/cell, "
                f"origin=({info.origin.position.x:+.2f}, "
                f"{info.origin.position.y:+.2f}), frame={msg.header.frame_id}")
            self.last_log_t = now


def main():
    rclpy.init()
    node = None
    try:
        node = MapRelay()
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("\n중단됨.")
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()