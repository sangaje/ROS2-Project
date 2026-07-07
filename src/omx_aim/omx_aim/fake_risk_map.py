#!/usr/bin/env python3
"""Fake Risk Map - patrol_planner 검증용 시뮬레이션 publisher.

여러 가우시안 hotspot 으로 합성한 가짜 위험도 맵을 발행한다.
실제 Burger 가 도착하기 전 patrol_planner 알고리즘 검증용.

토픽:
    Pub: /scout/risk_map  nav_msgs/OccupancyGrid  (transient_local)

파라미터:
    topic:        default '/scout/risk_map'
    map_frame:    default 'map'
    width:        100 (cells)
    height:       100
    resolution:   0.1 m/cell  (실제 크기 10m x 10m)
    origin_x:     -5.0
    origin_y:     -5.0
    publish_rate: 1.0 Hz
    hotspots:     hardcoded (시뮬 시나리오)
"""

from __future__ import annotations

import math
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy

from nav_msgs.msg import OccupancyGrid


# 시뮬 시나리오: (x_world, y_world, peak_risk, sigma_m)
DEFAULT_HOTSPOTS = [
    (+2.5, +2.0, 90, 0.6),   # 강한 hotspot
    (-2.0, +1.5, 70, 0.5),   # 중간
    (+1.0, -2.5, 85, 0.4),   # 좁고 강함
    (-3.0, -2.0, 50, 0.8),   # 넓고 약함
    (+3.0, -0.5, 60, 0.3),   # 좁고 중간
]


class FakeRiskMap(Node):
    def __init__(self):
        super().__init__('fake_risk_map')

        self.declare_parameter('topic', '/scout/risk_map')
        self.declare_parameter('map_frame', 'map')
        self.declare_parameter('width', 100)
        self.declare_parameter('height', 100)
        self.declare_parameter('resolution', 0.1)
        self.declare_parameter('origin_x', -5.0)
        self.declare_parameter('origin_y', -5.0)
        self.declare_parameter('publish_rate', 1.0)

        self.topic = self.get_parameter('topic').value
        self.map_frame = self.get_parameter('map_frame').value
        self.width = int(self.get_parameter('width').value)
        self.height = int(self.get_parameter('height').value)
        self.resolution = float(self.get_parameter('resolution').value)
        self.origin_x = float(self.get_parameter('origin_x').value)
        self.origin_y = float(self.get_parameter('origin_y').value)
        self.publish_rate = float(self.get_parameter('publish_rate').value)

        # latched QoS
        qos = QoSProfile(
            depth=1,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.pub = self.create_publisher(OccupancyGrid, self.topic, qos)

        self.data = self._build_risk_grid(DEFAULT_HOTSPOTS)
        self.publish_count = 0

        period = 1.0 / max(0.01, self.publish_rate)
        self.create_timer(period, self.on_publish)

        self.get_logger().info("=" * 50)
        self.get_logger().info(f"Fake Risk Map -> {self.topic}")
        self.get_logger().info(
            f"Grid: {self.width}x{self.height} @ {self.resolution}m/cell, "
            f"origin=({self.origin_x:+.1f}, {self.origin_y:+.1f})")
        self.get_logger().info(
            f"World: x=[{self.origin_x:+.1f}, "
            f"{self.origin_x + self.width * self.resolution:+.1f}], "
            f"y=[{self.origin_y:+.1f}, "
            f"{self.origin_y + self.height * self.resolution:+.1f}]")
        self.get_logger().info("Hotspots (world coords):")
        for x, y, peak, sigma in DEFAULT_HOTSPOTS:
            self.get_logger().info(
                f"  ({x:+.1f}, {y:+.1f}) peak={peak} sigma={sigma}m")
        self.get_logger().info(f"Rate: {self.publish_rate}Hz")
        self.get_logger().info("=== Ready ===")

    def _build_risk_grid(self, hotspots):
        """가우시안 합성으로 risk grid 생성."""
        w = self.width
        h = self.height
        res = self.resolution
        ox = self.origin_x
        oy = self.origin_y

        data = [0] * (w * h)
        for gy in range(h):
            wy = oy + (gy + 0.5) * res
            for gx in range(w):
                wx = ox + (gx + 0.5) * res
                total = 0.0
                for hx, hy, peak, sigma in hotspots:
                    dx = wx - hx
                    dy = wy - hy
                    d2 = dx*dx + dy*dy
                    total += peak * math.exp(-d2 / (2.0 * sigma * sigma))
                v = int(round(min(100.0, total)))
                data[gy * w + gx] = v
        return data

    def on_publish(self):
        msg = OccupancyGrid()
        msg.header.frame_id = self.map_frame
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.info.resolution = self.resolution
        msg.info.width = self.width
        msg.info.height = self.height
        msg.info.origin.position.x = self.origin_x
        msg.info.origin.position.y = self.origin_y
        msg.info.origin.position.z = 0.0
        msg.info.origin.orientation.w = 1.0
        msg.data = self.data

        self.pub.publish(msg)
        self.publish_count += 1
        if self.publish_count == 1 or self.publish_count % 10 == 0:
            self.get_logger().info(f"발행 #{self.publish_count}")


def main():
    rclpy.init()
    node = None
    try:
        node = FakeRiskMap()
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