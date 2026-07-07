#!/usr/bin/env python3
"""Fake Static Map - Burger SLAM map 시뮬 publisher.

Burger 가 SLAM 으로 만든 occupancy map 시뮬용. /scout/map 으로 발행.
fake_risk_map.py 와 같은 grid 사양 (origin, resolution) 으로 맞춤.

토픽:
    Pub: /scout/map  nav_msgs/OccupancyGrid  (transient_local)

좌표 의미 (OccupancyGrid):
    0    = free (자유공간)
    100  = occupied (장애물)
    -1   = unknown (미관측)

기본 설계:
    10m x 10m 방 (100x100 cells @ 0.1m/cell)
    origin: (-5, -5)
    외곽: 두께 0.2m (2 cell) 벽
    내부: 박스 장애물 2개 (hotspot 과 겹치지 않게)
    
파라미터:
    topic, map_frame, width, height, resolution, origin_x, origin_y,
    publish_rate
"""

from __future__ import annotations

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy

from nav_msgs.msg import OccupancyGrid


# 내부 장애물: (x_min, y_min, x_max, y_max) in world (m)
# hotspot 위치 피해서 배치
DEFAULT_OBSTACLES = [
    # 가운데 작은 박스
    (-0.5, -0.5, 0.5, 0.5),
    # 좌상단 (hotspot -2.0, +1.5 와 -3.0, -2.0 사이)
    (-3.5, 0.0, -2.5, 0.5),
    # 우하단 (hotspot +1.0, -2.5 와 +3.0, -0.5 사이)
    (+1.8, -1.5, +2.5, -0.8),
]


class FakeStaticMap(Node):
    def __init__(self):
        super().__init__('fake_static_map')

        self.declare_parameter('topic', '/scout/map')
        self.declare_parameter('map_frame', 'map')
        self.declare_parameter('width', 100)
        self.declare_parameter('height', 100)
        self.declare_parameter('resolution', 0.1)
        self.declare_parameter('origin_x', -5.0)
        self.declare_parameter('origin_y', -5.0)
        self.declare_parameter('publish_rate', 1.0)
        self.declare_parameter('wall_thickness_cells', 2)

        self.topic = self.get_parameter('topic').value
        self.map_frame = self.get_parameter('map_frame').value
        self.width = int(self.get_parameter('width').value)
        self.height = int(self.get_parameter('height').value)
        self.resolution = float(self.get_parameter('resolution').value)
        self.origin_x = float(self.get_parameter('origin_x').value)
        self.origin_y = float(self.get_parameter('origin_y').value)
        self.publish_rate = float(self.get_parameter('publish_rate').value)
        self.wall_thickness = int(self.get_parameter(
            'wall_thickness_cells').value)

        # latched QoS
        qos = QoSProfile(
            depth=1,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.pub = self.create_publisher(OccupancyGrid, self.topic, qos)

        self.data = self._build_map(DEFAULT_OBSTACLES)
        self.publish_count = 0

        period = 1.0 / max(0.01, self.publish_rate)
        self.create_timer(period, self.on_publish)

        # Stats
        occ = sum(1 for v in self.data if v == 100)
        free = sum(1 for v in self.data if v == 0)
        self.get_logger().info("=" * 50)
        self.get_logger().info(f"Fake Static Map -> {self.topic}")
        self.get_logger().info(
            f"Grid: {self.width}x{self.height} @ {self.resolution}m/cell")
        self.get_logger().info(
            f"World: {self.width * self.resolution}m x "
            f"{self.height * self.resolution}m, "
            f"origin=({self.origin_x:+.1f}, {self.origin_y:+.1f})")
        self.get_logger().info(
            f"Cells: {occ} occupied, {free} free")
        self.get_logger().info(f"Obstacles (xy bounds, world m):")
        for x1, y1, x2, y2 in DEFAULT_OBSTACLES:
            self.get_logger().info(
                f"  ({x1:+.1f},{y1:+.1f}) ~ ({x2:+.1f},{y2:+.1f})")
        self.get_logger().info(f"Rate: {self.publish_rate}Hz")
        self.get_logger().info("=== Ready ===")

    def _build_map(self, obstacles):
        w = self.width
        h = self.height
        res = self.resolution
        ox = self.origin_x
        oy = self.origin_y
        wt = self.wall_thickness

        # 전부 free (0) 으로 시작
        data = [0] * (w * h)

        # 외곽 벽 (두께 wt)
        for gy in range(h):
            for gx in range(w):
                if (gx < wt or gx >= w - wt
                        or gy < wt or gy >= h - wt):
                    data[gy * w + gx] = 100

        # 내부 장애물 (rectangle)
        for x1, y1, x2, y2 in obstacles:
            # world → grid 변환
            gx_min = max(0, int((x1 - ox) / res))
            gy_min = max(0, int((y1 - oy) / res))
            gx_max = min(w, int((x2 - ox) / res) + 1)
            gy_max = min(h, int((y2 - oy) / res) + 1)
            for gy in range(gy_min, gy_max):
                for gx in range(gx_min, gx_max):
                    data[gy * w + gx] = 100

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
        node = FakeStaticMap()
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