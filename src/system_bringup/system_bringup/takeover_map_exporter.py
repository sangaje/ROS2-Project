#!/usr/bin/env python3
"""Export takeover maps while preserving the cached leader map as baseline."""

from __future__ import annotations

import math
import time
import hashlib
from copy import deepcopy
from typing import Optional

import rclpy
from nav_msgs.msg import OccupancyGrid
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy


def _map_qos(depth: int = 1) -> QoSProfile:
    return QoSProfile(
        depth=depth,
        reliability=ReliabilityPolicy.RELIABLE,
        durability=DurabilityPolicy.TRANSIENT_LOCAL,
        history=HistoryPolicy.KEEP_LAST,
    )


def _volatile_map_qos(depth: int = 5) -> QoSProfile:
    return QoSProfile(
        depth=depth,
        reliability=ReliabilityPolicy.RELIABLE,
        durability=DurabilityPolicy.VOLATILE,
        history=HistoryPolicy.KEEP_LAST,
    )


def _valid_grid(msg: OccupancyGrid) -> bool:
    width = int(msg.info.width)
    height = int(msg.info.height)
    return width > 0 and height > 0 and len(msg.data) == width * height


def _signature(msg: OccupancyGrid) -> tuple:
    info = msg.info
    data_hash = hashlib.blake2b(
        bytes((int(value) + 1) & 0xFF for value in msg.data),
        digest_size=16,
    ).hexdigest()
    return (
        int(info.width),
        int(info.height),
        round(float(info.resolution), 9),
        round(float(info.origin.position.x), 4),
        round(float(info.origin.position.y), 4),
        data_hash,
    )


class TakeoverMapExporter(Node):
    def __init__(self) -> None:
        super().__init__('takeover_map_exporter')
        self.declare_parameter('input_topic', '/map')
        self.declare_parameter('output_topic', '/local_slam_map')
        self.declare_parameter('keepalive_period_sec', 1.0)
        self.declare_parameter('merge_with_baseline', True)

        get = self.get_parameter
        self.input_topic = str(get('input_topic').value).strip() or '/map'
        self.output_topic = str(get('output_topic').value).strip() or '/local_slam_map'
        self.keepalive_period = max(0.0, float(get('keepalive_period_sec').value))
        self.merge_with_baseline = bool(get('merge_with_baseline').value)

        self.pub = self.create_publisher(OccupancyGrid, self.output_topic, _map_qos())
        self.create_subscription(
            OccupancyGrid,
            self.input_topic,
            self._on_map,
            _map_qos(depth=5),
        )
        self.create_subscription(
            OccupancyGrid,
            self.input_topic,
            self._on_map,
            _volatile_map_qos(depth=5),
        )
        self.baseline: Optional[OccupancyGrid] = None
        self.latest: Optional[OccupancyGrid] = None
        self.latest_sig: Optional[tuple] = None
        self.last_publish_wall = -1.0e9
        self.create_timer(0.2, self._tick)
        self.get_logger().warning(
            'TAKEOVER_MAP_EXPORTER_READY | '
            f'input={self.input_topic} output={self.output_topic} '
            f'merge_with_baseline={self.merge_with_baseline} '
            f'keepalive={self.keepalive_period:.2f}'
        )

    def _on_map(self, msg: OccupancyGrid) -> None:
        if not _valid_grid(msg):
            self.get_logger().warning(
                'TAKEOVER_MAP_EXPORTER_INVALID_MAP | ignoring invalid grid',
                throttle_duration_sec=5.0,
            )
            return
        if self.baseline is None:
            self.baseline = msg
            self.latest = msg
            self.latest_sig = _signature(msg)
            self.get_logger().warning(
                'TAKEOVER_MAP_BASELINE_CAPTURED | '
                f'width={msg.info.width} height={msg.info.height} '
                f'resolution={msg.info.resolution:.3f}'
            )
            self._publish(msg)
            return

        sig = _signature(msg)
        if sig == self.latest_sig:
            return
        self.latest_sig = sig
        self.latest = (
            self._merge(self.baseline, msg)
            if self.merge_with_baseline else msg
        )
        self._publish(self.latest)

    def _tick(self) -> None:
        if self.keepalive_period <= 0.0 or self.latest is None:
            return
        now = time.monotonic()
        if now - self.last_publish_wall >= self.keepalive_period:
            self._publish(self.latest)

    def _publish(self, msg: OccupancyGrid) -> None:
        out = OccupancyGrid()
        out.header = msg.header
        out.header.stamp = self.get_clock().now().to_msg()
        out.info = deepcopy(msg.info)
        out.data = list(msg.data)
        self.pub.publish(out)
        self.last_publish_wall = time.monotonic()

    def _merge(self, baseline: OccupancyGrid, current: OccupancyGrid) -> OccupancyGrid:
        resolution = float(current.info.resolution)
        base_resolution = float(baseline.info.resolution)
        if resolution <= 0.0 or not math.isclose(resolution, base_resolution, rel_tol=1e-6):
            return current

        bx0 = float(baseline.info.origin.position.x)
        by0 = float(baseline.info.origin.position.y)
        cx0 = float(current.info.origin.position.x)
        cy0 = float(current.info.origin.position.y)
        bw = int(baseline.info.width)
        bh = int(baseline.info.height)
        cw = int(current.info.width)
        ch = int(current.info.height)
        x0 = min(bx0, cx0)
        y0 = min(by0, cy0)
        x1 = max(bx0 + bw * resolution, cx0 + cw * resolution)
        y1 = max(by0 + bh * resolution, cy0 + ch * resolution)
        width = max(1, int(math.ceil((x1 - x0) / resolution)))
        height = max(1, int(math.ceil((y1 - y0) / resolution)))
        data = [-1] * (width * height)

        def paste(src: OccupancyGrid, *, prefer_known: bool) -> None:
            sx0 = float(src.info.origin.position.x)
            sy0 = float(src.info.origin.position.y)
            sw = int(src.info.width)
            sh = int(src.info.height)
            ox = int(round((sx0 - x0) / resolution))
            oy = int(round((sy0 - y0) / resolution))
            for sy in range(sh):
                for sx in range(sw):
                    value = int(src.data[sy * sw + sx])
                    if value < 0:
                        continue
                    dx = ox + sx
                    dy = oy + sy
                    if dx < 0 or dy < 0 or dx >= width or dy >= height:
                        continue
                    idx = dy * width + dx
                    if prefer_known or data[idx] < 0:
                        data[idx] = value

        paste(baseline, prefer_known=False)
        paste(current, prefer_known=True)

        out = OccupancyGrid()
        out.header = current.header
        out.info = deepcopy(current.info)
        out.info.width = width
        out.info.height = height
        out.info.origin.position.x = x0
        out.info.origin.position.y = y0
        out.data = data
        return out


def main(args=None) -> None:
    rclpy.init(args=args)
    node = TakeoverMapExporter()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
