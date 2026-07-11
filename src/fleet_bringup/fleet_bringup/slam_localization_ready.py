#!/usr/bin/env python3
"""Publish localization_ready for a robot whose SLAM/TF owner is Cartographer.

global_localize_kickstart's whole state machine is AMCL-specific (spin to
help AMCL's particle filter converge, watch AMCL covariance, call
/reinitialize_global_localization). When a robot's map->odom->base_footprint
chain is instead owned directly by Cartographer -- e.g. a leader launched
with enable_cartographer:=true, or any other role that owns its own SLAM --
none of that applies and, critically, nothing in that path ever publishes to
ready_topic. A downstream consumer with require_bootstrap_complete=true
(e.g. scout_failover_coordinator's bootstrap gate) then waits forever.

This node has one job: watch for /map + a valid map->base_footprint TF +
fresh /scan continuously for stable_duration_sec, then latch ready_topic
true once, matching global_localize_kickstart's ready_topic contract so
downstream consumers do not need to know which of the two produced it.
"""

from __future__ import annotations

from typing import Optional

import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from nav_msgs.msg import OccupancyGrid
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Bool
from tf2_ros import Buffer, TransformListener, LookupException, ExtrapolationException


class SlamLocalizationReady(Node):
    def __init__(self) -> None:
        super().__init__('slam_localization_ready')
        self.declare_parameter('map_topic', '/map')
        self.declare_parameter('scan_topic', '/scan')
        self.declare_parameter('global_frame', 'map')
        self.declare_parameter('base_frame', 'base_footprint')
        self.declare_parameter('ready_topic', 'localization_ready')
        self.declare_parameter('min_known_map_cells', 100)
        self.declare_parameter('max_scan_age_sec', 2.0)
        self.declare_parameter('tf_timeout_sec', 0.2)
        self.declare_parameter('stable_duration_sec', 2.0)
        self.declare_parameter('check_period_sec', 0.5)

        get = self.get_parameter
        self.map_topic = str(get('map_topic').value)
        self.scan_topic = str(get('scan_topic').value)
        self.global_frame = str(get('global_frame').value).strip().lstrip('/')
        self.base_frame = str(get('base_frame').value).strip().lstrip('/')
        self.ready_topic = str(get('ready_topic').value)
        self.min_known_map_cells = max(1, int(get('min_known_map_cells').value))
        self.max_scan_age_sec = max(0.1, float(get('max_scan_age_sec').value))
        self.tf_timeout_sec = max(0.05, float(get('tf_timeout_sec').value))
        self.stable_duration_sec = max(0.0, float(get('stable_duration_sec').value))
        self.check_period_sec = max(0.1, float(get('check_period_sec').value))

        latched_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
        )
        self.ready_pub = self.create_publisher(Bool, self.ready_topic, latched_qos)

        self.map_known_cells = 0
        self.last_scan_wall: Optional[float] = None
        self.good_since_wall: Optional[float] = None
        self.done = False

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        map_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
        )
        self.create_subscription(OccupancyGrid, self.map_topic, self._on_map, map_qos)
        self.create_subscription(
            LaserScan, self.scan_topic, self._on_scan, ReliabilityPolicy.BEST_EFFORT
        )

        self._publish_ready(False)
        self.create_timer(self.check_period_sec, self._tick)

        self.get_logger().info(
            'SLAM_LOCALIZATION_READY_WATCHING | '
            f'map={self.map_topic} scan={self.scan_topic} '
            f'tf={self.global_frame}->{self.base_frame} '
            f'out={self.ready_topic}'
        )

    def _now(self) -> float:
        return self.get_clock().now().nanoseconds * 1.0e-9

    def _on_map(self, msg: OccupancyGrid) -> None:
        self.map_known_cells = sum(1 for cell in msg.data if cell >= 0)

    def _on_scan(self, msg: LaserScan) -> None:  # noqa: ARG002
        self.last_scan_wall = self._now()

    def _scan_fresh(self) -> bool:
        if self.last_scan_wall is None:
            return False
        return self._now() - self.last_scan_wall <= self.max_scan_age_sec

    def _tf_ok(self) -> bool:
        try:
            self.tf_buffer.lookup_transform(
                self.global_frame,
                self.base_frame,
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=self.tf_timeout_sec),
            )
            return True
        except (LookupException, ExtrapolationException, Exception):  # noqa: BLE001
            return False

    def _publish_ready(self, ready: bool) -> None:
        self.ready_pub.publish(Bool(data=ready))

    def _tick(self) -> None:
        if self.done:
            return
        ok = (
            self.map_known_cells >= self.min_known_map_cells
            and self._scan_fresh()
            and self._tf_ok()
        )
        if not ok:
            self.good_since_wall = None
            return
        if self.good_since_wall is None:
            self.good_since_wall = self._now()
        if self._now() - self.good_since_wall < self.stable_duration_sec:
            return
        self.done = True
        self._publish_ready(True)
        self.get_logger().warning(
            'SLAM_LOCALIZATION_READY | '
            f'map_known_cells={self.map_known_cells} '
            f'tf={self.global_frame}->{self.base_frame} scan_fresh=true'
        )


def main() -> None:
    rclpy.init()
    node = SlamLocalizationReady()
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
