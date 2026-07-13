#!/usr/bin/env python3
"""Latch localization_ready for fixed-seed AMCL without moving the robot.

This node is the non-invasive counterpart to global_localize_kickstart.  It
does not publish /initialpose, /cmd_vel, global-localization requests, or Nav2
goals.  It only watches the fixed-seed AMCL pipeline and latches ready once the
map, scan, odom, AMCL pose, lifecycle state, and map->base TF are coherent.
"""

from __future__ import annotations

import math
from typing import Optional

import rclpy
from geometry_msgs.msg import PoseWithCovarianceStamped
from lifecycle_msgs.srv import GetState
from nav_msgs.msg import OccupancyGrid, Odometry
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Bool
from tf2_ros import Buffer, ExtrapolationException, LookupException, TransformListener


class AmclFixedSeedReady(Node):
    def __init__(self) -> None:
        super().__init__('amcl_fixed_seed_ready')
        self.declare_parameter('map_topic', '/map')
        self.declare_parameter('scan_topic', '/scan')
        self.declare_parameter('odom_topic', '/odom')
        self.declare_parameter('amcl_pose_topic', '/amcl_pose')
        self.declare_parameter('amcl_get_state_service', '/amcl/get_state')
        self.declare_parameter('global_frame', 'map')
        self.declare_parameter('base_frame', 'base_footprint')
        self.declare_parameter('ready_topic', 'localization_ready')
        self.declare_parameter('min_known_map_cells', 100)
        self.declare_parameter('max_scan_age_sec', 1.5)
        self.declare_parameter('max_odom_age_sec', 1.5)
        self.declare_parameter('max_amcl_pose_age_sec', 2.0)
        self.declare_parameter('tf_timeout_sec', 0.2)
        self.declare_parameter('stable_duration_sec', 1.0)
        self.declare_parameter('check_period_sec', 0.25)
        self.declare_parameter('require_amcl_lifecycle_active', True)
        self.declare_parameter('fixed_seed_initial_pose_applied', True)

        get = self.get_parameter
        self.map_topic = str(get('map_topic').value)
        self.scan_topic = str(get('scan_topic').value)
        self.odom_topic = str(get('odom_topic').value)
        self.amcl_pose_topic = str(get('amcl_pose_topic').value)
        self.amcl_get_state_service = str(get('amcl_get_state_service').value)
        self.global_frame = str(get('global_frame').value).strip().lstrip('/')
        self.base_frame = str(get('base_frame').value).strip().lstrip('/')
        self.ready_topic = str(get('ready_topic').value)
        self.min_known_map_cells = max(1, int(get('min_known_map_cells').value))
        self.max_scan_age_sec = max(0.1, float(get('max_scan_age_sec').value))
        self.max_odom_age_sec = max(0.1, float(get('max_odom_age_sec').value))
        self.max_amcl_age_sec = max(0.1, float(get('max_amcl_pose_age_sec').value))
        self.tf_timeout_sec = max(0.05, float(get('tf_timeout_sec').value))
        self.stable_duration_sec = max(0.0, float(get('stable_duration_sec').value))
        self.check_period_sec = max(0.1, float(get('check_period_sec').value))
        self.require_amcl_active = bool(get('require_amcl_lifecycle_active').value)
        self.initial_pose_applied = bool(get('fixed_seed_initial_pose_applied').value)

        latched_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
        )
        map_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
        )
        self.ready_pub = self.create_publisher(Bool, self.ready_topic, latched_qos)
        self.create_subscription(OccupancyGrid, self.map_topic, self._on_map, map_qos)
        self.create_subscription(
            LaserScan, self.scan_topic, self._on_scan, ReliabilityPolicy.BEST_EFFORT
        )
        self.create_subscription(Odometry, self.odom_topic, self._on_odom, 10)
        self.create_subscription(
            PoseWithCovarianceStamped, self.amcl_pose_topic, self._on_amcl_pose, 10
        )

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.amcl_state_client = self.create_client(GetState, self.amcl_get_state_service)

        self.map_known_cells = 0
        self.last_scan_wall: Optional[float] = None
        self.last_odom_wall: Optional[float] = None
        self.last_amcl_wall: Optional[float] = None
        self.amcl_pose_finite = False
        self.amcl_active_cached = not self.require_amcl_active
        self.amcl_state_request_pending = False
        self.good_since_wall: Optional[float] = None
        self.done = False

        self._publish_ready(False)
        self.create_timer(self.check_period_sec, self._tick)
        self.get_logger().warning(
            'AMCL_FIXED_SEED_READY_WATCHING | '
            f'map={self.map_topic} scan={self.scan_topic} odom={self.odom_topic} '
            f'amcl={self.amcl_pose_topic} tf={self.global_frame}->{self.base_frame} '
            f'initial_pose_applied={self.initial_pose_applied} out={self.ready_topic}'
        )

    def _now(self) -> float:
        return self.get_clock().now().nanoseconds * 1.0e-9

    def _on_map(self, msg: OccupancyGrid) -> None:
        self.map_known_cells = sum(1 for cell in msg.data if cell >= 0)

    def _on_scan(self, msg: LaserScan) -> None:  # noqa: ARG002
        self.last_scan_wall = self._now()

    def _on_odom(self, msg: Odometry) -> None:  # noqa: ARG002
        self.last_odom_wall = self._now()

    def _on_amcl_pose(self, msg: PoseWithCovarianceStamped) -> None:
        pose = msg.pose.pose
        quat = pose.orientation
        values = (
            pose.position.x,
            pose.position.y,
            pose.position.z,
            quat.x,
            quat.y,
            quat.z,
            quat.w,
        )
        self.amcl_pose_finite = all(math.isfinite(float(value)) for value in values)
        self.last_amcl_wall = self._now()

    def _fresh(self, last_wall: Optional[float], max_age: float) -> bool:
        return last_wall is not None and self._now() - last_wall <= max_age

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

    def _amcl_active(self) -> bool:
        if not self.require_amcl_active:
            return True
        if not self.amcl_state_request_pending and self.amcl_state_client.service_is_ready():
            self.amcl_state_request_pending = True
            future = self.amcl_state_client.call_async(GetState.Request())
            future.add_done_callback(self._on_amcl_state)
        return self.amcl_active_cached

    def _on_amcl_state(self, future) -> None:
        self.amcl_state_request_pending = False
        try:
            response = future.result()
        except Exception:  # noqa: BLE001
            self.amcl_active_cached = False
            return
        state = getattr(response.current_state, 'label', '')
        self.amcl_active_cached = str(state).lower() == 'active'

    def _publish_ready(self, ready: bool) -> None:
        self.ready_pub.publish(Bool(data=bool(ready)))

    def _tick(self) -> None:
        if self.done:
            return
        ok = (
            self.initial_pose_applied
            and self.map_known_cells >= self.min_known_map_cells
            and self._fresh(self.last_scan_wall, self.max_scan_age_sec)
            and self._fresh(self.last_odom_wall, self.max_odom_age_sec)
            and self._fresh(self.last_amcl_wall, self.max_amcl_age_sec)
            and self.amcl_pose_finite
            and self._tf_ok()
            and self._amcl_active()
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
            'AMCL_FIXED_SEED_LOCALIZATION_READY | '
            f'map_known_cells={self.map_known_cells} '
            f'scan_fresh=true odom_fresh=true amcl_fresh=true '
            f'tf={self.global_frame}->{self.base_frame} lifecycle=active'
        )


def main() -> None:
    rclpy.init()
    node = AmclFixedSeedReady()
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
