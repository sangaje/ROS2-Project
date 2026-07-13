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
        self.declare_parameter('odom_frame', 'odom')
        self.declare_parameter('base_frame', 'base_footprint')
        self.declare_parameter('ready_topic', '/localization_ready')
        self.declare_parameter('min_known_map_cells', 100)
        self.declare_parameter('max_scan_age_sec', 1.5)
        self.declare_parameter('max_odom_age_sec', 1.5)
        self.declare_parameter('max_amcl_pose_age_sec', 2.0)
        self.declare_parameter('tf_timeout_sec', 0.2)
        self.declare_parameter('stable_duration_sec', 1.0)
        self.declare_parameter('check_period_sec', 0.25)
        self.declare_parameter('max_xy_covariance', 0.22)
        self.declare_parameter('max_yaw_covariance', 0.16)
        self.declare_parameter('require_amcl_lifecycle_active', True)
        self.declare_parameter('fixed_seed_initial_pose_applied', True)

        get = self.get_parameter
        self.map_topic = str(get('map_topic').value)
        self.scan_topic = str(get('scan_topic').value)
        self.odom_topic = str(get('odom_topic').value)
        self.amcl_pose_topic = str(get('amcl_pose_topic').value)
        self.amcl_get_state_service = str(get('amcl_get_state_service').value)
        self.global_frame = str(get('global_frame').value).strip().lstrip('/')
        self.odom_frame = str(get('odom_frame').value).strip().lstrip('/')
        self.base_frame = str(get('base_frame').value).strip().lstrip('/')
        self.ready_topic = str(get('ready_topic').value)
        self.min_known_map_cells = max(1, int(get('min_known_map_cells').value))
        self.max_scan_age_sec = max(0.1, float(get('max_scan_age_sec').value))
        self.max_odom_age_sec = max(0.1, float(get('max_odom_age_sec').value))
        self.max_amcl_age_sec = max(0.1, float(get('max_amcl_pose_age_sec').value))
        self.tf_timeout_sec = max(0.05, float(get('tf_timeout_sec').value))
        self.stable_duration_sec = max(0.0, float(get('stable_duration_sec').value))
        self.check_period_sec = max(0.1, float(get('check_period_sec').value))
        self.max_xy_cov = max(0.0, float(get('max_xy_covariance').value))
        self.max_yaw_cov = max(0.0, float(get('max_yaw_covariance').value))
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
        scan_qos = QoSProfile(
            depth=5,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
        )
        self.ready_pub = self.create_publisher(Bool, self.ready_topic, latched_qos)
        self.create_subscription(OccupancyGrid, self.map_topic, self._on_map, map_qos)
        self.scan_sub = self.create_subscription(
            LaserScan, self.scan_topic, self._on_scan, scan_qos
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
        self.xy_cov = float('inf')
        self.yaw_cov = float('inf')
        self.amcl_active_cached = not self.require_amcl_active
        self.amcl_state_label = (
            'not_required' if not self.require_amcl_active else 'unknown'
        )
        self.amcl_state_request_pending = False
        self.good_since_wall: Optional[float] = None
        self.done = False
        self.last_map_wall: Optional[float] = None
        self.map_valid = False
        self.last_scan_frame = ''
        self.last_scan_stamp_sec = -1.0
        self.last_scan_ranges = 0
        self.last_scan_finite_ranges = 0
        self.last_scan_range_min = float('nan')
        self.last_scan_range_max = float('nan')
        self.last_scan_source_age_ms = -1.0

        self._publish_ready(False)
        self.create_timer(self.check_period_sec, self._tick)
        self.get_logger().warning(
            'AMCL_FIXED_SEED_READY_WATCHING | '
            f'map={self.map_topic} scan={self.scan_topic} odom={self.odom_topic} '
            f'amcl={self.amcl_pose_topic} '
            f'tf={self.global_frame}->{self.odom_frame}->{self.base_frame} '
            f'initial_pose_applied={self.initial_pose_applied} out={self.ready_topic}'
        )

    def _now(self) -> float:
        return self.get_clock().now().nanoseconds * 1.0e-9

    def _on_map(self, msg: OccupancyGrid) -> None:
        width = int(msg.info.width)
        height = int(msg.info.height)
        self.map_valid = (
            width > 0
            and height > 0
            and float(msg.info.resolution) > 0.0
            and len(msg.data) == width * height
        )
        self.map_known_cells = sum(1 for cell in msg.data if cell >= 0)
        self.last_map_wall = self._now()

    def _on_scan(self, msg: LaserScan) -> None:
        now = self._now()
        self.last_scan_wall = now
        stamp = msg.header.stamp
        self.last_scan_stamp_sec = float(stamp.sec) + float(stamp.nanosec) * 1.0e-9
        self.last_scan_frame = str(msg.header.frame_id or '').strip().lstrip('/')
        self.last_scan_ranges = len(msg.ranges)
        finite = [
            float(value)
            for value in msg.ranges
            if math.isfinite(float(value))
        ]
        self.last_scan_finite_ranges = len(finite)
        self.last_scan_range_min = min(finite) if finite else float('nan')
        self.last_scan_range_max = max(finite) if finite else float('nan')
        self.last_scan_source_age_ms = (
            max(0.0, (now - self.last_scan_stamp_sec) * 1000.0)
            if self.last_scan_stamp_sec > 0.0
            else -1.0
        )

    def _on_odom(self, msg: Odometry) -> None:  # noqa: ARG002
        self.last_odom_wall = self._now()

    def _on_amcl_pose(self, msg: PoseWithCovarianceStamped) -> None:
        pose = msg.pose.pose
        cov = msg.pose.covariance
        self.xy_cov = max(abs(float(cov[0])), abs(float(cov[7])))
        self.yaw_cov = abs(float(cov[35]))
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

    def _tf_ok(self, target: str, source: str) -> bool:
        return self._tf_status(target, source)[0]

    def _tf_status(self, target: str, source: str) -> tuple[bool, float]:
        try:
            transform = self.tf_buffer.lookup_transform(
                target,
                source,
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=self.tf_timeout_sec),
            )
            stamp = transform.header.stamp
            stamp_sec = float(stamp.sec) + float(stamp.nanosec) * 1.0e-9
            age_ms = -1.0
            if stamp_sec > 0.0:
                age_ms = max(0.0, (self._now() - stamp_sec) * 1000.0)
            return True, age_ms
        except (LookupException, ExtrapolationException, Exception):  # noqa: BLE001
            return False, -1.0

    def _age(self, last_wall: Optional[float]) -> float:
        if last_wall is None:
            return -1.0
        return max(0.0, self._now() - last_wall)

    def _readiness_state(self) -> tuple[bool, str, dict]:
        scan_fresh = self._fresh(self.last_scan_wall, self.max_scan_age_sec)
        scan_rx = self.last_scan_wall is not None
        scan_nonempty = scan_rx and self.last_scan_ranges > 0 and self.last_scan_finite_ranges > 0
        scan_frame_ok = bool(self.last_scan_frame)
        scan_stamp_ok = (
            not scan_rx
            or self.last_scan_stamp_sec <= 0.0
            or self.last_scan_source_age_ms <= max(self.max_scan_age_sec * 1000.0, 5000.0)
        )
        if not scan_rx or not scan_frame_ok:
            scan_tf_ok = False
            scan_tf_age_ms = -1.0
        elif self.last_scan_frame == self.base_frame:
            scan_tf_ok = True
            scan_tf_age_ms = 0.0
        else:
            scan_tf_ok, scan_tf_age_ms = self._tf_status(self.base_frame, self.last_scan_frame)
        odom_fresh = self._fresh(self.last_odom_wall, self.max_odom_age_sec)
        cov_ok = self.xy_cov <= self.max_xy_cov and self.yaw_cov <= self.max_yaw_cov
        map_odom_tf, map_odom_age_ms = self._tf_status(
            self.global_frame,
            self.odom_frame,
        )
        # Nav2's AMCL only republishes /amcl_pose after the robot moves past
        # update_min_d/update_min_a -- while stationary (the normal state
        # right after a fixed-seed initial pose, before start_motion is
        # released) it legitimately never republishes again, so amcl_pose_age
        # grows without bound even though localization itself is fine. AMCL
        # still rebroadcasts map->odom TF every filter cycle regardless of
        # motion, so a fresh TF is direct evidence the last received pose and
        # covariance are still authoritative -- accept that as fresh too,
        # instead of waiting forever for a pose republish that requires
        # motion this same readiness gate is blocking.
        amcl_topic_fresh = self._fresh(self.last_amcl_wall, self.max_amcl_age_sec)
        amcl_pose_received = self.last_amcl_wall is not None
        amcl_fresh = amcl_topic_fresh or (amcl_pose_received and map_odom_tf)
        odom_base_tf, odom_base_age_ms = self._tf_status(
            self.odom_frame,
            self.base_frame,
        )
        lifecycle_active = self._amcl_active()
        checks = {
            'initial_pose_applied': self.initial_pose_applied,
            'map_valid': self.map_valid,
            'map_ready': self.map_known_cells >= self.min_known_map_cells,
            'scan_rx': scan_rx,
            'scan_fresh': scan_fresh,
            'scan_nonempty': scan_nonempty,
            'scan_frame_ok': scan_frame_ok,
            'scan_stamp_ok': scan_stamp_ok,
            'scan_tf_ok': scan_tf_ok,
            'odom_fresh': odom_fresh,
            'amcl_pose_fresh': amcl_fresh,
            'amcl_pose_finite': self.amcl_pose_finite,
            'covariance_ok': cov_ok,
            'map_odom_tf': map_odom_tf,
            'odom_base_tf': odom_base_tf,
            'amcl_lifecycle_active': lifecycle_active,
            '_map_odom_tf_age_ms': map_odom_age_ms,
            '_odom_base_tf_age_ms': odom_base_age_ms,
            '_scan_tf_age_ms': scan_tf_age_ms,
        }
        reason = self._blocking_reason(checks)
        return all(value for key, value in checks.items() if not key.startswith('_')), reason, checks

    def _blocking_reason(self, checks: dict) -> str:
        if not checks['initial_pose_applied']:
            return 'initial_pose_missing'
        if not checks['map_valid']:
            return 'map_invalid'
        if not checks['map_ready']:
            return 'map_not_ready'
        if not checks['scan_rx']:
            return 'scan_missing'
        if not checks['scan_fresh']:
            return 'scan_stale'
        if not checks['scan_nonempty']:
            return 'scan_empty'
        if not checks['scan_frame_ok']:
            return 'scan_frame_missing'
        if not checks['scan_stamp_ok']:
            return 'scan_timestamp_out_of_range'
        if not checks['scan_tf_ok']:
            return 'scan_tf_unavailable'
        if not checks['odom_fresh']:
            return 'odom_stale'
        if not checks['amcl_pose_fresh']:
            return 'amcl_pose_stale'
        if not checks['amcl_pose_finite']:
            return 'amcl_pose_invalid'
        if not checks['covariance_ok']:
            return 'covariance_unstable'
        if not checks['map_odom_tf']:
            return 'map_odom_tf_unavailable'
        if not checks['odom_base_tf']:
            return 'odom_base_tf_unavailable'
        if not checks['amcl_lifecycle_active']:
            return 'amcl_lifecycle_inactive'
        return 'none'

    def _log_localization_debug(self, *, ready: bool, reason: str, checks: dict) -> None:
        self._log_scan_debug(checks)
        self.get_logger().warning(
            'LEADER_LOCALIZATION_DEBUG | '
            f'mode=amcl '
            f'map_rx={self.last_map_wall is not None} '
            f'map_age_ms={self._age(self.last_map_wall) * 1000.0:.0f} '
            f'map_valid={self.map_valid} '
            f'scan_rx={self.last_scan_wall is not None} '
            f'map_known_cells={self.map_known_cells} '
            f'map_ready={checks["map_ready"]} '
            f'scan_age_ms={self._age(self.last_scan_wall) * 1000.0:.0f} '
            f'odom_rx={self.last_odom_wall is not None} '
            f'odom_age_ms={self._age(self.last_odom_wall) * 1000.0:.0f} '
            f'amcl_process_alive={self.amcl_state_client.service_is_ready() or self.last_amcl_wall is not None} '
            f'amcl_lifecycle_state={self.amcl_state_label} '
            f'amcl_pose_rx={self.last_amcl_wall is not None} '
            f'amcl_pose_age_ms={self._age(self.last_amcl_wall) * 1000.0:.0f} '
            f'cov_xy={self.xy_cov:.4f} cov_yaw={self.yaw_cov:.4f} '
            f'covariance_ok={checks["covariance_ok"]} '
            f'map_odom_tf={checks["map_odom_tf"]} '
            f'odom_base_tf={checks["odom_base_tf"]} '
            f'tf_age_ms={max(checks["_map_odom_tf_age_ms"], checks["_odom_base_tf_age_ms"]):.0f} '
            f'readiness_publisher_count={self.count_publishers(self.ready_topic)} '
            f'localization_ready={ready} '
            f'blocking_reason={reason}',
            throttle_duration_sec=2.0,
        )

    def _resolved_topic(self, topic: str) -> str:
        try:
            return self.resolve_topic_name(topic)
        except Exception:  # noqa: BLE001
            return topic

    def _log_scan_debug(self, checks: dict) -> None:
        self.get_logger().warning(
            'LEADER_SCAN_DEBUG | '
            f'configured_topic={self.scan_topic} '
            f'resolved_topic={self._resolved_topic(self.scan_topic)} '
            f'publisher_count={self.count_publishers(self.scan_topic)} '
            f'subscription_count={self.count_subscribers(self.scan_topic)} '
            f'frame_id={self.last_scan_frame or "(none)"} '
            f'source_stamp_age_ms={self.last_scan_source_age_ms:.0f} '
            f'receive_age_ms={self._age(self.last_scan_wall) * 1000.0:.0f} '
            f'ranges_count={self.last_scan_ranges} '
            f'range_min={self.last_scan_range_min:.3f} '
            f'range_max={self.last_scan_range_max:.3f} '
            f'finite_ranges={self.last_scan_finite_ranges} '
            f'scan_tf={checks["scan_tf_ok"]} '
            f'scan_tf_age_ms={checks["_scan_tf_age_ms"]:.0f} '
            f'qos_compatible=best_effort_sensor_data '
            f'blocking_reason={self._blocking_reason(checks)}',
            throttle_duration_sec=2.0,
        )

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
            self.amcl_state_label = 'service_error'
            return
        state = getattr(response.current_state, 'label', '')
        self.amcl_state_label = str(state).lower() or 'unknown'
        self.amcl_active_cached = str(state).lower() == 'active'

    def _publish_ready(self, ready: bool) -> None:
        self.ready_pub.publish(Bool(data=bool(ready)))

    def _tick(self) -> None:
        if self.done:
            return
        ok, reason, checks = self._readiness_state()
        self._log_localization_debug(ready=ok, reason=reason, checks=checks)
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
            f'xy_cov={self.xy_cov:.4f} yaw_cov={self.yaw_cov:.4f} '
            f'tf={self.global_frame}->{self.odom_frame}->{self.base_frame} lifecycle=active'
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
