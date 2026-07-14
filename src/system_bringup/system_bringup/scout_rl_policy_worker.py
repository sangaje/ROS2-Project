#!/usr/bin/env python3
"""Minimal ACTIVE_SCOUT SAC worker.

This worker is deliberately small.  It does not wait for Nav2, localization,
TF, dashboard readiness, failover recovery, or the old heavyweight confidence
pipeline.  It loads the frozen SAC checkpoint, builds a valid observation from
the latest LaserScan plus the latest SLAM map, derives/publishes a lightweight
confidence map, and publishes the resulting velocity command directly to
/cmd_vel.
"""

from __future__ import annotations

import math
import threading
import time
import traceback
from typing import Optional

import numpy as np
import rclpy
from geometry_msgs.msg import Twist, TwistStamped
from nav_msgs.msg import OccupancyGrid
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Bool, String

from turtlebot3_rl_training.observation import (
    LidarPreprocessorConfig,
    build_exploration_observation,
)

from .rl_policy_contract import active_scout_config, load_deployment_model, probe_checkpoint
from .role_contract import RoleMessage, parse_role_message


RoleUpdate = RoleMessage
parse_role_update = parse_role_message


class ScoutRLPolicyWorker(Node):
    """Run the deployment policy with only the checks needed to publish."""

    def __init__(self) -> None:
        super().__init__('scout_rl_policy_worker')
        self._declare_compatible_parameters()
        get = self.get_parameter

        self.robot_name = str(get('robot_name').value).strip() or 'scout'
        self.role_topic = str(get('role_topic').value).strip() or f'/{self.robot_name}/role'
        self.cmd_vel_topic = str(get('cmd_vel_topic').value).strip() or '/cmd_vel'
        self.config = active_scout_config()
        self.map_topic = str(get('map_topic').value).strip() or self.config.map_topic
        self.confidence_map_topic = str(get('confidence_map_topic').value).strip()
        self.use_stamped = bool(get('use_stamped_cmd_vel').value)
        self.direct_rl_start = bool(get('direct_rl_start').value)
        self.load_model_on_start = bool(get('load_model_on_start').value)
        self.require_system_ready = bool(get('require_system_ready').value)
        self.system_ready_topic = str(get('system_ready_topic').value).strip()
        self.emergency_stop_distance_m = max(
            0.0, float(get('emergency_stop_distance_m').value)
        )

        self.lidar = LidarPreprocessorConfig(
            canonical_front_zero=self.config.lidar.canonical_front_zero,
            front_index=self.config.lidar.front_index,
            angle_offset_deg=self.config.lidar.angle_offset_deg,
            flip_lr=self.config.lidar.flip_lr,
            uniform_angle_resample=self.config.lidar.uniform_angle_resample,
            median_kernel=self.config.lidar.median_kernel,
            lowpass_kernel=self.config.lidar.lowpass_kernel,
            obstacle_margin_m=self.config.lidar.obstacle_margin_m,
        )

        self.model = None
        self.model_error = ''
        self.model_loading = False
        self.model_loader_started = False
        self.model_load_started = time.monotonic()
        self.role_active = bool(get('initial_role_active').value) or self.direct_rl_start
        self.system_ready = not self.require_system_ready
        self.latest_scan: Optional[LaserScan] = None
        self.latest_scan_at = 0.0
        self.latest_map: Optional[OccupancyGrid] = None
        self.latest_map_at = 0.0
        self.latest_confidence_grid: Optional[np.ndarray] = None
        self.previous_action = np.zeros(2, dtype=np.float32)
        self.history_vector: list[np.ndarray] = []
        self.history_map: list[np.ndarray] = []
        self.predict_count = 0
        self.publish_count = 0
        self.first_command_logged = False
        self.last_wait_log_at = 0.0

        if self.use_stamped:
            self.cmd_pub = self.create_publisher(TwistStamped, self.cmd_vel_topic, 10)
        else:
            self.cmd_pub = self.create_publisher(Twist, self.cmd_vel_topic, 10)
        self.confidence_pub = self.create_publisher(
            OccupancyGrid,
            self.confidence_map_topic,
            1,
        )

        latched_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
        )
        self.create_subscription(String, self.role_topic, self._on_role, latched_qos)
        self.create_subscription(
            LaserScan,
            self.config.scan_topic,
            self._on_scan,
            qos_profile_sensor_data,
        )
        volatile_map_qos = QoSProfile(
            depth=1,
            history=HistoryPolicy.KEEP_LAST,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )
        latched_map_qos = QoSProfile(
            depth=1,
            history=HistoryPolicy.KEEP_LAST,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.create_subscription(
            OccupancyGrid,
            self.map_topic,
            self._on_map,
            volatile_map_qos,
        )
        self.create_subscription(
            OccupancyGrid,
            self.map_topic,
            self._on_map,
            latched_map_qos,
        )
        if self.require_system_ready:
            self.create_subscription(Bool, self.system_ready_topic, self._on_system_ready, latched_qos)

        self.create_timer(self.config.control_dt_sec, self._policy_tick)
        self.create_timer(1.0, self._status_tick)
        self.create_timer(1.0, self._publish_confidence_map)
        if self.load_model_on_start or self.role_active:
            self._ensure_model_loader_started('startup')

        self.get_logger().warning(
            'SCOUT_RL_MINIMAL_WORKER_READY | '
            f'robot={self.robot_name} role_active={self.role_active} '
            f'cmd_vel={self.cmd_vel_topic} stamped={self.use_stamped} '
            f'scan_topic={self.config.scan_topic} '
            f'map_topic={self.map_topic} confidence_topic={self.confidence_map_topic} '
            f'checkpoint={self.config.checkpoint} '
            f'direct_rl_start={self.direct_rl_start} '
            f'load_model_on_start={self.load_model_on_start} '
            f'emergency_stop_distance_m={self.emergency_stop_distance_m:.2f}'
        )

    def _declare_compatible_parameters(self) -> None:
        self.declare_parameter('robot_name', 'scout22')
        self.declare_parameter('role_topic', '')
        self.declare_parameter('initial_role_active', False)
        self.declare_parameter('failover_state_topic', '/failover/state')
        self.declare_parameter('active_scout_id_topic', '/failover/active_scout_id')
        self.declare_parameter('scout_epoch_topic', '/failover/scout_epoch')
        self.declare_parameter('localization_ready_topic', '/localization_ready')
        self.declare_parameter('field_robot_status_topic', '/fleet/field_robot_status')
        self.declare_parameter('require_failover_activation', True)
        self.declare_parameter('require_localization_ready', True)
        self.declare_parameter('require_system_ready', False)
        self.declare_parameter('system_ready_topic', '/system/ready')
        self.declare_parameter('require_start_motion', False)
        self.declare_parameter('start_motion_topic', '/fleet/start_motion')
        self.declare_parameter('direct_rl_start', True)
        self.declare_parameter('load_model_on_start', True)
        self.declare_parameter('motion_readiness_detail_topic', '/fleet/scout_motion_ready_detail')
        self.declare_parameter('motion_release_stable_sec', 0.0)
        self.declare_parameter('startup_sensor_max_age_sec', 2.0)
        self.declare_parameter('require_video_ready', False)
        self.declare_parameter('video_ready_topic', '/fleet/start_motion')
        self.declare_parameter('cmd_vel_topic', '/cmd_vel')
        self.declare_parameter('map_topic', '')
        self.declare_parameter('confidence_map_topic', '/rl_confidence_map')
        self.declare_parameter('use_stamped_cmd_vel', True)
        self.declare_parameter('enable_velocity_safety_filter', False)
        self.declare_parameter('odom_topic', '/odom')
        self.declare_parameter('max_odom_age_sec', 2.0)
        self.declare_parameter('emergency_stop_distance_m', 0.20)

    def _ensure_model_loader_started(self, reason: str) -> None:
        if self.model is not None or self.model_loading or self.model_loader_started:
            return
        self.model_loader_started = True
        self.model_loading = True
        self.model_load_started = time.monotonic()
        self._start_model_loader(reason)

    def _start_model_loader(self, reason: str) -> None:
        def load() -> None:
            self.get_logger().warning(
                f'SCOUT_MODEL_LOAD_START | reason={reason} model_path={self.config.checkpoint}'
            )
            try:
                model = load_deployment_model()
                probe_checkpoint(model=model)
            except Exception as exc:  # noqa: BLE001
                self.model_error = f'{exc}'
                self.get_logger().error(
                    'SCOUT_MODEL_LOAD_FAILED | '
                    f'error={exc}\n{traceback.format_exc()}'
                )
                model = None
            self.model = model
            self.model_loading = False
            elapsed_ms = int((time.monotonic() - self.model_load_started) * 1000.0)
            self.get_logger().warning(
                'SCOUT_MODEL_READY_PIPELINE | '
                f'load_success={model is not None} elapsed_ms={elapsed_ms} '
                f'error={self.model_error}'
            )

        threading.Thread(target=load, name='scout_rl_model_loader', daemon=True).start()

    def _on_role(self, msg: String) -> None:
        update = parse_role_message(msg.data, self.robot_name)
        if update is None:
            return
        if update.robot and update.robot != self.robot_name:
            return
        role = str(getattr(update.role, 'value', update.role)).strip().upper()
        was_active = self.role_active
        self.role_active = role == 'ACTIVE_SCOUT'
        if was_active != self.role_active:
            self.get_logger().warning(
                f'SCOUT_RL_ROLE | role={role} active={self.role_active}'
            )
            if self.role_active:
                self._ensure_model_loader_started('active_scout_role')
            if not self.role_active:
                self._publish_zero()

    def _on_system_ready(self, msg: Bool) -> None:
        self.system_ready = bool(msg.data)
        if not self.system_ready:
            self._publish_zero()

    def _on_scan(self, msg: LaserScan) -> None:
        self.latest_scan = msg
        self.latest_scan_at = time.monotonic()

    def _on_map(self, msg: OccupancyGrid) -> None:
        if int(msg.info.width) <= 0 or int(msg.info.height) <= 0:
            return
        if len(msg.data) != int(msg.info.width) * int(msg.info.height):
            return
        self.latest_map = msg
        self.latest_map_at = time.monotonic()
        self.latest_confidence_grid = self._confidence_from_map(msg)

    def _policy_tick(self) -> None:
        if self.role_active:
            self._ensure_model_loader_started('policy_tick_active')
        if not self._motion_allowed():
            self._wait_log(self._blocking_reason())
            self._publish_zero()
            return
        scan = self.latest_scan
        if scan is None:
            self._wait_log('scan_missing')
            self._publish_zero()
            return
        if self.latest_map is None:
            self._wait_log('map_missing')
            self._publish_zero()
            return
        if time.monotonic() - self.latest_scan_at > self.config.max_scan_age_sec:
            self._wait_log('scan_stale')
            self._publish_zero()
            return
        if time.monotonic() - self.latest_map_at > self.config.max_map_age_sec:
            self._wait_log('map_stale')
            self._publish_zero()
            return

        try:
            observation = self._build_observation(scan)
            started = time.monotonic()
            action, _ = self.model.predict(observation, deterministic=True)
            predict_ms = (time.monotonic() - started) * 1000.0
            command = self._command_from_action(np.asarray(action, dtype=np.float32))
        except Exception as exc:  # noqa: BLE001
            self.get_logger().error(
                f'SCOUT_RL_DIRECT_PREDICT_FAILED | {exc}\n{traceback.format_exc()}',
                throttle_duration_sec=1.0,
            )
            self._publish_zero()
            return

        self.previous_action = command.copy()
        self.predict_count += 1
        self._publish_command(float(command[0]), float(command[1]))
        if not self.first_command_logged or self.predict_count % 10 == 0:
            self.first_command_logged = True
            self.get_logger().warning(
                'SCOUT_RL_DIRECT_CMD | '
                f'predict_count={self.predict_count} predict_ms={predict_ms:.1f} '
                f'linear_x={float(command[0]):.3f} angular_z={float(command[1]):.3f} '
                f'publish_count={self.publish_count}'
            )

    def _motion_allowed(self) -> bool:
        return bool(
            self.role_active
            and self.model is not None
            and not self.model_loading
            and (self.system_ready or not self.require_system_ready)
        )

    def _blocking_reason(self) -> str:
        if not self.role_active:
            return 'role_not_active_scout'
        if not self.model_loader_started:
            return 'model_not_started'
        if self.model_loading:
            return 'model_loading'
        if self.model is None:
            return f'model_error:{self.model_error}'
        if self.require_system_ready and not self.system_ready:
            return 'system_not_ready'
        return 'unknown'

    def _wait_log(self, reason: str) -> None:
        now = time.monotonic()
        if now - self.last_wait_log_at < 1.0:
            return
        self.last_wait_log_at = now
        load_ms = int((now - self.model_load_started) * 1000.0)
        self.get_logger().warning(
            'SCOUT_RL_DIRECT_WAIT | '
            f'reason={reason} role_active={self.role_active} '
            f'model_loading={self.model_loading} model_ready={self.model is not None} '
            f'model_load_elapsed_ms={load_ms} scan_received={self.latest_scan is not None} '
            f'map_received={self.latest_map is not None}'
        )

    def _build_observation(self, scan: LaserScan) -> dict[str, np.ndarray]:
        vector = build_exploration_observation(
            scan_ranges=scan.ranges,
            coverage_ratio=1.0,
            coverage_delta=0.0,
            frontier_distance=3.5,
            frontier_angle=0.0,
            target_priority=0.0,
            mean_confidence=50.0,
            stale_ratio=0.0,
            low_confidence_ratio=0.0,
            prev_action=self.previous_action,
            num_lidar_bins=self.config.lidar_bins,
            max_linear_speed=self.config.action_high[0],
            max_angular_speed=self.config.action_high[1],
            scan_angle_min=scan.angle_min,
            scan_angle_increment=scan.angle_increment,
            scan_angle_max=scan.angle_max,
            include_target_priority=False,
            trim_extra_stats=self.config.trim_extra_stats,
            lidar_config=self.lidar,
        ).astype(np.float32)
        if vector.shape != (self.config.vector_dim,):
            raise RuntimeError(f'vector shape mismatch: {vector.shape}')

        map_obs = self._map_observation()
        if not self.history_vector:
            self.history_vector = [vector.copy() for _ in range(self.config.history_len)]
            self.history_map = [map_obs.copy() for _ in range(self.config.history_len)]
        else:
            self.history_vector.append(vector.copy())
            self.history_map.append(map_obs.copy())
            self.history_vector = self.history_vector[-self.config.history_len:]
            self.history_map = self.history_map[-self.config.history_len:]

        return {
            'map': map_obs,
            'map_seq': np.stack(self.history_map, axis=0).astype(np.float32),
            'seq': np.stack(self.history_vector, axis=0).astype(np.float32),
            'vector': vector,
        }

    def _map_observation(self) -> np.ndarray:
        slam_map = self.latest_map
        if slam_map is None:
            raise RuntimeError('map is not ready')
        width = int(slam_map.info.width)
        height = int(slam_map.info.height)
        grid = np.asarray(slam_map.data, dtype=np.int16).reshape((height, width))
        sample = self._resize_nearest(grid, self.config.map_obs_size)

        confidence = self.latest_confidence_grid
        if confidence is None:
            confidence = self._confidence_from_map(slam_map)
        confidence_sample = self._resize_nearest(
            confidence.astype(np.float32),
            self.config.map_obs_size,
        )

        map_obs = np.zeros(
            (4, self.config.map_obs_size, self.config.map_obs_size),
            dtype=np.float32,
        )
        known = sample >= 0
        occupied = sample >= 50
        free = known & ~occupied
        unknown = ~known
        map_obs[0, free] = 1.0
        map_obs[1, unknown] = 1.0
        map_obs[2, occupied] = 1.0
        map_obs[3, :, :] = np.clip(confidence_sample / 100.0, 0.0, 1.0)
        return map_obs

    @staticmethod
    def _resize_nearest(grid: np.ndarray, size: int) -> np.ndarray:
        height, width = grid.shape
        ys = np.linspace(0, max(height - 1, 0), int(size)).astype(np.int32)
        xs = np.linspace(0, max(width - 1, 0), int(size)).astype(np.int32)
        return grid[ys[:, None], xs[None, :]]

    def _confidence_from_map(self, slam_map: OccupancyGrid) -> np.ndarray:
        width = int(slam_map.info.width)
        height = int(slam_map.info.height)
        grid = np.asarray(slam_map.data, dtype=np.int16).reshape((height, width))
        confidence = np.zeros((height, width), dtype=np.float32)
        known = grid >= 0
        occupied = grid >= 50
        confidence[known] = 70.0
        confidence[occupied] = 100.0
        return confidence

    def _publish_confidence_map(self) -> None:
        slam_map = self.latest_map
        confidence = self.latest_confidence_grid
        if slam_map is None or confidence is None:
            return
        msg = OccupancyGrid()
        msg.header = slam_map.header
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.info = slam_map.info
        msg.data = np.clip(confidence, 0.0, 100.0).astype(np.int8).reshape(-1).tolist()
        self.confidence_pub.publish(msg)

    def _command_from_action(self, action: np.ndarray) -> np.ndarray:
        raw = action.reshape(-1)[:2].astype(np.float32)
        command = np.clip(
            raw,
            np.asarray(self.config.action_low, dtype=np.float32),
            np.asarray(self.config.action_high, dtype=np.float32),
        ).astype(np.float32)
        scan = self.latest_scan
        if scan is not None and self._front_min(scan) < self.emergency_stop_distance_m:
            command[:] = 0.0
        return command

    def _front_min(self, scan: LaserScan) -> float:
        ranges = np.asarray(scan.ranges, dtype=np.float32)
        if ranges.size == 0:
            return float('inf')
        ranges = np.nan_to_num(ranges, nan=float('inf'), posinf=float('inf'), neginf=0.0)
        angle_min = float(scan.angle_min)
        angle_increment = float(scan.angle_increment)
        if not math.isfinite(angle_increment) or abs(angle_increment) < 1.0e-9:
            return float(np.min(ranges))
        angles = angle_min + np.arange(ranges.size, dtype=np.float32) * angle_increment
        front = np.abs(np.arctan2(np.sin(angles), np.cos(angles))) <= math.radians(25.0)
        if not np.any(front):
            return float(np.min(ranges))
        return float(np.min(ranges[front]))

    def _publish_command(self, linear_x: float, angular_z: float) -> None:
        if self.use_stamped:
            msg = TwistStamped()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = 'base_footprint'
            msg.twist.linear.x = linear_x
            msg.twist.angular.z = angular_z
        else:
            msg = Twist()
            msg.linear.x = linear_x
            msg.angular.z = angular_z
        self.cmd_pub.publish(msg)
        self.publish_count += 1

    def _publish_zero(self) -> None:
        self._publish_command(0.0, 0.0)

    def _status_tick(self) -> None:
        scan_age_ms = (
            (time.monotonic() - self.latest_scan_at) * 1000.0
            if self.latest_scan is not None else -1.0
        )
        map_age_ms = (
            (time.monotonic() - self.latest_map_at) * 1000.0
            if self.latest_map is not None else -1.0
        )
        self.get_logger().info(
            'SCOUT_RL_DIRECT_STATUS | '
            f'role_active={self.role_active} model_ready={self.model is not None} '
            f'model_loading={self.model_loading} scan_age_ms={scan_age_ms:.0f} '
            f'map_age_ms={map_age_ms:.0f} confidence_ready={self.latest_confidence_grid is not None} '
            f'predict_count={self.predict_count} publish_count={self.publish_count}'
        )


def main(args=None) -> None:
    rclpy.init(args=args)
    node = ScoutRLPolicyWorker()
    executor = MultiThreadedExecutor(num_threads=3)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
