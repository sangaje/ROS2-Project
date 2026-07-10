"""Single-process deterministic SAC runtime for an ACTIVE_SCOUT.

This module intentionally does not create a ROS node, a subprocess, or a
Gazebo environment.  ``UnifiedFieldRobot`` owns the node and the only command
publisher; this object contributes bounded sensor snapshots plus two role-gated
callbacks to that node.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math
import threading
import time
import traceback
from typing import Callable, Optional

import numpy as np
from nav_msgs.msg import OccupancyGrid
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
from rclpy.duration import Duration
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import LaserScan
from tf2_ros import Buffer, TransformException, TransformListener

from turtlebot3_rl_training.exploration_map import ExplorationGridMap, MapUpdateStats
from turtlebot3_rl_training.observation import (
    LidarPreprocessorConfig,
    build_exploration_observation,
)

from .rl_policy_contract import (
    ActiveScoutPolicyConfig,
    active_scout_config,
    load_deployment_model,
    probe_checkpoint,
)


@dataclass(frozen=True)
class SensorSnapshot:
    scan: Optional[LaserScan]
    scan_received_at: float
    scan_generation: int
    slam_map: Optional[OccupancyGrid]
    map_received_at: float
    map_generation: int


@dataclass(frozen=True)
class MapSnapshot:
    stats: MapUpdateStats
    robot_xy: np.ndarray
    robot_yaw: float
    scan_generation: int
    map_generation: int
    updated_at: float


@dataclass
class RuntimeCounters:
    model_load_count: int = 0
    scan_callback_count: int = 0
    map_callback_count: int = 0
    pose_success_count: int = 0
    confidence_update_attempt_count: int = 0
    confidence_update_success_count: int = 0
    predict_attempt_count: int = 0
    predict_success_count: int = 0
    predict_failure_count: int = 0


def _stamp_time(message) -> Time:
    stamp = getattr(getattr(message, 'header', None), 'stamp', None)
    if stamp is None or (int(stamp.sec) == 0 and int(stamp.nanosec) == 0):
        return Time()
    return Time.from_msg(stamp)


def _yaw_from_quaternion(q) -> float:
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


def _scan_sector_min(
    scan: LaserScan,
    center: float,
    half_width: float,
    lidar: LidarPreprocessorConfig,
) -> float:
    ranges = np.asarray(scan.ranges, dtype=np.float32)
    if ranges.size == 0:
        return float('inf')
    range_min = max(float(scan.range_min), 0.03)
    range_max = float(scan.range_max)
    if not math.isfinite(range_max) or range_max <= range_min:
        range_max = 10.0
    angles = float(scan.angle_min) + np.arange(ranges.size, dtype=np.float32) * float(scan.angle_increment)
    if lidar.flip_lr:
        angles = -angles
    angles = angles + math.radians(lidar.angle_offset_deg)
    delta = np.arctan2(np.sin(angles - center), np.cos(angles - center))
    valid = np.isfinite(ranges) & (ranges >= range_min) & (ranges <= range_max)
    valid &= np.abs(delta) <= half_width
    return float(np.min(ranges[valid])) if np.any(valid) else float('inf')


class VelocitySafetyFilter:
    """Timer-safe projection of v132 velocity safety behavior.

    The training environment performs a synchronous backup sequence while it
    advances simulation.  A real robot must never block a callback for that
    sequence, so the same finite sequence is emitted one control tick at a
    time.  Stale input is handled by the caller before reaching this filter.
    """

    def __init__(self, config: ActiveScoutPolicyConfig, lidar: LidarPreprocessorConfig):
        self.config = config
        self.lidar = lidar
        self.backup_remaining = 0
        self.cooldown_remaining = 0
        self.turn_sign = 1.0

    def reset(self) -> None:
        self.backup_remaining = 0
        self.cooldown_remaining = 0
        self.turn_sign = 1.0

    def filter(self, raw_action: np.ndarray, scan: LaserScan) -> np.ndarray:
        action = np.asarray(raw_action, dtype=np.float32).reshape(-1)
        if action.size != 2 or not np.all(np.isfinite(action)):
            return np.zeros(2, dtype=np.float32)
        action = np.clip(
            action,
            np.asarray(self.config.action_low, dtype=np.float32),
            np.asarray(self.config.action_high, dtype=np.float32),
        )
        front = _scan_sector_min(scan, 0.0, math.pi / 4.0, self.lidar)
        rear = _scan_sector_min(scan, math.pi, math.pi / 4.0, self.lidar)
        left = _scan_sector_min(scan, math.pi / 2.0, math.pi / 4.0, self.lidar)
        right = _scan_sector_min(scan, -math.pi / 2.0, math.pi / 4.0, self.lidar)

        if self.backup_remaining > 0:
            if rear <= self.config.safety_stop_distance_m:
                self.reset()
                return np.zeros(2, dtype=np.float32)
            self.backup_remaining -= 1
            if self.backup_remaining == 0:
                self.cooldown_remaining = self.config.safety_cooldown_steps
            return np.array(
                [-self.config.safety_backup_speed_mps, self.turn_sign * self.config.safety_turn_speed],
                dtype=np.float32,
            )

        if self.cooldown_remaining > 0:
            self.cooldown_remaining -= 1
        forward_requested = float(action[0]) >= max(self.config.linear_deadband, 0.04)
        if (
            forward_requested
            and front < self.config.safety_trigger_distance_m
            and rear > self.config.safety_stop_distance_m
            and self.cooldown_remaining == 0
        ):
            self.turn_sign = 1.0 if left >= right else -1.0
            self.backup_remaining = max(self.config.safety_backup_steps - 1, 0)
            return np.array(
                [-self.config.safety_backup_speed_mps, self.turn_sign * self.config.safety_turn_speed],
                dtype=np.float32,
            )

        if forward_requested and front < self.config.safety_stop_distance_m:
            action[0] = 0.0
        elif (
            forward_requested
            and self.config.safety_slowdown
            and front < self.config.safety_slow_distance_m
        ):
            span = max(
                self.config.safety_slow_distance_m - self.config.safety_stop_distance_m,
                1.0e-6,
            )
            scale = (front - self.config.safety_stop_distance_m) / span
            scale = float(np.clip(scale, self.config.safety_slow_min_scale, 1.0))
            action[0] *= scale

        if 0.0 < float(action[0]) < self.config.linear_deadband:
            action[0] = 0.0
        if abs(float(action[1])) < self.config.angular_deadband:
            action[1] = 0.0
        return action.astype(np.float32, copy=False)


class ActiveScoutRLRuntime:
    """Role-gated deterministic SAC inference attached to one host node."""

    def __init__(
        self,
        node,
        publish_command: Callable[[float, float], None],
        *,
        config: Optional[ActiveScoutPolicyConfig] = None,
        model_loader=None,
        on_stop: Optional[Callable[[str], None]] = None,
        on_ready: Optional[Callable[[], None]] = None,
    ) -> None:
        self.node = node
        self.config = config or active_scout_config()
        self.publish_command = publish_command
        self.on_stop = on_stop
        self.on_ready = on_ready
        self._active = False
        self._lock = threading.Lock()
        self._map_state_lock = threading.Lock()
        self._model_lock = threading.Lock()
        self._scan: Optional[LaserScan] = None
        self._scan_received_at = 0.0
        self._scan_generation = 0
        self._map: Optional[OccupancyGrid] = None
        self._map_received_at = 0.0
        self._map_generation = 0
        self._map_snapshot: Optional[MapSnapshot] = None
        self._history_vector: deque[np.ndarray] = deque(maxlen=self.config.history_len)
        self._history_map: deque[np.ndarray] = deque(maxlen=self.config.history_len)
        self._previous_action = np.zeros(2, dtype=np.float32)
        self._last_command_at = 0.0
        self._activated_at = 0.0
        self._last_error_at = 0.0
        self._last_heartbeat_at = 0.0
        self._last_stop_reason = 'not_activated'
        self._model_error: Optional[str] = None
        self._last_error = ''
        self._model_loading = True
        self._model_ready_notified = False
        self.model = None
        self.counters = RuntimeCounters()
        self._sensor_group = ReentrantCallbackGroup()
        self._map_group = MutuallyExclusiveCallbackGroup()
        self._policy_group = MutuallyExclusiveCallbackGroup()
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self.node, spin_thread=False)
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
        self.safety = VelocitySafetyFilter(self.config, self.lidar)
        self.exploration_map = ExplorationGridMap(
            node=self.node,
            resolution=self.config.map_resolution_m,
            size_m=self.config.map_initial_size_m,
            origin_x=-self.config.map_initial_size_m * 0.5,
            origin_y=-self.config.map_initial_size_m * 0.5,
            frame_id=self.config.map_frame,
            publish_topic='/rl_task_map',
            confidence_publish_topic='/rl_confidence_map',
            priority_publish_topic='',
            disable_priority_map=True,
            path_publish_topic='',
            filtered_slam_publish_topic='',
            legacy_memory_publish_topic='',
            publish_slam_aligned=True,
            keepalive_publish_period_sec=self.config.map_keepalive_period_sec,
            lidar_stride=2,
            max_range=3.5,
            publish_every_n=self.config.map_publish_every_n,
            min_known_confidence=8.0,
            low_confidence_threshold=35.0,
            stale_after_steps=180,
            confidence_decay_per_step=0.0,
            logodds_decay_per_step=0.0008,
            distance_weight_beta=0.30,
            confidence_max_range=2.0,
            front_angle_sigma_deg=20.0,
            seen_confidence_floor=70.0,
            clear_confidence_on_slam_occupied=self.config.clear_confidence_on_slam_occupied,
            confidence_occupied_confirm_steps=self.config.confidence_occupied_confirm_steps,
            confidence_decay_near_obstacle_scale=self.config.confidence_decay_near_obstacle_scale,
            confidence_obstacle_ring_radius=self.config.confidence_obstacle_ring_radius_cells,
            confidence_obstacle_floor_ratio=self.config.confidence_obstacle_floor_ratio,
            confidence_lidar_hit_guard_m=self.config.confidence_lidar_hit_guard_m,
            confidence_lidar_occlusion_radius_cells=self.config.confidence_lidar_occlusion_radius_cells,
            use_slam_prior=True,
            front_fov_deg=60.0,
            lidar_policy_config=self.lidar,
            deployment_mode=True,
        )
        map_qos = QoSProfile(
            depth=1,
            history=HistoryPolicy.KEEP_LAST,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.scan_sub = self.node.create_subscription(
            LaserScan, self.config.scan_topic, self._on_scan, qos_profile_sensor_data,
            callback_group=self._sensor_group,
        )
        self.map_sub = self.node.create_subscription(
            OccupancyGrid, self.config.map_topic, self._on_map, map_qos,
            callback_group=self._sensor_group,
        )
        self.map_timer = self.node.create_timer(
            self.config.control_dt_sec / self.config.map_substeps_per_action,
            self._map_tick,
            callback_group=self._map_group,
        )
        self.policy_timer = self.node.create_timer(
            self.config.control_dt_sec,
            self._policy_tick,
            callback_group=self._policy_group,
        )
        self.watchdog_timer = self.node.create_timer(
            min(self.config.control_dt_sec * 0.5, 0.05),
            self._command_watchdog,
            callback_group=self._sensor_group,
        )
        self.model_state_timer = self.node.create_timer(
            0.1,
            self._model_state_tick,
            callback_group=self._sensor_group,
        )
        self._start_model_loader(model_loader)

    @property
    def ready(self) -> bool:
        with self._model_lock:
            return self.model is not None and self._model_error is None

    @property
    def active(self) -> bool:
        return self._active and self.ready

    @property
    def last_stop_reason(self) -> str:
        return self._last_stop_reason

    def activate(self) -> None:
        self._reset_episode_state()
        self._active = True
        self._activated_at = time.monotonic()
        self._last_stop_reason = ''
        if self.ready:
            self.node.get_logger().warning('SCOUT_RL_ACTIVE | deterministic=true map_substeps=2')
            return
        if self._model_loading:
            self.node.get_logger().warning(
                'SCOUT_RL_MODEL_LOADING | runtime keeps map/scan callbacks live'
            )
            self.publish_command(0.0, 0.0)
            return
        if not self.ready:
            self._stop('model_unavailable')
            self.node.get_logger().error(f'SCOUT_RL_UNAVAILABLE | {self._model_error}')
            return

    def deactivate(self, reason: str) -> None:
        self._active = False
        self._reset_episode_state()
        self._stop(reason)

    def shutdown(self) -> None:
        self.deactivate('runtime_shutdown')

    def _load_model(self, model_loader):
        try:
            model = (
                load_deployment_model()
                if model_loader is None
                else model_loader(str(self.config.checkpoint), device='cpu', buffer_size=1)
            )
            probe_checkpoint(model=model)
            self.counters.model_load_count += 1
            return model
        except Exception as exc:  # noqa: BLE001
            self._model_error = str(exc)
            self._last_error = traceback.format_exc()
            self.node.get_logger().error(
                f'SCOUT_RL_MODEL_LOAD_FAILED | {exc}\n{self._last_error}'
            )
            return None

    def _start_model_loader(self, model_loader) -> None:
        """Keep PyTorch/SB3 import and checkpoint deserialization off the ROS executor."""
        def load() -> None:
            model = self._load_model(model_loader)
            with self._model_lock:
                self.model = model
                self._model_loading = False

        threading.Thread(
            target=load,
            name='scout_rl_model_loader',
            daemon=True,
        ).start()

    def _model_state_tick(self) -> None:
        """Apply loader completion on an executor callback, never the loader thread."""
        if not self._active or self._model_loading:
            return
        if not self.ready:
            self._stop('model_unavailable')
            return
        if self._model_ready_notified:
            return
        self._model_ready_notified = True
        self.node.get_logger().warning(
            'SCOUT_RL_ACTIVE | deterministic=true map_substeps=2 model=ready'
        )
        if self.on_ready is not None:
            self.on_ready()

    def _reset_episode_state(self) -> None:
        self._history_vector.clear()
        self._history_map.clear()
        self._previous_action = np.zeros(2, dtype=np.float32)
        self._map_snapshot = None
        self.safety.reset()

    def _on_scan(self, message: LaserScan) -> None:
        with self._lock:
            self._scan = message
            self._scan_received_at = time.monotonic()
            self._scan_generation += 1
            self.counters.scan_callback_count += 1

    def _on_map(self, message: OccupancyGrid) -> None:
        with self._lock:
            self._map = message
            self._map_received_at = time.monotonic()
            self._map_generation += 1
            self.counters.map_callback_count += 1

    def _sensor_snapshot(self) -> SensorSnapshot:
        with self._lock:
            return SensorSnapshot(
                scan=self._scan,
                scan_received_at=self._scan_received_at,
                scan_generation=self._scan_generation,
                slam_map=self._map,
                map_received_at=self._map_received_at,
                map_generation=self._map_generation,
            )

    def _fresh(self, snapshot: SensorSnapshot, now: float) -> bool:
        return bool(
            snapshot.scan is not None
            and snapshot.slam_map is not None
            and now - snapshot.scan_received_at <= self.config.max_scan_age_sec
            and now - snapshot.map_received_at <= self.config.max_map_age_sec
        )

    def _lookup_pose(self, target_frame: str, source_frame: str, stamp: Time) -> tuple[np.ndarray, float]:
        transform = self.tf_buffer.lookup_transform(
            target_frame,
            source_frame,
            stamp,
            timeout=Duration(seconds=self.config.max_tf_age_sec),
        )
        translation = transform.transform.translation
        return (
            np.array([float(translation.x), float(translation.y)], dtype=np.float32),
            _yaw_from_quaternion(transform.transform.rotation),
        )

    def _map_tick(self) -> None:
        if not self._active:
            return
        snapshot = self._sensor_snapshot()
        now = time.monotonic()
        if not self._fresh(snapshot, now):
            self._hold('waiting_for_sensor_or_map')
            return
        assert snapshot.scan is not None and snapshot.slam_map is not None
        map_frame = str(snapshot.slam_map.header.frame_id or '').lstrip('/')
        if map_frame != self.config.map_frame.lstrip('/'):
            self._hold('waiting_for_expected_map_frame')
            return
        scan_frame = str(snapshot.scan.header.frame_id or self.config.scan_frame).lstrip('/')
        try:
            stamp = _stamp_time(snapshot.scan)
            robot_xy, robot_yaw = self._lookup_pose(self.config.map_frame, self.config.base_frame, stamp)
            sensor_xy, sensor_yaw = self._lookup_pose(self.config.map_frame, scan_frame, stamp)
            self.counters.pose_success_count += 1
            self.counters.confidence_update_attempt_count += 1
            # The exploration map is mutable, but the sensor lock is never held
            # while this CPU-heavy update runs.  Policy reads take the same lock
            # only while cropping an observation, then release it before predict.
            with self._map_state_lock:
                stats = self.exploration_map.update(
                    snapshot.scan,
                    robot_xy,
                    robot_yaw,
                    publish=True,
                    slam_map=snapshot.slam_map,
                    sensor_xy=sensor_xy,
                    sensor_yaw=sensor_yaw,
                )
                self._map_snapshot = MapSnapshot(
                    stats=stats,
                    robot_xy=robot_xy,
                    robot_yaw=robot_yaw,
                    scan_generation=snapshot.scan_generation,
                    map_generation=snapshot.map_generation,
                    updated_at=now,
                )
            self.counters.confidence_update_success_count += 1
            self._log_heartbeat()
        except TransformException as exc:
            self._last_error = f'TransformException: {exc}'
            self._warn_throttled(f'SCOUT_RL_WAIT_TF | {exc}')
            self._hold('waiting_for_tf')
            return
        except Exception as exc:  # noqa: BLE001
            self._last_error = traceback.format_exc()
            self._warn_throttled(f'SCOUT_RL_WAIT_MAP_UPDATE | {exc}\n{self._last_error}')
            self._hold('waiting_for_map_update')
            return

    def _policy_tick(self) -> None:
        if not self._active or self.model is None:
            return
        snapshot = self._sensor_snapshot()
        map_snapshot = self._map_snapshot
        now = time.monotonic()
        if (
            not self._fresh(snapshot, now)
            or map_snapshot is None
            or now - map_snapshot.updated_at > self.config.max_scan_age_sec
        ):
            self._hold('waiting_for_coherent_observation')
            return
        assert snapshot.scan is not None
        try:
            # Never retain the map lock over model.predict(): scan/map callbacks
            # must remain able to replace their latest snapshots while inference
            # runs on the other executor worker.
            with self._map_state_lock:
                observation = self._build_observation(snapshot.scan, map_snapshot)
            if not all(np.all(np.isfinite(value)) for value in observation.values()):
                raise RuntimeError('observation contains NaN or Inf')
            started = time.monotonic()
            self.counters.predict_attempt_count += 1
            action, _ = self.model.predict(observation, deterministic=True)
            elapsed = time.monotonic() - started
            if elapsed > self.config.max_inference_sec:
                self.counters.predict_failure_count += 1
                self._warn_throttled(f'SCOUT_RL_INFERENCE_TIMEOUT | sec={elapsed:.3f}')
                self._hold('inference_timeout')
                return
            command = self.safety.filter(np.asarray(action, dtype=np.float32), snapshot.scan)
        except Exception as exc:  # noqa: BLE001
            self.counters.predict_failure_count += 1
            self._last_error = traceback.format_exc()
            self._warn_throttled(f'SCOUT_RL_INFERENCE_FAILED | {exc}\n{self._last_error}')
            # A bad frame or a transient CPU error must not turn ACTIVE_SCOUT
            # into a permanent inactive latch.  The next coherent snapshot gets
            # another deterministic attempt.
            self._hold('inference_error')
            return
        self._previous_action = command.copy()
        self._last_command_at = now
        self.counters.predict_success_count += 1
        self.publish_command(float(command[0]), float(command[1]))
        self._log_heartbeat()

    def _command_watchdog(self) -> None:
        if not self._active:
            return
        # TF lookup can legitimately take up to max_tf_age_sec while
        # Cartographer creates its first map->base transform. Until a coherent
        # map snapshot exists, _map_tick() owns the zero-command safety hold.
        if self._map_snapshot is None:
            return
        now = time.monotonic()
        if (
            now - self._activated_at > self.config.command_timeout_sec
            and now - self._last_command_at > self.config.command_timeout_sec
        ):
            self._warn_throttled('SCOUT_RL_COMMAND_TIMEOUT')
            # A delayed first map crop or predict must not revoke ACTIVE_SCOUT.
            # Publish an explicit zero so no old action survives, then let the
            # next timer tick retry with the newest sensor snapshot.
            self._hold('command_timeout')

    def _hold(self, reason: str) -> None:
        """Keep the role active while a startup/transient data gate recovers."""
        self._last_stop_reason = reason
        self._last_command_at = time.monotonic()
        self.publish_command(0.0, 0.0)
        self._log_heartbeat()

    def _log_heartbeat(self) -> None:
        """Emit aggregate liveness evidence without per-tick log spam."""
        now = time.monotonic()
        if now - self._last_heartbeat_at < 1.0:
            return
        self._last_heartbeat_at = now
        snapshot = self._sensor_snapshot()
        map_snapshot = self._map_snapshot
        scan_age = (now - snapshot.scan_received_at) * 1000.0 if snapshot.scan else -1.0
        map_age = (now - snapshot.map_received_at) * 1000.0 if snapshot.slam_map else -1.0
        obs_ok = bool(map_snapshot is not None and now - map_snapshot.updated_at <= self.config.max_scan_age_sec)
        counters = self.counters
        self.node.get_logger().info(
            'RL_RUNTIME_HEARTBEAT | '
            f'active={self._active} scan_age_ms={scan_age:.0f} map_age_ms={map_age:.0f} '
            f'pose_ok={counters.pose_success_count > 0} obs_ok={obs_ok} '
            f'scan_cb={counters.scan_callback_count} map_cb={counters.map_callback_count} '
            f'model_load={counters.model_load_count} '
            f'conf_attempts={counters.confidence_update_attempt_count} '
            f'conf_success={counters.confidence_update_success_count} '
            f'predict_attempts={counters.predict_attempt_count} '
            f'predict_success={counters.predict_success_count} '
            f'predict_failure={counters.predict_failure_count} '
            f'last_error={self._last_error.splitlines()[-1] if self._last_error else ""}'
        )

    def _build_observation(self, scan: LaserScan, map_snapshot: MapSnapshot) -> dict[str, np.ndarray]:
        stats = map_snapshot.stats
        vector = build_exploration_observation(
            scan_ranges=scan.ranges,
            coverage_ratio=stats.coverage_ratio,
            coverage_delta=stats.coverage_delta,
            frontier_distance=stats.frontier_distance,
            frontier_angle=stats.frontier_angle,
            target_priority=stats.target_priority,
            mean_confidence=stats.mean_confidence,
            stale_ratio=stats.stale_ratio,
            low_confidence_ratio=stats.low_confidence_ratio,
            prev_action=self._previous_action,
            num_lidar_bins=self.config.lidar_bins,
            max_linear_speed=self.config.action_high[0],
            max_angular_speed=self.config.action_high[1],
            scan_angle_min=scan.angle_min,
            scan_angle_increment=scan.angle_increment,
            scan_angle_max=scan.angle_max,
            include_target_priority=False,
            lidar_config=self.lidar,
        ).astype(np.float32)
        map_observation = self.exploration_map.build_update_need_tensor(
            robot_xy=map_snapshot.robot_xy,
            robot_yaw=map_snapshot.robot_yaw,
            output_size=self.config.map_obs_size,
            size_m=self.config.map_crop_size_m,
            rotate_to_robot=True,
        ).astype(np.float32)
        if vector.shape != (self.config.lidar_bins + 9,):
            raise RuntimeError(f'vector shape mismatch: {vector.shape}')
        if map_observation.shape != (4, self.config.map_obs_size, self.config.map_obs_size):
            raise RuntimeError(f'map shape mismatch: {map_observation.shape}')
        if not self._history_vector:
            for _ in range(self.config.history_len):
                self._history_vector.append(vector.copy())
                self._history_map.append(map_observation.copy())
        else:
            self._history_vector.append(vector.copy())
            self._history_map.append(map_observation.copy())
        observation = {
            'vector': vector,
            'map': map_observation,
            'seq': np.stack(self._history_vector, axis=0).astype(np.float32),
            'map_seq': np.stack(self._history_map, axis=0).astype(np.float32),
        }
        if self.counters.predict_attempt_count > 0 and self.counters.predict_attempt_count % 10 == 0:
            self.node.get_logger().info(
                'OBS_RUNTIME | '
                f'map_shape={observation["map"].shape} '
                f'map_seq_shape={observation["map_seq"].shape} '
                f'vector_shape={observation["vector"].shape} '
                f'seq_shape={observation["seq"].shape} '
                f'finite={all(np.all(np.isfinite(value)) for value in observation.values())}'
            )
        return observation

    def _stop(self, reason: str) -> None:
        was_active = self._active
        self._active = False
        self._last_stop_reason = reason
        self._last_command_at = 0.0
        self.publish_command(0.0, 0.0)
        if was_active:
            self.node.get_logger().error(f'SCOUT_RL_STOP | reason={reason}')
        if was_active and self.on_stop is not None:
            self.on_stop(reason)

    def _warn_throttled(self, message: str) -> None:
        now = time.monotonic()
        if now - self._last_error_at >= 2.0:
            self._last_error_at = now
            self.node.get_logger().warning(message)
