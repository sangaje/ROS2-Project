from pathlib import Path
import threading

import numpy as np
from nav_msgs.msg import OccupancyGrid, Odometry
from sensor_msgs.msg import LaserScan

from system_bringup.rl_policy_contract import active_scout_config
from system_bringup.scout_rl_runtime import (
    ActiveScoutRLRuntime,
    RuntimeCounters,
    SensorSnapshot,
    VelocitySafetyFilter,
)
from system_bringup.scout_rl_policy_worker import (
    GateInputs,
    RLWorkerState,
    evaluate_activation_gate,
    parse_role_update,
)
from turtlebot3_rl_training.exploration_map import ExplorationGridMap
from turtlebot3_rl_training.observation import LidarPreprocessorConfig, downsample_lidar


def _scan(front_distance=3.0):
    message = LaserScan()
    message.angle_min = -np.pi
    message.angle_increment = 2.0 * np.pi / 360.0
    message.angle_max = np.pi - message.angle_increment
    message.range_min = 0.12
    message.range_max = 3.5
    message.ranges = [3.0] * 360
    message.ranges[180] = front_distance
    return message


def _odom():
    message = Odometry()
    message.header.frame_id = 'odom'
    message.child_frame_id = 'base_footprint'
    message.pose.pose.orientation.w = 1.0
    return message


def test_runtime_keeps_v132_backup_sequence_nonblocking():
    config = active_scout_config()
    lidar = LidarPreprocessorConfig(
        canonical_front_zero=config.lidar.canonical_front_zero,
        front_index=config.lidar.front_index,
        angle_offset_deg=config.lidar.angle_offset_deg,
        flip_lr=config.lidar.flip_lr,
        uniform_angle_resample=config.lidar.uniform_angle_resample,
        median_kernel=config.lidar.median_kernel,
        lowpass_kernel=config.lidar.lowpass_kernel,
        obstacle_margin_m=config.lidar.obstacle_margin_m,
    )
    safety = VelocitySafetyFilter(config, lidar)

    first = safety.filter(np.array([0.2, 0.0], dtype=np.float32), _scan(0.15))
    remaining = [
        safety.filter(np.array([0.2, 0.0], dtype=np.float32), _scan(0.15))
        for _ in range(config.safety_backup_steps - 1)
    ]

    assert np.isclose(first[0], -config.safety_backup_speed_mps)
    assert all(np.isclose(command[0], -config.safety_backup_speed_mps) for command in remaining)
    assert safety.backup_remaining == 0
    assert safety.cooldown_remaining == config.safety_cooldown_steps


def test_runtime_backs_up_briefly_when_front_is_already_pinned():
    config = active_scout_config()
    lidar = LidarPreprocessorConfig(
        canonical_front_zero=config.lidar.canonical_front_zero,
        front_index=config.lidar.front_index,
        angle_offset_deg=config.lidar.angle_offset_deg,
        flip_lr=config.lidar.flip_lr,
        uniform_angle_resample=config.lidar.uniform_angle_resample,
        median_kernel=config.lidar.median_kernel,
        lowpass_kernel=config.lidar.lowpass_kernel,
        obstacle_margin_m=config.lidar.obstacle_margin_m,
    )
    safety = VelocitySafetyFilter(config, lidar)

    first = safety.filter(np.array([0.0, 0.0], dtype=np.float32), _scan(0.15))
    second = safety.filter(np.array([0.0, 0.0], dtype=np.float32), _scan(0.15))

    assert np.allclose(first, [-config.safety_backup_speed_mps, 0.0])
    assert np.allclose(second, [-config.safety_backup_speed_mps, 0.0])
    assert safety.backup_remaining == 0
    assert safety.cooldown_remaining == config.safety_cooldown_steps


def test_raw_policy_command_only_clips_to_the_trained_action_box():
    runtime = ActiveScoutRLRuntime.__new__(ActiveScoutRLRuntime)
    runtime.config = active_scout_config()

    command = runtime._raw_policy_command(np.array([0.10, 0.03], dtype=np.float32))

    # The raw diagnostic path intentionally does not apply the 0.04rad/s
    # angular deadband used by VelocitySafetyFilter.
    assert np.allclose(command, [0.10, 0.03])


def test_contract_lidar_config_ignores_mutable_environment(monkeypatch):
    monkeypatch.setenv('TB3_RL_LIDAR_FLIP_LR', '1')
    frozen = LidarPreprocessorConfig(
        canonical_front_zero=True,
        front_index=0,
        angle_offset_deg=0.0,
        flip_lr=False,
        uniform_angle_resample=True,
        median_kernel=3,
        lowpass_kernel=5,
        obstacle_margin_m=0.08,
    )
    scan = _scan()
    result = downsample_lidar(
        scan.ranges,
        num_bins=60,
        min_range=scan.range_min,
        max_range=scan.range_max,
        scan_angle_min=scan.angle_min,
        scan_angle_increment=scan.angle_increment,
        scan_angle_max=scan.angle_max,
        config=frozen,
    )

    assert result.shape == (60,)
    assert np.all(np.isfinite(result))


class _Publisher:
    def __init__(self):
        self.messages = []

    def publish(self, message):
        self.messages.append(message)


class _Logger:
    def info(self, message):
        pass


class _MapNode:
    def create_publisher(self, *args, **kwargs):
        return _Publisher()

    def create_timer(self, *args, **kwargs):
        return object()

    def get_logger(self):
        return _Logger()


def test_deployment_confidence_map_uses_contract_not_environment(monkeypatch):
    monkeypatch.setenv('TB3_RL_CONFIDENCE_LIDAR_OCCLUSION_RADIUS_CELLS', '0')
    config = active_scout_config()
    grid = ExplorationGridMap(
        _MapNode(),
        disable_priority_map=True,
        deployment_mode=True,
        confidence_decay_near_obstacle_scale=config.confidence_decay_near_obstacle_scale,
        confidence_obstacle_ring_radius=config.confidence_obstacle_ring_radius_cells,
        confidence_obstacle_floor_ratio=config.confidence_obstacle_floor_ratio,
        confidence_lidar_occlusion_radius_cells=config.confidence_lidar_occlusion_radius_cells,
        lidar_policy_config=LidarPreprocessorConfig(),
    )

    assert grid.disable_priority_map is True
    assert grid.confidence_lidar_occlusion_radius_cells == 3
    assert grid.confidence_decay_near_obstacle_scale == 0.0
    assert grid.confidence_decay_obstacle_ring_radius == 5
    assert grid.confidence_obstacle_floor_ratio == 1.0


def test_confidence_map_updates_continuously_for_twenty_policy_cycles():
    config = active_scout_config()
    node = _MapNode()
    grid = ExplorationGridMap(
        node,
        resolution=config.map_resolution_m,
        size_m=8.0,
        origin_x=-4.0,
        origin_y=-4.0,
        frame_id='map',
        publish_topic='/rl_task_map',
        confidence_publish_topic='/rl_confidence_map',
        priority_publish_topic='',
        disable_priority_map=True,
        filtered_slam_publish_topic='',
        legacy_memory_publish_topic='',
        publish_every_n=1,
        lidar_stride=2,
        use_slam_prior=True,
        deployment_mode=True,
    )
    slam_map = OccupancyGrid()
    slam_map.header.frame_id = 'map'
    slam_map.info.resolution = config.map_resolution_m
    slam_map.info.width = 160
    slam_map.info.height = 160
    slam_map.info.origin.position.x = -4.0
    slam_map.info.origin.position.y = -4.0
    slam_map.info.origin.orientation.w = 1.0
    slam_map.data = [-1] * (slam_map.info.width * slam_map.info.height)

    for _ in range(20):
        grid.update(
            _scan(),
            np.array([0.0, 0.0], dtype=np.float32),
            0.0,
            publish=True,
            slam_map=slam_map,
            sensor_xy=np.array([0.0, 0.0], dtype=np.float32),
            sensor_yaw=0.0,
        )

    assert grid.update_count == 20
    assert len(grid.confidence_pub.messages) >= 20


def test_first_predict_exception_does_not_deactivate_active_scout():
    class _FlakyModel:
        def __init__(self):
            self.calls = 0

        def predict(self, observation, deterministic):
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError('transient inference failure')
            return np.array([0.1, 0.0], dtype=np.float32), None

    runtime = ActiveScoutRLRuntime.__new__(ActiveScoutRLRuntime)
    runtime.config = active_scout_config()
    runtime._active = True
    runtime.model = _FlakyModel()
    runtime._model_lock = threading.Lock()
    runtime._model_error = None
    runtime._map_state_lock = threading.Lock()
    runtime._map_snapshot = type('MapSnapshot', (), {'updated_at': __import__('time').monotonic()})()
    runtime._sensor_snapshot = lambda: SensorSnapshot(
        scan=_scan(), scan_received_at=__import__('time').monotonic(), scan_generation=1,
        odom=_odom(), odom_received_at=__import__('time').monotonic(),
        odom_generation=1, odom_source_stamp_age_ms=5.0,
        slam_map=object(), map_received_at=__import__('time').monotonic(), map_generation=1,
    )
    runtime._fresh = lambda snapshot, now: True
    runtime._build_observation = lambda scan, map_snapshot: {
        'map': np.zeros((4, 64, 64), dtype=np.float32),
        'map_seq': np.zeros((8, 4, 64, 64), dtype=np.float32),
        'seq': np.zeros((8, 63), dtype=np.float32),
        'vector': np.zeros((63,), dtype=np.float32),
    }
    runtime.safety = type('Safety', (), {'filter': lambda self, action, scan: action})()
    runtime.enable_velocity_safety_filter = True
    runtime.counters = RuntimeCounters()
    runtime._last_error = ''
    runtime._previous_action = np.zeros(2, dtype=np.float32)
    runtime._last_command_at = 0.0
    runtime._warn_throttled = lambda message: None
    holds = []
    runtime._hold = holds.append
    commands = []
    runtime.publish_command = lambda linear, angular: commands.append((linear, angular))
    runtime._log_heartbeat = lambda: None
    runtime._log_policy_tick = lambda **kwargs: None
    runtime._log_inference = lambda **kwargs: None

    runtime._policy_tick()
    runtime._policy_tick()

    assert runtime.active is True
    assert holds == ['inference_error']
    assert runtime.counters.predict_failure_count == 1
    assert runtime.counters.predict_success_count == 1
    assert len(commands) == 1
    assert np.isclose(commands[0][0], 0.1)
    assert commands[0][1] == 0.0


def test_command_watchdog_holds_and_retries_without_deactivating_active_scout():
    runtime = ActiveScoutRLRuntime.__new__(ActiveScoutRLRuntime)
    runtime.config = active_scout_config()
    runtime._active = True
    runtime.model = object()
    runtime._model_lock = threading.Lock()
    runtime._model_error = None
    runtime._map_snapshot = object()
    now = __import__('time').monotonic()
    runtime._activated_at = now - runtime.config.command_timeout_sec - 0.1
    runtime._last_command_at = now - runtime.config.command_timeout_sec - 0.1
    runtime._warn_throttled = lambda message: None
    holds = []
    runtime._hold = holds.append

    runtime._command_watchdog()

    assert runtime.active is True
    assert holds == ['command_timeout']


def test_runtime_warmup_updates_observation_without_motion_activation():
    runtime = (
        Path(__file__).parents[1]
        / 'system_bringup'
        / 'scout_rl_runtime.py'
    ).read_text(encoding='utf-8')
    worker = (
        Path(__file__).parents[1]
        / 'system_bringup'
        / 'scout_rl_policy_worker.py'
    ).read_text(encoding='utf-8')

    assert 'def warmup(self, reason: str = ' in runtime
    assert 'self._sensor_pipeline_enabled = True' in runtime
    assert 'if not self._sensor_pipeline_enabled:' in runtime
    assert 'publish=self._active' in runtime
    assert 'self._warm_observation(snapshot.scan)' in runtime
    assert 'if state == RLWorkerState.ACTIVE and not self.start_motion' in worker
    assert 'self.runtime.warmup(' in worker
    assert 'SCOUT_STARTUP_PIPELINE |' in worker
    assert 'SCOUT_OBSERVATION_PIPELINE |' in worker


def test_deployment_runtime_has_no_process_or_nondeterministic_predict_path():
    source = Path(__file__).parents[1] / 'system_bringup' / 'scout_rl_runtime.py'
    text = source.read_text(encoding='utf-8')

    assert 'import subprocess' not in text
    assert 'Popen(' not in text
    assert 'deterministic=True' in text
    assert 'reset_noise' not in text


def test_standalone_policy_worker_is_registered_as_the_separate_process():
    setup = (Path(__file__).parents[1] / 'setup.py').read_text(encoding='utf-8')
    worker = Path(__file__).parents[1] / 'system_bringup' / 'scout_rl_policy_worker.py'

    assert worker.is_file()
    assert 'scout_rl_policy_worker = system_bringup.scout_rl_policy_worker:main' in setup


def test_scout_rl_worker_requires_leader_start_motion_gate():
    worker = Path(__file__).parents[1] / 'system_bringup' / 'scout_rl_policy_worker.py'
    source = worker.read_text(encoding='utf-8')

    assert "require_start_motion" in source
    assert "start_motion_topic" in source
    assert "self.require_start_motion = True" in source
    assert "self.start_motion = False" in source
    assert "self.create_subscription(Bool, self.start_motion_topic" in source
    assert "def _on_start_motion" in source
    assert "if not self.start_motion" in source
    assert "self._publish_zero()" in source
    assert "require_video_ready" in source
    assert "video_ready_topic" in source
    assert "def _on_video_ready" in source
    assert "sensor_ready = self.runtime.sensor_ready()" in source


def test_scout_rl_worker_waits_for_global_system_ready_gate():
    worker = Path(__file__).parents[1] / 'system_bringup' / 'scout_rl_policy_worker.py'
    source = worker.read_text(encoding='utf-8')

    assert "require_system_ready" in source
    assert "system_ready_topic" in source
    assert "self.create_subscription(Bool, self.system_ready_topic" in source
    assert "def _on_system_ready" in source
    assert "not self.system_ready" in source
    assert "self._publish_zero()" in source


def _gate(**overrides):
    values = {
        'role': 'ACTIVE_SCOUT',
        'role_robot_matches': True,
        'role_epoch': 1,
        'failover_epoch': 1,
        'active_scout_matches': True,
        'failover_state': 'NEW_SCOUT_EXPLORING',
        'localization_ready': True,
        'recovery_complete': True,
        'nav_goal_inactive': True,
        'motion_authority': 'NONE',
        'model_ready': True,
        'sensor_ready': True,
        'tf_ready': True,
        'require_failover_activation': True,
        'require_localization_ready': True,
    }
    values.update(overrides)
    return GateInputs(**values)


def test_worker_gate_blocks_recovery_navigation_before_rl_activation():
    state, reason = evaluate_activation_gate(_gate(
        failover_state='RECOVERY_NAVIGATING',
        motion_authority='FAILOVER_RECOVERY_NAV',
    ))

    assert state == RLWorkerState.RECOVERY_NAVIGATING
    assert reason == 'recovery_or_non_scout_role'


def test_worker_gate_manual_active_scout_skips_failover_localization_gate():
    state, reason = evaluate_activation_gate(_gate(
        role_epoch=0,
        failover_epoch=1,
        active_scout_matches=False,
        failover_state='RECOVERY_NAVIGATING',
        localization_ready=False,
        recovery_complete=False,
        motion_authority='FAILOVER_RECOVERY_NAV',
        require_failover_activation=False,
        require_localization_ready=False,
    ))

    assert state == RLWorkerState.ACTIVE
    assert reason == 'all_runtime_inputs_ready'


def test_worker_gate_manual_mode_still_requires_active_scout_role():
    state, reason = evaluate_activation_gate(_gate(
        role='FOLLOWER',
        require_failover_activation=False,
    ))

    assert state == RLWorkerState.RECOVERY_NAVIGATING
    assert reason == 'recovery_or_non_scout_role'


def test_worker_gate_rejects_stale_active_scout_role_epoch():
    state, reason = evaluate_activation_gate(_gate(role_epoch=0, failover_epoch=1))

    assert state == RLWorkerState.STANDBY
    assert reason == 'stale_epoch'


def test_worker_gate_requires_localization_motion_release_and_runtime_inputs():
    assert evaluate_activation_gate(_gate(localization_ready=False))[0] == RLWorkerState.WAIT_LOCALIZATION
    assert evaluate_activation_gate(_gate(nav_goal_inactive=False))[0] == RLWorkerState.WAIT_MOTION_RELEASE
    assert evaluate_activation_gate(_gate(sensor_ready=False))[0] == RLWorkerState.WAIT_SENSOR_READY


def test_worker_gate_accepts_only_fully_ready_failover_owner():
    state, reason = evaluate_activation_gate(_gate())

    assert state == RLWorkerState.ACTIVE
    assert reason == 'all_runtime_inputs_ready'


def test_worker_gate_waits_for_observation_ready_even_when_raw_sensors_are_fresh():
    # Regression test for the exact contradiction reported on real hardware:
    # role/lease/motion/model/sensor/tf all pass, but the internal MapSnapshot
    # the RL policy predicts from is still stale. The gate must not report
    # ACTIVE in that case.
    state, reason = evaluate_activation_gate(_gate(observation_ready=False))

    assert state == RLWorkerState.WAIT_OBSERVATION_READY
    assert reason == 'observation_stale'


def test_gate_inputs_default_observation_ready_true_for_existing_callers():
    # Callers built before observation_ready existed (e.g. this test file's
    # own _gate() helper) must keep their prior ACTIVE-when-otherwise-ready
    # behavior without having to be updated for the new field.
    gate = _gate()
    assert gate.observation_ready is True


def test_fast_observation_tick_never_recomputes_the_heavy_confidence_grid():
    # The fast tick must only ever reuse whatever _confidence_tick last
    # produced; it must never itself call exploration_map.update() (the
    # CPU-heavy call that used to be fused into a single 10 Hz callback and
    # caused the reported snapshot-age stalls).
    deque = __import__('collections').deque
    runtime = ActiveScoutRLRuntime.__new__(ActiveScoutRLRuntime)
    runtime.config = active_scout_config()
    runtime._sensor_pipeline_enabled = True
    runtime._active = False
    runtime.counters = RuntimeCounters()
    runtime._last_fast_tick_mono = 0.0
    runtime._last_map_tick_timing_log_at = 0.0
    runtime._fast_interval_ms_samples = deque(maxlen=100)
    runtime._fast_tf_ms_samples = deque(maxlen=100)
    runtime._fast_lock_ms_samples = deque(maxlen=100)
    runtime._fast_total_ms_samples = deque(maxlen=100)
    runtime._confidence_update_ms_samples = deque(maxlen=100)
    runtime._map_state_lock = threading.Lock()
    runtime._map_snapshot = None
    slam_map = OccupancyGrid()
    slam_map.header.frame_id = runtime.config.map_frame
    scan = _scan()
    scan.header.frame_id = runtime.config.scan_frame
    now = __import__('time').monotonic()
    runtime._sensor_snapshot = lambda: SensorSnapshot(
        scan=scan, scan_received_at=now, scan_generation=1,
        odom=_odom(), odom_received_at=now, odom_generation=1,
        odom_source_stamp_age_ms=5.0,
        slam_map=slam_map, map_received_at=now, map_generation=1,
    )
    runtime._fresh = lambda snapshot, now: True
    runtime._lookup_pose = lambda *args, **kwargs: (np.zeros(2, dtype=np.float32), 0.0)
    runtime._warn_throttled = lambda message: None
    runtime._hold = lambda reason: None
    runtime._log_heartbeat = lambda: None
    runtime._warm_observation = lambda scan: None

    update_calls = []
    runtime.exploration_map = type(
        'Grid', (), {'update': staticmethod(lambda *a, **k: update_calls.append(1) or object())}
    )()
    runtime._merge_confidence_seed_locked = lambda: None

    # Before any confidence tick has ever produced stats, the fast tick must
    # not fabricate a snapshot out of nothing.
    runtime._latest_stats = None
    runtime._fast_observation_tick()
    assert runtime._map_snapshot is None
    assert update_calls == []

    # Once a confidence tick has produced stats, the fast tick commits a
    # fresh snapshot on its own cadence without ever calling .update() itself.
    runtime._latest_stats = object()
    runtime._fast_observation_tick()
    assert runtime._map_snapshot is not None
    assert update_calls == []


def test_worker_role_parser_supports_json_and_simple_strings():
    update = parse_role_update(
        '{"role":"ACTIVE_SCOUT","robot":"follower21","epoch":2,'
        '"active_scout_id":"follower21","localization_ready":true,'
        '"recovery_complete":true}',
        'follower21',
    )
    assert update.role == 'ACTIVE_SCOUT'
    assert update.robot == 'follower21'
    assert update.epoch == 2
    assert update.active_scout_id == 'follower21'
    assert update.localization_ready is True
    assert update.recovery_complete is True

    simple = parse_role_update('RECOVERY_NAVIGATING', 'follower21')
    assert simple.role == 'RECOVERY_NAVIGATING'
    assert simple.robot == 'follower21'
    assert simple.active_scout_id is None


def test_worker_cartographer_mode_does_not_require_amcl_ready_signal():
    state, reason = evaluate_activation_gate(_gate(
        localization_ready=False,
        require_localization_ready=False,
    ))

    assert state == RLWorkerState.ACTIVE
    assert reason == 'all_runtime_inputs_ready'


def test_runtime_map_subscription_accepts_cartographer_volatile_maps():
    source = (Path(__file__).parents[1] / 'system_bringup' / 'scout_rl_runtime.py').read_text(
        encoding='utf-8'
    )

    assert 'durability=DurabilityPolicy.VOLATILE' in source


def test_runtime_tf_gate_uses_a_short_probe_timeout_before_activation():
    source = (Path(__file__).parents[1] / 'system_bringup' / 'scout_rl_runtime.py').read_text(
        encoding='utf-8'
    )

    assert 'probe_timeout_sec = min(self.config.max_tf_age_sec, 0.05)' in source
    assert 'timeout_sec=probe_timeout_sec' in source


def test_hardware_contract_leaves_budget_for_slow_map_and_inference_callbacks():
    config = active_scout_config()

    assert config.map_substeps_per_action == 2
    assert config.max_scan_age_sec == 0.8
    assert config.max_odom_age_sec == 0.8
    assert config.max_map_age_sec == 5.0
    assert config.max_inference_sec == 2.0
    assert config.command_timeout_sec == 3.0
    assert config.odom_topic == '/odom'


def test_observation_snapshot_and_confidence_periods_are_bounded_and_derived():
    config = active_scout_config()
    fast_tick_period_sec = config.control_dt_sec / config.map_substeps_per_action

    # Deliberately not tied to max_scan_age_sec: this is the freshness bound
    # for the derived MapSnapshot, not for raw scan/odom/map messages.
    # Floor raised from 0.6s->1.5s after real-hardware SCOUT_MAP_TICK_TIMING
    # telemetry showed the heavy confidence tick's own update() cost (an
    # external, unmodifiable turtlebot3_rl_training call) spiking past 1s.
    assert config.max_observation_snapshot_age_sec >= 1.5
    assert config.max_observation_snapshot_age_sec <= fast_tick_period_sec * 20.0
    assert config.confidence_update_period_sec >= 1.5
    # The heavy confidence/publish pipeline must run at a bounded, slower
    # cadence than the fast observation tick, not on every fast tick.
    assert config.confidence_update_period_sec > fast_tick_period_sec


def test_runtime_publishes_the_exact_policy_lidar_debug_topic():
    source = (Path(__file__).parents[1] / 'system_bringup' / 'scout_rl_runtime.py').read_text(
        encoding='utf-8'
    )

    assert "'/rl_policy_scan_60'" in source


def test_runtime_accepts_takeover_confidence_seed_without_rl_package_changes():
    source = (Path(__file__).parents[1] / 'system_bringup' / 'scout_rl_runtime.py').read_text(
        encoding='utf-8'
    )

    assert "'/rl_confidence_seed'" in source
    assert 'def _on_confidence_seed' in source
    assert 'def _merge_confidence_seed_locked' in source
    assert 'np.maximum(target, merged, out=target)' in source
    assert 'SCOUT_RL_CONFIDENCE_SEED_APPLIED' in source
    assert 'def _publish_policy_scan_from_raw' in source


def test_scout_rl_worker_logs_recoverable_runtime_gate_debug():
    source = (
        Path(__file__).parents[1]
        / 'system_bringup'
        / 'scout_rl_policy_worker.py'
    ).read_text(encoding='utf-8')
    runtime = (
        Path(__file__).parents[1]
        / 'system_bringup'
        / 'scout_rl_runtime.py'
    ).read_text(encoding='utf-8')

    assert 'SCOUT_RL_DEBUG |' in source
    assert 'SCOUT_RL_GATE |' in source
    assert 'blocking_reason=' in source
    assert 'startup_not_released' in source
    assert 'start_motion_false' in source
    assert 'raw_action_linear=' in source
    assert 'raw_action_nonzero=' in source
    assert 'scan_stale' in source
    assert 'odom_stale' in source
    assert 'map_stale' in source
    assert 'policy_worker_dead' in source
    assert 'SCOUT_ODOM_DEBUG |' in source
    assert 'blocking_inputs=' in source
    assert 'sensor_pipeline_enabled=' in source
    assert 'SCOUT_RL_RESUME_REQUEST |' in source
    assert 'self.runtime.hold(reason)' in source
    assert 'def debug_snapshot' in runtime
    assert 'self.odom_sub = self.node.create_subscription' in runtime
    assert 'Odometry, self.config.odom_topic' in runtime
    assert 'def hold(self, reason: str)' in runtime
    assert "'observation_ready'" in runtime
    assert "'inference_age_ms'" in runtime
