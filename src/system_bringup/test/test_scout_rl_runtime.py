from pathlib import Path
import threading

import numpy as np
from nav_msgs.msg import OccupancyGrid
from sensor_msgs.msg import LaserScan

from system_bringup.rl_policy_contract import active_scout_config
from system_bringup.scout_rl_runtime import (
    ActiveScoutRLRuntime,
    RuntimeCounters,
    SensorSnapshot,
    VelocitySafetyFilter,
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
        slam_map=object(), map_received_at=__import__('time').monotonic(), map_generation=1,
    )
    runtime._fresh = lambda snapshot, now: True
    runtime._build_observation = lambda scan, map_snapshot: {
        'map': np.zeros((4, 64, 64), dtype=np.float32),
        'map_seq': np.zeros((8, 4, 64, 64), dtype=np.float32),
        'seq': np.zeros((8, 69), dtype=np.float32),
        'vector': np.zeros((69,), dtype=np.float32),
    }
    runtime.safety = type('Safety', (), {'filter': lambda self, action, scan: action})()
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
