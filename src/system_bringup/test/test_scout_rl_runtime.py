from pathlib import Path

import numpy as np
from sensor_msgs.msg import LaserScan

from system_bringup.rl_policy_contract import active_scout_config
from system_bringup.scout_rl_runtime import VelocitySafetyFilter
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
    def publish(self, message):
        pass


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


def test_deployment_runtime_has_no_process_or_nondeterministic_predict_path():
    source = Path(__file__).parents[1] / 'system_bringup' / 'scout_rl_runtime.py'
    text = source.read_text(encoding='utf-8')

    assert 'import subprocess' not in text
    assert 'Popen(' not in text
    assert 'deterministic=True' in text
    assert 'reset_noise' not in text
