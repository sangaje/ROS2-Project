"""Typed source of truth for the in-process ACTIVE_SCOUT SAC policy."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
from typing import Any, Optional


CONTRACT_FILENAME = 'scout_rl_policy_contract.json'


class PolicyContractError(RuntimeError):
    """A model, configuration, or checkpoint violates the frozen contract."""


@dataclass(frozen=True)
class LidarPolicyConfig:
    canonical_front_zero: bool
    front_index: int
    angle_offset_deg: float
    flip_lr: bool
    uniform_angle_resample: bool
    median_kernel: int
    lowpass_kernel: int
    obstacle_margin_m: float


@dataclass(frozen=True)
class ActiveScoutPolicyConfig:
    checkpoint: Path
    control_dt_sec: float
    map_substeps_per_action: int
    max_scan_age_sec: float
    max_map_age_sec: float
    max_tf_age_sec: float
    max_inference_sec: float
    command_timeout_sec: float
    scan_topic: str
    map_topic: str
    map_frame: str
    base_frame: str
    scan_frame: str
    map_resolution_m: float
    map_initial_size_m: float
    map_publish_every_n: int
    map_keepalive_period_sec: float
    clear_confidence_on_slam_occupied: bool
    confidence_occupied_confirm_steps: int
    confidence_decay_near_obstacle_scale: float
    confidence_obstacle_ring_radius_cells: int
    confidence_obstacle_floor_ratio: float
    confidence_lidar_hit_guard_m: float
    confidence_lidar_occlusion_radius_cells: int
    map_crop_size_m: float
    map_obs_size: int
    history_len: int
    lidar_bins: int
    lidar: LidarPolicyConfig
    action_low: tuple[float, float]
    action_high: tuple[float, float]
    linear_deadband: float
    angular_deadband: float
    smoothing_alpha: float
    safety_trigger_distance_m: float
    safety_stop_distance_m: float
    safety_slow_distance_m: float
    safety_backup_speed_mps: float
    safety_turn_speed: float
    safety_backup_steps: int
    safety_cooldown_steps: int
    safety_slowdown: bool
    safety_slow_min_scale: float


def workspace_root() -> Path:
    """Return the workspace containing this source-installed package."""
    override = os.environ.get('ROS2_PROJECT_ROOT', '').strip()
    if override:
        candidate = Path(override).expanduser().resolve()
        if (candidate / 'src' / 'system_bringup').is_dir():
            return candidate
        raise PolicyContractError(
            f'ROS2_PROJECT_ROOT does not contain src/system_bringup: {candidate}'
        )
    source_candidate = Path(__file__).resolve().parents[3]
    if (source_candidate / 'src' / 'system_bringup').is_dir():
        return source_candidate
    current = Path.cwd().resolve()
    for candidate in (current, *current.parents):
        if (candidate / 'src' / 'system_bringup').is_dir():
            return candidate
    raise PolicyContractError('cannot resolve ROS2 workspace root')


def contract_path() -> Path:
    path = workspace_root() / 'src' / 'system_bringup' / 'config' / CONTRACT_FILENAME
    if not path.is_file():
        raise PolicyContractError(f'RL policy contract is missing: {path}')
    return path


def _require(data: dict[str, Any], *parts: str) -> Any:
    value: Any = data
    for part in parts:
        if not isinstance(value, dict) or part not in value:
            raise PolicyContractError(
                f'RL policy contract missing {".".join(parts)}'
            )
        value = value[part]
    return value


def _validate_contract(data: dict[str, Any]) -> dict[str, Any]:
    for parts in (
        ('source_of_truth', 'training_runner'),
        ('source_of_truth', 'model_path'),
        ('policy', 'deterministic_inference'),
        ('policy', 'sde_inference'),
        ('runtime', 'control_dt_sec'),
        ('runtime', 'map_substeps_per_action'),
        ('runtime', 'max_inference_sec'),
        ('runtime', 'command_timeout_sec'),
        ('runtime', 'map', 'confidence'),
        ('runtime', 'lidar'),
        ('observation_contract', 'keys'),
        ('action_contract', 'shape'),
    ):
        _require(data, *parts)
    if data['policy']['deterministic_inference'] is not True:
        raise PolicyContractError('deterministic_inference must be true')
    if data['policy']['sde_inference'] is not False:
        raise PolicyContractError('sde_inference must be false')
    if int(data['runtime']['map_substeps_per_action']) != 2:
        raise PolicyContractError('map_substeps_per_action must match v132 value 2')
    return data


def load_contract(path: Optional[Path] = None) -> dict[str, Any]:
    """Read the immutable policy contract and validate deployment essentials."""
    resolved = Path(path) if path is not None else contract_path()
    try:
        data = json.loads(resolved.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError) as exc:
        raise PolicyContractError(f'invalid RL policy contract {resolved}: {exc}') from exc
    return _validate_contract(data)


def resolve_workspace_path(relative_path: str) -> Path:
    root = workspace_root()
    candidate = (root / str(relative_path)).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise PolicyContractError(f'contract path escapes workspace: {candidate}') from exc
    return candidate


def validate_static_assets(contract: Optional[dict[str, Any]] = None) -> dict[str, Path]:
    """Verify source provenance and the checkpoint without spawning a process."""
    data = load_contract() if contract is None else _validate_contract(contract)
    runner = resolve_workspace_path(data['source_of_truth']['training_runner'])
    checkpoint = resolve_workspace_path(data['source_of_truth']['model_path'])
    failures = []
    if not runner.is_file():
        failures.append(f'training runner missing: {runner}')
    if not checkpoint.is_file() or checkpoint.stat().st_size <= 0:
        failures.append(f'checkpoint missing/empty: {checkpoint}')
    if failures:
        raise PolicyContractError('; '.join(failures))
    return {'runner': runner, 'checkpoint': checkpoint}


def active_scout_config(
    contract: Optional[dict[str, Any]] = None,
) -> ActiveScoutPolicyConfig:
    """Compile JSON once into the values allowed on the deployment hot path."""
    data = load_contract() if contract is None else contract
    assets = validate_static_assets(data)
    runtime = data['runtime']
    observation = data['observation_contract']
    action = data['action_contract']
    safety = action['safety']
    lidar = runtime['lidar']
    topics = runtime['topics']
    map_config = runtime['map']
    confidence = map_config['confidence']
    low = tuple(float(value) for value in action['low'])
    high = tuple(float(value) for value in action['high'])
    if len(low) != 2 or len(high) != 2:
        raise PolicyContractError('velocity policy action bounds must have two values')
    if observation['keys'] != ['map', 'map_seq', 'seq', 'vector']:
        raise PolicyContractError('checkpoint requires map/map_seq/seq/vector observations')
    if int(observation['map']['shape'][0]) != 4:
        raise PolicyContractError('checkpoint requires the four-channel no-priority map')
    return ActiveScoutPolicyConfig(
        checkpoint=assets['checkpoint'],
        control_dt_sec=float(runtime['control_dt_sec']),
        map_substeps_per_action=int(runtime['map_substeps_per_action']),
        max_scan_age_sec=float(runtime['max_scan_age_sec']),
        max_map_age_sec=float(runtime['max_map_age_sec']),
        max_tf_age_sec=float(runtime['max_tf_age_sec']),
        max_inference_sec=float(runtime['max_inference_sec']),
        command_timeout_sec=float(runtime['command_timeout_sec']),
        scan_topic=str(topics['scan']),
        map_topic=str(topics['map']),
        map_frame=str(topics['map_frame']),
        base_frame=str(topics['base_frame']),
        scan_frame=str(topics['scan_frame']),
        map_resolution_m=float(map_config['resolution_m']),
        map_initial_size_m=float(map_config['initial_size_m']),
        map_publish_every_n=int(map_config['publish_every_n']),
        map_keepalive_period_sec=float(map_config['keepalive_period_sec']),
        clear_confidence_on_slam_occupied=bool(confidence['clear_on_slam_occupied']),
        confidence_occupied_confirm_steps=int(confidence['occupied_confirm_steps']),
        confidence_decay_near_obstacle_scale=float(confidence['decay_near_obstacle_scale']),
        confidence_obstacle_ring_radius_cells=int(confidence['obstacle_ring_radius_cells']),
        confidence_obstacle_floor_ratio=float(confidence['obstacle_floor_ratio']),
        confidence_lidar_hit_guard_m=float(confidence['lidar_hit_guard_m']),
        confidence_lidar_occlusion_radius_cells=int(confidence['lidar_occlusion_radius_cells']),
        map_crop_size_m=float(observation['map']['crop_size_m']),
        map_obs_size=int(observation['map']['shape'][1]),
        history_len=int(observation['temporal']['history_len']),
        lidar_bins=int(observation['vector']['lidar_bins']),
        lidar=LidarPolicyConfig(
            canonical_front_zero=bool(lidar['canonical_front_zero']),
            front_index=int(lidar['front_index']),
            angle_offset_deg=float(lidar['angle_offset_deg']),
            flip_lr=bool(lidar['flip_lr']),
            uniform_angle_resample=bool(lidar['uniform_angle_resample']),
            median_kernel=int(lidar['median_kernel']),
            lowpass_kernel=int(lidar['lowpass_kernel']),
            obstacle_margin_m=float(lidar['obstacle_margin_m']),
        ),
        action_low=(low[0], low[1]),
        action_high=(high[0], high[1]),
        linear_deadband=float(action['linear_deadband']),
        angular_deadband=float(action['angular_deadband']),
        smoothing_alpha=float(action['smoothing_alpha']),
        safety_trigger_distance_m=float(safety['trigger_distance_m']),
        safety_stop_distance_m=float(safety['stop_distance_m']),
        safety_slow_distance_m=float(safety['slow_distance_m']),
        safety_backup_speed_mps=float(safety['backup_speed_mps']),
        safety_turn_speed=float(safety['turn_speed']),
        safety_backup_steps=int(safety['backup_steps']),
        safety_cooldown_steps=int(safety['cooldown_steps']),
        safety_slowdown=bool(safety['slowdown']),
        safety_slow_min_scale=float(safety['slow_min_scale']),
    )


def _space_contract(space) -> dict[str, Any]:
    spaces = getattr(space, 'spaces', None)
    if isinstance(spaces, dict):
        return {
            str(key): {
                'shape': list(getattr(value, 'shape', ())),
                'dtype': str(getattr(value, 'dtype', '')),
            }
            for key, value in spaces.items()
        }
    return {'shape': list(getattr(space, 'shape', ())), 'dtype': str(getattr(space, 'dtype', ''))}


def probe_checkpoint(
    contract: Optional[dict[str, Any]] = None,
    model=None,
) -> dict[str, Any]:
    """Load the checkpoint once and compare its immutable metadata."""
    data = load_contract() if contract is None else contract
    assets = validate_static_assets(data)
    if model is None:
        try:
            from stable_baselines3 import SAC
        except ImportError as exc:
            raise PolicyContractError(f'stable_baselines3 unavailable: {exc}') from exc
        try:
            model = SAC.load(str(assets['checkpoint']), device='cpu', buffer_size=1)
        except Exception as exc:  # noqa: BLE001
            raise PolicyContractError(f'checkpoint load failed: {exc}') from exc

    expected_obs = data['observation_contract']
    actual_obs = _space_contract(model.observation_space)
    expected_by_key = {
        key: {'shape': list(expected_obs[key]['shape']), 'dtype': expected_obs[key]['dtype']}
        for key in expected_obs['keys']
    }
    failures = []
    if list(actual_obs) != list(expected_obs['keys']):
        failures.append(f'observation keys expected={expected_obs["keys"]} actual={list(actual_obs)}')
    if actual_obs != expected_by_key:
        failures.append(f'observation space expected={expected_by_key} actual={actual_obs}')
    action = data['action_contract']
    actual_action = _space_contract(model.action_space)
    if actual_action != {'shape': list(action['shape']), 'dtype': action['dtype']}:
        failures.append(f'action space expected={action["shape"]} actual={actual_action}')
    low = [float(value) for value in model.action_space.low.tolist()]
    high = [float(value) for value in model.action_space.high.tolist()]
    if not all(math.isclose(a, b, abs_tol=1.0e-6) for a, b in zip(low, action['low'])):
        failures.append(f'action low expected={action["low"]} actual={low}')
    if not all(math.isclose(a, b, abs_tol=1.0e-6) for a, b in zip(high, action['high'])):
        failures.append(f'action high expected={action["high"]} actual={high}')
    if bool(getattr(model, 'use_sde', False)) != bool(data['policy']['checkpoint_training_use_sde']):
        failures.append('checkpoint use_sde does not match policy contract')
    extractor = type(model.actor.features_extractor)
    extractor_name = f'{extractor.__module__}.{extractor.__name__}'
    if extractor_name != data['policy']['feature_extractor']:
        failures.append(f'feature extractor expected={data["policy"]["feature_extractor"]} actual={extractor_name}')
    if failures:
        raise PolicyContractError('; '.join(failures))
    return {
        'checkpoint': str(assets['checkpoint']),
        'observation_space': actual_obs,
        'action_space': {**actual_action, 'low': low, 'high': high},
        'checkpoint_training_use_sde': bool(getattr(model, 'use_sde', False)),
        'deterministic_inference': True,
        'sde_inference': False,
        'feature_extractor': extractor_name,
    }


def main() -> int:
    """Manual, in-process checkpoint audit; never called by launch."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--probe-checkpoint', action='store_true')
    args = parser.parse_args()
    try:
        result = probe_checkpoint() if args.probe_checkpoint else validate_static_assets()
    except PolicyContractError as exc:
        print(f'RL_POLICY_PREFLIGHT_FAILED | {exc}', flush=True)
        return 2
    print('RL_POLICY_PREFLIGHT_PASS | ' + json.dumps(result, default=str, sort_keys=True), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
