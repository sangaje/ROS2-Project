"""Load and validate the frozen ACTIVE_SCOUT policy contract."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import subprocess
from typing import Any, Optional


CONTRACT_FILENAME = 'scout_rl_policy_contract.json'


class PolicyContractError(RuntimeError):
    """A model, environment, or checkpoint violates the frozen contract."""


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
    """Return the source contract path, which is also symlink-installed."""
    path = workspace_root() / 'src' / 'system_bringup' / 'config' / CONTRACT_FILENAME
    if not path.is_file():
        raise PolicyContractError(f'RL policy contract is missing: {path}')
    return path


def load_contract(path: Optional[Path] = None) -> dict[str, Any]:
    """Read the contract and reject missing safety-critical fields."""
    resolved = Path(path) if path is not None else contract_path()
    try:
        data = json.loads(resolved.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError) as exc:
        raise PolicyContractError(f'invalid RL policy contract {resolved}: {exc}') from exc
    required_paths = (
        ('source_of_truth', 'training_runner'),
        ('source_of_truth', 'model_path'),
        ('runtime', 'python_executable'),
        ('policy', 'deterministic_inference'),
        ('policy', 'sde_inference'),
        ('observation_contract', 'keys'),
        ('action_contract', 'shape'),
    )
    for parts in required_paths:
        value: Any = data
        for part in parts:
            if not isinstance(value, dict) or part not in value:
                raise PolicyContractError(
                    f'RL policy contract missing {".".join(parts)}'
                )
            value = value[part]
    if data['policy']['deterministic_inference'] is not True:
        raise PolicyContractError('deterministic_inference must be true')
    if data['policy']['sde_inference'] is not False:
        raise PolicyContractError('sde_inference must be false')
    return data


def resolve_workspace_path(relative_path: str) -> Path:
    """Resolve a contract-owned file and keep it inside the workspace."""
    root = workspace_root()
    candidate = (root / str(relative_path)).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise PolicyContractError(f'contract path escapes workspace: {candidate}') from exc
    return candidate


def validate_static_assets(contract: Optional[dict[str, Any]] = None) -> dict[str, Path]:
    """Fail fast on missing runner, checkpoint, or compatible interpreter."""
    data = load_contract() if contract is None else contract
    source = data['source_of_truth']
    runner = resolve_workspace_path(source['training_runner'])
    checkpoint = resolve_workspace_path(source['model_path'])
    # Keep the venv symlink path intact.  Resolving it to /usr/bin/python3
    # silently drops the venv's NumPy 2 checkpoint compatibility.
    python_executable = Path(
        data['runtime']['python_executable']
    ).expanduser().absolute()
    failures = []
    if not runner.is_file():
        failures.append(f'training runner missing: {runner}')
    if not checkpoint.is_file() or checkpoint.stat().st_size <= 0:
        failures.append(f'checkpoint missing/empty: {checkpoint}')
    if not python_executable.is_file() or not os.access(python_executable, os.X_OK):
        failures.append(f'RL Python is not executable: {python_executable}')
    if failures:
        raise PolicyContractError('; '.join(failures))
    return {
        'runner': runner,
        'checkpoint': checkpoint,
        'python': python_executable,
    }


def run_checkpoint_preflight(
    contract: Optional[dict[str, Any]] = None,
    timeout_sec: float = 30.0,
) -> str:
    """Probe the checkpoint with its declared NumPy-compatible interpreter."""
    data = load_contract() if contract is None else contract
    assets = validate_static_assets(data)
    environment = os.environ.copy()
    environment.update(inference_environment(data))
    environment.setdefault('MPLCONFIGDIR', '/tmp/system_bringup_mplconfig')
    command = [
        str(assets['python']),
        '-m',
        'system_bringup.rl_policy_contract',
        '--probe-checkpoint',
    ]
    try:
        result = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=max(1.0, float(timeout_sec)),
            env=environment,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise PolicyContractError(f'checkpoint preflight could not run: {exc}') from exc
    output = '\n'.join(
        part.strip() for part in (result.stdout, result.stderr) if part.strip()
    )
    if result.returncode != 0:
        raise PolicyContractError(
            f'checkpoint preflight exit={result.returncode}: {output}'
        )
    if 'RL_POLICY_PREFLIGHT_PASS' not in output:
        raise PolicyContractError(f'checkpoint preflight returned no PASS marker: {output}')
    return output


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
    return {
        'shape': list(getattr(space, 'shape', ())),
        'dtype': str(getattr(space, 'dtype', '')),
    }


def probe_checkpoint(
    contract: Optional[dict[str, Any]] = None,
    model=None,
) -> dict[str, Any]:
    """Load the checkpoint once and compare model metadata byte-for-byte-ish."""
    data = load_contract() if contract is None else contract
    assets = validate_static_assets(data)
    if model is None:
        try:
            from stable_baselines3 import SAC
        except ImportError as exc:
            raise PolicyContractError(f'stable_baselines3 unavailable: {exc}') from exc
        try:
            model = SAC.load(str(assets['checkpoint']), device='cpu')
        except Exception as exc:  # noqa: BLE001
            raise PolicyContractError(
                f'checkpoint load failed with {Path(os.sys.executable)}: {exc}'
            ) from exc

    expected_obs = data['observation_contract']
    actual_obs = _space_contract(model.observation_space)
    expected_by_key = {
        key: {
            'shape': list(expected_obs[key]['shape']),
            'dtype': expected_obs[key]['dtype'],
        }
        for key in expected_obs['keys']
    }
    failures = []
    if list(actual_obs) != list(expected_obs['keys']):
        failures.append(
            f'observation keys expected={expected_obs["keys"]} actual={list(actual_obs)}'
        )
    if actual_obs != expected_by_key:
        failures.append(f'observation space expected={expected_by_key} actual={actual_obs}')

    action = data['action_contract']
    actual_action = _space_contract(model.action_space)
    expected_action = {
        'shape': list(action['shape']),
        'dtype': action['dtype'],
    }
    if actual_action != expected_action:
        failures.append(
            f'action space expected={expected_action} actual={actual_action}'
        )
    low = [float(value) for value in model.action_space.low.tolist()]
    high = [float(value) for value in model.action_space.high.tolist()]
    expected_low = [float(value) for value in action['low']]
    expected_high = [float(value) for value in action['high']]
    if len(low) != len(expected_low) or any(
        not math.isclose(left, right, abs_tol=1.0e-6)
        for left, right in zip(low, expected_low)
    ):
        failures.append(f'action low expected={expected_low} actual={low}')
    if len(high) != len(expected_high) or any(
        not math.isclose(left, right, abs_tol=1.0e-6)
        for left, right in zip(high, expected_high)
    ):
        failures.append(f'action high expected={expected_high} actual={high}')

    checkpoint_use_sde = bool(getattr(model, 'use_sde', False))
    expected_training_sde = bool(data['policy']['checkpoint_training_use_sde'])
    if checkpoint_use_sde != expected_training_sde:
        failures.append(
            'checkpoint use_sde expected='
            f'{expected_training_sde} actual={checkpoint_use_sde}'
        )
    actor_extractor = type(model.actor.features_extractor)
    extractor_name = f'{actor_extractor.__module__}.{actor_extractor.__name__}'
    expected_extractor = data['policy']['feature_extractor']
    if extractor_name != expected_extractor:
        failures.append(
            f'feature extractor expected={expected_extractor} actual={extractor_name}'
        )
    if failures:
        raise PolicyContractError('; '.join(failures))
    return {
        'checkpoint': str(assets['checkpoint']),
        'observation_space': actual_obs,
        'action_space': {
            **actual_action,
            'low': low,
            'high': high,
        },
        'checkpoint_training_use_sde': checkpoint_use_sde,
        'deterministic_inference': True,
        'sde_inference': False,
        'feature_extractor': extractor_name,
    }


def inference_environment(contract: Optional[dict[str, Any]] = None) -> dict[str, str]:
    """Return the frozen environment overrides for the inference process."""
    data = load_contract() if contract is None else contract
    environment = data.get('environment', {})
    return {str(key): str(value) for key, value in environment.items()}


def inference_command(contract: Optional[dict[str, Any]] = None) -> list[str]:
    """Build the model-sensitive CLI exclusively from the frozen contract."""
    data = load_contract() if contract is None else contract
    assets = validate_static_assets(data)
    observation = data['observation_contract']
    action = data['action_contract']
    runtime = data['runtime']
    temporal = observation['temporal']
    features = observation['feature_dimensions']
    command = [
        str(assets['python']), '-m', 'turtlebot3_rl_training.eval_policy',
        '--model', str(assets['checkpoint']),
        '--real-robot',
        '--episodes', '1000000000',
        '--no-manual-reset-prompt',
        '--no-real-robot-stop-between-episodes',
        '--control-dt', str(runtime['control_dt_sec']),
        '--physics-step-size', '0.005',
        '--max-episode-steps', '1000000000',
        '--entity-name', 'burger',
        '--set-pose-service', '/world/default/set_pose',
        '--world-control-service', '/world/default/control',
        '--action-mode', 'velocity',
        '--cmd-vel-topic', '/cmd_vel',
        '--max-linear-speed', str(action['high'][0]),
        '--max-angular-speed', str(action['high'][1]),
        '--velocity-command-linear-limit', str(action['high'][0]),
        '--velocity-command-angular-limit', str(action['high'][1]),
        '--linear-deadband', str(action['linear_deadband']),
        '--angular-deadband', str(action['angular_deadband']),
        '--action-smoothing-alpha', str(action['smoothing_alpha']),
        '--velocity-safety-backup',
        '--velocity-safety-trigger-distance-m', str(action['safety']['trigger_distance_m']),
        '--velocity-safety-stop-distance-m', str(action['safety']['stop_distance_m']),
        '--velocity-safety-slow-distance-m', str(action['safety']['slow_distance_m']),
        '--velocity-safety-backup-speed-mps', str(action['safety']['backup_speed_mps']),
        '--velocity-safety-turn-speed', str(action['safety']['turn_speed']),
        '--velocity-safety-backup-steps', str(action['safety']['backup_steps']),
        '--velocity-safety-cooldown-steps', str(action['safety']['cooldown_steps']),
        '--no-velocity-safety-slowdown',
        '--slam-backend', 'cartographer',
        '--slam-map-topic', '/map',
        '--map-frame', 'map',
        '--pose-frame', 'map',
        '--safety-boundary-frame', 'odom',
        '--wait-slam-map',
        '--strict-slam-map-required',
        '--strict-slam-map-wait-timeout-sec', '60.0',
        '--strict-slam-map-min-known-cells', '80',
        '--strict-slam-map-min-known-ratio', '0.003',
        '--post-reset-ready-gate',
        '--post-reset-ready-timeout-sec', '12.0',
        '--post-reset-ready-min-known-ratio', '0.003',
        '--post-reset-ready-min-known-cells', '40',
        '--post-reset-ready-min-lidar-beams', '30',
        '--no-post-reset-ready-require-priority',
        '--rl-map-topic', '/rl_task_map',
        '--rl-confidence-topic', '/rl_confidence_map',
        '--rl-priority-topic', '',
        '--rl-filtered-slam-topic', '',
        '--waypoint-marker-topic', '',
        '--map-publish-every-n', '1',
        '--map-live-update-period-sec', str(runtime['map_live_update_period_sec']),
        '--map-keepalive-period-sec', str(runtime['map_keepalive_period_sec']),
        '--use-map-cnn',
        '--map-obs-size', str(observation['map']['shape'][1]),
        '--map-obs-size-m', str(observation['map']['crop_size_m']),
        '--num-lidar-bins', str(observation['vector']['lidar_bins']),
        '--use-temporal-cnn',
        '--temporal-history-len', str(temporal['history_len']),
        '--temporal-features-dim', str(features['temporal']),
        '--cnn-features-dim', str(features['map']),
        '--vector-features-dim', str(features['vector']),
        '--combined-features-dim', str(features['combined']),
        '--real-robot-disable-priority',
        '--disable-priority-map',
        '--front-fov-deg', '60.0',
        '--front-angle-sigma-deg', '20.0',
        '--confidence-max-range', '2.0',
        '--seen-confidence-floor', str(runtime['seen_confidence_floor']),
        '--confidence-decay-per-step', '0.0',
        '--lidar-empty-restart',
        '--lidar-empty-timeout-sec', '2.5',
        '--lidar-empty-grace-sec', '1.0',
        '--lidar-empty-min-valid-beams', '2',
        '--collision-threshold', '0.14',
        '--restart-on-collision',
        '--no-terminate-on-out-of-bounds',
        '--safety-boundary-radius-m', '8.0',
        '--no-check-env',
    ]
    return command


def main() -> int:
    """CLI used by system_bringup's startup preflight."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--probe-checkpoint', action='store_true')
    args = parser.parse_args()
    try:
        result = probe_checkpoint() if args.probe_checkpoint else validate_static_assets()
    except PolicyContractError as exc:
        print(f'RL_POLICY_PREFLIGHT_FAILED | {exc}', flush=True)
        return 2
    serializable = {key: str(value) for key, value in result.items()}
    if args.probe_checkpoint:
        serializable = result
    print('RL_POLICY_PREFLIGHT_PASS | ' + json.dumps(serializable, sort_keys=True), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
