import argparse
import atexit
import os
import subprocess
import sys
import time
import math
import re
import types
import warnings
import zipfile
from collections import deque
from pathlib import Path
from typing import Optional

import rclpy
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.buffers import DictReplayBuffer
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import get_schedule_fn
from turtlebot3_rl_training.feature_extractor import MapVectorFeatureExtractor
from turtlebot3_rl_training.gazebo_nav_env import GazeboNavEnv
from turtlebot3_rl_training.ros_interface import TurtleBot3RosInterface

# v133: SB3's own get_schedule_fn()/constant_fn() deprecation warnings fire on
# every run regardless of our code; they are library-internal noise, not
# something we can fix here, and clutter the quiet "[SAC] ..." progress output
# the user wants. Placed *after* all imports (some packages reset the warnings
# filter list as a side effect of import) so nothing downstream can silently
# undo this.
warnings.filterwarnings("ignore", message=".*get_schedule_fn.*deprecated.*")
warnings.filterwarnings("ignore", message=".*constant_fn.*deprecated.*")



def _terminate_process(proc: Optional[subprocess.Popen], label: str = "process") -> None:
    if proc is None:
        return
    if proc.poll() is not None:
        return
    try:
        print(f"[AUTO LAUNCH] Stopping {label}...")
        proc.terminate()
        proc.wait(timeout=3.0)
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass


def _reset_sac_entropy_coefficient(model: SAC, ent_coef: Optional[float], logger=None) -> bool:
    if ent_coef is None:
        return False
    value = float(ent_coef)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError("--sac-reset-ent-coef must be a finite positive value")

    try:
        import torch as th
    except Exception as exc:
        raise RuntimeError(f"torch import failed while resetting SAC entropy coefficient: {exc}") from exc

    if getattr(model, "log_ent_coef", None) is not None:
        with th.no_grad():
            target = th.log(th.ones_like(model.log_ent_coef.data) * value)
            model.log_ent_coef.data.copy_(target)
        optimizer = getattr(model, "ent_coef_optimizer", None)
        if optimizer is not None:
            optimizer.state.clear()
        if logger is not None:
            logger.info(f"SAC_ENT_COEF_RESET | learned alpha reset to {value:.6f}; optimizer state cleared")
        return True

    if getattr(model, "ent_coef_tensor", None) is not None:
        model.ent_coef_tensor = th.tensor(value, device=getattr(model, "device", "cpu"))
        model.ent_coef = value
        if logger is not None:
            logger.info(f"SAC_ENT_COEF_RESET | fixed alpha set to {value:.6f}")
        return True

    if logger is not None:
        logger.warn("SAC_ENT_COEF_RESET_SKIPPED | model has no entropy coefficient attribute to reset")
    return False

def _reset_sac_critics(model: SAC, logger=None) -> bool:
    """Reinitialize SAC critics while keeping the loaded actor/extractor setup."""
    policy = getattr(model, "policy", None)
    critic = getattr(policy, "critic", None) if policy is not None else None
    critic_target = getattr(policy, "critic_target", None) if policy is not None else None
    if critic is None:
        if logger is not None:
            logger.warn("SAC_CRITIC_RESET_SKIPPED | model policy has no critic")
        return False

    def _reset_module(module) -> None:
        reset = getattr(module, "reset_parameters", None)
        if callable(reset):
            reset()

    try:
        critic.apply(_reset_module)
        if critic_target is not None:
            critic_target.load_state_dict(critic.state_dict())
        optimizer = getattr(critic, "optimizer", None)
        if optimizer is not None:
            optimizer.state.clear()
        if logger is not None:
            logger.warn("SAC_CRITIC_RESET | critic and critic_target reinitialized; actor kept")
        return True
    except Exception as exc:
        if logger is not None:
            logger.warn(f"SAC_CRITIC_RESET_FAILED | {type(exc).__name__}: {exc}")
        return False


def _install_warmup_action_mixer(
    model: SAC,
    *,
    warmup_steps: int,
    zero_linear_prob: float,
    random_prob: float,
    noise_prob: float,
    noise_std: float,
    logger=None,
) -> bool:
    steps = max(int(warmup_steps), 0)
    zero_linear_probability = min(max(float(zero_linear_prob), 0.0), 1.0)
    random_probability = min(max(float(random_prob), 0.0), max(1.0 - zero_linear_probability, 0.0))
    noise_probability = min(max(float(noise_prob), 0.0), max(1.0 - zero_linear_probability - random_probability, 0.0))
    std = max(float(noise_std), 0.0)
    if steps <= 0 or (zero_linear_probability <= 0.0 and random_probability <= 0.0 and (noise_probability <= 0.0 or std <= 0.0)):
        return False

    try:
        import numpy as np
    except Exception as exc:
        raise RuntimeError(f"numpy import failed while installing warmup action mixer: {exc}") from exc

    original_sample_action = getattr(model, "_tb3_original_sample_action", None)
    if original_sample_action is None:
        original_sample_action = model._sample_action
        model._tb3_original_sample_action = original_sample_action

    rng = np.random.default_rng()
    start_timesteps = int(getattr(model, "num_timesteps", 0))
    model._tb3_warmup_action_start_timesteps = start_timesteps
    model._tb3_warmup_action_steps = steps
    model._tb3_warmup_action_zero_linear_prob = zero_linear_probability
    model._tb3_warmup_action_random_prob = random_probability
    model._tb3_warmup_action_noise_prob = noise_probability
    model._tb3_warmup_action_noise_std = std
    model._tb3_warmup_action_zero_linear_count = 0
    model._tb3_warmup_action_random_count = 0
    model._tb3_warmup_action_noise_count = 0
    model._tb3_last_warmup_action_source = "policy"

    def _sample_action_with_warmup(self, learning_starts: int, action_noise=None, n_envs: int = 1):
        action, buffer_action = original_sample_action(learning_starts, action_noise, n_envs)
        elapsed = int(getattr(self, "num_timesteps", 0)) - int(getattr(self, "_tb3_warmup_action_start_timesteps", 0))
        if elapsed < 0 or elapsed >= int(getattr(self, "_tb3_warmup_action_steps", 0)):
            self._tb3_last_warmup_action_source = "policy"
            return action, buffer_action

        action_arr = np.asarray(action).copy()
        buffer_arr = np.asarray(buffer_action).copy()
        env_count = max(int(n_envs), 1)
        draws = rng.random(env_count)
        zero_linear_prob_now = float(getattr(self, "_tb3_warmup_action_zero_linear_prob", 0.0))
        random_cutoff = zero_linear_prob_now + float(getattr(self, "_tb3_warmup_action_random_prob", 0.0))
        noise_cutoff = random_cutoff + float(getattr(self, "_tb3_warmup_action_noise_prob", 0.0))
        zero_linear_mask = draws < zero_linear_prob_now
        random_mask = (draws >= zero_linear_prob_now) & (draws < random_cutoff)
        noisy_mask = (draws >= random_cutoff) & (draws < noise_cutoff)

        if np.any(zero_linear_mask):
            sampled_action = np.asarray([self.action_space.sample() for _ in range(env_count)], dtype=np.float32)
            sampled_action[:, 0] = 0.0
            sampled_buffer_action = self.policy.scale_action(sampled_action)
            action_arr[zero_linear_mask] = sampled_action[zero_linear_mask]
            buffer_arr[zero_linear_mask] = sampled_buffer_action[zero_linear_mask]
            self._tb3_warmup_action_zero_linear_count = int(getattr(self, "_tb3_warmup_action_zero_linear_count", 0)) + int(np.count_nonzero(zero_linear_mask))

        if np.any(random_mask):
            sampled_action = np.asarray([self.action_space.sample() for _ in range(env_count)], dtype=np.float32)
            sampled_buffer_action = self.policy.scale_action(sampled_action)
            action_arr[random_mask] = sampled_action[random_mask]
            buffer_arr[random_mask] = sampled_buffer_action[random_mask]
            self._tb3_warmup_action_random_count = int(getattr(self, "_tb3_warmup_action_random_count", 0)) + int(np.count_nonzero(random_mask))

        if np.any(noisy_mask):
            noise = rng.normal(0.0, float(getattr(self, "_tb3_warmup_action_noise_std", 0.0)), size=buffer_arr[noisy_mask].shape)
            noisy_buffer_action = np.clip(buffer_arr[noisy_mask] + noise, -1.0, 1.0)
            buffer_arr[noisy_mask] = noisy_buffer_action
            action_arr[noisy_mask] = self.policy.unscale_action(noisy_buffer_action)
            self._tb3_warmup_action_noise_count = int(getattr(self, "_tb3_warmup_action_noise_count", 0)) + int(np.count_nonzero(noisy_mask))

        if np.count_nonzero([np.any(zero_linear_mask), np.any(random_mask), np.any(noisy_mask)]) > 1:
            self._tb3_last_warmup_action_source = "mixed"
        elif np.any(zero_linear_mask):
            self._tb3_last_warmup_action_source = "zero_linear"
        elif np.any(random_mask):
            self._tb3_last_warmup_action_source = "random"
        elif np.any(noisy_mask):
            self._tb3_last_warmup_action_source = "noisy"
        else:
            self._tb3_last_warmup_action_source = "policy"

        return action_arr, buffer_arr

    model._sample_action = types.MethodType(_sample_action_with_warmup, model)
    if logger is not None:
        logger.info(
            "WARMUP_ACTION_MIXER | "
            f"resume_start={start_timesteps} steps={steps} "
            f"zero_linear_prob={zero_linear_probability:.3f} "
            f"random_prob={random_probability:.3f} noise_prob={noise_probability:.3f} noise_std={std:.3f}"
        )
    return True


def _start_gazebo_if_requested(cli_args) -> Optional[subprocess.Popen]:
    """Optionally start Gazebo/TurtleBot3 simulation from train/eval itself."""
    if not getattr(cli_args, "auto_start_gazebo", False):
        return None

    env = os.environ.copy()
    model = str(getattr(cli_args, "gazebo_turtlebot_model", "") or "").strip()
    if model:
        env["TURTLEBOT3_MODEL"] = model

    launch_package = str(getattr(cli_args, "gazebo_launch_package", "turtlebot3_gazebo") or "turtlebot3_gazebo").strip()
    launch_file = str(getattr(cli_args, "gazebo_launch_file", "turtlebot3_house.launch.py") or "turtlebot3_house.launch.py").strip()

    cmd = ["ros2", "launch", launch_package, launch_file]

    world_arg = str(getattr(cli_args, "gazebo_world", "") or "").strip()
    if world_arg:
        cmd.append(f"world:={world_arg}")

    use_sim_time = bool(getattr(cli_args, "gazebo_use_sim_time", True))
    cmd.append(f"use_sim_time:={'true' if use_sim_time else 'false'}")

    extra_args = getattr(cli_args, "gazebo_extra_arg", []) or []
    for arg in extra_args:
        arg = str(arg).strip()
        if arg:
            cmd.append(arg)

    print("[AUTO LAUNCH] Starting Gazebo internally:")
    print(" ".join(cmd))

    stdout = None if bool(getattr(cli_args, "gazebo_show_output", False)) else subprocess.DEVNULL
    stderr = None if bool(getattr(cli_args, "gazebo_show_output", False)) else subprocess.DEVNULL

    proc = subprocess.Popen(
        cmd,
        stdout=stdout,
        stderr=stderr,
        text=True,
        env=env,
        start_new_session=True,
    )
    atexit.register(_terminate_process, proc, "Gazebo")

    wait_sec = max(float(getattr(cli_args, "gazebo_startup_wait_sec", 4.0)), 0.0)
    if wait_sec > 0.0:
        time.sleep(wait_sec)
    return proc


def _scan_geometry_counts(scan_msg, *, max_valid_range_m: float = 3.35) -> tuple[int, int, float]:
    try:
        ranges = list(getattr(scan_msg, "ranges", []) or [])
    except Exception:
        ranges = []
    total = len(ranges)
    if total <= 0:
        return 0, 0, 999.0

    try:
        range_min = max(float(getattr(scan_msg, "range_min", 0.05) or 0.05), 0.0)
    except Exception:
        range_min = 0.05
    try:
        range_max = float(getattr(scan_msg, "range_max", 3.5) or 3.5)
    except Exception:
        range_max = 3.5
    max_valid = min(float(max_valid_range_m), max(range_max - 0.05, range_min + 1e-3))

    valid = 0
    nearest = 999.0
    for value in ranges:
        try:
            r = float(value)
        except Exception:
            continue
        if not math.isfinite(r):
            continue
        nearest = min(nearest, r)
        if range_min <= r <= max_valid:
            valid += 1
    return int(valid), int(total), float(nearest)


def _wait_for_training_geometry_ready(ros, cli_args) -> bool:
    """Reject non-trainable startup states such as an empty plane world.

    /scan and /odom existing is not enough for this exploration task.  The reward
    depends on obstacle/frontier/SLAM geometry; if almost every LiDAR ray is
    max-range, the agent receives mostly stall penalties and learns nonsense.
    """
    if not bool(getattr(cli_args, "startup_require_training_geometry", True)):
        return True

    timeout_sec = max(float(getattr(cli_args, "startup_geometry_wait_sec", 8.0)), 0.0)
    min_beams = max(int(getattr(cli_args, "startup_min_obstacle_beams", 20)), 1)
    max_valid_range = max(float(getattr(cli_args, "startup_max_valid_lidar_range_m", 3.35)), 0.1)
    start = time.time()
    last_counts = (0, 0, 999.0)

    while time.time() - start <= timeout_sec:
        rclpy.spin_once(ros, timeout_sec=0.05)
        scan = getattr(ros, "scan", None)
        if scan is not None:
            last_counts = _scan_geometry_counts(scan, max_valid_range_m=max_valid_range)
            valid, total, nearest = last_counts
            if total > 0 and valid >= min_beams:
                print(
                    "[TRAIN STARTUP] LiDAR geometry OK: "
                    f"valid_obstacle_beams={valid}/{total}, nearest={nearest:.3f}m",
                    flush=True,
                )
                return True
        time.sleep(0.05)

    valid, total, nearest = last_counts
    print(
        "\n[TRAIN STARTUP ERROR] LiDAR sees too little obstacle geometry for exploration training.\n"
        f"  valid_obstacle_beams={valid}/{total}, nearest={nearest:.3f}m, required>={min_beams}\n"
        "This usually means Gazebo is running an empty plane world or the robot spawned outside the training map.\n"
        "Use terminal 1 with the RL training world:\n"
        "  cd ~/Desktop/ROS2_Project\n"
        "  bash run_gazebo.sh\n"
        "The default simulator world should print:\n"
        "  world: .../src/turtlebot3_rl_training/world/training_house.sdf\n"
        "To bypass this guard only for diagnostics, pass --no-startup-require-training-geometry.\n",
        flush=True,
    )
    return False


def _model_step_from_name(path: Path) -> int:
    match = re.search(r"_(\d+)_steps(?:\.zip)?$", path.stem)
    if match:
        return int(match.group(1))
    try:
        return int(path.stat().st_mtime)
    except Exception:
        return -1


def _is_valid_zip_model(path: Path) -> bool:
    try:
        if not path.is_file() or path.stat().st_size <= 0:
            return False
        with zipfile.ZipFile(path) as archive:
            return archive.testzip() is None
    except Exception:
        return False


def _sb3_zip_path(base_path: Path) -> Path:
    path = Path(base_path)
    if path.suffix == ".zip":
        return path
    return Path(str(path) + ".zip")


def _detach_warmup_action_mixer_for_save(model) -> dict:
    """Temporarily remove runtime monkey-patched methods before SB3/cloudpickle save."""
    state = {}
    model_dict = getattr(model, "__dict__", None)
    if not isinstance(model_dict, dict):
        return state
    for attr in ("_sample_action", "_tb3_original_sample_action"):
        if attr in model_dict:
            state[attr] = model_dict.pop(attr)
    return state


def _restore_warmup_action_mixer_after_save(model, state: dict) -> None:
    for attr, value in state.items():
        try:
            setattr(model, attr, value)
        except Exception:
            pass


def _safe_save_sac_model(model, base_path: Path, logger=None) -> Path:
    """Save an SB3 model through a validated temp zip before replacing target."""
    target_base = Path(base_path)
    target_zip = _sb3_zip_path(target_base)
    target_zip.parent.mkdir(parents=True, exist_ok=True)
    tmp_zip = target_zip.parent / f".{target_zip.stem}.tmp_{os.getpid()}_{int(time.time() * 1000)}.zip"
    warmup_state = _detach_warmup_action_mixer_for_save(model)
    try:
        model.save(str(tmp_zip))
        if not _is_valid_zip_model(tmp_zip):
            size = tmp_zip.stat().st_size if tmp_zip.exists() else -1
            raise IOError(f"temporary model zip is invalid: {tmp_zip} size={size}")
        os.replace(tmp_zip, target_zip)
        return target_zip
    except BaseException:
        try:
            if tmp_zip.exists() and not _is_valid_zip_model(tmp_zip):
                tmp_zip.unlink()
        except Exception:
            pass
        raise
    finally:
        _restore_warmup_action_mixer_after_save(model, warmup_state)
        try:
            if tmp_zip.exists() and tmp_zip != target_zip:
                tmp_zip.unlink()
        except Exception:
            pass


def _find_latest_sac_model(model_dir: Path) -> Optional[Path]:
    candidates = []
    for pattern in ("sac_turtlebot3_burger_checkpoint_*_steps.zip", "sac_turtlebot3_burger.zip"):
        candidates.extend(model_dir.glob(pattern))
    candidates = [p for p in candidates if _is_valid_zip_model(p)]
    if not candidates:
        return None
    return max(candidates, key=lambda p: (_model_step_from_name(p), p.stat().st_mtime))


def _guess_replay_buffer_path(model_path: Path, model_dir: Path) -> Optional[Path]:
    explicit_final = model_dir / "sac_turtlebot3_burger_replay_buffer.pkl"
    if explicit_final.exists():
        return explicit_final

    stem = model_path.stem
    match = re.search(r"_(\d+)_steps$", stem)
    if match:
        step = match.group(1)
        same_step = model_dir / f"sac_turtlebot3_burger_checkpoint_replay_buffer_{step}_steps.pkl"
        if same_step.exists():
            return same_step

    candidates = list(model_dir.glob("sac_turtlebot3_burger_checkpoint_replay_buffer_*_steps.pkl"))
    candidates += list(model_dir.glob("*_replay_buffer*.pkl"))
    candidates = [p for p in candidates if p.is_file()]
    if not candidates:
        return None
    return max(candidates, key=lambda p: (_model_step_from_name(p), p.stat().st_mtime))

try:
    from turtlebot3_rl_training.process_cleanup import (
        clean_fastdds_shm,
        ensure_non_shm_fastdds_profile,
    )
except Exception:  # pragma: no cover - allow running outside the package layout
    try:
        from process_cleanup import (  # type: ignore
            clean_fastdds_shm,
            ensure_non_shm_fastdds_profile,
        )
    except Exception:
        clean_fastdds_shm = None
        ensure_non_shm_fastdds_profile = None


def _apply_training_profile(cli_args):
    """Apply coarse-grained runtime profiles after argparse parsing.

    This avoids passing dozens of throughput flags by hand.  The profile only
    changes runtime/training speed knobs; explicit reset coordinates remain the
    user's CLI values.
    """
    profile = str(getattr(cli_args, "training_profile", "normal") or "normal").strip().lower()
    if profile in {"", "normal", "none"}:
        return cli_args

    if profile == "ultrafast":
        # Maximum throughput: no SLAM process, no RViz maps, no world-step barrier,
        # no macro-action inner loop, small policy.  The internal LiDAR memory map
        # and scalar priority/target statistics are still used by reward/observation.
        cli_args.disable_world_step = True
        cli_args.disable_slam_map = True
        cli_args.auto_start_slam = False
        cli_args.wait_slam_map = False
        cli_args.reset_slam_on_reset = False
        cli_args.restart_slam_on_reset = True
        cli_args.reset_slam_every_n_episodes = 0
        cli_args.reset_tf_buffer_on_reset = False
        cli_args.pose_frame = "odom"
        cli_args.safety_boundary_frame = "odom"
        cli_args.action_mode = "nav2"
        cli_args.waypoint_action_type = "polar"
        cli_args.waypoint_execute_until_reached = False
        cli_args.waypoint_control_steps = 1
        cli_args.waypoint_max_control_steps = 1
        cli_args.waypoint_visual_publish_every_n = 1_000_000
        cli_args.control_dt = 0.04
        cli_args.physics_step_size = 0.02
        cli_args.realtime_spin_steps = 1
        cli_args.realtime_spin_timeout_sec = 0.0
        cli_args.realtime_sleep_sec = 0.001
        cli_args.debug_input_map = False
        cli_args.rl_map_topic = ""
        cli_args.rl_confidence_topic = ""
        cli_args.rl_priority_topic = ""
        cli_args.rl_filtered_slam_topic = ""
        cli_args.rl_path_topic = ""
        cli_args.waypoint_marker_topic = ""
        cli_args.waypoint_path_topic = ""
        cli_args.map_publish_every_n = 0
        cli_args.map_keepalive_period_sec = 0.0
        cli_args.path_visual_publish_every_n = 0
        cli_args.use_map_cnn = False
        cli_args.use_temporal_cnn = False
        cli_args.map_obs_size = 32
        cli_args.map_obs_size_m = 5.0
        cli_args.cnn_features_dim = 32
        cli_args.vector_features_dim = 96
        cli_args.combined_features_dim = 128
        cli_args.temporal_features_dim = 32
        cli_args.temporal_history_len = 2
        cli_args.priority_recompute_interval = 32
        cli_args.train_freq_steps = 32
        cli_args.gradient_steps = 1
        cli_args.batch_size = min(int(cli_args.batch_size), 128)
        cli_args.buffer_size = min(int(cli_args.buffer_size), 50_000)
        cli_args.learning_starts = min(int(cli_args.learning_starts), 500)
        cli_args.debug_print_freq = 0
        cli_args.progress_print_freq = max(int(cli_args.progress_print_freq), 5000)
        cli_args.checkpoint_freq = max(int(cli_args.checkpoint_freq), 100000)
        cli_args.save_replay_buffer = False
        cli_args.save_vecnormalize = False
        return cli_args

    if profile == "fast":
        # Balanced profile: keep map CNN and internal priority map, but avoid the
        # expensive SLAM/RViz/world-step components during training.
        cli_args.disable_world_step = True
        cli_args.disable_slam_map = True
        cli_args.auto_start_slam = False
        cli_args.wait_slam_map = False
        cli_args.reset_slam_on_reset = False
        cli_args.restart_slam_on_reset = True
        cli_args.reset_slam_every_n_episodes = 0
        cli_args.reset_tf_buffer_on_reset = False
        cli_args.pose_frame = "odom"
        cli_args.safety_boundary_frame = "odom"
        cli_args.action_mode = "nav2"
        cli_args.waypoint_action_type = "polar"
        cli_args.waypoint_execute_until_reached = False
        cli_args.waypoint_control_steps = 1
        cli_args.waypoint_max_control_steps = 1
        cli_args.control_dt = 0.06
        cli_args.realtime_spin_steps = 2
        cli_args.realtime_spin_timeout_sec = 0.0
        cli_args.realtime_sleep_sec = 0.001
        cli_args.debug_input_map = False
        cli_args.rl_map_topic = ""
        cli_args.rl_confidence_topic = ""
        cli_args.rl_priority_topic = ""
        cli_args.rl_filtered_slam_topic = ""
        cli_args.rl_path_topic = ""
        cli_args.waypoint_marker_topic = ""
        cli_args.waypoint_path_topic = ""
        cli_args.map_publish_every_n = 0
        cli_args.map_keepalive_period_sec = 0.0
        cli_args.path_visual_publish_every_n = 0
        cli_args.use_map_cnn = True
        cli_args.use_temporal_cnn = False
        cli_args.map_obs_size = min(int(cli_args.map_obs_size), 40)
        cli_args.map_obs_size_m = min(float(cli_args.map_obs_size_m), 5.5)
        cli_args.cnn_features_dim = min(int(cli_args.cnn_features_dim), 40)
        cli_args.combined_features_dim = min(int(cli_args.combined_features_dim), 160)
        cli_args.priority_recompute_interval = max(int(cli_args.priority_recompute_interval), 24)
        cli_args.train_freq_steps = max(int(cli_args.train_freq_steps), 16)
        cli_args.gradient_steps = min(max(int(cli_args.gradient_steps), 1), 1)
        cli_args.debug_print_freq = 0
        cli_args.progress_print_freq = max(int(cli_args.progress_print_freq), 3000)
        return cli_args

    if profile == "visual_lite":
        # For checking RViz occasionally.  Still avoid episode-level SLAM reset.
        cli_args.disable_world_step = True
        cli_args.disable_slam_map = False
        cli_args.auto_start_slam = True
        cli_args.wait_slam_map = True
        # Do not override action_mode, pose_frame, safety frame, or SLAM reset
        # settings here.  The Nav2/SLAM-stable command intentionally supplies
        # --action-mode nav2, --pose-frame map, and --reset-slam-on-reset.
        cli_args.waypoint_execute_until_reached = False
        cli_args.realtime_spin_steps = 2
        cli_args.realtime_spin_timeout_sec = 0.0
        cli_args.realtime_sleep_sec = 0.001
        cli_args.debug_input_map = False
        cli_args.rl_map_topic = "/rl_task_map"
        cli_args.rl_confidence_topic = "/rl_confidence_map"
        cli_args.rl_priority_topic = "/rl_priority_map"
        cli_args.rl_filtered_slam_topic = "/rl_filtered_slam_map"
        cli_args.rl_path_topic = ""
        cli_args.waypoint_marker_topic = "/rl_waypoint_marker"
        cli_args.waypoint_path_topic = ""
        # Keep RViz layers live independently of waypoint/action updates.
        # The confidence/priority maps are recomputed from the latest scan+pose
        # at 10 Hz by GazeboNavEnv, not only when SAC emits a new waypoint.
        cli_args.map_publish_every_n = 1
        cli_args.priority_recompute_interval = 1
        cli_args.map_keepalive_period_sec = max(float(getattr(cli_args, "map_keepalive_period_sec", 0.0)), 0.10)
        cli_args.map_live_update_period_sec = max(float(getattr(cli_args, "map_live_update_period_sec", 0.10)), 0.10)
        cli_args.waypoint_visual_publish_every_n = max(int(cli_args.waypoint_visual_publish_every_n), 1)
        cli_args.path_visual_publish_every_n = 0
        # Do not override --use-temporal-cnn here.  The feature extractor always
        # uses a LiDAR 1D CNN; this flag controls the additional temporal 1D CNN
        # over recent vector observations.
        cli_args.progress_print_freq = max(int(cli_args.progress_print_freq), 3000)
        return cli_args



    if profile in {"real_finetune", "real_robot_finetune", "real"}:
        # Continue an existing pure-velocity SAC checkpoint under conditions closer
        # to the real TurtleBot: short control period, slower executed command, fast
        # live RL-map refresh, and stronger priority-target hysteresis.  The SAC
        # action_space is intentionally kept at the old wide bounds so checkpoint
        # loading remains valid; velocity_command_* limits clamp only the executed
        # TwistStamped inside GazeboNavEnv.
        cli_args.disable_world_step = True
        cli_args.disable_slam_map = False
        cli_args.auto_start_slam = True
        cli_args.wait_slam_map = True
        cli_args.action_mode = "velocity"
        cli_args.pose_frame = str(getattr(cli_args, "map_frame", "map") or "map")
        cli_args.safety_boundary_frame = "odom"
        cli_args.max_linear_speed = max(float(getattr(cli_args, "max_linear_speed", 0.22)), 0.22)
        cli_args.max_angular_speed = max(float(getattr(cli_args, "max_angular_speed", 0.85)), 0.85)
        cli_args.velocity_command_linear_limit = min(max(float(getattr(cli_args, "velocity_command_linear_limit", 0.10)), 0.05), 0.12)
        cli_args.velocity_command_angular_limit = min(max(float(getattr(cli_args, "velocity_command_angular_limit", 0.35)), 0.20), 0.45)
        cli_args.control_dt = min(float(getattr(cli_args, "control_dt", 0.05)), 0.05)
        cli_args.realtime_spin_steps = max(int(getattr(cli_args, "realtime_spin_steps", 3)), 3)
        cli_args.realtime_spin_timeout_sec = 0.0
        cli_args.realtime_sleep_sec = max(float(getattr(cli_args, "realtime_sleep_sec", 0.01)), 0.01)
        cli_args.action_smoothing_alpha = min(float(getattr(cli_args, "action_smoothing_alpha", 0.35)), 0.35)
        cli_args.max_linear_delta = min(float(getattr(cli_args, "max_linear_delta", 0.025)), 0.025)
        cli_args.max_angular_delta = min(float(getattr(cli_args, "max_angular_delta", 0.06)), 0.06)
        cli_args.linear_deadband = min(float(getattr(cli_args, "linear_deadband", 0.005)), 0.005)
        cli_args.angular_deadband = min(float(getattr(cli_args, "angular_deadband", 0.025)), 0.025)
        cli_args.use_map_cnn = True
        cli_args.use_temporal_cnn = True
        cli_args.map_obs_size = 48
        cli_args.map_obs_size_m = 6.0
        cli_args.map_live_update_period_sec = min(float(getattr(cli_args, "map_live_update_period_sec", 0.05)), 0.05)
        cli_args.map_publish_every_n = 1
        cli_args.map_keepalive_period_sec = min(max(float(getattr(cli_args, "map_keepalive_period_sec", 0.25)), 0.05), 0.25)
        cli_args.priority_recompute_interval = 1
        cli_args.priority_target_lock_steps = max(int(getattr(cli_args, "priority_target_lock_steps", 40)), 40)
        cli_args.priority_target_switch_margin = max(float(getattr(cli_args, "priority_target_switch_margin", 0.18)), 0.18)
        cli_args.priority_clear_fov_deg = 360.0
        cli_args.priority_clear_max_range_m = max(float(getattr(cli_args, "priority_clear_max_range_m", 2.50)), 2.50)
        cli_args.priority_clear_robot_radius_m = max(float(getattr(cli_args, "priority_clear_robot_radius_m", 0.45)), 0.45)
        cli_args.priority_clear_angle_sigma_deg = 180.0
        cli_args.priority_clear_min_weight = min(float(getattr(cli_args, "priority_clear_min_weight", 0.05)), 0.05)
        cli_args.corridor_priority_reward_weight = min(float(getattr(cli_args, "corridor_priority_reward_weight", 0.20)), 0.20)
        cli_args.confidence_reward_weight = max(float(getattr(cli_args, "confidence_reward_weight", 2.0)), 2.0)
        cli_args.slam_map_update_reward = True
        cli_args.slam_map_update_reward_weight = min(float(getattr(cli_args, "slam_map_update_reward_weight", 0.10)), 0.10)
        cli_args.slam_map_update_reward_norm_cells = max(float(getattr(cli_args, "slam_map_update_reward_norm_cells", 120)), 120)
        cli_args.slam_map_update_reward_cap = min(float(getattr(cli_args, "slam_map_update_reward_cap", 0.5)), 0.5)
        cli_args.velocity_safety_backup = True
        cli_args.velocity_safety_trigger_distance_m = min(float(getattr(cli_args, "velocity_safety_trigger_distance_m", 0.20)), 0.20)
        cli_args.velocity_safety_stop_distance_m = min(float(getattr(cli_args, "velocity_safety_stop_distance_m", 0.22)), 0.22)
        cli_args.velocity_safety_slow_distance_m = min(float(getattr(cli_args, "velocity_safety_slow_distance_m", 0.40)), 0.40)
        cli_args.velocity_forward_assist_mps = max(float(getattr(cli_args, "velocity_forward_assist_mps", 0.030)), 0.030)
        cli_args.velocity_forward_assist_angular_threshold = max(float(getattr(cli_args, "velocity_forward_assist_angular_threshold", 0.30)), 0.30)
        cli_args.velocity_forward_assist_min_clearance_m = max(float(getattr(cli_args, "velocity_forward_assist_min_clearance_m", 0.50)), 0.50)
        cli_args.velocity_spin_breaker = True
        cli_args.velocity_spin_breaker_steps = min(int(getattr(cli_args, "velocity_spin_breaker_steps", 10)), 10)
        cli_args.velocity_spin_breaker_angular_ratio = min(float(getattr(cli_args, "velocity_spin_breaker_angular_ratio", 0.80)), 0.80)
        cli_args.velocity_spin_breaker_forward_mps = max(float(getattr(cli_args, "velocity_spin_breaker_forward_mps", 0.035)), 0.035)
        cli_args.velocity_spin_breaker_angular_scale = min(float(getattr(cli_args, "velocity_spin_breaker_angular_scale", 0.35)), 0.35)
        cli_args.train_freq_steps = min(max(int(getattr(cli_args, "train_freq_steps", 4)), 1), 4)
        cli_args.gradient_steps = max(int(getattr(cli_args, "gradient_steps", 2)), 2)
        cli_args.batch_size = max(int(getattr(cli_args, "batch_size", 256)), 256)
        cli_args.learning_starts = min(int(getattr(cli_args, "learning_starts", 1000)), 1000)
        cli_args.checkpoint_freq = min(int(getattr(cli_args, "checkpoint_freq", 25000)), 25000)
        cli_args.save_replay_buffer = True
        cli_args.progress_print_freq = min(int(getattr(cli_args, "progress_print_freq", 1000)), 1000)
        return cli_args

    raise ValueError("--training-profile must be one of: normal, fast, ultrafast, visual_lite, real_finetune")


def _apply_rviz_origin_policy(cli_args):
    """Force the settings needed for RViz/map base pose to become (0, 0) after each reset.

    This intentionally overrides training profiles because the visual invariant
    requires an active SLAM map->odom transform and per-reset SLAM reset.
    """
    if not bool(getattr(cli_args, "rviz_zero_robot_on_reset", False)):
        return cli_args

    cli_args.disable_slam_map = False
    cli_args.auto_start_slam = True
    cli_args.wait_slam_map = True
    cli_args.reset_slam_on_reset = True
    cli_args.restart_slam_on_reset = True
    cli_args.reset_slam_every_n_episodes = 1
    cli_args.reset_tf_buffer_on_reset = True
    cli_args.pose_frame = str(getattr(cli_args, "map_frame", "map") or "map")
    cli_args.safety_boundary_frame = str(getattr(cli_args, "map_frame", "map") or "map")
    # SLAM-stable control profile.  RViz/map-origin mode is visualization-heavy,
    # so the robot must move with short local waypoints and conservative speeds.
    cli_args.max_linear_speed = min(float(cli_args.max_linear_speed), 0.32)
    cli_args.max_angular_speed = min(float(cli_args.max_angular_speed), 0.90)
    cli_args.waypoint_min_distance = min(float(cli_args.waypoint_min_distance), 0.20)
    cli_args.waypoint_max_distance = min(float(cli_args.waypoint_max_distance), 0.65)
    cli_args.waypoint_reached_tolerance = max(float(cli_args.waypoint_reached_tolerance), 0.32)
    cli_args.waypoint_execute_until_reached = True
    cli_args.waypoint_control_steps = min(int(cli_args.waypoint_control_steps), 2)
    cli_args.waypoint_max_control_steps = min(int(cli_args.waypoint_max_control_steps), 8)
    cli_args.waypoint_front_stop_distance = max(float(cli_args.waypoint_front_stop_distance), 0.30)
    cli_args.waypoint_replan_distance_m = min(float(getattr(cli_args, "waypoint_replan_distance_m", 0.35)), 0.35)
    cli_args.waypoint_replan_heading_deg = min(float(getattr(cli_args, "waypoint_replan_heading_deg", 45.0)), 45.0)
    cli_args.reset_pose_min_clearance_m = max(float(getattr(cli_args, "reset_pose_min_clearance_m", 0.16)), 0.12)
    return cli_args


class SkipStoreDictReplayBuffer(DictReplayBuffer):
    """DictReplayBuffer that skips transitions flagged by the env.

    SB3 stores every env.step() transition unconditionally.  For this project a
    forced safety backup/hold is not a policy decision: the robot was reversed or
    held still by the safety shield, not by the actor.  Storing those transitions
    teaches SAC the wrong dynamics (a forward action that maps to a reverse/zero
    motion) and dilutes the forward-driving signal.

    When the env marks a step with info["skip_store"] == True (or
    info["velocity_safety_skip_store"]), this buffer drops the transition instead
    of adding it, so the safety motion never enters training.  The penalty still
    reaches the agent on the *next* stored step boundary because the env already
    applied it to the reward; here we simply avoid storing the forced-motion
    transition itself.
    """

    @staticmethod
    def _is_skip(info) -> bool:
        if not isinstance(info, dict):
            return False
        return bool(info.get("skip_store", False) or info.get("velocity_safety_skip_store", False))

    def add(self, obs, next_obs, action, reward, done, infos):
        # infos is a list (one per parallel env).  With a single env, skip when
        # that env's transition is flagged.  With multiple envs we conservatively
        # only skip if ALL sub-envs are flagged, because add() writes one shared
        # row for every env at once and partial skipping is not representable.
        try:
            if infos and all(self._is_skip(i) for i in infos):
                return
        except Exception:
            # Never let a malformed info crash training; fall through to normal add.
            pass
        return super().add(obs, next_obs, action, reward, done, infos)


class RotatingCheckpointCallback(CheckpointCallback):
    """CheckpointCallback that keeps only the newest ``keep_last`` checkpoints.

    After every periodic save it deletes older checkpoint files so the model
    directory never accumulates more than ``keep_last`` model snapshots (plus the
    matching replay-buffer / vecnormalize files for the same step).  Early in
    training there may legitimately be 0 or 1 checkpoints; deletion only happens
    once there are more than ``keep_last``.  All file removals tolerate a missing
    file, so a race or an already-deleted file never raises.
    """

    def __init__(self, *args, keep_last: int = 2, **kwargs):
        super().__init__(*args, **kwargs)
        self.keep_last = max(int(keep_last), 1)

    def _prune_old_checkpoints(self) -> None:
        try:
            save_dir = Path(self.save_path)
        except Exception:
            return
        if not save_dir.exists():
            return
        prefix = str(self.name_prefix)

        # Group every checkpoint artifact by its step number so a kept step keeps
        # all of its sidecar files and a pruned step removes all of them.
        model_glob = f"{prefix}_*_steps.zip"
        try:
            model_files = list(save_dir.glob(model_glob))
        except Exception:
            return

        def _step_of(p: Path) -> int:
            m = re.search(r"_(\d+)_steps\.zip$", p.name)
            if m:
                return int(m.group(1))
            try:
                return int(p.stat().st_mtime)
            except Exception:
                return -1

        def _unlink_checkpoint_group(model_path: Path) -> None:
            step = _step_of(model_path)
            sidecars = [
                model_path,
                save_dir / f"{prefix}_replay_buffer_{step}_steps.pkl",
                save_dir / f"{prefix}_vecnormalize_{step}_steps.pkl",
            ]
            for f in sidecars:
                try:
                    f.unlink(missing_ok=True)  # py3.8+: never raises if absent
                except TypeError:
                    try:
                        if f.exists():
                            f.unlink()
                    except Exception:
                        pass
                except Exception:
                    pass

        valid_model_files = []
        for model_path in model_files:
            if _is_valid_zip_model(model_path):
                valid_model_files.append(model_path)
                continue
            _unlink_checkpoint_group(model_path)
            try:
                if self.verbose:
                    print(f"[CKPT ROTATE] removed invalid checkpoint {model_path.name}", flush=True)
            except Exception:
                pass

        if len(valid_model_files) <= self.keep_last:
            return

        # Newest first; keep the first keep_last, delete the rest.
        valid_model_files.sort(key=_step_of, reverse=True)
        to_delete = valid_model_files[self.keep_last:]

        for model_path in to_delete:
            step = _step_of(model_path)
            # Remove the model zip and any sidecar files saved for the same step.
            _unlink_checkpoint_group(model_path)
            try:
                if self.verbose:
                    print(f"[CKPT ROTATE] removed old checkpoint step={step}", flush=True)
            except Exception:
                pass

    def _on_step(self) -> bool:
        # Run the normal periodic save first, then prune.
        warmup_state = {}
        save_due = bool(self.save_freq > 0 and (self.n_calls % self.save_freq == 0))
        if save_due:
            warmup_state = _detach_warmup_action_mixer_for_save(getattr(self, "model", None))
        try:
            result = super()._on_step()
        finally:
            if warmup_state:
                _restore_warmup_action_mixer_after_save(getattr(self, "model", None), warmup_state)
        try:
            if self.save_freq > 0 and (self.n_calls % self.save_freq == 0):
                self._prune_old_checkpoints()
        except Exception:
            # Pruning must never crash training.
            pass
        return result


class EntropyCoefficientFloorCallback(BaseCallback):
    """Keep learned SAC alpha above a configured floor during fine-tuning."""

    def __init__(self, min_ent_coef: Optional[float], verbose: int = 0):
        super().__init__(verbose=verbose)
        self.min_ent_coef = None if min_ent_coef is None else max(float(min_ent_coef), 0.0)
        self.clamp_count = 0

    def _clamp_entropy_coefficient(self) -> None:
        if self.min_ent_coef is None or self.min_ent_coef <= 0.0:
            return
        log_ent_coef = getattr(self.model, "log_ent_coef", None)
        if log_ent_coef is None:
            return
        try:
            import torch as th
            with th.no_grad():
                min_log = th.log(th.as_tensor(self.min_ent_coef, device=log_ent_coef.device, dtype=log_ent_coef.dtype))
                if bool(th.any(log_ent_coef.data < min_log)):
                    log_ent_coef.data.copy_(th.maximum(log_ent_coef.data, min_log.expand_as(log_ent_coef.data)))
                    self.clamp_count += 1
                    if self.verbose > 0 and (self.clamp_count == 1 or self.clamp_count % 1000 == 0):
                        print(
                            f"[SAC ENTROPY] clamped alpha floor={self.min_ent_coef:.6f} count={self.clamp_count}",
                            flush=True,
                        )
        except Exception:
            return

    def _on_training_start(self) -> None:
        self._clamp_entropy_coefficient()

    def _on_rollout_start(self) -> None:
        self._clamp_entropy_coefficient()

    def _on_step(self) -> bool:
        try:
            self._clamp_entropy_coefficient()
        except Exception:
            return True
        return True


class DebugCallback(BaseCallback):
    def __init__(self, print_freq: int = 100, verbose: int = 0):
        super().__init__(verbose)
        self.print_freq = int(print_freq)

    def _on_step(self) -> bool:
        if self.print_freq <= 0:
            return True

        if self.num_timesteps % self.print_freq == 0:
            infos = self.locals.get("infos", [])
            rewards = self.locals.get("rewards", [])
            dones = self.locals.get("dones", [])

            info = infos[0] if len(infos) > 0 else {}
            reward = rewards[0] if len(rewards) > 0 else 0.0
            done = dones[0] if len(dones) > 0 else False

            print(
                "[SAC DEBUG] "
                f"t={self.num_timesteps} | "
                f"reward={float(reward):.4f} | "
                f"done={bool(done)} | "
                f"coverage={info.get('coverage_ratio', -1.0):.4f} | "
                f"new_cells={info.get('new_known_cells', -1)} | "
                f"mean_conf={info.get('mean_confidence', -1.0):.2f} | "
                f"conf_gain={info.get('confidence_gain', -1.0):.3f} | "
                f"stale={info.get('stale_known_cells', -1)} | "
                f"stale_refresh={info.get('stale_refresh_cells', -1)} | "
                f"low_conf={info.get('low_confidence_cells', -1)} | "
                f"priority={info.get('priority_score', -1.0):.2f} | "
                f"clear={info.get('priority_cleared_cells', -1)}:{info.get('priority_clear_gain', -1.0):.2f} | "
                f"pclrR={float(info.get('priority_clear_reward', 0.0)):+.3f}/"
                f"{float(info.get('episode_priority_check_reward', 0.0)):+.3f} | "
                f"invalid={info.get('priority_invalidated_cells', -1)}:{info.get('priority_invalidated_gain', -1.0):.2f} | "
                f"wall={info.get('wall_support_score', -1.0):.2f} | "
                f"open={info.get('open_space_score', -1.0):.2f} | "
                f"obs={info.get('nearest_obstacle_distance', -1.0):.2f}:"
                f"{info.get('obstacle_proximity_score', -1.0):.2f} | "
                f"lidar_obs={info.get('lidar_action_obstacle_distance', -1.0):.2f}:"
                f"{info.get('lidar_action_obstacle_score', -1.0):.2f}:"
                f"front={info.get('lidar_front_obstacle_distance', -1.0):.2f} | "
                f"stall={info.get('explored_stall_steps', -1)} | "
                f"frontiers={info.get('frontier_count', -1)} | "
                f"target={info.get('target_type', 'none')}:{info.get('target_priority', -1.0):.2f} | "
                f"angle={info.get('frontier_angle', 0.0):+.2f} | "
                f"pdir={info.get('priority_direction_error', 0.0):+.2f}:"
                f"{info.get('priority_direction_alignment', 0.0):.2f}:"
                f"{info.get('priority_direction_signed', 0.0):+.2f} | "
                f"mode={info.get('action_mode', 'n/a')} | "
                f"pa={info.get('policy_action_0', 0.0):+.2f}:{info.get('policy_action_1', 0.0):+.2f} | "
                f"cmd={info.get('executed_linear_x', 0.0):+.2f}:{info.get('executed_angular_z', 0.0):+.2f} | "
                f"slam_q={info.get('slam_quality_score', 1.0):.2f}:"
                f"loc={info.get('slam_local_known_ratio', 1.0):.2f}/"
                f"{info.get('slam_local_linear_score', 1.0):.2f}:"
                f"front={info.get('slam_front_known_ratio', 1.0):.2f}/"
                f"{info.get('slam_front_linear_score', 1.0):.2f}:"
                f"fresh={info.get('slam_fresh_score', 1.0):.2f}/"
                f"{info.get('slam_fresh_linear_score', 1.0):.2f}:"
                f"raw={info.get('slam_speed_raw_scale', 1.0):.2f}:"
                f"vscale={info.get('slam_speed_scale', 1.0):.2f}:"
                f"vlim={info.get('slam_speed_limit', -1.0):.2f} | "
                f"wp={info.get('waypoint_action_type', 'n/a')}:"
                f"{info.get('waypoint_distance', 0.0):.2f}:{info.get('waypoint_angle', 0.0):+.2f}:"
                f"lat={info.get('waypoint_lateral_offset', 0.0):+.2f}:"
                f"dhead={info.get('waypoint_heading_delta', 0.0):+.2f}:"
                f"{info.get('waypoint_reached', False)}:{info.get('waypoint_timed_out', False)}:"
                f"tout={info.get('waypoint_timeout_sec', 0.0):.2f}:"
                f"err={info.get('waypoint_final_error', 0.0):.2f}:steps={info.get('controller_steps', 0)} | "
                f"nav2={info.get('nav2_goal_accepted', False)}:{info.get('nav2_status_name', 'none')}:"
                f"src={info.get('nav2_goal_source', 'none')}:"
                f"valid={info.get('nav2_goal_valid', False)}:"
                f"gate={info.get('nav2_goal_validation', 'none')} | "
                f"sw={info.get('target_switched', False)}:{info.get('target_lock_age', -1)} | "
                f"slam={info.get('slam_map_available', False)}:{info.get('slam_map_gate', 'n/a')} "
                f"age={info.get('slam_map_age_sec', -1.0):.2f} "
                f"delay={info.get('slam_map_delay_remaining_sec', 0.0):.2f} | "
                f"collision={info.get('collision', False)}:"
                f"restart={info.get('collision_restart_requested', False)} | "
                f"term={info.get('terminal_reason', 'none')} | "
                f"bound={info.get('safety_boundary_frame', 'n/a')}:"
                f"c=({info.get('safety_boundary_center_x', 0.0):+.2f},{info.get('safety_boundary_center_y', 0.0):+.2f}):"
                f"p=({info.get('out_of_bounds_x', 0.0):+.2f},{info.get('out_of_bounds_y', 0.0):+.2f}):"
                f"r={info.get('out_of_bounds_radius', 0.0):.2f}:"
                f"why={info.get('out_of_bounds_reason', 'none')} | "
                f"fallen={info.get('fallen', False)} | "
                f"mapdbg={info.get('debug_input_map', False)}:"
                f"{info.get('debug_input_map_published', False)}:"
                f"{info.get('debug_input_map_topic_prefix', '')} | "
                f"sim_time={info.get('sim_time', -1.0):.3f}",
                flush=True,
            )

        return True


class TrainingProgressCallback(BaseCallback):
    """
    Quiet SAC training progress reporter.

    Modes:
      - line : low-frequency one-line print
      - block: low-frequency multi-line compact status block
      - tqdm : single tqdm progress bar with compact postfix
      - quiet: CSV/TensorBoard only, no normal terminal progress

    Severe ROS/Gazebo errors still go through the normal logger.  This callback
    deliberately records only training history and compact state metrics.
    """

    def __init__(
        self,
        total_timesteps: int,
        print_freq: int = 500,
        window_size: int = 20,
        csv_path: str = "",
        progress_style: str = "line",
        csv_flush_every: int = 20,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.requested_total_timesteps = max(int(total_timesteps), 1)
        self.print_freq = max(int(print_freq), 1)
        self.window_size = max(int(window_size), 1)
        self.csv_path = str(csv_path or "").strip()
        self.progress_style = str(progress_style or "line").strip().lower()
        if self.progress_style not in {"line", "block", "tqdm", "quiet"}:
            self.progress_style = "line"
        self.csv_flush_every = max(int(csv_flush_every), 1)
        self.start_wall_time = 0.0
        self.start_timesteps = 0
        self.target_timesteps = self.requested_total_timesteps
        self.recent_episode_rewards = deque(maxlen=self.window_size)
        self.recent_episode_lengths = deque(maxlen=self.window_size)
        self.recent_episode_steps_per_wall_sec = deque(maxlen=self.window_size)
        self.recent_episode_steps_per_sim_sec = deque(maxlen=self.window_size)
        self.recent_terminal_reasons = deque(maxlen=self.window_size)
        self.recent_policy_w = deque(maxlen=max(self.print_freq, 100))
        self.recent_executed_w = deque(maxlen=max(self.print_freq, 100))
        self._current_episode_reward = 0.0
        self._current_episode_len = 0
        self._episode_start_wall_time = 0.0
        self._episode_start_sim_time = None
        self._last_episode_wall_sec = float("nan")
        self._last_episode_sim_sec = float("nan")
        self._last_episode_steps_per_wall_sec = float("nan")
        self._last_episode_steps_per_sim_sec = float("nan")
        self._episode_index = 0
        self._csv_file = None
        self._csv_rows_since_flush = 0
        self._pbar = None
        self._last_pbar_n = 0
        self._current_ep_safety_terminal_count = 0
        self._current_ep_collision_count = 0
        self._last_ep_safety_terminal_count = 0
        self._last_ep_collision_count = 0

    @staticmethod
    def _fmt_duration(seconds: float) -> str:
        seconds = max(float(seconds), 0.0)
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        sec = int(seconds % 60)
        if h > 0:
            return f"{h:d}h{m:02d}m{sec:02d}s"
        if m > 0:
            return f"{m:d}m{sec:02d}s"
        return f"{sec:d}s"

    def _logger_value(self, key: str, default: float = float("nan")) -> float:
        """Best-effort read from SB3 logger without forcing a dump."""
        try:
            logger = getattr(self.model, "logger", None)
            values = getattr(logger, "name_to_value", {}) if logger is not None else {}
            if key in values:
                return float(values[key])
            alt = key.replace("train/", "") if key.startswith("train/") else f"train/{key}"
            if alt in values:
                return float(values[alt])
        except Exception:
            pass
        return float(default)

    @staticmethod
    def _finite_or_blank(x: float) -> str:
        try:
            if math.isfinite(float(x)):
                return f"{float(x):.6f}"
        except Exception:
            pass
        return ""

    @staticmethod
    def _turn_balance(values, deadband: float = 0.04) -> tuple[float, float, float, float]:
        vals = [float(v) for v in values if math.isfinite(float(v))]
        if not vals:
            return float("nan"), float("nan"), float("nan"), float("nan")
        total = float(len(vals))
        left = sum(1 for v in vals if v > deadband) / total
        right = sum(1 for v in vals if v < -deadband) / total
        zero = sum(1 for v in vals if abs(v) <= deadband) / total
        mean_w = sum(vals) / total
        return left, right, zero, mean_w

    def _on_training_start(self) -> None:
        self.start_wall_time = time.time()
        self._episode_start_wall_time = self.start_wall_time
        self._episode_start_sim_time = None
        self.start_timesteps = int(self.model.num_timesteps)
        self.target_timesteps = self.start_timesteps + self.requested_total_timesteps

        if self.csv_path:
            path = Path(self.csv_path).expanduser()
            path.parent.mkdir(parents=True, exist_ok=True)
            csv_header = (
                "timesteps,progress_percent,fps,elapsed_sec,eta_sec,"
                "actor_loss,critic_loss,ent_coef_loss,ent_coef,learning_rate,"
                "mean_episode_reward,mean_episode_length,last_episode_wall_sec,"
                "last_episode_sps_wall,last_episode_sim_sec,last_episode_sps_sim,"
                "mean_episode_sps_wall,mean_episode_sps_sim,last_reward,last_done,"
                "terminal_reason,coverage_ratio,confidence_gain,confidence_fill_delta,confidence_fill_hold,confidence_updated_cells,"
                "confidence_observed_cells,priority_score,collision,fallen,out_of_bounds,"
                "coverage_stall_terminal,coverage_stall_active,coverage_stall_reason,"
                "coverage_stall_window_steps,coverage_stall_window_len,"
                "coverage_stall_slam_new_cells,coverage_stall_confidence_updated_cells,"
                "coverage_stall_coverage_delta,coverage_stall_consecutive_hits,"
                "coverage_stall_no_progress_steps,coverage_stall_hard_no_progress_steps,"
                "step_recovery,"
                "scan_age_sec,odom_age_sec,map_age_sec,obs_stale,scan_stale,map_stale,"
                "tf_pose_ok,confidence_pose_ok,"
                "policy_scan_front,policy_scan_left,policy_scan_rear,policy_scan_right,"
                "raw_scan_front,raw_scan_left,raw_scan_rear,raw_scan_right,"
                "lidar60_stamp_delta_sec,"
                "velocity_safety_terminal,velocity_safety_distance,"
                "executed_v,executed_w,policy_v,policy_w,"
                "r_confidence,r_confidence_fill,r_slam,r_priority,r_wall,"
                "r_safety_slow,r_safety_terminal,r_collision,r_step,reward_total,"
                "safety_terminal_count_ep,collision_count_ep\n"
            )
            append_mode = path.exists() and path.stat().st_size > 0
            if append_mode:
                try:
                    first_line = path.open("r", encoding="utf-8").readline()
                except Exception:
                    first_line = ""
                if first_line and first_line != csv_header:
                    rotated = path.with_suffix(path.suffix + f".legacy_{int(time.time())}")
                    path.rename(rotated)
                    append_mode = False
            self._csv_file = path.open("a" if append_mode else "w", encoding="utf-8")
            if not append_mode:
                self._csv_file.write(csv_header)
            self._csv_file.flush()

        if self.progress_style == "tqdm":
            try:
                from tqdm.auto import tqdm  # type: ignore
                self._pbar = tqdm(
                    total=self.requested_total_timesteps,
                    initial=0,
                    dynamic_ncols=True,
                    smoothing=0.05,
                    mininterval=0.5,
                    desc="SAC",
                    unit="step",
                    leave=True,
                )
            except Exception:
                self._pbar = None
                self.progress_style = "line"

        self._rich_live = None
        if self.progress_style == "block" and sys.stdout.isatty():
            # v133: a fixed dashboard window that updates in place. Manual
            # multi-line ANSI cursor-up/erase was tried first and reverted --
            # it silently assumes nothing else writes to stdout between our
            # own prints (checkpoint-save messages, warnings, etc. all break
            # that assumption and leave stale lines behind). rich.live.Live
            # is built to handle exactly this correctly.
            try:
                from rich.live import Live  # type: ignore
                self._rich_live = Live("", refresh_per_second=4, transient=False)
                self._rich_live.start()
            except Exception:
                self._rich_live = None

    def _on_training_end(self) -> None:
        if getattr(self, "_rich_live", None) is not None:
            try:
                self._rich_live.stop()
            except Exception:
                pass

    def _on_step(self) -> bool:
        rewards = self.locals.get("rewards", [])
        dones = self.locals.get("dones", [])
        infos = self.locals.get("infos", [])

        reward = float(rewards[0]) if len(rewards) > 0 else 0.0
        done = bool(dones[0]) if len(dones) > 0 else False
        info = infos[0] if len(infos) > 0 else {}
        if not isinstance(info, dict):
            info = {}

        if self._current_episode_len == 0:
            self._episode_start_wall_time = time.time()
            try:
                sim_start = float(info.get("sim_time", float("nan")))
            except Exception:
                sim_start = float("nan")
            self._episode_start_sim_time = sim_start if sim_start >= 0.0 and math.isfinite(sim_start) else None

        self._current_episode_reward += reward
        self._current_episode_len += 1

        # v131: per-episode safety terminal and collision counters
        if info.get("velocity_safety_terminal", False):
            self._current_ep_safety_terminal_count = int(getattr(self, "_current_ep_safety_terminal_count", 0)) + 1
        if info.get("collision", False):
            self._current_ep_collision_count = int(getattr(self, "_current_ep_collision_count", 0)) + 1

        try:
            self.recent_policy_w.append(float(info.get("policy_w", 0.0)))
            self.recent_executed_w.append(float(info.get("executed_w", 0.0)))
        except Exception:
            pass

        if done:
            episode_info = info.get("episode", {}) if isinstance(info, dict) else {}
            ep_reward = float(episode_info.get("r", self._current_episode_reward))
            ep_len = int(episode_info.get("l", self._current_episode_len))

            # v133: prefer the env's own reset-free timestamp (info["episode_wall_sec"],
            # set in gazebo_nav_env.py from when reset() actually returned). The
            # naive alternative -- time.time() - self._episode_start_wall_time --
            # is contaminated: SB3's VecEnv auto-resets inside the same step()
            # call that returns done=True, so that bracket silently includes the
            # *next* episode's full reset (world reset, obstacle spawn, SLAM
            # restart), making both fps and this "sps" read almost identical and
            # far too low.
            _env_ep_wall_sec = float("nan")
            try:
                _env_ep_wall_sec = float(info.get("episode_wall_sec", float("nan")))
            except Exception:
                pass
            if math.isfinite(_env_ep_wall_sec) and _env_ep_wall_sec >= 0.0:
                ep_wall_sec = max(_env_ep_wall_sec, 1e-6)
            else:
                ep_wall_sec = max(time.time() - float(self._episode_start_wall_time or self.start_wall_time), 1e-6)
            ep_sps_wall = float(ep_len) / ep_wall_sec if ep_len > 0 else 0.0
            ep_sim_sec = float("nan")
            ep_sps_sim = float("nan")
            try:
                sim_end = float(info.get("sim_time", float("nan")))
            except Exception:
                sim_end = float("nan")
            if self._episode_start_sim_time is not None and math.isfinite(sim_end):
                ep_sim_sec = max(sim_end - float(self._episode_start_sim_time), 1e-6)
                ep_sps_sim = float(ep_len) / ep_sim_sec if ep_len > 0 else 0.0

            self._episode_index += 1
            self._last_episode_wall_sec = ep_wall_sec
            self._last_episode_sim_sec = ep_sim_sec
            self._last_episode_steps_per_wall_sec = ep_sps_wall
            self._last_episode_steps_per_sim_sec = ep_sps_sim

            self.recent_episode_rewards.append(ep_reward)
            self.recent_episode_lengths.append(ep_len)
            self.recent_episode_steps_per_wall_sec.append(ep_sps_wall)
            if math.isfinite(ep_sps_sim):
                self.recent_episode_steps_per_sim_sec.append(ep_sps_sim)
            self.recent_terminal_reasons.append(str(info.get("terminal_reason", "done")))

            try:
                self.logger.record("rollout/ep_last_steps", ep_len)
                self.logger.record("rollout/ep_last_wall_sec", ep_wall_sec)
                self.logger.record("rollout/ep_last_sps_wall", ep_sps_wall)
                if math.isfinite(ep_sim_sec):
                    self.logger.record("rollout/ep_last_sim_sec", ep_sim_sec)
                if math.isfinite(ep_sps_sim):
                    self.logger.record("rollout/ep_last_sps_sim", ep_sps_sim)
                self.logger.record("rollout/ep_last_reward", ep_reward)
            except Exception:
                pass

            if str(os.environ.get("TB3_RL_EPISODE_SPS_LINE_LOG", "0")).strip().lower() in {"1", "true", "yes", "on"}:
                print(
                    "[SAC EPISODE] "
                    f"ep={self._episode_index:d} steps={ep_len:d} "
                    f"wall={self._fmt_duration(ep_wall_sec)} sps={ep_sps_wall:6.2f} "
                    f"epR={ep_reward:+9.2f} term={info.get('terminal_reason', 'done')}",
                    flush=True,
                )

            self._current_episode_reward = 0.0
            self._current_episode_len = 0
            self._last_ep_safety_terminal_count = int(getattr(self, "_current_ep_safety_terminal_count", 0))
            self._last_ep_collision_count = int(getattr(self, "_current_ep_collision_count", 0))
            self._current_ep_safety_terminal_count = 0
            self._current_ep_collision_count = 0
            self._episode_start_wall_time = time.time()
            self._episode_start_sim_time = None

        trained_steps = max(int(self.num_timesteps) - self.start_timesteps, 0)
        terminal_reason = str(info.get("terminal_reason", "none"))
        if done and terminal_reason == "none":
            if bool(info.get("step_recovery", False)):
                terminal_reason = "step_recovery"
            elif bool(info.get("TimeLimit.truncated", False)):
                terminal_reason = "time_limit"
            else:
                terminal_reason = "done_unknown"

        # v133: was episode-end-only for a while (see git history) on the
        # theory that per-step CSV/print I/O was contributing to a disk-full
        # crash. The actual cause turned out to be cartographer's own
        # untracked glog files (fixed separately with GLOG_logtostderr), so
        # that's no longer a reason to hide per-step Q-loss/progress visibility.
        should_emit = (
            trained_steps == 1
            or trained_steps % self.print_freq == 0
            or trained_steps >= self.requested_total_timesteps
            or done
        )

        if should_emit:
            elapsed = max(time.time() - self.start_wall_time, 1e-6)
            progress = min(trained_steps / max(self.requested_total_timesteps, 1), 1.0)
            fps = trained_steps / elapsed if trained_steps > 0 else 0.0
            remaining = max(self.requested_total_timesteps - trained_steps, 0)
            eta = remaining / fps if fps > 1e-6 else 0.0

            mean_ep_reward = (
                sum(self.recent_episode_rewards) / len(self.recent_episode_rewards)
                if self.recent_episode_rewards else float("nan")
            )
            mean_ep_len = (
                sum(self.recent_episode_lengths) / len(self.recent_episode_lengths)
                if self.recent_episode_lengths else float("nan")
            )
            mean_ep_sps_wall = (
                sum(self.recent_episode_steps_per_wall_sec) / len(self.recent_episode_steps_per_wall_sec)
                if self.recent_episode_steps_per_wall_sec else float("nan")
            )
            mean_ep_sps_sim = (
                sum(self.recent_episode_steps_per_sim_sec) / len(self.recent_episode_steps_per_sim_sec)
                if self.recent_episode_steps_per_sim_sec else float("nan")
            )

            actor_loss = self._logger_value("train/actor_loss")
            critic_loss = self._logger_value("train/critic_loss")
            ent_coef_loss = self._logger_value("train/ent_coef_loss")
            ent_coef = self._logger_value("train/ent_coef")
            try:
                log_ent_coef = getattr(self.model, "log_ent_coef", None)
                if log_ent_coef is not None:
                    ent_coef = float(log_ent_coef.detach().exp().mean().item())
                elif getattr(self.model, "ent_coef_tensor", None) is not None:
                    ent_coef = float(self.model.ent_coef_tensor.detach().mean().item())
            except Exception:
                pass
            learning_rate = self._logger_value("train/learning_rate")
            exec_left, exec_right, exec_zero, exec_mean_w = self._turn_balance(self.recent_executed_w)

        if self.progress_style == "tqdm" and self._pbar is not None:
            delta = trained_steps - int(self._last_pbar_n)
            if delta > 0:
                self._pbar.update(delta)
                self._last_pbar_n = trained_steps
            if should_emit:
                self._pbar.set_postfix({
                    "fps": f"{fps:.1f}",
                    "Qloss": "nan" if not math.isfinite(critic_loss) else f"{critic_loss:.2f}",
                    "Aloss": "nan" if not math.isfinite(actor_loss) else f"{actor_loss:.2f}",
                    "epR": "nan" if not math.isfinite(mean_ep_reward) else f"{mean_ep_reward:.1f}",
                    "wL/R": "nan" if not math.isfinite(exec_left) else f"{exec_left:.2f}/{exec_right:.2f}",
                    "conf": int(info.get("confidence_updated_cells", 0)),
                    "term": terminal_reason[:14],
                })
        elif self.progress_style == "block" and should_emit:
            def _num(x: float, width: int = 8, prec: int = 3) -> str:
                try:
                    if math.isfinite(float(x)):
                        return f"{float(x):{width}.{prec}f}"
                except Exception:
                    pass
                return "nan".rjust(width)

            block_text = (
                f"[SAC] {progress * 100.0:6.2f}%  step={trained_steps}/{self.requested_total_timesteps}  "
                f"fps={fps:6.2f}  eta={self._fmt_duration(eta)}\n"
                f"      loss: Q={_num(critic_loss)}  actor={_num(actor_loss)}  "
                f"alpha={_num(ent_coef, 7, 4)}  lr={_num(learning_rate, 9, 6)}\n"
                f"      ep  : R{self.window_size}={_num(mean_ep_reward, 9, 2)}  "
                f"len={_num(mean_ep_len, 7, 1)}  sps={_num(mean_ep_sps_wall, 7, 2)}  "
                f"r_now={reward:+8.3f}\n"
                f"      env : conf={int(info.get('confidence_updated_cells', 0)):5d}/"
                f"{int(info.get('confidence_observed_cells', 0)):5d}  "
                f"recent={int(info.get('recent_confidence_updated_cells', 0)):6d}  "
                f"conf_gain={float(info.get('confidence_gain', 0.0)):7.3f}  "
                f"cov={float(info.get('coverage_ratio', -1.0)):6.3f}  "
                f"term={terminal_reason}  err={bool(info.get('step_recovery', False))}\n"
                f"      act : policy=({float(info.get('policy_v', 0.0)):+.3f},{float(info.get('policy_w', 0.0)):+.3f})  "
                f"exec=({float(info.get('executed_v', 0.0)):+.3f},{float(info.get('executed_w', 0.0)):+.3f})  "
                f"w_recent L/R/Z={exec_left:4.2f}/{exec_right:4.2f}/{exec_zero:4.2f}  "
                f"mean_w={exec_mean_w:+.3f}"
            )
            if getattr(self, "_rich_live", None) is not None:
                try:
                    self._rich_live.update(block_text)
                except Exception:
                    print(block_text, flush=True)
            else:
                print(block_text, flush=True)
        elif self.progress_style == "line" and should_emit:
            line_text = (
                "[SAC] "
                f"{progress * 100.0:6.2f}% "
                f"step={trained_steps}/{self.requested_total_timesteps} "
                f"fps={fps:6.1f} eta={self._fmt_duration(eta)} "
                f"Qloss={critic_loss:7.3f} Aloss={actor_loss:7.3f} alpha={ent_coef:7.4f} "
                f"epR{self.window_size}={mean_ep_reward:8.2f} epLen={mean_ep_len:6.1f} "
                f"sps={mean_ep_sps_wall:5.2f} r={reward:+7.3f} "
                f"conf={int(info.get('confidence_updated_cells', 0))}/{int(info.get('confidence_observed_cells', 0))} "
                f"cov={float(info.get('coverage_ratio', -1.0)):.3f} "
                f"p=({float(info.get('policy_v', 0.0)):+.2f},{float(info.get('policy_w', 0.0)):+.2f}) "
                f"e=({float(info.get('executed_v', 0.0)):+.2f},{float(info.get('executed_w', 0.0)):+.2f}) "
                f"wL/R/Z={exec_left:.2f}/{exec_right:.2f}/{exec_zero:.2f} "
                f"term={terminal_reason} "
                f"err={bool(info.get('step_recovery', False))}"
            )
            if sys.stdout.isatty():
                # \r + pad-to-previous-length is a single-line overwrite that
                # every terminal handles the same way (unlike multi-line
                # cursor-up, which depends on the terminal's own tracked
                # cursor position and breaks under `make`/multiplexers).
                prev_len = int(getattr(self, "_line_prev_len", 0))
                pad = max(prev_len - len(line_text), 0)
                sys.stdout.write("\r" + line_text + (" " * pad))
                sys.stdout.flush()
                self._line_prev_len = len(line_text)
            else:
                print(line_text, flush=True)

        if self._csv_file is not None and should_emit:
            self._csv_file.write(
                f"{trained_steps},{progress * 100.0:.6f},{fps:.6f},{elapsed:.6f},{eta:.6f},"
                f"{self._finite_or_blank(actor_loss)},{self._finite_or_blank(critic_loss)},"
                f"{self._finite_or_blank(ent_coef_loss)},{self._finite_or_blank(ent_coef)},"
                f"{self._finite_or_blank(learning_rate)},"
                f"{self._finite_or_blank(mean_ep_reward)},{self._finite_or_blank(mean_ep_len)},"
                f"{self._finite_or_blank(self._last_episode_wall_sec)},"
                f"{self._finite_or_blank(self._last_episode_steps_per_wall_sec)},"
                f"{self._finite_or_blank(self._last_episode_sim_sec)},"
                f"{self._finite_or_blank(self._last_episode_steps_per_sim_sec)},"
                f"{self._finite_or_blank(mean_ep_sps_wall)},{self._finite_or_blank(mean_ep_sps_sim)},"
                f"{reward:.6f},{int(done)},"
                f"{terminal_reason},{float(info.get('coverage_ratio', -1.0)):.6f},"
                f"{float(info.get('confidence_gain', 0.0)):.6f},"
                f"{float(info.get('confidence_fill_delta', 0.0)):.6f},"
                f"{float(info.get('confidence_fill_hold', 0.0)):.6f},"
                f"{int(info.get('confidence_updated_cells', 0))},"
                f"{int(info.get('confidence_observed_cells', 0))},"
                f"{float(info.get('priority_score', -1.0)):.6f},"
                f"{int(bool(info.get('collision', False)))},"
                f"{int(bool(info.get('fallen', False)))},"
                f"{int(bool(info.get('out_of_bounds', False)))},"
                f"{int(bool(info.get('coverage_stall_terminal', False)))},"
                f"{int(bool(info.get('coverage_stall_active', False)))},"
                f"{str(info.get('coverage_stall_reason', 'none')).replace(',', ';')},"
                f"{int(info.get('coverage_stall_window_steps', 0))},"
                f"{int(info.get('coverage_stall_window_len', 0))},"
                f"{int(info.get('coverage_stall_slam_new_cells', 0))},"
                f"{int(info.get('coverage_stall_confidence_updated_cells', 0))},"
                f"{self._finite_or_blank(float(info.get('coverage_stall_coverage_delta', 0.0)))},"
                f"{int(info.get('coverage_stall_consecutive_hits', 0))},"
                f"{int(info.get('coverage_stall_no_progress_steps', 0))},"
                f"{int(info.get('coverage_stall_hard_no_progress_steps', 0))},"
                f"{int(bool(info.get('step_recovery', False)))},"
                f"{self._finite_or_blank(float(info.get('scan_age_sec', -1.0)))},"
                f"{self._finite_or_blank(float(info.get('odom_age_sec', -1.0)))},"
                f"{self._finite_or_blank(float(info.get('map_age_sec', -1.0)))},"
                f"{int(bool(info.get('obs_stale', False)))},"
                f"{int(bool(info.get('scan_stale', False)))},"
                f"{int(bool(info.get('map_stale', False)))},"
                f"{int(bool(info.get('tf_pose_ok', False)))},"
                f"{int(bool(info.get('confidence_pose_ok', False)))},"
                f"{self._finite_or_blank(float(info.get('policy_scan_front', 999.0)))},"
                f"{self._finite_or_blank(float(info.get('policy_scan_left', 999.0)))},"
                f"{self._finite_or_blank(float(info.get('policy_scan_rear', 999.0)))},"
                f"{self._finite_or_blank(float(info.get('policy_scan_right', 999.0)))},"
                f"{self._finite_or_blank(float(info.get('raw_scan_front', 999.0)))},"
                f"{self._finite_or_blank(float(info.get('raw_scan_left', 999.0)))},"
                f"{self._finite_or_blank(float(info.get('raw_scan_rear', 999.0)))},"
                f"{self._finite_or_blank(float(info.get('raw_scan_right', 999.0)))},"
                f"{self._finite_or_blank(float(info.get('lidar60_stamp_delta_sec', 0.0)))},"
                f"{int(bool(info.get('velocity_safety_terminal', False)))},"
                f"{self._finite_or_blank(float(info.get('velocity_safety_distance', 999.0)))},"
                f"{self._finite_or_blank(float(info.get('executed_v', 0.0)))},"
                f"{self._finite_or_blank(float(info.get('executed_w', 0.0)))},"
                f"{self._finite_or_blank(float(info.get('policy_v', 0.0)))},"
                f"{self._finite_or_blank(float(info.get('policy_w', 0.0)))},"
                f"{self._finite_or_blank(float(info.get('r_confidence', 0.0)))},"
                f"{self._finite_or_blank(float(info.get('r_confidence_fill', 0.0)))},"
                f"{self._finite_or_blank(float(info.get('r_slam', 0.0)))},"
                f"{self._finite_or_blank(float(info.get('r_priority', 0.0)))},"
                f"{self._finite_or_blank(float(info.get('r_wall', 0.0)))},"
                f"{self._finite_or_blank(float(info.get('r_safety_slow', 0.0)))},"
                f"{self._finite_or_blank(float(info.get('r_safety_terminal', 0.0)))},"
                f"{self._finite_or_blank(float(info.get('r_collision', 0.0)))},"
                f"{self._finite_or_blank(float(info.get('r_step', 0.0)))},"
                f"{self._finite_or_blank(float(info.get('reward_total', 0.0)))},"
                f"{int(getattr(self, '_last_ep_safety_terminal_count', 0))},"
                f"{int(getattr(self, '_last_ep_collision_count', 0))}\n"
            )
            self._csv_rows_since_flush += 1
            if self._csv_rows_since_flush >= self.csv_flush_every:
                self._csv_file.flush()
                self._csv_rows_since_flush = 0

        return True

    def _on_training_end(self) -> None:
        if self._pbar is not None:
            try:
                self._pbar.close()
            except Exception:
                pass
            self._pbar = None
        if self._csv_file is not None:
            self._csv_file.flush()
            self._csv_file.close()
            self._csv_file = None


def _force_nav2_only_policy(cli_args):
    """Hard runtime invariant: SAC may select only local goals; Nav2 owns all motion.

    Do not silently rewrite the user's Nav2 timing/preemption policy here.
    The previous fast-streaming patch clamped --nav2-send-goal-wait-sec to <=0.30s
    and forced preempt streaming.  On this TurtleBot3/Nav2 setup the FollowPath
    action server is ready, but goal acceptance often takes longer than 0.30s;
    treating that as timeout makes the env return zero motion forever.
    """
    requested = str(getattr(cli_args, "action_mode", "nav2") or "nav2").strip().lower()
    if requested != "nav2":
        # Direct-control modes are now first-class.  Do not rewrite them to Nav2.
        # This is required for pure velocity SAC, where the policy owns /cmd_vel
        # through TurtleBot3RosInterface.publish_cmd_vel(TwistStamped).
        cli_args.action_mode = requested
        cli_args.disable_world_step = True
        cli_args.disable_wall_proximity_penalty = False
        if requested == "velocity":
            print("[DIRECT_CONTROL] action_mode='velocity': Nav2 disabled, SAC publishes TwistStamped with safety backup shield")
            # Existing Nav2/waypoint checkpoints have incompatible action semantics.
            # Keep resume explicit only; do not silently auto-resume by default in velocity mode.
            if bool(getattr(cli_args, "resume_latest", False)) and not str(getattr(cli_args, "load_model", "") or "").strip():
                print("[DIRECT_CONTROL] WARNING: --resume-latest with velocity mode can load incompatible Nav2/waypoint weights.")
        return cli_args

    cli_args.action_mode = "nav2"

    # Nav2 action servers progress on wall-clock.  Do not run the paused-world
    # multi_step barrier in Nav2-only mode.
    cli_args.disable_world_step = True

    # Keep RViz waypoint/reward diagnostics visible by default.
    if not str(getattr(cli_args, "waypoint_marker_topic", "") or "").strip():
        cli_args.waypoint_marker_topic = "/rl_waypoint_marker"
    cli_args.waypoint_visual_publish_every_n = 1

    # Dense safety shaping is required even when Nav2 owns /cmd_vel; otherwise the
    # critic sees only sparse collision terminals.
    cli_args.disable_wall_proximity_penalty = False

    # Keep the user's explicit cancel/preempt policy from the command line.
    # Important: do NOT force these to fast streaming here.
    #   --no-nav2-continuous-goal-update
    #   --no-nav2-preempt-without-cancel
    #   --nav2-cancel-on-reached
    #   --nav2-cancel-on-timeout
    # must remain effective for the last stable cancel-sync mode.
    cli_args.nav2_goal_reached_tolerance = max(
        float(getattr(cli_args, "nav2_goal_reached_tolerance", 0.30)),
        0.30,
    )
    cli_args.waypoint_reached_tolerance = max(
        float(getattr(cli_args, "waypoint_reached_tolerance", 0.32)),
        0.32,
    )

    # Do not let action goal acceptance be killed before Nav2 has time to answer.
    # This is the direct fix for repeated "Nav2 goal send timed out" while the
    # FollowPath/NavigateToPose action server is already reported as ready.
    cli_args.nav2_send_goal_wait_sec = max(
        float(getattr(cli_args, "nav2_send_goal_wait_sec", 2.0)),
        2.0,
    )
    cli_args.nav2_wait_timeout_sec = max(
        float(getattr(cli_args, "nav2_wait_timeout_sec", 8.0)),
        8.0,
    )

    return cli_args

def _force_mandatory_slam_reset_policy(cli_args):
    """Hard invariant: every Gazebo respawn must reset SLAM and TF.

    This intentionally overrides speed profiles.  Running with stale SLAM maps is
    the main reason /map and /rl_priority_map drift apart in RViz over time.
    """
    cli_args.disable_slam_map = False
    cli_args.auto_start_slam = True
    cli_args.wait_slam_map = True
    cli_args.reset_slam_on_reset = True
    cli_args.restart_slam_on_reset = True
    cli_args.reset_slam_every_n_episodes = 1
    cli_args.reset_tf_buffer_on_reset = True

    # Keep RViz/debug layers map-locked.  The previous fast patch forced odom here,
    # which made /rl_waypoint_marker show frame=odom and made MarkerArray debugging
    # hard when RViz Fixed Frame was map.  SLAM is still reset at every respawn, but
    # RL maps, filtered SLAM, priority/confidence layers, Nav2 goals, and waypoint
    # markers are all published in the same map frame by default.
    map_frame = str(getattr(cli_args, "map_frame", "map") or "map").strip().lstrip("/") or "map"
    pose_frame = str(getattr(cli_args, "pose_frame", map_frame) or map_frame).strip().lstrip("/") or map_frame
    if pose_frame not in {"map", map_frame}:
        print(f"[FRAME_POLICY] overriding --pose-frame {pose_frame!r} -> {map_frame!r} for map-aligned RViz debugging")
        pose_frame = map_frame
    cli_args.map_frame = map_frame
    cli_args.pose_frame = pose_frame
    # Safety/out-of-bounds remains in odom unless the user explicitly set another
    # frame.  That avoids false resets while slam_toolbox corrects map->odom.
    cli_args.safety_boundary_frame = str(getattr(cli_args, "safety_boundary_frame", "odom") or "odom").strip().lstrip("/") or "odom"

    if not str(getattr(cli_args, "rl_filtered_slam_topic", "") or "").strip():
        cli_args.rl_filtered_slam_topic = "/rl_filtered_slam_map"
    if bool(getattr(cli_args, "disable_priority_map", False)):
        # Priority is intentionally removed: do not publish /rl_priority_map and
        # do not let the mandatory SLAM/map policy silently re-enable the topic.
        cli_args.rl_priority_topic = ""
        cli_args.enable_corridor_priority_reward = False
        cli_args.corridor_priority_reward_weight = 0.0
        cli_args.post_reset_ready_require_priority = False
        cli_args.priority_stuck_restart = False
    elif not str(getattr(cli_args, "rl_priority_topic", "") or "").strip():
        cli_args.rl_priority_topic = "/rl_priority_map"
    if not str(getattr(cli_args, "rl_map_topic", "") or "").strip():
        cli_args.rl_map_topic = "/rl_task_map"
    if not str(getattr(cli_args, "rl_confidence_topic", "") or "").strip():
        cli_args.rl_confidence_topic = "/rl_confidence_map"

    cli_args.slam_map_accept_delay_sec = max(float(getattr(cli_args, "slam_map_accept_delay_sec", 1.0)), 1.2)
    cli_args.slam_map_max_age_sec = max(float(getattr(cli_args, "slam_map_max_age_sec", 3.0)), 3.0)
    return cli_args




def _force_no_priority_policy(cli_args):
    """v93: hard-remove priority from training runtime.

    This is stronger than reward-only disablement:
      - no /rl_priority_map publisher
      - no priority channel in map CNN input
      - no target_priority scalar in vector observation
      - no priority reward / priority-stuck restart / priority reset gate

    The policy is activated by --disable-priority-map,
    TB3_RL_FORCE_NO_PRIORITY=1, or TB3_RL_NO_PRIORITY_MODEL_INPUT=1.
    """
    raw_force = str(os.environ.get("TB3_RL_FORCE_NO_PRIORITY", "0")).strip().lower()
    raw_no_model = str(os.environ.get("TB3_RL_NO_PRIORITY_MODEL_INPUT", "0")).strip().lower()
    force = raw_force not in {"0", "false", "no", "off", "disable", "disabled"}
    no_model = raw_no_model in {"1", "true", "yes", "on", "enable", "enabled"}
    if force or no_model or bool(getattr(cli_args, "disable_priority_map", False)):
        cli_args.disable_priority_map = True
        cli_args.enable_corridor_priority_reward = False
        cli_args.corridor_priority_reward_weight = 0.0
        cli_args.post_reset_ready_require_priority = False
        cli_args.priority_stuck_restart = False
        cli_args.priority_stuck_restart_sec = 0.0
        cli_args.priority_stuck_restart_steps = 1_000_000_000
        cli_args.priority_stuck_score_threshold = 1.0
        cli_args.priority_stuck_clear_gain_threshold = 1.0
        cli_args.priority_stuck_info_gain_threshold = 1.0
        cli_args.priority_recompute_interval = 1_000_000_000
        cli_args.priority_target_lock_steps = 1_000_000_000
        cli_args.priority_target_switch_margin = 1.0
        cli_args.priority_visit_suppression_gain = 0.0
        cli_args.priority_observed_suppression_gain = 0.0
        cli_args.priority_clear_min_weight = 1.0
        cli_args.rl_priority_topic = ""
        os.environ["TB3_RL_FORCE_NO_PRIORITY"] = "1"
        os.environ["TB3_RL_NO_PRIORITY_MODEL_INPUT"] = "1"
        os.environ["TB3_RL_PRIORITY_CLUSTER_SPAWN_INTERVAL_STEPS"] = "1000000000"
        os.environ["TB3_RL_PRIORITY_MAX_SEED_POINTS"] = "0"
    return cli_args

def _force_odom_coordinate_policy(cli_args):
    """Backward-compatible hook.

    이전 버전은 여기서 모든 프레임을 odom으로 강제했지만, SLAM /map과
    RL priority/confidence/task overlay를 함께 RViz에서 볼 때 map->odom TF가
    변하면서 layer가 서로 풀려 보이는 문제가 생긴다.

    이제 CLI가 지정한 --map-frame/--pose-frame/--safety-boundary-frame을
    그대로 보존한다. map-aligned 디버깅은 --map-frame map --pose-frame map
    --rviz-zero-robot-on-reset 조합으로 실행한다.
    """
    return cli_args


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--timesteps", type=int, default=100_000)
    parser.add_argument("--learning-starts", type=int, default=1_000)
    parser.add_argument("--buffer-size", type=int, default=100_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--sac-learning-rate", type=float, default=3e-4)
    # SAC discount factor. 0.995 gives a longer planning horizon than the old hard-coded 0.99.
    parser.add_argument("--sac-gamma", type=float, default=0.995)
    parser.add_argument("--sac-target-entropy", type=float, default=None)
    parser.add_argument("--sac-reset-ent-coef", type=float, default=None)
    parser.add_argument("--sac-min-ent-coef", type=float, default=None)
    parser.add_argument(
        "--sac-use-sde",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use generalized State-Dependent Exploration (gSDE) instead of per-step i.i.d. "
        "Gaussian action noise. At control_dt~0.1s, independent noise averages out over an "
        "episode (law of large numbers) and barely changes the macro-trajectory; gSDE holds a "
        "state-dependent noise matrix fixed for --sac-sde-sample-freq steps so exploration "
        "actually produces different trajectories. Only takes effect when starting a new model "
        "(architecture differs from non-SDE checkpoints, so it cannot be toggled on --load-model).",
    )
    parser.add_argument(
        "--sac-sde-sample-freq",
        type=int,
        default=32,
        help="Steps between gSDE noise matrix resamples. SB3's own default (-1) resamples once "
        "per collect_rollouts() call, which is only --train-freq-steps long (4-32 in this repo's "
        "profiles) -- too short to fix the law-of-large-numbers washout this flag exists for. "
        "Default here (32 steps, ~4s at control_dt=0.12s) is a fixed 'maneuver-length' window "
        "decoupled from train_freq, long enough to produce a different trajectory but short "
        "enough to recover from a bad noise draw before it turns into a sustained collision "
        "course. Ignored unless --sac-use-sde is set.",
    )
    parser.add_argument("--warmup-action-steps", type=int, default=0)
    parser.add_argument("--warmup-action-zero-linear-prob", type=float, default=0.0)
    parser.add_argument("--warmup-action-random-prob", type=float, default=0.0)
    parser.add_argument("--warmup-action-noise-prob", type=float, default=0.0)
    parser.add_argument("--warmup-action-noise-std", type=float, default=0.25)
    # Gamma used only for episode discounted-return diagnostics in the environment overlay/log.
    parser.add_argument("--reward-gamma", type=float, default=0.995)
    parser.add_argument("--control-dt", type=float, default=0.12)
    parser.add_argument("--physics-step-size", type=float, default=0.01)
    parser.add_argument("--max-episode-steps", type=int, default=300)

    parser.add_argument("--namespace", type=str, default="")
    parser.add_argument("--cmd-vel-topic", type=str, default="cmd_vel")

    parser.add_argument("--entity-name", type=str, default=os.environ.get("TB3_RL_ENTITY_NAME", "burger"))
    parser.add_argument(
        "--set-pose-service",
        type=str,
        default="/world/default/set_pose",
    )
    parser.add_argument(
        "--world-control-service",
        type=str,
        default="/world/default/control",
    )


    # One-command simulation launch. If enabled, train/eval starts Gazebo itself
    # before constructing the ROS interface, so /scan, /odom, /clock become available
    # without a separate terminal launch.
    parser.add_argument("--auto-start-gazebo", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--gazebo-launch-package", type=str, default="turtlebot3_gazebo")
    parser.add_argument("--gazebo-launch-file", type=str, default="turtlebot3_house.launch.py")
    parser.add_argument("--gazebo-world", type=str, default="")
    parser.add_argument("--gazebo-turtlebot-model", type=str, default="burger")
    parser.add_argument("--gazebo-use-sim-time", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--gazebo-startup-wait-sec", type=float, default=5.0)
    parser.add_argument("--gazebo-sensor-wait-timeout-sec", type=float, default=45.0)
    parser.add_argument("--gazebo-show-output", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--startup-require-training-geometry",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Before training, require LiDAR to see enough non-max-range obstacle geometry.",
    )
    parser.add_argument("--startup-geometry-wait-sec", type=float, default=8.0)
    parser.add_argument("--startup-min-obstacle-beams", type=int, default=20)
    parser.add_argument("--startup-max-valid-lidar-range-m", type=float, default=3.35)
    parser.add_argument(
        "--gazebo-extra-arg",
        action="append",
        default=[],
        help="Extra raw launch argument, e.g. --gazebo-extra-arg gui:=false",
    )

    parser.add_argument(
        "--training-profile",
        type=str,
        choices=["normal", "fast", "ultrafast", "visual_lite", "real_finetune", "real_robot_finetune", "real"],
        default="normal",
        help=(
            "Runtime throughput profile. ultrafast disables SLAM/RViz/world-step/map-CNN; "
            "fast keeps map-CNN but disables SLAM/RViz/world-step; visual_lite enables sparse RViz maps; real_finetune continues velocity SAC with real-robot-like command limits."
        ),
    )

    parser.add_argument("--disable-pose-reset", action="store_true")
    parser.add_argument("--disable-world-step", action="store_true")
    parser.add_argument("--world-step-target-fraction", type=float, default=0.05)
    parser.add_argument("--world-step-wait-timeout-sec", type=float, default=0.03)
    parser.add_argument("--world-step-sensor-timeout-sec", type=float, default=0.03)
    parser.add_argument("--world-step-stale-warn-every-n", type=int, default=500)
    parser.add_argument("--world-step-auto-disable", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--world-step-stale-limit", type=int, default=10)
    parser.add_argument(
        "--realtime-spin-steps",
        type=int,
        default=2,
        help="When --disable-world-step is active, number of nonblocking ROS spin calls after each command.",
    )
    parser.add_argument(
        "--realtime-spin-timeout-sec",
        type=float,
        default=0.0,
        help="Timeout per ROS spin call when --disable-world-step is active. Keep 0.0 for maximum Gazebo throughput.",
    )
    parser.add_argument(
        "--realtime-sleep-sec",
        type=float,
        default=0.001,
        help="Tiny CPU-yield sleep after command publish when --disable-world-step is active.",
    )
    parser.add_argument(
        "--realtime-enforce-control-dt",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "When --disable-world-step is active, throttle each env step/micro-step "
            "to the requested control_dt in wall-clock time while spinning ROS callbacks. "
            "Use this when step-count based timers must correspond to real seconds."
        ),
    )
    parser.add_argument(
        "--realtime-control-dt-wall-margin-sec",
        type=float,
        default=0.0,
        help="Optional extra wall-clock margin added to control_dt when realtime-enforce-control-dt is enabled.",
    )
    parser.add_argument(
        "--disable-path-reward",
        action="store_true",
        help="Deprecated. Path reward/planning is permanently disabled in this build.",
    )
    parser.add_argument(
        "--disable-wall-proximity-penalty",
        action="store_true",
        help="Disable reward penalties for being close to walls/obstacles. Collision/fallen terminal penalties remain enabled. Useful when Nav2 handles local obstacle clearance.",
    )
    parser.add_argument(
        "--disable-priority-map",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Disable the priority map completely. Keeps observation shape compatible "
            "by zeroing the priority channel, disables priority reward/restart logic, "
            "and suppresses /rl_priority_map publishing unless a topic is explicitly forced."
        ),
    )
    parser.add_argument(
        "--enable-corridor-priority-reward",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Add priority-map direction reward shaping for doorway/corridor-like gaps. Enabled by default.",
    )
    parser.add_argument(
        "--corridor-priority-reward-weight",
        type=float,
        default=0.55,
        help="Weight for doorway/corridor priority reward shaping when --enable-corridor-priority-reward is enabled. Default is reduced so priority clear does not dominate return.",
    )
    parser.add_argument("--fixed-reset-yaw", action="store_true")
    parser.add_argument("--reset-z", type=float, default=0.05)

    parser.add_argument("--collision-threshold", type=float, default=0.16)
    parser.add_argument("--restart-on-collision", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--collision-clear-nav2-costmaps", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--collision-cancel-nav2-goal", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fallen-roll-threshold", type=float, default=0.45)
    parser.add_argument("--fallen-pitch-threshold", type=float, default=0.45)
    parser.add_argument("--terminate-on-out-of-bounds", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--safety-boundary-radius-m", type=float, default=6.0)
    parser.add_argument("--safety-boundary-min-x", type=float, default=-6.0)
    parser.add_argument("--safety-boundary-max-x", type=float, default=6.0)
    parser.add_argument("--safety-boundary-min-y", type=float, default=-6.0)
    parser.add_argument("--safety-boundary-max-y", type=float, default=6.0)
    parser.add_argument("--safety-boundary-max-abs-z", type=float, default=0.45)
    parser.add_argument(
        "--safety-boundary-frame",
        type=str,
        default="odom",
        help="Frame used for out-of-bounds checks. Keep odom by default because SLAM map->odom can drift across resets.",
    )

    parser.add_argument("--max-linear-speed", type=float, default=0.32)
    parser.add_argument("--max-angular-speed", type=float, default=0.75)
    parser.add_argument("--velocity-command-linear-limit", type=float, default=0.0, help="Clamp executed TwistStamped linear.x without changing SAC action_space. Use for resume fine-tuning old wide-action checkpoints under real-robot speed limits.")
    parser.add_argument("--velocity-command-angular-limit", type=float, default=0.0, help="Clamp executed TwistStamped angular.z without changing SAC action_space. Use for resume fine-tuning old wide-action checkpoints under real-robot turn limits.")

    # Policy action 해석 방식.
    # waypoint + polar: SAC는 [거리 비율, 방향 비율]을 내고,
    #                  env 내부 controller가 해당 local waypoint까지 이동한다.
    # waypoint + path는 제거되었으며 입력되면 env에서 polar로 fallback한다.
    # velocity        : 기존 방식처럼 SAC가 [linear_x, angular_z]를 직접 낸다.
    parser.add_argument("--action-mode", type=str, choices=["waypoint", "velocity", "nav2"], default="nav2")
    parser.add_argument("--waypoint-action-type", type=str, choices=["path", "polar"], default="polar")
    parser.add_argument("--waypoint-lateral-max-offset", type=float, default=0.20)
    parser.add_argument("--waypoint-min-distance", type=float, default=1.00)
    parser.add_argument("--waypoint-max-distance", type=float, default=2.20)
    parser.add_argument("--waypoint-max-angle-deg", type=float, default=65.0)
    parser.add_argument("--waypoint-reached-tolerance", type=float, default=0.40)
    parser.add_argument("--waypoint-control-steps", type=int, default=1)
    parser.add_argument("--waypoint-execute-until-reached", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--waypoint-max-control-steps", type=int, default=2)
    parser.add_argument(
        "--waypoint-timeout-sec",
        type=float,
        default=0.20,
        help="Short receding-horizon waypoint timeout. If > 0, override waypoint-max-control-steps using ceil(timeout_sec / control_dt).",
    )
    parser.add_argument("--waypoint-timeout-stop", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--waypoint-linear-kp", type=float, default=0.90)
    parser.add_argument("--waypoint-angular-kp", type=float, default=2.80)
    parser.add_argument("--waypoint-max-yaw-error-for-linear-deg", type=float, default=75.0)
    parser.add_argument("--waypoint-slowdown-distance", type=float, default=0.45)
    parser.add_argument("--waypoint-min-linear-speed", type=float, default=0.08)
    parser.add_argument("--waypoint-disable-arrival-slowdown", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--waypoint-front-stop-distance", type=float, default=0.32)
    parser.add_argument("--waypoint-replan-distance-m", type=float, default=0.12)
    parser.add_argument("--waypoint-replan-heading-deg", type=float, default=18.0)
    parser.add_argument("--waypoint-direct-point-mode", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--waypoint-direct-heading-tolerance-deg", type=float, default=10.0)
    parser.add_argument("--waypoint-direct-drive-heading-limit-deg", type=float, default=28.0)
    parser.add_argument("--waypoint-direct-max-correction-angular", type=float, default=0.35)
    parser.add_argument("--waypoint-direct-min-drive-distance", type=float, default=0.05)
    parser.add_argument("--waypoint-direct-target-sector-deg", type=float, default=14.0)
    parser.add_argument("--waypoint-direct-turn-drive", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--waypoint-direct-turn-drive-max-yaw-deg", type=float, default=65.0)
    parser.add_argument("--waypoint-direct-turn-drive-speed-scale", type=float, default=0.45)
    parser.add_argument("--waypoint-direct-turn-drive-min-speed", type=float, default=0.06)
    parser.add_argument("--reset-pose-max-attempts", type=int, default=8)
    parser.add_argument("--reset-pose-min-clearance-m", type=float, default=0.15)
    parser.add_argument("--reset-pose-validation-wait-sec", type=float, default=0.20)

    # Nav2 NavigateToPose action mode.
    parser.add_argument("--nav2-action-name", type=str, default="/navigate_to_pose")
    parser.add_argument("--nav2-goal-timeout-sec", type=float, default=5.0)
    parser.add_argument("--nav2-control-window-sec", type=float, default=0.90)
    parser.add_argument("--nav2-replan-on-movement", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--nav2-replan-distance-m", type=float, default=0.45)
    parser.add_argument(
        "--nav2-early-replan-remaining-m",
        type=float,
        default=1.05,
        help="In Nav2 streaming mode, ask SAC for the next waypoint before Nav2 slows down at the current local goal.",
    )
    parser.add_argument(
        "--nav2-near-goal-replan-only",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="When true, replan only near the current waypoint. Default false uses movement-gated streaming to avoid stop-and-go.",
    )
    parser.add_argument("--nav2-continuous-goal-update", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--nav2-preempt-without-cancel", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--nav2-wait-timeout-sec", type=float, default=8.0)
    parser.add_argument("--nav2-goal-reached-tolerance", type=float, default=0.35)
    parser.add_argument("--nav2-cancel-on-timeout", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--nav2-cancel-on-reached", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--nav2-send-goal-wait-sec", type=float, default=2.0)
    parser.add_argument("--nav2-cancel-wait-sec", type=float, default=0.0, help="Seconds to wait for Nav2 cancel acknowledgement. 0.0 means fire-and-forget cancel/preempt streaming.")
    parser.add_argument("--nav2-use-goal-orientation", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--auto-start-nav2", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--nav2-launch-package", type=str, default="nav2_bringup")
    parser.add_argument("--nav2-launch-file", type=str, default="navigation_launch.py")
    parser.add_argument("--nav2-params-file", type=str, default="")
    parser.add_argument("--nav2-use-sim-time", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--nav2-startup-timeout-sec", type=float, default=25.0)

    # SLAM local-quality based adaptive speed limiter.
    parser.add_argument("--slam-adaptive-speed", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--slam-local-speed-radius", type=float, default=1.00)
    parser.add_argument("--slam-front-speed-distance", type=float, default=1.20)
    parser.add_argument("--slam-front-speed-half-angle-deg", type=float, default=45.0)
    parser.add_argument("--slam-speed-min-scale", type=float, default=0.30)
    parser.add_argument("--slam-speed-max-scale", type=float, default=1.00)
    parser.add_argument("--slam-speed-local-weight", type=float, default=0.55)
    parser.add_argument("--slam-speed-front-weight", type=float, default=0.25)
    parser.add_argument("--slam-speed-fresh-weight", type=float, default=0.20)
    parser.add_argument("--slam-speed-map-age-soft-limit-sec", type=float, default=3.0)
    parser.add_argument("--slam-speed-known-low-ratio", type=float, default=0.25)
    parser.add_argument("--slam-speed-known-high-ratio", type=float, default=0.85)
    parser.add_argument("--slam-speed-fresh-low-score", type=float, default=0.15)
    parser.add_argument("--slam-speed-smoothing-alpha", type=float, default=0.20)

    parser.add_argument("--waypoint-marker-topic", type=str, default="")
    parser.add_argument("--waypoint-path-topic", type=str, default="")
    parser.add_argument("--waypoint-visual-history-len", type=int, default=80)
    parser.add_argument("--waypoint-visual-publish-every-n", type=int, default=1000000)
    parser.add_argument(
        "--waypoint-show-history",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Show recent waypoint history in RViz. Default shows only current waypoint.",
    )

    # SLAM 관련 옵션.
    parser.add_argument("--slam-map-topic", type=str, default="/map")
    parser.add_argument("--map-frame", type=str, default="map")
    parser.add_argument(
        "--pose-frame",
        type=str,
        default="map",
        help="Runtime pose/control frame. Use map for RViz Fixed Frame=map alignment; odom remains available as a fallback.",
    )
    parser.add_argument("--disable-slam-map", action="store_true")
    parser.add_argument("--auto-start-slam", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--slam-backend",
        type=str,
        choices=["cartographer", "slam_toolbox"],
        default="cartographer",
        help="SLAM backend for /map. Default is Cartographer for both training and real robot.",
    )
    parser.add_argument("--wait-slam-map", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--reset-slam-on-reset", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--rviz-zero-robot-on-reset",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "After Gazebo teleport reset, reset SLAM so RViz/map frame sees "
            "base_footprint near (0,0). Gazebo spawn can still be nonzero/random."
        ),
    )
    parser.add_argument("--rviz-origin-wait-sec", type=float, default=2.0)
    parser.add_argument("--rviz-origin-tolerance-m", type=float, default=0.25)
    parser.add_argument("--restart-slam-on-reset", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--reset-slam-every-n-episodes",
        type=int,
        default=1,
        help="Forced to 1 in this build: SLAM reset is mandatory at every episode/reset.",
    )
    parser.add_argument("--reset-tf-buffer-on-reset", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--slam-reset-service", type=str, default="")
    parser.add_argument("--slam-reset-timeout", type=float, default=8.0)
    parser.add_argument("--slam-reset-warmup-steps", type=int, default=15)
    parser.add_argument("--rl-map-topic", type=str, default="")
    parser.add_argument("--rl-confidence-topic", type=str, default="")
    parser.add_argument("--rl-priority-topic", type=str, default="")
    parser.add_argument("--rl-path-topic", type=str, default="")
    parser.add_argument("--rl-filtered-slam-topic", type=str, default="")
    parser.add_argument("--slam-map-accept-delay-sec", type=float, default=1.0)
    parser.add_argument("--post-reset-stabilize-sec", type=float, default=2.0)
    parser.add_argument("--post-reset-stabilize-spin-steps", type=int, default=12)
    parser.add_argument("--post-reset-ready-gate", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--post-reset-ready-timeout-sec", type=float, default=7.0)
    parser.add_argument("--post-reset-ready-min-known-ratio", type=float, default=0.02)
    parser.add_argument("--post-reset-ready-min-known-cells", type=int, default=40)
    parser.add_argument("--post-reset-ready-min-lidar-beams", type=int, default=30)
    parser.add_argument("--post-reset-ready-require-priority", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--action-sync-reward-gate", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--action-sync-wait-timeout-sec", type=float, default=0.06)
    parser.add_argument("--action-sync-min-scan-age-sec", type=float, default=0.0)
    parser.add_argument("--map-bounds-restart", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--map-bounds-margin-cells", type=int, default=2)
    parser.add_argument("--map-bounds-min-local-known-ratio", type=float, default=0.04)
    parser.add_argument("--map-bounds-min-local-known-cells", type=int, default=12)
    parser.add_argument("--map-bounds-grace-steps", type=int, default=8)
    parser.add_argument("--map-bounds-restart-penalty", type=float, default=0.0)
    parser.add_argument("--slam-map-max-age-sec", type=float, default=3.0)
    # Strict SLAM map gating shared with eval_policy. Defaults are disabled for training unless explicitly enabled.
    parser.add_argument("--no-mandatory-slam-reset-policy", dest="no_mandatory_slam_reset_policy", action="store_true", default=False)
    parser.add_argument("--strict-slam-map-required", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--strict-slam-map-wait-timeout-sec", type=float, default=30.0)
    parser.add_argument("--strict-slam-map-retry-interval-sec", type=float, default=0.50)
    parser.add_argument("--strict-slam-map-min-known-cells", type=int, default=20)
    parser.add_argument("--strict-slam-map-min-known-ratio", type=float, default=0.001)
    parser.add_argument("--reset-x", type=float, default=-2.0)
    parser.add_argument("--reset-y", type=float, default=-0.5)
    parser.add_argument(
        "--reset-pose-mode",
        type=str,
        choices=["fixed", "corners", "house_random", "house_inside_random", "list"],
        default="house_inside_random",
        help=(
            "fixed: always reset to (--reset-x,--reset-y). "
            "corners: randomize around reset point by --reset-offset. "
            "house_random: mixed turtlebot3_house candidate list with LiDAR validation. "
            "house_inside_random: indoor-only turtlebot3_house candidate list. "
            "list: randomize from --reset-pose-list."
        ),
    )
    parser.add_argument("--reset-offset", type=float, default=0.3)
    parser.add_argument(
        "--reset-pose-list",
        type=str,
        default="",
        help='Custom random reset candidates as "x,y;x,y;...", used with --reset-pose-mode list.',
    )

    # Episode restart conditions beyond collision/fall.
    parser.add_argument("--priority-stuck-restart", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--priority-stuck-restart-sec",
        type=float,
        default=0.0,
        help="Restart episode if active nonzero priority remains unresolved for this many seconds. 0 uses --priority-stuck-restart-steps.",
    )
    parser.add_argument("--priority-stuck-restart-steps", type=int, default=100)
    parser.add_argument("--priority-stuck-score-threshold", type=float, default=0.15)
    parser.add_argument("--priority-stuck-clear-gain-threshold", type=float, default=0.03)
    parser.add_argument("--priority-stuck-info-gain-threshold", type=float, default=0.0005)
    parser.add_argument("--priority-stuck-restart-penalty", type=float, default=0.0)

    parser.add_argument("--lidar-empty-restart", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--lidar-empty-timeout-sec",
        type=float,
        default=2.5,
        help="Restart if LiDAR has no valid finite hit below --lidar-empty-max-valid-range-m for this long.",
    )
    parser.add_argument("--lidar-empty-grace-sec", type=float, default=1.0)
    parser.add_argument("--lidar-empty-min-valid-range-m", type=float, default=0.12)
    parser.add_argument("--lidar-empty-max-valid-range-m", type=float, default=3.35)
    parser.add_argument("--lidar-empty-min-valid-beams", type=int, default=2)
    parser.add_argument("--lidar-empty-restart-penalty", type=float, default=0.0)

    # v113: soft terminal for low-information episode tails.
    parser.add_argument("--coverage-stall-terminal", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--coverage-stall-start-steps", type=int, default=1000)
    parser.add_argument("--coverage-stall-window-steps", type=int, default=500)
    parser.add_argument("--coverage-stall-min-slam-new-cells", type=int, default=5)
    parser.add_argument("--coverage-stall-min-confidence-updated-cells", type=int, default=30)
    parser.add_argument("--coverage-stall-min-coverage-delta", type=float, default=0.002)
    parser.add_argument("--coverage-stall-required-consecutive-windows", type=int, default=2)
    parser.add_argument("--coverage-stall-terminal-penalty", type=float, default=0.0)

    # Pure velocity SAC safety shield.  Used only when --action-mode velocity.
    # Policy action remains forward-only; the shield may publish a short reverse
    # TwistStamped when the front sector is too close to an obstacle.
    parser.add_argument("--velocity-safety-backup", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--velocity-safety-trigger-distance-m", type=float, default=0.28)
    parser.add_argument("--velocity-safety-stop-distance-m", type=float, default=0.36)
    parser.add_argument("--velocity-safety-slow-distance-m", type=float, default=0.55)
    parser.add_argument("--velocity-safety-backup-speed-mps", type=float, default=0.08)
    parser.add_argument("--velocity-safety-turn-speed", type=float, default=0.35)
    parser.add_argument("--velocity-safety-backup-steps", type=int, default=4)
    parser.add_argument("--velocity-safety-cooldown-steps", type=int, default=8)
    parser.add_argument("--velocity-safety-penalty", type=float, default=10.0)
    parser.add_argument("--velocity-safety-block-penalty", type=float, default=0.80)
    parser.add_argument("--velocity-safety-slowdown", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--velocity-safety-slow-min-scale", type=float, default=0.20)
    parser.add_argument("--velocity-safety-slow-penalty", type=float, default=1.80)
    parser.add_argument("--velocity-safety-slow-speed-power", type=float, default=1.35)
    parser.add_argument("--velocity-safety-slow-danger-power", type=float, default=1.10)
    parser.add_argument("--velocity-safety-terminal", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--velocity-safety-terminal-distance-m", type=float, default=0.0)
    parser.add_argument("--velocity-safety-terminal-penalty", type=float, default=40.0)
    parser.add_argument("--velocity-safety-terminal-forward-min", type=float, default=0.03)
    # v131: explicit no-backup flag (alias for --no-velocity-safety-backup)
    # --no-velocity-safety-backup is already handled by BooleanOptionalAction above.
    # --max-scan-age-sec: observation freshness guard (sets TB3_RL_MAX_SCAN_AGE_SEC)
    parser.add_argument("--max-scan-age-sec", type=float, default=0.35,
                        help="Max allowed scan age (sec) before obs is marked stale. Set via env TB3_RL_MAX_SCAN_AGE_SEC. Default 0.35.")
    parser.add_argument("--velocity-forward-assist-mps", type=float, default=0.0)
    parser.add_argument("--velocity-forward-assist-angular-threshold", type=float, default=0.20)
    parser.add_argument("--velocity-forward-assist-min-clearance-m", type=float, default=0.45)
    parser.add_argument("--velocity-spin-breaker", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--velocity-spin-breaker-steps", type=int, default=14)
    parser.add_argument("--velocity-spin-breaker-angular-ratio", type=float, default=0.85)
    parser.add_argument("--velocity-spin-breaker-forward-mps", type=float, default=0.035)
    parser.add_argument("--velocity-spin-breaker-angular-scale", type=float, default=0.35)
    parser.add_argument("--velocity-spin-breaker-min-clearance-m", type=float, default=0.48)

    # Reset if the body is bouncing/tilting even before a full fall is detected.
    parser.add_argument("--shake-restart", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--shake-restart-steps", type=int, default=4)
    parser.add_argument("--shake-tilt-threshold", type=float, default=0.12)
    parser.add_argument("--shake-angular-xy-threshold", type=float, default=0.70)
    parser.add_argument("--shake-linear-z-threshold", type=float, default=0.08)
    parser.add_argument("--shake-z-deviation-threshold", type=float, default=0.05)
    parser.add_argument("--shake-ground-min-z", type=float, default=-0.02)
    parser.add_argument("--shake-ground-max-z", type=float, default=0.13)
    parser.add_argument("--shake-leaky-decay", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--shake-yaw-wobble", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--shake-yaw-wobble-grace-steps", type=int, default=80)
    parser.add_argument("--shake-yaw-rate-threshold", type=float, default=0.24)
    parser.add_argument("--shake-cmd-flip-threshold", type=float, default=0.16)
    parser.add_argument("--shake-wobble-window-steps", type=int, default=8)
    parser.add_argument("--shake-wobble-min-flips", type=int, default=2)
    parser.add_argument("--shake-wobble-max-net-motion-m", type=float, default=0.045)
    parser.add_argument("--shake-spin-stall-restart-steps", type=int, default=18)
    parser.add_argument("--shake-restart-penalty", type=float, default=100.0)
    parser.add_argument("--reset-hard-stabilize-reapply", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--reset-hard-stabilize-reapply-interval-sec", type=float, default=0.25)

    # Nav2-only escape behavior.  This uses behavior_server /backup, not direct /cmd_vel.
    parser.add_argument("--nav2-stuck-backup", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--nav2-stuck-backup-action-name", type=str, default="/backup")
    parser.add_argument(
        "--nav2-stuck-backup-sec",
        type=float,
        default=3.0,
        help="Run Nav2 BackUp if accepted Nav2 goals produce less than --nav2-stuck-backup-min-movement-m motion for this many seconds.",
    )
    parser.add_argument("--nav2-stuck-backup-steps", type=int, default=3)
    parser.add_argument("--nav2-stuck-backup-min-movement-m", type=float, default=0.020)
    parser.add_argument("--nav2-stuck-backup-stationary-sec", type=float, default=1.5)
    parser.add_argument("--nav2-stuck-backup-stationary-xy-m", type=float, default=0.025)
    parser.add_argument("--nav2-stuck-backup-stationary-yaw-deg", type=float, default=7.0)
    parser.add_argument("--nav2-stuck-backup-distance-m", type=float, default=0.24)
    parser.add_argument("--nav2-stuck-backup-speed-mps", type=float, default=0.07)
    parser.add_argument("--nav2-stuck-backup-timeout-sec", type=float, default=4.0)
    parser.add_argument("--nav2-stuck-backup-cooldown-sec", type=float, default=5.0)
    parser.add_argument("--nav2-stuck-backup-penalty", type=float, default=0.0)

    # CNN map observation.
    parser.add_argument("--use-map-cnn", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--map-obs-size", type=int, default=48)
    parser.add_argument("--map-obs-size-m", type=float, default=6.0)
    parser.add_argument(
        "--num-lidar-bins",
        type=int,
        default=60,
        help=(
            "Number of LiDAR bins in the policy observation. v5 default is 60. "
            "Checkpoints are only compatible with the same value used during training."
        ),
    )
    parser.add_argument("--cnn-features-dim", type=int, default=48)
    parser.add_argument("--vector-features-dim", type=int, default=128)
    parser.add_argument("--combined-features-dim", type=int, default=192)
    parser.add_argument("--policy-weight-decay", type=float, default=1e-5)
    parser.add_argument("--use-temporal-cnn", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--temporal-history-len", type=int, default=4)
    parser.add_argument("--temporal-features-dim", type=int, default=64)
    parser.add_argument("--front-fov-deg", type=float, default=80.0)
    parser.add_argument("--front-angle-sigma-deg", type=float, default=20.0)
    parser.add_argument("--confidence-max-range", type=float, default=2.0)
    parser.add_argument("--seen-confidence-floor", type=float, default=80.0)
    parser.add_argument("--confidence-decay-per-step", type=float, default=0.0)
    parser.add_argument("--confidence-reward-weight", type=float, default=1.0)
    parser.add_argument("--confidence-fill-reward-weight", type=float, default=0.0)
    parser.add_argument("--confidence-fill-hold-weight", type=float, default=0.0)
    parser.add_argument("--slam-map-update-reward", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--slam-map-update-reward-weight", type=float, default=0.65)
    parser.add_argument("--slam-map-update-reward-norm-cells", type=float, default=50.0)
    parser.add_argument("--slam-map-update-reward-cap", type=float, default=3.0)
    parser.add_argument("--slam-map-update-reward-grace-steps", type=int, default=10)
    parser.add_argument("--reward-positive-log-compress", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--reward-positive-log-alpha", type=float, default=0.50)
    parser.add_argument("--reward-positive-log-max", type=float, default=8.0)
    parser.add_argument("--suppress-gap-confidence", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--gap-occupied-threshold", type=float, default=65.0)
    parser.add_argument("--gap-check-radius-m", type=float, default=1.20)
    parser.add_argument("--gap-min-width-m", type=float, default=0.20)
    parser.add_argument("--gap-max-width-m", type=float, default=2.00)
    parser.add_argument("--map-expand-chunk-cells", type=int, default=64)
    parser.add_argument("--map-publish-every-n", type=int, default=0)
    parser.add_argument("--max-planned-candidates", type=int, default=8)
    parser.add_argument("--max-alternative-paths", type=int, default=5)
    parser.add_argument("--path-visual-publish-every-n", type=int, default=0)
    parser.add_argument("--priority-recompute-interval", type=int, default=16)
    parser.add_argument("--priority-target-lock-steps", type=int, default=16, help="Keep the selected priority/frontier target for at least this many updates unless a much better target appears.")
    parser.add_argument("--priority-target-switch-margin", type=float, default=0.12, help="New target must beat the locked target by this margin during the lock window.")
    parser.add_argument("--priority-visit-suppression-radius-m", type=float, default=0.55)
    parser.add_argument("--priority-visit-suppression-gain", type=float, default=0.35)
    parser.add_argument("--priority-visit-suppression-max", type=float, default=0.85)
    parser.add_argument("--priority-observed-suppression-gain", type=float, default=0.20)
    parser.add_argument("--priority-clear-fov-deg", type=float, default=360.0)
    parser.add_argument("--priority-clear-max-range-m", type=float, default=2.50)
    parser.add_argument("--priority-clear-robot-radius-m", type=float, default=0.45)
    parser.add_argument("--priority-clear-min-value", type=float, default=5.0)
    parser.add_argument("--priority-clear-sigma-m", type=float, default=0.35)
    parser.add_argument("--priority-clear-angle-sigma-deg", type=float, default=180.0)
    parser.add_argument("--priority-clear-min-weight", type=float, default=0.05)
    parser.add_argument("--priority-clear-visit-sigma-m", type=float, default=0.25)
    parser.add_argument("--wall-support-radius-m", type=float, default=0.70)
    parser.add_argument("--wall-support-density-threshold", type=float, default=0.025)
    parser.add_argument("--open-space-front-distance-m", type=float, default=1.80)
    parser.add_argument("--open-space-side-width-m", type=float, default=1.20)
    parser.add_argument("--open-space-forward-penalty", type=float, default=0.45)
    parser.add_argument("--map-keepalive-period-sec", type=float, default=0.0)
    parser.add_argument("--map-live-update-period-sec", type=float, default=0.10)
    parser.add_argument("--debug-input-map", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--debug-input-map-topic-prefix", type=str, default="/rl_debug_input")
    parser.add_argument("--debug-input-map-frame-id", type=str, default="base_link")
    parser.add_argument("--debug-input-map-publish-every-n", type=int, default=50)

    # Training throughput. Off-policy SAC does not require a gradient step after every env step.
    parser.add_argument("--train-freq-steps", type=int, default=8)
    parser.add_argument("--gradient-steps", type=int, default=1)

    # Action filtering / anti-jitter.
    parser.add_argument("--action-smoothing-alpha", type=float, default=0.30)
    parser.add_argument("--max-linear-delta", type=float, default=0.08)
    parser.add_argument("--max-angular-delta", type=float, default=0.20)
    parser.add_argument("--linear-deadband", type=float, default=0.015)
    parser.add_argument("--angular-deadband", type=float, default=0.04)
    parser.add_argument("--enable-motion-mode-hysteresis", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--explored-stall-start-steps", type=int, default=8)
    parser.add_argument("--explored-stall-growth", type=float, default=0.008)
    parser.add_argument("--explored-stall-power", type=float, default=1.45)
    parser.add_argument("--explored-stall-max-penalty", type=float, default=1.20)

    parser.add_argument("--no-check-env", action="store_true", default=True)
    parser.add_argument("--check-env", dest="no_check_env", action="store_false")

    parser.add_argument("--model-dir", type=str, default="rl_models")
    parser.add_argument("--log-dir", type=str, default="rl_logs")
    parser.add_argument("--debug-print-freq", type=int, default=0)
    parser.add_argument("--show-training-progress", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--progress-style", type=str, default="line", choices=["line", "block", "tqdm", "quiet"])
    parser.add_argument("--progress-print-freq", type=int, default=2000)
    parser.add_argument("--progress-window", type=int, default=20)
    parser.add_argument("--progress-csv", type=str, default="")
    parser.add_argument("--progress-csv-flush-every", type=int, default=20)
    parser.add_argument("--sac-verbose", type=int, default=0)
    parser.add_argument("--checkpoint-freq", type=int, default=50_000)
    parser.add_argument("--save-replay-buffer", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save-vecnormalize", action=argparse.BooleanOptionalAction, default=False)

    # Existing model continuation.
    # Observation/action space를 바꾸지 않는 reward-only 수정이면 기존 SAC policy를 그대로 이어 학습할 수 있다.
    parser.add_argument("--load-model", type=str, default="")
    parser.add_argument("--load-replay-buffer", type=str, default="")
    parser.add_argument("--resume-latest", action=argparse.BooleanOptionalAction, default=False, help="Load the newest SAC checkpoint/final model from --model-dir if --load-model is not given.")
    parser.add_argument("--resume-replay-buffer", action=argparse.BooleanOptionalAction, default=True, help="When resuming, auto-load the matching replay buffer if present.")
    parser.add_argument("--sac-reset-critics", action=argparse.BooleanOptionalAction, default=False, help="When loading a model, reinitialize SAC critics/targets but keep the actor. Useful after reward/action-execution changes.")
    parser.add_argument("--continue-timesteps", action=argparse.BooleanOptionalAction, default=True)

    parsed_args, _ = parser.parse_known_args()
    # Backfill options introduced by eval/real-robot patches so old commands do not crash training.
    _defaults = {
        "strict_slam_map_required": False,
        "strict_slam_map_wait_timeout_sec": 30.0,
        "strict_slam_map_retry_interval_sec": 0.50,
        "strict_slam_map_min_known_cells": 20,
        "strict_slam_map_min_known_ratio": 0.001,
        "velocity_spin_breaker": False,
        "velocity_spin_breaker_steps": 14,
        "velocity_spin_breaker_angular_ratio": 0.85,
        "velocity_spin_breaker_forward_mps": 0.035,
        "velocity_spin_breaker_angular_scale": 0.35,
        "velocity_spin_breaker_min_clearance_m": 0.48,
    }
    for _name, _value in _defaults.items():
        if not hasattr(parsed_args, _name):
            setattr(parsed_args, _name, _value)

    parsed_args = _apply_training_profile(parsed_args)
    parsed_args = _apply_rviz_origin_policy(parsed_args)
    parsed_args = _force_nav2_only_policy(parsed_args)
    if not bool(getattr(parsed_args, "no_mandatory_slam_reset_policy", False)):
        parsed_args = _force_mandatory_slam_reset_policy(parsed_args)
    parsed_args = _force_odom_coordinate_policy(parsed_args)
    parsed_args = _force_no_priority_policy(parsed_args)
    return parsed_args


def main(args=None):
    cli_args = parse_args()

    if bool(getattr(cli_args, "disable_priority_map", False)) or str(os.environ.get("TB3_RL_FORCE_NO_PRIORITY", "0")).strip().lower() not in {"0", "false", "no", "off"}:
        # v93 hard no-priority observation contract.
        os.environ["TB3_RL_FORCE_NO_PRIORITY"] = "1"
        os.environ["TB3_RL_NO_PRIORITY_MODEL_INPUT"] = "1"
        cli_args.disable_priority_map = True
        cli_args.enable_corridor_priority_reward = False
        cli_args.corridor_priority_reward_weight = 0.0
        cli_args.post_reset_ready_require_priority = False
        cli_args.priority_stuck_restart = False
        cli_args.rl_priority_topic = ""

    # Prevent FastDDS /dev/shm port exhaustion at the source: disable the
    # shared-memory transport (UDPv4 loopback only) BEFORE rclpy.init / any
    # participant is created.  This is the robust fix for the repeated
    # open_and_lock_file / init_port fastrtps_portNNNN crashes that occur when
    # Cartographer is restarted every episode.  Respects a user-provided
    # FASTRTPS_DEFAULT_PROFILES_FILE if one is already set.
    if str(os.environ.get("TB3_RL_DISABLE_SHM_TRANSPORT", "1")).strip().lower() not in {"0", "false", "no", "off"}:
        try:
            if ensure_non_shm_fastdds_profile is not None:
                ensure_non_shm_fastdds_profile(logger=None)
        except Exception:
            pass

    # One-time wipe of stale SHM lock files from previous crashed runs.  This is
    # safe ONLY here, before we start ROS/Gazebo ourselves.  If an external
    # Gazebo is already running we skip the wipe to avoid disturbing it.
    try:
        if bool(getattr(cli_args, "auto_start_gazebo", False)) and clean_fastdds_shm is not None:
            clean_fastdds_shm(logger=None)
    except Exception:
        pass

    gazebo_proc = _start_gazebo_if_requested(cli_args)

    # v131: propagate --max-scan-age-sec to environment variable used in _get_obs
    _max_scan_age = float(getattr(cli_args, "max_scan_age_sec", 0.35))
    if _max_scan_age > 0.0:
        os.environ["TB3_RL_MAX_SCAN_AGE_SEC"] = str(_max_scan_age)

    rclpy.init(args=args)

    use_slam_map = not cli_args.disable_slam_map
    map_topic = cli_args.slam_map_topic if use_slam_map else ""

    ros = TurtleBot3RosInterface(
        namespace=cli_args.namespace,
        cmd_vel_topic=cli_args.cmd_vel_topic,
        map_topic=map_topic,
        enable_tf=use_slam_map,
        enable_cmd_vel_pub=(cli_args.action_mode != "nav2"),
        auto_start_slam=use_slam_map and cli_args.auto_start_slam,
        slam_backend=cli_args.slam_backend,
        slam_reset_service=cli_args.slam_reset_service,
        use_sim_time=cli_args.gazebo_use_sim_time,
    )

    sensor_wait_timeout = max(10.0, float(getattr(cli_args, "gazebo_sensor_wait_timeout_sec", 45.0)))
    if not ros.wait_for_sensor_ready(timeout_sec=sensor_wait_timeout):
        if not bool(getattr(cli_args, "auto_start_gazebo", False)):
            print(
                "\n[TRAIN STARTUP ERROR] /scan and /odom were not received.\n"
                "This training process no longer starts Gazebo by default.\n"
                "Start the simulator in a separate terminal first:\n"
                "  cd ~/Desktop/ROS2_Project\n"
                "  bash run_gazebo.sh\n"
                "Then start training in another terminal:\n"
                "  cd ~/Desktop/ROS2_Project\n"
                "  bash run_train.sh\n",
                flush=True,
            )
        ros.destroy_node()
        rclpy.shutdown()
        _terminate_process(gazebo_proc, "Gazebo")
        return

    if not _wait_for_training_geometry_ready(ros, cli_args):
        ros.destroy_node()
        rclpy.shutdown()
        _terminate_process(gazebo_proc, "Gazebo")
        return

    if use_slam_map and cli_args.auto_start_slam:
        ros.ensure_slam_toolbox(timeout_sec=cli_args.slam_reset_timeout)

    if use_slam_map and cli_args.wait_slam_map:
        ros.wait_for_slam_map_ready(timeout_sec=max(10.0, cli_args.slam_reset_timeout))

    raw_env = GazeboNavEnv(
        ros_interface=ros,
        entity_name=cli_args.entity_name,
        set_pose_service=cli_args.set_pose_service,
        enable_pose_reset=not cli_args.disable_pose_reset,
        random_reset_yaw=not cli_args.fixed_reset_yaw,
        reset_z=cli_args.reset_z,
        control_dt=cli_args.control_dt,
        physics_step_size=cli_args.physics_step_size,
        max_episode_steps=cli_args.max_episode_steps,
        collision_threshold=cli_args.collision_threshold,
        restart_on_collision=cli_args.restart_on_collision,
        collision_clear_nav2_costmaps=cli_args.collision_clear_nav2_costmaps,
        collision_cancel_nav2_goal=cli_args.collision_cancel_nav2_goal,
        fallen_roll_threshold=cli_args.fallen_roll_threshold,
        fallen_pitch_threshold=cli_args.fallen_pitch_threshold,
        terminate_on_out_of_bounds=cli_args.terminate_on_out_of_bounds,
        safety_boundary_radius_m=cli_args.safety_boundary_radius_m,
        safety_boundary_min_x=cli_args.safety_boundary_min_x,
        safety_boundary_max_x=cli_args.safety_boundary_max_x,
        safety_boundary_min_y=cli_args.safety_boundary_min_y,
        safety_boundary_max_y=cli_args.safety_boundary_max_y,
        safety_boundary_max_abs_z=cli_args.safety_boundary_max_abs_z,
        safety_boundary_frame=cli_args.safety_boundary_frame,
        world_control_service=cli_args.world_control_service,
        use_world_step=not cli_args.disable_world_step,
        world_step_target_fraction=cli_args.world_step_target_fraction,
        world_step_wait_timeout_sec=cli_args.world_step_wait_timeout_sec,
        world_step_sensor_timeout_sec=cli_args.world_step_sensor_timeout_sec,
        world_step_stale_warn_every_n=cli_args.world_step_stale_warn_every_n,
        world_step_auto_disable_on_stale=cli_args.world_step_auto_disable,
        world_step_stale_limit=cli_args.world_step_stale_limit,
        realtime_spin_steps=cli_args.realtime_spin_steps,
        realtime_spin_timeout_sec=cli_args.realtime_spin_timeout_sec,
        realtime_sleep_sec=cli_args.realtime_sleep_sec,
        realtime_enforce_control_dt=cli_args.realtime_enforce_control_dt,
        realtime_control_dt_wall_margin_sec=cli_args.realtime_control_dt_wall_margin_sec,
        disable_path_reward=True,
        disable_wall_proximity_penalty=cli_args.disable_wall_proximity_penalty,
        enable_corridor_priority_reward=(
            bool(getattr(cli_args, "enable_corridor_priority_reward", True))
            and not bool(getattr(cli_args, "disable_priority_map", False))
        ),
        disable_priority_map=bool(getattr(cli_args, "disable_priority_map", False)),
        corridor_priority_reward_weight=(
            0.0 if bool(getattr(cli_args, "disable_priority_map", False))
            else cli_args.corridor_priority_reward_weight
        ),
        confidence_reward_weight=cli_args.confidence_reward_weight,
        confidence_fill_reward_weight=cli_args.confidence_fill_reward_weight,
        confidence_fill_hold_weight=cli_args.confidence_fill_hold_weight,
        slam_map_update_reward=cli_args.slam_map_update_reward,
        slam_map_update_reward_weight=cli_args.slam_map_update_reward_weight,
        slam_map_update_reward_norm_cells=cli_args.slam_map_update_reward_norm_cells,
        slam_map_update_reward_cap=cli_args.slam_map_update_reward_cap,
        slam_map_update_reward_grace_steps=cli_args.slam_map_update_reward_grace_steps,
        reward_positive_log_compress=cli_args.reward_positive_log_compress,
        reward_positive_log_alpha=cli_args.reward_positive_log_alpha,
        reward_positive_log_max=cli_args.reward_positive_log_max,
        max_linear_speed=cli_args.max_linear_speed,
        max_angular_speed=cli_args.max_angular_speed,
        velocity_command_linear_limit=cli_args.velocity_command_linear_limit,
        velocity_command_angular_limit=cli_args.velocity_command_angular_limit,
        action_mode=cli_args.action_mode,
        waypoint_action_type="polar",
        waypoint_lateral_max_offset=cli_args.waypoint_lateral_max_offset,
        waypoint_min_distance=cli_args.waypoint_min_distance,
        waypoint_max_distance=cli_args.waypoint_max_distance,
        waypoint_max_angle_deg=cli_args.waypoint_max_angle_deg,
        waypoint_reached_tolerance=cli_args.waypoint_reached_tolerance,
        waypoint_control_steps=cli_args.waypoint_control_steps,
        waypoint_execute_until_reached=cli_args.waypoint_execute_until_reached,
        waypoint_max_control_steps=cli_args.waypoint_max_control_steps,
        waypoint_timeout_sec=cli_args.waypoint_timeout_sec,
        waypoint_timeout_stop=cli_args.waypoint_timeout_stop,
        waypoint_linear_kp=cli_args.waypoint_linear_kp,
        waypoint_angular_kp=cli_args.waypoint_angular_kp,
        waypoint_max_yaw_error_for_linear_deg=cli_args.waypoint_max_yaw_error_for_linear_deg,
        waypoint_slowdown_distance=cli_args.waypoint_slowdown_distance,
        waypoint_min_linear_speed=cli_args.waypoint_min_linear_speed,
        waypoint_disable_arrival_slowdown=cli_args.waypoint_disable_arrival_slowdown,
        waypoint_front_stop_distance=cli_args.waypoint_front_stop_distance,
        waypoint_replan_distance_m=cli_args.waypoint_replan_distance_m,
        waypoint_replan_heading_deg=cli_args.waypoint_replan_heading_deg,
        waypoint_direct_point_mode=cli_args.waypoint_direct_point_mode,
        waypoint_direct_heading_tolerance_deg=cli_args.waypoint_direct_heading_tolerance_deg,
        waypoint_direct_drive_heading_limit_deg=cli_args.waypoint_direct_drive_heading_limit_deg,
        waypoint_direct_max_correction_angular=cli_args.waypoint_direct_max_correction_angular,
        waypoint_direct_min_drive_distance=cli_args.waypoint_direct_min_drive_distance,
        waypoint_direct_target_sector_deg=cli_args.waypoint_direct_target_sector_deg,
        waypoint_direct_turn_drive=cli_args.waypoint_direct_turn_drive,
        waypoint_direct_turn_drive_max_yaw_deg=cli_args.waypoint_direct_turn_drive_max_yaw_deg,
        waypoint_direct_turn_drive_speed_scale=cli_args.waypoint_direct_turn_drive_speed_scale,
        waypoint_direct_turn_drive_min_speed=cli_args.waypoint_direct_turn_drive_min_speed,
        reset_pose_max_attempts=cli_args.reset_pose_max_attempts,
        reset_pose_min_clearance_m=cli_args.reset_pose_min_clearance_m,
        reset_pose_validation_wait_sec=cli_args.reset_pose_validation_wait_sec,
        post_reset_stabilize_sec=cli_args.post_reset_stabilize_sec,
        post_reset_stabilize_spin_steps=cli_args.post_reset_stabilize_spin_steps,
        nav2_action_name=cli_args.nav2_action_name,
        nav2_goal_timeout_sec=cli_args.nav2_goal_timeout_sec,
        nav2_control_window_sec=cli_args.nav2_control_window_sec,
        nav2_replan_on_movement=cli_args.nav2_replan_on_movement,
        nav2_replan_distance_m=cli_args.nav2_replan_distance_m,
        nav2_early_replan_remaining_m=cli_args.nav2_early_replan_remaining_m,
        nav2_near_goal_replan_only=cli_args.nav2_near_goal_replan_only,
        nav2_continuous_goal_update=cli_args.nav2_continuous_goal_update,
        nav2_preempt_without_cancel=cli_args.nav2_preempt_without_cancel,
        nav2_wait_timeout_sec=cli_args.nav2_wait_timeout_sec,
        nav2_goal_reached_tolerance=cli_args.nav2_goal_reached_tolerance,
        nav2_cancel_on_timeout=cli_args.nav2_cancel_on_timeout,
        nav2_cancel_on_reached=cli_args.nav2_cancel_on_reached,
        nav2_send_goal_wait_sec=cli_args.nav2_send_goal_wait_sec,
        nav2_cancel_wait_sec=cli_args.nav2_cancel_wait_sec,
        nav2_use_goal_orientation=cli_args.nav2_use_goal_orientation,
        nav2_auto_start=cli_args.auto_start_nav2,
        nav2_launch_package=cli_args.nav2_launch_package,
        nav2_launch_file=cli_args.nav2_launch_file,
        nav2_params_file=cli_args.nav2_params_file,
        nav2_use_sim_time=cli_args.nav2_use_sim_time,
        nav2_startup_timeout_sec=cli_args.nav2_startup_timeout_sec,
        slam_adaptive_speed=cli_args.slam_adaptive_speed,
        slam_local_speed_radius=cli_args.slam_local_speed_radius,
        slam_front_speed_distance=cli_args.slam_front_speed_distance,
        slam_front_speed_half_angle_deg=cli_args.slam_front_speed_half_angle_deg,
        slam_speed_min_scale=cli_args.slam_speed_min_scale,
        slam_speed_max_scale=cli_args.slam_speed_max_scale,
        slam_speed_local_weight=cli_args.slam_speed_local_weight,
        slam_speed_front_weight=cli_args.slam_speed_front_weight,
        slam_speed_fresh_weight=cli_args.slam_speed_fresh_weight,
        slam_speed_map_age_soft_limit_sec=cli_args.slam_speed_map_age_soft_limit_sec,
        slam_speed_known_low_ratio=cli_args.slam_speed_known_low_ratio,
        slam_speed_known_high_ratio=cli_args.slam_speed_known_high_ratio,
        slam_speed_fresh_low_score=cli_args.slam_speed_fresh_low_score,
        slam_speed_smoothing_alpha=cli_args.slam_speed_smoothing_alpha,
        waypoint_marker_topic=cli_args.waypoint_marker_topic,
        waypoint_path_topic=cli_args.waypoint_path_topic,
        waypoint_visual_history_len=cli_args.waypoint_visual_history_len,
        waypoint_visual_publish_every_n=cli_args.waypoint_visual_publish_every_n,
        waypoint_show_history=cli_args.waypoint_show_history,
        use_slam_map=use_slam_map,
        map_frame=cli_args.map_frame,
        pose_frame=cli_args.pose_frame,
        rl_map_topic=cli_args.rl_map_topic,
        rl_confidence_topic=cli_args.rl_confidence_topic,
        rl_priority_topic=cli_args.rl_priority_topic,
        rl_path_topic=cli_args.rl_path_topic,
        rl_filtered_slam_topic=cli_args.rl_filtered_slam_topic,
        slam_map_accept_delay_sec=cli_args.slam_map_accept_delay_sec,
        slam_map_max_age_sec=cli_args.slam_map_max_age_sec,
        strict_slam_map_required=cli_args.strict_slam_map_required,
        strict_slam_map_wait_timeout_sec=cli_args.strict_slam_map_wait_timeout_sec,
        strict_slam_map_retry_interval_sec=cli_args.strict_slam_map_retry_interval_sec,
        strict_slam_map_min_known_cells=cli_args.strict_slam_map_min_known_cells,
        strict_slam_map_min_known_ratio=cli_args.strict_slam_map_min_known_ratio,
        post_reset_ready_gate=cli_args.post_reset_ready_gate,
        post_reset_ready_timeout_sec=cli_args.post_reset_ready_timeout_sec,
        post_reset_ready_min_known_ratio=cli_args.post_reset_ready_min_known_ratio,
        post_reset_ready_min_known_cells=cli_args.post_reset_ready_min_known_cells,
        post_reset_ready_min_lidar_beams=cli_args.post_reset_ready_min_lidar_beams,
        post_reset_ready_require_priority=cli_args.post_reset_ready_require_priority,
        action_sync_reward_gate=cli_args.action_sync_reward_gate,
        action_sync_wait_timeout_sec=cli_args.action_sync_wait_timeout_sec,
        action_sync_min_scan_age_sec=cli_args.action_sync_min_scan_age_sec,
        map_bounds_restart=cli_args.map_bounds_restart,
        map_bounds_margin_cells=cli_args.map_bounds_margin_cells,
        map_bounds_min_local_known_ratio=cli_args.map_bounds_min_local_known_ratio,
        map_bounds_min_local_known_cells=cli_args.map_bounds_min_local_known_cells,
        map_bounds_grace_steps=cli_args.map_bounds_grace_steps,
        map_bounds_restart_penalty=cli_args.map_bounds_restart_penalty,
        reset_x=cli_args.reset_x,
        reset_y=cli_args.reset_y,
        reset_pose_mode=cli_args.reset_pose_mode,
        reset_offset=cli_args.reset_offset,
        reset_pose_list=cli_args.reset_pose_list,
        rviz_zero_robot_on_reset=cli_args.rviz_zero_robot_on_reset,
        rviz_origin_wait_sec=cli_args.rviz_origin_wait_sec,
        rviz_origin_tolerance_m=cli_args.rviz_origin_tolerance_m,
        reset_slam_on_reset=use_slam_map and cli_args.reset_slam_on_reset,
        restart_slam_on_reset=use_slam_map and cli_args.restart_slam_on_reset,
        reset_slam_every_n_episodes=cli_args.reset_slam_every_n_episodes,
        slam_reset_timeout_sec=cli_args.slam_reset_timeout,
        slam_reset_warmup_steps=cli_args.slam_reset_warmup_steps,
        use_map_cnn=cli_args.use_map_cnn,
        map_obs_size=cli_args.map_obs_size,
        map_obs_size_m=cli_args.map_obs_size_m,
        use_temporal_cnn=cli_args.use_temporal_cnn,
        num_lidar_bins=cli_args.num_lidar_bins,
        temporal_history_len=cli_args.temporal_history_len,
        front_fov_deg=cli_args.front_fov_deg,
        front_angle_sigma_deg=cli_args.front_angle_sigma_deg,
        confidence_max_range=cli_args.confidence_max_range,
        seen_confidence_floor=cli_args.seen_confidence_floor,
        confidence_decay_per_step=cli_args.confidence_decay_per_step,
        suppress_gap_confidence=cli_args.suppress_gap_confidence,
        gap_occupied_threshold=cli_args.gap_occupied_threshold,
        gap_check_radius_m=cli_args.gap_check_radius_m,
        gap_min_width_m=cli_args.gap_min_width_m,
        gap_max_width_m=cli_args.gap_max_width_m,
        map_expand_chunk_cells=cli_args.map_expand_chunk_cells,
        map_publish_every_n=cli_args.map_publish_every_n,
        max_planned_candidates=cli_args.max_planned_candidates,
        max_alternative_paths=cli_args.max_alternative_paths,
        path_visual_publish_every_n=cli_args.path_visual_publish_every_n,
        priority_recompute_interval=cli_args.priority_recompute_interval,
        priority_target_lock_steps=cli_args.priority_target_lock_steps,
        priority_target_switch_margin=cli_args.priority_target_switch_margin,
        priority_visit_suppression_radius_m=cli_args.priority_visit_suppression_radius_m,
        priority_visit_suppression_gain=cli_args.priority_visit_suppression_gain,
        priority_visit_suppression_max=cli_args.priority_visit_suppression_max,
        priority_observed_suppression_gain=cli_args.priority_observed_suppression_gain,
        priority_clear_fov_deg=cli_args.priority_clear_fov_deg,
        priority_clear_max_range_m=cli_args.priority_clear_max_range_m,
        priority_clear_robot_radius_m=cli_args.priority_clear_robot_radius_m,
        priority_clear_min_value=cli_args.priority_clear_min_value,
        priority_clear_sigma_m=cli_args.priority_clear_sigma_m,
        priority_clear_angle_sigma_deg=cli_args.priority_clear_angle_sigma_deg,
        priority_clear_min_weight=cli_args.priority_clear_min_weight,
        priority_clear_visit_sigma_m=cli_args.priority_clear_visit_sigma_m,
        wall_support_radius_m=cli_args.wall_support_radius_m,
        wall_support_density_threshold=cli_args.wall_support_density_threshold,
        open_space_front_distance_m=cli_args.open_space_front_distance_m,
        open_space_side_width_m=cli_args.open_space_side_width_m,
        open_space_forward_penalty=cli_args.open_space_forward_penalty,
        map_keepalive_period_sec=cli_args.map_keepalive_period_sec,
        map_live_update_period_sec=cli_args.map_live_update_period_sec,
        debug_input_map=cli_args.debug_input_map,
        debug_input_map_topic_prefix=cli_args.debug_input_map_topic_prefix,
        debug_input_map_frame_id=cli_args.debug_input_map_frame_id,
        debug_input_map_publish_every_n=cli_args.debug_input_map_publish_every_n,
        action_smoothing_alpha=cli_args.action_smoothing_alpha,
        max_linear_delta=cli_args.max_linear_delta,
        max_angular_delta=cli_args.max_angular_delta,
        linear_deadband=cli_args.linear_deadband,
        angular_deadband=cli_args.angular_deadband,
        enable_motion_mode_hysteresis=cli_args.enable_motion_mode_hysteresis,
        explored_stall_start_steps=cli_args.explored_stall_start_steps,
        explored_stall_growth=cli_args.explored_stall_growth,
        explored_stall_power=cli_args.explored_stall_power,
        explored_stall_max_penalty=cli_args.explored_stall_max_penalty,
        priority_stuck_restart=cli_args.priority_stuck_restart,
        priority_stuck_restart_sec=cli_args.priority_stuck_restart_sec,
        priority_stuck_restart_steps=cli_args.priority_stuck_restart_steps,
        priority_stuck_score_threshold=cli_args.priority_stuck_score_threshold,
        priority_stuck_clear_gain_threshold=cli_args.priority_stuck_clear_gain_threshold,
        priority_stuck_info_gain_threshold=cli_args.priority_stuck_info_gain_threshold,
        priority_stuck_restart_penalty=cli_args.priority_stuck_restart_penalty,
        lidar_empty_restart=cli_args.lidar_empty_restart,
        lidar_empty_timeout_sec=cli_args.lidar_empty_timeout_sec,
        lidar_empty_grace_sec=cli_args.lidar_empty_grace_sec,
        lidar_empty_min_valid_range_m=cli_args.lidar_empty_min_valid_range_m,
        lidar_empty_max_valid_range_m=cli_args.lidar_empty_max_valid_range_m,
        lidar_empty_min_valid_beams=cli_args.lidar_empty_min_valid_beams,
        lidar_empty_restart_penalty=cli_args.lidar_empty_restart_penalty,
        coverage_stall_terminal=cli_args.coverage_stall_terminal,
        coverage_stall_start_steps=cli_args.coverage_stall_start_steps,
        coverage_stall_window_steps=cli_args.coverage_stall_window_steps,
        coverage_stall_min_slam_new_cells=cli_args.coverage_stall_min_slam_new_cells,
        coverage_stall_min_confidence_updated_cells=cli_args.coverage_stall_min_confidence_updated_cells,
        coverage_stall_min_coverage_delta=cli_args.coverage_stall_min_coverage_delta,
        coverage_stall_required_consecutive_windows=cli_args.coverage_stall_required_consecutive_windows,
        coverage_stall_terminal_penalty=cli_args.coverage_stall_terminal_penalty,
        velocity_safety_backup=cli_args.velocity_safety_backup,
        velocity_safety_trigger_distance_m=cli_args.velocity_safety_trigger_distance_m,
        velocity_safety_stop_distance_m=cli_args.velocity_safety_stop_distance_m,
        velocity_safety_slow_distance_m=cli_args.velocity_safety_slow_distance_m,
        velocity_safety_backup_speed_mps=cli_args.velocity_safety_backup_speed_mps,
        velocity_safety_turn_speed=cli_args.velocity_safety_turn_speed,
        velocity_safety_backup_steps=cli_args.velocity_safety_backup_steps,
        velocity_safety_cooldown_steps=cli_args.velocity_safety_cooldown_steps,
        velocity_safety_penalty=cli_args.velocity_safety_penalty,
        velocity_safety_block_penalty=cli_args.velocity_safety_block_penalty,
        velocity_safety_slowdown=cli_args.velocity_safety_slowdown,
        velocity_safety_slow_min_scale=cli_args.velocity_safety_slow_min_scale,
        velocity_safety_slow_penalty=cli_args.velocity_safety_slow_penalty,
        velocity_safety_slow_speed_power=cli_args.velocity_safety_slow_speed_power,
        velocity_safety_slow_danger_power=cli_args.velocity_safety_slow_danger_power,
        velocity_safety_terminal=cli_args.velocity_safety_terminal,
        velocity_safety_terminal_distance_m=cli_args.velocity_safety_terminal_distance_m,
        velocity_safety_terminal_penalty=cli_args.velocity_safety_terminal_penalty,
        velocity_safety_terminal_forward_min=cli_args.velocity_safety_terminal_forward_min,
        velocity_forward_assist_mps=cli_args.velocity_forward_assist_mps,
        velocity_forward_assist_angular_threshold=cli_args.velocity_forward_assist_angular_threshold,
        velocity_forward_assist_min_clearance_m=cli_args.velocity_forward_assist_min_clearance_m,
        velocity_spin_breaker=cli_args.velocity_spin_breaker,
        velocity_spin_breaker_steps=cli_args.velocity_spin_breaker_steps,
        velocity_spin_breaker_angular_ratio=cli_args.velocity_spin_breaker_angular_ratio,
        velocity_spin_breaker_forward_mps=cli_args.velocity_spin_breaker_forward_mps,
        velocity_spin_breaker_angular_scale=cli_args.velocity_spin_breaker_angular_scale,
        velocity_spin_breaker_min_clearance_m=cli_args.velocity_spin_breaker_min_clearance_m,
        shake_restart=cli_args.shake_restart,
        shake_restart_steps=cli_args.shake_restart_steps,
        shake_tilt_threshold=cli_args.shake_tilt_threshold,
        shake_angular_xy_threshold=cli_args.shake_angular_xy_threshold,
        shake_linear_z_threshold=cli_args.shake_linear_z_threshold,
        shake_z_deviation_threshold=cli_args.shake_z_deviation_threshold,
        shake_ground_min_z=cli_args.shake_ground_min_z,
        shake_ground_max_z=cli_args.shake_ground_max_z,
        shake_leaky_decay=cli_args.shake_leaky_decay,
        shake_yaw_wobble=cli_args.shake_yaw_wobble,
        shake_yaw_wobble_grace_steps=cli_args.shake_yaw_wobble_grace_steps,
        shake_yaw_rate_threshold=cli_args.shake_yaw_rate_threshold,
        shake_cmd_flip_threshold=cli_args.shake_cmd_flip_threshold,
        shake_wobble_window_steps=cli_args.shake_wobble_window_steps,
        shake_wobble_min_flips=cli_args.shake_wobble_min_flips,
        shake_wobble_max_net_motion_m=cli_args.shake_wobble_max_net_motion_m,
        shake_spin_stall_restart_steps=cli_args.shake_spin_stall_restart_steps,
        shake_restart_penalty=cli_args.shake_restart_penalty,
        reset_hard_stabilize_reapply=cli_args.reset_hard_stabilize_reapply,
        reset_hard_stabilize_reapply_interval_sec=cli_args.reset_hard_stabilize_reapply_interval_sec,
        nav2_stuck_backup=cli_args.nav2_stuck_backup,
        nav2_stuck_backup_action_name=cli_args.nav2_stuck_backup_action_name,
        nav2_stuck_backup_sec=cli_args.nav2_stuck_backup_sec,
        nav2_stuck_backup_steps=cli_args.nav2_stuck_backup_steps,
        nav2_stuck_backup_min_movement_m=cli_args.nav2_stuck_backup_min_movement_m,
        nav2_stuck_backup_stationary_sec=cli_args.nav2_stuck_backup_stationary_sec,
        nav2_stuck_backup_stationary_xy_m=cli_args.nav2_stuck_backup_stationary_xy_m,
        nav2_stuck_backup_stationary_yaw_deg=cli_args.nav2_stuck_backup_stationary_yaw_deg,
        nav2_stuck_backup_distance_m=cli_args.nav2_stuck_backup_distance_m,
        nav2_stuck_backup_speed_mps=cli_args.nav2_stuck_backup_speed_mps,
        nav2_stuck_backup_timeout_sec=cli_args.nav2_stuck_backup_timeout_sec,
        nav2_stuck_backup_cooldown_sec=cli_args.nav2_stuck_backup_cooldown_sec,
        nav2_stuck_backup_penalty=cli_args.nav2_stuck_backup_penalty,
        reward_gamma=max(0.0, min(0.9999, float(cli_args.reward_gamma))),
    )

    if not cli_args.no_check_env:
        ros.get_logger().info("Running Stable-Baselines3 env checker...")
        check_env(raw_env, warn=True)
        ros.get_logger().info("Env checker passed.")
    else:
        ros.get_logger().info(
            "Skipping Stable-Baselines3 env checker for live ROS/Gazebo training. "
            "Use --check-env only for offline API validation."
        )

    env = Monitor(raw_env)

    model_dir = Path(cli_args.model_dir)
    log_dir = Path(cli_args.log_dir)

    model_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    # Fail fast if the model directory is not writable or is nearly out of space,
    # so a long run does not crash only at the final save.
    try:
        _probe = model_dir / ".write_test"
        _probe.write_text("ok")
        _probe.unlink()
    except Exception as _werr:
        print(f"[WARN] model_dir may not be writable: {model_dir} | {type(_werr).__name__}: {_werr}", flush=True)
    try:
        import shutil as _shutil
        _free_gib = _shutil.disk_usage(str(model_dir)).free / (1024 ** 3)
        if _free_gib < 1.0:
            print(f"[WARN] Low disk space for model_dir: ~{_free_gib:.2f} GiB free. "
                  "Model/checkpoint saves may fail. Free up space before a long run.", flush=True)
        else:
            print(f"[INFO] model_dir free space: ~{_free_gib:.1f} GiB", flush=True)
    except Exception:
        pass

    if bool(getattr(cli_args, "resume_latest", False)) and not str(cli_args.load_model).strip():
        latest_model = _find_latest_sac_model(model_dir)
        if latest_model is None:
            ros.get_logger().warn(
                f"--resume-latest requested, but no SAC model was found in {model_dir}. Starting a new model."
            )
        else:
            cli_args.load_model = str(latest_model)
            ros.get_logger().info(f"RESUME_LATEST | selected model={latest_model}")

    if str(cli_args.load_model).strip():
        requested_model = Path(cli_args.load_model).expanduser()
        if not _is_valid_zip_model(requested_model):
            fallback_model = _find_latest_sac_model(model_dir)
            if fallback_model is not None and fallback_model != requested_model:
                ros.get_logger().warn(
                    "LOAD_MODEL_INVALID | "
                    f"requested={requested_model} is empty/corrupt; using fallback={fallback_model}"
                )
                cli_args.load_model = str(fallback_model)
            else:
                ros.get_logger().error(
                    "LOAD_MODEL_INVALID | "
                    f"requested={requested_model} is empty/corrupt and no valid fallback exists; starting a new model"
                )
                cli_args.load_model = ""

    if cli_args.load_model and bool(getattr(cli_args, "resume_replay_buffer", True)) and not str(cli_args.load_replay_buffer).strip():
        guessed_rb = _guess_replay_buffer_path(Path(cli_args.load_model).expanduser(), model_dir)
        if guessed_rb is not None:
            cli_args.load_replay_buffer = str(guessed_rb)
            ros.get_logger().info(f"RESUME_LATEST | selected replay_buffer={guessed_rb}")

    policy_kwargs = dict(
        # Smaller actor/critic. The feature extractor already encodes map, LiDAR,
        # and temporal state; oversized SAC heads tend to overfit a fixed Gazebo
        # world and amplify reward exploits.
        net_arch=dict(
            pi=[128, 128],
            qf=[128, 128],
        ),
        optimizer_kwargs=dict(
            weight_decay=float(cli_args.policy_weight_decay),
        ),
    )

    action_dim = int(math.prod(env.action_space.shape)) if hasattr(env.action_space, "shape") else 2
    effective_target_entropy = cli_args.sac_target_entropy
    if bool(getattr(cli_args, "sac_use_sde", False)) and not cli_args.load_model:
        # gSDE freezes a state-dependent noise matrix for sde_sample_freq steps instead of
        # resampling i.i.d. noise every step. The whole point is to make the existing entropy
        # budget translate into real trajectory diversity instead of self-cancelling jitter --
        # not to also cut that budget. So leave target_entropy at the standard SB3 default
        # (-action_dim); the only gSDE-specific safety knob here is use_expln=True, which keeps
        # the state-dependent noise scale from growing unbounded as latent features grow during
        # training (SB3's documented recommendation when use_sde=True).
        policy_kwargs.update(use_expln=True)
        if effective_target_entropy is None:
            ros.get_logger().info(
                "SAC_SDE_TARGET_ENTROPY | --sac-use-sde is on: keeping standard "
                f"target_entropy=auto ({-float(action_dim):.2f}) -- gSDE makes the same entropy "
                "budget more effective (correlated noise over "
                f"{cli_args.sac_sde_sample_freq} steps instead of self-cancelling i.i.d. noise), "
                "not a reason to explore less. policy_kwargs.use_expln=True bounds noise growth. "
                "Override with --sac-target-entropy if you want to tune this explicitly."
            )

    policy_name = "MlpPolicy"

    if cli_args.use_map_cnn:
        policy_name = "MultiInputPolicy"
        policy_kwargs.update(
            features_extractor_class=MapVectorFeatureExtractor,
            features_extractor_kwargs=dict(
                map_features_dim=cli_args.cnn_features_dim,
                vector_features_dim=cli_args.vector_features_dim,
                temporal_features_dim=cli_args.temporal_features_dim,
                combined_features_dim=cli_args.combined_features_dim,
                use_temporal_cnn=cli_args.use_temporal_cnn,
                lidar_dim=cli_args.num_lidar_bins,
            ),
        )

    ros.get_logger().info(
        "TRAIN_CONFIG | "
        f"profile={cli_args.training_profile} | "
        f"policy={policy_name} | "
        f"map_cnn={cli_args.use_map_cnn} map=({4 if bool(getattr(cli_args, 'disable_priority_map', False)) else 5},{cli_args.map_obs_size},{cli_args.map_obs_size}) lidar_bins={cli_args.num_lidar_bins} | "
        f"temporal={cli_args.use_temporal_cnn} hist={cli_args.temporal_history_len} | "
        f"frames map={cli_args.map_frame} pose={cli_args.pose_frame} safety={cli_args.safety_boundary_frame} | "
        f"priority={cli_args.rl_priority_topic or '(off)'} filtered_slam={cli_args.rl_filtered_slam_topic or '(off)'} | "
        f"timesteps={cli_args.timesteps} model_dir={cli_args.model_dir} | "
        f"train_freq={cli_args.train_freq_steps} grad_steps={cli_args.gradient_steps} | "
        f"lr={float(cli_args.sac_learning_rate):.6f} "
        f"sac_gamma={max(0.0, min(0.9999, float(cli_args.sac_gamma))):.4f} "
        f"target_entropy={effective_target_entropy if effective_target_entropy is not None else 'auto'} "
        f"reset_alpha={cli_args.sac_reset_ent_coef if cli_args.sac_reset_ent_coef is not None else 'off'} "
        f"warmup_mix=({int(cli_args.warmup_action_steps)},"
        f"rand={float(cli_args.warmup_action_random_prob):.2f},"
        f"noise={float(cli_args.warmup_action_noise_prob):.2f},"
        f"std={float(cli_args.warmup_action_noise_std):.2f}) "
        f"reward_gamma={max(0.0, min(0.9999, float(cli_args.reward_gamma))):.4f} | "
        f"cmd_limit=({float(getattr(cli_args, 'velocity_command_linear_limit', 0.0)):.3f},"
        f"{float(getattr(cli_args, 'velocity_command_angular_limit', 0.0)):.3f}) | "
        f"priority_lock={int(getattr(cli_args, 'priority_target_lock_steps', 16))}/"
        f"{float(getattr(cli_args, 'priority_target_switch_margin', 0.12)):.2f}"
    )
    ros.get_logger().debug(
        f"TRAIN_CONFIG_VERBOSE | "
        f"front_fov_deg={cli_args.front_fov_deg}, "
        f"cnn_features_dim={cli_args.cnn_features_dim}, "
        f"vector_features_dim={cli_args.vector_features_dim}, "
        f"temporal_features_dim={cli_args.temporal_features_dim}, "
        f"map_encoder=onehot_slam_geometry_robot_centric_5ch, "
        f"debug_input_map={cli_args.debug_input_map}:{cli_args.debug_input_map_topic_prefix}:"
        f"frame={cli_args.debug_input_map_frame_id}:every={cli_args.debug_input_map_publish_every_n}, "
        f"slam_map_accept_delay={cli_args.slam_map_accept_delay_sec:.2f}s, "
        f"slam_map_max_age={cli_args.slam_map_max_age_sec:.2f}s, "
        f"priority_gap_width=[{cli_args.gap_min_width_m:.2f},{cli_args.gap_max_width_m:.2f}]m, "
        f"map_expand_chunk_cells={cli_args.map_expand_chunk_cells}, "
        f"map_publish_every_n={cli_args.map_publish_every_n}, "
        f"max_planned_candidates={cli_args.max_planned_candidates}, "
        f"max_alternative_paths={cli_args.max_alternative_paths}, "
        f"priority_recompute_interval={cli_args.priority_recompute_interval}, "
        f"priority_clear_fov={cli_args.priority_clear_fov_deg:.1f}deg, "
        f"priority_clear_range={cli_args.priority_clear_max_range_m:.2f}m, "
        f"sac_net_arch=pi[128,128]/qf[128,128], "
        f"save_replay_buffer={cli_args.save_replay_buffer}, "
        f"load_model={cli_args.load_model or '(new)'}, "
        f"load_replay_buffer={cli_args.load_replay_buffer or '(none)'}, "
        f"continue_timesteps={cli_args.continue_timesteps}, "
        f"target_entropy={effective_target_entropy if effective_target_entropy is not None else 'auto'}, "
        f"reset_alpha={cli_args.sac_reset_ent_coef if cli_args.sac_reset_ent_coef is not None else 'off'}, "
        f"min_alpha={cli_args.sac_min_ent_coef if cli_args.sac_min_ent_coef is not None else 'off'}, "
        f"lr={float(cli_args.sac_learning_rate):.6f}, "
        f"warmup_actions={int(cli_args.warmup_action_steps)}, "
        f"weight_decay={cli_args.policy_weight_decay}"
    )

    sac_device = str(os.environ.get("TB3_RL_SAC_DEVICE", "auto") or "auto").strip() or "auto"
    ros.get_logger().info(f"SAC_DEVICE_REQUEST | device={sac_device}")

    # ---- Replay-buffer memory guard -------------------------------------------
    # SAC's DictReplayBuffer preallocates a dense numpy array for EVERY observation
    # key: shape (buffer_size, n_envs, *key_shape) float32, and again for the next
    # observation.  With temporal keys like map_seq (H,5,32,32) this explodes:
    # e.g. buffer_size=500000 * (16*5*32*32) * 4B * 2 ~= 300+ GiB and the process
    # is killed with numpy ArrayMemoryError before training starts.  Estimate the
    # requirement up front and clamp buffer_size to fit a memory budget so the run
    # starts instead of dying.  Disable with TB3_RL_DISABLE_BUFFER_GUARD=1.
    if str(os.environ.get("TB3_RL_DISABLE_BUFFER_GUARD", "0")).strip().lower() in {"0", "false", "no", "off", ""}:
        try:
            import numpy as _np

            obs_space = env.observation_space
            per_step_obs_bytes = 0
            if hasattr(obs_space, "spaces"):
                for _k, _sp in obs_space.spaces.items():
                    n = int(_np.prod(_sp.shape)) if _sp.shape else 1
                    per_step_obs_bytes += n * 4  # float32
            else:
                n = int(_np.prod(obs_space.shape)) if obs_space.shape else 1
                per_step_obs_bytes = n * 4

            # obs + next_obs are both stored.
            bytes_per_slot = per_step_obs_bytes * 2
            requested = int(cli_args.buffer_size)
            requested_bytes = bytes_per_slot * requested

            # Memory budget: env override, else a fraction of total system RAM.
            budget_gib_env = str(os.environ.get("TB3_RL_BUFFER_MEM_BUDGET_GIB", "")).strip()
            if budget_gib_env:
                budget_bytes = float(budget_gib_env) * (1024 ** 3)
            else:
                budget_bytes = None
                try:
                    page = os.sysconf("SC_PAGE_SIZE")
                    total = os.sysconf("SC_PHYS_PAGES")
                    if page > 0 and total > 0:
                        budget_bytes = 0.35 * float(page) * float(total)
                except Exception:
                    budget_bytes = None
                if budget_bytes is None:
                    budget_bytes = 8.0 * (1024 ** 3)  # conservative fallback

            req_gib = requested_bytes / (1024 ** 3)
            bud_gib = budget_bytes / (1024 ** 3)
            ros.get_logger().info(
                f"REPLAY_BUFFER_MEM | per_step_obs={per_step_obs_bytes/1024:.1f}KiB "
                f"requested_buffer={requested} (~{req_gib:.1f}GiB) budget=~{bud_gib:.1f}GiB"
            )

            if requested_bytes > budget_bytes and bytes_per_slot > 0:
                safe_size = max(int(budget_bytes // bytes_per_slot), 1000)
                safe_size = min(safe_size, requested)
                if safe_size < requested:
                    ros.get_logger().warn(
                        "REPLAY_BUFFER_CLAMP | requested buffer_size="
                        f"{requested} (~{req_gib:.1f}GiB) exceeds memory budget "
                        f"(~{bud_gib:.1f}GiB); clamping to {safe_size}. "
                        "Override with --buffer-size, TB3_RL_BUFFER_MEM_BUDGET_GIB, "
                        "or disable via TB3_RL_DISABLE_BUFFER_GUARD=1."
                    )
                    cli_args.buffer_size = int(safe_size)
        except Exception as _mem_exc:
            try:
                ros.get_logger().warn(f"REPLAY_BUFFER_MEM_GUARD_SKIPPED | {type(_mem_exc).__name__}: {_mem_exc}")
            except Exception:
                pass
    # ---------------------------------------------------------------------------

    if cli_args.load_model:
        load_path = Path(cli_args.load_model).expanduser()
        if not load_path.exists():
            raise FileNotFoundError(f"--load-model not found: {load_path}")

        ros.get_logger().info(f"Loading existing SAC model: {load_path}")
        model = SAC.load(
            str(load_path),
            env=env,
            tensorboard_log=str(log_dir),
            print_system_info=False,
            device=sac_device,
            custom_objects={"replay_buffer_class": SkipStoreDictReplayBuffer},
        )
        if bool(getattr(cli_args, "sac_use_sde", False)) and not bool(getattr(model, "use_sde", False)):
            ros.get_logger().warn(
                "SAC_USE_SDE_IGNORED | --sac-use-sde was requested but the loaded checkpoint "
                "was trained without gSDE. gSDE changes the actor's log_std parameter shape, so "
                "it cannot be toggled on a resumed model without breaking the loaded weights. "
                "Start a fresh model (no --load-model) to train with gSDE."
            )
        # Ensure the skip-store buffer class sticks even if the loaded checkpoint
        # recorded the default DictReplayBuffer.  If the buffer is (re)created
        # later it will use our subclass.
        try:
            model.replay_buffer_class = SkipStoreDictReplayBuffer
            if getattr(model, "replay_buffer", None) is not None and not isinstance(model.replay_buffer, SkipStoreDictReplayBuffer):
                # Rebuild an empty skip-store buffer with the same parameters so
                # subsequent transitions honor skip_store.  A loaded buffer (via
                # --load-replay-buffer below) overrides this afterward.
                model.replay_buffer = SkipStoreDictReplayBuffer(
                    model.buffer_size,
                    model.observation_space,
                    model.action_space,
                    device=model.device,
                    n_envs=model.n_envs,
                    optimize_memory_usage=getattr(model, "optimize_memory_usage", False),
                )
        except Exception as _rb_exc:
            try:
                ros.get_logger().warn(f"SKIP_STORE_BUFFER_SETUP | could not enforce on load: {type(_rb_exc).__name__}: {_rb_exc}")
            except Exception:
                pass
        # Keep the loaded weights, but apply safe current CLI throughput knobs.
        # SB3 stores train_freq as an internal TrainFreq object, so use the
        # algorithm helper when available instead of assigning a raw tuple.
        try:
            model.train_freq = model._convert_train_freq((max(int(cli_args.train_freq_steps), 1), "step"))
        except Exception:
            pass
        model.gradient_steps = max(int(cli_args.gradient_steps), 0)
        model.batch_size = int(cli_args.batch_size)
        model.learning_starts = max(int(cli_args.learning_starts), 0)
        model.learning_rate = float(cli_args.sac_learning_rate)
        model.lr_schedule = get_schedule_fn(float(cli_args.sac_learning_rate))
        # Allow longer-horizon fine-tuning even when loading an existing SAC checkpoint.
        model.gamma = max(0.0, min(0.9999, float(cli_args.sac_gamma)))
        if cli_args.sac_target_entropy is not None:
            model.target_entropy = float(cli_args.sac_target_entropy)

        # Reward-only 변경은 policy 구조를 바꾸지 않으므로 env만 교체해서 이어 학습 가능하다.
        # 단, replay buffer는 SAC.save()에 포함되지 않는다. 저장해 둔 buffer가 있으면 별도로 로드한다.
        if cli_args.load_replay_buffer:
            rb_path = Path(cli_args.load_replay_buffer).expanduser()
            if not rb_path.exists():
                raise FileNotFoundError(f"--load-replay-buffer not found: {rb_path}")
            model.load_replay_buffer(str(rb_path))
            ros.get_logger().info(f"Loaded replay buffer: {rb_path}")
        else:
            ros.get_logger().warn(
                "No replay buffer loaded. The policy/critics are continued, "
                "but the off-policy replay buffer starts from the loaded model default state. "
                "For future continuation, train with --save-replay-buffer."
            )
    else:
        model = SAC(
            policy=policy_name,
            env=env,
            learning_rate=float(cli_args.sac_learning_rate),
            buffer_size=cli_args.buffer_size,
            learning_starts=cli_args.learning_starts,
            batch_size=cli_args.batch_size,
            tau=0.005,
            gamma=max(0.0, min(0.9999, float(cli_args.sac_gamma))),
            train_freq=(max(int(cli_args.train_freq_steps), 1), "step"),
            gradient_steps=max(int(cli_args.gradient_steps), 0),
            ent_coef="auto",
            target_entropy=effective_target_entropy if effective_target_entropy is not None else "auto",
            use_sde=bool(getattr(cli_args, "sac_use_sde", False)),
            sde_sample_freq=int(getattr(cli_args, "sac_sde_sample_freq", -1)),
            verbose=int(getattr(cli_args, "sac_verbose", 0)),
            tensorboard_log=str(log_dir),
            policy_kwargs=policy_kwargs,
            device=sac_device,
            replay_buffer_class=SkipStoreDictReplayBuffer,
        )

    if bool(getattr(cli_args, "sac_reset_critics", False)):
        _reset_sac_critics(model, logger=ros.get_logger())
    _reset_sac_entropy_coefficient(model, cli_args.sac_reset_ent_coef, logger=ros.get_logger())
    _install_warmup_action_mixer(
        model,
        warmup_steps=cli_args.warmup_action_steps,
        zero_linear_prob=cli_args.warmup_action_zero_linear_prob,
        random_prob=cli_args.warmup_action_random_prob,
        noise_prob=cli_args.warmup_action_noise_prob,
        noise_std=cli_args.warmup_action_noise_std,
        logger=ros.get_logger(),
    )

    try:
        ros.get_logger().info(f"SAC_DEVICE_ACTIVE | device={getattr(model, 'device', 'unknown')}")
    except Exception:
        pass

    debug_callback = DebugCallback(print_freq=cli_args.debug_print_freq)

    progress_csv = str(cli_args.progress_csv or "").strip()
    if not progress_csv and bool(cli_args.show_training_progress):
        progress_csv = str(log_dir / "training_progress.csv")

    progress_callback = None
    if bool(cli_args.show_training_progress):
        progress_callback = TrainingProgressCallback(
            total_timesteps=cli_args.timesteps,
            print_freq=cli_args.progress_print_freq,
            window_size=cli_args.progress_window,
            csv_path=progress_csv,
            progress_style=str(getattr(cli_args, "progress_style", "line")),
            csv_flush_every=int(getattr(cli_args, "progress_csv_flush_every", 20)),
        )

    try:
        keep_last_ckpts = int(os.environ.get("TB3_RL_KEEP_LAST_CHECKPOINTS", "2") or 2)
    except Exception:
        keep_last_ckpts = 2
    keep_last_ckpts = max(int(keep_last_ckpts), 1)

    checkpoint_callback = RotatingCheckpointCallback(
        save_freq=cli_args.checkpoint_freq,
        save_path=str(model_dir),
        name_prefix="sac_turtlebot3_burger_checkpoint",
        save_replay_buffer=bool(cli_args.save_replay_buffer),
        save_vecnormalize=bool(cli_args.save_vecnormalize),
        keep_last=keep_last_ckpts,
        verbose=1,
    )

    try:
        callbacks = [checkpoint_callback]
        if cli_args.sac_min_ent_coef is not None and float(cli_args.sac_min_ent_coef) > 0.0:
            callbacks.append(EntropyCoefficientFloorCallback(cli_args.sac_min_ent_coef, verbose=1))
        if progress_callback is not None:
            callbacks.append(progress_callback)
        if cli_args.debug_print_freq > 0:
            callbacks.append(debug_callback)

        if str(os.environ.get("TB3_RL_TIME_GRAD_UPDATE", "0")).strip().lower() in {"1", "true", "yes", "on"}:
            _orig_train = model.train
            _grad_update_stats = {"n": 0, "acc": 0.0}

            def _timed_train(*_targs, **_tkwargs):
                _t0 = time.perf_counter()
                _result = _orig_train(*_targs, **_tkwargs)
                _dt = time.perf_counter() - _t0
                _grad_update_stats["n"] += 1
                _grad_update_stats["acc"] += _dt
                n = _grad_update_stats["n"]
                if n % 5 == 0 or _dt > 0.5:
                    ros.get_logger().warn(
                        "GRAD_UPDATE_TIME | "
                        f"call={n} last={_dt * 1000.0:.1f}ms avg={_grad_update_stats['acc'] / n * 1000.0:.1f}ms"
                    )
                return _result

            model.train = _timed_train

        learn_error = None
        try:
            model.learn(
                total_timesteps=cli_args.timesteps,
                callback=callbacks,
                log_interval=4,
                reset_num_timesteps=not bool(cli_args.continue_timesteps),
            )
        except KeyboardInterrupt:
            learn_error = "KeyboardInterrupt"
            # OffPolicyAlgorithm.learn() calls callback.on_training_end() only
            # on a *normal* loop exit, not when an exception unwinds out of it
            # -- so our rich.live.Live (if active) is never told to stop and
            # would otherwise keep holding the terminal region while the
            # emergency-save messages below try to print.
            if progress_callback is not None and getattr(progress_callback, "_rich_live", None) is not None:
                try:
                    progress_callback._rich_live.stop()
                except Exception:
                    pass
            try:
                ros.get_logger().warn(
                    "TRAINING_INTERRUPTED | KeyboardInterrupt received; saving emergency checkpoint."
                )
            except Exception:
                pass
        except Exception as exc:
            learn_error = f"{type(exc).__name__}: {exc}"
            if progress_callback is not None and getattr(progress_callback, "_rich_live", None) is not None:
                try:
                    progress_callback._rich_live.stop()
                except Exception:
                    pass
            try:
                ros.get_logger().error(
                    f"TRAINING_ABORTED | learn() raised; saving emergency checkpoint | {learn_error}"
                )
            except Exception:
                pass

        if learn_error is not None:
            # Best-effort emergency save so a long run is never lost on a late crash.
            try:
                emergency_path = model_dir / "sac_turtlebot3_burger_emergency"
                emergency_zip = _safe_save_sac_model(model, emergency_path, logger=ros.get_logger())
                ros.get_logger().info(f"Saved emergency model to {emergency_zip}")
                if bool(cli_args.save_replay_buffer):
                    em_replay = model_dir / "sac_turtlebot3_burger_emergency_replay_buffer.pkl"
                    model.save_replay_buffer(str(em_replay))
                    ros.get_logger().info(f"Saved emergency replay buffer to {em_replay}")
            except Exception as save_exc:
                try:
                    ros.get_logger().error(f"EMERGENCY_SAVE_FAILED | {type(save_exc).__name__}: {save_exc}")
                except Exception:
                    pass

        # Final save.  Wrap in try/except so a late disk/permission error does not
        # discard the trained model silently or crash before cleanup.  On failure,
        # retry once into a timestamped fallback path so a long run is never lost.
        save_path = model_dir / "sac_turtlebot3_burger"
        try:
            saved_zip = _safe_save_sac_model(model, save_path, logger=ros.get_logger())
            if bool(cli_args.save_replay_buffer):
                replay_path = model_dir / "sac_turtlebot3_burger_replay_buffer.pkl"
                model.save_replay_buffer(str(replay_path))
                ros.get_logger().info(f"Saved replay buffer to {replay_path}")
            ros.get_logger().info(f"Saved model to {saved_zip}")
        except Exception as final_exc:
            try:
                ros.get_logger().error(
                    f"FINAL_SAVE_FAILED | {type(final_exc).__name__}: {final_exc} | trying fallback path"
                )
            except Exception:
                pass
            try:
                import time as _t
                fb = model_dir / f"sac_turtlebot3_burger_savefail_{int(_t.time())}"
                fb_zip = _safe_save_sac_model(model, fb, logger=ros.get_logger())
                ros.get_logger().info(f"Saved model to fallback {fb_zip}")
            except Exception as fb_exc:
                # Last resort: try the home directory so the weights are not lost.
                try:
                    home_fb = Path.home() / f"sac_turtlebot3_burger_rescue_{int(time.time())}"
                    home_fb_zip = _safe_save_sac_model(model, home_fb, logger=ros.get_logger())
                    ros.get_logger().error(f"FINAL_SAVE_RESCUE | saved to {home_fb_zip}")
                except Exception as rescue_exc:
                    try:
                        ros.get_logger().error(
                            f"FINAL_SAVE_RESCUE_FAILED | {type(fb_exc).__name__}/{type(rescue_exc).__name__} | model weights could not be written"
                        )
                    except Exception:
                        pass

    finally:
        ros.stop_robot()
        env.close()
        ros.destroy_node()
        rclpy.shutdown()
        _terminate_process(gazebo_proc, "Gazebo")


if __name__ == "__main__":
    main()
