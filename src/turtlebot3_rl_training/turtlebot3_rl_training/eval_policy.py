import argparse
import atexit
import os
import signal
import subprocess
import time
from pathlib import Path
from typing import Optional

import numpy as np
import rclpy
from stable_baselines3 import SAC



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

from turtlebot3_rl_training.feature_extractor import MapVectorFeatureExtractor
from turtlebot3_rl_training.gazebo_nav_env import GazeboNavEnv
from turtlebot3_rl_training.ros_interface import TurtleBot3RosInterface

try:
    from turtlebot3_rl_training.process_cleanup import (
        clean_fastdds_shm,
        ensure_non_shm_fastdds_profile,
    )
except Exception:  # pragma: no cover
    try:
        from process_cleanup import clean_fastdds_shm, ensure_non_shm_fastdds_profile  # type: ignore
    except Exception:
        clean_fastdds_shm = None
        ensure_non_shm_fastdds_profile = None


def _apply_rviz_origin_policy(cli_args):
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
    return cli_args




def _force_mandatory_slam_reset_policy(cli_args):
    """Evaluation uses the same invariant as training: reset SLAM at every respawn."""
    cli_args.disable_slam_map = False
    cli_args.auto_start_slam = True
    cli_args.wait_slam_map = True
    cli_args.reset_slam_on_reset = True
    cli_args.restart_slam_on_reset = True
    cli_args.reset_slam_every_n_episodes = 1
    cli_args.reset_tf_buffer_on_reset = True
    cli_args.map_frame = "map"
    cli_args.pose_frame = "map"
    cli_args.safety_boundary_frame = "odom"
    if not str(getattr(cli_args, "rl_filtered_slam_topic", "") or "").strip():
        cli_args.rl_filtered_slam_topic = "/rl_filtered_slam_map"
    if not str(getattr(cli_args, "rl_priority_topic", "") or "").strip():
        cli_args.rl_priority_topic = "/rl_priority_map"
    if not str(getattr(cli_args, "rl_map_topic", "") or "").strip():
        cli_args.rl_map_topic = "/rl_task_map"
    if not str(getattr(cli_args, "rl_confidence_topic", "") or "").strip():
        cli_args.rl_confidence_topic = "/rl_confidence_map"
    cli_args.slam_map_accept_delay_sec = max(float(getattr(cli_args, "slam_map_accept_delay_sec", 1.0)), 1.2)
    cli_args.slam_map_max_age_sec = max(float(getattr(cli_args, "slam_map_max_age_sec", 3.0)), 3.0)
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


def _force_nav2_only_policy(cli_args):
    """Keep old Nav2 eval defaults only for Nav2 checkpoints.

    Pure-velocity SAC checkpoints use action=[linear_x, angular_z].  For those,
    forcing eval to Nav2 corrupts the action semantics.
    """
    requested = str(getattr(cli_args, "action_mode", "nav2") or "nav2").strip().lower()
    if requested == "velocity":
        cli_args.action_mode = "velocity"
        # In Gazebo velocity eval we can still use world stepping unless the user
        # explicitly disabled it.  Real-robot mode overrides this later.
        if not bool(getattr(cli_args, "real_robot", False)):
            cli_args.disable_world_step = bool(getattr(cli_args, "disable_world_step", False))
        if not str(getattr(cli_args, "waypoint_marker_topic", "") or "").strip():
            cli_args.waypoint_marker_topic = "/rl_debug_overlay"
        cli_args.waypoint_visual_publish_every_n = 1
        cli_args.disable_wall_proximity_penalty = False
        cli_args.collision_clear_nav2_costmaps = False
        cli_args.collision_cancel_nav2_goal = False
        cli_args.auto_start_nav2 = False
        return cli_args

    if requested != "nav2":
        print(f"[NAV2_ONLY] overriding --action-mode {requested!r} -> 'nav2'")
    cli_args.action_mode = "nav2"
    cli_args.disable_world_step = True
    if not str(getattr(cli_args, "waypoint_marker_topic", "") or "").strip():
        cli_args.waypoint_marker_topic = "/rl_waypoint_marker"
    cli_args.waypoint_visual_publish_every_n = 1
    cli_args.disable_wall_proximity_penalty = False
    cli_args.collision_clear_nav2_costmaps = False
    cli_args.collision_cancel_nav2_goal = True
    cli_args.nav2_continuous_goal_update = True
    cli_args.nav2_preempt_without_cancel = True
    cli_args.nav2_cancel_on_timeout = False
    cli_args.nav2_cancel_on_reached = False
    cli_args.nav2_goal_reached_tolerance = max(float(getattr(cli_args, "nav2_goal_reached_tolerance", 0.30)), 0.30)
    cli_args.waypoint_reached_tolerance = max(float(getattr(cli_args, "waypoint_reached_tolerance", 0.32)), 0.32)
    cli_args.nav2_goal_timeout_sec = min(max(float(getattr(cli_args, "nav2_goal_timeout_sec", 0.75)), 0.45), 1.00)
    cli_args.nav2_control_window_sec = min(max(float(getattr(cli_args, "nav2_control_window_sec", 0.18)), 0.12), 0.25)
    cli_args.nav2_replan_distance_m = min(max(float(getattr(cli_args, "nav2_replan_distance_m", 0.10)), 0.06), 0.16)
    cli_args.nav2_send_goal_wait_sec = min(max(float(getattr(cli_args, "nav2_send_goal_wait_sec", 0.25)), 0.05), 0.30)
    if hasattr(cli_args, "nav2_cancel_wait_sec"):
        cli_args.nav2_cancel_wait_sec = 0.0
    return cli_args


def _apply_real_robot_policy(cli_args):
    """Make eval_policy safe for a physical TurtleBot3.

    Gazebo-specific mechanisms are disabled.  The learned policy still publishes
    TwistStamped through TurtleBot3RosInterface; if the physical base only accepts
    geometry_msgs/Twist, run a separate adapter or switch the base driver to a
    stamped cmd_vel input.
    """
    if not bool(getattr(cli_args, "real_robot", False)):
        return cli_args

    cli_args.action_mode = "velocity"
    cli_args.disable_pose_reset = True
    cli_args.disable_world_step = True
    cli_args.auto_start_gazebo = False
    cli_args.gazebo_use_sim_time = False
    cli_args.nav2_use_sim_time = False
    cli_args.auto_start_nav2 = False
    # For the physical robot, launch SLAM internally using the real-robot-safe
    # scan-fixed + slam_toolbox config generated by TurtleBot3RosInterface.
    # This keeps the training/eval map pipeline identical and avoids manually
    # starting slam_toolbox with inconsistent parameters.
    cli_args.auto_start_slam = True
    cli_args.wait_slam_map = True
    cli_args.reset_slam_on_reset = True
    cli_args.restart_slam_on_reset = True
    cli_args.reset_slam_every_n_episodes = 1
    cli_args.reset_tf_buffer_on_reset = True

    cli_args.map_frame = "map"
    cli_args.pose_frame = "map"
    cli_args.safety_boundary_frame = "map"
    cli_args.waypoint_marker_topic = str(getattr(cli_args, "waypoint_marker_topic", "") or "/rl_debug_overlay")
    cli_args.waypoint_visual_publish_every_n = 1

    # v2: keep the policy observation distribution close to training.
    # Older real-robot eval disabled priority by default, which zeroed the 5th
    # CNN map channel for checkpoints trained with priority enabled.
    if bool(getattr(cli_args, "real_robot_disable_priority", False)):
        cli_args.disable_priority_map = True
        cli_args.enable_corridor_priority_reward = False
        cli_args.corridor_priority_reward_weight = 0.0
        cli_args.post_reset_ready_require_priority = False
        cli_args.priority_stuck_restart = False
        cli_args.rl_priority_topic = ""
    else:
        cli_args.disable_priority_map = False
        if not str(getattr(cli_args, "rl_priority_topic", "") or "").strip():
            cli_args.rl_priority_topic = "/rl_priority_map"

    # Real robots need wall-clock pacing.  The Gazebo real-time fallback sleeps only
    # briefly by default to maximize sim throughput; that is unsafe on hardware.
    cli_args.realtime_spin_steps = max(int(getattr(cli_args, "realtime_spin_steps", 8)), 8)
    cli_args.realtime_spin_timeout_sec = max(float(getattr(cli_args, "realtime_spin_timeout_sec", 0.002)), 0.002)
    cli_args.realtime_sleep_sec = max(float(getattr(cli_args, "realtime_sleep_sec", 0.0)), float(getattr(cli_args, "control_dt", 0.10)))

    # v2: respect CLI speed bounds so eval action_space can match the checkpoint.
    # The physical safety shield still clips dangerous commands downstream.
    cli_args.max_linear_speed = float(getattr(cli_args, "max_linear_speed", 0.14))
    cli_args.max_angular_speed = float(getattr(cli_args, "max_angular_speed", 0.50))
    cli_args.velocity_safety_backup = True
    cli_args.velocity_safety_penalty = float(getattr(cli_args, "velocity_safety_penalty", 10.0))

    if getattr(cli_args, "manual_reset_prompt", None) is None:
        cli_args.manual_reset_prompt = True
    return cli_args


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        type=str,
        default="rl_models/sac_turtlebot3_burger.zip",
    )
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument(
        "--eval-deterministic",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Use deterministic SAC actor during evaluation. Disable this with "
            "--no-eval-deterministic to sample the stochastic SAC policy. "
            "This is useful when a partially trained or priority-mismatched "
            "checkpoint saturates to a nearly constant action."
        ),
    )
    parser.add_argument(
        "--rescale-model-action-to-env-space",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "When checkpoint and eval action bounds differ, linearly remap "
            "model-space actions into the current env action space instead of "
            "hard clipping. This prevents low-speed eval from freezing at "
            "exact max commands such as (+0.100,-0.400)."
        ),
    )
    parser.add_argument("--control-dt", type=float, default=0.12)
    parser.add_argument("--physics-step-size", type=float, default=0.01)
    parser.add_argument("--max-episode-steps", type=int, default=300)

    parser.add_argument("--namespace", type=str, default="")
    parser.add_argument("--cmd-vel-topic", type=str, default="cmd_vel")

    parser.add_argument("--entity-name", type=str, default="turtlebot3_burger")
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
        "--gazebo-extra-arg",
        action="append",
        default=[],
        help="Extra raw launch argument, e.g. --gazebo-extra-arg gui:=false",
    )


    # Physical TurtleBot3 evaluation mode.  This disables Gazebo pose/world control
    # and uses manual reset prompts between episodes.
    parser.add_argument("--real-robot", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--disable-priority-map", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--real-robot-disable-priority", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--manual-reset-prompt",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Pause before each episode so the operator can place the real robot and press ENTER.",
    )
    parser.add_argument("--real-robot-start-delay-sec", type=float, default=0.0)
    parser.add_argument("--real-robot-stop-between-episodes", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--disable-pose-reset", action="store_true")
    parser.add_argument("--disable-world-step", action="store_true")
    parser.add_argument("--world-step-target-fraction", type=float, default=0.05)
    parser.add_argument("--world-step-wait-timeout-sec", type=float, default=0.03)
    parser.add_argument("--world-step-sensor-timeout-sec", type=float, default=0.03)
    parser.add_argument("--world-step-stale-warn-every-n", type=int, default=500)
    parser.add_argument("--world-step-auto-disable", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--world-step-stale-limit", type=int, default=10)
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
        "--enable-corridor-priority-reward",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Add priority-map direction reward shaping for doorway/corridor-like gaps. Enabled by default.",
    )
    parser.add_argument(
        "--corridor-priority-reward-weight",
        type=float,
        default=1.65,
        help="Weight for doorway/corridor priority reward shaping when --enable-corridor-priority-reward is enabled.",
    )
    parser.add_argument("--fixed-reset-yaw", action="store_true")
    parser.add_argument("--reset-z", type=float, default=0.05)

    parser.add_argument("--collision-threshold", type=float, default=0.10)
    parser.add_argument("--restart-on-collision", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--collision-clear-nav2-costmaps", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--collision-cancel-nav2-goal", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fallen-roll-threshold", type=float, default=0.45)
    parser.add_argument("--fallen-pitch-threshold", type=float, default=0.45)
    parser.add_argument("--terminate-on-out-of-bounds", action=argparse.BooleanOptionalAction, default=True)
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
    parser.add_argument("--velocity-command-linear-limit", type=float, default=0.0)
    parser.add_argument("--velocity-command-angular-limit", type=float, default=0.0)

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

    # Nav2 NavigateToPose action mode.
    parser.add_argument("--nav2-action-name", type=str, default="/navigate_to_pose")
    parser.add_argument("--nav2-goal-timeout-sec", type=float, default=3.0)
    parser.add_argument("--nav2-control-window-sec", type=float, default=1.35)
    parser.add_argument("--nav2-replan-on-movement", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--nav2-replan-distance-m", type=float, default=0.28)
    parser.add_argument("--nav2-continuous-goal-update", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--nav2-preempt-without-cancel", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--nav2-wait-timeout-sec", type=float, default=8.0)
    parser.add_argument("--nav2-goal-reached-tolerance", type=float, default=0.22)
    parser.add_argument("--nav2-cancel-on-timeout", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--nav2-cancel-on-reached", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--nav2-send-goal-wait-sec", type=float, default=2.0)
    parser.add_argument("--nav2-cancel-wait-sec", type=float, default=0.0)
    parser.add_argument("--nav2-use-goal-orientation", action=argparse.BooleanOptionalAction, default=True)
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
    parser.add_argument("--map-frame", type=str, default="odom")
    parser.add_argument(
        "--pose-frame",
        type=str,
        default="odom",
        help="Runtime pose/control frame. This build locks it to odom for RViz/RobotModel/map consistency.",
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
    parser.add_argument("--slam-map-max-age-sec", type=float, default=3.0)
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
        default="fixed",
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

    parser.add_argument("--reset-pose-max-attempts", type=int, default=60)
    parser.add_argument("--reset-pose-min-clearance-m", type=float, default=0.16)
    parser.add_argument("--reset-pose-validation-wait-sec", type=float, default=0.50)
    parser.add_argument("--post-reset-stabilize-sec", type=float, default=2.50)
    parser.add_argument("--post-reset-ready-gate", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--post-reset-ready-timeout-sec", type=float, default=7.0)
    parser.add_argument("--post-reset-ready-min-known-ratio", type=float, default=0.02)
    parser.add_argument("--post-reset-ready-min-known-cells", type=int, default=40)
    parser.add_argument("--post-reset-ready-min-lidar-beams", type=int, default=30)
    parser.add_argument("--post-reset-ready-require-priority", action=argparse.BooleanOptionalAction, default=True)

    # Episode restart conditions beyond collision/fall.
    parser.add_argument("--priority-stuck-restart", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--priority-stuck-restart-sec",
        type=float,
        default=0.0,
        help="Restart episode if active nonzero priority remains unresolved for this many seconds. 0 uses --priority-stuck-restart-steps.",
    )
    parser.add_argument("--priority-stuck-restart-steps", type=int, default=20)
    parser.add_argument("--priority-stuck-score-threshold", type=float, default=0.15)
    parser.add_argument("--priority-stuck-clear-gain-threshold", type=float, default=0.03)
    parser.add_argument("--priority-stuck-info-gain-threshold", type=float, default=0.0005)
    parser.add_argument("--priority-stuck-restart-penalty", type=float, default=45.0)

    parser.add_argument("--lidar-empty-restart", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--lidar-empty-timeout-sec",
        type=float,
        default=2.5,
        help="Restart with -100 if LiDAR has no valid finite hit below --lidar-empty-max-valid-range-m for this long.",
    )
    parser.add_argument("--lidar-empty-grace-sec", type=float, default=1.0)
    parser.add_argument("--lidar-empty-min-valid-range-m", type=float, default=0.12)
    parser.add_argument("--lidar-empty-max-valid-range-m", type=float, default=3.35)
    parser.add_argument("--lidar-empty-min-valid-beams", type=int, default=2)
    parser.add_argument("--lidar-empty-restart-penalty", type=float, default=100.0)


    parser.add_argument("--action-sync-reward-gate", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--action-sync-wait-timeout-sec", type=float, default=0.06)
    parser.add_argument("--action-sync-min-scan-age-sec", type=float, default=0.0)
    parser.add_argument("--map-bounds-restart", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--map-bounds-margin-cells", type=int, default=2)
    parser.add_argument("--map-bounds-min-local-known-ratio", type=float, default=0.04)
    parser.add_argument("--map-bounds-min-local-known-cells", type=int, default=12)
    parser.add_argument("--map-bounds-grace-steps", type=int, default=8)
    parser.add_argument("--map-bounds-restart-penalty", type=float, default=100.0)

    # CNN map observation. Must match the model used during training.
    parser.add_argument("--use-map-cnn", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--map-obs-size", type=int, default=48)
    parser.add_argument("--map-obs-size-m", type=float, default=6.0)
    parser.add_argument(
        "--num-lidar-bins",
        type=int,
        default=60,
        help=(
            "Number of LiDAR bins in the policy observation. Must match the checkpoint. "
            "v5 60-sector checkpoints need --num-lidar-bins 60. Legacy 360-bin checkpoints need 360."
        ),
    )
    parser.add_argument("--use-temporal-cnn", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--temporal-history-len", type=int, default=4)
    parser.add_argument("--front-fov-deg", type=float, default=80.0)
    parser.add_argument("--front-angle-sigma-deg", type=float, default=20.0)
    parser.add_argument("--confidence-max-range", type=float, default=2.0)
    parser.add_argument("--seen-confidence-floor", type=float, default=80.0)
    parser.add_argument("--confidence-decay-per-step", type=float, default=0.0)
    parser.add_argument("--suppress-gap-confidence", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--gap-occupied-threshold", type=float, default=65.0)
    parser.add_argument("--gap-check-radius-m", type=float, default=1.20)
    parser.add_argument("--gap-min-width-m", type=float, default=0.20)
    parser.add_argument("--gap-max-width-m", type=float, default=2.00)
    parser.add_argument("--map-expand-chunk-cells", type=int, default=64)
    parser.add_argument("--priority-recompute-interval", type=int, default=16)
    parser.add_argument("--priority-target-lock-steps", type=int, default=16)
    parser.add_argument("--priority-target-switch-margin", type=float, default=0.12)
    parser.add_argument("--priority-visit-suppression-radius-m", type=float, default=0.55)
    parser.add_argument("--priority-visit-suppression-gain", type=float, default=0.35)
    parser.add_argument("--priority-visit-suppression-max", type=float, default=0.85)
    parser.add_argument("--priority-observed-suppression-gain", type=float, default=0.20)
    parser.add_argument("--priority-clear-fov-deg", type=float, default=90.0)
    parser.add_argument("--priority-clear-max-range-m", type=float, default=1.20)
    parser.add_argument("--priority-clear-robot-radius-m", type=float, default=0.45)
    parser.add_argument("--priority-clear-min-value", type=float, default=5.0)
    parser.add_argument("--priority-clear-sigma-m", type=float, default=0.35)
    parser.add_argument("--priority-clear-angle-sigma-deg", type=float, default=30.0)
    parser.add_argument("--priority-clear-min-weight", type=float, default=0.18)
    parser.add_argument("--priority-clear-visit-sigma-m", type=float, default=0.25)
    parser.add_argument("--wall-support-radius-m", type=float, default=0.70)
    parser.add_argument("--wall-support-density-threshold", type=float, default=0.025)
    parser.add_argument("--open-space-front-distance-m", type=float, default=1.80)
    parser.add_argument("--open-space-side-width-m", type=float, default=1.20)
    parser.add_argument("--open-space-forward-penalty", type=float, default=0.45)
    parser.add_argument("--map-publish-every-n", type=int, default=0)
    parser.add_argument("--map-keepalive-period-sec", type=float, default=0.0)
    parser.add_argument("--debug-input-map", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--debug-input-map-topic-prefix", type=str, default="/rl_debug_input")
    parser.add_argument("--debug-input-map-frame-id", type=str, default="base_link")
    parser.add_argument("--debug-input-map-publish-every-n", type=int, default=50)

    # Action filtering / anti-jitter. Must match training-time semantics when possible.
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

    parser.add_argument("--realtime-spin-steps", type=int, default=2)
    parser.add_argument("--realtime-spin-timeout-sec", type=float, default=0.0)
    parser.add_argument("--realtime-sleep-sec", type=float, default=0.001)

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
    parser.add_argument("--velocity-forward-assist-mps", type=float, default=0.0)
    parser.add_argument("--velocity-forward-assist-angular-threshold", type=float, default=0.20)
    parser.add_argument("--velocity-forward-assist-min-clearance-m", type=float, default=0.45)
    parser.add_argument("--velocity-spin-breaker", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--velocity-spin-breaker-steps", type=int, default=14)
    parser.add_argument("--velocity-spin-breaker-angular-ratio", type=float, default=0.85)
    parser.add_argument("--velocity-spin-breaker-forward-mps", type=float, default=0.035)
    parser.add_argument("--velocity-spin-breaker-angular-scale", type=float, default=0.35)
    parser.add_argument("--velocity-spin-breaker-min-clearance-m", type=float, default=0.48)

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
    parser.add_argument("--shake-yaw-rate-threshold", type=float, default=0.24)
    parser.add_argument("--shake-cmd-flip-threshold", type=float, default=0.16)
    parser.add_argument("--shake-wobble-window-steps", type=int, default=8)
    parser.add_argument("--shake-wobble-min-flips", type=int, default=2)
    parser.add_argument("--shake-wobble-max-net-motion-m", type=float, default=0.045)
    parser.add_argument("--shake-spin-stall-restart-steps", type=int, default=18)
    parser.add_argument("--shake-restart-penalty", type=float, default=100.0)

    parser.add_argument("--confidence-reward-weight", type=float, default=2.0)
    parser.add_argument("--slam-map-update-reward", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--slam-map-update-reward-weight", type=float, default=0.65)
    parser.add_argument("--slam-map-update-reward-norm-cells", type=float, default=50.0)
    parser.add_argument("--slam-map-update-reward-cap", type=float, default=3.0)
    parser.add_argument("--slam-map-update-reward-grace-steps", type=int, default=10)
    parser.add_argument("--reward-positive-log-compress", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--reward-positive-log-alpha", type=float, default=0.50)
    parser.add_argument("--reward-positive-log-max", type=float, default=8.0)

    parsed_args, _ = parser.parse_known_args()
    parsed_args = _force_nav2_only_policy(parsed_args)
    parsed_args = _force_mandatory_slam_reset_policy(parsed_args)
    parsed_args = _apply_real_robot_policy(parsed_args)
    return parsed_args



def _maybe_rescale_model_action_to_env_space(action, model, env, enabled: bool = True):
    """Linearly remap actions from the checkpoint Box to the current env Box."""
    arr = np.asarray(action, dtype=np.float32)
    if not enabled:
        return arr
    try:
        model_space = getattr(model, "action_space", None)
        env_space = getattr(env, "action_space", None)
        ml = np.asarray(model_space.low, dtype=np.float32)
        mh = np.asarray(model_space.high, dtype=np.float32)
        el = np.asarray(env_space.low, dtype=np.float32)
        eh = np.asarray(env_space.high, dtype=np.float32)
    except Exception:
        return arr
    if ml.shape != arr.shape or mh.shape != arr.shape or el.shape != arr.shape or eh.shape != arr.shape:
        return arr
    if np.allclose(ml, el, atol=1e-6) and np.allclose(mh, eh, atol=1e-6):
        return np.clip(arr, el, eh).astype(np.float32)
    denom = np.maximum(mh - ml, 1e-6)
    unit = (arr - ml) / denom
    remapped = el + np.clip(unit, 0.0, 1.0) * (eh - el)
    return np.clip(remapped, el, eh).astype(np.float32)


def _describe_action_space_box(space) -> str:
    try:
        low = np.asarray(space.low, dtype=np.float32).tolist()
        high = np.asarray(space.high, dtype=np.float32).tolist()
        return f"low={low}, high={high}"
    except Exception:
        return "unavailable"

def main(args=None):
    cli_args = _apply_real_robot_policy(
        _force_odom_coordinate_policy(
            _force_mandatory_slam_reset_policy(_apply_rviz_origin_policy(parse_args()))
        )
    )

    # Disable FastDDS SHM transport before ROS init (same robust fix as training)
    # so long eval sessions never hit open_and_lock_file / init_port failures.
    if str(os.environ.get("TB3_RL_DISABLE_SHM_TRANSPORT", "1")).strip().lower() not in {"0", "false", "no", "off"}:
        try:
            if ensure_non_shm_fastdds_profile is not None:
                ensure_non_shm_fastdds_profile(logger=None)
        except Exception:
            pass

    try:
        if bool(getattr(cli_args, "auto_start_gazebo", False)) and clean_fastdds_shm is not None:
            clean_fastdds_shm(logger=None)
    except Exception:
        pass

    gazebo_proc = _start_gazebo_if_requested(cli_args)

    model_path = Path(cli_args.model)

    if not model_path.exists():
        print(f"Model file not found: {model_path}")
        return

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
        ros.destroy_node()
        rclpy.shutdown()
        _terminate_process(gazebo_proc, "Gazebo")
        _terminate_process(gazebo_proc, "Gazebo")
        return

    if use_slam_map and cli_args.auto_start_slam:
        ros.ensure_slam_toolbox(timeout_sec=cli_args.slam_reset_timeout)

    if use_slam_map and cli_args.wait_slam_map:
        ros.wait_for_slam_map_ready(timeout_sec=max(10.0, cli_args.slam_reset_timeout))

    env = GazeboNavEnv(
        ros_interface=ros,
        entity_name=cli_args.entity_name,
        set_pose_service=cli_args.set_pose_service,
        enable_pose_reset=not cli_args.disable_pose_reset,
        random_reset_yaw=not cli_args.fixed_reset_yaw,
        reset_z=cli_args.reset_z,
        control_dt=cli_args.control_dt,
        physics_step_size=cli_args.physics_step_size,
        realtime_spin_steps=cli_args.realtime_spin_steps,
        realtime_spin_timeout_sec=cli_args.realtime_spin_timeout_sec,
        realtime_sleep_sec=cli_args.realtime_sleep_sec,
        max_episode_steps=cli_args.max_episode_steps,
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
        disable_path_reward=True,
        disable_wall_proximity_penalty=cli_args.disable_wall_proximity_penalty,
        enable_corridor_priority_reward=bool(getattr(cli_args, "enable_corridor_priority_reward", True)),
        disable_priority_map=bool(getattr(cli_args, "disable_priority_map", False)),
        corridor_priority_reward_weight=cli_args.corridor_priority_reward_weight,
        confidence_reward_weight=cli_args.confidence_reward_weight,
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
        nav2_action_name=cli_args.nav2_action_name,
        nav2_goal_timeout_sec=cli_args.nav2_goal_timeout_sec,
        nav2_control_window_sec=cli_args.nav2_control_window_sec,
        nav2_replan_on_movement=cli_args.nav2_replan_on_movement,
        nav2_replan_distance_m=cli_args.nav2_replan_distance_m,
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
        shake_yaw_rate_threshold=cli_args.shake_yaw_rate_threshold,
        shake_cmd_flip_threshold=cli_args.shake_cmd_flip_threshold,
        shake_wobble_window_steps=cli_args.shake_wobble_window_steps,
        shake_wobble_min_flips=cli_args.shake_wobble_min_flips,
        shake_wobble_max_net_motion_m=cli_args.shake_wobble_max_net_motion_m,
        shake_spin_stall_restart_steps=cli_args.shake_spin_stall_restart_steps,
        shake_restart_penalty=cli_args.shake_restart_penalty,
        reward_gamma=0.99,
    )

    # Importing MapVectorFeatureExtractor above ensures custom extractor class is available
    # when loading a model trained with MultiInputPolicy.
    _ = MapVectorFeatureExtractor
    model = SAC.load(str(model_path))

    try:
        ros.get_logger().warn(
            "EVAL_ACTION_SPACE | "
            f"model={_describe_action_space_box(getattr(model, 'action_space', None))} | "
            f"env={_describe_action_space_box(getattr(env, 'action_space', None))} | "
            f"rescale={bool(getattr(cli_args, 'rescale_model_action_to_env_space', True))}"
        )
    except Exception:
        pass

    ros.get_logger().info(
        "EVAL_POLICY_MODE | "
        f"deterministic={bool(getattr(cli_args, 'eval_deterministic', True))} "
        f"action_mode={getattr(cli_args, 'action_mode', 'unknown')} "
        f"priority_disabled={bool(getattr(cli_args, 'disable_priority_map', False)) or bool(getattr(cli_args, 'real_robot_disable_priority', False) and getattr(cli_args, 'real_robot', False))}"
    )

    if bool(getattr(cli_args, "real_robot", False)):
        ros.get_logger().warn(
            "REAL_ROBOT_POLICY_READY | Gazebo reset/world-step disabled. "
            "Keep an emergency stop terminal ready. /cmd_vel type must be TwistStamped."
        )
        delay = max(float(getattr(cli_args, "real_robot_start_delay_sec", 0.0)), 0.0)
        if delay > 0.0:
            ros.stop_robot()
            time.sleep(delay)

    try:
        for episode in range(cli_args.episodes):
            if bool(getattr(cli_args, "real_robot_stop_between_episodes", True)):
                ros.stop_robot()
            if bool(getattr(cli_args, "manual_reset_prompt", False)):
                print(
                    f"\n[REAL_ROBOT] Episode {episode + 1}/{cli_args.episodes}: "
                    "put the robot at the start pose, clear the area, then press ENTER. "
                    "SLAM will be reset after ENTER."
                )
                try:
                    input()
                except EOFError:
                    pass
                ros.stop_robot()
            obs, info = env.reset()

            total_reward = 0.0
            done = False
            step_count = 0

            while not done:
                model_action, _ = model.predict(
                    obs,
                    deterministic=bool(getattr(cli_args, "eval_deterministic", True)),
                )
                action = _maybe_rescale_model_action_to_env_space(
                    model_action,
                    model=model,
                    env=env,
                    enabled=bool(getattr(cli_args, "rescale_model_action_to_env_space", True)),
                )

                obs, reward, terminated, truncated, info = env.step(action)

                total_reward += reward
                step_count += 1

                done = terminated or truncated

            ros.get_logger().info(
                "Episode "
                f"{episode + 1}/{cli_args.episodes} | "
                f"reward={total_reward:.3f} | "
                f"steps={step_count} | "
                f"coverage={info.get('coverage_ratio', -1.0):.4f} | "
                f"new_cells={info.get('new_known_cells', -1)} | "
                f"mean_conf={info.get('mean_confidence', -1.0):.2f} | "
                f"conf_gain={info.get('confidence_gain', -1.0):.3f} | "
                f"stale={info.get('stale_known_cells', -1)} | "
                f"stale_refresh={info.get('stale_refresh_cells', -1)} | "
                f"low_conf={info.get('low_confidence_cells', -1)} | "
                f"priority={info.get('priority_score', -1.0):.2f} | "
                f"clear={info.get('priority_cleared_cells', -1)}:{info.get('priority_clear_gain', -1.0):.2f} | "
                f"frontiers={info.get('frontier_count', -1)} | "
                f"target={info.get('target_type', 'none')}:{info.get('target_priority', -1.0):.2f} | "
                f"pdir={info.get('priority_direction_error', 0.0):+.2f}:{info.get('priority_direction_alignment', 0.0):.2f}:{info.get('priority_direction_signed', 0.0):+.2f} | "
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
                f"err={info.get('waypoint_final_error', 0.0):.2f}:steps={info.get('controller_steps', 0)} | "
                f"slam={info.get('slam_map_available', False)} | "
                f"collision={info.get('collision', False)} | "
                f"fallen={info.get('fallen', False)} | "
                f"coverage_done={info.get('coverage_done', False)} | "
                f"sim_time={info.get('sim_time', -1.0):.3f}"
            )

    finally:
        env.close()
        ros.stop_robot()
        ros.destroy_node()
        rclpy.shutdown()
        _terminate_process(gazebo_proc, "Gazebo")


if __name__ == "__main__":
    main()
