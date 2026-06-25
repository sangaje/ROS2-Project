import math
import os
import random
import re
import subprocess
import time
from copy import deepcopy
from dataclasses import replace
from collections import deque
from pathlib import Path
from typing import Optional

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import rclpy
from builtin_interfaces.msg import Duration, Time as RosTime
from geometry_msgs.msg import Point, PoseStamped, PoseWithCovarianceStamped
from nav_msgs.msg import OccupancyGrid, Path as NavPath
from sensor_msgs.msg import LaserScan
from visualization_msgs.msg import Marker, MarkerArray
from std_srvs.srv import Empty

try:
    from action_msgs.msg import GoalStatus
    from rclpy.action import ActionClient
except Exception:  # pragma: no cover - Nav2 optional dependency
    GoalStatus = None
    ActionClient = None

try:
    from nav2_msgs.action import NavigateToPose, FollowPath, BackUp
except Exception:  # pragma: no cover - Nav2 optional dependency
    NavigateToPose = None
    FollowPath = None
    BackUp = None

try:
    from nav2_msgs.action import Spin
except Exception:  # pragma: no cover - behavior_server optional action
    Spin = None

try:
    from nav2_msgs.action import DriveOnHeading
except Exception:  # pragma: no cover - behavior_server optional action
    DriveOnHeading = None

from turtlebot3_rl_training.exploration_map import ExplorationGridMap, MapUpdateStats
from turtlebot3_rl_training.observation import build_exploration_observation
from turtlebot3_rl_training.reset_manager import ResetManager
from turtlebot3_rl_training.reward import (
    compute_exploration_reward,
    compute_velocity_safety_slowdown_penalty,
    compute_waypoint_macro_reward_adjustment,
)
from turtlebot3_rl_training.sim_controller import GazeboSimController


def _quiet_reset_logs() -> bool:
    """When TB3_RL_QUIET_RESET_LOGS is set, suppress repetitive per-episode
    reset/SLAM-wait WARN spam (STRICT_SLAM_MAP_WAITING, RESET_CANDIDATE_CHECK,
    RESET_POSE_TRUTH, Safety boundary updates).  Diagnostic, not functional."""
    return str(os.environ.get("TB3_RL_QUIET_RESET_LOGS", "0")).strip().lower() in {"1", "true", "yes", "on"}


class RecoverableResetError(RuntimeError):
    """Reset/readiness failure that should retry the whole reset instead of killing SB3.

    These errors are caused by transient reset races such as delayed Cartographer
    /map, missing map->base TF immediately after teleport, or post-reset readiness
    timeout.  They must not return an empty observation, because that would insert
    a corrupted initial state into the replay buffer.  The environment reset wrapper
    catches this exception, holds the robot still, clears transient TF/Nav2 state,
    and retries the reset from the beginning.
    """


class GazeboNavEnv(gym.Env):
    # Conservative spawn candidates for turtlebot3_house-like indoor worlds.
    # These are Gazebo/world-frame coordinates. With --rviz-zero-robot-on-reset,
    # the selected Gazebo spawn is redefined as RViz/map-frame origin after reset.
    # TurtleBot3 house용 spawn 후보.
    # 이전 패치에서 벽/가구 근접 spawn을 피하려고 보수적으로 줄였는데,
    # 그러면 episode가 외부/문 근처에 치우쳐 실내 탐색 학습이 약해진다.
    # 따라서 house_random 기본 후보에 실내 후보를 다시 넣고, 실제 사용 가능 여부는
    # reset_pose_min_clearance_m 기반 LiDAR 검증으로 걸러낸다.
    # 주의: (0.20, -0.20), (0.20, -2.20)은 벽/우편함 근접 이슈가 반복되어 제외한다.
    # TurtleBot3 house 내부 시작 후보만 모은 리스트.
    # house_random은 문/외곽 후보도 섞이므로, 실내에서만 시작시키고 싶을 때는
    # --reset-pose-mode house_inside_random을 사용한다.
    HOUSE_INSIDE_RESET_CANDIDATES: tuple[tuple[float, float], ...] = (
        # User-selected safe indoor spawn points only.
        # Keep this list small and explicit; do not mix old corridor/door candidates.
        # Coordinates are Gazebo/world-frame x,y. The ResetManager still performs
        # LiDAR clearance validation before accepting a candidate.
        (-2.80,  0.96),
        ( 5.00,  0.86),
        ( 1.20, -1.60),
    )

    DEFAULT_HOUSE_RESET_CANDIDATES: tuple[tuple[float, float], ...] = (
        *HOUSE_INSIDE_RESET_CANDIDATES,
    )

    """
    TurtleBot3 Burger용 Gymnasium Env.

    구조:
      - SLAM /map은 실제 geometry/localization 품질을 담당한다.
      - RL memory map은 confidence, stale, revisit 정보를 담당한다.
      - SAC policy는 LiDAR + confidence/task-map 통계량을 보고 탐색 정책을 학습한다.

    action:
      기본값 action_mode="waypoint" + waypoint_action_type="path"에서는
      [lookahead_norm, lateral_norm]이다.
      - lookahead_norm: [0, 1]  -> 현재 planned path 방향으로 볼 거리
      - lateral_norm  : [-1, 1] -> path tangent 기준 좌/우 offset

      waypoint_action_type="polar"를 명시하면 이전처럼
      [distance_norm, heading_norm]을 로봇 기준 local waypoint로 해석한다.
      action_mode="velocity"를 명시하면 기존처럼 [linear_x, angular_z]를 직접 사용한다.
      action_mode="nav2"를 명시하면 SAC가 고른 waypoint를 Nav2 NavigateToPose goal로 보낸다.

    exploration observation:
      LiDAR 360
      + coverage_ratio
      + coverage_delta_norm
      + frontier_distance_norm
      + frontier_angle_norm
      + target_priority
      + mean_confidence_norm
      + stale_ratio
      + low_confidence_ratio
      + prev_linear_norm
      + prev_angular_norm
      = 370 dimensions

    핵심:
      use_world_step=True일 때는 Gazebo를 paused 상태로 두고,
      매 RL step마다 /world/default/control multi_step으로 physics를 전진시킨다.
      이후 /clock 또는 /odom stamp가 실제로 전진했는지 barrier를 건다.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        ros_interface,
        entity_name: str = "turtlebot3_burger",
        set_pose_service: str = "",
        enable_pose_reset: bool = True,
        random_reset_yaw: bool = True,
        reset_z: float = 0.05,
        control_dt: float = 0.12,
        max_episode_steps: int = 1000,
        goal_threshold: float = 0.25,
        collision_threshold: float = 0.16,
        restart_on_collision: bool = True,
        collision_clear_nav2_costmaps: bool = True,
        collision_cancel_nav2_goal: bool = True,
        fallen_roll_threshold: float = 0.45,
        fallen_pitch_threshold: float = 0.45,
        terminate_on_out_of_bounds: bool = False,
        safety_boundary_radius_m: float = 6.0,
        safety_boundary_min_x: float = -6.0,
        safety_boundary_max_x: float = 6.0,
        safety_boundary_min_y: float = -6.0,
        safety_boundary_max_y: float = 6.0,
        safety_boundary_max_abs_z: float = 0.45,
        safety_boundary_frame: str = "odom",
        world_control_service: str = "/world/default/control",
        physics_step_size: float = 0.01,
        use_world_step: bool = True,
        world_step_target_fraction: float = 0.05,
        world_step_wait_timeout_sec: float = 0.03,
        world_step_sensor_timeout_sec: float = 0.03,
        world_step_stale_warn_every_n: int = 500,
        world_step_auto_disable_on_stale: bool = True,
        world_step_stale_limit: int = 10,
        realtime_spin_steps: int = 2,
        realtime_spin_timeout_sec: float = 0.0,
        realtime_sleep_sec: float = 0.001,
        realtime_enforce_control_dt: bool = False,
        realtime_control_dt_wall_margin_sec: float = 0.0,
        disable_path_reward: bool = True,
        disable_wall_proximity_penalty: bool = False,
        enable_corridor_priority_reward: bool = True,
        disable_priority_map: bool = False,
        corridor_priority_reward_weight: float = 1.65,
        confidence_reward_weight: float = 1.0,
        slam_map_update_reward: bool = False,
        slam_map_update_reward_weight: float = 0.20,
        slam_map_update_reward_norm_cells: float = 80.0,
        slam_map_update_reward_cap: float = 1.0,
        slam_map_update_reward_grace_steps: int = 10,
        reward_positive_log_compress: bool = False,
        reward_positive_log_alpha: float = 0.50,
        reward_positive_log_max: float = 8.0,
        max_linear_speed: float = 0.32,
        max_angular_speed: float = 0.90,
        velocity_command_linear_limit: float = 0.0,
        velocity_command_angular_limit: float = 0.0,
        action_mode: str = "waypoint",
        waypoint_action_type: str = "polar",
        waypoint_lateral_max_offset: float = 0.20,
        waypoint_min_distance: float = 0.25,
        waypoint_max_distance: float = 0.65,
        waypoint_max_angle_deg: float = 60.0,
        waypoint_reached_tolerance: float = 0.40,
        waypoint_control_steps: int = 2,
        waypoint_execute_until_reached: bool = False,
        waypoint_max_control_steps: int = 2,
        waypoint_timeout_sec: float = 0.20,
        waypoint_timeout_stop: bool = False,
        waypoint_linear_kp: float = 0.90,
        waypoint_angular_kp: float = 2.80,
        waypoint_max_yaw_error_for_linear_deg: float = 75.0,
        waypoint_slowdown_distance: float = 0.45,
        waypoint_min_linear_speed: float = 0.08,
        waypoint_disable_arrival_slowdown: bool = True,
        waypoint_front_stop_distance: float = 0.32,
        waypoint_replan_distance_m: float = 0.12,
        waypoint_replan_heading_deg: float = 18.0,
        waypoint_direct_point_mode: bool = True,
        waypoint_direct_heading_tolerance_deg: float = 10.0,
        waypoint_direct_drive_heading_limit_deg: float = 28.0,
        waypoint_direct_max_correction_angular: float = 0.35,
        waypoint_direct_min_drive_distance: float = 0.05,
        waypoint_direct_target_sector_deg: float = 14.0,
        waypoint_direct_turn_drive: bool = True,
        waypoint_direct_turn_drive_max_yaw_deg: float = 65.0,
        waypoint_direct_turn_drive_speed_scale: float = 0.45,
        waypoint_direct_turn_drive_min_speed: float = 0.06,
        reset_pose_max_attempts: int = 8,
        reset_pose_min_clearance_m: float = 0.13,
        reset_pose_validation_wait_sec: float = 0.20,
        post_reset_stabilize_sec: float = 2.0,
        post_reset_stabilize_spin_steps: int = 12,
        post_reset_ready_gate: bool = True,
        post_reset_ready_timeout_sec: float = 7.0,
        post_reset_ready_min_known_ratio: float = 0.02,
        post_reset_ready_min_known_cells: int = 40,
        post_reset_ready_min_lidar_beams: int = 30,
        post_reset_ready_require_priority: bool = True,
        action_sync_reward_gate: bool = True,
        action_sync_wait_timeout_sec: float = 0.06,
        action_sync_min_scan_age_sec: float = 0.0,
        map_bounds_restart: bool = True,
        map_bounds_margin_cells: int = 2,
        map_bounds_min_local_known_ratio: float = 0.04,
        map_bounds_min_local_known_cells: int = 12,
        map_bounds_grace_steps: int = 8,
        map_bounds_restart_penalty: float = 100.0,
        nav2_action_name: str = "/navigate_to_pose",
        nav2_goal_timeout_sec: float = 2.0,
        nav2_control_window_sec: float = 1.35,
        nav2_replan_on_movement: bool = True,
        nav2_replan_distance_m: float = 0.28,
        nav2_early_replan_remaining_m: float = 0.55,
        nav2_near_goal_replan_only: bool = True,
        nav2_continuous_goal_update: bool = False,
        nav2_preempt_without_cancel: bool = True,
        nav2_wait_timeout_sec: float = 8.0,
        nav2_goal_reached_tolerance: float = 0.35,
        nav2_cancel_on_timeout: bool = True,
        nav2_cancel_on_reached: bool = True,
        nav2_send_goal_wait_sec: float = 1.0,
        nav2_cancel_wait_sec: float = 0.0,
        nav2_use_goal_orientation: bool = False,
        nav2_auto_start: bool = True,
        nav2_launch_package: str = "nav2_bringup",
        nav2_launch_file: str = "navigation_launch.py",
        nav2_params_file: str = "",
        nav2_use_sim_time: bool = True,
        nav2_startup_timeout_sec: float = 25.0,
        slam_adaptive_speed: bool = True,
        slam_local_speed_radius: float = 1.00,
        slam_front_speed_distance: float = 1.20,
        slam_front_speed_half_angle_deg: float = 45.0,
        slam_speed_min_scale: float = 0.30,
        slam_speed_max_scale: float = 1.00,
        slam_speed_local_weight: float = 0.55,
        slam_speed_front_weight: float = 0.25,
        slam_speed_fresh_weight: float = 0.20,
        slam_speed_map_age_soft_limit_sec: float = 3.0,
        slam_speed_known_low_ratio: float = 0.25,
        slam_speed_known_high_ratio: float = 0.85,
        slam_speed_fresh_low_score: float = 0.15,
        slam_speed_smoothing_alpha: float = 0.10,
        waypoint_marker_topic: str = "",
        waypoint_path_topic: str = "",
        waypoint_visual_history_len: int = 80,
        waypoint_visual_publish_every_n: int = 1000000,
        waypoint_show_history: bool = False,
        use_slam_map: bool = True,
        map_frame: str = "odom",
        pose_frame: str = "odom",
        rl_map_topic: str = "",
        rl_confidence_topic: str = "",
        rl_priority_topic: str = "",
        rl_path_topic: str = "",
        rl_filtered_slam_topic: str = "",
        slam_map_accept_delay_sec: float = 1.0,
        slam_map_max_age_sec: float = 3.0,
        strict_slam_map_required: bool = False,
        strict_slam_map_wait_timeout_sec: float = 30.0,
        strict_slam_map_retry_interval_sec: float = 0.50,
        strict_slam_map_min_known_cells: int = 20,
        strict_slam_map_min_known_ratio: float = 0.001,
        reset_x: float = 0.0,
        reset_y: float = 0.0,
        reset_pose_mode: str = "house_inside_random",
        reset_offset: float = 0.3,
        reset_pose_list: str = "",
        rviz_zero_robot_on_reset: bool = False,
        rviz_origin_wait_sec: float = 2.0,
        rviz_origin_tolerance_m: float = 0.25,
        reset_slam_on_reset: bool = False,
        restart_slam_on_reset: bool = False,
        reset_slam_every_n_episodes: int = 0,
        reset_tf_buffer_on_reset: bool = True,
        slam_reset_timeout_sec: float = 8.0,
        slam_reset_warmup_steps: int = 15,
        use_map_cnn: bool = True,
        map_obs_size: int = 48,
        map_obs_size_m: float = 6.0,
        use_temporal_cnn: bool = False,
        num_lidar_bins: int = 60,
        temporal_history_len: int = 4,
        front_fov_deg: float = 80.0,
        confidence_decay_per_step: float = 0.0,
        confidence_max_range: float = 2.0,
        front_angle_sigma_deg: float = 20.0,
        seen_confidence_floor: float = 80.0,
        suppress_gap_confidence: bool = False,
        gap_occupied_threshold: float = 65.0,
        gap_check_radius_m: float = 1.20,
        gap_min_width_m: float = 0.20,
        gap_max_width_m: float = 2.00,
        map_expand_chunk_cells: int = 64,
        map_publish_every_n: int = 0,
        max_planned_candidates: int = 8,
        max_alternative_paths: int = 5,
        path_visual_publish_every_n: int = 0,
        priority_recompute_interval: int = 16,
        priority_target_lock_steps: int = 16,
        priority_target_switch_margin: float = 0.12,
        priority_visit_suppression_radius_m: float = 0.55,
        priority_visit_suppression_gain: float = 0.35,
        priority_visit_suppression_max: float = 0.85,
        priority_observed_suppression_gain: float = 0.20,
        priority_clear_fov_deg: float = 90.0,
        priority_clear_max_range_m: float = 1.20,
        priority_clear_robot_radius_m: float = 0.45,
        priority_clear_min_value: float = 5.0,
        priority_clear_sigma_m: float = 0.35,
        priority_clear_angle_sigma_deg: float = 30.0,
        priority_clear_min_weight: float = 0.18,
        priority_clear_visit_sigma_m: float = 0.25,
        wall_support_radius_m: float = 0.70,
        wall_support_density_threshold: float = 0.025,
        open_space_front_distance_m: float = 1.80,
        open_space_side_width_m: float = 1.20,
        open_space_forward_penalty: float = 0.35,
        map_keepalive_period_sec: float = 0.0,
        map_live_update_period_sec: float = 0.10,
        debug_input_map: bool = False,
        debug_input_map_topic_prefix: str = "/rl_debug_input",
        debug_input_map_frame_id: str = "base_link",
        debug_input_map_publish_every_n: int = 50,
        action_smoothing_alpha: float = 0.30,
        max_linear_delta: float = 0.08,
        max_angular_delta: float = 0.20,
        linear_deadband: float = 0.015,
        angular_deadband: float = 0.04,
        enable_motion_mode_hysteresis: bool = True,
        explored_stall_start_steps: int = 8,
        explored_stall_growth: float = 0.008,
        explored_stall_power: float = 1.45,
        explored_stall_max_penalty: float = 1.20,
        confidence_stall_start_steps: int = 6,
        confidence_stall_growth: float = 0.010,
        confidence_stall_power: float = 1.35,
        confidence_stall_max_penalty: float = 1.60,
        confidence_stall_gain_threshold: float = 0.02,
        confidence_stall_low_ratio_threshold: float = 0.20,
        priority_stuck_restart: bool = True,
        priority_stuck_restart_sec: float = 0.0,
        priority_stuck_restart_steps: int = 100,
        priority_stuck_score_threshold: float = 0.15,
        priority_stuck_clear_gain_threshold: float = 0.03,
        priority_stuck_info_gain_threshold: float = 0.0005,
        priority_stuck_restart_penalty: float = 45.0,
        lidar_empty_restart: bool = True,
        lidar_empty_timeout_sec: float = 2.5,
        lidar_empty_grace_sec: float = 1.0,
        lidar_empty_min_valid_range_m: float = 0.12,
        lidar_empty_max_valid_range_m: float = 3.35,
        lidar_empty_min_valid_beams: int = 2,
        lidar_empty_restart_penalty: float = 100.0,
        coverage_stall_terminal: bool = False,
        coverage_stall_start_steps: int = 1000,
        coverage_stall_window_steps: int = 500,
        coverage_stall_min_slam_new_cells: int = 5,
        coverage_stall_min_confidence_updated_cells: int = 30,
        coverage_stall_terminal_penalty: float = -1.0,
        velocity_safety_backup: bool = True,
        velocity_safety_trigger_distance_m: float = 0.28,
        velocity_safety_stop_distance_m: float = 0.36,
        velocity_safety_slow_distance_m: float = 0.55,
        velocity_safety_backup_speed_mps: float = 0.08,
        velocity_safety_turn_speed: float = 0.35,
        velocity_safety_backup_steps: int = 4,
        velocity_safety_cooldown_steps: int = 8,
        velocity_safety_penalty: float = 10.0,
        velocity_safety_block_penalty: float = 0.80,
        velocity_safety_slowdown: bool = True,
        velocity_safety_slow_min_scale: float = 0.20,
        velocity_safety_slow_penalty: float = 1.80,
        velocity_safety_slow_speed_power: float = 1.35,
        velocity_safety_slow_danger_power: float = 1.10,
        velocity_forward_assist_mps: float = 0.0,
        velocity_forward_assist_angular_threshold: float = 0.20,
        velocity_forward_assist_min_clearance_m: float = 0.45,
        velocity_spin_breaker: bool = False,
        velocity_spin_breaker_steps: int = 14,
        velocity_spin_breaker_angular_ratio: float = 0.85,
        velocity_spin_breaker_forward_mps: float = 0.035,
        velocity_spin_breaker_angular_scale: float = 0.35,
        velocity_spin_breaker_min_clearance_m: float = 0.48,
        shake_restart: bool = True,
        shake_restart_steps: int = 4,
        shake_tilt_threshold: float = 0.12,
        shake_angular_xy_threshold: float = 0.70,
        shake_linear_z_threshold: float = 0.08,
        shake_z_deviation_threshold: float = 0.05,
        shake_ground_min_z: float = -0.02,
        shake_ground_max_z: float = 0.13,
        shake_leaky_decay: bool = True,
        shake_yaw_wobble: bool = False,
        shake_yaw_rate_threshold: float = 0.24,
        shake_cmd_flip_threshold: float = 0.16,
        shake_wobble_window_steps: int = 8,
        shake_wobble_min_flips: int = 2,
        shake_wobble_max_net_motion_m: float = 0.045,
        shake_spin_stall_restart_steps: int = 18,
        shake_restart_penalty: float = 100.0,
        reset_hard_stabilize_reapply: bool = True,
        reset_hard_stabilize_reapply_interval_sec: float = 0.25,
        nav2_stuck_backup: bool = True,
        nav2_stuck_backup_action_name: str = "/backup",
        nav2_stuck_backup_sec: float = 3.0,
        nav2_stuck_backup_steps: int = 4,
        nav2_stuck_backup_min_movement_m: float = 0.025,
        nav2_stuck_backup_stationary_sec: float = 1.5,
        nav2_stuck_backup_stationary_xy_m: float = 0.025,
        nav2_stuck_backup_stationary_yaw_deg: float = 7.0,
        nav2_stuck_backup_distance_m: float = 0.26,
        nav2_stuck_backup_speed_mps: float = 0.07,
        nav2_stuck_backup_timeout_sec: float = 4.0,
        nav2_stuck_backup_cooldown_sec: float = 2.5,
        nav2_stuck_backup_penalty: float = 3.0,
        reward_gamma: float = 0.99,
    ):
        super().__init__()

        self.ros = ros_interface

        self.entity_name = entity_name
        self.enable_pose_reset = bool(enable_pose_reset)
        self.random_reset_yaw = bool(random_reset_yaw)

        self.control_dt = float(control_dt)
        self.physics_step_size = float(physics_step_size)
        self.use_world_step = bool(use_world_step)
        # Gazebo ControlWorld multi_step is not perfectly synchronous on every
        # ros_gz/Gazebo setup.  Treat a small clock/odom advance as valid, keep
        # the barrier short, and automatically fall back to wall-clock spinning
        # if multi_step repeatedly returns stale observations.
        self.world_step_target_fraction = float(np.clip(world_step_target_fraction, 0.0, 1.0))
        self.world_step_wait_timeout_sec = max(float(world_step_wait_timeout_sec), 0.0)
        self.world_step_sensor_timeout_sec = max(float(world_step_sensor_timeout_sec), 0.0)
        self.world_step_stale_warn_every_n = max(int(world_step_stale_warn_every_n), 1)
        self.world_step_auto_disable_on_stale = bool(world_step_auto_disable_on_stale)
        self.world_step_stale_limit = max(int(world_step_stale_limit), 1)
        self.realtime_spin_steps = max(int(realtime_spin_steps), 0)
        self.realtime_spin_timeout_sec = max(float(realtime_spin_timeout_sec), 0.0)
        self.realtime_sleep_sec = max(float(realtime_sleep_sec), 0.0)
        self.realtime_enforce_control_dt = bool(realtime_enforce_control_dt)
        self.realtime_control_dt_wall_margin_sec = max(float(realtime_control_dt_wall_margin_sec), 0.0)
        self._last_realtime_step_wall_elapsed_sec = 0.0
        self._world_step_stale_count = 0
        # /rl_path based planning/reward is removed.  Keep the attribute for old
        # callers, but force it disabled at runtime.
        self.disable_path_reward = True
        # Nav2-only training still needs a dense safety signal.  Nav2 owns /cmd_vel,
        # but the critic must see wall/obstacle risk before a hard collision terminal.
        self.disable_wall_proximity_penalty = False
        # v93: hard no-priority mode.  When active, priority is removed from
        # publishers, reward, reset gates, stuck restarts, and policy input.
        force_no_priority = (
            str(os.environ.get("TB3_RL_FORCE_NO_PRIORITY", "0")).strip().lower()
            not in {"0", "false", "no", "off", "disable", "disabled"}
        ) or (
            str(os.environ.get("TB3_RL_NO_PRIORITY_MODEL_INPUT", "0")).strip().lower()
            in {"1", "true", "yes", "on", "enable", "enabled"}
        )
        self.disable_priority_map = bool(disable_priority_map) or bool(force_no_priority)
        if self.disable_priority_map:
            os.environ["TB3_RL_FORCE_NO_PRIORITY"] = "1"
            os.environ["TB3_RL_NO_PRIORITY_MODEL_INPUT"] = "1"
        self.enable_corridor_priority_reward = bool(enable_corridor_priority_reward) and (not self.disable_priority_map)
        self.corridor_priority_reward_weight = 0.0 if self.disable_priority_map else max(float(corridor_priority_reward_weight), 0.0)
        self.confidence_reward_weight = max(float(confidence_reward_weight), 0.0)
        self.slam_map_update_reward = bool(slam_map_update_reward)
        self.slam_map_update_reward_weight = max(float(slam_map_update_reward_weight), 0.0)
        self.slam_map_update_reward_norm_cells = max(float(slam_map_update_reward_norm_cells), 1.0)
        self.slam_map_update_reward_cap = max(float(slam_map_update_reward_cap), 0.0)
        self.slam_map_update_reward_grace_steps = max(int(slam_map_update_reward_grace_steps), 0)
        self._last_slam_map_update_reward = 0.0
        self._last_slam_map_update_reward_raw = 0.0
        self._last_slam_map_update_reward_reason = "init"
        self._episode_slam_map_update_reward = 0.0

        # v119: Optional positive reward log-compression.
        # This keeps early large map-building rewards from dominating critic targets
        # while preserving negative safety penalties and terminal penalties.
        self.reward_positive_log_compress = bool(reward_positive_log_compress)
        self.reward_positive_log_alpha = max(float(reward_positive_log_alpha), 1e-6)
        self.reward_positive_log_max = max(float(reward_positive_log_max), 0.0)
        self._last_reward_pre_log_compress = 0.0
        self._last_reward_post_log_compress = 0.0
        self._last_reward_log_compress_delta = 0.0

        self.max_linear_speed = float(max_linear_speed)
        self.max_angular_speed = float(max_angular_speed)
        # Optional command-space clamp used for continuing an existing SAC model
        # while exposing real-robot-like low-speed dynamics.  It deliberately does
        # not change action_space, so old checkpoints with wider action bounds can
        # still be loaded and fine-tuned safely.  The policy may output the old
        # [0,max_linear] x [-max_angular,max_angular] action, but the executed
        # TwistStamped is limited before the safety shield.
        self.velocity_command_linear_limit = (
            float(velocity_command_linear_limit) if float(velocity_command_linear_limit) > 0.0 else float(self.max_linear_speed)
        )
        self.velocity_command_angular_limit = (
            float(velocity_command_angular_limit) if float(velocity_command_angular_limit) > 0.0 else float(self.max_angular_speed)
        )
        self.velocity_command_linear_limit = float(np.clip(self.velocity_command_linear_limit, 0.0, self.max_linear_speed))
        self.velocity_command_angular_limit = float(np.clip(self.velocity_command_angular_limit, 0.0, self.max_angular_speed))

        # Episode safety envelope. Out-of-envelope poses are treated as
        # collision-like terminal states because they usually mean the robot has
        # escaped the useful map/world region or physics became unstable.
        self.terminate_on_out_of_bounds = bool(terminate_on_out_of_bounds)
        self.safety_boundary_radius_m = max(float(safety_boundary_radius_m), 0.0)
        self.safety_boundary_min_x = float(safety_boundary_min_x)
        self.safety_boundary_max_x = float(safety_boundary_max_x)
        self.safety_boundary_min_y = float(safety_boundary_min_y)
        self.safety_boundary_max_y = float(safety_boundary_max_y)
        self.safety_boundary_max_abs_z = max(float(safety_boundary_max_abs_z), 0.0)
        # Boundary checks must use a reset-stable frame.  When SLAM is active,
        # map->odom can drift or remain from the previous episode even after the
        # Gazebo model pose is teleported.  Therefore the safety envelope defaults
        # to odom, not map.  RViz /map may show a shifted robot while odom/Gazebo
        # reset is actually correct.
        # Force all runtime visualization/control frames to odom.
        # The user-facing invariant is: RViz Fixed Frame=odom, RobotModel in odom,
        # RL task/confidence/priority maps in odom, waypoint markers in odom.
        self.safety_boundary_frame = "odom"
        self.current_boundary_center_xy = np.array([0.0, 0.0], dtype=np.float32)
        self._last_out_of_bounds = False
        self._last_out_of_bounds_reason = "none"
        self._last_out_of_bounds_radius = 0.0
        self._last_out_of_bounds_x = 0.0
        self._last_out_of_bounds_y = 0.0
        self._last_out_of_bounds_z = 0.0

        requested_action_mode = str(action_mode or "nav2").strip().lower()
        if requested_action_mode not in {"waypoint", "velocity", "nav2"}:
            raise ValueError("action_mode must be one of: 'waypoint', 'velocity', 'nav2'")
        self.action_mode = requested_action_mode
        if self.action_mode == "velocity":
            self.ros.get_logger().warn(
                "PURE_VELOCITY_SAC_ENABLED | Nav2 motion executor is disabled; "
                "SAC publishes TwistStamped directly through the safety shield"
            )
        elif self.action_mode == "waypoint":
            self.ros.get_logger().warn(
                "INTERNAL_WAYPOINT_CONTROLLER_ENABLED | Nav2 motion executor is disabled; "
                "local waypoint controller publishes TwistStamped"
            )
        else:
            self.ros.get_logger().info(
                "NAV2_MOTION_EXECUTOR_ENABLED | Nav2 FollowPath/NavigateToPose owns motion"
            )

        self.waypoint_action_type = str(waypoint_action_type or "polar").strip().lower()
        if self.waypoint_action_type not in {"path", "polar"}:
            raise ValueError("waypoint_action_type must be either 'path' or 'polar'")
        if self.waypoint_action_type == "path":
            self.ros.get_logger().warn(
                "waypoint_action_type='path' is deprecated/disabled. "
                "Falling back to waypoint_action_type='polar'."
            )
            self.waypoint_action_type = "polar"
        self.waypoint_lateral_max_offset = max(float(waypoint_lateral_max_offset), 0.0)

        self.waypoint_min_distance = max(float(waypoint_min_distance), 0.0)
        self.waypoint_max_distance = max(float(waypoint_max_distance), self.waypoint_min_distance + 1e-3)
        self.waypoint_max_angle_rad = math.radians(max(float(waypoint_max_angle_deg), 1.0))
        self.waypoint_reached_tolerance = max(float(waypoint_reached_tolerance), 0.03)
        self.waypoint_control_steps = max(int(waypoint_control_steps), 1)
        self.waypoint_execute_until_reached = bool(waypoint_execute_until_reached)
        self.waypoint_timeout_sec = max(float(waypoint_timeout_sec), 0.0)
        if self.waypoint_timeout_sec > 1e-6:
            timeout_steps = int(math.ceil(self.waypoint_timeout_sec / max(self.control_dt, 1e-6)))
            self.waypoint_max_control_steps = max(timeout_steps, self.waypoint_control_steps)
        else:
            self.waypoint_max_control_steps = max(int(waypoint_max_control_steps), self.waypoint_control_steps)
        self.waypoint_timeout_stop = bool(waypoint_timeout_stop)
        self.waypoint_linear_kp = max(float(waypoint_linear_kp), 0.0)
        self.waypoint_angular_kp = max(float(waypoint_angular_kp), 0.0)
        self.waypoint_max_yaw_error_for_linear_rad = math.radians(
            max(float(waypoint_max_yaw_error_for_linear_deg), 1.0)
        )
        self.waypoint_slowdown_distance = max(float(waypoint_slowdown_distance), 0.05)
        self.waypoint_min_linear_speed = max(float(waypoint_min_linear_speed), 0.0)
        self.waypoint_disable_arrival_slowdown = bool(waypoint_disable_arrival_slowdown)
        self.waypoint_front_stop_distance = max(float(waypoint_front_stop_distance), 0.05)
        # Receding-horizon waypoint policy.  A waypoint is not held until exact
        # arrival if the robot already moved enough or rotated enough; this avoids
        # long single-target runs that destabilize SLAM after teleport reset.
        self.waypoint_replan_distance_m = max(float(waypoint_replan_distance_m), 0.03)
        self.waypoint_replan_heading_rad = math.radians(max(float(waypoint_replan_heading_deg), 1.0))
        # Direct-point primitive mode.  This removes Nav2/local-planner behavior
        # from the RL inner loop.  The policy picks a short local point; the
        # controller first aligns slowly, then drives almost straight with only
        # bounded yaw correction.  This is intentionally conservative for
        # slam_toolbox after teleport reset.
        self.waypoint_direct_point_mode = bool(waypoint_direct_point_mode)
        self.waypoint_direct_heading_tolerance_rad = math.radians(
            max(float(waypoint_direct_heading_tolerance_deg), 1.0)
        )
        self.waypoint_direct_drive_heading_limit_rad = math.radians(
            max(float(waypoint_direct_drive_heading_limit_deg), 1.0)
        )
        self.waypoint_direct_max_correction_angular = max(
            float(waypoint_direct_max_correction_angular),
            0.0,
        )
        self.waypoint_direct_min_drive_distance = max(float(waypoint_direct_min_drive_distance), 0.0)
        self.waypoint_direct_target_sector_rad = math.radians(
            max(float(waypoint_direct_target_sector_deg), 3.0)
        )
        self.waypoint_direct_turn_drive = bool(waypoint_direct_turn_drive)
        self.waypoint_direct_turn_drive_max_yaw_rad = math.radians(
            max(float(waypoint_direct_turn_drive_max_yaw_deg), 1.0)
        )
        self.waypoint_direct_turn_drive_speed_scale = float(
            np.clip(float(waypoint_direct_turn_drive_speed_scale), 0.0, 1.0)
        )
        self.waypoint_direct_turn_drive_min_speed = max(
            float(waypoint_direct_turn_drive_min_speed),
            0.0,
        )
        # Reset spawn safety validator.  Gazebo can teleport into a visually open
        # but LiDAR-colliding place; validate with fresh scan and resample.
        self.reset_pose_max_attempts = max(int(reset_pose_max_attempts), 1)
        self.reset_pose_min_clearance_m = max(float(reset_pose_min_clearance_m), 0.05)
        self.reset_pose_validation_wait_sec = max(float(reset_pose_validation_wait_sec), 0.0)
        self.post_reset_stabilize_sec = max(float(post_reset_stabilize_sec), 0.0)
        self.post_reset_stabilize_spin_steps = max(int(post_reset_stabilize_spin_steps), 1)
        self.post_reset_ready_gate = bool(post_reset_ready_gate)
        self.post_reset_ready_timeout_sec = max(float(post_reset_ready_timeout_sec), 0.0)
        self.post_reset_ready_min_known_ratio = float(np.clip(float(post_reset_ready_min_known_ratio), 0.0, 1.0))
        self.post_reset_ready_min_known_cells = max(int(post_reset_ready_min_known_cells), 0)
        self.post_reset_ready_min_lidar_beams = max(int(post_reset_ready_min_lidar_beams), 0)
        self.post_reset_ready_require_priority = bool(post_reset_ready_require_priority) and (not bool(getattr(self, "disable_priority_map", False)))

        # Action-synchronous reward gate.  SAC credit assignment becomes noisy if
        # reward is computed from a stale scan/pose while SLAM /map is still
        # lagging.  The step() path records the sensor callback times before an
        # action, waits briefly for a post-action scan/odom frame, then updates
        # the internal exploration map from that scan/pose before reward.py runs.
        # RViz /rl_* publishing may still be asynchronous; reward uses the just
        # computed internal delta, not a delayed visualized map message.
        self.action_sync_reward_gate = bool(action_sync_reward_gate)
        self.action_sync_wait_timeout_sec = max(float(action_sync_wait_timeout_sec), 0.0)
        self.action_sync_min_scan_age_sec = max(float(action_sync_min_scan_age_sec), 0.0)
        self._last_action_sync_ok = False
        self._last_action_sync_reason = "not_checked"
        self._last_action_sync_wait_sec = 0.0
        self._last_action_sync_scan_fresh = False
        self._last_action_sync_odom_fresh = False

        # Map-bounds/map-signal restart.  When reset or SLAM expansion leaves the
        # robot outside the useful /map canvas, the episode can otherwise spin
        # extremely fast with cov/prio both zero until max_episode_steps.  Cut
        # those transitions early so the replay buffer does not fill with
        # meaningless outside-map samples.
        self.map_bounds_restart = bool(map_bounds_restart)
        self.map_bounds_margin_cells = max(int(map_bounds_margin_cells), 0)
        self.map_bounds_min_local_known_ratio = float(np.clip(float(map_bounds_min_local_known_ratio), 0.0, 1.0))
        self.map_bounds_min_local_known_cells = max(int(map_bounds_min_local_known_cells), 0)
        self.map_bounds_grace_steps = max(int(map_bounds_grace_steps), 0)
        self.map_bounds_restart_penalty = max(float(map_bounds_restart_penalty), 0.0)
        self.map_bounds_bad_steps = 0
        # Cells past an edge still treated as a transient canvas/pose mismatch
        # (recoverable, grace-gated) rather than a genuine out-of-map (immediate
        # restart).  Overridable via env for tuning without code edits.
        try:
            self.map_bounds_transient_outside_cells = max(
                int(os.environ.get("TB3_RL_MAP_BOUNDS_TRANSIENT_OUTSIDE_CELLS", "24")), 0
            )
        except Exception:
            self.map_bounds_transient_outside_cells = 24
        # Pose that confidence/priority were last painted at, used to keep the
        # map-bounds check consistent with the confidence canvas.
        self._last_confidence_update_xy = None
        self._last_confidence_update_yaw = 0.0
        self._last_confidence_update_step = -10_000_000
        # Last valid TF(map->base) pose for the direct map_base_tf confidence mode
        # (used only as a short HOLD during brief TF gaps; never integrated).
        self._last_direct_tf_pose_xy = None
        self._last_direct_tf_pose_yaw = None
        self._last_direct_tf_pose_wall = 0.0
        # Exact pose confidence was painted at when unified with the TF cube; the
        # origin marker reuses this so the red sphere == the cube.
        self._confidence_unified_xy = None
        self._confidence_unified_yaw = None
        self._last_map_bounds_restart = False
        self._last_map_bounds_reason = "none"
        self._last_map_bounds_local_known_ratio = 0.0
        self._last_map_bounds_local_known_cells = 0

        self._last_post_reset_ready = False
        self._last_post_reset_ready_reason = "not_checked"
        self._last_post_reset_ready_known_ratio = 0.0
        self._last_post_reset_ready_known_cells = 0
        self._last_post_reset_ready_lidar_beams = 0
        self._last_post_reset_ready_priority = 0.0

        # Nav2 NavigateToPose mode parameters.
        # action_mode="nav2" still uses the same waypoint action encoding as
        # waypoint mode, but execution is delegated to Nav2 instead of the
        # internal /cmd_vel controller.
        self.nav2_action_name = str(nav2_action_name or "/navigate_to_pose").strip() or "/navigate_to_pose"
        self.nav2_backup_action_name = str(nav2_stuck_backup_action_name or "/backup").strip() or "/backup"
        # Nav2 behavior_server actions used only to unstick a controller_server
        # FollowPath goal that was accepted but produced neither xy nor yaw motion.
        # These remain Nav2-owned motion paths; the RL node still does not publish
        # direct /cmd_vel for the normal policy action.
        self.nav2_spin_action_name = "/spin"
        self.nav2_drive_on_heading_action_name = "/drive_on_heading"
        # Use Nav2 controller_server FollowPath for actual motion.
        # NavigateToPose BT can spend most of each short RL macro-step rotating/replanning.
        # FollowPath still uses Nav2 controller/costmap and publishes Nav2 /cmd_vel,
        # but bypasses BT goal-orientation oscillation for short local goals.
        self.nav2_follow_path_action_name = "/follow_path"
        # Keep RViz/learning maps locked to /map when requested, but feed Nav2
        # controller_server FollowPath in odom.  TurtleBot3 Nav2 local costmaps
        # commonly run with global_frame=odom; sending very short FollowPath
        # goals in map can be accepted by the action server but produce zero
        # /cmd_vel while transforms/costmaps wait or reject the path internally.
        # This does not change the RL waypoint, reward, priority map, or RViz
        # frame. It only converts the same local target from pose_frame(map) to
        # odom for the low-level Nav2 controller goal.
        self.nav2_goal_timeout_sec = max(float(nav2_goal_timeout_sec), 0.05)
        self.nav2_replan_on_movement = bool(nav2_replan_on_movement)
        self.nav2_replan_distance_m = max(float(nav2_replan_distance_m), 0.03)
        self.nav2_early_replan_remaining_m = max(float(nav2_early_replan_remaining_m), 0.03)
        self.nav2_near_goal_replan_only = bool(nav2_near_goal_replan_only)
        # Nav2 streaming macro-step mode.
        #
        # FollowPath already makes the robot translate reliably, but holding one
        # local path until reached/timeout creates stop-and-go motion: the robot
        # moves, pauses near/inside the local path, then the SAC loop samples a new
        # waypoint.  For RL exploration we want receding-horizon behavior instead:
        # keep Nav2 as the sole /cmd_vel owner, but return from env.step() after
        # the robot has moved a meaningful partial distance or after a short
        # control window with visible progress.  The next SAC action then sends a
        # new FollowPath goal, and controller_server preempts the old path.
        self.nav2_control_window_sec = max(float(nav2_control_window_sec), 0.05)
        self.nav2_continuous_goal_update = bool(nav2_continuous_goal_update)
        self.nav2_preempt_without_cancel = bool(nav2_preempt_without_cancel)
        if self.action_mode == "nav2":
            # Fast receding-horizon Nav2 mode: do not serialize SAC decisions behind
            # full goal completion or synchronous cancel.  Each accepted FollowPath /
            # NavigateToPose goal is allowed to run for only a short observation
            # window; the next SAC step sends a fresh goal and lets Nav2 preempt the
            # previous one.  This keeps Nav2 as the sole /cmd_vel owner while removing
            # the visible stop-go behavior caused by waiting for result/cancel.
            if self.nav2_continuous_goal_update:
                self.nav2_control_window_sec = float(
                    np.clip(self.nav2_control_window_sec, 0.08, 0.35)
                )
                self.nav2_replan_distance_m = float(
                    np.clip(self.nav2_replan_distance_m, 0.05, 0.18)
                )
                self.nav2_early_replan_remaining_m = float(
                    np.clip(self.nav2_early_replan_remaining_m, 0.12, 0.45)
                )
                self.nav2_goal_timeout_sec = float(
                    np.clip(self.nav2_goal_timeout_sec, 0.40, 1.20)
                )
            else:
                self.nav2_preempt_without_cancel = False
        self.nav2_wait_timeout_sec = max(float(nav2_wait_timeout_sec), 0.05)
        self.nav2_goal_reached_tolerance = max(float(nav2_goal_reached_tolerance), 0.03)
        self.nav2_cancel_on_timeout = bool(nav2_cancel_on_timeout)
        self.nav2_cancel_on_reached = bool(nav2_cancel_on_reached)
        self.nav2_send_goal_wait_sec = max(float(nav2_send_goal_wait_sec), 0.01)
        # 0.0 means fire-and-forget cancel.  Normal high-rate Nav2 streaming should
        # rely on action preemption, not on waiting for cancel acknowledgements.
        self.nav2_cancel_wait_sec = max(float(nav2_cancel_wait_sec), 0.0)
        self.nav2_use_goal_orientation = bool(nav2_use_goal_orientation)
        self.nav2_auto_start = bool(nav2_auto_start)
        self.nav2_launch_package = str(nav2_launch_package or "nav2_bringup").strip() or "nav2_bringup"
        self.nav2_launch_file = str(nav2_launch_file or "navigation_launch.py").strip() or "navigation_launch.py"
        self.nav2_params_file = str(nav2_params_file or "").strip()
        self._nav2_runtime_params_file = ""
        self._nav2_stamped_params_prepared = False
        self.nav2_use_sim_time = bool(nav2_use_sim_time)
        self.nav2_startup_timeout_sec = max(float(nav2_startup_timeout_sec), self.nav2_wait_timeout_sec)
        self.nav2_client = None
        self.nav2_follow_path_client = None
        self.nav2_backup_client = None
        self._nav2_use_follow_path_controller = False
        self.nav2_proc: Optional[subprocess.Popen] = None
        self._nav2_log_handle = None
        self._nav2_log_path = ""
        self._nav2_goal_handle = None
        self.nav2_clear_global_service = "/global_costmap/clear_entirely_global_costmap"
        self.nav2_clear_local_service = "/local_costmap/clear_entirely_local_costmap"
        self.nav2_clear_global_client = None
        self.nav2_clear_local_client = None
        self._last_nav2_goal_source = "none"
        self._last_nav2_goal_valid = False
        self._last_nav2_goal_validation = "none"
        self._last_nav2_moved_distance = 0.0
        self._last_nav2_yaw_delta_since_goal = 0.0
        self._last_nav2_unavailable_log_time = 0.0

        # SLAM local-quality based adaptive speed limiter.
        # When the local SLAM map is sparse/stale, the controller reduces only the
        # linear speed limit. It does not stop at waypoints; obstacle logic remains
        # the only hard forward-stop gate.
        self.slam_adaptive_speed = bool(slam_adaptive_speed)
        self.slam_local_speed_radius = max(float(slam_local_speed_radius), 0.10)
        self.slam_front_speed_distance = max(float(slam_front_speed_distance), 0.10)
        self.slam_front_speed_half_angle_rad = math.radians(
            max(float(slam_front_speed_half_angle_deg), 1.0)
        )
        self.slam_speed_min_scale = float(np.clip(slam_speed_min_scale, 0.05, 1.0))
        self.slam_speed_max_scale = float(np.clip(slam_speed_max_scale, self.slam_speed_min_scale, 1.0))
        self.slam_speed_local_weight = max(float(slam_speed_local_weight), 0.0)
        self.slam_speed_front_weight = max(float(slam_speed_front_weight), 0.0)
        self.slam_speed_fresh_weight = max(float(slam_speed_fresh_weight), 0.0)
        self.slam_speed_map_age_soft_limit_sec = max(float(slam_speed_map_age_soft_limit_sec), 0.05)

        # Continuous linear speed scaling parameters.
        # known_ratio <= low  -> min_scale
        # known_ratio >= high -> max_scale
        # between them        -> strictly linear interpolation.
        self.slam_speed_known_low_ratio = float(np.clip(slam_speed_known_low_ratio, 0.0, 1.0))
        self.slam_speed_known_high_ratio = float(np.clip(slam_speed_known_high_ratio, 0.0, 1.0))
        if self.slam_speed_known_high_ratio <= self.slam_speed_known_low_ratio + 1e-4:
            self.slam_speed_known_high_ratio = min(self.slam_speed_known_low_ratio + 1e-3, 1.0)
        self.slam_speed_fresh_low_score = float(np.clip(slam_speed_fresh_low_score, 0.0, 1.0))
        self.slam_speed_smoothing_alpha = float(np.clip(slam_speed_smoothing_alpha, 0.0, 0.95))

        self._last_slam_local_known_ratio = 1.0
        self._last_slam_front_known_ratio = 1.0
        self._last_slam_local_linear_score = 1.0
        self._last_slam_front_linear_score = 1.0
        self._last_slam_fresh_score = 1.0
        self._last_slam_fresh_linear_score = 1.0
        self._last_slam_quality_score = 1.0
        self._last_slam_speed_raw_scale = 1.0
        self._last_slam_speed_scale = 1.0
        self._last_slam_speed_limit = self.max_linear_speed

        self.waypoint_marker_topic = str(waypoint_marker_topic).strip()
        self.waypoint_path_topic = str(waypoint_path_topic).strip()
        self.waypoint_visual_history_len = max(int(waypoint_visual_history_len), 1)
        self.waypoint_visual_publish_every_n = max(int(waypoint_visual_publish_every_n), 1)
        # RViz 디버깅 기본값은 현재 waypoint만 표시한다.
        # 과거 waypoint 궤적은 정책이 와리가리칠 때 화면을 오염시키므로 옵션으로만 켠다.
        self.waypoint_show_history = bool(waypoint_show_history)

        self.use_slam_map = bool(use_slam_map)
        # Frame policy:
        #   - map-aligned debugging/training: map_frame == pose_frame == "map"
        #     SLAM /map, filtered SLAM, priority/confidence/task maps, waypoint
        #     markers all live in the same global frame. RViz Fixed Frame must be map.
        #   - odom-local fallback: pose_frame == "odom". In that mode SLAM /map is
        #     not injected unless a full grid transform is implemented.
        self.map_frame = str(map_frame or "map").strip().lstrip("/") or "map"
        self.pose_frame = str(pose_frame or self.map_frame).strip().lstrip("/") or self.map_frame
        # Keep RViz/learning maps in pose_frame.  Motion progress is measured in
        # odom because odom is continuous across short controller windows, but the
        # actual Nav2 goal is NOT sent as FollowPath in odom when pose_frame=map.
        # That combination was accepted by controller_server while producing zero
        # /cmd_vel on TurtleBot3/Nav2.  In map-locked mode, force NavigateToPose in
        # map and let Nav2 perform the map->odom->base_link transform internally.
        self.nav2_motion_frame = "odom" if self.pose_frame == "map" else self.pose_frame
        self._last_nav2_motion_frame = self.nav2_motion_frame
        # This training branch is map-locked and Nav2-only.  Do not use
        # controller_server /follow_path at all: in this workspace it can accept
        # short goals while publishing no /cmd_vel, producing moved=0.000 loops.
        # Always send NavigateToPose goals in the same frame as the RL/SLAM maps.
        self.nav2_prefer_navigate_to_pose = True
        self.nav2_force_navigate_to_pose = True
        self._nav2_use_follow_path_controller = False
        if self.nav2_prefer_navigate_to_pose:
            # BT NavigateToPose can spend the first few seconds rotating/replanning.
            # Do not abort at 5 s before any translation appears.  Streaming still
            # returns early once movement is detected.
            self.nav2_goal_timeout_sec = max(float(getattr(self, "nav2_goal_timeout_sec", 0.0)), 12.0)
        # True when every RL layer, waypoint marker, and accepted SLAM prior is
        # expressed in odom.  The raw slam_toolbox /map may still have frame_id=map,
        # so it must be transformed before being injected or used as RViz metadata.
        self.odom_unified_frame_mode = (self.pose_frame == "odom")
        # In odom-unified mode we freeze map->odom once per episode after SLAM
        # reset.  Using the continuously changing latest SLAM TF to reproject /map
        # every update is what made /rl_priority_map and /rl_filtered_slam_map drift.
        self._episode_slam_transform_source = ""
        self._episode_slam_transform_target = ""
        self._slam_transform_cache_key = None
        self._slam_transform_cache_msg = None
        self._episode_slam_transform = None
        self._slam_transform_cache_key = None
        self._slam_transform_cache_msg = None

        # v8 confidence pose policy.
        # Do NOT integrate commands.  Confidence/priority/task maps must be
        # physically aligned to the same /map canvas.  We therefore anchor one
        # valid SLAM map pose at episode start, then project the *actual robot
        # odometry* delta (/model/burger/odometry first, /odom message second)
        # into that map frame.  If the real odometry source is unavailable, the
        # update falls back to strict map TF and emits a warning instead of
        # silently inventing motion.
        self.confidence_pose_source = str(
            os.environ.get("TB3_RL_CONFIDENCE_POSE_SOURCE", "real_odom_anchored") or "real_odom_anchored"
        ).strip().lower()
        self.confidence_motion_source = str(
            os.environ.get("TB3_RL_CONFIDENCE_MOTION_SOURCE", "model_odom") or "model_odom"
        ).strip().lower()
        self._confidence_odom_anchor = None
        self._last_confidence_pose_log_step = -10_000_000
        self._last_confidence_pose_warn_time = 0.0
        # v7: command-integrated pose fallback for confidence only.
        # If /model odom, /odom msg, and odom TF are stale/frozen, this lets the
        # camera-front confidence cone follow the actually published cmd_vel.
        self._confidence_cmd_xy = np.zeros(2, dtype=np.float32)
        self._confidence_cmd_yaw = 0.0
        self._confidence_cmd_last_time = None
        self._confidence_cmd_last_step = -1

        # v101: AMCL pose source for confidence painting.
        # /amcl_pose is already in the map frame when AMCL is running, so this
        # mode can use the same pose that localization publishes instead of
        # Cartographer TF, Gazebo model odom, or anchored odom deltas.
        self.amcl_pose_topic = str(os.environ.get("TB3_RL_AMCL_POSE_TOPIC", "/amcl_pose") or "/amcl_pose").strip()
        try:
            self.amcl_pose_max_age_sec = max(float(os.environ.get("TB3_RL_AMCL_POSE_MAX_AGE_SEC", "1.0")), 0.0)
        except Exception:
            self.amcl_pose_max_age_sec = 1.0
        self.amcl_fallback_pose_source = str(os.environ.get("TB3_RL_AMCL_FALLBACK_POSE_SOURCE", "none") or "none").strip().lower()
        self._latest_amcl_pose = None
        self._amcl_pose_sub = None
        self._last_amcl_pose_warn_time = 0.0

        # v102: AMCL-compatible pose shim.  During SLAM-only runs there is no
        # real Nav2 AMCL node, but RViz still renders the robot from the TF chain
        # map -> base_footprint.  For confidence painting we need an explicit
        # /amcl_pose-style PoseWithCovarianceStamped source.  This publisher
        # mirrors exactly that RViz TF pose into /amcl_pose so the confidence cone
        # uses the same map-frame robot position visible in RViz, without starting
        # full Nav2 localization or map_server.
        try:
            _pose_src = str(os.environ.get("TB3_RL_CONFIDENCE_POSE_SOURCE", self.confidence_pose_source) or "").strip().lower()
        except Exception:
            _pose_src = ""
        _auto_amcl_default = _pose_src in {"amcl", "amcl_pose", "map_amcl", "amcl_map", "amcl_pose_topic", "nav2_amcl"}
        self.auto_publish_amcl_pose_from_tf = self._scan_bool_env(
            "TB3_RL_AUTO_PUBLISH_AMCL_POSE_FROM_TF",
            _auto_amcl_default,
        )
        self.amcl_pose_tf_target_frame = str(
            os.environ.get("TB3_RL_AMCL_POSE_TF_TARGET_FRAME", self.map_frame) or self.map_frame
        ).strip().lstrip("/") or "map"
        # v103 hotfix: GazeboNavEnv has no self.base_frame at this point.
        # Use environment defaults only; the ROS interface owns the real base_frame.
        # This prevents AttributeError during __init__ while keeping the same
        # effective default used by RViz/TF: base_footprint.
        _default_amcl_source_frame = str(
            os.environ.get(
                "TB3_RL_RVIZ_ROBOT_FRAME",
                os.environ.get("TB3_RL_BASE_FRAME", "base_footprint"),
            )
            or "base_footprint"
        ).strip().lstrip("/") or "base_footprint"
        self.amcl_pose_tf_source_frame = str(
            os.environ.get("TB3_RL_AMCL_POSE_TF_SOURCE_FRAME", _default_amcl_source_frame)
            or _default_amcl_source_frame
        ).strip().lstrip("/") or "base_footprint"
        try:
            self.amcl_pose_tf_publish_hz = max(float(os.environ.get("TB3_RL_AMCL_POSE_TF_PUBLISH_HZ", "20.0")), 0.0)
        except Exception:
            self.amcl_pose_tf_publish_hz = 20.0
        try:
            self.amcl_pose_tf_timeout_sec = max(float(os.environ.get("TB3_RL_AMCL_POSE_TF_TIMEOUT_SEC", "0.05")), 0.0)
        except Exception:
            self.amcl_pose_tf_timeout_sec = 0.05
        try:
            self.amcl_pose_cov_xy = max(float(os.environ.get("TB3_RL_AMCL_POSE_COV_XY", "0.0025")), 0.0)
        except Exception:
            self.amcl_pose_cov_xy = 0.0025
        try:
            self.amcl_pose_cov_yaw = max(float(os.environ.get("TB3_RL_AMCL_POSE_COV_YAW", "0.01")), 0.0)
        except Exception:
            self.amcl_pose_cov_yaw = 0.01
        self._amcl_pose_tf_pub = None
        self._amcl_pose_tf_timer = None
        self._last_amcl_pose_tf_warn_time = 0.0
        # v105: the TF bridge can be changed to an anchored odometry bridge.
        # This is useful when Gazebo/model odometry moves but TF(map->base) is
        # stale or delayed in the local TF buffer.  It still publishes a standard
        # /amcl_pose message, but the pose is generated as:
        #   initial map->base TF anchor + actual/model odom SE(2) delta.
        self.amcl_pose_bridge_source = str(
            os.environ.get("TB3_RL_AMCL_POSE_BRIDGE_SOURCE", "tf") or "tf"
        ).strip().lower()
        self._amcl_pose_odom_anchor = None

        # reset_x/reset_y는 실제 Gazebo reset target이다.  이전 패치에서
        # reset_pose_candidates가 (1.2, -2.2)로 하드코딩되어 CLI의
        # --reset-x/--reset-y가 무시되는 문제가 있었다.  기본값은 다시
        # 정확히 (0, 0) fixed reset으로 둔다.
        self.reset_x = float(reset_x)
        self.reset_y = float(reset_y)
        self.reset_pose_mode = str(reset_pose_mode or "fixed").strip().lower()
        if self.reset_pose_mode not in {"fixed", "corners", "house_random", "house_inside_random", "list"}:
            raise ValueError("reset_pose_mode must be one of: fixed, corners, house_random, house_inside_random, list")
        self.reset_offset = max(float(reset_offset), 0.0)
        self.reset_pose_list = str(reset_pose_list or "")
        self.rviz_zero_robot_on_reset = bool(rviz_zero_robot_on_reset)
        self.rviz_origin_wait_sec = max(float(rviz_origin_wait_sec), 0.0)
        self.rviz_origin_tolerance_m = max(float(rviz_origin_tolerance_m), 0.01)
        self.current_reset_xy = np.array(
            [self.reset_x, self.reset_y],
            dtype=np.float32,
        )
        self.current_boundary_center_xy = self.current_reset_xy.copy()
        # Hard invariant for this project: Gazebo pose reset must always be followed
        # by a fresh SLAM reset.  Otherwise raw /map, /rl_priority_map and Nav2
        # costmaps gradually diverge after repeated respawns.
        self.reset_slam_on_reset = True if self.use_slam_map else bool(reset_slam_on_reset)
        self.restart_slam_on_reset = True if self.use_slam_map else bool(restart_slam_on_reset)
        self.reset_slam_every_n_episodes = 1 if self.use_slam_map else max(int(reset_slam_every_n_episodes), 0)
        self.reset_tf_buffer_on_reset = True if self.use_slam_map else bool(reset_tf_buffer_on_reset)
        self.episode_index = 0
        self.slam_reset_timeout_sec = float(slam_reset_timeout_sec)
        self.slam_reset_warmup_steps = max(int(slam_reset_warmup_steps), 0)
        self.ignore_slam_prior_this_episode = False
        self.use_map_cnn = bool(use_map_cnn)
        self.map_obs_size = max(int(map_obs_size), 8)
        self.map_obs_size_m = max(float(map_obs_size_m), 1.0)
        self.use_temporal_cnn = bool(use_temporal_cnn)
        self.temporal_history_len = max(int(temporal_history_len), 2)
        self.front_fov_deg = float(front_fov_deg)
        self.confidence_decay_per_step = float(confidence_decay_per_step)
        self.confidence_max_range = float(confidence_max_range)
        self.front_angle_sigma_deg = float(front_angle_sigma_deg)
        self.seen_confidence_floor = float(seen_confidence_floor)
        self.priority_target_lock_steps = max(int(priority_target_lock_steps), 0)
        self.priority_target_switch_margin = float(np.clip(float(priority_target_switch_margin), 0.0, 1.0))
        self.rl_priority_topic = "" if self.disable_priority_map else str(rl_priority_topic).strip()
        # Empty string intentionally disables RViz path publishing. Publishing a
        # nav_msgs/Path every RL step is useful for debugging but slows training.
        self.rl_path_topic = str(rl_path_topic).strip()
        # Empty string intentionally disables filtered SLAM republishing.
        self.rl_filtered_slam_topic = str(rl_filtered_slam_topic).strip()
        self.slam_map_accept_delay_sec = max(float(slam_map_accept_delay_sec), 0.0)
        self.slam_map_max_age_sec = max(float(slam_map_max_age_sec), 0.0)
        self.strict_slam_map_required = bool(strict_slam_map_required) and bool(self.use_slam_map)
        self.strict_slam_map_wait_timeout_sec = max(float(strict_slam_map_wait_timeout_sec), 0.5)
        self.strict_slam_map_retry_interval_sec = max(float(strict_slam_map_retry_interval_sec), 0.10)
        self.strict_slam_map_min_known_cells = max(int(strict_slam_map_min_known_cells), 0)
        self.strict_slam_map_min_known_ratio = max(float(strict_slam_map_min_known_ratio), 0.0)
        self._strict_slam_map_ready_count = 0
        self._slam_map_min_wall_time = 0.0
        self._slam_map_accept_after_wall_time = 0.0
        self._last_slam_gate_reason = "not_initialized"
        self._last_slam_map_age_sec = -1.0
        self._last_slam_map_delay_remaining_sec = 0.0
        self.suppress_gap_confidence = bool(suppress_gap_confidence)
        self.gap_occupied_threshold = float(gap_occupied_threshold)
        self.gap_check_radius_m = float(gap_check_radius_m)
        self.gap_min_width_m = float(gap_min_width_m)
        self.gap_max_width_m = float(gap_max_width_m)
        self.map_expand_chunk_cells = max(int(map_expand_chunk_cells), 1)
        self.map_publish_every_n = max(int(map_publish_every_n), 1)
        self.max_planned_candidates = max(int(max_planned_candidates), 1)
        self.max_alternative_paths = max(int(max_alternative_paths), 1)
        self.path_visual_publish_every_n = max(int(path_visual_publish_every_n), 0)
        self.priority_recompute_interval = max(int(priority_recompute_interval), 1)
        self.priority_visit_suppression_radius_m = max(float(priority_visit_suppression_radius_m), 0.0)
        self.priority_visit_suppression_gain = float(np.clip(priority_visit_suppression_gain, 0.0, 1.0))
        self.priority_visit_suppression_max = float(np.clip(priority_visit_suppression_max, 0.0, 1.0))
        self.priority_observed_suppression_gain = float(np.clip(priority_observed_suppression_gain, 0.0, 1.0))
        self.priority_clear_fov_deg = float(priority_clear_fov_deg)
        self.priority_clear_max_range_m = max(float(priority_clear_max_range_m), 0.05)
        self.priority_clear_robot_radius_m = max(float(priority_clear_robot_radius_m), 0.05)
        self.priority_clear_min_value = float(np.clip(priority_clear_min_value, 0.0, 100.0))
        self.priority_clear_sigma_m = max(float(priority_clear_sigma_m), 0.05)
        self.priority_clear_angle_sigma_deg = max(float(priority_clear_angle_sigma_deg), 1e-3)
        self.priority_clear_min_weight = float(np.clip(priority_clear_min_weight, 0.0, 1.0))
        self.priority_clear_visit_sigma_m = max(float(priority_clear_visit_sigma_m), 0.05)
        self.wall_support_radius_m = max(float(wall_support_radius_m), 0.01)
        self.wall_support_density_threshold = max(float(wall_support_density_threshold), 1e-4)
        self.open_space_front_distance_m = max(float(open_space_front_distance_m), 0.05)
        self.open_space_side_width_m = max(float(open_space_side_width_m), 0.05)
        self.open_space_forward_penalty = max(float(open_space_forward_penalty), 0.0)
        self.map_keepalive_period_sec = max(float(map_keepalive_period_sec), 0.0)
        # Live map updater: confidence/priority layers are refreshed from the
        # latest LaserScan+pose on a wall-clock timer, independent of waypoint
        # generation. 0.10s == 10 Hz.
        self.map_live_update_period_sec = max(float(map_live_update_period_sec), 0.0)
        self._map_live_update_timer = None
        self._map_live_update_paused = True
        self._map_live_update_busy = False
        self._last_live_map_update_wall = 0.0
        self._last_live_map_update_error_wall = 0.0
        self._last_live_map_update_count = 0

        # Debug visualization of the exact CNN map observation.  These topics are
        # robot-local OccupancyGrid layers in debug_input_map_frame_id.  The frame
        # convention is REP-105 style: x=robot forward, y=robot left.  Therefore
        # RViz shows the same orientation the policy receives: front is +x, left
        # is +y, and the priority channel is channel 4 of obs["map"].
        self.debug_input_map = bool(debug_input_map)
        prefix = str(debug_input_map_topic_prefix or "/rl_debug_input").strip().rstrip("/")
        self.debug_input_map_topic_prefix = prefix or "/rl_debug_input"
        self.debug_input_map_frame_id = str(debug_input_map_frame_id or "base_link").strip() or "base_link"
        self.debug_input_map_publish_every_n = max(int(debug_input_map_publish_every_n), 1)
        self._last_debug_input_map_publish_step = -1
        self._last_debug_input_map_published = False
        self.debug_input_map_publishers = {}

        self.vector_history = deque(maxlen=self.temporal_history_len)
        self.map_history = deque(maxlen=self.temporal_history_len)

        # 이미 확인한 영역에서 새 정보 없이 머무는 시간을 누적한다.
        # 이 값은 reward에서 시간 증가형 penalty로 사용한다.
        self.explored_stall_steps = 0
        self.explored_stall_start_steps = max(int(explored_stall_start_steps), 0)
        self.explored_stall_growth = max(float(explored_stall_growth), 0.0)
        self.explored_stall_power = max(float(explored_stall_power), 1.0)
        self.explored_stall_max_penalty = max(float(explored_stall_max_penalty), 0.0)
        self.confidence_stall_steps = 0
        self.confidence_stall_start_steps = max(int(confidence_stall_start_steps), 0)
        self.confidence_stall_growth = max(float(confidence_stall_growth), 0.0)
        self.confidence_stall_power = max(float(confidence_stall_power), 1.0)
        self.confidence_stall_max_penalty = max(float(confidence_stall_max_penalty), 0.0)
        self.confidence_stall_gain_threshold = max(float(confidence_stall_gain_threshold), 0.0)
        self.confidence_stall_low_ratio_threshold = float(np.clip(float(confidence_stall_low_ratio_threshold), 0.0, 1.0))

        # 저속/무전진 상태에서 회전이 몇 step 연속되는지 누적한다.
        # reward.py에서 제자리 회전이 지속될수록 growing penalty로 사용한다.
        self.sustained_rotation_steps = 0

        # 전진은 하지만 작은 원/호 궤적으로 같은 공간을 도는 orbit-loop를 추적한다.
        # 제자리 회전(spinStall)과 다르게, 선속도가 있는 arc-loop를 잡기 위한 상태다.
        self.orbit_stall_steps = 0
        self._orbit_pose_history = deque(maxlen=48)
        self._last_orbit_path_efficiency = 1.0
        self._last_orbit_path_length = 0.0
        self._last_orbit_net_displacement = 0.0
        self._last_orbit_yaw_accum = 0.0
        self._last_orbit_reason = "init"

        # Policy 출력은 바로 cmd_vel로 보내지 않고, 물리적으로 가능한 제어 신호로 필터링한다.
        # 목적:
        #   - 좌우 각속도 sign flip으로 생기는 본체 떨림 억제
        #   - 직진/호 주행/제자리 회전 모드가 매 step 흔들리지 않게 hysteresis 부여
        #   - 후진은 action_space에서 이미 제거되어 있으므로 linear_x는 [0, max_linear_speed] 유지
        self.filtered_action = np.zeros(2, dtype=np.float32)
        self.raw_action = np.zeros(2, dtype=np.float32)
        self.prev_policy_action = np.zeros(2, dtype=np.float32)
        self._last_waypoint_local = np.zeros(2, dtype=np.float32)
        self._last_waypoint_world = np.zeros(2, dtype=np.float32)
        self._last_waypoint_distance = 0.0
        self._last_waypoint_angle = 0.0
        self._last_waypoint_lateral_offset = 0.0
        self._last_waypoint_heading_delta = 0.0
        self._prev_waypoint_angle_for_reward: Optional[float] = None
        self._last_waypoint_action_type = self.waypoint_action_type
        self._last_waypoint_reached = False
        self._last_waypoint_timed_out = False
        self._last_waypoint_final_error = 0.0
        self._last_nav2_goal_heading_error = 999.0
        self._last_nav2_goal_front_min = 999.0
        self._last_nav2_backup_gate_reason = "none"
        self._last_controller_steps = 0
        self._waypoint_history = deque(maxlen=self.waypoint_visual_history_len)
        self._last_lidar_action_obstacle_distance = 999.0
        self._last_lidar_action_obstacle_score = 0.0
        self._last_lidar_front_obstacle_distance = 999.0
        self.action_smoothing_alpha = float(np.clip(action_smoothing_alpha, 0.0, 1.0))
        self.max_linear_delta = max(float(max_linear_delta), 0.0)
        self.max_angular_delta = max(float(max_angular_delta), 0.0)
        self.linear_deadband = max(float(linear_deadband), 0.0)
        self.angular_deadband = max(float(angular_deadband), 0.0)
        self.enable_motion_mode_hysteresis = bool(enable_motion_mode_hysteresis)
        self.motion_mode = "STRAIGHT"

        if self.physics_step_size <= 0.0:
            raise ValueError("physics_step_size must be positive")

        if self.control_dt <= 0.0:
            raise ValueError("control_dt must be positive")

        self.sim_steps_per_action = max(
            int(round(self.control_dt / self.physics_step_size)),
            1,
        )

        self.reset_manager: Optional[ResetManager] = None
        self.sim_controller: Optional[GazeboSimController] = None

        if self.enable_pose_reset:
            self.reset_manager = ResetManager(
                node=self.ros,
                entity_name=self.entity_name,
                set_pose_service=set_pose_service,
                reset_z=reset_z,
            )

        if self.use_world_step:
            self.sim_controller = GazeboSimController(
                node=self.ros,
                control_service=world_control_service,
                auto_start_bridge=True,
            )

            self.sim_controller.pause(True)

            self.ros.get_logger().info(
                "World stepping enabled: "
                f"control_dt={self.control_dt}, "
                f"physics_step_size={self.physics_step_size}, "
                f"sim_steps_per_action={self.sim_steps_per_action}"
            )

        # v5: policy LiDAR input can be reduced to 60 sectors.
        # Existing 360-bin checkpoints are not compatible with this setting;
        # train/evaluate checkpoints with the same --num-lidar-bins.
        self.num_lidar_bins = max(int(num_lidar_bins), 1)
        self.no_priority_model_input = bool(getattr(self, "disable_priority_map", False)) or (
            str(os.environ.get("TB3_RL_NO_PRIORITY_MODEL_INPUT", "0")).strip().lower()
            in {"1", "true", "yes", "on", "enable", "enabled"}
        )
        # Normal vector extras are 10.  No-priority mode removes target_priority
        # from the policy vector, so the actor/critic sees lidar+9 instead.
        self.obs_extra_dim = 9 if self.no_priority_model_input else 10
        self.obs_dim = self.num_lidar_bins + self.obs_extra_dim
        self.map_channels = 4 if self.no_priority_model_input else 5

        self.exploration_map = ExplorationGridMap(
            node=self.ros,
            resolution=0.05,
            size_m=8.0,
            origin_x=-4.0,
            origin_y=-4.0,
            # v11: every OccupancyGrid layer is locked to the SLAM map frame.
            # Robot-centered crops use TF(map->base_footprint); LiDAR rays use
            # TF(map->base_scan).  Do not let pose_frame/odom redefine layer
            # coordinates.
            frame_id=self.map_frame,
            publish_topic=rl_map_topic,
            confidence_publish_topic=rl_confidence_topic,
            priority_publish_topic=self.rl_priority_topic,
            disable_priority_map=self.disable_priority_map,
            path_publish_topic=self.rl_path_topic,
            filtered_slam_publish_topic=self.rl_filtered_slam_topic,
            # Strict /map-locked publication is useful only when the runtime is
            # actually map-frame aligned.  In the current pure-velocity odom mode
            # we must NOT wait for /map metadata before publishing RL layers;
            # otherwise RViz repeatedly prints MAP_LOCKED_PUBLISH_WAITING_FOR_SLAM_REF
            # and /rl_* maps disappear until SLAM produces a fresh reference.
            # Therefore odom mode uses the common internal odom grid, while map
            # mode can still resample onto the exact /map canvas.
            # Always publish RL/confidence/priority/filtered-SLAM layers on the
            # accepted /map canvas when SLAM is enabled.  ExplorationGridMap now
            # transforms between map and odom internally, so pure velocity odom
            # control can still get RViz layers that are locked to /map metadata.
            publish_slam_aligned=True,  # v5: RViz /rl_* layers are reprojected onto the raw /map canvas so confidence remains visible/aligned after SLAM map growth.
            keepalive_publish_period_sec=self.map_keepalive_period_sec,
            lidar_stride=2,
            max_range=3.5,
            publish_every_n=self.map_publish_every_n,
            max_planned_candidates=self.max_planned_candidates,
            max_alternative_paths=self.max_alternative_paths,
            path_visual_publish_every_n=self.path_visual_publish_every_n,
            target_lock_steps=self.priority_target_lock_steps,
            target_switch_margin=self.priority_target_switch_margin,
            min_known_confidence=8.0,
            low_confidence_threshold=35.0,
            stale_after_steps=180,
            confidence_decay_per_step=self.confidence_decay_per_step,
            logodds_decay_per_step=0.0008,
            distance_weight_beta=0.30,
            confidence_max_range=self.confidence_max_range,
            front_angle_sigma_deg=self.front_angle_sigma_deg,
            seen_confidence_floor=self.seen_confidence_floor,
            suppress_gap_confidence=self.suppress_gap_confidence,
            gap_occupied_threshold=self.gap_occupied_threshold,
            gap_check_radius_m=self.gap_check_radius_m,
            gap_min_width_m=self.gap_min_width_m,
            gap_max_width_m=self.gap_max_width_m,
            priority_recompute_interval=self.priority_recompute_interval,
            priority_visit_suppression_radius_m=self.priority_visit_suppression_radius_m,
            priority_visit_suppression_gain=self.priority_visit_suppression_gain,
            priority_visit_suppression_max=self.priority_visit_suppression_max,
            priority_observed_suppression_gain=self.priority_observed_suppression_gain,
            priority_clear_fov_deg=self.priority_clear_fov_deg,
            priority_clear_max_range_m=self.priority_clear_max_range_m,
            priority_clear_robot_radius_m=self.priority_clear_robot_radius_m,
            priority_clear_min_value=self.priority_clear_min_value,
            priority_clear_sigma_m=self.priority_clear_sigma_m,
            priority_clear_angle_sigma_deg=self.priority_clear_angle_sigma_deg,
            priority_clear_min_weight=self.priority_clear_min_weight,
            priority_clear_visit_sigma_m=self.priority_clear_visit_sigma_m,
            wall_support_radius_m=self.wall_support_radius_m,
            wall_support_density_threshold=self.wall_support_density_threshold,
            open_space_front_distance_m=self.open_space_front_distance_m,
            open_space_side_width_m=self.open_space_side_width_m,
            map_expand_chunk_cells=self.map_expand_chunk_cells,
            use_slam_prior=self.use_slam_map,
            front_fov_deg=self.front_fov_deg,
        )

        self.last_map_stats = self._empty_map_stats()

        self.target_coverage_ratio = 0.90

        if self.use_map_cnn:
            obs_spaces = {
                "vector": spaces.Box(
                    low=-1.0,
                    high=1.0,
                    shape=(self.obs_dim,),
                    dtype=np.float32,
                ),
                "map": spaces.Box(
                    low=0.0,
                    high=1.0,
                    shape=(self.map_channels, self.map_obs_size, self.map_obs_size),
                    dtype=np.float32,
                ),
            }

            if self.use_temporal_cnn:
                obs_spaces["seq"] = spaces.Box(
                    low=-1.0,
                    high=1.0,
                    shape=(self.temporal_history_len, self.obs_dim),
                    dtype=np.float32,
                )
                obs_spaces["map_seq"] = spaces.Box(
                    low=0.0,
                    high=1.0,
                    shape=(self.temporal_history_len, self.map_channels, self.map_obs_size, self.map_obs_size),
                    dtype=np.float32,
                )

            self.observation_space = spaces.Dict(obs_spaces)
        else:
            self.observation_space = spaces.Box(
                low=-1.0,
                high=1.0,
                shape=(self.obs_dim,),
                dtype=np.float32,
            )

        if self.action_mode in {"waypoint", "nav2"}:
            # Policy는 더 이상 cmd_vel을 직접 출력하지 않는다.
            # waypoint_action_type="path"이면 action[0]=path lookahead 비율,
            # action[1]=path lateral offset 비율이다. "polar"이면 이전처럼
            # action[0]=local waypoint 거리 비율, action[1]=local waypoint 방향 비율이다.
            self.action_space = spaces.Box(
                low=np.array([0.0, -1.0], dtype=np.float32),
                high=np.array([1.0, 1.0], dtype=np.float32),
                dtype=np.float32,
            )
        else:
            # Backward-compatible direct velocity mode.
            self.action_space = spaces.Box(
                low=np.array([0.0, -self.max_angular_speed], dtype=np.float32),
                high=np.array(
                    [self.max_linear_speed, self.max_angular_speed], dtype=np.float32
                ),
                dtype=np.float32,
            )

        self.waypoint_marker_pub = None
        if self.waypoint_marker_topic:
            self.waypoint_marker_pub = self.ros.create_publisher(
                MarkerArray,
                self.waypoint_marker_topic,
                10,
            )

        # v108: publish the exact pose actually used for confidence painting.
        # This separates two failure classes in RViz:
        #   A) origin marker != RobotModel  -> pose lookup/source problem
        #   B) origin marker == RobotModel, but magenta grid elsewhere -> grid/canvas problem
        self.confidence_origin_marker_topic = str(
            os.environ.get("TB3_RL_CONFIDENCE_ORIGIN_TOPIC", "/rl_confidence_origin") or ""
        ).strip()
        try:
            self.confidence_origin_marker_publish_every_n = max(
                int(os.environ.get("TB3_RL_CONFIDENCE_ORIGIN_PUBLISH_EVERY_N", "1")),
                1,
            )
        except Exception:
            self.confidence_origin_marker_publish_every_n = 1
        self.confidence_origin_marker_pub = None
        if self.confidence_origin_marker_topic:
            self.confidence_origin_marker_pub = self.ros.create_publisher(
                MarkerArray,
                self.confidence_origin_marker_topic,
                10,
            )
        self._last_confidence_origin_marker_step = -10_000_000

        self.waypoint_path_pub = None
        if self.waypoint_path_topic:
            self.waypoint_path_pub = self.ros.create_publisher(
                NavPath,
                self.waypoint_path_topic,
                10,
            )

        # v2 real-robot LiDAR diagnostics: publish the exact 360-bin vector seen
        # by the policy as a LaserScan. This is not fed back into SLAM.
        self.policy_scan_topic = str(os.environ.get("TB3_RL_POLICY_SCAN_TOPIC", "/rl_policy_scan") or "").strip()
        self.policy_scan_publish_every_n = max(
            int(os.environ.get("TB3_RL_POLICY_SCAN_PUBLISH_EVERY_N", "5")),
            1,
        )
        self.policy_scan_pub = None
        if self.policy_scan_topic:
            self.policy_scan_pub = self.ros.create_publisher(
                LaserScan,
                self.policy_scan_topic,
                10,
            )

        # v5: explicit 60-sector debug topic.  When --num-lidar-bins=60 this is
        # the exact vector slice fed to the policy, rendered as LaserScan in RViz.
        self.policy_scan_60_topic = str(
            os.environ.get("TB3_RL_POLICY_SCAN_60_TOPIC", "/rl_policy_scan_60") or ""
        ).strip()
        self.policy_scan_60_pub = None
        if self.policy_scan_60_topic:
            self.policy_scan_60_pub = self.ros.create_publisher(
                LaserScan,
                self.policy_scan_60_topic,
                10,
            )

        # v7 RViz alignment debug: publish policy/raw scan endpoints directly in
        # the map frame as MarkerArray.  LaserScan displays depend on RViz TF
        # timing and scan frame orientation; map-frame markers make it explicit
        # whether the mismatch is from TF/map alignment or LiDAR preprocessing.
        self.policy_scan_marker_topic = str(
            os.environ.get("TB3_RL_POLICY_SCAN_MARKER_TOPIC", "/rl_policy_scan_60_points") or ""
        ).strip()
        self.policy_scan_marker_pub = None
        if self.policy_scan_marker_topic:
            self.policy_scan_marker_pub = self.ros.create_publisher(
                MarkerArray,
                self.policy_scan_marker_topic,
                10,
            )

        self.raw_scan_marker_topic = str(
            os.environ.get("TB3_RL_RAW_SCAN_MARKER_TOPIC", "/rl_raw_scan_points") or ""
        ).strip()
        self.raw_scan_marker_pub = None
        if self.raw_scan_marker_topic:
            self.raw_scan_marker_pub = self.ros.create_publisher(
                MarkerArray,
                self.raw_scan_marker_topic,
                10,
            )

        # v12: anchor both confidence/priority rays and RViz scan markers at the
        # robot position on /map by default. This uses the same TF(map->base)
        # center as the robot-centric crop. It prevents debug rays and semantic
        # maps from appearing at the SLAM canvas center when scan-frame TF is
        # stale or evaluated at a different timestamp.
        self.use_base_pose_for_raycast = self._scan_bool_env("TB3_RL_USE_BASE_POSE_FOR_RAYCAST", True)
        self.use_base_pose_for_scan_markers = self._scan_bool_env("TB3_RL_USE_BASE_POSE_FOR_SCAN_MARKERS", True)

        # v101: subscribe to AMCL pose if requested.  The subscription is cheap
        # even when /amcl_pose is absent; no messages simply means the confidence
        # update will wait or use the configured fallback.
        if self.amcl_pose_topic:
            try:
                self._amcl_pose_sub = self.ros.create_subscription(
                    PoseWithCovarianceStamped,
                    self.amcl_pose_topic,
                    self._amcl_pose_callback,
                    10,
                )
                self.ros.get_logger().info(
                    "AMCL_POSE_CONFIDENCE_SUBSCRIBED | "
                    f"topic={self.amcl_pose_topic} max_age={self.amcl_pose_max_age_sec:.2f}s "
                    f"fallback={self.amcl_fallback_pose_source}"
                )
            except Exception as exc:
                self._amcl_pose_sub = None
                try:
                    self.ros.get_logger().warn(
                        f"AMCL_POSE_CONFIDENCE_SUBSCRIBE_FAILED | topic={self.amcl_pose_topic} err={exc}"
                    )
                except Exception:
                    pass

        # v102: publish /amcl_pose from the same TF used by RViz RobotModel.
        # This is deliberately enabled by the run script for SLAM-only training,
        # because launching full Nav2 AMCL together with Cartographer SLAM is not
        # the right architecture.  The produced topic is AMCL-compatible and is
        # consumed by the confidence pose source.
        if bool(getattr(self, "auto_publish_amcl_pose_from_tf", False)) and self.amcl_pose_topic:
            try:
                self._amcl_pose_tf_pub = self.ros.create_publisher(
                    PoseWithCovarianceStamped,
                    self.amcl_pose_topic,
                    10,
                )
                period = 1.0 / max(float(getattr(self, "amcl_pose_tf_publish_hz", 20.0)), 1e-3)
                self._amcl_pose_tf_timer = self.ros.create_timer(
                    period,
                    self._publish_amcl_pose_from_tf_timer,
                )
                self.ros.get_logger().info(
                    "AMCL_POSE_TF_BRIDGE_ACTIVE | "
                    f"topic={self.amcl_pose_topic} "
                    f"target={self.amcl_pose_tf_target_frame} "
                    f"source={self.amcl_pose_tf_source_frame} "
                    f"hz={float(getattr(self, 'amcl_pose_tf_publish_hz', 20.0)):.1f}"
                )
            except Exception as exc:
                self._amcl_pose_tf_pub = None
                self._amcl_pose_tf_timer = None
                try:
                    self.ros.get_logger().warn(f"AMCL_POSE_TF_BRIDGE_FAILED | err={exc}")
                except Exception:
                    pass

        self._last_scan_geometry_log_time = 0.0
        self._last_scan_geometry_debug = {}

        if self.debug_input_map and self.use_map_cnn:
            self._init_debug_input_map_publishers()

        if self.map_live_update_period_sec > 0.0:
            self._map_live_update_timer = self.ros.create_timer(
                self.map_live_update_period_sec,
                self._live_map_update_timer_callback,
            )
            self.ros.get_logger().warn(
                "LIVE_MAP_UPDATE_TIMER_ACTIVE | "
                f"period={self.map_live_update_period_sec:.2f}s | "
                "updates confidence/priority/RL maps from latest scan+pose independent of waypoint refresh"
            )

        if self.action_mode == "nav2":
            if ActionClient is None or NavigateToPose is None:
                raise RuntimeError(
                    "action_mode='nav2' requires nav2_msgs and rclpy.action. "
                    "Install/launch Nav2 before using this mode."
                )
            self.nav2_client = ActionClient(
                self.ros,
                NavigateToPose,
                self.nav2_action_name,
            )
            if FollowPath is not None:
                self.nav2_follow_path_client = ActionClient(
                    self.ros,
                    FollowPath,
                    self.nav2_follow_path_action_name,
                )
            if BackUp is not None:
                self.nav2_backup_client = ActionClient(
                    self.ros,
                    BackUp,
                    self.nav2_backup_action_name,
                )
            if Spin is not None:
                self.nav2_spin_client = ActionClient(
                    self.ros,
                    Spin,
                    self.nav2_spin_action_name,
                )
            if DriveOnHeading is not None:
                self.nav2_drive_on_heading_client = ActionClient(
                    self.ros,
                    DriveOnHeading,
                    self.nav2_drive_on_heading_action_name,
                )
            self.nav2_clear_global_client = self.ros.create_client(
                Empty,
                self.nav2_clear_global_service,
            )
            self.nav2_clear_local_client = self.ros.create_client(
                Empty,
                self.nav2_clear_local_service,
            )
            self._ensure_nav2_motion_server(timeout_sec=self.nav2_startup_timeout_sec)

        self.max_episode_steps = int(max_episode_steps)

        self.goal_threshold = float(goal_threshold)
        self.collision_threshold = float(collision_threshold)
        self.restart_on_collision = bool(restart_on_collision)
        self.collision_clear_nav2_costmaps = bool(collision_clear_nav2_costmaps)
        self.collision_cancel_nav2_goal = bool(collision_cancel_nav2_goal)
        self._last_terminal_reason = "none"
        self._last_collision_restart_requested = False

        self.fallen_roll_threshold = float(fallen_roll_threshold)
        self.fallen_pitch_threshold = float(fallen_pitch_threshold)

        self.prev_action = np.zeros(2, dtype=np.float32)
        self.prev_distance = 0.0

        self.goal_xy = np.array([1.5, 0.0], dtype=np.float32)
        self.step_count = 0
        self._last_step_reward = 0.0
        self._episode_reward_sum = 0.0
        self.reward_gamma = float(np.clip(float(reward_gamma), 0.0, 1.0))
        # v112 return/debug metrics.
        # _episode_discounted_return is now the live/recent discounted return:
        #   G_live[t] = gamma * G_live[t-1] + r_t
        # This keeps late-episode rewards visible.  The old start-anchored return
        # is kept separately for reference only:
        #   G_start[t] = sum_i gamma^i r_i
        self._episode_discounted_return = 0.0
        self._episode_start_discounted_return = 0.0
        self._episode_reward_ema = 0.0
        self._reward_ema_beta = float(np.clip(float(os.environ.get("TB3_RL_REWARD_EMA_BETA", str(self.reward_gamma))), 0.0, 0.999999))
        self._reward_window_n = max(int(os.environ.get("TB3_RL_REWARD_WINDOW_N", "100") or "100"), 1)
        self._recent_reward_window = deque(maxlen=self._reward_window_n)
        self._recent_slam_new_window = deque(maxlen=self._reward_window_n)
        self._recent_conf_update_window = deque(maxlen=self._reward_window_n)
        self._last_priority_clear_reward = 0.0
        self._last_priority_recheck_reward = 0.0
        self._last_priority_check_reward = 0.0
        self._episode_priority_clear_reward = 0.0
        self._episode_priority_recheck_reward = 0.0
        self._episode_priority_check_reward = 0.0
        self._last_slam_map_update_reward = 0.0
        self._last_slam_map_update_reward_raw = 0.0
        self._last_slam_map_update_reward_reason = "reset"
        self._episode_slam_map_update_reward = 0.0
        self._last_reward_text_valid = False
        self._live_priority_reward_collection_enabled = False
        self._reset_priority_event_accumulators()

        # Priority-stuck restart gate.
        # Do not restart when the active priority map is zero.  Count only while
        # a non-trivial priority remains but the robot fails to clear/recheck it
        # and also fails to acquire meaningful new information.
        self.priority_stuck_restart = bool(priority_stuck_restart) and (not bool(getattr(self, "disable_priority_map", False)))
        self.priority_stuck_restart_sec = max(float(priority_stuck_restart_sec), 0.0)
        self.priority_stuck_restart_steps = self._seconds_to_step_count(
            self.priority_stuck_restart_sec,
            priority_stuck_restart_steps,
        )
        self.priority_stuck_score_threshold = float(np.clip(float(priority_stuck_score_threshold), 0.0, 1.0))
        self.priority_stuck_clear_gain_threshold = max(float(priority_stuck_clear_gain_threshold), 0.0)
        self.priority_stuck_info_gain_threshold = max(float(priority_stuck_info_gain_threshold), 0.0)
        self.priority_stuck_restart_penalty = max(float(priority_stuck_restart_penalty), 0.0)
        self.priority_stuck_steps = 0
        self._last_priority_stuck_active = False
        self._last_priority_stuck_restart = False
        self._last_priority_stuck_reason = "none"
        self.lidar_empty_restart = bool(lidar_empty_restart)
        self.lidar_empty_timeout_sec = max(float(lidar_empty_timeout_sec), 0.0)
        self.lidar_empty_grace_sec = max(float(lidar_empty_grace_sec), 0.0)
        self.lidar_empty_min_valid_range_m = max(float(lidar_empty_min_valid_range_m), 0.0)
        self.lidar_empty_max_valid_range_m = max(float(lidar_empty_max_valid_range_m), self.lidar_empty_min_valid_range_m + 1e-3)
        self.lidar_empty_min_valid_beams = max(int(lidar_empty_min_valid_beams), 1)
        self.lidar_empty_restart_penalty = max(float(lidar_empty_restart_penalty), 0.0)
        self.lidar_empty_timeout_steps = self._seconds_to_step_count(self.lidar_empty_timeout_sec, 4)
        self.lidar_empty_steps = 0
        self._last_lidar_empty_active = False
        self._last_lidar_empty_restart = False
        self._last_lidar_empty_reason = "none"

        # v113: coverage-stall terminal.
        # This is not an extra shaping reward. It cuts off long, low-information
        # episode tails when both SLAM-map growth and confidence-map growth have
        # stalled for a full window after warmup.
        self.coverage_stall_terminal = bool(coverage_stall_terminal)
        self.coverage_stall_start_steps = max(int(coverage_stall_start_steps), 0)
        self.coverage_stall_window_steps = max(int(coverage_stall_window_steps), 1)
        self.coverage_stall_min_slam_new_cells = max(int(coverage_stall_min_slam_new_cells), 0)
        self.coverage_stall_min_confidence_updated_cells = max(int(coverage_stall_min_confidence_updated_cells), 0)
        self.coverage_stall_terminal_penalty = float(coverage_stall_terminal_penalty)
        self._coverage_stall_slam_window = deque(maxlen=self.coverage_stall_window_steps)
        self._coverage_stall_conf_window = deque(maxlen=self.coverage_stall_window_steps)
        self._last_coverage_stall_terminal = False
        self._last_coverage_stall_active = False
        self._last_coverage_stall_reason = "disabled"
        self._last_coverage_stall_slam_window = 0
        self._last_coverage_stall_conf_window = 0
        self._last_coverage_stall_window_len = 0

        # Direct velocity safety shield.  The policy action space remains forward-only
        # [0, max_linear_speed] x [-max_angular_speed, max_angular_speed], but the
        # shield may temporarily publish a negative linear velocity to escape an
        # imminent front collision.  A dense penalty is added in step(); actual
        # collision still terminates/resets exactly as before.
        self.velocity_safety_backup = bool(velocity_safety_backup)
        self.velocity_safety_trigger_distance_m = max(float(velocity_safety_trigger_distance_m), 0.05)
        self.velocity_safety_stop_distance_m = max(float(velocity_safety_stop_distance_m), self.velocity_safety_trigger_distance_m)
        self.velocity_safety_slow_distance_m = max(float(velocity_safety_slow_distance_m), self.velocity_safety_stop_distance_m)
        self.velocity_safety_backup_speed_mps = max(float(velocity_safety_backup_speed_mps), 0.0)
        self.velocity_safety_turn_speed = max(float(velocity_safety_turn_speed), 0.0)
        self.velocity_safety_backup_steps = max(int(velocity_safety_backup_steps), 1)
        self.velocity_safety_cooldown_steps_cfg = max(int(velocity_safety_cooldown_steps), 0)
        self.velocity_safety_penalty = max(float(velocity_safety_penalty), 0.0)
        self.velocity_safety_block_penalty = max(float(velocity_safety_block_penalty), 0.0)
        self.velocity_safety_slowdown = bool(velocity_safety_slowdown)
        self.velocity_safety_slow_min_scale = float(np.clip(float(velocity_safety_slow_min_scale), 0.02, 1.0))
        self.velocity_safety_slow_penalty = max(float(velocity_safety_slow_penalty), 0.0)
        self.velocity_safety_slow_speed_power = max(float(velocity_safety_slow_speed_power), 0.10)
        self.velocity_safety_slow_danger_power = max(float(velocity_safety_slow_danger_power), 0.10)
        self.velocity_safety_cooldown_steps = 0
        # Sticky reverse escape lock.  When the safety shield starts a backup,
        # the following Gym steps ignore the policy action and continue the
        # reverse escape until the configured backup step budget is consumed or
        # the rear sector becomes unsafe.  This prevents the next SAC action from
        # immediately canceling a backup maneuver.
        self.velocity_safety_backup_lock_steps = 0
        self.velocity_safety_backup_lock_turn_sign = 1.0
        self._last_velocity_safety_backup_lock_active = False
        self._last_velocity_safety_backup_triggered = False
        self._last_velocity_safety_backup_lock_active = False
        self._last_velocity_safety_blocked = False
        self._last_velocity_safety_skip_store = False
        self._pending_skip_penalty = 0.0
        self._last_velocity_safety_slowdown = 1.0
        self._last_velocity_safety_slowdown_risk = 0.0
        self._last_velocity_safety_policy_v = 0.0
        self._last_velocity_safety_executed_v = 0.0
        self._last_velocity_safety_penalty = 0.0
        self._last_velocity_safety_reason = "none"

        self.shake_restart = bool(shake_restart)
        self.shake_restart_steps_limit = max(int(shake_restart_steps), 1)
        self.shake_tilt_threshold = max(float(shake_tilt_threshold), 0.01)
        self.shake_angular_xy_threshold = max(float(shake_angular_xy_threshold), 0.01)
        self.shake_linear_z_threshold = max(float(shake_linear_z_threshold), 0.01)
        self.shake_z_deviation_threshold = max(float(shake_z_deviation_threshold), 0.005)
        self.shake_ground_min_z = float(shake_ground_min_z)
        self.shake_ground_max_z = float(shake_ground_max_z)
        if self.shake_ground_min_z > self.shake_ground_max_z:
            self.shake_ground_min_z, self.shake_ground_max_z = self.shake_ground_max_z, self.shake_ground_min_z
        self.shake_leaky_decay = bool(shake_leaky_decay)
        # v10: "shake" means physical body instability: the robot is tilted,
        # bouncing in z, or not planted on the floor.  Planar yaw oscillation is
        # no longer counted as shake by default; use the spin breaker for that.
        self.shake_yaw_wobble = bool(shake_yaw_wobble)
        self.shake_yaw_rate_threshold = max(float(shake_yaw_rate_threshold), 0.01)
        self.shake_cmd_flip_threshold = max(float(shake_cmd_flip_threshold), 0.01)
        self.shake_wobble_window_steps = max(int(shake_wobble_window_steps), 3)
        self.shake_wobble_min_flips = max(int(shake_wobble_min_flips), 1)
        self.shake_wobble_max_net_motion_m = max(float(shake_wobble_max_net_motion_m), 0.0)
        self.shake_spin_stall_restart_steps = max(int(shake_spin_stall_restart_steps), 0)
        self.shake_restart_penalty = max(float(shake_restart_penalty), 0.0)
        self.reset_hard_stabilize_reapply = bool(reset_hard_stabilize_reapply)
        self.reset_hard_stabilize_reapply_interval_sec = max(float(reset_hard_stabilize_reapply_interval_sec), 0.05)
        self.shake_steps = 0
        self._last_shake_active = False
        self._last_shake_restart = False
        self._last_shake_reason = "none"
        self._shake_wobble_history = deque(maxlen=int(self.shake_wobble_window_steps))
        self._shake_last_wobble_reason = "none"
        self._reset_nominal_z = float(reset_z)

        self.nav2_stuck_steps = 0
        self.nav2_backup_cooldown_steps = 0
        self._last_nav2_stuck_active = False
        self._last_nav2_stuck_backup_triggered = False
        self._last_nav2_stuck_reason = "none"
        self._last_nav2_backup_status = "none"

        self.nav2_stuck_backup = bool(nav2_stuck_backup)
        self.nav2_stuck_backup_sec = max(float(nav2_stuck_backup_sec), 0.0)
        self.nav2_stuck_backup_steps = self._seconds_to_step_count(
            self.nav2_stuck_backup_sec,
            nav2_stuck_backup_steps,
        )
        self.nav2_stuck_backup_min_movement_m = max(float(nav2_stuck_backup_min_movement_m), 0.0)
        self.nav2_stuck_backup_stationary_sec = max(float(nav2_stuck_backup_stationary_sec), 0.1)
        self.nav2_stuck_backup_stationary_xy_m = max(float(nav2_stuck_backup_stationary_xy_m), 0.0)
        self.nav2_stuck_backup_stationary_yaw_rad = math.radians(max(float(nav2_stuck_backup_stationary_yaw_deg), 0.0))
        self._nav2_stationary_samples = deque(maxlen=240)
        self._last_nav2_stationary_gate_reason = "none"
        self.nav2_stuck_backup_distance_m = max(float(nav2_stuck_backup_distance_m), 0.01)
        self.nav2_stuck_backup_speed_mps = max(float(nav2_stuck_backup_speed_mps), 0.01)
        self.nav2_stuck_backup_timeout_sec = max(float(nav2_stuck_backup_timeout_sec), 0.1)
        self.nav2_stuck_backup_cooldown_sec = max(float(nav2_stuck_backup_cooldown_sec), 0.0)
        self.nav2_stuck_backup_cooldown_steps_limit = self._seconds_to_step_count(
            self.nav2_stuck_backup_cooldown_sec,
            3,
        )
        self.nav2_stuck_backup_penalty = max(float(nav2_stuck_backup_penalty), 0.0)
        self.nav2_stuck_steps = 0
        self.nav2_backup_cooldown_steps = 0
        self._last_nav2_stuck_active = False
        self._last_nav2_stuck_backup_triggered = False
        self._last_nav2_stuck_reason = "none"
        self._last_nav2_backup_status = "none"
        self._nav2_stationary_samples.clear()
        self._last_nav2_stationary_gate_reason = "none"
        self._last_lidar_valid_beams = 0
        self._last_lidar_nearest_detection = 999.0
        self._episode_start_wall_time = time.time()
        self._episode_start_sim_time = self._safe_sim_time()

        # Episode reset 위치 후보.
        # fixed      : 항상 --reset-x/--reset-y
        # corners    : reset_x/reset_y 주변 4점
        # house_random: turtlebot3_house용 사전 후보 중 랜덤
        # list       : --reset-pose-list "x,y;x,y;..."에서 랜덤
        self.reset_pose_candidates = self._build_reset_pose_candidates()

        self.goal_candidates = [
            np.array([1.5, 0.0], dtype=np.float32),
            np.array([1.2, 0.8], dtype=np.float32),
            np.array([1.2, -0.8], dtype=np.float32),
            np.array([2.0, 0.5], dtype=np.float32),
            np.array([2.0, -0.5], dtype=np.float32),
        ]

        self.ros.get_logger().info(
            "ENV_CONFIG | "
            f"mode={self.action_mode} | frames map={self.map_frame} pose={self.pose_frame} safety={self.safety_boundary_frame} | "
            f"obs={self.obs_dim} map_cnn={self.use_map_cnn} map=({self.map_channels},{self.map_obs_size},{self.map_obs_size}) | "
            f"temporal={self.use_temporal_cnn} hist={self.temporal_history_len} | "
            f"reset_mode={self.reset_pose_mode} candidates={len(self.reset_pose_candidates)} clearance={self.reset_pose_min_clearance_m:.2f}m | "
            f"priority_stuck={self.priority_stuck_restart}:{self.priority_stuck_restart_sec:.1f}s/{self.priority_stuck_restart_steps}steps | "
            f"lidar_empty={self.lidar_empty_restart}:{self.lidar_empty_timeout_sec:.1f}s/{self.lidar_empty_timeout_steps}steps | "
            f"stuck_backup={self.nav2_stuck_backup}:stationary>={self.nav2_stuck_backup_stationary_sec:.1f}s/{self.nav2_stuck_backup_steps}steps "
            f"xy<{self.nav2_stuck_backup_stationary_xy_m:.3f}m yaw<{math.degrees(self.nav2_stuck_backup_stationary_yaw_rad):.1f}deg dist={self.nav2_stuck_backup_distance_m:.2f}m | "
            f"nav2={self.nav2_action_name if self.action_mode == 'nav2' else '(off)'} window={self.nav2_control_window_sec:.2f}s tol={self.nav2_goal_reached_tolerance:.2f}m | "
            f"topics priority={self.rl_priority_topic or '(off)'} filtered_slam={self.rl_filtered_slam_topic or '(off)'} marker={self.waypoint_marker_topic or '(off)'}"
        )
        self.ros.get_logger().debug(
            "ENV_CONFIG_VERBOSE | "
            f"front_fov_deg={self.front_fov_deg:.1f}, confidence_max_range={self.confidence_max_range:.2f}, "
            f"seen_confidence_floor={self.seen_confidence_floor:.1f}, gap_width=[{self.gap_min_width_m:.2f},{self.gap_max_width_m:.2f}]m, "
            f"map_expand_chunk_cells={self.map_expand_chunk_cells}, max_planned_candidates={self.max_planned_candidates}, "
            f"max_alternative_paths={self.max_alternative_paths}, priority_clear_fov={self.priority_clear_fov_deg:.1f}deg, "
            f"priority_clear_range={self.priority_clear_max_range_m:.2f}m, priority_clear_sigma={self.priority_clear_sigma_m:.2f}m, "
            f"wall_support_radius={self.wall_support_radius_m:.2f}m, open_space_front_distance={self.open_space_front_distance_m:.2f}m, "
            f"waypoint_distance=[{self.waypoint_min_distance:.2f},{self.waypoint_max_distance:.2f}]m, "
            f"nav2_goal_timeout={self.nav2_goal_timeout_sec:.2f}s, nav2_replan_distance={self.nav2_replan_distance_m:.2f}m, "
            f"safety_radius={self.safety_boundary_radius_m:.2f}m, safety_xy=[{self.safety_boundary_min_x:.1f},{self.safety_boundary_max_x:.1f}]x[{self.safety_boundary_min_y:.1f},{self.safety_boundary_max_y:.1f}], "
            f"reset_candidates={self.reset_pose_candidates}, rviz_zero_robot_on_reset={self.rviz_zero_robot_on_reset}, "
            f"reset_slam_on_reset={self.reset_slam_on_reset}, post_reset_stabilize={self.post_reset_stabilize_sec:.2f}s, "
            f"strict_slam_map_required={self.strict_slam_map_required}:{self.strict_slam_map_wait_timeout_sec:.1f}s, "
            f"reset_tf_buffer_on_reset={self.reset_tf_buffer_on_reset}"
        )
        if self.action_mode == "nav2":
            self.ros.get_logger().info(
                "NAV2_STRICT_RESTORED | action_mode=nav2 | "
                "collision/fallen/drop reward=-100 | "
                f"reset_slam_on_reset={self.reset_slam_on_reset} | "
                f"post_reset_stabilize={self.post_reset_stabilize_sec:.2f}s | "
                f"out_of_bounds_terminal={self.terminate_on_out_of_bounds}"
            )
            self.ros.get_logger().info(
                "NAV2_MACRO_GOAL_POLICY | "
                f"reached_tol={self.nav2_goal_reached_tolerance:.2f}m | "
                f"timeout={self.nav2_goal_timeout_sec:.2f}s | "
                f"cancel_on_reached={self.nav2_cancel_on_reached} | "
                f"cancel_on_timeout={self.nav2_cancel_on_timeout} | "
                f"continuous_update={self.nav2_continuous_goal_update} | "
                f"use_goal_orientation={self.nav2_use_goal_orientation}"
            )
            self.ros.get_logger().info(
                "UNSAFE_TERMINAL_RESET_RESTORED | "
                f"collision_threshold={self.collision_threshold:.3f}m | "
                f"fallen_roll={self.fallen_roll_threshold:.3f}rad | "
                f"fallen_pitch={self.fallen_pitch_threshold:.3f}rad | "
                f"drop_abs_z={self.safety_boundary_max_abs_z:.3f}m | "
                "collision/fallen/drop -> stop + cancel_nav2 + clear_costmap + terminated + reward=-100"
            )
        if self.pose_frame == self.map_frame:
            self.ros.get_logger().warn(
                "POSE_FRAME_ALIGNED | RViz Fixed Frame should match pose_frame | "
                "SLAM prior is transformed into pose_frame before RL map/priority update"
            )
        else:
            self.ros.get_logger().warn(
                "ODOM_OR_POSE_FRAME_UNIFIED | RViz Fixed Frame should be pose_frame | "
                "SLAM /map origin is transformed before /rl_priority_map publication"
            )

    @staticmethod
    def _parse_reset_pose_list(reset_pose_list: str) -> list[tuple[float, float]]:
        """Parse "x,y;x,y;..." reset candidates from CLI."""
        text = str(reset_pose_list or "").strip()
        if not text:
            return []

        candidates: list[tuple[float, float]] = []
        for raw_item in text.replace("|", ";").split(";"):
            item = raw_item.strip()
            if not item:
                continue
            parts = [p.strip() for p in item.replace(":", ",").split(",") if p.strip()]
            if len(parts) < 2:
                raise ValueError(
                    "Each reset pose candidate must have at least x,y. "
                    f"Bad item: {raw_item!r}"
                )
            try:
                x = float(parts[0])
                y = float(parts[1])
            except ValueError as exc:
                raise ValueError(f"Invalid reset pose candidate: {raw_item!r}") from exc
            candidates.append((x, y))

        if not candidates:
            raise ValueError("--reset-pose-list was provided but no valid candidates were parsed")
        return candidates

    def _build_reset_pose_candidates(self) -> list[tuple[float, float]]:
        """Build Gazebo/world-frame reset candidates."""
        mode = self.reset_pose_mode
        if mode == "fixed":
            return [(self.reset_x, self.reset_y)]

        if mode == "corners":
            ox = self.reset_offset
            oy = self.reset_offset
            return [
                (self.reset_x + ox, self.reset_y + oy),
                (self.reset_x + ox, self.reset_y - oy),
                (self.reset_x - ox, self.reset_y + oy),
                (self.reset_x - ox, self.reset_y - oy),
            ]

        if mode == "house_random":
            return list(self.DEFAULT_HOUSE_RESET_CANDIDATES)

        if mode == "house_inside_random":
            return list(self.HOUSE_INSIDE_RESET_CANDIDATES)

        if mode == "list":
            return self._parse_reset_pose_list(self.reset_pose_list)

        raise ValueError(f"Unsupported reset_pose_mode={mode!r}")

    def _wait_for_rviz_map_origin_after_reset(self) -> bool:
        """Verify that map-frame base pose is near (0, 0) after SLAM reset."""
        if not self.rviz_zero_robot_on_reset:
            return True
        if self.pose_frame != self.map_frame:
            self.ros.get_logger().warn(
                "rviz_zero_robot_on_reset=True but pose_frame is not map. "
                f"pose_frame={self.pose_frame}, map_frame={self.map_frame}"
            )

        deadline = time.time() + self.rviz_origin_wait_sec
        last_xy = None
        while rclpy.ok() and time.time() < deadline:
            pose = self.ros.get_pose2d(frame_id=self.map_frame)
            if pose is not None:
                xy, yaw = pose
                last_xy = xy
                dist = float(np.linalg.norm(xy))
                if dist <= self.rviz_origin_tolerance_m:
                    self.ros.get_logger().info(
                        "RViz/map reset origin verified: "
                        f"map_base=({xy[0]:+.3f}, {xy[1]:+.3f}), "
                        f"yaw={yaw:+.3f}, dist={dist:.3f}m"
                    )
                    return True
            self.ros.spin_steps(num_spins=4, timeout_sec=0.005)
            time.sleep(0.01)

        if last_xy is None:
            self.ros.get_logger().warn(
                "RViz/map reset origin could not be verified: no map-frame pose available. "
                "Check slam_toolbox and TF map->odom."
            )
        else:
            dist = float(np.linalg.norm(last_xy))
            self.ros.get_logger().warn(
                "RViz/map reset origin not yet near zero: "
                f"map_base=({last_xy[0]:+.3f}, {last_xy[1]:+.3f}), "
                f"dist={dist:.3f}m > tolerance={self.rviz_origin_tolerance_m:.3f}m"
            )
        return False

    def _wait_for_fresh_reset_sensors(self, wait_sec: Optional[float] = None) -> None:
        """Let Gazebo/ROS publish fresh odom/scan after teleport before validation or motion."""
        deadline = time.time() + (self.reset_pose_validation_wait_sec if wait_sec is None else max(float(wait_sec), 0.0))
        # Publish zero velocity during the settle window so the controller cannot move
        # while scan/odom/TF are still reflecting the pre-teleport state.
        self.ros.stop_robot()
        while rclpy.ok() and time.time() < deadline:
            self.ros.spin_steps(num_spins=4, timeout_sec=0.005)
            if self.use_world_step:
                self._advance_world_after_command(target_delta_sec=min(self.control_dt, 0.03))
            else:
                time.sleep(0.01)

    def _reset_pose_scan_clearance(self) -> tuple[float, float, float]:
        """Return (global_min, front_min, rear_min) LiDAR clearance after reset."""
        if self.ros.scan is None:
            return 999.0, 999.0, 999.0
        ranges = np.asarray(self.ros.scan.ranges, dtype=np.float32)
        ranges = np.nan_to_num(ranges, nan=999.0, posinf=999.0, neginf=0.0)
        range_min = float(getattr(self.ros.scan, "range_min", 0.05))
        ranges = ranges[np.isfinite(ranges)]
        if ranges.size == 0:
            global_min = 999.0
        else:
            ranges = ranges[ranges >= max(0.0, range_min * 0.5)]
            global_min = float(np.min(ranges)) if ranges.size else 999.0
        front_min = self._scan_min_distance_in_sector(
            scan=self.ros.scan,
            center_angle=0.0,
            half_width_rad=math.radians(35.0),
            max_considered_range=1.00,
        )
        rear_min = self._scan_min_distance_in_sector(
            scan=self.ros.scan,
            center_angle=math.pi,
            half_width_rad=math.radians(35.0),
            max_considered_range=1.00,
        )
        return float(global_min), float(front_min), float(rear_min)

    def _is_reset_pose_clear(self) -> tuple[bool, str, float, float, float]:
        global_min, front_min, rear_min = self._reset_pose_scan_clearance()

        # TurtleBot3 house has several valid start poses where a side wall or
        # furniture leg appears at ~0.18-0.22 m in the 360-degree LiDAR minimum.
        # Treating the global minimum as a hard front-clearance test rejects
        # valid spawns repeatedly.  Use a two-level test instead:
        #   - side/global clearance: only reject true overlaps / near-contact
        #   - front clearance      : keep stricter because Nav2 starts moving there
        side_threshold = min(float(self.reset_pose_min_clearance_m), 0.10)
        front_rear_threshold = float(self.reset_pose_min_clearance_m)

        if global_min < side_threshold:
            return False, "scan_side_clearance_too_small", global_min, front_min, rear_min
        if front_min < front_rear_threshold:
            return False, "scan_front_clearance_too_small", global_min, front_min, rear_min
        if rear_min < side_threshold:
            return False, "scan_rear_clearance_too_small", global_min, front_min, rear_min
        if self._check_fallen():
            return False, "fallen_after_reset", global_min, front_min, rear_min
        return True, "clear", global_min, front_min, rear_min

    def _select_reset_candidate_order(self) -> list[tuple[float, float]]:
        candidates = list(self.reset_pose_candidates)
        if not candidates:
            candidates = [(self.reset_x, self.reset_y)]
        random.shuffle(candidates)
        # If max_attempts exceeds the candidate count, repeat the shuffled list.
        attempts = []
        while len(attempts) < self.reset_pose_max_attempts:
            attempts.extend(candidates)
        return attempts[: self.reset_pose_max_attempts]

    def _post_reset_stabilize(self, reset_pose=None) -> None:
        """Hold the robot still after pose/SLAM reset before returning first observation.

        v48 safety fix:
          - The previous second flat-reset loop could extend its own deadline by
            resetting ``second_start`` whenever a benign z-deviation was reported.
            With TurtleBot3 Burger this can keep reset() at step=0 forever, so
            the SAC loop never publishes a velocity command or RViz overlay.
          - This implementation uses absolute deadlines and a bounded reapply
            count.  A reset can delay the episode, but it cannot permanently
            block motion.
        """
        sec = float(getattr(self, "post_reset_stabilize_sec", 0.0))
        if sec <= 0.0:
            return

        try:
            stable_window_sec = float(os.environ.get("TB3_RL_POST_RESET_STABLE_WINDOW_SEC", "0.35") or 0.35)
        except Exception:
            stable_window_sec = 0.35
        stable_window_sec = float(np.clip(stable_window_sec, 0.10, 1.00))

        try:
            max_sec = float(os.environ.get("TB3_RL_POST_RESET_MAX_STABILIZE_SEC", str(max(sec + 2.0, 3.0))) or max(sec + 2.0, 3.0))
        except Exception:
            max_sec = max(sec + 2.0, 3.0)
        max_sec = max(float(max_sec), min(sec + 2.0, 3.0))

        try:
            max_reapply = int(os.environ.get("TB3_RL_POST_RESET_MAX_REAPPLY", "4") or 4)
        except Exception:
            max_reapply = 4
        max_reapply = max(int(max_reapply), 0)

        self.ros.get_logger().info(
            f"POST_RESET_STABILIZE | holding /cmd_vel=0 for {sec:.2f}s before episode start "
            f"| stable_window={stable_window_sec:.2f}s max={max_sec:.2f}s max_reapply={max_reapply}"
        )

        start = time.time()
        deadline = start + max_sec
        next_reapply = start
        reapply_count = 0
        stable_since = None
        last_reason = "none"
        ever_unstable = False

        while time.time() < deadline:
            self.ros.stop_robot()
            now = time.time()
            if (
                bool(getattr(self, "reset_hard_stabilize_reapply", True))
                and reset_pose is not None
                and self.reset_manager is not None
                and reapply_count < max_reapply
                and now >= next_reapply
            ):
                try:
                    self.reset_manager.reset_to_pose(reset_pose, timeout_sec=0.35)
                    reapply_count += 1
                except Exception:
                    pass
                try:
                    interval = float(getattr(self, "reset_hard_stabilize_reapply_interval_sec", 0.25))
                except Exception:
                    interval = 0.25
                next_reapply = time.time() + float(np.clip(interval, 0.18, 0.75))

            self.ros.spin_steps(
                num_spins=int(getattr(self, "post_reset_stabilize_spin_steps", 12)),
                timeout_sec=0.002,
            )
            reason = self._instantaneous_shake_reason()
            last_reason = reason
            if reason == "none":
                if stable_since is None:
                    stable_since = time.time()
                if (time.time() - start) >= sec and (time.time() - stable_since) >= stable_window_sec:
                    break
            else:
                ever_unstable = True
                stable_since = None
            time.sleep(0.03)

        # Bounded second flat reset.  This is allowed to improve pose settling,
        # but it must never keep the environment at step=0 forever.
        if (
            bool(getattr(self, "reset_hard_stabilize_reapply", True))
            and reset_pose is not None
            and self.reset_manager is not None
            and ever_unstable
            and max_reapply > 0
        ):
            self.ros.get_logger().warn(
                f"POST_RESET_SECOND_FLAT_RESET | first_stabilize_reason={last_reason}; "
                "bounded re-apply; reset will not block episode start indefinitely"
            )
            try:
                self.ros.stop_robot()
                self.reset_manager.reset_to_pose(reset_pose, timeout_sec=0.60)
            except Exception:
                pass

            second_deadline = time.time() + max(0.80, min(1.60, sec))
            second_stable_since = None
            second_reapply_count = 0
            second_next_reapply = time.time() + 0.35
            while time.time() < second_deadline:
                self.ros.stop_robot()
                self.ros.spin_steps(
                    num_spins=int(getattr(self, "post_reset_stabilize_spin_steps", 12)),
                    timeout_sec=0.002,
                )
                reason2 = self._instantaneous_shake_reason()
                last_reason = reason2
                if reason2 == "none":
                    if second_stable_since is None:
                        second_stable_since = time.time()
                    if time.time() - second_stable_since >= 0.30:
                        break
                else:
                    second_stable_since = None
                    if second_reapply_count < 2 and time.time() >= second_next_reapply:
                        try:
                            self.reset_manager.reset_to_pose(reset_pose, timeout_sec=0.35)
                            second_reapply_count += 1
                        except Exception:
                            pass
                        second_next_reapply = time.time() + 0.45
                time.sleep(0.03)

            if last_reason != "none":
                self.ros.get_logger().warn(
                    "POST_RESET_STABILIZE_SOFT_CONTINUE | "
                    f"shake={last_reason}; starting episode anyway to avoid reset deadlock"
                )

        # One final quiet hold after the last pose reapply.
        final_hold_start = time.time()
        while time.time() - final_hold_start < 0.15:
            self.ros.stop_robot()
            self.ros.spin_steps(num_spins=8, timeout_sec=0.002)
            time.sleep(0.02)

        try:
            reason = self._instantaneous_shake_reason()
            if reason == "none":
                self.shake_steps = 0
                self._last_shake_active = False
                self._last_shake_restart = False
                self._last_shake_reason = "none"
            else:
                # Keep diagnostic state, but do not block reset() here.  If the
                # instability is real, _update_shake_restart_state() can still end
                # the episode after actual control steps.
                self.shake_steps = max(int(getattr(self, "shake_steps", 0)), 1)
                self._last_shake_active = True
                self._last_shake_reason = reason
                last_reason = reason
        except Exception:
            pass
        self.ros.get_logger().info(
            "POST_RESET_STABILIZE_DONE | "
            f"scan={self.ros.scan is not None}, odom={self.ros.odom is not None}, "
            f"slam_map={self.ros.slam_map is not None}, shake={last_reason}"
        )

    def _post_reset_ready_metrics(self, map_stats: Optional[MapUpdateStats] = None) -> dict:
        """Return readiness metrics for the first trainable observation after reset.

        Reset must not return an observation while SLAM just emitted an empty or
        old /map.  The policy should see a real local map, valid LiDAR and, in
        the simplified priority setup, at least one active random priority target
        whenever possible.
        """
        metrics = {
            "ready": False,
            "reason": "unknown",
            "inside": False,
            "known_ratio": 0.0,
            "known_cells": 0,
            "lidar_beams": 0,
            "priority_score": 0.0,
            "confidence_cells": 0,
            "confidence_updated_cells": 0,
            "slam_gate": str(getattr(self, "_last_slam_gate_reason", "unknown")),
        }
        try:
            beams, nearest = self._lidar_detection_stats()
            metrics["lidar_beams"] = int(beams)
            self._last_lidar_valid_beams = int(beams)
            self._last_lidar_nearest_detection = float(nearest)
        except Exception:
            metrics["lidar_beams"] = 0

        try:
            pose = self._get_robot_pose2d()
            if pose is None or self.exploration_map is None:
                metrics["reason"] = "missing_pose_or_map"
                return metrics
            robot_xy, _ = pose
            emap = self.exploration_map
            rix, riy = emap.world_to_map(float(robot_xy[0]), float(robot_xy[1]))
            inside = bool(emap.in_bounds(int(rix), int(riy)))
            metrics["inside"] = inside
            if not inside:
                metrics["reason"] = "pose_outside_rl_map"
                return metrics

            radius_cells = max(2, int(math.ceil(1.40 / max(float(emap.resolution), 1e-6))))
            x0 = max(0, int(rix) - radius_cells)
            x1 = min(int(emap.width), int(rix) + radius_cells + 1)
            y0 = max(0, int(riy) - radius_cells)
            y1 = min(int(emap.height), int(riy) + radius_cells + 1)
            local = np.asarray(emap.base_grid[y0:y1, x0:x1]) if x1 > x0 and y1 > y0 else np.empty((0, 0), dtype=np.int16)
            if local.size > 0:
                known = int(np.count_nonzero(local >= 0))
                ratio = float(known) / float(local.size)
            else:
                known = 0
                ratio = 0.0
            # Confidence is a stronger indication that the current LiDAR was
            # integrated after reset.  Use it as a fallback when SLAM /map is
            # sparse but the local ray update is already valid.
            conf_local = np.asarray(emap.confidence_grid[y0:y1, x0:x1]) if x1 > x0 and y1 > y0 else np.empty((0, 0), dtype=np.float32)
            conf_known = int(np.count_nonzero(conf_local >= max(float(emap.min_known_confidence), 1.0))) if conf_local.size else 0
            known_eff = max(known, conf_known)
            ratio_eff = max(ratio, float(conf_known) / float(conf_local.size) if conf_local.size else 0.0)
            metrics["known_cells"] = int(known_eff)
            metrics["known_ratio"] = float(ratio_eff)
            metrics["confidence_cells"] = int(conf_known)
            try:
                metrics["confidence_updated_cells"] = int(getattr(map_stats, "confidence_updated_cells", 0) if map_stats is not None else 0)
            except Exception:
                metrics["confidence_updated_cells"] = 0

            try:
                metrics["priority_score"] = float(emap.priority_score())
            except Exception:
                metrics["priority_score"] = float(getattr(map_stats, "priority_score", 0.0) if map_stats is not None else 0.0)
        except Exception as exc:
            metrics["reason"] = f"metrics_error:{type(exc).__name__}"
            return metrics

        min_beams = int(getattr(self, "post_reset_ready_min_lidar_beams", 0))
        min_cells = int(getattr(self, "post_reset_ready_min_known_cells", 0))
        min_ratio = float(getattr(self, "post_reset_ready_min_known_ratio", 0.0))
        require_prio = bool(getattr(self, "post_reset_ready_require_priority", True))
        try:
            min_conf_cells = int(os.environ.get("TB3_RL_POST_RESET_READY_MIN_CONFIDENCE_CELLS", "0") or 0)
        except Exception:
            min_conf_cells = 0

        if metrics["lidar_beams"] < min_beams:
            metrics["reason"] = f"lidar_warmup:{metrics['lidar_beams']}/{min_beams}"
            return metrics
        if metrics["known_cells"] < min_cells and metrics["known_ratio"] < min_ratio:
            metrics["reason"] = f"map_warmup:known={metrics['known_cells']},ratio={metrics['known_ratio']:.3f}"
            return metrics
        if min_conf_cells > 0 and int(metrics.get("confidence_cells", 0)) < min_conf_cells:
            metrics["reason"] = f"confidence_warmup:{int(metrics.get('confidence_cells', 0))}/{min_conf_cells}"
            return metrics
        if require_prio and float(metrics["priority_score"]) <= 0.01:
            metrics["reason"] = "priority_warmup"
            return metrics

        metrics["ready"] = True
        metrics["reason"] = "ready"
        return metrics

    def _slam_map_known_stats(self, slam_map) -> tuple[int, float, int]:
        """Return (known_cells, known_ratio, total_cells) for an OccupancyGrid."""
        if slam_map is None:
            return 0, 0.0, 0
        try:
            data = np.asarray(slam_map.data, dtype=np.int16)
            total = int(data.size)
            known = int(np.count_nonzero(data >= 0))
            ratio = float(known) / float(max(total, 1))
            return known, ratio, total
        except Exception:
            return 0, 0.0, 0

    def _strict_wait_for_accepted_slam_map(self, stage: str = "reset"):
        """Block until a post-reset SLAM map is actually accepted by the RL map gate.

        In evaluation on the real robot, falling back to LiDAR-only maps changes
        the observation distribution relative to training.  When strict mode is
        enabled, reset() must not return until _filtered_slam_map_for_update()
        returns a map that passed: reset wall-time barrier, accept delay, age gate,
        frame transform, and minimum known-cell threshold.
        """
        if not bool(getattr(self, "strict_slam_map_required", False)):
            return None
        if not bool(getattr(self, "use_slam_map", True)):
            raise RuntimeError("strict_slam_map_required=True but use_slam_map=False")

        timeout_sec = max(float(getattr(self, "strict_slam_map_wait_timeout_sec", 30.0)), 0.5)
        retry_interval = max(float(getattr(self, "strict_slam_map_retry_interval_sec", 0.50)), 0.10)
        min_cells = max(int(getattr(self, "strict_slam_map_min_known_cells", 20)), 0)
        min_ratio = max(float(getattr(self, "strict_slam_map_min_known_ratio", 0.001)), 0.0)

        start = time.time()
        last_service_try = 0.0
        last_log = 0.0
        last_reason = "init"
        self.ros.get_logger().warn(
            "STRICT_SLAM_MAP_WAIT_START | "
            f"stage={stage} timeout={timeout_sec:.1f}s min_known_cells={min_cells} "
            f"min_known_ratio={min_ratio:.4f} topic={getattr(self.ros, 'map_topic', '')}"
        )

        while time.time() - start < timeout_sec:
            self.ros.stop_robot()
            try:
                self.ros.spin_steps(num_spins=max(int(getattr(self, "post_reset_stabilize_spin_steps", 12)), 8), timeout_sec=0.002)
            except Exception:
                pass

            now = time.time()
            if now - last_service_try >= retry_interval:
                last_service_try = now
                try:
                    fetch = getattr(self.ros, "_try_fetch_slam_map_service", None)
                    if callable(fetch):
                        fetch(timeout_sec=min(0.90, max(0.25, retry_interval)), reason=f"strict_{stage}")
                except Exception as exc:
                    if now - last_log > 2.0:
                        self.ros.get_logger().warn(
                            f"STRICT_SLAM_MAP_SERVICE_TRY_ERROR | stage={stage} {type(exc).__name__}: {exc}"
                        )

            accepted = self._filtered_slam_map_for_update()
            if accepted is not None:
                known, ratio, total = self._slam_map_known_stats(getattr(self.ros, "slam_map", None))
                if known >= min_cells or ratio >= min_ratio:
                    self._strict_slam_map_ready_count += 1
                    self.ignore_slam_prior_this_episode = False
                    if hasattr(self.exploration_map, "set_slam_publish_reference"):
                        self.exploration_map.set_slam_publish_reference(getattr(self.ros, "slam_map", None))
                    self.ros.get_logger().warn(
                        "STRICT_SLAM_MAP_READY | "
                        f"stage={stage} dt={time.time() - start:.2f}s gate={self._last_slam_gate_reason} "
                        f"known={known}/{total} ratio={ratio:.4f} ready_count={self._strict_slam_map_ready_count}"
                    )
                    return accepted
                last_reason = f"known_warmup:{known}/{min_cells},ratio={ratio:.4f}/{min_ratio:.4f}"
            else:
                last_reason = str(getattr(self, "_last_slam_gate_reason", "missing"))

            if now - last_log >= 1.0:
                last_log = now
                if not _quiet_reset_logs():
                    raw = getattr(self.ros, "slam_map", None)
                    known, ratio, total = self._slam_map_known_stats(raw)
                    last_time = getattr(self.ros, "last_slam_map_time", None)
                    age = (now - float(last_time)) if last_time is not None else -1.0
                    self.ros.get_logger().warn(
                        "STRICT_SLAM_MAP_WAITING | "
                        f"stage={stage} dt={now - start:.1f}/{timeout_sec:.1f}s gate={last_reason} "
                        f"raw_map={raw is not None} age={age:.2f}s known={known}/{total} ratio={ratio:.4f} "
                        f"delay_left={float(getattr(self, '_last_slam_map_delay_remaining_sec', 0.0)):.2f}s"
                    )
            time.sleep(0.03)

        self.ros.stop_robot()
        msg = (
            "STRICT_SLAM_MAP_TIMEOUT | "
            f"stage={stage} timeout={timeout_sec:.1f}s last_gate={last_reason}. "
            "Policy step is blocked because fallback LiDAR-only maps would corrupt the trained observation."
        )
        self.ros.get_logger().error(msg)
        raise RecoverableResetError(msg)

    def _wait_post_reset_ready(self, reset_pose=None) -> MapUpdateStats:
        """Block reset() until the first observation is trainable.

        During this gate no SAC action is requested, no reward is computed, and
        no transition can enter the replay buffer because Gym reset() has not
        returned yet.  /cmd_vel is held at zero while callbacks, SLAM, LiDAR and
        priority generation warm up.
        """
        if not bool(getattr(self, "post_reset_ready_gate", True)):
            if bool(getattr(self, "strict_slam_map_required", False)):
                # Strict mode overrides the disabled warmup gate: reset() may not
                # return a policy observation until the SLAM map path is valid.
                accepted = self._strict_wait_for_accepted_slam_map(stage="post_reset_ready_disabled")
                self._last_post_reset_ready = True
                self._last_post_reset_ready_reason = "strict_slam_ready"
                return self._update_exploration_map_with_unified_tf(slam_map=accepted)
            self._last_post_reset_ready = True
            self._last_post_reset_ready_reason = "disabled"
            return self._update_exploration_map()

        timeout_sec = max(float(getattr(self, "post_reset_ready_timeout_sec", 7.0)), 0.1)
        start = time.time()
        last_log = 0.0
        last_stats = self._empty_map_stats()
        best_metrics = None
        soft_ready_stats = None
        soft_ready_metrics = None

        self.ros.get_logger().info(
            "POST_RESET_READY_WAIT | holding /cmd_vel=0 until SLAM/LiDAR/map/priority are ready "
            f"| timeout={timeout_sec:.2f}s | min_known_ratio={float(getattr(self, 'post_reset_ready_min_known_ratio', 0.0)):.3f} "
            f"min_known_cells={int(getattr(self, 'post_reset_ready_min_known_cells', 0))} "
            f"min_lidar_beams={int(getattr(self, 'post_reset_ready_min_lidar_beams', 0))} "
            f"require_priority={bool(getattr(self, 'post_reset_ready_require_priority', True))}"
        )

        while time.time() - start < timeout_sec:
            self.ros.stop_robot()
            self.ros.spin_steps(num_spins=max(int(getattr(self, "post_reset_stabilize_spin_steps", 12)), 8), timeout_sec=0.002)

            # Prefer the accepted SLAM /map after reset, but never block real-robot
            # or dry-run evaluation forever if slam_toolbox has not produced /map.
            # In that case run the same LiDAR-only internal map update that training
            # used before /map became available.  This lets confidence/priority and
            # debug overlays start from scan+odom instead of staying at known=0.
            use_slam = bool(getattr(self, "use_slam_map", True))
            slam_map = self._filtered_slam_map_for_update() if use_slam else None
            if bool(getattr(self, "strict_slam_map_required", False)) and use_slam and slam_map is None:
                # Do not publish/update LiDAR-only fallback layers in strict mode;
                # they are explicitly not the observation distribution the policy
                # was trained on.  Keep waiting and keep /cmd_vel=0.
                # Do not hammer dynamic_map from the warmup loop.  If strict
                # SLAM is required, _strict_wait_for_accepted_slam_map() is the
                # single owner of service retries.  Here we only wait on topic
                # callbacks/mirror updates and keep cmd_vel at zero.
                self._last_post_reset_ready_reason = f"strict_wait_slam:{self._last_slam_gate_reason}"
            else:
                try:
                    last_stats = self._update_exploration_map_with_unified_tf(slam_map=slam_map)
                    if use_slam and slam_map is None:
                        self._last_post_reset_ready_reason = "lidar_only_slam_fallback"
                except Exception as exc:
                    self._last_post_reset_ready_reason = f"map_update_error:{type(exc).__name__}"

            metrics = self._post_reset_ready_metrics(last_stats)
            best_metrics = metrics
            # Soft-ready ignores priority.  If priority cannot spawn at a specific
            # valid pose, do not lock reset forever; start with a warning after
            # the timeout.  The live updater can spawn the next cluster.
            if (
                bool(metrics.get("inside"))
                and int(metrics.get("lidar_beams", 0)) >= int(getattr(self, "post_reset_ready_min_lidar_beams", 0))
                and (
                    int(metrics.get("known_cells", 0)) >= int(getattr(self, "post_reset_ready_min_known_cells", 0))
                    or float(metrics.get("known_ratio", 0.0)) >= float(getattr(self, "post_reset_ready_min_known_ratio", 0.0))
                )
            ):
                soft_ready_stats = last_stats
                soft_ready_metrics = dict(metrics)

            if bool(metrics.get("ready", False)):
                self._last_post_reset_ready = True
                self._last_post_reset_ready_reason = str(metrics.get("reason", "ready"))
                self._last_post_reset_ready_known_ratio = float(metrics.get("known_ratio", 0.0))
                self._last_post_reset_ready_known_cells = int(metrics.get("known_cells", 0))
                self._last_post_reset_ready_lidar_beams = int(metrics.get("lidar_beams", 0))
                self._last_post_reset_ready_priority = float(metrics.get("priority_score", 0.0))
                self.ros.get_logger().info(
                    "POST_RESET_READY | "
                    f"dt={time.time() - start:.2f}s | slam={metrics.get('slam_gate')} | "
                    f"known={metrics.get('known_cells')},ratio={float(metrics.get('known_ratio', 0.0)):.3f} | "
                    f"beams={metrics.get('lidar_beams')} | conf={int(metrics.get('confidence_cells', 0))} | prio={float(metrics.get('priority_score', 0.0)):.3f}"
                )
                return last_stats

            now = time.time()
            if now - last_log >= 0.75:
                last_log = now
                self.ros.get_logger().info(
                    "POST_RESET_READY_WAITING | "
                    f"reason={metrics.get('reason')} | slam={metrics.get('slam_gate')} | "
                    f"known={metrics.get('known_cells')},ratio={float(metrics.get('known_ratio', 0.0)):.3f} | "
                    f"beams={metrics.get('lidar_beams')} | conf={int(metrics.get('confidence_cells', 0))} | prio={float(metrics.get('priority_score', 0.0)):.3f}"
                )
            time.sleep(0.03)

        if bool(getattr(self, "strict_slam_map_required", False)):
            metrics = best_metrics or {"reason": "timeout_no_metrics"}
            # v14: Strict SLAM should block missing/invalid /map, but it must not
            # crash training only because the initial camera-front confidence cone
            # is smaller than the requested warmup threshold.  In tight indoor
            # poses the forward 60-degree view may contain only a few cells, or
            # even 0 cells when facing a very close wall.  If SLAM, LiDAR and
            # local map readiness are already valid, let the episode start and
            # allow live confidence to grow after the policy moves/turns.
            reason = str(metrics.get("reason", ""))
            known_ok = (
                int(metrics.get("known_cells", 0)) >= int(getattr(self, "post_reset_ready_min_known_cells", 0))
                or float(metrics.get("known_ratio", 0.0)) >= float(getattr(self, "post_reset_ready_min_known_ratio", 0.0))
            )
            beams_ok = int(metrics.get("lidar_beams", 0)) >= int(getattr(self, "post_reset_ready_min_lidar_beams", 0))
            slam_ok = str(metrics.get("slam_gate", "")).startswith("accepted") or str(metrics.get("slam_gate", "")) == "accepted"
            allow_conf_timeout = bool(int(os.environ.get("TB3_RL_POST_RESET_READY_ALLOW_CONFIDENCE_TIMEOUT", "1") or 1))
            if allow_conf_timeout and reason.startswith("confidence_warmup") and known_ok and beams_ok and bool(metrics.get("inside", False)):
                self._last_post_reset_ready = True
                self._last_post_reset_ready_reason = "timeout_confidence_soft_ready"
                self._last_post_reset_ready_known_ratio = float(metrics.get("known_ratio", 0.0))
                self._last_post_reset_ready_known_cells = int(metrics.get("known_cells", 0))
                self._last_post_reset_ready_lidar_beams = int(metrics.get("lidar_beams", 0))
                self._last_post_reset_ready_priority = float(metrics.get("priority_score", 0.0))
                self.ros.get_logger().warn(
                    "POST_RESET_READY_CONFIDENCE_TIMEOUT_SOFT_READY | "
                    f"reason={reason} slam={metrics.get('slam_gate')} "
                    f"known={metrics.get('known_cells')},ratio={float(metrics.get('known_ratio', 0.0)):.3f} "
                    f"beams={metrics.get('lidar_beams')} conf={int(metrics.get('confidence_cells', 0))} "
                    "| starting episode; live confidence will update after motion"
                )
                return soft_ready_stats if soft_ready_stats is not None else last_stats

            self.ros.stop_robot()
            msg = (
                "POST_RESET_READY_STRICT_TIMEOUT | "
                f"reason={metrics.get('reason')} slam={metrics.get('slam_gate')} "
                f"known={metrics.get('known_cells')},ratio={float(metrics.get('known_ratio', 0.0)):.3f} "
                f"beams={metrics.get('lidar_beams')} conf={int(metrics.get('confidence_cells', 0))} prio={float(metrics.get('priority_score', 0.0)):.3f}. "
                "Policy step blocked because SLAM map is required."
            )
            self.ros.get_logger().error(msg)
            raise RecoverableResetError(msg)

        # Timeout handling: prefer a map/LiDAR-ready observation even if priority
        # did not spawn.  Never return a completely empty post-reset observation
        # when a soft-ready snapshot exists.
        metrics = soft_ready_metrics or best_metrics or {"reason": "timeout_no_metrics"}
        self._last_post_reset_ready = bool(soft_ready_metrics is not None)
        self._last_post_reset_ready_reason = "timeout_soft_ready" if soft_ready_metrics is not None else str(metrics.get("reason", "timeout"))
        self._last_post_reset_ready_known_ratio = float(metrics.get("known_ratio", 0.0))
        self._last_post_reset_ready_known_cells = int(metrics.get("known_cells", 0))
        self._last_post_reset_ready_lidar_beams = int(metrics.get("lidar_beams", 0))
        self._last_post_reset_ready_priority = float(metrics.get("priority_score", 0.0))
        self.ros.get_logger().warn(
            "POST_RESET_READY_TIMEOUT | "
            f"soft_ready={soft_ready_metrics is not None} | reason={metrics.get('reason')} | "
            f"slam={metrics.get('slam_gate')} | known={metrics.get('known_cells')},"
            f"ratio={float(metrics.get('known_ratio', 0.0)):.3f} | beams={metrics.get('lidar_beams')} | "
            f"conf={int(metrics.get('confidence_cells', 0))} | prio={float(metrics.get('priority_score', 0.0)):.3f}"
        )
        return soft_ready_stats if soft_ready_stats is not None else last_stats

    def _reset_confidence_pose_runtime_state(self) -> None:
        """Reset per-episode confidence pose anchoring state.

        This must run after every Gazebo/SLAM reset.  The confidence pose is
        anchored from the first valid real odometry sample of the current
        episode to the current /map pose.  Keeping the previous episode anchor
        makes episode 2+ paint at the old position even though the maps publish
        on the new /map canvas.
        """
        self._confidence_odom_anchor = None
        self._last_confidence_pose_log_step = -10_000_000
        self._confidence_cmd_xy = np.zeros(2, dtype=np.float32)
        self._confidence_cmd_yaw = 0.0
        self._confidence_cmd_last_time = None
        self._confidence_cmd_last_step = -1
        self._last_conf_rate_wall_time = 0.0
        self._last_conf_rate_step = -1
        self._last_conf_rate_ema = 0.0
        try:
            self._last_confidence_pose_warn_time = 0.0
        except Exception:
            pass

    def _sync_exploration_canvas_to_current_slam(self, *, reason: str = "", publish: bool = True) -> bool:
        """Force all RL map layers to the current raw /map canvas.

        v9 fix: reset_centered_at() intentionally clears the RL/confidence grids,
        but it also clears the stored SLAM publish reference.  In episode 2+, the
        first post-reset /rl_confidence_map can therefore be published on a stale
        or internal canvas until another SLAM sample path refreshes the reference.
        This helper makes the invariant explicit after every reset:

            /map, /rl_task_map, /rl_confidence_map, /rl_priority_map
            share exactly the same frame_id, origin, resolution, width, height.
        """
        try:
            emap = getattr(self, "exploration_map", None)
            slam = getattr(self.ros, "slam_map", None)
            if emap is None or slam is None:
                return False
            # Clear publication/resampling caches so a new episode cannot reuse
            # previous /map metadata after Cartographer restart.
            for name, value in (
                ("_publish_resample_cache_key", None),
                ("_publish_resample_cache", None),
                ("_last_direct_map_publish_step", -1),
                ("_last_confidence_publish_debug_step", -10_000_000),
            ):
                try:
                    setattr(emap, name, value)
                except Exception:
                    pass
            if hasattr(emap, "set_slam_publish_reference"):
                emap.set_slam_publish_reference(slam)
            # Also lock the internal persistent arrays to the same SLAM canvas.
            # This is not just an RViz visual fix; it makes reward/CNN/confidence
            # use the same global grid metadata as /map.
            try:
                emap._sample_slam_base(slam)
            except Exception:
                pass
            if publish:
                try:
                    emap.publish()
                except Exception:
                    pass
            # Optional one-line debug only when explicitly requested.
            try:
                dbg = str(os.environ.get("TB3_RL_EPISODE_CANVAS_DEBUG", "0")).strip().lower() in {"1", "true", "yes", "on"}
                if dbg:
                    info = slam.info
                    self.ros.get_logger().warn(
                        "EPISODE_CANVAS_SYNC | "
                        f"reason={reason} frame={getattr(slam.header, 'frame_id', '')} "
                        f"size={int(info.width)}x{int(info.height)} "
                        f"origin=({float(info.origin.position.x):+.2f},{float(info.origin.position.y):+.2f}) "
                        f"res={float(info.resolution):.3f}"
                    )
            except Exception:
                pass
            return True
        except Exception:
            return False

    def reset(self, seed=None, options=None):
        """Gym reset with non-fatal recovery for transient SLAM/TF readiness failures.

        Stable-Baselines3 calls reset() automatically after an episode ends.  A
        transient Cartographer/TF/map race inside reset() must therefore not raise
        out of the Gym API; otherwise the whole training job exits.  For
        recoverable reset-readiness failures, retry the complete reset sequence
        instead of returning a fake/empty observation.

        Environment variables:
          TB3_RL_RESET_RECOVERY_MAX_RETRIES_PER_CYCLE : default 8
          TB3_RL_RESET_RECOVERY_FATAL_AFTER           : default 0 (never fatal)
          TB3_RL_RESET_RECOVERY_BACKOFF_SEC           : default 0.35
          TB3_RL_RESET_RECOVERY_HARD_BACKOFF_SEC      : default 2.0
        """
        try:
            max_per_cycle = int(os.environ.get("TB3_RL_RESET_RECOVERY_MAX_RETRIES_PER_CYCLE", "8") or 8)
        except Exception:
            max_per_cycle = 8
        max_per_cycle = max(int(max_per_cycle), 1)

        try:
            fatal_after = int(os.environ.get("TB3_RL_RESET_RECOVERY_FATAL_AFTER", "0") or 0)
        except Exception:
            fatal_after = 0
        fatal_after = max(int(fatal_after), 0)

        attempt = 0
        last_exc = None
        while rclpy.ok():
            episode_before = int(getattr(self, "episode_index", 0))
            try:
                obs, info = self._reset_once(seed=seed, options=options)
                if attempt > 0:
                    try:
                        if info is None:
                            info = {}
                        if isinstance(info, dict):
                            info["reset_recovery_attempts"] = int(attempt)
                            info["reset_recovery_last_error"] = str(last_exc) if last_exc is not None else ""
                    except Exception:
                        pass
                    try:
                        self.ros.get_logger().warn(
                            "RESET_RECOVERY_SUCCEEDED | "
                            f"attempts={attempt} episode={int(getattr(self, 'episode_index', 0))}"
                        )
                    except Exception:
                        pass
                return obs, info
            except RecoverableResetError as exc:
                last_exc = exc
                attempt += 1
                # The failed reset attempt did not produce a valid Gym episode;
                # keep the public episode counter stable until reset succeeds.
                try:
                    self.episode_index = episode_before
                except Exception:
                    pass

                try:
                    self.ros.get_logger().warn(
                        "RESET_RECOVERY_RETRY | "
                        f"attempt={attempt} cycle_attempt={((attempt - 1) % max_per_cycle) + 1}/{max_per_cycle} "
                        f"fatal_after={fatal_after if fatal_after > 0 else 'never'} | {exc}"
                    )
                except Exception:
                    pass

                if fatal_after > 0 and attempt >= fatal_after:
                    try:
                        self.ros.get_logger().error(
                            "RESET_RECOVERY_FATAL | "
                            f"attempt={attempt}/{fatal_after} | raising after configured fatal limit | {exc}"
                        )
                    except Exception:
                        pass
                    raise

                hard_cycle = (attempt % max_per_cycle) == 0
                self._prepare_reset_recovery(attempt=attempt, exc=exc, hard_cycle=hard_cycle)
                continue

            except KeyboardInterrupt:
                raise

            except Exception as exc:
                # Non-Recoverable exceptions (transient ROS/Gazebo/SHM failures that
                # were not wrapped as RecoverableResetError) must also not kill the
                # whole training job.  Retry with the same recovery machinery, but
                # cap unexpected failures separately so a genuine permanent fault
                # still surfaces eventually.
                last_exc = exc
                attempt += 1
                try:
                    self.episode_index = episode_before
                except Exception:
                    pass

                try:
                    unexpected_fatal_after = int(
                        os.environ.get("TB3_RL_RESET_UNEXPECTED_FATAL_AFTER", "40") or 40
                    )
                except Exception:
                    unexpected_fatal_after = 40
                unexpected_fatal_after = max(int(unexpected_fatal_after), 1)

                try:
                    self.ros.get_logger().warn(
                        "RESET_RECOVERY_RETRY_UNEXPECTED | "
                        f"attempt={attempt} unexpected_fatal_after={unexpected_fatal_after} | "
                        f"{type(exc).__name__}: {exc}"
                    )
                except Exception:
                    pass

                if attempt >= unexpected_fatal_after:
                    try:
                        self.ros.get_logger().error(
                            "RESET_RECOVERY_FATAL_UNEXPECTED | "
                            f"attempt={attempt}/{unexpected_fatal_after} | "
                            f"raising after unexpected-failure limit | {type(exc).__name__}: {exc}"
                        )
                    except Exception:
                        pass
                    raise

                # Force a hard recovery cycle (includes SHM/TF/SLAM cleanup) for
                # unexpected faults regardless of the normal cycle counter.
                self._prepare_reset_recovery(attempt=attempt, exc=exc, hard_cycle=True)
                continue

        # rclpy shutdown/cancel path.  At this point the training process is
        # already terminating externally, so propagating is appropriate.
        raise RuntimeError("ROS shutdown while reset recovery was waiting")

    def _prepare_reset_recovery(self, attempt: int, exc: Exception | None = None, hard_cycle: bool = False) -> None:
        """Hold the robot still and clear transient state before retrying reset().

        This function deliberately does not manufacture an observation.  Its job
        is only to make the next full reset attempt more likely to succeed.
        """
        try:
            self._map_live_update_paused = True
        except Exception:
            pass
        try:
            self.ros.stop_robot()
        except Exception:
            pass
        try:
            self._clear_waypoint_visualization()
        except Exception:
            pass
        try:
            if str(getattr(self, "action_mode", "")) == "nav2":
                self._cancel_nav2_goal()
                self._clear_nav2_costmaps(wait_timeout_sec=0.50)
        except Exception:
            pass
        try:
            if bool(getattr(self, "reset_tf_buffer_on_reset", True)) and hasattr(self.ros, "reset_tf_buffer"):
                self.ros.reset_tf_buffer()
        except Exception:
            pass
        try:
            if hasattr(self.ros, "reset_slam_state"):
                # Drop stale cached /map signatures.  The next _reset_once() still
                # performs the mandatory per-episode SLAM reset/restart policy.
                self.ros.reset_slam_state()
        except Exception:
            pass
        if hard_cycle:
            try:
                rm = getattr(self, "reset_manager", None)
                if rm is not None and hasattr(rm, "failed_entity_names"):
                    rm.failed_entity_names.clear()
            except Exception:
                pass
            # NOTE: No /dev/shm sweep here.  SHM exhaustion is prevented by forcing
            # a non-SHM FastDDS transport before ROS init; an age-based sweep during
            # active ROS can destroy a live participant's segment and is unsafe.
            try:
                self.ros.get_logger().warn(
                    "RESET_RECOVERY_HARD_CYCLE | "
                    f"attempt={attempt} | cleared transient TF/SLAM/entity blacklist; waiting before next reset"
                )
            except Exception:
                pass
        try:
            self.ros.spin_steps(num_spins=max(int(getattr(self, "post_reset_stabilize_spin_steps", 12)), 8), timeout_sec=0.005)
        except Exception:
            pass
        try:
            self._advance_world_after_command(target_delta_sec=0.03)
        except Exception:
            pass
        try:
            backoff = float(os.environ.get(
                "TB3_RL_RESET_RECOVERY_HARD_BACKOFF_SEC" if hard_cycle else "TB3_RL_RESET_RECOVERY_BACKOFF_SEC",
                "2.0" if hard_cycle else "0.35",
            ) or (2.0 if hard_cycle else 0.35))
        except Exception:
            backoff = 2.0 if hard_cycle else 0.35
        if backoff > 0.0:
            time.sleep(max(float(backoff), 0.0))

    def _reset_once(self, seed=None, options=None):
        super().reset(seed=seed)
        _reset_prof_on = str(os.environ.get("TB3_RL_RESET_PROFILER", "0")).strip().lower() in {"1", "true", "yes", "on"}
        if _reset_prof_on:
            _reset_prof_t0 = time.perf_counter()
            _reset_prof_last = _reset_prof_t0
            _reset_prof_parts = {}

            def _reset_prof_mark(name: str) -> None:
                nonlocal _reset_prof_last
                now = time.perf_counter()
                _reset_prof_parts[name] = _reset_prof_parts.get(name, 0.0) + float(now - _reset_prof_last)
                _reset_prof_last = now
        else:
            _reset_prof_t0 = 0.0
            _reset_prof_parts = {}

            def _reset_prof_mark(name: str) -> None:
                return

        self._map_live_update_paused = True
        self.episode_index += 1

        self.ros.stop_robot()
        self._advance_world_after_command(target_delta_sec=0.02)

        # 1) Burger를 episode 시작점으로 보낸다.
        #    house_random/list 모드에서는 후보를 무작위 순서로 검증한다.
        #    Gazebo pose reset 자체가 성공해도 벽/기둥에 너무 가까우면 SLAM 첫 scan이
        #    깨지므로 fresh scan clearance가 충분한 후보만 채택한다.
        reset_pose = None
        reset_x, reset_y = random.choice(self.reset_pose_candidates)
        accepted_reason = "not_validated"

        if self.enable_pose_reset:
            if self.reset_manager is None:
                raise RuntimeError("pose reset is enabled but reset_manager is None")

            last_error = "unknown"
            for attempt_idx, (cand_x, cand_y) in enumerate(self._select_reset_candidate_order(), start=1):
                reset_x, reset_y = float(cand_x), float(cand_y)
                reset_pose = self.reset_manager.reset_center_pose(
                    x=reset_x,
                    y=reset_y,
                    random_yaw=self.random_reset_yaw,
                    fixed_yaw=0.0,
                )
                if reset_pose is None:
                    last_error = "set_entity_pose_failed"
                    continue

                self.ros.stop_robot()
                self._advance_world_after_command(target_delta_sec=self.control_dt)
                if self.reset_tf_buffer_on_reset and hasattr(self.ros, "reset_tf_buffer"):
                    self.ros.reset_tf_buffer()
                    self.ros.spin_steps(num_spins=8, timeout_sec=0.002)
                self._wait_for_fresh_reset_sensors()
                clear, reason, gmin, fmin, rmin = self._is_reset_pose_clear()
                last_error = reason
                reset_check_msg = (
                    "RESET_CANDIDATE_CHECK | "
                    f"attempt={attempt_idx}/{self.reset_pose_max_attempts} | "
                    f"requested=(x={reset_x:.3f}, y={reset_y:.3f}) | "
                    f"clear={clear} | reason={reason} | "
                    f"lidar_min={gmin:.3f}, front={fmin:.3f}, rear={rmin:.3f}"
                )
                if clear:
                    if _quiet_reset_logs():
                        self.ros.get_logger().debug(reset_check_msg)
                    else:
                        self.ros.get_logger().info(reset_check_msg)
                    accepted_reason = reason
                    break
                else:
                    self.ros.get_logger().debug(reset_check_msg)

            if reset_pose is None:
                raise RuntimeError(
                    "Failed to reset TurtleBot pose. "
                    "Check SetEntityPose service/entity name. "
                    "The ResetManager now auto-tries gz model --list candidates."
                )
            if accepted_reason != "clear":
                self.ros.get_logger().warn(
                    "No reset candidate passed LiDAR clearance validation. "
                    f"Continuing with last candidate requested=(x={reset_x:.3f}, y={reset_y:.3f}), "
                    f"last_error={last_error}. Consider editing --reset-pose-list."
                )

            self._log_reset_pose_truth(
                requested_x=reset_x,
                requested_y=reset_y,
                reset_pose=reset_pose,
            )
        else:
            self.ros.spin_for(0.2)
            self._log_reset_pose_truth(
                requested_x=reset_x,
                requested_y=reset_y,
                reset_pose=None,
            )
        _reset_prof_mark("pose_reset")

        self.current_reset_xy = np.array([reset_x, reset_y], dtype=np.float32)

        self.prev_action = np.zeros(2, dtype=np.float32)
        self.filtered_action = np.zeros(2, dtype=np.float32)
        self.raw_action = np.zeros(2, dtype=np.float32)
        self.prev_policy_action = np.zeros(2, dtype=np.float32)
        self._last_waypoint_local = np.zeros(2, dtype=np.float32)
        self._last_waypoint_world = np.zeros(2, dtype=np.float32)
        self._last_waypoint_distance = 0.0
        self._last_waypoint_angle = 0.0
        self._last_waypoint_lateral_offset = 0.0
        self._last_waypoint_heading_delta = 0.0
        self._prev_waypoint_angle_for_reward: Optional[float] = None
        self._last_waypoint_action_type = self.waypoint_action_type
        self._last_waypoint_reached = False
        self._last_nav2_goal_heading_error = 999.0
        self._last_nav2_goal_front_min = 999.0
        self._last_nav2_backup_gate_reason = "none"
        self._last_controller_steps = 0
        self.motion_mode = "STRAIGHT"
        self.step_count = 0
        self._last_step_reward = 0.0
        self._episode_reward_sum = 0.0
        self._episode_discounted_return = 0.0
        self._episode_start_discounted_return = 0.0
        self._episode_reward_ema = 0.0
        self._recent_reward_window.clear()
        self._recent_slam_new_window.clear()
        self._recent_conf_update_window.clear()
        self._coverage_stall_slam_window.clear()
        self._coverage_stall_conf_window.clear()
        self._last_coverage_stall_terminal = False
        self._last_coverage_stall_active = False
        self._last_coverage_stall_reason = "reset"
        self._last_coverage_stall_slam_window = 0
        self._last_coverage_stall_conf_window = 0
        self._last_coverage_stall_window_len = 0
        self._last_priority_clear_reward = 0.0
        self._last_priority_recheck_reward = 0.0
        self._last_priority_check_reward = 0.0
        self._episode_priority_clear_reward = 0.0
        self._episode_priority_recheck_reward = 0.0
        self._episode_priority_check_reward = 0.0
        self._last_reward_text_valid = False
        self.priority_stuck_steps = 0
        self._last_priority_stuck_active = False
        self._last_priority_stuck_restart = False
        self._last_priority_stuck_reason = "none"
        self.lidar_empty_steps = 0
        self._last_lidar_empty_active = False
        self._last_lidar_empty_restart = False
        self._last_lidar_empty_reason = "none"
        self.velocity_safety_cooldown_steps = 0
        self.velocity_safety_backup_lock_steps = 0
        self.velocity_safety_backup_lock_turn_sign = 1.0
        self._last_velocity_safety_backup_lock_active = False
        self._last_velocity_safety_backup_triggered = False
        self._last_velocity_safety_blocked = False
        self._last_velocity_safety_skip_store = False
        self._pending_skip_penalty = 0.0
        self._last_velocity_safety_slowdown = 1.0
        self._last_velocity_safety_slowdown_risk = 0.0
        self._last_velocity_safety_policy_v = 0.0
        self._last_velocity_safety_executed_v = 0.0
        self._last_velocity_safety_penalty = 0.0
        self._last_velocity_safety_reason = "none"
        self.shake_steps = 0
        self._last_shake_active = False
        self._last_shake_restart = False
        self._last_shake_reason = "none"
        self._shake_last_wobble_reason = "none"
        try:
            self._shake_wobble_history.clear()
        except Exception:
            self._shake_wobble_history = deque(maxlen=int(getattr(self, "shake_wobble_window_steps", 8)))
        self.nav2_stuck_steps = 0
        self.nav2_backup_cooldown_steps = 0
        self._last_nav2_stuck_active = False
        self._last_nav2_stuck_backup_triggered = False
        self._last_nav2_stuck_reason = "none"
        self._last_nav2_backup_status = "none"
        self._last_lidar_valid_beams = 0
        self._last_lidar_nearest_detection = 999.0
        self._episode_start_wall_time = time.time()
        self._episode_start_sim_time = self._safe_sim_time()
        self.explored_stall_steps = 0
        self.confidence_stall_steps = 0
        self.sustained_rotation_steps = 0
        self.orbit_stall_steps = 0
        try:
            self._orbit_pose_history.clear()
        except Exception:
            pass
        self._last_orbit_path_efficiency = 1.0
        self._last_orbit_path_length = 0.0
        self._last_orbit_net_displacement = 0.0
        self._last_orbit_yaw_accum = 0.0
        self._last_orbit_reason = "reset"
        self.last_map_stats = self._empty_map_stats()
        self.vector_history.clear()
        try:
            self.map_history.clear()
        except Exception:
            pass
        self._waypoint_history.clear()
        self._clear_waypoint_visualization()
        self._last_terminal_reason = "none"
        self._last_collision_restart_requested = False
        self._last_out_of_bounds = False
        self._last_out_of_bounds_reason = "none"
        self._last_out_of_bounds_radius = 0.0
        self._last_out_of_bounds_x = 0.0
        self._last_out_of_bounds_y = 0.0
        self._last_out_of_bounds_z = 0.0
        self.map_bounds_bad_steps = 0
        self._last_map_bounds_restart = False
        self._last_map_bounds_reason = "none"
        self._last_map_bounds_local_known_ratio = 0.0
        self._last_map_bounds_local_known_cells = 0
        self._last_confidence_update_xy = None
        self._last_confidence_update_step = -10_000_000
        self._last_direct_tf_pose_xy = None
        self._last_direct_tf_pose_yaw = None
        self._last_direct_tf_pose_wall = 0.0
        self._confidence_unified_xy = None
        self._confidence_unified_yaw = None

        # Boundary center는 SLAM map frame이 아니라 reset-stable frame
        # self.safety_boundary_frame에서 실제 reset 후 pose로 잡는다.
        self._update_boundary_center_after_reset(requested_xy=self.current_reset_xy.copy())

        # 2) SLAM reset is mandatory at every respawn.
        #    The SLAM process itself is not restarted unless explicitly requested,
        #    but the map state is cleared on every episode/reset so RViz layers and
        #    Nav2 costmaps do not accumulate stale map->odom drift.
        self.ignore_slam_prior_this_episode = False
        self._episode_slam_transform_source = ""
        self._episode_slam_transform_target = ""
        self._episode_slam_transform = None
        self._slam_transform_cache_key = None
        self._slam_transform_cache_msg = None
        self._reset_confidence_pose_runtime_state()
        should_reset_slam = bool(self.use_slam_map)

        if should_reset_slam:
            slam_reset_ok = self._reset_slam_map_after_pose_reset()
            if slam_reset_ok and self.rviz_zero_robot_on_reset:
                self._wait_for_rviz_map_origin_after_reset()
            if (
                slam_reset_ok
                and self.ros.slam_map is not None
                and hasattr(self.exploration_map, "set_slam_publish_reference")
            ):
                # RViz/debug layers must lock to the raw SLAM /map canvas, not to
                # the odom-transformed learning grid.  Internal control still uses
                # pose_frame=odom, but published /rl_* OccupancyGrid metadata now
                # matches /map exactly so confidence/priority/filtered-slam align.
                self.exploration_map.set_slam_publish_reference(self.ros.slam_map)
            self.ignore_slam_prior_this_episode = not bool(slam_reset_ok)

        # Gate SLAM maps after teleport/reset. slam_toolbox may still deliver an
        # old /scan-/tf-derived /map for a short time. Only maps received after
        # this wall-clock barrier and after accept_delay are allowed into the RL
        # filtered base_grid used by priority/path/CNN.
        now_wall = time.time()
        self._slam_map_min_wall_time = now_wall
        self._slam_map_accept_after_wall_time = now_wall + self.slam_map_accept_delay_sec
        self._last_slam_gate_reason = "reset_delay"
        self._last_slam_map_age_sec = -1.0
        self._last_slam_map_delay_remaining_sec = self.slam_map_accept_delay_sec

        if bool(getattr(self, "strict_slam_map_required", False)):
            # Make the reset barrier explicit: do not let a policy observation be
            # generated from an old pre-reset map or from LiDAR-only fallback.
            self._strict_wait_for_accepted_slam_map(stage="after_slam_reset")
        _reset_prof_mark("slam_reset")

        # 3) Nav2는 Gazebo pose reset 이전 episode의 goal/costmap 상태를 물고 있을 수 있다.
        #    reset마다 goal을 취소하고 costmap을 비워야 reset 후 ABORTED 루프가 줄어든다.
        if self.action_mode == "nav2":
            self._cancel_nav2_goal()
            self._clear_nav2_costmaps(wait_timeout_sec=0.60)

        # 4) RESET 직후 SLAM/TF/Nav2 costmap 안정화 대기.
        #    첫 Nav2 goal을 보내기 전에 /cmd_vel=0으로 고정하고 callback만 돌린다.
        self._post_reset_stabilize(reset_pose=reset_pose)

        # 4.5) Some valid spawn poses still leave Nav2's local/global costmaps with
        # stale obstacle inflation from the previous episode or from the teleport
        # instant.  Clear once more *after* the post-reset wait so every accepted
        # spawn gets a clean first NavigateToPose goal.  This is intentionally not
        # tied to the boundary terminal; collision/fallen/drop are still hard
        # terminals in step().
        if self.action_mode == "nav2":
            self._cancel_nav2_goal()
            cleared_after_reset = self._clear_nav2_costmaps(wait_timeout_sec=0.80)
            self.ros.spin_steps(num_spins=12, timeout_sec=0.005)
            if bool(getattr(self, "collision_clear_nav2_costmaps", True)):
                self.ros.get_logger().info(
                    f"NAV2_RESET_READY | goal canceled, costmap_clear_attempted={cleared_after_reset}"
                )
            else:
                self.ros.get_logger().info(
                    "NAV2_RESET_READY | goal canceled, manual costmap clear disabled"
                )
        _reset_prof_mark("post_reset_stabilize")

        # 5) RL memory/confidence map도 전부 초기화하고 Burger가 중앙에 오도록 origin을 재설정한다.
        robot_pose = self._get_robot_pose2d()

        if robot_pose is not None:
            robot_xy, _ = robot_pose
            self.exploration_map.reset_centered_at(robot_xy)
        else:
            self.exploration_map.reset_centered_at(self.current_reset_xy.copy())

        # v9: reset_centered_at() clears the grids and the SLAM publish reference.
        # Re-lock all persistent RL layers to the current /map canvas immediately,
        # then clear the confidence pose anchor so the first confidence update of
        # this episode anchors real odometry to this episode's map pose.
        self._sync_exploration_canvas_to_current_slam(reason="after_exploration_reset", publish=True)
        self._reset_confidence_pose_runtime_state()
        _reset_prof_mark("map_reset")

        # 6) RESET 직후에는 바로 episode를 열지 않는다.
        #    SLAM /map이 reset 이후 실제로 채워지고, LiDAR/confidence/priority가
        #    최소 기준을 만족할 때까지 /cmd_vel=0으로 대기한다. Gym reset()이
        #    아직 반환되지 않았으므로 이 구간은 reward/replay에 들어가지 않는다.
        self.last_map_stats = self._wait_post_reset_ready(reset_pose=reset_pose)
        # v9: after the ready gate, Cartographer may have grown /map again.
        # Refresh the shared /map canvas reference without clearing confidence.
        self._sync_exploration_canvas_to_current_slam(reason="after_post_reset_ready", publish=True)
        self._last_live_map_update_wall = time.time()
        self._map_live_update_paused = False
        _reset_prof_mark("ready_gate")

        obs = self._get_obs()

        info = self._build_info(
            map_stats=self.last_map_stats,
            collision=self._check_collision(),
            fallen=self._check_fallen(),
            coverage_done=False,
        )
        _reset_prof_mark("obs_info")

        if _reset_prof_on:
            total = float(time.perf_counter() - _reset_prof_t0)
            _reset_prof_parts["total"] = total
            order = ["pose_reset", "slam_reset", "post_reset_stabilize", "map_reset", "ready_gate", "obs_info", "total"]
            txt = " ".join(
                f"{k}={float(_reset_prof_parts.get(k, 0.0))*1000.0:.1f}ms"
                for k in order if k in _reset_prof_parts
            )
            self.ros.get_logger().warn(
                "RESET_PROFILE | "
                f"episode={self.episode_index} reason={self._last_terminal_reason} | {txt}"
            )

        return obs, info

    def _effective_rl_step_sec(self) -> float:
        """Approximate simulated control duration represented by one Gym step."""
        base = max(float(getattr(self, "control_dt", 0.05)), 1e-3)
        if str(getattr(self, "action_mode", "")) == "nav2":
            base = max(base, float(getattr(self, "nav2_control_window_sec", base)))
        return base

    def _seconds_to_step_count(self, sec: float, fallback_steps: int) -> int:
        sec = float(sec)
        if sec <= 0.0:
            return max(int(fallback_steps), 1)
        return max(int(math.ceil(sec / self._effective_rl_step_sec())), 1)

    def _lidar_detection_stats(self) -> tuple[int, float]:
        """Return (valid_detection_beams, nearest_detection_m).

        A beam is a detection only if it is finite and strictly below the
        configured no-hit cutoff.  Gazebo/TurtleBot3 LiDAR often encodes
        no-hit beams as +inf or near range_max; those must not count as objects.
        """
        scan = self.ros.scan
        if scan is None:
            return 0, 999.0

        ranges = np.asarray(scan.ranges, dtype=np.float32)
        if ranges.size == 0:
            return 0, 999.0

        range_min = max(float(getattr(scan, "range_min", 0.0)), 0.0)
        range_max = float(getattr(scan, "range_max", 3.5))
        min_valid = max(float(getattr(self, "lidar_empty_min_valid_range_m", 0.12)), range_min)
        max_valid = min(
            float(getattr(self, "lidar_empty_max_valid_range_m", 3.35)),
            max(range_min + 1e-3, range_max - 0.03),
        )

        finite = np.isfinite(ranges)
        valid = finite & (ranges >= min_valid) & (ranges < max_valid)
        count = int(np.count_nonzero(valid))
        nearest = float(np.min(ranges[valid])) if count > 0 else 999.0
        return count, nearest

    def _nav2_backup_time_allowance(self) -> Duration:
        sec = max(float(getattr(self, "nav2_stuck_backup_timeout_sec", 4.0)), 0.1)
        whole = int(math.floor(sec))
        nanosec = int(round((sec - whole) * 1_000_000_000))
        if nanosec >= 1_000_000_000:
            whole += 1
            nanosec -= 1_000_000_000
        msg = Duration()
        msg.sec = whole
        msg.nanosec = nanosec
        return msg

    def _signed_heading_error_to_world_target(self, target_world_xy: np.ndarray | None) -> float:
        """Signed yaw error from robot heading to a map/pose-frame target."""
        if target_world_xy is None:
            return 0.0
        robot_pose = self._get_robot_pose2d()
        if robot_pose is None:
            return 0.0
        robot_xy, robot_yaw = robot_pose
        target = np.asarray(target_world_xy, dtype=np.float32)
        if target.shape[0] < 2 or not np.all(np.isfinite(target[:2])):
            return 0.0
        dx = float(target[0]) - float(robot_xy[0])
        dy = float(target[1]) - float(robot_xy[1])
        if not np.isfinite(dx) or not np.isfinite(dy) or math.hypot(dx, dy) < 1e-4:
            return 0.0
        return float(self._normalize_angle(math.atan2(dy, dx) - float(robot_yaw)))

    def _nav2_behavior_time_allowance(self, sec: float) -> Duration:
        sec = max(float(sec), 0.1)
        whole = int(math.floor(sec))
        nanosec = int(round((sec - whole) * 1_000_000_000))
        if nanosec >= 1_000_000_000:
            whole += 1
            nanosec -= 1_000_000_000
        msg = Duration()
        msg.sec = whole
        msg.nanosec = nanosec
        return msg

    def _execute_nav2_spin_unstall(self, angle_rad: float, reason: str = "followpath_rotation_stall") -> bool:
        """Use Nav2 behavior_server Spin when FollowPath is accepted but yaw is frozen.

        This handles the exact failure mode visible in logs:
        gate=rotating_to_goal, moved=0.000m, yaw_delta=0.0deg.  Waiting on the
        same FollowPath goal is useless there because controller_server is not
        producing rotation.  Spin is a Nav2 action, so motion is still Nav2-owned.
        """
        if self.action_mode != "nav2" or Spin is None:
            return False
        client = getattr(self, "nav2_spin_client", None)
        if client is None:
            return False
        if not client.wait_for_server(timeout_sec=0.20):
            self.ros.get_logger().warn(
                "NAV2_SPIN_UNSTALL_UNAVAILABLE | /spin action not ready; will replan FollowPath"
            )
            return False

        # Do not spin more than a bounded heading correction in one macro-step.
        angle = float(np.clip(float(angle_rad), -0.85, 0.85))
        if abs(angle) < math.radians(4.0):
            return False

        self._cancel_nav2_goal_sync(reason="nav2_followpath_rotation_stall_spin")
        goal = Spin.Goal()
        try:
            goal.target_yaw = float(angle)
        except Exception:
            return False
        try:
            goal.time_allowance = self._nav2_behavior_time_allowance(2.5)
        except Exception:
            pass

        start_pose = self._get_robot_pose2d()
        self.ros.get_logger().warn(
            "NAV2_SPIN_UNSTALL_START | "
            f"angle={math.degrees(angle):+.1f}deg | reason={reason}"
        )
        send_future = client.send_goal_async(goal)
        if not self._wait_future_done(send_future, timeout_sec=0.80):
            self.ros.get_logger().warn("NAV2_SPIN_UNSTALL_REJECTED | send timeout")
            return False
        try:
            goal_handle = send_future.result()
        except Exception as exc:
            self.ros.get_logger().warn(f"NAV2_SPIN_UNSTALL_REJECTED | error={exc}")
            return False
        if goal_handle is None or not bool(getattr(goal_handle, "accepted", False)):
            self.ros.get_logger().warn("NAV2_SPIN_UNSTALL_REJECTED | goal not accepted")
            return False

        result_future = goal_handle.get_result_async()
        self._wait_future_done(result_future, timeout_sec=2.8)
        end_pose = self._get_robot_pose2d()
        yaw_moved = 0.0
        if start_pose is not None and end_pose is not None:
            yaw_moved = abs(self._normalize_angle(float(end_pose[1]) - float(start_pose[1])))
        self.ros.get_logger().warn(
            "NAV2_SPIN_UNSTALL_DONE | "
            f"requested={math.degrees(angle):+.1f}deg | yaw_moved={math.degrees(yaw_moved):.1f}deg"
        )
        return True

    def _execute_nav2_drive_on_heading_unstall(self, reason: str = "followpath_no_translation") -> bool:
        """Use Nav2 DriveOnHeading when front is open but FollowPath outputs no progress."""
        if self.action_mode != "nav2" or DriveOnHeading is None:
            return False
        client = getattr(self, "nav2_drive_on_heading_client", None)
        if client is None:
            return False
        if not client.wait_for_server(timeout_sec=0.20):
            self.ros.get_logger().warn(
                "NAV2_DRIVE_ON_HEADING_UNSTALL_UNAVAILABLE | /drive_on_heading action not ready; will replan FollowPath"
            )
            return False

        self._cancel_nav2_goal_sync(reason="nav2_followpath_no_translation_drive_on_heading")
        goal = DriveOnHeading.Goal()
        try:
            goal.target.x = 0.28
            goal.target.y = 0.0
            goal.target.z = 0.0
            goal.speed = 0.08
        except Exception:
            return False
        try:
            goal.time_allowance = self._nav2_behavior_time_allowance(3.0)
        except Exception:
            pass

        start_pose = self._get_robot_pose2d()
        self.ros.get_logger().warn(
            "NAV2_DRIVE_ON_HEADING_UNSTALL_START | "
            f"distance=0.28m speed=0.08mps | reason={reason}"
        )
        send_future = client.send_goal_async(goal)
        if not self._wait_future_done(send_future, timeout_sec=0.80):
            self.ros.get_logger().warn("NAV2_DRIVE_ON_HEADING_UNSTALL_REJECTED | send timeout")
            return False
        try:
            goal_handle = send_future.result()
        except Exception as exc:
            self.ros.get_logger().warn(f"NAV2_DRIVE_ON_HEADING_UNSTALL_REJECTED | error={exc}")
            return False
        if goal_handle is None or not bool(getattr(goal_handle, "accepted", False)):
            self.ros.get_logger().warn("NAV2_DRIVE_ON_HEADING_UNSTALL_REJECTED | goal not accepted")
            return False

        result_future = goal_handle.get_result_async()
        self._wait_future_done(result_future, timeout_sec=3.4)
        end_pose = self._get_robot_pose2d()
        moved = 0.0
        if start_pose is not None and end_pose is not None:
            moved = float(np.linalg.norm(np.asarray(end_pose[0], dtype=np.float32) - np.asarray(start_pose[0], dtype=np.float32)))
        self.ros.get_logger().warn(
            "NAV2_DRIVE_ON_HEADING_UNSTALL_DONE | "
            f"moved={moved:.3f}m"
        )
        return True

    def _execute_nav2_backup_behavior(self, reason: str = "motion_stuck") -> bool:
        """Run Nav2 behavior_server BackUp action as the only escape motion.

        This keeps the earlier Nav2-only invariant: the env does not publish
        /cmd_vel directly.  If /backup is unavailable, we only cancel the active
        controller goal and clear costmaps; the next SAC step will sample a new
        Nav2 goal.
        """
        if self.action_mode != "nav2":
            return False

        self._last_nav2_backup_status = "unavailable"
        self._cancel_nav2_goal()

        client = getattr(self, "nav2_backup_client", None)
        if client is None or BackUp is None:
            self.ros.get_logger().warn(
                "NAV2_STUCK_BACKUP_UNAVAILABLE | /backup action type/client is not available; "
                "clearing costmaps and continuing with next Nav2 goal"
            )
            self._clear_nav2_costmaps(wait_timeout_sec=0.35)
            return False

        if not client.wait_for_server(timeout_sec=0.25):
            self.ros.get_logger().warn(
                f"NAV2_STUCK_BACKUP_UNAVAILABLE | action={self.nav2_backup_action_name} not ready; "
                "clearing costmaps and continuing with next Nav2 goal"
            )
            self._clear_nav2_costmaps(wait_timeout_sec=0.35)
            return False

        goal = BackUp.Goal()
        goal.target.x = -float(getattr(self, "nav2_stuck_backup_distance_m", 0.26))
        goal.target.y = 0.0
        goal.target.z = 0.0
        goal.speed = float(getattr(self, "nav2_stuck_backup_speed_mps", 0.07))
        try:
            goal.time_allowance = self._nav2_backup_time_allowance()
        except Exception:
            pass

        start_pose = self._get_robot_pose2d()
        self.ros.get_logger().warn(
            "NAV2_STUCK_BACKUP_START | "
            f"reason={reason} | action={self.nav2_backup_action_name} | "
            f"distance={abs(goal.target.x):.2f}m | speed={goal.speed:.2f}m/s | "
            f"timeout={float(getattr(self, 'nav2_stuck_backup_timeout_sec', 0.0)):.2f}s"
        )

        send_future = client.send_goal_async(goal)
        if not self._wait_future_done(send_future, timeout_sec=0.75):
            self._last_nav2_backup_status = "send_timeout"
            self.ros.get_logger().warn("NAV2_STUCK_BACKUP_FAIL | send_timeout")
            self._clear_nav2_costmaps(wait_timeout_sec=0.35)
            return False

        goal_handle = send_future.result()
        if goal_handle is None or not goal_handle.accepted:
            self._last_nav2_backup_status = "rejected"
            self.ros.get_logger().warn("NAV2_STUCK_BACKUP_FAIL | rejected")
            self._clear_nav2_costmaps(wait_timeout_sec=0.35)
            return False

        result_future = goal_handle.get_result_async()
        ok = self._wait_future_done(
            result_future,
            timeout_sec=float(getattr(self, "nav2_stuck_backup_timeout_sec", 4.0)) + 0.5,
        )
        status_name = "timeout"
        if ok:
            try:
                wrapped = result_future.result()
                status = int(getattr(wrapped, "status", -1))
                status_name = self._nav2_status_name(status)
            except Exception:
                status_name = "result_error"
        else:
            try:
                goal_handle.cancel_goal_async()
            except Exception:
                pass

        moved = 0.0
        end_pose = self._get_robot_pose2d()
        if start_pose is not None and end_pose is not None:
            moved = float(np.linalg.norm(np.asarray(end_pose[0]) - np.asarray(start_pose[0])))

        self._last_nav2_backup_status = status_name
        self._clear_nav2_costmaps(wait_timeout_sec=0.35)
        self.ros.get_logger().warn(
            "NAV2_STUCK_BACKUP_DONE | "
            f"status={status_name} | moved={moved:.3f}m"
        )
        return bool(status_name in {"SUCCEEDED", "STATUS_SUCCEEDED", "4"} or moved >= 0.03)

    def _update_nav2_stuck_backup_state(self) -> bool:
        """Back up when Nav2 accepted goals but the robot has not translated.

        This does not restart the episode.  It performs one bounded Nav2 BackUp
        behavior and then lets the SAC loop continue with a new goal.
        """
        self._last_nav2_stuck_backup_triggered = False
        self._last_nav2_stuck_reason = "none"

        if self.action_mode != "nav2" or not bool(getattr(self, "nav2_stuck_backup", True)):
            self.nav2_stuck_steps = 0
            self._last_nav2_stuck_active = False
            return False

        if int(getattr(self, "nav2_backup_cooldown_steps", 0)) > 0:
            self.nav2_backup_cooldown_steps -= 1
            self.nav2_stuck_steps = 0
            self._last_nav2_stuck_active = False
            self._last_nav2_stuck_reason = f"cooldown:{self.nav2_backup_cooldown_steps}"
            return False

        if bool(getattr(self, "_last_waypoint_reached", False)):
            self.nav2_stuck_steps = 0
            self._last_nav2_stuck_active = False
            return False

        if not bool(getattr(self, "_last_nav2_goal_accepted", False)):
            self.nav2_stuck_steps = 0
            self._last_nav2_stuck_active = False
            return False

        if float(getattr(self, "_last_waypoint_final_error", 0.0)) <= float(getattr(self, "nav2_goal_reached_tolerance", 0.24)):
            self.nav2_stuck_steps = 0
            self._last_nav2_stuck_active = False
            return False

        if bool(getattr(self, "_last_waypoint_timed_out", False)):
            # Timeout is handled by normal Nav2 replanning/restart logic.
            self.nav2_stuck_steps = 0
            self._last_nav2_stuck_active = False
            return False

        moved = float(getattr(self, "_last_nav2_moved_distance", 0.0))
        min_move = float(getattr(self, "nav2_stuck_backup_min_movement_m", 0.025))
        if moved >= min_move:
            self.nav2_stuck_steps = 0
            self._last_nav2_stuck_active = False
            self._last_nav2_stuck_reason = f"moving:{moved:.3f}m"
            return False

        allowed, gate_reason = self._nav2_backup_allowed_now(
            getattr(self, "_last_waypoint_world", None),
            moved_since_goal=moved,
            elapsed_wall=float(getattr(self, "nav2_stuck_backup_sec", 2.2)),
        )
        self._last_nav2_backup_gate_reason = gate_reason
        if not allowed:
            # Normal initial heading alignment often has near-zero xy movement.
            # Do not accumulate stuck steps or reverse unless the front is actually blocked.
            self.nav2_stuck_steps = 0
            self._last_nav2_stuck_active = False
            self._last_nav2_stuck_reason = f"backup_gated:{gate_reason}"
            return False

        self.nav2_stuck_steps += 1
        self._last_nav2_stuck_active = True
        limit = max(int(getattr(self, "nav2_stuck_backup_steps", 1)), 1)
        self._last_nav2_stuck_reason = (
            f"counting:{self.nav2_stuck_steps}/{limit},moved={moved:.3f}m,"
            f"err={float(getattr(self, '_last_waypoint_final_error', 0.0)):.3f}m"
        )

        if self.nav2_stuck_steps < limit:
            return False

        self._last_nav2_stuck_backup_triggered = True
        self._last_nav2_stuck_reason = (
            f"stuck:{self.nav2_stuck_steps}/{limit},moved={moved:.3f}m,"
            f"err={float(getattr(self, '_last_waypoint_final_error', 0.0)):.3f}m"
        )
        self.ros.get_logger().warn(
            "NAV2_STUCK_BACKUP_TRIGGER | "
            f"steps={self.nav2_stuck_steps}/{limit} | moved={moved:.3f}m | "
            f"min_move={min_move:.3f}m | err={float(getattr(self, '_last_waypoint_final_error', 0.0)):.3f}m"
        )
        executed = self._execute_nav2_backup_behavior(reason=self._last_nav2_stuck_reason)
        self.nav2_stuck_steps = 0
        self.nav2_backup_cooldown_steps = max(
            int(getattr(self, "nav2_stuck_backup_cooldown_steps_limit", 1)),
            1,
        )
        return bool(executed)

    def _update_lidar_empty_restart_state(self) -> bool:
        """Return True if LiDAR sees no valid obstacle/surface for too long."""
        self._last_lidar_empty_restart = False
        self._last_lidar_empty_reason = "none"

        if not bool(getattr(self, "lidar_empty_restart", True)):
            self.lidar_empty_steps = 0
            self._last_lidar_empty_active = False
            return False

        valid_beams, nearest = self._lidar_detection_stats()
        self._last_lidar_valid_beams = int(valid_beams)
        self._last_lidar_nearest_detection = float(nearest)

        # Ignore the initial sensor/teleport settling period.
        now_sim = self._safe_sim_time()
        sim_elapsed = now_sim - float(getattr(self, "_episode_start_sim_time", now_sim))
        wall_elapsed = time.time() - float(getattr(self, "_episode_start_wall_time", time.time()))
        elapsed = sim_elapsed if sim_elapsed >= 0.0 else wall_elapsed
        if elapsed < float(getattr(self, "lidar_empty_grace_sec", 1.0)):
            self.lidar_empty_steps = 0
            self._last_lidar_empty_active = False
            self._last_lidar_empty_reason = "grace"
            return False

        min_beams = max(int(getattr(self, "lidar_empty_min_valid_beams", 2)), 1)
        if valid_beams >= min_beams:
            self.lidar_empty_steps = 0
            self._last_lidar_empty_active = False
            self._last_lidar_empty_reason = f"detected:{valid_beams}"
            return False

        self.lidar_empty_steps += 1
        self._last_lidar_empty_active = True

        limit = max(int(getattr(self, "lidar_empty_timeout_steps", 1)), 1)
        if self.lidar_empty_steps >= limit:
            self._last_lidar_empty_restart = True
            self._last_lidar_empty_reason = (
                f"lidar_empty:{self.lidar_empty_steps}/{limit},"
                f"valid_beams={valid_beams},nearest={nearest:.2f}"
            )
            self.ros.get_logger().warn(
                "LIDAR_EMPTY_RESTART | "
                f"steps={self.lidar_empty_steps}/{limit} | "
                f"timeout={float(getattr(self, 'lidar_empty_timeout_sec', 0.0)):.2f}s | "
                f"valid_beams={valid_beams} | nearest={nearest:.3f}"
            )
            return True

        self._last_lidar_empty_reason = (
            f"counting:{self.lidar_empty_steps}/{limit},valid_beams={valid_beams}"
        )
        return False

    def _update_priority_stuck_restart_state(self, map_stats: MapUpdateStats) -> bool:
        """Return True when an active priority target stayed unresolved too long.

        This is deliberately different from "priority_score == 0".  Zero means
        there is no active priority candidate to solve, so the counter is reset.
        The counter increases only while active priority remains above threshold
        and neither priority clear/recheck nor meaningful information gain occurs.
        """
        self._last_priority_stuck_restart = False
        self._last_priority_stuck_reason = "none"

        if not bool(getattr(self, "priority_stuck_restart", True)):
            self.priority_stuck_steps = 0
            self._last_priority_stuck_active = False
            return False

        priority_score = float(np.clip(float(getattr(map_stats, "priority_score", 0.0)), 0.0, 1.0))
        target_priority = float(np.clip(float(getattr(map_stats, "target_priority", 0.0)), 0.0, 1.0))
        target_type = str(getattr(map_stats, "target_type", "none") or "none")
        active_strength = max(priority_score, target_priority)

        # 핵심: priority가 0이거나 threshold 미만이면 restart 대상이 아니다.
        if active_strength < self.priority_stuck_score_threshold:
            self.priority_stuck_steps = 0
            self._last_priority_stuck_active = False
            return False

        # priority_gap만 강제 restart 대상으로 본다. unknown/low_confidence는
        # 탐색 후보이지, 반드시 지워야 하는 구조적 priority가 아니다.
        if target_type not in {"priority_gap", "stale", "low_confidence"} and priority_score < self.priority_stuck_score_threshold:
            self.priority_stuck_steps = 0
            self._last_priority_stuck_active = False
            return False

        clear_gain = float(getattr(map_stats, "priority_clear_gain", 0.0))
        recheck_gain = float(getattr(map_stats, "priority_rechecked_gain", 0.0))
        cleared_cells = int(getattr(map_stats, "priority_cleared_cells", 0))
        rechecked_cells = int(getattr(map_stats, "priority_rechecked_cells", 0))

        # Small information gains mean the robot is still doing useful work, even
        # if the active priority has not disappeared yet.  Do not restart then.
        info_gain = max(
            float(getattr(map_stats, "coverage_delta", 0.0)),
            float(getattr(map_stats, "confidence_gain", 0.0)) / 100.0,
            float(getattr(map_stats, "stale_refresh_cells", 0)) / 200.0,
            float(getattr(map_stats, "new_known_cells", 0)) / 200.0,
        )

        resolved = (
            clear_gain >= self.priority_stuck_clear_gain_threshold
            or recheck_gain >= self.priority_stuck_clear_gain_threshold
            or cleared_cells > 0
            or rechecked_cells > 0
        )

        if resolved or info_gain >= self.priority_stuck_info_gain_threshold:
            self.priority_stuck_steps = 0
            self._last_priority_stuck_active = True
            self._last_priority_stuck_reason = "progress"
            return False

        self.priority_stuck_steps += 1
        self._last_priority_stuck_active = True

        if self.priority_stuck_steps >= self.priority_stuck_restart_steps:
            self._last_priority_stuck_restart = True
            self._last_priority_stuck_reason = (
                f"priority_unresolved:{self.priority_stuck_steps}/"
                f"{self.priority_stuck_restart_steps},score={active_strength:.2f}"
            )
            self.ros.get_logger().warn(
                "PRIORITY_STUCK_RESTART | "
                f"steps={self.priority_stuck_steps}/{self.priority_stuck_restart_steps} | "
                f"priority_score={priority_score:.3f} | target_priority={target_priority:.3f} | "
                f"target_type={target_type} | clear_gain={clear_gain:.3f} | "
                f"recheck_gain={recheck_gain:.3f} | info_gain={info_gain:.6f}"
            )
            return True

        self._last_priority_stuck_reason = (
            f"counting:{self.priority_stuck_steps}/{self.priority_stuck_restart_steps},"
            f"score={active_strength:.2f}"
        )
        return False

    def _backup_instead_of_priority_restart(self, reason: str) -> bool:
        """Try Nav2 BackUp before terminating an unresolved-priority episode."""
        if self.action_mode != "nav2" or not bool(getattr(self, "nav2_stuck_backup", True)):
            return False
        if int(getattr(self, "nav2_backup_cooldown_steps", 0)) > 0:
            return False
        try:
            self._cancel_nav2_goal_sync(reason=reason)
        except Exception:
            pass
        executed = self._execute_nav2_backup_behavior(reason=reason)
        if executed:
            self._last_nav2_stuck_backup_triggered = True
            self._last_nav2_stuck_active = True
            self._last_nav2_stuck_reason = reason
            self.nav2_stuck_steps = 0
            self.priority_stuck_steps = 0
            self.nav2_backup_cooldown_steps = max(
                int(getattr(self, "nav2_stuck_backup_cooldown_steps_limit", 1)),
                1,
            )
            try:
                self._clear_nav2_costmaps(wait_timeout_sec=0.35)
            except Exception:
                pass
            self.ros.get_logger().warn(
                "PRIORITY_STUCK_BACKUP_INSTEAD_OF_RESET | "
                f"reason={reason} | backup_status={self._last_nav2_backup_status}"
            )
            return True
        return False

    def step(self, action):
        """Crash-safe wrapper around the real step implementation.

        A transient ROS/Gazebo failure raised from inside the step (lost service,
        dropped /scan, world-control timeout, SHM hiccup, etc.) would otherwise
        propagate out of the Gym API and terminate the entire SB3 training run.
        Here we catch any unexpected exception, stop the robot, and end the
        episode via truncation so SB3 calls reset() and training continues.

        The normal (no-exception) path is unchanged: it returns exactly what the
        underlying implementation returns, and the model/observation/reward logic
        is never altered.
        """
        try:
            result = self._step_impl(action)
        except KeyboardInterrupt:
            raise
        except Exception as exc:
            try:
                self.ros.get_logger().error(
                    "STEP_RECOVERY | unexpected exception in step(); truncating episode "
                    f"to keep training alive | {type(exc).__name__}: {exc}"
                )
            except Exception:
                pass
            try:
                self.ros.stop_robot()
            except Exception:
                pass
            obs = self._safe_fallback_observation()
            info = {
                "step_recovery": True,
                "step_recovery_error": f"{type(exc).__name__}: {exc}",
                "TimeLimit.truncated": True,
            }
            try:
                self.step_count = int(getattr(self, "step_count", 0)) + 1
            except Exception:
                pass
            # terminated=False, truncated=True -> SB3 bootstraps then resets.
            return obs, 0.0, False, True, info

        # Cache the most recent valid observation for fallback reuse.
        try:
            self._last_valid_obs = result[0]
        except Exception:
            pass
        return result

    def _safe_fallback_observation(self):
        """Return the last valid observation, or a zero-sample of the space."""
        cached = getattr(self, "_last_valid_obs", None)
        if cached is not None:
            return cached
        try:
            return self.observation_space.sample() * 0
        except Exception:
            pass
        try:
            import numpy as _np
            space = self.observation_space
            if isinstance(space, spaces.Dict):
                return {
                    k: _np.zeros(v.shape, dtype=v.dtype)
                    for k, v in space.spaces.items()
                }
            return _np.zeros(space.shape, dtype=space.dtype)
        except Exception:
            return None

    def _step_impl(self, action):
        if not hasattr(self, "_step_profiler_initialized"):
            raw = str(os.environ.get("TB3_RL_STEP_PROFILER", "0")).strip().lower()
            self._step_profiler_enabled = raw in {"1", "true", "yes", "on"}
            try:
                self._step_profiler_every_n = max(int(os.environ.get("TB3_RL_STEP_PROFILER_EVERY_N", "50")), 1)
            except Exception:
                self._step_profiler_every_n = 50
            try:
                self._step_profiler_slow_ms = max(float(os.environ.get("TB3_RL_STEP_PROFILER_SLOW_MS", "700")), 0.0)
            except Exception:
                self._step_profiler_slow_ms = 700.0
            self._step_profiler_acc = {}
            self._step_profiler_count = 0
            self._step_profiler_initialized = True

        _prof_on = bool(getattr(self, "_step_profiler_enabled", False))
        if _prof_on:
            _prof_t0 = time.perf_counter()
            _prof_last = _prof_t0
            _prof_parts = {}

            def _prof_mark(name: str) -> None:
                nonlocal _prof_last
                now = time.perf_counter()
                _prof_parts[name] = _prof_parts.get(name, 0.0) + float(now - _prof_last)
                _prof_last = now
        else:
            _prof_t0 = 0.0
            _prof_parts = {}

            def _prof_mark(name: str) -> None:
                return

        policy_action = np.asarray(action, dtype=np.float32)
        policy_action = np.clip(policy_action, self.action_space.low, self.action_space.high)
        self.raw_action = policy_action.astype(np.float32).copy()
        self.prev_policy_action = self.raw_action.copy()
        # While Nav2 is moving toward this waypoint, the 10Hz live map timer may
        # clear/recheck priority cells.  Those event rewards belong to this SAC
        # action, so collect them until reward calculation drains the accumulator.
        self._live_priority_reward_collection_enabled = True
        action_prev_scan_wall = getattr(self.ros, "last_scan_time", None)
        action_prev_odom_wall = getattr(self.ros, "last_odom_time", None)

        if self.action_mode == "waypoint":
            executed_action, lidar_action_obstacle_distance, lidar_action_obstacle_score, lidar_front_obstacle_distance = (
                self._execute_waypoint_action(policy_action)
            )
        elif self.action_mode == "nav2":
            executed_action, lidar_action_obstacle_distance, lidar_action_obstacle_score, lidar_front_obstacle_distance = (
                self._execute_nav2_goal_action(policy_action)
            )
        else:
            executed_action, lidar_action_obstacle_distance, lidar_action_obstacle_score, lidar_front_obstacle_distance = (
                self._execute_velocity_action(policy_action)
            )

        nav2_stuck_backup_triggered = False
        if self.action_mode == "nav2":
            nav2_stuck_backup_triggered = self._update_nav2_stuck_backup_state()
            if nav2_stuck_backup_triggered:
                # After backing up, refresh obstacle estimates around the new pose.
                self.ros.spin_steps(num_spins=2, timeout_sec=0.02)
                d_obs, s_obs, f_obs = self._compute_lidar_action_obstacle_risk(np.zeros(2, dtype=np.float32))
                lidar_action_obstacle_distance = min(float(lidar_action_obstacle_distance), float(d_obs))
                lidar_action_obstacle_score = max(float(lidar_action_obstacle_score), float(s_obs))
                lidar_front_obstacle_distance = float(f_obs)

        _prof_mark("action_execute")

        # Reward와 observation에는 policy raw action이 아니라 실제 controller가 수행한 cmd_vel 평균을 넣는다.
        # 이렇게 해야 path alignment / obstacle penalty가 실제 궤적 기준으로 계산된다.
        action_for_reward = executed_action.astype(np.float32)
        self.filtered_action = action_for_reward.copy()

        self._last_lidar_action_obstacle_distance = float(lidar_action_obstacle_distance)
        self._last_lidar_action_obstacle_score = float(lidar_action_obstacle_score)
        self._last_lidar_front_obstacle_distance = float(lidar_front_obstacle_distance)

        # Do not compute the map-delta reward from stale pre-action sensors.
        # We wait briefly for a post-action scan/odom frame, then immediately
        # ray-cast the current scan into the internal confidence/priority map.
        # This is the B-plan sync path: reward uses scan/pose deltas, not delayed
        # RViz /map publication timing.
        self._wait_for_action_synced_observation(action_prev_scan_wall, action_prev_odom_wall)
        _prof_mark("wait_sync")

        map_stats = self._update_exploration_map()
        self._update_explored_stall_steps(map_stats=map_stats, action=action_for_reward)
        self._update_confidence_stall_steps(map_stats=map_stats, action=action_for_reward)
        self._update_sustained_rotation_steps(action=action_for_reward)
        self._update_orbit_stall_steps(map_stats=map_stats, action=action_for_reward)
        _prof_mark("map_update")

        collision = self._check_collision()
        fallen = self._check_fallen()
        boundary_out_of_bounds = self._check_out_of_bounds()
        map_bounds_restart = self._update_map_bounds_restart_state(map_stats)
        out_of_bounds = bool(boundary_out_of_bounds or map_bounds_restart)
        shake_restart = self._update_shake_restart_state()
        collision_like = bool(collision or out_of_bounds or shake_restart)

        coverage_done = map_stats.coverage_ratio >= self.target_coverage_ratio
        priority_stuck_restart = self._update_priority_stuck_restart_state(map_stats)
        lidar_empty_restart = self._update_lidar_empty_restart_state()
        coverage_stall_terminal = False

        self._last_terminal_reason = "none"
        self._last_collision_restart_requested = False
        if collision:
            # Physical contact / near-contact must always end the episode.
            # Do not gate this on restart_on_collision; the user expects a reset
            # and reward.py gives -200 when collision=True.
            self._last_terminal_reason = "collision"
            self._last_collision_restart_requested = True
            self._handle_unsafe_terminal("collision")
        elif out_of_bounds:
            if bool(getattr(self, "_last_map_bounds_restart", False)):
                self._last_terminal_reason = f"map_bounds_restart:{self._last_map_bounds_reason}"
            else:
                self._last_terminal_reason = "out_of_bounds"
            self._last_collision_restart_requested = True
            self._handle_unsafe_terminal(self._last_terminal_reason)
        elif fallen:
            # Roll/pitch fall or z-drop.  This is also an unsafe physical terminal
            # with reward=-100 and a mandatory reset on the next Gym reset().
            self._last_terminal_reason = "fallen_or_drop"
            self._last_collision_restart_requested = True
            self._handle_unsafe_terminal("fallen_or_drop")
        elif shake_restart:
            # The robot is not fully fallen yet, but roll/pitch/body velocity has
            # remained unstable for several control ticks.  Reset before the scan
            # and SLAM map are corrupted by a bouncing robot.
            self._last_terminal_reason = "shake_restart"
            self._last_collision_restart_requested = True
            self._handle_unsafe_terminal("shake_restart")
        elif priority_stuck_restart:
            # Active priority exists, but it was not cleared/rechecked for too long.
            # Before resetting the episode, try Nav2 BackUp once.  This keeps the
            # robot from getting reset immediately at door/wall-adjacent priority
            # blobs and gives Nav2 a chance to escape the local minimum.
            if self._backup_instead_of_priority_restart("priority_stuck_restart"):
                priority_stuck_restart = False
                self._last_terminal_reason = "priority_stuck_backup"
                self._last_collision_restart_requested = False
            else:
                self._last_terminal_reason = "priority_stuck_restart"
                self._last_collision_restart_requested = False
                try:
                    self._cancel_nav2_goal_sync(reason="priority_stuck_restart")
                except Exception:
                    pass
                try:
                    self._clear_nav2_costmaps()
                except Exception:
                    pass
                self.ros.stop_robot()
        elif lidar_empty_restart:
            # All LiDAR beams are no-hit/max-range for too long.  In the house
            # world this usually means bad spawn, sensor failure, or escaped useful geometry.
            self._last_terminal_reason = "lidar_empty_restart"
            self._last_collision_restart_requested = False
            try:
                self._cancel_nav2_goal_sync(reason="lidar_empty_restart")
            except Exception:
                pass
            try:
                self._clear_nav2_costmaps()
            except Exception:
                pass
            self.ros.stop_robot()
        elif coverage_done:
            self._last_terminal_reason = "coverage_done"

        _prof_mark("terminal_checks")

        # Merge all priority clear/recheck events produced by the live 10Hz map
        # update timer while this waypoint was being executed.  Without this,
        # only the final map_stats snapshot would be rewarded and intermediate
        # priority-clearing work would be lost.
        reward_map_stats = self._merge_and_drain_pending_priority_events(map_stats)
        self._live_priority_reward_collection_enabled = False

        # Reward의 action smoothness penalty는 실제로 수행된 controller command 기준으로 계산한다.
        prev_action_for_reward = self.prev_action.copy()

        reward = compute_exploration_reward(
            new_known_cells=reward_map_stats.new_known_cells,
            coverage_delta=reward_map_stats.coverage_delta,
            coverage_ratio=reward_map_stats.coverage_ratio,
            frontier_count=reward_map_stats.frontier_count,
            robot_visit_count=reward_map_stats.robot_visit_count,
            action=action_for_reward,
            prev_action=prev_action_for_reward,
            collision=collision_like,
            fallen=fallen,
            stale_refresh_cells=reward_map_stats.stale_refresh_cells,
            confidence_gain=reward_map_stats.confidence_gain,
            mean_confidence=reward_map_stats.mean_confidence,
            stale_ratio=reward_map_stats.stale_ratio,
            low_confidence_ratio=reward_map_stats.low_confidence_ratio,
            target_priority=reward_map_stats.target_priority,
            frontier_angle=reward_map_stats.frontier_angle,
            target_type=reward_map_stats.target_type,
            target_switched=reward_map_stats.target_switched,
            target_lock_age=reward_map_stats.target_lock_age,
            target_reachable=reward_map_stats.target_reachable,
            path_distance=reward_map_stats.path_distance,
            path_angle=reward_map_stats.path_angle,
            path_progress=reward_map_stats.path_progress,
            alternative_path_angles=reward_map_stats.alternative_path_angles,
            priority_score=reward_map_stats.priority_score,
            priority_gain=reward_map_stats.priority_gain,
            priority_cleared_cells=reward_map_stats.priority_cleared_cells,
            priority_clear_gain=reward_map_stats.priority_clear_gain,
            priority_rechecked_cells=int(getattr(reward_map_stats, "priority_rechecked_cells", 0)),
            priority_rechecked_gain=float(getattr(reward_map_stats, "priority_rechecked_gain", 0.0)),
            wall_support_score=reward_map_stats.wall_support_score,
            open_space_score=reward_map_stats.open_space_score,
            open_space_forward_penalty=self.open_space_forward_penalty,
            explored_stall_steps=self.explored_stall_steps,
            explored_stall_start_steps=self.explored_stall_start_steps,
            explored_stall_growth=self.explored_stall_growth,
            explored_stall_power=self.explored_stall_power,
            explored_stall_max_penalty=self.explored_stall_max_penalty,
            confidence_stall_steps=self.confidence_stall_steps,
            confidence_stall_start_steps=self.confidence_stall_start_steps,
            confidence_stall_growth=self.confidence_stall_growth,
            confidence_stall_power=self.confidence_stall_power,
            confidence_stall_max_penalty=self.confidence_stall_max_penalty,
            confidence_stall_gain_threshold=self.confidence_stall_gain_threshold,
            confidence_stall_low_ratio_threshold=self.confidence_stall_low_ratio_threshold,
            sustained_rotation_steps=self.sustained_rotation_steps,
            orbit_stall_steps=int(getattr(self, "orbit_stall_steps", 0)),
            orbit_path_efficiency=float(getattr(self, "_last_orbit_path_efficiency", 1.0)),
            orbit_path_length=float(getattr(self, "_last_orbit_path_length", 0.0)),
            orbit_yaw_accum=float(getattr(self, "_last_orbit_yaw_accum", 0.0)),
            max_linear_speed=self.max_linear_speed,
            max_angular_speed=self.max_angular_speed,
            nearest_obstacle_distance=reward_map_stats.nearest_obstacle_distance,
            obstacle_proximity_score=reward_map_stats.obstacle_proximity_score,
            lidar_action_obstacle_distance=lidar_action_obstacle_distance,
            lidar_action_obstacle_score=lidar_action_obstacle_score,
            lidar_front_obstacle_distance=lidar_front_obstacle_distance,
            use_path_reward=not self.disable_path_reward,
            use_wall_proximity_penalty=not self.disable_wall_proximity_penalty,
            use_corridor_priority_reward=self.enable_corridor_priority_reward,
            corridor_priority_reward_weight=self.corridor_priority_reward_weight,
            confidence_reward_weight=self.confidence_reward_weight,
        )
        _prof_mark("reward_compute")

        # Delayed SLAM /map update bonus.  This is intentionally separate from
        # the immediate LiDAR/confidence reward: it credits unknown->known cells
        # that arrive later through slam_toolbox, but it is small/capped because
        # exact action credit is less precise than the post-action scan reward.
        reward += self._compute_delayed_slam_map_update_reward(
            reward_map_stats,
            unsafe_terminal=bool(collision_like or fallen or lidar_empty_restart),
        )

        # v113: terminate low-information episode tails. This runs after the
        # delayed SLAM map update has been merged into reward_map_stats, so the
        # decision sees both SLAM growth and confidence growth for the action
        # that just executed.
        if not bool(collision_like or fallen or coverage_done or priority_stuck_restart or lidar_empty_restart):
            coverage_stall_terminal = self._update_coverage_stall_terminal_state(reward_map_stats)
            if coverage_stall_terminal:
                reward += float(getattr(self, "coverage_stall_terminal_penalty", -1.0))
                self._last_terminal_reason = "coverage_stall_terminal"
                self._last_collision_restart_requested = False
                try:
                    self.ros.stop_robot()
                except Exception:
                    pass
        else:
            # Keep diagnostics current, but do not let this soft terminal
            # override hard safety/success terminals.
            self._update_coverage_stall_terminal_state(reward_map_stats)
            coverage_stall_terminal = False

        if self.action_mode in {"waypoint", "nav2"}:
            reward += compute_waypoint_macro_reward_adjustment(
                collision=collision_like,
                fallen=fallen,
                target_reachable=map_stats.target_reachable,
                path_progress=map_stats.path_progress,
                waypoint_reached=self._last_waypoint_reached,
                waypoint_timed_out=self._last_waypoint_timed_out,
                waypoint_distance=self._last_waypoint_distance,
                waypoint_final_error=self._last_waypoint_final_error,
                waypoint_reached_tolerance=self.waypoint_reached_tolerance,
                controller_steps=self._last_controller_steps,
                waypoint_max_control_steps=self.waypoint_max_control_steps,
                waypoint_heading_delta=self._last_waypoint_heading_delta,
                waypoint_lateral_offset=self._last_waypoint_lateral_offset,
                waypoint_lateral_max_offset=self.waypoint_lateral_max_offset,
                waypoint_path_conditioned=(self.waypoint_action_type == "path"),
                use_path_reward=not self.disable_path_reward,
            )

        if priority_stuck_restart:
            reward -= self.priority_stuck_restart_penalty
        if nav2_stuck_backup_triggered:
            reward -= float(getattr(self, "nav2_stuck_backup_penalty", 0.0))
        if self.action_mode == "velocity" and float(getattr(self, "_last_velocity_safety_penalty", 0.0)) > 0.0:
            # v117: a forced safety backup/hold step is dropped from the replay
            # buffer (info.skip_store), so applying its penalty to THIS step's
            # reward would discard the penalty too.  Instead, defer it: accumulate
            # the penalty and subtract it from the next stored (non-skipped) step,
            # so the policy still pays for driving into the wall, but SAC never
            # learns from the forced-motion transition itself.
            if bool(getattr(self, "_last_velocity_safety_skip_store", False)):
                self._pending_skip_penalty = float(getattr(self, "_pending_skip_penalty", 0.0)) + float(getattr(self, "_last_velocity_safety_penalty", 0.0))
            else:
                # Non-skipped safety event (e.g. slowdown): apply immediately.
                reward -= float(getattr(self, "_last_velocity_safety_penalty", 0.0))
        # Drain any deferred backup penalty onto this step if it will be stored.
        if self.action_mode == "velocity" and not bool(getattr(self, "_last_velocity_safety_skip_store", False)):
            _pending = float(getattr(self, "_pending_skip_penalty", 0.0))
            if _pending > 0.0:
                reward -= _pending
                self._pending_skip_penalty = 0.0
        if shake_restart:
            reward -= float(getattr(self, "shake_restart_penalty", 100.0))
        if lidar_empty_restart:
            reward = -float(self.lidar_empty_restart_penalty)

        # v119: compress positive total reward after all positive bonuses and
        # soft penalties have been combined.  Negative safety/terminal rewards
        # are left unchanged by _apply_positive_reward_log_compression().
        reward = self._apply_positive_reward_log_compression(reward)
        _prof_mark("reward_post")

        # v6: Time-limit/episode timeout is a neutral truncation, not a penalty.
        # Collision/fallen/drop/lidar-empty/stuck restarts can still terminate with
        # their own penalties, but simply reaching max_episode_steps must not inject
        # an additional negative reward.  This keeps long but safe exploration runs
        # from being punished only because the episode horizon expired.
        will_truncate = bool((self.step_count + 1) >= self.max_episode_steps)
        if will_truncate and not (collision_like or fallen or coverage_done or coverage_stall_terminal or priority_stuck_restart or lidar_empty_restart):
            if self._last_terminal_reason == "none":
                self._last_terminal_reason = "time_limit"

        if bool(getattr(self, "disable_priority_map", False)):
            priority_clear_reward = priority_recheck_reward = priority_check_reward = 0.0
        else:
            priority_clear_reward, priority_recheck_reward, priority_check_reward = self._estimate_priority_check_reward_component(reward_map_stats)
        self._last_priority_clear_reward = float(priority_clear_reward)
        self._last_priority_recheck_reward = float(priority_recheck_reward)
        self._last_priority_check_reward = float(priority_check_reward)
        self._episode_priority_clear_reward += float(priority_clear_reward)
        self._episode_priority_recheck_reward += float(priority_recheck_reward)
        self._episode_priority_check_reward += float(priority_check_reward)

        reward = float(reward)
        self._last_step_reward = reward
        self._episode_reward_sum += reward

        # v112: use a live/recent discounted return for RViz/debug G.
        # Old code used: G += gamma**step * r, which becomes visually flat in
        # long episodes because gamma**step underflows toward zero.
        gamma_dbg = float(np.clip(float(getattr(self, "reward_gamma", 0.99)), 0.0, 1.0))
        self._episode_start_discounted_return += (gamma_dbg ** int(self.step_count)) * reward
        self._episode_discounted_return = gamma_dbg * float(getattr(self, "_episode_discounted_return", 0.0)) + reward
        beta_dbg = float(np.clip(float(getattr(self, "_reward_ema_beta", gamma_dbg)), 0.0, 0.999999))
        self._episode_reward_ema = beta_dbg * float(getattr(self, "_episode_reward_ema", 0.0)) + (1.0 - beta_dbg) * reward
        self._recent_reward_window.append(float(reward))
        try:
            self._recent_slam_new_window.append(int(max(0, getattr(reward_map_stats, "slam_update_new_known_cells", 0))))
        except Exception:
            self._recent_slam_new_window.append(0)
        try:
            self._recent_conf_update_window.append(int(max(0, getattr(reward_map_stats, "confidence_updated_cells", 0))))
        except Exception:
            self._recent_conf_update_window.append(0)
        self._last_reward_text_valid = True

        # Reward is only known after map update + reward.py evaluation.
        # Republish the same local waypoint marker once more with the current
        # step reward so RViz debug text matches the action that just executed.
        if self.action_mode in {"waypoint", "nav2"} and self.waypoint_marker_pub is not None:
            if (
                self._last_waypoint_local is not None
                and self._last_waypoint_world is not None
                and np.all(np.isfinite(self._last_waypoint_local[:2]))
                and np.all(np.isfinite(self._last_waypoint_world[:2]))
            ):
                self._publish_waypoint_visualization(
                    waypoint_world_xy=self._last_waypoint_world,
                    waypoint_local_xy=self._last_waypoint_local,
                    distance=float(self._last_waypoint_final_error),
                    heading=float(self._last_waypoint_angle),
                    append_history=False,
                    reward_value=reward,
                    episode_reward=self._episode_discounted_return,
                )
        elif self.action_mode == "velocity" and self.waypoint_marker_pub is not None:
            self._publish_velocity_debug_overlay(
                reward_value=reward,
                episode_reward=self._episode_discounted_return,
                map_stats=reward_map_stats,
                raw_action=policy_action,
                executed_action=action_for_reward,
                lidar_action_obstacle_distance=lidar_action_obstacle_distance,
                lidar_action_obstacle_score=lidar_action_obstacle_score,
                lidar_front_obstacle_distance=lidar_front_obstacle_distance,
            )
        _prof_mark("debug_publish")

        self.step_count += 1

        # collision/fallen/drop are hard terminals.  restart_on_collision is kept
        # only for backward-compatible logging/config, not for suppressing reset.
        terminated = bool(collision or out_of_bounds or fallen or shake_restart or coverage_done or coverage_stall_terminal or priority_stuck_restart or lidar_empty_restart)
        truncated = bool(self.step_count >= self.max_episode_steps)
        if truncated and self._last_terminal_reason == "none":
            self._last_terminal_reason = "time_limit"

        self.prev_action = action_for_reward.copy()
        self.last_map_stats = map_stats

        obs = self._get_obs()
        _prof_mark("obs_build")

        if terminated or truncated:
            self.ros.stop_robot()
            self._warn_episode_reset_reason(
                reason=self._last_terminal_reason,
                reward=reward,
                terminated=terminated,
                truncated=truncated,
                map_stats=reward_map_stats,
            )

        info = self._build_info(
            map_stats=reward_map_stats,
            collision=collision,
            fallen=fallen,
            coverage_done=coverage_done,
        )
        _prof_mark("info_build")

        if _prof_on:
            total = float(time.perf_counter() - _prof_t0)
            _prof_parts["total"] = total
            self._step_profiler_count = int(getattr(self, "_step_profiler_count", 0)) + 1
            acc = getattr(self, "_step_profiler_acc", {})
            for k, v in _prof_parts.items():
                acc[k] = float(acc.get(k, 0.0)) + float(v)
            self._step_profiler_acc = acc
            every_n = max(int(getattr(self, "_step_profiler_every_n", 50)), 1)
            slow_ms = float(getattr(self, "_step_profiler_slow_ms", 700.0))
            should_log = (self._step_profiler_count % every_n == 0) or (total * 1000.0 >= slow_ms)
            if should_log:
                order = [
                    "action_execute", "wait_sync", "map_update", "terminal_checks",
                    "reward_compute", "reward_post", "debug_publish", "obs_build",
                    "info_build", "total",
                ]
                last_txt = " ".join(
                    f"{k}={float(_prof_parts.get(k, 0.0))*1000.0:.1f}ms" for k in order if k in _prof_parts
                )
                avg_txt = " ".join(
                    f"{k}={float(acc.get(k, 0.0))/max(self._step_profiler_count,1)*1000.0:.1f}ms"
                    for k in order if k in acc
                )
                self.ros.get_logger().warn(
                    "STEP_PROFILE | "
                    f"episode={self.episode_index} step={self.step_count} count={self._step_profiler_count} "
                    f"term={self._last_terminal_reason} collision={collision} truncated={truncated} | "
                    f"last {last_txt} | avg {avg_txt}"
                )

        return obs, float(reward), terminated, truncated, info

    def _update_coverage_stall_terminal_state(self, map_stats: MapUpdateStats) -> bool:
        """Return True when the episode tail has stopped producing new information.

        The terminal is based on raw progress counts, not G/Glive, critic loss,
        or total reward.  It fires only after warmup and after a full window is
        available:

            recent_slam_new <= threshold AND recent_conf_updated <= threshold

        One progress source being alive is enough to keep the episode running.
        """
        self._last_coverage_stall_terminal = False
        self._last_coverage_stall_active = False
        self._last_coverage_stall_reason = "disabled"

        slam_new = max(int(getattr(map_stats, "slam_update_new_known_cells", 0)), 0)
        conf_new = max(int(getattr(map_stats, "confidence_updated_cells", 0)), 0)

        if not bool(getattr(self, "coverage_stall_terminal", False)):
            self._last_coverage_stall_slam_window = int(sum(getattr(self, "_coverage_stall_slam_window", [])))
            self._last_coverage_stall_conf_window = int(sum(getattr(self, "_coverage_stall_conf_window", [])))
            self._last_coverage_stall_window_len = int(len(getattr(self, "_coverage_stall_slam_window", [])))
            return False

        self._coverage_stall_slam_window.append(int(slam_new))
        self._coverage_stall_conf_window.append(int(conf_new))

        slam_sum = int(sum(self._coverage_stall_slam_window))
        conf_sum = int(sum(self._coverage_stall_conf_window))
        win_len = int(min(len(self._coverage_stall_slam_window), len(self._coverage_stall_conf_window)))
        self._last_coverage_stall_slam_window = slam_sum
        self._last_coverage_stall_conf_window = conf_sum
        self._last_coverage_stall_window_len = win_len

        step_after = int(getattr(self, "step_count", 0)) + 1
        start_steps = int(getattr(self, "coverage_stall_start_steps", 0))
        window_steps = int(getattr(self, "coverage_stall_window_steps", 1))

        if step_after < start_steps:
            self._last_coverage_stall_reason = f"warmup:{step_after}/{start_steps}"
            return False
        if win_len < window_steps:
            self._last_coverage_stall_reason = f"window:{win_len}/{window_steps}"
            return False

        # v114: do not end an episode as coverage-stalled while an emergency
        # priority spot is still active.  Priority is now an explicit task signal,
        # so an unresolved active spot is progress-relevant even if SLAM/confidence
        # are flat.
        try:
            if not bool(getattr(self, "disable_priority_map", False)):
                prio = self.exploration_map._active_priority_grid()
                prio_thr = max(float(getattr(self.exploration_map, "priority_clear_min_value", 5.0)), 1.0)
                active_prio_cells = int(np.count_nonzero(np.asarray(prio) >= prio_thr))
                if active_prio_cells > 0:
                    self._last_coverage_stall_reason = f"priority_active:{active_prio_cells}"
                    return False
        except Exception:
            pass

        min_slam = int(getattr(self, "coverage_stall_min_slam_new_cells", 0))
        min_conf = int(getattr(self, "coverage_stall_min_confidence_updated_cells", 0))
        active = bool(slam_sum <= min_slam and conf_sum <= min_conf)
        self._last_coverage_stall_active = active
        self._last_coverage_stall_terminal = active
        self._last_coverage_stall_reason = (
            f"stall:slam={slam_sum}<={min_slam},conf={conf_sum}<={min_conf}"
            if active else
            f"progress:slam={slam_sum}>{min_slam} or conf={conf_sum}>{min_conf}"
        )
        return active

    def _execute_velocity_safety_backup_sequence(
        self,
        *,
        front_min: float,
        rear_min: float,
        turn_sign: float,
        backup_v: float,
        backup_turn_speed: float,
        backup_steps_cfg: int,
        rear_stop_dist: float,
        lidar_action_obstacle_distance: float,
        lidar_action_obstacle_score: float,
        lidar_front_obstacle_distance: float,
        reason_prefix: str,
        log_label: str,
    ) -> tuple[np.ndarray, float, float, float]:
        """Execute the full safety backup as one environment transition.

        This is intentionally synchronous.  Standard SB3 SAC samples an action
        before every env.step(), so a sticky backup implemented over many Gym
        steps would still call actor.forward() and store arbitrary policy actions
        for forced-backup motion.  Running all backup ticks inside this one step
        keeps replay cleaner: one unsafe policy action -> one forced recovery
        transition -> one explicit penalty.
        """
        backup_steps_cfg = max(int(backup_steps_cfg), 1)
        turn_sign = float(turn_sign)
        if abs(turn_sign) < 1e-6:
            turn_sign = 1.0
        # v117: pure straight reverse only.  Mixing an angular term into the
        # backup made the robot pivot while reversing, which both (a) changed the
        # heading the policy is trying to learn forward motion for and (b) could
        # swing the front back toward the wall.  The user wants "just back up a
        # little", so the backup twist has zero angular component.
        backup_w = 0.0
        backup_v = -abs(float(backup_v))
        cmd = np.array([backup_v, backup_w], dtype=np.float32)
        rear_during_backup = float(rear_min)
        stopped_by_rear = False
        steps_done = 0

        def _refresh_confidence_during_backup():
            # confidence/priority maps are normally refreshed by the ROS live-update
            # timer, but the backup loop only spins briefly, so the timer barely
            # fires and the confidence origin lags behind the reversing robot.
            # Drive the same update explicitly each micro-step so the overlay keeps
            # up with the backup motion.
            try:
                self.ros.spin_steps(num_spins=4, timeout_sec=0.002)
                if not bool(getattr(self, "_map_live_update_paused", False)):
                    self._live_map_update_timer_callback()
            except Exception:
                pass

        for _ in range(backup_steps_cfg):
            self.ros.spin_steps(num_spins=2, timeout_sec=0.001)
            scan_now = self.ros.scan
            if scan_now is not None:
                rear_during_backup = self._scan_min_distance_in_sector(
                    scan=scan_now,
                    center_angle=math.pi,
                    half_width_rad=math.radians(45.0),
                    max_considered_range=0.90,
                )

            if float(rear_during_backup) <= float(rear_stop_dist):
                stopped_by_rear = True
                cmd = np.array([0.0, backup_w], dtype=np.float32)
                self._last_velocity_safety_executed_v = float(cmd[0])
                self.ros.publish_cmd_vel(float(cmd[0]), float(cmd[1]))
                prev_scan_wall = getattr(self.ros, "last_scan_time", None)
                self.ros.spin_steps(num_spins=2, timeout_sec=0.001)
                self._advance_world_after_command(target_delta_sec=self.control_dt)
                try:
                    self.ros.wait_for_new_sensor_frame(prev_scan_wall, None, timeout_wall_sec=0.04)
                except Exception:
                    pass
                _refresh_confidence_during_backup()
                break

            cmd = np.array([backup_v, backup_w], dtype=np.float32)
            prev_scan_wall = getattr(self.ros, "last_scan_time", None)
            self.ros.publish_cmd_vel(float(cmd[0]), float(cmd[1]))
            self.ros.spin_steps(num_spins=2, timeout_sec=0.001)
            self._advance_world_after_command(target_delta_sec=self.control_dt)
            try:
                self.ros.wait_for_new_sensor_frame(prev_scan_wall, None, timeout_wall_sec=0.04)
            except Exception:
                pass
            _refresh_confidence_during_backup()
            steps_done += 1

        # CRITICAL: explicitly stop the robot at the end of the backup.
        # cmd_vel is latched by the driver, so if we leave the last reverse
        # command active the robot keeps coasting backward through the cooldown
        # window and the next world-step advances, which is what produced the
        # runaway multi-meter reverse.  Force a zero twist and settle it.
        cmd = np.array([0.0, 0.0], dtype=np.float32)
        prev_scan_wall = getattr(self.ros, "last_scan_time", None)
        self.ros.publish_cmd_vel(0.0, 0.0)
        self.ros.spin_steps(num_spins=2, timeout_sec=0.001)
        self._advance_world_after_command(target_delta_sec=self.control_dt)
        try:
            self.ros.wait_for_new_sensor_frame(prev_scan_wall, None, timeout_wall_sec=0.06)
        except Exception:
            pass
        try:
            self.ros.stop_robot()
        except Exception:
            pass
        _refresh_confidence_during_backup()

        front_after_backup = float(front_min)
        try:
            scan_after = self.ros.scan
            if scan_after is not None:
                front_after_backup = self._scan_min_distance_in_sector(
                    scan=scan_after,
                    center_angle=0.0,
                    half_width_rad=math.radians(28.0),
                    max_considered_range=max(1.20, self.velocity_safety_slow_distance_m + 0.20),
                )
        except Exception:
            pass

        self.velocity_safety_backup_lock_steps = 0
        self.velocity_safety_cooldown_steps = int(getattr(self, "velocity_safety_cooldown_steps_cfg", 8))
        self._last_velocity_safety_backup_triggered = True
        self._last_velocity_safety_backup_lock_active = False
        self._last_velocity_safety_skip_store = True
        self._last_velocity_safety_penalty = float(getattr(self, "velocity_safety_penalty", 10.0))
        self._last_velocity_safety_sync_steps = int(steps_done)
        self._last_velocity_safety_reason = (
            f"{reason_prefix}{'_rear_stop' if stopped_by_rear else ''}:"
            f"front0={float(front_min):.3f},front1={float(front_after_backup):.3f},"
            f"rear0={float(rear_min):.3f},rear={float(rear_during_backup):.3f},"
            f"sync_steps={int(steps_done)}/{int(backup_steps_cfg)}"
        )
        self.ros.get_logger().warn(
            f"{log_label} | "
            f"front0={float(front_min):.3f}m front1={float(front_after_backup):.3f}m "
            f"rear0={float(rear_min):.3f}m rear={float(rear_during_backup):.3f}m "
            f"sync_steps={int(steps_done)}/{int(backup_steps_cfg)} "
            f"cmd_last=({float(cmd[0]):+.3f},{float(cmd[1]):+.3f}) "
            f"rear_stop={float(rear_stop_dist):.3f} "
            f"penalty={float(self._last_velocity_safety_penalty):.2f} "
            f"actor_forward_skipped_during_backup=1"
        )
        return (
            cmd.astype(np.float32),
            float(lidar_action_obstacle_distance),
            float(max(lidar_action_obstacle_score, 1.0)),
            float(lidar_front_obstacle_distance),
        )

    def _execute_velocity_action(self, policy_action: np.ndarray) -> tuple[np.ndarray, float, float, float]:
        """
        Direct pure-velocity SAC executor.

        Policy action meaning:
          action[0] = commanded forward linear velocity in [0, max_linear_speed]
          action[1] = commanded angular velocity in [-max_angular_speed, max_angular_speed]

        The safety shield does not change the observation/action space.  When an
        unsafe forward command would collide, v115 executes the whole reverse
        recovery inside this single Gym step.  That prevents SB3 from calling the
        actor and adding replay-buffer transitions during the individual backup
        ticks.  The replay buffer sees one transition: the unsafe action caused a
        synchronous forced backup and a safety penalty.
        Collisions are still handled by _check_collision() and terminate/reset the
        episode exactly as before.
        """
        action = np.asarray(policy_action, dtype=np.float32).copy()
        action = np.clip(
            action,
            np.array([0.0, -self.max_angular_speed], dtype=np.float32),
            np.array([self.max_linear_speed, self.max_angular_speed], dtype=np.float32),
        )

        self._last_velocity_safety_backup_triggered = False
        self._last_velocity_safety_blocked = False
        self._last_velocity_safety_skip_store = False
        self._last_velocity_safety_slowdown = 1.0
        self._last_velocity_safety_slowdown_risk = 0.0
        self._last_velocity_safety_policy_v = float(action[0]) if action.size > 0 else 0.0
        self._last_velocity_safety_executed_v = float(action[0]) if action.size > 0 else 0.0
        self._last_velocity_safety_penalty = 0.0
        self._last_velocity_safety_reason = "none"
        self._last_velocity_safety_sync_steps = 0
        self._last_velocity_forward_assist = False
        self._last_velocity_spin_breaker = False

        # Cooldown prevents repeated new backup starts.  It must not consume the
        # already-started sticky backup lock; otherwise the policy can re-enter
        # and cancel the escape before the robot has physically moved backward.
        if int(getattr(self, "velocity_safety_backup_lock_steps", 0)) <= 0:
            if int(getattr(self, "velocity_safety_cooldown_steps", 0)) > 0:
                self.velocity_safety_cooldown_steps = max(
                    int(getattr(self, "velocity_safety_cooldown_steps", 0)) - 1,
                    0,
                )

        # Drain callbacks before reading safety distances.  The previous build
        # could reuse a LaserScan from before/inside the synchronous backup, so
        # front_min looked pinned (e.g. 0.206/0.224m) and the next policy step
        # immediately triggered another backup.  Safety decisions below are made
        # from the freshest live LaserScan available at this point.
        self.ros.spin_steps(num_spins=4, timeout_sec=0.001)
        scan = self.ros.scan
        front_min = 999.0
        rear_min = 999.0
        left_min = 999.0
        right_min = 999.0
        if scan is not None:
            front_min = self._scan_min_distance_in_sector(
                scan=scan,
                center_angle=0.0,
                half_width_rad=math.radians(28.0),
                max_considered_range=max(1.20, self.velocity_safety_slow_distance_m + 0.20),
            )
            rear_min = self._scan_min_distance_in_sector(
                scan=scan,
                center_angle=math.pi,
                half_width_rad=math.radians(35.0),
                max_considered_range=0.85,
            )
            left_min = self._scan_min_distance_in_sector(
                scan=scan,
                center_angle=math.radians(45.0),
                half_width_rad=math.radians(35.0),
                max_considered_range=0.90,
            )
            right_min = self._scan_min_distance_in_sector(
                scan=scan,
                center_angle=math.radians(-45.0),
                half_width_rad=math.radians(35.0),
                max_considered_range=0.90,
            )

        scan_front_min = float(front_min)
        scan_rear_min = float(rear_min)

        live_action_obstacle_distance, live_action_obstacle_score, live_front_obstacle_distance = (
            self._compute_lidar_action_obstacle_risk(action)
        )
        slam_action_distance, slam_front_distance = self._compute_slam_action_obstacle_distance(action)

        # Keep SLAM/map obstacle distances for reward diagnostics, but do NOT use
        # them to trigger forced safety backup.  SLAM/map distance can lag after a
        # reset/backup because it depends on map pose and occupancy-grid latency;
        # using min(scan, slam) for the backup condition caused repeated reverse
        # commands even when the live front scan had already cleared.
        safety_front_min = float(scan_front_min)
        safety_action_obstacle_distance = float(live_action_obstacle_distance)
        safety_rear_min = float(scan_rear_min)

        lidar_action_obstacle_distance = min(float(live_action_obstacle_distance), float(slam_action_distance))
        lidar_front_obstacle_distance = min(float(live_front_obstacle_distance), float(scan_front_min), float(slam_front_distance))
        front_min = float(scan_front_min)
        if float(lidar_action_obstacle_distance) < 0.60:
            warn_distance = 0.60
            hard_distance = 0.22
            obstacle_risk = (warn_distance - float(lidar_action_obstacle_distance)) / max(warn_distance - hard_distance, 1e-6)
            lidar_action_obstacle_score = max(float(live_action_obstacle_score), float(np.clip(obstacle_risk, 0.0, 1.0)))
        else:
            lidar_action_obstacle_score = float(live_action_obstacle_score)

        cmd = action.astype(np.float32).copy()
        self._last_velocity_command_limited = False
        self._last_velocity_command_limit_reason = "none"
        linear_limit = float(getattr(self, "velocity_command_linear_limit", self.max_linear_speed))
        angular_limit = float(getattr(self, "velocity_command_angular_limit", self.max_angular_speed))
        if linear_limit < float(self.max_linear_speed) - 1e-6 or angular_limit < float(self.max_angular_speed) - 1e-6:
            before = cmd.copy()
            cmd[0] = float(np.clip(float(cmd[0]), 0.0, linear_limit))
            cmd[1] = float(np.clip(float(cmd[1]), -angular_limit, angular_limit))
            if abs(float(before[0]) - float(cmd[0])) > 1e-6 or abs(float(before[1]) - float(cmd[1])) > 1e-6:
                self._last_velocity_command_limited = True
                self._last_velocity_command_limit_reason = (
                    f"cmd_limit old=({float(before[0]):+.3f},{float(before[1]):+.3f}) "
                    f"limit=({linear_limit:.3f},{angular_limit:.3f})"
                )
        forward = float(cmd[0])
        angular = float(cmd[1])
        # Safety backup must be causally tied to a meaningful forward policy command.
        # A near-zero positive v from SAC exploration/noise should not repeatedly
        # trigger forced backup transitions near a wall.  Those forced transitions
        # are skip_store=True, but they still consume wall-clock time and can trap
        # the robot in backup loops.
        forward_backup_min = max(float(getattr(self, "linear_deadband", 0.015)), 0.04)
        forward_backup_requested = bool(forward > forward_backup_min)
        turn_sign = 1.0 if float(left_min) >= float(right_min) else -1.0
        if abs(angular) > 1e-4:
            # Prefer the policy's turn direction if it already chose one.
            turn_sign = 1.0 if angular > 0.0 else -1.0

        trigger_dist = float(getattr(self, "velocity_safety_trigger_distance_m", 0.28))
        stop_dist = float(getattr(self, "velocity_safety_stop_distance_m", 0.36))
        slow_dist = float(getattr(self, "velocity_safety_slow_distance_m", 0.55))
        # Rear safety for forced backup.  The backup escape is allowed only when
        # the rear sector has enough clearance.  This is checked once before the
        # escape and again at every reverse micro-step, so the robot does not back
        # into a wall or object while trying to avoid a front obstacle.
        rear_stop_dist = max(float(self.collision_threshold) + 0.14, 0.30)
        rear_warn_dist = max(rear_stop_dist + 0.10, 0.42)

        backup_v_cfg = float(getattr(self, "velocity_safety_backup_speed_mps", 0.08))
        backup_v = -min(max(backup_v_cfg, 0.0), 0.10)
        backup_turn_speed = float(getattr(self, "velocity_safety_turn_speed", 0.35))
        backup_steps_cfg = max(int(getattr(self, "velocity_safety_backup_steps", 4)), 1)

        # v115: backup is synchronous inside the triggering Gym step.
        # If a leftover lock exists from an older run/checkpoint, clear it so
        # no future actor-forward step is consumed by forced backup.
        lock_steps = int(getattr(self, "velocity_safety_backup_lock_steps", 0))
        if lock_steps > 0:
            self.velocity_safety_backup_lock_steps = 0
            self.velocity_safety_cooldown_steps = int(getattr(self, "velocity_safety_cooldown_steps_cfg", 8))
            self._last_velocity_safety_reason = f"cleared_legacy_backup_lock:{lock_steps}"

        # Safety-training policy:
        #   - Do not use cooldown to force cmd=(0,0) holds.
        #   - If rear is clear, execute backup immediately as a skip-store
        #     recovery transition.
        #   - If backup is unavailable, let the policy command pass through so
        #     the normal collision/terminal reward, not a hidden command clamp,
        #     provides the learning signal.
        can_backup = (
            bool(getattr(self, "velocity_safety_backup", True))
            and float(safety_rear_min) > rear_warn_dist
        )

        cooldown_active = int(getattr(self, "velocity_safety_cooldown_steps", 0)) > 0
        rear_ok = float(safety_rear_min) > rear_warn_dist

        # Continuous danger score for the soft safety layer.
        #   0.0: outside slow band
        #   1.0: at/inside stop band
        # We use the minimum of live front-scan and action-arc obstacle distance,
        # so straight frontal collision and curved-action collision are both covered.
        safety_slow_reference_distance = min(
            float(safety_front_min),
            float(safety_action_obstacle_distance),
        )
        slow_den = max(float(slow_dist) - float(stop_dist), 1e-6)
        velocity_safety_slow_risk = float(
            np.clip((float(slow_dist) - safety_slow_reference_distance) / slow_den, 0.0, 1.0)
        )
        self._last_velocity_safety_slowdown_risk = velocity_safety_slow_risk

        # 1) Hard trigger band only: start a synchronous reverse recovery only
        # when the policy is actually requesting meaningful forward motion.
        #
        # Important fix:
        #   stop_dist is a diagnostic/warning boundary.  It must not itself start
        #   backup.  With stop_dist=0.24 and trigger_dist=0.19, logs such as
        #   front=0.224 were repeatedly entering the old STOP_BAND_BACKUP branch
        #   even though they were outside the intended hard backup trigger.
        #
        # Cooldown is also backup-only: it suppresses repeated backup starts but
        # never publishes a forced hold or clamp.  During cooldown, the policy
        # command passes through unchanged and normal collision/terminal learning
        # applies.
        hard_backup_condition = bool(
            forward_backup_requested
            and (
                float(safety_front_min) < float(trigger_dist)
                or float(safety_action_obstacle_distance) < float(trigger_dist)
            )
        )

        if hard_backup_condition and bool(getattr(self, "velocity_safety_backup", True)) and rear_ok and not cooldown_active:
            return self._execute_velocity_safety_backup_sequence(
                front_min=float(safety_front_min),
                rear_min=float(safety_rear_min),
                turn_sign=float(turn_sign),
                backup_v=float(backup_v),
                backup_turn_speed=float(backup_turn_speed),
                backup_steps_cfg=int(backup_steps_cfg),
                rear_stop_dist=float(rear_stop_dist),
                lidar_action_obstacle_distance=float(safety_action_obstacle_distance),
                lidar_action_obstacle_score=float(live_action_obstacle_score),
                lidar_front_obstacle_distance=float(live_front_obstacle_distance),
                reason_prefix="sync_live_lidar_hard_trigger_backup",
                log_label="VELOCITY_SAFETY_SYNC_LIVE_LIDAR_BACKUP",
            )

        if hard_backup_condition:
            # No forced hold here.  If cooldown is active, rear is unsafe, backup
            # is disabled, or the forward command is below the meaningful-action
            # threshold, pass the policy command through unchanged.
            self._last_velocity_safety_blocked = False
            self._last_velocity_safety_skip_store = False
            self._last_velocity_safety_penalty = 0.0
            self._last_velocity_safety_reason = (
                f"hard_trigger_no_backup_policy_passthrough:front_scan={safety_front_min:.3f},"
                f"action_scan={safety_action_obstacle_distance:.3f},trigger={trigger_dist:.3f},"
                f"rear_scan={safety_rear_min:.3f},rear_warn={rear_warn_dist:.3f},"
                f"cooldown={cooldown_active},forward={forward:.3f},"
                f"forward_min={forward_backup_min:.3f}"
            )

        # 2) Stop band: diagnostic only.  Do not start backup, hold, rotate, or
        # scale v/w here.  This preserves action->executed-command consistency in
        # replay.  If the policy actually drives into collision, the normal
        # collision terminal/reward path supplies the learning signal.
        if (not hard_backup_condition) and forward_backup_requested and (
            float(safety_front_min) < float(stop_dist)
            or float(safety_action_obstacle_distance) < float(stop_dist)
        ):
            self._last_velocity_safety_blocked = False
            self._last_velocity_safety_skip_store = False
            self._last_velocity_safety_penalty = 0.0
            self._last_velocity_safety_reason = (
                f"stop_band_policy_passthrough:front_scan={safety_front_min:.3f},"
                f"action_scan={safety_action_obstacle_distance:.3f},stop={stop_dist:.3f},"
                f"trigger={trigger_dist:.3f},rear_scan={safety_rear_min:.3f},"
                f"cooldown={cooldown_active},forward={forward:.3f},"
                f"forward_min={forward_backup_min:.3f}"
            )

        # 3) Soft slowdown band: reduce v continuously and penalize the raw
        # policy-requested v.  This is the behavior requested for near-risk states:
        # high v near an obstacle receives a large penalty, small v receives a
        # small penalty, and the executed robot command is kept physically safer.
        if (
            bool(getattr(self, "velocity_safety_slowdown", True))
            and forward > 0.0
            and velocity_safety_slow_risk > 0.0
            and not bool(getattr(self, "_last_velocity_safety_backup_triggered", False))
        ):
            old_v = float(cmd[0])
            min_scale = float(getattr(self, "velocity_safety_slow_min_scale", 0.20))
            # risk=0 -> 1.0, risk=1 -> min_scale.  Squaring the risk keeps the
            # outer warning band permissive but clamps aggressively near stop_dist.
            slowdown_scale = float(np.clip(1.0 - (1.0 - min_scale) * (velocity_safety_slow_risk ** 1.20), min_scale, 1.0))
            cmd[0] = float(np.clip(float(cmd[0]) * slowdown_scale, 0.0, float(self.max_linear_speed)))
            self._last_velocity_safety_slowdown = slowdown_scale
            self._last_velocity_safety_executed_v = float(cmd[0])
            slow_penalty = compute_velocity_safety_slowdown_penalty(
                policy_linear_x=float(action[0]) if action.size > 0 else old_v,
                max_linear_speed=float(self.max_linear_speed),
                danger_score=float(velocity_safety_slow_risk),
                penalty_scale=float(getattr(self, "velocity_safety_slow_penalty", 1.80)),
                speed_power=float(getattr(self, "velocity_safety_slow_speed_power", 1.35)),
                danger_power=float(getattr(self, "velocity_safety_slow_danger_power", 1.10)),
            )
            self._last_velocity_safety_penalty = max(
                float(getattr(self, "_last_velocity_safety_penalty", 0.0)),
                float(slow_penalty),
            )
            self._last_velocity_safety_reason = (
                f"soft_slowdown:risk={velocity_safety_slow_risk:.3f},scale={slowdown_scale:.3f},"
                f"v_policy={float(action[0]) if action.size > 0 else old_v:.3f},"
                f"v_exec={float(cmd[0]):.3f},penalty={float(self._last_velocity_safety_penalty):.3f},"
                f"front_scan={safety_front_min:.3f},action_scan={safety_action_obstacle_distance:.3f},"
                f"slow={slow_dist:.3f},stop={stop_dist:.3f}"
            )
        elif forward > 0.0 and lidar_action_obstacle_distance < slow_dist:
            self._last_velocity_safety_slowdown = 1.0
            self._last_velocity_safety_reason = (
                f"warning_no_slowdown_disabled:action_scan={safety_action_obstacle_distance:.3f},"
                f"action_reward={lidar_action_obstacle_distance:.3f},slow={slow_dist:.3f},"
                f"cmd=({float(cmd[0]):+.3f},{float(cmd[1]):+.3f})"
            )

        # Optional evaluation-time forward assist.  A weak checkpoint can learn to
        # rotate in place, especially when priority is disabled even though the
        # model was trained with priority channels.  This assist does not change
        # the action space; it only raises a tiny safe forward component while
        # the policy is turning, and only when the front/action sectors are clear.
        assist_v = float(getattr(self, "velocity_forward_assist_mps", 0.0))
        assist_w_thr = float(getattr(self, "velocity_forward_assist_angular_threshold", 0.20))
        assist_clear = float(getattr(self, "velocity_forward_assist_min_clearance_m", 0.45))
        if (
            assist_v > 0.0
            and float(cmd[0]) < assist_v
            and abs(float(cmd[1])) >= assist_w_thr
            and float(front_min) > assist_clear
            and float(lidar_action_obstacle_distance) > assist_clear
            and not bool(getattr(self, "_last_velocity_safety_blocked", False))
        ):
            cmd[0] = min(float(assist_v), float(self.max_linear_speed))
            self._last_velocity_forward_assist = True
            self._last_velocity_safety_reason = (
                "forward_assist:"
                f"v={cmd[0]:.3f},w={cmd[1]:.3f},front={front_min:.3f},action={lidar_action_obstacle_distance:.3f}"
            )

        # Evaluation-only anti-spin guard.  This does not affect the observation
        # space or training; it only prevents a weak/OOD checkpoint from holding
        # an angular command at the physical limit forever on the real robot.
        # It activates only after the same saturated turn sign has persisted for
        # several consecutive policy steps and the front/action sectors are clear.
        spin_sign = 1 if float(cmd[1]) > 0.0 else (-1 if float(cmd[1]) < 0.0 else 0)
        spin_ratio = abs(float(cmd[1])) / max(float(self.max_angular_speed), 1e-6)
        if spin_sign != 0 and spin_sign == int(getattr(self, "_velocity_spin_last_sign", 0)):
            self._velocity_spin_same_sign_steps = int(getattr(self, "_velocity_spin_same_sign_steps", 0)) + 1
        elif spin_sign != 0:
            self._velocity_spin_same_sign_steps = 1
            self._velocity_spin_last_sign = spin_sign
        else:
            self._velocity_spin_same_sign_steps = 0
            self._velocity_spin_last_sign = 0

        if (
            bool(getattr(self, "velocity_spin_breaker", False))
            and int(getattr(self, "_velocity_spin_same_sign_steps", 0)) >= int(getattr(self, "velocity_spin_breaker_steps", 14))
            and spin_ratio >= float(getattr(self, "velocity_spin_breaker_angular_ratio", 0.85))
            and float(front_min) > float(getattr(self, "velocity_spin_breaker_min_clearance_m", 0.48))
            and float(lidar_action_obstacle_distance) > float(getattr(self, "velocity_spin_breaker_min_clearance_m", 0.48))
            and not bool(getattr(self, "_last_velocity_safety_blocked", False))
        ):
            old_v = float(cmd[0])
            old_w = float(cmd[1])
            cmd[0] = min(
                float(self.max_linear_speed),
                max(float(cmd[0]), float(getattr(self, "velocity_spin_breaker_forward_mps", 0.035))),
            )
            cmd[1] = float(cmd[1]) * float(getattr(self, "velocity_spin_breaker_angular_scale", 0.35))
            self._last_velocity_spin_breaker = True
            self._last_velocity_safety_reason = (
                "spin_breaker:"
                f"steps={int(getattr(self, '_velocity_spin_same_sign_steps', 0))},"
                f"old=({old_v:+.3f},{old_w:+.3f}),new=({cmd[0]:+.3f},{cmd[1]:+.3f}),"
                f"front={front_min:.3f},action={lidar_action_obstacle_distance:.3f}"
            )
            if int(getattr(self, "_velocity_spin_same_sign_steps", 0)) % max(int(getattr(self, "velocity_spin_breaker_steps", 14)), 1) == 0:
                self.ros.get_logger().warn(
                    "VELOCITY_SPIN_BREAKER | "
                    f"steps={int(getattr(self, '_velocity_spin_same_sign_steps', 0))} "
                    f"cmd_old=({old_v:+.3f},{old_w:+.3f}) "
                    f"cmd_new=({cmd[0]:+.3f},{cmd[1]:+.3f}) "
                    f"front={front_min:.3f} action={lidar_action_obstacle_distance:.3f}"
                )

        self._last_velocity_safety_executed_v = float(cmd[0])
        self.ros.publish_cmd_vel(float(cmd[0]), float(cmd[1]))
        self.ros.spin_steps(num_spins=5, timeout_sec=0.001)
        self._advance_world_after_command(target_delta_sec=self.control_dt)

        return (
            cmd.astype(np.float32),
            float(lidar_action_obstacle_distance),
            float(lidar_action_obstacle_score),
            float(lidar_front_obstacle_distance),
        )

    def _decode_waypoint_action(self, policy_action: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, float]:
        """
        policy action을 local/world waypoint로 변환한다.

        waypoint_action_type="path":
          action[0] -> planned path 방향 lookahead 거리 비율
          action[1] -> planned path tangent 기준 lateral offset 비율

        waypoint_action_type="polar":
          action[0] -> 로봇 기준 waypoint 거리 비율
          action[1] -> 로봇 기준 waypoint heading 비율
        """
        action = np.asarray(policy_action, dtype=np.float32)
        if action.size < 2:
            action = np.pad(action, (0, 2 - action.size), mode="constant")

        a0 = float(np.clip(action[0], 0.0, 1.0))
        a1 = float(np.clip(action[1], -1.0, 1.0))

        if self.waypoint_action_type == "path":
            local_xy, distance, heading, lateral_offset = self._decode_path_conditioned_waypoint(
                lookahead_norm=a0,
                lateral_norm=a1,
            )
        else:
            local_xy, distance, heading, lateral_offset = self._decode_polar_waypoint(
                distance_norm=a0,
                heading_norm=a1,
            )

        robot_pose = self._get_robot_pose2d()
        if robot_pose is None:
            # Never reinterpret robot-local coordinates as odom/map coordinates.
            # That was the failure mode where goals were sent around the world
            # origin/center whenever odom pose was temporarily unavailable.
            world_xy = np.array([np.nan, np.nan], dtype=np.float32)
        else:
            robot_xy, robot_yaw = robot_pose
            c = math.cos(float(robot_yaw))
            s = math.sin(float(robot_yaw))
            world_xy = np.array(
                [
                    float(robot_xy[0]) + c * float(local_xy[0]) - s * float(local_xy[1]),
                    float(robot_xy[1]) + s * float(local_xy[0]) + c * float(local_xy[1]),
                ],
                dtype=np.float32,
            )

        self._last_waypoint_lateral_offset = float(lateral_offset)
        self._last_waypoint_action_type = str(self.waypoint_action_type)
        if self._prev_waypoint_angle_for_reward is None:
            heading_delta = 0.0
        else:
            heading_delta = self._normalize_angle(float(heading) - float(self._prev_waypoint_angle_for_reward))
        self._last_waypoint_heading_delta = float(heading_delta)
        self._prev_waypoint_angle_for_reward = float(heading)

        return local_xy, world_xy.astype(np.float32), float(distance), float(heading)

    def _decode_polar_waypoint(
        self,
        distance_norm: float,
        heading_norm: float,
    ) -> tuple[np.ndarray, float, float, float]:
        distance = self.waypoint_min_distance + float(distance_norm) * (
            self.waypoint_max_distance - self.waypoint_min_distance
        )
        heading = float(heading_norm) * self.waypoint_max_angle_rad
        local_xy = np.array(
            [distance * math.cos(heading), distance * math.sin(heading)],
            dtype=np.float32,
        )
        return local_xy, float(distance), float(heading), 0.0

    def _decode_path_conditioned_waypoint(
        self,
        lookahead_norm: float,
        lateral_norm: float,
    ) -> tuple[np.ndarray, float, float, float]:
        """
        planned path tangent 주변에 waypoint를 만든다.

        path_angle은 ExplorationGridMap이 계산한 reachable path lookahead 방향이다.
        이 방식은 policy가 임의 polar 방향을 찍는 것을 막고, path 위를 따라가며
        lateral offset만 미세 조정하게 만든다. path가 아직 없으면 frontier_angle을
        약한 fallback으로 쓰고, 그것도 없으면 정면을 사용한다.
        """
        lookahead = self.waypoint_min_distance + float(np.clip(lookahead_norm, 0.0, 1.0)) * (
            self.waypoint_max_distance - self.waypoint_min_distance
        )
        lateral = float(np.clip(lateral_norm, -1.0, 1.0)) * self.waypoint_lateral_max_offset

        stats = self.last_map_stats
        if stats is not None and bool(getattr(stats, "target_reachable", False)):
            path_angle = float(getattr(stats, "path_angle", 0.0))
        elif stats is not None and int(getattr(stats, "frontier_count", 0)) > 0:
            # fallback: 아직 BFS path가 없을 때만 frontier bearing을 제한적으로 사용한다.
            path_angle = float(getattr(stats, "frontier_angle", 0.0))
        else:
            path_angle = 0.0

        # 너무 후방 waypoint를 찍으면 controller가 제자리 회전에 가까워지므로 제한한다.
        path_angle = float(np.clip(path_angle, -self.waypoint_max_angle_rad, self.waypoint_max_angle_rad))

        tx = math.cos(path_angle)
        ty = math.sin(path_angle)
        # robot local frame에서 path tangent의 좌측 normal.
        nx = -ty
        ny = tx

        local_xy = np.array(
            [
                lookahead * tx + lateral * nx,
                lookahead * ty + lateral * ny,
            ],
            dtype=np.float32,
        )

        distance = float(np.linalg.norm(local_xy))
        if distance < self.waypoint_min_distance:
            # lateral cancellation 등으로 너무 가까우면 tangent 방향 최소거리로 보정한다.
            local_xy = np.array(
                [self.waypoint_min_distance * tx, self.waypoint_min_distance * ty],
                dtype=np.float32,
            )
            distance = float(self.waypoint_min_distance)
            lateral = 0.0

        heading = math.atan2(float(local_xy[1]), float(local_xy[0]))
        return local_xy, float(distance), float(heading), float(lateral)


    def _reset_nav2_stationary_window(self) -> None:
        """Start a fresh motion window for one Nav2 goal.

        BackUp must only run when the robot has been physically stationary for a
        continuous window.  Heading alignment or normal slow turning is not a
        backup condition, even if xy translation is small.
        """
        try:
            self._nav2_stationary_samples.clear()
        except Exception:
            self._nav2_stationary_samples = deque(maxlen=240)
        self._sample_nav2_stationary_window()
        self._last_nav2_stationary_gate_reason = "window_reset"

    def _sample_nav2_stationary_window(self) -> None:
        pose = self._get_robot_pose2d()
        if pose is None:
            return
        xy, yaw = pose
        xy = np.asarray(xy, dtype=np.float32)
        if xy.shape[0] < 2 or not np.all(np.isfinite(xy[:2])) or not np.isfinite(float(yaw)):
            return
        self._nav2_stationary_samples.append((time.time(), float(xy[0]), float(xy[1]), float(yaw)))

    def _nav2_stationary_for_backup(self, required_sec: float | None = None) -> tuple[bool, str]:
        """Return True only if xy and yaw stayed almost unchanged for required_sec.

        This is the hard gate for BackUp.  It prevents random reverse while Nav2
        is rotating in place or still activating the controller.
        """
        required = max(float(required_sec if required_sec is not None else getattr(self, "nav2_stuck_backup_stationary_sec", 1.5)), 0.1)
        now = time.time()
        samples = list(getattr(self, "_nav2_stationary_samples", []))
        if len(samples) < 2:
            return False, f"stationary_window_short:0.00/{required:.2f}s"

        # Keep only recent samples; deque is bounded, but pruning makes the
        # measured window explicit and stable.
        cutoff = now - max(required * 2.0, required + 0.5)
        samples = [x for x in samples if x[0] >= cutoff]
        if len(samples) < 2:
            return False, f"stationary_window_short:0.00/{required:.2f}s"

        newest = samples[-1]
        oldest = None
        for sample in reversed(samples):
            if newest[0] - sample[0] >= required:
                oldest = sample
                break
        if oldest is None:
            span = newest[0] - samples[0][0]
            return False, f"stationary_window_short:{span:.2f}/{required:.2f}s"

        dx = float(newest[1]) - float(oldest[1])
        dy = float(newest[2]) - float(oldest[2])
        xy_delta = math.hypot(dx, dy)
        yaw_delta = abs(self._normalize_angle(float(newest[3]) - float(oldest[3])))
        xy_limit = max(float(getattr(self, "nav2_stuck_backup_stationary_xy_m", 0.025)), 0.0)
        yaw_limit = max(float(getattr(self, "nav2_stuck_backup_stationary_yaw_rad", math.radians(7.0))), 0.0)

        if xy_delta > xy_limit:
            return False, f"not_stationary_xy:{xy_delta:.3f}>{xy_limit:.3f}m/{required:.2f}s"
        if yaw_delta > yaw_limit:
            return False, f"not_stationary_yaw:{math.degrees(yaw_delta):.1f}>{math.degrees(yaw_limit):.1f}deg/{required:.2f}s"
        return True, f"stationary:{xy_delta:.3f}m,{math.degrees(yaw_delta):.1f}deg/{required:.2f}s"

    def _waypoint_distance_to_target(self, target_world_xy: np.ndarray) -> float:
        """현재 로봇 pose 기준 target_world_xy까지의 남은 거리[m]를 계산한다."""
        robot_pose = self._get_robot_pose2d()
        if robot_pose is None:
            return 999.0
        robot_xy, _ = robot_pose
        diff = np.asarray(target_world_xy, dtype=np.float32) - np.asarray(robot_xy, dtype=np.float32)
        return float(np.linalg.norm(diff))

    def _heading_error_to_world_target(self, target_world_xy: np.ndarray | None) -> float:
        """Robot yaw 기준 target 방향까지의 절대 heading error[rad].

        Nav2는 goal을 받은 직후 제자리 회전을 먼저 할 수 있다. 이때 xy 이동량은
        거의 0이므로 단순 `moved < epsilon` stuck 판정은 정상 회전을 stuck으로
        오판한다. backup은 heading 정렬이 대체로 끝났고 정면이 실제로 막혔을 때만
        허용한다.
        """
        if target_world_xy is None:
            return 999.0
        robot_pose = self._get_robot_pose2d()
        if robot_pose is None:
            return 999.0
        robot_xy, robot_yaw = robot_pose
        target = np.asarray(target_world_xy, dtype=np.float32)
        if target.shape[0] < 2 or not np.all(np.isfinite(target[:2])):
            return 999.0
        dx = float(target[0]) - float(robot_xy[0])
        dy = float(target[1]) - float(robot_xy[1])
        if not np.isfinite(dx) or not np.isfinite(dy) or math.hypot(dx, dy) < 1e-4:
            return 0.0
        target_yaw = math.atan2(dy, dx)
        return abs(self._normalize_angle(target_yaw - float(robot_yaw)))

    def _nav2_backup_allowed_now(self, target_world_xy: np.ndarray | None, moved_since_goal: float, elapsed_wall: float) -> tuple[bool, str]:
        """Gate Nav2 BackUp so normal heading-alignment rotation is not treated as stuck."""
        heading_err = self._heading_error_to_world_target(target_world_xy)
        front_min = self._scan_min_distance_in_sector(
            scan=self.ros.scan,
            center_angle=0.0,
            half_width_rad=math.radians(32.0),
            max_considered_range=1.20,
        )
        self._last_nav2_goal_heading_error = float(heading_err)
        self._last_nav2_goal_front_min = float(front_min)

        # If the robot is still turning toward the waypoint, xy motion can be near
        # zero. Do not reverse in this phase.
        if heading_err > math.radians(38.0):
            return False, f"rotating_to_goal:herr={math.degrees(heading_err):.1f}deg,front={front_min:.3f}m"

        # Random reverse in open space was caused by a movement-only stuck test.
        # Require a real near-front obstacle before using BackUp.
        if front_min > 0.34:
            return False, f"not_front_blocked:front={front_min:.3f}m,herr={math.degrees(heading_err):.1f}deg"

        # Very early after sending a goal, Nav2 may still be activating the
        # controller. Avoid backing up before the controller has had time to issue
        # a valid command.
        if elapsed_wall < max(1.5, float(getattr(self, "nav2_stuck_backup_stationary_sec", 1.5))):
            return False, f"warmup:{elapsed_wall:.2f}s,front={front_min:.3f}m,herr={math.degrees(heading_err):.1f}deg"

        stationary_ok, stationary_reason = self._nav2_stationary_for_backup(
            required_sec=float(getattr(self, "nav2_stuck_backup_stationary_sec", 1.5))
        )
        self._last_nav2_stationary_gate_reason = stationary_reason
        if not stationary_ok:
            return False, f"not_stationary:{stationary_reason},front={front_min:.3f}m,herr={math.degrees(heading_err):.1f}deg"

        return True, f"front_blocked:{stationary_reason},front={front_min:.3f}m,herr={math.degrees(heading_err):.1f}deg,moved={moved_since_goal:.3f}m"


    @staticmethod
    def _yaw_from_quaternion_xyzw(x: float, y: float, z: float, w: float) -> float:
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        return math.atan2(siny_cosp, cosy_cosp)

    @staticmethod
    def _linear_ramp01(value: float, low: float, high: float) -> float:
        """
        Piecewise-linear ramp used by the SLAM speed limiter.

        value <= low  -> 0.0
        value >= high -> 1.0
        otherwise     -> (value - low) / (high - low)

        This is intentionally not a discrete gate. It prevents behavior such as
        0 / 0.5 / 1 speed buckets and makes speed change continuously with local
        SLAM map completeness.
        """
        v = float(value)
        lo = float(low)
        hi = float(high)
        if hi <= lo + 1e-6:
            return 1.0 if v >= hi else 0.0
        return float(np.clip((v - lo) / (hi - lo), 0.0, 1.0))

    def _compute_slam_adaptive_speed_limit(self) -> tuple[float, float]:
        """
        Compute a continuous linear-speed cap from local SLAM map quality.

        Important policy:
          - The configured max_linear_speed itself is not changed.
          - Only the current controller speed cap is scaled down.
          - Scaling is continuous and linear with respect to known-cell ratios.

        Let k_local and k_front be known-cell ratios around/in front of the robot.
        Instead of mapping them to discrete buckets, each ratio is converted by
        a linear ramp:

            score = clamp((known_ratio - low) / (high - low), 0, 1)

        The weighted score then linearly interpolates the velocity scale:

            scale = min_scale + (max_scale - min_scale) * quality
        """
        if not self.slam_adaptive_speed or not self.use_slam_map:
            self._last_slam_local_known_ratio = 1.0
            self._last_slam_front_known_ratio = 1.0
            self._last_slam_local_linear_score = 1.0
            self._last_slam_front_linear_score = 1.0
            self._last_slam_fresh_score = 1.0
            self._last_slam_fresh_linear_score = 1.0
            self._last_slam_quality_score = 1.0
            self._last_slam_speed_raw_scale = 1.0
            self._last_slam_speed_scale = 1.0
            self._last_slam_speed_limit = float(self.max_linear_speed)
            return float(self.max_linear_speed), 1.0

        slam_map = getattr(self.ros, "slam_map", None)
        robot_pose = self._get_robot_pose2d()
        if slam_map is None or robot_pose is None:
            local_known = 0.0
            front_known = 0.0
            fresh_score = 0.0
            local_linear = 0.0
            front_linear = 0.0
            fresh_linear = 0.0
            quality = 0.0
            raw_scale = self.slam_speed_min_scale
            prev_scale = getattr(self, "_last_slam_speed_scale", raw_scale)
            alpha = self.slam_speed_smoothing_alpha
            scale = alpha * prev_scale + (1.0 - alpha) * raw_scale
            scale = float(np.clip(scale, self.slam_speed_min_scale, self.slam_speed_max_scale))

            self._last_slam_local_known_ratio = local_known
            self._last_slam_front_known_ratio = front_known
            self._last_slam_local_linear_score = local_linear
            self._last_slam_front_linear_score = front_linear
            self._last_slam_fresh_score = fresh_score
            self._last_slam_fresh_linear_score = fresh_linear
            self._last_slam_quality_score = quality
            self._last_slam_speed_raw_scale = float(raw_scale)
            self._last_slam_speed_scale = float(scale)
            self._last_slam_speed_limit = float(self.max_linear_speed * scale)
            return self._last_slam_speed_limit, float(scale)

        robot_xy, robot_yaw = robot_pose
        info = slam_map.info
        width = int(info.width)
        height = int(info.height)
        resolution = float(info.resolution)
        if width <= 0 or height <= 0 or resolution <= 1e-6:
            return float(self.max_linear_speed), 1.0

        try:
            grid = np.asarray(slam_map.data, dtype=np.int16).reshape((height, width))
        except Exception:
            return float(self.max_linear_speed), 1.0

        origin = info.origin
        origin_x = float(origin.position.x)
        origin_y = float(origin.position.y)
        origin_yaw = self._yaw_from_quaternion_xyzw(
            float(origin.orientation.x),
            float(origin.orientation.y),
            float(origin.orientation.z),
            float(origin.orientation.w),
        )

        # World -> map grid center index. OccupancyGrid origins are usually yaw=0,
        # but keeping yaw support prevents subtle errors in custom maps.
        dx = float(robot_xy[0]) - origin_x
        dy = float(robot_xy[1]) - origin_y
        c = math.cos(-origin_yaw)
        s = math.sin(-origin_yaw)
        mx = (c * dx - s * dy) / resolution
        my = (s * dx + c * dy) / resolution
        cx = int(math.floor(mx))
        cy = int(math.floor(my))

        radius_m = max(self.slam_local_speed_radius, self.slam_front_speed_distance)
        radius_cells = int(math.ceil(radius_m / resolution)) + 2
        x0 = max(cx - radius_cells, 0)
        x1 = min(cx + radius_cells + 1, width)
        y0 = max(cy - radius_cells, 0)
        y1 = min(cy + radius_cells + 1, height)

        if x0 >= x1 or y0 >= y1:
            local_known = 0.0
            front_known = 0.0
        else:
            ys, xs = np.mgrid[y0:y1, x0:x1]
            # Cell centers in map-local metric coordinates.
            map_local_x = (xs.astype(np.float32) + 0.5) * resolution
            map_local_y = (ys.astype(np.float32) + 0.5) * resolution
            # Map local -> world.
            co = math.cos(origin_yaw)
            so = math.sin(origin_yaw)
            wx = origin_x + co * map_local_x - so * map_local_y
            wy = origin_y + so * map_local_x + co * map_local_y

            rel_x = wx - float(robot_xy[0])
            rel_y = wy - float(robot_xy[1])
            dist = np.sqrt(rel_x * rel_x + rel_y * rel_y)
            angle = np.arctan2(rel_y, rel_x) - float(robot_yaw)
            angle = np.arctan2(np.sin(angle), np.cos(angle))

            sub = grid[y0:y1, x0:x1]
            known = sub >= 0

            local_mask = dist <= self.slam_local_speed_radius
            front_mask = (
                (dist <= self.slam_front_speed_distance)
                & (dist >= max(0.10, self.waypoint_reached_tolerance * 0.5))
                & (np.abs(angle) <= self.slam_front_speed_half_angle_rad)
            )

            local_total = int(np.count_nonzero(local_mask))
            front_total = int(np.count_nonzero(front_mask))
            local_known = (
                float(np.count_nonzero(known & local_mask)) / float(local_total)
                if local_total > 0
                else 0.0
            )
            front_known = (
                float(np.count_nonzero(known & front_mask)) / float(front_total)
                if front_total > 0
                else local_known
            )

        now_wall = time.time()
        last_map_wall = getattr(self.ros, "last_slam_map_time", None)
        if last_map_wall is None:
            map_age = self.slam_speed_map_age_soft_limit_sec
        else:
            map_age = max(now_wall - float(last_map_wall), 0.0)
        fresh_score = float(
            np.clip(
                1.0 - map_age / max(self.slam_speed_map_age_soft_limit_sec, 1e-6),
                0.0,
                1.0,
            )
        )

        # Continuous linear remapping. This is the part that removes bucket-like
        # behavior. A known ratio of 0.55 with low=0.25, high=0.85 becomes 0.50,
        # not a hard class such as 0 / 0.5 / 1.
        local_linear = self._linear_ramp01(
            local_known,
            self.slam_speed_known_low_ratio,
            self.slam_speed_known_high_ratio,
        )
        front_linear = self._linear_ramp01(
            front_known,
            self.slam_speed_known_low_ratio,
            self.slam_speed_known_high_ratio,
        )
        fresh_linear = self._linear_ramp01(
            fresh_score,
            self.slam_speed_fresh_low_score,
            1.0,
        )

        total_weight = (
            self.slam_speed_local_weight
            + self.slam_speed_front_weight
            + self.slam_speed_fresh_weight
        )
        if total_weight <= 1e-6:
            quality = 1.0
        else:
            quality = (
                self.slam_speed_local_weight * local_linear
                + self.slam_speed_front_weight * front_linear
                + self.slam_speed_fresh_weight * fresh_linear
            ) / total_weight
        quality = float(np.clip(quality, 0.0, 1.0))

        raw_scale = float(
            self.slam_speed_min_scale
            + (self.slam_speed_max_scale - self.slam_speed_min_scale) * quality
        )
        raw_scale = float(np.clip(raw_scale, self.slam_speed_min_scale, self.slam_speed_max_scale))

        # Optional first-order smoothing to prevent speed cap jitter when the SLAM
        # map alternates between adjacent known/unknown cells.
        prev_scale = float(getattr(self, "_last_slam_speed_scale", raw_scale))
        alpha = self.slam_speed_smoothing_alpha
        scale = alpha * prev_scale + (1.0 - alpha) * raw_scale
        scale = float(np.clip(scale, self.slam_speed_min_scale, self.slam_speed_max_scale))
        limit = float(self.max_linear_speed * scale)

        self._last_slam_local_known_ratio = float(local_known)
        self._last_slam_front_known_ratio = float(front_known)
        self._last_slam_local_linear_score = float(local_linear)
        self._last_slam_front_linear_score = float(front_linear)
        self._last_slam_fresh_score = float(fresh_score)
        self._last_slam_fresh_linear_score = float(fresh_linear)
        self._last_slam_quality_score = float(quality)
        self._last_slam_speed_raw_scale = float(raw_scale)
        self._last_slam_speed_scale = float(scale)
        self._last_slam_speed_limit = float(limit)
        return limit, scale

    def _waypoint_controller_command(self, target_world_xy: np.ndarray) -> tuple[np.ndarray, bool]:
        """
        현재 pose에서 target_world_xy까지 가는 local primitive command를 만든다.

        기본은 direct-point primitive다.
          1) 목표 heading 오차가 크면 rotate-only.
          2) 충분히 정렬되면 거의 직선으로 전진한다.
          3) 전진 중 yaw 보정은 작은 angular cap으로 제한한다.
          4) 목표 방향 LiDAR sector 또는 정면 emergency sector가 막히면 전진을 멈춘다.

        의도:
          - Nav2/DWB/BT의 spin/recovery/preemption을 RL inner loop에서 제거한다.
          - SLAM reset 직후 빠른 회전/곡선 주행으로 map이 찢어지는 문제를 줄인다.
        """
        robot_pose = self._get_robot_pose2d()
        if robot_pose is None:
            return np.zeros(2, dtype=np.float32), False

        robot_xy, robot_yaw = robot_pose
        diff = np.asarray(target_world_xy, dtype=np.float32) - np.asarray(robot_xy, dtype=np.float32)
        distance = float(np.linalg.norm(diff))

        if distance <= self.waypoint_reached_tolerance:
            return np.zeros(2, dtype=np.float32), True

        target_yaw = math.atan2(float(diff[1]), float(diff[0]))
        yaw_error = self._normalize_angle(target_yaw - float(robot_yaw))
        yaw_abs = abs(yaw_error)

        # Always keep SLAM-adaptive debug fields updated; the returned speed limit
        # is used only when forward motion is allowed.
        speed_limit, _ = self._compute_slam_adaptive_speed_limit()
        speed_limit = float(np.clip(speed_limit, 0.0, self.max_linear_speed))

        # LiDAR clearance along the target direction and along the robot front.
        # target_min protects the chosen point ray; front_min is an emergency stop
        # for cases where the robot is already facing an obstacle.
        target_sector_half_width = self.waypoint_direct_target_sector_rad
        target_min = self._scan_min_distance_in_sector(
            scan=self.ros.scan,
            center_angle=yaw_error,
            half_width_rad=target_sector_half_width,
            max_considered_range=1.20,
        )
        front_min = self._scan_min_distance_in_sector(
            scan=self.ros.scan,
            center_angle=0.0,
            half_width_rad=math.radians(18.0),
            max_considered_range=1.20,
        )
        emergency_stop_distance = max(
            self.collision_threshold + 0.04,
            0.80 * self.waypoint_front_stop_distance,
        )
        target_blocked = target_min < self.waypoint_front_stop_distance
        front_emergency = front_min < emergency_stop_distance

        # ------------------------------------------------------------------
        # Last-line safety guard for the internal controller.
        # Nav2 is disabled in this mode, so a bad SAC waypoint can still point
        # directly into a wall.  If the forward sector is already too close, do
        # not merely stop in place; back out a little and yaw toward the more
        # open side.  Hit endpoints/walls are not marked as priority-checked;
        # this is purely a motor safety primitive.
        # ------------------------------------------------------------------
        if front_emergency:
            left_clear = self._scan_min_distance_in_sector(
                scan=self.ros.scan,
                center_angle=math.radians(55.0),
                half_width_rad=math.radians(25.0),
                max_considered_range=1.20,
            )
            right_clear = self._scan_min_distance_in_sector(
                scan=self.ros.scan,
                center_angle=-math.radians(55.0),
                half_width_rad=math.radians(25.0),
                max_considered_range=1.20,
            )
            rear_clear = self._scan_min_distance_in_sector(
                scan=self.ros.scan,
                center_angle=math.pi,
                half_width_rad=math.radians(28.0),
                max_considered_range=1.00,
            )
            turn_sign = 1.0 if left_clear >= right_clear else -1.0
            angular_z = float(turn_sign * min(self.max_angular_speed, 0.55))
            if rear_clear > max(0.26, self.collision_threshold + 0.08):
                linear_x = -min(0.08, max(self.max_linear_speed * 0.35, 0.04))
            else:
                linear_x = 0.0
            return np.array([linear_x, angular_z], dtype=np.float32), False

        if not self.waypoint_direct_point_mode:
            # Backward-compatible continuous arc-follow controller.
            angular_z = float(np.clip(
                self.waypoint_angular_kp * yaw_error,
                -self.max_angular_speed,
                self.max_angular_speed,
            ))
            forward_limit = max(float(self.waypoint_max_yaw_error_for_linear_rad), math.radians(1.0))
            if yaw_abs >= forward_limit:
                heading_gate = 0.0
                linear_x = 0.0
            else:
                c = math.cos(yaw_abs)
                c_min = math.cos(forward_limit)
                heading_gate = float(np.clip((c - c_min) / max(1.0 - c_min, 1e-6), 0.0, 1.0))
                if self.waypoint_disable_arrival_slowdown:
                    linear_x = self.waypoint_linear_kp * distance * heading_gate
                else:
                    distance_gate = float(np.clip(distance / self.waypoint_slowdown_distance, 0.15, 1.0))
                    linear_x = self.waypoint_linear_kp * distance * heading_gate * distance_gate
            if target_blocked or front_emergency:
                linear_x = 0.0
            elif linear_x > 0.0:
                linear_x = min(float(linear_x), speed_limit)
                if self.waypoint_disable_arrival_slowdown and heading_gate > 0.20:
                    requested_min = self.waypoint_min_linear_speed * heading_gate
                    effective_min = min(float(requested_min), speed_limit)
                    linear_x = max(float(linear_x), float(effective_min))
            linear_x = float(np.clip(linear_x, 0.0, self.max_linear_speed))
            if linear_x < self.linear_deadband:
                linear_x = 0.0
            if abs(angular_z) < self.angular_deadband:
                angular_z = 0.0
            return np.array([linear_x, angular_z], dtype=np.float32), False

        # ------------------------------------------------------------------
        # Direct-point primitive.
        # ------------------------------------------------------------------
        # Phase A: rotate toward the selected point.  Unlike the previous
        # rotate-only primitive, this version allows a small forward component
        # while turning when the target is not too far from the front axis.
        #
        # This avoids the slow pattern:
        #   rotate -> stop -> move -> stop -> rotate ...
        # while still protecting SLAM by forbidding forward creep for very large
        # yaw errors or blocked target rays.
        if yaw_abs > self.waypoint_direct_heading_tolerance_rad:
            angular_z = float(np.clip(
                self.waypoint_angular_kp * yaw_error,
                -self.max_angular_speed,
                self.max_angular_speed,
            ))
            spin_cap = max(0.20, min(self.max_angular_speed, 1.05))
            angular_z = float(np.clip(angular_z, -spin_cap, spin_cap))

            linear_x = 0.0
            turn_drive_limit = max(
                self.waypoint_direct_turn_drive_max_yaw_rad,
                self.waypoint_direct_heading_tolerance_rad + math.radians(1.0),
            )
            can_creep = (
                self.waypoint_direct_turn_drive
                and yaw_abs <= turn_drive_limit
                and not target_blocked
                and not front_emergency
                and distance > self.waypoint_direct_min_drive_distance
                and speed_limit > self.linear_deadband
            )
            if can_creep:
                # Gate: 1.0 near heading tolerance, 0.0 at turn_drive_limit.
                gate = (turn_drive_limit - yaw_abs) / max(
                    turn_drive_limit - self.waypoint_direct_heading_tolerance_rad,
                    1e-6,
                )
                gate = float(np.clip(gate, 0.0, 1.0))
                # Keep some forward creep even during moderate turns, but never
                # exceed the SLAM-adaptive speed limit.
                scale = self.waypoint_direct_turn_drive_speed_scale * (0.30 + 0.70 * gate)
                requested = self.waypoint_linear_kp * distance * scale
                min_creep = min(self.waypoint_direct_turn_drive_min_speed, speed_limit)
                linear_x = max(float(requested), float(min_creep))
                linear_x = min(float(linear_x), float(speed_limit), self.max_linear_speed)

            if linear_x < self.linear_deadband:
                linear_x = 0.0
            if abs(angular_z) < self.angular_deadband:
                angular_z = 0.0
            return np.array([linear_x, angular_z], dtype=np.float32), False

        # Phase B: drive mostly straight.  Only bounded heading correction is allowed.
        if target_blocked or front_emergency:
            return np.zeros(2, dtype=np.float32), False

        if distance <= self.waypoint_direct_min_drive_distance:
            return np.zeros(2, dtype=np.float32), True

        # If the yaw error suddenly becomes larger than the drive limit, avoid
        # arc-driving and go back to rotate-only on the next controller tick.
        if yaw_abs > self.waypoint_direct_drive_heading_limit_rad:
            angular_z = float(np.clip(
                self.waypoint_angular_kp * yaw_error,
                -self.max_angular_speed,
                self.max_angular_speed,
            ))
            spin_cap = max(0.15, min(self.max_angular_speed, 0.75))
            angular_z = float(np.clip(angular_z, -spin_cap, spin_cap))
            if abs(angular_z) < self.angular_deadband:
                angular_z = 0.0
            return np.array([0.0, angular_z], dtype=np.float32), False

        # Nearly straight forward speed.  Avoid arrival slowdown by default, but
        # keep a moderate speed cap for SLAM.
        linear_x = self.waypoint_linear_kp * distance
        linear_x = min(float(linear_x), speed_limit, self.max_linear_speed)
        if self.waypoint_disable_arrival_slowdown:
            linear_x = max(float(linear_x), min(self.waypoint_min_linear_speed, speed_limit))
        linear_x = float(np.clip(linear_x, 0.0, self.max_linear_speed))

        angular_z = float(np.clip(
            self.waypoint_angular_kp * yaw_error,
            -self.waypoint_direct_max_correction_angular,
            self.waypoint_direct_max_correction_angular,
        ))

        if linear_x < self.linear_deadband:
            linear_x = 0.0
        if abs(angular_z) < self.angular_deadband:
            angular_z = 0.0

        return np.array([linear_x, angular_z], dtype=np.float32), False


    def _clear_nav2_costmaps(self, wait_timeout_sec: float = 0.50) -> bool:
        """
        Clear Nav2 local/global costmaps after teleport reset.

        Gazebo SetEntityPose teleports the robot, but Nav2 costmaps and behavior-tree
        state are not automatically reset. If we do not clear them, the next
        NavigateToPose goal can be rejected/aborted even though the action server is
        active. Missing services are tolerated because some Nav2 launches use
        different names or are not ready yet.
        """
        if self.action_mode != "nav2":
            return False
        # This flag disables *all* explicit Nav2 costmap clear attempts, not only
        # collision-time clears.  Some Nav2 launches expose clear services under
        # different names or do not bring them up before the action server appears.
        # Repeated wait_for_service() calls then stall training and spam warnings.
        # With this disabled, Nav2 keeps ownership of motion while we skip the
        # manual clear step entirely.
        if not bool(getattr(self, "collision_clear_nav2_costmaps", True)):
            return False

        ok_any = False
        for name, client in (
            (self.nav2_clear_global_service, self.nav2_clear_global_client),
            (self.nav2_clear_local_service, self.nav2_clear_local_client),
        ):
            if client is None:
                continue
            try:
                if not client.wait_for_service(timeout_sec=float(wait_timeout_sec)):
                    self.ros.get_logger().debug(f"Nav2 costmap clear service not ready: {name}")
                    continue
                future = client.call_async(Empty.Request())
                self._wait_future_done(future, timeout_sec=float(wait_timeout_sec))
                ok_any = True
            except Exception as exc:
                self.ros.get_logger().warn(f"Failed to clear Nav2 costmap service {name}: {exc}")
        return bool(ok_any)

    def _world_from_local_xy(self, local_xy: np.ndarray) -> Optional[np.ndarray]:
        robot_pose = self._get_robot_pose2d()
        if robot_pose is None:
            return None
        robot_xy, robot_yaw = robot_pose
        c = math.cos(float(robot_yaw))
        s = math.sin(float(robot_yaw))
        lx = float(local_xy[0])
        ly = float(local_xy[1])
        return np.array(
            [
                float(robot_xy[0]) + c * lx - s * ly,
                float(robot_xy[1]) + s * lx + c * ly,
            ],
            dtype=np.float32,
        )

    def _slam_occupancy_at_world(self, world_xy: np.ndarray) -> Optional[int]:
        """Return occupancy at a target expressed in self.pose_frame.

        Nav2 goals and RL waypoints are expressed in self.pose_frame.  When
        self.pose_frame == "odom", raw slam_toolbox /map is normally still in
        frame_id="map".  Sampling that raw grid with odom coordinates produces
        shifted occupancy tests and makes the visual layers look inconsistent.
        Therefore this function samples a SLAM grid transformed into pose_frame.
        """
        slam_map = getattr(self, "_slam_transform_cache_msg", None)
        if slam_map is None:
            slam_map = getattr(self.ros, "slam_map", None)
        if slam_map is None:
            return None

        source_frame = str(getattr(getattr(slam_map, "header", None), "frame_id", "") or "map").strip().lstrip("/") or "map"
        target_frame = str(getattr(self, "pose_frame", "odom") or "odom").strip().lstrip("/") or "odom"
        if source_frame != target_frame:
            slam_map = self._transform_slam_map_to_frame(slam_map, target_frame=target_frame)
            if slam_map is None:
                return None

        info = slam_map.info
        width = int(info.width)
        height = int(info.height)
        resolution = float(info.resolution)
        if width <= 0 or height <= 0 or resolution <= 1e-9:
            return None

        origin = info.origin
        ox = float(origin.position.x)
        oy = float(origin.position.y)
        oyaw = self._yaw_from_quaternion_xyzw(
            float(origin.orientation.x),
            float(origin.orientation.y),
            float(origin.orientation.z),
            float(origin.orientation.w),
        )
        dx = float(world_xy[0]) - ox
        dy = float(world_xy[1]) - oy
        c = math.cos(-oyaw)
        s = math.sin(-oyaw)
        mx = int(math.floor((c * dx - s * dy) / resolution))
        my = int(math.floor((s * dx + c * dy) / resolution))
        if mx < 0 or mx >= width or my < 0 or my >= height:
            return None
        idx = my * width + mx
        try:
            return int(slam_map.data[idx])
        except Exception:
            return None

    def _local_lidar_goal_clear(self, local_xy: np.ndarray, margin_m: float = 0.10) -> bool:
        distance = float(np.linalg.norm(np.asarray(local_xy, dtype=np.float32)))
        if distance <= 1e-6:
            return False
        heading = math.atan2(float(local_xy[1]), float(local_xy[0]))
        half_width = math.radians(12.0)
        required_clear = min(1.20, max(0.12, distance + float(margin_m)))
        sector_min = self._scan_min_distance_in_sector(
            scan=self.ros.scan,
            center_angle=heading,
            half_width_rad=half_width,
            max_considered_range=1.20,
        )
        return bool(sector_min >= required_clear)

    def _is_nav2_goal_valid(self, local_xy: np.ndarray, world_xy: np.ndarray) -> tuple[bool, str]:
        """
        Validate a short Nav2 goal before sending it.

        Important policy:
          - LiDAR collision check is a hard gate.
          - SLAM occupancy is a soft gate for short local goals.

        The earlier version rejected ``outside_slam_map`` and ``unknown_slam_cell``
        as hard failures. In this project that caused a deadlock after reset:
        the SLAM gate may still be ``pre_reset``/``accept_delay`` while the local
        LiDAR clearly says the short goal is free. The env then skipped every Nav2
        goal and produced zero motion. Therefore, unknown/out-of-bounds SLAM cells
        are allowed when the local LiDAR ray to the goal is clear. Occupied cells
        remain rejected.
        """
        local_xy = np.asarray(local_xy, dtype=np.float32)
        world_xy = np.asarray(world_xy, dtype=np.float32)
        if local_xy.shape[0] < 2 or world_xy.shape[0] < 2:
            return False, "bad_shape"
        if not np.all(np.isfinite(local_xy[:2])) or not np.all(np.isfinite(world_xy[:2])):
            return False, "nonfinite"

        dist = float(np.linalg.norm(local_xy[:2]))
        if dist < max(0.05, self.waypoint_reached_tolerance * 0.5):
            return False, "too_close"
        if dist > max(self.waypoint_max_distance * 1.35, self.waypoint_max_distance + 0.25):
            return False, "too_far"

        # Local LiDAR is the strongest short-horizon safety signal.
        if not self._local_lidar_goal_clear(local_xy, margin_m=0.08):
            return False, "lidar_blocked"

        # If the accepted SLAM snapshot is not ready after reset, do not use the
        # raw /map bounds as a hard rejection criterion. Nav2 can still consume a
        # short map-frame goal once TF is available; if it cannot, the fallback
        # nudge below prevents a zero-motion training loop.
        slam_gate = str(getattr(self, "_last_slam_gate_reason", "unknown"))
        occ = None if slam_gate != "accepted" else self._slam_occupancy_at_world(world_xy)

        if occ is None:
            return True, f"lidar_only_{slam_gate}"
        if occ < 0:
            return True, "lidar_only_unknown"
        if occ >= 65:
            return False, "occupied_slam_cell"
        return True, "valid"

    def _find_safe_local_fallback_goal(
        self,
        preferred_heading: float = 0.0,
        preferred_distance: Optional[float] = None,
    ) -> tuple[np.ndarray, np.ndarray, float, float, str, bool, str]:
        """
        Find a short local goal that is likely to be accepted by Nav2 and does not
        point into a nearby wall.

        This is intentionally LiDAR-first. Around walls, the SLAM occupancy/costmap
        may be conservative or slightly misaligned; the policy-selected polar goal
        can point into the inflated wall side. If we simply skip the goal or run the
        direct controller toward that blocked point, the robot stays still. Instead,
        sample a small set of local goals and choose the clearest sector.
        """
        min_d = max(float(self.waypoint_min_distance), 0.16)
        max_d = max(float(self.waypoint_max_distance), min_d + 0.05)
        base_d = float(preferred_distance) if preferred_distance is not None else min(0.38, max_d)
        base_d = float(np.clip(base_d, min_d, max_d))

        # Prefer forward-ish arcs, but include wider side angles so a robot next to
        # a wall can slide along the corridor instead of repeatedly aiming into the wall.
        angle_degs = [0, 12, -12, 24, -24, 36, -36, 50, -50, 65, -65, 80, -80]
        # When the policy suggested a non-zero heading, test nearby headings first.
        preferred_deg = math.degrees(float(preferred_heading))
        if np.isfinite(preferred_deg) and abs(preferred_deg) > 1.0:
            for off in (0, 12, -12, 24, -24):
                a = int(round(preferred_deg + off))
                if -90 <= a <= 90 and a not in angle_degs:
                    angle_degs.insert(0, a)

        distances = sorted({
            round(min(max_d, max(min_d, base_d)), 3),
            round(min(max_d, max(min_d, 0.28)), 3),
            round(min(max_d, max(min_d, 0.42)), 3),
            round(min(max_d, max(min_d, 0.55)), 3),
        })

        best: tuple[float, np.ndarray, np.ndarray, float, float, str] | None = None
        best_reason = "no_candidate"
        max_angle = max(float(self.waypoint_max_angle_rad), math.radians(85.0))

        for deg in angle_degs:
            ang = math.radians(float(deg))
            if abs(ang) > max_angle:
                continue
            # Estimate clearance before checking each distance. This lets us score
            # candidates by how far they are from the wall/inflated obstacle side.
            clearance = self._scan_min_distance_in_sector(
                scan=self.ros.scan,
                center_angle=ang,
                half_width_rad=math.radians(18.0),
                max_considered_range=1.25,
            )
            for d in distances:
                # Keep a small safety margin: do not choose a point deeper than the
                # measured clearance in that direction.
                if clearance < d + 0.10:
                    best_reason = "fallback_lidar_blocked"
                    continue
                local = np.array([float(d) * math.cos(ang), float(d) * math.sin(ang)], dtype=np.float32)
                world = self._world_from_local_xy(local)
                if world is None:
                    best_reason = "no_pose"
                    continue
                valid, reason = self._is_nav2_goal_valid(local, world)
                if not valid:
                    best_reason = reason
                    continue

                # Score: clearance first, then forward preference, then proximity to
                # the policy-suggested heading. This avoids sharp side goals unless
                # the wall makes them necessary.
                forward_bias = 0.25 * math.cos(ang)
                policy_bias = 0.12 * math.cos(self._normalize_angle(ang - float(preferred_heading)))
                distance_bias = 0.08 * (float(d) / max(max_d, 1e-6))
                score = float(clearance) + forward_bias + policy_bias + distance_bias - 0.002 * abs(deg)
                if best is None or score > best[0]:
                    best = (score, local, world.astype(np.float32), float(d), float(ang), reason)

        if best is None:
            dummy_local = np.array([min_d, 0.0], dtype=np.float32)
            dummy_world = self._world_from_local_xy(dummy_local)
            if dummy_world is None:
                dummy_world = np.array([np.nan, np.nan], dtype=np.float32)
            return dummy_local, np.asarray(dummy_world, dtype=np.float32), float(min_d), 0.0, "no_free_fallback", False, best_reason

        _, local, world, d, ang, reason = best
        return local, world, d, ang, "free_space_fallback", True, reason

    def _execute_wall_escape_motion(self, max_steps: int = 6) -> tuple[np.ndarray, float, float, float]:
        """
        Last-resort motion when both Nav2 and local goal validation reject the
        waypoint near a wall. Rotate toward the more open side and add a small
        forward component only if the forward sector is not blocked.
        """
        executed: list[np.ndarray] = []
        min_action_obstacle_distance = 999.0
        max_action_obstacle_score = 0.0
        last_front_obstacle_distance = 999.0

        left = self._scan_min_distance_in_sector(
            scan=self.ros.scan,
            center_angle=math.radians(55.0),
            half_width_rad=math.radians(25.0),
            max_considered_range=1.20,
        )
        right = self._scan_min_distance_in_sector(
            scan=self.ros.scan,
            center_angle=-math.radians(55.0),
            half_width_rad=math.radians(25.0),
            max_considered_range=1.20,
        )
        front = self._scan_min_distance_in_sector(
            scan=self.ros.scan,
            center_angle=0.0,
            half_width_rad=math.radians(22.0),
            max_considered_range=1.20,
        )
        turn_sign = 1.0 if left >= right else -1.0
        angular = float(turn_sign * min(self.max_angular_speed, 0.70))
        # Small forward nudge only when the front is not too close. This prevents
        # pure zero-motion near an inflated wall while still avoiding collision.
        linear = 0.0
        if front > max(0.24, self.waypoint_front_stop_distance + 0.03):
            linear = min(self.max_linear_speed, max(self.waypoint_min_linear_speed, 0.06))

        cmd = np.array([linear, angular], dtype=np.float32)
        for _ in range(max(int(max_steps), 1)):
            d_obs, s_obs, f_obs = self._compute_lidar_action_obstacle_risk(cmd)
            min_action_obstacle_distance = min(float(min_action_obstacle_distance), float(d_obs))
            max_action_obstacle_score = max(float(max_action_obstacle_score), float(s_obs))
            last_front_obstacle_distance = float(f_obs)
            if linear > 0.0 and f_obs < max(0.18, self.waypoint_front_stop_distance * 0.80):
                cmd = np.array([0.0, angular], dtype=np.float32)
            self.ros.publish_cmd_vel(float(cmd[0]), float(cmd[1]))
            executed.append(cmd.copy())
            self.ros.spin_steps(num_spins=1, timeout_sec=0.01)
            if self.use_world_step:
                self.ros.step_simulation(self.sim_steps_per_action)
            else:
                time.sleep(self.control_dt)
            self._last_controller_steps += 1
            if self._check_collision() or self._check_fallen():
                self.ros.stop_robot()
                break

        executed_action = np.mean(np.stack(executed, axis=0), axis=0).astype(np.float32) if executed else np.zeros(2, dtype=np.float32)
        if min_action_obstacle_distance >= 999.0:
            d_obs, s_obs, f_obs = self._compute_lidar_action_obstacle_risk(executed_action)
            min_action_obstacle_distance = float(d_obs)
            max_action_obstacle_score = float(s_obs)
            last_front_obstacle_distance = float(f_obs)
        return executed_action, float(min_action_obstacle_distance), float(max_action_obstacle_score), float(last_front_obstacle_distance)

    def _decode_nav2_goal_action(
        self,
        policy_action: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, float, float, str, bool, str]:
        """
        Decode a SAC action into a Nav2 goal with validity gating and fallback.

        For action_mode=nav2, path-conditioned goals are only used when the internal
        ExplorationGridMap has a reachable path. If target_reachable=False, using the
        path-conditioned angle is dangerous because it can point to an unreachable
        priority/frontier target. In that case we fall back to a local free-space goal
        selected from short LiDAR/SLAM-validated candidates.
        """
        stats = self.last_map_stats
        action = np.asarray(policy_action, dtype=np.float32)
        if action.size < 2:
            action = np.pad(action, (0, 2 - action.size), mode="constant")
        a0 = float(np.clip(action[0], 0.0, 1.0))
        a1 = float(np.clip(action[1], -1.0, 1.0))

        # 1) Use path-conditioned waypoint only when the path is actually reachable.
        if self.waypoint_action_type == "path" and stats is not None and bool(getattr(stats, "target_reachable", False)):
            local_xy, world_xy, distance, heading = self._decode_waypoint_action(action)
            valid, reason = self._is_nav2_goal_valid(local_xy, world_xy)
            if valid:
                return local_xy, world_xy, distance, heading, "path", True, reason

        # 2) Try the policy as a polar local goal. This is safer when path=False.
        old_type = self.waypoint_action_type
        try:
            self.waypoint_action_type = "polar"
            local_xy, world_xy, distance, heading = self._decode_waypoint_action(action)
        finally:
            self.waypoint_action_type = old_type
        valid, reason = self._is_nav2_goal_valid(local_xy, world_xy)
        if valid:
            self._last_waypoint_action_type = "polar_fallback"
            return local_xy, world_xy, distance, heading, "polar_fallback", True, reason

        # 3) Clearance-scored local fallback. This is the important wall case:
        #    if the policy-selected goal is close to an inflated wall, do not keep
        #    trying that blocked point. Select a short goal in the clearest local sector.
        fb_local, fb_world, fb_dist, fb_heading, fb_src, fb_valid, fb_reason = self._find_safe_local_fallback_goal(
            preferred_heading=float(heading),
            preferred_distance=float(distance),
        )
        if fb_valid:
            self._last_waypoint_action_type = fb_src
            return fb_local, fb_world, fb_dist, fb_heading, fb_src, True, fb_reason

        # 4) No safe goal. Return the original polar goal for visualization, but mark invalid.
        self._last_waypoint_action_type = "wall_escape"
        return local_xy, world_xy, float(distance), float(heading), "wall_escape", False, fb_reason

    def _warn_episode_reset_reason(
        self,
        *,
        reason: str,
        reward: float,
        terminated: bool,
        truncated: bool,
        map_stats: MapUpdateStats,
    ) -> None:
        """Emit one compact warning explaining why Gym will reset the episode."""
        reason = str(reason or "unknown")
        if reason == "none":
            reason = "time_limit" if truncated else "terminated"

        parts = [
            "EPISODE_RESET_REASON",
            f"episode={int(getattr(self, 'episode_index', 0))}",
            f"reason={reason}",
            f"terminated={bool(terminated)}",
            f"truncated={bool(truncated)}",
            f"step={int(getattr(self, 'step_count', 0))}/{int(getattr(self, 'max_episode_steps', 0))}",
            f"r={float(reward):+.3f}",
            f"Glive={float(getattr(self, '_episode_discounted_return', 0.0)):+.3f}",
            f"Gstart={float(getattr(self, '_episode_start_discounted_return', 0.0)):+.3f}",
            f"Gsum={float(getattr(self, '_episode_reward_sum', 0.0)):+.3f}",
            f"R{int(getattr(self, '_reward_window_n', 100))}={sum(getattr(self, '_recent_reward_window', [])):+.3f}",
            f"coverage={float(getattr(map_stats, 'coverage_ratio', 0.0)):.3f}",
            f"priority={float(getattr(map_stats, 'priority_score', 0.0)):.3f}",
            f"target={str(getattr(map_stats, 'target_type', 'none'))}",
        ]

        if reason in {"collision", "out_of_bounds", "fallen_or_drop"}:
            parts.extend([
                f"collision_min={float(getattr(self, '_last_collision_global_min', 999.0)):.3f}",
                f"front_min={float(getattr(self, '_last_collision_front_min', 999.0)):.3f}",
                f"fallen_reason={str(getattr(self, '_last_fallen_reason', 'none'))}",
                f"oob={str(getattr(self, '_last_out_of_bounds_reason', 'none'))}",
            ])
        elif reason == "priority_stuck_restart":
            parts.extend([
                f"pstuck={int(getattr(self, 'priority_stuck_steps', 0))}/{int(getattr(self, 'priority_stuck_restart_steps', 0))}",
                f"priority_reason={str(getattr(self, '_last_priority_stuck_reason', 'none'))}",
                f"clear_gain={float(getattr(map_stats, 'priority_clear_gain', 0.0)):.4f}",
                f"recheck_gain={float(getattr(map_stats, 'priority_rechecked_gain', 0.0)):.4f}",
            ])
        elif reason == "lidar_empty_restart":
            parts.extend([
                f"lempty={int(getattr(self, 'lidar_empty_steps', 0))}/{int(getattr(self, 'lidar_empty_timeout_steps', 0))}",
                f"lidar_reason={str(getattr(self, '_last_lidar_empty_reason', 'none'))}",
                f"valid_beams={int(getattr(self, '_last_lidar_valid_beams', 0))}",
                f"nearest={float(getattr(self, '_last_lidar_nearest_detection', 999.0)):.3f}",
            ])
        elif reason in {"timeout_low_coverage", "time_limit"}:
            parts.extend([
                f"stall={int(getattr(self, 'explored_stall_steps', 0))}",
                f"target_cov={float(getattr(self, 'target_coverage_ratio', 0.0)):.3f}",
            ])

        self.ros.get_logger().warn(" | ".join(parts))

    def _handle_unsafe_terminal(self, reason: str) -> None:
        """Prepare a clean next episode after collision-like unsafe states.

        This is used for physical collision, out-of-bounds escape, and tilt/fall.
        All three should cancel the active Nav2 goal and publish zero velocity before
        Gym/SB3 calls reset(), otherwise Nav2/controller state can keep rotating the
        robot during the next episode.
        """
        self.ros.get_logger().warn(
            f"Unsafe terminal state detected: {reason}. Stopping robot and preparing reset."
        )
        self.ros.stop_robot()

        if self.action_mode == "nav2":
            if self.collision_cancel_nav2_goal:
                self._cancel_nav2_goal()
            if self.collision_clear_nav2_costmaps:
                self._clear_nav2_costmaps(wait_timeout_sec=0.35)

        self.ros.spin_steps(num_spins=5, timeout_sec=0.001)

    def _handle_collision_terminal(self) -> None:
        """Backward-compatible wrapper."""
        self._handle_unsafe_terminal("collision")


    def _ensure_yaml_node_param(self, text: str, node: str, key: str, value: str) -> str:
        """Ensure a top-level Nav2 node has a ros__parameters entry.

        This is deliberately text-based so the project does not depend on
        PyYAML at runtime.  It preserves the upstream nav2_bringup parameter
        file and only forces the few parameters required for TurtleBot3 Jazzy
        simulation.
        """
        node_re = re.compile(rf"(^|\n){re.escape(node)}:\n\s+ros__parameters:\n")
        m = node_re.search(text)
        if not m:
            if not text.endswith("\n"):
                text += "\n"
            return text + f"\n{node}:\n  ros__parameters:\n    {key}: {value}\n"

        start = m.end()
        next_node = re.search(r"\n\S[^:\n]*:\n\s+ros__parameters:\n", text[start:])
        end = start + next_node.start() if next_node else len(text)
        block = text[start:end]

        key_re = re.compile(rf"(^\s*{re.escape(key)}\s*:\s*).*$", re.MULTILINE)
        if key_re.search(block):
            block = key_re.sub(rf"\g<1>{value}", block)
            return text[:start] + block + text[end:]

        return text[:start] + f"    {key}: {value}\n" + text[start:]

    def _ensure_yaml_nested_param(self, text: str, node: str, section: str, key: str, value: str) -> str:
        """Ensure node.ros__parameters.section.key exists in a Nav2 YAML file.

        This stays text-based to avoid a runtime PyYAML dependency. It is used
        for controller plugin parameters such as controller_server/FollowPath.
        """
        node_re = re.compile(rf"(^|\n){re.escape(node)}:\n\s+ros__parameters:\n")
        m = node_re.search(text)
        if not m:
            if not text.endswith("\n"):
                text += "\n"
            return text + f"\n{node}:\n  ros__parameters:\n    {section}:\n      {key}: {value}\n"

        start = m.end()
        next_node = re.search(r"\n\S[^:\n]*:\n\s+ros__parameters:\n", text[start:])
        end = start + next_node.start() if next_node else len(text)
        block = text[start:end]

        section_re = re.compile(rf"(^    {re.escape(section)}:\n)", re.MULTILINE)
        sm = section_re.search(block)
        if not sm:
            block = f"    {section}:\n      {key}: {value}\n" + block
            return text[:start] + block + text[end:]

        sec_start = sm.end()
        # Next sibling at exactly four spaces, or end of the node block.
        next_sec = re.search(r"\n    \S[^:\n]*:\n", block[sec_start:])
        sec_end = sec_start + next_sec.start() if next_sec else len(block)
        sec_block = block[sec_start:sec_end]
        key_re = re.compile(rf"(^      {re.escape(key)}\s*:\s*).*$", re.MULTILINE)
        if key_re.search(sec_block):
            sec_block = key_re.sub(rf"\g<1>{value}", sec_block)
        else:
            sec_block = f"      {key}: {value}\n" + sec_block
        block = block[:sec_start] + sec_block + block[sec_end:]
        return text[:start] + block + text[end:]

    def _resolve_default_nav2_params_file(self) -> str:
        """Return a TurtleBot3-tuned Nav2 params file when available."""
        candidates: list[str] = []
        model = os.environ.get("TURTLEBOT3_MODEL", "burger").strip() or "burger"
        try:
            from ament_index_python.packages import get_package_share_directory
            tb3_share = get_package_share_directory("turtlebot3_navigation2")
            for name in (f"{model}.yaml", "burger.yaml", "waffle.yaml", "nav2_params.yaml"):
                candidates.append(str(Path(tb3_share) / "param" / name))
                candidates.append(str(Path(tb3_share) / "params" / name))
        except Exception:
            pass
        try:
            from ament_index_python.packages import get_package_share_directory
            share_dir = get_package_share_directory("nav2_bringup")
            candidates.append(str(Path(share_dir) / "params" / "nav2_params.yaml"))
        except Exception:
            pass

        for pkg, rels in (
            ("turtlebot3_navigation2", [f"share/turtlebot3_navigation2/param/{model}.yaml", "share/turtlebot3_navigation2/param/burger.yaml"]),
            ("nav2_bringup", ["share/nav2_bringup/params/nav2_params.yaml"]),
        ):
            try:
                prefix = subprocess.check_output(
                    ["ros2", "pkg", "prefix", pkg],
                    stderr=subprocess.DEVNULL,
                    text=True,
                    timeout=2.0,
                ).strip()
                for rel in rels:
                    candidates.append(str(Path(prefix) / rel))
            except Exception:
                pass

        for c in candidates:
            if c and Path(c).is_file():
                return c
        return ""

    def _prepare_nav2_stamped_params_file(self) -> str:
        """Create a runtime Nav2 params file that publishes TwistStamped /cmd_vel.

        This project uses geometry_msgs/msg/TwistStamped on /cmd_vel.
        Only force enable_stamped_cmd_vel=true and keep all controller plugin
        parameters from the upstream TurtleBot3 YAML intact. Changing controller
        plugin parameters here can prevent controller_server from configuring,
        which makes /follow_path unavailable.
        """
        if self._nav2_stamped_params_prepared and self._nav2_runtime_params_file:
            if Path(self._nav2_runtime_params_file).is_file():
                return self._nav2_runtime_params_file

        src = str(self.nav2_params_file or "").strip()
        if not src:
            src = self._resolve_default_nav2_params_file()

        if not src or not Path(src).is_file():
            self.ros.get_logger().warn(
                "Nav2 params file was not found. Starting Nav2 without stamped cmd_vel params; "
                "if /cmd_vel is not geometry_msgs/msg/TwistStamped, this project may not move."
            )
            self._nav2_stamped_params_prepared = True
            self._nav2_runtime_params_file = ""
            return ""

        try:
            text = Path(src).read_text()
        except Exception as exc:
            self.ros.get_logger().warn(f"Failed to read Nav2 params file '{src}': {exc}")
            self._nav2_stamped_params_prepared = True
            self._nav2_runtime_params_file = ""
            return ""

        # Patch all existing explicit entries first, then ensure key Nav2 nodes
        # have the parameter even if the distro's default file omits it.
        # This project expects geometry_msgs/msg/TwistStamped on /cmd_vel.
        text = re.sub(
            r"(enable_stamped_cmd_vel\s*:\s*)(true|True|false|False)\b",
            r"\1true",
            text,
            flags=re.IGNORECASE,
        )

        # Keep Nav2 controller parameters intact.
        #
        # The previous forward-motion patch injected RPP/DWB-specific parameters
        # into whatever controller plugin was installed.  On some TurtleBot3/Jazzy
        # setups this prevents controller_server from configuring, so /follow_path
        # never appears.  For runtime launch safety, only force the message type
        # compatibility knob below; all controller tuning remains exactly as the
        # installed TurtleBot3 YAML defines it.
        for node in (
            "controller_server",
            "velocity_smoother",
            "behavior_server",
            "collision_monitor",
        ):
            text = self._ensure_yaml_node_param(text, node, "enable_stamped_cmd_vel", "true")
            text = self._ensure_yaml_node_param(
                text,
                node,
                "use_sim_time",
                "true" if self.nav2_use_sim_time else "false",
            )

        # Keep the rest of the upstream file intact.  Use a deterministic tmp name
        # so repeated resets do not create unbounded files.
        out = Path("/tmp") / f"turtlebot3_rl_nav2_stamped_params_{os.getpid()}.yaml"
        try:
            out.write_text(text)
        except Exception as exc:
            self.ros.get_logger().warn(f"Failed to write runtime Nav2 params file '{out}': {exc}")
            self._nav2_stamped_params_prepared = True
            self._nav2_runtime_params_file = ""
            return ""

        self._nav2_stamped_params_prepared = True
        self._nav2_runtime_params_file = str(out)
        self.ros.get_logger().info(
            f"NAV2_TWIST_STAMPED_CMD_VEL_PARAMS | source={src} | runtime={self._nav2_runtime_params_file} | "
            "enable_stamped_cmd_vel=true | cmd_vel_type=geometry_msgs/msg/TwistStamped | "
            "controller_params=upstream_safe | follow_path_server_expected | cancel_sync_restore"
        )
        return self._nav2_runtime_params_file

    def _start_nav2_process_if_needed(self) -> None:
        """Start Nav2 once, with stamped cmd_vel params and a visible log path."""
        if not self.nav2_auto_start or self.nav2_proc is not None:
            return

        cmd = [
            "ros2",
            "launch",
            self.nav2_launch_package,
            self.nav2_launch_file,
            f"use_sim_time:={'true' if self.nav2_use_sim_time else 'false'}",
            "autostart:=true",
        ]
        nav2_runtime_params = self._prepare_nav2_stamped_params_file()
        if nav2_runtime_params:
            cmd.append(f"params_file:={nav2_runtime_params}")
        elif self.nav2_params_file:
            cmd.append(f"params_file:={self.nav2_params_file}")

        self._nav2_log_path = f"/tmp/turtlebot3_rl_nav2_{os.getpid()}.log"
        self.ros.get_logger().warn(
            "Nav2 motion server is not available. Starting Nav2 internally:\n"
            + " ".join(cmd)
            + f"\nNAV2_LOG_FILE | {self._nav2_log_path}"
        )
        try:
            self._nav2_log_handle = open(self._nav2_log_path, "a", buffering=1)
            self.nav2_proc = subprocess.Popen(
                cmd,
                stdout=self._nav2_log_handle,
                stderr=subprocess.STDOUT,
                text=True,
                start_new_session=True,
            )
        except Exception as exc:
            self.ros.get_logger().error(f"Failed to start Nav2 internally: {exc}")
            self.nav2_proc = None
            try:
                if self._nav2_log_handle is not None:
                    self._nav2_log_handle.close()
            except Exception:
                pass
            self._nav2_log_handle = None

    def _ensure_nav2_action_server(self, timeout_sec: Optional[float] = None) -> bool:
        """Ensure NavigateToPose action server is available; optionally launch Nav2."""
        if self.nav2_client is None:
            return False

        wait_timeout = self.nav2_wait_timeout_sec if timeout_sec is None else float(timeout_sec)

        if self.nav2_client.wait_for_server(timeout_sec=0.05):
            return True

        self._start_nav2_process_if_needed()

        start = time.time()
        while time.time() - start < wait_timeout:
            self.ros.spin_steps(num_spins=1, timeout_sec=0.05)
            if self.nav2_client.wait_for_server(timeout_sec=0.05):
                self.ros.get_logger().info(
                    f"Nav2 NavigateToPose action server is now available: {self.nav2_action_name}"
                )
                return True

        now = time.time()
        if now - float(getattr(self, "_last_nav2_unavailable_log_time", 0.0)) > 8.0:
            self._last_nav2_unavailable_log_time = now
            self.ros.get_logger().error(
                f"Nav2 NavigateToPose action server not available: {self.nav2_action_name}. "
                f"Nav2 log: {self._nav2_log_path or '(none)'}"
            )
        return False

    def _ensure_nav2_follow_path_server(self, timeout_sec: Optional[float] = None) -> bool:
        """Ensure Nav2 controller_server /follow_path is available; optionally launch Nav2."""
        if self.nav2_follow_path_client is None:
            return False

        wait_timeout = self.nav2_wait_timeout_sec if timeout_sec is None else float(timeout_sec)

        if self.nav2_follow_path_client.wait_for_server(timeout_sec=0.05):
            return True

        self._start_nav2_process_if_needed()

        start = time.time()
        while time.time() - start < wait_timeout:
            self.ros.spin_steps(num_spins=1, timeout_sec=0.05)
            if self.nav2_follow_path_client.wait_for_server(timeout_sec=0.05):
                self.ros.get_logger().info(
                    f"Nav2 FollowPath action server is now available: {self.nav2_follow_path_action_name}"
                )
                return True

        now = time.time()
        if now - float(getattr(self, "_last_nav2_unavailable_log_time", 0.0)) > 8.0:
            self._last_nav2_unavailable_log_time = now
            self.ros.get_logger().error(
                f"Nav2 FollowPath action server not available: {self.nav2_follow_path_action_name}. "
                f"Nav2 log: {self._nav2_log_path or '(none)'}"
            )
        return False

    def _ensure_nav2_motion_server(self, timeout_sec: Optional[float] = None) -> bool:
        """Ensure a Nav2-owned motion server is available.

        For short RL waypoint actions on a freshly reset SLAM map, FollowPath is
        the reliable motion primitive: it lets controller_server track a local
        path while all RViz/RL map layers stay locked to SLAM /map.  The path is
        expressed in odom after transforming the selected map-frame waypoint.
        Direct /cmd_vel is still never published by the RL node.
        """
        wait_timeout = self.nav2_wait_timeout_sec if timeout_sec is None else float(timeout_sec)

        if self._ensure_nav2_follow_path_server(timeout_sec=wait_timeout):
            self._nav2_use_follow_path_controller = True
            motion_frame = str(getattr(self, "nav2_motion_frame", "odom") or "odom").strip().lstrip("/")
            self.ros.get_logger().info(
                "NAV2_MOTION_SERVER_READY | using /follow_path; "
                "controller_server owns /cmd_vel; "
                f"path_frame={motion_frame} | map layers remain locked to /map | cmd_vel_type=TwistStamped"
            )
            return True

        # Fallback only if controller_server is genuinely unavailable.
        if self._ensure_nav2_action_server(timeout_sec=max(0.25, wait_timeout)):
            self._nav2_use_follow_path_controller = False
            self.ros.get_logger().warn(
                "NAV2_FOLLOW_PATH_UNAVAILABLE | fallback to /navigate_to_pose; "
                "BT/planner owns /cmd_vel; goal_frame=map | map layers remain locked to /map"
            )
            return True

        self._nav2_use_follow_path_controller = False
        self.ros.get_logger().error(
            "NAV2_MOTION_SERVER_UNAVAILABLE | neither /follow_path nor /navigate_to_pose is ready. "
            f"Nav2 log: {self._nav2_log_path or '(none)'}"
        )
        return False


    @staticmethod
    def _nav2_status_name(status: int) -> str:
        if GoalStatus is None:
            return str(status)
        names = {
            GoalStatus.STATUS_UNKNOWN: "UNKNOWN",
            GoalStatus.STATUS_ACCEPTED: "ACCEPTED",
            GoalStatus.STATUS_EXECUTING: "EXECUTING",
            GoalStatus.STATUS_CANCELING: "CANCELING",
            GoalStatus.STATUS_SUCCEEDED: "SUCCEEDED",
            GoalStatus.STATUS_CANCELED: "CANCELED",
            GoalStatus.STATUS_ABORTED: "ABORTED",
        }
        return names.get(int(status), str(status))

    @staticmethod
    def _quaternion_from_yaw(yaw: float) -> tuple[float, float, float, float]:
        half = float(yaw) * 0.5
        return 0.0, 0.0, math.sin(half), math.cos(half)


    def _wait_future_done(self, future, timeout_sec: float) -> bool:
        start = time.time()
        while time.time() - start < float(timeout_sec):
            self.ros.spin_steps(num_spins=1, timeout_sec=0.01)
            if future.done():
                return True
        return bool(future.done())

    def _cancel_nav2_goal(self, wait_sec: Optional[float] = None) -> None:
        goal_handle = getattr(self, "_nav2_goal_handle", None)
        if goal_handle is None:
            return
        try:
            cancel_future = goal_handle.cancel_goal_async()
            timeout = self.nav2_cancel_wait_sec if wait_sec is None else max(float(wait_sec), 0.0)
            if timeout > 0.0:
                self._wait_future_done(cancel_future, timeout_sec=timeout)
            else:
                # Let rclpy enqueue/send the cancel request once, but do not block
                # the RL loop on Nav2's cancel response.  The next goal can preempt
                # the old one anyway.
                self.ros.spin_steps(num_spins=1, timeout_sec=0.0)
        except Exception:
            pass
        self._nav2_goal_handle = None

    def _cancel_nav2_goal_sync(self, reason: str = "") -> None:
        """Backward-compatible synchronous Nav2 goal cancel wrapper.

        Several escape paths call this method before BackUp/timeout handling.
        Older versions only had _cancel_nav2_goal(), which caused an
        AttributeError inside check_env or the first real Nav2 timeout.  Keep
        this wrapper tiny and exception-safe so cancel never kills training.
        """
        try:
            if reason:
                self._last_nav2_cancel_reason = str(reason)
            self._cancel_nav2_goal(wait_sec=self.nav2_cancel_wait_sec)
        except Exception as exc:
            try:
                self.ros.get_logger().warn(
                    f"NAV2_CANCEL_IGNORED | reason={reason} | error={exc}"
                )
            except Exception:
                pass

    def _execute_direct_fallback_to_goal(
        self,
        target_world_xy: np.ndarray,
        max_steps: int = 6,
    ) -> tuple[np.ndarray, float, float, float]:
        """
        Conservative last-resort fallback used only when Nav2 rejects/aborts a
        very short local goal. This keeps the RL loop from degenerating into
        thousands of zero-motion steps while Nav2/SLAM/costmap recover.

        It still respects the same LiDAR obstacle gates as the waypoint
        controller and runs for a small bounded number of control ticks.
        """
        executed: list[np.ndarray] = []
        min_action_obstacle_distance = 999.0
        max_action_obstacle_score = 0.0
        last_front_obstacle_distance = 999.0

        for _ in range(max(int(max_steps), 1)):
            cmd, reached = self._waypoint_controller_command(target_world_xy)
            if reached:
                self._last_waypoint_reached = True
                break

            d_obs, s_obs, f_obs = self._compute_lidar_action_obstacle_risk(cmd)
            min_action_obstacle_distance = min(float(min_action_obstacle_distance), float(d_obs))
            max_action_obstacle_score = max(float(max_action_obstacle_score), float(s_obs))
            last_front_obstacle_distance = float(f_obs)

            # If the local controller also decides no safe forward motion exists,
            # do not spin forever. Return zero and let the next SAC action replan.
            if abs(float(cmd[0])) < self.linear_deadband and abs(float(cmd[1])) < self.angular_deadband:
                break

            self.ros.publish_cmd_vel(float(cmd[0]), float(cmd[1]))
            executed.append(np.asarray(cmd, dtype=np.float32))
            self._advance_world_after_command(target_delta_sec=self.control_dt)
            self._last_controller_steps += 1

            if self._check_collision() or self._check_fallen():
                break

        if executed:
            executed_action = np.mean(np.stack(executed, axis=0), axis=0).astype(np.float32)
        else:
            executed_action = np.zeros(2, dtype=np.float32)

        if min_action_obstacle_distance >= 999.0:
            d_obs, s_obs, f_obs = self._compute_lidar_action_obstacle_risk(executed_action)
            min_action_obstacle_distance = float(d_obs)
            max_action_obstacle_score = float(s_obs)
            last_front_obstacle_distance = float(f_obs)

        return (
            executed_action,
            float(min_action_obstacle_distance),
            float(max_action_obstacle_score),
            float(last_front_obstacle_distance),
        )


    def _estimate_executed_action_from_poses(
        self,
        start_pose,
        start_time: float,
    ) -> np.ndarray:
        """Estimate the macro-action actually executed by Nav2 during this env step."""
        end_pose = self._get_robot_pose2d()
        elapsed = max(time.time() - float(start_time), 1e-3)
        if start_pose is not None and end_pose is not None:
            start_xy, start_yaw = start_pose
            end_xy, end_yaw = end_pose
            dist_moved = float(np.linalg.norm(np.asarray(end_xy) - np.asarray(start_xy)))
            yaw_delta = self._normalize_angle(float(end_yaw) - float(start_yaw))
            linear_x = float(np.clip(dist_moved / elapsed, 0.0, self.max_linear_speed))
            angular_z = float(np.clip(yaw_delta / elapsed, -self.max_angular_speed, self.max_angular_speed))
            return np.array([linear_x, angular_z], dtype=np.float32)
        return np.zeros(2, dtype=np.float32)

    def _transform_pose_xy_yaw_between_frames(
        self,
        xy: np.ndarray,
        yaw: float,
        source_frame: str,
        target_frame: str,
    ) -> Optional[tuple[np.ndarray, float]]:
        """Transform a planar pose between ROS frames using the node TF buffer.

        Returns None instead of falling back to the source frame.  Silent fallback
        is exactly what makes RViz layers and Nav2 goals drift apart when map and
        odom are mixed.
        """
        source = str(source_frame or "").strip().lstrip("/")
        target = str(target_frame or "").strip().lstrip("/")
        if not source or not target or source == target:
            return np.asarray(xy, dtype=np.float32).copy(), float(yaw)
        try:
            fn = getattr(self.ros, "_transform_pose2d", None)
            if fn is None:
                return None
            return fn(
                xy=np.asarray(xy, dtype=np.float32),
                yaw=float(yaw),
                source_frame=source,
                target_frame=target,
            )
        except Exception:
            return None

    def _build_nav2_follow_path_goal(self, world_xy: np.ndarray) -> Optional[object]:
        """Build the older stable short FollowPath goal: start -> midpoint -> target.

        This intentionally avoids the forward-seed / spin-unstall experiments.  The
        robot may stop between local goals, but controller_server gets a normal short
        path and the next goal is sent only after cancel-sync.
        """
        if FollowPath is None:
            return None
        robot_pose = self._get_robot_pose2d()
        if robot_pose is None:
            return None
        robot_xy, robot_yaw = robot_pose
        target = np.asarray(world_xy, dtype=np.float32)
        if target.shape[0] < 2 or not np.all(np.isfinite(target[:2])):
            return None

        dx = float(target[0] - robot_xy[0])
        dy = float(target[1] - robot_xy[1])
        path_yaw = math.atan2(dy, dx) if (dx * dx + dy * dy) > 1e-8 else float(robot_yaw)

        goal = FollowPath.Goal()
        goal.path = NavPath()
        goal.path.header.frame_id = self.pose_frame
        goal.path.header.stamp = self._latest_tf_stamp()
        self._last_nav2_motion_frame = self.pose_frame

        def make_pose(x: float, y: float, yaw: float) -> PoseStamped:
            ps = PoseStamped()
            ps.header.frame_id = self.pose_frame
            ps.header.stamp = goal.path.header.stamp
            ps.pose.position.x = float(x)
            ps.pose.position.y = float(y)
            ps.pose.position.z = 0.0
            qx, qy, qz, qw = self._quaternion_from_yaw(float(yaw))
            ps.pose.orientation.x = qx
            ps.pose.orientation.y = qy
            ps.pose.orientation.z = qz
            ps.pose.orientation.w = qw
            return ps

        start_ps = make_pose(float(robot_xy[0]), float(robot_xy[1]), path_yaw)
        mid_xy = np.asarray(robot_xy, dtype=np.float32) + 0.55 * (
            target[:2] - np.asarray(robot_xy, dtype=np.float32)
        )
        mid_ps = make_pose(float(mid_xy[0]), float(mid_xy[1]), path_yaw)
        goal_ps = make_pose(float(target[0]), float(target[1]), path_yaw)
        goal.path.poses = [start_ps, mid_ps, goal_ps]

        # Empty IDs use the controller_server defaults from the TurtleBot3/Nav2 params file.
        try:
            goal.controller_id = ""
            goal.goal_checker_id = ""
            goal.progress_checker_id = ""
        except Exception:
            pass
        return goal

    def _execute_nav2_goal_action(self, policy_action: np.ndarray) -> tuple[np.ndarray, float, float, float]:
        """
        SAC가 고른 waypoint를 Nav2 NavigateToPose goal로 보낸다.

        이 모드는 더 이상 env 내부 P-controller가 /cmd_vel을 직접 만들지 않는다.
        Nav2 planner/controller/recovery가 실제 이동을 담당하고, env는 "도착 판정" 또는
        "시간 제한" 중 먼저 걸리는 조건에서 goal을 cancel하고 다음 SAC decision으로 넘어간다.

        주의:
          - Gazebo를 paused + multi_step으로 강제 전진시키면 Nav2 action server가 wall-clock 기준으로
            제대로 동작하지 않을 수 있으므로 학습 커맨드에서는 --disable-world-step 사용을 강하게 권장한다.
          - goal에 정확히 정차하지 않게 하기 위해 nav2_goal_reached_tolerance 안에 들어오면 goal을 cancel하고
            다음 waypoint를 바로 받는다.
        """
        if self.nav2_client is None:
            raise RuntimeError("Nav2 action client is not initialized. Use action_mode='nav2'.")

        local_xy, world_xy, distance, heading, goal_source, goal_valid, validation_reason = (
            self._decode_nav2_goal_action(policy_action)
        )

        self._last_nav2_goal_source = str(goal_source)
        self._last_nav2_goal_valid = bool(goal_valid)
        self._last_nav2_goal_validation = str(validation_reason)

        self._last_waypoint_local = local_xy.astype(np.float32).copy()
        self._last_waypoint_world = world_xy.astype(np.float32).copy()
        self._last_waypoint_distance = float(distance)
        self._last_waypoint_angle = float(heading)
        self._last_waypoint_reached = False
        self._last_waypoint_timed_out = False
        self._last_waypoint_final_error = float(self._waypoint_distance_to_target(world_xy))
        self._last_controller_steps = 0
        self._last_nav2_goal_accepted = False
        self._last_nav2_status = -1
        self._last_nav2_status_name = "none"

        self._publish_waypoint_visualization(
            waypoint_world_xy=world_xy,
            waypoint_local_xy=local_xy,
            distance=distance,
            heading=heading,
        )

        if not goal_valid:
            # Strict Nav2 mode: do not use internal /cmd_vel wall-escape fallback.
            # Return zero for this step and let the next SAC action generate another
            # Nav2 goal candidate.  If this repeats, the visible marker/status makes
            # the invalid target obvious instead of silently switching controller.
            self._last_nav2_goal_accepted = False
            self._last_nav2_status = -10
            self._last_nav2_status_name = f"SKIPPED_{validation_reason}_NAV2_ONLY"
            zero = np.zeros(2, dtype=np.float32)
            _, _, front = self._compute_lidar_action_obstacle_risk(zero)
            self._last_waypoint_timed_out = True
            return zero, 999.0, 0.0, float(front)

        # Use odom for motion monitoring even when the learning/RViz frame is map.
        # map->odom can be corrected by SLAM while the robot is physically moving;
        # odom is the stable frame for detecting whether Nav2 actually translated.
        motion_monitor_frame = str(getattr(self, "nav2_motion_frame", self.pose_frame) or self.pose_frame).strip().lstrip("/")
        start_pose = self._get_robot_pose2d(frame_id=motion_monitor_frame)
        start_time = time.time()

        if not self._ensure_nav2_motion_server(timeout_sec=self.nav2_wait_timeout_sec):
            self._last_nav2_goal_accepted = False
            self._last_nav2_status = -20
            self._last_nav2_status_name = "NAV2_UNAVAILABLE"
            raise RuntimeError(
                "Nav2 motion server is unavailable. /follow_path is preferred for short local waypoints; "
                "/navigate_to_pose is only a fallback. No internal /cmd_vel fallback is allowed."
            )

        # Prefer Nav2 controller_server FollowPath for short-horizon motion.
        # The RL/RViz maps stay in map, but the local path is transformed to odom
        # before it is sent to controller_server.
        use_follow_path = bool(getattr(self, "_nav2_use_follow_path_controller", False))
        if use_follow_path and not self._ensure_nav2_follow_path_server(timeout_sec=0.25):
            use_follow_path = False
            self._nav2_use_follow_path_controller = False

        if not self.nav2_preempt_without_cancel:
            self._cancel_nav2_goal_sync(reason="pre_send_nav2_goal_cancel_sync_restore")

        if use_follow_path:
            goal = self._build_nav2_follow_path_goal(world_xy)
            if goal is None:
                use_follow_path = False
                self._nav2_use_follow_path_controller = False

        if use_follow_path:
            self._last_nav2_status_name = "FOLLOW_PATH_SENT"
            send_future = self.nav2_follow_path_client.send_goal_async(goal)
        else:
            goal = NavigateToPose.Goal()
            goal.pose = PoseStamped()
            goal.pose.header.frame_id = self.pose_frame
            goal.pose.header.stamp = self._latest_tf_stamp()
            goal.pose.pose.position.x = float(world_xy[0])
            goal.pose.pose.position.y = float(world_xy[1])
            goal.pose.pose.position.z = 0.0

            robot_pose = self._get_robot_pose2d()
            if robot_pose is not None:
                robot_xy, robot_yaw = robot_pose
                # For short local goals, orient the goal along the target direction.
                # Holding the old robot yaw makes BT/controller rotate around the goal
                # instead of committing to forward path tracking.
                goal_yaw = math.atan2(float(world_xy[1] - robot_xy[1]), float(world_xy[0] - robot_xy[0]))
            else:
                goal_yaw = float(heading)
            qx, qy, qz, qw = self._quaternion_from_yaw(goal_yaw)
            goal.pose.pose.orientation.x = qx
            goal.pose.pose.orientation.y = qy
            goal.pose.pose.orientation.z = qz
            goal.pose.pose.orientation.w = qw
            self._last_nav2_status_name = "NAVIGATE_TO_POSE_SENT"
            if not bool(getattr(self, "_nav2_navigate_to_pose_logged", False)):
                self._nav2_navigate_to_pose_logged = True
                self.ros.get_logger().info(
                    "NAV2_NAVIGATE_TO_POSE_ACTIVE | bt_navigator owns motion; "
                    f"goal_frame={goal.pose.header.frame_id} | RL map layers use the unified pose frame"
                )
            send_future = self.nav2_client.send_goal_async(goal)

        if not self._wait_future_done(send_future, timeout_sec=self.nav2_send_goal_wait_sec):
            self.ros.get_logger().warn("Nav2 goal send timed out")
            zero = np.zeros(2, dtype=np.float32)
            _, _, front = self._compute_lidar_action_obstacle_risk(zero)
            self._last_waypoint_timed_out = True
            return zero, 999.0, 0.0, float(front)

        goal_handle = send_future.result()
        self._nav2_goal_handle = goal_handle
        if goal_handle is None or not goal_handle.accepted:
            self._last_nav2_goal_accepted = False
            self._last_nav2_status = int(GoalStatus.STATUS_UNKNOWN) if GoalStatus is not None else -1
            self._last_nav2_status_name = "REJECTED_NAV2_ONLY"
            self._clear_nav2_costmaps(wait_timeout_sec=0.35)
            zero = np.zeros(2, dtype=np.float32)
            _, _, front = self._compute_lidar_action_obstacle_risk(zero)
            self._last_waypoint_timed_out = True
            return zero, 999.0, 0.0, float(front)

        self._last_nav2_goal_accepted = True
        self._reset_nav2_stationary_window()
        if self._nav2_use_follow_path_controller and not bool(getattr(self, "_nav2_follow_path_logged", False)):
            self._nav2_follow_path_logged = True
            self.ros.get_logger().info(
                f"NAV2_FOLLOW_PATH_ACTIVE | controller_server /follow_path owns motion; "
                f"path_frame={getattr(self, '_last_nav2_motion_frame', self.pose_frame)} | "
                "NavigateToPose BT bypassed | cmd_vel_type=TwistStamped | preempt_streaming"
            )
        result_future = goal_handle.get_result_async()

        min_action_obstacle_distance = 999.0
        max_action_obstacle_score = 0.0
        last_front_obstacle_distance = 999.0
        moved_since_goal = 0.0
        # Streaming mode should return quickly so the next SAC action can refresh
        # the local path.  The previous threshold was conservative and produced
        # visible stop-and-go.
        partial_move_threshold = max(0.045, 0.35 * self.nav2_replan_distance_m)
        # If Nav2 accepts a goal but does not produce translation, do not wait for
        # the full goal timeout.  Trigger the Nav2 BackUp behavior and return to
        # SAC; otherwise training stalls at ~0.1 fps while the robot only jitters.
        # Do not even consider BackUp until the robot has had enough time to
        # prove it is stationary.  The actual stationary window is checked again
        # inside _nav2_backup_allowed_now().
        spawn_stuck_fallback_sec = max(
            float(getattr(self, "nav2_stuck_backup_stationary_sec", 1.5)),
            min(float(getattr(self, "nav2_stuck_backup_sec", 2.2)), 0.60 * self.nav2_goal_timeout_sec),
        )
        spawn_stuck_move_epsilon = max(0.010, float(getattr(self, "nav2_stuck_backup_min_movement_m", 0.02)))
        nav2_goal_backup_attempted = False

        # Nav2가 실제 /cmd_vel을 내므로 env는 wall-clock을 기준으로 action result를 기다린다.
        # 종료 조건은 명시적으로 3개다:
        #   1) local goal tolerance 이내 도착
        #   2) nav2_goal_timeout_sec 경과
        #   3) collision/fallen/drop terminal
        # 이렇게 해야 goal 근처에서 orientation 정렬 때문에 뱅글뱅글 도는 구간을 시간으로 끊을 수 있다.
        while True:
            self.ros.spin_steps(num_spins=1, timeout_sec=0.02)
            # Guarantee live map refresh even if the ROS timer is delayed by a long
            # Nav2 wait loop. This keeps /rl_priority_map and /rl_confidence_map
            # close to 10 Hz while a waypoint is being followed.
            self._live_map_update_timer_callback()
            self._sample_nav2_stationary_window()
            self._last_controller_steps += 1
            self._last_waypoint_final_error = float(self._waypoint_distance_to_target(world_xy))
            moved_since_goal = 0.0
            yaw_delta_since_goal = 0.0
            if start_pose is not None:
                cur_pose_for_move = self._get_robot_pose2d(frame_id=motion_monitor_frame)
                if cur_pose_for_move is not None:
                    moved_since_goal = float(
                        np.linalg.norm(
                            np.asarray(cur_pose_for_move[0], dtype=np.float32)
                            - np.asarray(start_pose[0], dtype=np.float32)
                        )
                    )
                    try:
                        yaw_delta_since_goal = abs(
                            self._normalize_angle(float(cur_pose_for_move[1]) - float(start_pose[1]))
                        )
                    except Exception:
                        yaw_delta_since_goal = 0.0
            self._last_nav2_moved_distance = float(moved_since_goal)
            self._last_nav2_yaw_delta_since_goal = float(yaw_delta_since_goal)

            # 디버그 시 현재 goal marker가 로봇 위치와 함께 갱신되도록 주기적으로 republish.
            if self._last_controller_steps % self.waypoint_visual_publish_every_n == 0:
                self._publish_waypoint_visualization(
                    waypoint_world_xy=world_xy,
                    waypoint_local_xy=local_xy,
                    distance=self._last_waypoint_final_error,
                    heading=heading,
                    append_history=False,
                )

            approx_to_goal = np.array([
                min(self.max_linear_speed, max(0.0, self._last_waypoint_distance / max(self.nav2_goal_timeout_sec, 1e-6))),
                float(np.clip(heading / max(self.control_dt, 1e-6), -self.max_angular_speed, self.max_angular_speed)),
            ], dtype=np.float32)
            d_obs, s_obs, f_obs = self._compute_lidar_action_obstacle_risk(approx_to_goal)
            min_action_obstacle_distance = min(min_action_obstacle_distance, float(d_obs))
            max_action_obstacle_score = max(max_action_obstacle_score, float(s_obs))
            last_front_obstacle_distance = float(f_obs)

            if self._last_waypoint_final_error <= self.nav2_goal_reached_tolerance:
                self._last_waypoint_reached = True
                self._last_nav2_status = int(GoalStatus.STATUS_SUCCEEDED) if GoalStatus is not None else 4
                self._last_nav2_status_name = "LOCAL_TOL_REACHED"
                if self.nav2_cancel_on_reached and not (
                    self.nav2_continuous_goal_update and self.nav2_preempt_without_cancel
                ):
                    self._cancel_nav2_goal_sync(reason="nav2_goal_reached")
                break

            if result_future.done():
                try:
                    wrapped = result_future.result()
                    status = int(getattr(wrapped, "status", -1))
                except Exception:
                    status = -1
                self._last_nav2_status = status
                self._last_nav2_status_name = self._nav2_status_name(status)
                self._last_waypoint_reached = (
                    GoalStatus is not None and status == GoalStatus.STATUS_SUCCEEDED
                )

                # Strict Nav2 mode: if Nav2 aborts, do not switch to the internal
                # waypoint controller. Clear costmaps and let the next SAC step send
                # a new NavigateToPose goal.
                if GoalStatus is not None and status == GoalStatus.STATUS_ABORTED:
                    self._last_nav2_status_name = "ABORTED_NAV2_ONLY"
                    self._clear_nav2_costmaps(wait_timeout_sec=0.35)
                break

            elapsed_wall = time.time() - start_time

            # Strict Nav2 mode: no internal /cmd_vel rescue.  If the controller
            # accepts a goal but remains motionless, clear costmaps and replan on
            # the next SAC step through NavigateToPose only.
            if (
                elapsed_wall >= spawn_stuck_fallback_sec
                and moved_since_goal < spawn_stuck_move_epsilon
                and self._last_waypoint_final_error > self.nav2_goal_reached_tolerance
            ):
                backup_allowed, backup_gate_reason = self._nav2_backup_allowed_now(
                    world_xy, moved_since_goal=moved_since_goal, elapsed_wall=elapsed_wall
                )
                self._last_nav2_backup_gate_reason = backup_gate_reason
                if (
                    backup_allowed
                    and not nav2_goal_backup_attempted
                    and bool(getattr(self, "nav2_stuck_backup", True))
                ):
                    nav2_goal_backup_attempted = True
                    self._last_nav2_status = -4
                    self._last_nav2_status_name = "STUCK_BACKUP"
                    self._last_nav2_stuck_reason = (
                        f"goal_stuck:{elapsed_wall:.2f}s,moved={moved_since_goal:.3f},"
                        f"err={self._last_waypoint_final_error:.3f},{backup_gate_reason}"
                    )
                    self._cancel_nav2_goal_sync(reason="nav2_goal_stuck_backup")
                    self.ros.get_logger().warn(
                        "NAV2_GOAL_STUCK_BACKUP | "
                        f"elapsed={elapsed_wall:.2f}s | moved={moved_since_goal:.3f}m | "
                        f"err={self._last_waypoint_final_error:.3f}m | gate={backup_gate_reason}"
                    )
                    self._execute_nav2_backup_behavior(reason=self._last_nav2_stuck_reason)
                    break
                if getattr(self, "_last_nav2_stuck_log_goal_time", None) != start_time:
                    self._last_nav2_stuck_log_goal_time = start_time
                    self.ros.get_logger().warn(
                        "NAV2_GOAL_STUCK_WAIT | backup gated or already attempted | "
                        f"gate={backup_gate_reason}"
                    )

                # Legacy cancel-sync restore: do not cancel during initial rotation/no-progress.
                # The older moving behavior held the same short FollowPath goal until
                # local tolerance or timeout.  This can look stop-and-go, but it avoids
                # repeatedly killing a valid controller goal before it can translate.
                if getattr(self, "_last_nav2_initial_wait_log_goal_time", None) != start_time:
                    self._last_nav2_initial_wait_log_goal_time = start_time
                    self.ros.get_logger().warn(
                        "NAV2_INITIAL_ROTATION_WAIT | holding same FollowPath goal until reached/timeout | "
                        f"elapsed={elapsed_wall:.2f}s | moved={moved_since_goal:.3f}m | "
                        f"err={self._last_waypoint_final_error:.3f}m | gate={backup_gate_reason} | cancel_sync_restore"
                    )

            # Continuous goal update mode: do not hold env.step() until Nav2
            # reaches/aborts/times out. Return after a small control window so SAC
            # can emit the next nearby goal before the robot stops at the current one.
            # We intentionally keep the active Nav2 goal alive here; the next
            # NavigateToPose goal will preempt/update it. This is the key for
            # visually continuous motion.
            if self.nav2_continuous_goal_update:
                near_replan = (
                    self._last_waypoint_final_error <= self.nav2_early_replan_remaining_m
                    and moved_since_goal >= 0.03
                )
                if near_replan:
                    self._last_nav2_status = int(GoalStatus.STATUS_EXECUTING) if GoalStatus is not None else 2
                    self._last_nav2_status_name = "STREAMING_NEAR_GOAL_REPLAN"
                    break

                if self.nav2_replan_on_movement and moved_since_goal >= self.nav2_replan_distance_m:
                    self._last_nav2_status = int(GoalStatus.STATUS_EXECUTING) if GoalStatus is not None else 2
                    self._last_nav2_status_name = "STREAMING_MOVED_REPLAN"
                    break

                # Critical speed path: do not wait for result, full timeout, or a
                # cancel acknowledgement.  Even if the robot is still rotating or
                # has not translated yet, return after a small wall-clock window so
                # SAC can immediately refresh the local target.  The next action goal
                # preempts the old one inside Nav2.
                if elapsed_wall >= self.nav2_control_window_sec:
                    self._last_nav2_status = int(GoalStatus.STATUS_EXECUTING) if GoalStatus is not None else 2
                    if moved_since_goal >= partial_move_threshold:
                        self._last_nav2_status_name = "STREAMING_WINDOW_MOVED_REPLAN"
                    else:
                        self._last_nav2_status_name = "STREAMING_WINDOW_REPLAN"
                    break

            if elapsed_wall >= self.nav2_goal_timeout_sec:
                self._last_waypoint_timed_out = True
                self._last_nav2_status = -2
                self._last_nav2_status_name = "TIMEOUT"
                if self.nav2_cancel_on_timeout and not (
                    self.nav2_continuous_goal_update and self.nav2_preempt_without_cancel
                ):
                    self._cancel_nav2_goal_sync(reason="nav2_timeout")
                if moved_since_goal < 0.05 and self._last_waypoint_final_error > self.nav2_goal_reached_tolerance:
                    backup_allowed, backup_gate_reason = self._nav2_backup_allowed_now(
                        world_xy, moved_since_goal=moved_since_goal, elapsed_wall=elapsed_wall
                    )
                    self._last_nav2_backup_gate_reason = backup_gate_reason
                    if backup_allowed and bool(getattr(self, "nav2_stuck_backup", True)):
                        self._last_nav2_status_name = "TIMEOUT_BACKUP"
                        self._execute_nav2_backup_behavior(
                            reason=(
                                f"timeout,moved={moved_since_goal:.3f},"
                                f"err={self._last_waypoint_final_error:.3f},{backup_gate_reason}"
                            )
                        )
                    else:
                        self._last_nav2_status_name = "TIMEOUT_REPLAN_NO_BACKUP"
                        self.ros.get_logger().warn(
                            "NAV2_TIMEOUT_NO_BACKUP | "
                            f"moved={moved_since_goal:.3f}m | err={self._last_waypoint_final_error:.3f}m | "
                            f"gate={backup_gate_reason}"
                        )
                        self._clear_nav2_costmaps(wait_timeout_sec=0.05)
                break

            if self._check_collision() or self._check_fallen():
                self._last_nav2_status = -3
                self._last_nav2_status_name = "ENV_TERMINAL"
                self._cancel_nav2_goal_sync(reason="nav2_goal_reached_cancel_sync_restore")
                break

        end_pose = self._get_robot_pose2d()
        elapsed = max(time.time() - start_time, 1e-3)
        if start_pose is not None and end_pose is not None:
            start_xy, start_yaw = start_pose
            end_xy, end_yaw = end_pose
            dist_moved = float(np.linalg.norm(np.asarray(end_xy) - np.asarray(start_xy)))
            yaw_delta = self._normalize_angle(float(end_yaw) - float(start_yaw))
            linear_x = float(np.clip(dist_moved / elapsed, 0.0, self.max_linear_speed))
            angular_z = float(np.clip(yaw_delta / elapsed, -self.max_angular_speed, self.max_angular_speed))
            executed_action = np.array([linear_x, angular_z], dtype=np.float32)
        else:
            executed_action = np.zeros(2, dtype=np.float32)

        if min_action_obstacle_distance >= 999.0:
            d_obs, s_obs, f_obs = self._compute_lidar_action_obstacle_risk(executed_action)
            min_action_obstacle_distance = float(d_obs)
            max_action_obstacle_score = float(s_obs)
            last_front_obstacle_distance = float(f_obs)

        return (
            executed_action.astype(np.float32),
            float(min_action_obstacle_distance),
            float(max_action_obstacle_score),
            float(last_front_obstacle_distance),
        )

    def _execute_waypoint_action(self, policy_action: np.ndarray) -> tuple[np.ndarray, float, float, float]:
        """
        SAC가 선택한 local waypoint를 짧은 시간만 추종한 뒤 즉시 다음 waypoint를 받는다.

        핵심:
          - policy는 한 번에 하나의 waypoint 좌표를 고른다.
          - 같은 env.step() 안에서는 새 waypoint를 받지 않는다.
          - 기본값은 receding-horizon 방식이다. waypoint에 정확히 도달할 때까지 붙잡지 않는다.
          - waypoint_timeout_sec > 0이면 execute_until_reached를 켰을 때도 짧은 시간 제한으로 끊는다.
          - timeout/step budget이 끝나면 같은 waypoint를 버리고 다음 SAC action으로 넘어간다.
          - reward에는 실제 수행된 cmd_vel 평균을 넘긴다.
        """
        local_xy, world_xy, distance, heading, goal_source, goal_valid, validation_reason = (
            self._decode_nav2_goal_action(policy_action)
        )

        self._last_nav2_goal_source = str(goal_source)
        self._last_nav2_goal_valid = bool(goal_valid)
        self._last_nav2_goal_validation = str(validation_reason)

        self._last_waypoint_local = local_xy.astype(np.float32).copy()
        self._last_waypoint_world = world_xy.astype(np.float32).copy()
        self._last_waypoint_distance = float(distance)
        self._last_waypoint_angle = float(heading)
        self._last_waypoint_reached = False
        self._last_waypoint_timed_out = False
        self._last_waypoint_final_error = float(self._waypoint_distance_to_target(world_xy))
        self._last_controller_steps = 0
        self._publish_waypoint_visualization(
            waypoint_world_xy=world_xy,
            waypoint_local_xy=local_xy,
            distance=distance,
            heading=heading,
        )

        executed: list[np.ndarray] = []
        min_action_obstacle_distance = 999.0
        max_action_obstacle_score = 0.0
        last_front_obstacle_distance = 999.0

        start_pose = self._get_robot_pose2d()
        if start_pose is not None:
            start_xy, start_yaw = start_pose
            start_xy = np.asarray(start_xy, dtype=np.float32).copy()
            start_yaw = float(start_yaw)
        else:
            start_xy = None
            start_yaw = 0.0

        # Local receding-horizon policy.  The important anti-spin rule is:
        # do not treat a waypoint as a hard sub-goal that must be reached.  In
        # normal mode, execute only waypoint_control_steps tick(s), then return
        # to SAC so the next waypoint can be sampled immediately.  If the user
        # explicitly enables execute_until_reached, waypoint_timeout_sec still
        # caps the hold time through waypoint_max_control_steps.
        max_steps = self.waypoint_max_control_steps if self.waypoint_execute_until_reached else self.waypoint_control_steps

        for low_level_i in range(max_steps):
            if self._check_collision() or self._check_fallen():
                self.ros.stop_robot()
                break

            if start_xy is not None:
                curr_pose = self._get_robot_pose2d()
                if curr_pose is not None:
                    curr_xy, curr_yaw = curr_pose
                    moved = float(np.linalg.norm(np.asarray(curr_xy, dtype=np.float32) - start_xy))
                    heading_delta = abs(self._normalize_angle(float(curr_yaw) - start_yaw))
                    if moved >= self.waypoint_replan_distance_m:
                        self._last_waypoint_timed_out = False
                        break
                    if heading_delta >= self.waypoint_replan_heading_rad:
                        self._last_waypoint_timed_out = False
                        break

            cmd, reached = self._waypoint_controller_command(world_xy)
            self._last_waypoint_final_error = float(self._waypoint_distance_to_target(world_xy))
            self._last_waypoint_reached = bool(reached)

            if reached:
                # 도달 시 절대 stop_robot()을 호출하지 않는다.
                # waypoint는 정차 지점이 아니라 경유점이다. 이 env.step()은 여기서 끝나고,
                # 다음 SAC action이 즉시 다음 waypoint를 생성한다.
                break

            lidar_action_obstacle_distance, lidar_action_obstacle_score, lidar_front_obstacle_distance = (
                self._compute_lidar_action_obstacle_risk(cmd)
            )
            min_action_obstacle_distance = min(
                float(min_action_obstacle_distance),
                float(lidar_action_obstacle_distance),
            )
            max_action_obstacle_score = max(
                float(max_action_obstacle_score),
                float(lidar_action_obstacle_score),
            )
            last_front_obstacle_distance = float(lidar_front_obstacle_distance)

            self.ros.publish_cmd_vel(float(cmd[0]), float(cmd[1]))
            executed.append(cmd.astype(np.float32).copy())
            self._last_controller_steps += 1

            self.ros.spin_steps(num_spins=5, timeout_sec=0.001)
            self._advance_world_after_command(target_delta_sec=self.control_dt)

            # waypoint가 고정된 상태에서도 marker를 계속 갱신해야 RViz에서 목표점과 로봇 사이 선이 따라 움직인다.
            if (low_level_i + 1) % self.waypoint_visual_publish_every_n == 0:
                self._publish_waypoint_visualization(
                    waypoint_world_xy=world_xy,
                    waypoint_local_xy=local_xy,
                    distance=self._last_waypoint_final_error,
                    heading=heading,
                    append_history=False,
                )

            if self._check_collision() or self._check_fallen():
                self.ros.stop_robot()
                break

        if not self._last_waypoint_reached and self._last_controller_steps >= max_steps:
            self._last_waypoint_timed_out = bool(self.waypoint_execute_until_reached)
            # 기본값은 timeout에서도 정지하지 않는다. 계속 움직이게 두면 다음 waypoint가
            # 바로 이어져 stop-and-go가 줄어든다. 장애물 앞에서 반드시 멈추고 싶을 때만
            # --waypoint-timeout-stop을 켠다.
            if self.waypoint_timeout_stop:
                self.ros.stop_robot()

        self._last_waypoint_final_error = float(self._waypoint_distance_to_target(world_xy))

        if executed:
            executed_action = np.mean(np.stack(executed, axis=0), axis=0).astype(np.float32)
        else:
            executed_action = np.zeros(2, dtype=np.float32)
            # 정지 상태에서도 front distance는 debug/reward에 넣는다.
            _, _, last_front_obstacle_distance = self._compute_lidar_action_obstacle_risk(executed_action)

        return (
            executed_action,
            float(min_action_obstacle_distance),
            float(max_action_obstacle_score),
            float(last_front_obstacle_distance),
        )


    @staticmethod
    def _latest_tf_stamp() -> RosTime:
        """Return stamp=0 so RViz/tf2 resolves visualization transforms at latest time.

        These waypoint markers are purely debug visualization.  Using wall-time
        or stale sim-time stamps is what triggers RViz MarkerArray "No transform
        to fixed frame" errors when Gazebo publishes /clock.
        """
        return RosTime(sec=0, nanosec=0)

    @staticmethod
    def _zero_stamp() -> RosTime:
        """Return a ROS zero timestamp for robot-frame debug markers.

        v14 switched scan/debug markers to robot frames, but some call sites
        referenced _zero_stamp() while only _latest_tf_stamp() existed.  Keep
        both names because zero stamp is intentional for RViz latest-TF marker
        visualization in sim time and real time.
        """
        return RosTime(sec=0, nanosec=0)

    @staticmethod
    def _point_xyz(x: float, y: float, z: float = 0.0) -> Point:
        p = Point()
        p.x = float(x)
        p.y = float(y)
        p.z = float(z)
        return p

    @staticmethod
    def _set_marker_color(marker: Marker, r: float, g: float, b: float, a: float = 1.0) -> None:
        marker.color.r = float(r)
        marker.color.g = float(g)
        marker.color.b = float(b)
        marker.color.a = float(a)

    def _make_waypoint_marker(
        self,
        marker_id: int,
        marker_type: int,
        frame_id: str,
        stamp,
    ) -> Marker:
        marker = Marker()
        marker.header.frame_id = frame_id
        marker.header.stamp = stamp
        marker.ns = "rl_waypoint"
        marker.id = int(marker_id)
        marker.type = marker_type
        marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0
        return marker

    def _waypoint_visualization_frame(self) -> str:
        """Single frame for waypoint/velocity MarkerArray debugging.

        Normal training/eval uses map frame so RL map layers and debug overlay are
        aligned.  During quick Gazebo/real-robot dry runs, however, SLAM /map can
        be temporarily unavailable after a reset.  If we keep publishing markers
        in map while map->odom is absent, RViz cannot render the arrows/text even
        though /cmd_vel is being published.  In that specific case, fall back to
        pose_frame/odom for markers only.  This does not disable SLAM reset or
        change the policy/control path; it only keeps the debug overlay visible.
        """
        map_frame = str(getattr(self, "map_frame", "map") or "map").strip().lstrip("/") or "map"
        pose_frame = str(getattr(self, "pose_frame", map_frame) or map_frame).strip().lstrip("/") or map_frame
        frame = map_frame
        if pose_frame != frame:
            frame = pose_frame

        # Marker-only fallback: when SLAM has not produced a usable /map yet,
        # prefer odom/pose frame so RViz can still show the velocity arrows.
        try:
            slam_map = getattr(getattr(self, "ros", None), "slam_map", None)
            slam_gate = str(getattr(self, "_last_slam_gate_reason", ""))
            if frame == map_frame and map_frame == "map" and (slam_map is None or slam_gate in {"pre_reset", "not_initialized"}):
                fallback = pose_frame if pose_frame and pose_frame != "map" else str(getattr(self, "safety_boundary_frame", "odom") or "odom").strip().lstrip("/")
                return fallback or "odom"
        except Exception:
            pass
        return frame

    def _transform_xy_between_frames(
        self,
        xy: np.ndarray,
        source_frame: str,
        target_frame: str,
    ) -> Optional[np.ndarray]:
        """Transform an XY point between frames; return None rather than guessing."""
        src = str(source_frame or "").strip().lstrip("/")
        dst = str(target_frame or "").strip().lstrip("/")
        arr = np.asarray(xy, dtype=np.float32)
        if arr.size < 2 or not np.all(np.isfinite(arr[:2])):
            return None
        if not src or not dst or src == dst:
            return arr[:2].copy()
        try:
            transformed = self._transform_pose_xy_yaw_between_frames(
                xy=arr[:2],
                yaw=0.0,
                source_frame=src,
                target_frame=dst,
            )
            if transformed is None:
                return None
            out_xy, _ = transformed
            return np.asarray(out_xy, dtype=np.float32)[:2].copy()
        except Exception:
            return None

    def _clear_waypoint_visualization(self) -> None:
        """RViz에 남아 있는 이전 episode waypoint marker/path를 지운다."""
        stamp = self._latest_tf_stamp()

        if self.waypoint_marker_pub is not None:
            clear = Marker()
            clear.header.frame_id = self._waypoint_visualization_frame()
            clear.header.stamp = stamp
            clear.ns = "rl_waypoint"
            clear.action = Marker.DELETEALL
            arr = MarkerArray()
            arr.markers.append(clear)
            self.waypoint_marker_pub.publish(arr)

        if self.waypoint_path_pub is not None:
            msg = NavPath()
            msg.header.frame_id = self._waypoint_visualization_frame()
            msg.header.stamp = stamp
            self.waypoint_path_pub.publish(msg)

    def _publish_waypoint_visualization(
        self,
        waypoint_world_xy: np.ndarray,
        waypoint_local_xy: np.ndarray,
        distance: float,
        heading: float,
        append_history: bool = True,
        reward_value: Optional[float] = None,
        episode_reward: Optional[float] = None,
    ) -> None:
        """
        Publish the current SAC/Nav2 waypoint in a fixed global frame.

        Invariant:
          - MarkerArray frame is self.pose_frame, normally "odom" for the user's
            current command.
          - The waypoint sphere is the actual Nav2 goal point and is therefore
            fixed in odom/map after it is selected.
          - The line/arrow start follows the current robot pose only because it is
            a live debug vector from robot -> fixed goal.
          - No extra robot marker is published; RViz RobotModel already shows the
            robot.  This avoids the duplicated red-dot failure mode.
          - Robot-local coordinates are shown in the text for debugging.
        """
        if self.waypoint_marker_pub is None and self.waypoint_path_pub is None:
            return

        if self.step_count % self.waypoint_visual_publish_every_n != 0:
            return

        if reward_value is None and getattr(self, "_last_reward_text_valid", False):
            reward_value = float(getattr(self, "_last_step_reward", 0.0))
            if episode_reward is None:
                episode_reward = float(getattr(self, "_episode_discounted_return", 0.0))

        target_xy = np.asarray(waypoint_world_xy, dtype=np.float32)
        local_xy = np.asarray(waypoint_local_xy, dtype=np.float32)
        if local_xy.size < 2 or not np.all(np.isfinite(local_xy[:2])):
            return
        if target_xy.size < 2 or not np.all(np.isfinite(target_xy[:2])):
            return

        stamp = self._latest_tf_stamp()
        marker_frame_id = self._waypoint_visualization_frame()
        path_frame_id = marker_frame_id

        # waypoint_world_xy is produced in self.pose_frame.  Transform only if the
        # visualization frame differs.  Do not fall back to base_footprint: a goal
        # marker in a robot frame moves with the robot and is useless for Nav2
        # debugging.
        target_marker_xy = self._transform_xy_between_frames(
            target_xy,
            source_frame=self.pose_frame,
            target_frame=marker_frame_id,
        )
        if target_marker_xy is None or not np.all(np.isfinite(target_marker_xy[:2])):
            return

        if append_history and self.waypoint_show_history:
            self._waypoint_history.append((float(target_marker_xy[0]), float(target_marker_xy[1])))
        elif not self.waypoint_show_history:
            self._waypoint_history.clear()
            self._waypoint_history.append((float(target_marker_xy[0]), float(target_marker_xy[1])))

        robot_pose = self._get_robot_pose2d(frame_id=marker_frame_id)
        robot_xy = None
        if robot_pose is not None:
            rxy = np.asarray(robot_pose[0], dtype=np.float32)
            if rxy.size >= 2 and np.all(np.isfinite(rxy[:2])):
                robot_xy = rxy[:2].copy()

        wx = float(target_marker_xy[0])
        wy = float(target_marker_xy[1])
        lx = float(local_xy[0])
        ly = float(local_xy[1])
        rx = float(robot_xy[0]) if robot_xy is not None else float("nan")
        ry = float(robot_xy[1]) if robot_xy is not None else float("nan")

        if self.waypoint_marker_pub is not None:
            markers = MarkerArray()

            # Clear stale markers from older debug builds: history line, robot dot,
            # and robot heading arrow.  Only the target/line/arrow/text remain.
            for stale_id, stale_type in ((4, Marker.LINE_STRIP), (5, Marker.CYLINDER), (6, Marker.ARROW)):
                stale = self._make_waypoint_marker(stale_id, stale_type, marker_frame_id, stamp)
                stale.action = Marker.DELETE
                markers.markers.append(stale)

            target_marker = self._make_waypoint_marker(0, Marker.SPHERE, marker_frame_id, stamp)
            target_marker.frame_locked = False
            target_marker.pose.position.x = wx
            target_marker.pose.position.y = wy
            target_marker.pose.position.z = 0.10
            target_marker.scale.x = 0.18
            target_marker.scale.y = 0.18
            target_marker.scale.z = 0.18
            self._set_marker_color(target_marker, 0.05, 0.85, 1.00, 0.95)
            markers.markers.append(target_marker)

            if robot_xy is not None:
                line_marker = self._make_waypoint_marker(1, Marker.LINE_STRIP, marker_frame_id, stamp)
                line_marker.frame_locked = False
                line_marker.points = [
                    self._point_xyz(rx, ry, 0.06),
                    self._point_xyz(wx, wy, 0.06),
                ]
                line_marker.scale.x = 0.045
                self._set_marker_color(line_marker, 1.00, 0.82, 0.05, 0.95)
                markers.markers.append(line_marker)

                arrow_marker = self._make_waypoint_marker(2, Marker.ARROW, marker_frame_id, stamp)
                arrow_marker.frame_locked = False
                arrow_marker.points = [
                    self._point_xyz(rx, ry, 0.12),
                    self._point_xyz(wx, wy, 0.12),
                ]
                arrow_marker.scale.x = 0.055
                arrow_marker.scale.y = 0.14
                arrow_marker.scale.z = 0.18
                self._set_marker_color(arrow_marker, 0.10, 1.00, 0.25, 0.95)
                markers.markers.append(arrow_marker)
            else:
                # Clear vector markers if robot pose is temporarily unavailable.
                for stale_id, stale_type in ((1, Marker.LINE_STRIP), (2, Marker.ARROW)):
                    stale = self._make_waypoint_marker(stale_id, stale_type, marker_frame_id, stamp)
                    stale.action = Marker.DELETE
                    markers.markers.append(stale)

            text_marker = self._make_waypoint_marker(3, Marker.TEXT_VIEW_FACING, marker_frame_id, stamp)
            text_marker.frame_locked = False
            text_marker.pose.position.x = wx
            text_marker.pose.position.y = wy
            text_marker.pose.position.z = 0.35
            text_marker.scale.z = 0.18
            reward_line = f"marker={marker_frame_id} pose={self.pose_frame} nav2=NavigateToPose"
            if reward_value is not None:
                try:
                    rv = float(reward_value)
                except (TypeError, ValueError):
                    rv = float("nan")
                if math.isfinite(rv):
                    reward_line = f"r={rv:+.3f} " + reward_line
                    if episode_reward is not None:
                        try:
                            er = float(episode_reward)
                        except (TypeError, ValueError):
                            er = float("nan")
                        if math.isfinite(er):
                            reward_line = f"r={rv:+.3f} Glive={er:+.2f} γ={self.reward_gamma:.2f} marker={marker_frame_id} pose={self.pose_frame} nav2=NavigateToPose"

            step_pclear = float(getattr(self, "_last_step_priority_clear_gain", 0.0)) + float(getattr(self, "_last_step_priority_rechecked_gain", 0.0))
            ep_pclear = float(getattr(self, "_episode_priority_clear_gain", 0.0)) + float(getattr(self, "_episode_priority_rechecked_gain", 0.0))
            live_pclear = float(getattr(self, "_last_pending_priority_clear_gain", 0.0)) + float(getattr(self, "_last_pending_priority_rechecked_gain", 0.0))
            if step_pclear > 1e-6 or ep_pclear > 1e-6:
                reward_line += f" Wclr={step_pclear:.2f}/{ep_pclear:.1f}"
                if live_pclear > 1e-6:
                    reward_line += f" live={live_pclear:.2f}"
            if bool(getattr(self, "_last_priority_stuck_active", False)):
                reward_line += (
                    f" Pstuck={int(getattr(self, 'priority_stuck_steps', 0))}/"
                    f"{int(getattr(self, 'priority_stuck_restart_steps', 0))}"
                )
            if bool(getattr(self, "_last_nav2_stuck_active", False)):
                reward_line += (
                    f" Bstuck={int(getattr(self, 'nav2_stuck_steps', 0))}/"
                    f"{int(getattr(self, 'nav2_stuck_backup_steps', 0))}"
                )
            if bool(getattr(self, "_last_nav2_stuck_backup_triggered", False)):
                reward_line += f" Bup={str(getattr(self, '_last_nav2_backup_status', 'none'))}"
            if bool(getattr(self, "_last_lidar_empty_active", False)):
                reward_line += (
                    f" Lempty={int(getattr(self, 'lidar_empty_steps', 0))}/"
                    f"{int(getattr(self, 'lidar_empty_timeout_steps', 0))}"
                    f" beams={int(getattr(self, '_last_lidar_valid_beams', 0))}"
                )

            text_marker.text = (
                f"NEXT WP fixed goal\n"
                f"local base_footprint=({lx:+.2f},{ly:+.2f}) "
                f"d={float(distance):.2f}m a={math.degrees(float(heading)):+.0f}deg\n"
                f"goal {marker_frame_id}=({wx:+.2f},{wy:+.2f}) "
                f"robot {marker_frame_id}=({rx:+.2f},{ry:+.2f}) {reward_line}"
            )
            self._set_marker_color(text_marker, 1.00, 1.00, 1.00, 0.95)
            markers.markers.append(text_marker)

            self.waypoint_marker_pub.publish(markers)

        if self.waypoint_path_pub is not None:
            path_msg = NavPath()
            path_msg.header.frame_id = path_frame_id
            path_msg.header.stamp = stamp

            points = list(self._waypoint_history) if self.waypoint_show_history else [(float(wx), float(wy))]
            for x, y in points:
                pose = PoseStamped()
                pose.header.frame_id = path_frame_id
                pose.header.stamp = stamp
                pose.pose.position.x = float(x)
                pose.pose.position.y = float(y)
                pose.pose.position.z = 0.04
                pose.pose.orientation.w = 1.0
                path_msg.poses.append(pose)

            self.waypoint_path_pub.publish(path_msg)

    def _publish_velocity_debug_overlay(
        self,
        reward_value: float,
        episode_reward: float,
        map_stats: MapUpdateStats,
        raw_action: np.ndarray,
        executed_action: np.ndarray,
        lidar_action_obstacle_distance: float,
        lidar_action_obstacle_score: float,
        lidar_front_obstacle_distance: float,
    ) -> None:
        """Publish a robot-attached RViz debug overlay for pure velocity SAC.

        This is intentionally separate from waypoint visualization.  In velocity
        mode there is no fixed global waypoint, so the useful debug object is a
        large TEXT_VIEW_FACING marker above the robot plus a small local command
        arrow in base_footprint.  The marker moves with the robot by design.
        """
        if self.waypoint_marker_pub is None:
            return
        # Velocity debug overlay publish rate.  In velocity mode there is no
        # waypoint, so reuse a dedicated env knob (default: every step) instead of
        # the waypoint_visual_publish_every_n gate which defaults to ~never.
        try:
            overlay_every_n = int(os.environ.get("TB3_RL_DEBUG_OVERLAY_EVERY_N", "1"))
        except Exception:
            overlay_every_n = 1
        overlay_every_n = max(int(overlay_every_n), 1)
        if (int(self.step_count) % overlay_every_n) != 0:
            return

        # v14 default: velocity debug overlay is robot-attached.  The old map-frame
        # overlay could appear at a stale/global location when map->odom changed,
        # which made the user think the MarkerArray was not robot based.
        if self._scan_bool_env("TB3_RL_DEBUG_OVERLAY_IN_BASE_FRAME", True):
            stamp = self._zero_stamp()
            frame_id = str(os.environ.get("TB3_RL_BASE_FRAME", getattr(self.ros, "base_frame", "base_footprint")) or "base_footprint").strip().lstrip("/") or "base_footprint"
            robot_xy_overlay = np.array([0.0, 0.0], dtype=np.float32)
            robot_yaw_overlay = 0.0
        else:
            stamp = self._latest_tf_stamp()
            frame_id = self._waypoint_visualization_frame()
            robot_pose_for_overlay = self._get_robot_pose2d(frame_id=frame_id)
            if robot_pose_for_overlay is None:
                frame_id = "base_footprint"
                robot_xy_overlay = np.array([0.0, 0.0], dtype=np.float32)
                robot_yaw_overlay = 0.0
            else:
                robot_xy_overlay = np.asarray(robot_pose_for_overlay[0], dtype=np.float32)[:2]
                robot_yaw_overlay = float(robot_pose_for_overlay[1])
        raw = np.asarray(raw_action, dtype=np.float32)
        exe = np.asarray(executed_action, dtype=np.float32)
        if raw.size < 2:
            raw = np.pad(raw, (0, 2 - raw.size), mode="constant")
        if exe.size < 2:
            exe = np.pad(exe, (0, 2 - exe.size), mode="constant")

        def fnum(value, default=0.0):
            try:
                out = float(value)
                return out if math.isfinite(out) else float(default)
            except Exception:
                return float(default)

        reward = fnum(reward_value)
        g_return = fnum(episode_reward)
        g_sum = fnum(getattr(self, "_episode_reward_sum", 0.0), 0.0)
        g_start = fnum(getattr(self, "_episode_start_discounted_return", 0.0), 0.0)
        g_ema = fnum(getattr(self, "_episode_reward_ema", 0.0), 0.0)
        r_window_n = int(getattr(self, "_reward_window_n", 100))
        r_window = fnum(sum(getattr(self, "_recent_reward_window", [])), 0.0)
        slam_window = int(sum(getattr(self, "_recent_slam_new_window", [])))
        conf_window = int(sum(getattr(self, "_recent_conf_update_window", [])))
        front = fnum(lidar_front_obstacle_distance, 999.0)
        act_obs = fnum(lidar_action_obstacle_distance, 999.0)
        act_score = fnum(lidar_action_obstacle_score, 0.0)
        nearest = fnum(getattr(map_stats, "nearest_obstacle_distance", 999.0), 999.0)
        cov = 100.0 * fnum(getattr(map_stats, "coverage_ratio", 0.0), 0.0)
        dcov = 100.0 * fnum(getattr(map_stats, "coverage_delta", 0.0), 0.0)
        conf = fnum(getattr(map_stats, "mean_confidence", 0.0), 0.0)
        stale_pct = 100.0 * fnum(getattr(map_stats, "stale_ratio", 0.0), 0.0)
        low = 100.0 * fnum(getattr(map_stats, "low_confidence_ratio", 0.0), 0.0)
        conf_updated_cells = int(max(0, fnum(getattr(map_stats, "confidence_updated_cells", 0), 0.0)))
        now_conf_rate = time.time()
        last_conf_rate_t = float(getattr(self, "_last_conf_rate_wall_time", 0.0) or 0.0)
        last_conf_rate_step = int(getattr(self, "_last_conf_rate_step", -1))
        conf_rate_ema = float(getattr(self, "_last_conf_rate_ema", 0.0) or 0.0)
        if int(getattr(self, "step_count", 0)) != last_conf_rate_step:
            dt_conf_rate = max(now_conf_rate - last_conf_rate_t, 1e-3) if last_conf_rate_t > 0.0 else max(float(getattr(self, "control_dt", 0.06)), 1e-3)
            inst_conf_rate = float(conf_updated_cells) / dt_conf_rate
            conf_rate_ema = inst_conf_rate if last_conf_rate_t <= 0.0 else (0.30 * inst_conf_rate + 0.70 * conf_rate_ema)
            self._last_conf_rate_wall_time = now_conf_rate
            self._last_conf_rate_step = int(getattr(self, "step_count", 0))
            self._last_conf_rate_ema = conf_rate_ema
        no_priority_overlay = bool(getattr(self, "disable_priority_map", False)) or (
            str(os.environ.get("TB3_RL_FORCE_NO_PRIORITY", "0")).strip().lower()
            not in {"0", "false", "no", "off", "disable", "disabled"}
        )
        if no_priority_overlay:
            prio_score = prio_gain = prio_clear = prio_recheck = target_prio = 0.0
            target_type = str(getattr(map_stats, "target_type", "none"))
            if target_type == "priority_gap":
                target_type = "none"
        else:
            prio_score = fnum(getattr(map_stats, "priority_score", 0.0), 0.0)
            prio_gain = fnum(getattr(map_stats, "priority_gain", 0.0), 0.0)
            prio_clear = fnum(getattr(map_stats, "priority_clear_gain", 0.0), 0.0)
            prio_recheck = fnum(getattr(map_stats, "priority_rechecked_gain", 0.0), 0.0)
            target_type = str(getattr(map_stats, "target_type", "none"))
            target_prio = fnum(getattr(map_stats, "target_priority", 0.0), 0.0)

        l_empty_steps = int(getattr(self, "lidar_empty_steps", 0))
        l_empty_timeout = int(getattr(self, "lidar_empty_timeout_steps", 0))
        valid_beams = int(getattr(self, "_last_lidar_valid_beams", 0))
        p_steps = 0 if no_priority_overlay else int(getattr(self, "priority_stuck_steps", 0))
        p_limit = 0 if no_priority_overlay else int(getattr(self, "priority_stuck_restart_steps", 0))
        safety_reason = str(getattr(self, "_last_velocity_safety_reason", "none"))
        safety_pen = fnum(getattr(self, "_last_velocity_safety_penalty", 0.0), 0.0)
        safety_cd = int(getattr(self, "velocity_safety_cooldown_steps", 0))
        slowdown = fnum(getattr(self, "_last_velocity_safety_slowdown", 1.0), 1.0)
        term = str(getattr(self, "_last_terminal_reason", "none"))
        shake_steps = int(getattr(self, "shake_steps", 0))
        shake_limit = int(getattr(self, "shake_restart_steps_limit", 0))
        shake_reason = str(getattr(self, "_last_shake_reason", "none"))
        slam_gate = str(getattr(self, "_last_slam_gate_reason", "none"))
        map_age = fnum(getattr(self, "_last_slam_map_age_sec", -1.0), -1.0)
        slam_new_known = int(getattr(map_stats, "slam_update_new_known_cells", 0))
        slam_r = fnum(getattr(self, "_last_slam_map_update_reward", 0.0), 0.0)

        arr = MarkerArray()

        # NOTE: Do NOT mix a DELETEALL marker with ADD markers in the same
        # MarkerArray.  RViz can report "Duplicate Marker in the same namespace"
        # and drop the overlay.  Each ADD marker below uses a fixed (ns,id) so it
        # is overwritten in place every frame; no DELETEALL is needed.

        # Delete old waypoint/goal markers from previous modes on this topic.
        for stale_id, stale_type in ((0, Marker.SPHERE), (1, Marker.LINE_STRIP), (2, Marker.ARROW), (3, Marker.TEXT_VIEW_FACING), (4, Marker.LINE_STRIP), (5, Marker.CYLINDER), (6, Marker.ARROW)):
            stale_marker = Marker()
            stale_marker.header.frame_id = frame_id
            stale_marker.header.stamp = stamp
            stale_marker.ns = "rl_waypoint"
            stale_marker.id = int(stale_id)
            stale_marker.type = stale_type
            stale_marker.action = Marker.DELETE
            arr.markers.append(stale_marker)

        text = Marker()
        text.header.frame_id = frame_id
        text.header.stamp = stamp
        text.ns = "rl_velocity_debug"
        text.id = 0
        text.type = Marker.TEXT_VIEW_FACING
        text.action = Marker.ADD
        text.pose.orientation.w = 1.0
        text.pose.position.x = float(robot_xy_overlay[0])
        text.pose.position.y = float(robot_xy_overlay[1])
        text.pose.position.z = 0.95
        text.scale.z = 0.20
        self._set_marker_color(text, 1.0, 1.0, 1.0, 0.98)
        text.text = (
            f"PURE VELOCITY SAC  step={int(self.step_count)}\n"
            f"r={reward:+.3f}  Glive={g_return:+.2f}  Gsum={g_sum:+.1f}  R{r_window_n}={r_window:+.2f}  γ={self.reward_gamma:.3f}  term={term}\n"
            f"Gema={g_ema:+.3f}  Gstart(old)={g_start:+.2f}  slam{r_window_n}={slam_window}  conf{r_window_n}={conf_window}\n"
            f"raw(v,w)=({fnum(raw[0]):+.3f},{fnum(raw[1]):+.3f})  "
            f"cmd=({fnum(exe[0]):+.3f},{fnum(exe[1]):+.3f})\n"
            f"front={front:.2f}m  actionObs={act_obs:.2f}m  score={act_score:.2f}  near={nearest:.2f}m\n"
            f"scan raw={int(getattr(self, '_last_scan_geometry_debug', {}).get('raw_count', 0))} "
            f"exp={int(getattr(self, '_last_scan_geometry_debug', {}).get('expected_by_meta', 0))} "
            f"bins={int(getattr(self, '_last_scan_geometry_debug', {}).get('policy_bins', self.num_lidar_bins))} "
            f"amin={float(getattr(self, '_last_scan_geometry_debug', {}).get('angle_min', 0.0)):+.2f} "
            f"amax={float(getattr(self, '_last_scan_geometry_debug', {}).get('angle_max', 0.0)):+.2f} "
            f"canon={int(bool(getattr(self, '_last_scan_geometry_debug', {}).get('canonical', False)))} "
            f"frontIdx={int(getattr(self, '_last_scan_geometry_debug', {}).get('front_index', 0))} "
            f"sector={int(getattr(self, '_last_scan_geometry_debug', {}).get('sector_bins', 0))} "
            f"lp={int(getattr(self, '_last_scan_geometry_debug', {}).get('sector_lowpass_kernel', 0))} "
            f"expand={str(getattr(self, '_last_scan_geometry_debug', {}).get('sector_expand_mode', ''))} "
            f"pmin={float(getattr(self, '_last_scan_geometry_debug', {}).get('policy_min', 0.0)):.2f}\n"
            f"anchor=base scanYaw={float(getattr(self, '_last_hard_map_scan_yaw', 0.0)):+.2f} overlayFrame={frame_id}\n"
            f"SAFE pen={safety_pen:.2f} slow={slowdown:.2f} cd={safety_cd} assist={int(bool(getattr(self, '_last_velocity_forward_assist', False)))} spinfix={int(bool(getattr(self, '_last_velocity_spin_breaker', False)))} limit={int(bool(getattr(self, '_last_velocity_command_limited', False)))}  {safety_reason} {str(getattr(self, '_last_velocity_command_limit_reason', 'none'))}\n"
            f"Shake={shake_steps}/{shake_limit} {shake_reason}  "
            f"Lempty={l_empty_steps}/{l_empty_timeout} beams={valid_beams}  "
            f"confStall={int(getattr(self, 'confidence_stall_steps', 0))} "
            f"spinStall={int(getattr(self, 'sustained_rotation_steps', 0))} "
            f"orbitStall={int(getattr(self, 'orbit_stall_steps', 0))} "
            f"eff={float(getattr(self, '_last_orbit_path_efficiency', 1.0)):.2f}\n"
            f"cov={cov:.1f}% Δ={dcov:+.2f}%  conf={conf:.1f} conf/s={conf_rate_ema:.1f}  stale={stale_pct:.1f}% low={low:.1f}%\n"
            f"slamNew={slam_new_known} slamR={slam_r:+.3f}  "
            f"target={target_type}  slam={slam_gate} age={map_age:.1f}s "
            f"ready={str(getattr(self, '_last_post_reset_ready_reason', 'none'))}"
        )
        arr.markers.append(text)

        # Command arrow in the same global frame as the map.  The start is the
        # robot pose and the endpoint is a rotated local velocity vector, so it
        # remains visible even when base_footprint TF display is unreliable.
        arrow = Marker()
        arrow.header.frame_id = frame_id
        arrow.header.stamp = stamp
        arrow.ns = "rl_velocity_debug"
        arrow.id = 1
        arrow.type = Marker.ARROW
        arrow.action = Marker.ADD
        arrow.pose.orientation.w = 1.0
        sx = float(robot_xy_overlay[0])
        sy = float(robot_xy_overlay[1])
        arrow.points = [self._point_xyz(sx, sy, 0.32)]
        vx_scale = 0.90 / max(float(self.max_linear_speed), 1e-6)
        wz_scale = 0.45 / max(float(self.max_angular_speed), 1e-6)
        # Green arrow is linear.x only.  Older overlays mixed angular.z into
        # the arrow endpoint, so a pure spin could look like a forward command.
        local_x = float(np.clip(fnum(exe[0]) * vx_scale, -0.35, 0.90))
        cy = math.cos(robot_yaw_overlay)
        syaw = math.sin(robot_yaw_overlay)
        if abs(local_x) < 0.025:
            local_x = 0.0
        end_x = sx + cy * local_x
        end_y = sy + syaw * local_x
        arrow.points.append(self._point_xyz(end_x, end_y, 0.32))
        arrow.scale.x = 0.065
        arrow.scale.y = 0.18
        arrow.scale.z = 0.24
        if bool(getattr(self, "_last_velocity_safety_backup_triggered", False)):
            self._set_marker_color(arrow, 1.0, 0.15, 0.05, 0.95)
        elif bool(getattr(self, "_last_velocity_safety_blocked", False)):
            self._set_marker_color(arrow, 1.0, 0.75, 0.05, 0.95)
        elif float(getattr(self, "_last_velocity_safety_slowdown", 1.0)) < 0.999:
            self._set_marker_color(arrow, 1.0, 0.95, 0.10, 0.95)
        else:
            self._set_marker_color(arrow, 0.10, 1.0, 0.25, 0.95)
        arr.markers.append(arrow)

        yaw_arrow = Marker()
        yaw_arrow.header.frame_id = frame_id
        yaw_arrow.header.stamp = stamp
        yaw_arrow.ns = "rl_velocity_debug"
        yaw_arrow.id = 3
        yaw_arrow.type = Marker.ARROW
        yaw_arrow.action = Marker.ADD
        yaw_arrow.pose.orientation.w = 1.0
        yaw_mag = float(np.clip(abs(fnum(exe[1])) * wz_scale, 0.0, 0.45))
        yaw_sign = 1.0 if fnum(exe[1]) >= 0.0 else -1.0
        # Yellow arrow is angular.z only, drawn sideways from the robot.
        yaw_local_x = 0.0
        yaw_local_y = yaw_sign * yaw_mag
        yaw_end_x = sx + cy * yaw_local_x - syaw * yaw_local_y
        yaw_end_y = sy + syaw * yaw_local_x + cy * yaw_local_y
        yaw_arrow.points = [self._point_xyz(sx, sy, 0.36), self._point_xyz(yaw_end_x, yaw_end_y, 0.36)]
        yaw_arrow.scale.x = 0.045
        yaw_arrow.scale.y = 0.13
        yaw_arrow.scale.z = 0.18
        self._set_marker_color(yaw_arrow, 1.0, 0.85, 0.05, 0.92)
        arr.markers.append(yaw_arrow)

        heading = Marker()
        heading.header.frame_id = frame_id
        heading.header.stamp = stamp
        heading.ns = "rl_velocity_debug"
        heading.id = 2
        heading.type = Marker.ARROW
        heading.action = Marker.ADD
        heading.pose.orientation.w = 1.0
        hx0 = float(robot_xy_overlay[0])
        hy0 = float(robot_xy_overlay[1])
        hx1 = hx0 + 0.55 * math.cos(robot_yaw_overlay)
        hy1 = hy0 + 0.55 * math.sin(robot_yaw_overlay)
        heading.points = [self._point_xyz(hx0, hy0, 0.40), self._point_xyz(hx1, hy1, 0.40)]
        heading.scale.x = 0.045
        heading.scale.y = 0.13
        heading.scale.z = 0.17
        self._set_marker_color(heading, 0.05, 0.55, 1.00, 0.95)
        arr.markers.append(heading)

        self.waypoint_marker_pub.publish(arr)

    def _update_explored_stall_steps(
        self,
        map_stats: MapUpdateStats,
        action: np.ndarray,
    ) -> int:
        """
        이미 탐색된 영역에서 새 정보 없이 머무는 연속 step 수를 누적한다.

        기준:
          - 새 known cell 거의 없음
          - stale refresh 거의 없음
          - confidence 증가 거의 없음
          - 같은 local cell 방문 횟수가 누적됨 또는 의미 있는 frontier target을 못 따라감

        이 값은 reward에서 시간이 갈수록 증가하는 penalty로 사용된다.
        새 정보를 얻으면 즉시 0으로 reset한다.
        """
        new_info = (
            int(map_stats.new_known_cells) > 1
            or int(map_stats.stale_refresh_cells) > 1
            or float(map_stats.confidence_gain) > 0.02
        )

        if new_info:
            self.explored_stall_steps = 0
            return self.explored_stall_steps

        linear_x = float(action[0])
        angular_z = float(action[1])
        normalized_turn = abs(angular_z) / max(self.max_angular_speed, 1e-6)

        # 같은 cell에서 계속 머물거나, 전진/회전은 하는데 새 정보가 없으면
        # 이미 확인된 영역에서 정책이 정체된 것으로 본다.
        local_revisit = int(map_stats.robot_visit_count) >= 4
        active_but_no_info = linear_x > 0.03 or normalized_turn > 0.20

        if local_revisit or active_but_no_info:
            self.explored_stall_steps += 1
        else:
            # 완전 정지 상태는 너무 빨리 벌을 누적하면 초기 reset 직후가 불안정하므로
            # 완만하게만 줄인다.
            self.explored_stall_steps = max(self.explored_stall_steps - 1, 0)

        return self.explored_stall_steps

    def _update_confidence_stall_steps(
        self,
        map_stats: MapUpdateStats,
        action: np.ndarray,
    ) -> int:
        """Track consecutive steps where the confidence layer is not updated.

        Unlike explored_stall_steps, this counter is focused specifically on
        confidence_gain.  It grows when the robot is moving/turning or low
        confidence remains, but confidence_gain stays below the configured
        threshold.  It resets immediately when confidence_gain becomes
        meaningful.
        """
        try:
            conf_gain = float(getattr(map_stats, "confidence_gain", 0.0))
        except Exception:
            conf_gain = 0.0
        if conf_gain > float(self.confidence_stall_gain_threshold):
            self.confidence_stall_steps = 0
            return self.confidence_stall_steps

        try:
            low_ratio = float(getattr(map_stats, "low_confidence_ratio", 0.0))
        except Exception:
            low_ratio = 0.0
        try:
            linear_x = float(action[0])
            angular_z = float(action[1])
        except Exception:
            linear_x = 0.0
            angular_z = 0.0
        normalized_turn = abs(angular_z) / max(float(self.max_angular_speed), 1e-6)

        active_motion = bool(linear_x > 0.025 or normalized_turn > 0.15)
        confidence_problem = bool(low_ratio >= float(self.confidence_stall_low_ratio_threshold))
        local_revisit = bool(int(getattr(map_stats, "robot_visit_count", 0)) >= 3)

        if active_motion or confidence_problem or local_revisit:
            self.confidence_stall_steps += 1
        else:
            self.confidence_stall_steps = max(self.confidence_stall_steps - 1, 0)

        return self.confidence_stall_steps


    def _update_sustained_rotation_steps(self, action: np.ndarray) -> int:
        """Track consecutive low-translation/high-angular commands.

        A short turn is normal for heading alignment.  Continuous rotation with
        little forward motion is the failure mode where the robot spins in place
        and harvests LiDAR/confidence/priority deltas.  This counter feeds a
        growing reward penalty and is reset/decayed when the robot actually
        translates.
        """
        try:
            linear_x = float(action[0])
            angular_z = float(action[1])
        except Exception:
            linear_x = 0.0
            angular_z = 0.0

        forward_norm = float(
            np.clip(max(linear_x, 0.0) / max(float(self.max_linear_speed), 1e-6), 0.0, 1.0)
        )
        turn_norm = float(
            np.clip(abs(angular_z) / max(float(self.max_angular_speed), 1e-6), 0.0, 1.0)
        )

        # Thresholds are intentionally permissive: necessary heading correction is
        # allowed for a few steps.  Only persistent low-forward/high-turn behavior
        # accumulates.
        rotating_in_place = bool(forward_norm < 0.16 and turn_norm > 0.34)
        if rotating_in_place:
            self.sustained_rotation_steps += 1
        elif forward_norm > 0.22 or turn_norm < 0.18:
            self.sustained_rotation_steps = 0
        else:
            self.sustained_rotation_steps = max(int(self.sustained_rotation_steps) - 1, 0)

        return int(self.sustained_rotation_steps)

    def _update_orbit_stall_steps(self, map_stats: MapUpdateStats, action: np.ndarray) -> int:
        """Track forward arc-loops around the same local area.

        sustained_rotation_steps catches in-place spinning.  This counter catches
        the harder exploit: the robot drives forward while continuously turning,
        so path_length grows but net displacement remains small.  A normal curved
        avoidance maneuver is exempted when it actually moves away or creates
        meaningful confidence/map/priority information.
        """
        try:
            pose = self._get_robot_pose2d()
        except Exception:
            pose = None

        if pose is None:
            self.orbit_stall_steps = max(int(getattr(self, "orbit_stall_steps", 0)) - 1, 0)
            self._last_orbit_reason = "missing_pose"
            return int(self.orbit_stall_steps)

        try:
            xy, yaw = pose
            x = float(xy[0])
            y = float(xy[1])
            yaw = float(yaw)
        except Exception:
            self.orbit_stall_steps = max(int(getattr(self, "orbit_stall_steps", 0)) - 1, 0)
            self._last_orbit_reason = "bad_pose"
            return int(self.orbit_stall_steps)

        if not (np.isfinite(x) and np.isfinite(y) and np.isfinite(yaw)):
            self.orbit_stall_steps = max(int(getattr(self, "orbit_stall_steps", 0)) - 1, 0)
            self._last_orbit_reason = "nonfinite_pose"
            return int(self.orbit_stall_steps)

        try:
            linear_x = float(action[0])
            angular_z = float(action[1])
        except Exception:
            linear_x = 0.0
            angular_z = 0.0

        forward_norm = float(
            np.clip(max(linear_x, 0.0) / max(float(self.max_linear_speed), 1e-6), 0.0, 1.0)
        )
        turn_norm = float(
            np.clip(abs(angular_z) / max(float(self.max_angular_speed), 1e-6), 0.0, 1.0)
        )

        hist = getattr(self, "_orbit_pose_history", None)
        if hist is None:
            hist = deque(maxlen=48)
            self._orbit_pose_history = hist
        hist.append((int(getattr(self, "step_count", 0)), x, y, yaw))

        samples = list(hist)
        if len(samples) < 10:
            self._last_orbit_path_efficiency = 1.0
            self._last_orbit_path_length = 0.0
            self._last_orbit_net_displacement = 0.0
            self._last_orbit_yaw_accum = 0.0
            self._last_orbit_reason = "warming"
            self.orbit_stall_steps = max(int(getattr(self, "orbit_stall_steps", 0)) - 1, 0)
            return int(self.orbit_stall_steps)

        # Use the recent trajectory window.  The deque maxlen already bounds it;
        # keep the code explicit so future maxlen changes remain safe.
        samples = samples[-48:]
        path_len = 0.0
        yaw_accum = 0.0
        for a, b in zip(samples[:-1], samples[1:]):
            dx = float(b[1]) - float(a[1])
            dy = float(b[2]) - float(a[2])
            ds = math.hypot(dx, dy)
            if np.isfinite(ds):
                path_len += ds
            dyaw = self._normalize_angle(float(b[3]) - float(a[3]))
            if np.isfinite(dyaw):
                yaw_accum += abs(float(dyaw))

        net_disp = math.hypot(float(samples[-1][1]) - float(samples[0][1]), float(samples[-1][2]) - float(samples[0][2]))
        eff = float(net_disp / max(path_len, 1e-6)) if path_len > 1e-6 else 1.0
        eff = float(np.clip(eff, 0.0, 1.0))

        self._last_orbit_path_efficiency = eff
        self._last_orbit_path_length = float(path_len)
        self._last_orbit_net_displacement = float(net_disp)
        self._last_orbit_yaw_accum = float(yaw_accum)

        try:
            conf_gain = float(getattr(map_stats, "confidence_gain", 0.0))
        except Exception:
            conf_gain = 0.0
        try:
            new_known = int(getattr(map_stats, "new_known_cells", 0))
        except Exception:
            new_known = 0
        try:
            coverage_delta = float(getattr(map_stats, "coverage_delta", 0.0))
        except Exception:
            coverage_delta = 0.0
        try:
            priority_clear_gain = float(getattr(map_stats, "priority_clear_gain", 0.0))
            priority_rechecked_gain = float(getattr(map_stats, "priority_rechecked_gain", 0.0))
        except Exception:
            priority_clear_gain = 0.0
            priority_rechecked_gain = 0.0

        meaningful_info = bool(
            conf_gain > max(float(getattr(self, "confidence_stall_gain_threshold", 0.02)), 0.02)
            or new_known >= 6
            or coverage_delta >= 2e-4
            or priority_clear_gain >= 0.20
            or priority_rechecked_gain >= 0.35
        )

        arc_motion = bool(forward_norm > 0.12 and turn_norm > 0.24)
        low_eff_loop = bool(
            path_len >= 0.28
            and eff <= 0.38
            and yaw_accum >= math.radians(95.0)
        )

        if arc_motion and low_eff_loop and not meaningful_info:
            self.orbit_stall_steps += 1
            self._last_orbit_reason = (
                f"orbit_loop:eff={eff:.2f},path={path_len:.2f},net={net_disp:.2f},"
                f"yaw={math.degrees(yaw_accum):.1f}"
            )
        elif meaningful_info or eff >= 0.55 or path_len < 0.18 or turn_norm < 0.16:
            self.orbit_stall_steps = 0
            self._last_orbit_reason = (
                "reset:" + ("info" if meaningful_info else f"eff={eff:.2f},path={path_len:.2f},turn={turn_norm:.2f}")
            )
        else:
            self.orbit_stall_steps = max(int(getattr(self, "orbit_stall_steps", 0)) - 1, 0)
            self._last_orbit_reason = f"decay:eff={eff:.2f},path={path_len:.2f},yaw={math.degrees(yaw_accum):.1f}"

        return int(self.orbit_stall_steps)


    def _filter_action(self, raw_action: np.ndarray) -> np.ndarray:
        """
        Backward-compatible no-op action filter.

        이전 버전에서는 여기서 smoothing/rate-limit/hysteresis를 적용했지만,
        현재 버전은 Gazebo로 나가는 제어 입력을 직접 막지 않는다.
        action은 action_space 범위로만 clip하고 그대로 반환한다.
        """
        action = np.asarray(raw_action, dtype=np.float32)
        action = np.clip(action, self.action_space.low, self.action_space.high)
        self.filtered_action = action.astype(np.float32).copy()
        return self.filtered_action.copy()

    def _apply_motion_mode_hysteresis(self, action: np.ndarray) -> np.ndarray:
        """No-op. Kept only for CLI/API compatibility."""
        return np.asarray(action, dtype=np.float32).copy()

    def _advance_world_after_command(self, target_delta_sec: float):
        prev_sim_time = self.ros.get_sim_time_sec()
        prev_odom_stamp = self.ros.get_odom_stamp_sec()
        prev_scan_wall = self.ros.last_scan_time
        prev_odom_wall = self.ros.last_odom_time

        if self.use_world_step and self.sim_controller is not None:
            ok = self.sim_controller.step(self.sim_steps_per_action)

            if not ok:
                self.ros.get_logger().warn(
                    "World multi_step failed. Falling back to short ROS spin."
                )
                self.ros.spin_steps(num_spins=20, timeout_sec=0.001)
                return

            target_fraction = self.world_step_target_fraction
            time_advanced = True
            if target_fraction > 0.0:
                time_advanced = self.ros.wait_for_time_advance(
                    start_sim_time_sec=prev_sim_time,
                    start_odom_stamp_sec=prev_odom_stamp,
                    target_delta_sec=max(float(target_delta_sec) * target_fraction, 1e-4),
                    timeout_wall_sec=self.world_step_wait_timeout_sec,
                )

            sensor_updated = self.ros.wait_for_new_sensor_frame(
                prev_scan_wall_time=prev_scan_wall,
                prev_odom_wall_time=prev_odom_wall,
                timeout_wall_sec=self.world_step_sensor_timeout_sec,
            )

            if time_advanced or sensor_updated:
                self._world_step_stale_count = 0
            else:
                self._world_step_stale_count += 1
                if self.step_count % self.world_step_stale_warn_every_n == 0:
                    self.ros.get_logger().warn(
                        "Gazebo multi_step returned before clock/odom/sensor callbacks advanced. "
                        "Continuing, but observation may be stale. "
                        f"stale_count={self._world_step_stale_count}, "
                        f"sim_time={self.ros.get_sim_time_sec()}, "
                        f"odom_stamp={self.ros.get_odom_stamp_sec()}. "
                        "If this repeats, run with --disable-world-step."
                    )
                if (
                    self.world_step_auto_disable_on_stale
                    and self._world_step_stale_count >= self.world_step_stale_limit
                ):
                    self.use_world_step = False
                    self._world_step_stale_count = 0
                    self.ros.get_logger().error(
                        "Disabling Gazebo multi_step for this run because observations stayed stale. "
                        "Falling back to wall-clock ROS spinning. For future runs, add --disable-world-step."
                    )

            self.ros.spin_steps(num_spins=5, timeout_sec=0.0)

        else:
            # Real-time fallback: Gazebo is already running unpaused in its own
            # process/thread.  Previous high-throughput builds only drained ROS
            # callbacks and yielded the CPU briefly here, which made one Gym step
            # much shorter than control_dt.  That is good for throughput but bad
            # when step-count based events such as emergency priority spawning or
            # synchronous backup are expected to represent physical time.
            #
            # v118: optional strict wall-clock control period.  When enabled, each
            # env step/micro-step keeps the command active for approximately
            # target_delta_sec seconds of wall-clock time while continuously
            # spinning callbacks.  This makes, for example, control_dt=0.10 mean
            # about 10 Hz from the policy/environment point of view.
            start_wall = time.monotonic()

            if self.realtime_spin_steps > 0:
                self.ros.spin_steps(
                    num_spins=self.realtime_spin_steps,
                    timeout_sec=self.realtime_spin_timeout_sec,
                )

            if bool(getattr(self, "realtime_enforce_control_dt", False)):
                target_wall = max(float(target_delta_sec), 0.0) + float(getattr(self, "realtime_control_dt_wall_margin_sec", 0.0))
                # Keep small spin/sleep slices so /clock, odom, scan, SLAM, and
                # map timers can update while the command is being held.
                while rclpy.ok():
                    elapsed = time.monotonic() - start_wall
                    remaining = target_wall - elapsed
                    if remaining <= 0.0:
                        break
                    spin_timeout = min(max(float(remaining), 0.0), 0.005)
                    self.ros.spin_steps(num_spins=1, timeout_sec=spin_timeout)
                    if remaining > 0.001:
                        time.sleep(min(0.001, max(remaining, 0.0)))
                self._last_realtime_step_wall_elapsed_sec = float(time.monotonic() - start_wall)
            else:
                if self.realtime_sleep_sec > 0.0:
                    time.sleep(self.realtime_sleep_sec)
                self._last_realtime_step_wall_elapsed_sec = float(time.monotonic() - start_wall)

    def close(self):
        self._clear_waypoint_visualization()
        self.ros.stop_robot()

        if self.sim_controller is not None and hasattr(self.sim_controller, "close"):
            self.sim_controller.close()

        if self.reset_manager is not None and hasattr(self.reset_manager, "close"):
            self.reset_manager.close()

        if self.nav2_proc is not None and self.nav2_proc.poll() is None:
            self.ros.get_logger().info("Stopping internal Nav2 process...")
            self.nav2_proc.terminate()
            try:
                self.nav2_proc.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                self.nav2_proc.kill()
        self.nav2_proc = None

    def _log_reset_pose_truth(
        self,
        requested_x: float,
        requested_y: float,
        reset_pose,
    ):
        """
        reset target과 실제 pose를 분리해서 로그로 남긴다.

        Gazebo GUI, odom, map frame은 서로 좌표계가 다를 수 있다.
        따라서 디버깅 시에는 requested pose / Gazebo actual pose / odom pose / map pose를
        한 줄에 같이 봐야 한다.
        """
        entity = "unknown"
        actual_gz = "unverified"

        if self.reset_manager is not None:
            entity = (
                self.reset_manager.last_reset_entity_name
                or self.reset_manager.entity_name
            )
            actual_gz = (
                self.reset_manager._format_pose(  # pylint: disable=protected-access
                    self.reset_manager.last_actual_pose
                )
            )

        odom_pose = self.ros.get_pose2d(frame_id="odom")
        map_pose = self.ros.get_pose2d(frame_id=self.pose_frame)

        if odom_pose is None:
            odom_text = "unavailable"
        else:
            odom_xy, odom_yaw = odom_pose
            odom_text = f"(x={float(odom_xy[0]):.3f}, y={float(odom_xy[1]):.3f}, yaw={float(odom_yaw):.3f})"

        if map_pose is None:
            map_text = "unavailable"
        else:
            map_xy, map_yaw = map_pose
            map_text = f"(x={float(map_xy[0]):.3f}, y={float(map_xy[1]):.3f}, yaw={float(map_yaw):.3f})"

        if reset_pose is None:
            reset_text = "pose reset disabled"
        else:
            reset_text = (
                f"returned=(x={float(reset_pose.x):.3f}, y={float(reset_pose.y):.3f}, "
                f"z={float(reset_pose.z):.3f}, yaw={float(reset_pose.yaw):.3f})"
            )

        _reset_truth_msg = (
            "RESET_POSE_TRUTH | "
            f"candidate_requested=(x={float(requested_x):.3f}, y={float(requested_y):.3f}) | "
            f"entity='{entity}' | "
            f"{reset_text} | "
            f"actual_gazebo={actual_gz} | "
            f"odom_pose={odom_text} | "
            f"{self.pose_frame}_pose={map_text}"
        )
        if _quiet_reset_logs():
            self.ros.get_logger().debug(_reset_truth_msg)
        else:
            self.ros.get_logger().info(_reset_truth_msg)

    def _reset_slam_map_after_pose_reset(self) -> bool:
        if not hasattr(self.ros, "reset_slam_mapping"):
            return False

        ok = self.ros.reset_slam_mapping(
            timeout_sec=self.slam_reset_timeout_sec,
            allow_process_restart=self.restart_slam_on_reset,
        )

        # Gazebo world가 paused 상태일 수 있으므로, SLAM 재시작 후 scan/odom/map이
        # 들어오도록 physics를 조금 전진시킨다.
        for _ in range(self.slam_reset_warmup_steps):
            self._advance_world_after_command(
                target_delta_sec=min(self.control_dt, 0.05)
            )
            self.ros.spin_steps(num_spins=10, timeout_sec=0.001)
            if self.ros.slam_map is not None:
                return True

        if self.ros.slam_map is None:
            self.ros.wait_for_slam_map_ready(timeout_sec=self.slam_reset_timeout_sec)

        return bool(ok and self.ros.slam_map is not None)

    def _init_debug_input_map_publishers(self) -> None:
        channel_topics = {
            "free": "slam_free",
            "unknown": "slam_unknown",
            "occupied": "slam_occupied",
            "confidence": "confidence",
            "priority": "priority",
        }
        for key, suffix in channel_topics.items():
            topic = f"{self.debug_input_map_topic_prefix}/{suffix}"
            self.debug_input_map_publishers[key] = self.ros.create_publisher(
                OccupancyGrid,
                topic,
                10,
            )
        self.ros.get_logger().info(
            "Debug CNN input map topics enabled: "
            f"prefix={self.debug_input_map_topic_prefix}, "
            f"frame={self.debug_input_map_frame_id}, "
            f"every_n={self.debug_input_map_publish_every_n}. "
            "OccupancyGrid frame uses x=robot_forward, y=robot_left."
        )

    @staticmethod
    def _tensor_channel_to_robot_local_grid(channel: np.ndarray) -> np.ndarray:
        """
        Convert obs["map"][ch] to OccupancyGrid row-major layout.

        Tensor convention from ExplorationGridMap.build_update_need_tensor():
          - row 0 is robot forward
          - last row is robot backward
          - column 0 is robot left
          - last column is robot right

        OccupancyGrid in base_link-like frame:
          - grid x increases forward
          - grid y increases left
          - data index = x + y * width

        Therefore grid[y, x] = tensor[H - 1 - x, W - 1 - y].
        """
        arr = np.asarray(channel, dtype=np.float32)
        if arr.ndim != 2:
            return np.zeros((1, 1), dtype=np.float32)
        return np.flip(arr, axis=(0, 1)).T.astype(np.float32, copy=False)

    def _publish_debug_input_map(self, map_obs: np.ndarray) -> None:
        if not self.debug_input_map or not self.use_map_cnn:
            self._last_debug_input_map_published = False
            return
        if not self.debug_input_map_publishers:
            self._last_debug_input_map_published = False
            return
        if self.step_count % self.debug_input_map_publish_every_n != 0:
            self._last_debug_input_map_published = False
            return
        if self._last_debug_input_map_publish_step == self.step_count:
            self._last_debug_input_map_published = False
            return

        arr = np.asarray(map_obs, dtype=np.float32)
        if arr.ndim != 3 or arr.shape[0] < 5:
            self._last_debug_input_map_published = False
            return

        _, height, width = arr.shape
        if height <= 0 or width <= 0:
            self._last_debug_input_map_published = False
            return

        resolution = float(self.map_obs_size_m) / max(float(width), 1.0)
        half = float(self.map_obs_size_m) * 0.5
        stamp = self.ros.get_clock().now().to_msg()

        channel_order = [
            ("free", 0),
            ("unknown", 1),
            ("occupied", 2),
            ("confidence", 3),
            ("priority", 4),
        ]

        for name, ch in channel_order:
            pub = self.debug_input_map_publishers.get(name)
            if pub is None:
                continue
            grid_local = self._tensor_channel_to_robot_local_grid(arr[ch])
            values = np.clip(np.rint(grid_local * 100.0), 0, 100).astype(np.int8)

            msg = OccupancyGrid()
            msg.header.stamp = stamp
            msg.header.frame_id = self.debug_input_map_frame_id
            msg.info.resolution = resolution
            msg.info.width = int(width)
            msg.info.height = int(height)
            msg.info.origin.position.x = -half
            msg.info.origin.position.y = -half
            msg.info.origin.position.z = 0.0
            msg.info.origin.orientation.w = 1.0
            msg.data = values.reshape(-1).tolist()
            pub.publish(msg)

        self._last_debug_input_map_publish_step = self.step_count
        self._last_debug_input_map_published = True

    def _scan_float(self, value, default=0.0):
        try:
            out = float(value)
            return out if math.isfinite(out) else float(default)
        except Exception:
            return float(default)

    def _scan_bool_env(self, name: str, default: bool) -> bool:
        raw = os.environ.get(name, "1" if default else "0")
        return str(raw).strip().lower() not in {"0", "false", "no", "off", "disable", "disabled"}

    def _policy_scan_front_index(self) -> int:
        try:
            return int(os.environ.get("TB3_RL_LIDAR_FRONT_INDEX", "0")) % max(int(self.num_lidar_bins), 1)
        except Exception:
            return 0

    def _update_scan_geometry_debug(self, scan_msg, vector_obs: np.ndarray) -> None:
        raw_ranges = getattr(scan_msg, "ranges", []) or []
        raw_count = len(raw_ranges)
        angle_min = self._scan_float(getattr(scan_msg, "angle_min", 0.0), 0.0)
        angle_max = self._scan_float(getattr(scan_msg, "angle_max", 0.0), 0.0)
        angle_inc = self._scan_float(getattr(scan_msg, "angle_increment", 0.0), 0.0)
        canonical = self._scan_bool_env("TB3_RL_LIDAR_CANONICAL_FRONT_ZERO", True)
        front_index = self._policy_scan_front_index()
        metadata_valid = raw_count > 0 and math.isfinite(angle_min) and math.isfinite(angle_inc) and abs(angle_inc) > 1e-12
        expected_by_meta = 0
        if metadata_valid:
            expected_by_meta = int(round(abs(angle_max - angle_min) / max(abs(angle_inc), 1e-12))) + 1

        lidar_vec = np.asarray(vector_obs[: self.num_lidar_bins], dtype=np.float32)
        finite = lidar_vec[np.isfinite(lidar_vec)]
        lidar_min = float(np.min(finite)) if finite.size else float("nan")
        lidar_mean = float(np.mean(finite)) if finite.size else float("nan")

        self._last_scan_geometry_debug = {
            "raw_count": int(raw_count),
            "expected_by_meta": int(expected_by_meta),
            "angle_min": float(angle_min),
            "angle_max": float(angle_max),
            "angle_increment": float(angle_inc),
            "metadata_valid": bool(metadata_valid),
            "canonical": bool(canonical),
            "front_index": int(front_index),
            "angle_offset_deg": float(math.degrees(self._policy_scan_angle_offset_rad())),
            "flip_lr": bool(self._policy_scan_flip_lr()),
            "sector_bins": int(os.environ.get("TB3_RL_LIDAR_SECTOR_BINS", "0") or "0") if str(os.environ.get("TB3_RL_LIDAR_SECTOR_BINS", "0")).lstrip("-+").isdigit() else 0,
            "sector_lowpass_kernel": int(os.environ.get("TB3_RL_LIDAR_SECTOR_LOWPASS_KERNEL", "0") or "0") if str(os.environ.get("TB3_RL_LIDAR_SECTOR_LOWPASS_KERNEL", "0")).lstrip("-+").isdigit() else 0,
            "sector_expand_mode": str(os.environ.get("TB3_RL_LIDAR_SECTOR_EXPAND_MODE", "")),
            "policy_bins": int(self.num_lidar_bins),
            "input_lidar_bins": int(self.num_lidar_bins),
            "policy_min": float(lidar_min),
            "policy_mean": float(lidar_mean),
        }

        try:
            obs_scan_dbg_sec = float(os.environ.get("TB3_RL_OBS_SCAN_GEOMETRY_DEBUG_SEC", "0.0"))
        except Exception:
            obs_scan_dbg_sec = 0.0
        now = time.time()
        if obs_scan_dbg_sec > 0.0 and now - float(getattr(self, "_last_scan_geometry_log_time", 0.0)) >= obs_scan_dbg_sec:
            self._last_scan_geometry_log_time = now
            try:
                self.ros.get_logger().warn(
                    "OBS_SCAN_GEOMETRY | "
                    f"raw={raw_count} expected={expected_by_meta} input_bins={self.num_lidar_bins} "
                    f"angle_min={angle_min:.6f} angle_max={angle_max:.6f} inc={angle_inc:.9f} "
                    f"metadata_valid={metadata_valid} canonical={canonical} front_index={front_index} "
                    f"policy_min={lidar_min:.3f} policy_mean={lidar_mean:.3f}"
                )
            except Exception:
                pass


    def _get_tf_cube_confidence_pose2d(self) -> Optional[tuple[np.ndarray, float]]:
        """Pose used to UNIFY the confidence origin with the visible TF CUBE.

        v125 rule:
          - The green CUBE is the source of truth, because that is the marker the
            user verified against RViz RobotModel.
          - The confidence cone uses the exact same manual-TF lookup path as the
            CUBE marker used to use: target=/map, source=base_footprint by default.
          - tf2 Buffer lookup is only an optional fallback.  In the failing run the
            tf2 path and manual path diverged, so the default must be manual-first.
          - A short hold keeps both confidence and the marker from blinking when one
            TF sample is late.  If the hold expires, unified mode skips the update
            instead of falling back to an unrelated odom/anchor pose.
        """
        target_frame = str(os.environ.get(
            "TB3_RL_CONFIDENCE_TARGET_FRAME", self.map_frame or "map"
        ) or "map").strip().lstrip("/") or "map"

        source_frame = str(os.environ.get(
            "TB3_RL_CONFIDENCE_CUBE_FRAME",
            os.environ.get("TB3_RL_CONFIDENCE_BASE_FRAME", "base_footprint"),
        ) or "base_footprint").strip().lstrip("/") or "base_footprint"

        try:
            max_age = float(os.environ.get("TB3_RL_MANUAL_TF_MAX_AGE_SEC", "15.0") or 15.0)
        except Exception:
            max_age = 15.0

        pose = None
        # This is the important change: manual TF cache first, because the green
        # compare CUBE that looked correct was generated from get_frame_pose2d_manual().
        try:
            if hasattr(self.ros, "get_frame_pose2d_manual"):
                pose = self.ros.get_frame_pose2d_manual(
                    target_frame=target_frame,
                    source_frame=source_frame,
                    max_age_sec=max_age,
                )
        except Exception:
            pose = None

        # Optional only.  Leave disabled by default so a tf2/manual mismatch cannot
        # silently move confidence away from the verified CUBE pose.
        try:
            buffer_fallback = str(os.environ.get(
                "TB3_RL_CONFIDENCE_CUBE_TF_BUFFER_FALLBACK", "0"
            ) or "0").strip().lower() in {"1", "true", "yes", "on"}
        except Exception:
            buffer_fallback = False
        if pose is None and buffer_fallback:
            try:
                if hasattr(self.ros, "get_frame_pose2d"):
                    pose = self.ros.get_frame_pose2d(
                        target_frame=target_frame,
                        source_frame=source_frame,
                        stamp=None,
                        timeout_sec=0.02,
                        allow_latest_fallback=True,
                    )
            except Exception:
                pose = None

        if pose is not None:
            xy, yaw = pose
            xy = np.asarray(xy, dtype=np.float32).reshape(-1)[:2]
            if xy.size >= 2 and np.all(np.isfinite(xy)) and math.isfinite(float(yaw)):
                self._last_direct_tf_pose_xy = xy.copy()
                self._last_direct_tf_pose_yaw = float(yaw)
                self._last_direct_tf_pose_wall = time.time()
                self._last_direct_tf_pose_frame = source_frame
                self._last_direct_tf_pose_source = "manual_cube" if not buffer_fallback else "manual_or_buffer_cube"
                return xy.copy(), float(yaw)

        # Hold the last verified CUBE pose across brief TF gaps.  This prevents the
        # RViz marker from being DELETEALL-cleared and prevents one missed TF sample
        # from forcing the confidence updater into a different pose source.
        try:
            hold_sec = float(os.environ.get("TB3_RL_CONFIDENCE_TF_HOLD_SEC", "1.5") or 1.5)
        except Exception:
            hold_sec = 1.5
        held_xy = getattr(self, "_last_direct_tf_pose_xy", None)
        held_yaw = getattr(self, "_last_direct_tf_pose_yaw", None)
        held_wall = float(getattr(self, "_last_direct_tf_pose_wall", 0.0))
        if (
            isinstance(held_xy, np.ndarray)
            and held_xy.size >= 2
            and held_yaw is not None
            and (time.time() - held_wall) <= max(hold_sec, 0.0)
        ):
            return held_xy.copy(), float(held_yaw)

        try:
            if self._scan_bool_env("TB3_RL_TF_CUBE_POSE_WARN", False):
                now = time.time()
                if now - float(getattr(self, "_last_tf_cube_pose_warn_time", 0.0)) > 2.0:
                    self._last_tf_cube_pose_warn_time = now
                    self.ros.get_logger().warn(
                        "TF_CUBE_POSE_UNAVAILABLE | "
                        f"target={target_frame} source={source_frame} max_age={max_age:.1f}s | "
                        "skip unified confidence update until manual TF returns"
                    )
        except Exception:
            pass
        return None

    def _get_map_base_pose2d_hard(self) -> Optional[tuple[np.ndarray, float]]:
        """Strict canonical robot anchor for every /map-aligned RL layer.

        v29 rule: confidence/priority map writes use the *robot pose on the
        SLAM map*, not the OccupancyGrid origin and not odometry treated as map.
        This is exactly the TF transform RViz uses to place the robot model:

            target_frame = map
            source_frame = base_footprint   # or configured base frame

        Internally this reads /tf through the tf2 buffer.  If that TF is
        unavailable, semantic layers skip the update instead of painting at a
        guessed map origin.
        """
        target_frame = str(os.environ.get("TB3_RL_CONFIDENCE_TARGET_FRAME", self.map_frame or "map") or "map").strip().lstrip("/") or "map"
        base_candidates = []
        # Highest priority: explicit confidence base frame.  This prevents an
        # old launch environment or a fallback list from accidentally using a
        # different base frame than RViz.
        try:
            env_base = str(os.environ.get("TB3_RL_CONFIDENCE_BASE_FRAME", "") or "").strip().lstrip("/")
            if env_base:
                base_candidates.append(env_base)
        except Exception:
            pass
        try:
            strict_base = str(os.environ.get("TB3_RL_CONFIDENCE_STRICT_BASE_FRAME", "1") or "1").strip().lower() in {"1", "true", "yes", "on"}
        except Exception:
            strict_base = True
        if not strict_base:
            try:
                base_candidates.extend(list(getattr(self.ros, "base_frame_fallbacks", []) or []))
            except Exception:
                pass
            try:
                base_candidates.append(str(getattr(self.ros, "base_frame", "base_footprint") or "base_footprint"))
            except Exception:
                pass
            base_candidates.extend(["base_footprint", "base_link", "base_scan"])

        seen = set()
        for source_frame in base_candidates:
            source_frame = str(source_frame or "").strip().lstrip("/")
            if not source_frame or source_frame in seen:
                continue
            seen.add(source_frame)
            pose = None
            try:
                # RViz places the robot model using the live tf2 buffer
                # (target=map, source=base_footprint), which time-synchronizes the
                # full map->odom->base chain.  The manual /tf cache instead
                # composes the latest value of each edge independently, so when
                # map->odom (SLAM, low rate) and odom->base (odom, high rate) have
                # different timestamps the confidence anchor lags / offsets from
                # the RViz robot.  To make the confidence map origin EXACTLY match
                # the RViz robot position, prefer the tf2 buffer and keep the
                # manual cache only as a fallback for brief TF gaps.
                prefer_tf_buffer = str(os.environ.get("TB3_RL_CONFIDENCE_PREFER_TF_BUFFER", "1") or "1").strip().lower() in {"1", "true", "yes", "on"}
                use_manual = str(os.environ.get("TB3_RL_USE_MANUAL_TF_CACHE_FOR_CONFIDENCE", "1") or "1").strip().lower() in {"1", "true", "yes", "on"}
                buffer_fallback = str(os.environ.get("TB3_RL_CONFIDENCE_TF_BUFFER_FALLBACK", "0") or "0").strip().lower() in {"1", "true", "yes", "on"}
                max_age = float(os.environ.get("TB3_RL_MANUAL_TF_MAX_AGE_SEC", "5.0") or 5.0)

                def _lookup_tf_buffer():
                    if not hasattr(self.ros, "get_frame_pose2d"):
                        return None
                    return self.ros.get_frame_pose2d(
                        target_frame=target_frame,
                        source_frame=source_frame,
                        stamp=None,
                        timeout_sec=0.05,
                        allow_latest_fallback=True,
                    )

                def _lookup_manual():
                    if not hasattr(self.ros, "get_frame_pose2d_manual"):
                        return None
                    return self.ros.get_frame_pose2d_manual(
                        target_frame=target_frame,
                        source_frame=source_frame,
                        max_age_sec=max_age,
                    )

                if prefer_tf_buffer:
                    # tf2 buffer first (RViz-identical), manual cache as fallback.
                    pose = _lookup_tf_buffer()
                    if pose is None and use_manual:
                        pose = _lookup_manual()
                else:
                    # Legacy order: manual cache first, tf2 buffer as fallback.
                    if use_manual:
                        pose = _lookup_manual()
                    if pose is None and buffer_fallback:
                        pose = _lookup_tf_buffer()
            except Exception:
                pose = None
            if pose is not None:
                xy, yaw = pose
                xy = np.asarray(xy, dtype=np.float32)
                if xy.shape[0] >= 2 and np.all(np.isfinite(xy[:2])) and math.isfinite(float(yaw)):
                    self._last_hard_map_base_frame = source_frame
                    self._last_hard_map_base_xy = xy[:2].copy()
                    self._last_hard_map_base_yaw = float(yaw)
                    return xy[:2], float(yaw)

        # Do not spam this during normal training.  Cartographer may briefly drop
        # map->base TF during per-episode SLAM restart; v11 waits for a verified
        # map->base anchor instead of writing at a guessed SLAM-local origin.
        try:
            warn_enabled = str(os.environ.get("TB3_RL_MAP_BASE_TF_WARN", "0")).strip().lower() in {"1", "true", "yes", "on"}
        except Exception:
            warn_enabled = False
        now = time.time()
        if warn_enabled and now - float(getattr(self, "_last_hard_map_base_tf_warn_time", 0.0)) > 2.0:
            self._last_hard_map_base_tf_warn_time = now
            try:
                self.ros.get_logger().warn(
                    "TF_MAP_BASE_POSE_UNAVAILABLE | "
                    f"target={target_frame} candidates={list(seen)} | "
                    "will skip semantic map write until map TF is available"
                )
            except Exception:
                pass
        return None

    def _get_slam_origin_anchor_pose2d(self, motion_yaw: Optional[float] = None) -> Optional[tuple[np.ndarray, float, str]]:
        """Fallback anchor for per-episode Cartographer TF gaps.

        In this project every persistent semantic layer is published on the raw
        /map OccupancyGrid canvas.  The preferred anchor is TF(map->base).  During
        Cartographer restart this TF can be unavailable even though /map and the
        actual odometry stream are alive.  In that case, do not skip confidence
        updates indefinitely: anchor the actual odometry pose to the SLAM-local
        origin.  For a freshly restarted Cartographer map this is normally where
        the robot starts.

        This is not command integration and not an odom-as-map fallback.  It only
        supplies the initial map-frame anchor; subsequent motion still comes from
        actual odometry (/model odom preferred).
        """
        try:
            enabled = str(os.environ.get("TB3_RL_CONFIDENCE_ALLOW_SLAM_ORIGIN_ANCHOR", "0")).strip().lower() in {"1", "true", "yes", "on"}
        except Exception:
            enabled = False
        if not enabled:
            return None
        try:
            slam = getattr(self.ros, "slam_map", None)
            if slam is None:
                return None
            info = slam.info
            width = int(info.width)
            height = int(info.height)
            res = float(info.resolution)
            if width <= 0 or height <= 0 or not math.isfinite(res) or res <= 0.0:
                return None
        except Exception:
            return None

        try:
            ax = float(os.environ.get("TB3_RL_CONFIDENCE_SLAM_ANCHOR_X", "0.0") or 0.0)
            ay = float(os.environ.get("TB3_RL_CONFIDENCE_SLAM_ANCHOR_Y", "0.0") or 0.0)
        except Exception:
            ax, ay = 0.0, 0.0

        # If a yaw is explicitly provided use it; otherwise align map yaw with
        # the actual odometry yaw at the anchor so camera/front rays rotate with
        # the real robot pose.
        try:
            if "TB3_RL_CONFIDENCE_SLAM_ANCHOR_YAW_DEG" in os.environ:
                yaw = math.radians(float(os.environ.get("TB3_RL_CONFIDENCE_SLAM_ANCHOR_YAW_DEG", "0.0") or 0.0))
            elif motion_yaw is not None and math.isfinite(float(motion_yaw)):
                yaw = float(motion_yaw)
            else:
                yaw = 0.0
        except Exception:
            yaw = float(motion_yaw) if motion_yaw is not None and math.isfinite(float(motion_yaw)) else 0.0

        try:
            dbg = str(os.environ.get("TB3_RL_CONFIDENCE_ANCHOR_DEBUG", "0")).strip().lower() in {"1", "true", "yes", "on"}
            if dbg:
                now = time.time()
                if now - float(getattr(self, "_last_slam_origin_anchor_log_time", 0.0)) > 2.0:
                    self._last_slam_origin_anchor_log_time = now
                    self.ros.get_logger().warn(
                        "CONFIDENCE_SLAM_ORIGIN_ANCHOR | "
                        f"map_xy=({ax:+.3f},{ay:+.3f}) yaw={math.degrees(yaw):+.1f}deg "
                        f"canvas={width}x{height} res={res:.3f}"
                    )
        except Exception:
            pass

        return np.array([ax, ay], dtype=np.float32), float(yaw), "slam_origin"

    def _get_map_scan_yaw_hard(self, raw_scan_msg=None, base_yaw: Optional[float] = None) -> float:
        """Return the current scan-frame yaw expressed in /map.

        Important distinction for this project:
          - ray *origin* is the robot anchor: TF(map -> base_footprint).
          - ray *orientation* must still follow the LaserScan frame: TF(map -> base_scan).

        The old v13 path used base yaw for both origin and scan angles.  If
        base_scan has any yaw offset relative to base_footprint, confidence and
        priority rays rotate away from the real scan and appear not to update at
        the robot.
        """
        target_frame = str(self.map_frame or "map").strip().lstrip("/") or "map"
        scan_frame = ""
        try:
            scan_frame = str(getattr(getattr(raw_scan_msg, "header", None), "frame_id", "") or "").strip().lstrip("/")
        except Exception:
            scan_frame = ""
        if not scan_frame:
            scan_frame = str(os.environ.get("TB3_RL_SCAN_FRAME", getattr(self, "scan_frame", "base_scan")) or "base_scan").strip().lstrip("/") or "base_scan"

        pose = None
        try:
            if hasattr(self.ros, "get_frame_pose2d"):
                # Use latest TF for semantic map updates.  All persistent layers
                # are updated from the same current TF snapshot; do not mix old
                # scan stamps with current /map canvas.
                pose = self.ros.get_frame_pose2d(
                    target_frame=target_frame,
                    source_frame=scan_frame,
                    stamp=None,
                    timeout_sec=0.05,
                    allow_latest_fallback=True,
                )
        except Exception:
            pose = None
        if pose is not None:
            try:
                _, yaw = pose
                if math.isfinite(float(yaw)):
                    self._last_hard_map_scan_frame = scan_frame
                    self._last_hard_map_scan_yaw = float(yaw)
                    return float(yaw)
            except Exception:
                pass

        # Fall back to base yaw only for orientation, never for XY/map origin.
        # This keeps confidence/priority anchored at the robot even if the
        # optional base_scan TF is temporarily missing.
        if base_yaw is not None and math.isfinite(float(base_yaw)):
            now = time.time()
            if now - float(getattr(self, "_last_scan_yaw_tf_warn_time", 0.0)) > 2.0:
                self._last_scan_yaw_tf_warn_time = now
                try:
                    self.ros.get_logger().warn(
                        "TF_MAP_SCAN_YAW_UNAVAILABLE | "
                        f"target={target_frame} source={scan_frame} | use base yaw fallback for ray direction"
                    )
                except Exception:
                    pass
            return float(base_yaw)
        return 0.0

    def _select_confidence_ray_yaw(
        self,
        *,
        base_yaw: float,
        scan_yaw: Optional[float] = None,
        mode_hint: str = "",
    ) -> float:
        """Select the yaw used to paint the confidence cone.

        v111 fix:
          In map_base_tf mode the origin already follows TF(map->base_footprint),
          but older code used TF(map->base_scan) yaw for the ray.  In the current
          Gazebo/Cartographer setup that scan yaw can differ from the RobotModel
          body yaw and the magenta cone points in the wrong direction.  Default
          back to the robot/base yaw, while keeping a scan-yaw option for explicit
          experiments.

        Environment:
          TB3_RL_CONFIDENCE_RAY_YAW_SOURCE=base|scan    default: base
          TB3_RL_CONFIDENCE_YAW_OFFSET_DEG=<degrees>    default: 0
        """
        try:
            source = str(os.environ.get("TB3_RL_CONFIDENCE_RAY_YAW_SOURCE", "base") or "base").strip().lower()
        except Exception:
            source = "base"
        yaw = float(base_yaw) if math.isfinite(float(base_yaw)) else 0.0
        if source in {"scan", "base_scan", "lidar", "laser", "sensor"}:
            try:
                if scan_yaw is not None and math.isfinite(float(scan_yaw)):
                    yaw = float(scan_yaw)
            except Exception:
                pass
        try:
            offset_deg = float(os.environ.get("TB3_RL_CONFIDENCE_YAW_OFFSET_DEG", "0.0") or 0.0)
        except Exception:
            offset_deg = 0.0
        yaw = float(yaw) + math.radians(float(offset_deg))
        return math.atan2(math.sin(yaw), math.cos(yaw))

    def _pose2d_from_odometry_msg(self, msg, label: str) -> Optional[tuple[np.ndarray, float, str]]:
        """Extract an SE(2) pose from an Odometry-like message.

        The frame does not have to be /map.  For the confidence v6 odom-delta
        policy, it only needs to be a self-consistent moving coordinate system.
        """
        if msg is None:
            return None
        try:
            p = msg.pose.pose.position
            q = msg.pose.pose.orientation
            xy = np.array([float(p.x), float(p.y)], dtype=np.float32)
            yaw = self._yaw_from_quaternion_xyzw_static(float(q.x), float(q.y), float(q.z), float(q.w))
            if not (np.all(np.isfinite(xy)) and math.isfinite(float(yaw))):
                return None
            return xy, float(yaw), str(label)
        except Exception:
            return None

    def _confidence_time_sec(self) -> float:
        """Return a monotonic-ish time for command integration.

        Prefer simulation time when it is advancing; otherwise use wall time.
        """
        try:
            t = self.ros.get_sim_time_sec()
            if t is not None and math.isfinite(float(t)) and float(t) > 0.0:
                return float(t)
        except Exception:
            pass
        return float(time.time())

    def _get_confidence_cmd_integrated_pose2d(self) -> Optional[tuple[np.ndarray, float, str]]:
        """Integrate the actually published cmd_vel for confidence pose only.

        This is deliberately not used for robot localization, reward terminal
        checks, or SLAM.  It is a visualization/task-map fallback for the case
        seen in this run: map->base, odom->base, and model odom are unavailable
        or frozen while the direct velocity controller is still publishing
        non-zero commands.
        """
        try:
            now = self._confidence_time_sec()
        except Exception:
            now = float(time.time())

        try:
            xy = np.asarray(getattr(self, "_confidence_cmd_xy", np.zeros(2, dtype=np.float32)), dtype=np.float32)[:2]
            yaw = float(getattr(self, "_confidence_cmd_yaw", 0.0))
        except Exception:
            xy = np.zeros(2, dtype=np.float32)
            yaw = 0.0

        last_t = getattr(self, "_confidence_cmd_last_time", None)
        if last_t is None or not math.isfinite(float(last_t)):
            self._confidence_cmd_last_time = now
            self._confidence_cmd_xy = xy.copy()
            self._confidence_cmd_yaw = float(yaw)
            return xy.copy(), float(yaw), "cmd_integrated"

        dt = float(now) - float(last_t)
        # Timer jitter and reset pauses can create large wall-time gaps; never
        # integrate a huge jump into the map.
        dt = float(np.clip(dt, 0.0, float(os.environ.get("TB3_RL_CONFIDENCE_CMD_DT_CAP", "0.12") or 0.12)))
        self._confidence_cmd_last_time = now

        try:
            v = float(getattr(self.ros, "last_cmd_linear_x", 0.0))
            w = float(getattr(self.ros, "last_cmd_angular_z", 0.0))
        except Exception:
            try:
                fa = np.asarray(getattr(self, "filtered_action", np.zeros(2, dtype=np.float32)), dtype=np.float32)
                v = float(fa[0])
                w = float(fa[1])
            except Exception:
                v = 0.0
                w = 0.0

        # Avoid accumulating numerical noise when the command is effectively zero.
        if abs(v) < 1e-4:
            v = 0.0
        if abs(w) < 1e-4:
            w = 0.0

        if dt > 0.0 and (v != 0.0 or w != 0.0):
            mid_yaw = self._normalize_angle(float(yaw) + 0.5 * float(w) * dt)
            xy = xy.copy()
            xy[0] += float(v) * dt * math.cos(mid_yaw)
            xy[1] += float(v) * dt * math.sin(mid_yaw)
            yaw = self._normalize_angle(float(yaw) + float(w) * dt)

        self._confidence_cmd_xy = xy.copy()
        self._confidence_cmd_yaw = float(yaw)
        return xy.copy(), float(yaw), "cmd_integrated"

    def _amcl_pose_callback(self, msg) -> None:
        self._latest_amcl_pose = msg

    def _get_amcl_bridge_anchored_odom_pose2d(self, target_frame: str = "map") -> Optional[tuple[np.ndarray, float]]:
        """Return an AMCL-compatible map pose from an anchored odometry delta.

        This is intentionally not particle-filter AMCL.  It exists for SLAM
        training runs where the user wants a /amcl_pose topic but the live
        TF(map->base_footprint) seen by this node is stale.  The pose starts from
        a verified map->base TF anchor and then follows actual/model odometry
        deltas, which is the confidence mode that previously tracked movement
        best in Gazebo.
        """
        try:
            motion_pose = self._get_confidence_motion_pose2d()
        except Exception:
            motion_pose = None
        if motion_pose is None:
            return None
        motion_xy, motion_yaw, motion_label = motion_pose
        motion_xy = np.asarray(motion_xy, dtype=np.float32).reshape(-1)[:2]
        motion_yaw = float(motion_yaw)
        if motion_xy.size < 2 or not (np.all(np.isfinite(motion_xy)) and math.isfinite(motion_yaw)):
            return None

        anchor = getattr(self, "_amcl_pose_odom_anchor", None)
        if anchor is None or str(anchor.get("motion_label", "")) != str(motion_label):
            try:
                hard_map_pose = self._get_map_base_pose2d_hard()
            except Exception:
                hard_map_pose = None
            if hard_map_pose is None:
                return None
            map_xy0, map_yaw0 = hard_map_pose
            map_xy0 = np.asarray(map_xy0, dtype=np.float32).reshape(-1)[:2]
            map_yaw0 = float(map_yaw0)
            if map_xy0.size < 2 or not (np.all(np.isfinite(map_xy0)) and math.isfinite(map_yaw0)):
                return None
            anchor = {
                "map_xy0": map_xy0.copy(),
                "map_yaw0": float(map_yaw0),
                "motion_xy0": motion_xy.copy(),
                "motion_yaw0": float(motion_yaw),
                "motion_label": str(motion_label),
            }
            self._amcl_pose_odom_anchor = anchor
            try:
                if str(os.environ.get("TB3_RL_CONFIDENCE_POSE_WARN", "0")).strip().lower() in {"1", "true", "yes", "on"}:
                    self.ros.get_logger().warn(
                        "AMCL_POSE_ANCHORED_ODOM_ANCHOR_SET | "
                        f"target={target_frame} source={motion_label} "
                        f"map=({float(map_xy0[0]):+.3f},{float(map_xy0[1]):+.3f},{math.degrees(map_yaw0):+.1f}deg) "
                        f"motion=({float(motion_xy[0]):+.3f},{float(motion_xy[1]):+.3f},{math.degrees(motion_yaw):+.1f}deg)"
                    )
            except Exception:
                pass

        theta = self._normalize_angle(float(anchor["map_yaw0"]) - float(anchor["motion_yaw0"]))
        c = math.cos(theta)
        ss = math.sin(theta)
        dx = float(motion_xy[0]) - float(anchor["motion_xy0"][0])
        dy = float(motion_xy[1]) - float(anchor["motion_xy0"][1])
        out_x = float(anchor["map_xy0"][0]) + c * dx - ss * dy
        out_y = float(anchor["map_xy0"][1]) + ss * dx + c * dy
        out_yaw = self._normalize_angle(float(motion_yaw) + theta)
        if not (math.isfinite(out_x) and math.isfinite(out_y) and math.isfinite(out_yaw)):
            return None
        return np.array([out_x, out_y], dtype=np.float32), float(out_yaw)

    def _publish_amcl_pose_from_tf_timer(self) -> None:
        """Publish an AMCL-compatible /amcl_pose from TF(map->base_footprint).

        This is not a particle-filter AMCL estimate.  It is a deterministic
        PoseWithCovarianceStamped bridge for SLAM training runs where the user
        wants confidence to consume /amcl_pose but the pose that is actually
        trustworthy is the same TF RViz uses to render RobotModel.
        """
        pub = getattr(self, "_amcl_pose_tf_pub", None)
        if pub is None:
            return
        target_frame = str(getattr(self, "amcl_pose_tf_target_frame", "map") or "map").strip().lstrip("/") or "map"
        source_frame = str(getattr(self, "amcl_pose_tf_source_frame", "base_footprint") or "base_footprint").strip().lstrip("/") or "base_footprint"
        pose = None
        bridge_source = str(getattr(self, "amcl_pose_bridge_source", "tf") or "tf").strip().lower()
        if bridge_source in {"anchored", "anchored_odom", "real_odom_anchored", "actual_odom", "model_odom"}:
            pose = self._get_amcl_bridge_anchored_odom_pose2d(target_frame=target_frame)
        else:
            try:
                if hasattr(self.ros, "get_frame_pose2d"):
                    pose = self.ros.get_frame_pose2d(
                        target_frame=target_frame,
                        source_frame=source_frame,
                        stamp=None,
                        timeout_sec=float(getattr(self, "amcl_pose_tf_timeout_sec", 0.05)),
                        allow_latest_fallback=True,
                    )
            except Exception:
                pose = None
        if pose is None:
            now = time.time()
            if now - float(getattr(self, "_last_amcl_pose_tf_warn_time", 0.0)) > 2.0:
                self._last_amcl_pose_tf_warn_time = now
                try:
                    self.ros.get_logger().warn(
                        "AMCL_POSE_TF_BRIDGE_WAITING | "
                        f"bridge_source={bridge_source} target={target_frame} source={source_frame} topic={getattr(self, 'amcl_pose_topic', '/amcl_pose')}"
                    )
                except Exception:
                    pass
            return
        try:
            # ros_interface.get_frame_pose2d() returns (xy: np.ndarray[2], yaw).
            # Older v102/v103 code incorrectly treated it as (x, y, yaw), so
            # float(pose[0]) tried to convert a 2-D numpy array and the AMCL
            # bridge never published /amcl_pose.
            xy = np.asarray(pose[0], dtype=np.float64).reshape(-1)
            if xy.size < 2:
                return
            x = float(xy[0])
            y = float(xy[1])
            yaw = float(pose[1])
            if not (math.isfinite(x) and math.isfinite(y) and math.isfinite(yaw)):
                return
            msg = PoseWithCovarianceStamped()
            try:
                msg.header.stamp = self.ros.get_clock().now().to_msg()
            except Exception:
                pass
            msg.header.frame_id = target_frame
            msg.pose.pose.position.x = x
            msg.pose.pose.position.y = y
            msg.pose.pose.position.z = 0.0
            qx, qy, qz, qw = self._yaw_to_quaternion_xyzw_static(yaw)
            msg.pose.pose.orientation.x = float(qx)
            msg.pose.pose.orientation.y = float(qy)
            msg.pose.pose.orientation.z = float(qz)
            msg.pose.pose.orientation.w = float(qw)
            cov = [0.0] * 36
            cov[0] = float(getattr(self, "amcl_pose_cov_xy", 0.0025))
            cov[7] = float(getattr(self, "amcl_pose_cov_xy", 0.0025))
            cov[35] = float(getattr(self, "amcl_pose_cov_yaw", 0.01))
            msg.pose.covariance = cov
            pub.publish(msg)
            # Keep the confidence source immediately usable even before the ROS
            # subscription callback loops this message back.
            self._latest_amcl_pose = msg
        except Exception as exc:
            now = time.time()
            if now - float(getattr(self, "_last_amcl_pose_tf_warn_time", 0.0)) > 2.0:
                self._last_amcl_pose_tf_warn_time = now
                try:
                    self.ros.get_logger().warn(f"AMCL_POSE_TF_BRIDGE_PUBLISH_FAILED | bridge_source={bridge_source} err={exc}")
                except Exception:
                    pass

    def _get_amcl_pose2d_map(self) -> Optional[tuple[np.ndarray, float, str]]:
        """Return /amcl_pose as an SE(2) pose in the current map frame."""
        msg = getattr(self, "_latest_amcl_pose", None)
        if msg is None:
            self._warn_amcl_pose_once("AMCL_POSE_UNAVAILABLE", "no /amcl_pose message received yet")
            return None

        try:
            frame = str(getattr(getattr(msg, "header", None), "frame_id", "") or "map").strip().lstrip("/") or "map"
            target = str(getattr(self, "map_frame", "map") or "map").strip().lstrip("/") or "map"
            if frame != target:
                self._warn_amcl_pose_once(
                    "AMCL_POSE_FRAME_MISMATCH",
                    f"header.frame_id={frame} expected={target}; skipping confidence update",
                )
                return None

            # Reject stale AMCL poses when simulation time is available.  AMCL
            # sometimes publishes with stamp=0 during startup; accept that.
            stamp = getattr(getattr(msg, "header", None), "stamp", None)
            if stamp is not None:
                t_msg = float(getattr(stamp, "sec", 0)) + 1e-9 * float(getattr(stamp, "nanosec", 0))
                if t_msg > 0.0 and float(getattr(self, "amcl_pose_max_age_sec", 0.0)) > 0.0:
                    now = self._confidence_time_sec()
                    if now > 0.0 and math.isfinite(float(now)):
                        age = float(now) - float(t_msg)
                        if age > float(self.amcl_pose_max_age_sec):
                            self._warn_amcl_pose_once(
                                "AMCL_POSE_STALE",
                                f"age={age:.2f}s max_age={self.amcl_pose_max_age_sec:.2f}s",
                            )
                            return None

            p = msg.pose.pose.position
            q = msg.pose.pose.orientation
            xy = np.array([float(p.x), float(p.y)], dtype=np.float32)
            yaw = self._yaw_from_quaternion_xyzw_static(float(q.x), float(q.y), float(q.z), float(q.w))
            if not (np.all(np.isfinite(xy)) and math.isfinite(float(yaw))):
                self._warn_amcl_pose_once("AMCL_POSE_INVALID", "non-finite xy/yaw")
                return None
            return xy[:2], float(yaw), "amcl_pose"
        except Exception as exc:
            self._warn_amcl_pose_once("AMCL_POSE_PARSE_FAILED", str(exc))
            return None

    def _warn_amcl_pose_once(self, tag: str, detail: str) -> None:
        try:
            dbg = str(os.environ.get("TB3_RL_CONFIDENCE_POSE_WARN", "0")).strip().lower() in {"1", "true", "yes", "on"}
        except Exception:
            dbg = False
        if not dbg:
            return
        now = time.time()
        if now - float(getattr(self, "_last_amcl_pose_warn_time", 0.0)) < 2.0:
            return
        self._last_amcl_pose_warn_time = now
        try:
            self.ros.get_logger().warn(f"{tag} | {detail}")
        except Exception:
            pass

    def _get_confidence_motion_pose2d(self) -> Optional[tuple[np.ndarray, float, str]]:
        """Return an actual robot pose source for confidence map alignment.

        v8 rule: do not use command integration.  The confidence layers are
        persistent map layers, so their robot anchor must come from a physical
        pose measurement:
          1) /model/<name>/odometry    (Gazebo model odom / simulated truth-ish)
          2) /odom message             (wheel/Gazebo odometry)
          3) odom TF                   (last-resort actual TF)

        The returned pose is not assumed to be in /map.  The caller anchors it
        once to the current map pose and applies SE(2) deltas so all published
        maps remain on the same /map OccupancyGrid canvas.
        """
        source = str(getattr(self, "confidence_motion_source", "model_odom") or "model_odom").strip().lower()
        if source in {"", "auto", "real", "actual", "real_odom", "actual_odom"}:
            order = ["model_odom", "odom_msg", "odom_tf"]
        elif source in {"model", "model_odom", "gazebo", "truth", "gazebo_model"}:
            order = ["model_odom", "odom_msg", "odom_tf"]
        elif source in {"model_strict", "model_odom_strict", "gazebo_strict", "truth_strict"}:
            order = ["model_odom"]
        elif source in {"odom", "odom_msg", "wheel_odom", "msg"}:
            order = ["odom_msg", "odom_tf", "model_odom"]
        elif source in {"odom_tf", "tf"}:
            order = ["odom_tf", "odom_msg", "model_odom"]
        elif source in {"cmd", "cmd_vel", "cmd_integrated", "integrated_cmd"}:
            # v7 used this as a fallback, but it breaks true map alignment.
            # Keep the alias only to prevent hard crashes from an old shell env.
            try:
                now = time.time()
                if now - float(getattr(self, "_last_confidence_pose_warn_time", 0.0)) > 2.0:
                    self._last_confidence_pose_warn_time = now
                    self.ros.get_logger().warn(
                        "CONFIDENCE_CMD_INTEGRATION_DISABLED | "
                        "cmd_integrated is not a valid aligned map pose source; "
                        "using actual odometry sources instead"
                    )
            except Exception:
                pass
            order = ["model_odom", "odom_msg", "odom_tf"]
        else:
            order = ["model_odom", "odom_msg", "odom_tf"]

        for item in order:
            if item == "model_odom":
                pose = self._pose2d_from_odometry_msg(getattr(self.ros, "model_odom", None), "model_odom")
                if pose is not None:
                    return pose
            elif item == "odom_msg":
                pose = self._pose2d_from_odometry_msg(getattr(self.ros, "odom", None), "odom_msg")
                if pose is not None:
                    return pose
            elif item == "odom_tf":
                try:
                    pose = self.ros.get_pose2d(frame_id="odom") if hasattr(self.ros, "get_pose2d") else None
                    if pose is not None:
                        xy, yaw = pose
                        xy = np.asarray(xy, dtype=np.float32)
                        if xy.size >= 2 and np.all(np.isfinite(xy[:2])) and math.isfinite(float(yaw)):
                            return xy[:2], float(yaw), "odom_tf"
                except Exception:
                    pass
        return None

    def _get_confidence_scan_stamped_tf_pose2d(self) -> Optional[tuple[np.ndarray, float, np.ndarray, float, str]]:
        """Return a scan-timestamped map pose for confidence painting.

        v107 rule:
          - Confidence is generated from LaserScan data, therefore the pose used
            to paint confidence must be the LiDAR pose at LaserScan.header.stamp.
          - Do not mix a scan from time t with the latest map->base TF from time
            t+dt.  That is what makes the magenta cone appear detached from the
            RobotModel/map while the robot is moving.
          - Ray origin/yaw: TF(map -> scan.header.frame_id) at scan.header.stamp.
          - Robot/crop pose: TF(map -> base_footprint) at the same stamp when
            available; otherwise use the scan pose as a conservative fallback
            unless TB3_RL_SCAN_STAMPED_TF_REQUIRE_BASE=1.
        """
        try:
            scan = getattr(self.ros, "scan", None)
            if scan is None:
                return None
            header = getattr(scan, "header", None)
            stamp = getattr(header, "stamp", None)
            scan_frame = str(getattr(header, "frame_id", "") or "").strip().lstrip("/")
            if not scan_frame:
                scan_frame = str(os.environ.get("TB3_RL_SCAN_FRAME", getattr(self, "scan_frame", "base_scan")) or "base_scan").strip().lstrip("/") or "base_scan"
            target_frame = str(os.environ.get("TB3_RL_CONFIDENCE_TARGET_FRAME", self.map_frame or "map") or "map").strip().lstrip("/") or "map"
            base_frame = str(os.environ.get("TB3_RL_CONFIDENCE_BASE_FRAME", os.environ.get("TB3_RL_BASE_FRAME", "base_footprint")) or "base_footprint").strip().lstrip("/") or "base_footprint"
            timeout_sec = float(os.environ.get("TB3_RL_SCAN_STAMPED_TF_TIMEOUT_SEC", "0.08") or 0.08)
            allow_latest = str(os.environ.get("TB3_RL_SCAN_STAMPED_TF_ALLOW_LATEST_FALLBACK", "0") or "0").strip().lower() in {"1", "true", "yes", "on"}
            require_base = str(os.environ.get("TB3_RL_SCAN_STAMPED_TF_REQUIRE_BASE", "0") or "0").strip().lower() in {"1", "true", "yes", "on"}
        except Exception:
            return None

        if not hasattr(self.ros, "get_frame_pose2d"):
            return None

        scan_pose = None
        try:
            scan_pose = self.ros.get_frame_pose2d(
                target_frame=target_frame,
                source_frame=scan_frame,
                stamp=stamp,
                timeout_sec=max(float(timeout_sec), 0.0),
                allow_latest_fallback=allow_latest,
            )
        except Exception:
            scan_pose = None

        if scan_pose is None:
            try:
                dbg = str(os.environ.get("TB3_RL_CONFIDENCE_POSE_WARN", "0")).strip().lower() in {"1", "true", "yes", "on"}
            except Exception:
                dbg = False
            if dbg:
                now = time.time()
                if now - float(getattr(self, "_last_confidence_pose_warn_time", 0.0)) > 2.0:
                    self._last_confidence_pose_warn_time = now
                    try:
                        self.ros.get_logger().warn(
                            "CONFIDENCE_SCAN_STAMPED_TF_UNAVAILABLE | "
                            f"target={target_frame} source={scan_frame} allow_latest={allow_latest}; "
                            "skip confidence update"
                        )
                    except Exception:
                        pass
            return None

        base_pose = None
        try:
            base_pose = self.ros.get_frame_pose2d(
                target_frame=target_frame,
                source_frame=base_frame,
                stamp=stamp,
                timeout_sec=max(float(timeout_sec), 0.0),
                allow_latest_fallback=allow_latest,
            )
        except Exception:
            base_pose = None

        if base_pose is None and require_base:
            try:
                dbg = str(os.environ.get("TB3_RL_CONFIDENCE_POSE_WARN", "0")).strip().lower() in {"1", "true", "yes", "on"}
            except Exception:
                dbg = False
            if dbg:
                now = time.time()
                if now - float(getattr(self, "_last_confidence_pose_warn_time", 0.0)) > 2.0:
                    self._last_confidence_pose_warn_time = now
                    try:
                        self.ros.get_logger().warn(
                            "CONFIDENCE_BASE_STAMPED_TF_UNAVAILABLE | "
                            f"target={target_frame} source={base_frame} allow_latest={allow_latest}; "
                            "skip confidence update"
                        )
                    except Exception:
                        pass
            return None

        try:
            scan_xy = np.asarray(scan_pose[0], dtype=np.float32).reshape(-1)[:2]
            scan_yaw = float(scan_pose[1])
            if scan_xy.size < 2 or not np.all(np.isfinite(scan_xy)) or not math.isfinite(scan_yaw):
                return None
            if base_pose is not None:
                robot_xy = np.asarray(base_pose[0], dtype=np.float32).reshape(-1)[:2]
                robot_yaw = float(base_pose[1])
                if robot_xy.size < 2 or not np.all(np.isfinite(robot_xy)) or not math.isfinite(robot_yaw):
                    robot_xy = scan_xy.copy()
                    robot_yaw = scan_yaw
                    label = f"scan_stamped_tf:{scan_frame}:base_fallback"
                else:
                    label = f"scan_stamped_tf:{scan_frame}:{base_frame}"
            else:
                robot_xy = scan_xy.copy()
                robot_yaw = scan_yaw
                label = f"scan_stamped_tf:{scan_frame}:scan_as_robot"
            self._last_confidence_scan_stamped_frame = scan_frame
            self._last_confidence_scan_stamped_xy = scan_xy.copy()
            self._last_confidence_scan_stamped_yaw = float(scan_yaw)
            return robot_xy.astype(np.float32), float(robot_yaw), scan_xy.astype(np.float32), float(scan_yaw), label
        except Exception:
            return None

    def _get_confidence_update_pose2d(self) -> Optional[tuple[np.ndarray, float, np.ndarray, float, str]]:
        """Return the pose used to paint camera-front confidence in /map.

        v11 rule:
          - All persistent maps stay on the same raw /map OccupancyGrid canvas.
          - The initial anchor MUST be a verified TF(map->base) pose.
          - Do not paint confidence/priority at a guessed SLAM origin.  That
            created ghost confidence blobs away from the robot when Cartographer
            had not produced map->base TF yet.
          - After the anchor is created, actual odometry deltas move the pose.
          - No cmd_vel integration is used.
        """
        mode = str(os.environ.get("TB3_RL_CONFIDENCE_POSE_SOURCE", getattr(self, "confidence_pose_source", "real_odom_anchored")) or "real_odom_anchored").strip().lower()
        self.confidence_pose_source = mode

        # ── Hard unification with the TF compare CUBE / RViz RobotModel ──────────
        # The user wants ONE origin: the confidence cone must start at exactly the
        # pose marked by the TF compare cube (live TF map->base_footprint), which
        # is what RViz uses to place the robot.  When enabled (default), bypass all
        # anchored/odom estimators and return that TF pose directly, with a short
        # hold across brief TF gaps so confidence does not blink.  The marker code
        # reuses self._confidence_unified_xy/_yaw so the red sphere is drawn from
        # the SAME value as the cube -> they are guaranteed identical, not merely
        # "close".  Disable with TB3_RL_CONFIDENCE_UNIFY_WITH_TF_CUBE=0.
        try:
            unify_cube = str(os.environ.get("TB3_RL_CONFIDENCE_UNIFY_WITH_TF_CUBE", "1") or "1").strip().lower() in {"1", "true", "yes", "on"}
        except Exception:
            unify_cube = True
        if unify_cube:
            cube_pose = self._get_tf_cube_confidence_pose2d()
            if cube_pose is not None:
                cxy, cyaw = cube_pose
                raw_scan_yaw = self._get_map_scan_yaw_hard(raw_scan_msg=self.ros.scan, base_yaw=float(cyaw))
                scan_yaw = self._select_confidence_ray_yaw(base_yaw=float(cyaw), scan_yaw=raw_scan_yaw, mode_hint="tf_cube_unified")
                # Persist so the origin marker draws the sphere at the cube pose.
                self._confidence_unified_xy = np.asarray(cxy, dtype=np.float32).copy()
                self._confidence_unified_yaw = float(cyaw)
                self.confidence_pose_source = "tf_cube_unified"
                return (
                    np.asarray(cxy, dtype=np.float32),
                    float(cyaw),
                    np.asarray(cxy, dtype=np.float32),
                    float(scan_yaw),
                    "tf_cube_unified",
                )
            # If the CUBE pose is unavailable, do NOT fall through to the legacy
            # odom/anchor estimators by default.  The whole point of v125 is that
            # confidence must use the same pose as the verified green CUBE, or skip
            # the semantic write for this tick.  This prevents hidden source swaps.
            if self._scan_bool_env("TB3_RL_CONFIDENCE_UNIFY_STRICT", True):
                self._confidence_unified_xy = None
                self._confidence_unified_yaw = None
                return None
            # Legacy escape hatch only when explicitly disabled.
        # ────────────────────────────────────────────────────────────────────────

        anchor = getattr(self, "_confidence_odom_anchor", None)
        hard_map_pose = None

        # Direct TF robot-pose mode.  This is the intended map-aligned mode for
        # confidence: OccupancyGrid frame stays /map, but the cone origin is read
        # from TF instead of odometry integration.
        #
        # Important v96 split:
        #   - map_base_tf:  robot/base origin + scan-frame yaw
        #   - map_scan_tf:  robot/base pose for visit/crop, but ray origin+yaw
        #                   from TF(map -> base_scan).  This is the correct mode
        #                   when the confidence cone must coincide with the
        #                   LaserScan geometry seen in RViz.
        direct_tf_modes = {
            "map", "map_tf", "tf", "cartographer",
            "map_base_tf", "base_tf", "robot_tf", "tf_base",
        }
        # v100: exact RViz robot pose mode.  This mode deliberately bypasses
        # the odom/model anchored estimators and also ignores
        # TB3_RL_CONFIDENCE_TARGET_FRAME.  It samples the latest TF exactly as
        # RViz places RobotModel in Fixed Frame=map:
        #
        #     target_frame = map
        #     source_frame = base_footprint
        #
        # Use this when the confidence cone must originate from the robot that
        # is visibly drawn in RViz, not from model odometry or an episode anchor.
        rviz_tf_modes = {
            "rviz_tf", "rviz_base_tf", "rviz_map_tf",
            "map_rviz_tf", "actual_tf", "robot_model_tf",
        }
        scan_tf_modes = {
            "map_scan_tf", "scan_tf", "base_scan_tf", "tf_scan",
            "lidar_tf", "laser_tf", "map_lidar_tf", "map_laser_tf",
        }
        # v107: timestamp-synchronized scan pose mode.  This is the most
        # consistent confidence-painting source for Cartographer: the pose is
        # looked up at LaserScan.header.stamp, not at the latest TF time.
        # The ray origin/yaw use TF(map -> scan.header.frame_id) at the scan
        # stamp, so the confidence cone is painted from the sensor pose that
        # actually produced the scan.
        scan_stamped_tf_modes = {
            "scan_stamped_tf", "stamped_scan_tf", "scan_stamp_tf",
            "map_scan_stamped_tf", "map_base_scan_stamped_tf",
            "lidar_stamped_tf", "laser_stamped_tf",
        }
        amcl_modes = {
            "amcl", "amcl_pose", "map_amcl", "amcl_map",
            "amcl_pose_topic", "nav2_amcl",
        }

        # In the normal anchored mode, map->base TF is only needed to create the
        # anchor.  In direct TF mode, it is read on every update.
        if mode in scan_stamped_tf_modes:
            # v107 uses timestamped scan/base TF directly and should not prime a
            # latest map->base anchor.
            hard_map_pose = None
        elif mode in direct_tf_modes or mode in scan_tf_modes or mode in rviz_tf_modes or anchor is None:
            hard_map_pose = self._get_map_base_pose2d_hard()

        if mode in amcl_modes:
            amcl_pose = self._get_amcl_pose2d_map()
            if amcl_pose is not None:
                xy, yaw, label = amcl_pose
                raw_scan_yaw = self._get_map_scan_yaw_hard(raw_scan_msg=self.ros.scan, base_yaw=float(yaw))
                scan_yaw = self._select_confidence_ray_yaw(base_yaw=float(yaw), scan_yaw=raw_scan_yaw, mode_hint="amcl_pose")
                try:
                    dbg_n = int(os.environ.get("TB3_RL_CONFIDENCE_POSE_DEBUG_EVERY_N", "0"))
                except Exception:
                    dbg_n = 0
                step = int(getattr(getattr(self, "exploration_map", None), "step_index", 0))
                if dbg_n > 0 and (step <= 3 or step - int(getattr(self, "_last_confidence_pose_log_step", -10_000_000)) >= dbg_n):
                    self._last_confidence_pose_log_step = step
                    try:
                        self.ros.get_logger().warn(
                            "CONFIDENCE_POSE | "
                            f"mode=amcl_pose topic={getattr(self, 'amcl_pose_topic', '/amcl_pose')} step={step} "
                            f"map_xy=({float(xy[0]):+.3f},{float(xy[1]):+.3f}) "
                            f"yaw={math.degrees(float(yaw)):+.1f}deg "
                            f"scan_yaw={math.degrees(float(scan_yaw)):+.1f}deg"
                        )
                    except Exception:
                        pass
                return np.asarray(xy, dtype=np.float32), float(yaw), np.asarray(xy, dtype=np.float32), float(scan_yaw), "amcl_pose"

            fallback = str(getattr(self, "amcl_fallback_pose_source", "none") or "none").strip().lower()
            if fallback in {"", "none", "off", "disable", "disabled", "no"} or fallback in amcl_modes:
                return None
            mode = fallback
            self.confidence_pose_source = mode
            if mode in direct_tf_modes or mode in scan_tf_modes or mode in rviz_tf_modes or anchor is None:
                hard_map_pose = self._get_map_base_pose2d_hard()

        if mode in scan_stamped_tf_modes:
            pose_pack = self._get_confidence_scan_stamped_tf_pose2d()
            if pose_pack is None:
                return None
            robot_xy, robot_yaw, sensor_xy, sensor_yaw, label = pose_pack
            try:
                dbg_n = int(os.environ.get("TB3_RL_CONFIDENCE_POSE_DEBUG_EVERY_N", "0"))
            except Exception:
                dbg_n = 0
            step = int(getattr(getattr(self, "exploration_map", None), "step_index", 0))
            if dbg_n > 0 and (step <= 3 or step - int(getattr(self, "_last_confidence_pose_log_step", -10_000_000)) >= dbg_n):
                self._last_confidence_pose_log_step = step
                try:
                    self.ros.get_logger().warn(
                        "CONFIDENCE_POSE | "
                        f"mode=scan_stamped_tf label={label} step={step} "
                        f"robot_xy=({float(robot_xy[0]):+.3f},{float(robot_xy[1]):+.3f}) "
                        f"sensor_xy=({float(sensor_xy[0]):+.3f},{float(sensor_xy[1]):+.3f}) "
                        f"robot_yaw={math.degrees(float(robot_yaw)):+.1f}deg "
                        f"sensor_yaw={math.degrees(float(sensor_yaw)):+.1f}deg"
                    )
                except Exception:
                    pass
            return robot_xy, float(robot_yaw), sensor_xy, float(sensor_yaw), label

        if mode in rviz_tf_modes:
            # Hard-code the target/source pair to the same frames the user sees
            # in RViz.  This prevents an old confidence target-frame env var or
            # fallback source-frame ordering from silently selecting odom,
            # base_link, or base_scan.
            target_frame = "map"
            try:
                source_frame = str(os.environ.get("TB3_RL_RVIZ_ROBOT_FRAME", "base_footprint") or "base_footprint").strip().lstrip("/")
            except Exception:
                source_frame = "base_footprint"
            pose = None
            try:
                if hasattr(self.ros, "get_frame_pose2d"):
                    pose = self.ros.get_frame_pose2d(
                        target_frame=target_frame,
                        source_frame=source_frame,
                        stamp=None,
                        timeout_sec=0.10,
                        allow_latest_fallback=True,
                    )
            except Exception:
                pose = None
            if pose is None:
                try:
                    dbg = str(os.environ.get("TB3_RL_CONFIDENCE_POSE_WARN", "0")).strip().lower() in {"1", "true", "yes", "on"}
                except Exception:
                    dbg = False
                if dbg:
                    now = time.time()
                    if now - float(getattr(self, "_last_confidence_pose_warn_time", 0.0)) > 2.0:
                        self._last_confidence_pose_warn_time = now
                        try:
                            self.ros.get_logger().warn(
                                "CONFIDENCE_RVIZ_TF_UNAVAILABLE | "
                                f"target={target_frame} source={source_frame}; skip confidence update"
                            )
                        except Exception:
                            pass
                return None
            xy, yaw = pose
            xy = np.asarray(xy, dtype=np.float32)[:2]
            try:
                raw_scan_yaw = self._get_map_scan_yaw_hard(raw_scan_msg=self.ros.scan, base_yaw=float(yaw))
                scan_yaw = self._select_confidence_ray_yaw(base_yaw=float(yaw), scan_yaw=raw_scan_yaw, mode_hint="amcl_pose")
            except Exception:
                scan_yaw = float(yaw)
            try:
                dbg_n = int(os.environ.get("TB3_RL_CONFIDENCE_POSE_DEBUG_EVERY_N", "0"))
            except Exception:
                dbg_n = 0
            step = int(getattr(getattr(self, "exploration_map", None), "step_index", 0))
            if dbg_n > 0 and (step <= 3 or step - int(getattr(self, "_last_confidence_pose_log_step", -10_000_000)) >= dbg_n):
                self._last_confidence_pose_log_step = step
                try:
                    self.ros.get_logger().warn(
                        "CONFIDENCE_POSE | "
                        f"mode=rviz_base_tf source={source_frame} target={target_frame} step={step} "
                        f"map_xy=({float(xy[0]):+.3f},{float(xy[1]):+.3f}) "
                        f"yaw={math.degrees(float(yaw)):+.1f}deg "
                        f"scan_yaw={math.degrees(float(scan_yaw)):+.1f}deg"
                    )
                except Exception:
                    pass
            return xy, float(yaw), xy.copy(), float(scan_yaw), "rviz_base_tf"

        if mode in scan_tf_modes:
            if hard_map_pose is None:
                return None
            scan_pose = self._get_scan_pose2d_for_map_update()
            if scan_pose is None:
                try:
                    dbg = str(os.environ.get("TB3_RL_CONFIDENCE_POSE_WARN", "0")).strip().lower() in {"1", "true", "yes", "on"}
                except Exception:
                    dbg = False
                if dbg:
                    now = time.time()
                    if now - float(getattr(self, "_last_confidence_pose_warn_time", 0.0)) > 2.0:
                        self._last_confidence_pose_warn_time = now
                        try:
                            self.ros.get_logger().warn(
                                "CONFIDENCE_SCAN_TF_UNAVAILABLE | "
                                "mode=map_scan_tf; skip confidence update until TF(map->base_scan) is valid"
                            )
                        except Exception:
                            pass
                return None
            base_xy, base_yaw = hard_map_pose
            scan_xy, scan_yaw = scan_pose
            return (
                np.asarray(base_xy, dtype=np.float32),
                float(base_yaw),
                np.asarray(scan_xy, dtype=np.float32),
                float(scan_yaw),
                "map_scan_tf",
            )

        if mode in direct_tf_modes:
            # Unify confidence origin with the TF compare cube / RViz RobotModel:
            # position AND yaw come straight from TF(map->base_footprint) every
            # step, with no odometry integration and no anchor.  The confidence
            # cone therefore expands from exactly the pose the cube marks.
            if hard_map_pose is not None:
                xy, yaw = hard_map_pose
                # Cache the last valid TF pose so a brief TF gap (e.g. paused
                # Gazebo multi_step) does not skip the confidence update.  This is
                # a short HOLD of the last true pose, NOT odometry integration, so
                # it never drifts: at worst it repaints the same spot for a few ms
                # until live TF returns.
                try:
                    self._last_direct_tf_pose_xy = np.asarray(xy, dtype=np.float32).reshape(-1)[:2].copy()
                    self._last_direct_tf_pose_yaw = float(yaw)
                    self._last_direct_tf_pose_wall = time.time()
                except Exception:
                    pass
            else:
                # No live TF this step: hold the last valid TF pose for a short
                # window instead of returning None (which would skip the update).
                try:
                    hold_sec = float(os.environ.get("TB3_RL_CONFIDENCE_TF_HOLD_SEC", "0.5") or 0.5)
                except Exception:
                    hold_sec = 0.5
                held_xy = getattr(self, "_last_direct_tf_pose_xy", None)
                held_yaw = getattr(self, "_last_direct_tf_pose_yaw", None)
                held_wall = float(getattr(self, "_last_direct_tf_pose_wall", 0.0))
                if (
                    isinstance(held_xy, np.ndarray)
                    and held_xy.size >= 2
                    and held_yaw is not None
                    and (time.time() - held_wall) <= max(hold_sec, 0.0)
                ):
                    xy = held_xy.copy()
                    yaw = float(held_yaw)
                else:
                    # TF has been missing too long; skip rather than paint stale.
                    return None
            raw_scan_yaw = self._get_map_scan_yaw_hard(raw_scan_msg=self.ros.scan, base_yaw=yaw)
            scan_yaw = self._select_confidence_ray_yaw(base_yaw=float(yaw), scan_yaw=raw_scan_yaw, mode_hint="map_base_tf")
            # sensor_xy deliberately equals base xy in this mode.  Use
            # TB3_RL_CONFIDENCE_POSE_SOURCE=map_scan_tf if the ray origin must be
            # the LaserScan frame itself.
            return np.asarray(xy, dtype=np.float32), float(yaw), np.asarray(xy, dtype=np.float32), float(scan_yaw), "map_base_tf"

        motion_pose = self._get_confidence_motion_pose2d()
        if motion_pose is None:
            # Without actual odometry we cannot maintain true map alignment.  If
            # TF exists, use it directly for this update; otherwise skip.
            if hard_map_pose is None:
                try:
                    dbg = str(os.environ.get("TB3_RL_CONFIDENCE_POSE_WARN", "0")).strip().lower() in {"1", "true", "yes", "on"}
                except Exception:
                    dbg = False
                if dbg:
                    now = time.time()
                    if now - float(getattr(self, "_last_confidence_pose_warn_time", 0.0)) > 2.0:
                        self._last_confidence_pose_warn_time = now
                        try:
                            self.ros.get_logger().warn(
                                "CONFIDENCE_POSE_SOURCE_UNAVAILABLE | "
                                f"mode={mode} motion_source={getattr(self, 'confidence_motion_source', '')}"
                            )
                        except Exception:
                            pass
                return None
            xy, yaw = hard_map_pose
            raw_scan_yaw = self._get_map_scan_yaw_hard(raw_scan_msg=self.ros.scan, base_yaw=yaw)
            scan_yaw = self._select_confidence_ray_yaw(base_yaw=float(yaw), scan_yaw=raw_scan_yaw, mode_hint="map_tf_fallback")
            return np.asarray(xy, dtype=np.float32), float(yaw), np.asarray(xy, dtype=np.float32), float(scan_yaw), "map_tf_fallback"

        motion_xy, motion_yaw, motion_label = motion_pose
        motion_xy = np.asarray(motion_xy, dtype=np.float32)[:2]
        motion_yaw = float(motion_yaw)

        anchor = getattr(self, "_confidence_odom_anchor", None)
        if anchor is None or str(anchor.get("motion_label", "")) != str(motion_label):
            # v11: do not create a new anchor from a guessed map origin.
            # A persistent map layer can only be considered aligned if the first
            # anchor came from the same map->base TF used by RViz.  If TF is not
            # available yet, skip this update and wait; otherwise confidence can
            # be painted at (0,0) or another stale location with no robot there.
            if hard_map_pose is None:
                try:
                    dbg = str(os.environ.get("TB3_RL_CONFIDENCE_ANCHOR_DEBUG", "0")).strip().lower() in {"1", "true", "yes", "on"}
                except Exception:
                    dbg = False
                if dbg:
                    now = time.time()
                    if now - float(getattr(self, "_last_confidence_pose_warn_time", 0.0)) > 2.0:
                        self._last_confidence_pose_warn_time = now
                        try:
                            self.ros.get_logger().warn(
                                "CONFIDENCE_ANCHOR_WAIT_MAP_TF | "
                                f"mode={mode} motion_source={motion_label}; skip map write until map->base TF is valid"
                            )
                        except Exception:
                            pass
                return None
            anchor_source = "map_tf"
            map_xy0, map_yaw0 = hard_map_pose

            map_xy0 = np.asarray(map_xy0, dtype=np.float32)[:2]
            map_yaw0 = float(map_yaw0)
            try:
                if hard_map_pose is not None:
                    scan_yaw0 = self._get_map_scan_yaw_hard(raw_scan_msg=self.ros.scan, base_yaw=map_yaw0)
                    scan_offset = self._normalize_angle(float(scan_yaw0) - float(map_yaw0))
                else:
                    scan_offset = 0.0
            except Exception:
                scan_offset = 0.0
            anchor = {
                "map_xy0": map_xy0.copy(),
                "map_yaw0": float(map_yaw0),
                "motion_xy0": motion_xy.copy(),
                "motion_yaw0": float(motion_yaw),
                "motion_label": str(motion_label),
                "scan_offset": float(scan_offset),
                "anchor_source": str(anchor_source),
            }
            self._confidence_odom_anchor = anchor
            try:
                _anchor_dbg_n = int(os.environ.get("TB3_RL_CONFIDENCE_POSE_DEBUG_EVERY_N", "0"))
            except Exception:
                _anchor_dbg_n = 0
            if _anchor_dbg_n > 0:
                try:
                    self.ros.get_logger().warn(
                        "CONFIDENCE_POSE_ANCHOR_SET | "
                        f"mode=real_odom_anchored source={motion_label} anchor={anchor_source} "
                        f"map=({map_xy0[0]:+.3f},{map_xy0[1]:+.3f},{math.degrees(map_yaw0):+.1f}deg) "
                        f"motion=({motion_xy[0]:+.3f},{motion_xy[1]:+.3f},{math.degrees(motion_yaw):+.1f}deg) "
                        f"scan_offset={math.degrees(scan_offset):+.1f}deg"
                    )
                except Exception:
                    pass

        theta = self._normalize_angle(float(anchor["map_yaw0"]) - float(anchor["motion_yaw0"]))
        c = math.cos(theta)
        s = math.sin(theta)
        dx = float(motion_xy[0]) - float(anchor["motion_xy0"][0])
        dy = float(motion_xy[1]) - float(anchor["motion_xy0"][1])
        out_x = float(anchor["map_xy0"][0]) + c * dx - s * dy
        out_y = float(anchor["map_xy0"][1]) + s * dx + c * dy
        out_yaw = self._normalize_angle(float(motion_yaw) + theta)
        scan_yaw = self._normalize_angle(out_yaw + float(anchor.get("scan_offset", 0.0)))
        out_xy = np.array([out_x, out_y], dtype=np.float32)

        # v122 anchor re-sync (fixes the confidence-origin "jump"):
        #
        # The anchor map_xy0 is captured once from TF(map->base).  After that the
        # pose is propagated purely by integrating odometry deltas.  But every
        # time Cartographer runs a pose-graph correction / loop closure, the live
        # map->base TF jumps, while the frozen anchor does not.  The integrated
        # confidence origin (red sphere) then drifts away from the actual robot
        # (the TF compare cube), which is exactly the reported symptom: the cube
        # is correct, the sphere is offset.
        #
        # Fix: whenever a fresh, valid TF(map->base) is available, snap the anchor
        # back onto it.  This keeps the odometry-integration benefit during brief
        # TF gaps (paused Gazebo multi_step), but removes any accumulated drift
        # the instant TF returns, so the sphere tracks the cube.  Disable with
        # TB3_RL_CONFIDENCE_ANCHOR_RESYNC=0.
        try:
            resync_enabled = str(os.environ.get("TB3_RL_CONFIDENCE_ANCHOR_RESYNC", "1") or "1").strip().lower() in {"1", "true", "yes", "on"}
        except Exception:
            resync_enabled = True
        if resync_enabled and hard_map_pose is None:
            # hard_map_pose was only fetched when the anchor was (re)created above;
            # in steady state we must read the live TF here to compare/resync.
            try:
                hard_map_pose = self._get_map_base_pose2d_hard()
            except Exception:
                hard_map_pose = None
        if resync_enabled and hard_map_pose is not None:
            try:
                tf_xy, tf_yaw = hard_map_pose
                tf_xy = np.asarray(tf_xy, dtype=np.float32).reshape(-1)[:2]
                tf_yaw = float(tf_yaw)
                if tf_xy.size >= 2 and np.all(np.isfinite(tf_xy)) and math.isfinite(tf_yaw):
                    # Distance between the integrated origin and the live TF pose.
                    drift = math.hypot(float(out_x) - float(tf_xy[0]), float(out_y) - float(tf_xy[1]))
                    try:
                        resync_tol = float(os.environ.get("TB3_RL_CONFIDENCE_ANCHOR_RESYNC_TOL_M", "0.03") or 0.03)
                    except Exception:
                        resync_tol = 0.03
                    if drift > max(resync_tol, 0.0):
                        # Re-anchor onto the live TF and zero the integration base
                        # so subsequent odom deltas accumulate from here.
                        try:
                            scan_yaw0 = self._get_map_scan_yaw_hard(raw_scan_msg=self.ros.scan, base_yaw=tf_yaw)
                            new_scan_offset = self._normalize_angle(float(scan_yaw0) - float(tf_yaw))
                        except Exception:
                            new_scan_offset = float(anchor.get("scan_offset", 0.0))
                        anchor["map_xy0"] = tf_xy.astype(np.float32).copy()
                        anchor["map_yaw0"] = float(tf_yaw)
                        anchor["motion_xy0"] = motion_xy.copy()
                        anchor["motion_yaw0"] = float(motion_yaw)
                        anchor["scan_offset"] = float(new_scan_offset)
                        anchor["anchor_source"] = "map_tf_resync"
                        self._confidence_odom_anchor = anchor
                        # Recompute the output pose from the freshly synced anchor.
                        out_x = float(tf_xy[0])
                        out_y = float(tf_xy[1])
                        out_yaw = float(tf_yaw)
                        scan_yaw = self._normalize_angle(out_yaw + float(new_scan_offset))
                        out_xy = np.array([out_x, out_y], dtype=np.float32)
            except Exception:
                pass

        try:
            dbg_n = int(os.environ.get("TB3_RL_CONFIDENCE_POSE_DEBUG_EVERY_N", "0"))
        except Exception:
            dbg_n = 0
        step = int(getattr(getattr(self, "exploration_map", None), "step_index", 0))
        if dbg_n > 0 and (step <= 3 or step - int(getattr(self, "_last_confidence_pose_log_step", -10_000_000)) >= dbg_n):
            self._last_confidence_pose_log_step = step
            try:
                self.ros.get_logger().info(
                    "CONFIDENCE_POSE | "
                    f"mode=real_odom_anchored source={motion_label} anchor={anchor.get('anchor_source', 'unknown')} step={step} "
                    f"map_xy=({out_x:+.3f},{out_y:+.3f}) yaw={math.degrees(out_yaw):+.1f}deg "
                    f"motion_delta=({dx:+.3f},{dy:+.3f}) "
                    f"theta={math.degrees(theta):+.1f}deg"
                )
            except Exception:
                pass

        return out_xy, float(out_yaw), out_xy.copy(), float(scan_yaw), f"real_odom_anchored:{motion_label}:{anchor.get('anchor_source', 'unknown')}"

    def _publish_confidence_origin_marker(
        self,
        *,
        robot_xy: np.ndarray,
        robot_yaw: float,
        sensor_xy: np.ndarray,
        sensor_yaw: float,
        label: str,
    ) -> None:
        """Publish the exact map-frame pose used by confidence update.

        This marker is diagnostic only.  It must be compared against RViz
        RobotModel with Fixed Frame=map:
          - If the marker is not at the RobotModel, the pose source is wrong.
          - If the marker is at the RobotModel but /rl_confidence_map is offset,
            the OccupancyGrid origin/canvas conversion is wrong.
        """
        pub = getattr(self, "confidence_origin_marker_pub", None)
        if pub is None:
            return
        try:
            step = int(getattr(self, "step_count", 0))
            every_n = max(int(getattr(self, "confidence_origin_marker_publish_every_n", 1)), 1)
            if step > 3 and (step % every_n) != 0:
                return
            self._last_confidence_origin_marker_step = step
            frame_id = str(os.environ.get("TB3_RL_CONFIDENCE_TARGET_FRAME", self.map_frame or "map") or "map").strip().lstrip("/") or "map"
            stamp = self.ros.get_clock().now().to_msg()
            rxy = np.asarray(robot_xy, dtype=np.float32).reshape(-1)[:2]
            sxy = np.asarray(sensor_xy, dtype=np.float32).reshape(-1)[:2]
            ryaw = float(robot_yaw)
            syaw = float(sensor_yaw)

            # When confidence is unified with the TF compare cube, draw the origin
            # markers from the EXACT same value the confidence update used (which is
            # the live TF map->base pose).  This guarantees the red sphere sits on
            # the cube/RobotModel instead of merely near it.
            unified_xy = getattr(self, "_confidence_unified_xy", None)
            unified_yaw = getattr(self, "_confidence_unified_yaw", None)
            if (
                isinstance(unified_xy, np.ndarray)
                and unified_xy.size >= 2
                and np.all(np.isfinite(unified_xy[:2]))
                and unified_yaw is not None
                and math.isfinite(float(unified_yaw))
            ):
                rxy = np.asarray(unified_xy, dtype=np.float32).reshape(-1)[:2]
                ryaw = float(unified_yaw)
                sxy = rxy.copy()

            if rxy.size < 2 or sxy.size < 2:
                return
            if not (np.all(np.isfinite(rxy)) and np.all(np.isfinite(sxy)) and math.isfinite(ryaw) and math.isfinite(syaw)):
                return

            unified_now = bool(
                isinstance(getattr(self, "_confidence_unified_xy", None), np.ndarray)
                and getattr(self, "_confidence_unified_xy").size >= 2
            )
            single_cube = self._scan_bool_env("TB3_RL_CONFIDENCE_SINGLE_CUBE_MARKER", True)
            if unified_now and single_cube:
                arr = MarkerArray()

                # Clear old red/cyan spheres/arrows/text and old compare markers,
                # then publish exactly one marker: the verified CUBE pose that the
                # confidence update actually used.  Do not perform another TF lookup
                # here; repeated lookups were the source of sphere/cube divergence
                # and can add latency in long runs.
                clear = Marker()
                clear.header.frame_id = frame_id
                clear.header.stamp = stamp
                clear.action = Marker.DELETEALL
                arr.markers.append(clear)

                m = Marker()
                m.header.frame_id = frame_id
                m.header.stamp = stamp
                m.ns = "rl_confidence_tf_compare"
                m.id = 20
                m.type = Marker.CUBE
                m.action = Marker.ADD
                m.pose.position.x = float(rxy[0])
                m.pose.position.y = float(rxy[1])
                m.pose.position.z = 0.62
                m.pose.orientation.w = 1.0
                m.scale.x = 0.26
                m.scale.y = 0.26
                m.scale.z = 0.26
                self._set_marker_color(m, 0.0, 1.0, 0.0, 0.98)
                arr.markers.append(m)

                pub.publish(arr)
                return

            arr = MarkerArray()

            clear = Marker()
            clear.header.frame_id = frame_id
            clear.header.stamp = stamp
            clear.ns = "rl_confidence_origin"
            clear.action = Marker.DELETEALL
            arr.markers.append(clear)

            # Robot/base confidence origin: red sphere.
            base = Marker()
            base.header.frame_id = frame_id
            base.header.stamp = stamp
            base.ns = "rl_confidence_origin"
            base.id = 0
            base.type = Marker.SPHERE
            base.action = Marker.ADD
            base.pose.position.x = float(rxy[0])
            base.pose.position.y = float(rxy[1])
            base.pose.position.z = 0.28
            base.pose.orientation.w = 1.0
            base.scale.x = 0.22
            base.scale.y = 0.22
            base.scale.z = 0.22
            self._set_marker_color(base, 1.0, 0.05, 0.05, 0.95)
            arr.markers.append(base)

            # Sensor/raycast origin: cyan sphere.  In base-origin mode this will
            # overlap the red sphere; in scan-frame mode it shows the LiDAR offset.
            sensor = Marker()
            sensor.header.frame_id = frame_id
            sensor.header.stamp = stamp
            sensor.ns = "rl_confidence_origin"
            sensor.id = 1
            sensor.type = Marker.SPHERE
            sensor.action = Marker.ADD
            sensor.pose.position.x = float(sxy[0])
            sensor.pose.position.y = float(sxy[1])
            sensor.pose.position.z = 0.34
            sensor.pose.orientation.w = 1.0
            sensor.scale.x = 0.16
            sensor.scale.y = 0.16
            sensor.scale.z = 0.16
            self._set_marker_color(sensor, 0.0, 0.95, 1.0, 0.95)
            arr.markers.append(sensor)

            # Sensor forward arrow: same yaw that confidence raycast uses.
            arrow = Marker()
            arrow.header.frame_id = frame_id
            arrow.header.stamp = stamp
            arrow.ns = "rl_confidence_origin"
            arrow.id = 2
            arrow.type = Marker.ARROW
            arrow.action = Marker.ADD
            arrow.pose.orientation.w = 1.0
            sx = float(sxy[0])
            sy = float(sxy[1])
            ex = sx + 0.85 * math.cos(syaw)
            ey = sy + 0.85 * math.sin(syaw)
            arrow.points = [self._point_xyz(sx, sy, 0.42), self._point_xyz(ex, ey, 0.42)]
            arrow.scale.x = 0.055
            arrow.scale.y = 0.16
            arrow.scale.z = 0.22
            self._set_marker_color(arrow, 1.0, 0.85, 0.0, 0.98)
            arr.markers.append(arrow)

            # Robot/body yaw arrow: blue.  If this differs from the yellow arrow,
            # confidence is using a scan yaw different from base yaw.
            body = Marker()
            body.header.frame_id = frame_id
            body.header.stamp = stamp
            body.ns = "rl_confidence_origin"
            body.id = 3
            body.type = Marker.ARROW
            body.action = Marker.ADD
            body.pose.orientation.w = 1.0
            bx = float(rxy[0])
            by = float(rxy[1])
            bex = bx + 0.60 * math.cos(ryaw)
            bey = by + 0.60 * math.sin(ryaw)
            body.points = [self._point_xyz(bx, by, 0.50), self._point_xyz(bex, bey, 0.50)]
            body.scale.x = 0.040
            body.scale.y = 0.13
            body.scale.z = 0.18
            self._set_marker_color(body, 0.05, 0.35, 1.0, 0.95)
            arr.markers.append(body)

            text = Marker()
            text.header.frame_id = frame_id
            text.header.stamp = stamp
            text.ns = "rl_confidence_origin"
            text.id = 4
            text.type = Marker.TEXT_VIEW_FACING
            text.action = Marker.ADD
            text.pose.position.x = float(rxy[0])
            text.pose.position.y = float(rxy[1])
            text.pose.position.z = 0.90
            text.pose.orientation.w = 1.0
            text.scale.z = 0.16
            self._set_marker_color(text, 1.0, 1.0, 1.0, 0.98)
            text.text = (
                f"CONF ORIGIN step={step}\n"
                f"mode={label}\n"
                f"base=({float(rxy[0]):+.2f},{float(rxy[1]):+.2f}) yaw={math.degrees(ryaw):+.1f}\n"
                f"sensor=({float(sxy[0]):+.2f},{float(sxy[1]):+.2f}) yaw={math.degrees(syaw):+.1f}"
            )
            arr.markers.append(text)

            # v109: publish raw TF comparison markers in the same MarkerArray.
            # These markers are independent of the confidence pose calculation.
            # If one of these matches RViz RobotModel while the red CONF marker
            # does not, the confidence source-frame/env selection is wrong.
            # If none of these match RobotModel, TF publishers are conflicting or
            # RViz is displaying a different domain/process.
            #
            # When confidence is unified with the TF cube, the red sphere is drawn
            # from the live TF(map->base_footprint) pose itself, so only the
            # base_footprint cube is meaningful here (it must coincide exactly with
            # the sphere).  The base_link/base_scan cubes are at rigid offsets and
            # only add visual confusion in unified mode, so they are dropped.
            unified_now = bool(
                isinstance(getattr(self, "_confidence_unified_xy", None), np.ndarray)
                and getattr(self, "_confidence_unified_xy").size >= 2
            )
            if unified_now:
                compare_frames = [
                    ("base_footprint", 20, (0.0, 1.0, 0.0, 0.95), 0.24),
                ]
            else:
                compare_frames = [
                    ("base_footprint", 20, (0.0, 1.0, 0.0, 0.95), 0.24),
                    ("base_link", 21, (1.0, 0.0, 1.0, 0.95), 0.24),
                    ("base_scan", 22, (0.0, 0.75, 1.0, 0.95), 0.20),
                ]
            compare_text_lines = []
            for fname, mid, rgba, size in compare_frames:
                try:
                    pose_cmp = None
                    if hasattr(self.ros, "get_frame_pose2d_manual"):
                        pose_cmp = self.ros.get_frame_pose2d_manual(
                            target_frame=frame_id,
                            source_frame=fname,
                            max_age_sec=float(os.environ.get("TB3_RL_MANUAL_TF_MAX_AGE_SEC", "5.0") or 5.0),
                        )
                    if pose_cmp is None and str(os.environ.get("TB3_RL_CONFIDENCE_TF_BUFFER_FALLBACK", "0") or "0").strip().lower() in {"1", "true", "yes", "on"} and hasattr(self.ros, "get_frame_pose2d"):
                        pose_cmp = self.ros.get_frame_pose2d(
                            target_frame=frame_id,
                            source_frame=fname,
                            stamp=None,
                            timeout_sec=0.05,
                            allow_latest_fallback=True,
                        )
                    if pose_cmp is None:
                        compare_text_lines.append(f"{fname}=NA")
                        continue
                    cxy, cyaw = pose_cmp
                    cxy = np.asarray(cxy, dtype=np.float32).reshape(-1)[:2]
                    if cxy.size < 2 or not np.all(np.isfinite(cxy)):
                        compare_text_lines.append(f"{fname}=bad")
                        continue
                    m = Marker()
                    m.header.frame_id = frame_id
                    m.header.stamp = stamp
                    m.ns = "rl_confidence_tf_compare"
                    m.id = int(mid)
                    m.type = Marker.CUBE
                    m.action = Marker.ADD
                    m.pose.position.x = float(cxy[0])
                    m.pose.position.y = float(cxy[1])
                    m.pose.position.z = 0.62 + 0.04 * float(mid - 20)
                    m.pose.orientation.w = 1.0
                    m.scale.x = float(size)
                    m.scale.y = float(size)
                    m.scale.z = float(size)
                    self._set_marker_color(m, float(rgba[0]), float(rgba[1]), float(rgba[2]), float(rgba[3]))
                    arr.markers.append(m)
                    compare_text_lines.append(f"{fname}=({float(cxy[0]):+.2f},{float(cxy[1]):+.2f})")
                except Exception:
                    compare_text_lines.append(f"{fname}=ERR")

            if compare_text_lines:
                ct = Marker()
                ct.header.frame_id = frame_id
                ct.header.stamp = stamp
                ct.ns = "rl_confidence_tf_compare"
                ct.id = 29
                ct.type = Marker.TEXT_VIEW_FACING
                ct.action = Marker.ADD
                ct.pose.position.x = float(rxy[0])
                ct.pose.position.y = float(rxy[1])
                ct.pose.position.z = 1.20
                ct.pose.orientation.w = 1.0
                ct.scale.z = 0.13
                self._set_marker_color(ct, 1.0, 1.0, 1.0, 0.95)
                ct.text = "TF compare\n" + "\n".join(compare_text_lines)
                arr.markers.append(ct)

            pub.publish(arr)
        except Exception as exc:
            try:
                if str(os.environ.get("TB3_RL_CONFIDENCE_ORIGIN_WARN", "0")).strip().lower() in {"1", "true", "yes", "on"}:
                    self.ros.get_logger().warn(f"CONFIDENCE_ORIGIN_MARKER_FAILED | err={exc}")
            except Exception:
                pass

    def _make_scan_points_marker_array_at_pose(
        self,
        *,
        ranges: np.ndarray,
        angles: np.ndarray,
        target_frame: str,
        origin_xy: np.ndarray,
        origin_yaw: float,
        namespace: str,
        point_rgb: tuple[float, float, float],
        ray_rgb: tuple[float, float, float],
        point_scale: float = 0.055,
        ray_scale: float = 0.010,
        max_points: int = 720,
        stamp=None,
    ) -> Optional[MarkerArray]:
        """Create map-frame scan endpoint/ray markers from one hard robot anchor.

        Unlike the old TF-per-marker path, the origin is explicitly the same
        map->base pose used for confidence/priority updates and robot-centric
        crop.  This guarantees /rl_*_points starts at the robot location on /map.
        """
        ranges = np.asarray(ranges, dtype=np.float32)
        angles = np.asarray(angles, dtype=np.float32)
        if ranges.size == 0 or angles.size == 0:
            return None
        n = int(min(ranges.size, angles.size))
        if n <= 0:
            return None
        xy = np.asarray(origin_xy, dtype=np.float32).reshape(-1)
        if xy.size < 2 or not np.all(np.isfinite(xy[:2])) or not math.isfinite(float(origin_yaw)):
            return None
        ox, oy, yaw = float(xy[0]), float(xy[1]), float(origin_yaw)

        step = max(1, int(math.ceil(float(n) / float(max(max_points, 1)))))
        idxs = range(0, n, step)

        arr = MarkerArray()
        marker_stamp = stamp if stamp is not None else self._latest_tf_stamp()
        lifetime = self._marker_lifetime(0.45)

        pts = Marker()
        pts.header.frame_id = target_frame
        pts.header.stamp = marker_stamp
        pts.ns = namespace
        pts.id = 0
        pts.type = Marker.POINTS
        pts.action = Marker.ADD
        pts.pose.orientation.w = 1.0
        pts.scale.x = float(point_scale)
        pts.scale.y = float(point_scale)
        self._set_marker_color(pts, point_rgb[0], point_rgb[1], point_rgb[2], 0.95)
        pts.lifetime = lifetime

        rays = Marker()
        rays.header.frame_id = target_frame
        rays.header.stamp = marker_stamp
        rays.ns = namespace
        rays.id = 1
        rays.type = Marker.LINE_LIST
        rays.action = Marker.ADD
        rays.pose.orientation.w = 1.0
        rays.scale.x = float(ray_scale)
        self._set_marker_color(rays, ray_rgb[0], ray_rgb[1], ray_rgb[2], 0.35)
        rays.lifetime = lifetime

        anchor = Marker()
        anchor.header.frame_id = target_frame
        anchor.header.stamp = marker_stamp
        anchor.ns = namespace
        anchor.id = 2
        anchor.type = Marker.SPHERE
        anchor.action = Marker.ADD
        anchor.pose.position = self._point_xyz(ox, oy, 0.11)
        anchor.pose.orientation.w = 1.0
        anchor.scale.x = 0.16
        anchor.scale.y = 0.16
        anchor.scale.z = 0.16
        self._set_marker_color(anchor, 1.0, 1.0, 0.0, 0.95)
        anchor.lifetime = lifetime

        origin = self._point_xyz(ox, oy, 0.07)
        for i in idxs:
            rr = float(ranges[i])
            aa = float(angles[i])
            if not math.isfinite(rr) or rr <= 0.0:
                continue
            gx = ox + rr * math.cos(yaw + aa)
            gy = oy + rr * math.sin(yaw + aa)
            p = self._point_xyz(gx, gy, 0.07)
            pts.points.append(p)
            rays.points.append(origin)
            rays.points.append(p)

        arr.markers.append(pts)
        arr.markers.append(rays)
        arr.markers.append(anchor)
        return arr

    def _tf_pose2d_at(self, target_frame: str, source_frame: str, stamp=None) -> Optional[tuple[float, float, float]]:
        """Return source_frame pose expressed in target_frame at the scan timestamp.

        v8 used the latest TF for markers/policy-scan debug. During Cartographer
        updates, map->odom can change even if the robot is physically stationary;
        combining an old LaserScan with a latest pose makes debug points look like
        they rotate or slide relative to /map. v9 uses the raw scan stamp first.
        """
        target_frame = str(target_frame or "map").strip().lstrip("/") or "map"
        source_frame = str(source_frame or "base_scan").strip().lstrip("/") or "base_scan"
        tf_buffer = getattr(self.ros, "tf_buffer", None)
        if tf_buffer is None:
            return None

        query_time = rclpy.time.Time()
        if stamp is not None:
            try:
                if int(getattr(stamp, "sec", 0)) != 0 or int(getattr(stamp, "nanosec", 0)) != 0:
                    query_time = rclpy.time.Time.from_msg(stamp)
            except Exception:
                query_time = rclpy.time.Time()

        try:
            transform = tf_buffer.lookup_transform(
                target_frame,
                source_frame,
                query_time,
                timeout=rclpy.duration.Duration(seconds=0.04),
            )
        except Exception:
            try:
                transform = tf_buffer.lookup_transform(
                    target_frame,
                    source_frame,
                    rclpy.time.Time(),
                    timeout=rclpy.duration.Duration(seconds=0.02),
                )
            except Exception:
                return None
        t = transform.transform.translation
        q = transform.transform.rotation
        yaw = self._yaw_from_quaternion_xyzw_static(q.x, q.y, q.z, q.w)
        return float(t.x), float(t.y), float(yaw)

    def _tf_pose2d_latest(self, target_frame: str, source_frame: str) -> Optional[tuple[float, float, float]]:
        return self._tf_pose2d_at(target_frame=target_frame, source_frame=source_frame, stamp=None)

    def _policy_scan_angle_offset_rad(self) -> float:
        try:
            return math.radians(float(os.environ.get("TB3_RL_LIDAR_ANGLE_OFFSET_DEG", "0.0")))
        except Exception:
            return 0.0

    def _policy_scan_flip_lr(self) -> bool:
        return self._scan_bool_env("TB3_RL_LIDAR_FLIP_LR", False)

    def _correct_scan_angles_for_policy(self, angles: np.ndarray) -> np.ndarray:
        out = np.asarray(angles, dtype=np.float32)
        if self._policy_scan_flip_lr():
            out = -out
        offset = self._policy_scan_angle_offset_rad()
        if abs(offset) > 1.0e-12:
            out = out + float(offset)
        return np.arctan2(np.sin(out), np.cos(out)).astype(np.float32)

    @staticmethod
    def _marker_lifetime(sec: float = 0.35) -> Duration:
        sec = max(float(sec), 0.0)
        whole = int(sec)
        nano = int(round((sec - whole) * 1_000_000_000.0))
        return Duration(sec=whole, nanosec=nano)

    def _make_scan_points_marker_array(
        self,
        *,
        ranges: np.ndarray,
        angles: np.ndarray,
        target_frame: str,
        source_frame: str,
        namespace: str,
        point_rgb: tuple[float, float, float],
        ray_rgb: tuple[float, float, float],
        point_scale: float = 0.055,
        ray_scale: float = 0.010,
        max_points: int = 720,
        stamp=None,
    ) -> Optional[MarkerArray]:
        """Create map-frame scan endpoint/ray markers for RViz debugging.

        The key difference from the LaserScan debug topic is that these markers
        are already transformed into `target_frame` at publish time. If raw scan
        markers fit the map but policy markers do not, the problem is the policy
        resampler convention. If both are shifted, the problem is TF/Cartographer
        pose/map alignment.
        """
        ranges = np.asarray(ranges, dtype=np.float32)
        angles = np.asarray(angles, dtype=np.float32)
        if ranges.size == 0 or angles.size == 0:
            return None
        n = int(min(ranges.size, angles.size))
        if n <= 0:
            return None
        pose = self._tf_pose2d_at(target_frame=target_frame, source_frame=source_frame, stamp=stamp)
        if pose is None:
            return None
        ox, oy, yaw = pose

        # Downsample only if somebody accidentally publishes a very dense scan.
        step = max(1, int(math.ceil(float(n) / float(max(max_points, 1)))))
        idxs = range(0, n, step)

        arr = MarkerArray()
        marker_stamp = stamp if stamp is not None else self._latest_tf_stamp()
        lifetime = self._marker_lifetime(0.45)

        pts = Marker()
        pts.header.frame_id = target_frame
        pts.header.stamp = marker_stamp
        pts.ns = namespace
        pts.id = 0
        pts.type = Marker.POINTS
        pts.action = Marker.ADD
        pts.pose.orientation.w = 1.0
        pts.scale.x = float(point_scale)
        pts.scale.y = float(point_scale)
        self._set_marker_color(pts, point_rgb[0], point_rgb[1], point_rgb[2], 0.95)
        pts.lifetime = lifetime

        rays = Marker()
        rays.header.frame_id = target_frame
        rays.header.stamp = marker_stamp
        rays.ns = namespace
        rays.id = 1
        rays.type = Marker.LINE_LIST
        rays.action = Marker.ADD
        rays.pose.orientation.w = 1.0
        rays.scale.x = float(ray_scale)
        self._set_marker_color(rays, ray_rgb[0], ray_rgb[1], ray_rgb[2], 0.35)
        rays.lifetime = lifetime

        origin = self._point_xyz(ox, oy, 0.07)
        for i in idxs:
            rr = float(ranges[i])
            aa = float(angles[i])
            if not math.isfinite(rr) or rr <= 0.0:
                continue
            gx = ox + rr * math.cos(yaw + aa)
            gy = oy + rr * math.sin(yaw + aa)
            p = self._point_xyz(gx, gy, 0.07)
            pts.points.append(p)
            rays.points.append(origin)
            rays.points.append(p)

        arr.markers.append(pts)
        arr.markers.append(rays)
        return arr

    def _publish_policy_scan_map_debug(self, vector_obs: np.ndarray, raw_scan_msg) -> None:
        """Publish raw/policy scan debug markers anchored at TF(map->base).

        v13 hard rule:
          - marker origin = the same strict map->base pose used by
            confidence/priority and robot-centric crop;
          - raw marker uses driver-native /scan angles by default;
          - policy marker uses the canonical policy 60-bin angles.

        This prevents the previous failure where marker rays appeared around the
        /map canvas center or used a different TF timestamp than the semantic
        layer update.
        """
        if self.policy_scan_marker_pub is None and self.raw_scan_marker_pub is None:
            return
        try:
            raw_header = getattr(raw_scan_msg, "header", None)
            raw_stamp = getattr(raw_header, "stamp", None)
            raw_scan_frame = str(getattr(raw_header, "frame_id", "") or "").strip().lstrip("/")
            if not raw_scan_frame:
                raw_scan_frame = str(os.environ.get("TB3_RL_SCAN_FRAME", "base_scan") or "base_scan").strip().lstrip("/") or "base_scan"

            # v14 default: scan debug markers are robot-frame markers, not
            # precomputed absolute map points.  RViz then uses the same TF path
            # as the RobotModel, so the marker fan must start at the robot.
            robot_frame_debug = self._scan_bool_env("TB3_RL_SCAN_MARKERS_IN_ROBOT_FRAME", True)
            if robot_frame_debug:
                target_frame = raw_scan_frame
                origin_xy = np.array([0.0, 0.0], dtype=np.float32)
                origin_yaw = 0.0
                marker_stamp = self._zero_stamp()
            else:
                target_frame = str(self.map_frame or "map").strip().lstrip("/") or "map"
                pose = self._get_map_base_pose2d_hard()
                if pose is None:
                    return
                origin_xy, base_yaw = pose
                origin_yaw = self._get_map_scan_yaw_hard(raw_scan_msg=raw_scan_msg, base_yaw=base_yaw)
                marker_stamp = self._zero_stamp()


            raw_min = self._scan_float(getattr(raw_scan_msg, "range_min", 0.12), 0.12)
            raw_max = self._scan_float(getattr(raw_scan_msg, "range_max", 3.5), 3.5)
            if raw_max <= raw_min + 1e-6:
                raw_min, raw_max = 0.12, 3.5

            # Policy vector markers: exactly the LiDAR slice the policy sees,
            # interpreted in the canonical policy convention.
            if self.policy_scan_marker_pub is not None:
                num_bins = max(int(self.num_lidar_bins), 1)
                norm = np.asarray(vector_obs[:num_bins], dtype=np.float32)
                norm = np.nan_to_num(norm, nan=1.0, posinf=1.0, neginf=0.0)
                policy_ranges = np.clip(norm, 0.0, 1.0) * (raw_max - raw_min) + raw_min
                front_index = self._policy_scan_front_index()
                policy_angles = 2.0 * math.pi * (np.arange(num_bins, dtype=np.float32) - float(front_index)) / float(num_bins)
                policy_angles = np.arctan2(np.sin(policy_angles), np.cos(policy_angles)).astype(np.float32)
                arr = self._make_scan_points_marker_array_at_pose(
                    ranges=policy_ranges,
                    angles=policy_angles,
                    target_frame=target_frame,
                    origin_xy=origin_xy,
                    origin_yaw=origin_yaw,
                    namespace="rl_policy_scan_60_map_points",
                    point_rgb=(0.0, 1.0, 1.0),
                    ray_rgb=(0.0, 0.7, 1.0),
                    point_scale=0.075,
                    ray_scale=0.012,
                    max_points=max(num_bins, 1),
                    # v16: in robot-frame debug mode use a zero stamp for
                    # policy markers too.  Mixing a base_scan frame with an old
                    # raw scan stamp can make RViz render the marker fan detached
                    # from the current robot pose even though the semantic update
                    # is robot-anchored.
                    stamp=marker_stamp,
                )
                if arr is not None:
                    self.policy_scan_marker_pub.publish(arr)

            # Raw scan marker: reference geometry from /scan.  Do not apply the
            # policy offset/flip by default; this should match raw /scan in RViz
            # and the SLAM map if TF is healthy.  Set the env below to 0 only
            # when intentionally comparing policy-corrected raw angles.
            if self.raw_scan_marker_pub is not None:
                raw_ranges = np.asarray(getattr(raw_scan_msg, "ranges", []) or [], dtype=np.float32)
                if raw_ranges.size > 0:
                    raw_ranges = np.nan_to_num(raw_ranges, nan=raw_max, posinf=raw_max, neginf=raw_min)
                    raw_ranges = np.clip(raw_ranges, raw_min, raw_max).astype(np.float32)
                    angle_min = self._scan_float(getattr(raw_scan_msg, "angle_min", 0.0), 0.0)
                    angle_inc = self._scan_float(getattr(raw_scan_msg, "angle_increment", 0.0), 0.0)
                    raw_angles = angle_min + np.arange(raw_ranges.size, dtype=np.float32) * float(angle_inc)
                    raw_angles = np.arctan2(np.sin(raw_angles), np.cos(raw_angles)).astype(np.float32)
                    if not self._scan_bool_env("TB3_RL_RAW_SCAN_MARKER_UNCORRECTED", True):
                        raw_angles = self._correct_scan_angles_for_policy(raw_angles)
                    arr = self._make_scan_points_marker_array_at_pose(
                        ranges=raw_ranges,
                        angles=raw_angles,
                        target_frame=target_frame,
                        origin_xy=origin_xy,
                        origin_yaw=origin_yaw,
                        namespace="rl_raw_scan_map_points",
                        point_rgb=(1.0, 1.0, 1.0),
                        ray_rgb=(1.0, 1.0, 1.0),
                        point_scale=0.035,
                        ray_scale=0.006,
                        max_points=360,
                        stamp=marker_stamp,
                    )
                    if arr is not None:
                        self.raw_scan_marker_pub.publish(arr)
        except Exception as exc:
            try:
                self.ros.get_logger().warn(f"SCAN_MAP_DEBUG_PUBLISH_FAILED | {exc}")
            except Exception:
                pass

    def _publish_policy_scan(self, vector_obs: np.ndarray, raw_scan_msg) -> None:
        if self.policy_scan_pub is None and self.policy_scan_60_pub is None:
            return
        if self.step_count % self.policy_scan_publish_every_n != 0:
            return
        try:
            num_bins = max(int(self.num_lidar_bins), 1)
            min_range = 0.12
            max_range = 3.5
            norm = np.asarray(vector_obs[:num_bins], dtype=np.float32)
            norm = np.nan_to_num(norm, nan=1.0, posinf=1.0, neginf=0.0)
            ranges = np.clip(norm, 0.0, 1.0) * (max_range - min_range) + min_range

            msg = LaserScan()
            raw_header = getattr(raw_scan_msg, "header", None)
            raw_stamp = getattr(raw_header, "stamp", None)
            msg.header.stamp = raw_stamp if raw_stamp is not None else self._latest_tf_stamp()
            msg.header.frame_id = str(getattr(raw_header, "frame_id", "") or "base_scan")
            front_index = self._policy_scan_front_index()
            msg.angle_increment = float(2.0 * math.pi / float(num_bins))
            msg.angle_min = float(-msg.angle_increment * float(front_index))
            msg.angle_max = float(msg.angle_min + msg.angle_increment * float(num_bins - 1))
            msg.time_increment = 0.0
            msg.scan_time = 0.0
            msg.range_min = min_range
            msg.range_max = max_range
            msg.ranges = [float(x) for x in ranges.tolist()]
            msg.intensities = []
            if self.policy_scan_pub is not None:
                self.policy_scan_pub.publish(msg)
            if self.policy_scan_60_pub is not None:
                self.policy_scan_60_pub.publish(msg)
            self._publish_policy_scan_map_debug(vector_obs, raw_scan_msg)
        except Exception as exc:
            try:
                self.ros.get_logger().warn(f"POLICY_SCAN_PUBLISH_FAILED | {exc}")
            except Exception:
                pass

    def _get_obs(self):
        if self.ros.scan is None or self.ros.odom is None:
            return self._empty_observation()

        stats = self.last_map_stats

        scan_msg = self.ros.scan
        scan_angle_min = getattr(scan_msg, "angle_min", None)
        scan_angle_increment = getattr(scan_msg, "angle_increment", None)
        scan_angle_max = getattr(scan_msg, "angle_max", None)

        vector_obs = build_exploration_observation(
            scan_ranges=scan_msg.ranges,
            coverage_ratio=stats.coverage_ratio,
            coverage_delta=stats.coverage_delta,
            frontier_distance=stats.frontier_distance,
            frontier_angle=stats.frontier_angle,
            target_priority=stats.target_priority,
            mean_confidence=stats.mean_confidence,
            stale_ratio=stats.stale_ratio,
            low_confidence_ratio=stats.low_confidence_ratio,
            prev_action=self.prev_action,
            num_lidar_bins=self.num_lidar_bins,
            max_linear_speed=self.max_linear_speed,
            max_angular_speed=self.max_angular_speed,
            scan_angle_min=scan_angle_min,
            scan_angle_increment=scan_angle_increment,
            scan_angle_max=scan_angle_max,
            include_target_priority=not bool(getattr(self, "no_priority_model_input", False)),
        ).astype(np.float32)

        self._update_scan_geometry_debug(scan_msg, vector_obs)
        self._publish_policy_scan(vector_obs, scan_msg)

        if not self.use_map_cnn:
            self._push_vector_history(vector_obs)
            return vector_obs

        robot_pose = self._get_robot_pose2d()

        if robot_pose is None:
            map_obs = np.zeros(
                (self.map_channels, self.map_obs_size, self.map_obs_size),
                dtype=np.float32,
            )
        else:
            robot_xy, robot_yaw = robot_pose
            map_obs = self.exploration_map.build_update_need_tensor(
                robot_xy=robot_xy,
                robot_yaw=robot_yaw,
                output_size=self.map_obs_size,
                size_m=self.map_obs_size_m,
                rotate_to_robot=True,
            ).astype(np.float32)

        self._publish_debug_input_map(map_obs)

        self._push_observation_history(vector_obs, map_obs)

        obs = {
            "vector": vector_obs,
            "map": map_obs,
        }

        if self.use_temporal_cnn:
            obs["seq"] = self._sequence_observation()
            obs["map_seq"] = self._map_sequence_observation()

        return obs

    def _empty_observation(self):
        vector_obs = np.zeros(self.obs_dim, dtype=np.float32)

        if not self.use_map_cnn:
            return vector_obs

        obs = {
            "vector": vector_obs,
            "map": np.zeros(
                (self.map_channels, self.map_obs_size, self.map_obs_size),
                dtype=np.float32,
            ),
        }

        if self.use_temporal_cnn:
            obs["seq"] = np.zeros(
                (self.temporal_history_len, self.obs_dim),
                dtype=np.float32,
            )
            obs["map_seq"] = np.zeros(
                (self.temporal_history_len, self.map_channels, self.map_obs_size, self.map_obs_size),
                dtype=np.float32,
            )

        return obs

    def _push_vector_history(self, vector_obs: np.ndarray):
        # Backward-compatible helper for vector-only observations.
        if not self.use_temporal_cnn:
            return
        vector_obs = np.asarray(vector_obs, dtype=np.float32)
        if not self.vector_history:
            for _ in range(self.temporal_history_len):
                self.vector_history.append(vector_obs.copy())
            return
        self.vector_history.append(vector_obs.copy())

    def _push_observation_history(self, vector_obs: np.ndarray, map_obs: np.ndarray):
        if not self.use_temporal_cnn:
            return
        vector_obs = np.asarray(vector_obs, dtype=np.float32)
        map_obs = np.asarray(map_obs, dtype=np.float32)
        if not self.vector_history:
            for _ in range(self.temporal_history_len):
                self.vector_history.append(vector_obs.copy())
                self.map_history.append(map_obs.copy())
        else:
            self.vector_history.append(vector_obs.copy())
            self.map_history.append(map_obs.copy())

    def _sequence_observation(self) -> np.ndarray:
        if not self.vector_history:
            return np.zeros(
                (self.temporal_history_len, self.obs_dim),
                dtype=np.float32,
            )
        seq = list(self.vector_history)
        while len(seq) < self.temporal_history_len:
            seq.insert(0, seq[0].copy())
        return np.stack(seq[-self.temporal_history_len :], axis=0).astype(np.float32)

    def _map_sequence_observation(self) -> np.ndarray:
        shape = (self.temporal_history_len, self.map_channels, self.map_obs_size, self.map_obs_size)
        if not hasattr(self, "map_history") or not self.map_history:
            return np.zeros(shape, dtype=np.float32)
        seq = list(self.map_history)
        while len(seq) < self.temporal_history_len:
            seq.insert(0, seq[0].copy())
        out = np.stack(seq[-self.temporal_history_len :], axis=0).astype(np.float32)
        if out.shape != shape:
            fixed = np.zeros(shape, dtype=np.float32)
            try:
                t = min(fixed.shape[0], out.shape[0])
                c = min(fixed.shape[1], out.shape[1])
                h = min(fixed.shape[2], out.shape[2])
                w = min(fixed.shape[3], out.shape[3])
                fixed[-t:, :c, :h, :w] = out[-t:, :c, :h, :w]
            except Exception:
                pass
            return fixed
        return np.clip(out, 0.0, 1.0)

    def _distance_to_goal(self, robot_xy: Optional[np.ndarray]) -> float:
        if robot_xy is None:
            return 999.0

        diff = self.goal_xy - robot_xy
        return float(np.linalg.norm(diff))

    def _check_collision(self) -> bool:
        if self.ros.scan is None:
            return False

        ranges = np.asarray(self.ros.scan.ranges, dtype=np.float32)
        ranges = np.nan_to_num(
            ranges,
            nan=10.0,
            posinf=10.0,
            neginf=0.0,
        )
        finite = ranges[np.isfinite(ranges)]
        global_min = float(np.min(finite)) if finite.size else 10.0
        front_min = self._scan_min_distance_in_sector(
            scan=self.ros.scan,
            center_angle=0.0,
            half_width_rad=math.radians(28.0),
            max_considered_range=0.80,
        )
        # LiDAR collision threshold is the hard terminal condition.  Use both
        # global min and front-sector min so a wall hit in front terminates even
        # when a single noisy global beam is filtered by the controller.
        hit = bool(min(global_min, front_min) < self.collision_threshold)
        if hit:
            self._last_collision_global_min = float(global_min)
            self._last_collision_front_min = float(front_min)
        return hit

    def _instantaneous_shake_reason(self) -> str:
        """Return a non-'none' reason when the body is not stably planted.

        In this project "shake" is not planar spinning.  It means the physical
        TurtleBot body is wobbling, tilted, bouncing, airborne, or otherwise not
        properly attached to the floor.  Use the best available 6-DoF source:
        Gazebo model odometry, IMU, then normal odometry as a fallback.
        """
        if not bool(getattr(self, "shake_restart", True)):
            return "none"
        try:
            rpy = None
            if hasattr(self.ros, "get_body_roll_pitch_yaw"):
                rpy = self.ros.get_body_roll_pitch_yaw()
            if rpy is None:
                rpy = self.ros.get_roll_pitch_yaw()
            roll = pitch = 0.0
            if rpy is not None:
                roll, pitch, _ = rpy
            tilt = max(abs(float(roll)), abs(float(pitch)))

            wx = wy = 0.0
            if hasattr(self.ros, "get_body_angular_xy"):
                wx, wy = self.ros.get_body_angular_xy()
            elif self.ros.odom is not None:
                twist = self.ros.odom.twist.twist
                wx = float(getattr(twist.angular, "x", 0.0))
                wy = float(getattr(twist.angular, "y", 0.0))

            vz = 0.0
            if hasattr(self.ros, "get_body_vertical_velocity"):
                vz = self.ros.get_body_vertical_velocity()
            elif self.ros.odom is not None:
                vz = float(getattr(self.ros.odom.twist.twist.linear, "z", 0.0))

            body_z = None
            if hasattr(self.ros, "get_body_z"):
                body_z = self.ros.get_body_z()
            elif self.ros.odom is not None:
                body_z = float(self.ros.odom.pose.pose.position.z)

            nominal_z = float(getattr(self, "_reset_nominal_z", 0.05))
            z_dev = 0.0 if body_z is None else abs(float(body_z) - nominal_z)
            z_min = float(getattr(self, "shake_ground_min_z", -0.02))
            z_max = float(getattr(self, "shake_ground_max_z", 0.13))

            if tilt >= float(getattr(self, "shake_tilt_threshold", 0.12)):
                return f"body_tilt:roll={roll:.3f},pitch={pitch:.3f}"
            if body_z is not None and (float(body_z) < z_min or float(body_z) > z_max):
                return f"body_z:{float(body_z):.3f}[{z_min:.2f},{z_max:.2f}]"
            # v48: Gazebo TurtleBot3 Burger normally reports body z around the
            # wheel-contact height, not exactly the SetEntityPose request z.
            # Treat z-deviation from reset_z as diagnostic only by default; only
            # hard-fail it when explicitly requested.  The absolute ground range
            # check above still catches airborne/fallen states.
            z_dev_strict = str(os.environ.get("TB3_RL_SHAKE_Z_DEV_STRICT", "0")).strip().lower() in {"1", "true", "yes", "on"}
            if z_dev_strict and body_z is not None and z_dev >= float(getattr(self, "shake_z_deviation_threshold", 0.05)):
                return f"body_z_dev:{z_dev:.3f}"
            ang_xy = math.hypot(float(wx), float(wy))
            if ang_xy >= float(getattr(self, "shake_angular_xy_threshold", 0.70)):
                return f"body_ang_xy:{ang_xy:.3f}"
            if abs(float(vz)) >= float(getattr(self, "shake_linear_z_threshold", 0.08)):
                return f"body_vz:{float(vz):.3f}"
        except Exception as exc:
            return f"shake_check_error:{type(exc).__name__}"
        return "none"

    def _update_yaw_wobble_reason(self) -> str:
        """Detect planar left/right wobble that does not show up as roll/pitch.

        In Gazebo and on TurtleBot3 odometry, angular.x/y are often zero even
        when the robot visibly wiggles.  The failure mode is repeated angular.z
        sign flips or long in-place spinning with very small net displacement.
        This helper uses a short pose/command window and returns a reason string
        only when the motion is persistent enough to be unsafe or useless.
        """
        if not bool(getattr(self, "shake_yaw_wobble", True)):
            return "none"

        try:
            cmd_v = float(getattr(self, "filtered_action", np.zeros(2, dtype=np.float32))[0])
            cmd_w = float(getattr(self, "filtered_action", np.zeros(2, dtype=np.float32))[1])
        except Exception:
            cmd_v = 0.0
            cmd_w = 0.0

        odom_wz = 0.0
        try:
            if self.ros.odom is not None:
                odom_wz = float(self.ros.odom.twist.twist.angular.z)
        except Exception:
            odom_wz = 0.0

        # Prefer the actually published command when it is meaningful.  Fall
        # back to odom angular.z when the controller or driver has already
        # changed the command.
        cmd_thr = float(getattr(self, "shake_cmd_flip_threshold", 0.16))
        yaw_thr = float(getattr(self, "shake_yaw_rate_threshold", 0.24))
        yaw_signal = cmd_w if abs(cmd_w) >= cmd_thr else odom_wz
        yaw_mag = abs(float(yaw_signal))
        yaw_sign = 1 if yaw_signal > yaw_thr else (-1 if yaw_signal < -yaw_thr else 0)

        pose = None
        try:
            pose = self._get_robot_pose2d(frame_id=self.pose_frame)
        except Exception:
            pose = None

        x = y = yaw = 0.0
        if pose is not None:
            try:
                xy, yyaw = pose
                x = float(xy[0])
                y = float(xy[1])
                yaw = float(yyaw)
            except Exception:
                x = y = yaw = 0.0

        hist = getattr(self, "_shake_wobble_history", None)
        if hist is None or getattr(hist, "maxlen", None) != int(getattr(self, "shake_wobble_window_steps", 8)):
            hist = deque(maxlen=int(getattr(self, "shake_wobble_window_steps", 8)))
            self._shake_wobble_history = hist

        hist.append((int(getattr(self, "step_count", 0)), x, y, yaw, int(yaw_sign), float(yaw_mag), float(cmd_v), float(odom_wz)))
        samples = list(hist)
        if len(samples) < 3:
            self._shake_last_wobble_reason = "warming"
            return "none"

        flips = 0
        prev_sign = 0
        for item in samples:
            sign = int(item[4])
            if sign == 0:
                continue
            if prev_sign != 0 and sign != prev_sign:
                flips += 1
            prev_sign = sign

        net_disp = math.hypot(float(samples[-1][1]) - float(samples[0][1]), float(samples[-1][2]) - float(samples[0][2]))
        yaw_accum = 0.0
        for a, b in zip(samples[:-1], samples[1:]):
            dyaw = self._normalize_angle(float(b[3]) - float(a[3]))
            if math.isfinite(dyaw):
                yaw_accum += abs(dyaw)

        max_net = float(getattr(self, "shake_wobble_max_net_motion_m", 0.045))
        min_flips = int(getattr(self, "shake_wobble_min_flips", 2))
        low_motion = bool(net_disp <= max_net or abs(cmd_v) <= 0.025)
        if flips >= min_flips and low_motion:
            reason = f"yaw_wobble:flips={flips},net={net_disp:.3f},w={yaw_mag:.2f}"
            self._shake_last_wobble_reason = reason
            return reason

        spin_limit = int(getattr(self, "shake_spin_stall_restart_steps", 0))
        spin_steps = int(getattr(self, "sustained_rotation_steps", 0))
        if spin_limit > 0 and spin_steps >= spin_limit:
            reason = f"spin_stall:{spin_steps}/{spin_limit},net={net_disp:.3f},yaw={math.degrees(yaw_accum):.1f}"
            self._shake_last_wobble_reason = reason
            return reason

        self._shake_last_wobble_reason = f"ok:flips={flips},net={net_disp:.3f},spin={spin_steps}"
        return "none"

    def _update_shake_restart_state(self) -> bool:
        """Restart when body shake or yaw wobble persists for N effective ticks.

        v8 changes this from strict consecutive counting to a leaky integrator.
        Real wobble often alternates between one bad and one borderline step, so
        the old code reset the counter back to 0/8 forever.
        """
        self._last_shake_restart = False
        physical_reason = self._instantaneous_shake_reason()
        wobble_reason = self._update_yaw_wobble_reason() if bool(getattr(self, "shake_yaw_wobble", False)) else "none"
        reason = physical_reason if physical_reason != "none" else wobble_reason

        limit = int(getattr(self, "shake_restart_steps_limit", 8))
        limit = max(limit, 1)
        current = int(getattr(self, "shake_steps", 0))

        if reason == "none":
            if bool(getattr(self, "shake_leaky_decay", True)):
                self.shake_steps = max(current - 1, 0)
            else:
                self.shake_steps = 0
            self._last_shake_active = bool(self.shake_steps > 0)
            if self.shake_steps <= 0:
                self._last_shake_reason = "none"
            else:
                self._last_shake_reason = "decay:body_not_planted"
            return False

        inc = 1
        self.shake_steps = min(current + inc, limit)
        self._last_shake_active = True
        self._last_shake_reason = reason
        if self.shake_steps >= limit:
            self._last_shake_restart = True
            self.ros.get_logger().warn(
                "SHAKE_RESTART | "
                f"steps={self.shake_steps}/{limit} | "
                f"reason={reason} | "
                f"body_contact=tilt/z/angular_xy/vz | yaw_wobble={getattr(self, '_shake_last_wobble_reason', 'disabled')}"
            )
            return True
        return False

    def _check_fallen(self) -> bool:
        self._last_fallen_reason = "none"
        tilted = self.ros.is_fallen(
            max_abs_roll=self.fallen_roll_threshold,
            max_abs_pitch=self.fallen_pitch_threshold,
        )
        if tilted:
            self._last_fallen_reason = "roll_pitch"
            return True

        # Boundary terminal may be disabled, but physical drop/vertical fall must
        # still terminate the episode with -200.  Odom z is available in the
        # standard TurtleBot3 Gazebo odometry path and is cheap enough to check
        # every step.
        try:
            if self.ros.odom is not None:
                z = float(self.ros.odom.pose.pose.position.z)
                max_abs_z = float(getattr(self, "safety_boundary_max_abs_z", 0.45))
                if not np.isfinite(z):
                    self._last_fallen_reason = "nonfinite_z"
                    return True
                if max_abs_z > 0.0 and abs(z) > max_abs_z:
                    self._last_fallen_reason = f"z_abs:{z:.3f}"
                    return True
                # Extra floor-drop guard: if the robot falls below the nominal
                # floor by a noticeable amount, reset even when abs(z) threshold
                # was set large for x/y boundary experiments.
                if z < -0.08:
                    self._last_fallen_reason = f"z_below_floor:{z:.3f}"
                    return True
        except Exception as exc:
            self._last_fallen_reason = f"z_check_error:{type(exc).__name__}"
            return False
        return False

    def _check_out_of_bounds(self) -> bool:
        """Return True when the robot leaves the configured safe training envelope."""
        self._last_out_of_bounds = False
        self._last_out_of_bounds_reason = "none"

        if not self.terminate_on_out_of_bounds:
            return False

        pose = self._get_robot_pose2d(frame_id=self.safety_boundary_frame)
        if pose is None:
            return False

        xy, _ = pose
        x = float(xy[0])
        y = float(xy[1])
        z = 0.0
        try:
            if self.ros.odom is not None:
                z = float(self.ros.odom.pose.pose.position.z)
        except Exception:
            z = 0.0

        self._last_out_of_bounds_x = x
        self._last_out_of_bounds_y = y
        self._last_out_of_bounds_z = z

        if not (np.isfinite(x) and np.isfinite(y) and np.isfinite(z)):
            self._last_out_of_bounds = True
            self._last_out_of_bounds_reason = "nonfinite_pose"
            return True

        reasons: list[str] = []
        if x < self.safety_boundary_min_x:
            reasons.append("x_min")
        if x > self.safety_boundary_max_x:
            reasons.append("x_max")
        if y < self.safety_boundary_min_y:
            reasons.append("y_min")
        if y > self.safety_boundary_max_y:
            reasons.append("y_max")
        if self.safety_boundary_max_abs_z > 0.0 and abs(z) > self.safety_boundary_max_abs_z:
            reasons.append("z_abs")

        if self.safety_boundary_radius_m > 0.0:
            center = np.asarray(self.current_boundary_center_xy, dtype=np.float32)
            radius = float(np.linalg.norm(np.asarray([x, y], dtype=np.float32) - center))
            self._last_out_of_bounds_radius = radius
            if radius > self.safety_boundary_radius_m:
                reasons.append("radius")
        else:
            self._last_out_of_bounds_radius = 0.0

        if reasons:
            self._last_out_of_bounds = True
            self._last_out_of_bounds_reason = "+".join(reasons)
            return True

        return False

    def _sample_goal(self) -> np.ndarray:
        return random.choice(self.goal_candidates)

    def _safe_sim_time(self) -> float:
        sim_time = self.ros.get_sim_time_sec()

        if sim_time is None:
            return -1.0

        return float(sim_time)

    def _get_robot_pose2d(self, frame_id: Optional[str] = None) -> Optional[tuple[np.ndarray, float]]:
        # v11: never use Gazebo model pose for map layers.  This returns the
        # same TF-based base pose that RViz uses, normally map->base_footprint.
        return self.ros.get_pose2d(frame_id=(frame_id or self.map_frame or self.pose_frame))

    def _update_boundary_center_after_reset(self, requested_xy: np.ndarray) -> None:
        # Give /odom callbacks a short chance to reflect the Gazebo teleport.
        self.ros.spin_steps(num_spins=12, timeout_sec=0.01)
        pose = self._get_robot_pose2d(frame_id=self.safety_boundary_frame)
        if pose is not None:
            xy, _ = pose
            self.current_boundary_center_xy = np.asarray(xy, dtype=np.float32).copy()
            source = "actual_pose"
        else:
            self.current_boundary_center_xy = np.asarray(requested_xy, dtype=np.float32).copy()
            source = "requested_pose_fallback"

        _boundary_msg = (
            "Safety boundary center updated: "
            f"frame={self.safety_boundary_frame}, "
            f"source={source}, "
            f"center=({self.current_boundary_center_xy[0]:.3f}, {self.current_boundary_center_xy[1]:.3f}), "
            f"requested=({float(requested_xy[0]):.3f}, {float(requested_xy[1]):.3f})"
        )
        if _quiet_reset_logs():
            self.ros.get_logger().debug(_boundary_msg)
        else:
            self.ros.get_logger().info(_boundary_msg)


    @staticmethod
    def _yaw_from_quaternion_xyzw_static(x: float, y: float, z: float, w: float) -> float:
        siny_cosp = 2.0 * (float(w) * float(z) + float(x) * float(y))
        cosy_cosp = 1.0 - 2.0 * (float(y) * float(y) + float(z) * float(z))
        return math.atan2(siny_cosp, cosy_cosp)

    @staticmethod
    def _yaw_to_quaternion_xyzw_static(yaw: float) -> tuple[float, float, float, float]:
        half = 0.5 * float(yaw)
        return 0.0, 0.0, math.sin(half), math.cos(half)

    def _transform_slam_map_to_frame(
        self,
        slam_map: OccupancyGrid,
        target_frame: str,
    ) -> Optional[OccupancyGrid]:
        """
        Convert a slam_toolbox OccupancyGrid into target_frame by transforming
        only the grid origin metadata.

        The cell array itself is unchanged.  OccupancyGrid cells are defined in
        the local coordinate system of info.origin; for a rigid SE(2) transform
        between frames, changing origin pose is sufficient.  This is the key
        invariant for odom-unified debugging: /map may be frame_id=map, but the
        RL task/confidence/priority maps are frame_id=odom, so the accepted SLAM
        prior and the RViz publication reference must also be expressed in odom.
        """
        if slam_map is None:
            return None

        source_frame = str(getattr(getattr(slam_map, "header", None), "frame_id", "") or "map").strip().lstrip("/") or "map"
        target_frame = str(target_frame or self.pose_frame or "odom").strip().lstrip("/") or "odom"
        if source_frame == target_frame:
            return slam_map

        tf_buffer = getattr(self.ros, "tf_buffer", None)
        if tf_buffer is None:
            self._last_slam_gate_reason = "missing_tf_buffer"
            return None

        last_exc = None
        transform = None

        cache_key_match = (
            bool(getattr(self, "odom_unified_frame_mode", False))
            and target_frame == "odom"
            and source_frame == "map"
            and self._episode_slam_transform is not None
            and self._episode_slam_transform_source == source_frame
            and self._episode_slam_transform_target == target_frame
        )
        if cache_key_match:
            transform = self._episode_slam_transform
        else:
            for _ in range(6):
                try:
                    # Capture the first valid map->odom transform after reset.
                    # Do not keep chasing latest TF in odom mode: that makes the
                    # SLAM-derived base grid move underneath persistent priority /
                    # confidence state and RViz layers appear to drift.
                    transform = tf_buffer.lookup_transform(
                        target_frame,
                        source_frame,
                        rclpy.time.Time(),
                    )
                    if (
                        bool(getattr(self, "odom_unified_frame_mode", False))
                        and target_frame == "odom"
                        and source_frame == "map"
                    ):
                        self._episode_slam_transform = transform
                        self._episode_slam_transform_source = source_frame
                        self._episode_slam_transform_target = target_frame
                    break
                except Exception as exc:  # TF exceptions differ across distros.
                    last_exc = exc
                    try:
                        self.ros.spin_steps(num_spins=2, timeout_sec=0.002)
                    except Exception:
                        pass
                    time.sleep(0.01)

        if transform is None:
            self._last_slam_gate_reason = "missing_slam_to_pose_frame_tf"
            self.ros.get_logger().warn(
                "SLAM_GRID_FRAME_TRANSFORM_MISSING | "
                f"source={source_frame} target={target_frame} err={last_exc}"
            )
            return None

        t = transform.transform.translation
        q = transform.transform.rotation
        tf_yaw = self._yaw_from_quaternion_xyzw_static(q.x, q.y, q.z, q.w)

        origin = slam_map.info.origin
        ox = float(origin.position.x)
        oy = float(origin.position.y)
        oyaw = self._yaw_from_quaternion_xyzw_static(
            origin.orientation.x,
            origin.orientation.y,
            origin.orientation.z,
            origin.orientation.w,
        )

        c = math.cos(tf_yaw)
        s = math.sin(tf_yaw)
        target_origin_x = float(t.x) + c * ox - s * oy
        target_origin_y = float(t.y) + s * ox + c * oy
        target_origin_yaw = self._normalize_angle(tf_yaw + oyaw)
        qx, qy, qz, qw = self._yaw_to_quaternion_xyzw_static(target_origin_yaw)

        out = deepcopy(slam_map)
        out.header.frame_id = target_frame
        out.info.origin.position.x = target_origin_x
        out.info.origin.position.y = target_origin_y
        out.info.origin.position.z = float(origin.position.z)
        out.info.origin.orientation.x = qx
        out.info.origin.orientation.y = qy
        out.info.origin.orientation.z = qz
        out.info.origin.orientation.w = qw
        return out

    def _transform_slam_map_to_odom(self, slam_map: OccupancyGrid) -> Optional[OccupancyGrid]:
        # Backward-compatible wrapper used by older patches/logging.
        return self._transform_slam_map_to_frame(slam_map, target_frame="odom")

    def _slam_transform_cache_key_for(self, slam_map: OccupancyGrid, target_frame: str):
        """Return an exact cache key for transforming a SLAM OccupancyGrid origin.

        The transformed OccupancyGrid is reused only when the raw map stamp,
        dimensions, resolution, origin metadata and target frame are identical.
        Therefore this is a performance cache, not a behavior change.
        """
        try:
            hdr = getattr(slam_map, "header", None)
            stamp = getattr(hdr, "stamp", None)
            info = slam_map.info
            origin = info.origin
            q = origin.orientation
            return (
                str(getattr(hdr, "frame_id", "") or ""),
                str(target_frame or ""),
                int(getattr(stamp, "sec", 0)) if stamp is not None else 0,
                int(getattr(stamp, "nanosec", 0)) if stamp is not None else 0,
                int(info.width),
                int(info.height),
                float(info.resolution),
                float(origin.position.x),
                float(origin.position.y),
                float(origin.position.z),
                float(getattr(q, "x", 0.0)),
                float(getattr(q, "y", 0.0)),
                float(getattr(q, "z", 0.0)),
                float(getattr(q, "w", 1.0)),
                id(getattr(slam_map, "data", None)),
                getattr(self, "_episode_slam_transform_source", ""),
                getattr(self, "_episode_slam_transform_target", ""),
            )
        except Exception:
            return None

    def _filtered_slam_map_for_update(self):
        """
        Return a SLAM map snapshot that is safe to inject into the RL base_grid.

        This is deliberately a gate in front of ExplorationGridMap rather than a
        publisher that overwrites /map. The original /map remains owned by
        slam_toolbox. RL priority/path/CNN use only accepted snapshots and publish
        the resulting filtered view on /rl_filtered_slam_map.
        """
        self._last_slam_map_age_sec = -1.0
        self._last_slam_map_delay_remaining_sec = 0.0

        if not self.use_slam_map:
            self._last_slam_gate_reason = "disabled"
            return None

        # If the reset service failed early, older code ignored SLAM for the whole
        # episode.  On the real robot, however, /map can appear a few seconds later
        # through the background map mirror.  Do not keep the episode permanently
        # map-blind once a fresh post-reset map has arrived.
        if self.ignore_slam_prior_this_episode:
            maybe_map = getattr(self.ros, "slam_map", None)
            maybe_time = getattr(self.ros, "last_slam_map_time", None)
            if maybe_map is not None and maybe_time is not None and float(maybe_time) >= float(getattr(self, "_slam_map_min_wall_time", 0.0)):
                self.ignore_slam_prior_this_episode = False
                self._last_slam_gate_reason = "late_map_recovered"
                if not bool(getattr(self, "_late_slam_recovery_logged", False)):
                    self._late_slam_recovery_logged = True
                    self.ros.get_logger().warn(
                        "SLAM_LATE_MAP_RECOVERED | reset had marked this episode map-blind, "
                        "but a fresh /map arrived; re-enabling SLAM prior for this episode"
                    )
            else:
                self._last_slam_gate_reason = "ignored_after_reset"
                return None

        # The raw slam_toolbox OccupancyGrid is often frame_id=map even when the
        # RL runtime frame is odom.  Never inject that grid directly into odom
        # maps.  After freshness checks below, transform the OccupancyGrid origin
        # into self.pose_frame and use that transformed map both for learning and
        # for RViz publication reference metadata.

        slam_map = self.ros.slam_map
        if slam_map is None:
            self._last_slam_gate_reason = "missing"
            return None

        now_wall = time.time()
        last_map_wall = getattr(self.ros, "last_slam_map_time", None)
        if last_map_wall is None:
            self._last_slam_gate_reason = "missing_stamp"
            return None

        age = now_wall - float(last_map_wall)
        self._last_slam_map_age_sec = float(age)

        if float(last_map_wall) < self._slam_map_min_wall_time:
            self._last_slam_gate_reason = "pre_reset"
            return None

        if self.slam_map_max_age_sec > 0.0 and age > self.slam_map_max_age_sec:
            # Do not stop map-layer alignment just because /map publication became
            # temporarily stale.  The latest /map metadata is still the only
            # valid canvas for /rl_confidence_map and /rl_priority_map.  Keep
            # using it for map growth/alignment while exposing the stale state in
            # the debug overlay.
            self._last_slam_gate_reason = "stale_using_last"
            # Continue with the latest cached slam_map instead of returning None.

        raw_slam_map_for_rviz = slam_map
        if hasattr(self.exploration_map, "set_slam_publish_reference"):
            self.exploration_map.set_slam_publish_reference(raw_slam_map_for_rviz)

        target_pose_frame = self.pose_frame
        cache_key = self._slam_transform_cache_key_for(slam_map, target_pose_frame)
        if cache_key is not None and cache_key == getattr(self, "_slam_transform_cache_key", None):
            slam_map_in_pose_frame = getattr(self, "_slam_transform_cache_msg", None)
        else:
            slam_map_in_pose_frame = self._transform_slam_map_to_frame(
                slam_map,
                target_frame=target_pose_frame,
            )
            if slam_map_in_pose_frame is not None:
                self._slam_transform_cache_key = cache_key
                self._slam_transform_cache_msg = slam_map_in_pose_frame
        if slam_map_in_pose_frame is None:
            return None
        slam_map = slam_map_in_pose_frame

        # The RViz publication reference was already taken from the raw /map
        # metadata before transform.  Do not overwrite it with the odom-transformed
        # learning map, otherwise /rl_priority_map and /map use different canvases.

        if now_wall < self._slam_map_accept_after_wall_time:
            self._last_slam_map_delay_remaining_sec = float(
                self._slam_map_accept_after_wall_time - now_wall
            )
            self._last_slam_gate_reason = "accept_delay"
            return None

        self._last_slam_gate_reason = "accepted"
        return slam_map

    def _apply_positive_reward_log_compression(self, reward: float) -> float:
        """Compress only positive non-terminal reward using sign-preserving log1p.

        Large early map-discovery bonuses can produce high-variance TD targets.
        Negative penalties are intentionally left unchanged so safety backup,
        collision-like penalties, and terminal penalties keep their full scale.
        
        Formula for r > 0:
            r' = log(1 + alpha * r) / alpha
        with an optional positive cap after compression.
        """
        try:
            r = float(reward)
        except Exception:
            return reward
        self._last_reward_pre_log_compress = r
        self._last_reward_post_log_compress = r
        self._last_reward_log_compress_delta = 0.0
        if not bool(getattr(self, "reward_positive_log_compress", False)):
            return r
        if not math.isfinite(r) or r <= 0.0:
            return r
        alpha = max(float(getattr(self, "reward_positive_log_alpha", 0.50)), 1e-6)
        out = float(math.log1p(alpha * r) / alpha)
        cap = max(float(getattr(self, "reward_positive_log_max", 0.0)), 0.0)
        if cap > 0.0:
            out = min(out, cap)
        self._last_reward_post_log_compress = out
        self._last_reward_log_compress_delta = out - r
        return out

    def _compute_delayed_slam_map_update_reward(self, stats: MapUpdateStats, unsafe_terminal: bool = False) -> float:
        """Small capped bonus for delayed SLAM /map unknown->known updates.

        Immediate scan/pose rewards are handled through confidence_gain.  This
        reward is intentionally smaller and capped because the SLAM OccupancyGrid
        can arrive one or more steps after the action that caused it.
        """
        self._last_slam_map_update_reward = 0.0
        self._last_slam_map_update_reward_raw = 0.0
        self._last_slam_map_update_reward_reason = "disabled"
        if not bool(getattr(self, "slam_map_update_reward", False)):
            return 0.0
        if unsafe_terminal:
            self._last_slam_map_update_reward_reason = "unsafe_terminal"
            return 0.0
        if int(getattr(self, "step_count", 0)) < int(getattr(self, "slam_map_update_reward_grace_steps", 10)):
            self._last_slam_map_update_reward_reason = "grace"
            return 0.0
        if not bool(getattr(self, "_last_post_reset_ready", True)):
            self._last_slam_map_update_reward_reason = "post_reset_not_ready"
            return 0.0
        new_known = max(int(getattr(stats, "slam_update_new_known_cells", 0)), 0)
        if new_known <= 0:
            self._last_slam_map_update_reward_reason = "no_new_known"
            return 0.0
        norm = max(float(getattr(self, "slam_map_update_reward_norm_cells", 80.0)), 1.0)
        cap = max(float(getattr(self, "slam_map_update_reward_cap", 1.0)), 0.0)
        weight = max(float(getattr(self, "slam_map_update_reward_weight", 0.20)), 0.0)
        raw = float(np.clip(float(new_known) / norm, 0.0, cap))
        bonus = float(weight * raw)
        self._last_slam_map_update_reward_raw = raw
        self._last_slam_map_update_reward = bonus
        self._last_slam_map_update_reward_reason = "new_known"
        self._episode_slam_map_update_reward += bonus
        slam_update_log_enabled = str(
            os.environ.get("TB3_RL_SLAM_MAP_UPDATE_REWARD_LOG", "0")
        ).strip().lower() in {"1", "true", "yes", "on"}
        if slam_update_log_enabled and bonus > 0.0 and int(getattr(self, "step_count", 0)) % 25 == 0:
            try:
                self.ros.get_logger().info(
                    "SLAM_MAP_UPDATE_REWARD | "
                    f"bonus={bonus:.3f} raw={raw:.3f} new_known={new_known} "
                    f"free={int(getattr(stats, 'slam_update_new_free_cells', 0))} "
                    f"occ={int(getattr(stats, 'slam_update_new_occupied_cells', 0))} "
                    f"expand={int(getattr(stats, 'slam_update_expand_known_cells', 0))}"
                )
            except Exception:
                pass
        return bonus

    def _accumulate_pending_slam_map_update_events(self, stats: MapUpdateStats) -> None:
        """Accumulate delayed /map update events produced by the live map timer."""
        if stats is None:
            return
        new_known = max(int(getattr(stats, "slam_update_new_known_cells", 0)), 0)
        if new_known <= 0:
            return
        self._pending_slam_update_new_known_cells += new_known
        self._pending_slam_update_new_free_cells += max(int(getattr(stats, "slam_update_new_free_cells", 0)), 0)
        self._pending_slam_update_new_occupied_cells += max(int(getattr(stats, "slam_update_new_occupied_cells", 0)), 0)
        self._pending_slam_update_expand_known_cells += max(int(getattr(stats, "slam_update_expand_known_cells", 0)), 0)
        self._pending_slam_update_count += 1

    def _reset_priority_event_accumulators(self) -> None:
        """Reset per-episode and pending priority clear/recheck accounting.

        The pending fields are drained once per Gym step.  They collect events
        produced by the live 10Hz map update timer while Nav2 is moving toward
        the current waypoint.  Episode fields are diagnostics: total priority
        work actually credited to rewards in this episode.
        """
        self._pending_priority_cleared_cells = 0
        self._pending_priority_clear_gain = 0.0
        self._pending_priority_rechecked_cells = 0
        self._pending_priority_rechecked_gain = 0.0
        self._pending_priority_update_count = 0
        self._pending_slam_update_new_known_cells = 0
        self._pending_slam_update_new_free_cells = 0
        self._pending_slam_update_new_occupied_cells = 0
        self._pending_slam_update_expand_known_cells = 0
        self._pending_slam_update_count = 0

        self._last_pending_priority_cleared_cells = 0
        self._last_pending_priority_clear_gain = 0.0
        self._last_pending_priority_rechecked_cells = 0
        self._last_pending_priority_rechecked_gain = 0.0
        self._last_pending_priority_update_count = 0
        self._last_pending_slam_update_new_known_cells = 0
        self._last_pending_slam_update_new_free_cells = 0
        self._last_pending_slam_update_new_occupied_cells = 0
        self._last_pending_slam_update_expand_known_cells = 0
        self._last_pending_slam_update_count = 0

        self._last_step_priority_cleared_cells = 0
        self._last_step_priority_clear_gain = 0.0
        self._last_step_priority_rechecked_cells = 0
        self._last_step_priority_rechecked_gain = 0.0

        self._episode_priority_cleared_cells = 0
        self._episode_priority_clear_gain = 0.0
        self._episode_priority_rechecked_cells = 0
        self._episode_priority_rechecked_gain = 0.0

    def _accumulate_pending_priority_events(self, stats: MapUpdateStats) -> None:
        """Accumulate live-timer priority clear/recheck events for next reward.

        MapUpdateStats is event-like: clear/recheck gains describe what happened
        in that specific update call.  Because live map updates are decoupled
        from SAC steps, these events must be buffered until step() computes the
        reward for the currently active waypoint/action.
        """
        if stats is None:
            return
        cleared_cells = max(int(getattr(stats, "priority_cleared_cells", 0)), 0)
        clear_gain = max(float(getattr(stats, "priority_clear_gain", 0.0)), 0.0)
        rechecked_cells = max(int(getattr(stats, "priority_rechecked_cells", 0)), 0)
        rechecked_gain = max(float(getattr(stats, "priority_rechecked_gain", 0.0)), 0.0)
        if cleared_cells <= 0 and clear_gain <= 0.0 and rechecked_cells <= 0 and rechecked_gain <= 0.0:
            return
        self._pending_priority_cleared_cells += cleared_cells
        self._pending_priority_clear_gain += clear_gain
        self._pending_priority_rechecked_cells += rechecked_cells
        self._pending_priority_rechecked_gain += rechecked_gain
        self._pending_priority_update_count += 1

    def _merge_and_drain_pending_priority_events(self, map_stats: MapUpdateStats) -> MapUpdateStats:
        """Return reward stats including all live priority events, then drain.

        This is the critical accounting point: a single SAC action may run Nav2
        for several seconds.  During that interval, the 10Hz live map timer may
        clear/recheck many priority cells.  The returned MapUpdateStats folds all
        those event gains into the final step reward so the policy is credited
        for the total priority removed while executing the waypoint.
        """
        pending_cleared = int(getattr(self, "_pending_priority_cleared_cells", 0))
        pending_clear_gain = float(getattr(self, "_pending_priority_clear_gain", 0.0))
        pending_rechecked = int(getattr(self, "_pending_priority_rechecked_cells", 0))
        pending_recheck_gain = float(getattr(self, "_pending_priority_rechecked_gain", 0.0))
        pending_updates = int(getattr(self, "_pending_priority_update_count", 0))
        pending_slam_known = max(int(getattr(self, "_pending_slam_update_new_known_cells", 0)), 0)
        pending_slam_free = max(int(getattr(self, "_pending_slam_update_new_free_cells", 0)), 0)
        pending_slam_occ = max(int(getattr(self, "_pending_slam_update_new_occupied_cells", 0)), 0)
        pending_slam_expand = max(int(getattr(self, "_pending_slam_update_expand_known_cells", 0)), 0)
        pending_slam_updates = int(getattr(self, "_pending_slam_update_count", 0))

        current_cleared = max(int(getattr(map_stats, "priority_cleared_cells", 0)), 0)
        current_clear_gain = max(float(getattr(map_stats, "priority_clear_gain", 0.0)), 0.0)
        current_rechecked = max(int(getattr(map_stats, "priority_rechecked_cells", 0)), 0)
        current_recheck_gain = max(float(getattr(map_stats, "priority_rechecked_gain", 0.0)), 0.0)

        total_cleared = current_cleared + pending_cleared
        total_clear_gain = current_clear_gain + pending_clear_gain
        total_rechecked = current_rechecked + pending_rechecked
        total_recheck_gain = current_recheck_gain + pending_recheck_gain

        self._last_pending_priority_cleared_cells = pending_cleared
        self._last_pending_priority_clear_gain = pending_clear_gain
        self._last_pending_priority_rechecked_cells = pending_rechecked
        self._last_pending_priority_rechecked_gain = pending_recheck_gain
        self._last_pending_priority_update_count = pending_updates
        self._last_pending_slam_update_new_known_cells = pending_slam_known
        self._last_pending_slam_update_new_free_cells = pending_slam_free
        self._last_pending_slam_update_new_occupied_cells = pending_slam_occ
        self._last_pending_slam_update_expand_known_cells = pending_slam_expand
        self._last_pending_slam_update_count = pending_slam_updates

        self._last_step_priority_cleared_cells = total_cleared
        self._last_step_priority_clear_gain = total_clear_gain
        self._last_step_priority_rechecked_cells = total_rechecked
        self._last_step_priority_rechecked_gain = total_recheck_gain

        self._episode_priority_cleared_cells += total_cleared
        self._episode_priority_clear_gain += total_clear_gain
        self._episode_priority_rechecked_cells += total_rechecked
        self._episode_priority_rechecked_gain += total_recheck_gain

        self._pending_priority_cleared_cells = 0
        self._pending_priority_clear_gain = 0.0
        self._pending_priority_rechecked_cells = 0
        self._pending_priority_rechecked_gain = 0.0
        self._pending_priority_update_count = 0
        self._pending_slam_update_new_known_cells = 0
        self._pending_slam_update_new_free_cells = 0
        self._pending_slam_update_new_occupied_cells = 0
        self._pending_slam_update_expand_known_cells = 0
        self._pending_slam_update_count = 0

        total_slam_known = max(int(getattr(map_stats, "slam_update_new_known_cells", 0)), 0) + pending_slam_known
        total_slam_free = max(int(getattr(map_stats, "slam_update_new_free_cells", 0)), 0) + pending_slam_free
        total_slam_occ = max(int(getattr(map_stats, "slam_update_new_occupied_cells", 0)), 0) + pending_slam_occ
        total_slam_expand = max(int(getattr(map_stats, "slam_update_expand_known_cells", 0)), 0) + pending_slam_expand

        if (
            total_cleared == current_cleared
            and total_clear_gain == current_clear_gain
            and total_rechecked == current_rechecked
            and total_recheck_gain == current_recheck_gain
            and total_slam_known == max(int(getattr(map_stats, "slam_update_new_known_cells", 0)), 0)
            and total_slam_free == max(int(getattr(map_stats, "slam_update_new_free_cells", 0)), 0)
            and total_slam_occ == max(int(getattr(map_stats, "slam_update_new_occupied_cells", 0)), 0)
            and total_slam_expand == max(int(getattr(map_stats, "slam_update_expand_known_cells", 0)), 0)
        ):
            return map_stats

        return replace(
            map_stats,
            priority_cleared_cells=total_cleared,
            priority_clear_gain=total_clear_gain,
            priority_rechecked_cells=total_rechecked,
            priority_rechecked_gain=total_recheck_gain,
            slam_update_new_known_cells=total_slam_known,
            slam_update_new_free_cells=total_slam_free,
            slam_update_new_occupied_cells=total_slam_occ,
            slam_update_expand_known_cells=total_slam_expand,
        )

    def _live_map_update_timer_callback(self) -> None:
        """Refresh confidence/priority maps at a fixed wall-clock rate.

        This callback is intentionally decoupled from SAC waypoint generation.
        It is executed by rclpy spin callbacks while Nav2 is moving, so RViz and
        the internal priority/confidence state update every ~map_live_update_period_sec
        instead of only when env.step() returns with a new waypoint.
        """
        if self._map_live_update_paused or self._map_live_update_busy:
            return
        if self.ros.scan is None or self.ros.odom is None:
            return
        now = time.time()
        if self.map_live_update_period_sec > 0.0:
            if now - self._last_live_map_update_wall < 0.80 * self.map_live_update_period_sec:
                return
        self._map_live_update_busy = True
        try:
            prev_stats = self.last_map_stats
            stats = self._update_exploration_map()
            # Accumulate every timer-generated priority clear/recheck event.
            # Priority must be cleared continuously from LiDAR at 10Hz, not only
            # when a new waypoint is sampled. The next env.step() drains this
            # accumulator and credits the SAC action with the total weighted
            # priority removed since the previous reward computation.
            if stats is not prev_stats:
                self._accumulate_pending_priority_events(stats)
                self._accumulate_pending_slam_map_update_events(stats)
            self.last_map_stats = stats
            self._last_live_map_update_wall = now
            self._last_live_map_update_count += 1
        except Exception as exc:
            # Do not kill training because an RViz/debug refresh failed. Throttle.
            if now - self._last_live_map_update_error_wall > 2.0:
                self._last_live_map_update_error_wall = now
                self.ros.get_logger().warn(f"LIVE_MAP_UPDATE_FAILED | err={exc}")
        finally:
            self._map_live_update_busy = False

    def _wait_for_action_synced_observation(
        self,
        prev_scan_wall_time: Optional[float],
        prev_odom_wall_time: Optional[float],
    ) -> None:
        """Wait briefly for a post-action scan/odom frame before reward update.

        This implements the practical sync plan: do not wait for SLAM /map
        publication on every SAC step, but do require the immediate sensors that
        explain the action outcome.  ExplorationGridMap.update() then ray-casts
        this scan into the internal confidence/priority state, and reward.py sees
        that step-local delta.
        """
        self._last_action_sync_ok = False
        self._last_action_sync_reason = "disabled"
        self._last_action_sync_wait_sec = 0.0
        self._last_action_sync_scan_fresh = False
        self._last_action_sync_odom_fresh = False

        if not bool(getattr(self, "action_sync_reward_gate", True)):
            return

        start = time.time()
        timeout = float(getattr(self, "action_sync_wait_timeout_sec", 0.06))
        deadline = start + max(timeout, 0.0)
        scan_fresh = False
        odom_fresh = False

        def check_fresh() -> tuple[bool, bool]:
            sf = (
                prev_scan_wall_time is not None
                and getattr(self.ros, "last_scan_time", None) is not None
                and float(self.ros.last_scan_time) > float(prev_scan_wall_time) + float(getattr(self, "action_sync_min_scan_age_sec", 0.0))
            )
            of = (
                prev_odom_wall_time is not None
                and getattr(self.ros, "last_odom_time", None) is not None
                and float(self.ros.last_odom_time) > float(prev_odom_wall_time)
            )
            return bool(sf), bool(of)

        while time.time() <= deadline:
            self.ros.spin_steps(num_spins=4, timeout_sec=0.001)
            scan_fresh, odom_fresh = check_fresh()
            # For reward credit assignment, scan freshness is the important one;
            # pose can still be obtained from the latest odom/TF.  Prefer both,
            # but do not block indefinitely when SLAM/odom callbacks are slower.
            if scan_fresh and (odom_fresh or prev_odom_wall_time is None):
                break
            time.sleep(0.002)

        self._last_action_sync_wait_sec = float(time.time() - start)
        self._last_action_sync_scan_fresh = bool(scan_fresh)
        self._last_action_sync_odom_fresh = bool(odom_fresh)
        if scan_fresh:
            self._last_action_sync_ok = True
            self._last_action_sync_reason = "fresh_scan" if not odom_fresh else "fresh_scan_odom"
        else:
            self._last_action_sync_ok = False
            self._last_action_sync_reason = "scan_stale"
            if int(getattr(self, "step_count", 0)) % 250 == 0:
                self.ros.get_logger().warn(
                    "ACTION_SYNC_STALE | reward will use latest available scan/pose; "
                    f"wait={self._last_action_sync_wait_sec:.3f}s "
                    f"scan_fresh={scan_fresh} odom_fresh={odom_fresh}"
                )

    def _map_bounds_metrics(self) -> dict:
        metrics = {
            "inside": False,
            "near_edge": False,
            "reason": "unknown",
            "known_ratio": 0.0,
            "known_cells": 0,
            "outside_cells": 0,
        }
        try:
            if self.exploration_map is None:
                metrics["reason"] = "missing_exploration_map"
                return metrics
            emap = self.exploration_map

            # Prefer the exact pose that the confidence/priority update just used
            # on THIS canvas.  Falling back to a freshly fetched TF pose can read
            # a slightly different stamp/anchor than the one the canvas was locked
            # to, which (right after a SLAM canvas grow/origin-shift) yields a
            # negative or out-of-range index even though the robot is inside the
            # map.  That mismatch is the root cause of both the spurious
            # pose_outside_rl_map restart and the confidence-origin "jump".
            conf_xy = getattr(self, "_last_confidence_update_xy", None)
            conf_step = int(getattr(self, "_last_confidence_update_step", -10_000_000))
            use_conf_pose = (
                isinstance(conf_xy, np.ndarray)
                and conf_xy.size >= 2
                and bool(np.all(np.isfinite(conf_xy[:2])))
                and (int(getattr(self, "step_count", 0)) - conf_step) <= 1
            )

            pose = self._get_robot_pose2d(frame_id=self.pose_frame)
            if pose is None and not use_conf_pose:
                metrics["reason"] = "missing_pose"
                return metrics

            if use_conf_pose:
                robot_xy = np.asarray(conf_xy[:2], dtype=np.float32)
            else:
                robot_xy, _ = pose

            # If the exploration map already recorded the cell it painted the
            # robot at this update, trust that index: it is guaranteed consistent
            # with the current canvas origin/size.  Only fall back to recomputing
            # via world_to_map when that record is missing or stale.
            rec_ix = getattr(emap, "_last_robot_ix", None)
            rec_iy = getattr(emap, "_last_robot_iy", None)
            if use_conf_pose and isinstance(rec_ix, (int, np.integer)) and isinstance(rec_iy, (int, np.integer)):
                rix, riy = int(rec_ix), int(rec_iy)
            else:
                rix, riy = emap.world_to_map(float(robot_xy[0]), float(robot_xy[1]))
            margin = max(int(getattr(self, "map_bounds_margin_cells", 2)), 0)
            inside = bool(emap.in_bounds(int(rix), int(riy)))
            metrics["inside"] = inside
            if not inside:
                # Quantify how far outside the canvas the index is, in cells.
                # A robot that is genuinely off the map is many cells out; a
                # transient canvas/pose mismatch is typically only a few cells
                # past an edge (often a small negative index).  The restart logic
                # uses this to avoid killing the episode on a brief mismatch.
                ox = 0
                oy = 0
                if int(rix) < 0:
                    ox = -int(rix)
                elif int(rix) >= int(emap.width):
                    ox = int(rix) - int(emap.width) + 1
                if int(riy) < 0:
                    oy = -int(riy)
                elif int(riy) >= int(emap.height):
                    oy = int(riy) - int(emap.height) + 1
                metrics["outside_cells"] = int(max(ox, oy))
                metrics["reason"] = f"pose_outside_rl_map:ix={int(rix)},iy={int(riy)},size={int(emap.width)}x{int(emap.height)}"
                return metrics

            near_edge = bool(
                int(rix) < margin
                or int(riy) < margin
                or int(rix) >= int(emap.width) - margin
                or int(riy) >= int(emap.height) - margin
            )
            metrics["near_edge"] = near_edge

            # Local map-valid metric.  Unknown-only/outside-like crops are a bad
            # training state even when the index is technically inside the canvas.
            radius_m = max(float(getattr(self, "map_obs_size_m", 6.0)) * 0.35, 1.0)
            radius_cells = max(2, int(math.ceil(radius_m / max(float(emap.resolution), 1e-6))))
            x0 = max(0, int(rix) - radius_cells)
            x1 = min(int(emap.width), int(rix) + radius_cells + 1)
            y0 = max(0, int(riy) - radius_cells)
            y1 = min(int(emap.height), int(riy) + radius_cells + 1)
            if x1 <= x0 or y1 <= y0:
                metrics["reason"] = "empty_local_crop"
                return metrics
            local = np.asarray(emap.base_grid[y0:y1, x0:x1])
            conf = np.asarray(emap.confidence_grid[y0:y1, x0:x1])
            known_map = local >= 0
            known_conf = conf >= max(float(getattr(emap, "min_known_confidence", 1.0)), 1.0)
            known = int(np.count_nonzero(known_map | known_conf))
            ratio = float(known) / float(local.size) if local.size else 0.0
            metrics["known_cells"] = int(known)
            metrics["known_ratio"] = float(ratio)
            if near_edge:
                metrics["reason"] = "near_rl_map_edge"
            elif known < int(getattr(self, "map_bounds_min_local_known_cells", 12)):
                metrics["reason"] = "local_known_cells_low"
            elif ratio < float(getattr(self, "map_bounds_min_local_known_ratio", 0.04)):
                metrics["reason"] = "local_known_ratio_low"
            else:
                metrics["reason"] = "ok"
            return metrics
        except Exception as exc:
            metrics["reason"] = f"map_bounds_error:{type(exc).__name__}"
            return metrics

    def _update_map_bounds_restart_state(self, map_stats: Optional[MapUpdateStats] = None) -> bool:
        self._last_map_bounds_restart = False
        self._last_map_bounds_reason = "none"
        if not bool(getattr(self, "map_bounds_restart", True)):
            self.map_bounds_bad_steps = 0
            return False
        metrics = self._map_bounds_metrics()
        reason = str(metrics.get("reason", "unknown"))
        known_ratio = float(metrics.get("known_ratio", 0.0))
        known_cells = int(metrics.get("known_cells", 0))
        outside_cells = int(metrics.get("outside_cells", 0))
        self._last_map_bounds_reason = reason
        self._last_map_bounds_local_known_ratio = known_ratio
        self._last_map_bounds_local_known_cells = known_cells

        # How many cells past an edge still counts as a transient canvas/pose
        # mismatch rather than the robot truly leaving the map.  A brief mismatch
        # right after a SLAM /map grow/origin-shift is only a few cells out and
        # must NOT instantly end the episode; it recovers within a step or two
        # once confidence re-locks to the new canvas.
        transient_outside_tol = max(int(getattr(self, "map_bounds_transient_outside_cells", 24)), 0)

        is_outside = reason.startswith("pose_outside_rl_map")
        transient_outside = bool(is_outside and outside_cells <= transient_outside_tol)
        genuine_outside = bool(is_outside and not transient_outside)

        # Only a clearly out-of-map pose (or a missing/empty local crop) is an
        # immediate hard restart now.  A small out-of-range index and being near
        # the edge are treated as soft/recoverable and must persist for the grace
        # window before they end the episode.
        hard_bad = genuine_outside or reason in {"missing_pose", "empty_local_crop"}
        soft_bad = reason in {"local_known_cells_low", "local_known_ratio_low", "near_rl_map_edge"} or transient_outside
        # No target and no map delta for many steps is effectively the same
        # failure mode: the robot is training in an uninformative outside-map
        # state.  Do not wait until max_episode_steps.
        try:
            no_target = str(getattr(map_stats, "target_type", "none")) in {"none", "unknown", ""}
            no_prio = float(getattr(map_stats, "priority_score", 0.0)) <= 1e-5
            no_delta = abs(float(getattr(map_stats, "coverage_delta", 0.0))) <= 1e-6 and float(getattr(map_stats, "confidence_gain", 0.0)) <= 1e-6
            very_stale = float(getattr(map_stats, "stale_ratio", 0.0)) >= 0.98
            soft_bad = bool(soft_bad or (no_target and no_prio and no_delta and very_stale))
            if soft_bad and reason == "ok":
                reason = "no_map_reward_signal"
                self._last_map_bounds_reason = reason
        except Exception:
            pass

        if hard_bad or soft_bad:
            self.map_bounds_bad_steps += 1
        else:
            self.map_bounds_bad_steps = 0

        grace = max(int(getattr(self, "map_bounds_grace_steps", 8)), 0)
        restart = bool(hard_bad or self.map_bounds_bad_steps >= grace)
        if restart:
            self._last_map_bounds_restart = True
            self._last_map_bounds_reason = reason
            self.ros.get_logger().warn(
                "MAP_BOUNDS_RESTART | "
                f"reason={reason} bad={self.map_bounds_bad_steps}/{grace} "
                f"known={known_cells} ratio={known_ratio:.3f} "
                f"outside_cells={outside_cells} transient_tol={transient_outside_tol}"
            )
        return restart

    def _get_scan_pose2d_for_map_update(self) -> Optional[tuple[np.ndarray, float]]:
        """Return LiDAR frame pose in the map frame at the scan timestamp.

        Confidence/priority ray-casting must use the same map-frame scan origin
        that RViz uses for /scan.  The robot base pose is still used for visit
        count, crop center and actor observation centering.
        """
        try:
            scan = self.ros.scan
            if scan is None:
                return None
            header = getattr(scan, "header", None)
            scan_frame = str(getattr(header, "frame_id", "") or "base_scan").strip().lstrip("/") or "base_scan"
            stamp = getattr(header, "stamp", None)
            if hasattr(self.ros, "get_frame_pose2d"):
                return self.ros.get_frame_pose2d(
                    target_frame=str(self.map_frame or "map"),
                    source_frame=scan_frame,
                    stamp=stamp,
                    timeout_sec=0.04,
                    allow_latest_fallback=True,
                )
        except Exception:
            return None
        return None

    def _update_exploration_map_with_unified_tf(self, slam_map=None) -> MapUpdateStats:
        """Single owner for confidence/priority/task map update frames.

        All persistent layers live in map_frame and all coordinates are generated
        through TF:
          - base pose:  map -> base_footprint/base_link
          - scan pose:  map -> scan.header.frame_id, at scan timestamp
          - grid:       latest accepted /map metadata
        """
        if self.ros.scan is None or self.ros.odom is None:
            return self.last_map_stats

        # v8 pose anchor rule:
        #   - confidence/priority/task layers live in /map and publish on the same
        #     SLAM OccupancyGrid canvas.
        #   - use actual odometry deltas (/model odom preferred, /odom msg fallback)
        #     anchored once to the current map pose.
        #   - no cmd_vel integration.  Set TB3_RL_CONFIDENCE_POSE_SOURCE=map_base_tf
        #     to force TF(map -> base_footprint) on every confidence update.
        pose_pack = self._get_confidence_update_pose2d()
        if pose_pack is None:
            return self.last_map_stats
        robot_xy, robot_yaw, sensor_xy, sensor_yaw, confidence_pose_mode = pose_pack

        # Record the exact base pose that confidence/priority were painted at.
        # The map-bounds restart check must evaluate the robot against the SAME
        # pose source and the SAME canvas that confidence used; otherwise a
        # transient mismatch between this pose and a separately-fetched TF pose
        # (especially right after the SLAM /map canvas grows or its origin shifts)
        # maps the robot to a negative cell index (e.g. iy=-12) and triggers a
        # spurious pose_outside_rl_map restart.  It is also what makes the
        # /rl_confidence_origin marker appear to "jump".
        try:
            self._last_confidence_update_xy = np.asarray(robot_xy, dtype=np.float32).copy()
            self._last_confidence_update_yaw = float(robot_yaw)
            self._last_confidence_update_step = int(getattr(self, "step_count", 0))
        except Exception:
            pass

        # v108 diagnostic: show the exact pose/yaw used by confidence in RViz.
        # Compare /rl_confidence_origin against RobotModel in Fixed Frame=map.
        try:
            self._publish_confidence_origin_marker(
                robot_xy=robot_xy,
                robot_yaw=robot_yaw,
                sensor_xy=sensor_xy,
                sensor_yaw=sensor_yaw,
                label=str(confidence_pose_mode),
            )
        except Exception:
            pass

        # Optional legacy scan-pose origin path.  Default remains base-anchored:
        # confidence is conceptually camera/front FOV, but the ray starts at the
        # robot anchor so the grid does not jump due to tiny sensor-frame offsets.
        if not bool(getattr(self, "use_base_pose_for_raycast", True)):
            scan_pose = self._get_scan_pose2d_for_map_update()
            if scan_pose is not None:
                sensor_xy, sensor_yaw = scan_pose

        if slam_map is None:
            slam_map = self._filtered_slam_map_for_update()

        try:
            stats = self.exploration_map.update(
                scan=self.ros.scan,
                robot_xy=robot_xy,
                robot_yaw=robot_yaw,
                publish=True,
                slam_map=slam_map,
                sensor_xy=sensor_xy,
                sensor_yaw=sensor_yaw,
            )
        except TypeError as exc:
            # Do not silently fall back to the old robot_yaw-only update path.
            # If this fires, colcon/symlink-install is still loading an older
            # ExplorationGridMap and confidence direction/pose will be wrong.
            try:
                self.ros.get_logger().error(
                    "CONFIDENCE_UPDATE_API_MISMATCH | "
                    f"ExplorationGridMap.update() does not accept sensor_xy/sensor_yaw: {exc}"
                )
            except Exception:
                pass
            return self.last_map_stats

        try:
            if str(os.environ.get("TB3_RL_FORCE_MAP_PUBLISH_EVERY_UPDATE", "0")).strip().lower() in {"1", "true", "yes", "on"}:
                self.exploration_map.publish()
        except Exception:
            pass
        return stats

    def _update_exploration_map(self) -> MapUpdateStats:
        """Canonical camera-front confidence update path.

        v4 rule:
          - Do not use Odometry/Gazebo fallback for semantic map writes.
          - Use the same TF(map -> base_footprint) pose that RViz uses for the robot.
          - v111 default: use base_footprint yaw for the camera-front confidence ray;
            base_scan yaw remains optional via TB3_RL_CONFIDENCE_RAY_YAW_SOURCE=scan.

        This fixes the failure mode where /rl_confidence_map keeps being painted
        around the first-step pose while the RViz robot has already moved/rotated.
        """
        return self._update_exploration_map_with_unified_tf()

    @staticmethod
    def _normalize_angle(angle_rad: float) -> float:
        return math.atan2(math.sin(float(angle_rad)), math.cos(float(angle_rad)))

    @staticmethod
    def _scan_min_distance_in_sector(
        scan,
        center_angle: float,
        half_width_rad: float,
        max_considered_range: float = 1.20,
    ) -> float:
        """
        LaserScan에서 center_angle 주변 sector의 최소 range를 구한다.

        ROS LaserScan 관례상 angle=0은 로봇 전방, +angle은 좌측이다.
        반환값은 meter 단위이며, 유효한 beam이 없으면 max_considered_range를 반환한다.
        """
        if scan is None:
            return float(max_considered_range)

        ranges = np.asarray(scan.ranges, dtype=np.float32)
        if ranges.size == 0:
            return float(max_considered_range)

        angle_min = float(getattr(scan, "angle_min", -math.pi))
        angle_increment = float(getattr(scan, "angle_increment", 0.0))
        if not np.isfinite(angle_increment) or abs(angle_increment) < 1e-9:
            return float(max_considered_range)

        range_min = float(getattr(scan, "range_min", 0.05))
        range_max = float(getattr(scan, "range_max", max_considered_range))
        usable_max = min(
            float(max_considered_range),
            range_max if np.isfinite(range_max) else max_considered_range,
        )

        idx = np.arange(ranges.size, dtype=np.float32)
        angles = angle_min + idx * angle_increment
        angle_error = np.arctan2(
            np.sin(angles - float(center_angle)),
            np.cos(angles - float(center_angle)),
        )

        valid = np.isfinite(ranges) & (ranges >= max(range_min, 0.03)) & (ranges <= usable_max)
        sector = np.abs(angle_error) <= float(half_width_rad)
        mask = valid & sector
        if not np.any(mask):
            return float(max_considered_range)

        return float(np.min(ranges[mask]))

    def _compute_slam_action_obstacle_distance(self, action: np.ndarray) -> tuple[float, float]:
        """Return (action_dir_dist, front_dist) from the current SLAM structural grid.

        LaserScan is the first safety source, but /map can contain a wall that is
        outside the current sparse LiDAR sector or temporarily hidden by scan noise.
        This lightweight ray check makes the velocity safety shield use both.
        """
        try:
            pose = self._get_robot_pose2d(frame_id=self.pose_frame)
            if pose is None or self.exploration_map is None:
                return 999.0, 999.0
            robot_xy, robot_yaw = pose
            emap = self.exploration_map
            rix, riy = emap.world_to_map(float(robot_xy[0]), float(robot_xy[1]))
            if not emap.in_bounds(rix, riy):
                return 999.0, 999.0
            struct = emap._structural_grid()
            occ_thr = emap._slam_occupied_threshold()

            action = np.asarray(action, dtype=np.float32)
            linear_x = float(action[0]) if action.size > 0 else 0.0
            angular_z = float(action[1]) if action.size > 1 else 0.0
            arc_angle, ok = self._commanded_arc_angle(linear_x, angular_z, horizon_sec=0.45)
            if not ok:
                arc_angle = 0.0

            def cast(center_rel: float, half_width_rad: float, max_range_m: float) -> float:
                max_cells = max(int(math.ceil(float(max_range_m) / max(float(emap.resolution), 1e-6))), 1)
                samples = 7
                best = 999.0
                for rel in np.linspace(center_rel - half_width_rad, center_rel + half_width_rad, samples):
                    yaw = float(robot_yaw) + float(rel)
                    ex = float(robot_xy[0]) + float(max_range_m) * math.cos(yaw)
                    ey = float(robot_xy[1]) + float(max_range_m) * math.sin(yaw)
                    eix, eiy = emap.world_to_map(ex, ey)
                    cells = emap.bresenham(int(rix), int(riy), int(eix), int(eiy))
                    for idx, (cx, cy) in enumerate(cells[: max_cells + 1]):
                        if idx == 0:
                            continue
                        if not emap.in_bounds(cx, cy):
                            continue
                        if struct[int(cy), int(cx)] >= occ_thr:
                            d = math.hypot(int(cx) - int(rix), int(cy) - int(riy)) * float(emap.resolution)
                            best = min(best, float(d))
                            break
                return best

            turn_norm = float(np.clip(abs(angular_z) / max(float(self.max_angular_speed), 1e-6), 0.0, 1.0))
            action_half = math.radians(18.0 + 18.0 * turn_norm)
            action_d = cast(float(arc_angle), action_half, 1.20)
            front_d = cast(0.0, math.radians(25.0), 1.20)
            return float(action_d), float(front_d)
        except Exception:
            return 999.0, 999.0

    def _compute_lidar_action_obstacle_risk(self, action: np.ndarray) -> tuple[float, float, float]:
        """
        현재 LaserScan 기준으로, 이번 action이 향하는 방향에 장애물이 가까우면 위험도를 계산한다.

        반환:
          action_min_dist:
            command arc 방향 sector 안의 최소 LiDAR 거리.
          action_risk:
            [0, 1]. 0이면 안전, 1이면 매우 위험.
          front_min_dist:
            정면 sector 안의 최소 LiDAR 거리.

        이 값은 observation 차원을 바꾸지 않고 reward에만 들어간다.
        따라서 처음부터 새로 학습할 때 구조 mismatch를 만들지 않는다.
        """
        scan = self.ros.scan
        if scan is None:
            return 999.0, 0.0, 999.0

        action = np.asarray(action, dtype=np.float32)
        linear_x = float(action[0]) if action.size > 0 else 0.0
        angular_z = float(action[1]) if action.size > 1 else 0.0

        forward_norm = float(
            np.clip(max(linear_x, 0.0) / max(float(self.max_linear_speed), 1e-6), 0.0, 1.0)
        )
        if forward_norm <= 0.02:
            # 거의 정지/제자리 회전은 벽으로 밀고 들어가는 행동이 아니다.
            front_min = self._scan_min_distance_in_sector(
                scan=scan,
                center_angle=0.0,
                half_width_rad=math.radians(25.0),
                max_considered_range=1.20,
            )
            return 999.0, 0.0, float(front_min)

        arc_angle, ok = self._commanded_arc_angle(linear_x, angular_z, horizon_sec=0.45)
        if not ok:
            arc_angle = 0.0

        # 고속/회전 action일수록 실제 swept volume이 넓어지므로 sector도 약간 넓힌다.
        turn_norm = float(
            np.clip(abs(angular_z) / max(float(self.max_angular_speed), 1e-6), 0.0, 1.0)
        )
        half_width = math.radians(18.0 + 18.0 * turn_norm)

        action_min = self._scan_min_distance_in_sector(
            scan=scan,
            center_angle=arc_angle,
            half_width_rad=half_width,
            max_considered_range=1.20,
        )
        front_min = self._scan_min_distance_in_sector(
            scan=scan,
            center_angle=0.0,
            half_width_rad=math.radians(25.0),
            max_considered_range=1.20,
        )

        # 0.60m부터 경고, 0.22m 이하는 최대 위험.
        warn_distance = 0.60
        hard_distance = 0.22
        risk = (warn_distance - float(action_min)) / max(warn_distance - hard_distance, 1e-6)
        risk = float(np.clip(risk, 0.0, 1.0))

        # forward speed가 클수록 같은 거리도 더 위험하다.
        risk *= float(0.35 + 0.65 * forward_norm)
        risk = float(np.clip(risk, 0.0, 1.0))

        return float(action_min), risk, float(front_min)

    @staticmethod
    def _commanded_arc_angle(linear_x: float, angular_z: float, horizon_sec: float = 0.35) -> tuple[float, bool]:
        v = max(float(linear_x), 0.0)
        w = float(angular_z)
        if v < 1e-3:
            return 0.0, False
        t = max(float(horizon_sec), 1e-3)
        wt = w * t
        if abs(w) < 1e-5:
            return 0.0, True
        dx = (v / w) * math.sin(wt)
        dy = (v / w) * (1.0 - math.cos(wt))
        if abs(dx) < 1e-9 and abs(dy) < 1e-9:
            return 0.0, False
        return float(np.clip(math.atan2(dy, dx), -math.radians(75.0), math.radians(75.0))), True

    def _current_action_priority_error(self, map_stats: MapUpdateStats) -> tuple[float, float, float]:
        """
        Debug metric: executed command arc vs priority/frontier target bearing.

        - frontier_angle is robot-relative: 0 = robot front, + = left, - = right.
        - commanded_arc_angle is the short-horizon direction implied by (v, w).
        - alignment uses the same Gaussian idea as reward.py's corridor block.
        """
        if int(getattr(map_stats, "frontier_count", 0)) <= 0:
            return 0.0, 0.0, 0.0

        action = np.asarray(self.prev_action, dtype=np.float32)
        arc_angle, ok = self._commanded_arc_angle(float(action[0]), float(action[1]))
        if not ok:
            return float(getattr(map_stats, "frontier_angle", 0.0)), 0.0, 0.0

        err = self._normalize_angle(float(getattr(map_stats, "frontier_angle", 0.0)) - arc_angle)
        align = math.exp(-0.5 * (err / max(math.radians(24.0), 1e-6)) ** 2)

        threshold_cos = math.cos(math.radians(60.0))
        c = math.cos(err)
        if c >= threshold_cos:
            signed = (c - threshold_cos) / max(1.0 - threshold_cos, 1e-6)
        else:
            signed = -((threshold_cos - c) / max(threshold_cos + 1.0, 1e-6))
        signed = float(np.clip(signed, -1.0, 1.0))
        return float(err), float(align), signed

    def _estimate_priority_check_reward_component(self, map_stats: MapUpdateStats) -> tuple[float, float, float]:
        """Estimate the reward.py priority clear/recheck component for logging.

        compute_exploration_reward() returns only the total scalar reward.  For debugging
        whether clearing priority cells actually contributes to return, reproduce only
        the priority-clear/recheck terms here and accumulate them in info/logs.
        This is not an extra reward path; it is telemetry for the already-applied reward.
        """
        try:
            target_priority_norm = float(np.clip(float(getattr(map_stats, "target_priority", 0.0)), 0.0, 1.0))
            priority_score_norm = float(np.clip(float(getattr(map_stats, "priority_score", 0.0)), 0.0, 1.0))
            priority_clear_gain = float(getattr(map_stats, "priority_clear_gain", 0.0))
            priority_cleared_cells = int(getattr(map_stats, "priority_cleared_cells", 0))
            priority_rechecked_gain = float(getattr(map_stats, "priority_rechecked_gain", 0.0))
            priority_rechecked_cells = int(getattr(map_stats, "priority_rechecked_cells", 0))
        except Exception:
            return 0.0, 0.0, 0.0

        priority_strength = float(np.clip(max(target_priority_norm, priority_score_norm), 0.0, 1.0))
        priority_check_multiplier = 0.75 + 1.75 * priority_strength
        corridor_priority_weight = max(float(getattr(self, "corridor_priority_reward_weight", 1.0)), 0.0)

        priority_clear_weighted_sum = max(priority_clear_gain, 0.0)

        # Must mirror reward.py approximately.  This is telemetry only, not an
        # extra reward path.  reward.py additionally multiplies by priority_motion_gate.
        clear_reward = (
            corridor_priority_weight
            * 0.045
            * priority_clear_weighted_sum
            * priority_check_multiplier
        )
        clear_reward += 0.0015 * corridor_priority_weight * float(max(priority_cleared_cells, 0))
        clear_reward = min(float(clear_reward), 5.0)

        # v114: recheck is no longer a positive reward path.
        recheck_reward = -min(
            0.35,
            0.003 * max(float(priority_rechecked_gain), 0.0)
            + 0.00008 * min(max(int(priority_rechecked_cells), 0), 400),
        ) if (priority_rechecked_gain > 0.0 or priority_rechecked_cells > 0) else 0.0

        total = float(clear_reward + recheck_reward)
        return float(clear_reward), float(recheck_reward), total

    def _build_info(
        self,
        map_stats: MapUpdateStats,
        collision: bool,
        fallen: bool,
        coverage_done: bool,
    ) -> dict:
        priority_direction_error, priority_direction_alignment, priority_direction_signed = self._current_action_priority_error(map_stats)
        return {
            "coverage_ratio": float(map_stats.coverage_ratio),
            "episode_reward_sum": float(getattr(self, "_episode_reward_sum", 0.0)),
            "episode_discounted_return": float(getattr(self, "_episode_discounted_return", 0.0)),
            "episode_live_discounted_return": float(getattr(self, "_episode_discounted_return", 0.0)),
            "episode_start_discounted_return": float(getattr(self, "_episode_start_discounted_return", 0.0)),
            "episode_reward_ema": float(getattr(self, "_episode_reward_ema", 0.0)),
            "recent_reward_window_sum": float(sum(getattr(self, "_recent_reward_window", []))),
            "recent_slam_new_cells": int(sum(getattr(self, "_recent_slam_new_window", []))),
            "recent_confidence_updated_cells": int(sum(getattr(self, "_recent_conf_update_window", []))),
            "coverage_stall_terminal_enabled": bool(getattr(self, "coverage_stall_terminal", False)),
            "coverage_stall_terminal": bool(getattr(self, "_last_coverage_stall_terminal", False)),
            "coverage_stall_active": bool(getattr(self, "_last_coverage_stall_active", False)),
            "coverage_stall_reason": str(getattr(self, "_last_coverage_stall_reason", "none")),
            "coverage_stall_window_steps": int(getattr(self, "coverage_stall_window_steps", 0)),
            "coverage_stall_window_len": int(getattr(self, "_last_coverage_stall_window_len", 0)),
            "coverage_stall_slam_new_cells": int(getattr(self, "_last_coverage_stall_slam_window", 0)),
            "coverage_stall_confidence_updated_cells": int(getattr(self, "_last_coverage_stall_conf_window", 0)),
            "coverage_stall_min_slam_new_cells": int(getattr(self, "coverage_stall_min_slam_new_cells", 0)),
            "coverage_stall_min_confidence_updated_cells": int(getattr(self, "coverage_stall_min_confidence_updated_cells", 0)),
            "coverage_stall_terminal_penalty": float(getattr(self, "coverage_stall_terminal_penalty", 0.0)),
            "reward_window_n": int(getattr(self, "_reward_window_n", 100)),
            "reward_gamma": float(getattr(self, "reward_gamma", 0.99)),
            "coverage_delta": float(map_stats.coverage_delta),
            "new_known_cells": int(map_stats.new_known_cells),
            "known_cells": int(map_stats.known_cells),
            "frontier_count": int(map_stats.frontier_count),
            "frontier_distance": float(map_stats.frontier_distance),
            "frontier_angle": float(map_stats.frontier_angle),
            "robot_visit_count": int(map_stats.robot_visit_count),
            "explored_stall_steps": int(self.explored_stall_steps),
            "confidence_stall_steps": int(getattr(self, "confidence_stall_steps", 0)),
            "sustained_rotation_steps": int(getattr(self, "sustained_rotation_steps", 0)),
            "orbit_stall_steps": int(getattr(self, "orbit_stall_steps", 0)),
            "orbit_path_efficiency": float(getattr(self, "_last_orbit_path_efficiency", 1.0)),
            "orbit_path_length": float(getattr(self, "_last_orbit_path_length", 0.0)),
            "orbit_net_displacement": float(getattr(self, "_last_orbit_net_displacement", 0.0)),
            "orbit_yaw_accum": float(getattr(self, "_last_orbit_yaw_accum", 0.0)),
            "orbit_reason": str(getattr(self, "_last_orbit_reason", "none")),
            "mean_confidence": float(map_stats.mean_confidence),
            "stale_known_cells": int(map_stats.stale_known_cells),
            "stale_ratio": float(map_stats.stale_ratio),
            "low_confidence_cells": int(map_stats.low_confidence_cells),
            "low_confidence_ratio": float(map_stats.low_confidence_ratio),
            "stale_refresh_cells": int(map_stats.stale_refresh_cells),
            "confidence_gain": float(map_stats.confidence_gain),
            "target_priority": float(map_stats.target_priority),
            "target_type": str(map_stats.target_type),
            "target_switched": bool(map_stats.target_switched),
            "target_lock_age": int(map_stats.target_lock_age),
            "priority_stuck_steps": int(getattr(self, "priority_stuck_steps", 0)),
            "priority_stuck_restart_steps": int(getattr(self, "priority_stuck_restart_steps", 0)),
            "priority_stuck_active": bool(getattr(self, "_last_priority_stuck_active", False)),
            "priority_stuck_restart": bool(getattr(self, "_last_priority_stuck_restart", False)),
            "priority_stuck_reason": str(getattr(self, "_last_priority_stuck_reason", "none")),
            "lidar_empty_steps": int(getattr(self, "lidar_empty_steps", 0)),
            "lidar_empty_timeout_steps": int(getattr(self, "lidar_empty_timeout_steps", 0)),
            "lidar_empty_active": bool(getattr(self, "_last_lidar_empty_active", False)),
            "lidar_empty_restart": bool(getattr(self, "_last_lidar_empty_restart", False)),
            "lidar_empty_reason": str(getattr(self, "_last_lidar_empty_reason", "none")),
            "nav2_stuck_steps": int(getattr(self, "nav2_stuck_steps", 0)),
            "nav2_stuck_backup_steps": int(getattr(self, "nav2_stuck_backup_steps", 0)),
            "nav2_stuck_active": bool(getattr(self, "_last_nav2_stuck_active", False)),
            "nav2_stuck_backup_triggered": bool(getattr(self, "_last_nav2_stuck_backup_triggered", False)),
            "nav2_stuck_reason": str(getattr(self, "_last_nav2_stuck_reason", "none")),
            "nav2_backup_status": str(getattr(self, "_last_nav2_backup_status", "none")),
            "lidar_valid_beams": int(getattr(self, "_last_lidar_valid_beams", 0)),
            "lidar_nearest_detection": float(getattr(self, "_last_lidar_nearest_detection", 999.0)),
            "target_reachable": bool(map_stats.target_reachable),
            "disable_path_reward": True,
            "path_reward_enabled": False,
            "path_distance": 0.0,
            "path_angle": 0.0,
            "path_progress": 0.0,
            "alternative_path_count": 0,
            "alternative_path_angles": (),
            "priority_direction_error": float(priority_direction_error),
            "priority_direction_alignment": float(priority_direction_alignment),
            "priority_direction_signed": float(priority_direction_signed),
            "action_mode": str(self.action_mode),
            "policy_action_0": float(self.raw_action[0]) if self.raw_action.size > 0 else 0.0,
            "policy_action_1": float(self.raw_action[1]) if self.raw_action.size > 1 else 0.0,
            "executed_linear_x": float(self.prev_action[0]) if self.prev_action.size > 0 else 0.0,
            "executed_angular_z": float(self.prev_action[1]) if self.prev_action.size > 1 else 0.0,
            "velocity_spin_breaker": bool(getattr(self, "_last_velocity_spin_breaker", False)),
            "velocity_spin_same_sign_steps": int(getattr(self, "_velocity_spin_same_sign_steps", 0)),
            "waypoint_local_x": float(self._last_waypoint_local[0]),
            "waypoint_local_y": float(self._last_waypoint_local[1]),
            "waypoint_world_x": float(self._last_waypoint_world[0]),
            "waypoint_world_y": float(self._last_waypoint_world[1]),
            "waypoint_distance": float(self._last_waypoint_distance),
            "waypoint_angle": float(self._last_waypoint_angle),
            "waypoint_action_type": str(self._last_waypoint_action_type),
            "waypoint_lateral_offset": float(self._last_waypoint_lateral_offset),
            "waypoint_heading_delta": float(self._last_waypoint_heading_delta),
            "waypoint_reached": bool(self._last_waypoint_reached),
            "waypoint_timed_out": bool(self._last_waypoint_timed_out),
            "waypoint_timeout_sec": float(self.waypoint_timeout_sec),
            "waypoint_final_error": float(self._last_waypoint_final_error),
            "controller_steps": int(self._last_controller_steps),
            "nav2_goal_accepted": bool(getattr(self, "_last_nav2_goal_accepted", False)),
            "nav2_status": int(getattr(self, "_last_nav2_status", -1)),
            "nav2_status_name": str(getattr(self, "_last_nav2_status_name", "none")),
            "nav2_goal_source": str(getattr(self, "_last_nav2_goal_source", "none")),
            "nav2_goal_valid": bool(getattr(self, "_last_nav2_goal_valid", False)),
            "nav2_goal_validation": str(getattr(self, "_last_nav2_goal_validation", "none")),
            "nav2_moved_distance": float(getattr(self, "_last_nav2_moved_distance", 0.0)),
            "nav2_replan_distance_m": float(getattr(self, "nav2_replan_distance_m", 0.0)),
            "slam_local_known_ratio": float(getattr(self, "_last_slam_local_known_ratio", 1.0)),
            "slam_front_known_ratio": float(getattr(self, "_last_slam_front_known_ratio", 1.0)),
            "slam_fresh_score": float(getattr(self, "_last_slam_fresh_score", 1.0)),
            "slam_local_linear_score": float(getattr(self, "_last_slam_local_linear_score", 1.0)),
            "slam_front_linear_score": float(getattr(self, "_last_slam_front_linear_score", 1.0)),
            "slam_fresh_linear_score": float(getattr(self, "_last_slam_fresh_linear_score", 1.0)),
            "slam_quality_score": float(getattr(self, "_last_slam_quality_score", 1.0)),
            "slam_speed_raw_scale": float(getattr(self, "_last_slam_speed_raw_scale", 1.0)),
            "slam_speed_scale": float(getattr(self, "_last_slam_speed_scale", 1.0)),
            "slam_speed_limit": float(getattr(self, "_last_slam_speed_limit", self.max_linear_speed)),
            "priority_score": float(map_stats.priority_score),
            "priority_gain": float(map_stats.priority_gain),
            "priority_cleared_cells": int(map_stats.priority_cleared_cells),
            "priority_clear_gain": float(map_stats.priority_clear_gain),
            "priority_step_cleared_cells": int(getattr(self, "_last_step_priority_cleared_cells", int(map_stats.priority_cleared_cells))),
            "priority_step_clear_gain": float(getattr(self, "_last_step_priority_clear_gain", float(map_stats.priority_clear_gain))),
            "priority_live_cleared_cells": int(getattr(self, "_last_pending_priority_cleared_cells", 0)),
            "priority_live_clear_gain": float(getattr(self, "_last_pending_priority_clear_gain", 0.0)),
            "priority_live_update_count": int(getattr(self, "_last_pending_priority_update_count", 0)),
            "episode_priority_cleared_cells": int(getattr(self, "_episode_priority_cleared_cells", 0)),
            "episode_priority_clear_gain": float(getattr(self, "_episode_priority_clear_gain", 0.0)),
            "priority_clear_reward": float(getattr(self, "_last_priority_clear_reward", 0.0)),
            "priority_recheck_reward": float(getattr(self, "_last_priority_recheck_reward", 0.0)),
            "priority_check_reward": float(getattr(self, "_last_priority_check_reward", 0.0)),
            "episode_priority_clear_reward": float(getattr(self, "_episode_priority_clear_reward", 0.0)),
            "episode_priority_recheck_reward": float(getattr(self, "_episode_priority_recheck_reward", 0.0)),
            "episode_priority_check_reward": float(getattr(self, "_episode_priority_check_reward", 0.0)),
            "priority_invalidated_cells": int(getattr(map_stats, 'priority_invalidated_cells', 0)),
            "priority_invalidated_gain": float(getattr(map_stats, 'priority_invalidated_gain', 0.0)),
            "priority_rechecked_cells": int(getattr(map_stats, 'priority_rechecked_cells', 0)),
            "priority_rechecked_gain": float(getattr(map_stats, 'priority_rechecked_gain', 0.0)),
            "priority_step_rechecked_cells": int(getattr(self, "_last_step_priority_rechecked_cells", int(getattr(map_stats, 'priority_rechecked_cells', 0)))),
            "priority_step_rechecked_gain": float(getattr(self, "_last_step_priority_rechecked_gain", float(getattr(map_stats, 'priority_rechecked_gain', 0.0)))),
            "priority_live_rechecked_cells": int(getattr(self, "_last_pending_priority_rechecked_cells", 0)),
            "priority_live_rechecked_gain": float(getattr(self, "_last_pending_priority_rechecked_gain", 0.0)),
            "episode_priority_rechecked_cells": int(getattr(self, "_episode_priority_rechecked_cells", 0)),
            "episode_priority_rechecked_gain": float(getattr(self, "_episode_priority_rechecked_gain", 0.0)),
            "wall_support_score": float(map_stats.wall_support_score),
            "open_space_score": float(map_stats.open_space_score),
            "nearest_obstacle_distance": float(map_stats.nearest_obstacle_distance),
            "obstacle_proximity_score": float(map_stats.obstacle_proximity_score),
            "lidar_action_obstacle_distance": float(getattr(self, "_last_lidar_action_obstacle_distance", 999.0)),
            "lidar_action_obstacle_score": float(getattr(self, "_last_lidar_action_obstacle_score", 0.0)),
            "lidar_front_obstacle_distance": float(getattr(self, "_last_lidar_front_obstacle_distance", 999.0)),
            "velocity_safety_backup_triggered": bool(getattr(self, "_last_velocity_safety_backup_triggered", False)),
            "velocity_safety_blocked": bool(getattr(self, "_last_velocity_safety_blocked", False)),
            "velocity_safety_slowdown": float(getattr(self, "_last_velocity_safety_slowdown", 1.0)),
            "velocity_safety_slowdown_risk": float(getattr(self, "_last_velocity_safety_slowdown_risk", 0.0)),
            "velocity_safety_policy_v": float(getattr(self, "_last_velocity_safety_policy_v", 0.0)),
            "velocity_safety_executed_v": float(getattr(self, "_last_velocity_safety_executed_v", 0.0)),
            "velocity_safety_penalty": float(getattr(self, "_last_velocity_safety_penalty", 0.0)),
            "velocity_safety_reason": str(getattr(self, "_last_velocity_safety_reason", "none")),
            "velocity_safety_cooldown_steps": int(getattr(self, "velocity_safety_cooldown_steps", 0)),
            "shake_steps": int(getattr(self, "shake_steps", 0)),
            "shake_restart_steps": int(getattr(self, "shake_restart_steps_limit", 0)),
            "shake_active": bool(getattr(self, "_last_shake_active", False)),
            "shake_restart": bool(getattr(self, "_last_shake_restart", False)),
            "shake_reason": str(getattr(self, "_last_shake_reason", "none")),
            "collision": bool(collision),
            "out_of_bounds": bool(getattr(self, "_last_out_of_bounds", False)),
            "out_of_bounds_reason": str(getattr(self, "_last_out_of_bounds_reason", "none")),
            "out_of_bounds_radius": float(getattr(self, "_last_out_of_bounds_radius", 0.0)),
            "out_of_bounds_x": float(getattr(self, "_last_out_of_bounds_x", 0.0)),
            "out_of_bounds_y": float(getattr(self, "_last_out_of_bounds_y", 0.0)),
            "out_of_bounds_z": float(getattr(self, "_last_out_of_bounds_z", 0.0)),
            "safety_boundary_frame": str(getattr(self, "safety_boundary_frame", "odom")),
            "safety_boundary_center_x": float(getattr(self, "current_boundary_center_xy", np.zeros(2))[0]),
            "safety_boundary_center_y": float(getattr(self, "current_boundary_center_xy", np.zeros(2))[1]),
            "reset_target_x": float(getattr(self, "current_reset_xy", np.zeros(2))[0]),
            "reset_target_y": float(getattr(self, "current_reset_xy", np.zeros(2))[1]),
            "restart_on_collision": bool(self.restart_on_collision),
            "collision_restart_requested": bool(self._last_collision_restart_requested),
            "terminal_reason": str(self._last_terminal_reason),
            "fallen": bool(fallen),
            "fallen_reason": str(getattr(self, "_last_fallen_reason", "none")),
            "collision_global_min": float(getattr(self, "_last_collision_global_min", 999.0)),
            "collision_front_min": float(getattr(self, "_last_collision_front_min", 999.0)),
            "coverage_done": bool(coverage_done),
            "step_count": int(self.step_count),
            "sim_time": self._safe_sim_time(),
            "episode_index": int(getattr(self, "episode_index", 0)),
            "use_slam_map": bool(self.use_slam_map),
            "map_frame": str(self.map_frame),
            "pose_frame": str(self.pose_frame),
            "reset_slam_on_reset": bool(self.reset_slam_on_reset),
            "reset_slam_every_n_episodes": int(self.reset_slam_every_n_episodes),
            "slam_map_available": bool(self.ros.slam_map is not None),
            "slam_map_gate": str(self._last_slam_gate_reason),
            "slam_map_age_sec": float(self._last_slam_map_age_sec),
            "slam_map_delay_remaining_sec": float(self._last_slam_map_delay_remaining_sec),
            "post_reset_ready": bool(getattr(self, "_last_post_reset_ready", False)),
            "post_reset_ready_reason": str(getattr(self, "_last_post_reset_ready_reason", "none")),
            "post_reset_ready_known_ratio": float(getattr(self, "_last_post_reset_ready_known_ratio", 0.0)),
            "post_reset_ready_known_cells": int(getattr(self, "_last_post_reset_ready_known_cells", 0)),
            "post_reset_ready_lidar_beams": int(getattr(self, "_last_post_reset_ready_lidar_beams", 0)),
            "post_reset_ready_priority": float(getattr(self, "_last_post_reset_ready_priority", 0.0)),
            "action_sync_ok": bool(getattr(self, "_last_action_sync_ok", False)),
            "action_sync_reason": str(getattr(self, "_last_action_sync_reason", "none")),
            "action_sync_wait_sec": float(getattr(self, "_last_action_sync_wait_sec", 0.0)),
            "action_sync_scan_fresh": bool(getattr(self, "_last_action_sync_scan_fresh", False)),
            "action_sync_odom_fresh": bool(getattr(self, "_last_action_sync_odom_fresh", False)),
            "map_bounds_restart": bool(getattr(self, "_last_map_bounds_restart", False)),
            "map_bounds_reason": str(getattr(self, "_last_map_bounds_reason", "none")),
            "map_bounds_bad_steps": int(getattr(self, "map_bounds_bad_steps", 0)),
            "map_bounds_local_known_ratio": float(getattr(self, "_last_map_bounds_local_known_ratio", 0.0)),
            "map_bounds_local_known_cells": int(getattr(self, "_last_map_bounds_local_known_cells", 0)),
            "slam_update_new_known_cells": int(getattr(map_stats, "slam_update_new_known_cells", 0)),
            "slam_update_new_free_cells": int(getattr(map_stats, "slam_update_new_free_cells", 0)),
            "slam_update_new_occupied_cells": int(getattr(map_stats, "slam_update_new_occupied_cells", 0)),
            "slam_update_expand_known_cells": int(getattr(map_stats, "slam_update_expand_known_cells", 0)),
            "slam_map_update_reward": float(getattr(self, "_last_slam_map_update_reward", 0.0)),
            "slam_map_update_reward_raw": float(getattr(self, "_last_slam_map_update_reward_raw", 0.0)),
            "slam_map_update_reward_reason": str(getattr(self, "_last_slam_map_update_reward_reason", "none")),
            "slam_map_update_reward_episode": float(getattr(self, "_episode_slam_map_update_reward", 0.0)),
            "slam_live_update_new_known_cells": int(getattr(self, "_last_pending_slam_update_new_known_cells", 0)),
            "slam_live_update_count": int(getattr(self, "_last_pending_slam_update_count", 0)),
            "debug_input_map": bool(self.debug_input_map and self.use_map_cnn),
            "debug_input_map_published": bool(getattr(self, "_last_debug_input_map_published", False)),
            "debug_input_map_topic_prefix": str(getattr(self, "debug_input_map_topic_prefix", "")),
            "debug_input_map_frame_id": str(getattr(self, "debug_input_map_frame_id", "")),
            # v117: when a forced safety backup/hold ran this step, mark the
            # transition so the custom replay buffer skips storing it.  SAC must
            # not learn from forced-recovery motion that the policy did not choose.
            "skip_store": bool(getattr(self, "_last_velocity_safety_skip_store", False)),
            "velocity_safety_skip_store": bool(getattr(self, "_last_velocity_safety_skip_store", False)),
        }

        return info

    @staticmethod
    def _empty_map_stats() -> MapUpdateStats:
        return MapUpdateStats(
            known_cells=0,
            new_known_cells=0,
            coverage_ratio=0.0,
            coverage_delta=0.0,
            frontier_count=0,
            frontier_distance=5.0,
            frontier_angle=0.0,
            robot_visit_count=0,
            mean_confidence=0.0,
            stale_known_cells=0,
            stale_ratio=0.0,
            low_confidence_cells=0,
            low_confidence_ratio=0.0,
            stale_refresh_cells=0,
            confidence_gain=0.0,
            target_priority=0.0,
            target_type="none",
            target_switched=False,
            target_lock_age=0,
            target_reachable=False,
            path_distance=5.0,
            path_angle=0.0,
            path_progress=0.0,
            alternative_path_count=0,
            alternative_path_angles=(),
            priority_score=0.0,
            priority_gain=0.0,
            priority_cleared_cells=0,
            priority_clear_gain=0.0,
            priority_invalidated_cells=0,
            priority_invalidated_gain=0.0,
            priority_rechecked_cells=0,
            priority_rechecked_gain=0.0,
            wall_support_score=0.0,
            open_space_score=0.0,
            nearest_obstacle_distance=999.0,
            obstacle_proximity_score=0.0,
        )
