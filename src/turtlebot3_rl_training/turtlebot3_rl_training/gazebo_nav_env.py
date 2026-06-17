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
from geometry_msgs.msg import Point, PoseStamped
from nav_msgs.msg import OccupancyGrid, Path as NavPath
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
    compute_waypoint_macro_reward_adjustment,
)
from turtlebot3_rl_training.sim_controller import GazeboSimController


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
        action_sync_wait_timeout_sec: float = 0.18,
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
        shake_restart_steps: int = 8,
        shake_tilt_threshold: float = 0.22,
        shake_angular_xy_threshold: float = 1.80,
        shake_linear_z_threshold: float = 0.22,
        shake_z_deviation_threshold: float = 0.12,
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
        self._world_step_stale_count = 0
        # /rl_path based planning/reward is removed.  Keep the attribute for old
        # callers, but force it disabled at runtime.
        self.disable_path_reward = True
        # Nav2-only training still needs a dense safety signal.  Nav2 owns /cmd_vel,
        # but the critic must see wall/obstacle risk before a hard collision terminal.
        self.disable_wall_proximity_penalty = False
        self.disable_priority_map = bool(disable_priority_map)
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
        self.rl_priority_topic = str(rl_priority_topic).strip()
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

        self.num_lidar_bins = 360
        self.obs_extra_dim = 10
        self.obs_dim = self.num_lidar_bins + self.obs_extra_dim

        self.exploration_map = ExplorationGridMap(
            node=self.ros,
            resolution=0.05,
            size_m=8.0,
            origin_x=-4.0,
            origin_y=-4.0,
            frame_id=self.pose_frame,
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
            publish_slam_aligned=bool(self.use_slam_map),
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
                    shape=(5, self.map_obs_size, self.map_obs_size),
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

        self.waypoint_path_pub = None
        if self.waypoint_path_topic:
            self.waypoint_path_pub = self.ros.create_publisher(
                NavPath,
                self.waypoint_path_topic,
                10,
            )

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
        self._episode_discounted_return = 0.0
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
        self.velocity_safety_cooldown_steps = 0
        self._last_velocity_safety_backup_triggered = False
        self._last_velocity_safety_blocked = False
        self._last_velocity_safety_slowdown = 1.0
        self._last_velocity_safety_penalty = 0.0
        self._last_velocity_safety_reason = "none"

        self.shake_restart = bool(shake_restart)
        self.shake_restart_steps_limit = max(int(shake_restart_steps), 1)
        self.shake_tilt_threshold = max(float(shake_tilt_threshold), 0.01)
        self.shake_angular_xy_threshold = max(float(shake_angular_xy_threshold), 0.01)
        self.shake_linear_z_threshold = max(float(shake_linear_z_threshold), 0.01)
        self.shake_z_deviation_threshold = max(float(shake_z_deviation_threshold), 0.01)
        self.shake_restart_penalty = max(float(shake_restart_penalty), 0.0)
        self.reset_hard_stabilize_reapply = bool(reset_hard_stabilize_reapply)
        self.reset_hard_stabilize_reapply_interval_sec = max(float(reset_hard_stabilize_reapply_interval_sec), 0.05)
        self.shake_steps = 0
        self._last_shake_active = False
        self._last_shake_restart = False
        self._last_shake_reason = "none"
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
            f"obs={self.obs_dim} map_cnn={self.use_map_cnn} map=(5,{self.map_obs_size},{self.map_obs_size}) | "
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

        This version waits for an actual stable window, not only for a fixed
        duration.  If Gazebo leaves residual roll/pitch/angular-x/y motion after
        SetEntityPose, the same pose is re-applied while /cmd_vel is held at zero
        until the instantaneous shake detector reports `none` continuously.
        """
        sec = float(getattr(self, "post_reset_stabilize_sec", 0.0))
        if sec <= 0.0:
            return
        stable_window_sec = 0.70
        max_sec = max(sec + 3.0, 4.0)
        self.ros.get_logger().info(
            f"POST_RESET_STABILIZE | holding /cmd_vel=0 for {sec:.2f}s before episode start "
            f"| stable_window={stable_window_sec:.2f}s max={max_sec:.2f}s"
        )
        start = time.time()
        next_reapply = start
        stable_since = None
        last_reason = "none"
        ever_unstable = False
        while time.time() - start < max_sec:
            self.ros.stop_robot()
            now = time.time()
            if (
                bool(getattr(self, "reset_hard_stabilize_reapply", True))
                and reset_pose is not None
                and self.reset_manager is not None
                and now >= next_reapply
            ):
                try:
                    self.reset_manager.reset_to_pose(reset_pose, timeout_sec=0.45)
                except Exception:
                    pass
                # Re-apply faster than before while unstable.  This is the only
                # available way with SetEntityPose to damp residual body wobble.
                next_reapply = time.time() + min(
                    float(getattr(self, "reset_hard_stabilize_reapply_interval_sec", 0.25)),
                    0.18,
                )
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

        # If the robot was wobbling/spinning during stabilization, do a second
        # flat teleport after it has calmed down.  SetEntityPose can leave
        # residual body twist; the second reset after a zero-cmd settle is the
        # most reliable way to remove roll/pitch and angular-x/y motion.
        if (
            bool(getattr(self, "reset_hard_stabilize_reapply", True))
            and reset_pose is not None
            and self.reset_manager is not None
            and ever_unstable
        ):
            self.ros.get_logger().warn(
                f"POST_RESET_SECOND_FLAT_RESET | first_stabilize_reason={last_reason}; "
                "re-applying flat reset pose after zero-cmd settle"
            )
            try:
                self.ros.stop_robot()
                self.reset_manager.reset_to_pose(reset_pose, timeout_sec=0.80)
            except Exception:
                pass
            second_start = time.time()
            second_stable_since = None
            while time.time() - second_start < max(1.20, min(2.50, sec)):
                self.ros.stop_robot()
                self.ros.spin_steps(num_spins=int(getattr(self, "post_reset_stabilize_spin_steps", 12)), timeout_sec=0.002)
                reason2 = self._instantaneous_shake_reason()
                last_reason = reason2
                if reason2 == "none":
                    if second_stable_since is None:
                        second_stable_since = time.time()
                    if time.time() - second_stable_since >= 0.45:
                        break
                else:
                    second_stable_since = None
                    try:
                        if time.time() - second_start > 0.35:
                            self.reset_manager.reset_to_pose(reset_pose, timeout_sec=0.45)
                            second_start = time.time() - 0.10
                    except Exception:
                        pass
                time.sleep(0.03)

        # One final quiet hold after the last pose reapply.
        final_hold_start = time.time()
        while time.time() - final_hold_start < 0.25:
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
                # Do not hide an unstable reset from the first training step.
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

        if metrics["lidar_beams"] < min_beams:
            metrics["reason"] = f"lidar_warmup:{metrics['lidar_beams']}/{min_beams}"
            return metrics
        if metrics["known_cells"] < min_cells and metrics["known_ratio"] < min_ratio:
            metrics["reason"] = f"map_warmup:known={metrics['known_cells']},ratio={metrics['known_ratio']:.3f}"
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
        raise RuntimeError(msg)

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
                pose = self._get_robot_pose2d()
                if pose is not None:
                    robot_xy, robot_yaw = pose
                    return self.exploration_map.update(
                        scan=self.ros.scan,
                        robot_xy=robot_xy,
                        robot_yaw=robot_yaw,
                        publish=True,
                        slam_map=accepted,
                    )
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
                pose = self._get_robot_pose2d()
                if pose is not None:
                    robot_xy, robot_yaw = pose
                    try:
                        last_stats = self.exploration_map.update(
                            scan=self.ros.scan,
                            robot_xy=robot_xy,
                            robot_yaw=robot_yaw,
                            publish=True,
                            slam_map=slam_map,
                        )
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
                    f"beams={metrics.get('lidar_beams')} | prio={float(metrics.get('priority_score', 0.0)):.3f}"
                )
                return last_stats

            now = time.time()
            if now - last_log >= 0.75:
                last_log = now
                self.ros.get_logger().info(
                    "POST_RESET_READY_WAITING | "
                    f"reason={metrics.get('reason')} | slam={metrics.get('slam_gate')} | "
                    f"known={metrics.get('known_cells')},ratio={float(metrics.get('known_ratio', 0.0)):.3f} | "
                    f"beams={metrics.get('lidar_beams')} | prio={float(metrics.get('priority_score', 0.0)):.3f}"
                )
            time.sleep(0.03)

        if bool(getattr(self, "strict_slam_map_required", False)):
            self.ros.stop_robot()
            metrics = best_metrics or {"reason": "timeout_no_metrics"}
            msg = (
                "POST_RESET_READY_STRICT_TIMEOUT | "
                f"reason={metrics.get('reason')} slam={metrics.get('slam_gate')} "
                f"known={metrics.get('known_cells')},ratio={float(metrics.get('known_ratio', 0.0)):.3f} "
                f"beams={metrics.get('lidar_beams')} prio={float(metrics.get('priority_score', 0.0)):.3f}. "
                "Policy step blocked because SLAM map is required."
            )
            self.ros.get_logger().error(msg)
            raise RuntimeError(msg)

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
            f"prio={float(metrics.get('priority_score', 0.0)):.3f}"
        )
        return soft_ready_stats if soft_ready_stats is not None else last_stats

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
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
        self._last_velocity_safety_backup_triggered = False
        self._last_velocity_safety_blocked = False
        self._last_velocity_safety_slowdown = 1.0
        self._last_velocity_safety_penalty = 0.0
        self._last_velocity_safety_reason = "none"
        self.shake_steps = 0
        self._last_shake_active = False
        self._last_shake_restart = False
        self._last_shake_reason = "none"
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

        # 5) RL memory/confidence map도 전부 초기화하고 Burger가 중앙에 오도록 origin을 재설정한다.
        robot_pose = self._get_robot_pose2d()

        if robot_pose is not None:
            robot_xy, _ = robot_pose
            self.exploration_map.reset_centered_at(robot_xy)
        else:
            self.exploration_map.reset_centered_at(self.current_reset_xy.copy())

        # 6) RESET 직후에는 바로 episode를 열지 않는다.
        #    SLAM /map이 reset 이후 실제로 채워지고, LiDAR/confidence/priority가
        #    최소 기준을 만족할 때까지 /cmd_vel=0으로 대기한다. Gym reset()이
        #    아직 반환되지 않았으므로 이 구간은 reward/replay에 들어가지 않는다.
        self.last_map_stats = self._wait_post_reset_ready(reset_pose=reset_pose)
        self._last_live_map_update_wall = time.time()
        self._map_live_update_paused = False

        obs = self._get_obs()

        info = self._build_info(
            map_stats=self.last_map_stats,
            collision=self._check_collision(),
            fallen=self._check_fallen(),
            coverage_done=False,
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

        map_stats = self._update_exploration_map()
        self._update_explored_stall_steps(map_stats=map_stats, action=action_for_reward)
        self._update_confidence_stall_steps(map_stats=map_stats, action=action_for_reward)
        self._update_sustained_rotation_steps(action=action_for_reward)
        self._update_orbit_stall_steps(map_stats=map_stats, action=action_for_reward)

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

        # Delayed SLAM /map update bonus.  This is intentionally separate from
        # the immediate LiDAR/confidence reward: it credits unknown->known cells
        # that arrive later through slam_toolbox, but it is small/capped because
        # exact action credit is less precise than the post-action scan reward.
        reward += self._compute_delayed_slam_map_update_reward(
            reward_map_stats,
            unsafe_terminal=bool(collision_like or fallen or lidar_empty_restart),
        )

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
            reward -= float(getattr(self, "_last_velocity_safety_penalty", 0.0))
        if shake_restart:
            reward -= float(getattr(self, "shake_restart_penalty", 100.0))
        if lidar_empty_restart:
            reward = -float(self.lidar_empty_restart_penalty)

        # Time-limit is not a neutral ending in exploration.  Without this,
        # a policy can prefer a low-risk local loop with many small negative
        # rewards over taking informative Nav2 goals.  Penalize only non-success
        # timeouts; collision/fallen/out-of-bounds are already -100 hard terminals.
        will_truncate = bool((self.step_count + 1) >= self.max_episode_steps)
        if will_truncate and not (collision_like or fallen or coverage_done or priority_stuck_restart or lidar_empty_restart):
            target_cov = max(float(self.target_coverage_ratio), 1e-6)
            missing_cov = float(np.clip((target_cov - float(reward_map_stats.coverage_ratio)) / target_cov, 0.0, 1.0))
            stall_norm = float(np.clip(float(self.explored_stall_steps) / 80.0, 0.0, 1.0))
            timeout_penalty = 35.0 + 45.0 * missing_cov + 15.0 * stall_norm
            reward -= timeout_penalty
            self._last_terminal_reason = "timeout_low_coverage"

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
        self._episode_discounted_return += (self.reward_gamma ** int(self.step_count)) * reward
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

        self.step_count += 1

        # collision/fallen/drop are hard terminals.  restart_on_collision is kept
        # only for backward-compatible logging/config, not for suppressing reset.
        terminated = bool(collision or out_of_bounds or fallen or shake_restart or coverage_done or priority_stuck_restart or lidar_empty_restart)
        truncated = bool(self.step_count >= self.max_episode_steps)
        if truncated and self._last_terminal_reason == "none":
            self._last_terminal_reason = "time_limit"

        self.prev_action = action_for_reward.copy()
        self.last_map_stats = map_stats

        obs = self._get_obs()

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

        return obs, float(reward), terminated, truncated, info

    def _execute_velocity_action(self, policy_action: np.ndarray) -> tuple[np.ndarray, float, float, float]:
        """
        Direct pure-velocity SAC executor.

        Policy action meaning:
          action[0] = commanded forward linear velocity in [0, max_linear_speed]
          action[1] = commanded angular velocity in [-max_angular_speed, max_angular_speed]

        The safety shield does not change the observation/action space.  It only
        clamps unsafe forward commands and, when the front sector is already too
        close to an obstacle, publishes a short reverse TwistStamped escape.
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
        self._last_velocity_safety_slowdown = 1.0
        self._last_velocity_safety_penalty = 0.0
        self._last_velocity_safety_reason = "none"
        self._last_velocity_forward_assist = False
        self._last_velocity_spin_breaker = False

        if int(getattr(self, "velocity_safety_cooldown_steps", 0)) > 0:
            self.velocity_safety_cooldown_steps = max(
                int(getattr(self, "velocity_safety_cooldown_steps", 0)) - 1,
                0,
            )

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

        lidar_action_obstacle_distance, lidar_action_obstacle_score, lidar_front_obstacle_distance = (
            self._compute_lidar_action_obstacle_risk(action)
        )
        slam_action_distance, slam_front_distance = self._compute_slam_action_obstacle_distance(action)
        lidar_action_obstacle_distance = min(float(lidar_action_obstacle_distance), float(slam_action_distance))
        lidar_front_obstacle_distance = min(float(lidar_front_obstacle_distance), float(front_min), float(slam_front_distance))
        front_min = min(float(front_min), float(slam_front_distance))
        if float(lidar_action_obstacle_distance) < 0.60:
            warn_distance = 0.60
            hard_distance = 0.22
            slam_risk = (warn_distance - float(lidar_action_obstacle_distance)) / max(warn_distance - hard_distance, 1e-6)
            lidar_action_obstacle_score = max(float(lidar_action_obstacle_score), float(np.clip(slam_risk, 0.0, 1.0)))

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
        can_backup = (
            bool(getattr(self, "velocity_safety_backup", True))
            and int(getattr(self, "velocity_safety_cooldown_steps", 0)) <= 0
            and float(rear_min) > rear_warn_dist
        )

        # 1) Imminent front collision: override with short reverse recovery, but
        # never continue reversing once the rear sector becomes unsafe.
        if front_min < trigger_dist and can_backup:
            backup_v_cfg = float(getattr(self, "velocity_safety_backup_speed_mps", 0.08))
            backup_v = -min(max(backup_v_cfg, 0.0), 0.10)
            backup_w = turn_sign * float(getattr(self, "velocity_safety_turn_speed", 0.35))
            backup_steps = int(getattr(self, "velocity_safety_backup_steps", 4))
            executed_backup_steps = 0
            stopped_by_rear = False
            rear_during_backup = float(rear_min)
            for _ in range(max(backup_steps, 1)):
                self.ros.spin_steps(num_spins=2, timeout_sec=0.001)
                scan_now = self.ros.scan
                if scan_now is not None:
                    rear_during_backup = self._scan_min_distance_in_sector(
                        scan=scan_now,
                        center_angle=math.pi,
                        half_width_rad=math.radians(45.0),
                        max_considered_range=0.90,
                    )
                if float(rear_during_backup) <= rear_stop_dist:
                    stopped_by_rear = True
                    self.ros.publish_cmd_vel(0.0, backup_w)
                    self._advance_world_after_command(target_delta_sec=self.control_dt)
                    break
                self.ros.publish_cmd_vel(backup_v, backup_w)
                self.ros.spin_steps(num_spins=2, timeout_sec=0.001)
                self._advance_world_after_command(target_delta_sec=self.control_dt)
                executed_backup_steps += 1
                if self._check_collision() or self._check_fallen():
                    break

            if stopped_by_rear or executed_backup_steps <= 0:
                cmd = np.array([0.0, backup_w], dtype=np.float32)
                reason_prefix = "backup_blocked_rear"
            else:
                cmd = np.array([backup_v, backup_w], dtype=np.float32)
                reason_prefix = "backup"
            self.ros.publish_cmd_vel(float(cmd[0]), float(cmd[1]))
            self.velocity_safety_cooldown_steps = int(getattr(self, "velocity_safety_cooldown_steps_cfg", 8))
            self._last_velocity_safety_backup_triggered = True
            self._last_velocity_safety_penalty = float(getattr(self, "velocity_safety_penalty", 10.0))
            self._last_velocity_safety_reason = (
                f"{reason_prefix}:front={front_min:.3f},rear0={rear_min:.3f},"
                f"rear={rear_during_backup:.3f},steps={executed_backup_steps}"
            )
            self.ros.get_logger().warn(
                "VELOCITY_SAFETY_BACKUP | "
                f"front={front_min:.3f}m rear0={rear_min:.3f}m rear={rear_during_backup:.3f}m "
                f"steps={executed_backup_steps}/{backup_steps} cmd=({cmd[0]:+.3f},{cmd[1]:+.3f}) "
                f"rear_stop={rear_stop_dist:.3f} penalty={self._last_velocity_safety_penalty:.2f}"
            )
            return (
                cmd.astype(np.float32),
                float(lidar_action_obstacle_distance),
                float(max(lidar_action_obstacle_score, 1.0)),
                float(lidar_front_obstacle_distance),
            )

        if front_min < trigger_dist and not can_backup:
            # Front is dangerous but rear is not safe enough for reverse.  Rotate
            # in place instead of backing into an unknown/occupied rear sector.
            cmd = np.array([0.0, turn_sign * float(getattr(self, "velocity_safety_turn_speed", 0.35))], dtype=np.float32)
            self.ros.publish_cmd_vel(float(cmd[0]), float(cmd[1]))
            self.ros.spin_steps(num_spins=5, timeout_sec=0.001)
            self._advance_world_after_command(target_delta_sec=self.control_dt)
            self._last_velocity_safety_blocked = True
            self._last_velocity_safety_penalty = float(getattr(self, "velocity_safety_penalty", 10.0))
            self._last_velocity_safety_reason = (
                f"backup_denied_rear:front={front_min:.3f},rear={rear_min:.3f},rear_warn={rear_warn_dist:.3f}"
            )
            self.ros.get_logger().warn(
                "VELOCITY_SAFETY_BACKUP_DENIED | "
                f"front={front_min:.3f}m rear={rear_min:.3f}m rear_warn={rear_warn_dist:.3f} "
                f"cmd=({cmd[0]:+.3f},{cmd[1]:+.3f}) penalty={self._last_velocity_safety_penalty:.2f}"
            )
            return (
                cmd.astype(np.float32),
                float(lidar_action_obstacle_distance),
                float(max(lidar_action_obstacle_score, 1.0)),
                float(lidar_front_obstacle_distance),
            )

        # 2) Too close but cannot/should not reverse: block forward motion and rotate.
        if forward > 0.0 and (front_min < stop_dist or lidar_action_obstacle_distance < stop_dist):
            cmd[0] = 0.0
            if abs(float(cmd[1])) < 0.08:
                cmd[1] = turn_sign * min(float(self.max_angular_speed), float(getattr(self, "velocity_safety_turn_speed", 0.35)))
            self._last_velocity_safety_blocked = True
            self._last_velocity_safety_penalty = float(getattr(self, "velocity_safety_block_penalty", 0.80))
            self._last_velocity_safety_reason = (
                f"block:front={front_min:.3f},action={lidar_action_obstacle_distance:.3f}"
            )

        # 3) Warning band: smoothly reduce forward speed before the hard stop band.
        elif forward > 0.0 and lidar_action_obstacle_distance < slow_dist:
            denom = max(slow_dist - stop_dist, 1e-6)
            scale = float(np.clip((float(lidar_action_obstacle_distance) - stop_dist) / denom, 0.15, 1.0))
            cmd[0] = float(cmd[0]) * scale
            self._last_velocity_safety_slowdown = scale
            if scale < 0.99:
                self._last_velocity_safety_penalty = 0.25 * (1.0 - scale)
                self._last_velocity_safety_reason = (
                    f"slowdown:{scale:.2f},action={lidar_action_obstacle_distance:.3f}"
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
            f"Ggamma={float(getattr(self, '_episode_discounted_return', 0.0)):+.3f}",
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
                            reward_line = f"r={rv:+.3f} Gγ={er:+.2f} γ={self.reward_gamma:.2f} marker={marker_frame_id} pose={self.pose_frame} nav2=NavigateToPose"

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
        if self.step_count % self.waypoint_visual_publish_every_n != 0:
            return

        stamp = self._latest_tf_stamp()
        # For real-robot RViz debugging, publish markers directly in map/pose frame
        # instead of base_footprint.  This avoids losing arrows/text when TF is
        # temporarily unavailable or RViz cannot transform robot-local markers.
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
        front = fnum(lidar_front_obstacle_distance, 999.0)
        act_obs = fnum(lidar_action_obstacle_distance, 999.0)
        act_score = fnum(lidar_action_obstacle_score, 0.0)
        nearest = fnum(getattr(map_stats, "nearest_obstacle_distance", 999.0), 999.0)
        cov = 100.0 * fnum(getattr(map_stats, "coverage_ratio", 0.0), 0.0)
        dcov = 100.0 * fnum(getattr(map_stats, "coverage_delta", 0.0), 0.0)
        conf = fnum(getattr(map_stats, "mean_confidence", 0.0), 0.0)
        stale_pct = 100.0 * fnum(getattr(map_stats, "stale_ratio", 0.0), 0.0)
        low = 100.0 * fnum(getattr(map_stats, "low_confidence_ratio", 0.0), 0.0)
        prio_score = fnum(getattr(map_stats, "priority_score", 0.0), 0.0)
        prio_gain = fnum(getattr(map_stats, "priority_gain", 0.0), 0.0)
        prio_clear = fnum(getattr(map_stats, "priority_clear_gain", 0.0), 0.0)
        prio_recheck = fnum(getattr(map_stats, "priority_rechecked_gain", 0.0), 0.0)
        target_type = str(getattr(map_stats, "target_type", "none"))
        target_prio = fnum(getattr(map_stats, "target_priority", 0.0), 0.0)

        l_empty_steps = int(getattr(self, "lidar_empty_steps", 0))
        l_empty_timeout = int(getattr(self, "lidar_empty_timeout_steps", 0))
        valid_beams = int(getattr(self, "_last_lidar_valid_beams", 0))
        p_steps = int(getattr(self, "priority_stuck_steps", 0))
        p_limit = int(getattr(self, "priority_stuck_restart_steps", 0))
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

        # Clear stale velocity markers from previous frame modes.
        stale_velocity = Marker()
        stale_velocity.header.frame_id = frame_id
        stale_velocity.header.stamp = stamp
        stale_velocity.ns = "rl_velocity_debug"
        stale_velocity.action = Marker.DELETEALL
        arr.markers.append(stale_velocity)

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
            f"r={reward:+.3f}  Gγ={g_return:+.2f}  γ={self.reward_gamma:.2f}  term={term}\n"
            f"raw(v,w)=({fnum(raw[0]):+.3f},{fnum(raw[1]):+.3f})  "
            f"cmd=({fnum(exe[0]):+.3f},{fnum(exe[1]):+.3f})\n"
            f"front={front:.2f}m  actionObs={act_obs:.2f}m  score={act_score:.2f}  near={nearest:.2f}m\n"
            f"SAFE pen={safety_pen:.2f} slow={slowdown:.2f} cd={safety_cd} assist={int(bool(getattr(self, '_last_velocity_forward_assist', False)))} spinfix={int(bool(getattr(self, '_last_velocity_spin_breaker', False)))} limit={int(bool(getattr(self, '_last_velocity_command_limited', False)))}  {safety_reason} {str(getattr(self, '_last_velocity_command_limit_reason', 'none'))}\n"
            f"Shake={shake_steps}/{shake_limit} {shake_reason}  "
            f"Lempty={l_empty_steps}/{l_empty_timeout} beams={valid_beams}  "
            f"Pstuck={p_steps}/{p_limit} confStall={int(getattr(self, 'confidence_stall_steps', 0))} "
            f"spinStall={int(getattr(self, 'sustained_rotation_steps', 0))} "
            f"orbitStall={int(getattr(self, 'orbit_stall_steps', 0))} "
            f"eff={float(getattr(self, '_last_orbit_path_efficiency', 1.0)):.2f}\n"
            f"cov={cov:.1f}% Δ={dcov:+.2f}%  conf={conf:.1f}  stale={stale_pct:.1f}% low={low:.1f}%\n"
            f"prio score={prio_score:.2f} gain={prio_gain:.2f} clr={prio_clear:.2f} rechk={prio_recheck:.2f}\n"
            f"slamNew={slam_new_known} slamR={slam_r:+.3f}  "
            f"target={target_type} tp={target_prio:.2f}  slam={slam_gate} age={map_age:.1f}s "
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
            # process/thread.  Do not block for control_dt here; blocking the RL
            # process competes with Gazebo/SLAM/RViz on the same CPU and can drop
            # the simulator real-time factor.  We only drain pending callbacks and
            # yield the CPU briefly.
            if self.realtime_spin_steps > 0:
                self.ros.spin_steps(
                    num_spins=self.realtime_spin_steps,
                    timeout_sec=self.realtime_spin_timeout_sec,
                )
            if self.realtime_sleep_sec > 0.0:
                time.sleep(self.realtime_sleep_sec)

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

        self.ros.get_logger().info(
            "RESET_POSE_TRUTH | "
            f"candidate_requested=(x={float(requested_x):.3f}, y={float(requested_y):.3f}) | "
            f"entity='{entity}' | "
            f"{reset_text} | "
            f"actual_gazebo={actual_gz} | "
            f"odom_pose={odom_text} | "
            f"{self.pose_frame}_pose={map_text}"
        )

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

    def _get_obs(self):
        if self.ros.scan is None or self.ros.odom is None:
            return self._empty_observation()

        stats = self.last_map_stats

        vector_obs = build_exploration_observation(
            scan_ranges=self.ros.scan.ranges,
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
        ).astype(np.float32)

        self._push_vector_history(vector_obs)

        if not self.use_map_cnn:
            return vector_obs

        robot_pose = self._get_robot_pose2d()

        if robot_pose is None:
            map_obs = np.zeros(
                (5, self.map_obs_size, self.map_obs_size),
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

        obs = {
            "vector": vector_obs,
            "map": map_obs,
        }

        if self.use_temporal_cnn:
            obs["seq"] = self._sequence_observation()

        return obs

    def _empty_observation(self):
        vector_obs = np.zeros(self.obs_dim, dtype=np.float32)

        if not self.use_map_cnn:
            return vector_obs

        obs = {
            "vector": vector_obs,
            "map": np.zeros(
                (5, self.map_obs_size, self.map_obs_size),
                dtype=np.float32,
            ),
        }

        if self.use_temporal_cnn:
            obs["seq"] = np.zeros(
                (self.temporal_history_len, self.obs_dim),
                dtype=np.float32,
            )

        return obs

    def _push_vector_history(self, vector_obs: np.ndarray):
        if not self.use_temporal_cnn:
            return

        vector_obs = np.asarray(vector_obs, dtype=np.float32)

        if not self.vector_history:
            for _ in range(self.temporal_history_len):
                self.vector_history.append(vector_obs.copy())
            return

        self.vector_history.append(vector_obs.copy())

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
        """Return a non-'none' reason when the robot body is dynamically unstable."""
        if not bool(getattr(self, "shake_restart", True)):
            return "none"
        try:
            rpy = self.ros.get_roll_pitch_yaw()
            roll = pitch = 0.0
            if rpy is not None:
                roll, pitch, _ = rpy
            tilt = max(abs(float(roll)), abs(float(pitch)))

            wx = wy = vz = z_dev = 0.0
            if self.ros.odom is not None:
                twist = self.ros.odom.twist.twist
                wx = float(getattr(twist.angular, "x", 0.0))
                wy = float(getattr(twist.angular, "y", 0.0))
                vz = float(getattr(twist.linear, "z", 0.0))
                z = float(self.ros.odom.pose.pose.position.z)
                nominal_z = float(getattr(self, "_reset_nominal_z", 0.05))
                z_dev = abs(z - nominal_z)

            if tilt >= float(getattr(self, "shake_tilt_threshold", 0.22)):
                return f"tilt:{tilt:.3f}"
            if math.hypot(wx, wy) >= float(getattr(self, "shake_angular_xy_threshold", 1.80)):
                return f"ang_xy:{math.hypot(wx, wy):.3f}"
            if abs(vz) >= float(getattr(self, "shake_linear_z_threshold", 0.22)):
                return f"vz:{vz:.3f}"
            if z_dev >= float(getattr(self, "shake_z_deviation_threshold", 0.12)):
                return f"z_dev:{z_dev:.3f}"
        except Exception as exc:
            return f"shake_check_error:{type(exc).__name__}"
        return "none"

    def _update_shake_restart_state(self) -> bool:
        """Restart when non-yaw body shake persists for N consecutive steps."""
        self._last_shake_restart = False
        reason = self._instantaneous_shake_reason()
        if reason == "none":
            self.shake_steps = 0
            self._last_shake_active = False
            self._last_shake_reason = "none"
            return False
        self.shake_steps += 1
        self._last_shake_active = True
        self._last_shake_reason = reason
        if self.shake_steps >= int(getattr(self, "shake_restart_steps_limit", 8)):
            self._last_shake_restart = True
            self.ros.get_logger().warn(
                "SHAKE_RESTART | "
                f"steps={self.shake_steps}/{int(getattr(self, 'shake_restart_steps_limit', 8))} | "
                f"reason={reason}"
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
        return self.ros.get_pose2d(frame_id=(frame_id or self.pose_frame))

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

        self.ros.get_logger().info(
            "Safety boundary center updated: "
            f"frame={self.safety_boundary_frame}, "
            f"source={source}, "
            f"center=({self.current_boundary_center_xy[0]:.3f}, {self.current_boundary_center_xy[1]:.3f}), "
            f"requested=({float(requested_xy[0]):.3f}, {float(requested_xy[1]):.3f})"
        )


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
        if bonus > 0.0 and int(getattr(self, "step_count", 0)) % 25 == 0:
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
        timeout = float(getattr(self, "action_sync_wait_timeout_sec", 0.18))
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
        }
        try:
            if self.exploration_map is None:
                metrics["reason"] = "missing_exploration_map"
                return metrics
            pose = self._get_robot_pose2d(frame_id=self.pose_frame)
            if pose is None:
                metrics["reason"] = "missing_pose"
                return metrics
            robot_xy, _ = pose
            emap = self.exploration_map
            rix, riy = emap.world_to_map(float(robot_xy[0]), float(robot_xy[1]))
            margin = max(int(getattr(self, "map_bounds_margin_cells", 2)), 0)
            inside = bool(emap.in_bounds(int(rix), int(riy)))
            metrics["inside"] = inside
            if not inside:
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
        self._last_map_bounds_reason = reason
        self._last_map_bounds_local_known_ratio = known_ratio
        self._last_map_bounds_local_known_cells = known_cells

        hard_bad = reason.startswith("pose_outside_rl_map") or reason in {"missing_pose", "empty_local_crop", "near_rl_map_edge"}
        soft_bad = reason in {"local_known_cells_low", "local_known_ratio_low"}
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
                f"known={known_cells} ratio={known_ratio:.3f}"
            )
        return restart

    def _update_exploration_map(self) -> MapUpdateStats:
        if self.ros.scan is None or self.ros.odom is None:
            return self.last_map_stats

        robot_pose = self._get_robot_pose2d()

        if robot_pose is None:
            return self.last_map_stats

        robot_xy, robot_yaw = robot_pose

        # v23.5: Do NOT poll /slam_toolbox/dynamic_map during live policy
        # steps.  Dynamic_map service calls are relatively heavy and, if made
        # every second while the node is already spinning, can both trigger
        # "Executor is already spinning" in rclpy and make slam_toolbox print
        # service-response timeout warnings.  After strict startup/reset gates,
        # live map updates must come from /map topic/mirror content updates.
        # Service fetch remains available only inside explicit strict gates.

        slam_map = self._filtered_slam_map_for_update()

        # If SLAM /map is missing/stale/ignored after reset, still update the
        # internal exploration layers from the latest LaserScan and pose.  This is
        # essential for real-robot tests where slam_toolbox may fail to publish
        # /map because of sensor/TF/QoS problems; otherwise known=0, prio=0 and
        # MAP_LOCKED_PUBLISH_WAITING_FOR_SLAM_REF repeat forever.
        return self.exploration_map.update(
            scan=self.ros.scan,
            robot_xy=robot_xy,
            robot_yaw=robot_yaw,
            publish=True,
            slam_map=slam_map,
        )

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
        priority_check_multiplier = 0.50 + 1.50 * priority_strength
        corridor_priority_weight = float(getattr(self, "corridor_priority_reward_weight", 1.0))

        priority_clear_weighted_sum = max(priority_clear_gain, 0.0)
        priority_rechecked_weighted_sum = max(priority_rechecked_gain, 0.0)

        # Must mirror reward.py exactly.  This is telemetry only, not an extra
        # reward path.  The gain terms are additive weighted sums over all cells
        # cleared/rechecked during the current SAC macro-action.
        clear_reward = (
            0.042
            * corridor_priority_weight
            * priority_clear_weighted_sum
            * priority_check_multiplier
        )
        clear_reward += 0.0012 * corridor_priority_weight * float(max(priority_cleared_cells, 0))

        recheck_reward = (
            0.016
            * corridor_priority_weight
            * priority_rechecked_weighted_sum
            * (0.50 + 0.75 * priority_strength)
        )
        recheck_reward += 0.004 * corridor_priority_weight * float(max(priority_rechecked_cells, 0))

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
