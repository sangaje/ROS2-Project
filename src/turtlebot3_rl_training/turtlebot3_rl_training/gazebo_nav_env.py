import math
import random
import time
from collections import deque
from typing import Optional

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from turtlebot3_rl_training.exploration_map import ExplorationGridMap, MapUpdateStats
from turtlebot3_rl_training.observation import build_exploration_observation
from turtlebot3_rl_training.reset_manager import ResetManager
from turtlebot3_rl_training.reward import compute_exploration_reward
from turtlebot3_rl_training.sim_controller import GazeboSimController


class GazeboNavEnv(gym.Env):
    """
    TurtleBot3 Burger용 Gymnasium Env.

    구조:
      - SLAM /map은 실제 geometry/localization 품질을 담당한다.
      - RL memory map은 confidence, stale, revisit 정보를 담당한다.
      - SAC policy는 LiDAR + confidence/task-map 통계량을 보고 탐색 정책을 학습한다.

    action:
      [linear_x, angular_z]

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
        control_dt: float = 0.1,
        max_episode_steps: int = 1000,
        goal_threshold: float = 0.25,
        collision_threshold: float = 0.10,
        fallen_roll_threshold: float = 0.7,
        fallen_pitch_threshold: float = 0.7,
        world_control_service: str = "/world/default/control",
        physics_step_size: float = 0.005,
        use_world_step: bool = True,
        max_linear_speed: float = 0.6,
        max_angular_speed: float = 1.5,
        use_slam_map: bool = True,
        map_frame: str = "map",
        rl_map_topic: str = "/rl_task_map",
        rl_confidence_topic: str = "/rl_confidence_map",
        rl_priority_topic: str = "/rl_priority_map",
        rl_path_topic: str = "/rl_path",
        rl_filtered_slam_topic: str = "/rl_filtered_slam_map",
        slam_map_accept_delay_sec: float = 1.0,
        slam_map_max_age_sec: float = 3.0,
        reset_x: float = 0.0,
        reset_y: float = 0.0,
        reset_slam_on_reset: bool = False,
        restart_slam_on_reset: bool = False,
        slam_reset_timeout_sec: float = 8.0,
        slam_reset_warmup_steps: int = 15,
        use_map_cnn: bool = True,
        map_obs_size: int = 64,
        map_obs_size_m: float = 6.4,
        use_temporal_cnn: bool = True,
        temporal_history_len: int = 8,
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
        map_publish_every_n: int = 10,
        priority_recompute_interval: int = 8,
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
        map_keepalive_period_sec: float = 1.0,
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
    ):
        super().__init__()

        self.ros = ros_interface

        self.entity_name = entity_name
        self.enable_pose_reset = bool(enable_pose_reset)
        self.random_reset_yaw = bool(random_reset_yaw)

        self.control_dt = float(control_dt)
        self.physics_step_size = float(physics_step_size)
        self.use_world_step = bool(use_world_step)

        self.max_linear_speed = float(max_linear_speed)
        self.max_angular_speed = float(max_angular_speed)

        self.use_slam_map = bool(use_slam_map)
        self.map_frame = str(map_frame).strip() or "map"
        self.pose_frame = self.map_frame if self.use_slam_map else "odom"
        # reset_x/reset_y는 CLI 호환성 때문에 유지하지만,
        # 실제 episode reset 위치는 0.0을 쓰지 않고 ±reset_offset 후보만 사용한다.
        # Burger가 SLAM map과 RL memory map의 경계 근처에서 시작하지 않도록
        # x, y 각각 {-0.3, +0.3}만 허용한다.
        self.reset_x = float(reset_x)
        self.reset_y = float(reset_y)
        self.reset_offset = 0.3
        self.current_reset_xy = np.array(
            [self.reset_offset, self.reset_offset],
            dtype=np.float32,
        )
        self.reset_slam_on_reset = bool(reset_slam_on_reset)
        self.restart_slam_on_reset = bool(restart_slam_on_reset)
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
        self.rl_priority_topic = str(rl_priority_topic).strip() or "/rl_priority_map"
        self.rl_path_topic = str(rl_path_topic).strip() or "/rl_path"
        self.rl_filtered_slam_topic = str(rl_filtered_slam_topic).strip() or "/rl_filtered_slam_map"
        self.slam_map_accept_delay_sec = max(float(slam_map_accept_delay_sec), 0.0)
        self.slam_map_max_age_sec = max(float(slam_map_max_age_sec), 0.0)
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
        self.vector_history = deque(maxlen=self.temporal_history_len)

        # 이미 확인한 영역에서 새 정보 없이 머무는 시간을 누적한다.
        # 이 값은 reward에서 시간 증가형 penalty로 사용한다.
        self.explored_stall_steps = 0
        self.explored_stall_start_steps = max(int(explored_stall_start_steps), 0)
        self.explored_stall_growth = max(float(explored_stall_growth), 0.0)
        self.explored_stall_power = max(float(explored_stall_power), 1.0)
        self.explored_stall_max_penalty = max(float(explored_stall_max_penalty), 0.0)

        # Policy 출력은 바로 cmd_vel로 보내지 않고, 물리적으로 가능한 제어 신호로 필터링한다.
        # 목적:
        #   - 좌우 각속도 sign flip으로 생기는 본체 떨림 억제
        #   - 직진/호 주행/제자리 회전 모드가 매 step 흔들리지 않게 hysteresis 부여
        #   - 후진은 action_space에서 이미 제거되어 있으므로 linear_x는 [0, max_linear_speed] 유지
        self.filtered_action = np.zeros(2, dtype=np.float32)
        self.raw_action = np.zeros(2, dtype=np.float32)
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
            path_publish_topic=self.rl_path_topic,
            filtered_slam_publish_topic=self.rl_filtered_slam_topic,
            keepalive_publish_period_sec=self.map_keepalive_period_sec,
            lidar_stride=2,
            max_range=3.5,
            publish_every_n=self.map_publish_every_n,
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

        self.action_space = spaces.Box(
            low=np.array([0.0, -self.max_angular_speed], dtype=np.float32),
            high=np.array(
                [self.max_linear_speed, self.max_angular_speed], dtype=np.float32
            ),
            dtype=np.float32,
        )

        self.max_episode_steps = int(max_episode_steps)

        self.goal_threshold = float(goal_threshold)
        self.collision_threshold = float(collision_threshold)

        self.fallen_roll_threshold = float(fallen_roll_threshold)
        self.fallen_pitch_threshold = float(fallen_pitch_threshold)

        self.prev_action = np.zeros(2, dtype=np.float32)
        self.prev_distance = 0.0

        self.goal_xy = np.array([1.5, 0.0], dtype=np.float32)
        self.step_count = 0

        # Episode reset 위치 후보.
        # 요청 조건:
        #   - x는 +0.3 또는 -0.3만 사용
        #   - y는 +0.3 또는 -0.3만 사용
        #   - (0.0, 0.0)은 절대 사용하지 않음
        self.reset_pose_candidates = [
            (1.2, -2.2),
            # (self.reset_offset, self.reset_offset),
            # (self.reset_offset, -self.reset_offset),
            # (-self.reset_offset, self.reset_offset),
            # (-self.reset_offset, -self.reset_offset),
        ]

        self.goal_candidates = [
            np.array([1.5, 0.0], dtype=np.float32),
            np.array([1.2, 0.8], dtype=np.float32),
            np.array([1.2, -0.8], dtype=np.float32),
            np.array([2.0, 0.5], dtype=np.float32),
            np.array([2.0, -0.5], dtype=np.float32),
        ]

        self.ros.get_logger().info(
            "GazeboNavEnv SLAM-memory mode: "
            f"use_slam_map={self.use_slam_map}, "
            f"pose_frame={self.pose_frame}, "
            f"obs_dim={self.obs_dim}, "
            f"use_map_cnn={self.use_map_cnn}, "
            f"map_obs_shape=(5,{self.map_obs_size},{self.map_obs_size}), "
            f"use_temporal_cnn={self.use_temporal_cnn}, "
            f"temporal_history_len={self.temporal_history_len}, "
            f"front_fov_deg={self.front_fov_deg:.1f}, "
            f"front_angle_sigma_deg={self.front_angle_sigma_deg:.1f}, "
            f"confidence_max_range={self.confidence_max_range:.2f}, "
            f"seen_confidence_floor={self.seen_confidence_floor:.1f}, "
            f"confidence_decay={'disabled' if self.confidence_decay_per_step <= 0.0 else self.confidence_decay_per_step}, "
            f"suppress_gap_confidence={self.suppress_gap_confidence}, "
            f"gap_width=[{self.gap_min_width_m:.2f},{self.gap_max_width_m:.2f}]m, "
            f"map_expand_chunk_cells={self.map_expand_chunk_cells}, "
            f"priority_suppression_radius={self.priority_visit_suppression_radius_m:.2f}m, "
            f"priority_suppression_gain={self.priority_visit_suppression_gain:.2f}, "
            f"priority_observed_suppression_gain={self.priority_observed_suppression_gain:.2f}, "
            f"priority_clear_fov={self.priority_clear_fov_deg:.1f}deg, "
            f"priority_clear_range={self.priority_clear_max_range_m:.2f}m, "
            f"priority_clear_sigma={self.priority_clear_sigma_m:.2f}m, "
            f"priority_clear_min_weight={self.priority_clear_min_weight:.2f}, "
            f"wall_support_radius={self.wall_support_radius_m:.2f}m, "
            f"wall_support_density_threshold={self.wall_support_density_threshold:.3f}, "
            f"open_space_front_distance={self.open_space_front_distance_m:.2f}m, "
            f"open_space_side_width={self.open_space_side_width_m:.2f}m, "
            f"open_space_forward_penalty={self.open_space_forward_penalty:.2f}, "
            f"action_filter=disabled, "
            f"max_linear_delta={self.max_linear_delta:.3f}, "
            f"max_angular_delta={self.max_angular_delta:.3f}, "
            f"motion_mode_hysteresis={self.enable_motion_mode_hysteresis}, "
            f"remember_map=disabled, "
            f"explored_stall_start={self.explored_stall_start_steps}, "
            f"explored_stall_growth={self.explored_stall_growth:.4f}, "
            f"explored_stall_power={self.explored_stall_power:.2f}, "
            f"explored_stall_max={self.explored_stall_max_penalty:.2f}, "
            f"reset_candidates={self.reset_pose_candidates}, "
            f"rl_path_topic={self.rl_path_topic}, "
            f"rl_filtered_slam_topic={self.rl_filtered_slam_topic}, "
            f"slam_map_accept_delay={self.slam_map_accept_delay_sec:.2f}s, "
            f"slam_map_max_age={self.slam_map_max_age_sec:.2f}s, "
            f"reset_slam_on_reset={self.reset_slam_on_reset}, "
            f"restart_slam_on_reset={self.restart_slam_on_reset}"
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.ros.stop_robot()
        self._advance_world_after_command(target_delta_sec=0.02)

        # 1) Burger를 episode 시작점으로 보낸다.
        #    (0, 0)은 사용하지 않고, x/y 각각 ±0.3 조합만 사용한다.
        reset_x, reset_y = random.choice(self.reset_pose_candidates)
        self.current_reset_xy = np.array([reset_x, reset_y], dtype=np.float32)

        if self.enable_pose_reset:
            if self.reset_manager is None:
                raise RuntimeError("pose reset is enabled but reset_manager is None")

            reset_pose = self.reset_manager.reset_center_pose(
                x=reset_x,
                y=reset_y,
                random_yaw=self.random_reset_yaw,
                fixed_yaw=0.0,
            )

            if reset_pose is None:
                raise RuntimeError(
                    "Failed to reset TurtleBot pose. "
                    "Check SetEntityPose service/entity name. "
                    "The ResetManager now auto-tries gz model --list candidates."
                )

            self.ros.stop_robot()
            self._advance_world_after_command(target_delta_sec=self.control_dt)
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

        self.prev_action = np.zeros(2, dtype=np.float32)
        self.filtered_action = np.zeros(2, dtype=np.float32)
        self.raw_action = np.zeros(2, dtype=np.float32)
        self.motion_mode = "STRAIGHT"
        self.step_count = 0
        self.explored_stall_steps = 0
        self.last_map_stats = self._empty_map_stats()
        self.vector_history.clear()

        # 2) 기본은 SLAM 프로세스를 재시작하지 않는다.
        #    /map reset service가 있을 때만 service로 비우고, 없으면 RL memory/confidence map만 초기화한다.
        #    slam_toolbox 프로세스 재시작은 --restart-slam-on-reset을 명시했을 때만 허용한다.
        self.ignore_slam_prior_this_episode = False
        if self.use_slam_map and self.reset_slam_on_reset:
            slam_reset_ok = self._reset_slam_map_after_pose_reset()
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

        # 3) RL memory/confidence map도 전부 초기화하고 Burger가 중앙에 오도록 origin을 재설정한다.
        robot_pose = self._get_robot_pose2d()

        if robot_pose is not None:
            robot_xy, _ = robot_pose
            self.exploration_map.reset_centered_at(robot_xy)
        else:
            self.exploration_map.reset_centered_at(self.current_reset_xy.copy())

        # 4) 초기 scan 한 번으로 첫 memory map을 채운다.
        self.last_map_stats = self._update_exploration_map()

        obs = self._get_obs()

        info = self._build_info(
            map_stats=self.last_map_stats,
            collision=self._check_collision(),
            fallen=self._check_fallen(),
            coverage_done=False,
        )

        return obs, info

    def step(self, action):
        raw_action = np.asarray(action, dtype=np.float32)
        raw_action = np.clip(raw_action, self.action_space.low, self.action_space.high)
        self.raw_action = raw_action.copy()

        # Gazebo로 나가는 제어 입력은 policy가 낸 action을 그대로 사용한다.
        # smoothing/rate-limit/hysteresis 같은 직접 제어 차단은 적용하지 않는다.
        # 흔들림/불필요한 회전은 reward shaping으로만 억제한다.
        action = raw_action.astype(np.float32).copy()

        linear_x = float(action[0])
        angular_z = float(action[1])

        self.ros.publish_cmd_vel(linear_x, angular_z)

        self.ros.spin_steps(num_spins=5, timeout_sec=0.001)

        self._advance_world_after_command(target_delta_sec=self.control_dt)

        map_stats = self._update_exploration_map()
        self._update_explored_stall_steps(map_stats=map_stats, action=action)

        collision = self._check_collision()
        fallen = self._check_fallen()

        coverage_done = map_stats.coverage_ratio >= self.target_coverage_ratio

        # Reward의 action smoothness penalty는 "이전 출력"과 현재 action을 비교해야 하므로
        # self.prev_action을 갱신하기 전에 별도로 보관한다.
        prev_action_for_reward = self.prev_action.copy()

        reward = compute_exploration_reward(
            new_known_cells=map_stats.new_known_cells,
            coverage_delta=map_stats.coverage_delta,
            coverage_ratio=map_stats.coverage_ratio,
            frontier_count=map_stats.frontier_count,
            robot_visit_count=map_stats.robot_visit_count,
            action=action,
            prev_action=prev_action_for_reward,
            collision=collision,
            fallen=fallen,
            stale_refresh_cells=map_stats.stale_refresh_cells,
            confidence_gain=map_stats.confidence_gain,
            mean_confidence=map_stats.mean_confidence,
            stale_ratio=map_stats.stale_ratio,
            low_confidence_ratio=map_stats.low_confidence_ratio,
            target_priority=map_stats.target_priority,
            frontier_angle=map_stats.frontier_angle,
            target_type=map_stats.target_type,
            target_switched=map_stats.target_switched,
            target_reachable=map_stats.target_reachable,
            path_distance=map_stats.path_distance,
            path_angle=map_stats.path_angle,
            path_progress=map_stats.path_progress,
            priority_score=map_stats.priority_score,
            priority_gain=map_stats.priority_gain,
            priority_cleared_cells=map_stats.priority_cleared_cells,
            priority_clear_gain=map_stats.priority_clear_gain,
            wall_support_score=map_stats.wall_support_score,
            open_space_score=map_stats.open_space_score,
            open_space_forward_penalty=self.open_space_forward_penalty,
            explored_stall_steps=self.explored_stall_steps,
            explored_stall_start_steps=self.explored_stall_start_steps,
            explored_stall_growth=self.explored_stall_growth,
            explored_stall_power=self.explored_stall_power,
            explored_stall_max_penalty=self.explored_stall_max_penalty,
            max_linear_speed=self.max_linear_speed,
            max_angular_speed=self.max_angular_speed,
        )

        self.step_count += 1

        terminated = bool(collision or fallen or coverage_done)
        truncated = bool(self.step_count >= self.max_episode_steps)

        # 다음 observation의 vector/seq에는 "방금 낸 출력(action_t) + 현재 환경(state_{t+1})"이 들어가게 한다.
        # 이렇게 해야 1D CNN이 이전 출력과 현재 환경의 시간적 연결을 학습할 수 있다.
        self.prev_action = action.copy()
        self.last_map_stats = map_stats

        obs = self._get_obs()

        if terminated or truncated:
            self.ros.stop_robot()

        info = self._build_info(
            map_stats=map_stats,
            collision=collision,
            fallen=fallen,
            coverage_done=coverage_done,
        )

        return obs, float(reward), terminated, truncated, info

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

            time_advanced = self.ros.wait_for_time_advance(
                start_sim_time_sec=prev_sim_time,
                start_odom_stamp_sec=prev_odom_stamp,
                target_delta_sec=float(target_delta_sec) * 0.5,
                timeout_wall_sec=0.2,
            )

            sensor_updated = self.ros.wait_for_new_sensor_frame(
                prev_scan_wall_time=prev_scan_wall,
                prev_odom_wall_time=prev_odom_wall,
                timeout_wall_sec=0.2,
            )

            if not time_advanced and not sensor_updated:
                if self.step_count % 50 == 0:
                    self.ros.get_logger().warn(
                        "Sim/odom time and sensor frame did not advance after multi_step. "
                        "Observation may be stale. "
                        f"sim_time={self.ros.get_sim_time_sec()}, "
                        f"odom_stamp={self.ros.get_odom_stamp_sec()}"
                    )

            self.ros.spin_steps(num_spins=5, timeout_sec=0.0)

        else:
            self.ros.spin_steps(num_spins=20, timeout_sec=0.001)

    def close(self):
        self.ros.stop_robot()

        if self.sim_controller is not None and hasattr(self.sim_controller, "close"):
            self.sim_controller.close()

        if self.reset_manager is not None and hasattr(self.reset_manager, "close"):
            self.reset_manager.close()

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

        return bool(np.min(ranges) < self.collision_threshold)

    def _check_fallen(self) -> bool:
        return self.ros.is_fallen(
            max_abs_roll=self.fallen_roll_threshold,
            max_abs_pitch=self.fallen_pitch_threshold,
        )

    def _sample_goal(self) -> np.ndarray:
        return random.choice(self.goal_candidates)

    def _safe_sim_time(self) -> float:
        sim_time = self.ros.get_sim_time_sec()

        if sim_time is None:
            return -1.0

        return float(sim_time)

    def _get_robot_pose2d(self) -> Optional[tuple[np.ndarray, float]]:
        return self.ros.get_pose2d(frame_id=self.pose_frame)

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

        if self.ignore_slam_prior_this_episode:
            self._last_slam_gate_reason = "reset_failed"
            return None

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

        if now_wall < self._slam_map_accept_after_wall_time:
            self._last_slam_map_delay_remaining_sec = float(
                self._slam_map_accept_after_wall_time - now_wall
            )
            self._last_slam_gate_reason = "accept_delay"
            return None

        if self.slam_map_max_age_sec > 0.0 and age > self.slam_map_max_age_sec:
            self._last_slam_gate_reason = "stale"
            return None

        self._last_slam_gate_reason = "accepted"
        return slam_map

    def _update_exploration_map(self) -> MapUpdateStats:
        if self.ros.scan is None or self.ros.odom is None:
            return self.last_map_stats

        robot_pose = self._get_robot_pose2d()

        if robot_pose is None:
            return self.last_map_stats

        robot_xy, robot_yaw = robot_pose

        slam_map = self._filtered_slam_map_for_update()

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

    def _current_action_path_error(self, map_stats: MapUpdateStats) -> tuple[float, float, float]:
        if not bool(map_stats.target_reachable):
            return 0.0, 0.0, 0.0
        action = np.asarray(self.prev_action, dtype=np.float32)
        arc_angle, ok = self._commanded_arc_angle(float(action[0]), float(action[1]))
        if not ok:
            return 0.0, 0.0, 0.0
        err = self._normalize_angle(float(map_stats.path_angle) - arc_angle)
        # reward.py와 동일하게 60도 기준 signed alignment를 debug에 표시한다.
        align = math.exp(-0.5 * (err / max(math.radians(16.0), 1e-6)) ** 2)
        threshold_cos = math.cos(math.radians(60.0))
        c = math.cos(err)
        if c >= threshold_cos:
            signed = (c - threshold_cos) / max(1.0 - threshold_cos, 1e-6)
        else:
            signed = -((threshold_cos - c) / max(threshold_cos + 1.0, 1e-6))
        signed = float(np.clip(signed, -1.0, 1.0))
        return float(err), float(align), signed

    def _build_info(
        self,
        map_stats: MapUpdateStats,
        collision: bool,
        fallen: bool,
        coverage_done: bool,
    ) -> dict:
        action_path_error, action_path_alignment, action_path_signed = self._current_action_path_error(map_stats)
        return {
            "coverage_ratio": float(map_stats.coverage_ratio),
            "coverage_delta": float(map_stats.coverage_delta),
            "new_known_cells": int(map_stats.new_known_cells),
            "known_cells": int(map_stats.known_cells),
            "frontier_count": int(map_stats.frontier_count),
            "frontier_distance": float(map_stats.frontier_distance),
            "frontier_angle": float(map_stats.frontier_angle),
            "robot_visit_count": int(map_stats.robot_visit_count),
            "explored_stall_steps": int(self.explored_stall_steps),
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
            "target_reachable": bool(map_stats.target_reachable),
            "path_distance": float(map_stats.path_distance),
            "path_angle": float(map_stats.path_angle),
            "path_progress": float(map_stats.path_progress),
            "action_path_error": float(action_path_error),
            "action_path_alignment": float(action_path_alignment),
            "action_path_signed": float(action_path_signed),
            "priority_score": float(map_stats.priority_score),
            "priority_gain": float(map_stats.priority_gain),
            "priority_cleared_cells": int(map_stats.priority_cleared_cells),
            "priority_clear_gain": float(map_stats.priority_clear_gain),
            "priority_invalidated_cells": int(getattr(map_stats, 'priority_invalidated_cells', 0)),
            "priority_invalidated_gain": float(getattr(map_stats, 'priority_invalidated_gain', 0.0)),
            "wall_support_score": float(map_stats.wall_support_score),
            "open_space_score": float(map_stats.open_space_score),
            "collision": bool(collision),
            "fallen": bool(fallen),
            "coverage_done": bool(coverage_done),
            "step_count": int(self.step_count),
            "sim_time": self._safe_sim_time(),
            "use_slam_map": bool(self.use_slam_map),
            "pose_frame": str(self.pose_frame),
            "slam_map_available": bool(self.ros.slam_map is not None),
            "slam_map_gate": str(self._last_slam_gate_reason),
            "slam_map_age_sec": float(self._last_slam_map_age_sec),
            "slam_map_delay_remaining_sec": float(self._last_slam_map_delay_remaining_sec),
        }

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
            priority_score=0.0,
            priority_gain=0.0,
            priority_cleared_cells=0,
            priority_clear_gain=0.0,
            priority_invalidated_cells=0,
            priority_invalidated_gain=0.0,
            wall_support_score=0.0,
            open_space_score=0.0,
        )
