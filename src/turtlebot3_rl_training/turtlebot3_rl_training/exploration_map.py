#!/usr/bin/env python3

import heapq
import math
import os
import time
from collections import deque
from dataclasses import dataclass
from typing import Optional

import numpy as np

try:
    from scipy.ndimage import binary_dilation as _scipy_binary_dilation
except ImportError:
    _scipy_binary_dilation = None

from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import OccupancyGrid, Path as NavPath
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)
from sensor_msgs.msg import LaserScan


@dataclass
class MapUpdateStats:
    known_cells: int
    new_known_cells: int
    coverage_ratio: float
    coverage_delta: float
    frontier_count: int
    frontier_distance: float
    frontier_angle: float
    robot_visit_count: int
    mean_confidence: float
    stale_known_cells: int
    stale_ratio: float
    low_confidence_cells: int
    low_confidence_ratio: float
    stale_refresh_cells: int
    confidence_gain: float
    target_priority: float
    target_type: str
    target_switched: bool
    target_lock_age: int
    target_reachable: bool
    path_distance: float
    path_angle: float
    path_progress: float
    alternative_path_count: int
    alternative_path_angles: tuple[float, ...]
    priority_score: float
    priority_gain: float
    priority_cleared_cells: int
    priority_clear_gain: float
    priority_invalidated_cells: int
    priority_invalidated_gain: float
    priority_rechecked_cells: int
    priority_rechecked_gain: float
    wall_support_score: float
    open_space_score: float
    nearest_obstacle_distance: float
    obstacle_proximity_score: float
    # Delayed SLAM /map update diagnostics.  These describe new structural
    # information that arrived through the SLAM OccupancyGrid, not through the
    # immediate LiDAR ray-cast confidence update.  They default to zero so older
    # callers that build MapUpdateStats manually remain compatible.
    slam_update_new_known_cells: int = 0
    slam_update_new_free_cells: int = 0
    slam_update_new_occupied_cells: int = 0
    slam_update_expand_known_cells: int = 0
    # Confidence update diagnostics for RViz overlay/debug.
    # observed = cells inside the camera/front ray mask, updated = cells whose
    # confidence value actually increased in this update.
    confidence_observed_cells: int = 0
    confidence_updated_cells: int = 0


class ExplorationGridMap:
    """
    SLAM-base + auto-expanding task/confidence/priority maps.

    Map policy input is a robot-centric local crop.

      channel 0: SLAM free mask
      channel 1: SLAM unknown mask
      channel 2: SLAM occupied mask
      channel 3: confidence map, normalized 0..1

    If priority is enabled for backward compatibility, channel 4 may contain
    the priority map.  In the no-priority training path, channel 4 is removed
    entirely rather than zero-filled.

    The first three channels are one-hot geometry channels rather than a
    scalar-coded occupancy value. This avoids imposing a fake ordering such as
    free < unknown < occupied and lets the CNN learn frontier-like boundaries
    from free/unknown edges directly.

    Important conventions:
      - Internal global maps auto-expand when SLAM/map or robot movement exceeds
        current bounds.
      - CNN input is always robot-centric:
          top    = robot forward
          bottom = robot backward
          left   = robot left
          right  = robot right
      - confidence map is published as 0..100, never -1.
      - gap/door suppression of confidence is removed.
      - door-like gaps are represented in a separate priority map instead.
    """

    UNKNOWN = -1

    TARGET_NONE = "none"
    TARGET_UNKNOWN = "unknown"
    TARGET_STALE = "stale"
    TARGET_LOW_CONFIDENCE = "low_confidence"
    TARGET_PRIORITY_GAP = "priority_gap"

    def __init__(
        self,
        node,
        resolution: float = 0.05,
        size_m: float = 8.0,
        origin_x: float = -4.0,
        origin_y: float = -4.0,
        frame_id: str = "odom",
        publish_topic: str = "/rl_task_map",
        confidence_publish_topic: str = "/rl_confidence_map",
        priority_publish_topic: str = "/rl_priority_map",
        disable_priority_map: bool = False,
        path_publish_topic: str = "",
        filtered_slam_publish_topic: str = "/rl_filtered_slam_map",
        publish_slam_aligned: bool = False,
        legacy_memory_publish_topic: str = "/rl_memory_map",
        keepalive_publish_period_sec: float = 0.50,
        lidar_stride: int = 2,
        max_range: float = 3.5,
        publish_every_n: int = 5,
        min_known_confidence: float = 8.0,
        low_confidence_threshold: float = 35.0,
        stale_after_steps: int = 180,
        confidence_decay_per_step: float = 0.0,
        logodds_decay_per_step: float = 0.0015,
        distance_weight_beta: float = 0.30,
        confidence_max_range: float = 2.0,
        front_angle_sigma_deg: float = 20.0,
        seen_confidence_floor: float = 80.0,
        # Kept for CLI backward compatibility. Not used for confidence suppression.
        suppress_gap_confidence: bool = False,
        gap_occupied_threshold: float = 65.0,
        gap_check_radius_m: float = 1.20,
        gap_min_width_m: float = 0.25,
        gap_max_width_m: float = 2.00,
        free_logodds_delta: float = -0.16,
        occupied_logodds_delta: float = 0.28,
        max_logodds_abs: float = 1.25,
        slam_prior_confidence: float = 0.0,
        use_slam_prior: bool = True,
        front_fov_deg: float = 80.0,
        priority_recompute_interval: int = 3,
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
        map_expand_chunk_cells: int = 64,
        max_planned_candidates: int = 8,
        max_alternative_paths: int = 5,
        path_visual_publish_every_n: int = 5,
        target_lock_steps: int = 16,
        target_switch_margin: float = 0.12,
    ):
        self.node = node
        self.resolution = float(resolution)
        self.initial_size_m = float(size_m)
        self.size_m = float(size_m)
        self.origin_x = float(origin_x)
        self.origin_y = float(origin_y)
        self.frame_id = str(frame_id or "odom").strip() or "odom"
        self.disable_priority_map = bool(disable_priority_map) or (
            str(os.environ.get("TB3_RL_FORCE_NO_PRIORITY", "0")).strip().lower()
            not in {"0", "false", "no", "off", "disable", "disabled"}
        ) or (
            str(os.environ.get("TB3_RL_NO_PRIORITY_MODEL_INPUT", "0")).strip().lower()
            in {"1", "true", "yes", "on", "enable", "enabled"}
        )
        if self.disable_priority_map:
            os.environ["TB3_RL_FORCE_NO_PRIORITY"] = "1"
            os.environ["TB3_RL_NO_PRIORITY_MODEL_INPUT"] = "1"
            priority_publish_topic = ""
            priority_recompute_interval = 1_000_000_000
            priority_visit_suppression_gain = 0.0
            priority_observed_suppression_gain = 0.0
        self.fast_no_priority_stats = (
            str(os.environ.get("TB3_RL_FAST_NO_PRIORITY_STATS", "0")).strip().lower()
            in {"1", "true", "yes", "on", "enable", "enabled"}
        )
        self.lidar_stride = max(int(lidar_stride), 1)
        self.max_range = float(max_range)

        self.publish_every_n = max(int(publish_every_n), 0)
        self.update_count = 0
        self.step_index = 0
        # Priority RViz diagnostics. This is deliberately low frequency so it
        # does not dominate training logs, but it tells us whether /rl_priority_map
        # is empty because generation failed or because RViz is not drawing it.
        self._last_priority_publish_debug_step = -10_000_000
        # v5 RViz diagnostics: report the exact canvas and nonzero counts that are
        # sent on /rl_confidence_map. This separates internal update bugs from
        # RViz display/canvas mismatch bugs.
        self._last_confidence_publish_debug_step = -10_000_000

        # Cached index grids for vectorized SLAM sampling. Rebuilt only when
        # the auto-expanding map changes shape.
        self._index_cache_shape: tuple[int, int] | None = None
        self._index_cache: tuple[np.ndarray, np.ndarray] | None = None
        self._last_slam_sample_key = None
        # Delayed SLAM /map reward bookkeeping.  Width/height are initialized
        # later in __init__, so keep the dimensions at 0 here.  The first valid
        # SLAM /map callback initializes the previous-known mask and is not paid
        # as exploration reward.
        self._slam_reward_prev_known_mask: np.ndarray | None = None
        self._slam_reward_prev_origin_x = float(self.origin_x)
        self._slam_reward_prev_origin_y = float(self.origin_y)
        self._slam_reward_prev_resolution = float(self.resolution)
        self._slam_reward_prev_width = 0
        self._slam_reward_prev_height = 0
        self._slam_reward_prev_frame_id = str(self.frame_id)
        self._slam_reward_prev_stamp_key = None
        self._last_slam_update_new_known_cells = 0
        self._last_slam_update_new_free_cells = 0
        self._last_slam_update_new_occupied_cells = 0
        self._last_slam_update_expand_known_cells = 0
        self._base_grid_needs_resample = True
        self._publish_resample_cache_key = None
        self._publish_resample_cache = None
        # v18 runtime optimization: if the same OccupancyGrid object/stamp/canvas
        # is passed through multiple env steps, do not re-copy and re-invalidate
        # the SLAM-locked base grid every step.  ROS callbacks create a new message
        # object when /map changes, so this is a safe no-op for unchanged maps.
        self._last_slam_fast_lock_key = None

        self.min_known_confidence = float(min_known_confidence)
        self.low_confidence_threshold = float(low_confidence_threshold)
        self.stale_after_steps = max(int(stale_after_steps), 1)
        self.confidence_decay_per_step = float(confidence_decay_per_step)
        self.logodds_decay_per_step = float(logodds_decay_per_step)
        self.distance_weight_beta = float(distance_weight_beta)
        self.confidence_max_range = max(float(confidence_max_range), 0.1)
        self.front_angle_sigma_rad = max(math.radians(float(front_angle_sigma_deg)), 1e-6)
        self.seen_confidence_floor = float(
            np.clip(seen_confidence_floor, self.min_known_confidence, 100.0)
        )
        try:
            self.clear_confidence_on_slam_occupied = str(
                os.environ.get("TB3_RL_CLEAR_CONFIDENCE_ON_SLAM_OCCUPIED", "0")
            ).strip().lower() in {"1", "true", "yes", "on", "enable", "enabled"}
        except Exception:
            self.clear_confidence_on_slam_occupied = False
        try:
            self.confidence_occupied_confirm_steps = int(
                os.environ.get("TB3_RL_CONFIDENCE_OCCUPIED_CONFIRM_STEPS", "3")
            )
        except Exception:
            self.confidence_occupied_confirm_steps = 3
        self.confidence_occupied_confirm_steps = int(
            np.clip(self.confidence_occupied_confirm_steps, 1, 20)
        )
        try:
            self.confidence_decay_near_obstacle_scale = float(
                os.environ.get("TB3_RL_CONFIDENCE_DECAY_NEAR_OBSTACLE_SCALE", "0.20")
            )
        except Exception:
            self.confidence_decay_near_obstacle_scale = 0.20
        self.confidence_decay_near_obstacle_scale = float(
            np.clip(self.confidence_decay_near_obstacle_scale, 0.0, 1.0)
        )
        try:
            self.confidence_decay_obstacle_ring_radius = int(
                os.environ.get("TB3_RL_CONFIDENCE_OBSTACLE_RING_RADIUS", "3")
            )
        except Exception:
            self.confidence_decay_obstacle_ring_radius = 3
        self.confidence_decay_obstacle_ring_radius = int(
            np.clip(self.confidence_decay_obstacle_ring_radius, 1, 16)
        )
        try:
            self.confidence_obstacle_floor_ratio = float(
                os.environ.get("TB3_RL_CONFIDENCE_OBSTACLE_FLOOR_RATIO", "0.70")
            )
        except Exception:
            self.confidence_obstacle_floor_ratio = 0.70
        self.confidence_obstacle_floor_ratio = float(
            np.clip(self.confidence_obstacle_floor_ratio, 0.0, 1.0)
        )

        # Gap confidence suppression is intentionally disabled.
        self.suppress_gap_confidence = False

        # Gap/door priority-map parameters.
        self.gap_occupied_threshold = float(np.clip(gap_occupied_threshold, 0.0, 100.0))
        self.gap_check_radius_m = max(float(gap_check_radius_m), self.resolution)
        self.gap_min_width_m = max(float(gap_min_width_m), self.resolution)
        self.gap_max_width_m = max(float(gap_max_width_m), self.gap_min_width_m)
        self.priority_recompute_interval = max(int(priority_recompute_interval), 1)
        # v25.7: Priority recompute/birth throttles.
        #
        # `priority_recompute_interval` is a candidate-field refresh throttle.
        # Dirty flags request a refresh, but must not bypass the interval.
        #
        # `TB3_RL_PRIORITY_BIRTH_DELTA` is a separate region-birth throttle:
        # new/high priority regions ramp up by at most this many priority points
        # per recompute.  Clears/decreases are still immediate.
        self._last_priority_recompute_step = -10_000_000
        self._last_priority_recompute_debug_step = -10_000_000
        try:
            self.priority_birth_max_delta_per_recompute = float(
                os.environ.get("TB3_RL_PRIORITY_BIRTH_DELTA", "6.0")
            )
        except Exception:
            self.priority_birth_max_delta_per_recompute = 6.0
        self.priority_birth_max_delta_per_recompute = float(
            np.clip(self.priority_birth_max_delta_per_recompute, 0.05, 100.0)
        )
        self._last_priority_birth_debug_step = -10_000_000
        self.priority_visit_suppression_radius_m = max(
            float(priority_visit_suppression_radius_m),
            self.resolution,
        )
        self.priority_visit_suppression_gain = float(
            np.clip(priority_visit_suppression_gain, 0.0, 1.0)
        )
        self.priority_visit_suppression_max = float(
            np.clip(priority_visit_suppression_max, 0.0, 1.0)
        )
        self.priority_observed_suppression_gain = float(
            np.clip(priority_observed_suppression_gain, 0.0, 1.0)
        )

        # Priority clearing model. Once the robot physically reaches a priority
        # region, or checks it with a short front-FOV cone, that region becomes
        # checked and must not be regenerated as a high-priority target. The
        # published priority map marks checked cells as -1 for RViz debugging.
        self.priority_clear_fov_rad = math.radians(float(priority_clear_fov_deg))
        self.priority_clear_max_range_m = max(float(priority_clear_max_range_m), self.resolution)
        self.priority_clear_robot_radius_m = max(float(priority_clear_robot_radius_m), self.resolution)
        self.priority_clear_min_value = float(np.clip(priority_clear_min_value, 0.0, 100.0))
        # Gaussian priority clearing. A checked priority region is not a hard 1-cell ray anymore;
        # nearby cells receive a continuous observation probability and are marked checked
        # when the Gaussian weight is sufficiently high.
        self.priority_clear_sigma_m = max(float(priority_clear_sigma_m), self.resolution)
        self.priority_clear_angle_sigma_rad = max(
            math.radians(float(priority_clear_angle_sigma_deg)),
            1e-6,
        )
        self.priority_clear_min_weight = float(np.clip(priority_clear_min_weight, 0.0, 1.0))
        self.priority_clear_visit_sigma_m = max(float(priority_clear_visit_sigma_m), self.resolution)

        # Wall/open-space support parameters.
        # These attributes are used by _occupied_density_score_map() and
        # compute_forward_structure_scores(). Keep them in ExplorationGridMap,
        # not only in GazeboNavEnv, because priority/reward scoring is computed here.
        self.wall_support_radius_m = max(float(wall_support_radius_m), self.resolution)
        self.wall_support_density_threshold = max(
            float(wall_support_density_threshold),
            1e-6,
        )
        self.open_space_front_distance_m = max(
            float(open_space_front_distance_m),
            self.resolution,
        )
        self.open_space_side_width_m = max(
            float(open_space_side_width_m),
            self.resolution,
        )

        # Expand maps in fixed chunks to reduce full-array realloc/copy overhead.
        # 64 cells at 0.05 m resolution equals 3.2 m per expansion chunk.
        self.map_expand_chunk_cells = max(int(map_expand_chunk_cells), 1)

        # Hot-path planning controls. Multi-path reward is useful, but planning
        # too many BFS paths every RL step dominates wall-clock time. These
        # values cap the number of candidate paths while still exposing several
        # distinct valid directions to reward.py.
        self.max_planned_candidates = max(int(max_planned_candidates), 1)
        self.max_alternative_paths = max(int(max_alternative_paths), 1)
        self.path_visual_publish_every_n = max(int(path_visual_publish_every_n), 0)

        self.free_logodds_delta = float(free_logodds_delta)
        self.occupied_logodds_delta = float(occupied_logodds_delta)
        self.max_logodds_abs = float(max_logodds_abs)
        self.slam_prior_confidence = float(slam_prior_confidence)
        self.use_slam_prior = bool(use_slam_prior)
        self.front_fov_rad = math.radians(float(front_fov_deg))

        self.width = int(round(self.size_m / self.resolution))
        self.height = int(round(self.size_m / self.resolution))

        self.base_grid = np.full((self.height, self.width), self.UNKNOWN, dtype=np.int16)
        self.correction_logodds_grid = np.zeros((self.height, self.width), dtype=np.float32)
        self.confidence_grid = np.zeros((self.height, self.width), dtype=np.float32)
        self._occupied_persistence_grid = np.zeros((self.height, self.width), dtype=np.uint8)
        try:
            _hit_guard_m = float(os.environ.get("TB3_RL_CONFIDENCE_LIDAR_HIT_GUARD_M", "0.05"))
        except Exception:
            _hit_guard_m = 0.05
        self.confidence_lidar_hit_guard_m = float(np.clip(_hit_guard_m, 0.0, 0.50))
        try:
            _lidar_occ_radius = int(os.environ.get("TB3_RL_CONFIDENCE_LIDAR_OCCLUSION_RADIUS_CELLS", "2"))
        except Exception:
            _lidar_occ_radius = 2
        self.confidence_lidar_occlusion_radius_cells = int(np.clip(_lidar_occ_radius, 0, 4))
        self._last_confidence_lidar_blocked_beams = 0
        self._last_confidence_lidar_guarded_cells = 0
        self._last_confidence_endpoint_confidence_skips = 0
        self.priority_grid = np.zeros((self.height, self.width), dtype=np.float32)
        self._persistent_priority_seed_grid = np.zeros((self.height, self.width), dtype=np.float32)
        self._persistent_priority_allowed_grid = np.zeros((self.height, self.width), dtype=bool)
        self._last_priority_cluster_spawn_step = -10_000_000
        # 0..1 persistent mask that suppresses priority around already explored / visited regions.
        # It prevents a door-like gap from remaining attractive after the robot has already checked it.
        self.priority_suppression_grid = np.zeros((self.height, self.width), dtype=np.float32)
        # True means this area has already been checked for priority purposes.
        # It is a hard exclusion for future priority recomputation.
        self.priority_checked_grid = np.zeros((self.height, self.width), dtype=bool)
        # One-shot reward bookkeeping for checked(-1) cells that would later
        # become valid priority candidates again.  Without this mask the same
        # checked cell can pay a positive reward on every priority recompute.
        self.priority_rechecked_rewarded_grid = np.zeros((self.height, self.width), dtype=bool)
        self.visit_grid = np.zeros((self.height, self.width), dtype=np.int32)
        self.last_seen_grid = np.full((self.height, self.width), -1, dtype=np.int32)
        self.grid = np.full((self.height, self.width), self.UNKNOWN, dtype=np.int8)

        self.prev_known_cells = 0
        self.prev_mean_confidence = 0.0
        self.prev_priority_score = 0.0
        self._priority_dirty = True
        self._last_priority_invalidated_cells = 0
        self._last_priority_invalidated_gain = 0.0
        self._last_priority_rechecked_cells = 0
        self._last_priority_rechecked_gain = 0.0

        # Target hysteresis state. Target selection used to be a per-step global
        # argmax over unknown/low-confidence/priority candidates, which made
        # frontier_angle flip between left/right candidates. Keep a selected
        # target for a short horizon unless a clearly better target appears.
        self.target_lock_steps = max(int(target_lock_steps), 0)
        self.target_switch_margin = float(np.clip(float(target_switch_margin), 0.0, 1.0))
        self._locked_target_ix: Optional[int] = None
        self._locked_target_iy: Optional[int] = None
        self._locked_target_type: str = self.TARGET_NONE
        self._target_lock_age: int = 0
        self._last_target_switched: bool = False
        self._prev_path_target_key: Optional[tuple[int, int, str]] = None
        self._prev_path_distance: Optional[float] = None

        map_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )

        self.publish_topic = str(publish_topic).strip()
        self.confidence_publish_topic = str(confidence_publish_topic).strip()
        self.priority_publish_topic = "" if self.disable_priority_map else str(priority_publish_topic).strip()
        self.map_pub = None
        self.confidence_pub = None
        self.priority_pub = None
        if self.publish_topic:
            self.map_pub = self.node.create_publisher(OccupancyGrid, self.publish_topic, map_qos)
        if self.confidence_publish_topic:
            self.confidence_pub = self.node.create_publisher(
                OccupancyGrid,
                self.confidence_publish_topic,
                map_qos,
            )
        if self.priority_publish_topic:
            self.priority_pub = self.node.create_publisher(
                OccupancyGrid,
                self.priority_publish_topic,
                map_qos,
            )

        # RL-only filtered SLAM view. Do not overwrite /map published by slam_toolbox.
        # This topic mirrors the post-filter base_grid actually used by priority/path/CNN logic.
        self.filtered_slam_publish_topic = str(filtered_slam_publish_topic).strip()
        self.publish_slam_aligned = bool(publish_slam_aligned)
        self._slam_publish_ref = None
        self.filtered_slam_pub = None
        if self.filtered_slam_publish_topic:
            self.filtered_slam_pub = self.node.create_publisher(
                OccupancyGrid,
                self.filtered_slam_publish_topic,
                map_qos,
            )

        self.path_publish_topic = str(path_publish_topic).strip()
        self.path_pub = None
        self._last_path_world: list[tuple[float, float]] = []
        if self.path_publish_topic:
            self.path_pub = self.node.create_publisher(
                NavPath,
                self.path_publish_topic,
                map_qos,
            )

        self.legacy_memory_topic = str(legacy_memory_publish_topic).strip()
        self.legacy_memory_pub = None
        if self.legacy_memory_topic:
            # Backward-compatible alias for old RViz configs.
            # The old independent remember map is removed; this topic now mirrors
            # the 5-channel task-map visualization layer, so stale RViz displays
            # using /rl_memory_map still receive a valid OccupancyGrid.
            self.legacy_memory_pub = self.node.create_publisher(
                OccupancyGrid,
                self.legacy_memory_topic,
                map_qos,
            )

        self.keepalive_publish_period_sec = max(float(keepalive_publish_period_sec), 0.0)
        self._keepalive_timer = None
        self._has_any_map_publisher = any(
            pub is not None
            for pub in (
                self.map_pub,
                self.confidence_pub,
                self.priority_pub,
                self.filtered_slam_pub,
                self.path_pub,
                self.legacy_memory_pub,
            )
        )
        if self._has_any_map_publisher and self.keepalive_publish_period_sec > 0.0:
            # RViz sometimes subscribes after the first reset publish. Re-publish
            # full maps periodically so Map displays do not stay at "No map received".
            self._keepalive_timer = self.node.create_timer(
                self.keepalive_publish_period_sec,
                self.publish,
            )

        # Publish a valid empty initial map only when a visualization topic exists.
        if self._has_any_map_publisher:
            self.publish()

        self.node.get_logger().info(
            f"SLAM task/confidence map publishers: "
            f"task={publish_topic}, confidence={confidence_publish_topic}, priority={(self.priority_publish_topic or '(disabled)')}, "
            f"frame_id={self.frame_id}, size={self.width}x{self.height}, resolution={self.resolution}, "
            f"front_fov_deg={math.degrees(self.front_fov_rad):.1f}, "
            f"front_angle_sigma_deg={math.degrees(self.front_angle_sigma_rad):.1f}, "
            f"confidence_max_range={self.confidence_max_range:.2f}, "
            f"confidence_lidar_hit_guard_m={self.confidence_lidar_hit_guard_m:.2f}, "
            f"confidence_lidar_occ_radius={self.confidence_lidar_occlusion_radius_cells}, "
            f"seen_confidence_floor={self.seen_confidence_floor:.1f}, "
            f"confidence_decay={'disabled' if self.confidence_decay_per_step <= 0.0 else self.confidence_decay_per_step}, "
            f"clear_conf_on_slam_occupied={self.clear_confidence_on_slam_occupied}, "
            f"conf_occ_confirm_steps={self.confidence_occupied_confirm_steps}, "
            f"conf_decay_near_obs_scale={self.confidence_decay_near_obstacle_scale:.2f}, "
            f"conf_decay_obs_ring_radius={self.confidence_decay_obstacle_ring_radius}, "
            f"conf_obs_floor_ratio={self.confidence_obstacle_floor_ratio:.2f}, "
            f"gap_confidence_suppression=disabled, "
            f"priority_disabled={self.disable_priority_map}, priority_birth_delta={self.priority_birth_max_delta_per_recompute:.2f}/recompute, "
            f"priority_gap_width=[{self.gap_min_width_m:.2f},{self.gap_max_width_m:.2f}]m, "
            f"priority_visit_suppression_radius={self.priority_visit_suppression_radius_m:.2f}m, "
            f"priority_visit_suppression_gain={self.priority_visit_suppression_gain:.2f}, "
            f"priority_observed_suppression_gain={self.priority_observed_suppression_gain:.2f}, "
            f"priority_clear_fov_deg={math.degrees(self.priority_clear_fov_rad):.1f}, "
            f"priority_clear_max_range={self.priority_clear_max_range_m:.2f}m, "
            f"priority_clear_robot_radius={self.priority_clear_robot_radius_m:.2f}m, "
            f"priority_clear_sigma={self.priority_clear_sigma_m:.2f}m, "
            f"priority_clear_angle_sigma={math.degrees(self.priority_clear_angle_sigma_rad):.1f}deg, "
            f"priority_clear_min_weight={self.priority_clear_min_weight:.2f}, "
            f"auto_expand=local_robot_window, expand_chunk_cells={self.map_expand_chunk_cells}, "
            f"max_planned_candidates={self.max_planned_candidates}, "
            f"max_alternative_paths={self.max_alternative_paths}, "
            f"path_visual_publish_every_n={self.path_visual_publish_every_n}, "
            f"path_topic={self.path_publish_topic or '(disabled)'}, "
            f"filtered_slam_topic={self.filtered_slam_publish_topic or '(disabled)'}, "
            f"publish_slam_aligned={self.publish_slam_aligned}, "
            f"cnn_channels={4 if self.disable_priority_map else 5}, "
            f"legacy_memory_alias={self.legacy_memory_topic or '(disabled)'}, "
            f"keepalive_publish_period={self.keepalive_publish_period_sec:.2f}s"
        )

        try:
            _canon_default = str(os.environ.get("TB3_RL_LIDAR_CANONICAL_FRONT_ZERO", "0") or "0")
            self._cached_use_canonical_scan_angles = str(os.environ.get(
                "TB3_RL_CONFIDENCE_USE_CANONICAL_SCAN_ANGLES", _canon_default
            )).strip().lower() in {"1", "true", "yes", "on", "enable", "enabled"}
        except Exception:
            self._cached_use_canonical_scan_angles = False
        try:
            self._cached_canonical_front_index = int(os.environ.get("TB3_RL_LIDAR_FRONT_INDEX", "0") or 0)
        except Exception:
            self._cached_canonical_front_index = 0
        try:
            self._cached_canonical_flip_lr = str(os.environ.get("TB3_RL_LIDAR_FLIP_LR", "0") or "0").strip().lower() in {"1", "true", "yes", "on"}
        except Exception:
            self._cached_canonical_flip_lr = False
        try:
            _off_default = str(os.environ.get("TB3_RL_LIDAR_ANGLE_OFFSET_DEG", "0.0") or "0.0")
            self._cached_canonical_angle_offset_rad = math.radians(float(os.environ.get(
                "TB3_RL_CONFIDENCE_SCAN_ANGLE_OFFSET_DEG", _off_default
            ) or 0.0))
        except Exception:
            self._cached_canonical_angle_offset_rad = 0.0

    def reset(self):
        self.base_grid.fill(self.UNKNOWN)
        self.correction_logodds_grid.fill(0.0)
        self.confidence_grid.fill(0.0)
        if self._occupied_persistence_grid.shape == self.confidence_grid.shape:
            self._occupied_persistence_grid.fill(0)
        else:
            self._occupied_persistence_grid = np.zeros_like(self.confidence_grid, dtype=np.uint8)
        self.priority_grid.fill(0.0)
        if hasattr(self, "_persistent_priority_seed_grid") and isinstance(self._persistent_priority_seed_grid, np.ndarray):
            self._persistent_priority_seed_grid.fill(0.0)
        if hasattr(self, "_persistent_priority_allowed_grid") and isinstance(self._persistent_priority_allowed_grid, np.ndarray):
            self._persistent_priority_allowed_grid.fill(False)
        self._last_priority_cluster_spawn_step = -10_000_000
        self.priority_suppression_grid.fill(0.0)
        self.priority_checked_grid.fill(False)
        if hasattr(self, "priority_rechecked_rewarded_grid"):
            self.priority_rechecked_rewarded_grid.fill(False)
        self._last_lidar_visible_free_mask = np.zeros_like(self.priority_grid, dtype=bool)
        self._last_lidar_priority_clear_weight = np.zeros_like(self.priority_grid, dtype=np.float32)
        self._last_lidar_hit_wall_mask = np.zeros_like(self.priority_grid, dtype=bool)
        self._last_lidar_hit_cells_for_priority = []
        self._last_lidar_visible_step = -1
        self._last_confidence_lidar_blocked_beams = 0
        self._last_confidence_lidar_guarded_cells = 0
        self._last_confidence_endpoint_confidence_skips = 0
        self._random_priority_epoch = None
        self._random_priority_seed_cache = None
        self.visit_grid.fill(0)
        self.last_seen_grid.fill(-1)
        self.grid.fill(self.UNKNOWN)
        self.prev_known_cells = 0
        self.prev_mean_confidence = 0.0
        self.prev_priority_score = 0.0
        self._last_priority_invalidated_cells = 0
        self._last_priority_invalidated_gain = 0.0
        self._last_priority_rechecked_cells = 0
        self._last_priority_rechecked_gain = 0.0
        self.update_count = 0
        self.step_index = 0
        self._priority_dirty = True
        self._last_slam_sample_key = None
        self._base_grid_needs_resample = True
        self._publish_resample_cache_key = None
        self._publish_resample_cache = None
        self._slam_publish_ref = None
        self._prev_wall_clamp_mask = None
        self._reset_slam_update_reward_bookkeeping()
        self._reset_target_lock()
        self.publish()

    def reset_centered_at(self, robot_xy: np.ndarray):
        """
        Reset maps and center the initial grid around the robot. After reset the
        maps are still allowed to auto-expand as SLAM/map or robot motion grows.
        """
        cx = float(robot_xy[0])
        cy = float(robot_xy[1])

        self.origin_x = cx - self.initial_size_m * 0.5
        self.origin_y = cy - self.initial_size_m * 0.5
        self.size_m = self.initial_size_m
        self.width = int(round(self.size_m / self.resolution))
        self.height = int(round(self.size_m / self.resolution))

        self.base_grid = np.full((self.height, self.width), self.UNKNOWN, dtype=np.int16)
        self.correction_logodds_grid = np.zeros((self.height, self.width), dtype=np.float32)
        self.confidence_grid = np.zeros((self.height, self.width), dtype=np.float32)
        self._occupied_persistence_grid = np.zeros((self.height, self.width), dtype=np.uint8)
        self.priority_grid = np.zeros((self.height, self.width), dtype=np.float32)
        self._persistent_priority_seed_grid = np.zeros((self.height, self.width), dtype=np.float32)
        self._persistent_priority_allowed_grid = np.zeros((self.height, self.width), dtype=bool)
        self._last_priority_cluster_spawn_step = -10_000_000
        # 0..1 persistent mask that suppresses priority around already explored / visited regions.
        # It prevents a door-like gap from remaining attractive after the robot has already checked it.
        self.priority_suppression_grid = np.zeros((self.height, self.width), dtype=np.float32)
        self.priority_checked_grid = np.zeros((self.height, self.width), dtype=bool)
        self.priority_rechecked_rewarded_grid = np.zeros((self.height, self.width), dtype=bool)
        self._last_lidar_visible_free_mask = np.zeros((self.height, self.width), dtype=bool)
        self._last_lidar_priority_clear_weight = np.zeros((self.height, self.width), dtype=np.float32)
        self._last_lidar_visible_step = -1
        self._random_priority_epoch = None
        self._random_priority_seed_cache = None
        self.visit_grid = np.zeros((self.height, self.width), dtype=np.int32)
        self.last_seen_grid = np.full((self.height, self.width), -1, dtype=np.int32)
        self.grid = np.full((self.height, self.width), self.UNKNOWN, dtype=np.int8)

        self.reset()

    def update(
        self,
        scan: LaserScan,
        robot_xy: np.ndarray,
        robot_yaw: float,
        publish: bool = True,
        slam_map: Optional[OccupancyGrid] = None,
        sensor_xy: Optional[np.ndarray] = None,
        sensor_yaw: Optional[float] = None,
    ) -> MapUpdateStats:
        self.step_index += 1
        self._last_priority_invalidated_cells = 0
        self._last_priority_invalidated_gain = 0.0
        self._last_priority_rechecked_cells = 0
        self._last_priority_rechecked_gain = 0.0
        self._apply_temporal_decay()

        # In map/map/map mode the internal confidence/priority canvas must be
        # the exact SLAM /map canvas.  Lock to the latest /map before any local
        # LiDAR/robot bounds logic; otherwise the RL maps grow in a separate
        # robot-centered rectangle and RViz alignment drifts.
        if self.use_slam_prior and slam_map is not None:
            self._sample_slam_base(slam_map)
        elif self.use_slam_prior and slam_map is None:
            pass  # Keep base_grid as-is during SLAM gap; skip canvas switch
        elif not self.use_slam_prior:
            self.base_grid.fill(self.UNKNOWN)

        def _same_frame_name(a, b) -> bool:
            return str(a or "").strip().lstrip("/") == str(b or "").strip().lstrip("/") and bool(str(a or "").strip())

        slam_frame_for_update = str(getattr(getattr(slam_map, "header", None), "frame_id", "") or "").strip() if slam_map is not None else ""
        map_canvas_locked_mode = bool(self.use_slam_prior and slam_map is not None and _same_frame_name(self.frame_id, slam_frame_for_update))

        # Only non-map-locked operation may grow from LiDAR/robot bounds.  In
        # map-locked mode, growth is exclusively driven by SLAM /map metadata.
        if not map_canvas_locked_mode:
            local_pad = max(self.initial_size_m * 0.5, self.confidence_max_range + 0.5)
            self._ensure_world_bounds(
                float(robot_xy[0]) - local_pad,
                float(robot_xy[0]) + local_pad,
                float(robot_xy[1]) - local_pad,
                float(robot_xy[1]) + local_pad,
                padding_m=0.0,
            )

        prev_known = self.known_cell_count()
        prev_mean_conf = self.mean_confidence()
        confidence_gain_accum = 0.0
        prev_priority_score = self.priority_score()

        stale_before = self._stale_mask()
        observed_mask = np.zeros((self.height, self.width), dtype=bool)
        confidence_updated_mask = np.zeros((self.height, self.width), dtype=bool)
        priority_clear_mask = np.zeros((self.height, self.width), dtype=np.float32)

        def _remap_update_mask(mask: np.ndarray, old_ox: float, old_oy: float, old_w: int, old_h: int, fill, dtype):
            """Remap a per-update temporary mask after the persistent map canvas changes.

            update() may lock the internal canvas to a newly grown SLAM /map or, in
            odom mode, expand the canvas for a LiDAR ray.  Temporary masks such as
            priority_clear_mask must follow the same origin/size shift; otherwise
            later visibility masking crashes or, worse, clears confidence/priority
            in the wrong physical cells.
            """
            if isinstance(mask, np.ndarray) and mask.shape == (self.height, self.width):
                return mask
            out = np.full((self.height, self.width), fill, dtype=dtype)
            if not isinstance(mask, np.ndarray) or mask.shape != (int(old_h), int(old_w)):
                return out
            try:
                off_x = int(round((float(old_ox) - float(self.origin_x)) / max(float(self.resolution), 1e-9)))
                off_y = int(round((float(old_oy) - float(self.origin_y)) / max(float(self.resolution), 1e-9)))
                src_x0 = max(0, -off_x)
                src_y0 = max(0, -off_y)
                dst_x0 = max(0, off_x)
                dst_y0 = max(0, off_y)
                cols = min(int(old_w) - src_x0, int(self.width) - dst_x0)
                rows = min(int(old_h) - src_y0, int(self.height) - dst_y0)
                if cols > 0 and rows > 0:
                    out[dst_y0:dst_y0 + rows, dst_x0:dst_x0 + cols] = mask[src_y0:src_y0 + rows, src_x0:src_x0 + cols]
            except Exception:
                pass
            return out

        def _is_slam_canvas_locked() -> bool:
            try:
                if slam_map is None:
                    return False
                frame = str(self.frame_id or "").strip().lstrip("/")
                slam_frame = str(getattr(getattr(slam_map, "header", None), "frame_id", "") or "").strip().lstrip("/")
                return bool(frame and slam_frame and frame == slam_frame)
            except Exception:
                return False

        slam_canvas_locked = _is_slam_canvas_locked()

        robot_ix, robot_iy = self.world_to_map(float(robot_xy[0]), float(robot_xy[1]))
        _prev_robot_ix_dbg = getattr(self, "_last_robot_ix", None)
        _prev_robot_iy_dbg = getattr(self, "_last_robot_iy", None)
        self._last_robot_ix = int(robot_ix)
        self._last_robot_iy = int(robot_iy)
        self._last_robot_yaw = float(robot_yaw)

        # Camera/front confidence rule:
        #   - FOV selection stays in LaserScan local coordinates: rel_angle around 0 rad only.
        #   - World/map projection uses the current scan-frame yaw, not motion direction.
        #   - No backward/motion-aligned confidence painting is performed here.
        # The base pose remains the owner for visit count, crop center and episode
        # bookkeeping; the scan pose owns only LiDAR/camera-front ray orientation.
        ray_xy = np.asarray(robot_xy, dtype=np.float32)
        ray_yaw = float(robot_yaw)
        try:
            if sensor_xy is not None:
                _sxy = np.asarray(sensor_xy, dtype=np.float32).reshape(-1)
                if _sxy.size >= 2 and np.all(np.isfinite(_sxy[:2])):
                    ray_xy = _sxy[:2].astype(np.float32, copy=True)
            if sensor_yaw is not None and math.isfinite(float(sensor_yaw)):
                ray_yaw = float(sensor_yaw)
        except Exception:
            ray_xy = np.asarray(robot_xy, dtype=np.float32)
            ray_yaw = float(robot_yaw)

        ray_ix, ray_iy = self.world_to_map(float(ray_xy[0]), float(ray_xy[1]))
        self._last_ray_ix = int(ray_ix)
        self._last_ray_iy = int(ray_iy)
        self._last_ray_yaw = float(ray_yaw)

        robot_visit_count = 0
        lidar_hit_cells_for_priority: list[tuple[int, int, int, float]] = []
        lidar_hit_wall_mask = np.zeros((self.height, self.width), dtype=bool)
        confidence_lidar_blocked_beams = 0
        confidence_lidar_guarded_cells = 0
        confidence_endpoint_confidence_skips = 0
        confidence_hit_guard_cells = max(
            int(math.ceil(float(getattr(self, "confidence_lidar_hit_guard_m", 0.0)) / max(float(self.resolution), 1e-6))),
            0,
        )
        if self.in_bounds(robot_ix, robot_iy):
            self.visit_grid[robot_iy, robot_ix] += 1
            robot_visit_count = int(self.visit_grid[robot_iy, robot_ix])
            _robot_conf_delta = self._observe_cell(
                robot_ix,
                robot_iy,
                logodds_delta=self.free_logodds_delta,
                confidence_gain=16.0,
                observed_mask=observed_mask,
                confidence_floor=100.0,
            )
            if float(_robot_conf_delta or 0.0) > 1e-6:
                confidence_gain_accum += float(_robot_conf_delta) / 100.0
                if confidence_updated_mask.shape == self.confidence_grid.shape:
                    confidence_updated_mask[int(robot_iy), int(robot_ix)] = True
            if not self.disable_priority_map:
                self._mark_priority_clear_visit(priority_clear_mask, robot_ix, robot_iy)

        ranges = np.asarray(scan.ranges, dtype=np.float32)
        angle_min = float(scan.angle_min)
        angle_increment = float(scan.angle_increment)
        range_min = max(float(scan.range_min), 0.05)
        range_max = min(float(scan.range_max), self.max_range)
        confirmation_range_max = min(range_max, self.confidence_max_range)
        lidar_barrier_range_max = min(
            range_max,
            max(float(confirmation_range_max), float(getattr(self, "priority_clear_max_range_m", 0.0))),
        )

        def _is_lidar_hit_range(r_value: float, max_range_m: float) -> bool:
            if not math.isfinite(float(r_value)):
                return False
            if float(r_value) < float(range_min):
                return False
            if float(r_value) > float(max_range_m):
                return False
            # A finite value near scan.range_max is usually a max-range return,
            # not an obstacle.  Anything clearly below range_max is a physical hit.
            return float(r_value) < float(range_max) * 0.995

        # v111: use the same "front index = 0" convention as the policy LiDAR
        # pipeline when requested.  Gazebo /scan metadata may describe angles as
        # -pi..pi while the physical/policy convention treats beam 0 as the robot
        # front.  If confidence uses raw LaserScan angles while the policy/debug
        # scan uses canonical front-zero angles, the magenta cone follows the
        # robot position but points in the wrong direction.
        use_canonical_scan_angles = getattr(self, "_cached_use_canonical_scan_angles", False)
        canonical_front_index = getattr(self, "_cached_canonical_front_index", 0)
        canonical_flip_lr = getattr(self, "_cached_canonical_flip_lr", False)
        canonical_angle_offset_rad = getattr(self, "_cached_canonical_angle_offset_rad", 0.0)

        def _confidence_rel_angle_for_index(idx: int) -> float:
            if use_canonical_scan_angles and ranges.size > 0:
                n = max(int(ranges.size), 1)
                fi = int(canonical_front_index) % n
                a = (float(int(idx) - fi) * (2.0 * math.pi / float(n)))
                if canonical_flip_lr:
                    a = -a
                a += float(canonical_angle_offset_rad)
                return normalize_angle(a)
            return normalize_angle(angle_min + float(idx) * angle_increment)

        # Hot path optimization only: structural occupancy is immutable during
        # this update call, so reuse it for all LiDAR ray occlusion checks.
        # This preserves semantics while avoiding thousands of _structural_grid()
        # rebuilds per second at 10Hz live-map update.
        occlusion_struct = self._structural_grid()
        occlusion_threshold = self._slam_occupied_threshold()
        # Keep the raw (non-inflated) occupancy so close-range rays can be tested
        # against real walls only.  Without this, the robot's own inflation halo
        # truncates every confidence/priority ray at the first step whenever the
        # robot is within the inflation radius of a wall (tight corridors, corners,
        # reset noise), which is the "confidence stops updating" failure.
        raw_occlusion_struct = occlusion_struct
        occlusion_inflate_radius = 2
        # Use an inflated SLAM wall as the ray barrier.  A single-cell /map wall
        # can otherwise be bypassed by diagonal Bresenham rays and the auxiliary
        # confidence/priority layers look like they are going through the wall.
        try:
            _occ = occlusion_struct >= occlusion_threshold
            _inflated = self._dilate_bool(_occ, radius=occlusion_inflate_radius)
            if np.any(_inflated):
                occlusion_struct = np.asarray(occlusion_struct, dtype=np.int16)
                occlusion_struct[_inflated] = max(int(occlusion_threshold) + 20, 100)
        except Exception:
            pass
        # Allow rays to pass the synthetic inflation halo for a few cells around
        # the robot (radius + 1), but never past a real wall.
        occlusion_near_skip_cells = int(occlusion_inflate_radius) + 1

        # Pre-compute the bounding box of all ray endpoints so _ensure_world_bounds
        # is called at most once before the ray loop, avoiding repeated array
        # reallocations inside the loop.
        if not slam_canvas_locked and ranges.size > 0:
            _pre_min_x = float(ray_xy[0])
            _pre_max_x = float(ray_xy[0])
            _pre_min_y = float(ray_xy[1])
            _pre_max_y = float(ray_xy[1])
            for _pi in range(0, ranges.size, self.lidar_stride):
                _pr = _confidence_rel_angle_for_index(_pi)
                _in_cfov = abs(_pr) <= self.front_fov_rad * 0.5
                _in_pfov = abs(_pr) <= self.priority_clear_fov_rad * 0.5
                if not _in_cfov and not _in_pfov:
                    continue
                _pr_raw = float(ranges[_pi])
                _pr_r = confirmation_range_max if not np.isfinite(_pr_raw) else float(np.clip(_pr_raw, range_min, confirmation_range_max))
                _ba = ray_yaw + _pr
                _ex = float(ray_xy[0]) + _pr_r * math.cos(_ba)
                _ey = float(ray_xy[1]) + _pr_r * math.sin(_ba)
                _pre_min_x = min(_pre_min_x, _ex)
                _pre_max_x = max(_pre_max_x, _ex)
                _pre_min_y = min(_pre_min_y, _ey)
                _pre_max_y = max(_pre_max_y, _ey)
            old_ox, old_oy = float(self.origin_x), float(self.origin_y)
            old_w, old_h = int(self.width), int(self.height)
            self._ensure_world_bounds(
                _pre_min_x, _pre_max_x,
                _pre_min_y, _pre_max_y,
                padding_m=0.50,
            )
            if (old_w, old_h) != (int(self.width), int(self.height)) or abs(old_ox - float(self.origin_x)) > 1e-9 or abs(old_oy - float(self.origin_y)) > 1e-9:
                observed_mask = _remap_update_mask(observed_mask, old_ox, old_oy, old_w, old_h, False, bool)
                confidence_updated_mask = _remap_update_mask(confidence_updated_mask, old_ox, old_oy, old_w, old_h, False, bool)
                priority_clear_mask = _remap_update_mask(priority_clear_mask, old_ox, old_oy, old_w, old_h, 0.0, np.float32)
                lidar_hit_wall_mask = _remap_update_mask(lidar_hit_wall_mask, old_ox, old_oy, old_w, old_h, False, bool)
                occlusion_struct = self._structural_grid()
                occlusion_threshold = self._slam_occupied_threshold()
                raw_occlusion_struct = occlusion_struct
                try:
                    _occ = occlusion_struct >= occlusion_threshold
                    _inflated = self._dilate_bool(_occ, radius=occlusion_inflate_radius)
                    if np.any(_inflated):
                        occlusion_struct = np.asarray(occlusion_struct, dtype=np.int16)
                        occlusion_struct[_inflated] = max(int(occlusion_threshold) + 20, 100)
                except Exception:
                    pass

        current_lidar_hit_wall_mask = np.zeros((self.height, self.width), dtype=bool)
        current_lidar_barrier = np.zeros((self.height, self.width), dtype=bool)
        if ranges.size > 0 and lidar_barrier_range_max > 0.0:
            current_lidar_fov = max(float(self.front_fov_rad), float(self.priority_clear_fov_rad))
            for _hi in range(0, ranges.size):
                _hr = float(ranges[_hi])
                if not _is_lidar_hit_range(_hr, lidar_barrier_range_max):
                    continue
                _hrel = _confidence_rel_angle_for_index(_hi)
                if abs(_hrel) > current_lidar_fov * 0.5:
                    continue
                _hx = float(ray_xy[0]) + _hr * math.cos(float(ray_yaw) + _hrel)
                _hy = float(ray_xy[1]) + _hr * math.sin(float(ray_yaw) + _hrel)
                _hix, _hiy = self.world_to_map(_hx, _hy)
                if self.in_bounds(_hix, _hiy):
                    current_lidar_hit_wall_mask[int(_hiy), int(_hix)] = True

        # Preserve the SLAM-only raw structure for destructive cleanup decisions.
        # LiDAR hit cells below are temporary ray barriers; they must block new
        # confidence writes but must not erase existing confidence memory.
        raw_slam_only_occlusion_struct = raw_occlusion_struct
        if np.any(current_lidar_hit_wall_mask):
            occ_value = max(int(occlusion_threshold) + 20, 100)
            raw_with_lidar = np.asarray(raw_occlusion_struct, dtype=np.int16).copy()
            raw_with_lidar[current_lidar_hit_wall_mask] = occ_value
            raw_occlusion_struct = raw_with_lidar

            radius = max(int(getattr(self, "confidence_lidar_occlusion_radius_cells", 1)), 0)
            if radius > 0:
                current_lidar_barrier = self._dilate_bool(current_lidar_hit_wall_mask, radius=radius)
            else:
                current_lidar_barrier = current_lidar_hit_wall_mask
            struct_with_lidar = np.asarray(occlusion_struct, dtype=np.int16).copy()
            struct_with_lidar[current_lidar_barrier] = occ_value
            occlusion_struct = struct_with_lidar

        for i in range(0, ranges.size, self.lidar_stride):
            rel_angle = _confidence_rel_angle_for_index(i)

            in_confidence_fov = abs(rel_angle) <= self.front_fov_rad * 0.5
            in_priority_clear_fov = abs(rel_angle) <= self.priority_clear_fov_rad * 0.5

            # Confidence and priority clearing are intentionally decoupled.
            # Priority clearing may use a wider Gaussian front cone than the confidence map,
            # while confidence itself still uses the stricter front FOV.
            if not in_confidence_fov and not in_priority_clear_fov:
                continue

            angle_weight = self._front_angle_weight(rel_angle) if in_confidence_fov else 0.0

            r_raw = float(ranges[i])
            if not np.isfinite(r_raw):
                r = confirmation_range_max
                hit = False
            else:
                if r_raw > confirmation_range_max:
                    r = confirmation_range_max
                    hit = False
                else:
                    hit = _is_lidar_hit_range(r_raw, confirmation_range_max)
                    r = float(np.clip(r_raw, range_min, confirmation_range_max))

            beam_angle = ray_yaw + rel_angle
            end_x = float(ray_xy[0]) + r * math.cos(beam_angle)
            end_y = float(ray_xy[1]) + r * math.sin(beam_angle)

            robot_ix, robot_iy = self.world_to_map(float(robot_xy[0]), float(robot_xy[1]))
            ray_ix, ray_iy = self.world_to_map(float(ray_xy[0]), float(ray_xy[1]))
            end_ix, end_iy = self.world_to_map(end_x, end_y)

            if not self.in_bounds(ray_ix, ray_iy):
                continue

            cells = self.bresenham(ray_ix, ray_iy, end_ix, end_iy)
            if not cells:
                continue

            # /map visibility gate. Even if a LaserScan beam or Gaussian priority
            # clear region geometrically extends farther, confidence and priority
            # checked(-1) must stop at the first occupied SLAM /map cell.
            visible_cells, slam_blocked = self._truncate_ray_by_slam_occlusion(
                cells,
                include_blocking_cell=True,
                struct=occlusion_struct,
                occupied_threshold=occlusion_threshold,
                near_skip_cells=occlusion_near_skip_cells,
                raw_struct=raw_occlusion_struct,
            )
            if not visible_cells:
                continue

            # Priority checked(-1) must mean "directly inspected free/known
            # space".  Do not include the blocking endpoint: if the beam stopped
            # because of a LiDAR hit or a SLAM wall, that endpoint is obstacle
            # evidence, not a visited/free cell.  Also do not use Gaussian spill
            # around the endpoint; wide coverage comes from the front FOV's many
            # individual rays.
            effective_hit = bool(hit or slam_blocked)
            free_cells = visible_cells[:-1] if effective_hit else visible_cells

            if in_priority_clear_fov and not self.disable_priority_map:
                self._mark_priority_clear_ray(
                    priority_clear_mask,
                    ray_ix,
                    ray_iy,
                    free_cells,
                    angle_weight=self._priority_clear_angle_weight(rel_angle),
                )

            if not in_confidence_fov:
                continue

            # LiDAR/SLAM-blocked beams are obstacle evidence, not visual
            # confirmation of the obstacle cell or the thin band right next to it.
            # Keep confidence updates for definitely visible free-space before the
            # hit so open side-rays still refresh normally. Keep only a one-cell
            # default guard band; too much guard makes narrow spaces look unseen.
            confidence_free_cells = free_cells
            if effective_hit:
                confidence_lidar_blocked_beams += 1
                if confidence_hit_guard_cells > 0 and len(confidence_free_cells) > 0:
                    keep_count = max(0, len(confidence_free_cells) - int(confidence_hit_guard_cells))
                    confidence_lidar_guarded_cells += int(len(confidence_free_cells) - keep_count)
                    confidence_free_cells = confidence_free_cells[:keep_count]

            total_free = max(len(confidence_free_cells), 1)

            for j, (cx, cy) in enumerate(confidence_free_cells):
                if not self.in_bounds(cx, cy):
                    continue
                if (
                    isinstance(current_lidar_barrier, np.ndarray)
                    and current_lidar_barrier.shape == self.confidence_grid.shape
                    and current_lidar_barrier[int(cy), int(cx)]
                ):
                    continue

                dist = r * (float(j + 1) / float(total_free))
                weight = self._distance_weight(dist) * angle_weight
                _conf_delta = self._observe_cell(
                    cx,
                    cy,
                    logodds_delta=self.free_logodds_delta * weight,
                    confidence_gain=12.0 * weight,
                    observed_mask=observed_mask,
                    confidence_floor=self.seen_confidence_floor * angle_weight,
                )
                if float(_conf_delta or 0.0) > 1e-6:
                    confidence_gain_accum += float(_conf_delta) / 100.0
                    if confidence_updated_mask.shape == self.confidence_grid.shape:
                        confidence_updated_mask[int(cy), int(cx)] = True

            if effective_hit:
                ox, oy = visible_cells[-1]
                if self.in_bounds(ox, oy):
                    # Store actual LiDAR/SLAM hit endpoints.  Priority entrance seeds
                    # are built from pairs of these black obstacle dots, and the hit
                    # mask is also used as an extra wall barrier so confidence/priority
                    # cannot survive behind a physical obstacle that LiDAR saw before
                    # SLAM fully closes the wall.
                    try:
                        lidar_hit_cells_for_priority.append((int(ox), int(oy), int(i), float(r)))
                        lidar_hit_wall_mask[int(oy), int(ox)] = True
                    except Exception:
                        pass
                    # Use the actual grid distance to the visible endpoint, because
                    # /map occlusion can truncate the scan before the LaserScan range.
                    d_cells = math.sqrt(float((int(ox) - int(robot_ix)) ** 2 + (int(oy) - int(robot_iy)) ** 2))
                    endpoint_dist = d_cells * self.resolution
                    weight = self._distance_weight(endpoint_dist) * angle_weight
                    clear_real_slam_wall_confidence = bool(slam_blocked) and self._is_occupied_in_structural_grid(
                        raw_slam_only_occlusion_struct,
                        occlusion_threshold,
                        int(ox),
                        int(oy),
                    )
                    if clear_real_slam_wall_confidence and not (
                        (int(ox) == int(ray_ix) and int(oy) == int(ray_iy))
                        or (int(ox) == int(robot_ix) and int(oy) == int(robot_iy))
                    ):
                        self.confidence_grid[int(oy), int(ox)] = 0.0
                    self._observe_cell(
                        ox,
                        oy,
                        logodds_delta=self.occupied_logodds_delta * weight,
                        confidence_gain=0.0,
                        observed_mask=None,
                        confidence_floor=0.0,
                    )
                    confidence_endpoint_confidence_skips += 1

        self._last_confidence_lidar_blocked_beams = int(confidence_lidar_blocked_beams)
        self._last_confidence_lidar_guarded_cells = int(confidence_lidar_guarded_cells)
        self._last_confidence_endpoint_confidence_skips = int(confidence_endpoint_confidence_skips)

        # Clip Gaussian priority clear by actual /map line-of-sight. This prevents
        # cells behind walls from becoming checked(-1) just because a Gaussian blob
        # overlapped them.
        if priority_clear_mask.shape != self.priority_grid.shape:
            priority_clear_mask = _remap_update_mask(
                priority_clear_mask,
                float(self.origin_x),
                float(self.origin_y),
                int(priority_clear_mask.shape[1]) if isinstance(priority_clear_mask, np.ndarray) and priority_clear_mask.ndim == 2 else 0,
                int(priority_clear_mask.shape[0]) if isinstance(priority_clear_mask, np.ndarray) and priority_clear_mask.ndim == 2 else 0,
                0.0,
                np.float32,
            )

        if np.any(priority_clear_mask):
            extra_vis_raw = os.environ.get("TB3_RL_PRIORITY_CLEAR_EXTRA_VISIBILITY_MASK", "0")
            extra_visibility_mask = str(extra_vis_raw).strip().lower() in {"1", "true", "yes", "on", "enable", "enabled"}
            if extra_visibility_mask:
                visibility_mask = self._slam_visibility_mask_from_robot(
                    robot_ix=robot_ix,
                    robot_iy=robot_iy,
                    robot_yaw=robot_yaw,
                    max_range_m=self.priority_clear_max_range_m,
                    fov_rad=self.priority_clear_fov_rad,
                )
                if visibility_mask.shape == priority_clear_mask.shape:
                    priority_clear_mask *= visibility_mask
                else:
                    # Last-resort protection against stale temporary masks after a
                    # SLAM resize: never clear priority through an unknown-sized mask.
                    priority_clear_mask = np.zeros_like(self.priority_grid, dtype=np.float32)

        # Store the current LiDAR-visible free-space mask for priority generation.
        # SLAM line-of-sight alone is not enough: freshly observed physical obstacles
        # may not yet be in /map, so random priority targets must also be constrained
        # by the latest LaserScan free rays.  This is also what stops Gaussian
        # priority from appearing behind a wall/hit that LiDAR just saw.
        try:
            if bool(getattr(self, "disable_priority_map", False)):
                self._last_lidar_visible_free_mask = np.zeros_like(self.priority_grid, dtype=bool)
                self._last_lidar_priority_clear_weight = np.zeros_like(self.priority_grid, dtype=np.float32)
                self._last_lidar_visible_step = int(self.step_index)
            else:
                lidar_visible_free = priority_clear_mask >= max(float(self.priority_clear_min_weight), 1e-4)
                struct_now = self._structural_grid()
                occ_now = struct_now >= self._slam_occupied_threshold()
                if lidar_visible_free.shape == occ_now.shape:
                    lidar_visible_free &= ~occ_now
                self._last_lidar_visible_free_mask = lidar_visible_free.astype(bool, copy=False)
                self._last_lidar_priority_clear_weight = np.clip(priority_clear_mask, 0.0, 1.0).astype(np.float32, copy=False)
                self._last_lidar_visible_step = int(self.step_index)
        except Exception:
            self._last_lidar_visible_free_mask = np.zeros_like(self.priority_grid, dtype=bool)
            self._last_lidar_priority_clear_weight = np.zeros_like(self.priority_grid, dtype=np.float32)
            self._last_lidar_visible_step = int(self.step_index)

        # Store latest physical hit wall before the final clamp.  SLAM can lag
        # behind LaserScan; this mask closes those gaps immediately for auxiliary
        # maps.
        try:
            if lidar_hit_wall_mask.shape == self.priority_grid.shape:
                self._last_lidar_hit_wall_mask = self._dilate_bool(lidar_hit_wall_mask.astype(bool), radius=2)
                self._last_lidar_hit_cells_for_priority = list(lidar_hit_cells_for_priority[-720:])
        except Exception:
            self._last_lidar_hit_wall_mask = np.zeros_like(self.priority_grid, dtype=bool)
            self._last_lidar_hit_cells_for_priority = []

        # Hard wall/component clamp after applying this LiDAR frame.  This removes
        # stale confidence/priority that survived behind newly drawn SLAM/LiDAR walls.
        strict_component_mask = self._apply_strict_wall_visibility_clamp(robot_ix, robot_iy)
        if strict_component_mask is not None and priority_clear_mask.shape == strict_component_mask.shape:
            priority_clear_mask *= strict_component_mask.astype(np.float32)

        # Priority can be disabled for real-robot safety/eval.  Keep the map channel
        # shape unchanged for the trained policy, but force it to all zeros and do
        # not let priority affect reward, target selection, or reset gates.
        if self.disable_priority_map:
            if self.priority_grid.shape != self.base_grid.shape:
                self.priority_grid = np.zeros_like(self.base_grid, dtype=np.float32)
            else:
                self.priority_grid.fill(0.0)
            if self.priority_checked_grid.shape != self.base_grid.shape:
                self.priority_checked_grid = np.zeros_like(self.base_grid, dtype=bool)
            else:
                self.priority_checked_grid.fill(False)
            if self.priority_suppression_grid.shape != self.base_grid.shape:
                self.priority_suppression_grid = np.zeros_like(self.base_grid, dtype=np.float32)
            else:
                self.priority_suppression_grid.fill(0.0)
            if hasattr(self, "_persistent_priority_seed_grid") and isinstance(self._persistent_priority_seed_grid, np.ndarray):
                if self._persistent_priority_seed_grid.shape != self.base_grid.shape:
                    self._persistent_priority_seed_grid = np.zeros_like(self.base_grid, dtype=np.float32)
                else:
                    self._persistent_priority_seed_grid.fill(0.0)
            if hasattr(self, "_persistent_priority_allowed_grid") and isinstance(self._persistent_priority_allowed_grid, np.ndarray):
                if self._persistent_priority_allowed_grid.shape != self.base_grid.shape:
                    self._persistent_priority_allowed_grid = np.zeros_like(self.base_grid, dtype=bool)
                else:
                    self._persistent_priority_allowed_grid.fill(False)
            priority_cleared_cells, priority_clear_gain = 0, 0.0
            self._last_priority_invalidated_cells = 0
            self._last_priority_invalidated_gain = 0.0
            self._last_priority_rechecked_cells = 0
            self._last_priority_rechecked_gain = 0.0
            self.prev_priority_score = 0.0
            self._priority_dirty = False
        else:
            # Lower priority around regions that the robot has already physically reached
            # or has just confirmed with the front-FOV sensor model. This is separate
            # from confidence: a gap may remain structurally important, but once the
            # robot has explored it, it should stop being repeatedly selected as a
            # high-priority target.
            priority_cleared_cells, priority_clear_gain = self._update_priority_checked(priority_clear_mask)

            self._update_priority_suppression(robot_ix, robot_iy, observed_mask)

            # Recompute priority map after confidence/SLAM update. Track priority that
            # disappears because the newer SLAM geometry no longer supports the old
            # door/gap hypothesis. This is diagnostic only; it is not rewarded.
            #
            # v25.7: respect --priority-recompute-interval strictly. `_priority_dirty`
            # means "refresh eventually", not "refresh immediately". Without this,
            # confidence/SLAM updates set dirty almost every step and the interval flag
            # looked ineffective.
            priority_interval = max(int(self.priority_recompute_interval), 1)
            steps_since_priority = int(self.step_index) - int(getattr(self, "_last_priority_recompute_step", -10_000_000))
            interval_tick = (int(self.step_index) % priority_interval) == 0
            dirty_ready = bool(self._priority_dirty) and steps_since_priority >= priority_interval
            initial_priority = getattr(self, "_last_priority_recompute_step", -10_000_000) < 0

            if initial_priority or interval_tick or dirty_ready:
                old_active_priority = np.clip(self._active_priority_grid(), 0.0, 100.0)
                self._recompute_priority_grid()
                self._last_priority_recompute_step = int(self.step_index)
                new_active_priority = np.clip(self._active_priority_grid(), 0.0, 100.0)
                dropped = (old_active_priority >= self.priority_clear_min_value) & (new_active_priority < 1.0)
                if self.priority_checked_grid.shape == dropped.shape:
                    dropped &= ~self.priority_checked_grid
                dropped_cells = int(np.count_nonzero(dropped))
                dropped_gain = float(np.sum(old_active_priority[dropped] / 100.0))
                if dropped_cells > 0:
                    self._last_priority_invalidated_cells += dropped_cells
                    self._last_priority_invalidated_gain += dropped_gain
                self._priority_dirty = False

                if os.environ.get("TB3_RL_QUIET_PRIORITY_LOGS", "0").strip().lower() not in {"1", "true", "yes", "on"} and (int(self.step_index) - int(getattr(self, "_last_priority_recompute_debug_step", -10_000_000))) >= max(priority_interval, 200):
                    try:
                        active_cells = int(np.count_nonzero(new_active_priority >= max(self.priority_clear_min_value, 1.0)))
                        self.node.get_logger().info(
                            "PRIORITY_RECOMPUTE | step=%d interval=%d dirty=%s active=%d"
                            % (int(self.step_index), int(priority_interval), str(bool(self._priority_dirty)), active_cells)
                        )
                        self._last_priority_recompute_debug_step = int(self.step_index)
                    except Exception:
                        pass

        stale_refresh_cells = int(np.count_nonzero(stale_before & observed_mask))

        known = self.known_cell_count()
        new_known = max(known - prev_known, 0)

        total_cells = float(self.width * self.height)
        coverage = known / max(total_cells, 1.0)
        coverage_delta = coverage - (self.prev_known_cells / max(total_cells, 1.0))
        self.prev_known_cells = known

        mean_conf = self.mean_confidence()
        confidence_observed_cells = 0
        confidence_updated_cells = 0
        try:
            confidence_observed_cells = int(np.count_nonzero(observed_mask)) if isinstance(observed_mask, np.ndarray) else 0
        except Exception:
            confidence_observed_cells = 0
        try:
            if isinstance(confidence_updated_mask, np.ndarray) and confidence_updated_mask.shape == self.confidence_grid.shape:
                confidence_gain = float(max(confidence_gain_accum, 0.0))
                confidence_updated_cells = int(np.count_nonzero(confidence_updated_mask))
            else:
                confidence_gain = max(mean_conf - prev_mean_conf, 0.0)
                confidence_updated_cells = int(confidence_observed_cells)
        except Exception:
            confidence_gain = max(mean_conf - prev_mean_conf, 0.0)
        self.prev_mean_confidence = mean_conf
        try:
            self._last_confidence_observed_cells = int(confidence_observed_cells)
            self._last_confidence_updated_cells = int(confidence_updated_cells)
            self._last_confidence_gain_cells = float(confidence_gain)
        except Exception:
            pass

        priority_score = self.priority_score()
        priority_gain = max(priority_score - prev_priority_score, 0.0)
        self.prev_priority_score = priority_score

        if bool(getattr(self, "fast_no_priority_stats", False)) and bool(getattr(self, "disable_priority_map", False)):
            wall_support_score, open_space_score = 0.0, 0.0
            nearest_obstacle_distance, obstacle_proximity_score = 999.0, 0.0
            frontier_count = 0
            frontier_distance = self.size_m
            frontier_angle = 0.0
            target_priority = 0.0
            target_type = self.TARGET_NONE
            target_reachable = False
            path_distance = self.size_m
            path_angle = 0.0
            path_progress = 0.0
            alternative_path_count = 0
            alternative_path_angles = ()
        else:
            wall_support_score, open_space_score = self.compute_forward_structure_scores(
                robot_xy=robot_xy,
                robot_yaw=robot_yaw,
            )
            nearest_obstacle_distance, obstacle_proximity_score = self.compute_obstacle_proximity_score(
                robot_xy=robot_xy,
                warning_radius_m=0.55,
                hard_radius_m=0.20,
            )

            (
                frontier_count,
                frontier_distance,
                frontier_angle,
                target_priority,
                target_type,
                target_reachable,
                path_distance,
                path_angle,
                path_progress,
                alternative_path_count,
                alternative_path_angles,
            ) = self.compute_frontier_info(robot_xy=robot_xy, robot_yaw=robot_yaw)

        stale_known_cells = self.stale_known_count()
        stale_ratio = stale_known_cells / max(float(known), 1.0)

        low_confidence_cells = self.low_confidence_count()
        base_free_cells = int(np.count_nonzero((self.base_grid >= 0) & (self.base_grid <= 35)))
        low_confidence_ratio = low_confidence_cells / max(float(base_free_cells), 1.0)

        try:
            dbg_n = int(os.environ.get("TB3_RL_CONFIDENCE_DEBUG_EVERY_N", "100"))
        except Exception:
            dbg_n = 100
        if dbg_n > 0 and (int(self.step_index) <= 5 or int(self.step_index) % dbg_n == 0):
            try:
                observed_cells_dbg = int(np.count_nonzero(observed_mask)) if isinstance(observed_mask, np.ndarray) else -1
                yaw_delta_dbg = normalize_angle(float(ray_yaw) - float(robot_yaw))
                self.node.get_logger().info(
                    "CONFIDENCE_UPDATE | "
                    f"step={int(self.step_index)} mode=camera_front gain={float(confidence_gain):.4f} "
                    f"mean={float(mean_conf):.3f} observed={observed_cells_dbg} "
                    f"blocked_beams={int(getattr(self, '_last_confidence_lidar_blocked_beams', 0))} "
                    f"guarded_cells={int(getattr(self, '_last_confidence_lidar_guarded_cells', 0))} "
                    f"endpoint_skips={int(getattr(self, '_last_confidence_endpoint_confidence_skips', 0))} "
                    f"known={int(known)} low={int(low_confidence_cells)} "
                    f"fov={math.degrees(float(self.front_fov_rad)):.1f}deg "
                    f"robot=({int(robot_ix)},{int(robot_iy)}) ray=({int(ray_ix)},{int(ray_iy)}) "
                    f"dCell=({int(robot_ix) - int(_prev_robot_ix_dbg) if _prev_robot_ix_dbg is not None else 0},"
                    f"{int(robot_iy) - int(_prev_robot_iy_dbg) if _prev_robot_iy_dbg is not None else 0}) "
                    f"baseYaw={math.degrees(float(robot_yaw)):.1f}deg "
                    f"scanYaw={math.degrees(float(ray_yaw)):.1f}deg "
                    f"scanBaseDelta={math.degrees(float(yaw_delta_dbg)):.1f}deg "
                    f"slam_locked={bool(slam_canvas_locked)}"
                )
            except Exception:
                pass

        stats = MapUpdateStats(
            known_cells=int(known),
            new_known_cells=int(new_known),
            coverage_ratio=float(coverage),
            coverage_delta=float(coverage_delta),
            frontier_count=int(frontier_count),
            frontier_distance=float(frontier_distance),
            frontier_angle=float(frontier_angle),
            robot_visit_count=int(robot_visit_count),
            mean_confidence=float(mean_conf),
            stale_known_cells=int(stale_known_cells),
            stale_ratio=float(stale_ratio),
            low_confidence_cells=int(low_confidence_cells),
            low_confidence_ratio=float(low_confidence_ratio),
            stale_refresh_cells=int(stale_refresh_cells),
            confidence_gain=float(confidence_gain),
            confidence_observed_cells=int(confidence_observed_cells),
            confidence_updated_cells=int(confidence_updated_cells),
            target_priority=float(target_priority),
            target_type=str(target_type),
            target_switched=bool(self._last_target_switched),
            target_lock_age=int(self._target_lock_age),
            target_reachable=bool(target_reachable),
            path_distance=float(path_distance),
            path_angle=float(path_angle),
            path_progress=float(path_progress),
            alternative_path_count=int(alternative_path_count),
            alternative_path_angles=tuple(float(a) for a in alternative_path_angles),
            priority_score=float(priority_score),
            priority_gain=float(priority_gain),
            priority_cleared_cells=int(priority_cleared_cells),
            priority_clear_gain=float(priority_clear_gain),
            priority_invalidated_cells=int(self._last_priority_invalidated_cells),
            priority_invalidated_gain=float(self._last_priority_invalidated_gain),
            priority_rechecked_cells=int(getattr(self, "_last_priority_rechecked_cells", 0)),
            priority_rechecked_gain=float(getattr(self, "_last_priority_rechecked_gain", 0.0)),
            wall_support_score=float(wall_support_score),
            open_space_score=float(open_space_score),
            nearest_obstacle_distance=float(nearest_obstacle_distance),
            obstacle_proximity_score=float(obstacle_proximity_score),
            slam_update_new_known_cells=int(getattr(self, "_last_slam_update_new_known_cells", 0)),
            slam_update_new_free_cells=int(getattr(self, "_last_slam_update_new_free_cells", 0)),
            slam_update_new_occupied_cells=int(getattr(self, "_last_slam_update_new_occupied_cells", 0)),
            slam_update_expand_known_cells=int(getattr(self, "_last_slam_update_expand_known_cells", 0)),
        )

        self.update_count += 1
        if (
            publish
            and self.publish_every_n > 0
            and self._has_any_map_publisher
            and (self.update_count == 1 or self.update_count % self.publish_every_n == 0)
        ):
            self.publish()

        return stats

    def _apply_temporal_decay(self):
        if self.confidence_decay_per_step > 0.0:
            decay = float(np.clip(self.confidence_decay_per_step, 0.0, 1.0))
            keep_default = np.float32(max(0.0, 1.0 - decay))
            self.confidence_grid *= keep_default

            # Near static obstacles, map alignment can jitter by a few cells and
            # confidence tends to be erased too aggressively. Apply weaker decay
            # and a soft floor around occupied rings.
            try:
                struct = self._structural_grid()
                occ = np.asarray(struct >= self._slam_occupied_threshold(), dtype=bool)
                if occ.shape == self.confidence_grid.shape and np.any(occ):
                    ring = self._dilate_bool(occ, radius=max(int(self.confidence_decay_obstacle_ring_radius), 1))
                    ring &= ~occ
                    if np.any(ring):
                        near_scale = float(np.clip(self.confidence_decay_near_obstacle_scale, 0.0, 1.0))
                        if near_scale < 1.0:
                            keep_near = np.float32(max(0.0, 1.0 - decay * near_scale))
                            self.confidence_grid[ring] /= keep_default
                            self.confidence_grid[ring] *= keep_near
                        floor = np.float32(
                            np.clip(
                                float(self.seen_confidence_floor) * float(self.confidence_obstacle_floor_ratio),
                                0.0,
                                100.0,
                            )
                        )
                        if floor > 0.0:
                            self.confidence_grid[ring] = np.maximum(self.confidence_grid[ring], floor)
            except Exception:
                pass

            np.clip(self.confidence_grid, 0.0, 100.0, out=self.confidence_grid)

        if self.logodds_decay_per_step > 0.0:
            self.correction_logodds_grid *= np.float32(max(0.0, 1.0 - float(self.logodds_decay_per_step)))
            np.clip(
                self.correction_logodds_grid,
                -float(self.max_logodds_abs),
                float(self.max_logodds_abs),
                out=self.correction_logodds_grid,
            )

    def _stable_occupied_mask_for_confidence(self, occupied_mask: np.ndarray) -> np.ndarray:
        """Return occupied cells that remained occupied for N consecutive updates.

        SLAM walls can oscillate by 1-2 cells while pose-graph is settling. Clearing
        confidence on every one-frame occupied flicker causes near-wall confidence
        to repeatedly disappear. This keeps confidence cleanup conservative.
        """
        occ = np.asarray(occupied_mask, dtype=bool)
        if occ.shape != (self.height, self.width):
            self._occupied_persistence_grid = np.zeros((self.height, self.width), dtype=np.uint8)
            return np.zeros((self.height, self.width), dtype=bool)

        if (
            not isinstance(getattr(self, "_occupied_persistence_grid", None), np.ndarray)
            or self._occupied_persistence_grid.shape != occ.shape
        ):
            self._occupied_persistence_grid = np.zeros_like(occ, dtype=np.uint8)

        streak = self._occupied_persistence_grid
        np.add(streak, np.uint8(1), out=streak, where=occ)
        streak[~occ] = np.uint8(0)

        confirm_steps = max(int(getattr(self, "confidence_occupied_confirm_steps", 3)), 1)
        if confirm_steps <= 1:
            return occ
        return occ & (streak >= np.uint8(confirm_steps))

    def _observe_cell(
        self,
        ix: int,
        iy: int,
        logodds_delta: float,
        confidence_gain: float,
        observed_mask: Optional[np.ndarray] = None,
        confidence_floor: float = 0.0,
    ) -> float:
        if not self.in_bounds(ix, iy):
            return 0.0

        if abs(float(logodds_delta)) > 1e-9:
            lo = float(self.correction_logodds_grid[iy, ix]) + float(logodds_delta)
            if not math.isfinite(lo):
                lo = 0.0
            self.correction_logodds_grid[iy, ix] = max(-self.max_logodds_abs, min(self.max_logodds_abs, lo))

        cg = float(confidence_gain)
        if not math.isfinite(cg):
            cg = 0.0
        old_confidence = float(self.confidence_grid[iy, ix])
        new_confidence = old_confidence + cg
        if confidence_floor > 0.0:
            new_confidence = max(new_confidence, float(confidence_floor))
        clipped_confidence = max(0.0, min(100.0, new_confidence))
        self.confidence_grid[iy, ix] = clipped_confidence
        self.last_seen_grid[iy, ix] = self.step_index

        if observed_mask is not None and observed_mask.shape == self.confidence_grid.shape:
            observed_mask[iy, ix] = True
        return max(float(clipped_confidence) - old_confidence, 0.0)

    def _distance_weight(self, distance: float) -> float:
        d = max(float(distance), 0.0)
        return float(1.0 / (1.0 + self.distance_weight_beta * d * d))

    def _front_angle_weight(self, rel_angle: float) -> float:
        a = float(rel_angle)
        return float(math.exp(-0.5 * (a / self.front_angle_sigma_rad) ** 2))

    def _grid_index_arrays(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Return cached (yy, xx) index arrays for the current internal map size.

        np.indices() allocation was happening inside multiple hot paths. The map
        size changes only when auto-expand occurs, so caching avoids repeated
        2D array allocation during every RL step.
        """
        shape = (self.height, self.width)
        if self._index_cache_shape != shape or self._index_cache is None:
            self._index_cache = np.indices(shape, dtype=np.float32)
            self._index_cache_shape = shape
        yy, xx = self._index_cache
        return yy, xx


    @staticmethod
    def _quat_to_yaw(q) -> float:
        """Return yaw from a geometry_msgs/Quaternion-like object."""
        try:
            x = float(getattr(q, "x", 0.0))
            y = float(getattr(q, "y", 0.0))
            z = float(getattr(q, "z", 0.0))
            w = float(getattr(q, "w", 1.0))
            return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
        except Exception:
            return 0.0

    def set_slam_publish_reference(self, slam_map: Optional[OccupancyGrid]) -> None:
        """
        Store the current /map metadata as the RViz publication reference.

        Internal RL/confidence/priority maps have their own dynamically centered
        grid. Publishing those grids directly is valid mathematically, but RViz
        overlays can look shifted after slam_toolbox resets /map origin.  For
        visualization, resample the RL layers onto the exact current /map grid
        metadata so /rl_priority_map and /map share resolution, origin, and frame.
        """
        if not bool(getattr(self, "publish_slam_aligned", False)):
            # The RL debug maps are now intentionally published in the same
            # internal odom grid.  Keeping a stale SLAM publication reference
            # around can make /rl_priority_map and /rl_filtered_slam_map use
            # different RViz origins after SLAM reset.
            self._slam_publish_ref = None
            return
        if slam_map is None:
            return
        try:
            width = int(slam_map.info.width)
            height = int(slam_map.info.height)
            resolution = float(slam_map.info.resolution)
        except Exception:
            return
        if width <= 0 or height <= 0 or not np.isfinite(resolution) or resolution <= 0.0:
            return
        origin = slam_map.info.origin
        q = origin.orientation
        stamp = getattr(getattr(slam_map, "header", None), "stamp", None)

        # Keep an exact copy of the latest SLAM grid used as the RViz reference.
        # In map-fixed mode /rl_filtered_slam_map must be byte-for-byte sampled on
        # the same /map canvas, not rebuilt from the internal robot-centered RL
        # grid.  Otherwise RViz can show the filtered layer drifting or changing
        # size while raw /map remains fixed.
        try:
            self._slam_publish_raw_grid = np.asarray(slam_map.data, dtype=np.int8).reshape((height, width)).copy()
        except Exception:
            self._slam_publish_raw_grid = None

        self._slam_publish_ref = {
            "frame_id": str(getattr(getattr(slam_map, "header", None), "frame_id", self.frame_id) or self.frame_id),
            "stamp_sec": int(getattr(stamp, "sec", 0)) if stamp is not None else 0,
            "stamp_nanosec": int(getattr(stamp, "nanosec", 0)) if stamp is not None else 0,
            "width": width,
            "height": height,
            "resolution": resolution,
            "origin_x": float(origin.position.x),
            "origin_y": float(origin.position.y),
            "origin_z": float(origin.position.z),
            "origin_qx": float(getattr(q, "x", 0.0)),
            "origin_qy": float(getattr(q, "y", 0.0)),
            "origin_qz": float(getattr(q, "z", 0.0)),
            "origin_qw": float(getattr(q, "w", 1.0)),
            "origin_yaw": self._quat_to_yaw(q),
        }
        try:
            self._last_valid_slam_publish_ref = dict(self._slam_publish_ref)
        except Exception:
            pass

    def _resample_grid_to_slam_reference(
        self,
        grid: np.ndarray,
        fill_value,
        dtype=np.int8,
    ) -> Optional[np.ndarray]:
        """Nearest-neighbor sample an internal RL layer onto current /map metadata."""
        ref = getattr(self, "_slam_publish_ref", None)
        if not self.publish_slam_aligned or ref is None:
            return None
        try:
            width = int(ref["width"])
            height = int(ref["height"])
            res = float(ref["resolution"])
            ox = float(ref["origin_x"])
            oy = float(ref["origin_y"])
            yaw = float(ref.get("origin_yaw", 0.0))
        except Exception:
            return None
        if width <= 0 or height <= 0 or res <= 0.0:
            return None

        ref_frame = str(ref.get("frame_id", self.frame_id) or self.frame_id)
        cache_key = (
            width,
            height,
            round(res, 9),
            round(ox, 6),
            round(oy, 6),
            round(yaw, 9),
            self.width,
            self.height,
            round(float(self.resolution), 9),
            round(float(self.origin_x), 6),
            round(float(self.origin_y), 6),
            str(ref_frame),
            str(self.frame_id),
        )
        cached = getattr(self, "_publish_resample_cache", None)
        if cache_key == getattr(self, "_publish_resample_cache_key", None) and cached is not None:
            valid, ix_valid, iy_valid = cached
        else:
            yy, xx = np.indices((height, width), dtype=np.float32)
            local_x = (xx + 0.5) * res
            local_y = (yy + 0.5) * res
            c = math.cos(yaw)
            s = math.sin(yaw)
            wx_ref = ox + local_x * c - local_y * s
            wy_ref = oy + local_x * s + local_y * c

            ref_to_internal = self._lookup_2d_transform(self.frame_id, ref_frame)
            if ref_to_internal is None:
                return None
            wx, wy = self._apply_2d_transform(wx_ref, wy_ref, ref_to_internal)

            ix = np.floor((wx - self.origin_x) / max(self.resolution, 1e-6)).astype(np.int32)
            iy = np.floor((wy - self.origin_y) / max(self.resolution, 1e-6)).astype(np.int32)
            valid = (ix >= 0) & (ix < self.width) & (iy >= 0) & (iy < self.height)
            ix_valid = ix[valid]
            iy_valid = iy[valid]
            self._publish_resample_cache_key = cache_key
            self._publish_resample_cache = (valid, ix_valid, iy_valid)

        out = np.full((height, width), fill_value, dtype=dtype)
        if np.any(valid):
            out[valid] = grid[iy_valid, ix_valid].astype(dtype, copy=False)
        return out


    def _lookup_2d_transform(self, target_frame: str, source_frame: str) -> Optional[tuple[float, float, float]]:
        """
        Return planar transform target_T_source = (tx, ty, yaw).

        This is intentionally local to ExplorationGridMap so SLAM /map can be
        sampled into an odom-locked internal grid without assuming that map and
        odom are numerically identical.  If frames are identical the identity
        transform is returned.  If TF is unavailable, None is returned and the
        caller must skip the SLAM sample rather than silently misalign layers.
        """
        target = str(target_frame or "").strip()
        source = str(source_frame or "").strip()
        if not target or not source or target == source:
            return 0.0, 0.0, 0.0

        tf_buffer = getattr(self.node, "tf_buffer", None)
        if tf_buffer is None:
            return None

        try:
            import rclpy
            transform = tf_buffer.lookup_transform(
                target,
                source,
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.03),
            )
        except Exception:
            return None

        t = transform.transform.translation
        q = transform.transform.rotation
        yaw = self._quat_to_yaw(q)
        return float(t.x), float(t.y), float(yaw)

    @staticmethod
    def _apply_2d_transform(
        x,
        y,
        tf: tuple[float, float, float],
    ):
        """Vectorized planar transform for scalars or numpy arrays."""
        tx, ty, yaw = tf
        c = math.cos(float(yaw))
        s = math.sin(float(yaw))
        return float(tx) + c * x - s * y, float(ty) + s * x + c * y

    def _ensure_slam_known_bounds(
        self,
        data: np.ndarray,
        slam_res: float,
        slam_origin_x: float,
        slam_origin_y: float,
        slam_origin_yaw: float,
        slam_frame_id: str = "",
    ) -> None:
        """Expand all internal RL maps to cover the known SLAM bounding box.

        This expands base/confidence/priority/suppression/checked/visit grids
        together through _ensure_world_bounds(), preserving alignment.  Unknown
        SLAM canvas is ignored so a large empty /map does not explode the task
        map size immediately after reset.
        """
        if data.size == 0 or float(slam_res) <= 0.0:
            return

        known_y, known_x = np.nonzero(np.asarray(data) >= 0)
        if known_x.size == 0:
            return

        min_sx = int(np.min(known_x))
        max_sx = int(np.max(known_x)) + 1
        min_sy = int(np.min(known_y))
        max_sy = int(np.max(known_y)) + 1

        # Use cell-boundary corners, not centers, so the whole known SLAM extent
        # is covered after discretization.
        local_corners = (
            (min_sx * slam_res, min_sy * slam_res),
            (max_sx * slam_res, min_sy * slam_res),
            (min_sx * slam_res, max_sy * slam_res),
            (max_sx * slam_res, max_sy * slam_res),
        )
        c = math.cos(float(slam_origin_yaw))
        s = math.sin(float(slam_origin_yaw))
        # Corners above are first expressed in the SLAM map frame.  Convert them
        # into the internal map frame before expanding the RL/confidence/priority
        # arrays.  This fixes the common RViz failure where /map is in `map` but
        # the RL layers are maintained in `odom`.
        slam_frame = str(slam_frame_id or self.frame_id).strip() or self.frame_id
        slam_to_internal = self._lookup_2d_transform(self.frame_id, slam_frame)
        if slam_to_internal is None:
            # Avoid expanding the odom grid with raw map-frame coordinates; that
            # is exactly what creates the large skewed/shifted overlays.
            if self.node is not None:
                now = time.time()
                if now - float(getattr(self, "_last_slam_bounds_tf_warn_time", 0.0)) > 2.0:
                    self._last_slam_bounds_tf_warn_time = now
                    self.node.get_logger().warn(
                        f"SLAM_BOUNDS_TF_WAIT | cannot transform {slam_frame} -> {self.frame_id}; "
                        "skipping SLAM bounds expansion this cycle"
                    )
            return

        world_x: list[float] = []
        world_y: list[float] = []
        for lx, ly in local_corners:
            sx = float(slam_origin_x) + float(lx) * c - float(ly) * s
            sy = float(slam_origin_y) + float(lx) * s + float(ly) * c
            ix, iy = self._apply_2d_transform(sx, sy, slam_to_internal)
            world_x.append(float(ix))
            world_y.append(float(iy))

        pad = max(0.25, 4.0 * self.resolution)
        self._ensure_world_bounds(
            min(world_x),
            max(world_x),
            min(world_y),
            max(world_y),
            padding_m=pad,
        )

    def _reset_slam_update_reward_bookkeeping(self) -> None:
        """Clear delayed-SLAM reward state for a new episode/reset."""
        self._slam_reward_prev_known_mask = None
        self._slam_reward_prev_origin_x = float(getattr(self, "origin_x", 0.0))
        self._slam_reward_prev_origin_y = float(getattr(self, "origin_y", 0.0))
        self._slam_reward_prev_resolution = float(getattr(self, "resolution", 0.05))
        self._slam_reward_prev_width = int(getattr(self, "width", 0))
        self._slam_reward_prev_height = int(getattr(self, "height", 0))
        self._slam_reward_prev_frame_id = str(getattr(self, "frame_id", "map"))
        self._slam_reward_prev_stamp_key = None
        self._last_slam_update_new_known_cells = 0
        self._last_slam_update_new_free_cells = 0
        self._last_slam_update_new_occupied_cells = 0
        self._last_slam_update_expand_known_cells = 0

    def _record_slam_map_update_delta(
        self,
        data: np.ndarray,
        slam_res: float,
        slam_origin_x: float,
        slam_origin_y: float,
        slam_frame_id: str,
        slam_stamp_key,
    ) -> None:
        """Record unknown->known cells introduced by a newly sampled SLAM /map.

        This is deliberately separate from confidence gain.  Confidence is updated
        synchronously from the latest LaserScan; this method tracks delayed SLAM
        OccupancyGrid updates so the env can give a small capped bonus when the
        global map finally incorporates new free/occupied cells.

        The first map after reset only initializes the previous-known mask.  This
        prevents reset warmup / map bootstrap from being rewarded as exploration.
        """
        self._last_slam_update_new_known_cells = 0
        self._last_slam_update_new_free_cells = 0
        self._last_slam_update_new_occupied_cells = 0
        self._last_slam_update_expand_known_cells = 0
        try:
            cur = np.asarray(data, dtype=np.int16)
            if cur.ndim != 2 or cur.size == 0:
                return
            h, w = int(cur.shape[0]), int(cur.shape[1])
            if h <= 0 or w <= 0:
                return
            # Avoid paying the same /map message more than once when the live
            # map timer and env.step both sample it.
            stamp_key = (
                slam_stamp_key,
                int(w),
                int(h),
                round(float(slam_res), 9),
                round(float(slam_origin_x), 6),
                round(float(slam_origin_y), 6),
                str(slam_frame_id or ""),
                int(np.count_nonzero(cur >= 0)),
            )

            cur_known = cur >= 0
            occ_thr = int(self._slam_occupied_threshold())
            cur_occ = cur >= occ_thr
            cur_free = cur_known & (~cur_occ)

            prev = getattr(self, "_slam_reward_prev_known_mask", None)
            prev_w = int(getattr(self, "_slam_reward_prev_width", 0))
            prev_h = int(getattr(self, "_slam_reward_prev_height", 0))
            prev_res = float(getattr(self, "_slam_reward_prev_resolution", float(slam_res)))
            prev_ox = float(getattr(self, "_slam_reward_prev_origin_x", float(slam_origin_x)))
            prev_oy = float(getattr(self, "_slam_reward_prev_origin_y", float(slam_origin_y)))

            if not isinstance(prev, np.ndarray) or prev.shape != (prev_h, prev_w) or prev_w <= 0 or prev_h <= 0:
                # First accepted SLAM map after reset: initialize only, no reward.
                self._slam_reward_prev_known_mask = cur_known.astype(bool, copy=True)
                self._slam_reward_prev_origin_x = float(slam_origin_x)
                self._slam_reward_prev_origin_y = float(slam_origin_y)
                self._slam_reward_prev_resolution = float(slam_res)
                self._slam_reward_prev_width = int(w)
                self._slam_reward_prev_height = int(h)
                self._slam_reward_prev_frame_id = str(slam_frame_id or self.frame_id)
                self._slam_reward_prev_stamp_key = stamp_key
                return

            # Project previous known mask onto the current SLAM canvas.  This
            # handles /map origin/size expansion without treating all shifted cells
            # as new information.
            yy, xx = np.indices((h, w), dtype=np.float32)
            wx = float(slam_origin_x) + (xx + 0.5) * float(slam_res)
            wy = float(slam_origin_y) + (yy + 0.5) * float(slam_res)
            pix = np.floor((wx - prev_ox) / max(prev_res, 1e-9)).astype(np.int32)
            piy = np.floor((wy - prev_oy) / max(prev_res, 1e-9)).astype(np.int32)
            prev_valid = (pix >= 0) & (pix < prev_w) & (piy >= 0) & (piy < prev_h)
            prev_known_on_cur = np.zeros((h, w), dtype=bool)
            if np.any(prev_valid):
                prev_known_on_cur[prev_valid] = prev[piy[prev_valid], pix[prev_valid]]

            new_known = cur_known & (~prev_known_on_cur)
            expanded_known = cur_known & (~prev_valid)
            if np.any(new_known):
                self._last_slam_update_new_known_cells = int(np.count_nonzero(new_known))
                self._last_slam_update_new_free_cells = int(np.count_nonzero(new_known & cur_free))
                self._last_slam_update_new_occupied_cells = int(np.count_nonzero(new_known & cur_occ))
                self._last_slam_update_expand_known_cells = int(np.count_nonzero(expanded_known))

            self._slam_reward_prev_known_mask = cur_known.astype(bool, copy=True)
            self._slam_reward_prev_origin_x = float(slam_origin_x)
            self._slam_reward_prev_origin_y = float(slam_origin_y)
            self._slam_reward_prev_resolution = float(slam_res)
            self._slam_reward_prev_width = int(w)
            self._slam_reward_prev_height = int(h)
            self._slam_reward_prev_frame_id = str(slam_frame_id or self.frame_id)
            self._slam_reward_prev_stamp_key = stamp_key
        except Exception:
            # Reward bookkeeping must never break map update/training.
            self._last_slam_update_new_known_cells = 0
            self._last_slam_update_new_free_cells = 0
            self._last_slam_update_new_occupied_cells = 0
            self._last_slam_update_expand_known_cells = 0

    def _try_lock_internal_grid_to_slam_canvas(
        self,
        data: np.ndarray,
        slam_res: float,
        slam_origin_x: float,
        slam_origin_y: float,
        slam_origin_yaw: float,
        slam_frame_id: str,
    ) -> bool:
        """Make the internal RL/confidence/priority canvas exactly match /map.

        This path is used when the environment frame is already the SLAM frame
        (the recommended pure-velocity debug command uses map/map/map).  It
        removes the last source of RViz drift: an expanding robot-centered RL
        grid projected onto a separately expanding SLAM grid.  The arrays are
        resampled onto the new /map canvas so confidence follows SLAM growth.
        """
        try:
            frame = str(self.frame_id or "").strip().lstrip("/")
            slam_frame = str(slam_frame_id or "").strip().lstrip("/")
            if not frame or not slam_frame or frame != slam_frame:
                return False
            if abs(float(slam_origin_yaw)) > 1e-3:
                return False
            h, w = int(data.shape[0]), int(data.shape[1])
            if h <= 0 or w <= 0 or not np.isfinite(slam_res) or float(slam_res) <= 0.0:
                return False
        except Exception:
            return False

        old_w = int(getattr(self, "width", 0))
        old_h = int(getattr(self, "height", 0))
        old_res = float(getattr(self, "resolution", slam_res))
        old_ox = float(getattr(self, "origin_x", slam_origin_x))
        old_oy = float(getattr(self, "origin_y", slam_origin_y))

        same_canvas = (
            old_w == w
            and old_h == h
            and abs(old_res - float(slam_res)) < 1e-9
            and abs(old_ox - float(slam_origin_x)) < 1e-6
            and abs(old_oy - float(slam_origin_y)) < 1e-6
        )
        if same_canvas:
            return True

        def resample_old(name: str, fill, dtype):
            old = getattr(self, name, None)
            out = np.full((h, w), fill, dtype=dtype)
            if not isinstance(old, np.ndarray) or old.shape != (old_h, old_w) or old_w <= 0 or old_h <= 0:
                return out
            yy, xx = np.indices((h, w), dtype=np.float32)
            wx = float(slam_origin_x) + (xx + 0.5) * float(slam_res)
            wy = float(slam_origin_y) + (yy + 0.5) * float(slam_res)
            ix = np.floor((wx - old_ox) / max(old_res, 1e-9)).astype(np.int32)
            iy = np.floor((wy - old_oy) / max(old_res, 1e-9)).astype(np.int32)
            valid = (ix >= 0) & (ix < old_w) & (iy >= 0) & (iy < old_h)
            if np.any(valid):
                out[valid] = old[iy[valid], ix[valid]].astype(dtype, copy=False)
            return out

        self.origin_x = float(slam_origin_x)
        self.origin_y = float(slam_origin_y)
        self.resolution = float(slam_res)
        self.width = w
        self.height = h
        self.size_m = max(float(w), float(h)) * float(slam_res)

        self.base_grid = np.full((h, w), self.UNKNOWN, dtype=np.int16)
        self.correction_logodds_grid = resample_old("correction_logodds_grid", 0.0, np.float32)

        # v106: Confidence is not SLAM geometry; it is a record of policy-visible
        # cells.  When Cartographer changes the /map canvas origin/size after a
        # reset or pose-graph correction, nearest-neighbor resampling of old
        # confidence can leave magenta blobs at physically wrong cells.  In the
        # unified mode, keep /rl_confidence_map on the exact current /map canvas
        # and clear the old confidence memory whenever the SLAM canvas changes.
        # This makes the current confidence cone use the same pose/canvas that
        # RViz uses for RobotModel and /map.
        try:
            _clear_conf_on_canvas_change = str(os.environ.get(
                "TB3_RL_CLEAR_CONFIDENCE_ON_SLAM_CANVAS_CHANGE", "0"
            )).strip().lower() in {"1", "true", "yes", "on", "enable", "enabled"}
        except Exception:
            _clear_conf_on_canvas_change = False

        if _clear_conf_on_canvas_change:
            self.confidence_grid = np.zeros((h, w), dtype=np.float32)
            try:
                self.last_seen_grid = np.full((h, w), -1, dtype=np.int32)
                self.visit_grid = np.zeros((h, w), dtype=np.int32)
                self._last_lidar_visible_free_mask = np.zeros((h, w), dtype=bool)
                self._last_lidar_hit_wall_mask = np.zeros((h, w), dtype=bool)
                self._last_lidar_priority_clear_weight = np.zeros((h, w), dtype=np.float32)
            except Exception:
                pass
            try:
                if str(os.environ.get("TB3_RL_CONFIDENCE_CANVAS_DEBUG", "0")).strip().lower() in {"1", "true", "yes", "on"}:
                    self.node.get_logger().warn(
                        "CONFIDENCE_CLEARED_ON_SLAM_CANVAS_CHANGE | "
                        f"old={old_w}x{old_h}@({old_ox:+.3f},{old_oy:+.3f}) "
                        f"new={w}x{h}@({float(slam_origin_x):+.3f},{float(slam_origin_y):+.3f})"
                    )
            except Exception:
                pass
        else:
            self.confidence_grid = resample_old("confidence_grid", 0.0, np.float32)

        self.priority_grid = resample_old("priority_grid", 0.0, np.float32)
        # Occupancy persistence must restart when SLAM canvas changes; old streaks
        # are tied to the previous grid alignment.
        self._occupied_persistence_grid = np.zeros((h, w), dtype=np.uint8)
        self._persistent_priority_seed_grid = resample_old("_persistent_priority_seed_grid", 0.0, np.float32)
        self._persistent_priority_allowed_grid = resample_old("_persistent_priority_allowed_grid", False, bool)
        self.priority_suppression_grid = resample_old("priority_suppression_grid", 0.0, np.float32)
        self.priority_checked_grid = resample_old("priority_checked_grid", False, bool)
        self.priority_rechecked_rewarded_grid = resample_old("priority_rechecked_rewarded_grid", False, bool)
        if _clear_conf_on_canvas_change:
            self._last_lidar_visible_free_mask = np.zeros((h, w), dtype=bool)
            self._last_lidar_priority_clear_weight = np.zeros((h, w), dtype=np.float32)
            self._last_lidar_hit_wall_mask = np.zeros((h, w), dtype=bool)
            self.visit_grid = np.zeros((h, w), dtype=np.int32)
            self.last_seen_grid = np.full((h, w), -1, dtype=np.int32)
        else:
            self._last_lidar_visible_free_mask = resample_old("_last_lidar_visible_free_mask", False, bool)
            self._last_lidar_priority_clear_weight = resample_old("_last_lidar_priority_clear_weight", 0.0, np.float32)
            self._last_lidar_hit_wall_mask = resample_old("_last_lidar_hit_wall_mask", False, bool)
            self.visit_grid = resample_old("visit_grid", 0, np.int32)
            self.last_seen_grid = resample_old("last_seen_grid", -1, np.int32)
        self._last_lidar_hit_cells_for_priority = []
        self.grid = np.full((h, w), self.UNKNOWN, dtype=np.int8)

        self._index_cache_shape = None
        self._index_cache = None
        self._publish_resample_cache_key = None
        self._publish_resample_cache = None
        self._last_slam_fast_lock_key = None
        self._base_grid_needs_resample = True
        return True

    def _sample_slam_base(self, slam_map: OccupancyGrid):
        self.set_slam_publish_reference(slam_map)
        slam_width = int(slam_map.info.width)
        slam_height = int(slam_map.info.height)
        slam_stamp = getattr(getattr(slam_map, "header", None), "stamp", None)
        slam_stamp_key = (getattr(slam_stamp, "sec", 0), getattr(slam_stamp, "nanosec", 0))

        if slam_width <= 0 or slam_height <= 0:
            self.base_grid.fill(self.UNKNOWN)
            return

        try:
            data = np.asarray(slam_map.data, dtype=np.int16).reshape((slam_height, slam_width))
        except Exception:
            self.base_grid.fill(self.UNKNOWN)
            return

        slam_res = float(slam_map.info.resolution)
        slam_origin_x = float(slam_map.info.origin.position.x)
        slam_origin_y = float(slam_map.info.origin.position.y)
        slam_origin_yaw = self._quat_to_yaw(slam_map.info.origin.orientation)
        slam_frame = str(getattr(getattr(slam_map, "header", None), "frame_id", "") or self.frame_id).strip() or self.frame_id

        # v18 fast path: in map-locked mode the env calls update() much faster
        # than Cartographer publishes a new /map.  Re-processing the same
        # OccupancyGrid every step costs tens of milliseconds and repeatedly marks
        # priority dirty even though no SLAM data changed.
        try:
            frame_fast = str(self.frame_id or "").strip().lstrip("/")
            slam_frame_fast = str(slam_frame or "").strip().lstrip("/")
            same_frame_fast = bool(frame_fast and slam_frame_fast and frame_fast == slam_frame_fast)
            same_canvas_fast = (
                int(getattr(self, "width", 0)) == int(slam_width)
                and int(getattr(self, "height", 0)) == int(slam_height)
                and abs(float(getattr(self, "resolution", slam_res)) - float(slam_res)) < 1e-9
                and abs(float(getattr(self, "origin_x", slam_origin_x)) - float(slam_origin_x)) < 1e-6
                and abs(float(getattr(self, "origin_y", slam_origin_y)) - float(slam_origin_y)) < 1e-6
            )
            fast_key = (
                id(slam_map),
                slam_stamp_key,
                int(slam_width),
                int(slam_height),
                round(float(slam_res), 9),
                round(float(slam_origin_x), 6),
                round(float(slam_origin_y), 6),
                str(slam_frame),
            )
            if (
                same_frame_fast
                and same_canvas_fast
                and not bool(getattr(self, "_base_grid_needs_resample", False))
                and getattr(self, "_last_slam_fast_lock_key", None) == fast_key
            ):
                self._last_slam_update_new_known_cells = 0
                self._last_slam_update_new_free_cells = 0
                self._last_slam_update_new_occupied_cells = 0
                self._last_slam_update_expand_known_cells = 0
                return
        except Exception:
            fast_key = None

        self._record_slam_map_update_delta(
            data=data,
            slam_res=slam_res,
            slam_origin_x=slam_origin_x,
            slam_origin_y=slam_origin_y,
            slam_frame_id=slam_frame,
            slam_stamp_key=slam_stamp_key,
        )

        # Preferred pure-velocity debugging path: when the env frame is map,
        # lock the internal RL/confidence/priority canvas to the exact SLAM /map
        # canvas.  This makes /rl_confidence_map and /rl_priority_map grow with
        # /map instead of with a separate robot-centered grid.
        if self._try_lock_internal_grid_to_slam_canvas(
            data=data,
            slam_res=slam_res,
            slam_origin_x=slam_origin_x,
            slam_origin_y=slam_origin_y,
            slam_origin_yaw=slam_origin_yaw,
            slam_frame_id=slam_frame,
        ):
            self.base_grid = np.asarray(data, dtype=np.int16).copy()
            self._last_slam_sample_key = None
            self._base_grid_needs_resample = False
            try:
                occupied_internal = self.base_grid >= self._slam_occupied_threshold()
                if np.any(occupied_internal):
                    if bool(getattr(self, "clear_confidence_on_slam_occupied", False)):
                        stable_occ = self._stable_occupied_mask_for_confidence(occupied_internal)
                        if stable_occ.shape == self.confidence_grid.shape and np.any(stable_occ):
                            self.confidence_grid[stable_occ] = 0.0
                    self.priority_grid[occupied_internal] = 0.0
                    self.priority_checked_grid[occupied_internal] = False
                    self.priority_suppression_grid[occupied_internal] = 0.0
            except Exception:
                pass
            self._invalidate_priority_from_slam_geometry()
            self._priority_dirty = True
            try:
                self._last_slam_fast_lock_key = fast_key
            except Exception:
                self._last_slam_fast_lock_key = None
            return

        # Fallback for odom/internal-frame operation: expand to known SLAM bounds
        # and resample through TF.
        self._ensure_slam_known_bounds(
            data=data,
            slam_res=slam_res,
            slam_origin_x=slam_origin_x,
            slam_origin_y=slam_origin_y,
            slam_origin_yaw=slam_origin_yaw,
            slam_frame_id=slam_frame,
        )

        # Cache key must be computed after possible expansion; otherwise the
        # sample cache can say "already sampled" while the internal map size has
        # changed.
        sample_key = (
            slam_stamp_key,
            slam_width,
            slam_height,
            slam_res,
            slam_origin_x,
            slam_origin_y,
            self.width,
            self.height,
            round(float(self.origin_x), 6),
            round(float(self.origin_y), 6),
            str(slam_frame),
            str(self.frame_id),
        )

        if (
            not self._base_grid_needs_resample
            and self._last_slam_sample_key == sample_key
        ):
            return

        yy, xx = self._grid_index_arrays()
        wx_internal = self.origin_x + (xx + 0.5) * self.resolution
        wy_internal = self.origin_y + (yy + 0.5) * self.resolution

        internal_to_slam = self._lookup_2d_transform(slam_frame, self.frame_id)
        if internal_to_slam is None:
            # Do not sample map-frame /map as if it were odom.  Keep the previous
            # accepted base grid and try again when TF becomes available.
            if self.node is not None:
                now = time.time()
                if now - float(getattr(self, "_last_slam_sample_tf_warn_time", 0.0)) > 2.0:
                    self._last_slam_sample_tf_warn_time = now
                    self.node.get_logger().warn(
                        f"SLAM_SAMPLE_TF_WAIT | cannot transform {self.frame_id} -> {slam_frame}; "
                        "holding previous RL structural map"
                    )
            return

        wx, wy = self._apply_2d_transform(wx_internal, wy_internal, internal_to_slam)

        dx = wx - slam_origin_x
        dy = wy - slam_origin_y
        cyaw = math.cos(slam_origin_yaw)
        syaw = math.sin(slam_origin_yaw)
        slam_local_x = dx * cyaw + dy * syaw
        slam_local_y = -dx * syaw + dy * cyaw
        sx = np.floor(slam_local_x / max(slam_res, 1e-6)).astype(np.int32)
        sy = np.floor(slam_local_y / max(slam_res, 1e-6)).astype(np.int32)

        valid = (sx >= 0) & (sx < slam_width) & (sy >= 0) & (sy < slam_height)

        self.base_grid.fill(self.UNKNOWN)
        self.base_grid[valid] = data[sy[valid], sx[valid]]
        try:
            self._last_slam_fast_lock_key = fast_key
        except Exception:
            self._last_slam_fast_lock_key = None

        # Hard wall mask for auxiliary layers. Confidence and priority are
        # traversable-space quantities; they must never occupy SLAM wall cells.
        # This also prevents old confidence/checked cells from surviving after a
        # SLAM reset or after a wall is redrawn in a slightly different place.
        try:
            occupied_internal = self.base_grid >= self._slam_occupied_threshold()
            if np.any(occupied_internal):
                if bool(getattr(self, "clear_confidence_on_slam_occupied", False)) and self.confidence_grid.shape == occupied_internal.shape:
                    stable_occ = self._stable_occupied_mask_for_confidence(occupied_internal)
                    if stable_occ.shape == self.confidence_grid.shape and np.any(stable_occ):
                        self.confidence_grid[stable_occ] = 0.0
                if self.priority_grid.shape == occupied_internal.shape:
                    self.priority_grid[occupied_internal] = 0.0
                if self.priority_checked_grid.shape == occupied_internal.shape:
                    self.priority_checked_grid[occupied_internal] = False
                if self.priority_suppression_grid.shape == occupied_internal.shape:
                    self.priority_suppression_grid[occupied_internal] = 0.0
        except Exception:
            pass

        self._last_slam_sample_key = sample_key
        self._base_grid_needs_resample = False
        self._invalidate_priority_from_slam_geometry()
        self._priority_dirty = True


    def _invalidate_priority_from_slam_geometry(self):
        """
        Immediately remove stale priority state that became impossible after a
        newer SLAM /map sample.

        Example: early SLAM sees two separated obstacle points and the gap looks
        like a doorway, so priority appears between them. Later SLAM fills that
        area as occupied wall. Waiting for the normal priority recomputation can
        leave a locked target/path pointing into a now-blocked cell for a few
        steps. This method uses only the SLAM-derived base_grid and performs a
        hard invalidation before reward/target selection.
        """
        self._last_priority_invalidated_cells = 0
        self._last_priority_invalidated_gain = 0.0
        self._last_priority_rechecked_cells = 0
        self._last_priority_rechecked_gain = 0.0

        if self.priority_grid.shape != self.base_grid.shape:
            return

        if self.priority_checked_grid.shape != self.base_grid.shape:
            self.priority_checked_grid = np.zeros_like(self.base_grid, dtype=bool)
        if self.priority_suppression_grid.shape != self.base_grid.shape:
            self.priority_suppression_grid = np.zeros_like(self.base_grid, dtype=np.float32)

        occupied = self.base_grid >= self._slam_occupied_threshold()
        if not np.any(occupied):
            return

        previous_priority = np.clip(self.priority_grid.astype(np.float32, copy=False), 0.0, 100.0)
        active_priority_on_wall = occupied & (previous_priority >= max(self.priority_clear_min_value, 1.0))
        stale_checked_on_wall = occupied & self.priority_checked_grid
        stale_suppression_on_wall = occupied & (self.priority_suppression_grid > 1e-3)
        invalid = active_priority_on_wall | stale_checked_on_wall | stale_suppression_on_wall

        if not np.any(invalid):
            # Locked target may still become invalid even if priority_grid was already zero.
            if (
                self._locked_target_ix is not None
                and self._locked_target_iy is not None
                and self.in_bounds(int(self._locked_target_ix), int(self._locked_target_iy))
                and bool(occupied[int(self._locked_target_iy), int(self._locked_target_ix)])
            ):
                self._reset_target_lock()
                self._clear_path_visualization()
            return

        counted = active_priority_on_wall
        self._last_priority_invalidated_cells = int(np.count_nonzero(counted))
        self._last_priority_invalidated_gain = float(np.sum(previous_priority[counted] / 100.0))

        # Occupied cells are not “checked priority”; they are invalid geometry.
        # Keep them out of RViz priority(-1) and out of future priority targets.
        self.priority_grid[invalid] = 0.0
        if hasattr(self, "_persistent_priority_seed_grid") and isinstance(self._persistent_priority_seed_grid, np.ndarray) and self._persistent_priority_seed_grid.shape == self.priority_grid.shape:
            self._persistent_priority_seed_grid[invalid] = 0.0
        self.priority_checked_grid[invalid] = False
        self.priority_suppression_grid[invalid] = 0.0

        if (
            self._locked_target_ix is not None
            and self._locked_target_iy is not None
            and self.in_bounds(int(self._locked_target_ix), int(self._locked_target_iy))
            and bool(occupied[int(self._locked_target_iy), int(self._locked_target_ix)])
        ):
            self._reset_target_lock()
            self._clear_path_visualization()

        self._priority_dirty = True

    def _ensure_world_bounds(
        self,
        min_x: float,
        max_x: float,
        min_y: float,
        max_y: float,
        padding_m: float = 0.0,
    ):
        max_map_cells = 800
        if self.width >= max_map_cells and self.height >= max_map_cells:
            return

        pad = max(float(padding_m), 0.0)
        min_x = float(min_x) - pad
        max_x = float(max_x) + pad
        min_y = float(min_y) - pad
        max_y = float(max_y) + pad

        cur_min_x = self.origin_x
        cur_min_y = self.origin_y
        cur_max_x = self.origin_x + self.width * self.resolution
        cur_max_y = self.origin_y + self.height * self.resolution

        if (
            min_x >= cur_min_x
            and max_x <= cur_max_x
            and min_y >= cur_min_y
            and max_y <= cur_max_y
        ):
            return

        # Expand by fixed chunks instead of the minimum number of cells.
        # This avoids repeated full-array realloc/copy when exploration slowly crosses
        # the current boundary. The chunk unit is measured in cells, but bounds are
        # still aligned to the map resolution.
        chunk_m = max(self.map_expand_chunk_cells * self.resolution, self.resolution)

        left_chunks = 0
        right_chunks = 0
        down_chunks = 0
        up_chunks = 0

        if min_x < cur_min_x:
            left_chunks = int(math.ceil((cur_min_x - min_x) / chunk_m))
        if max_x > cur_max_x:
            right_chunks = int(math.ceil((max_x - cur_max_x) / chunk_m))
        if min_y < cur_min_y:
            down_chunks = int(math.ceil((cur_min_y - min_y) / chunk_m))
        if max_y > cur_max_y:
            up_chunks = int(math.ceil((max_y - cur_max_y) / chunk_m))

        if left_chunks == 0 and right_chunks == 0 and down_chunks == 0 and up_chunks == 0:
            return

        new_min_x = cur_min_x - left_chunks * chunk_m
        new_min_y = cur_min_y - down_chunks * chunk_m

        # Keep exact cell counts. Because chunk_m is an integer multiple of resolution,
        # width/height increase by multiples of map_expand_chunk_cells.
        # Capture the pre-expansion shape before mutating any arrays.  Persistent
        # priority state is validated against this old shape, then copied into
        # the expanded canvas below.
        old_w = int(self.width)
        old_h = int(self.height)
        new_width = old_w + (left_chunks + right_chunks) * self.map_expand_chunk_cells
        new_height = old_h + (down_chunks + up_chunks) * self.map_expand_chunk_cells

        off_x = left_chunks * self.map_expand_chunk_cells
        off_y = down_chunks * self.map_expand_chunk_cells

        def expand_array(arr: np.ndarray, fill_value, dtype=None) -> np.ndarray:
            dtype = arr.dtype if dtype is None else dtype
            new_arr = np.full((new_height, new_width), fill_value, dtype=dtype)
            new_arr[off_y:off_y + self.height, off_x:off_x + self.width] = arr
            return new_arr

        self.base_grid = expand_array(self.base_grid, self.UNKNOWN, np.int16)
        self.correction_logodds_grid = expand_array(self.correction_logodds_grid, 0.0, np.float32)
        self.confidence_grid = expand_array(self.confidence_grid, 0.0, np.float32)
        if not hasattr(self, "_occupied_persistence_grid") or not isinstance(self._occupied_persistence_grid, np.ndarray) or self._occupied_persistence_grid.shape != (old_h, old_w):
            self._occupied_persistence_grid = np.zeros((old_h, old_w), dtype=np.uint8)
        self._occupied_persistence_grid = expand_array(self._occupied_persistence_grid, 0, np.uint8)
        self.priority_grid = expand_array(self.priority_grid, 0.0, np.float32)
        if not hasattr(self, "_persistent_priority_seed_grid") or self._persistent_priority_seed_grid.shape != (old_h, old_w):
            self._persistent_priority_seed_grid = np.zeros((old_h, old_w), dtype=np.float32)
        self._persistent_priority_seed_grid = expand_array(self._persistent_priority_seed_grid, 0.0, np.float32)
        if not hasattr(self, "_persistent_priority_allowed_grid") or self._persistent_priority_allowed_grid.shape != (old_h, old_w):
            self._persistent_priority_allowed_grid = np.zeros((old_h, old_w), dtype=bool)
        self._persistent_priority_allowed_grid = expand_array(self._persistent_priority_allowed_grid, False, bool)
        self.priority_suppression_grid = expand_array(self.priority_suppression_grid, 0.0, np.float32)
        self.priority_checked_grid = expand_array(self.priority_checked_grid, False, bool)
        if not hasattr(self, "_last_lidar_visible_free_mask") or self._last_lidar_visible_free_mask.shape != (self.height, self.width):
            self._last_lidar_visible_free_mask = np.zeros((self.height, self.width), dtype=bool)
        if not hasattr(self, "_last_lidar_priority_clear_weight") or self._last_lidar_priority_clear_weight.shape != (self.height, self.width):
            self._last_lidar_priority_clear_weight = np.zeros((self.height, self.width), dtype=np.float32)
        self._last_lidar_visible_free_mask = expand_array(self._last_lidar_visible_free_mask, False, bool)
        self._last_lidar_priority_clear_weight = expand_array(self._last_lidar_priority_clear_weight, 0.0, np.float32)
        if not hasattr(self, "_last_lidar_hit_wall_mask") or not isinstance(self._last_lidar_hit_wall_mask, np.ndarray) or self._last_lidar_hit_wall_mask.shape != (old_h, old_w):
            self._last_lidar_hit_wall_mask = np.zeros((old_h, old_w), dtype=bool)
        self._last_lidar_hit_wall_mask = expand_array(self._last_lidar_hit_wall_mask, False, bool)
        self._last_lidar_hit_cells_for_priority = []
        if not hasattr(self, "priority_rechecked_rewarded_grid") or self.priority_rechecked_rewarded_grid.shape != (self.height, self.width):
            self.priority_rechecked_rewarded_grid = np.zeros((self.height, self.width), dtype=bool)
        self.priority_rechecked_rewarded_grid = expand_array(self.priority_rechecked_rewarded_grid, False, bool)
        self.visit_grid = expand_array(self.visit_grid, 0, np.int32)
        self.last_seen_grid = expand_array(self.last_seen_grid, -1, np.int32)
        self.grid = expand_array(self.grid, self.UNKNOWN, np.int8)

        if self._locked_target_ix is not None and self._locked_target_iy is not None:
            self._locked_target_ix += int(off_x)
            self._locked_target_iy += int(off_y)

        self.origin_x = float(new_min_x)
        self.origin_y = float(new_min_y)
        self.width = int(new_width)
        self.height = int(new_height)
        self.size_m = max(self.width, self.height) * self.resolution
        self._index_cache_shape = None
        self._index_cache = None
        self._base_grid_needs_resample = True
        self._priority_dirty = True
        self._prev_wall_clamp_mask = None
        self._structural_grid_cache = None
        self._inflated_wall_cache = None
        self._density_score_cache = None
        self._stale_mask_cache = None
        self._low_confidence_mask_cache = None

        # Expansion is expected during exploration. Keep this at debug level so
        # normal training logs are not flooded every time the auto-growing maps
        # add a few cells.
        self.node.get_logger().debug(
            f"Auto-expanded RL maps: size={self.width}x{self.height}, "
            f"origin=({self.origin_x:.2f},{self.origin_y:.2f}), "
            f"chunk_cells={self.map_expand_chunk_cells}"
        )

    def known_cell_count(self) -> int:
        return int(np.count_nonzero(self.confidence_grid >= self.min_known_confidence))

    def mean_confidence(self) -> float:
        if self.confidence_grid.size == 0:
            return 0.0
        return float(np.mean(np.clip(self.confidence_grid, 0.0, 100.0)))

    def priority_score(self) -> float:
        if bool(getattr(self, "disable_priority_map", False)):
            return 0.0
        if self.priority_grid.size == 0:
            return 0.0
        active = self._active_priority_grid()
        return float(np.max(np.clip(active / 100.0, 0.0, 1.0)))

    def _active_priority_grid(self) -> np.ndarray:
        """Return priority candidates that are still actionable. Cached per step."""
        cached = getattr(self, "_active_priority_cache", None)
        cached_step = getattr(self, "_active_priority_cache_step", -1)
        if cached is not None and cached_step == self.step_index and cached.shape == (self.height, self.width):
            return cached

        if bool(getattr(self, "disable_priority_map", False)):
            result = np.zeros_like(self.priority_grid, dtype=np.float32)
        else:
            result = np.clip(self.priority_grid, 0.0, 100.0).astype(np.float32)
            if self.priority_checked_grid.shape == result.shape:
                result[self.priority_checked_grid] = 0.0

        self._active_priority_cache = result
        self._active_priority_cache_step = self.step_index
        return result

    def stale_known_count(self) -> int:
        return int(np.count_nonzero(self._stale_mask()))

    def low_confidence_count(self) -> int:
        return int(np.count_nonzero(self._low_confidence_mask()))

    def _stale_mask(self) -> np.ndarray:
        cache_key = (self.step_index, self.height, self.width)
        cached = getattr(self, "_stale_mask_cache", None)
        if cached is not None and getattr(self, "_stale_mask_cache_key", None) == cache_key:
            return cached
        known = self.confidence_grid >= self.min_known_confidence
        seen = self.last_seen_grid >= 0
        age = self.step_index - self.last_seen_grid
        result = known & seen & (age >= self.stale_after_steps)
        self._stale_mask_cache = result
        self._stale_mask_cache_key = cache_key
        return result

    def _low_confidence_mask(self) -> np.ndarray:
        cache_key = (self.step_index, self.height, self.width)
        cached = getattr(self, "_low_conf_mask_cache", None)
        if cached is not None and getattr(self, "_low_conf_mask_cache_key", None) == cache_key:
            return cached
        struct = self._structural_grid()
        base_free = (struct >= 0) & (struct <= 35)
        result = base_free & (self.confidence_grid < self.low_confidence_threshold)
        self._low_conf_mask_cache = result
        self._low_conf_mask_cache_key = cache_key
        return result

    def _structural_grid(self) -> np.ndarray:
        """Occupancy geometry used by priority/RViz/CNN. Cached per step_index."""
        cached = getattr(self, "_structural_grid_cache", None)
        cached_step = getattr(self, "_structural_grid_cache_step", -1)
        cached_shape = getattr(self, "_structural_grid_cache_shape", None)
        if (
            cached is not None
            and cached_step == self.step_index
            and cached_shape == (self.height, self.width)
        ):
            return cached

        grid = np.array(self.base_grid, dtype=np.int16, copy=True)

        if not bool(getattr(self, "use_slam_prior", True)):
            if self.correction_logodds_grid.shape == grid.shape:
                logodds = self.correction_logodds_grid.astype(np.float32, copy=False)
                evidence = np.abs(logodds) >= 0.03
                if np.any(evidence):
                    prob = 1.0 / (1.0 + np.exp(-3.0 * np.clip(logodds, -4.0, 4.0)))
                    occ = np.clip(np.round(prob * 100.0), 0, 100).astype(np.int16)
                    grid[evidence] = occ[evidence]

        self._structural_grid_cache = grid
        self._structural_grid_cache_step = self.step_index
        self._structural_grid_cache_shape = (self.height, self.width)
        return grid

    def _modified_grid(self) -> np.ndarray:
        struct = self._structural_grid()
        grid = np.full((self.height, self.width), self.UNKNOWN, dtype=np.int8)
        known = struct >= 0
        if np.any(known):
            grid[known] = np.clip(np.round(struct[known]), 0, 100).astype(np.int8)
        return grid

    def world_to_map(self, x: float, y: float) -> tuple[int, int]:
        ix = int(math.floor((float(x) - self.origin_x) / self.resolution))
        iy = int(math.floor((float(y) - self.origin_y) / self.resolution))
        return ix, iy

    def map_to_world(self, ix: int, iy: int) -> tuple[float, float]:
        x = self.origin_x + (float(ix) + 0.5) * self.resolution
        y = self.origin_y + (float(iy) + 0.5) * self.resolution
        return x, y

    def in_bounds(self, ix: int, iy: int) -> bool:
        return 0 <= int(ix) < self.width and 0 <= int(iy) < self.height

    @staticmethod
    def bresenham(x0: int, y0: int, x1: int, y1: int) -> list[tuple[int, int]]:
        cells: list[tuple[int, int]] = []
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy
        x, y = x0, y0

        while True:
            cells.append((x, y))
            if x == x1 and y == y1:
                break

            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x += sx
            if e2 < dx:
                err += dx
                y += sy

            if len(cells) > 10000:
                break

        return cells


    def _slam_occupied_threshold(self) -> float:
        """Occupancy threshold used only for visibility/occlusion against SLAM /map."""
        return float(np.clip(getattr(self, "gap_occupied_threshold", 65.0), 0.0, 100.0))

    @staticmethod
    def _is_occupied_in_structural_grid(
        struct: np.ndarray,
        occupied_threshold: float,
        ix: int,
        iy: int,
    ) -> bool:
        """Fast occupied-cell test for callers that already own struct.

        This is behavior-equivalent to _is_slam_occupied_cell(), but avoids
        rebuilding _structural_grid() for every cell along every ray.
        """
        try:
            return bool(struct[int(iy), int(ix)] >= float(occupied_threshold))
        except Exception:
            return False

    def _is_slam_occupied_cell(self, ix: int, iy: int) -> bool:
        """
        True only when the SLAM /map-derived base_grid says this cell is occupied.

        This intentionally does not use confidence_grid or priority_grid. The user-facing
        rule is: confidence/priority-clear must not pass through walls, and the wall
        test comes from the original SLAM /map only.
        """
        if not self.in_bounds(ix, iy):
            return False
        struct = self._structural_grid()
        return self._is_occupied_in_structural_grid(
            struct,
            self._slam_occupied_threshold(),
            int(ix),
            int(iy),
        )

    def _inflated_wall_mask(
        self,
        radius: int = 2,
        struct: Optional[np.ndarray] = None,
        include_lidar: bool = True,
    ) -> np.ndarray:
        """Return an inflated wall barrier for update-time masking.

        SLAM occupied cells are stable geometry.  Recent LiDAR hit endpoints are
        useful as a *temporary* barrier for ray updates/priority clear, but they
        are too noisy to destructively erase old confidence.  `include_lidar`
        therefore stays opt-in for temporary visibility checks, while persistent
        map-memory cleanup uses SLAM-only walls.
        """
        if struct is None:
            cache_key = (int(radius), bool(include_lidar), self.step_index, self.height, self.width)
            cached = getattr(self, "_inflated_wall_cache", None)
            if cached is not None and getattr(self, "_inflated_wall_cache_key", None) == cache_key:
                return cached
        try:
            base = self._structural_grid() if struct is None else np.asarray(struct)
            occ = base >= self._slam_occupied_threshold()
            wall = self._dilate_bool(occ, radius=max(int(radius), 0))
            if bool(include_lidar):
                # Fuse recent LiDAR hit endpoints only for temporary visibility
                # masks.  Do not use this path for persistent confidence erasure,
                # otherwise one noisy scan can make the published confidence map
                # look as if it reset mid-episode.
                lidar_wall = getattr(self, "_last_lidar_hit_wall_mask", None)
                if isinstance(lidar_wall, np.ndarray) and lidar_wall.shape == wall.shape:
                    wall |= lidar_wall.astype(bool, copy=False)
            if struct is None:
                self._inflated_wall_cache = wall
                self._inflated_wall_cache_key = cache_key
            return wall
        except Exception:
            return np.zeros((self.height, self.width), dtype=bool)

    def _reachable_component_from_robot(self, robot_ix: int, robot_iy: int, wall_mask: Optional[np.ndarray] = None) -> np.ndarray:
        """4-connected traversable component containing the robot.

        This is intentionally strict.  It is used only as a safety clamp for the
        auxiliary confidence/priority layers so those layers cannot visually or
        numerically leak through a wall if SLAM closes a passage after earlier
        LiDAR updates.
        """
        # Cache keyed by (robot_ix, robot_iy, step_index) so the expensive BFS
        # is reused when called multiple times within the same step.
        cache_key = (int(robot_ix), int(robot_iy), int(getattr(self, "step_index", -1)))
        cached = getattr(self, "_reachable_cache", None)
        cached_key = getattr(self, "_reachable_cache_key", None)
        if cached is not None and cached_key == cache_key and isinstance(cached, np.ndarray):
            if cached.shape == (int(self.height), int(self.width)):
                return cached

        h = int(self.height)
        w = int(self.width)
        out = np.zeros((h, w), dtype=bool)
        if h <= 0 or w <= 0:
            return out
        rx = int(robot_ix)
        ry = int(robot_iy)
        if rx < 0 or rx >= w or ry < 0 or ry >= h:
            return out
        if wall_mask is None or not isinstance(wall_mask, np.ndarray) or wall_mask.shape != (h, w):
            wall_mask = self._inflated_wall_mask(radius=2)
        blocked = wall_mask.astype(bool, copy=False)
        if blocked[ry, rx]:
            # After a teleport/reset SLAM can mark the robot cell as occupied for
            # a few frames.  Open a tiny local hole around the robot, otherwise
            # the whole component becomes empty and all debug layers disappear.
            blocked = blocked.copy()
            y0 = max(0, ry - 1)
            y1 = min(h, ry + 2)
            x0 = max(0, rx - 1)
            x1 = min(w, rx + 2)
            blocked[y0:y1, x0:x1] = False
        from collections import deque
        q = deque([(rx, ry)])
        out[ry, rx] = True
        while q:
            x, y = q.popleft()
            for nx, ny in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
                if nx < 0 or nx >= w or ny < 0 or ny >= h:
                    continue
                if out[ny, nx] or blocked[ny, nx]:
                    continue
                out[ny, nx] = True
                q.append((nx, ny))
        self._reachable_cache = out
        self._reachable_cache_key = cache_key
        return out

    def _apply_strict_wall_visibility_clamp(self, robot_ix: int, robot_iy: int) -> np.ndarray:
        """Return a wall-separated component and erase only confirmed wall cells.

        Earlier patches destructively zeroed every confidence/priority cell outside
        the robot's *current* component.  That did stop wall leaks, but it also made
        RViz look as if /rl_confidence_map reset whenever a temporary LiDAR wall or
        slightly changed SLAM wall split the component.

        Current policy:
          - Use SLAM+LiDAR component only as a temporary mask for *new writes* and
            priority clear.
          - Persist old confidence memory across components.
          - Destructively erase only confirmed SLAM wall cells so confidence never
            remains inside occupied geometry.
        """
        fast_clamp_raw = os.environ.get("TB3_RL_FAST_STRICT_WALL_CLAMP", "1")
        fast_clamp = str(fast_clamp_raw).strip().lower() not in {"0", "false", "no", "off", "disable", "disabled"}
        if fast_clamp:
            # v18: ray updates are already clipped by direct LiDAR/SLAM line-of-sight.
            # The expensive full-map connected-component BFS is only needed as an
            # extra conservative guard.  In the fast path, keep the stable wall
            # cleanup but return a local write mask that excludes confirmed walls.
            stable_wall = self._inflated_wall_mask(radius=1, include_lidar=False)
            self._prev_wall_clamp_mask = stable_wall.copy()
            try:
                if bool(getattr(self, "clear_confidence_on_slam_occupied", False)):
                    occ_now = self._structural_grid() >= self._slam_occupied_threshold()
                    stable_occ = self._stable_occupied_mask_for_confidence(occ_now)
                    if stable_occ.shape == self.confidence_grid.shape and np.any(stable_occ):
                        self.confidence_grid[stable_occ] = 0.0
                if stable_wall.shape == self.priority_grid.shape:
                    self.priority_grid[stable_wall] = 0.0
                if hasattr(self, "_persistent_priority_seed_grid") and isinstance(self._persistent_priority_seed_grid, np.ndarray) and self._persistent_priority_seed_grid.shape == stable_wall.shape:
                    self._persistent_priority_seed_grid[stable_wall] = 0.0
                if hasattr(self, "_persistent_priority_allowed_grid") and isinstance(self._persistent_priority_allowed_grid, np.ndarray) and self._persistent_priority_allowed_grid.shape == stable_wall.shape:
                    self._persistent_priority_allowed_grid[stable_wall] = False
                if self.priority_checked_grid.shape == stable_wall.shape:
                    self.priority_checked_grid[stable_wall] = False
                if self.priority_suppression_grid.shape == stable_wall.shape:
                    self.priority_suppression_grid[stable_wall] = 0.0
            except Exception:
                pass
            if stable_wall.shape == self.priority_grid.shape:
                return (~stable_wall).astype(bool, copy=False)
            return np.ones_like(self.priority_grid, dtype=bool)

        temp_wall = self._inflated_wall_mask(radius=2, include_lidar=True)
        component = self._reachable_component_from_robot(robot_ix, robot_iy, temp_wall)
        if component.shape != self.priority_grid.shape:
            component = np.zeros_like(self.priority_grid, dtype=bool)

        # Stable, destructive cleanup: SLAM walls only.  Do not include transient
        # LiDAR endpoints and do not erase the whole outside-component area.
        stable_wall = self._inflated_wall_mask(radius=1, include_lidar=False)
        self._prev_wall_clamp_mask = stable_wall.copy()
        try:
            if bool(getattr(self, "clear_confidence_on_slam_occupied", False)):
                occ_now = self._structural_grid() >= self._slam_occupied_threshold()
                stable_occ = self._stable_occupied_mask_for_confidence(occ_now)
                if stable_occ.shape == self.confidence_grid.shape and np.any(stable_occ):
                    self.confidence_grid[stable_occ] = 0.0
            if stable_wall.shape == self.priority_grid.shape:
                self.priority_grid[stable_wall] = 0.0
            if hasattr(self, "_persistent_priority_seed_grid") and isinstance(self._persistent_priority_seed_grid, np.ndarray) and self._persistent_priority_seed_grid.shape == stable_wall.shape:
                self._persistent_priority_seed_grid[stable_wall] = 0.0
            if hasattr(self, "_persistent_priority_allowed_grid") and isinstance(self._persistent_priority_allowed_grid, np.ndarray) and self._persistent_priority_allowed_grid.shape == stable_wall.shape:
                self._persistent_priority_allowed_grid[stable_wall] = False
            if self.priority_checked_grid.shape == stable_wall.shape:
                self.priority_checked_grid[stable_wall] = False
            if self.priority_suppression_grid.shape == stable_wall.shape:
                self.priority_suppression_grid[stable_wall] = 0.0
        except Exception:
            pass
        return component

    def _truncate_ray_by_slam_occlusion(
        self,
        cells: list[tuple[int, int]],
        include_blocking_cell: bool = True,
        struct: Optional[np.ndarray] = None,
        occupied_threshold: Optional[float] = None,
        near_skip_cells: int = 0,
        raw_struct: Optional[np.ndarray] = None,
    ) -> tuple[list[tuple[int, int]], bool]:
        """
        Cut a ray at the first occupied SLAM /map cell.

        Returns:
          visible_cells: cells up to the first /map obstacle. If include_blocking_cell
                         is True, the blocking wall cell is included as the final cell.
          blocked:       True if a /map obstacle truncated the ray.

        This prevents both confidence increase and priority checked(-1) updates from
        leaking into rooms/empty space behind a wall.

        near_skip_cells / raw_struct:
          ``struct`` is normally the *inflated* SLAM wall mask (occupied cells are
          dilated by a few cells so a single-cell wall cannot be bypassed by a
          diagonal Bresenham ray).  When the robot stands within the inflation
          radius of a wall, the cell immediately after the ray origin becomes an
          *inflated* (synthetic) wall, which truncates every ray at idx==1 and
          stops confidence/priority from ever being painted in tight corridors,
          corners, or near reset noise.

          To fix this without letting rays leak through real walls, the first
          ``near_skip_cells`` cells after the ray origin are tested against the
          *raw* (non-inflated) occupancy ``raw_struct`` instead of the inflated
          ``struct``.  Real SLAM walls still block immediately; only the synthetic
          inflation halo around the robot is ignored at close range.
        """
        if not cells:
            return [], False

        if struct is None:
            struct = self._structural_grid()
        occ_thr = self._slam_occupied_threshold() if occupied_threshold is None else float(occupied_threshold)
        near_skip_cells = max(int(near_skip_cells), 0)
        use_near_skip = near_skip_cells > 0 and isinstance(raw_struct, np.ndarray) and raw_struct.shape == struct.shape

        visible: list[tuple[int, int]] = []
        width = int(self.width)
        height = int(self.height)
        for idx, (cx, cy) in enumerate(cells):
            cx_i = int(cx)
            cy_i = int(cy)
            if cx_i < 0 or cx_i >= width or cy_i < 0 or cy_i >= height:
                break

            # The robot's own cell can be occupied/noisy in SLAM after reset; do not
            # let it occlude the ray.
            if idx > 0:
                # Within the near zone, test against the raw (non-inflated) wall so
                # the robot's own inflation halo does not block close-range rays.
                if use_near_skip and idx <= near_skip_cells:
                    occluded = self._is_occupied_in_structural_grid(raw_struct, occ_thr, cx_i, cy_i)
                else:
                    occluded = self._is_occupied_in_structural_grid(struct, occ_thr, cx_i, cy_i)
                if occluded:
                    if include_blocking_cell:
                        visible.append((cx_i, cy_i))
                    return visible, True

            visible.append((cx_i, cy_i))

        return visible, False

    def _has_slam_line_of_sight(
        self,
        robot_ix: int,
        robot_iy: int,
        ix: int,
        iy: int,
        struct: Optional[np.ndarray] = None,
        occupied_threshold: Optional[float] = None,
    ) -> bool:
        """
        Segment visibility test using only SLAM /map occupied cells.

        The endpoint is also tested. Therefore a cell inside or behind a wall is not
        considered visible. Unknown/free cells do not block.
        """
        if not (self.in_bounds(robot_ix, robot_iy) and self.in_bounds(ix, iy)):
            return False

        cells = self.bresenham(int(robot_ix), int(robot_iy), int(ix), int(iy))
        if not cells:
            return False

        if struct is None:
            struct = self._structural_grid()
        occ_thr = self._slam_occupied_threshold() if occupied_threshold is None else float(occupied_threshold)

        for idx, (cx, cy) in enumerate(cells):
            if idx == 0:
                continue
            if self._is_occupied_in_structural_grid(struct, occ_thr, int(cx), int(cy)):
                return False
        return True

    def _slam_visibility_mask_from_robot(
        self,
        robot_ix: int,
        robot_iy: int,
        robot_yaw: float,
        max_range_m: float,
        fov_rad: float,
    ) -> np.ndarray:
        """
        Build a local visibility mask for priority clearing.

        Priority clear uses Gaussian blobs, so without this mask a blob near a doorway
        can spill through a wall and mark cells as checked(-1) even though the robot
        did not actually see them. This mask clips the Gaussian field by line-of-sight
        against the SLAM /map.
        """
        mask = np.zeros((self.height, self.width), dtype=np.float32)
        if not self.in_bounds(robot_ix, robot_iy):
            return mask

        max_cells = max(int(math.ceil(float(max_range_m) / max(self.resolution, 1e-6))), 1)
        x0 = max(0, int(robot_ix) - max_cells)
        x1 = min(self.width, int(robot_ix) + max_cells + 1)
        y0 = max(0, int(robot_iy) - max_cells)
        y1 = min(self.height, int(robot_iy) + max_cells + 1)
        if x1 <= x0 or y1 <= y0:
            return mask

        # Same candidate set as the old nested loop, but the range/FOV filtering
        # is vectorized and the structural occupancy grid is computed once.
        yy, xx = np.ogrid[y0:y1, x0:x1]
        dx_cells = xx.astype(np.float32) - float(robot_ix)
        dy_cells = yy.astype(np.float32) - float(robot_iy)
        dx_m = dx_cells * float(self.resolution)
        dy_m = dy_cells * float(self.resolution)
        dist2 = dx_m * dx_m + dy_m * dy_m
        max_range2 = float(max_range_m) * float(max_range_m)
        candidate = dist2 <= max_range2

        half_fov = float(fov_rad) * 0.5
        if half_fov < math.pi:
            rel = np.arctan2(
                np.sin(np.arctan2(dy_m, dx_m) - float(robot_yaw)),
                np.cos(np.arctan2(dy_m, dx_m) - float(robot_yaw)),
            )
            candidate &= np.abs(rel) <= half_fov

        cy, cx = np.nonzero(candidate)
        if cx.size == 0:
            return mask

        struct = self._structural_grid()
        occ_thr = self._slam_occupied_threshold()
        rix = int(robot_ix)
        riy = int(robot_iy)
        for ly, lx in zip(cy.tolist(), cx.tolist(), strict=False):
            x = int(x0 + lx)
            y = int(y0 + ly)
            if self._has_slam_line_of_sight(
                rix,
                riy,
                x,
                y,
                struct=struct,
                occupied_threshold=occ_thr,
            ):
                mask[y, x] = 1.0

        return mask

    def _paint_gaussian_blob(
        self,
        weight_grid: np.ndarray,
        ix: int,
        iy: int,
        sigma_m: float,
        max_radius_m: float,
        base_weight: float = 1.0,
    ):
        """
        Max-composite a Gaussian observation weight into weight_grid.

        This is deterministic, but it behaves like an observation probability field:
            w(d) = base_weight * exp(-d^2 / (2 sigma^2))
        Cells with weight >= priority_clear_min_weight become checked.
        """
        if weight_grid.shape != self.priority_grid.shape or not self.in_bounds(ix, iy):
            return

        sigma_cells = max(float(sigma_m) / max(self.resolution, 1e-6), 1e-6)
        radius_cells = max(int(math.ceil(float(max_radius_m) / max(self.resolution, 1e-6))), 1)
        y0 = max(0, int(iy) - radius_cells)
        y1 = min(self.height, int(iy) + radius_cells + 1)
        x0 = max(0, int(ix) - radius_cells)
        x1 = min(self.width, int(ix) + radius_cells + 1)

        yy, xx = np.ogrid[y0:y1, x0:x1]
        dist2 = (xx - int(ix)) ** 2 + (yy - int(iy)) ** 2
        local_weight = float(base_weight) * np.exp(-0.5 * dist2 / (sigma_cells * sigma_cells))
        local_weight = np.where(dist2 <= radius_cells * radius_cells, local_weight, 0.0)

        view = weight_grid[y0:y1, x0:x1]
        np.maximum(view, local_weight.astype(np.float32), out=view)

    def _priority_robot_reached_mask(self, ix: int, iy: int) -> np.ndarray:
        """Cells physically reached by the robot for priority clearing.

        Priority is now a *target to reach*, not a target to merely look at.
        This mask is centered on the robot body and clipped by the SLAM/LiDAR
        wall barrier through a 4-neighbor reachable component, so it does not
        clear priority through a wall.
        """
        mask = np.zeros_like(self.priority_grid, dtype=bool)
        if mask.size == 0 or not self.in_bounds(int(ix), int(iy)):
            return mask

        radius_cells = max(
            int(math.ceil(float(self.priority_clear_robot_radius_m) / max(float(self.resolution), 1e-6))),
            1,
        )
        x0 = max(0, int(ix) - radius_cells)
        x1 = min(int(self.width), int(ix) + radius_cells + 1)
        y0 = max(0, int(iy) - radius_cells)
        y1 = min(int(self.height), int(iy) + radius_cells + 1)

        yy, xx = np.ogrid[y0:y1, x0:x1]
        disk = ((xx - int(ix)) ** 2 + (yy - int(iy)) ** 2) <= int(radius_cells) ** 2
        if not np.any(disk):
            return mask

        try:
            struct = self._structural_grid()
            wall = self._inflated_wall_mask(radius=1, struct=struct)
            # Also treat latest LiDAR hit endpoints as temporary walls when the
            # mask exists; this prevents a radius clear from crossing a freshly
            # observed obstacle that SLAM has not fully integrated yet.
            hit_wall = getattr(self, "_last_lidar_hit_wall_mask", None)
            if isinstance(hit_wall, np.ndarray) and hit_wall.shape == wall.shape:
                wall = wall | self._dilate_bool(hit_wall.astype(bool, copy=False), radius=1)
            reachable = self._reachable_component_from_robot(int(ix), int(iy), wall)
            if reachable.shape == mask.shape:
                local = disk & reachable[y0:y1, x0:x1]
            else:
                local = disk
        except Exception:
            local = disk

        mask[y0:y1, x0:x1] = local
        return mask

    def _mark_priority_clear_visit(self, weight_grid: np.ndarray, ix: int, iy: int):
        """Mark robot-body reached cells for priority clearing.

        Priority is no longer cleared by merely looking at it with LiDAR.
        It is cleared only when the robot physically reaches the target
        neighborhood.  The neighborhood is wall-clipped, so it cannot erase a
        priority cluster behind a wall.
        """
        if weight_grid.shape != self.priority_grid.shape:
            return
        reached = self._priority_robot_reached_mask(int(ix), int(iy))
        if np.any(reached):
            weight_grid[reached] = 1.0

    def _priority_clear_angle_weight(self, rel_angle: float) -> float:
        angle = abs(normalize_angle(float(rel_angle)))
        return float(math.exp(-0.5 * (angle / self.priority_clear_angle_sigma_rad) ** 2))

    def _mark_priority_clear_ray(
        self,
        weight_grid: np.ndarray,
        robot_ix: int,
        robot_iy: int,
        cells: list[tuple[int, int]],
        angle_weight: float = 1.0,
    ):
        """
        Mark only directly visible free-space cells as checked candidates.

        Important semantics for /rl_priority_map:
          -1 must not mean "wall" or "LiDAR blocked here".
          It means "the robot has directly inspected this traversable cell".

        Therefore this function writes weights only to cells on the visible ray
        before the blocking endpoint.  No Gaussian side-spill is used here; a
        wide front sector is obtained by many LiDAR rays across the FOV, not by
        painting around obstacles.
        """
        if weight_grid.shape != self.priority_grid.shape or not cells:
            return

        max_cells = max(
            int(math.ceil(self.priority_clear_max_range_m / max(self.resolution, 1e-6))),
            1,
        )
        rix = int(robot_ix)
        riy = int(robot_iy)
        base_angle_weight = float(np.clip(angle_weight, 0.0, 1.0))
        if base_angle_weight <= 0.0:
            return

        max_cells2 = int(max_cells) * int(max_cells)
        max_range2 = max(float(self.priority_clear_max_range_m) * float(self.priority_clear_max_range_m), 1e-12)
        res2 = float(self.resolution) * float(self.resolution)
        for cx, cy in cells:
            cx_i = int(cx)
            cy_i = int(cy)
            if cx_i < 0 or cx_i >= int(self.width) or cy_i < 0 or cy_i >= int(self.height):
                continue

            dx = cx_i - rix
            dy = cy_i - riy
            d_cells2 = int(dx * dx + dy * dy)
            if d_cells2 > max_cells2:
                break

            d_m2 = float(d_cells2) * res2
            radial_weight = 0.65 + 0.35 * math.exp(-0.5 * d_m2 / max_range2)
            w = base_angle_weight * radial_weight
            if w > 1.0:
                w = 1.0
            elif w < 0.0:
                w = 0.0
            if w > weight_grid[cy_i, cx_i]:
                weight_grid[cy_i, cx_i] = np.float32(w)

    def _update_priority_checked(self, clear_weight: Optional[np.ndarray]) -> tuple[int, float]:
        """
        Clear active priority when the robot directly checks it.

        v116 semantics:
          - robot-body reach still clears priority;
          - front-FOV LiDAR/camera-style visibility also clears priority;
          - both paths are clipped by SLAM/LiDAR wall occlusion before reaching here.

        This matches the confidence-map semantics: a priority spot is considered
        checked when it lies in directly visible front free-space, not only when
        the robot physically drives onto it.
        """
        if clear_weight is None or clear_weight.shape != self.priority_grid.shape:
            return 0, 0.0
        if self.priority_checked_grid.shape != self.priority_grid.shape:
            self.priority_checked_grid = np.zeros_like(self.priority_grid, dtype=bool)

        clear_weight = np.clip(clear_weight.astype(np.float32, copy=False), 0.0, 1.0)
        previous_priority = np.clip(self.priority_grid.astype(np.float32, copy=False), 0.0, 100.0)

        # v116: clear by direct front-FOV check as well as physical reach.
        # `clear_weight` is produced only from:
        #   1) robot-body reached cells, and
        #   2) visible front-FOV ray free-space cells.
        # The ray component is already truncated by SLAM/LiDAR occlusion, so this
        # does not erase priority behind walls.
        visible_clear_mask = clear_weight >= self.priority_clear_min_weight

        # Final wall guard against stale/reshaped maps.
        try:
            struct = self._structural_grid()
            occ_thr = min(float(self.gap_occupied_threshold), 45.0)
            visible_clear_mask &= ~(struct >= occ_thr)
        except Exception:
            pass

        if not np.any(visible_clear_mask):
            self._priority_dirty = True
            return 0, 0.0

        active_before = previous_priority >= self.priority_clear_min_value
        check_mask = visible_clear_mask & active_before
        cleared_cells = int(np.count_nonzero(check_mask))
        clear_gain = float(np.sum((previous_priority[check_mask] / 100.0) * clear_weight[check_mask]))

        if np.any(check_mask):
            # Checked is internal only: it blocks regeneration, but RViz renders it as 0.
            clear_blob = self._dilate_bool(check_mask, radius=max(1, int(round(0.20 / max(self.resolution, 1e-6)))))
            self.priority_checked_grid[clear_blob] = True
            self.priority_grid[clear_blob] = 0.0
            if hasattr(self, "_persistent_priority_seed_grid") and isinstance(self._persistent_priority_seed_grid, np.ndarray) and self._persistent_priority_seed_grid.shape == self.priority_grid.shape:
                self._persistent_priority_seed_grid[clear_blob] = 0.0
                if hasattr(self, "_persistent_priority_allowed_grid") and isinstance(self._persistent_priority_allowed_grid, np.ndarray) and self._persistent_priority_allowed_grid.shape == self.priority_grid.shape:
                    self._persistent_priority_allowed_grid[clear_blob] = False

        # Visible free space may reduce priority slightly, but it must not erase
        # future doorway/frontier seeds.  This prevents /rl_priority_map from
        # degenerating into a huge -1 canvas.
        if self.priority_suppression_grid.shape == self.priority_grid.shape:
            self.priority_suppression_grid[visible_clear_mask] = np.maximum(
                self.priority_suppression_grid[visible_clear_mask],
                min(float(self.priority_visit_suppression_max), 0.35),
            )

        # If the currently locked target has just been checked, drop it immediately.
        if (
            np.any(check_mask)
            and self._locked_target_ix is not None
            and self._locked_target_iy is not None
            and self.in_bounds(int(self._locked_target_ix), int(self._locked_target_iy))
            and bool(self.priority_checked_grid[int(self._locked_target_iy), int(self._locked_target_ix)])
        ):
            self._reset_target_lock()
            self._clear_path_visualization()

        self._priority_dirty = True
        return cleared_cells, clear_gain

    def _update_priority_suppression(
        self,
        robot_ix: int,
        robot_iy: int,
        observed_mask: Optional[np.ndarray],
    ):
        """
        Persistently lower priority around already explored regions.

        - robot radius suppression: strong, because the robot physically reached
          this area and should not keep selecting the same door/gap target.
        - observed front-FOV suppression: weaker, because the robot has visually
          checked these cells but may not have passed through them yet.

        Values are stored as a 0..1 multiplicative suppression mask and applied
        during priority_grid recomputation.
        """
        if self.priority_suppression_grid.shape != self.priority_grid.shape:
            self.priority_suppression_grid = np.zeros_like(self.priority_grid, dtype=np.float32)

        max_supp = float(self.priority_visit_suppression_max)

        if (
            observed_mask is not None
            and observed_mask.shape == self.priority_suppression_grid.shape
            and self.priority_observed_suppression_gain > 0.0
        ):
            observed_gain = min(float(self.priority_observed_suppression_gain), max_supp)
            np.maximum(
                self.priority_suppression_grid,
                np.where(observed_mask, observed_gain, 0.0).astype(np.float32),
                out=self.priority_suppression_grid,
            )

        if not self.in_bounds(robot_ix, robot_iy) or self.priority_visit_suppression_gain <= 0.0:
            self._priority_dirty = True
            return

        radius_cells = max(
            int(math.ceil(self.priority_visit_suppression_radius_m / self.resolution)),
            1,
        )
        y0 = max(0, int(robot_iy) - radius_cells)
        y1 = min(self.height, int(robot_iy) + radius_cells + 1)
        x0 = max(0, int(robot_ix) - radius_cells)
        x1 = min(self.width, int(robot_ix) + radius_cells + 1)

        yy, xx = np.ogrid[y0:y1, x0:x1]
        dist = np.sqrt((xx - int(robot_ix)) ** 2 + (yy - int(robot_iy)) ** 2).astype(np.float32)
        mask = dist <= float(radius_cells)
        if not np.any(mask):
            self._priority_dirty = True
            return

        # Radial falloff: center gets full suppression, edge gets about 55%.
        falloff = 1.0 - 0.45 * np.clip(dist / max(float(radius_cells), 1.0), 0.0, 1.0)
        local = np.zeros((y1 - y0, x1 - x0), dtype=np.float32)
        local[mask] = np.clip(
            self.priority_visit_suppression_gain * falloff[mask],
            0.0,
            max_supp,
        )
        region = self.priority_suppression_grid[y0:y1, x0:x1]
        np.maximum(region, local, out=region)
        self._priority_dirty = True


    def _occupied_density_score_map(self, radius_m: Optional[float] = None) -> np.ndarray:
        """
        Return 0..1 wall-support score for each cell.

        A cell is well supported when there is occupied SLAM geometry in a
        local neighborhood. This prevents frontier/unconfirmed-free rewards from
        pulling the policy into featureless empty space. The implementation uses
        an integral image, so it is much cheaper than repeated dilation with a
        large radius.
        """
        r = float(radius_m) if radius_m is not None else float(self.wall_support_radius_m)
        cache_key = (round(r, 6), self.step_index, self.height, self.width)
        cached = getattr(self, "_density_score_cache", None)
        if cached is not None and getattr(self, "_density_score_cache_key", None) == cache_key:
            return cached

        struct = self._structural_grid()
        occ = (struct >= min(float(self.gap_occupied_threshold), 55.0)).astype(np.float32)
        if occ.size == 0 or not np.any(occ > 0.0):
            result = np.zeros((self.height, self.width), dtype=np.float32)
            self._density_score_cache = result
            self._density_score_cache_key = cache_key
            return result

        radius_cells = max(int(math.ceil(r / max(self.resolution, 1e-6))), 1)
        k = 2 * radius_cells + 1

        padded = np.pad(occ, ((radius_cells, radius_cells), (radius_cells, radius_cells)), mode="constant")
        integ = np.pad(padded, ((1, 0), (1, 0)), mode="constant").cumsum(axis=0).cumsum(axis=1)
        sums = integ[k:, k:] - integ[:-k, k:] - integ[k:, :-k] + integ[:-k, :-k]
        density = sums / float(k * k)
        result = np.clip(density / max(self.wall_support_density_threshold, 1e-6), 0.0, 1.0).astype(np.float32)
        self._density_score_cache = result
        self._density_score_cache_key = cache_key
        return result

    def _sample_robot_centric_scalar(
        self,
        grid: np.ndarray,
        robot_xy: np.ndarray,
        robot_yaw: float,
        output_size: int = 32,
        size_m: float = 3.2,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Sample a scalar global grid into a small robot-centric crop."""
        output_size = int(max(output_size, 4))
        size_m = float(max(size_m, self.resolution))
        half = size_m * 0.5
        step = size_m / float(output_size)

        coords = (np.arange(output_size, dtype=np.float32) + 0.5) * step
        local_forward = half - coords
        local_right = coords - half
        lf, lr = np.meshgrid(local_forward, local_right, indexing="ij")

        cos_yaw = math.cos(float(robot_yaw))
        sin_yaw = math.sin(float(robot_yaw))
        wx = float(robot_xy[0]) + lf * cos_yaw + lr * sin_yaw
        wy = float(robot_xy[1]) + lf * sin_yaw - lr * cos_yaw

        ix = np.floor((wx - self.origin_x) / max(self.resolution, 1e-6)).astype(np.int32)
        iy = np.floor((wy - self.origin_y) / max(self.resolution, 1e-6)).astype(np.int32)
        valid = (ix >= 0) & (ix < self.width) & (iy >= 0) & (iy < self.height)

        out = np.zeros((output_size, output_size), dtype=np.float32)
        if np.any(valid):
            out[valid] = grid[iy[valid], ix[valid]].astype(np.float32)
        return out, lf, lr

    def compute_obstacle_proximity_score(
        self,
        robot_xy: np.ndarray,
        warning_radius_m: float = 0.55,
        hard_radius_m: float = 0.20,
        occupied_threshold: float | None = None,
    ) -> tuple[float, float]:
        """
        SLAM /map 기반 장애물 근접도를 계산한다.

        반환:
          nearest_distance_m:
            robot 중심에서 가장 가까운 /map occupied cell까지의 거리.
            occupied cell이 warning_radius_m 안에 없으면 warning_radius_m 이상 값.

          proximity_score:
            [0, 1] 범위의 위험도.
              0: warning_radius_m 밖
              1: hard_radius_m 안쪽

        이 값은 reward에서만 사용하고 observation 차원에는 넣지 않는다.
        따라서 기존 SAC 모델과 observation/action space 호환성이 깨지지 않는다.
        """
        warning_radius_m = max(float(warning_radius_m), self.resolution)
        hard_radius_m = max(float(hard_radius_m), 0.0)
        if hard_radius_m >= warning_radius_m:
            hard_radius_m = max(0.0, warning_radius_m - self.resolution)

        threshold = self._slam_occupied_threshold() if occupied_threshold is None else float(occupied_threshold)
        occupied = self._structural_grid() >= threshold
        if not np.any(occupied):
            return float(warning_radius_m), 0.0

        robot_ix, robot_iy = self.world_to_map(float(robot_xy[0]), float(robot_xy[1]))
        if not self.in_bounds(robot_ix, robot_iy):
            return float(warning_radius_m), 0.0

        radius_cells = max(int(math.ceil(warning_radius_m / max(self.resolution, 1e-6))), 1)
        x0 = max(robot_ix - radius_cells, 0)
        x1 = min(robot_ix + radius_cells + 1, self.width)
        y0 = max(robot_iy - radius_cells, 0)
        y1 = min(robot_iy + radius_cells + 1, self.height)

        local_occ = occupied[y0:y1, x0:x1]
        if not np.any(local_occ):
            return float(warning_radius_m), 0.0

        yy, xx = np.nonzero(local_occ)
        cell_x = xx + x0
        cell_y = yy + y0

        # cell center 기준 거리. /map occupied 셀이 robot 주변 어느 방향에 있든 벌점화한다.
        wx = self.origin_x + (cell_x.astype(np.float32) + 0.5) * self.resolution
        wy = self.origin_y + (cell_y.astype(np.float32) + 0.5) * self.resolution
        dx = wx - float(robot_xy[0])
        dy = wy - float(robot_xy[1])
        dists = np.sqrt(dx * dx + dy * dy)
        nearest = float(np.min(dists))

        if nearest >= warning_radius_m:
            return nearest, 0.0

        denom = max(warning_radius_m - hard_radius_m, 1e-6)
        score = (warning_radius_m - nearest) / denom
        score = float(np.clip(score, 0.0, 1.0))
        return nearest, score

    def compute_forward_structure_scores(
        self,
        robot_xy: np.ndarray,
        robot_yaw: float,
    ) -> tuple[float, float]:
        """
        Estimate whether the current forward direction is supported by walls.

        wall_support_score ~= 1 when the area in front of the robot contains
        nearby occupied geometry or a priority door/gap candidate.
        open_space_score ~= 1 when the robot is facing featureless empty space.
        """
        support_map = self._occupied_density_score_map()
        support_crop, lf, lr = self._sample_robot_centric_scalar(
            support_map,
            robot_xy=robot_xy,
            robot_yaw=robot_yaw,
            output_size=32,
            size_m=max(self.open_space_front_distance_m * 2.0, self.open_space_side_width_m * 2.0, 2.5),
        )
        priority_crop, _, _ = self._sample_robot_centric_scalar(
            np.clip(self._active_priority_grid() / 100.0, 0.0, 1.0),
            robot_xy=robot_xy,
            robot_yaw=robot_yaw,
            output_size=32,
            size_m=max(self.open_space_front_distance_m * 2.0, self.open_space_side_width_m * 2.0, 2.5),
        )

        forward_mask = (
            (lf >= 0.20)
            & (lf <= self.open_space_front_distance_m)
            & (np.abs(lr) <= self.open_space_side_width_m * 0.5)
        )
        if not np.any(forward_mask):
            return 0.0, 0.0

        support_vals = support_crop[forward_mask]
        priority_vals = priority_crop[forward_mask]

        # Combine average context with the strongest local wall/gap cue. A single
        # nearby wall should matter, but pure isolated noise should not dominate.
        wall_support = float(
            np.clip(0.55 * float(np.mean(support_vals)) + 0.45 * float(np.max(support_vals)), 0.0, 1.0)
        )
        forward_priority = float(np.clip(np.max(priority_vals), 0.0, 1.0))
        structured = max(wall_support, forward_priority)
        open_space = float(np.clip(1.0 - structured, 0.0, 1.0))
        return float(structured), open_space

    def _nearest_occupied_steps_grid(self, occupied: np.ndarray, dx: int, dy: int, max_steps: int) -> np.ndarray:
        """Nearest occupied cell distance in a discrete direction. 0 means not found."""
        steps = np.zeros(occupied.shape, dtype=np.int16)
        for step in range(1, max(int(max_steps), 1) + 1):
            hit = self._shift_bool(occupied, int(dx) * step, int(dy) * step)
            set_mask = (steps == 0) & hit
            if np.any(set_mask):
                steps[set_mask] = step
        return steps

    def _corner_priority_score_map(self, base_occupied: np.ndarray, candidate: np.ndarray) -> np.ndarray:
        """
        Score L-shaped corner / corridor-turn cells.

        Door/gap priority uses opposite occupied supports.  Corners are different:
        the useful target cell is near two perpendicular wall supports and usually
        has an open diagonal continuation.  This helper therefore looks for
        occupied support in two orthogonal directions, then gates it by the absence
        of an immediate diagonal obstacle.
        """
        if base_occupied.shape != self.base_grid.shape or candidate.shape != self.base_grid.shape:
            return np.zeros((self.height, self.width), dtype=np.float32)
        if not np.any(base_occupied) or not np.any(candidate):
            return np.zeros((self.height, self.width), dtype=np.float32)

        max_steps = max(int(math.ceil(self.gap_check_radius_m / max(self.resolution, 1e-6))), 1)
        min_wall_m = max(0.05, self.resolution)
        max_wall_m = max(float(self.gap_check_radius_m), min_wall_m + self.resolution)
        sigma_m = max(0.36 * max_wall_m, 0.22, self.resolution)

        right = self._nearest_occupied_steps_grid(base_occupied, 1, 0, max_steps).astype(np.float32) * self.resolution
        left = self._nearest_occupied_steps_grid(base_occupied, -1, 0, max_steps).astype(np.float32) * self.resolution
        up = self._nearest_occupied_steps_grid(base_occupied, 0, 1, max_steps).astype(np.float32) * self.resolution
        down = self._nearest_occupied_steps_grid(base_occupied, 0, -1, max_steps).astype(np.float32) * self.resolution

        def support_score(dist: np.ndarray) -> np.ndarray:
            valid = (dist >= min_wall_m) & (dist <= max_wall_m)
            # Stronger when the wall is close enough to define a corner, but not
            # exactly on the candidate cell.
            score = np.exp(-0.5 * (dist / sigma_m) ** 2).astype(np.float32)
            return np.where(valid, score, 0.0).astype(np.float32)

        sr = support_score(right)
        sl = support_score(left)
        su = support_score(up)
        sd = support_score(down)

        # For each L orientation, require two perpendicular supports.  Also avoid
        # cells where the diagonal continuation is immediately occupied, because
        # those are usually inside/behind a wall rather than an explorable corner.
        diag_ru_open = ~self._shift_bool(base_occupied, 1, 1)
        diag_rd_open = ~self._shift_bool(base_occupied, 1, -1)
        diag_lu_open = ~self._shift_bool(base_occupied, -1, 1)
        diag_ld_open = ~self._shift_bool(base_occupied, -1, -1)

        ru = np.sqrt(sr * su).astype(np.float32) * diag_ru_open.astype(np.float32)
        rd = np.sqrt(sr * sd).astype(np.float32) * diag_rd_open.astype(np.float32)
        lu = np.sqrt(sl * su).astype(np.float32) * diag_lu_open.astype(np.float32)
        ld = np.sqrt(sl * sd).astype(np.float32) * diag_ld_open.astype(np.float32)

        corner = np.maximum.reduce([ru, rd, lu, ld]).astype(np.float32)
        corner *= candidate.astype(np.float32)

        # Give the corner a small spatial footprint so RViz shows a usable region
        # instead of a single unstable pixel.
        if np.any(corner > 0.0):
            corner = self._max_filter_float(corner, radius=2, decay=0.82)
            corner *= candidate.astype(np.float32)
        return np.clip(corner, 0.0, 1.0).astype(np.float32)


    def _doorway_frontier_priority_score_map(
        self,
        base_occupied: np.ndarray,
        base_free: np.ndarray,
        base_unknown: np.ndarray,
        candidate: np.ndarray,
        unknown_near: np.ndarray,
        free_near: np.ndarray,
    ) -> np.ndarray:
        """
        Soft doorway / entrance seed detector.

        The strict two-sided gap detector is precise but often too brittle on an
        early SLAM map: doorway posts may be sparse, one wall may be missing, or
        the opening may be represented as a free/unknown mouth rather than a clean
        gap between two occupied supports.  This detector creates lower-amplitude
        seeds where traversable cells sit on a free/unknown boundary and have
        nearby occupied structural support.
        """
        if base_occupied.shape != candidate.shape:
            return np.zeros((self.height, self.width), dtype=np.float32)
        if not np.any(candidate):
            return np.zeros((self.height, self.width), dtype=np.float32)

        support = self._occupied_density_score_map(radius_m=max(self.wall_support_radius_m, 0.65))
        support = np.clip(support, 0.0, 1.0).astype(np.float32)
        if not np.any(support > 0.0):
            return np.zeros((self.height, self.width), dtype=np.float32)

        traversable = candidate & (base_free | base_unknown)
        boundary = traversable & (unknown_near | self._dilate_bool(base_unknown, radius=3)) & free_near
        wall_context = support >= 0.10
        seed_mask = boundary & wall_context
        if not np.any(seed_mask):
            return np.zeros((self.height, self.width), dtype=np.float32)

        low_conf = self.confidence_grid < max(self.low_confidence_threshold, self.min_known_confidence)
        low_conf_near = self._dilate_bool(low_conf, radius=3)
        exploration = np.where(base_unknown | unknown_near | low_conf_near, 1.0, 0.55).astype(np.float32)

        # Soft doorway/frontier values should still be visible in RViz.  Keep them
        # below true two-sided gaps but not in the almost-black 1..10 range.
        score = np.where(seed_mask, 0.32 + 0.48 * support * exploration, 0.0).astype(np.float32)
        score[base_occupied] = 0.0
        return np.clip(score, 0.0, 0.80).astype(np.float32)

    def _wall_mouth_priority_score_map(
        self,
        base_occupied: np.ndarray,
        base_free: np.ndarray,
        base_unknown: np.ndarray,
        candidate: np.ndarray,
        unknown_near: np.ndarray,
        free_near: np.ndarray,
    ) -> np.ndarray:
        """
        Entrance/frontier-mouth priority detector.

        The strict two-sided detector gives strong priority between obstacle
        supports. Real SLAM entrances often appear as a free/unknown mouth next
        to one or two wall fragments instead of a clean pair of posts. This
        restores that older useful behavior: cells at a free/unknown boundary
        with nearby wall support become visible purple priority; the same
        wall-constrained Gaussian spread then makes the blob broad without
        crossing walls.
        """
        out = np.zeros((self.height, self.width), dtype=np.float32)
        if candidate.shape != out.shape or base_occupied.shape != out.shape:
            return out
        if not np.any(candidate) or not np.any(base_occupied):
            return out

        support = self._occupied_density_score_map(radius_m=max(self.wall_support_radius_m, 0.75))
        support = np.clip(support.astype(np.float32, copy=False), 0.0, 1.0)

        unknown_mouth = unknown_near | self._dilate_bool(base_unknown, radius=5)
        free_context = free_near | self._dilate_bool(base_free, radius=2)
        low_conf = self.confidence_grid < max(self.low_confidence_threshold, self.min_known_confidence)
        low_conf_near = self._dilate_bool(low_conf, radius=3)

        wall_context = support >= max(0.055, 0.55 * float(self.wall_support_density_threshold))
        mouth = candidate & (~base_occupied) & (base_free | base_unknown | low_conf_near) & free_context & wall_context & (unknown_mouth | low_conf_near)
        if not np.any(mouth):
            return out

        occ_near_1 = self._dilate_bool(base_occupied, radius=1)
        occ_near_3 = self._dilate_bool(base_occupied, radius=3)
        corner_like = occ_near_3 & (~occ_near_1)
        mouth &= (corner_like | unknown_mouth | low_conf_near)
        if not np.any(mouth):
            return out

        score = (0.44 + 0.50 * support).astype(np.float32)
        score = np.where(mouth, score, 0.0).astype(np.float32)
        score[base_occupied] = 0.0
        return np.clip(score, 0.0, 0.94).astype(np.float32)

    def _random_priority_spot_seed_map(
        self,
        candidate: np.ndarray,
        base_occupied: np.ndarray,
        base_free: np.ndarray,
        base_unknown: np.ndarray,
        unknown_near: np.ndarray,
        free_near: np.ndarray,
    ) -> np.ndarray:
        """
        Add a few weak stochastic exploration-prior spots.

        These are deliberately low amplitude and sparse.  They make the priority
        layer non-empty in large partially-known rooms but should not dominate
        real door/gap seeds or change reward.py itself.
        """
        out = np.zeros((self.height, self.width), dtype=np.float32)
        if candidate.shape != out.shape or not np.any(candidate):
            return out

        # Refresh slowly to avoid frame-to-frame flicker and repeated priority_gain.
        epoch = int(self.step_index) // 80
        if getattr(self, "_random_priority_epoch", None) == epoch:
            cached = getattr(self, "_random_priority_seed_cache", None)
            if isinstance(cached, np.ndarray) and cached.shape == out.shape:
                return cached.copy()

        low_conf = self.confidence_grid < max(self.low_confidence_threshold, self.min_known_confidence)
        spot_candidates = candidate & (~base_occupied) & (base_free | base_unknown | low_conf) & (unknown_near | free_near | low_conf)
        if self.priority_checked_grid.shape == spot_candidates.shape:
            spot_candidates &= ~self.priority_checked_grid
        if self.priority_suppression_grid.shape == spot_candidates.shape:
            spot_candidates &= self.priority_suppression_grid < 0.65

        ys, xs = np.nonzero(spot_candidates)
        n_available = int(xs.size)
        if n_available <= 0:
            self._random_priority_epoch = epoch
            self._random_priority_seed_cache = out.copy()
            return out

        # Sparse by area: weak exploration prior only.  Keep it visible, but do
        # not flood the map; real door/gap priority remains stronger.
        n_spots = max(1, min(6, n_available // 5000 + 1))
        seed = (
            int(epoch) * 73856093
            ^ int(self.width) * 19349663
            ^ int(self.height) * 83492791
            ^ int(round(float(self.origin_x) * 100.0)) * 2654435761
            ^ int(round(float(self.origin_y) * 100.0)) * 97531
        ) & 0xFFFFFFFF
        rng = np.random.default_rng(seed)

        order = rng.permutation(n_available)
        chosen: list[tuple[int, int]] = []
        min_spacing_cells = max(int(round(0.90 / max(self.resolution, 1e-6))), 1)
        min_spacing2 = min_spacing_cells * min_spacing_cells
        for idx in order.tolist():
            x = int(xs[idx])
            y = int(ys[idx])
            ok = True
            for px, py in chosen:
                if (x - px) * (x - px) + (y - py) * (y - py) < min_spacing2:
                    ok = False
                    break
            if not ok:
                continue
            chosen.append((x, y))
            if len(chosen) >= n_spots:
                break

        for x, y in chosen:
            out[y, x] = float(rng.uniform(0.20, 0.35))

        self._random_priority_epoch = epoch
        self._random_priority_seed_cache = out.copy()
        return out

    def _two_wall_entrance_priority_seed(self, base_occupied: np.ndarray, candidate: np.ndarray) -> np.ndarray:
        """High priority in cells geometrically between two nearby wall supports.

        This restores the old useful behavior for door/entrance-like gaps, but it
        is intentionally clipped later by LiDAR/map visibility and 4-neighbor wall
        constraints.  It never paints directly on occupied/inflated wall cells.
        """
        out = np.zeros((self.height, self.width), dtype=np.float32)
        if base_occupied.shape != out.shape or candidate.shape != out.shape:
            return out
        if not np.any(base_occupied) or not np.any(candidate):
            return out

        max_steps = max(int(math.ceil(1.45 / max(self.resolution, 1e-6))), 2)
        min_gap_m = 0.22
        max_gap_m = 1.35

        right = self._nearest_occupied_steps_grid(base_occupied, 1, 0, max_steps).astype(np.float32)
        left = self._nearest_occupied_steps_grid(base_occupied, -1, 0, max_steps).astype(np.float32)
        up = self._nearest_occupied_steps_grid(base_occupied, 0, 1, max_steps).astype(np.float32)
        down = self._nearest_occupied_steps_grid(base_occupied, 0, -1, max_steps).astype(np.float32)

        # Diagonal/skew supports catch doorway posts that are not axis-aligned.
        ru = self._nearest_occupied_steps_grid(base_occupied, 1, 1, max_steps).astype(np.float32)
        ld = self._nearest_occupied_steps_grid(base_occupied, -1, -1, max_steps).astype(np.float32)
        rd = self._nearest_occupied_steps_grid(base_occupied, 1, -1, max_steps).astype(np.float32)
        lu = self._nearest_occupied_steps_grid(base_occupied, -1, 1, max_steps).astype(np.float32)

        def pair_score(a: np.ndarray, b: np.ndarray, diag: bool = False) -> np.ndarray:
            scale = math.sqrt(2.0) if diag else 1.0
            da = a * float(self.resolution) * scale
            db = b * float(self.resolution) * scale
            gap = da + db
            ok = (a > 0) & (b > 0) & (gap >= min_gap_m) & (gap <= max_gap_m)
            center_balance = 1.0 - np.clip(np.abs(da - db) / np.maximum(gap, 1e-6), 0.0, 1.0)
            # Prefer roughly doorway-sized openings.  Still allow broader mouths.
            target_gap = 0.72
            width_score = np.exp(-0.5 * ((gap - target_gap) / 0.38) ** 2).astype(np.float32)
            score = (0.45 + 0.55 * center_balance.astype(np.float32)) * width_score
            return np.where(ok, score, 0.0).astype(np.float32)

        score = np.maximum.reduce([
            pair_score(left, right, False),
            pair_score(up, down, False),
            pair_score(ru, ld, True),
            pair_score(rd, lu, True),
        ]).astype(np.float32)

        # Need at least a little unknown/low-confidence context so an already
        # fully explored empty corridor does not dominate forever.
        unknown = self.base_grid < 0
        low_conf = self.confidence_grid < max(self.low_confidence_threshold, self.min_known_confidence)
        frontier_context = self._dilate_bool(unknown | low_conf, radius=4)
        score *= candidate.astype(np.float32)
        score *= np.where(frontier_context, 1.0, 0.40).astype(np.float32)
        if np.any(score > 0.0):
            score = self._max_filter_float(score, radius=1, decay=0.88)
            score *= candidate.astype(np.float32)
        return np.clip(score, 0.0, 1.0).astype(np.float32)

    def _lidar_wall_pair_priority_seed(
        self,
        base_occupied: np.ndarray,
        candidate: np.ndarray,
        struct: Optional[np.ndarray] = None,
        occupied_threshold: Optional[float] = None,
    ) -> np.ndarray:
        """Seed priority by connecting pairs of LiDAR obstacle-hit cells.

        The useful entrance/corridor signal here is not a single obstacle pixel;
        it is the free segment between two LiDAR hit points.  We therefore find
        nearby pairs of hit endpoints, draw the segment between them, and place a
        high Gaussian seed on the traversable part of that segment.  The segment
        is later spread only through wall-constrained 4-neighbor free space.
        """
        out = np.zeros((self.height, self.width), dtype=np.float32)
        if base_occupied.shape != out.shape or candidate.shape != out.shape:
            return out
        hits = getattr(self, "_last_lidar_hit_cells_for_priority", [])
        if not hits:
            return out
        if struct is None:
            struct = self._structural_grid()
        occ_thr = self._slam_occupied_threshold() if occupied_threshold is None else float(occupied_threshold)
        # Deduplicate while keeping scan order.
        uniq = []
        seen = set()
        for item in hits:
            try:
                x, y, idx, rr = int(item[0]), int(item[1]), int(item[2]), float(item[3])
            except Exception:
                continue
            if not self.in_bounds(x, y):
                continue
            key = (x, y)
            if key in seen:
                continue
            seen.add(key)
            uniq.append((x, y, idx, rr))
        if len(uniq) < 2:
            return out

        pairs: list[tuple[float, int, int, list[tuple[int, int]]]] = []
        max_pairs = 80
        min_gap_m = max(0.28, float(getattr(self, "gap_min_width_m", 0.25)))
        max_gap_m = min(1.65, max(0.75, float(getattr(self, "gap_max_width_m", 1.25)) + 0.35))
        # Limit pair search to nearby scan hits, which represent adjacent doorway/wall returns.
        n = len(uniq)
        for a in range(n):
            x1, y1, i1, r1 = uniq[a]
            for b in range(a + 1, min(n, a + 35)):
                x2, y2, i2, r2 = uniq[b]
                if abs(int(i2) - int(i1)) < 3:
                    continue
                dx = (x2 - x1) * self.resolution
                dy = (y2 - y1) * self.resolution
                gap = math.hypot(dx, dy)
                if gap < min_gap_m or gap > max_gap_m:
                    continue
                mx = int(round((x1 + x2) * 0.5))
                my = int(round((y1 + y2) * 0.5))
                if not self.in_bounds(mx, my) or not candidate[my, mx]:
                    continue
                # Robot must have line-of-sight to the midpoint; otherwise the segment is behind a wall.
                rix = int(getattr(self, "_last_robot_ix", self.width // 2))
                riy = int(getattr(self, "_last_robot_iy", self.height // 2))
                if not self._has_slam_line_of_sight(rix, riy, mx, my, struct=struct, occupied_threshold=occ_thr):
                    continue
                line = self.bresenham(x1, y1, x2, y2)
                if len(line) < 3:
                    continue
                passable_line = []
                blocked = False
                for lx, ly in line[1:-1]:
                    if not self.in_bounds(lx, ly):
                        blocked = True
                        break
                    if base_occupied[ly, lx]:
                        blocked = True
                        break
                    if candidate[ly, lx]:
                        passable_line.append((lx, ly))
                if blocked or len(passable_line) == 0:
                    continue
                # Prefer gap sizes around a TurtleBot-scale doorway / passage.
                width_score = math.exp(-0.5 * ((gap - 0.72) / 0.33) ** 2)
                range_score = math.exp(-0.5 * (((r1 + r2) * 0.5 - 1.45) / 0.85) ** 2)
                score = float(0.35 + 0.65 * width_score) * float(0.55 + 0.45 * range_score)
                pairs.append((score, mx, my, passable_line))
        if not pairs:
            return out
        pairs.sort(key=lambda t: -t[0])
        for score, mx, my, line_cells in pairs[:max_pairs]:
            # Put the strongest seed at the midpoint and a weaker ridge along the segment.
            if self.in_bounds(mx, my) and candidate[my, mx]:
                out[my, mx] = max(float(out[my, mx]), min(1.0, 0.75 + 0.25 * float(score)))
            L = max(len(line_cells) - 1, 1)
            for k, (lx, ly) in enumerate(line_cells):
                if not candidate[ly, lx]:
                    continue
                center_w = math.exp(-0.5 * (((k / L) - 0.5) / 0.28) ** 2)
                out[ly, lx] = max(float(out[ly, lx]), float(score) * (0.40 + 0.45 * center_w))
        return np.clip(out, 0.0, 1.0).astype(np.float32)

    def _recompute_priority_grid(self):
        """Simple persistent random priority clusters only.

        User-facing policy for /rl_priority_map:
          0      = no active priority
          1..100 = active persistent random target priority

        The checked mask is internal only. Checked cells are not rendered as -1;
        they simply suppress future priority regeneration and stay zero in RViz.

        Priority generation/clearing is constrained by latest LiDAR-visible free
        rays, SLAM/LiDAR inflated wall barriers, and the robot 4-connected free
        component. Door/entrance heuristics are intentionally disabled here.
        """
        if self.priority_grid.shape != self.base_grid.shape:
            self.priority_grid = np.zeros_like(self.base_grid, dtype=np.float32)
        if self.priority_checked_grid.shape != self.priority_grid.shape:
            self.priority_checked_grid = np.zeros_like(self.priority_grid, dtype=bool)
        if not hasattr(self, "_persistent_priority_seed_grid") or not isinstance(self._persistent_priority_seed_grid, np.ndarray) or self._persistent_priority_seed_grid.shape != self.priority_grid.shape:
            self._persistent_priority_seed_grid = np.zeros_like(self.priority_grid, dtype=np.float32)
        if not hasattr(self, "_persistent_priority_allowed_grid") or not isinstance(self._persistent_priority_allowed_grid, np.ndarray) or self._persistent_priority_allowed_grid.shape != self.priority_grid.shape:
            self._persistent_priority_allowed_grid = np.zeros_like(self.priority_grid, dtype=bool)

        struct = self._structural_grid()
        occ_thr = self._slam_occupied_threshold()
        base_occupied = np.asarray(struct >= occ_thr, dtype=bool)
        wall_inflated = self._inflated_wall_mask(radius=2, struct=struct)

        rix = int(getattr(self, "_last_robot_ix", self.width // 2))
        riy = int(getattr(self, "_last_robot_iy", self.height // 2))
        reachable = self._reachable_component_from_robot(rix, riy, wall_inflated)
        if reachable.shape != self.priority_grid.shape:
            reachable = np.zeros_like(self.priority_grid, dtype=bool)

        lidar_visible = getattr(self, "_last_lidar_visible_free_mask", None)
        if not isinstance(lidar_visible, np.ndarray) or lidar_visible.shape != self.priority_grid.shape:
            lidar_visible = np.zeros_like(self.priority_grid, dtype=bool)
        else:
            lidar_visible = lidar_visible.astype(bool, copy=False)

        base_free = np.asarray((self.base_grid >= 0) & (self.base_grid < occ_thr), dtype=bool)
        base_unknown = np.asarray(self.base_grid < 0, dtype=bool)
        low_conf = np.asarray(self.confidence_grid < max(self.low_confidence_threshold, self.min_known_confidence), dtype=bool)
        passable = (base_free | base_unknown | low_conf) & reachable & (~base_occupied) & (~wall_inflated)
        if self.priority_checked_grid.shape == passable.shape:
            passable &= ~self.priority_checked_grid

        seed_grid = self._persistent_priority_seed_grid.astype(np.float32, copy=True)
        allowed_grid = self._persistent_priority_allowed_grid.astype(bool, copy=True)

        invalid = (~passable) | base_occupied | wall_inflated | (~reachable)
        seed_grid[invalid] = 0.0
        allowed_grid[invalid] = False

        # v24: keep the stable v5 priority mechanism, but slow down new cluster
        # generation by default.  This is intentionally env-configurable so the
        # algorithm can be tuned without changing the learned observation/reward
        # semantics.
        def _env_int(name: str, default: int) -> int:
            try:
                return int(os.environ.get(name, str(default)))
            except Exception:
                return int(default)

        def _env_float(name: str, default: float) -> float:
            try:
                return float(os.environ.get(name, str(default)))
            except Exception:
                return float(default)

        def _env_bool(name: str, default: bool = False) -> bool:
            raw = os.environ.get(name, "1" if default else "0")
            return str(raw).strip().lower() in {"1", "true", "yes", "on", "enable", "enabled"}

        # `step_index` advances once per RL env update, not once per 0.01s live-map
        # timer tick.  With the normal control_dt=0.10, 100 steps ~= 10 seconds.
        spawn_interval_steps = max(_env_int("TB3_RL_PRIORITY_CLUSTER_SPAWN_INTERVAL_STEPS", 100), 1)
        max_seed_points = max(_env_int("TB3_RL_PRIORITY_MAX_SEED_POINTS", 6), 0)
        spawn_min_range_m = max(_env_float("TB3_RL_PRIORITY_SPAWN_MIN_RANGE_M", 0.90), 0.0)
        spawn_max_range_m = max(_env_float("TB3_RL_PRIORITY_SPAWN_MAX_RANGE_M", 2.40), spawn_min_range_m)

        active_seed_points = int(np.count_nonzero(seed_grid > 1e-4))
        due = int(self.step_index) - int(getattr(self, "_last_priority_cluster_spawn_step", -10_000_000)) >= spawn_interval_steps

        if due and active_seed_points < max_seed_points:
            yy, xx = np.ogrid[:self.height, :self.width]
            dist_m = np.sqrt((xx.astype(np.float32) - float(rix)) ** 2 + (yy.astype(np.float32) - float(riy)) ** 2) * float(self.resolution)

            # v117: do not spawn a new random emergency priority inside the same
            # front-FOV clear mask.  v116 used `passable & lidar_visible`, but that
            # makes a freshly spawned spot immediately eligible for the confidence-like
            # priority clear rule on the next update.  The default now places random
            # spots in reachable, not-currently-visible cells.  They are still clipped
            # by SLAM walls/reachability and are cleared only after the robot turns
            # toward them and confirms them with the front FOV.
            spawn_in_current_fov = _env_bool("TB3_RL_PRIORITY_SPAWN_IN_CURRENT_FOV", False)
            spawn_visibility_mask = lidar_visible if spawn_in_current_fov else (~lidar_visible)
            spawn_mask = passable & spawn_visibility_mask & (dist_m >= spawn_min_range_m) & (dist_m <= spawn_max_range_m)

            if np.any(seed_grid > 1e-4):
                near_existing = self._dilate_bool(seed_grid > 1e-4, radius=max(3, int(round(0.55 / max(self.resolution, 1e-6)))))
                spawn_mask &= ~near_existing
            if np.any(self.priority_grid > 0.5):
                near_active = self._dilate_bool(self.priority_grid > 0.5, radius=max(2, int(round(0.35 / max(self.resolution, 1e-6)))))
                spawn_mask &= ~near_active

            ys, xs = np.nonzero(spawn_mask)
            if xs.size > 0:
                seed = (
                    int(self.step_index) * 73856093
                    ^ int(self.width) * 19349663
                    ^ int(self.height) * 83492791
                    ^ int(round(float(self.origin_x) * 100.0)) * 2654435761
                    ^ int(round(float(self.origin_y) * 100.0)) * 97531
                ) & 0xFFFFFFFF
                rng = np.random.default_rng(seed)
                pick = int(rng.integers(0, int(xs.size)))
                cx = int(xs[pick])
                cy = int(ys[pick])

                allow_radius = max(3, int(round(0.55 / max(self.resolution, 1e-6))))
                cluster_allowed = passable & spawn_visibility_mask & (
                    ((xx.astype(np.float32) - float(cx)) ** 2 + (yy.astype(np.float32) - float(cy)) ** 2)
                    <= float(allow_radius * allow_radius)
                )
                # Keep only cells connected to the selected center within the local
                # LiDAR-visible cluster. This prevents diagonal wall/corner leaks.
                if self.in_bounds(cx, cy) and bool(cluster_allowed[cy, cx]):
                    local_comp = self._reachable_component_from_robot(cx, cy, ~cluster_allowed)
                    if local_comp.shape == cluster_allowed.shape:
                        cluster_allowed &= local_comp

                if int(np.count_nonzero(cluster_allowed)) >= 3:
                    allowed_grid[cluster_allowed] = True
                    seed_grid[cy, cx] = max(float(seed_grid[cy, cx]), float(rng.uniform(0.78, 1.00)))
                    made = 1
                    attempts = 0
                    seed_radius = max(1, int(round(0.24 / max(self.resolution, 1e-6))))
                    while made < 3 and attempts < 60:
                        attempts += 1
                        px = cx + int(rng.integers(-seed_radius, seed_radius + 1))
                        py = cy + int(rng.integers(-seed_radius, seed_radius + 1))
                        if px < 0 or px >= self.width or py < 0 or py >= self.height:
                            continue
                        if not bool(cluster_allowed[py, px]):
                            continue
                        seed_grid[py, px] = max(float(seed_grid[py, px]), float(rng.uniform(0.60, 0.92)))
                        made += 1
                    self._last_priority_cluster_spawn_step = int(self.step_index)

        # Existing clusters persist in their stored allowed mask, but remain clipped
        # by current wall geometry, reachable component, and internal checked mask.
        allowed_grid &= passable
        seed_grid[~allowed_grid] = 0.0
        self._persistent_priority_seed_grid = seed_grid.astype(np.float32, copy=False)
        self._persistent_priority_allowed_grid = allowed_grid.astype(bool, copy=False)

        if not np.any(seed_grid > 1e-4):
            self.priority_grid.fill(0.0)
            self._reset_target_lock()
            self._clear_path_visualization()
            return

        random_prio = self._wall_constrained_gaussian_priority_spread(
            seed_grid=seed_grid,
            candidate_mask=passable & allowed_grid,
            occupied_mask=base_occupied | wall_inflated,
        )
        random_prio[~reachable] = 0.0
        random_prio[base_occupied | wall_inflated] = 0.0
        if self.priority_checked_grid.shape == random_prio.shape:
            random_prio[self.priority_checked_grid] = 0.0
        target_priority = np.clip(random_prio * 100.0, 0.0, 100.0).astype(np.float32)

        # v25.7: slow the birth/growth of newly generated priority regions.
        # Recompute interval controls how often target_priority is recalculated;
        # birth_delta controls how fast a new high-value area becomes visible to
        # RViz/CNN. Decreases/clears are immediate.
        if self.priority_grid.shape == target_priority.shape:
            previous_priority = np.clip(self.priority_grid.astype(np.float32, copy=False), 0.0, 100.0)
            birth_delta = float(
                np.clip(getattr(self, "priority_birth_max_delta_per_recompute", 6.0), 0.05, 100.0)
            )
            growing = target_priority > previous_priority
            grown_priority = np.minimum(target_priority, previous_priority + birth_delta)
            new_priority = np.where(growing, grown_priority, target_priority).astype(np.float32)

            new_priority[~reachable] = 0.0
            new_priority[base_occupied | wall_inflated] = 0.0
            if self.priority_checked_grid.shape == new_priority.shape:
                new_priority[self.priority_checked_grid] = 0.0

            if os.environ.get("TB3_RL_QUIET_PRIORITY_LOGS", "0").strip().lower() not in {"1", "true", "yes", "on"} and (int(self.step_index) - int(getattr(self, "_last_priority_birth_debug_step", -10_000_000))) >= max(int(self.priority_recompute_interval), 200):
                try:
                    target_active = int(np.count_nonzero(target_priority >= max(self.priority_clear_min_value, 1.0)))
                    new_active = int(np.count_nonzero(new_priority >= max(self.priority_clear_min_value, 1.0)))
                    self.node.get_logger().info(
                        "PRIORITY_BIRTH_THROTTLE | step=%d delta=%.2f target_active=%d active=%d"
                        % (int(self.step_index), birth_delta, target_active, new_active)
                    )
                    self._last_priority_birth_debug_step = int(self.step_index)
                except Exception:
                    pass

            self.priority_grid = np.clip(new_priority, 0.0, 100.0).astype(np.float32)
        else:
            self.priority_grid = target_priority

        # Invalidate the per-step active-priority cache so the next call to
        # _active_priority_grid() re-reads the freshly mutated priority_grid.
        self._active_priority_cache = None
        self._active_priority_cache_step = -1

    def _wall_constrained_gaussian_priority_spread(
        self,
        seed_grid: np.ndarray,
        candidate_mask: np.ndarray,
        occupied_mask: np.ndarray,
    ) -> np.ndarray:
        """
        Wall-tight Gaussian spread.

        This version deliberately uses 4-neighbor connectivity only.  The earlier
        8-neighbor spread could visually tunnel through one-cell diagonal wall
        corners in RViz.  With 4-neighbor expansion, a seed can spread only
        through cells connected by free/unknown traversable space and never across
        SLAM occupied cells.
        """
        if seed_grid.shape != occupied_mask.shape or candidate_mask.shape != occupied_mask.shape:
            return np.asarray(seed_grid, dtype=np.float32)

        seeds = np.asarray(seed_grid, dtype=np.float32)
        occupied_bool = np.asarray(occupied_mask, dtype=bool)
        passable = np.asarray(candidate_mask, dtype=bool) & (~occupied_bool)
        seed_mask = (seeds > 1e-4) & passable
        out = np.zeros_like(seeds, dtype=np.float32)
        if not np.any(seed_mask):
            return out

        sigma_m = max(0.18, min(0.36, 0.26 * max(float(self.gap_max_width_m), 1e-6)))
        radius_m = max(0.42, min(0.82, 2.4 * sigma_m))
        radius_cells = max(int(math.ceil(radius_m / max(self.resolution, 1e-6))), 1)

        ys, xs = np.nonzero(seed_mask)
        vals = seeds[ys, xs]
        max_seeds = 900
        if vals.size > max_seeds:
            keep = np.argpartition(vals, -max_seeds)[-max_seeds:]
            xs = xs[keep]
            ys = ys[keep]
            vals = vals[keep]
        order = np.argsort(-vals)

        neighbors = ((1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0))

        for idx in order:
            sx = int(xs[idx])
            sy = int(ys[idx])
            seed_value = float(vals[idx])
            if seed_value <= 0.0 or not self.in_bounds(sx, sy) or not passable[sy, sx]:
                continue

            x0 = max(0, sx - radius_cells)
            x1 = min(self.width, sx + radius_cells + 1)
            y0 = max(0, sy - radius_cells)
            y1 = min(self.height, sy + radius_cells + 1)
            dist = np.full((y1 - y0, x1 - x0), np.inf, dtype=np.float32)
            heap: list[tuple[float, int, int]] = []
            dist[sy - y0, sx - x0] = 0.0
            heapq.heappush(heap, (0.0, sx, sy))

            while heap:
                d_cells, cx, cy = heapq.heappop(heap)
                if d_cells > float(dist[cy - y0, cx - x0]) + 1e-6:
                    continue
                if not passable[cy, cx]:
                    continue
                d_m = float(d_cells) * self.resolution
                if d_m > radius_m:
                    continue

                weight = seed_value * math.exp(-0.5 * (d_m / max(sigma_m, 1e-6)) ** 2)
                if weight > float(out[cy, cx]):
                    out[cy, cx] = weight

                for dx, dy, step_cost in neighbors:
                    nx = cx + dx
                    ny = cy + dy
                    if nx < x0 or nx >= x1 or ny < y0 or ny >= y1:
                        continue
                    if not passable[ny, nx]:
                        continue
                    nd = float(d_cells) + float(step_cost)
                    if nd * self.resolution > radius_m:
                        continue
                    ly = ny - y0
                    lx = nx - x0
                    if nd + 1e-6 < float(dist[ly, lx]):
                        dist[ly, lx] = nd
                        heapq.heappush(heap, (nd, nx, ny))

        out[occupied_bool] = 0.0
        out[~passable] = 0.0
        return np.clip(out, 0.0, 1.0).astype(np.float32)

    @staticmethod
    def _shift_bool(mask: np.ndarray, dx: int, dy: int) -> np.ndarray:
        """
        out[y, x] == mask[y + dy, x + dx], out-of-bounds -> False.
        Used for vectorized directional neighborhood tests.
        """
        h, w = mask.shape
        out = np.zeros_like(mask, dtype=bool)

        src_y0 = max(0, dy)
        src_y1 = min(h, h + dy)
        dst_y0 = max(0, -dy)
        dst_y1 = min(h, h - dy)

        src_x0 = max(0, dx)
        src_x1 = min(w, w + dx)
        dst_x0 = max(0, -dx)
        dst_x1 = min(w, w - dx)

        if src_y0 < src_y1 and src_x0 < src_x1:
            out[dst_y0:dst_y1, dst_x0:dst_x1] = mask[src_y0:src_y1, src_x0:src_x1]
        return out

    @staticmethod
    def _shift_float(values: np.ndarray, dx: int, dy: int, fill: float = 0.0) -> np.ndarray:
        """out[y, x] == values[y + dy, x + dx], out-of-bounds -> fill."""
        h, w = values.shape
        out = np.full_like(values, fill_value=fill, dtype=np.float32)

        src_y0 = max(0, dy)
        src_y1 = min(h, h + dy)
        dst_y0 = max(0, -dy)
        dst_y1 = min(h, h - dy)

        src_x0 = max(0, dx)
        src_x1 = min(w, w + dx)
        dst_x0 = max(0, -dx)
        dst_x1 = min(w, w - dx)

        if src_y0 < src_y1 and src_x0 < src_x1:
            out[dst_y0:dst_y1, dst_x0:dst_x1] = values[src_y0:src_y1, src_x0:src_x1]
        return out

    @classmethod
    def _dilate_bool(cls, mask: np.ndarray, radius: int = 1) -> np.ndarray:
        radius = int(max(radius, 0))
        if radius <= 0:
            return mask.astype(bool, copy=True)

        if _scipy_binary_dilation is not None:
            # Build a disk structuring element of the given radius.
            yy, xx = np.ogrid[-radius:radius + 1, -radius:radius + 1]
            struct_elem = (xx * xx + yy * yy) <= radius * radius
            return _scipy_binary_dilation(mask.astype(bool, copy=False),
                                          structure=struct_elem).astype(bool)

        # Fallback: loop-based shift+OR when scipy is unavailable.
        out = mask.astype(bool, copy=True)
        for oy in range(-radius, radius + 1):
            for ox in range(-radius, radius + 1):
                if ox == 0 and oy == 0:
                    continue
                if ox * ox + oy * oy > radius * radius:
                    continue
                out |= cls._shift_bool(mask, ox, oy)
        return out

    @classmethod
    def _max_filter_float(cls, values: np.ndarray, radius: int = 1, decay: float = 1.0) -> np.ndarray:
        radius = int(max(radius, 0))
        if radius <= 0:
            return values.astype(np.float32, copy=True)

        base = values.astype(np.float32, copy=False)
        out = base.copy()
        for oy in range(-radius, radius + 1):
            for ox in range(-radius, radius + 1):
                if ox == 0 and oy == 0:
                    continue
                dist = math.sqrt(float(ox * ox + oy * oy))
                if dist > float(radius) + 1e-6:
                    continue
                scale = float(decay) ** dist
                out = np.maximum(out, cls._shift_float(base, ox, oy, fill=0.0) * scale)
        return out.astype(np.float32, copy=False)

    def _nearest_occupied_steps(
        self,
        ix: int,
        iy: int,
        dx: int,
        dy: int,
        max_steps: int,
        occupied: np.ndarray,
    ) -> Optional[int]:
        for step in range(1, max_steps + 1):
            sx = ix + dx * step
            sy = iy + dy * step
            if not self.in_bounds(sx, sy):
                return None
            if occupied[sy, sx]:
                return step
        return None

    def _reset_target_lock(self):
        self._locked_target_ix = None
        self._locked_target_iy = None
        self._locked_target_type = self.TARGET_NONE
        self._target_lock_age = 0
        self._last_target_switched = False
        self._prev_path_target_key = None
        self._prev_path_distance = None

    def _target_type_rank(self, target_type: str) -> int:
        """Lower rank means stronger semantic priority."""
        if target_type == self.TARGET_PRIORITY_GAP:
            return 0
        if target_type == self.TARGET_UNKNOWN:
            return 1
        if target_type == self.TARGET_LOW_CONFIDENCE:
            return 2
        if target_type == self.TARGET_STALE:
            return 3
        return 9

    def _ranked_targets_from_mask(
        self,
        mask: np.ndarray,
        base_score: np.ndarray,
        robot_xy: np.ndarray,
        robot_yaw: float,
        top_k: int = 8,
    ) -> list[tuple[int, int, float, float, float]]:
        """Return candidate targets sorted by semantic score.

        Each tuple is (ix, iy, score, direct_distance, direct_angle_robot).
        The direct angle is used only as a fallback/debug value; reward uses the
        path-next-waypoint angle when a reachable path exists.

        The selector is front-biased, but does not blindly keep a rear target if
        front/side candidates exist. This prevents old priority hypotheses behind
        the robot from dominating target lock.
        """
        ys, xs = np.where(mask & (base_score > 0.01))
        if xs.size == 0:
            return []

        wx = self.origin_x + (xs.astype(np.float32) + 0.5) * self.resolution
        wy = self.origin_y + (ys.astype(np.float32) + 0.5) * self.resolution
        dx = wx - float(robot_xy[0])
        dy = wy - float(robot_xy[1])
        dist = np.sqrt(dx * dx + dy * dy)
        angle_world = np.arctan2(dy, dx)
        angle_robot = np.arctan2(
            np.sin(angle_world - float(robot_yaw)),
            np.cos(angle_world - float(robot_yaw)),
        )
        angle_abs = np.abs(angle_robot)

        # Prefer front/side semantic targets. A path can still initially bend
        # backward if the valid route requires it; this filter only prevents the
        # *semantic target* from being an old rear candidate while front options
        # are available.
        forward_sector = angle_abs <= math.radians(120.0)
        if np.any(forward_sector):
            xs = xs[forward_sector]
            ys = ys[forward_sector]
            dist = dist[forward_sector]
            angle_robot = angle_robot[forward_sector]
            angle_abs = angle_abs[forward_sector]

        front_align = np.exp(-0.5 * (angle_robot / max(math.radians(48.0), 1e-6)) ** 2)
        rear_penalty = np.clip(
            (angle_abs - math.radians(90.0)) / max(math.radians(90.0), 1e-6),
            0.0,
            1.0,
        )
        distance_cost = np.clip(dist / max(self.size_m, 1e-6), 0.0, 1.0)
        visit_cost = np.clip(self.visit_grid[ys, xs].astype(np.float32) / 30.0, 0.0, 0.25)

        score = (
            base_score[ys, xs]
            + 0.22 * front_align
            - 0.34 * rear_penalty
            - 0.14 * distance_cost
            - 0.22 * visit_cost
        )

        order = np.argsort(-score)
        k = min(max(int(top_k), 1), int(order.size))
        ranked: list[tuple[int, int, float, float, float]] = []
        for idx in order[:k]:
            ranked.append(
                (
                    int(xs[idx]),
                    int(ys[idx]),
                    float(score[idx]),
                    float(dist[idx]),
                    float(angle_robot[idx]),
                )
            )
        return ranked

    def _select_best_target_from_mask(
        self,
        mask: np.ndarray,
        base_score: np.ndarray,
        robot_xy: np.ndarray,
        robot_yaw: float,
    ) -> Optional[tuple[int, int, float, float, float]]:
        ranked = self._ranked_targets_from_mask(
            mask=mask,
            base_score=base_score,
            robot_xy=robot_xy,
            robot_yaw=robot_yaw,
            top_k=1,
        )
        return ranked[0] if ranked else None

    def _locked_target_score(
        self,
        selected_mask: np.ndarray,
        base_score: np.ndarray,
        robot_xy: np.ndarray,
        robot_yaw: float,
        selected_type: str,
    ) -> Optional[tuple[int, int, float, float, float]]:
        if self._locked_target_ix is None or self._locked_target_iy is None:
            return None

        ix = int(self._locked_target_ix)
        iy = int(self._locked_target_iy)
        if not self.in_bounds(ix, iy):
            return None
        if self._locked_target_type != selected_type:
            return None
        if not bool(selected_mask[iy, ix]):
            return None
        if float(base_score[iy, ix]) <= 0.01:
            return None

        wx, wy = self.map_to_world(ix, iy)
        dx = float(wx) - float(robot_xy[0])
        dy = float(wy) - float(robot_xy[1])
        dist = math.sqrt(dx * dx + dy * dy)
        angle_world = math.atan2(dy, dx)
        angle_robot = normalize_angle(angle_world - float(robot_yaw))
        angle_abs = abs(float(angle_robot))
        front_align = math.exp(-0.5 * (angle_robot / max(math.radians(48.0), 1e-6)) ** 2)
        rear_penalty = float(
            np.clip(
                (angle_abs - math.radians(90.0)) / max(math.radians(90.0), 1e-6),
                0.0,
                1.0,
            )
        )
        distance_cost = float(np.clip(dist / max(self.size_m, 1e-6), 0.0, 1.0))
        visit_cost = float(np.clip(float(self.visit_grid[iy, ix]) / 30.0, 0.0, 0.25))
        score = (
            float(base_score[iy, ix])
            + 0.22 * front_align
            - 0.34 * rear_penalty
            - 0.14 * distance_cost
            - 0.22 * visit_cost
        )
        return ix, iy, score, dist, angle_robot

    def _nearest_traversable_cell(
        self,
        ix: int,
        iy: int,
        traversable: np.ndarray,
        radius_cells: int = 6,
    ) -> Optional[tuple[int, int]]:
        if self.in_bounds(ix, iy) and bool(traversable[iy, ix]):
            return int(ix), int(iy)

        best: Optional[tuple[int, int, int]] = None
        rmax = max(int(radius_cells), 0)
        for r in range(1, rmax + 1):
            for dy in range(-r, r + 1):
                for dx in range(-r, r + 1):
                    if dx * dx + dy * dy > r * r:
                        continue
                    x = int(ix) + dx
                    y = int(iy) + dy
                    if not self.in_bounds(x, y) or not bool(traversable[y, x]):
                        continue
                    d2 = dx * dx + dy * dy
                    if best is None or d2 < best[0]:
                        best = (d2, x, y)
            if best is not None:
                return best[1], best[2]
        return None

    def _publish_path(self, path_world: Optional[list[tuple[float, float]]] = None):
        """
        Publish the currently selected reachable path as nav_msgs/Path for RViz.

        The path is expressed in the same frame as the internal SLAM/RL maps
        (normally "map"). It is only a visualization/debug artifact; the policy
        still receives compact path statistics through the observation/reward.
        """
        if self.path_pub is None:
            return

        if path_world is None:
            path_world = self._last_path_world

        msg = NavPath()
        msg.header.frame_id = self.frame_id
        # stamp=0: RViz uses the latest odom transform and does not extrapolate.
        msg.header.stamp.sec = 0
        msg.header.stamp.nanosec = 0

        for x, y in path_world:
            pose = PoseStamped()
            pose.header = msg.header
            pose.pose.position.x = float(x)
            pose.pose.position.y = float(y)
            pose.pose.position.z = 0.02
            pose.pose.orientation.w = 1.0
            msg.poses.append(pose)

        self.path_pub.publish(msg)

    def _set_path_cells_for_visualization(self, path_cells: list[tuple[int, int]]):
        if not path_cells:
            self._last_path_world = []
            self._publish_path([])
            return

        # Downsample long paths so RViz remains responsive while preserving shape.
        max_points = 180
        if len(path_cells) > max_points:
            step = max(1, int(math.ceil(len(path_cells) / max_points)))
            sampled = path_cells[::step]
            if sampled[-1] != path_cells[-1]:
                sampled.append(path_cells[-1])
        else:
            sampled = path_cells

        self._last_path_world = [self.map_to_world(ix, iy) for ix, iy in sampled]
        self._publish_path(self._last_path_world)

    def _clear_path_visualization(self):
        self._last_path_world = []
        self._publish_path([])

    def _planner_traversable_grid(self) -> np.ndarray:
        """Build the traversability grid once per frontier-info computation.

        Older code rebuilt occupied/free/inflated masks inside every candidate
        path query. With multi-path reward that means many full-map allocations
        per RL step. This helper centralizes that work so all candidate BFS
        calls share the same traversability layer.
        """
        struct = self._structural_grid()
        occupied = struct >= min(float(self.gap_occupied_threshold), 55.0)
        known_free = (struct >= 0) & (struct <= 35)
        confirmed_free = self.confidence_grid >= self.min_known_confidence
        traversable_base = known_free | confirmed_free
        inflated_occupied = self._dilate_bool(occupied, radius=1)
        return traversable_base & (~inflated_occupied)


    def _path_guidance_to_target(
        self,
        robot_xy: np.ndarray,
        robot_yaw: float,
        target_ix: int,
        target_iy: int,
        selected_type: str,
        commit: bool = True,
        traversable: Optional[np.ndarray] = None,
    ) -> tuple[bool, float, float, float]:
        """
        Compute an actionable path direction to the selected target.

        The old reward used the straight bearing robot->target. That is wrong
        when a wall is between the robot and the target. This helper plans on
        the current SLAM occupancy layer and returns the direction of a reachable
        next waypoint along the path. Reward then follows this path angle, not
        the raw target bearing.

        Returns:
          reachable, path_distance_m, path_angle_robot, signed_path_progress_m
        """
        robot_ix, robot_iy = self.world_to_map(float(robot_xy[0]), float(robot_xy[1]))
        if not self.in_bounds(robot_ix, robot_iy) or not self.in_bounds(target_ix, target_iy):
            if commit:
                self._prev_path_target_key = None
                self._prev_path_distance = None
                self._clear_path_visualization()
            return False, self.size_m, 0.0, 0.0

        if traversable is None:
            traversable = self._planner_traversable_grid()

        start = self._nearest_traversable_cell(robot_ix, robot_iy, traversable, radius_cells=4)
        goal = self._nearest_traversable_cell(target_ix, target_iy, traversable, radius_cells=8)
        if start is None or goal is None:
            if commit:
                self._prev_path_target_key = None
                self._prev_path_distance = None
                self._clear_path_visualization()
            return False, self.size_m, 0.0, 0.0

        sx, sy = start
        gx, gy = goal
        if sx == gx and sy == gy:
            target_key = (int(target_ix), int(target_iy), str(selected_type))
            if commit:
                self._prev_path_target_key = target_key
                self._prev_path_distance = 0.0
                self._set_path_cells_for_visualization([(sx, sy)])
            return True, 0.0, 0.0, 0.0

        direct_cells = math.sqrt(float((gx - sx) ** 2 + (gy - sy) ** 2))
        pad_cells = int(
            np.clip(
                max(40.0, direct_cells * 0.55),
                40.0,
                140.0,
            )
        )
        x0 = max(0, min(sx, gx) - pad_cells)
        x1 = min(self.width - 1, max(sx, gx) + pad_cells)
        y0 = max(0, min(sy, gy) - pad_cells)
        y1 = min(self.height - 1, max(sy, gy) + pad_cells)

        local = traversable[y0 : y1 + 1, x0 : x1 + 1]
        h, w = local.shape
        if h <= 0 or w <= 0:
            if commit:
                self._prev_path_target_key = None
                self._prev_path_distance = None
                self._clear_path_visualization()
            return False, self.size_m, 0.0, 0.0

        lsx, lsy = sx - x0, sy - y0
        lgx, lgy = gx - x0, gy - y0
        if not (0 <= lsx < w and 0 <= lsy < h and 0 <= lgx < w and 0 <= lgy < h):
            if commit:
                self._prev_path_target_key = None
                self._prev_path_distance = None
                self._clear_path_visualization()
            return False, self.size_m, 0.0, 0.0

        total = int(h * w)
        # Avoid pathological per-step planning cost if the map expanded very far.
        if total > 90000:
            if commit:
                self._prev_path_target_key = None
                self._prev_path_distance = None
                self._clear_path_visualization()
            return False, self.size_m, 0.0, 0.0

        start_flat = int(lsy * w + lsx)
        goal_flat = int(lgy * w + lgx)
        parent = np.full(total, -1, dtype=np.int32)
        parent[start_flat] = start_flat
        q: deque[int] = deque([start_flat])
        neighbor_offsets = (
            (-1, 0),
            (1, 0),
            (0, -1),
            (0, 1),
            (-1, -1),
            (1, -1),
            (-1, 1),
            (1, 1),
        )

        found = False
        while q:
            cur = q.popleft()
            if cur == goal_flat:
                found = True
                break
            cy = cur // w
            cx = cur - cy * w
            for ndx, ndy in neighbor_offsets:
                nx = cx + ndx
                ny = cy + ndy
                if nx < 0 or nx >= w or ny < 0 or ny >= h:
                    continue
                nf = ny * w + nx
                if parent[nf] != -1 or not bool(local[ny, nx]):
                    continue
                # Prevent diagonal corner cutting through two occupied cells.
                if ndx != 0 and ndy != 0:
                    if not bool(local[cy, nx]) or not bool(local[ny, cx]):
                        continue
                parent[nf] = cur
                q.append(nf)

        if not found and parent[goal_flat] == -1:
            if commit:
                self._prev_path_target_key = None
                self._prev_path_distance = None
                self._clear_path_visualization()
            return False, self.size_m, 0.0, 0.0

        path: list[tuple[int, int]] = []
        cur = goal_flat
        guard = 0
        while cur != start_flat and cur >= 0 and guard < total:
            cy = cur // w
            cx = cur - cy * w
            path.append((cx + x0, cy + y0))
            cur = int(parent[cur])
            guard += 1
        path.append((sx, sy))
        path.reverse()
        if (
            commit
            and self.path_pub is not None
            and self.path_visual_publish_every_n > 0
            and (int(self.step_index) % int(self.path_visual_publish_every_n) == 0)
        ):
            self._set_path_cells_for_visualization(path)

        if len(path) < 2:
            path_distance = 0.0
            path_angle = 0.0
        else:
            # Full path distance is used for reward progress. The previous
            # version accidentally stopped accumulating when the lookahead
            # waypoint was found, which made progress nearly constant around the
            # lookahead distance and weakened the path-following signal.
            path_distance = 0.0
            lookahead_m = 0.45
            chosen_ix, chosen_iy = path[-1]
            prev_x, prev_y = path[0]
            accumulated = 0.0
            chosen_set = False
            for px, py in path[1:]:
                step_m = self.resolution * math.sqrt(float((px - prev_x) ** 2 + (py - prev_y) ** 2))
                path_distance += step_m
                accumulated += step_m
                if (not chosen_set) and accumulated >= lookahead_m:
                    chosen_ix, chosen_iy = px, py
                    chosen_set = True
                prev_x, prev_y = px, py

            wx, wy = self.map_to_world(chosen_ix, chosen_iy)
            path_angle_world = math.atan2(float(wy) - float(robot_xy[1]), float(wx) - float(robot_xy[0]))
            path_angle = normalize_angle(path_angle_world - float(robot_yaw))

        target_key = (int(target_ix), int(target_iy), str(selected_type))
        if commit and self._prev_path_target_key == target_key and self._prev_path_distance is not None:
            path_progress = float(self._prev_path_distance) - float(path_distance)
        else:
            path_progress = 0.0
        if commit:
            self._prev_path_target_key = target_key
            self._prev_path_distance = float(path_distance)

        return True, float(path_distance), float(path_angle), float(path_progress)


    @staticmethod
    def _append_diverse_path_angle(
        angles: list[float],
        angle: float,
        min_separation_rad: float = math.radians(18.0),
        max_count: int = 8,
    ) -> None:
        """
        Append a robot-relative path lookahead angle if it adds a meaningfully
        different branch. This keeps multi-path reward from being dominated by
        near-duplicate paths to adjacent cells.
        """
        if len(angles) >= int(max_count):
            return
        try:
            a = float(angle)
        except Exception:
            return
        if not np.isfinite(a):
            return
        for old in angles:
            diff = math.atan2(math.sin(a - float(old)), math.cos(a - float(old)))
            if abs(diff) < float(min_separation_rad):
                return
        angles.append(a)

    def compute_frontier_info(
        self,
        robot_xy: np.ndarray,
        robot_yaw: float,
    ) -> tuple[int, float, float, float, str, bool, float, float, float, int, tuple[float, ...]]:
        """
        Select a stable exploration target without /rl_path planning.

        Path/BFS guidance is intentionally removed.  The returned target is a
        robot-relative semantic target bearing:

          1. active priority_gap cells from priority_map
          2. unknown frontier cells
          3. low-confidence / unconfirmed free cells

        The angle returned in frontier_angle is direct target bearing in the
        robot frame.  Positive angle means target is to the robot's left, and
        negative angle means target is to the robot's right.  All path_* fields
        are kept only for API compatibility and are always neutral.
        """
        struct = self._structural_grid()
        base_unknown = struct < 0
        base_free = (struct >= 0) & (struct <= 35)
        base_occupied = struct >= min(float(self.gap_occupied_threshold), 55.0)

        confirmed = self.confidence_grid >= self.min_known_confidence
        traversable_anchor = base_free | confirmed

        adj_anchor = np.zeros_like(base_unknown, dtype=bool)
        adj_anchor[1:, :] |= traversable_anchor[:-1, :]
        adj_anchor[:-1, :] |= traversable_anchor[1:, :]
        adj_anchor[:, 1:] |= traversable_anchor[:, :-1]
        adj_anchor[:, :-1] |= traversable_anchor[:, 1:]

        unknown_frontier = base_unknown & adj_anchor
        unconfirmed_free = base_free & (self.confidence_grid < self.min_known_confidence)

        unconfirmed_weight = np.zeros((self.height, self.width), dtype=np.float32)
        if np.any(unconfirmed_free):
            unconfirmed_weight[unconfirmed_free] = (
                1.0
                - np.clip(
                    self.confidence_grid[unconfirmed_free] / max(self.min_known_confidence, 1e-6),
                    0.0,
                    1.0,
                )
            )

        active_priority = np.clip(self._active_priority_grid() / 100.0, 0.0, 1.0)
        priority_gap = active_priority > 0.05
        wall_support = self._occupied_density_score_map()
        structure_gate = 0.10 + 0.90 * wall_support

        priority_gap_score = np.zeros((self.height, self.width), dtype=np.float32)
        priority_gap_score[priority_gap] = active_priority[priority_gap]
        priority_gap_score[base_occupied] = 0.0

        unknown_score = np.zeros((self.height, self.width), dtype=np.float32)
        unknown_score[unknown_frontier] = 0.80 * structure_gate[unknown_frontier]
        unknown_score[base_occupied] = 0.0

        low_conf_score = np.zeros((self.height, self.width), dtype=np.float32)
        low_conf_score[unconfirmed_free] = (
            0.55 * unconfirmed_weight[unconfirmed_free] * structure_gate[unconfirmed_free]
        )
        low_conf_score[base_occupied] = 0.0

        visit_penalty = np.clip(self.visit_grid.astype(np.float32) / 24.0, 0.0, 0.35)
        priority_gap_score = np.clip(priority_gap_score - 0.40 * visit_penalty, 0.0, 1.0)
        unknown_score = np.clip(unknown_score - visit_penalty, 0.0, 1.0)
        low_conf_score = np.clip(low_conf_score - visit_penalty, 0.0, 1.0)

        candidate_classes: list[tuple[str, np.ndarray, np.ndarray]] = []
        if np.any(priority_gap_score > 0.05):
            candidate_classes.append((self.TARGET_PRIORITY_GAP, priority_gap_score > 0.05, priority_gap_score))
        if np.any(unknown_score > 0.05):
            candidate_classes.append((self.TARGET_UNKNOWN, unknown_score > 0.05, unknown_score))
        if np.any(low_conf_score > 0.05):
            candidate_classes.append((self.TARGET_LOW_CONFIDENCE, low_conf_score > 0.05, low_conf_score))

        if not candidate_classes:
            self._reset_target_lock()
            self._prev_path_target_key = None
            self._prev_path_distance = None
            self._clear_path_visualization()
            return 0, self.size_m, 0.0, 0.0, self.TARGET_NONE, False, self.size_m, 0.0, 0.0, 0, ()

        chosen = None
        for candidate_type, candidate_mask, candidate_score in candidate_classes:
            count = int(np.count_nonzero(candidate_mask))
            ranked = self._ranked_targets_from_mask(
                mask=candidate_mask,
                base_score=candidate_score,
                robot_xy=robot_xy,
                robot_yaw=robot_yaw,
                top_k=8,
            )
            if not ranked:
                continue

            old = self._locked_target_score(
                selected_mask=candidate_mask,
                base_score=candidate_score,
                robot_xy=robot_xy,
                robot_yaw=robot_yaw,
                selected_type=candidate_type,
            )
            if old is not None:
                ranked = [old] + [r for r in ranked if not (r[0] == old[0] and r[1] == old[1])]

            chosen = (candidate_type, candidate_mask, candidate_score, count, ranked[0], old)
            break

        if chosen is None:
            self._reset_target_lock()
            self._prev_path_target_key = None
            self._prev_path_distance = None
            self._clear_path_visualization()
            return 0, self.size_m, 0.0, 0.0, self.TARGET_NONE, False, self.size_m, 0.0, 0.0, 0, ()

        selected_type, selected_mask, selected_score, count, best, old = chosen
        target_x, target_y, target_score, target_dist, angle_robot = best

        use_old = False
        if old is not None:
            old_ix, old_iy, old_score, old_dist, old_angle = old
            _, _, best_score, _, _ = best
            if self._target_lock_age < self.target_lock_steps:
                use_old = old_score >= (best_score - self.target_switch_margin)
            else:
                use_old = old_score >= (best_score + 0.03)
            if use_old:
                target_x, target_y, target_score, target_dist, angle_robot = old

        if use_old and old is not None:
            self._target_lock_age += 1
            self._last_target_switched = False
        else:
            switched = (
                self._locked_target_ix is not None
                and self._locked_target_iy is not None
                and (
                    int(target_x) != int(self._locked_target_ix)
                    or int(target_y) != int(self._locked_target_iy)
                    or str(selected_type) != str(self._locked_target_type)
                )
            )
            self._locked_target_ix = int(target_x)
            self._locked_target_iy = int(target_y)
            self._locked_target_type = str(selected_type)
            self._target_lock_age = 0
            self._last_target_switched = bool(switched)

        # /rl_path has been removed from the active algorithm.  Clear stale
        # path visualization/state once and expose only direct semantic bearing.
        self._prev_path_target_key = None
        self._prev_path_distance = None
        self._clear_path_visualization()

        target_priority = float(np.clip(selected_score[int(target_y), int(target_x)], 0.0, 1.0))

        return (
            int(count),
            float(target_dist),
            float(angle_robot),
            target_priority,
            str(selected_type),
            False,          # target_reachable: no path planner is used
            self.size_m,    # path_distance: neutral compatibility value
            0.0,            # path_angle
            0.0,            # path_progress
            0,              # alternative_path_count
            (),             # alternative_path_angles
        )

    def publish(self):
        if not getattr(self, "_has_any_map_publisher", False):
            return

        # Hot path optimization only: publish() used to call _structural_grid() once
        # through _modified_grid() and then again for /rl_filtered_slam_map.  At
        # 10Hz this duplicated full-grid copies.  Use one structural snapshot for
        # both outputs so the published layers are sampled from exactly the same
        # internal state.
        structural_grid = self._structural_grid()
        self.grid = self._modified_grid_from_structural(structural_grid)

        confidence_grid = np.clip(np.round(self.confidence_grid), 0, 100).astype(np.int8)
        # Defer expensive grid computations until we know the publishers exist.
        priority_grid = self._priority_viz_grid() if self.priority_pub is not None else None
        filtered_slam_grid = np.clip(structural_grid, -1, 100).astype(np.int8) if self.filtered_slam_pub is not None else None

        # Map-fixed RViz path:
        # When publish_slam_aligned=True, all debug/RL maps are reprojected onto
        # the current accepted SLAM /map metadata.  This is the only mode in which
        # raw /map and /rl_priority_map should be viewed together in RViz.  It
        # prevents the previous failure mode where the internal RL grid expanded
        # or shifted with robot motion while raw /map stayed in a separate grid.
        ref = getattr(self, "_slam_publish_ref", None)
        if bool(getattr(self, "publish_slam_aligned", False)):
            # /map-locked RViz publication path.  Always try to refresh the raw
            # /map reference directly from the ROS interface before falling back
            # to the internal odom grid.  In pure-velocity mode the learning map
            # stays in odom, but RViz layers must be projected onto raw /map; if
            # this refresh is skipped, confidence/priority rectangles look shifted
            # even though the internal control frame is correct.
            if ref is None:
                try:
                    latest_raw_map = getattr(self.node, "slam_map", None)
                    if latest_raw_map is not None:
                        self.set_slam_publish_reference(latest_raw_map)
                        ref = getattr(self, "_slam_publish_ref", None)
                except Exception:
                    ref = getattr(self, "_slam_publish_ref", None)

            # /map-locked RViz publication path.  Once /map metadata exists, every
            # /rl_* layer is resampled onto that exact canvas.  While SLAM is still
            # booting, publish the internal odom grid instead of hiding all debug
            # layers and spamming MAP_LOCKED_PUBLISH_WAITING_FOR_SLAM_REF.
            if ref is None:
                # v12 strict-map safety: never publish internal pose-frame RL layers
                # under /rl_confidence_map, /rl_priority_map, or /rl_task_map while
                # publish_slam_aligned=True.  That fallback was the source of
                # long-run RViz drift/ghost confidence after SLAM reset failures.
                # If a previous valid /map canvas exists, publish blank layers on
                # that same canvas to clear stale RViz overlays; otherwise publish
                # nothing until an accepted raw /map reference is available.
                try:
                    clear_stale = str(os.environ.get("TB3_RL_CLEAR_RVIZ_WHEN_MAP_MISSING", "1")).strip().lower() not in ("0", "false", "no", "off")
                except Exception:
                    clear_stale = True
                last_ref = getattr(self, "_last_valid_slam_publish_ref", None)
                if clear_stale and isinstance(last_ref, dict):
                    try:
                        h = int(last_ref["height"])
                        w = int(last_ref["width"])
                        if h > 0 and w > 0:
                            blank_task = np.full((h, w), self.UNKNOWN, dtype=np.int8)
                            blank = np.zeros((h, w), dtype=np.int8)
                            self._publish_grid_with_ref(blank_task, self.map_pub, last_ref)
                            if self.legacy_memory_pub is not None:
                                self._publish_grid_with_ref(blank_task, self.legacy_memory_pub, last_ref)
                            self._publish_grid_with_ref(blank, self.confidence_pub, last_ref)
                            if self.priority_pub is not None:
                                self._publish_grid_with_ref(blank, self.priority_pub, last_ref)
                            if self.filtered_slam_pub is not None:
                                self._publish_grid_with_ref(blank_task, self.filtered_slam_pub, last_ref)
                    except Exception:
                        pass
                try:
                    warn_enabled = str(os.environ.get("TB3_RL_MAP_REF_MISSING_WARN", "0")).strip().lower() in ("1", "true", "yes", "on")
                except Exception:
                    warn_enabled = False
                if warn_enabled:
                    now = time.time()
                    if now - float(getattr(self, "_last_waiting_slam_ref_log_time", 0.0)) > 5.0:
                        self._last_waiting_slam_ref_log_time = now
                        if self.node is not None:
                            self.node.get_logger().warn(
                                "MAP_LOCKED_PUBLISH_WAITING_FOR_SLAM_REF | "
                                "raw /map metadata unavailable; not publishing internal fallback layers"
                            )
                return

            # v25 safety path: if the internal canvas is already locked to the
            # exact /map metadata, do not resample.  Resampling failures were the
            # main reason /rl_confidence_map and /rl_priority_map appeared once
            # and then stopped in RViz.  In this direct path every publish call
            # sends the latest internal confidence/priority grids on the /map
            # canvas.
            try:
                same_canvas = (
                    int(self.width) == int(ref["width"])
                    and int(self.height) == int(ref["height"])
                    and abs(float(self.resolution) - float(ref["resolution"])) < 1e-9
                    and abs(float(self.origin_x) - float(ref["origin_x"])) < max(1e-6, float(self.resolution) * 0.25)
                    and abs(float(self.origin_y) - float(ref["origin_y"])) < max(1e-6, float(self.resolution) * 0.25)
                    and str(self.frame_id or "").strip().lstrip("/") == str(ref.get("frame_id", "") or "").strip().lstrip("/")
                )
            except Exception:
                same_canvas = False
            if same_canvas:
                raw_slam = getattr(self, "_slam_publish_raw_grid", None)
                try:
                    if raw_slam is not None and raw_slam.shape == confidence_grid.shape:
                        raw_occ = np.asarray(raw_slam, dtype=np.int16) >= self._slam_occupied_threshold()
                        if np.any(raw_occ):
                            conf_pub_grid = np.asarray(confidence_grid, dtype=np.int8).copy()
                            raw_wall = self._dilate_bool(raw_occ, radius=1)
                            conf_pub_grid[raw_wall] = 0
                            if priority_grid is not None:
                                prio_pub_grid = np.asarray(priority_grid, dtype=np.int8).copy()
                                prio_pub_grid[raw_wall] = 0
                            else:
                                prio_pub_grid = None
                        else:
                            conf_pub_grid = confidence_grid
                            prio_pub_grid = priority_grid
                    else:
                        conf_pub_grid = confidence_grid
                        prio_pub_grid = priority_grid
                except Exception:
                    conf_pub_grid = confidence_grid
                    prio_pub_grid = priority_grid
                filt_pub_grid = filtered_slam_grid
                if filtered_slam_grid is not None and raw_slam is not None and getattr(raw_slam, "shape", None) == filtered_slam_grid.shape:
                    try:
                        filt_pub_grid = np.asarray(raw_slam, dtype=np.int8)
                    except Exception:
                        pass
                self._publish_grid_with_ref(self.grid, self.map_pub, ref)
                if self.legacy_memory_pub is not None:
                    self._publish_grid_with_ref(self.grid, self.legacy_memory_pub, ref)
                try:
                    dbg_n = int(os.environ.get("TB3_RL_CONFIDENCE_PUBLISH_DEBUG_EVERY_N", "20"))
                except Exception:
                    dbg_n = 20
                if dbg_n > 0 and (int(getattr(self, "step_index", 0)) <= 5 or int(getattr(self, "step_index", 0)) - int(getattr(self, "_last_confidence_publish_debug_step", -10_000_000)) >= dbg_n):
                    self._last_confidence_publish_debug_step = int(getattr(self, "step_index", 0))
                    try:
                        self.node.get_logger().info(
                            "CONFIDENCE_PUBLISH | mode=slam_ref_direct "
                            f"step={int(getattr(self, 'step_index', 0))} "
                            f"nonzero={int(np.count_nonzero(conf_pub_grid > 0))} "
                            f"max={float(np.max(conf_pub_grid)) if conf_pub_grid.size else 0.0:.1f} "
                            f"frame={str(ref.get('frame_id', self.frame_id))} "
                            f"size={int(ref['width'])}x{int(ref['height'])} "
                            f"origin=({float(ref['origin_x']):.2f},{float(ref['origin_y']):.2f})"
                        )
                    except Exception:
                        pass
                self._publish_grid_with_ref(conf_pub_grid, self.confidence_pub, ref)
                if self.priority_pub is not None:
                    self._publish_grid_with_ref(prio_pub_grid, self.priority_pub, ref)
                if self.filtered_slam_pub is not None:
                    self._publish_grid_with_ref(filt_pub_grid, self.filtered_slam_pub, ref)
                self._last_direct_map_publish_step = int(getattr(self, "step_index", 0))
                return

            task_ref = self._resample_grid_to_slam_reference(self.grid, self.UNKNOWN, np.int8)
            conf_ref = self._resample_grid_to_slam_reference(confidence_grid, 0, np.int8)
            prio_ref = self._resample_grid_to_slam_reference(priority_grid, 0, np.int8) if priority_grid is not None else None

            raw_slam = getattr(self, "_slam_publish_raw_grid", None)
            filt_ref = None
            try:
                if raw_slam is not None and raw_slam.shape == (int(ref["height"]), int(ref["width"])):
                    filt_ref = np.asarray(raw_slam, dtype=np.int8)
            except Exception:
                filt_ref = None
            if filt_ref is None and filtered_slam_grid is not None:
                # Fallback still uses the same /map metadata, never the internal
                # grid metadata.  This preserves RViz alignment even if the raw
                # SLAM data copy was unavailable for one callback.
                filt_ref = self._resample_grid_to_slam_reference(filtered_slam_grid, self.UNKNOWN, np.int8)

            if task_ref is not None and conf_ref is not None:
                # Final wall clamp on the exact /map canvas. This is a
                # publication safety net; internal confidence/priority already use
                # wall-aware ray tracing/spread, but clamping here guarantees RViz
                # never shows confidence or priority inside /map occupied cells.
                try:
                    _wall_source = filt_ref if filt_ref is not None else conf_ref
                    raw_occ = np.asarray(_wall_source, dtype=np.int16) >= self._slam_occupied_threshold()
                    if raw_occ.shape == conf_ref.shape:
                        conf_ref = np.asarray(conf_ref, dtype=np.int8).copy()
                        raw_wall = self._dilate_bool(raw_occ, radius=1)
                        conf_ref[raw_wall] = 0
                        if prio_ref is not None:
                            prio_ref = np.asarray(prio_ref, dtype=np.int8).copy()
                            prio_ref[raw_wall] = 0
                        # Do not mask the published confidence map by the robot's
                        # current connected component.  That made old explored
                        # rooms disappear in RViz when a transient LiDAR wall or
                        # inflated doorway split the component.  Wall cells are
                        # still removed above, and new writes/priority clears are
                        # still component-gated inside update().
                except Exception:
                    pass

                self._publish_grid_with_ref(task_ref, self.map_pub, ref)
                if self.legacy_memory_pub is not None:
                    self._publish_grid_with_ref(task_ref, self.legacy_memory_pub, ref)
                try:
                    dbg_n = int(os.environ.get("TB3_RL_CONFIDENCE_PUBLISH_DEBUG_EVERY_N", "20"))
                except Exception:
                    dbg_n = 20
                if dbg_n > 0 and (int(getattr(self, "step_index", 0)) <= 5 or int(getattr(self, "step_index", 0)) - int(getattr(self, "_last_confidence_publish_debug_step", -10_000_000)) >= dbg_n):
                    self._last_confidence_publish_debug_step = int(getattr(self, "step_index", 0))
                    try:
                        self.node.get_logger().info(
                            "CONFIDENCE_PUBLISH | mode=slam_ref_resampled "
                            f"step={int(getattr(self, 'step_index', 0))} "
                            f"nonzero={int(np.count_nonzero(conf_ref > 0))} "
                            f"max={float(np.max(conf_ref)) if conf_ref.size else 0.0:.1f} "
                            f"frame={str(ref.get('frame_id', self.frame_id))} "
                            f"size={int(ref['width'])}x{int(ref['height'])} "
                            f"origin=({float(ref['origin_x']):.2f},{float(ref['origin_y']):.2f})"
                        )
                    except Exception:
                        pass
                self._publish_grid_with_ref(conf_ref, self.confidence_pub, ref)
                if self.priority_pub is not None and prio_ref is not None:
                    self._publish_grid_with_ref(prio_ref, self.priority_pub, ref)
                if filt_ref is not None:
                    self._publish_grid_with_ref(filt_ref, self.filtered_slam_pub, ref)
                return

            # v25 fallback: never let RViz map topics stop publishing.  If
            # resampling failed, publish the common internal grid rather than
            # silently skipping.  The direct same-canvas path above handles the
            # normal map-locked case; this fallback keeps diagnostics alive if a
            # transient /map metadata mismatch occurs.
            self._publish_grid(self.grid, self.map_pub)
            if self.legacy_memory_pub is not None:
                self._publish_grid(self.grid, self.legacy_memory_pub)
            self._publish_grid(confidence_grid, self.confidence_pub)
            if self.priority_pub is not None:
                self._publish_grid(priority_grid, self.priority_pub)
            if self.filtered_slam_pub is not None:
                self._publish_grid(filtered_slam_grid, self.filtered_slam_pub)
            return

        # Fallback path for non-map-locked modes only: publish all RL layers on
        # the common internal grid.
        self._publish_grid(self.grid, self.map_pub)
        if self.legacy_memory_pub is not None:
            self._publish_grid(self.grid, self.legacy_memory_pub)
        self._publish_grid(confidence_grid, self.confidence_pub)
        if self.priority_pub is not None and priority_grid is not None:
            self._publish_grid(priority_grid, self.priority_pub)
        if filtered_slam_grid is not None:
            self._publish_grid(filtered_slam_grid, self.filtered_slam_pub)

    def _refresh_publish_grid(self):
        self.grid = self._modified_grid()

    def _modified_grid_from_structural(self, struct: np.ndarray) -> np.ndarray:
        grid = np.full((self.height, self.width), self.UNKNOWN, dtype=np.int8)
        known = struct >= 0
        if np.any(known):
            grid[known] = np.clip(np.round(struct[known]), 0, 100).astype(np.int8)
        return grid

    def _priority_viz_grid(self) -> np.ndarray:
        """RViz priority layer: 0=no priority, 1..100=active priority.

        priority_checked_grid remains an internal regeneration-block mask only.
        Checked/cleared areas are rendered as 0, not -1.
        """
        if bool(getattr(self, "disable_priority_map", False)):
            return np.zeros_like(self.base_grid, dtype=np.int8)
        raw = np.clip(self._active_priority_grid(), 0.0, 100.0).astype(np.float32)
        visible = np.clip(np.round(raw), 0, 100).astype(np.int16)
        active_mask = raw > 0.5
        visible[active_mask] = np.maximum(visible[active_mask], 25)
        try:
            wall = self._inflated_wall_mask(radius=1)
            if wall.shape == visible.shape:
                visible[wall] = 0
        except Exception:
            pass
        return np.clip(visible, 0, 100).astype(np.int8)

    def _publish_grid_with_ref(self, grid: np.ndarray, publisher, ref: dict):
        if publisher is None or ref is None:
            return
        msg = OccupancyGrid()
        # stamp=0 avoids RViz TF extrapolation while metadata still matches /map.
        msg.header.stamp.sec = 0
        msg.header.stamp.nanosec = 0
        msg.header.frame_id = str(ref.get("frame_id", self.frame_id) or self.frame_id)
        msg.info.resolution = float(ref["resolution"])
        msg.info.width = int(ref["width"])
        msg.info.height = int(ref["height"])
        msg.info.origin.position.x = float(ref["origin_x"])
        msg.info.origin.position.y = float(ref["origin_y"])
        msg.info.origin.position.z = float(ref.get("origin_z", 0.0))
        msg.info.origin.orientation.x = float(ref.get("origin_qx", 0.0))
        msg.info.origin.orientation.y = float(ref.get("origin_qy", 0.0))
        msg.info.origin.orientation.z = float(ref.get("origin_qz", 0.0))
        msg.info.origin.orientation.w = float(ref.get("origin_qw", 1.0))
        msg.data = np.ravel(grid).astype(np.int8, copy=False).tolist()
        publisher.publish(msg)

    def _publish_grid(self, grid: np.ndarray, publisher):
        if publisher is None:
            return
        msg = OccupancyGrid()
        # Use stamp=0 for RViz Map displays. With frame_id == Fixed Frame
        # this avoids TF/timestamp edge cases while still carrying the latest
        # full OccupancyGrid through TRANSIENT_LOCAL QoS and periodic keepalive.
        msg.header.stamp.sec = 0
        msg.header.stamp.nanosec = 0
        msg.header.frame_id = self.frame_id
        msg.info.resolution = self.resolution
        msg.info.width = self.width
        msg.info.height = self.height
        msg.info.origin.position.x = self.origin_x
        msg.info.origin.position.y = self.origin_y
        msg.info.origin.position.z = 0.0
        msg.info.origin.orientation.w = 1.0
        msg.data = np.ravel(grid).astype(np.int8, copy=False).tolist()
        publisher.publish(msg)

    def build_update_need_tensor(
        self,
        robot_xy: np.ndarray,
        robot_yaw: float,
        output_size: int = 64,
        size_m: float = 6.4,
        rotate_to_robot: bool = True,
    ) -> np.ndarray:
        """
        CNN input: robot-centric local crop.

        No-priority mode shape = (4, output_size, output_size)
          ch0: SLAM free mask
          ch1: SLAM unknown mask
          ch2: SLAM occupied mask
          ch3: confidence map

        Backward-compatible priority mode shape = (5, output_size, output_size)
          ch4: priority/door-gap map

        The geometry part is one-hot encoded. Frontier-like structure is not a
        separate hand-crafted channel; it is represented implicitly by local
        free/unknown boundaries in ch0/ch1.

        Vectorized sampling keeps the exact robot-centric convention while
        avoiding the previous 64x64 Python nested loop.
        """
        output_size = int(output_size)
        size_m = float(size_m)
        channels = self._update_need_channels()

        half = size_m * 0.5
        step = size_m / max(float(output_size), 1.0)

        coords = (np.arange(output_size, dtype=np.float32) + 0.5) * step
        local_forward = half - coords                      # top = forward
        local_right = coords - half                        # right = robot right
        lf, lr = np.meshgrid(local_forward, local_right, indexing="ij")

        if rotate_to_robot:
            cos_yaw = math.cos(robot_yaw)
            sin_yaw = math.sin(robot_yaw)
            wx = float(robot_xy[0]) + lf * cos_yaw + lr * sin_yaw
            wy = float(robot_xy[1]) + lf * sin_yaw - lr * cos_yaw
        else:
            wx = float(robot_xy[0]) + lr
            wy = float(robot_xy[1]) + lf

        ix = np.floor((wx - self.origin_x) / max(self.resolution, 1e-6)).astype(np.int32)
        iy = np.floor((wy - self.origin_y) / max(self.resolution, 1e-6)).astype(np.int32)
        valid = (ix >= 0) & (ix < self.width) & (iy >= 0) & (iy < self.height)

        out = np.zeros((int(channels.shape[0]), output_size, output_size), dtype=np.float32)
        if np.any(valid):
            out[:, valid] = channels[:, iy[valid], ix[valid]]

        return np.clip(out, 0.0, 1.0).astype(np.float32, copy=False)

    def _update_need_channels(self) -> np.ndarray:
        struct = self._structural_grid()
        base_unknown = struct < 0
        base_free = (struct >= 0) & (struct <= 35)
        base_occupied = struct >= min(float(self.gap_occupied_threshold), 55.0)

        no_priority_policy_input = bool(getattr(self, "disable_priority_map", False)) or (
            str(os.environ.get("TB3_RL_NO_PRIORITY_MODEL_INPUT", "0")).strip().lower()
            in {"1", "true", "yes", "on"}
        )
        channel_count = 4 if no_priority_policy_input else 5
        channels = np.zeros((channel_count, self.height, self.width), dtype=np.float32)

        # One-hot SLAM geometry.
        # ch0/ch1/ch2 are mutually exclusive category masks rather than a scalar
        # occupancy code. This is better for CNN learning because unknown is not
        # numerically halfway between free and occupied; it is a separate state.
        channels[0, base_free] = 1.0
        channels[1, base_unknown] = 1.0
        channels[2, base_occupied] = 1.0

        # Semantic state map.  In no-priority mode, the actor/critic never sees
        # a priority channel at all.  This is intentionally not a zero-filled
        # compatibility channel, because the user wants priority removed from the
        # model input rather than merely disabled at runtime.
        channels[3, :, :] = np.clip(self.confidence_grid / 100.0, 0.0, 1.0)
        if channel_count > 4:
            channels[4, :, :] = np.clip(self._active_priority_grid() / 100.0, 0.0, 1.0)

        return channels


def normalize_angle(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))
