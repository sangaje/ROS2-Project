#!/usr/bin/env python3

import math
from collections import deque
from dataclasses import dataclass
from typing import Optional

import numpy as np
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
    priority_score: float
    priority_gain: float
    priority_cleared_cells: int
    priority_clear_gain: float
    priority_invalidated_cells: int
    priority_invalidated_gain: float
    wall_support_score: float
    open_space_score: float


class ExplorationGridMap:
    """
    SLAM-base + auto-expanding task/confidence/priority maps.

    Map policy input is now 5-channel robot-centric local crop:

      channel 0: SLAM free mask
      channel 1: SLAM unknown mask
      channel 2: SLAM occupied mask
      channel 3: confidence map, normalized 0..1
      channel 4: priority map, normalized 0..1

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
        frame_id: str = "map",
        publish_topic: str = "/rl_task_map",
        confidence_publish_topic: str = "/rl_confidence_map",
        priority_publish_topic: str = "/rl_priority_map",
        path_publish_topic: str = "/rl_path",
        filtered_slam_publish_topic: str = "/rl_filtered_slam_map",
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
    ):
        self.node = node
        self.resolution = float(resolution)
        self.initial_size_m = float(size_m)
        self.size_m = float(size_m)
        self.origin_x = float(origin_x)
        self.origin_y = float(origin_y)
        self.frame_id = str(frame_id)
        self.lidar_stride = max(int(lidar_stride), 1)
        self.max_range = float(max_range)

        self.publish_every_n = max(int(publish_every_n), 1)
        self.update_count = 0
        self.step_index = 0

        # Cached index grids for vectorized SLAM sampling. Rebuilt only when
        # the auto-expanding map changes shape.
        self._index_cache_shape: tuple[int, int] | None = None
        self._index_cache: tuple[np.ndarray, np.ndarray] | None = None
        self._last_slam_sample_key = None
        self._base_grid_needs_resample = True

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

        # Gap confidence suppression is intentionally disabled.
        self.suppress_gap_confidence = False

        # Gap/door priority-map parameters.
        self.gap_occupied_threshold = float(np.clip(gap_occupied_threshold, 0.0, 100.0))
        self.gap_check_radius_m = max(float(gap_check_radius_m), self.resolution)
        self.gap_min_width_m = max(float(gap_min_width_m), self.resolution)
        self.gap_max_width_m = max(float(gap_max_width_m), self.gap_min_width_m)
        self.priority_recompute_interval = max(int(priority_recompute_interval), 1)
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
        self.priority_grid = np.zeros((self.height, self.width), dtype=np.float32)
        # 0..1 persistent mask that suppresses priority around already explored / visited regions.
        # It prevents a door-like gap from remaining attractive after the robot has already checked it.
        self.priority_suppression_grid = np.zeros((self.height, self.width), dtype=np.float32)
        # True means this area has already been checked for priority purposes.
        # It is a hard exclusion for future priority recomputation.
        self.priority_checked_grid = np.zeros((self.height, self.width), dtype=bool)
        self.visit_grid = np.zeros((self.height, self.width), dtype=np.int32)
        self.last_seen_grid = np.full((self.height, self.width), -1, dtype=np.int32)
        self.grid = np.full((self.height, self.width), self.UNKNOWN, dtype=np.int8)

        self.prev_known_cells = 0
        self.prev_mean_confidence = 0.0
        self.prev_priority_score = 0.0
        self._priority_dirty = True
        self._last_priority_invalidated_cells = 0
        self._last_priority_invalidated_gain = 0.0

        # Target hysteresis state. Target selection used to be a per-step global
        # argmax over unknown/low-confidence/priority candidates, which made
        # frontier_angle flip between left/right candidates. Keep a selected
        # target for a short horizon unless a clearly better target appears.
        self.target_lock_steps = 16
        self.target_switch_margin = 0.12
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

        self.map_pub = self.node.create_publisher(OccupancyGrid, publish_topic, map_qos)
        self.confidence_pub = self.node.create_publisher(
            OccupancyGrid,
            confidence_publish_topic,
            map_qos,
        )
        self.priority_pub = self.node.create_publisher(
            OccupancyGrid,
            priority_publish_topic,
            map_qos,
        )

        # RL-only filtered SLAM view. Do not overwrite /map published by slam_toolbox.
        # This topic mirrors the post-filter base_grid actually used by priority/path/CNN logic.
        self.filtered_slam_publish_topic = str(filtered_slam_publish_topic).strip()
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
        if self.keepalive_publish_period_sec > 0.0:
            # RViz sometimes subscribes after the first reset publish. Re-publish
            # full maps periodically so Map displays do not stay at "No map received".
            self._keepalive_timer = self.node.create_timer(
                self.keepalive_publish_period_sec,
                self.publish,
            )

        # Publish a valid empty initial map immediately. This also seeds the
        # TRANSIENT_LOCAL history before the first environment reset/update.
        self.publish()

        self.node.get_logger().info(
            f"SLAM task/confidence/priority map publishers: "
            f"task={publish_topic}, confidence={confidence_publish_topic}, priority={priority_publish_topic}, "
            f"frame_id={self.frame_id}, size={self.width}x{self.height}, resolution={self.resolution}, "
            f"front_fov_deg={math.degrees(self.front_fov_rad):.1f}, "
            f"front_angle_sigma_deg={math.degrees(self.front_angle_sigma_rad):.1f}, "
            f"confidence_max_range={self.confidence_max_range:.2f}, "
            f"seen_confidence_floor={self.seen_confidence_floor:.1f}, "
            f"confidence_decay={'disabled' if self.confidence_decay_per_step <= 0.0 else self.confidence_decay_per_step}, "
            f"gap_confidence_suppression=disabled, "
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
            f"path_topic={self.path_publish_topic or '(disabled)'}, "
            f"filtered_slam_topic={self.filtered_slam_publish_topic or '(disabled)'}, "
            f"cnn_channels=5, "
            f"legacy_memory_alias={self.legacy_memory_topic or '(disabled)'}, "
            f"keepalive_publish_period={self.keepalive_publish_period_sec:.2f}s"
        )

    def reset(self):
        self.base_grid.fill(self.UNKNOWN)
        self.correction_logodds_grid.fill(0.0)
        self.confidence_grid.fill(0.0)
        self.priority_grid.fill(0.0)
        self.priority_suppression_grid.fill(0.0)
        self.priority_checked_grid.fill(False)
        self.visit_grid.fill(0)
        self.last_seen_grid.fill(-1)
        self.grid.fill(self.UNKNOWN)
        self.prev_known_cells = 0
        self.prev_mean_confidence = 0.0
        self.prev_priority_score = 0.0
        self._last_priority_invalidated_cells = 0
        self._last_priority_invalidated_gain = 0.0
        self.update_count = 0
        self.step_index = 0
        self._priority_dirty = True
        self._last_slam_sample_key = None
        self._base_grid_needs_resample = True
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
        self.priority_grid = np.zeros((self.height, self.width), dtype=np.float32)
        # 0..1 persistent mask that suppresses priority around already explored / visited regions.
        # It prevents a door-like gap from remaining attractive after the robot has already checked it.
        self.priority_suppression_grid = np.zeros((self.height, self.width), dtype=np.float32)
        self.priority_checked_grid = np.zeros((self.height, self.width), dtype=bool)
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
    ) -> MapUpdateStats:
        self.step_index += 1
        self._last_priority_invalidated_cells = 0
        self._last_priority_invalidated_gain = 0.0
        self._apply_temporal_decay()

        # Ensure the internal maps can hold the robot-centered local operating region.
        # Do not add padding here: reset_centered_at() already creates exactly this
        # initial window. Adding padding made the map grow immediately at episode start.
        local_pad = max(self.initial_size_m * 0.5, self.confidence_max_range + 0.5)
        self._ensure_world_bounds(
            float(robot_xy[0]) - local_pad,
            float(robot_xy[0]) + local_pad,
            float(robot_xy[1]) - local_pad,
            float(robot_xy[1]) + local_pad,
            padding_m=0.0,
        )

        if self.use_slam_prior and slam_map is not None:
            self._sample_slam_base(slam_map)
        elif not self.use_slam_prior:
            self.base_grid.fill(self.UNKNOWN)

        prev_known = self.known_cell_count()
        prev_mean_conf = self.mean_confidence()
        prev_priority_score = self.priority_score()

        stale_before = self._stale_mask()
        observed_mask = np.zeros((self.height, self.width), dtype=bool)
        priority_clear_mask = np.zeros((self.height, self.width), dtype=np.float32)

        robot_ix, robot_iy = self.world_to_map(float(robot_xy[0]), float(robot_xy[1]))

        robot_visit_count = 0
        if self.in_bounds(robot_ix, robot_iy):
            self.visit_grid[robot_iy, robot_ix] += 1
            robot_visit_count = int(self.visit_grid[robot_iy, robot_ix])
            self._observe_cell(
                robot_ix,
                robot_iy,
                logodds_delta=self.free_logodds_delta,
                confidence_gain=16.0,
                observed_mask=observed_mask,
                confidence_floor=100.0,
            )
            self._mark_priority_clear_visit(priority_clear_mask, robot_ix, robot_iy)

        ranges = np.asarray(scan.ranges, dtype=np.float32)
        angle_min = float(scan.angle_min)
        angle_increment = float(scan.angle_increment)
        range_min = max(float(scan.range_min), 0.05)
        range_max = min(float(scan.range_max), self.max_range)
        confirmation_range_max = min(range_max, self.confidence_max_range)

        for i in range(0, ranges.size, self.lidar_stride):
            rel_angle = normalize_angle(angle_min + float(i) * angle_increment)

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
                    hit = range_min <= r_raw < min(range_max, confirmation_range_max) * 0.98
                    r = float(np.clip(r_raw, range_min, confirmation_range_max))

            beam_angle = robot_yaw + rel_angle
            end_x = float(robot_xy[0]) + r * math.cos(beam_angle)
            end_y = float(robot_xy[1]) + r * math.sin(beam_angle)

            # Rays may point outside the current grid; expand before conversion.
            self._ensure_world_bounds(
                min(float(robot_xy[0]), end_x),
                max(float(robot_xy[0]), end_x),
                min(float(robot_xy[1]), end_y),
                max(float(robot_xy[1]), end_y),
                padding_m=0.50,
            )

            robot_ix, robot_iy = self.world_to_map(float(robot_xy[0]), float(robot_xy[1]))
            end_ix, end_iy = self.world_to_map(end_x, end_y)

            if not self.in_bounds(robot_ix, robot_iy):
                continue

            cells = self.bresenham(robot_ix, robot_iy, end_ix, end_iy)
            if not cells:
                continue

            # /map visibility gate. Even if a LaserScan beam or Gaussian priority
            # clear region geometrically extends farther, confidence and priority
            # checked(-1) must stop at the first occupied SLAM /map cell.
            visible_cells, slam_blocked = self._truncate_ray_by_slam_occlusion(
                cells,
                include_blocking_cell=True,
            )
            if not visible_cells:
                continue

            # Priority clearing uses a short Gaussian front cone, but only along
            # the /map-visible part of the ray. A second visibility mask below
            # clips Gaussian spill-over across walls.
            if in_priority_clear_fov:
                self._mark_priority_clear_ray(
                    priority_clear_mask,
                    robot_ix,
                    robot_iy,
                    visible_cells,
                    angle_weight=self._priority_clear_angle_weight(rel_angle),
                )

            if not in_confidence_fov:
                continue

            effective_hit = bool(hit or slam_blocked)
            free_cells = visible_cells[:-1] if effective_hit else visible_cells
            total_free = max(len(free_cells), 1)

            for j, (cx, cy) in enumerate(free_cells):
                if not self.in_bounds(cx, cy):
                    continue

                dist = r * (float(j + 1) / float(total_free))
                weight = self._distance_weight(dist) * angle_weight
                self._observe_cell(
                    cx,
                    cy,
                    logodds_delta=self.free_logodds_delta * weight,
                    confidence_gain=12.0 * weight,
                    observed_mask=observed_mask,
                    confidence_floor=self.seen_confidence_floor * angle_weight,
                )

            if effective_hit:
                ox, oy = visible_cells[-1]
                if self.in_bounds(ox, oy):
                    # Use the actual grid distance to the visible endpoint, because
                    # /map occlusion can truncate the scan before the LaserScan range.
                    d_cells = math.sqrt(float((int(ox) - int(robot_ix)) ** 2 + (int(oy) - int(robot_iy)) ** 2))
                    endpoint_dist = d_cells * self.resolution
                    weight = self._distance_weight(endpoint_dist) * angle_weight
                    self._observe_cell(
                        ox,
                        oy,
                        logodds_delta=self.occupied_logodds_delta * weight,
                        confidence_gain=16.0 * weight,
                        observed_mask=observed_mask,
                        confidence_floor=self.seen_confidence_floor * angle_weight,
                    )

        # Clip Gaussian priority clear by actual /map line-of-sight. This prevents
        # cells behind walls from becoming checked(-1) just because a Gaussian blob
        # overlapped them.
        if np.any(priority_clear_mask):
            priority_clear_mask *= self._slam_visibility_mask_from_robot(
                robot_ix=robot_ix,
                robot_iy=robot_iy,
                robot_yaw=robot_yaw,
                max_range_m=self.priority_clear_max_range_m,
                fov_rad=self.priority_clear_fov_rad,
            )

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
        if self._priority_dirty or self.step_index % self.priority_recompute_interval == 0:
            old_active_priority = np.clip(self._active_priority_grid(), 0.0, 100.0)
            self._recompute_priority_grid()
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

        stale_refresh_cells = int(np.count_nonzero(stale_before & observed_mask))

        known = self.known_cell_count()
        new_known = max(known - prev_known, 0)

        total_cells = float(self.width * self.height)
        coverage = known / max(total_cells, 1.0)
        coverage_delta = coverage - (self.prev_known_cells / max(total_cells, 1.0))
        self.prev_known_cells = known

        mean_conf = self.mean_confidence()
        confidence_gain = max(mean_conf - prev_mean_conf, 0.0)
        self.prev_mean_confidence = mean_conf

        priority_score = self.priority_score()
        priority_gain = max(priority_score - prev_priority_score, 0.0)
        self.prev_priority_score = priority_score

        wall_support_score, open_space_score = self.compute_forward_structure_scores(
            robot_xy=robot_xy,
            robot_yaw=robot_yaw,
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
        ) = self.compute_frontier_info(robot_xy=robot_xy, robot_yaw=robot_yaw)

        stale_known_cells = self.stale_known_count()
        stale_ratio = stale_known_cells / max(float(known), 1.0)

        low_confidence_cells = self.low_confidence_count()
        base_free_cells = int(np.count_nonzero((self.base_grid >= 0) & (self.base_grid <= 35)))
        low_confidence_ratio = low_confidence_cells / max(float(base_free_cells), 1.0)

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
            target_priority=float(target_priority),
            target_type=str(target_type),
            target_switched=bool(self._last_target_switched),
            target_lock_age=int(self._target_lock_age),
            target_reachable=bool(target_reachable),
            path_distance=float(path_distance),
            path_angle=float(path_angle),
            path_progress=float(path_progress),
            priority_score=float(priority_score),
            priority_gain=float(priority_gain),
            priority_cleared_cells=int(priority_cleared_cells),
            priority_clear_gain=float(priority_clear_gain),
            priority_invalidated_cells=int(self._last_priority_invalidated_cells),
            priority_invalidated_gain=float(self._last_priority_invalidated_gain),
            wall_support_score=float(wall_support_score),
            open_space_score=float(open_space_score),
        )

        self.update_count += 1
        if publish and (self.update_count == 1 or self.update_count % self.publish_every_n == 0):
            self.publish()

        return stats

    def _apply_temporal_decay(self):
        # confidence decay disabled by default. Keep clamp for numerical safety.
        if self.confidence_decay_per_step > 0.0:
            decay = max(0.0, 1.0 - self.confidence_decay_per_step)
            self.confidence_grid *= np.float32(decay)

        np.clip(self.confidence_grid, 0.0, 100.0, out=self.confidence_grid)
        self.correction_logodds_grid.fill(0.0)

    def _observe_cell(
        self,
        ix: int,
        iy: int,
        logodds_delta: float,
        confidence_gain: float,
        observed_mask: Optional[np.ndarray] = None,
        confidence_floor: float = 0.0,
    ):
        _ = logodds_delta
        if not self.in_bounds(ix, iy):
            return

        new_confidence = self.confidence_grid[iy, ix] + float(confidence_gain)
        if confidence_floor > 0.0:
            new_confidence = max(new_confidence, float(confidence_floor))

        self.confidence_grid[iy, ix] = np.clip(new_confidence, 0.0, 100.0)
        self.last_seen_grid[iy, ix] = int(self.step_index)

        if observed_mask is not None and observed_mask.shape == self.confidence_grid.shape:
            observed_mask[iy, ix] = True

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

    def _sample_slam_base(self, slam_map: OccupancyGrid):
        slam_width = int(slam_map.info.width)
        slam_height = int(slam_map.info.height)
        slam_stamp = getattr(getattr(slam_map, "header", None), "stamp", None)
        slam_stamp_key = (getattr(slam_stamp, "sec", 0), getattr(slam_stamp, "nanosec", 0))
        sample_key = (
            slam_stamp_key,
            slam_width,
            slam_height,
            float(slam_map.info.resolution),
            float(slam_map.info.origin.position.x),
            float(slam_map.info.origin.position.y),
            self.width,
            self.height,
            round(float(self.origin_x), 6),
            round(float(self.origin_y), 6),
        )

        if (
            not self._base_grid_needs_resample
            and self._last_slam_sample_key == sample_key
        ):
            return

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

        # Important: do NOT expand the RL maps to the full SLAM /map extent.
        # slam_toolbox may publish a large transient-local map from a previous run
        # immediately after reset. Expanding to that full extent made /rl_* maps
        # jump to huge sizes at episode start. The RL maps now grow only from
        # robot motion and local ray bounds; SLAM is sampled only inside the
        # current RL map window.

        yy, xx = self._grid_index_arrays()
        wx = self.origin_x + (xx + 0.5) * self.resolution
        wy = self.origin_y + (yy + 0.5) * self.resolution

        sx = np.floor((wx - slam_origin_x) / max(slam_res, 1e-6)).astype(np.int32)
        sy = np.floor((wy - slam_origin_y) / max(slam_res, 1e-6)).astype(np.int32)

        valid = (sx >= 0) & (sx < slam_width) & (sy >= 0) & (sy < slam_height)

        self.base_grid.fill(self.UNKNOWN)
        self.base_grid[valid] = data[sy[valid], sx[valid]]
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
        new_max_x = cur_max_x + right_chunks * chunk_m
        new_min_y = cur_min_y - down_chunks * chunk_m
        new_max_y = cur_max_y + up_chunks * chunk_m

        # Keep exact cell counts. Because chunk_m is an integer multiple of resolution,
        # width/height increase by multiples of map_expand_chunk_cells.
        new_width = self.width + (left_chunks + right_chunks) * self.map_expand_chunk_cells
        new_height = self.height + (down_chunks + up_chunks) * self.map_expand_chunk_cells

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
        self.priority_grid = expand_array(self.priority_grid, 0.0, np.float32)
        self.priority_suppression_grid = expand_array(self.priority_suppression_grid, 0.0, np.float32)
        self.priority_checked_grid = expand_array(self.priority_checked_grid, False, bool)
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
        if self.priority_grid.size == 0:
            return 0.0
        active = self._active_priority_grid()
        return float(np.max(np.clip(active / 100.0, 0.0, 1.0)))

    def _active_priority_grid(self) -> np.ndarray:
        if getattr(self, "priority_checked_grid", None) is None:
            return np.clip(self.priority_grid, 0.0, 100.0)
        if self.priority_checked_grid.shape != self.priority_grid.shape:
            self.priority_checked_grid = np.zeros_like(self.priority_grid, dtype=bool)
        return np.where(self.priority_checked_grid, 0.0, np.clip(self.priority_grid, 0.0, 100.0)).astype(np.float32)

    def stale_known_count(self) -> int:
        return int(np.count_nonzero(self._stale_mask()))

    def low_confidence_count(self) -> int:
        return int(np.count_nonzero(self._low_confidence_mask()))

    def _stale_mask(self) -> np.ndarray:
        known = self.confidence_grid >= self.min_known_confidence
        seen = self.last_seen_grid >= 0
        age = self.step_index - self.last_seen_grid
        return known & seen & (age >= self.stale_after_steps)

    def _low_confidence_mask(self) -> np.ndarray:
        base_free = (self.base_grid >= 0) & (self.base_grid <= 35)
        return base_free & (self.confidence_grid < self.low_confidence_threshold)

    def _modified_grid(self) -> np.ndarray:
        grid = np.full((self.height, self.width), self.UNKNOWN, dtype=np.int8)
        base_known = self.base_grid >= 0
        if np.any(base_known):
            grid[base_known] = np.clip(
                np.round(self.base_grid[base_known]),
                0,
                100,
            ).astype(np.int8)
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

    def _is_slam_occupied_cell(self, ix: int, iy: int) -> bool:
        """
        True only when the SLAM /map-derived base_grid says this cell is occupied.

        This intentionally does not use confidence_grid or priority_grid. The user-facing
        rule is: confidence/priority-clear must not pass through walls, and the wall
        test comes from the original SLAM /map only.
        """
        if not self.in_bounds(ix, iy):
            return False
        return bool(self.base_grid[int(iy), int(ix)] >= self._slam_occupied_threshold())

    def _truncate_ray_by_slam_occlusion(
        self,
        cells: list[tuple[int, int]],
        include_blocking_cell: bool = True,
    ) -> tuple[list[tuple[int, int]], bool]:
        """
        Cut a ray at the first occupied SLAM /map cell.

        Returns:
          visible_cells: cells up to the first /map obstacle. If include_blocking_cell
                         is True, the blocking wall cell is included as the final cell.
          blocked:       True if a /map obstacle truncated the ray.

        This prevents both confidence increase and priority checked(-1) updates from
        leaking into rooms/empty space behind a wall.
        """
        if not cells:
            return [], False

        visible: list[tuple[int, int]] = []
        for idx, (cx, cy) in enumerate(cells):
            if not self.in_bounds(cx, cy):
                break

            # The robot's own cell can be occupied/noisy in SLAM after reset; do not
            # let it occlude the ray.
            if idx > 0 and self._is_slam_occupied_cell(cx, cy):
                if include_blocking_cell:
                    visible.append((int(cx), int(cy)))
                return visible, True

            visible.append((int(cx), int(cy)))

        return visible, False

    def _has_slam_line_of_sight(self, robot_ix: int, robot_iy: int, ix: int, iy: int) -> bool:
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

        for idx, (cx, cy) in enumerate(cells):
            if idx == 0:
                continue
            if self._is_slam_occupied_cell(cx, cy):
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
        max_range2 = float(max_range_m) * float(max_range_m)
        half_fov = float(fov_rad) * 0.5

        for y in range(y0, y1):
            wy = self.origin_y + (float(y) + 0.5) * self.resolution
            dy = wy - (self.origin_y + (float(robot_iy) + 0.5) * self.resolution)
            for x in range(x0, x1):
                wx = self.origin_x + (float(x) + 0.5) * self.resolution
                dx = wx - (self.origin_x + (float(robot_ix) + 0.5) * self.resolution)
                dist2 = dx * dx + dy * dy
                if dist2 > max_range2:
                    continue
                rel = normalize_angle(math.atan2(dy, dx) - float(robot_yaw))
                if abs(rel) > half_fov:
                    continue
                if self._has_slam_line_of_sight(robot_ix, robot_iy, x, y):
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

    def _mark_priority_clear_visit(self, weight_grid: np.ndarray, ix: int, iy: int):
        """
        Direct robot visit clears priority with a compact Gaussian footprint.
        """
        self._paint_gaussian_blob(
            weight_grid=weight_grid,
            ix=ix,
            iy=iy,
            sigma_m=self.priority_clear_visit_sigma_m,
            max_radius_m=self.priority_clear_robot_radius_m,
            base_weight=1.0,
        )

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
        Mark a short front cone as checked using Gaussian blobs along the beam.

        Compared with the previous binary ray, this clears a wider and smoother
        region around a door/gap candidate, so the policy receives reward when it
        actually inspects the region rather than needing to hit exactly one cell.
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

        for cx, cy in cells:
            if not self.in_bounds(cx, cy):
                continue

            d_cells = math.sqrt(float((int(cx) - rix) ** 2 + (int(cy) - riy) ** 2))
            if d_cells > max_cells:
                break

            d_m = d_cells * self.resolution
            # Keep the ray core mostly strong but reduce the far tail mildly.
            radial_weight = 0.55 + 0.45 * math.exp(
                -0.5 * (d_m / max(self.priority_clear_max_range_m, 1e-6)) ** 2
            )
            self._paint_gaussian_blob(
                weight_grid=weight_grid,
                ix=int(cx),
                iy=int(cy),
                sigma_m=self.priority_clear_sigma_m,
                max_radius_m=3.0 * self.priority_clear_sigma_m,
                base_weight=base_angle_weight * radial_weight,
            )

    def _update_priority_checked(self, clear_weight: Optional[np.ndarray]) -> tuple[int, float]:
        if clear_weight is None or clear_weight.shape != self.priority_grid.shape:
            return 0, 0.0
        if self.priority_checked_grid.shape != self.priority_grid.shape:
            self.priority_checked_grid = np.zeros_like(self.priority_grid, dtype=bool)

        clear_weight = np.clip(clear_weight.astype(np.float32, copy=False), 0.0, 1.0)
        newly_checked = (clear_weight >= self.priority_clear_min_weight) & (~self.priority_checked_grid)
        if not np.any(newly_checked):
            return 0, 0.0

        previous_priority = np.clip(self.priority_grid.astype(np.float32), 0.0, 100.0)
        counted = newly_checked & (previous_priority >= self.priority_clear_min_value)
        cleared_cells = int(np.count_nonzero(counted))
        # Gaussian-weighted clear gain: high-priority cells close to the ray/robot center pay more reward.
        clear_gain = float(np.sum((previous_priority[counted] / 100.0) * clear_weight[counted]))

        self.priority_checked_grid[newly_checked] = True
        self.priority_grid[newly_checked] = 0.0
        if self.priority_suppression_grid.shape == self.priority_grid.shape:
            self.priority_suppression_grid[newly_checked] = np.maximum(
                self.priority_suppression_grid[newly_checked],
                self.priority_visit_suppression_max,
            )
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
        occ = (self.base_grid >= min(float(self.gap_occupied_threshold), 55.0)).astype(np.float32)
        if occ.size == 0 or not np.any(occ > 0.0):
            return np.zeros((self.height, self.width), dtype=np.float32)

        radius_m = self.wall_support_radius_m if radius_m is None else float(radius_m)
        radius_cells = max(int(math.ceil(radius_m / max(self.resolution, 1e-6))), 1)
        k = 2 * radius_cells + 1

        padded = np.pad(occ, ((radius_cells, radius_cells), (radius_cells, radius_cells)), mode="constant")
        integ = np.pad(padded, ((1, 0), (1, 0)), mode="constant").cumsum(axis=0).cumsum(axis=1)
        sums = integ[k:, k:] - integ[:-k, k:] - integ[k:, :-k] + integ[:-k, :-k]
        density = sums / float(k * k)
        return np.clip(density / max(self.wall_support_density_threshold, 1e-6), 0.0, 1.0).astype(np.float32)

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

    def _recompute_priority_grid(self):
        """
        Door/gap/opening priority map, vectorized.

        The scoring is intentionally equivalent to the previous permissive logic,
        but it avoids a Python loop over every candidate cell. For each direction
        it finds the nearest occupied cell on both sides using shifted boolean
        masks and combines the resulting width/symmetry scores with NumPy.
        """
        occ_threshold = min(float(self.gap_occupied_threshold), 55.0)
        base_occupied = self.base_grid >= occ_threshold
        if self.priority_checked_grid.shape != self.base_grid.shape:
            self.priority_checked_grid = np.zeros_like(self.base_grid, dtype=bool)
        candidate = (~base_occupied) & (~self.priority_checked_grid)

        if not np.any(base_occupied) or not np.any(candidate):
            self.priority_grid.fill(0.0)
            return

        max_steps = max(int(math.ceil(self.gap_check_radius_m / self.resolution)), 1)
        dirs = [
            (1, 0, 1.0),
            (0, 1, 1.0),
            (1, 1, math.sqrt(2.0)),
            (1, -1, math.sqrt(2.0)),
            (2, 1, math.sqrt(5.0)),
            (1, 2, math.sqrt(5.0)),
            (2, -1, math.sqrt(5.0)),
            (1, -2, math.sqrt(5.0)),
            (3, 1, math.sqrt(10.0)),
            (1, 3, math.sqrt(10.0)),
            (3, -1, math.sqrt(10.0)),
            (1, -3, math.sqrt(10.0)),
        ]

        min_width = max(float(self.gap_min_width_m), self.resolution)
        max_width = max(float(self.gap_max_width_m), min_width + self.resolution)
        soft_upper = max(max_width * 1.80, max_width + 0.60)
        soft_lower = max(min_width * 0.50, self.resolution)
        width_center = 0.5 * (min_width + max_width)
        width_sigma = max(0.45 * (max_width - min_width), 0.35, self.resolution)

        prio = np.zeros((self.height, self.width), dtype=np.float32)

        # First occupied distance in +dir / -dir. 0 means not found.
        for dx, dy, step_scale in dirs:
            pos_steps = np.zeros((self.height, self.width), dtype=np.int16)
            neg_steps = np.zeros((self.height, self.width), dtype=np.int16)

            for step in range(1, max_steps + 1):
                pos_hit = self._shift_bool(base_occupied, dx * step, dy * step)
                neg_hit = self._shift_bool(base_occupied, -dx * step, -dy * step)

                pos_set = (pos_steps == 0) & pos_hit
                neg_set = (neg_steps == 0) & neg_hit
                if np.any(pos_set):
                    pos_steps[pos_set] = step
                if np.any(neg_set):
                    neg_steps[neg_set] = step

            valid = candidate & (pos_steps > 0) & (neg_steps > 0)
            if not np.any(valid):
                continue

            m_pos = pos_steps.astype(np.float32) * self.resolution * float(step_scale)
            m_neg = neg_steps.astype(np.float32) * self.resolution * float(step_scale)
            width = m_pos + m_neg

            width_valid = valid & (width >= soft_lower) & (width <= soft_upper)
            if not np.any(width_valid):
                continue

            symmetry = 1.0 - np.minimum(np.abs(m_pos - m_neg) / np.maximum(width, 1e-6), 1.0)
            width_score = np.exp(-0.5 * ((width - width_center) / width_sigma) ** 2)
            nominal_boost = np.where((width >= min_width) & (width <= max_width), 1.0, 0.60).astype(np.float32)
            score = nominal_boost * width_score * (0.45 + 0.55 * symmetry)
            prio = np.maximum(prio, np.where(width_valid, score, 0.0).astype(np.float32))

        if not np.any(prio > 0.0):
            self.priority_grid.fill(0.0)
            return

        # Require structural wall support. This prevents priority from appearing
        # in featureless empty voids while preserving door/gap candidates between
        # two obstacle boundaries.
        wall_support = self._occupied_density_score_map()
        prio = prio * (0.20 + 0.80 * wall_support)

        confidence_deficit = 1.0 - np.clip(self.confidence_grid / 100.0, 0.0, 1.0)
        base_unknown = self.base_grid < 0
        low_confirmed = self.confidence_grid < max(self.low_confidence_threshold, self.min_known_confidence)
        near_exploration_edge = self._dilate_bool(base_unknown | low_confirmed, radius=2)

        edge_boost = np.where(near_exploration_edge, 1.0, 0.72).astype(np.float32)
        conf_boost = (0.78 + 0.22 * confidence_deficit).astype(np.float32)
        prio = prio * edge_boost * conf_boost

        # Persistent suppression: robot-visited / front-FOV observed regions are
        # less attractive in the priority map. We leave a small residual priority
        # instead of hard-zeroing it, so if the environment changes or every other
        # candidate is worse, the policy can still pass through the area.
        suppression = np.clip(
            self.priority_suppression_grid,
            0.0,
            self.priority_visit_suppression_max,
        ).astype(np.float32)
        prio = prio * (1.0 - suppression)
        prio[self.priority_checked_grid] = 0.0

        prio = self._max_filter_float(prio, radius=2, decay=0.86)
        prio[self.priority_checked_grid] = 0.0
        self.priority_grid = np.clip(prio * 100.0, 0.0, 100.0).astype(np.float32)

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
        msg.header.stamp = self.node.get_clock().now().to_msg()

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

    def _path_guidance_to_target(
        self,
        robot_xy: np.ndarray,
        robot_yaw: float,
        target_ix: int,
        target_iy: int,
        selected_type: str,
        commit: bool = True,
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

        occupied = self.base_grid >= 65
        # Plan only through cells that are actually known/confirmed as traversable.
        # The previous version used ~occupied, which made unknown space traversable
        # and could create misleading paths through unmapped areas or behind walls.
        known_free = (self.base_grid >= 0) & (self.base_grid <= 35)
        confirmed_free = self.confidence_grid >= self.min_known_confidence
        traversable_base = known_free | confirmed_free

        # Light inflation avoids rewarding paths that graze a wall. Keep it small
        # because TurtleBot house door/gap candidates may be narrow.
        inflated_occupied = self._dilate_bool(occupied, radius=1)
        traversable = traversable_base & (~inflated_occupied)

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
        if commit:
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

    def compute_frontier_info(
        self,
        robot_xy: np.ndarray,
        robot_yaw: float,
    ) -> tuple[int, float, float, float, str, bool, float, float, float]:
        """
        Select a stable exploration target and compute path guidance.

        Priority order is now explicit:
          1. active priority_gap cells from priority_map
          2. unknown frontier cells
          3. low-confidence / unconfirmed free cells

        Previous versions mixed all candidates in one argmax, so target_type and
        frontier_angle could oscillate every step. This function first selects a
        semantic class, then applies target hysteresis within that class.
        """
        base_unknown = self.base_grid < 0
        base_free = (self.base_grid >= 0) & (self.base_grid <= 35)
        base_occupied = self.base_grid >= 65

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

        # Separate score maps. Do not let unknown frontier masquerade as a
        # priority_gap target in reward.py.
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

        # Visit penalty applies to all target classes, but it should not erase
        # priority_gap completely because checked_grid already handles hard removal.
        visit_penalty = np.clip(self.visit_grid.astype(np.float32) / 24.0, 0.0, 0.35)
        priority_gap_score = np.clip(priority_gap_score - 0.40 * visit_penalty, 0.0, 1.0)
        unknown_score = np.clip(unknown_score - visit_penalty, 0.0, 1.0)
        low_conf_score = np.clip(low_conf_score - visit_penalty, 0.0, 1.0)

        # Semantic priority order is still priority_gap -> unknown -> low-confidence,
        # but unreachable priority_gap must not block reachable unknown/frontier
        # targets. Therefore we evaluate a small ranked list per class and pick
        # the first reachable path target.
        candidate_classes: list[tuple[str, np.ndarray, np.ndarray]] = []
        if np.any(priority_gap_score > 0.05):
            candidate_classes.append((self.TARGET_PRIORITY_GAP, priority_gap_score > 0.05, priority_gap_score))
        if np.any(unknown_score > 0.05):
            candidate_classes.append((self.TARGET_UNKNOWN, unknown_score > 0.05, unknown_score))
        if np.any(low_conf_score > 0.05):
            candidate_classes.append((self.TARGET_LOW_CONFIDENCE, low_conf_score > 0.05, low_conf_score))

        if not candidate_classes:
            self._reset_target_lock()
            return 0, self.size_m, 0.0, 0.0, self.TARGET_NONE, False, self.size_m, 0.0, 0.0

        chosen = None
        fallback = None

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
                # Put a still-valid locked target into the comparison set.
                ranked = [old] + [r for r in ranked if not (r[0] == old[0] and r[1] == old[1])]

            # Keep the best unreachable candidate only as a last-resort debug target.
            if fallback is None:
                fallback = (candidate_type, candidate_mask, candidate_score, count, ranked[0], False, self.size_m, 0.0, 0.0)

            for cand in ranked:
                cx, cy, cscore, cdist, cangle = cand
                reachable, pdist, pangle, pprogress = self._path_guidance_to_target(
                    robot_xy=robot_xy,
                    robot_yaw=robot_yaw,
                    target_ix=int(cx),
                    target_iy=int(cy),
                    selected_type=str(candidate_type),
                    commit=False,
                )
                if reachable:
                    chosen = (candidate_type, candidate_mask, candidate_score, count, cand, True, pdist, pangle, 0.0)
                    break

            if chosen is not None:
                break

        if chosen is None:
            # No candidate has a reachable path on the current SLAM map. Do not
            # let an unreachable priority_gap dominate reward; expose it only as
            # a weak/unreachable debug target.
            if fallback is None:
                self._reset_target_lock()
                return 0, self.size_m, 0.0, 0.0, self.TARGET_NONE, False, self.size_m, 0.0, 0.0
            chosen = fallback

        selected_type, selected_mask, selected_score, count, best, target_reachable, path_distance, path_angle, path_progress = chosen
        target_x, target_y, target_score, target_dist, angle_robot = best

        # Hysteresis: only keep the previous target when it is from the same
        # semantic class and still reachable. Otherwise an unreachable stale
        # lock can force the robot to rotate toward a dead target.
        old = self._locked_target_score(
            selected_mask=selected_mask,
            base_score=selected_score,
            robot_xy=robot_xy,
            robot_yaw=robot_yaw,
            selected_type=selected_type,
        )

        use_old = False
        if old is not None and bool(target_reachable):
            old_ix, old_iy, old_score, old_dist, old_angle = old
            old_reachable, old_pdist, old_pangle, old_pprogress = self._path_guidance_to_target(
                robot_xy=robot_xy,
                robot_yaw=robot_yaw,
                target_ix=int(old_ix),
                target_iy=int(old_iy),
                selected_type=str(selected_type),
                commit=False,
            )
            if old_reachable:
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

        (
            target_reachable,
            path_distance,
            path_angle,
            path_progress,
        ) = self._path_guidance_to_target(
            robot_xy=robot_xy,
            robot_yaw=robot_yaw,
            target_ix=int(target_x),
            target_iy=int(target_y),
            selected_type=str(selected_type),
            commit=True,
        )

        target_priority = float(np.clip(selected_score[int(target_y), int(target_x)], 0.0, 1.0))

        # frontier_angle is now the actionable path-next-waypoint angle when the
        # target is reachable. If no path exists, return the direct angle only
        # for logging/observation; reward gates it out using target_reachable.
        actionable_angle = float(path_angle) if bool(target_reachable) else float(angle_robot)
        actionable_distance = float(path_distance) if bool(target_reachable) else float(target_dist)

        return (
            int(count),
            float(actionable_distance),
            float(actionable_angle),
            target_priority,
            str(selected_type),
            bool(target_reachable),
            float(path_distance),
            float(path_angle),
            float(path_progress),
        )

    def publish(self):
        self._refresh_publish_grid()
        self._publish_grid(self.grid, self.map_pub)

        # Backward-compatible RViz alias. This is NOT the old remember map;
        # it mirrors /rl_task_map so old RViz configs do not show "No map received".
        if self.legacy_memory_pub is not None:
            self._publish_grid(self.grid, self.legacy_memory_pub)

        confidence_grid = np.clip(np.round(self.confidence_grid), 0, 100).astype(np.int8)
        self._publish_grid(confidence_grid, self.confidence_pub)

        priority_grid = self._priority_viz_grid()
        self._publish_grid(priority_grid, self.priority_pub)

        if self.filtered_slam_pub is not None:
            filtered_slam_grid = np.clip(self.base_grid, -1, 100).astype(np.int8)
            self._publish_grid(filtered_slam_grid, self.filtered_slam_pub)

    def _refresh_publish_grid(self):
        self.grid = self._modified_grid()

    def _priority_viz_grid(self) -> np.ndarray:
        """
        Return a high-contrast RViz visualization of the priority map.

        The internal priority_grid remains continuous 0..100 and is still used
        for reward/CNN input. RViz Map displays, however, can make weak door-gap
        candidates almost invisible. For visualization only, any non-zero
        priority is lifted to a visible floor and strong priorities remain near
        100.
        """
        raw = self._active_priority_grid()
        visible = np.zeros_like(raw, dtype=np.float32)

        mask = raw > 0.5
        # 35 is deliberately above the map display's near-white range, so even
        # weak priority cells are visible in RViz. Strong candidates still reach
        # 100. Checked priority cells are published as -1 so they are visibly
        # distinct and, more importantly, never regenerate as targets.
        visible[mask] = 35.0 + 0.65 * raw[mask]

        out = np.clip(np.round(visible), 0, 100).astype(np.int8)
        if self.priority_checked_grid.shape == out.shape:
            out[self.priority_checked_grid] = -1
        return out

    def _publish_grid(self, grid: np.ndarray, publisher):
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
        msg.data = grid.flatten().astype(np.int8).tolist()
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
        CNN input: 5-channel robot-centric local crop.

        shape = (5, output_size, output_size)
          ch0: SLAM free mask
          ch1: SLAM unknown mask
          ch2: SLAM occupied mask
          ch3: confidence map
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

        out = np.zeros((5, output_size, output_size), dtype=np.float32)
        if np.any(valid):
            out[:, valid] = channels[:, iy[valid], ix[valid]]

        return np.clip(out, 0.0, 1.0).astype(np.float32, copy=False)

    def _update_need_channels(self) -> np.ndarray:
        base_unknown = self.base_grid < 0
        base_free = (self.base_grid >= 0) & (self.base_grid <= 35)
        base_occupied = self.base_grid >= 65

        channels = np.zeros((5, self.height, self.width), dtype=np.float32)

        # One-hot SLAM geometry.
        # ch0/ch1/ch2 are mutually exclusive category masks rather than a scalar
        # occupancy code. This is better for CNN learning because unknown is not
        # numerically halfway between free and occupied; it is a separate state.
        channels[0, base_free] = 1.0
        channels[1, base_unknown] = 1.0
        channels[2, base_occupied] = 1.0

        # Semantic state maps. Checked priority cells are already removed by
        # _active_priority_grid(), so CNN sees them as zero-priority regions.
        channels[3, :, :] = np.clip(self.confidence_grid / 100.0, 0.0, 1.0)
        channels[4, :, :] = np.clip(self._active_priority_grid() / 100.0, 0.0, 1.0)

        return channels


def normalize_angle(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))
