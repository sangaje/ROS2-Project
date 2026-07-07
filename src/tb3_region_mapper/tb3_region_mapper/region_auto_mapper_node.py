#!/usr/bin/env python3
"""Region-aware autonomous SLAM mapper for TurtleBot3.

This node is deliberately self-contained: it does not require Nav2.  It uses the
region map only as a *task allocation / coverage layer* and performs simple grid
A* + reactive velocity control on /cmd_vel.  TurtleBot3 Jazzy/Gazebo expects
geometry_msgs/msg/TwistStamped on /cmd_vel, so the command publisher is stamped.

Pipeline:
  /map + /slam_region_graph/region_map + /scan + TF
    -> accumulate coverage by line-of-sight raycast
    -> choose active region
    -> choose next-best-view inside active region
    -> A* path to viewpoint
    -> pure-pursuit-like velocity control
    -> when region coverage is sufficient, switch to next region

It is meant for simulation exploration.  v11 uses committed goals, continuous pure-pursuit control, LiDAR arc
avoidance, conservative A* costs, dense coverage painting, moderate 0.3 m/s cruise,
mission-lock mode, front-blocked waypoint abandonment, front-only coverage, cross-region frontier seeking, dense waypoint path publishing, and always-search keep-moving behavior.  The A* planner treats walls as inflated
obstacles and prefers high-clearance cells instead of simply minimizing distance.
"""

from __future__ import annotations

import heapq
import json
import math
import time
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import rclpy
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy

from std_msgs.msg import String, ColorRGBA
from nav_msgs.msg import OccupancyGrid, Path
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Point, PoseStamped, Quaternion, TwistStamped
from visualization_msgs.msg import Marker, MarkerArray

import tf2_ros
from tf2_ros import TransformException

Cell = Tuple[int, int]


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def angle_wrap(a: float) -> float:
    return math.atan2(math.sin(a), math.cos(a))


def yaw_from_quaternion(q) -> float:
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


def quaternion_from_yaw(yaw: float) -> Quaternion:
    return Quaternion(x=0.0, y=0.0, z=math.sin(yaw * 0.5), w=math.cos(yaw * 0.5))


@dataclass
class GridGeom:
    width: int
    height: int
    resolution: float
    origin_x: float
    origin_y: float

    @classmethod
    def from_msg(cls, msg: OccupancyGrid) -> 'GridGeom':
        return cls(
            width=int(msg.info.width),
            height=int(msg.info.height),
            resolution=float(msg.info.resolution),
            origin_x=float(msg.info.origin.position.x),
            origin_y=float(msg.info.origin.position.y),
        )

    def same_geometry(self, other: Optional['GridGeom']) -> bool:
        return (
            other is not None
            and self.width == other.width
            and self.height == other.height
            and abs(self.resolution - other.resolution) < 1e-9
            and abs(self.origin_x - other.origin_x) < 1e-6
            and abs(self.origin_y - other.origin_y) < 1e-6
        )


@dataclass
class RobotPose:
    x: float
    y: float
    yaw: float


@dataclass
class RegionStats:
    label: int
    decoded_id: int
    cells: List[Cell]
    centroid: Tuple[float, float]
    total: int
    covered: int
    coverage_ratio: float
    frontier_count: int
    area_m2: float


@dataclass
class Candidate:
    cell: Cell
    x: float
    y: float
    yaw: float
    score: float
    visible_unknown: int
    visible_uncovered: int
    visible_frontier: int
    euclid_cost: float
    clearance_m: float


class RegionAutoMapperNode(Node):
    def __init__(self):
        super().__init__('region_auto_mapper')

        # Topics / frames
        self.declare_parameter('map_topic', '/map')
        self.declare_parameter('region_map_topic', '/slam_region_graph/region_map')
        self.declare_parameter('scan_topic', '/scan')
        self.declare_parameter('cmd_vel_topic', '/cmd_vel')
        self.declare_parameter('coverage_map_topic', '/region_auto_mapper/coverage_map')
        self.declare_parameter('state_topic', '/region_auto_mapper/state')
        self.declare_parameter('markers_topic', '/region_auto_mapper/markers')
        self.declare_parameter('path_topic', '/region_auto_mapper/path')
        self.declare_parameter('global_frame', 'map')
        self.declare_parameter('robot_frame', 'base_footprint')

        # Execution
        self.declare_parameter('auto_start', True)
        self.declare_parameter('timer_period_sec', 0.03)
        self.declare_parameter('planning_period_sec', 2.50)
        self.declare_parameter('control_period_sec', 0.03)
        self.declare_parameter('replan_if_goal_older_sec', 180.0)
        self.declare_parameter('goal_reached_radius_m', 0.22)
        self.declare_parameter('waypoint_reached_radius_m', 0.18)
        self.declare_parameter('path_lookahead_m', 1.55)
        self.declare_parameter('goal_lock_min_sec', 90.0)
        self.declare_parameter('goal_progress_epsilon_m', 0.025)
        self.declare_parameter('goal_progress_stall_sec', 25.0)
        self.declare_parameter('only_replan_when_reached_or_stalled', True)

        # Map interpretation
        self.declare_parameter('free_threshold', 68)
        self.declare_parameter('occupied_threshold', 72)
        self.declare_parameter('min_region_cells', 20)
        self.declare_parameter('a_star_unknown_allowed', False)
        self.declare_parameter('a_star_allow_outside_active_region', True)
        self.declare_parameter('a_star_max_expansions', 25000)

        # Conservative A* safety layer.  This is separate from the reactive
        # LiDAR stop/arc controller: A* should already choose paths with large
        # obstacle clearance, so the local controller is not forced to stop-go
        # near walls.  If the map is tight, start/goal cells are allowed but the
        # intermediate path is kept away from obstacles when possible.
        self.declare_parameter('conservative_astar', True)
        self.declare_parameter('path_min_clearance_m', 0.30)
        self.declare_parameter('path_prefer_clearance_m', 0.85)
        self.declare_parameter('path_wall_cost_weight', 18.0)
        self.declare_parameter('path_unknown_cost', 80.0)
        self.declare_parameter('path_region_boundary_cost', 8.0)
        self.declare_parameter('path_diagonal_cost_multiplier', 1.08)
        self.declare_parameter('path_clearance_cache_max', 80000)

        # Coverage raycast
        self.declare_parameter('coverage_use_lidar_scan', True)
        self.declare_parameter('coverage_max_range_m', 3.2)
        self.declare_parameter('coverage_ray_step_m', 0.04)
        self.declare_parameter('coverage_downsample_angle_step_deg', 1.0)
        self.declare_parameter('dense_coverage_marking', True)
        self.declare_parameter('coverage_brush_radius_m', 0.18)
        self.declare_parameter('coverage_robot_radius_m', 0.26)
        self.declare_parameter('coverage_stop_on_obstacle', True)
        # v12: coverage should represent what the robot deliberately inspected
        # in its forward sensor sector, not the full 360-degree LiDAR halo.
        # This makes region completion require actually facing/sweeping through
        # a room and prevents side/rear beams from instantly filling regions.
        self.declare_parameter('coverage_front_only', True)
        self.declare_parameter('coverage_fov_deg', 90.0)
        self.declare_parameter('coverage_yaw_offset_deg', 0.0)
        self.declare_parameter('coverage_mark_robot_footprint', False)

        # Candidate / utility
        self.declare_parameter('candidate_grid_step_m', 0.25)
        self.declare_parameter('candidate_min_clearance_m', 0.30)
        self.declare_parameter('candidate_max_count', 450)
        self.declare_parameter('candidate_top_k_for_astar', 30)
        self.declare_parameter('view_fov_deg', 100.0)
        self.declare_parameter('view_max_range_m', 3.2)
        self.declare_parameter('view_eval_angle_step_deg', 4.0)
        # Count unknown/frontier visible outside the current region as gain so
        # the mapper actively opens new regions instead of only polishing the
        # current label.
        self.declare_parameter('allow_cross_region_view_gain', True)
        self.declare_parameter('w_cross_region_unknown', 2.0)
        self.declare_parameter('w_cross_region_frontier', 1.5)
        # v13: when the live room/region graph is still under-segmented
        # (often only one large R1 while Cartographer has not yet closed walls),
        # do not restrict viewpoint candidates to the active region only.
        # Also sample safe free cells near unknown frontiers so the mapper
        # actively drives toward new rooms/corridors and creates new regions.
        self.declare_parameter('frontier_candidate_sampling', True)
        self.declare_parameter('frontier_candidate_max_count', 260)
        self.declare_parameter('frontier_candidate_min_unknown_neighbors', 1)
        # Keep intermediate waypoints in RViz/path following.  A fully sparsified
        # A* path often contains only start/end on long straight segments.
        self.declare_parameter('path_sparsify_max_step_m', 0.35)
        self.declare_parameter('w_unknown', 3.0)
        self.declare_parameter('w_unseen', 2.0)
        self.declare_parameter('w_frontier', 2.5)
        self.declare_parameter('w_path', 0.9)
        self.declare_parameter('w_clearance', 0.35)

        # Region completion / switching
        self.declare_parameter('region_coverage_threshold', 0.92)
        self.declare_parameter('region_frontier_threshold', 10)
        self.declare_parameter('soft_complete_if_no_gain', True)
        self.declare_parameter('min_goal_gain', 6.0)
        self.declare_parameter('completed_region_revisit_delay_sec', 25.0)
        self.declare_parameter('region_completion_min_active_sec', 18.0)
        self.declare_parameter('region_completion_min_cells', 160)
        self.declare_parameter('select_next_region_policy', 'nearest_uncovered')

        # Velocity control
        self.declare_parameter('max_linear_x', 0.30)
        self.declare_parameter('max_angular_z', 1.20)
        self.declare_parameter('linear_k', 1.80)
        self.declare_parameter('angular_k', 1.85)
        self.declare_parameter('heading_slowdown_angle_rad', 1.75)
        self.declare_parameter('heading_stop_angle_rad', 3.05)
        self.declare_parameter('front_stop_distance_m', 0.20)
        self.declare_parameter('front_slow_distance_m', 0.48)
        self.declare_parameter('side_stop_distance_m', 0.16)
        self.declare_parameter('recovery_rotate_speed', 0.45)
        self.declare_parameter('recovery_reverse_speed', -0.060)
        self.declare_parameter('recovery_reverse_time_sec', 0.35)
        self.declare_parameter('recovery_rotate_time_sec', 0.75)
        self.declare_parameter('spin_in_place_after_goal_sec', 0.0)
        # v16: never rotate in place just because there is no valid goal.
        # Spinning made the robot look busy while not expanding the map; stop and
        # force frontier replanning instead.
        self.declare_parameter('idle_spin_enabled', False)
        self.declare_parameter('reopen_completed_region_on_frontier', True)
        self.declare_parameter('reopen_frontier_margin', 1.15)

        # v17: exploration should never sit idle just because the high-level
        # planner temporarily cannot produce a valid region/frontier goal.
        # When no A* goal/path is available, use a very slow LiDAR-guided
        # search motion that keeps the robot scanning and exposing new map
        # frontiers.  This is intentionally slower than normal path following.
        self.declare_parameter('keep_moving_when_no_goal', True)
        self.declare_parameter('clear_completed_regions_when_no_goal', True)
        self.declare_parameter('search_motion_linear_x', 0.075)
        self.declare_parameter('search_motion_angular_z', 0.32)
        self.declare_parameter('search_motion_front_clearance_m', 0.34)
        self.declare_parameter('search_motion_side_balance_gain', 0.45)
        self.declare_parameter('search_motion_min_turn_z', 0.18)

        # Continuous-drive shaping.  The original v4 controller frequently
        # dropped to exactly zero at short waypoints or during replanning, which
        # made the robot move in a stop-go pattern.  These parameters keep a
        # nonzero crawl speed when the heading is not too bad and rate-limit the
        # final TwistStamped command instead of emitting discontinuous commands.
        self.declare_parameter('min_linear_x', 0.10)
        self.declare_parameter('creep_linear_x', 0.06)
        self.declare_parameter('linear_accel_limit', 1.20)      # m/s^2
        self.declare_parameter('angular_accel_limit', 3.20)    # rad/s^2
        self.declare_parameter('cmd_smoothing_alpha', 0.98)    # 1.0 = no low-pass
        self.declare_parameter('continuous_goal_handoff', True)
        self.declare_parameter('lidar_arc_avoidance', True)
        self.declare_parameter('arc_avoidance_gain', 1.25)
        self.declare_parameter('arc_avoidance_max_wz', 0.85)
        self.declare_parameter('arc_clearance_balance_gain', 0.55)
        self.declare_parameter('arc_min_linear_scale', 0.45)
        self.declare_parameter('front_arc_distance_m', 0.58)
        self.declare_parameter('preserve_goal_on_map_resize', True)
        self.declare_parameter('lock_goal_until_reached', True)
        self.declare_parameter('cruise_linear_x', 0.24)
        self.declare_parameter('cruise_heading_limit_rad', 1.10)
        self.declare_parameter('scan_sector_percentile', 0.18)
        self.declare_parameter('direct_goal_fallback', True)
        self.declare_parameter('mission_lock_skip_region_replan', True)
        self.declare_parameter('no_stop_on_missing_region_stats', True)
        self.declare_parameter('emergency_stop_distance_m', 0.10)
        self.declare_parameter('fast_cruise_when_locked', True)

        # v11 safety behavior: if the committed viewpoint becomes unsafe in
        # the live LiDAR front sector, abandon that waypoint, temporarily
        # blacklist nearby candidate cells, and force immediate replanning.
        # This prevents the robot from insisting on a goal that requires
        # squeezing into a wall/furniture edge.
        self.declare_parameter('waypoint_abandon_enabled', True)
        self.declare_parameter('waypoint_abandon_front_distance_m', 0.24)
        self.declare_parameter('waypoint_abandon_time_sec', 0.70)
        self.declare_parameter('waypoint_abandon_cooldown_sec', 2.0)
        self.declare_parameter('abandoned_goal_radius_m', 0.70)
        self.declare_parameter('abandoned_goal_memory_sec', 55.0)
        self.declare_parameter('abandoned_goal_max_count', 48)
        self.declare_parameter('abandon_turn_speed', 0.38)

        # Read params
        self.map_topic = str(self.get_parameter('map_topic').value)
        self.region_map_topic = str(self.get_parameter('region_map_topic').value)
        self.scan_topic = str(self.get_parameter('scan_topic').value)
        self.cmd_vel_topic = str(self.get_parameter('cmd_vel_topic').value)
        self.coverage_map_topic = str(self.get_parameter('coverage_map_topic').value)
        self.state_topic = str(self.get_parameter('state_topic').value)
        self.markers_topic = str(self.get_parameter('markers_topic').value)
        self.path_topic = str(self.get_parameter('path_topic').value)
        self.global_frame = str(self.get_parameter('global_frame').value)
        self.robot_frame = str(self.get_parameter('robot_frame').value)

        self.auto_start = bool(self.get_parameter('auto_start').value)
        self.timer_period_sec = float(self.get_parameter('timer_period_sec').value)
        self.planning_period_sec = float(self.get_parameter('planning_period_sec').value)
        self.control_period_sec = float(self.get_parameter('control_period_sec').value)
        self.replan_if_goal_older_sec = float(self.get_parameter('replan_if_goal_older_sec').value)
        self.goal_reached_radius_m = float(self.get_parameter('goal_reached_radius_m').value)
        self.waypoint_reached_radius_m = float(self.get_parameter('waypoint_reached_radius_m').value)
        self.path_lookahead_m = float(self.get_parameter('path_lookahead_m').value)
        self.goal_lock_min_sec = float(self.get_parameter('goal_lock_min_sec').value)
        self.goal_progress_epsilon_m = float(self.get_parameter('goal_progress_epsilon_m').value)
        self.goal_progress_stall_sec = float(self.get_parameter('goal_progress_stall_sec').value)
        self.only_replan_when_reached_or_stalled = bool(self.get_parameter('only_replan_when_reached_or_stalled').value)

        self.free_threshold = int(self.get_parameter('free_threshold').value)
        self.occupied_threshold = int(self.get_parameter('occupied_threshold').value)
        self.min_region_cells = int(self.get_parameter('min_region_cells').value)
        self.a_star_unknown_allowed = bool(self.get_parameter('a_star_unknown_allowed').value)
        self.a_star_allow_outside_active_region = bool(self.get_parameter('a_star_allow_outside_active_region').value)
        self.a_star_max_expansions = int(self.get_parameter('a_star_max_expansions').value)
        self.conservative_astar = bool(self.get_parameter('conservative_astar').value)
        self.path_min_clearance_m = float(self.get_parameter('path_min_clearance_m').value)
        self.path_prefer_clearance_m = float(self.get_parameter('path_prefer_clearance_m').value)
        self.path_wall_cost_weight = float(self.get_parameter('path_wall_cost_weight').value)
        self.path_unknown_cost = float(self.get_parameter('path_unknown_cost').value)
        self.path_region_boundary_cost = float(self.get_parameter('path_region_boundary_cost').value)
        self.path_diagonal_cost_multiplier = float(self.get_parameter('path_diagonal_cost_multiplier').value)
        self.path_clearance_cache_max = int(self.get_parameter('path_clearance_cache_max').value)

        self.coverage_use_lidar_scan = bool(self.get_parameter('coverage_use_lidar_scan').value)
        self.coverage_max_range_m = float(self.get_parameter('coverage_max_range_m').value)
        self.coverage_ray_step_m = float(self.get_parameter('coverage_ray_step_m').value)
        self.coverage_downsample_angle_step_deg = float(self.get_parameter('coverage_downsample_angle_step_deg').value)
        self.dense_coverage_marking = bool(self.get_parameter('dense_coverage_marking').value)
        self.coverage_brush_radius_m = float(self.get_parameter('coverage_brush_radius_m').value)
        self.coverage_robot_radius_m = float(self.get_parameter('coverage_robot_radius_m').value)
        self.coverage_stop_on_obstacle = bool(self.get_parameter('coverage_stop_on_obstacle').value)
        self.coverage_front_only = bool(self.get_parameter('coverage_front_only').value)
        self.coverage_fov_deg = float(self.get_parameter('coverage_fov_deg').value)
        self.coverage_yaw_offset_deg = float(self.get_parameter('coverage_yaw_offset_deg').value)
        self.coverage_mark_robot_footprint = bool(self.get_parameter('coverage_mark_robot_footprint').value)

        self.candidate_grid_step_m = float(self.get_parameter('candidate_grid_step_m').value)
        self.candidate_min_clearance_m = float(self.get_parameter('candidate_min_clearance_m').value)
        self.candidate_max_count = int(self.get_parameter('candidate_max_count').value)
        self.candidate_top_k_for_astar = int(self.get_parameter('candidate_top_k_for_astar').value)
        self.view_fov_deg = float(self.get_parameter('view_fov_deg').value)
        self.view_max_range_m = float(self.get_parameter('view_max_range_m').value)
        self.view_eval_angle_step_deg = float(self.get_parameter('view_eval_angle_step_deg').value)
        self.allow_cross_region_view_gain = bool(self.get_parameter('allow_cross_region_view_gain').value)
        self.w_cross_region_unknown = float(self.get_parameter('w_cross_region_unknown').value)
        self.w_cross_region_frontier = float(self.get_parameter('w_cross_region_frontier').value)
        self.frontier_candidate_sampling = bool(self.get_parameter('frontier_candidate_sampling').value)
        self.frontier_candidate_max_count = int(self.get_parameter('frontier_candidate_max_count').value)
        self.frontier_candidate_min_unknown_neighbors = int(self.get_parameter('frontier_candidate_min_unknown_neighbors').value)
        self.path_sparsify_max_step_m = float(self.get_parameter('path_sparsify_max_step_m').value)
        self.w_unknown = float(self.get_parameter('w_unknown').value)
        self.w_unseen = float(self.get_parameter('w_unseen').value)
        self.w_frontier = float(self.get_parameter('w_frontier').value)
        self.w_path = float(self.get_parameter('w_path').value)
        self.w_clearance = float(self.get_parameter('w_clearance').value)

        self.region_coverage_threshold = float(self.get_parameter('region_coverage_threshold').value)
        self.region_frontier_threshold = int(self.get_parameter('region_frontier_threshold').value)
        self.soft_complete_if_no_gain = bool(self.get_parameter('soft_complete_if_no_gain').value)
        self.min_goal_gain = float(self.get_parameter('min_goal_gain').value)
        self.completed_region_revisit_delay_sec = float(self.get_parameter('completed_region_revisit_delay_sec').value)
        self.region_completion_min_active_sec = float(self.get_parameter('region_completion_min_active_sec').value)
        self.region_completion_min_cells = int(self.get_parameter('region_completion_min_cells').value)
        self.select_next_region_policy = str(self.get_parameter('select_next_region_policy').value)

        self.max_linear_x = float(self.get_parameter('max_linear_x').value)
        self.max_angular_z = float(self.get_parameter('max_angular_z').value)
        self.linear_k = float(self.get_parameter('linear_k').value)
        self.angular_k = float(self.get_parameter('angular_k').value)
        self.heading_slowdown_angle_rad = float(self.get_parameter('heading_slowdown_angle_rad').value)
        self.heading_stop_angle_rad = float(self.get_parameter('heading_stop_angle_rad').value)
        self.front_stop_distance_m = float(self.get_parameter('front_stop_distance_m').value)
        self.front_slow_distance_m = float(self.get_parameter('front_slow_distance_m').value)
        self.side_stop_distance_m = float(self.get_parameter('side_stop_distance_m').value)
        self.recovery_rotate_speed = float(self.get_parameter('recovery_rotate_speed').value)
        self.recovery_reverse_speed = float(self.get_parameter('recovery_reverse_speed').value)
        self.recovery_reverse_time_sec = float(self.get_parameter('recovery_reverse_time_sec').value)
        self.recovery_rotate_time_sec = float(self.get_parameter('recovery_rotate_time_sec').value)
        self.spin_in_place_after_goal_sec = float(self.get_parameter('spin_in_place_after_goal_sec').value)
        self.idle_spin_enabled = bool(self.get_parameter('idle_spin_enabled').value)
        self.reopen_completed_region_on_frontier = bool(self.get_parameter('reopen_completed_region_on_frontier').value)
        self.reopen_frontier_margin = float(self.get_parameter('reopen_frontier_margin').value)
        self.keep_moving_when_no_goal = bool(self.get_parameter('keep_moving_when_no_goal').value)
        self.clear_completed_regions_when_no_goal = bool(self.get_parameter('clear_completed_regions_when_no_goal').value)
        self.search_motion_linear_x = float(self.get_parameter('search_motion_linear_x').value)
        self.search_motion_angular_z = float(self.get_parameter('search_motion_angular_z').value)
        self.search_motion_front_clearance_m = float(self.get_parameter('search_motion_front_clearance_m').value)
        self.search_motion_side_balance_gain = float(self.get_parameter('search_motion_side_balance_gain').value)
        self.search_motion_min_turn_z = float(self.get_parameter('search_motion_min_turn_z').value)
        self.min_linear_x = float(self.get_parameter('min_linear_x').value)
        self.creep_linear_x = float(self.get_parameter('creep_linear_x').value)
        self.linear_accel_limit = float(self.get_parameter('linear_accel_limit').value)
        self.angular_accel_limit = float(self.get_parameter('angular_accel_limit').value)
        self.cmd_smoothing_alpha = float(self.get_parameter('cmd_smoothing_alpha').value)
        self.continuous_goal_handoff = bool(self.get_parameter('continuous_goal_handoff').value)
        self.lidar_arc_avoidance = bool(self.get_parameter('lidar_arc_avoidance').value)
        self.arc_avoidance_gain = float(self.get_parameter('arc_avoidance_gain').value)
        self.arc_avoidance_max_wz = float(self.get_parameter('arc_avoidance_max_wz').value)
        self.arc_clearance_balance_gain = float(self.get_parameter('arc_clearance_balance_gain').value)
        self.arc_min_linear_scale = float(self.get_parameter('arc_min_linear_scale').value)
        self.front_arc_distance_m = float(self.get_parameter('front_arc_distance_m').value)
        self.preserve_goal_on_map_resize = bool(self.get_parameter('preserve_goal_on_map_resize').value)
        self.lock_goal_until_reached = bool(self.get_parameter('lock_goal_until_reached').value)
        self.cruise_linear_x = float(self.get_parameter('cruise_linear_x').value)
        self.cruise_heading_limit_rad = float(self.get_parameter('cruise_heading_limit_rad').value)
        self.scan_sector_percentile = float(self.get_parameter('scan_sector_percentile').value)
        self.direct_goal_fallback = bool(self.get_parameter('direct_goal_fallback').value)
        self.mission_lock_skip_region_replan = bool(self.get_parameter('mission_lock_skip_region_replan').value)
        self.no_stop_on_missing_region_stats = bool(self.get_parameter('no_stop_on_missing_region_stats').value)
        self.emergency_stop_distance_m = float(self.get_parameter('emergency_stop_distance_m').value)
        self.fast_cruise_when_locked = bool(self.get_parameter('fast_cruise_when_locked').value)
        self.waypoint_abandon_enabled = bool(self.get_parameter('waypoint_abandon_enabled').value)
        self.waypoint_abandon_front_distance_m = float(self.get_parameter('waypoint_abandon_front_distance_m').value)
        self.waypoint_abandon_time_sec = float(self.get_parameter('waypoint_abandon_time_sec').value)
        self.waypoint_abandon_cooldown_sec = float(self.get_parameter('waypoint_abandon_cooldown_sec').value)
        self.abandoned_goal_radius_m = float(self.get_parameter('abandoned_goal_radius_m').value)
        self.abandoned_goal_memory_sec = float(self.get_parameter('abandoned_goal_memory_sec').value)
        self.abandoned_goal_max_count = int(self.get_parameter('abandoned_goal_max_count').value)
        self.abandon_turn_speed = float(self.get_parameter('abandon_turn_speed').value)

        # QoS
        map_qos = QoSProfile(reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL, history=HistoryPolicy.KEEP_LAST, depth=1)
        scan_qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, durability=DurabilityPolicy.VOLATILE, history=HistoryPolicy.KEEP_LAST, depth=5)
        pub_qos = QoSProfile(reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL, history=HistoryPolicy.KEEP_LAST, depth=1)

        self.map_sub = self.create_subscription(OccupancyGrid, self.map_topic, self._on_map, map_qos)
        self.region_sub = self.create_subscription(OccupancyGrid, self.region_map_topic, self._on_region_map, map_qos)
        self.scan_sub = self.create_subscription(LaserScan, self.scan_topic, self._on_scan, scan_qos)

        # TurtleBot3 Jazzy/Gazebo command path expects geometry_msgs/msg/TwistStamped.
        # Publishing geometry_msgs/msg/Twist on the same topic name will not match the
        # simulator/controller subscriber, so the robot will not move.
        self.cmd_pub = self.create_publisher(TwistStamped, self.cmd_vel_topic, 10)
        self.coverage_pub = self.create_publisher(OccupancyGrid, self.coverage_map_topic, pub_qos)
        self.state_pub = self.create_publisher(String, self.state_topic, 10)
        self.marker_pub = self.create_publisher(MarkerArray, self.markers_topic, 10)
        self.path_pub = self.create_publisher(Path, self.path_topic, 10)

        self.tf_buffer = tf2_ros.Buffer(cache_time=Duration(seconds=10.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.map_msg: Optional[OccupancyGrid] = None
        self.region_msg: Optional[OccupancyGrid] = None
        self.scan_msg: Optional[LaserScan] = None
        self.geom: Optional[GridGeom] = None
        self.coverage: List[bool] = []

        self.active_region_label: Optional[int] = None
        self.active_region_set_time = time.time()
        self.completed_regions: Dict[int, float] = {}
        self.last_planning_time = 0.0
        self.last_control_time = 0.0
        self.current_goal: Optional[Candidate] = None
        self.current_path: List[Cell] = []
        self.path_index = 0
        self.goal_set_time = 0.0
        self.goal_best_distance = 1e18
        self.goal_last_progress_time = time.time()
        self.last_candidates: List[Candidate] = []
        self.last_stats: Dict[int, RegionStats] = {}
        self.last_state = 'INIT'
        self.recovery_mode: Optional[str] = None
        self.recovery_until = 0.0
        self.spin_until = 0.0
        self.tick = 0
        self.last_cmd_vx = 0.0
        self.last_cmd_wz = 0.0
        self.last_cmd_time = time.time()
        self._path_clearance_cache: Dict[Cell, float] = {}
        self.abandoned_goals: Dict[Cell, float] = {}
        self.front_blocked_since: Optional[float] = None
        self.last_abandon_time = 0.0
        self.last_abandon_reason = ''

        self.timer = self.create_timer(self.timer_period_sec, self._on_timer)
        self.get_logger().info(
            'REGION_AUTO_MAPPER_READY | '
            f'auto_start={self.auto_start} cmd_vel={self.cmd_vel_topic} domain must be set externally | '
            f'map={self.map_topic} region={self.region_map_topic}'
        )

    def _refresh_runtime_params(self):
        """Refresh motion parameters so ros2 param set takes effect live.

        Exploration parameters that affect data structures are still mostly read
        at startup, but the velocity / smoothness knobs are deliberately hot so
        the robot can be tuned without restarting Cartographer.
        """
        try:
            self.max_linear_x = float(self.get_parameter('max_linear_x').value)
            self.max_angular_z = float(self.get_parameter('max_angular_z').value)
            self.linear_k = float(self.get_parameter('linear_k').value)
            self.angular_k = float(self.get_parameter('angular_k').value)
            self.heading_slowdown_angle_rad = float(self.get_parameter('heading_slowdown_angle_rad').value)
            self.heading_stop_angle_rad = float(self.get_parameter('heading_stop_angle_rad').value)
            self.front_stop_distance_m = float(self.get_parameter('front_stop_distance_m').value)
            self.front_slow_distance_m = float(self.get_parameter('front_slow_distance_m').value)
            self.side_stop_distance_m = float(self.get_parameter('side_stop_distance_m').value)
            self.min_linear_x = float(self.get_parameter('min_linear_x').value)
            self.creep_linear_x = float(self.get_parameter('creep_linear_x').value)
            self.linear_accel_limit = float(self.get_parameter('linear_accel_limit').value)
            self.angular_accel_limit = float(self.get_parameter('angular_accel_limit').value)
            self.cmd_smoothing_alpha = float(self.get_parameter('cmd_smoothing_alpha').value)
            self.continuous_goal_handoff = bool(self.get_parameter('continuous_goal_handoff').value)
            self.goal_lock_min_sec = float(self.get_parameter('goal_lock_min_sec').value)
            self.goal_progress_epsilon_m = float(self.get_parameter('goal_progress_epsilon_m').value)
            self.goal_progress_stall_sec = float(self.get_parameter('goal_progress_stall_sec').value)
            self.only_replan_when_reached_or_stalled = bool(self.get_parameter('only_replan_when_reached_or_stalled').value)
            self.lidar_arc_avoidance = bool(self.get_parameter('lidar_arc_avoidance').value)
            self.arc_avoidance_gain = float(self.get_parameter('arc_avoidance_gain').value)
            self.arc_avoidance_max_wz = float(self.get_parameter('arc_avoidance_max_wz').value)
            self.arc_clearance_balance_gain = float(self.get_parameter('arc_clearance_balance_gain').value)
            self.arc_min_linear_scale = float(self.get_parameter('arc_min_linear_scale').value)
            self.front_arc_distance_m = float(self.get_parameter('front_arc_distance_m').value)
            self.preserve_goal_on_map_resize = bool(self.get_parameter('preserve_goal_on_map_resize').value)
            self.lock_goal_until_reached = bool(self.get_parameter('lock_goal_until_reached').value)
            self.cruise_linear_x = float(self.get_parameter('cruise_linear_x').value)
            self.cruise_heading_limit_rad = float(self.get_parameter('cruise_heading_limit_rad').value)
            self.scan_sector_percentile = float(self.get_parameter('scan_sector_percentile').value)
            self.direct_goal_fallback = bool(self.get_parameter('direct_goal_fallback').value)
            self.idle_spin_enabled = bool(self.get_parameter('idle_spin_enabled').value)
            self.reopen_completed_region_on_frontier = bool(self.get_parameter('reopen_completed_region_on_frontier').value)
            self.reopen_frontier_margin = float(self.get_parameter('reopen_frontier_margin').value)
            self.keep_moving_when_no_goal = bool(self.get_parameter('keep_moving_when_no_goal').value)
            self.clear_completed_regions_when_no_goal = bool(self.get_parameter('clear_completed_regions_when_no_goal').value)
            self.search_motion_linear_x = float(self.get_parameter('search_motion_linear_x').value)
            self.search_motion_angular_z = float(self.get_parameter('search_motion_angular_z').value)
            self.search_motion_front_clearance_m = float(self.get_parameter('search_motion_front_clearance_m').value)
            self.search_motion_side_balance_gain = float(self.get_parameter('search_motion_side_balance_gain').value)
            self.search_motion_min_turn_z = float(self.get_parameter('search_motion_min_turn_z').value)
            self.mission_lock_skip_region_replan = bool(self.get_parameter('mission_lock_skip_region_replan').value)
            self.no_stop_on_missing_region_stats = bool(self.get_parameter('no_stop_on_missing_region_stats').value)
            self.emergency_stop_distance_m = float(self.get_parameter('emergency_stop_distance_m').value)
            self.fast_cruise_when_locked = bool(self.get_parameter('fast_cruise_when_locked').value)
            self.waypoint_abandon_enabled = bool(self.get_parameter('waypoint_abandon_enabled').value)
            self.waypoint_abandon_front_distance_m = float(self.get_parameter('waypoint_abandon_front_distance_m').value)
            self.waypoint_abandon_time_sec = float(self.get_parameter('waypoint_abandon_time_sec').value)
            self.waypoint_abandon_cooldown_sec = float(self.get_parameter('waypoint_abandon_cooldown_sec').value)
            self.abandoned_goal_radius_m = float(self.get_parameter('abandoned_goal_radius_m').value)
            self.abandoned_goal_memory_sec = float(self.get_parameter('abandoned_goal_memory_sec').value)
            self.abandoned_goal_max_count = int(self.get_parameter('abandoned_goal_max_count').value)
            self.abandon_turn_speed = float(self.get_parameter('abandon_turn_speed').value)
            self.conservative_astar = bool(self.get_parameter('conservative_astar').value)
            self.path_min_clearance_m = float(self.get_parameter('path_min_clearance_m').value)
            self.path_prefer_clearance_m = float(self.get_parameter('path_prefer_clearance_m').value)
            self.path_wall_cost_weight = float(self.get_parameter('path_wall_cost_weight').value)
            self.path_unknown_cost = float(self.get_parameter('path_unknown_cost').value)
            self.path_region_boundary_cost = float(self.get_parameter('path_region_boundary_cost').value)
            self.path_diagonal_cost_multiplier = float(self.get_parameter('path_diagonal_cost_multiplier').value)
            self.region_coverage_threshold = float(self.get_parameter('region_coverage_threshold').value)
            self.region_completion_min_active_sec = float(self.get_parameter('region_completion_min_active_sec').value)
            self.region_completion_min_cells = int(self.get_parameter('region_completion_min_cells').value)
            self.dense_coverage_marking = bool(self.get_parameter('dense_coverage_marking').value)
            self.coverage_brush_radius_m = float(self.get_parameter('coverage_brush_radius_m').value)
            self.coverage_robot_radius_m = float(self.get_parameter('coverage_robot_radius_m').value)
            self.coverage_front_only = bool(self.get_parameter('coverage_front_only').value)
            self.coverage_fov_deg = float(self.get_parameter('coverage_fov_deg').value)
            self.coverage_yaw_offset_deg = float(self.get_parameter('coverage_yaw_offset_deg').value)
            self.coverage_mark_robot_footprint = bool(self.get_parameter('coverage_mark_robot_footprint').value)
            self.view_fov_deg = float(self.get_parameter('view_fov_deg').value)
            self.allow_cross_region_view_gain = bool(self.get_parameter('allow_cross_region_view_gain').value)
            self.w_cross_region_unknown = float(self.get_parameter('w_cross_region_unknown').value)
            self.w_cross_region_frontier = float(self.get_parameter('w_cross_region_frontier').value)
            self.frontier_candidate_sampling = bool(self.get_parameter('frontier_candidate_sampling').value)
            self.frontier_candidate_max_count = int(self.get_parameter('frontier_candidate_max_count').value)
            self.frontier_candidate_min_unknown_neighbors = int(self.get_parameter('frontier_candidate_min_unknown_neighbors').value)
            self.path_sparsify_max_step_m = float(self.get_parameter('path_sparsify_max_step_m').value)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------
    def _on_map(self, msg: OccupancyGrid):
        new_geom = GridGeom.from_msg(msg)
        if not new_geom.same_geometry(self.geom):
            old_geom = self.geom
            old_cov = self.coverage
            self.geom = new_geom
            self.coverage = [False] * (new_geom.width * new_geom.height)
            if old_geom is not None and old_cov:
                self._copy_coverage_overlap(old_geom, old_cov, new_geom, self.coverage)
            if self.preserve_goal_on_map_resize and self.current_goal is not None:
                # SLAM OccupancyGrid can grow/shift its origin while the robot is moving.
                # Do not drop the selected viewpoint; only invalidate the cell path and
                # replan to the same world-space goal on the new grid.
                self.current_path = []
                self.path_index = 0
                self.last_planning_time = 0.0
            else:
                self.current_path = []
                self.path_index = 0
                self.current_goal = None
            self.get_logger().info(
                f'AUTO_MAPPER_GRID_RESET | size={new_geom.width}x{new_geom.height} res={new_geom.resolution:.3f} '
                f'origin=({new_geom.origin_x:.2f},{new_geom.origin_y:.2f})'
            )
        self.map_msg = msg
        # Occupancy probabilities change even when map geometry does not, so
        # cached wall clearances must not survive across map updates.
        self._path_clearance_cache.clear()

    def _on_region_map(self, msg: OccupancyGrid):
        self.region_msg = msg

    def _on_scan(self, msg: LaserScan):
        self.scan_msg = msg

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------
    def _on_timer(self):
        self.tick += 1
        self._refresh_runtime_params()
        robot = self._lookup_robot_pose()
        ready = self.map_msg is not None and self.region_msg is not None and self.geom is not None and robot is not None
        if not ready:
            self._publish_state({'state': 'WAIT_INPUT', 'has_map': self.map_msg is not None, 'has_region_map': self.region_msg is not None, 'has_tf': robot is not None})
            self._publish_stop()
            return

        assert robot is not None
        self._update_coverage(robot)
        self._publish_coverage_map()

        now = time.time()
        if now - self.last_planning_time >= self.planning_period_sec:
            self.last_planning_time = now
            self._planning_step(robot)

        if self.auto_start and now - self.last_control_time >= self.control_period_sec:
            self.last_control_time = now
            self._control_step(robot)
        elif not self.auto_start:
            self._publish_stop()

        if self.tick % 5 == 0:
            self._publish_markers(robot)
            self._publish_path()

    # ------------------------------------------------------------------
    # Grid helpers
    # ------------------------------------------------------------------
    def _idx(self, c: Cell) -> int:
        assert self.geom is not None
        return c[1] * self.geom.width + c[0]

    def _in_bounds(self, x: int, y: int) -> bool:
        assert self.geom is not None
        return 0 <= x < self.geom.width and 0 <= y < self.geom.height

    def _world_to_cell(self, wx: float, wy: float) -> Optional[Cell]:
        assert self.geom is not None
        x = int(math.floor((wx - self.geom.origin_x) / self.geom.resolution))
        y = int(math.floor((wy - self.geom.origin_y) / self.geom.resolution))
        if not self._in_bounds(x, y):
            return None
        return (x, y)

    def _cell_to_world(self, c: Cell) -> Tuple[float, float]:
        assert self.geom is not None
        return (
            self.geom.origin_x + (c[0] + 0.5) * self.geom.resolution,
            self.geom.origin_y + (c[1] + 0.5) * self.geom.resolution,
        )

    def _neighbors4(self, c: Cell) -> Iterable[Cell]:
        x, y = c
        for nx, ny in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
            if self._in_bounds(nx, ny):
                yield (nx, ny)

    def _neighbors8(self, c: Cell) -> Iterable[Tuple[Cell, float]]:
        x, y = c
        for dx, dy, cost in ((1,0,1.0),(-1,0,1.0),(0,1,1.0),(0,-1,1.0),(1,1,1.414),(-1,1,1.414),(1,-1,1.414),(-1,-1,1.414)):
            nx, ny = x + dx, y + dy
            if self._in_bounds(nx, ny):
                yield (nx, ny), cost

    def _copy_coverage_overlap(self, old_geom: GridGeom, old_cov: List[bool], new_geom: GridGeom, new_cov: List[bool]):
        for oy in range(old_geom.height):
            for ox in range(old_geom.width):
                oi = oy * old_geom.width + ox
                if not old_cov[oi]:
                    continue
                wx = old_geom.origin_x + (ox + 0.5) * old_geom.resolution
                wy = old_geom.origin_y + (oy + 0.5) * old_geom.resolution
                nx = int(math.floor((wx - new_geom.origin_x) / new_geom.resolution))
                ny = int(math.floor((wy - new_geom.origin_y) / new_geom.resolution))
                if 0 <= nx < new_geom.width and 0 <= ny < new_geom.height:
                    new_cov[ny * new_geom.width + nx] = True

    def _lookup_robot_pose(self) -> Optional[RobotPose]:
        try:
            tf = self.tf_buffer.lookup_transform(self.global_frame, self.robot_frame, rclpy.time.Time(), timeout=Duration(seconds=0.03))
        except TransformException:
            return None
        t = tf.transform.translation
        q = tf.transform.rotation
        return RobotPose(float(t.x), float(t.y), yaw_from_quaternion(q))

    # ------------------------------------------------------------------
    # Map / region interpretation
    # ------------------------------------------------------------------
    def _map_value(self, c: Cell) -> int:
        assert self.map_msg is not None
        return int(self.map_msg.data[self._idx(c)])

    def _is_obstacle(self, c: Cell) -> bool:
        return self._map_value(c) >= self.occupied_threshold

    def _is_free_candidate(self, c: Cell) -> bool:
        v = self._map_value(c)
        if v < 0:
            return bool(self.a_star_unknown_allowed)
        return v < self.occupied_threshold

    def _region_label_value(self, c: Cell) -> int:
        if self.region_msg is None:
            return -1
        i = self._idx(c)
        if i < 0 or i >= len(self.region_msg.data):
            return -1
        return int(self.region_msg.data[i])

    def _decode_region_id(self, label: int) -> int:
        if label <= 0:
            return -1
        rid = ((label - 1) * 53) % 98
        return 98 if rid == 0 else rid

    def _region_at_pose(self, robot: RobotPose) -> Optional[int]:
        c = self._world_to_cell(robot.x, robot.y)
        if c is None:
            return None
        label = self._region_label_value(c)
        if label > 0:
            return label
        best = None
        best_d2 = 10**9
        cx, cy = c
        for r in range(1, 7):
            for dy in range(-r, r + 1):
                for dx in range(-r, r + 1):
                    cc = (cx + dx, cy + dy)
                    if not self._in_bounds(cc[0], cc[1]):
                        continue
                    lab = self._region_label_value(cc)
                    if lab <= 0:
                        continue
                    d2 = dx * dx + dy * dy
                    if d2 < best_d2:
                        best = lab
                        best_d2 = d2
            if best is not None:
                return best
        return None

    # ------------------------------------------------------------------
    # Coverage
    # ------------------------------------------------------------------
    def _update_coverage(self, robot: RobotPose):
        if self.map_msg is None or self.geom is None:
            return
        rc = self._world_to_cell(robot.x, robot.y)
        if rc is not None and self.coverage_mark_robot_footprint:
            self._mark_coverage_disk(rc, self.coverage, self.coverage_robot_radius_m)
        if self.coverage_use_lidar_scan and self.scan_msg is not None:
            self._update_coverage_from_scan(robot, self.scan_msg)
        else:
            if self.coverage_front_only:
                half = math.radians(max(1.0, self.coverage_fov_deg)) * 0.5
                start, stop = -half, half
            else:
                start, stop = -math.pi, math.pi
            offset = math.radians(self.coverage_yaw_offset_deg)
            a = start
            step = math.radians(max(1.0, self.coverage_downsample_angle_step_deg))
            while a <= stop + 1e-6:
                self._mark_ray(robot.x, robot.y, robot.yaw + offset + a, self.coverage_max_range_m, self.coverage)
                a += step

    def _update_coverage_from_scan(self, robot: RobotPose, scan: LaserScan):
        stride = max(1, int(math.ceil(math.radians(max(0.5, self.coverage_downsample_angle_step_deg)) / max(abs(scan.angle_increment), 1e-6))))
        half_fov = math.radians(max(1.0, self.coverage_fov_deg)) * 0.5
        offset = math.radians(self.coverage_yaw_offset_deg)
        for i in range(0, len(scan.ranges), stride):
            rel_a = float(scan.angle_min) + i * float(scan.angle_increment)
            # Normalize against the desired forward sector.  TurtleBot3 scan is
            # assumed to be aligned to base_footprint, so rel_a=0 is front.
            if self.coverage_front_only and abs(angle_wrap(rel_a - offset)) > half_fov:
                continue
            r = float(scan.ranges[i])
            if math.isnan(r) or math.isinf(r) or r <= max(0.0, scan.range_min):
                r = self.coverage_max_range_m
            else:
                r = min(r, self.coverage_max_range_m, float(scan.range_max) if scan.range_max > 0 else self.coverage_max_range_m)
            a = robot.yaw + rel_a
            self._mark_ray(robot.x, robot.y, a, r, self.coverage)

    def _mark_ray(self, x0: float, y0: float, yaw: float, max_range: float, target: List[bool]) -> List[Cell]:
        assert self.geom is not None
        out: List[Cell] = []
        step = max(self.coverage_ray_step_m, self.geom.resolution * 0.5)
        n = max(1, int(max_range / step))
        ca, sa = math.cos(yaw), math.sin(yaw)
        last = None
        for k in range(n + 1):
            r = k * step
            c = self._world_to_cell(x0 + r * ca, y0 + r * sa)
            if c is None:
                break
            if c == last:
                continue
            last = c
            if self.dense_coverage_marking and target is self.coverage:
                self._mark_coverage_disk(c, target, self.coverage_brush_radius_m)
            else:
                target[self._idx(c)] = True
            out.append(c)
            if self.coverage_stop_on_obstacle and self._is_obstacle(c):
                break
        return out

    def _mark_coverage_disk(self, center: Cell, target: List[bool], radius_m: float):
        if self.geom is None or not target:
            return
        r_cells = max(0, int(math.ceil(max(0.0, radius_m) / max(1e-9, self.geom.resolution))))
        cx, cy = center
        rr = r_cells * r_cells
        for dy in range(-r_cells, r_cells + 1):
            for dx in range(-r_cells, r_cells + 1):
                if dx * dx + dy * dy > rr:
                    continue
                cc = (cx + dx, cy + dy)
                if not self._in_bounds(cc[0], cc[1]):
                    continue
                # Do not paint walls as covered free-space.  This keeps dense
                # coverage from bleeding through thin walls while still filling
                # the visible room interior between sparse scan rays.
                if self._is_obstacle(cc):
                    continue
                target[self._idx(cc)] = True

    def _publish_coverage_map(self):
        if self.map_msg is None or self.geom is None or not self.coverage:
            return
        msg = OccupancyGrid()
        msg.header = self.map_msg.header
        msg.header.frame_id = self.global_frame
        msg.info = self.map_msg.info
        data: List[int] = []
        for i, cov in enumerate(self.coverage):
            mv = int(self.map_msg.data[i]) if i < len(self.map_msg.data) else -1
            if cov:
                data.append(100)
            elif mv < 0:
                data.append(-1)
            else:
                data.append(0)
        msg.data = data
        self.coverage_pub.publish(msg)

    # ------------------------------------------------------------------
    # Region stats
    # ------------------------------------------------------------------
    def _compute_region_cells(self) -> Dict[int, List[Cell]]:
        if self.geom is None or self.region_msg is None:
            return {}
        out: Dict[int, List[Cell]] = {}
        for y in range(self.geom.height):
            for x in range(self.geom.width):
                c = (x, y)
                label = self._region_label_value(c)
                if label <= 0:
                    continue
                if not self._is_free_candidate(c):
                    continue
                out.setdefault(label, []).append(c)
        return {k: v for k, v in out.items() if len(v) >= self.min_region_cells}

    def _frontier_count_for_region(self, cells: Sequence[Cell]) -> int:
        cell_set = set(cells)
        cnt = 0
        for c in cells:
            for nb in self._neighbors4(c):
                if nb not in cell_set and self._map_value(nb) < 0:
                    cnt += 1
                    break
        return cnt

    def _compute_region_stats(self) -> Dict[int, RegionStats]:
        if self.geom is None:
            return {}
        regions = self._compute_region_cells()
        stats: Dict[int, RegionStats] = {}
        for label, cells in regions.items():
            total = len(cells)
            covered = sum(1 for c in cells if self.coverage and self.coverage[self._idx(c)])
            cx = cy = 0.0
            for c in cells:
                wx, wy = self._cell_to_world(c)
                cx += wx
                cy += wy
            cx /= max(1, total)
            cy /= max(1, total)
            stats[label] = RegionStats(
                label=label,
                decoded_id=self._decode_region_id(label),
                cells=list(cells),
                centroid=(cx, cy),
                total=total,
                covered=covered,
                coverage_ratio=covered / max(1, total),
                frontier_count=self._frontier_count_for_region(cells),
                area_m2=total * (self.geom.resolution ** 2),
            )
        return stats

    def _region_complete(self, st: RegionStats) -> bool:
        # Do not mark a region complete immediately after it appears/relabels.
        # With online SLAM the region graph can split a room into temporary small
        # labels; if we accept cov=1.0 on a tiny label, the robot switches tasks
        # before it has actually swept the room.
        if st.total < max(1, self.region_completion_min_cells):
            return False
        if st.label == self.active_region_label:
            active_age = time.time() - self.active_region_set_time
            if active_age < self.region_completion_min_active_sec:
                return False
        return st.coverage_ratio >= self.region_coverage_threshold and st.frontier_count <= self.region_frontier_threshold


    def _set_active_region_label(self, label: Optional[int]):
        if label != self.active_region_label:
            self.active_region_label = label
            self.active_region_set_time = time.time()
            # New region means a new committed coverage task.  Keep motion state
            # clean so the robot does not chase an old viewpoint from another
            # label after a deliberate region switch.
            self.current_goal = None
            self.current_path = []
            self.path_index = 0

    # ------------------------------------------------------------------
    # Planning
    # ------------------------------------------------------------------
    def _planning_step(self, robot: RobotPose):
        now = time.time()

        # v10 mission lock: while a viewpoint is already selected and the robot
        # is still making progress, do not even look at the current region map.
        # The region layer is allowed to be stale/frozen while moving; clearing
        # the goal because of a transient region-map mismatch is what produced
        # the short stop-go cycle.
        if self.mission_lock_skip_region_replan and self.lock_goal_until_reached and self.current_goal is not None:
            goal_dist = math.hypot(self.current_goal.x - robot.x, self.current_goal.y - robot.y)
            stalled = (now - self.goal_last_progress_time) > self.goal_progress_stall_sec
            stale = (now - self.goal_set_time) > self.replan_if_goal_older_sec
            if goal_dist > self.goal_reached_radius_m and not stalled and not stale:
                if not self.current_path:
                    self.current_path = self._repath_to_current_goal(robot)
                    self.path_index = 0
                self._publish_state({
                    'state': 'FOLLOWING_MISSION_LOCKED_GOAL',
                    'active_region_label': self.active_region_label,
                    'goal_x': round(self.current_goal.x, 3),
                    'goal_y': round(self.current_goal.y, 3),
                    'goal_dist': round(goal_dist, 3),
                    'path_cells': len(self.current_path),
                    'last_cmd_vx': round(self.last_cmd_vx, 3),
                    'last_cmd_wz': round(self.last_cmd_wz, 3),
                    'mission_lock_skip_region_replan': True,
                })
                return

        stats = self._compute_region_stats()
        self.last_stats = stats
        robot_region = self._region_at_pose(robot)

        if not stats:
            if self.no_stop_on_missing_region_stats and self.current_goal is not None:
                self._publish_state({
                    'state': 'KEEP_GOAL_WAIT_REGION_STATS',
                    'region_count': 0,
                    'goal_x': round(self.current_goal.x, 3),
                    'goal_y': round(self.current_goal.y, 3),
                    'path_cells': len(self.current_path),
                })
                return
            self.current_goal = None
            self.current_path = []
            self._publish_state({'state': 'WAIT_REGION_STATS', 'region_count': 0})
            return

        # Remove old completion marks if the region expanded, lost coverage,
        # or regained frontiers.  Online SLAM can mark a narrow slice as DONE
        # and then reveal a long frontier behind it; if DONE is kept, the robot
        # can sit/spin near that boundary instead of reopening exploration.
        for label in list(self.completed_regions.keys()):
            st = stats.get(label)
            if st is None:
                continue
            lost_coverage = st.coverage_ratio < max(0.55, self.region_coverage_threshold - 0.20)
            reopened_frontier = (
                self.reopen_completed_region_on_frontier
                and st.frontier_count > max(1, int(math.ceil(self.region_frontier_threshold * self.reopen_frontier_margin)))
            )
            if lost_coverage or reopened_frontier:
                del self.completed_regions[label]

        if self.active_region_label is None:
            self._set_active_region_label(robot_region if robot_region in stats else self._select_nearest_region(robot, stats))

        active = stats.get(self.active_region_label) if self.active_region_label is not None else None
        if active is None:
            self._set_active_region_label(robot_region if robot_region in stats else self._select_nearest_region(robot, stats))
            active = stats.get(self.active_region_label) if self.active_region_label is not None else None
        if active is None:
            self._publish_state({'state': 'NO_ACTIVE_REGION'})
            return

        # Hard sticky-goal mode: once a candidate viewpoint is selected, keep
        # driving to that exact world-space point.  Region-map relabeling,
        # SLAM grid resizing, and small coverage changes must not replace the
        # goal before it is reached or truly stalled.
        if self.lock_goal_until_reached and self.current_goal is not None:
            goal_dist = math.hypot(self.current_goal.x - robot.x, self.current_goal.y - robot.y)
            stalled = (now - self.goal_last_progress_time) > self.goal_progress_stall_sec
            stale = (now - self.goal_set_time) > self.replan_if_goal_older_sec
            if goal_dist > self.goal_reached_radius_m and not stalled and not stale:
                if not self.current_path:
                    self.current_path = self._repath_to_current_goal(robot)
                    self.path_index = 0
                self._publish_state(self._state_payload('FOLLOWING_STICKY_GOAL', robot, active, self.current_goal, stats))
                return

        if self._region_complete(active):
            self.completed_regions[active.label] = now
            nxt = self._select_next_region(robot, active, stats)
            if nxt is not None:
                self._set_active_region_label(nxt.label)
                active = nxt
            else:
                # v17: do not end the mission just because all current labels
                # look complete.  Online SLAM often needs the robot to keep
                # moving/scanning before a new frontier or room label appears.
                # Clear the completion cache and keep searching from the same
                # active region instead of entering a stopped terminal state.
                if self.clear_completed_regions_when_no_goal:
                    self.completed_regions.clear()
                self.last_state = 'ALL_KNOWN_REGIONS_COMPLETE_KEEP_SEARCHING'
                self._publish_state(self._state_payload(self.last_state, robot, active, None, stats))

        # Do not continuously replace goals.  The mapper should commit to a
        # selected viewpoint until it is reached, clearly stale, or the robot has
        # stopped making progress.  Without this hysteresis the region graph/map
        # updates can cause a new best viewpoint every few seconds, producing
        # the observed "go-stop-new-goal" behavior.
        if self.current_goal is not None and self.current_path:
            goal_age = now - self.goal_set_time
            goal_dist = math.hypot(self.current_goal.x - robot.x, self.current_goal.y - robot.y)
            stalled = (now - self.goal_last_progress_time) > self.goal_progress_stall_sec
            locked = goal_age < self.goal_lock_min_sec
            stale = goal_age > self.replan_if_goal_older_sec
            if self.only_replan_when_reached_or_stalled:
                if locked or (not stale and not stalled):
                    self._publish_state(self._state_payload('FOLLOWING_LOCKED_GOAL', robot, active, self.current_goal, stats))
                    return
            elif not stale:
                self._publish_state(self._state_payload('FOLLOWING_EXISTING_GOAL', robot, active, self.current_goal, stats))
                return

        best, path = self._find_best_candidate_with_path(robot, active)
        if best is None or not path:
            if self.keep_moving_when_no_goal:
                # Do not mark the active label complete and stop.  A transient
                # lack of safe A* candidates is common while Cartographer is
                # still exposing the next room.  Clear completed marks so newly
                # revealed frontiers can be considered immediately, then let the
                # control loop perform slow LiDAR-guided search motion.
                if self.clear_completed_regions_when_no_goal:
                    self.completed_regions.clear()
                self.current_goal = None
                self.current_path = []
                self.path_index = 0
                self.last_planning_time = 0.0
                self.last_state = 'NO_VALID_GOAL_KEEP_MOVING_SEARCH'
                self._publish_state(self._state_payload(self.last_state, robot, active, None, stats))
                return
            if self.soft_complete_if_no_gain:
                self.completed_regions[active.label] = now
                nxt = self._select_next_region(robot, active, stats)
                if nxt is not None:
                    self._set_active_region_label(nxt.label)
            self.current_goal = None
            self.current_path = []
            self.path_index = 0
            self.last_state = 'NO_VALID_GOAL_RECOVERY'
            self._publish_state(self._state_payload(self.last_state, robot, active, None, stats))
            return

        self.current_goal = best
        self.current_path = path
        self.path_index = 0
        self.goal_set_time = now
        self.goal_best_distance = math.hypot(best.x - robot.x, best.y - robot.y)
        self.goal_last_progress_time = now
        self.last_state = 'NEW_REGION_GOAL'
        self.get_logger().info(
            f'NEW_AUTO_GOAL | active=R{active.decoded_id} label={active.label} '
            f'x={best.x:.2f} y={best.y:.2f} score={best.score:.1f} '
            f'unk={best.visible_unknown} unseen={best.visible_uncovered} frontier={best.visible_frontier} path_cells={len(path)}'
        )
        self._publish_state(self._state_payload(self.last_state, robot, active, best, stats))

    def _select_nearest_region(self, robot: RobotPose, stats: Dict[int, RegionStats]) -> Optional[int]:
        best = None
        best_d = 1e18
        for label, st in stats.items():
            d = math.hypot(st.centroid[0] - robot.x, st.centroid[1] - robot.y)
            if d < best_d:
                best = label
                best_d = d
        return best

    def _select_next_region(self, robot: RobotPose, current: RegionStats, stats: Dict[int, RegionStats]) -> Optional[RegionStats]:
        now = time.time()
        candidates: List[RegionStats] = []
        for st in stats.values():
            if st.label == current.label:
                continue
            done_t = self.completed_regions.get(st.label)
            if done_t is not None and now - done_t < self.completed_region_revisit_delay_sec:
                continue
            if st.coverage_ratio < self.region_coverage_threshold:
                candidates.append(st)
        if not candidates:
            return None
        if self.select_next_region_policy == 'largest_uncovered':
            candidates.sort(key=lambda s: (s.total - s.covered, -math.hypot(s.centroid[0] - robot.x, s.centroid[1] - robot.y)), reverse=True)
            return candidates[0]
        candidates.sort(key=lambda s: math.hypot(s.centroid[0] - robot.x, s.centroid[1] - robot.y) - 0.5 * (1.0 - s.coverage_ratio))
        return candidates[0]

    def _repath_to_current_goal(self, robot: RobotPose) -> List[Cell]:
        if self.current_goal is None or self.geom is None:
            return []
        start = self._world_to_cell(robot.x, robot.y)
        goal = self._world_to_cell(self.current_goal.x, self.current_goal.y)
        if start is None or goal is None:
            return []
        # Update candidate cell to the current grid geometry; keep the same
        # world-space x/y and score.  We allow path outside the active region
        # because the region map can relabel during SLAM expansion.
        self.current_goal.cell = goal
        path = self._astar(start, goal, active_label=None)
        if path:
            return path
        # If the exact goal cell is newly considered blocked by a noisy map
        # update, try a small nearby free cell so the robot still drives toward
        # the locked viewpoint instead of selecting a new goal every tick.
        gx, gy = goal
        best = None
        best_d2 = 10**9
        for r in range(1, 9):
            for dy in range(-r, r + 1):
                for dx in range(-r, r + 1):
                    c = (gx + dx, gy + dy)
                    if not self._in_bounds(c[0], c[1]) or not self._is_traversable(c, None):
                        continue
                    p = self._astar(start, c, active_label=None)
                    if p:
                        d2 = dx * dx + dy * dy
                        if d2 < best_d2:
                            best = (c, p)
                            best_d2 = d2
            if best is not None:
                self.current_goal.cell = best[0]
                return best[1]
        return []

    def _prune_abandoned_goals(self):
        """Drop old abandoned waypoints from the temporary blacklist."""
        if not self.abandoned_goals:
            return
        now = time.time()
        for c, t in list(self.abandoned_goals.items()):
            if now - t > self.abandoned_goal_memory_sec:
                del self.abandoned_goals[c]
        if len(self.abandoned_goals) > max(1, self.abandoned_goal_max_count):
            oldest = sorted(self.abandoned_goals.items(), key=lambda kv: kv[1])
            for c, _ in oldest[:len(self.abandoned_goals) - self.abandoned_goal_max_count]:
                self.abandoned_goals.pop(c, None)

    def _is_near_abandoned_goal(self, c: Cell) -> bool:
        if not self.abandoned_goals or self.geom is None:
            return False
        self._prune_abandoned_goals()
        r_cells = max(1, int(math.ceil(self.abandoned_goal_radius_m / max(1e-9, self.geom.resolution))))
        rr = r_cells * r_cells
        cx, cy = c
        for ac in self.abandoned_goals.keys():
            dx = cx - ac[0]
            dy = cy - ac[1]
            if dx * dx + dy * dy <= rr:
                return True
        return False

    def _abandon_current_goal(self, robot: RobotPose, reason: str):
        """Give up the current waypoint and force a different replanning target.

        This is deliberately separate from emergency reverse.  Emergency reverse
        is only for immediate collision.  Abandoning a waypoint is used when the
        front LiDAR has been persistently blocked while the planner still wants
        to reach the same committed viewpoint.  The rejected goal cell is
        temporarily blacklisted so the next planning tick does not choose it
        again.
        """
        now = time.time()
        if self.current_goal is not None:
            c = self._world_to_cell(self.current_goal.x, self.current_goal.y)
            if c is not None:
                self.abandoned_goals[c] = now
        self._prune_abandoned_goals()
        self.current_goal = None
        self.current_path = []
        self.path_index = 0
        self.front_blocked_since = None
        self.last_abandon_time = now
        self.last_abandon_reason = reason
        # Force immediate planning on the next timer tick instead of waiting
        # planning_period_sec.  This is what makes "give up and go elsewhere"
        # feel responsive.
        self.last_planning_time = 0.0
        self.goal_set_time = 0.0
        self.goal_best_distance = 1e18
        self.goal_last_progress_time = now
        self.get_logger().warn(
            f'ABANDON_CURRENT_GOAL | reason={reason} front_blocked | '
            f'abandoned_count={len(self.abandoned_goals)} robot=({robot.x:.2f},{robot.y:.2f})'
        )

    def _find_best_candidate_with_path(self, robot: RobotPose, active: RegionStats) -> Tuple[Optional[Candidate], List[Cell]]:
        candidates = self._sample_candidates(robot, active)
        self.last_candidates = candidates[:120]
        if not candidates:
            return None, []
        start = self._world_to_cell(robot.x, robot.y)
        if start is None:
            return None, []
        best: Optional[Candidate] = None
        best_path: List[Cell] = []
        for cand in candidates[:max(1, self.candidate_top_k_for_astar)]:
            path = self._astar(start, cand.cell, active_label=active.label if not self.a_star_allow_outside_active_region else None)
            if not path:
                continue
            path_cost_m = len(path) * (self.geom.resolution if self.geom else 0.05)
            score = cand.score - self.w_path * path_cost_m
            cand2 = Candidate(cand.cell, cand.x, cand.y, cand.yaw, score, cand.visible_unknown, cand.visible_uncovered, cand.visible_frontier, path_cost_m, cand.clearance_m)
            if best is None or cand2.score > best.score:
                best = cand2
                best_path = path
        if best is not None and best.score < self.min_goal_gain and self.soft_complete_if_no_gain:
            return None, []
        return best, best_path

    def _sample_candidates(self, robot: RobotPose, region: RegionStats) -> List[Candidate]:
        assert self.geom is not None
        step_cells = max(1, int(round(self.candidate_grid_step_m / self.geom.resolution)))
        sampled: List[Cell] = []
        seen: Set[Cell] = set()
        region_set = set(region.cells)

        def add_if_valid(c: Cell) -> bool:
            if c in seen:
                return False
            if c[0] % step_cells != 0 or c[1] % step_cells != 0:
                return False
            if not self._is_free_candidate(c):
                return False
            clearance = self._approx_clearance(c, max_radius_m=0.65)
            if clearance < self.candidate_min_clearance_m:
                return False
            if self._is_near_abandoned_goal(c):
                return False
            seen.add(c)
            sampled.append(c)
            return True

        # 1) Normal region-local candidates: finish the active room/zone.
        for c in region.cells:
            add_if_valid(c)

        # 2) v13 frontier-sector candidates: while the region graph is still
        # under-segmented or a doorway/corridor has just appeared, sample safe
        # free cells near unknown boundaries globally, not only inside R_i.
        # A* reachability still filters the final top candidates, so this does
        # not make the robot teleport through walls; it simply gives the planner
        # useful targets that can create the next region.
        if self.frontier_candidate_sampling and self.map_msg is not None:
            extra: List[Cell] = []
            for y in range(0, self.geom.height, step_cells):
                for x in range(0, self.geom.width, step_cells):
                    c = (x, y)
                    if c in seen:
                        continue
                    if not self._is_free_candidate(c):
                        continue
                    unk_n = self._unknown_neighbor_count(c)
                    if unk_n < self.frontier_candidate_min_unknown_neighbors:
                        continue
                    if self._approx_clearance(c, max_radius_m=0.65) < self.candidate_min_clearance_m:
                        continue
                    if self._is_near_abandoned_goal(c):
                        continue
                    extra.append(c)
            if len(extra) > self.frontier_candidate_max_count:
                extra.sort(key=lambda c: self._cheap_cell_priority(c, robot, region_set), reverse=True)
                extra = extra[:self.frontier_candidate_max_count]
            for c in extra:
                if c not in seen:
                    seen.add(c)
                    sampled.append(c)

        if len(sampled) > self.candidate_max_count:
            sampled.sort(key=lambda c: self._cheap_cell_priority(c, robot, region_set), reverse=True)
            sampled = sampled[:self.candidate_max_count]

        candidates: List[Candidate] = []
        for c in sampled:
            wx, wy = self._cell_to_world(c)
            yaw = math.atan2(wy - robot.y, wx - robot.x)
            candidates.append(self._evaluate_candidate(robot, c, wx, wy, yaw, region_set))
        candidates.sort(key=lambda cc: cc.score, reverse=True)
        return candidates

    def _unknown_neighbor_count(self, c: Cell) -> int:
        cnt = 0
        # _neighbors8() returns (cell, move_cost).  v14 accidentally passed the
        # whole (cell, cost) pair into _map_value(), which then reached _idx()
        # as a malformed cell and crashed during frontier candidate sampling.
        for nb, _move_cost in self._neighbors8(c):
            if self._map_value(nb) < 0:
                cnt += 1
        return cnt

    def _cheap_cell_priority(self, c: Cell, robot: RobotPose, region_set: Set[Cell]) -> float:
        score = 0.0
        if not self.coverage[self._idx(c)]:
            score += 6.0
        for nb in self._neighbors4(c):
            if self._map_value(nb) < 0:
                score += 4.0
        wx, wy = self._cell_to_world(c)
        score -= 0.4 * math.hypot(wx - robot.x, wy - robot.y)
        score += 0.1 * self._approx_clearance(c, max_radius_m=0.5)
        return score

    def _evaluate_candidate(self, robot: RobotPose, c: Cell, wx: float, wy: float, yaw: float, region_set: Set[Cell]) -> Candidate:
        visible = self._visible_cells_from_pose(wx, wy, yaw, self.view_fov_deg, self.view_max_range_m)
        visible_unknown = 0
        visible_uncovered = 0
        visible_frontier = 0
        cross_unknown = 0
        cross_frontier = 0
        for vc in visible:
            in_region = vc in region_set
            if not in_region and not self.allow_cross_region_view_gain:
                continue
            mv = self._map_value(vc)
            if mv < 0:
                if in_region:
                    visible_unknown += 1
                else:
                    cross_unknown += 1
            elif in_region and not self.coverage[self._idx(vc)]:
                visible_uncovered += 1
            frontier_here = False
            for nb in self._neighbors4(vc):
                if self._map_value(nb) < 0:
                    frontier_here = True
                    break
            if frontier_here:
                if in_region:
                    visible_frontier += 1
                else:
                    cross_frontier += 1
        clearance = self._approx_clearance(c, max_radius_m=0.7)
        euclid = math.hypot(wx - robot.x, wy - robot.y)
        score = (
            self.w_unknown * visible_unknown
            + self.w_unseen * visible_uncovered
            + self.w_frontier * visible_frontier
            + self.w_cross_region_unknown * cross_unknown
            + self.w_cross_region_frontier * cross_frontier
            + self.w_clearance * clearance
            - 0.35 * euclid
        )
        return Candidate(c, wx, wy, yaw, score, visible_unknown, visible_uncovered, visible_frontier, euclid, clearance)

    def _visible_cells_from_pose(self, x: float, y: float, yaw: float, fov_deg: float, max_range: float) -> Set[Cell]:
        visible: Set[Cell] = set()
        if fov_deg >= 359.0:
            start, stop = -math.pi, math.pi
        else:
            half = math.radians(fov_deg) * 0.5
            start, stop = -half, half
        step = math.radians(max(1.0, self.view_eval_angle_step_deg))
        a = start
        while a <= stop + 1e-6:
            visible.update(self._trace_ray_cells(x, y, yaw + a, max_range))
            a += step
        return visible

    def _trace_ray_cells(self, x0: float, y0: float, yaw: float, max_range: float) -> List[Cell]:
        assert self.geom is not None
        out: List[Cell] = []
        step = max(self.coverage_ray_step_m, self.geom.resolution * 0.5)
        n = max(1, int(max_range / step))
        ca, sa = math.cos(yaw), math.sin(yaw)
        last = None
        for k in range(n + 1):
            r = k * step
            c = self._world_to_cell(x0 + r * ca, y0 + r * sa)
            if c is None:
                break
            if c == last:
                continue
            last = c
            out.append(c)
            if self._is_obstacle(c):
                break
        return out

    def _approx_clearance(self, c: Cell, max_radius_m: float = 0.6) -> float:
        assert self.geom is not None
        max_r = max(1, int(max_radius_m / self.geom.resolution))
        best = max_r + 1
        cx, cy = c
        for dy in range(-max_r, max_r + 1):
            for dx in range(-max_r, max_r + 1):
                if dx * dx + dy * dy > max_r * max_r:
                    continue
                cc = (cx + dx, cy + dy)
                if not self._in_bounds(cc[0], cc[1]):
                    best = min(best, int(math.hypot(dx, dy)))
                    continue
                if self._is_obstacle(cc):
                    best = min(best, int(math.hypot(dx, dy)))
        if best == max_r + 1:
            return max_radius_m
        return best * self.geom.resolution

    # ------------------------------------------------------------------
    # Conservative path costs
    # ------------------------------------------------------------------
    def _cached_path_clearance(self, c: Cell) -> float:
        """Obstacle clearance used by A*.

        This is intentionally cached per /map update because conservative A*
        calls it many times.  The cache is cleared in _on_map() whenever new
        occupancy data arrives.
        """
        v = self._path_clearance_cache.get(c)
        if v is not None:
            return v
        if len(self._path_clearance_cache) > self.path_clearance_cache_max:
            self._path_clearance_cache.clear()
        r = max(self.path_prefer_clearance_m, self.path_min_clearance_m)
        v = self._approx_clearance(c, max_radius_m=r)
        self._path_clearance_cache[c] = v
        return v

    def _is_conservative_path_cell(self, c: Cell, start: Cell, goal: Cell, active_label: Optional[int]) -> bool:
        if not self._is_traversable(c, active_label):
            return False
        if not self.conservative_astar:
            return True
        # Do not reject start/goal only because SLAM puts them near a wall; the
        # intermediate path is what must be conservative.  The goal candidate
        # itself is already filtered by candidate_min_clearance_m.
        if c == start or c == goal:
            return True
        return self._cached_path_clearance(c) >= self.path_min_clearance_m

    def _conservative_path_extra_cost(self, c: Cell, move_cost: float, active_label: Optional[int]) -> float:
        if not self.conservative_astar:
            extra = 0.0
            if self._map_value(c) < 0:
                extra += 3.0
            if active_label is not None and self._region_label_value(c) != active_label:
                extra += 5.0
            return extra

        extra = 0.0
        mv = self._map_value(c)
        if mv < 0:
            extra += self.path_unknown_cost
        elif mv > 0:
            # Cartographer often produces soft occupancy values, not just 0/100.
            # Slightly prefer lower-probability free cells.
            extra += 0.015 * float(mv)

        clearance = self._cached_path_clearance(c)
        prefer = max(self.path_prefer_clearance_m, self.path_min_clearance_m + 1e-3)
        if clearance < prefer:
            # Quadratic inflation cost.  Cells close to walls remain traversable
            # only when the environment is narrow, but A* strongly prefers the
            # center of corridors/rooms.
            danger = (prefer - clearance) / prefer
            extra += self.path_wall_cost_weight * danger * danger
        if clearance < self.path_min_clearance_m:
            extra += self.path_wall_cost_weight * 10.0

        if active_label is not None and self._region_label_value(c) != active_label:
            extra += self.path_region_boundary_cost
        return extra * move_cost

    # ------------------------------------------------------------------
    # A*
    # ------------------------------------------------------------------
    def _astar(self, start: Cell, goal: Cell, active_label: Optional[int] = None) -> List[Cell]:
        if self.geom is None:
            return []
        if not self._is_traversable(start, active_label) or not self._is_traversable(goal, active_label):
            return []
        def h(a: Cell, b: Cell) -> float:
            return math.hypot(a[0] - b[0], a[1] - b[1])
        open_heap: List[Tuple[float, int, Cell]] = []
        seq = 0
        heapq.heappush(open_heap, (h(start, goal), seq, start))
        came: Dict[Cell, Cell] = {}
        g: Dict[Cell, float] = {start: 0.0}
        closed: Set[Cell] = set()
        expansions = 0
        while open_heap and expansions < self.a_star_max_expansions:
            _, _, cur = heapq.heappop(open_heap)
            if cur in closed:
                continue
            if cur == goal:
                return self._reconstruct_path(came, cur)
            closed.add(cur)
            expansions += 1
            for nb, move_cost in self._neighbors8(cur):
                if nb in closed or not self._is_conservative_path_cell(nb, start, goal, active_label):
                    continue
                # Avoid diagonal corner cutting and require both adjacent cells
                # to also satisfy the conservative clearance gate.
                if nb[0] != cur[0] and nb[1] != cur[1]:
                    side_a = (nb[0], cur[1])
                    side_b = (cur[0], nb[1])
                    if (not self._is_conservative_path_cell(side_a, start, goal, active_label)
                            or not self._is_conservative_path_cell(side_b, start, goal, active_label)):
                        continue
                    move_cost *= self.path_diagonal_cost_multiplier
                extra = self._conservative_path_extra_cost(nb, move_cost, active_label)
                tentative = g[cur] + move_cost + extra
                if tentative < g.get(nb, 1e18):
                    came[nb] = cur
                    g[nb] = tentative
                    seq += 1
                    heapq.heappush(open_heap, (tentative + h(nb, goal), seq, nb))
        return []

    def _is_traversable(self, c: Cell, active_label: Optional[int] = None) -> bool:
        if not self._is_free_candidate(c):
            return False
        # Clearance check in 8-neighborhood to avoid rubbing walls.
        if self._is_obstacle(c):
            return False
        if active_label is not None and self._region_label_value(c) != active_label:
            return False
        return True

    def _reconstruct_path(self, came: Dict[Cell, Cell], cur: Cell) -> List[Cell]:
        path = [cur]
        while cur in came:
            cur = came[cur]
            path.append(cur)
        path.reverse()
        return self._sparsify_path(path)

    def _sparsify_path(self, path: List[Cell]) -> List[Cell]:
        if len(path) <= 2 or self.geom is None:
            return path
        # Keep direction-change points, but also keep intermediate cells at a
        # bounded spacing.  Without this, a long straight A* segment collapses
        # to [start, goal], RViz appears to have no middle waypoint, and the
        # pure-pursuit target can jump too far.
        max_step_cells = max(1, int(math.ceil(self.path_sparsify_max_step_m / max(1e-9, self.geom.resolution))))
        out = [path[0]]
        last_dir = None
        since_emit = 0
        for i in range(1, len(path)):
            dx = path[i][0] - path[i - 1][0]
            dy = path[i][1] - path[i - 1][1]
            cur_dir = (int(math.copysign(1, dx)) if dx != 0 else 0, int(math.copysign(1, dy)) if dy != 0 else 0)
            since_emit += 1
            emit_prev = False
            if last_dir is None:
                last_dir = cur_dir
            elif cur_dir != last_dir:
                emit_prev = True
                last_dir = cur_dir
            elif since_emit >= max_step_cells:
                emit_prev = True
            if emit_prev:
                p = path[i - 1]
                if p != out[-1]:
                    out.append(p)
                since_emit = 0
        if path[-1] != out[-1]:
            out.append(path[-1])
        return out

    # ------------------------------------------------------------------
    # Velocity control
    # ------------------------------------------------------------------
    def _control_step(self, robot: RobotPose):
        now = time.time()
        front, left, right = self._scan_sectors()

        if self.recovery_mode is not None and now < self.recovery_until:
            if self.recovery_mode == 'reverse':
                self._publish_cmd(self.recovery_reverse_speed, 0.0, immediate=True)
            else:
                turn = self.recovery_rotate_speed if left >= right else -self.recovery_rotate_speed
                self._publish_cmd(0.0, turn, immediate=True)
            return
        elif self.recovery_mode is not None:
            self.recovery_mode = None

        if front < self.emergency_stop_distance_m:
            # v10: reserve reverse recovery for a genuine emergency only.  The
            # normal front_stop/front_slow bands are handled by arc avoidance so
            # a single close LiDAR sample does not create rapid stop-go pulses.
            self.recovery_mode = 'reverse'
            self.recovery_until = now + self.recovery_reverse_time_sec
            self._publish_cmd(self.recovery_reverse_speed, 0.0, immediate=True)
            self.last_state = 'RECOVERY_REVERSE_EMERGENCY_FRONT_OBSTACLE'
            return

        if self.waypoint_abandon_enabled and self.current_goal is not None and self.current_path:
            blocked = front < self.waypoint_abandon_front_distance_m
            cooldown_ok = (now - self.last_abandon_time) >= self.waypoint_abandon_cooldown_sec
            if blocked:
                if self.front_blocked_since is None:
                    self.front_blocked_since = now
                blocked_for = now - self.front_blocked_since
                if cooldown_ok and blocked_for >= self.waypoint_abandon_time_sec:
                    left_open = left >= right
                    turn = self.abandon_turn_speed if left_open else -self.abandon_turn_speed
                    self._abandon_current_goal(robot, f'front<{self.waypoint_abandon_front_distance_m:.2f}m_for_{blocked_for:.2f}s')
                    self._publish_cmd(0.0, turn, immediate=True)
                    self.last_state = 'ABANDON_WAYPOINT_FRONT_BLOCKED'
                    return
            else:
                self.front_blocked_since = None

        if self.current_goal is None or not self.current_path:
            if self.current_goal is not None:
                self.current_path = self._repath_to_current_goal(robot)
                self.path_index = 0
            if self.current_goal is None:
                self.last_planning_time = 0.0
                self._planning_step(robot)
            if self.current_goal is None:
                if self.keep_moving_when_no_goal:
                    self._publish_search_motion(robot, front, left, right, 'planner_returned_no_goal')
                    return
                self._publish_state({'state': 'IDLE_NO_VALID_GOAL_STOPPED', 'reason': 'planner_returned_no_goal'})
                self._publish_stop()
                return
            if not self.current_path and not self.direct_goal_fallback:
                if self.keep_moving_when_no_goal:
                    self._publish_search_motion(robot, front, left, right, 'no_path_and_direct_fallback_disabled')
                    return
                self._publish_state({'state': 'IDLE_NO_PATH_STOPPED', 'reason': 'no_path_and_direct_fallback_disabled'})
                self._publish_stop()
                return

        goal_dist = math.hypot(self.current_goal.x - robot.x, self.current_goal.y - robot.y)
        if goal_dist < self.goal_best_distance - self.goal_progress_epsilon_m:
            self.goal_best_distance = goal_dist
            self.goal_last_progress_time = now
        if goal_dist < self.goal_reached_radius_m:
            self.current_goal = None
            self.current_path = []
            self.path_index = 0
            if self.spin_in_place_after_goal_sec > 0:
                self.spin_until = now + self.spin_in_place_after_goal_sec
            if self.continuous_goal_handoff:
                self.last_planning_time = 0.0
                self._planning_step(robot)
                if self.current_goal is None or not self.current_path:
                    if self.keep_moving_when_no_goal:
                        self._publish_search_motion(robot, front, left, right, 'goal_reached_no_next_goal')
                        return
                    self._publish_state({'state': 'GOAL_REACHED_NO_NEXT_GOAL_STOPPED'})
                    self._publish_stop()
                    return
            else:
                self._publish_stop()
                return

        if now < self.spin_until:
            if self.idle_spin_enabled:
                self._publish_cmd(0.0, 0.18)
            else:
                self._publish_stop()
            return

        target = self._select_lookahead_waypoint(robot)
        if target is None:
            if self.direct_goal_fallback and self.current_goal is not None:
                tx, ty = self.current_goal.x, self.current_goal.y
            else:
                if self.keep_moving_when_no_goal:
                    self._publish_search_motion(robot, front, left, right, 'no_lookahead_target')
                    return
                self._publish_state({'state': 'NO_LOOKAHEAD_TARGET_STOPPED'})
                self._publish_stop()
                return
        else:
            tx, ty = self._cell_to_world(target)
        desired = math.atan2(ty - robot.y, tx - robot.x)
        heading_err = angle_wrap(desired - robot.yaw)
        dist = math.hypot(tx - robot.x, ty - robot.y)

        wz = clamp(self.angular_k * heading_err, -self.max_angular_z, self.max_angular_z)
        if abs(heading_err) > self.heading_stop_angle_rad:
            # Do not fully stop unless the front sector is actually dangerous.
            # Crawling while turning produces an arc rather than rotate-then-go.
            vx = self.creep_linear_x if front > self.front_stop_distance_m else 0.0
        else:
            vx = clamp(self.linear_k * max(dist, self.path_lookahead_m * 0.65), self.min_linear_x, self.max_linear_x)
            if abs(heading_err) > self.heading_slowdown_angle_rad:
                vx *= 0.72

        if front > self.front_slow_distance_m and abs(heading_err) < self.cruise_heading_limit_rad:
            vx = max(vx, min(self.cruise_linear_x, self.max_linear_x))
        if self.fast_cruise_when_locked and self.current_goal is not None and front > self.front_slow_distance_m:
            # Keep the velocity floor high while following a committed goal.
            # This is intentionally after heading slowdown so minor path jitter
            # does not collapse vx to a crawl.
            if abs(heading_err) < self.cruise_heading_limit_rad:
                vx = max(vx, min(self.cruise_linear_x, self.max_linear_x))

        if self.lidar_arc_avoidance:
            arc_wz, arc_scale = self._lidar_arc_avoidance()
            # Blend path heading and obstacle avoidance.  Positive wz is left.
            # If the right side is more open, arc_wz becomes negative and the
            # robot bends right around the obstacle instead of stopping.
            wz = clamp(wz + arc_wz, -self.max_angular_z, self.max_angular_z)
            vx *= arc_scale
            if front > self.front_slow_distance_m and abs(heading_err) < self.cruise_heading_limit_rad:
                vx = max(vx, min(self.cruise_linear_x * arc_scale, self.max_linear_x))
        else:
            if front < self.front_slow_distance_m:
                vx *= clamp((front - self.front_stop_distance_m) / max(1e-3, self.front_slow_distance_m - self.front_stop_distance_m), 0.0, 1.0)

        # Final reactive safety gate. A* clearance is map-based and the arc
        # layer is front-biased, so keep a hard side guard for door frames and
        # wall corners that enter the lateral scan sectors while turning.
        if front < self.front_stop_distance_m:
            vx = 0.0
            if abs(wz) < self.search_motion_min_turn_z:
                wz = self.search_motion_min_turn_z if left >= right else -self.search_motion_min_turn_z
        if left < self.side_stop_distance_m or right < self.side_stop_distance_m:
            vx = min(vx, self.creep_linear_x * 0.35)
            if left < right:
                wz = min(wz, -self.search_motion_min_turn_z)
            elif right < left:
                wz = max(wz, self.search_motion_min_turn_z)
        self._publish_cmd(vx, wz)

    def _publish_search_motion(self, robot: RobotPose, front: float, left: float, right: float, reason: str):
        """Keep exploring when no committed A* goal exists.

        This is a deliberately simple local behavior, not a replacement for A*.
        It keeps the robot moving slowly so Cartographer can reveal more map and
        the next planning tick has fresh frontier candidates.  If the front is
        clear, move forward with a gentle bias toward the more open side; if the
        front is blocked, rotate toward the open side without driving into it.
        """
        if self.clear_completed_regions_when_no_goal and self.completed_regions:
            self.completed_regions.clear()

        open_left = left >= right
        sign = 1.0 if open_left else -1.0
        balance = clamp((left - right) / max(0.20, max(left, right, 0.20)), -1.0, 1.0)

        if front > self.search_motion_front_clearance_m:
            vx = min(self.search_motion_linear_x, self.max_linear_x)
            # A small nonzero curve lets the front-only coverage sweep rather
            # than drawing a single narrow wedge forever.
            wz = clamp(
                self.search_motion_side_balance_gain * balance,
                -self.search_motion_angular_z,
                self.search_motion_angular_z,
            )
            if abs(wz) < self.search_motion_min_turn_z:
                wz = self.search_motion_min_turn_z * sign
            state = 'SEARCH_MOTION_FORWARD_ARC'
        else:
            vx = 0.0 if front < self.front_stop_distance_m else min(self.creep_linear_x, self.max_linear_x)
            wz = self.search_motion_angular_z * sign
            state = 'SEARCH_MOTION_TURN_TO_OPEN_SIDE'

        self.last_state = state
        self._publish_state({
            'state': state,
            'reason': reason,
            'front': round(front, 3),
            'left': round(left, 3),
            'right': round(right, 3),
            'cmd_vx': round(vx, 3),
            'cmd_wz': round(wz, 3),
            'keep_moving_when_no_goal': self.keep_moving_when_no_goal,
            'completed_region_count': len(self.completed_regions),
        })
        self._publish_cmd(vx, wz)

    def _select_lookahead_waypoint(self, robot: RobotPose) -> Optional[Cell]:
        if not self.current_path:
            return None
        # Advance past reached path cells.
        while self.path_index < len(self.current_path) - 1:
            wx, wy = self._cell_to_world(self.current_path[self.path_index])
            if math.hypot(wx - robot.x, wy - robot.y) < self.waypoint_reached_radius_m:
                self.path_index += 1
            else:
                break
        # Pick first cell at lookahead distance.
        for j in range(self.path_index, len(self.current_path)):
            wx, wy = self._cell_to_world(self.current_path[j])
            if math.hypot(wx - robot.x, wy - robot.y) >= self.path_lookahead_m:
                return self.current_path[j]
        return self.current_path[-1]

    def _lidar_arc_avoidance(self) -> Tuple[float, float]:
        """Return (additional_wz, linear_scale) from LiDAR sectors.

        This is not a full local planner.  It is a continuous reactive layer
        over the A* pure-pursuit command.  When an obstacle enters the forward
        cone but is not close enough for emergency recovery, we bias angular
        velocity toward the side with larger clearance and keep a reduced
        forward velocity.  That makes the trajectory an arc around furniture or
        a wall edge instead of repeated stop/rotate/drive cycles.
        """
        if self.scan_msg is None or not self.scan_msg.ranges:
            return 0.0, 1.0
        scan = self.scan_msg
        front: List[float] = []
        front_left: List[float] = []
        front_right: List[float] = []
        left: List[float] = []
        right: List[float] = []
        for i, rr in enumerate(scan.ranges):
            if math.isnan(rr) or math.isinf(rr) or rr <= scan.range_min:
                continue
            r = float(rr)
            a = angle_wrap(float(scan.angle_min) + i * float(scan.angle_increment))
            aa = abs(a)
            if aa <= math.radians(18.0):
                front.append(r)
            elif math.radians(18.0) < a <= math.radians(62.0):
                front_left.append(r)
            elif -math.radians(62.0) <= a < -math.radians(18.0):
                front_right.append(r)
            elif math.radians(62.0) < a <= math.radians(115.0):
                left.append(r)
            elif -math.radians(115.0) <= a < -math.radians(62.0):
                right.append(r)

        def pct(vals: List[float], q: float, default: float = 9.9) -> float:
            if not vals:
                return default
            vals = sorted(vals)
            k = int(clamp(q * (len(vals) - 1), 0, len(vals) - 1))
            return vals[k]

        f = pct(front, 0.15)
        fl = pct(front_left, 0.20)
        fr = pct(front_right, 0.20)
        l = pct(left, 0.20)
        r = pct(right, 0.20)

        arc_dist = max(self.front_arc_distance_m, self.front_slow_distance_m)
        if min(f, fl, fr) >= arc_dist:
            return 0.0, 1.0

        # Positive means turn left.  If front-left is closer than front-right,
        # bias right; if right side is more blocked than left side, bias left.
        near_left = min(fl, l)
        near_right = min(fr, r)
        side_balance = clamp((near_right - near_left) / max(0.05, arc_dist), -1.0, 1.0)
        # If the frontal cone is close, choose the globally more open side.
        if f < arc_dist:
            open_side = 1.0 if near_left > near_right else -1.0
            frontal_pressure = clamp((arc_dist - f) / max(0.05, arc_dist - self.front_stop_distance_m), 0.0, 1.0)
        else:
            open_side = 0.0
            frontal_pressure = 0.0

        bias = self.arc_clearance_balance_gain * side_balance + self.arc_avoidance_gain * frontal_pressure * open_side
        add_wz = clamp(bias, -self.arc_avoidance_max_wz, self.arc_avoidance_max_wz)

        closest = min(f, fl, fr)
        scale = clamp((closest - self.front_stop_distance_m) / max(0.05, arc_dist - self.front_stop_distance_m), self.arc_min_linear_scale, 1.0)
        if closest < self.front_stop_distance_m:
            scale = 0.0
        return add_wz, scale

    def _scan_sectors(self) -> Tuple[float, float, float]:
        if self.scan_msg is None or not self.scan_msg.ranges:
            return 9.9, 9.9, 9.9
        scan = self.scan_msg
        front_vals: List[float] = []
        left_vals: List[float] = []
        right_vals: List[float] = []
        for i, rr in enumerate(scan.ranges):
            if math.isnan(rr) or math.isinf(rr) or rr <= scan.range_min:
                continue
            a = float(scan.angle_min) + i * float(scan.angle_increment)
            # Normalize scan angle around robot forward.
            a = angle_wrap(a)
            if abs(a) < math.radians(22.0):
                front_vals.append(float(rr))
            elif math.radians(22.0) <= a <= math.radians(95.0):
                left_vals.append(float(rr))
            elif -math.radians(95.0) <= a <= -math.radians(22.0):
                right_vals.append(float(rr))
        def robust(vals: List[float]) -> float:
            if not vals:
                return 9.9
            vals = sorted(vals)
            q = clamp(self.scan_sector_percentile, 0.0, 0.50)
            k = int(clamp(q * (len(vals) - 1), 0, len(vals) - 1))
            return vals[k]
        return robust(front_vals), robust(left_vals), robust(right_vals)

    def _publish_cmd(self, vx: float, wz: float, immediate: bool = False):
        # Clamp first, then apply acceleration limiting and a light low-pass.
        # This prevents the stop-go waveform caused by frequent replans and
        # short waypoint handoffs.  Emergency stop/recovery can bypass smoothing.
        target_vx = float(clamp(vx, self.recovery_reverse_speed, self.max_linear_x))
        target_wz = float(clamp(wz, -self.max_angular_z, self.max_angular_z))

        now = time.time()
        dt = max(1e-3, min(0.25, now - self.last_cmd_time))
        self.last_cmd_time = now

        if immediate:
            out_vx = target_vx
            out_wz = target_wz
        else:
            dv = clamp(target_vx - self.last_cmd_vx, -self.linear_accel_limit * dt, self.linear_accel_limit * dt)
            dw = clamp(target_wz - self.last_cmd_wz, -self.angular_accel_limit * dt, self.angular_accel_limit * dt)
            limited_vx = self.last_cmd_vx + dv
            limited_wz = self.last_cmd_wz + dw
            alpha = clamp(self.cmd_smoothing_alpha, 0.0, 1.0)
            out_vx = alpha * limited_vx + (1.0 - alpha) * self.last_cmd_vx
            out_wz = alpha * limited_wz + (1.0 - alpha) * self.last_cmd_wz

        self.last_cmd_vx = out_vx
        self.last_cmd_wz = out_wz

        msg = TwistStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_footprint'
        msg.twist.linear.x = float(out_vx)
        msg.twist.angular.z = float(out_wz)
        self.cmd_pub.publish(msg)

    def _publish_stop(self):
        self._publish_cmd(0.0, 0.0, immediate=True)

    # ------------------------------------------------------------------
    # Publications
    # ------------------------------------------------------------------
    def _state_payload(self, state: str, robot: RobotPose, active: Optional[RegionStats], goal: Optional[Candidate], stats: Dict[int, RegionStats]) -> Dict:
        payload = {
            'state': state,
            'auto_start': self.auto_start,
            'region_count': len(stats),
            'completed_region_count': len(self.completed_regions),
            'active_region_label': active.label if active else None,
            'active_region_id': active.decoded_id if active else None,
            'robot_region_label': self._region_at_pose(robot),
            'cmd_vel_topic': self.cmd_vel_topic,
            'last_cmd_vx': round(self.last_cmd_vx, 3),
            'last_cmd_wz': round(self.last_cmd_wz, 3),
            'max_linear_x': round(self.max_linear_x, 3),
            'continuous_goal_handoff': self.continuous_goal_handoff,
            'lock_goal_until_reached': self.lock_goal_until_reached,
            'cruise_linear_x': round(self.cruise_linear_x, 3),
            'goal_age_sec': round(time.time() - self.goal_set_time, 2) if self.current_goal is not None else None,
            'goal_best_distance': round(self.goal_best_distance, 3) if self.current_goal is not None else None,
            'goal_progress_stall_sec': round(time.time() - self.goal_last_progress_time, 2) if self.current_goal is not None else None,
            'lidar_arc_avoidance': self.lidar_arc_avoidance,
            'conservative_astar': self.conservative_astar,
            'path_min_clearance_m': round(self.path_min_clearance_m, 3),
            'path_prefer_clearance_m': round(self.path_prefer_clearance_m, 3),
            'path_wall_cost_weight': round(self.path_wall_cost_weight, 2),
            'path_cells': len(self.current_path),
            'path_index': self.path_index,
            'active_region_age_sec': round(time.time() - self.active_region_set_time, 2),
            'region_completion_min_active_sec': round(self.region_completion_min_active_sec, 2),
            'region_completion_min_cells': int(self.region_completion_min_cells),
            'dense_coverage_marking': self.dense_coverage_marking,
            'coverage_front_only': self.coverage_front_only,
            'coverage_fov_deg': round(self.coverage_fov_deg, 1),
            'coverage_mark_robot_footprint': self.coverage_mark_robot_footprint,
            'coverage_brush_radius_m': round(self.coverage_brush_radius_m, 3),
            'allow_cross_region_view_gain': self.allow_cross_region_view_gain,
            'frontier_candidate_sampling': self.frontier_candidate_sampling,
            'frontier_candidate_max_count': self.frontier_candidate_max_count,
            'view_fov_deg': round(self.view_fov_deg, 1),
            'path_sparsify_max_step_m': round(self.path_sparsify_max_step_m, 3),
            'waypoint_abandon_enabled': self.waypoint_abandon_enabled,
            'waypoint_abandon_front_distance_m': round(self.waypoint_abandon_front_distance_m, 3),
            'front_blocked_for_sec': round(time.time() - self.front_blocked_since, 2) if self.front_blocked_since is not None else 0.0,
            'abandoned_goal_count': len(self.abandoned_goals),
            'last_abandon_reason': self.last_abandon_reason,
            'idle_spin_enabled': self.idle_spin_enabled,
            'keep_moving_when_no_goal': self.keep_moving_when_no_goal,
            'search_motion_linear_x': round(self.search_motion_linear_x, 3),
            'search_motion_angular_z': round(self.search_motion_angular_z, 3),
            'reopen_completed_region_on_frontier': self.reopen_completed_region_on_frontier,
            'reopen_frontier_margin': round(self.reopen_frontier_margin, 2),
        }
        if active:
            payload.update({
                'coverage_ratio': round(active.coverage_ratio, 4),
                'covered_cells': active.covered,
                'total_cells': active.total,
                'frontier_count': active.frontier_count,
                'area_m2': round(active.area_m2, 3),
            })
        if goal:
            payload['goal'] = {
                'x': round(goal.x, 3),
                'y': round(goal.y, 3),
                'yaw_deg': round(math.degrees(goal.yaw), 1),
                'score': round(goal.score, 2),
                'unknown': goal.visible_unknown,
                'uncovered': goal.visible_uncovered,
                'frontier': goal.visible_frontier,
                'path_cost_m': round(goal.euclid_cost, 3),
                'clearance_m': round(goal.clearance_m, 3),
            }
        return payload

    def _publish_state(self, payload: Dict):
        self.last_state = str(payload.get('state', self.last_state))
        msg = String()
        msg.data = json.dumps(payload, ensure_ascii=False)
        self.state_pub.publish(msg)

    def _publish_markers(self, robot: RobotPose):
        ma = MarkerArray()
        now = self.get_clock().now().to_msg()
        clear = Marker()
        clear.action = Marker.DELETEALL
        ma.markers.append(clear)
        mid = 1

        # Active region / completed region text.
        for label, st in self.last_stats.items():
            if label != self.active_region_label and label not in self.completed_regions:
                continue
            text = Marker()
            text.header.frame_id = self.global_frame
            text.header.stamp = now
            text.ns = 'auto_mapper_region_text'
            text.id = mid; mid += 1
            text.type = Marker.TEXT_VIEW_FACING
            text.action = Marker.ADD
            text.pose.position.x = st.centroid[0]
            text.pose.position.y = st.centroid[1]
            text.pose.position.z = 0.62
            text.pose.orientation.w = 1.0
            text.scale.z = 0.18
            if label == self.active_region_label:
                text.color = ColorRGBA(r=1.0, g=1.0, b=1.0, a=1.0)
                prefix = 'AUTO ACTIVE'
            else:
                text.color = ColorRGBA(r=0.2, g=1.0, b=0.3, a=0.8)
                prefix = 'DONE'
            text.text = f'{prefix} R{st.decoded_id}\ncov={st.coverage_ratio:.2f} fr={st.frontier_count}'
            ma.markers.append(text)

        # Candidate points.
        if self.last_candidates:
            pts = Marker()
            pts.header.frame_id = self.global_frame
            pts.header.stamp = now
            pts.ns = 'auto_mapper_candidates'
            pts.id = mid; mid += 1
            pts.type = Marker.POINTS
            pts.action = Marker.ADD
            pts.pose.orientation.w = 1.0
            pts.scale.x = 0.045
            pts.scale.y = 0.045
            pts.color = ColorRGBA(r=0.2, g=0.75, b=1.0, a=0.55)
            for c in self.last_candidates[:120]:
                pts.points.append(Point(x=c.x, y=c.y, z=0.12))
            ma.markers.append(pts)

        # Current goal.
        if self.current_goal is not None:
            g = self.current_goal
            sph = Marker()
            sph.header.frame_id = self.global_frame
            sph.header.stamp = now
            sph.ns = 'auto_mapper_current_goal'
            sph.id = mid; mid += 1
            sph.type = Marker.SPHERE
            sph.action = Marker.ADD
            sph.pose.position.x = g.x
            sph.pose.position.y = g.y
            sph.pose.position.z = 0.22
            sph.pose.orientation.w = 1.0
            sph.scale.x = 0.20
            sph.scale.y = 0.20
            sph.scale.z = 0.20
            sph.color = ColorRGBA(r=1.0, g=0.85, b=0.05, a=0.95)
            ma.markers.append(sph)

            arr = Marker()
            arr.header.frame_id = self.global_frame
            arr.header.stamp = now
            arr.ns = 'auto_mapper_goal_yaw'
            arr.id = mid; mid += 1
            arr.type = Marker.ARROW
            arr.action = Marker.ADD
            arr.pose.position.x = g.x
            arr.pose.position.y = g.y
            arr.pose.position.z = 0.28
            arr.pose.orientation = quaternion_from_yaw(g.yaw)
            arr.scale.x = 0.36
            arr.scale.y = 0.045
            arr.scale.z = 0.045
            arr.color = ColorRGBA(r=1.0, g=0.6, b=0.0, a=0.95)
            ma.markers.append(arr)

        # Path as line strip.
        if self.current_path:
            line = Marker()
            line.header.frame_id = self.global_frame
            line.header.stamp = now
            line.ns = 'auto_mapper_path'
            line.id = mid; mid += 1
            line.type = Marker.LINE_STRIP
            line.action = Marker.ADD
            line.pose.orientation.w = 1.0
            line.scale.x = 0.035
            line.color = ColorRGBA(r=1.0, g=1.0, b=0.0, a=0.85)
            for c in self.current_path:
                wx, wy = self._cell_to_world(c)
                line.points.append(Point(x=wx, y=wy, z=0.10))
            ma.markers.append(line)

        self.marker_pub.publish(ma)

    def _publish_path(self):
        if self.geom is None or not self.current_path:
            return
        msg = Path()
        msg.header.frame_id = self.global_frame
        msg.header.stamp = self.get_clock().now().to_msg()
        for c in self.current_path:
            wx, wy = self._cell_to_world(c)
            ps = PoseStamped()
            ps.header = msg.header
            ps.pose.position.x = wx
            ps.pose.position.y = wy
            ps.pose.orientation.w = 1.0
            msg.poses.append(ps)
        self.path_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = RegionAutoMapperNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node._publish_stop()
        except Exception:
            pass
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
