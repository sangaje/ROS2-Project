#!/usr/bin/env python3
"""Region-aware Nav2 exploration for TurtleBot3 + Cartographer.

This node deliberately does NOT publish /cmd_vel.  It only selects semantic / topological
exploration goals and sends them to Nav2 NavigateToPose.  Nav2 is responsible for path
planning, local control, recovery behaviors, and TwistStamped velocity publication.
"""

from __future__ import annotations

import json
import math
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.duration import Duration
from rclpy.time import Time
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor

from action_msgs.msg import GoalStatus
from geometry_msgs.msg import PoseStamped, Quaternion
from nav2_msgs.action import NavigateToPose
from nav_msgs.msg import OccupancyGrid
from sensor_msgs.msg import LaserScan
from std_msgs.msg import String
from visualization_msgs.msg import Marker, MarkerArray

import tf2_ros


Cell = Tuple[int, int]
Pose2D = Tuple[float, float, float]


def _yaw_from_quaternion_msg(q: Quaternion) -> float:
    """Return yaw from a geometry_msgs/Quaternion without tf_transformations.

    This avoids the optional transforms3d runtime dependency used by
    tf_transformations on some ROS 2 Jazzy installs.
    """
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


def _quaternion_msg_from_yaw(yaw: float) -> Quaternion:
    half = 0.5 * yaw
    return Quaternion(x=0.0, y=0.0, z=math.sin(half), w=math.cos(half))


@dataclass
class MapGeom:
    width: int = 0
    height: int = 0
    resolution: float = 0.05
    origin_x: float = 0.0
    origin_y: float = 0.0


@dataclass
class RegionStats:
    label: int
    total_free: int = 0
    covered: int = 0
    frontier: int = 0
    centroid_x: float = 0.0
    centroid_y: float = 0.0
    coverage_ratio: float = 0.0


@dataclass
class Candidate:
    cell: Cell
    x: float
    y: float
    yaw: float
    score: float
    region_cov_gain: int
    region_frontier_gain: int
    cross_unknown_gain: int
    cross_frontier_gain: int
    distance: float
    reason: str
    # v28 diagnostics. Positional constructors remain compatible because these
    # fields have defaults.
    distance_cost: float = 0.0
    obstacle_cost: float = 0.0
    unknown_cost: float = 0.0
    clearance_risk_cost: float = 0.0
    key: str = ''
    candidate_type: str = 'COVERAGE_FILL'
    target_region_id: int = 0
    base_score: float = 0.0
    dynamic_score: float = 0.0
    created_time: float = 0.0
    last_refresh_time: float = 0.0
    last_revalidate_time: float = 0.0
    stale: bool = False
    blacklisted: bool = False
    debug: str = ''
    # v29 priority-queue cache. The view-gain raycast is the expensive part of
    # scoring; we stash the mode + clearance computed at rebuild time so the
    # high-rate re-score can recompute only the cheap pose-dependent terms
    # (distance + obstacle/clearance penalty) without raycasting again.
    score_mode: str = 'map'
    cached_clearance: float = 0.0


class RegionNav2ExplorerNode(Node):
    def __init__(self) -> None:
        super().__init__('region_nav2_explorer')

        # Topics / frames
        self.declare_parameter('map_topic', '/map')
        self.declare_parameter('region_map_topic', '/slam_region_graph/region_map')
        self.declare_parameter('scan_topic', '/scan')
        self.declare_parameter('global_frame', 'map')
        self.declare_parameter('robot_frame', 'base_footprint')
        self.declare_parameter('navigate_action_name', 'navigate_to_pose')

        # Update cadence
        self.declare_parameter('timer_period_sec', 0.05)
        self.declare_parameter('planning_period_sec', 0.35)
        self.declare_parameter('goal_dispatch_period_sec', 0.10)
        self.declare_parameter('min_goal_interval_sec', 0.05)
        self.declare_parameter('nav2_goal_timeout_sec', 45.0)
        self.declare_parameter('wait_nav2_server_timeout_sec', 0.15)
        self.declare_parameter('auto_start', True)
        # Fast rolling-goal mode.  Without this, NavigateToPose waits until the
        # current viewpoint fully succeeds before selecting the next viewpoint,
        # which creates a stop-go-stop pattern.
        self.declare_parameter('rolling_goal_update_enabled', True)
        self.declare_parameter('rolling_goal_period_sec', 0.12)
        self.declare_parameter('rolling_goal_near_distance_m', 1.15)
        self.declare_parameter('rolling_goal_min_shift_m', 0.12)
        self.declare_parameter('rolling_goal_min_score_improvement', 0.50)
        self.declare_parameter('rolling_goal_ignore_scan_turn', True)
        self.declare_parameter('dynamic_timer_period_update', True)
        self.declare_parameter('rolling_goal_force_period_sec', 0.55)
        self.declare_parameter('rolling_goal_force_near_distance_m', 1.35)
        self.declare_parameter('rolling_goal_yaw_change_rad', 0.45)
        self.declare_parameter('rolling_goal_reason_change_min_shift_m', 0.08)
        self.declare_parameter('rolling_goal_preempt_cooldown_sec', 0.10)
        self.declare_parameter('rolling_goal_debug_throttle_sec', 0.75)

        # Dedicated coverage update.  Coverage must be painted while the robot is
        # moving, not only after a Nav2 goal finishes or after a planning cycle.
        # This timer is intentionally separate from semantic planning.
        self.declare_parameter('coverage_update_period_sec', 0.10)
        self.declare_parameter('coverage_publish_period_sec', 0.10)
        self.declare_parameter('coverage_interpolate_motion_enabled', True)
        self.declare_parameter('coverage_motion_sample_step_m', 0.04)
        self.declare_parameter('coverage_motion_yaw_step_deg', 3.0)
        self.declare_parameter('coverage_motion_max_gap_m', 0.80)
        self.declare_parameter('coverage_motion_max_samples', 32)
        self.declare_parameter('coverage_live_debug_throttle_sec', 1.0)

        # Candidate priority queue.  Full candidate generation is expensive and
        # waiting for it on every Nav2 result causes stop-go motion.  Keep a
        # small rolling pool of high-value candidates, re-score it frequently,
        # and stream the current best candidate immediately.
        self.declare_parameter('candidate_priority_queue_enabled', True)
        self.declare_parameter('candidate_queue_size', 30)
        self.declare_parameter('candidate_queue_refresh_period_sec', 0.80)
        self.declare_parameter('candidate_queue_revalidate_period_sec', 0.05)
        self.declare_parameter('candidate_queue_max_age_sec', 8.0)
        self.declare_parameter('candidate_queue_min_score', 1.0)
        self.declare_parameter('candidate_queue_include_global_frontier', True)
        self.declare_parameter('candidate_queue_include_next_region', True)
        self.declare_parameter('candidate_queue_log_throttle_sec', 0.75)

        # Route beam-search lookahead queue.  Instead of showing the top-3
        # independent candidates (which cluster near the same high-gain area),
        # beam search builds a route [G1->G2->G3] that maximises *total*
        # discounted coverage over the next three hops while penalising overlap
        # between them and redundant same-region micro-steps.
        # RViz markers always show the best route; Nav2 still receives only G1.
        self.declare_parameter('route_queue_enabled', True)
        self.declare_parameter('route_horizon', 3)
        self.declare_parameter('route_beam_width', 6)
        self.declare_parameter('route_candidate_pool_size', 24)
        self.declare_parameter('route_spatial_suppression_m', 0.60)
        self.declare_parameter('route_discount', 0.65)
        self.declare_parameter('route_same_region_penalty', 15.0)
        self.declare_parameter('route_overlap_penalty_per_cell', 0.8)

        # Frontier lookahead / push-forward map expansion.  A frontier viewpoint
        # that is too close makes the robot stop at the free/unknown boundary.
        # When LiDAR says the corridor/doorway ahead is clear, push the Nav2 goal
        # farther forward so Cartographer receives scans from deeper inside the
        # opening.  Coverage remains conservative; this only affects goal choice.
        self.declare_parameter('map_expand_lidar_push_enabled', True)
        self.declare_parameter('map_expand_push_into_lidar_clear_unknown', True)
        self.declare_parameter('map_expand_goal_push_max_m', 1.20)
        self.declare_parameter('map_expand_unknown_push_max_m', 0.55)
        self.declare_parameter('map_expand_goal_push_step_m', 0.10)
        self.declare_parameter('map_expand_goal_push_min_m', 0.25)
        self.declare_parameter('map_expand_goal_push_lidar_margin_m', 0.28)
        self.declare_parameter('map_expand_goal_push_score_bonus_per_m', 10.0)
        self.declare_parameter('lidar_push_log_throttle_sec', 1.0)

        # Occupancy interpretation
        self.declare_parameter('free_threshold', 62)
        self.declare_parameter('occupied_threshold', 65)
        self.declare_parameter('unknown_is_frontier', True)
        self.declare_parameter('candidate_min_clearance_m', 0.22)
        self.declare_parameter('candidate_max_goal_distance_m', 5.0)

        # Front-only coverage
        self.declare_parameter('coverage_front_only', True)
        self.declare_parameter('coverage_fov_deg', 55.0)
        self.declare_parameter('coverage_max_range_m', 1.45)
        self.declare_parameter('coverage_ray_step_m', 0.025)
        self.declare_parameter('coverage_brush_radius_m', 0.06)
        self.declare_parameter('coverage_downsample_angle_step_deg', 1.0)
        self.declare_parameter('coverage_mark_robot_footprint', False)
        self.declare_parameter('coverage_robot_radius_m', 0.22)
        self.declare_parameter('coverage_only_known_free', True)
        self.declare_parameter('coverage_stop_at_unknown', True)
        self.declare_parameter('coverage_require_lidar_clear', True)
        self.declare_parameter('coverage_obstacle_margin_m', 0.12)

        # Candidate generation
        self.declare_parameter('candidate_grid_step_m', 0.25)
        self.declare_parameter('candidate_max_count', 120)
        self.declare_parameter('candidate_yaw_samples', 12)
        self.declare_parameter('candidate_frontier_ring_radius_m', 0.55)
        self.declare_parameter('frontier_candidate_sampling', True)
        self.declare_parameter('frontier_candidate_max_count', 120)
        self.declare_parameter('frontier_candidate_min_unknown_neighbors', 1)
        self.declare_parameter('allow_cross_region_view_gain', True)
        self.declare_parameter('view_fov_deg', 100.0)
        self.declare_parameter('view_max_range_m', 3.0)
        self.declare_parameter('view_ray_step_m', 0.05)
        self.declare_parameter('view_angle_step_deg', 5.0)
        self.declare_parameter('view_stop_at_unknown', True)

        # Scoring
        self.declare_parameter('w_region_coverage_gain', 4.0)
        self.declare_parameter('w_region_frontier_gain', 3.0)
        self.declare_parameter('w_cross_region_unknown', 1.5)
        self.declare_parameter('w_cross_region_frontier', 1.0)
        self.declare_parameter('w_path_cost', 0.8)
        self.declare_parameter('w_clearance_bonus', 0.25)
        self.declare_parameter('w_candidate_distance_cost', 2.5)
        self.declare_parameter('w_candidate_distance_sq_cost', 0.45)
        self.declare_parameter('w_candidate_obstacle_density_cost', 180.0)
        self.declare_parameter('w_candidate_unknown_density_cost', 30.0)
        self.declare_parameter('w_candidate_clearance_risk_cost', 45.0)
        self.declare_parameter('candidate_obstacle_line_step_m', 0.05)
        self.declare_parameter('candidate_obstacle_check_radius_m', 0.18)
        self.declare_parameter('region_candidate_gain_threshold', 24.0)
        self.declare_parameter('region_map_gain_threshold', 14.0)
        self.declare_parameter('region_coverage_gain_threshold', 12.0)

        # Region lifecycle
        self.declare_parameter('region_coverage_threshold', 0.82)
        self.declare_parameter('region_frontier_threshold', 12)
        self.declare_parameter('region_min_active_time_sec', 15.0)
        self.declare_parameter('region_max_active_time_sec', 55.0)
        self.declare_parameter('region_switch_on_stall', True)
        self.declare_parameter('region_switch_gain_ratio', 0.65)
        self.declare_parameter('region_switch_coverage_ratio', 0.72)
        self.declare_parameter('prefer_unvisited_region', True)
        self.declare_parameter('region_min_cells', 80)
        self.declare_parameter('active_region_iou_match_threshold', 0.15)
        self.declare_parameter('reopen_completed_region_on_frontier', True)
        self.declare_parameter('reopen_frontier_margin', 1.20)
        self.declare_parameter('clear_completed_regions_when_no_goal', True)

        # Next-region selection
        self.declare_parameter('w_next_region_uncovered', 4.0)
        self.declare_parameter('w_next_region_frontier', 3.0)
        self.declare_parameter('w_next_region_unvisited', 2.0)
        self.declare_parameter('w_next_region_path_cost', 0.8)
        self.declare_parameter('next_region_entry_offset_m', 0.40)

        # Failure / blacklist handling
        self.declare_parameter('goal_blacklist_radius_m', 0.70)
        self.declare_parameter('goal_blacklist_time_sec', 60.0)
        self.declare_parameter('blacklist_failed_nav2_goals', True)
        self.declare_parameter('allow_nav2_scan_turn_goal', True)
        self.declare_parameter('scan_turn_yaw_step_deg', 135.0)
        self.declare_parameter('allow_nav2_forward_probe_goal', True)
        self.declare_parameter('forward_probe_distance_m', 0.55)
        self.declare_parameter('forward_probe_min_distance_m', 0.28)
        self.declare_parameter('forward_probe_lidar_margin_m', 0.18)

        # How often the (expensive) full ROS-parameter reload runs. Reading ~120
        # parameters through the parameter API on *every* timer tick added real
        # latency/jitter and was a hidden contributor to the slow feel. We still
        # support live tuning, just at a sane cadence instead of every tick.
        self.declare_parameter('param_refresh_period_sec', 0.5)

        # Throttle guard state for _refresh_params. Must exist before the first
        # call below, which always performs a full load (_params_loaded == False).
        self._last_param_refresh_time: float = 0.0
        self._params_loaded: bool = False
        self.param_refresh_period_sec: float = 0.5

        # Read parameters into attributes (refreshed periodically for live tuning)
        self._refresh_params()

        self.map_msg: Optional[OccupancyGrid] = None
        self.region_msg: Optional[OccupancyGrid] = None
        self.scan_msg: Optional[LaserScan] = None
        self.geom = MapGeom()
        self.coverage: List[int] = []  # -1 unknown/uncovered, 100 covered
        self._last_map_signature: Optional[Tuple[int, int, float, float, float]] = None
        self._last_coverage_pose: Optional[Pose2D] = None
        self.last_coverage_publish_time: float = 0.0
        self.coverage_update_count: int = 0
        self.last_live_coverage_debug_time: float = 0.0
        self.candidate_queue: List[Candidate] = []
        self.last_candidate_queue_refresh_time: float = 0.0
        self.last_candidate_queue_revalidate_time: float = 0.0
        self.last_candidate_queue_log_time: float = 0.0
        self.last_candidate_prune_log_time: float = 0.0
        # Route beam-search queue: best [G1, G2, G3] exploration plan.
        self.route_queue: List[Candidate] = []
        self.last_route_queue_build_time: float = 0.0
        self.last_lidar_push_log_time: float = 0.0
        self.last_goal_dispatch_debug_time: float = 0.0
        self.lidar_push_since_log: int = 0
        self._shutting_down: bool = False

        self.active_region: Optional[int] = None
        self.active_region_started: float = self._now_sec()
        self.completed_regions: Set[int] = set()
        self.visited_regions: Set[int] = set()
        self.blacklist: List[Tuple[float, float, float]] = []  # x, y, expire time
        self.last_region_cells: Set[Cell] = set()

        self.active_goal = False
        self.goal_handle = None
        self.current_goal_pose: Optional[PoseStamped] = None
        self.current_goal_reason = ''
        self.current_goal_sent_time = 0.0
        self.current_goal_seq = 0
        self.current_goal_score = -1e9
        self.last_plan_time = 0.0
        self.last_goal_time = 0.0
        self.last_state = 'INIT'
        self.goal_seq = 0
        self.last_best: Optional[Candidate] = None
        self.last_stats: Dict[int, RegionStats] = {}

        map_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
        )
        normal_qos = QoSProfile(depth=10)

        # Two callback groups so the high-rate coverage painter is never blocked
        # by a long planning callback. Under a MultiThreadedExecutor each group
        # gets its own thread, so coverage keeps painting at ~10 Hz even while
        # candidate re-scoring is busy. Sensor/map subscriptions live in the
        # coverage group so fresh scans keep flowing to the painter.
        self.coverage_cb_group = MutuallyExclusiveCallbackGroup()
        self.planning_cb_group = MutuallyExclusiveCallbackGroup()
        self.dispatch_cb_group = MutuallyExclusiveCallbackGroup()

        self.create_subscription(
            OccupancyGrid, self.map_topic, self._on_map, map_qos,
            callback_group=self.coverage_cb_group,
        )
        self.create_subscription(
            OccupancyGrid, self.region_map_topic, self._on_region_map, map_qos,
            callback_group=self.coverage_cb_group,
        )
        self.create_subscription(
            LaserScan, self.scan_topic, self._on_scan, sensor_qos,
            callback_group=self.coverage_cb_group,
        )

        self.coverage_pub = self.create_publisher(OccupancyGrid, '/region_nav2_explorer/coverage_map', map_qos)
        self.state_pub = self.create_publisher(String, '/region_nav2_explorer/state', normal_qos)
        self.marker_pub = self.create_publisher(MarkerArray, '/region_nav2_explorer/markers', normal_qos)
        self.goal_debug_pub = self.create_publisher(PoseStamped, '/region_nav2_explorer/goal_pose', normal_qos)

        self.tf_buffer = tf2_ros.Buffer(cache_time=Duration(seconds=10.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.nav_client = ActionClient(self, NavigateToPose, self.navigate_action_name)
        self._active_timer_period_sec = max(0.02, float(self.timer_period_sec))
        self.timer = self.create_timer(
            self._active_timer_period_sec, self._on_timer,
            callback_group=self.planning_cb_group,
        )
        self._active_goal_dispatch_period_sec = max(0.02, float(self.goal_dispatch_period_sec))
        self.goal_dispatch_timer = self.create_timer(
            self._active_goal_dispatch_period_sec, self._on_goal_dispatch_timer,
            callback_group=self.dispatch_cb_group,
        )
        self._active_coverage_timer_period_sec = max(0.02, float(self.coverage_update_period_sec))
        self.coverage_timer = self.create_timer(
            self._active_coverage_timer_period_sec, self._on_coverage_timer,
            callback_group=self.coverage_cb_group,
        )
        self.last_rolling_debug_time = 0.0

        self.get_logger().info(
            'REGION_NAV2_EXPLORER_READY | does_not_publish_cmd_vel=True | '
            f'action={self.navigate_action_name} map={self.map_topic} region={self.region_map_topic} '
            f'coverage_front_only={self.coverage_front_only} fov={self.coverage_fov_deg:.1f}'
        )

    # ------------------------------------------------------------------
    # Parameters
    # ------------------------------------------------------------------
    def _refresh_params(self) -> None:
        # Skip the heavy full reload if we did one recently. The first call
        # (during __init__) always runs because _params_loaded starts False.
        if self._params_loaded:
            now = self._now_sec()
            if (now - self._last_param_refresh_time) < self.param_refresh_period_sec:
                return
        gp = self.get_parameter
        self.param_refresh_period_sec = float(gp('param_refresh_period_sec').value)
        self.map_topic = str(gp('map_topic').value)
        self.region_map_topic = str(gp('region_map_topic').value)
        self.scan_topic = str(gp('scan_topic').value)
        self.global_frame = str(gp('global_frame').value)
        self.robot_frame = str(gp('robot_frame').value)
        self.navigate_action_name = str(gp('navigate_action_name').value)

        self.timer_period_sec = float(gp('timer_period_sec').value)
        self.planning_period_sec = float(gp('planning_period_sec').value)
        self.goal_dispatch_period_sec = float(gp('goal_dispatch_period_sec').value)
        self.min_goal_interval_sec = float(gp('min_goal_interval_sec').value)
        self.nav2_goal_timeout_sec = float(gp('nav2_goal_timeout_sec').value)
        self.wait_nav2_server_timeout_sec = float(gp('wait_nav2_server_timeout_sec').value)
        self.auto_start = bool(gp('auto_start').value)
        self.rolling_goal_update_enabled = bool(gp('rolling_goal_update_enabled').value)
        self.rolling_goal_period_sec = float(gp('rolling_goal_period_sec').value)
        self.rolling_goal_near_distance_m = float(gp('rolling_goal_near_distance_m').value)
        self.rolling_goal_min_shift_m = float(gp('rolling_goal_min_shift_m').value)
        self.rolling_goal_min_score_improvement = float(gp('rolling_goal_min_score_improvement').value)
        self.rolling_goal_ignore_scan_turn = bool(gp('rolling_goal_ignore_scan_turn').value)
        self.dynamic_timer_period_update = bool(gp('dynamic_timer_period_update').value)
        self.rolling_goal_force_period_sec = float(gp('rolling_goal_force_period_sec').value)
        self.rolling_goal_force_near_distance_m = float(gp('rolling_goal_force_near_distance_m').value)
        self.rolling_goal_yaw_change_rad = float(gp('rolling_goal_yaw_change_rad').value)
        self.rolling_goal_reason_change_min_shift_m = float(gp('rolling_goal_reason_change_min_shift_m').value)
        self.rolling_goal_preempt_cooldown_sec = float(gp('rolling_goal_preempt_cooldown_sec').value)
        self.rolling_goal_debug_throttle_sec = float(gp('rolling_goal_debug_throttle_sec').value)
        self.coverage_update_period_sec = float(gp('coverage_update_period_sec').value)
        self.coverage_publish_period_sec = float(gp('coverage_publish_period_sec').value)
        self.coverage_interpolate_motion_enabled = bool(gp('coverage_interpolate_motion_enabled').value)
        self.coverage_motion_sample_step_m = float(gp('coverage_motion_sample_step_m').value)
        self.coverage_motion_yaw_step_deg = float(gp('coverage_motion_yaw_step_deg').value)
        self.coverage_motion_max_gap_m = float(gp('coverage_motion_max_gap_m').value)
        self.coverage_motion_max_samples = int(gp('coverage_motion_max_samples').value)
        self.coverage_live_debug_throttle_sec = float(gp('coverage_live_debug_throttle_sec').value)
        self.candidate_priority_queue_enabled = bool(gp('candidate_priority_queue_enabled').value)
        self.candidate_queue_size = int(gp('candidate_queue_size').value)
        self.candidate_queue_refresh_period_sec = float(gp('candidate_queue_refresh_period_sec').value)
        self.candidate_queue_revalidate_period_sec = float(gp('candidate_queue_revalidate_period_sec').value)
        self.candidate_queue_max_age_sec = float(gp('candidate_queue_max_age_sec').value)
        self.candidate_queue_min_score = float(gp('candidate_queue_min_score').value)
        self.candidate_queue_include_global_frontier = bool(gp('candidate_queue_include_global_frontier').value)
        self.candidate_queue_include_next_region = bool(gp('candidate_queue_include_next_region').value)
        self.candidate_queue_log_throttle_sec = float(gp('candidate_queue_log_throttle_sec').value)
        self.route_queue_enabled = bool(gp('route_queue_enabled').value)
        self.route_horizon = int(gp('route_horizon').value)
        self.route_beam_width = int(gp('route_beam_width').value)
        self.route_candidate_pool_size = int(gp('route_candidate_pool_size').value)
        self.route_spatial_suppression_m = float(gp('route_spatial_suppression_m').value)
        self.route_discount = float(gp('route_discount').value)
        self.route_same_region_penalty = float(gp('route_same_region_penalty').value)
        self.route_overlap_penalty_per_cell = float(gp('route_overlap_penalty_per_cell').value)
        self.map_expand_lidar_push_enabled = bool(gp('map_expand_lidar_push_enabled').value)
        self.map_expand_push_into_lidar_clear_unknown = bool(gp('map_expand_push_into_lidar_clear_unknown').value)
        self.map_expand_goal_push_max_m = float(gp('map_expand_goal_push_max_m').value)
        self.map_expand_unknown_push_max_m = float(gp('map_expand_unknown_push_max_m').value)
        self.map_expand_goal_push_step_m = float(gp('map_expand_goal_push_step_m').value)
        self.map_expand_goal_push_min_m = float(gp('map_expand_goal_push_min_m').value)
        self.map_expand_goal_push_lidar_margin_m = float(gp('map_expand_goal_push_lidar_margin_m').value)
        self.map_expand_goal_push_score_bonus_per_m = float(gp('map_expand_goal_push_score_bonus_per_m').value)
        self.lidar_push_log_throttle_sec = float(gp('lidar_push_log_throttle_sec').value)

        self.free_threshold = int(gp('free_threshold').value)
        self.occupied_threshold = int(gp('occupied_threshold').value)
        self.unknown_is_frontier = bool(gp('unknown_is_frontier').value)
        self.candidate_min_clearance_m = float(gp('candidate_min_clearance_m').value)
        self.candidate_max_goal_distance_m = float(gp('candidate_max_goal_distance_m').value)

        self.coverage_front_only = bool(gp('coverage_front_only').value)
        self.coverage_fov_deg = float(gp('coverage_fov_deg').value)
        self.coverage_max_range_m = float(gp('coverage_max_range_m').value)
        self.coverage_ray_step_m = float(gp('coverage_ray_step_m').value)
        self.coverage_brush_radius_m = float(gp('coverage_brush_radius_m').value)
        self.coverage_downsample_angle_step_deg = float(gp('coverage_downsample_angle_step_deg').value)
        self.coverage_mark_robot_footprint = bool(gp('coverage_mark_robot_footprint').value)
        self.coverage_robot_radius_m = float(gp('coverage_robot_radius_m').value)
        self.coverage_only_known_free = bool(gp('coverage_only_known_free').value)
        self.coverage_stop_at_unknown = bool(gp('coverage_stop_at_unknown').value)
        self.coverage_require_lidar_clear = bool(gp('coverage_require_lidar_clear').value)
        self.coverage_obstacle_margin_m = float(gp('coverage_obstacle_margin_m').value)

        self.candidate_grid_step_m = float(gp('candidate_grid_step_m').value)
        self.candidate_max_count = int(gp('candidate_max_count').value)
        self.candidate_yaw_samples = int(gp('candidate_yaw_samples').value)
        self.candidate_frontier_ring_radius_m = float(gp('candidate_frontier_ring_radius_m').value)
        self.frontier_candidate_sampling = bool(gp('frontier_candidate_sampling').value)
        self.frontier_candidate_max_count = int(gp('frontier_candidate_max_count').value)
        self.frontier_candidate_min_unknown_neighbors = int(gp('frontier_candidate_min_unknown_neighbors').value)
        self.allow_cross_region_view_gain = bool(gp('allow_cross_region_view_gain').value)
        self.view_fov_deg = float(gp('view_fov_deg').value)
        self.view_max_range_m = float(gp('view_max_range_m').value)
        self.view_ray_step_m = float(gp('view_ray_step_m').value)
        self.view_angle_step_deg = float(gp('view_angle_step_deg').value)
        self.view_stop_at_unknown = bool(gp('view_stop_at_unknown').value)

        self.w_region_coverage_gain = float(gp('w_region_coverage_gain').value)
        self.w_region_frontier_gain = float(gp('w_region_frontier_gain').value)
        self.w_cross_region_unknown = float(gp('w_cross_region_unknown').value)
        self.w_cross_region_frontier = float(gp('w_cross_region_frontier').value)
        self.w_path_cost = float(gp('w_path_cost').value)
        self.w_clearance_bonus = float(gp('w_clearance_bonus').value)
        self.w_candidate_distance_cost = float(gp('w_candidate_distance_cost').value)
        self.w_candidate_distance_sq_cost = float(gp('w_candidate_distance_sq_cost').value)
        self.w_candidate_obstacle_density_cost = float(gp('w_candidate_obstacle_density_cost').value)
        self.w_candidate_unknown_density_cost = float(gp('w_candidate_unknown_density_cost').value)
        self.w_candidate_clearance_risk_cost = float(gp('w_candidate_clearance_risk_cost').value)
        self.candidate_obstacle_line_step_m = float(gp('candidate_obstacle_line_step_m').value)
        self.candidate_obstacle_check_radius_m = float(gp('candidate_obstacle_check_radius_m').value)
        self.region_candidate_gain_threshold = float(gp('region_candidate_gain_threshold').value)
        self.region_map_gain_threshold = float(gp('region_map_gain_threshold').value)
        self.region_coverage_gain_threshold = float(gp('region_coverage_gain_threshold').value)

        self.region_coverage_threshold = float(gp('region_coverage_threshold').value)
        self.region_frontier_threshold = int(gp('region_frontier_threshold').value)
        self.region_min_active_time_sec = float(gp('region_min_active_time_sec').value)
        self.region_max_active_time_sec = float(gp('region_max_active_time_sec').value)
        self.region_switch_on_stall = bool(gp('region_switch_on_stall').value)
        self.region_switch_gain_ratio = float(gp('region_switch_gain_ratio').value)
        self.region_switch_coverage_ratio = float(gp('region_switch_coverage_ratio').value)
        self.prefer_unvisited_region = bool(gp('prefer_unvisited_region').value)
        self.region_min_cells = int(gp('region_min_cells').value)
        self.active_region_iou_match_threshold = float(gp('active_region_iou_match_threshold').value)
        self.reopen_completed_region_on_frontier = bool(gp('reopen_completed_region_on_frontier').value)
        self.reopen_frontier_margin = float(gp('reopen_frontier_margin').value)
        self.clear_completed_regions_when_no_goal = bool(gp('clear_completed_regions_when_no_goal').value)

        self.w_next_region_uncovered = float(gp('w_next_region_uncovered').value)
        self.w_next_region_frontier = float(gp('w_next_region_frontier').value)
        self.w_next_region_unvisited = float(gp('w_next_region_unvisited').value)
        self.w_next_region_path_cost = float(gp('w_next_region_path_cost').value)
        self.next_region_entry_offset_m = float(gp('next_region_entry_offset_m').value)

        self.goal_blacklist_radius_m = float(gp('goal_blacklist_radius_m').value)
        self.goal_blacklist_time_sec = float(gp('goal_blacklist_time_sec').value)
        self.blacklist_failed_nav2_goals = bool(gp('blacklist_failed_nav2_goals').value)
        self.allow_nav2_scan_turn_goal = bool(gp('allow_nav2_scan_turn_goal').value)
        self.scan_turn_yaw_step_deg = float(gp('scan_turn_yaw_step_deg').value)
        self.allow_nav2_forward_probe_goal = bool(gp('allow_nav2_forward_probe_goal').value)
        self.forward_probe_distance_m = float(gp('forward_probe_distance_m').value)
        self.forward_probe_min_distance_m = float(gp('forward_probe_min_distance_m').value)
        self.forward_probe_lidar_margin_m = float(gp('forward_probe_lidar_margin_m').value)

        self._params_loaded = True
        self._last_param_refresh_time = self._now_sec()

    def _maybe_update_timer_period(self) -> None:
        if not getattr(self, 'dynamic_timer_period_update', True):
            return
        requested = max(0.02, float(self.timer_period_sec))
        current = float(getattr(self, '_active_timer_period_sec', requested))
        if abs(requested - current) < 0.002:
            return
        old_timer = getattr(self, 'timer', None)
        self._active_timer_period_sec = requested
        self.timer = self.create_timer(requested, self._on_timer, callback_group=self.planning_cb_group)
        if old_timer is not None:
            try:
                old_timer.cancel()
                self.destroy_timer(old_timer)
            except Exception:
                pass
        self.get_logger().warn(f'DYNAMIC_TIMER_PERIOD_UPDATE | timer_period_sec={requested:.3f}s')

    def _maybe_update_goal_dispatch_timer_period(self) -> None:
        if not getattr(self, 'dynamic_timer_period_update', True):
            return
        requested = max(0.02, float(self.goal_dispatch_period_sec))
        current = float(getattr(self, '_active_goal_dispatch_period_sec', requested))
        if abs(requested - current) < 0.002:
            return
        old_timer = getattr(self, 'goal_dispatch_timer', None)
        self._active_goal_dispatch_period_sec = requested
        self.goal_dispatch_timer = self.create_timer(
            requested,
            self._on_goal_dispatch_timer,
            callback_group=self.dispatch_cb_group,
        )
        if old_timer is not None:
            try:
                old_timer.cancel()
                self.destroy_timer(old_timer)
            except Exception:
                pass
        self.get_logger().warn(f'DYNAMIC_GOAL_DISPATCH_TIMER_PERIOD_UPDATE | goal_dispatch_period_sec={requested:.3f}s')

    def _maybe_update_coverage_timer_period(self) -> None:
        requested = max(0.02, float(self.coverage_update_period_sec))
        current = float(getattr(self, '_active_coverage_timer_period_sec', requested))
        if abs(requested - current) < 0.002:
            return
        old_timer = getattr(self, 'coverage_timer', None)
        self._active_coverage_timer_period_sec = requested
        self.coverage_timer = self.create_timer(requested, self._on_coverage_timer, callback_group=self.coverage_cb_group)
        if old_timer is not None:
            try:
                old_timer.cancel()
                self.destroy_timer(old_timer)
            except Exception:
                pass
        self.get_logger().warn(f'DYNAMIC_COVERAGE_TIMER_PERIOD_UPDATE | coverage_update_period_sec={requested:.3f}s')

    def _on_coverage_timer(self) -> None:
        if self._shutting_down or not rclpy.ok():
            return
        # Independent high-rate coverage painting.  This continues while Nav2 is
        # moving toward a goal and while semantic candidate scoring is busy.
        self._refresh_params()
        self._maybe_update_coverage_timer_period()
        robot = self._lookup_robot_pose()
        if robot is not None:
            self._update_coverage_live(robot)
        now = self._now_sec()
        if now - self.last_coverage_publish_time >= self.coverage_publish_period_sec:
            self.last_coverage_publish_time = now
            self._publish_coverage()

    # ------------------------------------------------------------------
    # ROS callbacks
    # ------------------------------------------------------------------
    def _on_map(self, msg: OccupancyGrid) -> None:
        self.map_msg = msg
        self.geom = MapGeom(
            width=msg.info.width,
            height=msg.info.height,
            resolution=msg.info.resolution,
            origin_x=msg.info.origin.position.x,
            origin_y=msg.info.origin.position.y,
        )
        sig = (self.geom.width, self.geom.height, self.geom.resolution, self.geom.origin_x, self.geom.origin_y)
        if sig != self._last_map_signature:
            old_cov = self.coverage
            old_geom = getattr(self, '_coverage_geom', None)
            self.coverage = [-1] * (self.geom.width * self.geom.height)
            if old_cov and old_geom is not None:
                self._transfer_coverage(old_cov, old_geom)
            self._coverage_geom = MapGeom(**self.geom.__dict__)
            self._last_coverage_pose = None
            self._last_map_signature = sig
            self.get_logger().info(
                f'NAV2_EXPLORER_GRID_RESET | size={self.geom.width}x{self.geom.height} '
                f'res={self.geom.resolution:.3f} origin=({self.geom.origin_x:.2f},{self.geom.origin_y:.2f})'
            )

    def _on_region_map(self, msg: OccupancyGrid) -> None:
        self.region_msg = msg

    def _on_scan(self, msg: LaserScan) -> None:
        self.scan_msg = msg

    def _on_timer(self) -> None:
        if self._shutting_down or not rclpy.ok():
            return
        self._refresh_params()
        self._maybe_update_timer_period()
        self._maybe_update_goal_dispatch_timer_period()
        now = self._now_sec()

        robot = self._lookup_robot_pose()

        # Coverage is handled by the dedicated live coverage timer.  Keeping it
        # out of the semantic planning loop prevents candidate generation from
        # blocking 10Hz coverage painting.

        if not self.auto_start:
            self._publish_state('PAUSED', robot, extra={'auto_start': False})
            return

        if not self._ready(robot):
            self._publish_state('WAIT_READY', robot)
            return

        if not self.nav_client.server_is_ready():
            self.nav_client.wait_for_server(timeout_sec=self.wait_nav2_server_timeout_sec)
            if not self.nav_client.server_is_ready():
                self._publish_state('WAIT_NAV2_ACTION_SERVER', robot)
                return

        self._expire_blacklist(now)
        if now - self.last_plan_time < self.planning_period_sec:
            return
        self.last_plan_time = now

        # Producer only: refresh/revalidate the persistent candidate priority
        # queue. Goal dispatch/preemption is handled by _on_goal_dispatch_timer,
        # so a slow candidate sweep never blocks live coverage or dispatch.
        planned = self._plan_next_candidate(robot, allow_scan_fallback=False)
        if planned is None:
            self._publish_state('NO_MEANINGFUL_CANDIDATE', robot)
        else:
            best = self.candidate_queue[0] if self.candidate_queue else self.last_best
            self._publish_markers(robot, best, self.last_stats)
        if planned is not None and not self.active_goal:
            # Send the first available goal immediately from the producer tick.
            # The separate dispatch timer still handles rolling updates, but
            # this prevents the robot from sitting still if the planning sweep
            # is heavy enough to starve the dispatch timer.
            self._dispatch_best_goal(robot)

    def _on_goal_dispatch_timer(self) -> None:
        if self._shutting_down or not rclpy.ok():
            return
        self._refresh_params()
        self._maybe_update_goal_dispatch_timer_period()
        now = self._now_sec()
        robot = self._lookup_robot_pose()

        if not self.auto_start:
            return
        if not self._ready(robot):
            return
        if not self.nav_client.server_is_ready():
            self.nav_client.wait_for_server(timeout_sec=self.wait_nav2_server_timeout_sec)
            if not self.nav_client.server_is_ready():
                return

        self._expire_blacklist(now)
        if self.active_goal and now - self.current_goal_sent_time > self.nav2_goal_timeout_sec:
            self._cancel_current_goal('NAV2_GOAL_TIMEOUT')
            return

        self._dispatch_best_goal(robot)

    def _dispatch_best_goal(self, robot: Pose2D) -> None:
        now = self._now_sec()
        top = self._top_candidate_from_queue(robot)
        if top is None:
            if not self.active_goal:
                top = self._make_forward_probe_goal(robot)
                if top is None and self.allow_nav2_scan_turn_goal:
                    top = self._make_scan_turn_goal(robot)
            else:
                if self.active_goal:
                    self._publish_state('NAV2_GOAL_ACTIVE_QUEUE_EMPTY', robot)
                return
            if top is None:
                self._publish_state('NO_NAV2_FALLBACK_GOAL', robot)
                return

        if not self.active_goal:
            if now - self.last_goal_time < self.min_goal_interval_sec:
                return
            if now - self.last_goal_dispatch_debug_time >= self.candidate_queue_log_throttle_sec:
                self.last_goal_dispatch_debug_time = now
                self.get_logger().info(
                    f'GOAL_DISPATCH_READY | type={top.candidate_type} reason={top.reason} '
                    f'x={top.x:.2f} y={top.y:.2f} active_goal={self.active_goal}'
                )
            self._send_nav2_goal(top, top.reason)
            self._publish_markers(robot, top if 'SCAN_TURN' not in top.reason else None, self.last_stats)
            return

        if not self.rolling_goal_update_enabled:
            self._publish_state('NAV2_GOAL_ACTIVE_ROLLING_DISABLED', robot)
            return
        if now - self.last_goal_time < self.rolling_goal_preempt_cooldown_sec:
            return
        if self._should_preempt_current_goal(robot, top):
            self._send_nav2_goal(top, top.reason + '_ROLLING_UPDATE')
            self._publish_markers(robot, top, self.last_stats)
            return
        self._publish_state('NAV2_GOAL_ACTIVE_ROLLING', robot)

    # ------------------------------------------------------------------
    # Planning / rolling goal selection
    # ------------------------------------------------------------------
    def _plan_next_candidate(self, robot: Pose2D, allow_scan_fallback: bool = True) -> Optional[Tuple[Candidate, str, Dict[int, RegionStats]]]:
        """Return the next semantic Nav2 goal candidate without sending it.

        This is shared by normal planning and rolling-goal preemption so the
        robot does not wait for a full NavigateToPose success before thinking
        about the next viewpoint.
        """
        if self.candidate_priority_queue_enabled:
            return self._plan_next_candidate_from_priority_queue(robot, allow_scan_fallback=allow_scan_fallback)

        stats = self._compute_region_stats()
        self.last_stats = stats
        self._reopen_completed_regions(stats)

        active = self._select_or_track_active_region(robot, stats)
        if active is None:
            goal = self._select_global_frontier_goal(robot, stats)
            if goal is not None:
                return goal, 'GLOBAL_FRONTIER_FALLBACK', stats
            if allow_scan_fallback and self.allow_nav2_scan_turn_goal:
                return self._make_scan_turn_goal(robot), 'NAV2_SCAN_TURN_FALLBACK', stats
            return None

        self.active_region = active
        self.visited_regions.add(active)

        best_map = self._find_best_candidate_by_mode(robot, active, stats, mode='map')
        best_cov = self._find_best_candidate_by_mode(robot, active, stats, mode='coverage')
        best = best_map if (best_map is not None and (best_cov is None or best_map.score >= best_cov.score)) else best_cov
        self.last_best = best
        region_age = self._now_sec() - self.active_region_started
        active_stats = stats.get(active)

        forced_switch_reason = self._should_switch_region(active, active_stats, best, region_age, stats)
        if forced_switch_reason:
            next_goal = self._select_next_region_goal(robot, active, stats)
            if next_goal is not None:
                self.completed_regions.add(active)
                next_goal.reason = forced_switch_reason
                return next_goal, forced_switch_reason, stats

        if best_map is not None and best_map.score >= self.region_map_gain_threshold:
            return best_map, 'REGION_MAP_EXPAND_GOAL', stats

        if best_cov is not None and best_cov.score >= self.region_coverage_gain_threshold:
            return best_cov, 'REGION_COVERAGE_FILL_GOAL', stats

        if best_map is not None and active_stats is not None and active_stats.frontier > self.region_frontier_threshold:
            return best_map, 'REGION_MAP_EXPAND_LOW_GAIN', stats

        if self._region_complete(active, active_stats, best, region_age):
            self.completed_regions.add(active)
            next_goal = self._select_next_region_goal(robot, active, stats)
            if next_goal is not None:
                return next_goal, 'NEXT_REGION_ENTRY_GOAL', stats

            global_goal = self._select_global_frontier_goal(robot, stats)
            if global_goal is not None:
                if self.clear_completed_regions_when_no_goal:
                    self.completed_regions.clear()
                return global_goal, 'GLOBAL_FRONTIER_FALLBACK', stats

            if allow_scan_fallback and self.allow_nav2_scan_turn_goal:
                if self.clear_completed_regions_when_no_goal:
                    self.completed_regions.clear()
                return self._make_scan_turn_goal(robot), 'NAV2_SCAN_TURN_FALLBACK', stats

            return None

        global_goal = self._select_global_frontier_goal(robot, stats)
        if global_goal is not None:
            return global_goal, 'GLOBAL_FRONTIER_FALLBACK_UNBLOCK', stats

        if allow_scan_fallback and self.allow_nav2_scan_turn_goal:
            return self._make_scan_turn_goal(robot), 'NAV2_SCAN_TURN_FALLBACK', stats

        return None

    def _plan_next_candidate_from_priority_queue(self, robot: Pose2D, allow_scan_fallback: bool = True) -> Optional[Tuple[Candidate, str, Dict[int, RegionStats]]]:
        """Fast candidate selection through a rolling top-K priority queue.

        The queue stores about candidate_queue_size semantic candidates.  It is
        rebuilt at a moderate rate from the current region/global frontier pool,
        then re-scored at high rate against live coverage/map/robot pose.  This
        avoids waiting for a full candidate sweep before every rolling Nav2 goal.
        """
        now = self._now_sec()
        stats = self._compute_region_stats()
        self.last_stats = stats
        self._reopen_completed_regions(stats)

        active = self._select_or_track_active_region(robot, stats)
        if active is None:
            self.candidate_queue = []
            goal = self._select_global_frontier_goal(robot, stats)
            if goal is not None:
                stamped = self._stamp_candidate(goal, 0, refreshed=True)
                self.candidate_queue = self._prune_and_sort_candidates([stamped], robot, 0)[:1]
                self.last_candidate_queue_refresh_time = now
                self.last_candidate_queue_revalidate_time = now
                return goal, 'GLOBAL_FRONTIER_FALLBACK_PQ_EMPTY', stats
            if allow_scan_fallback and self.allow_nav2_scan_turn_goal:
                scan_turn = self._make_scan_turn_goal(robot)
                self.candidate_queue = [self._stamp_candidate(scan_turn, 0, refreshed=True)]
                self.last_candidate_queue_refresh_time = now
                self.last_candidate_queue_revalidate_time = now
                return scan_turn, 'NAV2_SCAN_TURN_FALLBACK', stats
            return None

        self.active_region = active
        self.visited_regions.add(active)

        must_refresh = (
            not self.candidate_queue
            or (now - self.last_candidate_queue_refresh_time) >= self.candidate_queue_refresh_period_sec
        )
        if must_refresh:
            self._rebuild_candidate_priority_queue(robot, active, stats)
        elif (now - self.last_candidate_queue_revalidate_time) >= self.candidate_queue_revalidate_period_sec:
            self._rescore_candidate_priority_queue(robot, active, stats)

        if self.candidate_queue:
            top = self._top_candidate_from_queue(robot)
            if top is not None and top.score >= self.candidate_queue_min_score:
                if now - self.last_candidate_queue_log_time >= self.candidate_queue_log_throttle_sec:
                    self.last_candidate_queue_log_time = now
                    self.get_logger().info(
                        f'CANDIDATE_PQ_TOP | type={top.candidate_type} score={top.dynamic_score:.1f} '
                        f'base={top.base_score:.1f} n={len(self.candidate_queue)} active={active} '
                        f'reason={top.reason} x={top.x:.2f} y={top.y:.2f} '
                        f'cov_gain={top.region_cov_gain} rfr={top.region_frontier_gain} '
                        f'xunk={top.cross_unknown_gain} xfr={top.cross_frontier_gain} '
                        f'dcost={top.distance_cost:.1f} ocost={top.obstacle_cost:.1f} '
                        f'ucost={top.unknown_cost:.1f} crisk={top.clearance_risk_cost:.1f}'
                    )
                return top, top.reason + '_PQ', stats

        # Queue is empty or all priorities collapsed after live re-score.
        self.candidate_queue = []
        global_goal = self._select_global_frontier_goal(robot, stats)
        if global_goal is not None:
            return global_goal, 'GLOBAL_FRONTIER_FALLBACK_PQ_REBUILD_FAIL', stats
        if allow_scan_fallback and self.allow_nav2_scan_turn_goal:
            return self._make_scan_turn_goal(robot), 'NAV2_SCAN_TURN_FALLBACK', stats
        return None

    def _rebuild_candidate_priority_queue(self, robot: Pose2D, active: int, stats: Dict[int, RegionStats]) -> None:
        now = self._now_sec()
        raw: List[Candidate] = []

        # Active-region map expansion and coverage candidates.
        for c in self._candidate_cells_for_region(robot, active):
            wx, wy = self._cell_to_world(c)
            if self._is_blacklisted(wx, wy):
                continue
            map_cand = self._score_candidate_mode(robot, c, active, mode='map')
            if map_cand is not None:
                raw.append(map_cand)
            cov_cand = self._score_candidate_mode(robot, c, active, mode='coverage')
            if cov_cand is not None:
                raw.append(cov_cand)

        # Keep next-region and global-frontier options in the same priority pool
        # so switching does not wait for a full active-region failure cycle.
        if self.candidate_queue_include_next_region:
            nxt = self._select_next_region_goal(robot, active, stats)
            if nxt is not None:
                raw.append(nxt)
        if self.candidate_queue_include_global_frontier:
            glob = self._select_global_frontier_goal(robot, stats)
            if glob is not None:
                raw.append(glob)

        old_by_key = {self._candidate_key(c): c for c in self.candidate_queue}
        updated = 0
        stamped: List[Candidate] = []
        refreshed_keys: Set[str] = set()
        for cand in raw:
            stamped_cand = self._stamp_candidate(cand, active, old=None, refreshed=True)
            old = old_by_key.get(stamped_cand.key)
            if old is not None:
                stamped_cand = self._stamp_candidate(stamped_cand, active, key=stamped_cand.key, old=old, refreshed=True)
            refreshed_keys.add(stamped_cand.key)
            stamped.append(stamped_cand)
            updated += 1
        # Persistent queue: keep still-valid old entries that were not refreshed
        # this producer tick, then let the fast revalidator rank/prune them.
        for key, old in old_by_key.items():
            if key not in refreshed_keys:
                stamped.append(old)
        self.candidate_queue = self._prune_and_sort_candidates(stamped, robot, active)[: max(1, self.candidate_queue_size)]
        self.last_candidate_queue_refresh_time = now
        self.last_candidate_queue_revalidate_time = now
        if now - self.last_candidate_queue_log_time >= self.candidate_queue_log_throttle_sec:
            self.last_candidate_queue_log_time = now
            top = self.candidate_queue[0] if self.candidate_queue else None
            self.get_logger().info(
                f'CANDIDATE_PQ_REFRESH | raw={len(raw)} updated={updated} kept={len(self.candidate_queue)} active={active} '
                + (f'top={top.candidate_type} score={top.dynamic_score:.1f} x={top.x:.2f} y={top.y:.2f}' if top else 'top=None')
            )

        # Build the lookahead route queue from the NMS-filtered candidate pool.
        # This is the expensive beam-search pass; it runs only here (moderate
        # rate), not in the high-rate rescore path.
        if self.route_queue_enabled and self.candidate_queue:
            pool = self._spatial_nms_candidates(
                list(self.candidate_queue[: max(1, self.route_candidate_pool_size)]),
                self.route_spatial_suppression_m,
            )
            self.route_queue = self._build_route_beam_search(robot, pool, active)
            self.last_route_queue_build_time = now
            if self.route_queue and now - self.last_candidate_queue_log_time < 0.01:
                self.get_logger().info(
                    f'ROUTE_QUEUE_BUILT | horizon={len(self.route_queue)} pool={len(pool)} '
                    + ' | '.join(
                        f'G{i+1}={c.candidate_type}({c.x:.2f},{c.y:.2f})s={c.score:.1f}'
                        for i, c in enumerate(self.route_queue)
                    )
                )

    def _rescore_candidate_priority_queue(self, robot: Pose2D, active: int, stats: Dict[int, RegionStats]) -> None:
        """Cheap, high-rate re-score of the existing queue.

        The expensive part of scoring is the view-gain raycast in _view_gain.
        The information gain of a target barely changes between two re-score
        ticks ~50 ms apart, but the robot pose (hence distance and the
        obstacle/clearance penalty along the path) does change. So we REUSE the
        cached view-gains and re-evaluate only the cheap, pose-dependent terms.
        This is what lets the priority queue actually update the goal quickly
        instead of paying a full raycast sweep every tick."""
        rescored: List[Candidate] = []
        for old in self.candidate_queue:
            if old.cell is None or not self._in_bounds(old.cell):
                continue
            wx, wy = old.x, old.y
            if self._is_blacklisted(wx, wy):
                continue
            dist = self._distance_world((robot[0], robot[1]), (wx, wy))
            if dist < 0.20:
                continue
            if self.candidate_max_goal_distance_m > 0.0 and dist > self.candidate_max_goal_distance_m:
                continue
            # Live penalty re-evaluation keeps "too far" (distance) and "too many
            # obstacles" (obstacle density along the path) honest as the robot
            # moves, without recomputing information gain.
            penalty, distance_cost, obstacle_cost, unknown_cost, clearance_risk_cost = self._candidate_penalty(
                robot, old.cell, wx, wy
            )
            score = self._compose_score(
                old.score_mode,
                old.region_cov_gain,
                old.region_frontier_gain,
                old.cross_unknown_gain,
                old.cross_frontier_gain,
                old.cached_clearance,
                dist,
                penalty,
            )
            cand = Candidate(
                    cell=old.cell,
                    x=wx,
                    y=wy,
                    yaw=old.yaw,
                    score=score,
                    region_cov_gain=old.region_cov_gain,
                    region_frontier_gain=old.region_frontier_gain,
                    cross_unknown_gain=old.cross_unknown_gain,
                    cross_frontier_gain=old.cross_frontier_gain,
                    distance=dist,
                    reason=old.reason,
                    distance_cost=distance_cost,
                    obstacle_cost=obstacle_cost,
                    unknown_cost=unknown_cost,
                    clearance_risk_cost=clearance_risk_cost,
                    key=old.key,
                    candidate_type=old.candidate_type,
                    target_region_id=old.target_region_id,
                    base_score=old.base_score,
                    dynamic_score=score,
                    created_time=old.created_time,
                    last_refresh_time=old.last_refresh_time,
                    last_revalidate_time=self._now_sec(),
                    stale=old.stale,
                    blacklisted=old.blacklisted,
                    debug=old.debug,
                    score_mode=old.score_mode,
                    cached_clearance=old.cached_clearance,
                )
            rescored.append(cand)
        self.candidate_queue = self._prune_and_sort_candidates(rescored, robot, active)[: max(1, self.candidate_queue_size)]
        self.last_candidate_queue_revalidate_time = self._now_sec()

    def _dedupe_and_sort_candidates(self, candidates: Sequence[Candidate], robot: Pose2D) -> List[Candidate]:
        best_by_key: Dict[Tuple[int, int, str], Candidate] = {}
        for cand in candidates:
            if cand is None:
                continue
            if not self._in_bounds(cand.cell):
                continue
            # Use a coarse spatial key so the queue contains diverse targets
            # rather than 30 yaw variants of the same cell.
            qx = int(round(cand.x / 0.20))
            qy = int(round(cand.y / 0.20))
            mode = 'coverage' if 'COVERAGE' in cand.reason else ('next' if 'NEXT_REGION' in cand.reason else ('global' if 'GLOBAL' in cand.reason else 'map'))
            key = (qx, qy, mode)
            old = best_by_key.get(key)
            if old is None or cand.score > old.score:
                best_by_key[key] = cand
        out = list(best_by_key.values())
        # Mild distance tiebreaker only.  Score remains dominant.
        out.sort(key=lambda c: (c.score, -self._distance_world((robot[0], robot[1]), (c.x, c.y))), reverse=True)
        return out

    def _candidate_type_from_reason(self, reason: str) -> str:
        if 'MAP_EXPAND' in reason:
            return 'MAP_EXPAND'
        if 'COVERAGE' in reason:
            return 'COVERAGE_FILL'
        if 'NEXT_REGION' in reason:
            return 'NEXT_REGION'
        if 'GLOBAL_FRONTIER' in reason:
            return 'GLOBAL_FRONTIER'
        return 'GLOBAL_FRONTIER'

    def _candidate_key(self, cand: Candidate) -> str:
        if cand.key:
            return cand.key
        ctype = self._candidate_type_from_reason(cand.reason)
        qx = int(round(cand.x / 0.20))
        qy = int(round(cand.y / 0.20))
        qyaw = int(round(self._angle_wrap(cand.yaw) / 0.35))
        return f'{ctype}:{cand.target_region_id}:{qx}:{qy}:{qyaw}'

    def _stamp_candidate(
        self,
        cand: Candidate,
        active: int,
        key: Optional[str] = None,
        old: Optional[Candidate] = None,
        refreshed: bool = False,
    ) -> Candidate:
        now = self._now_sec()
        ctype = self._candidate_type_from_reason(cand.reason)
        base_score = (
            self.w_region_coverage_gain * cand.region_cov_gain
            + self.w_region_frontier_gain * cand.region_frontier_gain
            + self.w_cross_region_unknown * cand.cross_unknown_gain
            + self.w_cross_region_frontier * cand.cross_frontier_gain
            + self.w_clearance_bonus * cand.cached_clearance
        )
        cand.candidate_type = ctype
        cand.target_region_id = active if ctype in ('MAP_EXPAND', 'COVERAGE_FILL') else int(cand.target_region_id or 0)
        cand.key = key or self._candidate_key(cand)
        cand.base_score = base_score
        cand.dynamic_score = cand.score
        cand.created_time = old.created_time if old is not None and old.created_time > 0.0 else now
        cand.last_refresh_time = now if refreshed else (old.last_refresh_time if old is not None else now)
        cand.last_revalidate_time = now
        cand.stale = False
        cand.blacklisted = False
        cand.debug = (
            f'{ctype} rgain={cand.region_cov_gain} rfr={cand.region_frontier_gain} '
            f'xunk={cand.cross_unknown_gain} xfr={cand.cross_frontier_gain}'
        )
        return cand

    def _prune_and_sort_candidates(self, candidates: Sequence[Candidate], robot: Pose2D, active: int) -> List[Candidate]:
        now = self._now_sec()
        counts: Dict[str, int] = defaultdict(int)
        best_by_key: Dict[str, Candidate] = {}
        for cand in candidates:
            key = self._candidate_key(cand)
            cand.key = key
            if now - cand.created_time > self.candidate_queue_max_age_sec:
                counts['max_age'] += 1
                continue
            if cand.cell is None or not self._in_bounds(cand.cell):
                counts['invalid_map_cell'] += 1
                continue
            if self._is_blacklisted(cand.x, cand.y):
                counts['blacklist'] += 1
                continue
            if self._is_occupied(cand.cell):
                counts['occupied_target'] += 1
                continue
            if self._is_unknown(cand.cell) and cand.candidate_type != 'MAP_EXPAND':
                counts['unknown_target'] += 1
                continue
            if cand.candidate_type != 'MAP_EXPAND' and not self._has_clearance(cand.cell, self.candidate_min_clearance_m):
                counts['low_clearance'] += 1
                continue
            if self.candidate_max_goal_distance_m > 0.0 and cand.distance > self.candidate_max_goal_distance_m:
                counts['too_far'] += 1
                continue
            if cand.score < self.candidate_queue_min_score:
                counts['low_score'] += 1
                continue
            old = best_by_key.get(key)
            if old is None or cand.score > old.score:
                best_by_key[key] = cand

        out = list(best_by_key.values())
        priority = {'MAP_EXPAND': 3, 'COVERAGE_FILL': 2, 'NEXT_REGION': 1, 'GLOBAL_FRONTIER': 0}
        out.sort(
            key=lambda c: (
                priority.get(c.candidate_type, 0),
                c.score,
                -self._distance_world((robot[0], robot[1]), (c.x, c.y)),
            ),
            reverse=True,
        )
        if counts and now - self.last_candidate_prune_log_time >= self.candidate_queue_log_throttle_sec:
            self.last_candidate_prune_log_time = now
            for reason, count in counts.items():
                if count > 0 and rclpy.ok() and not self._shutting_down:
                    self.get_logger().info(f'CANDIDATE_PQ_PRUNE | reason={reason} count={count}')
        return out

    def _top_candidate_from_queue(self, robot: Pose2D) -> Optional[Candidate]:
        active = self.active_region or 0
        if self.candidate_queue:
            self.candidate_queue = self._prune_and_sort_candidates(self.candidate_queue, robot, active)[: max(1, self.candidate_queue_size)]
        if not self.candidate_queue:
            return None
        top = self.candidate_queue[0]
        now = self._now_sec()
        if now - self.last_candidate_queue_log_time >= self.candidate_queue_log_throttle_sec:
            self.last_candidate_queue_log_time = now
            self.get_logger().info(
                f'CANDIDATE_PQ_TOP | type={top.candidate_type} score={top.dynamic_score:.1f} '
                f'base={top.base_score:.1f} dcost={top.distance_cost:.1f} ocost={top.obstacle_cost:.1f} '
                f'ucost={top.unknown_cost:.1f} crisk={top.clearance_risk_cost:.1f} key={top.key}'
            )
        return top

    def _should_preempt_current_goal(self, robot: Pose2D, candidate: Candidate) -> bool:
        now = self._now_sec()
        elapsed_since_goal = now - self.last_goal_time
        goal_age = now - self.current_goal_sent_time
        if elapsed_since_goal < max(self.rolling_goal_period_sec, self.rolling_goal_preempt_cooldown_sec):
            return False
        if self.current_goal_pose is None:
            return True

        gx = self.current_goal_pose.pose.position.x
        gy = self.current_goal_pose.pose.position.y
        goal_yaw = _yaw_from_quaternion_msg(self.current_goal_pose.pose.orientation)
        dist_to_current = self._distance_world((robot[0], robot[1]), (gx, gy))
        cand_shift = self._distance_world((candidate.x, candidate.y), (gx, gy))
        yaw_delta = abs(self._angle_wrap(candidate.yaw - goal_yaw))
        prev_score = self.current_goal_score
        score_improved = candidate.score >= prev_score + self.rolling_goal_min_score_improvement
        current_type = self._candidate_type_from_reason(self.current_goal_reason)
        reason_changed = candidate.candidate_type != current_type

        # Main anti stop-go rule: preempt well before the controller fully
        # satisfies the current NavigateToPose goal.  With the previous logic the
        # robot often reached the goal checker, stopped, then waited for a new
        # semantic goal.  This one starts streaming the next goal when the robot
        # is within a broad approach radius.
        if dist_to_current <= self.rolling_goal_near_distance_m and cand_shift >= self.rolling_goal_min_shift_m:
            self.get_logger().info(
                f'ROLLING_GOAL_PREEMPT_NEAR | dist_current={dist_to_current:.2f} '
                f'shift={cand_shift:.2f} old_score={prev_score:.1f} new_score={candidate.score:.1f} '
                f'goal_age={goal_age:.2f}s'
            )
            return True

        # If the semantic mode changes, do not keep following an old coverage
        # target while a map-expansion or next-region target is available.
        if reason_changed and cand_shift >= self.rolling_goal_reason_change_min_shift_m:
            self.get_logger().info(
                f'ROLLING_GOAL_PREEMPT_REASON_CHANGE | shift={cand_shift:.2f} '
                f'old_reason={self.current_goal_reason} new_reason={candidate.reason}'
            )
            return True

        # Also allow preemption if a substantially better map/coverage target
        # appears due to Cartographer map growth.
        if score_improved and cand_shift >= self.rolling_goal_min_shift_m:
            self.get_logger().info(
                f'ROLLING_GOAL_PREEMPT_BETTER | dist_current={dist_to_current:.2f} '
                f'shift={cand_shift:.2f} old_score={prev_score:.1f} new_score={candidate.score:.1f}'
            )
            return True

        # If the pose is nearly identical but the desired viewing direction
        # changes, still update the goal orientation.  This is important for
        # front-only coverage and Cartographer map expansion at doorways.
        if dist_to_current <= self.rolling_goal_force_near_distance_m and yaw_delta >= self.rolling_goal_yaw_change_rad:
            self.get_logger().info(
                f'ROLLING_GOAL_PREEMPT_YAW | dist_current={dist_to_current:.2f} '
                f'yaw_delta={math.degrees(yaw_delta):.1f}deg shift={cand_shift:.2f}'
            )
            return True

        # Last resort: force periodic preemption while approaching a goal if the
        # next semantic target is not exactly the same.  This prevents long waits
        # for Nav2 action success/result callbacks before the explorer streams a
        # new target.
        if (
            self.rolling_goal_force_period_sec > 0.0
            and goal_age >= self.rolling_goal_force_period_sec
            and dist_to_current <= self.rolling_goal_force_near_distance_m
            and cand_shift >= max(0.05, self.rolling_goal_reason_change_min_shift_m)
        ):
            self.get_logger().info(
                f'ROLLING_GOAL_PREEMPT_FORCE_PERIODIC | goal_age={goal_age:.2f}s '
                f'dist_current={dist_to_current:.2f} shift={cand_shift:.2f}'
            )
            return True

        # Throttled diagnostic so slow updates are visible in logs.
        if now - getattr(self, 'last_rolling_debug_time', 0.0) >= self.rolling_goal_debug_throttle_sec:
            self.last_rolling_debug_time = now
            self.get_logger().info(
                f'ROLLING_GOAL_HOLD | age={goal_age:.2f}s since_last={elapsed_since_goal:.2f}s '
                f'dist_current={dist_to_current:.2f} shift={cand_shift:.2f} '
                f'yaw_delta={math.degrees(yaw_delta):.1f} old_score={prev_score:.1f} new_score={candidate.score:.1f}'
            )
        return False

    # ------------------------------------------------------------------
    # Action handling
    # ------------------------------------------------------------------
    def _send_nav2_goal(self, candidate: Candidate, reason: str) -> None:
        now = self._now_sec()
        pose = self._candidate_to_pose(candidate)
        goal_msg = NavigateToPose.Goal()
        goal_msg.pose = pose

        self.goal_seq += 1
        seq = self.goal_seq
        self.current_goal_seq = seq
        self.active_goal = True
        self.current_goal_pose = pose
        self.current_goal_reason = reason
        self.current_goal_sent_time = now
        self.last_goal_time = now
        self.last_best = candidate
        self.current_goal_score = candidate.score

        self.goal_debug_pub.publish(pose)
        self.get_logger().info(
            f'NAV2_SEND_GOAL | seq={seq} type={candidate.candidate_type} reason={reason} '
            f'x={candidate.x:.2f} y={candidate.y:.2f} yaw={candidate.yaw:.2f} '
            f'score={candidate.score:.1f} base={candidate.base_score:.1f} cov_gain={candidate.region_cov_gain} '
            f'rfr={candidate.region_frontier_gain} xunk={candidate.cross_unknown_gain} xfr={candidate.cross_frontier_gain}'
        )
        future = self.nav_client.send_goal_async(goal_msg, feedback_callback=self._on_nav_feedback)
        future.add_done_callback(lambda fut, seq=seq: self._on_goal_response(fut, seq))
        self._publish_state(reason, self._lookup_robot_pose())

    def _on_goal_response(self, future, seq: int) -> None:
        if seq != self.current_goal_seq:
            self.get_logger().info(f'NAV2_STALE_GOAL_RESPONSE_IGNORED | seq={seq} current={self.current_goal_seq}')
            return
        try:
            goal_handle = future.result()
        except Exception as exc:
            self.get_logger().error(f'NAV2_GOAL_RESPONSE_EXCEPTION | {exc}')
            self._finish_goal(False, 'GOAL_RESPONSE_EXCEPTION')
            return
        if not goal_handle.accepted:
            self.get_logger().warn('NAV2_GOAL_REJECTED')
            self._finish_goal(False, 'GOAL_REJECTED')
            return
        self.goal_handle = goal_handle
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(lambda fut, seq=seq: self._on_nav_result(fut, seq))

    def _on_nav_feedback(self, _feedback_msg) -> None:
        # Feedback can be added to state later. Keep callback light.
        pass

    def _on_nav_result(self, future, seq: int) -> None:
        if seq != self.current_goal_seq:
            self.get_logger().info(f'NAV2_STALE_RESULT_IGNORED | seq={seq} current={self.current_goal_seq}')
            return
        try:
            result = future.result()
            status = result.status
        except Exception as exc:
            self.get_logger().error(f'NAV2_RESULT_EXCEPTION | {exc}')
            self._finish_goal(False, 'RESULT_EXCEPTION')
            return

        if status == GoalStatus.STATUS_SUCCEEDED:
            self.get_logger().info(f'NAV2_GOAL_SUCCEEDED | reason={self.current_goal_reason}')
            self._finish_goal(True, 'SUCCEEDED')
        else:
            self.get_logger().warn(f'NAV2_GOAL_FAILED | status={status} reason={self.current_goal_reason}')
            self._finish_goal(False, f'FAILED_STATUS_{status}')

    def _cancel_current_goal(self, reason: str) -> None:
        self.get_logger().warn(f'NAV2_CANCEL_GOAL | reason={reason}')
        if self.goal_handle is not None:
            try:
                self.goal_handle.cancel_goal_async()
            except Exception:
                pass
        self._finish_goal(False, reason)

    def _finish_goal(self, success: bool, reason: str) -> None:
        if (not success) and self.blacklist_failed_nav2_goals and self.current_goal_pose is not None:
            x = self.current_goal_pose.pose.position.x
            y = self.current_goal_pose.pose.position.y
            self.blacklist.append((x, y, self._now_sec() + self.goal_blacklist_time_sec))
        self.active_goal = False
        self.goal_handle = None
        self.current_goal_pose = None
        self.current_goal_reason = ''
        self.current_goal_seq = 0
        self.current_goal_score = -1e9
        self.last_state = f'NAV2_GOAL_{reason}'

    # ------------------------------------------------------------------
    # Readiness / transforms
    # ------------------------------------------------------------------
    def _ready(self, robot: Optional[Pose2D]) -> bool:
        return (
            self.map_msg is not None
            and self.region_msg is not None
            and self.scan_msg is not None
            and robot is not None
            and self.geom.width > 0
            and bool(self.coverage)
        )

    def _lookup_robot_pose(self) -> Optional[Pose2D]:
        try:
            tf = self.tf_buffer.lookup_transform(
                self.global_frame,
                self.robot_frame,
                Time(),
                timeout=Duration(seconds=0.03),
            )
            x = tf.transform.translation.x
            y = tf.transform.translation.y
            q = tf.transform.rotation
            yaw = _yaw_from_quaternion_msg(q)
            return (x, y, yaw)
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Geometry / map utilities
    # ------------------------------------------------------------------
    def _now_sec(self) -> float:
        if self.get_clock().ros_time_is_active:
            return self.get_clock().now().nanoseconds * 1e-9
        return time.time()

    def _in_bounds(self, c: Cell) -> bool:
        return 0 <= c[0] < self.geom.width and 0 <= c[1] < self.geom.height

    def _idx(self, c: Cell) -> int:
        return c[1] * self.geom.width + c[0]

    def _world_to_cell(self, x: float, y: float) -> Optional[Cell]:
        if self.geom.resolution <= 0.0:
            return None
        ix = int(math.floor((x - self.geom.origin_x) / self.geom.resolution))
        iy = int(math.floor((y - self.geom.origin_y) / self.geom.resolution))
        c = (ix, iy)
        return c if self._in_bounds(c) else None

    def _cell_to_world(self, c: Cell) -> Tuple[float, float]:
        return (
            self.geom.origin_x + (c[0] + 0.5) * self.geom.resolution,
            self.geom.origin_y + (c[1] + 0.5) * self.geom.resolution,
        )

    def _map_value(self, c: Cell) -> int:
        if self.map_msg is None or not self._in_bounds(c):
            return 100
        return int(self.map_msg.data[self._idx(c)])

    def _region_value(self, c: Cell) -> int:
        if self.region_msg is None or not self._in_bounds(c):
            return 0
        if self.region_msg.info.width != self.geom.width or self.region_msg.info.height != self.geom.height:
            # The region graph normally uses the same grid. If it lags during a
            # Cartographer resize, ignore until it catches up.
            return 0
        return int(self.region_msg.data[self._idx(c)])

    def _is_free(self, c: Cell) -> bool:
        v = self._map_value(c)
        return 0 <= v < self.free_threshold

    def _is_unknown(self, c: Cell) -> bool:
        return self._map_value(c) < 0

    def _is_occupied(self, c: Cell) -> bool:
        return self._map_value(c) >= self.occupied_threshold

    def _neighbors4(self, c: Cell) -> Iterable[Cell]:
        x, y = c
        for nb in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
            if self._in_bounds(nb):
                yield nb

    def _neighbors8(self, c: Cell) -> Iterable[Cell]:
        x, y = c
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                nb = (x + dx, y + dy)
                if self._in_bounds(nb):
                    yield nb

    def _covered(self, c: Cell) -> bool:
        # Read into a local first: under the MultiThreadedExecutor the coverage
        # group may rebind self.coverage to a different-length list on a map
        # resize while the planning thread is iterating here. Binding locally
        # plus an explicit length check keeps this read crash-proof.
        cov = self.coverage
        if not cov or not self._in_bounds(c):
            return False
        i = self._idx(c)
        return 0 <= i < len(cov) and cov[i] == 100

    def _distance_world(self, a: Tuple[float, float], b: Tuple[float, float]) -> float:
        return math.hypot(a[0] - b[0], a[1] - b[1])

    def _angle_wrap(self, a: float) -> float:
        while a > math.pi:
            a -= 2.0 * math.pi
        while a < -math.pi:
            a += 2.0 * math.pi
        return a

    def _clearance_m(self, c: Cell, max_cells: Optional[int] = None) -> float:
        if max_cells is None:
            max_cells = max(1, int(math.ceil(1.20 / max(self.geom.resolution, 1e-6))))
        best2 = max_cells * max_cells
        cx, cy = c
        for dy in range(-max_cells, max_cells + 1):
            for dx in range(-max_cells, max_cells + 1):
                nb = (cx + dx, cy + dy)
                if not self._in_bounds(nb):
                    continue
                if self._is_occupied(nb):
                    d2 = dx * dx + dy * dy
                    if d2 < best2:
                        best2 = d2
        return math.sqrt(best2) * self.geom.resolution

    def _has_clearance(self, c: Cell, clearance_m: float) -> bool:
        radius = max(1, int(math.ceil(clearance_m / max(self.geom.resolution, 1e-6))))
        cx, cy = c
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                if dx * dx + dy * dy > radius * radius:
                    continue
                nb = (cx + dx, cy + dy)
                if not self._in_bounds(nb) or self._is_occupied(nb):
                    return False
        return True

    def _is_frontier_cell(self, c: Cell, region_label: Optional[int] = None) -> bool:
        if not self._is_free(c):
            return False
        if region_label is not None and self._region_value(c) != region_label:
            return False
        unk = 0
        for nb in self._neighbors4(c):
            if self._is_unknown(nb):
                unk += 1
        return unk >= self.frontier_candidate_min_unknown_neighbors

    def _transfer_coverage(self, old_cov: List[int], old_geom: MapGeom) -> None:
        if not old_cov:
            return
        for oy in range(old_geom.height):
            for ox in range(old_geom.width):
                oi = oy * old_geom.width + ox
                if old_cov[oi] != 100:
                    continue
                wx = old_geom.origin_x + (ox + 0.5) * old_geom.resolution
                wy = old_geom.origin_y + (oy + 0.5) * old_geom.resolution
                nc = self._world_to_cell(wx, wy)
                if nc is not None:
                    self.coverage[self._idx(nc)] = 100

    # ------------------------------------------------------------------
    # Coverage
    # ------------------------------------------------------------------
    def _update_coverage_live(self, robot: Pose2D) -> None:
        """Paint coverage continuously at the coverage timer rate.

        RViz should not show isolated sector wedges only at old Nav2 goals.  This
        interpolates between the previous and current robot pose and paints each
        intermediate scan pose, so coverage accumulates along the motion path.
        """
        t0 = time.time()
        before = self._covered_cell_count()
        samples = 1

        if not self.coverage_interpolate_motion_enabled:
            self._update_coverage(robot)
            self._last_coverage_pose = robot
            self._log_live_coverage(samples, self._covered_cell_count() - before, (time.time() - t0) * 1000.0, 0.0, 0.0)
            return

        last = self._last_coverage_pose
        if last is None:
            self._update_coverage(robot)
            self._last_coverage_pose = robot
            self._log_live_coverage(samples, self._covered_cell_count() - before, (time.time() - t0) * 1000.0, 0.0, 0.0)
            return

        dx = robot[0] - last[0]
        dy = robot[1] - last[1]
        dist = math.hypot(dx, dy)
        dyaw = self._angle_wrap(robot[2] - last[2])

        # A large jump is usually a reset/spawn/TF recovery.  Do not smear a
        # false coverage strip across the map.
        if dist > max(0.05, self.coverage_motion_max_gap_m):
            self._update_coverage(robot)
            self._last_coverage_pose = robot
            self._log_live_coverage(samples, self._covered_cell_count() - before, (time.time() - t0) * 1000.0, dist, dyaw)
            return

        yaw_step = math.radians(max(0.5, self.coverage_motion_yaw_step_deg))
        trans_n = int(math.ceil(dist / max(0.01, self.coverage_motion_sample_step_m)))
        yaw_n = int(math.ceil(abs(dyaw) / max(1e-3, yaw_step)))
        samples = max(1, trans_n, yaw_n)
        samples = min(max(1, self.coverage_motion_max_samples), samples)

        for i in range(1, samples + 1):
            t = i / samples
            pose = (
                last[0] + t * dx,
                last[1] + t * dy,
                self._angle_wrap(last[2] + t * dyaw),
            )
            self._update_coverage(pose)

        self._last_coverage_pose = robot
        self._log_live_coverage(samples, self._covered_cell_count() - before, (time.time() - t0) * 1000.0, dist, dyaw)

    def _covered_cell_count(self) -> int:
        return sum(1 for v in self.coverage if v == 100)

    def _log_live_coverage(self, samples: int, newly_covered: int, elapsed_ms: float, move_dist: float, dyaw: float) -> None:
        self.coverage_update_count += 1
        now = self._now_sec()
        if now - self.last_live_coverage_debug_time >= self.coverage_live_debug_throttle_sec:
            self.last_live_coverage_debug_time = now
            self.get_logger().info(
                f'LIVE_COVERAGE_10FPS | updates={self.coverage_update_count} samples={samples} '
                f'new={newly_covered} elapsed_ms={elapsed_ms:.2f} '
                f'fps={1.0 / max(1e-3, self.coverage_update_period_sec):.1f}'
            )
            self.get_logger().info(
                f'LIVE_COVERAGE_INTERP | updates={self.coverage_update_count} samples={samples} '
                f'new={newly_covered} elapsed_ms={elapsed_ms:.2f} '
                f'move={move_dist:.3f}m dyaw={math.degrees(abs(dyaw)):.1f}deg '
                f'fov={self.coverage_fov_deg:.1f} range={self.coverage_max_range_m:.2f} brush={self.coverage_brush_radius_m:.2f}'
            )

    def _update_coverage(self, robot: Pose2D) -> None:
        if self.scan_msg is None or self.map_msg is None or not self.coverage:
            return
        rx, ry, ryaw = robot
        fov = math.radians(self.coverage_fov_deg)
        angle_skip = max(1, int(round(math.radians(self.coverage_downsample_angle_step_deg) / max(self.scan_msg.angle_increment, 1e-6))))
        max_range = min(self.coverage_max_range_m, float(self.scan_msg.range_max) if self.scan_msg.range_max > 0.0 else self.coverage_max_range_m)
        for i in range(0, len(self.scan_msg.ranges), angle_skip):
            a = self.scan_msg.angle_min + i * self.scan_msg.angle_increment
            if self.coverage_front_only and abs(self._angle_wrap(a)) > 0.5 * fov:
                continue

            raw_r = self.scan_msg.ranges[i]
            finite_hit = math.isfinite(raw_r) and raw_r > 0.02
            if finite_hit:
                # Conservative rule: never paint cells close to the LiDAR hit.
                # This prevents coverage from bleeding through/onto walls.
                r = max(0.0, min(raw_r, max_range) - max(0.0, self.coverage_obstacle_margin_m))
            else:
                r = max_range

            if self.coverage_require_lidar_clear and r <= 0.02:
                continue

            global_a = ryaw + a
            steps = max(1, int(math.floor(r / max(self.coverage_ray_step_m, 1e-3))))
            for s in range(steps + 1):
                d = min(r, s * self.coverage_ray_step_m)
                wx = rx + d * math.cos(global_a)
                wy = ry + d * math.sin(global_a)
                c = self._world_to_cell(wx, wy)
                if c is None:
                    break
                if self._is_occupied(c):
                    break
                if self.coverage_stop_at_unknown and self._is_unknown(c):
                    break
                if self.coverage_only_known_free and not self._is_free(c):
                    break
                self._paint_coverage(c, self.coverage_brush_radius_m)
        if self.coverage_mark_robot_footprint:
            rc = self._world_to_cell(rx, ry)
            if rc:
                self._paint_coverage(rc, self.coverage_robot_radius_m)

    def _paint_coverage(self, c: Cell, radius_m: float) -> None:
        radius = max(0, int(math.ceil(radius_m / max(self.geom.resolution, 1e-6))))
        cx, cy = c
        if radius <= 0:
            if self._in_bounds(c) and self._is_free(c):
                self.coverage[self._idx(c)] = 100
            return
        r2 = radius * radius
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                if dx * dx + dy * dy > r2:
                    continue
                nb = (cx + dx, cy + dy)
                if self._in_bounds(nb) and self._is_free(nb):
                    self.coverage[self._idx(nb)] = 100

    def _publish_coverage(self) -> None:
        if self.map_msg is None or not self.coverage:
            return
        msg = OccupancyGrid()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.global_frame
        msg.info = self.map_msg.info
        msg.data = list(self.coverage)
        self.coverage_pub.publish(msg)

    # ------------------------------------------------------------------
    # Region stats and active-region tracking
    # ------------------------------------------------------------------
    def _compute_region_stats(self) -> Dict[int, RegionStats]:
        stats: Dict[int, RegionStats] = {}
        if self.map_msg is None or self.region_msg is None:
            return stats
        if self.region_msg.info.width != self.geom.width or self.region_msg.info.height != self.geom.height:
            return stats
        sums: Dict[int, Tuple[float, float]] = defaultdict(lambda: (0.0, 0.0))
        for y in range(self.geom.height):
            for x in range(self.geom.width):
                c = (x, y)
                label = self._region_value(c)
                if label <= 0 or not self._is_free(c):
                    continue
                st = stats.setdefault(label, RegionStats(label=label))
                st.total_free += 1
                if self._covered(c):
                    st.covered += 1
                if self._is_frontier_cell(c, label):
                    st.frontier += 1
                wx, wy = self._cell_to_world(c)
                sx, sy = sums[label]
                sums[label] = (sx + wx, sy + wy)
        for label, st in stats.items():
            if st.total_free > 0:
                sx, sy = sums[label]
                st.centroid_x = sx / st.total_free
                st.centroid_y = sy / st.total_free
                st.coverage_ratio = st.covered / max(1, st.total_free)
        return {k: v for k, v in stats.items() if v.total_free >= self.region_min_cells}

    def _region_cells(self, label: int) -> Set[Cell]:
        cells: Set[Cell] = set()
        if self.region_msg is None:
            return cells
        for y in range(self.geom.height):
            for x in range(self.geom.width):
                c = (x, y)
                if self._region_value(c) == label and self._is_free(c):
                    cells.add(c)
        return cells

    def _select_or_track_active_region(self, robot: Pose2D, stats: Dict[int, RegionStats]) -> Optional[int]:
        rc = self._world_to_cell(robot[0], robot[1])
        robot_label = self._region_value(rc) if rc is not None else 0

        # Keep old active label only while it still has useful local work.
        # v22: do not cling to a far/old region forever. If the robot has already
        # moved into another valid region and the previous active region is mostly
        # covered, switch to the robot's current region so exploration can proceed
        # through the house instead of bouncing around the old entry area.
        if self.active_region in stats and self.active_region not in self.completed_regions:
            cur = stats.get(self.active_region)
            age = self._now_sec() - self.active_region_started
            if robot_label in stats and robot_label != self.active_region and robot_label not in self.completed_regions:
                if cur is None or cur.coverage_ratio >= min(0.65, self.region_switch_coverage_ratio) or age >= 0.5 * self.region_max_active_time_sec:
                    self.active_region = robot_label
                    self.active_region_started = self._now_sec()
                    self.last_region_cells = self._region_cells(robot_label)
                    self.get_logger().info(f'SWITCH_ACTIVE_TO_ROBOT_REGION | active=R{robot_label}')
                    return robot_label
            return self.active_region

        # Track by IoU if labels were renumbered by the region graph.
        if self.last_region_cells:
            best_label = None
            best_iou = 0.0
            old = self.last_region_cells
            for label in stats:
                if label in self.completed_regions:
                    continue
                new = self._region_cells(label)
                if not new:
                    continue
                inter = len(old & new)
                union = len(old | new)
                iou = inter / max(1, union)
                if iou > best_iou:
                    best_iou = iou
                    best_label = label
            if best_label is not None and best_iou >= self.active_region_iou_match_threshold:
                self.active_region = best_label
                self.last_region_cells = self._region_cells(best_label)
                return best_label

        # Prefer robot's current region if not completed.
        if robot_label in stats and robot_label not in self.completed_regions:
            self.active_region = robot_label
            self.active_region_started = self._now_sec()
            self.last_region_cells = self._region_cells(robot_label)
            return robot_label

        # Otherwise choose the most valuable region.
        best_label = None
        best_score = -1e9
        for label, st in stats.items():
            if label in self.completed_regions:
                continue
            dist = self._distance_world((robot[0], robot[1]), (st.centroid_x, st.centroid_y))
            centroid_cell = self._world_to_cell(st.centroid_x, st.centroid_y)
            region_penalty, _, _, _, _ = self._candidate_penalty(robot, centroid_cell, st.centroid_x, st.centroid_y)
            unvisited_bonus = (self.w_next_region_unvisited * 35.0) if (self.prefer_unvisited_region and label not in self.visited_regions) else 0.0
            score = (
                self.w_next_region_uncovered * (1.0 - st.coverage_ratio) * 100.0
                + self.w_next_region_frontier * st.frontier
                + unvisited_bonus
                - self.w_next_region_path_cost * dist
                - 0.60 * region_penalty
            )
            if score > best_score:
                best_score = score
                best_label = label
        if best_label is not None:
            self.active_region = best_label
            self.active_region_started = self._now_sec()
            self.last_region_cells = self._region_cells(best_label)
        return best_label

    def _reopen_completed_regions(self, stats: Dict[int, RegionStats]) -> None:
        if not self.reopen_completed_region_on_frontier:
            return
        reopened = []
        threshold = int(math.ceil(self.region_frontier_threshold * self.reopen_frontier_margin))
        for label in list(self.completed_regions):
            st = stats.get(label)
            if st is not None and st.frontier > threshold:
                self.completed_regions.remove(label)
                reopened.append(label)
        if reopened:
            self.get_logger().info(f'REOPEN_COMPLETED_REGIONS_ON_FRONTIER | labels={reopened}')

    def _should_switch_region(self, label: int, st: Optional[RegionStats], best: Optional[Candidate], active_age: float, stats: Dict[int, RegionStats]) -> Optional[str]:
        if not self.region_switch_on_stall or st is None:
            return None
        # Do not switch before the region has had a minimal chance to collect coverage.
        if active_age < self.region_min_active_time_sec:
            return None
        # If there is no other useful region, stay here and keep trying local/global frontier.
        other_exists = any((r != label and r not in self.completed_regions) for r in stats.keys())
        if not other_exists:
            return None
        gain = best.score if best is not None else 0.0
        low_gain = gain < (self.region_candidate_gain_threshold * self.region_switch_gain_ratio)
        enough_covered = st.coverage_ratio >= self.region_switch_coverage_ratio
        # Hard upper bound: avoid spending forever in one region when labels/frontiers oscillate.
        if active_age >= self.region_max_active_time_sec and (enough_covered or low_gain):
            self.get_logger().info(
                f'FORCE_SWITCH_REGION_TIMEOUT | active=R{label} age={active_age:.1f}s '
                f'cov={st.coverage_ratio:.2f} fr={st.frontier} gain={gain:.1f}'
            )
            return 'NEXT_REGION_FORCED_TIMEOUT'
        # Soft switch: if region is already reasonably covered and best new coverage gain is weak,
        # try another region instead of repeatedly sending goals inside the same region.
        if enough_covered and low_gain:
            self.get_logger().info(
                f'SWITCH_REGION_LOW_GAIN | active=R{label} age={active_age:.1f}s '
                f'cov={st.coverage_ratio:.2f} fr={st.frontier} gain={gain:.1f}'
            )
            return 'NEXT_REGION_LOW_GAIN'
        return None

    def _region_complete(self, label: int, st: Optional[RegionStats], best: Optional[Candidate], active_age: float) -> bool:
        if st is None:
            return False
        if active_age < self.region_min_active_time_sec:
            return False
        gain = best.score if best is not None else 0.0
        return (
            st.coverage_ratio >= self.region_coverage_threshold
            and st.frontier <= self.region_frontier_threshold
            and gain <= self.region_candidate_gain_threshold
            and st.total_free >= self.region_min_cells
        )

    # ------------------------------------------------------------------
    # Candidate generation / scoring
    # ------------------------------------------------------------------
    def _candidate_cells_for_region(self, robot: Pose2D, active: int) -> List[Cell]:
        candidates = self._sample_region_candidates(robot, active)
        if self.frontier_candidate_sampling:
            candidates.extend(self._sample_frontier_candidates(robot, active, max_count=self.frontier_candidate_max_count))
        seen: Set[Cell] = set()
        unique: List[Cell] = []
        for c in candidates:
            if c in seen:
                continue
            seen.add(c)
            unique.append(c)
        if len(unique) > self.candidate_max_count:
            stride = max(1, int(math.ceil(len(unique) / self.candidate_max_count)))
            unique = unique[::stride][:self.candidate_max_count]
        return unique

    def _line_risk_ratios(self, robot: Pose2D, x: float, y: float) -> Tuple[float, float, float]:
        """Return occupied/unknown/narrow ratios on the robot→candidate segment."""
        if self.map_msg is None:
            return 0.0, 0.0, 0.0
        rx, ry, _ = robot
        dist = self._distance_world((rx, ry), (x, y))
        if dist <= 0.05:
            return 0.0, 0.0, 0.0
        step = max(self.geom.resolution, self.candidate_obstacle_line_step_m, 1e-3)
        n = max(1, int(math.ceil(dist / step)))
        occ = 0
        unk = 0
        narrow = 0
        total = 0
        check_clearance = max(0.0, self.candidate_obstacle_check_radius_m)
        for i in range(1, n + 1):
            t = i / n
            wx = rx + t * (x - rx)
            wy = ry + t * (y - ry)
            pc = self._world_to_cell(wx, wy)
            if pc is None:
                unk += 1
                total += 1
                continue
            total += 1
            if self._is_occupied(pc):
                occ += 1
            elif self._is_unknown(pc):
                unk += 1
            if check_clearance > 0.0 and not self._has_clearance(pc, check_clearance):
                narrow += 1
        denom = max(1, total)
        return occ / denom, unk / denom, narrow / denom

    def _compose_score(
        self,
        mode: str,
        rgain: int,
        rfr: int,
        xunk: int,
        xfr: int,
        clearance: float,
        dist: float,
        penalty: float,
    ) -> float:
        """Combine cached view-gains with live distance/penalty into a score.

        Used by both the full scoring path and the cheap priority-queue
        re-score, so the two never drift apart. `penalty` already contains the
        distance and obstacle-density costs (see _candidate_penalty)."""
        if mode == 'map':
            return (
                self.w_region_frontier_gain * rfr
                + self.w_cross_region_unknown * xunk
                + self.w_cross_region_frontier * xfr
                + self.w_clearance_bonus * clearance
                - self.w_path_cost * dist
                - penalty
            )
        if mode == 'coverage':
            return (
                self.w_region_coverage_gain * rgain
                + 0.5 * self.w_region_frontier_gain * rfr
                + self.w_clearance_bonus * clearance
                - self.w_path_cost * dist
                - penalty
            )
        # generic (region / next-region / global-frontier)
        return (
            self.w_region_coverage_gain * rgain
            + self.w_region_frontier_gain * rfr
            + self.w_cross_region_unknown * xunk
            + self.w_cross_region_frontier * xfr
            + self.w_clearance_bonus * clearance
            - self.w_path_cost * dist
            - penalty
        )

    def _candidate_penalty(self, robot: Pose2D, c: Optional[Cell], x: float, y: float) -> Tuple[float, float, float, float, float]:
        """Distance + obstacle-density penalty used by all semantic candidates."""
        dist = self._distance_world((robot[0], robot[1]), (x, y))
        occ_ratio, unk_ratio, narrow_ratio = self._line_risk_ratios(robot, x, y)
        local_risk = 0.0
        if c is not None and self._in_bounds(c):
            clearance = self._clearance_m(c)
            target = max(0.05, self.candidate_min_clearance_m)
            local_risk = max(0.0, (target - clearance) / target)
        distance_cost = self.w_candidate_distance_cost * dist + self.w_candidate_distance_sq_cost * dist * dist
        obstacle_cost = self.w_candidate_obstacle_density_cost * occ_ratio
        unknown_cost = self.w_candidate_unknown_density_cost * unk_ratio
        clearance_cost = self.w_candidate_clearance_risk_cost * (0.65 * narrow_ratio + 0.35 * local_risk)
        penalty = distance_cost + obstacle_cost + unknown_cost + clearance_cost
        return penalty, distance_cost, obstacle_cost, unknown_cost, clearance_cost

    # ------------------------------------------------------------------
    # Route beam-search (lookahead route queue)
    # ------------------------------------------------------------------
    def _view_gain_cells(
        self, c: Cell, yaw: float, active: int
    ) -> Tuple[Set[Cell], Set[Cell], Set[Cell], Set[Cell]]:
        """Like _view_gain but returns the actual cell *sets* for beam search.

        Used only during the moderate-rate rebuild so the extra allocation cost
        is acceptable (one call per pool candidate, not per rescore tick).
        """
        fov = math.radians(self.view_fov_deg)
        step_a = math.radians(max(1.0, self.view_angle_step_deg))
        n = max(1, int(math.ceil(fov / step_a)))
        wx, wy = self._cell_to_world(c)
        region_uncovered: Set[Cell] = set()
        region_frontier: Set[Cell] = set()
        cross_unknown: Set[Cell] = set()
        cross_frontier: Set[Cell] = set()
        for k in range(n + 1):
            rel = -0.5 * fov + k * (fov / max(1, n))
            a = yaw + rel
            steps = max(1, int(math.floor(self.view_max_range_m / max(self.view_ray_step_m, 1e-3))))
            for s in range(1, steps + 1):
                d = s * self.view_ray_step_m
                pc = self._world_to_cell(wx + d * math.cos(a), wy + d * math.sin(a))
                if pc is None:
                    break
                if self._is_occupied(pc):
                    break
                label = self._region_value(pc)
                if label == active and self._is_free(pc) and not self._covered(pc):
                    region_uncovered.add(pc)
                if label == active and self._is_frontier_cell(pc, active):
                    region_frontier.add(pc)
                if self.allow_cross_region_view_gain:
                    if label != active and self._is_unknown(pc):
                        cross_unknown.add(pc)
                    if label != active and self._is_frontier_cell(pc, None):
                        cross_frontier.add(pc)
                if self.view_stop_at_unknown and self._is_unknown(pc):
                    break
        return region_uncovered, region_frontier, cross_unknown, cross_frontier

    def _spatial_nms_candidates(
        self, candidates: List[Candidate], suppression_radius_m: float
    ) -> List[Candidate]:
        """Non-maximum suppression: keep the best candidate per spatial cluster.

        Eliminates the "three copies of the same doorway" problem that arises
        when many candidates score highly around the same frontier.
        """
        if not candidates:
            return candidates
        sorted_cands = sorted(candidates, key=lambda c: c.score, reverse=True)
        kept: List[Candidate] = []
        for cand in sorted_cands:
            too_close = any(
                self._distance_world((cand.x, cand.y), (k.x, k.y)) < suppression_radius_m
                for k in kept
            )
            if not too_close:
                kept.append(cand)
        return kept

    def _build_route_beam_search(
        self, robot: Pose2D, pool: List[Candidate], active: int
    ) -> List[Candidate]:
        """Return a [G1, G2, G3] route that maximises total discounted gain.

        score(route) =
          sum_i( discount^i * new_gain(Gi | G0..G{i-1} already seen) )
          - sum_i( discount^i * w_path_cost * dist(G{i-1} -> Gi) )
          - same_region_penalty  (if consecutive hops stay in the same
                                   small area without adding new info)
          - overlap_penalty_per_cell * (cells seen twice across the route)

        Nav2 always receives only G1; G2/G3 are for RViz and guide the next
        rebuild so the robot does not cycle around the same region.
        """
        if not pool:
            return []

        horizon = max(1, self.route_horizon)
        beam_width = max(1, self.route_beam_width)
        discount = float(self.route_discount)
        suppression_r = float(self.route_spatial_suppression_m)
        same_region_pen = float(self.route_same_region_penalty)
        overlap_pen = float(self.route_overlap_penalty_per_cell)

        # Pre-compute view-gain cell sets for all pool candidates.
        gain_cells: Dict[str, Tuple[Set[Cell], Set[Cell], Set[Cell], Set[Cell]]] = {}
        for cand in pool:
            k = cand.key
            if k and k not in gain_cells:
                gain_cells[k] = self._view_gain_cells(cand.cell, cand.yaw, active)

        def _incremental_gain(
            cand: Candidate,
            seen_cov: Set[Cell],
            seen_fr: Set[Cell],
            seen_xunk: Set[Cell],
            seen_xfr: Set[Cell],
        ) -> float:
            cells = gain_cells.get(cand.key, (set(), set(), set(), set()))
            new_cov  = len(cells[0] - seen_cov)
            new_fr   = len(cells[1] - seen_fr)
            new_xunk = len(cells[2] - seen_xunk)
            new_xfr  = len(cells[3] - seen_xfr)
            return (
                self.w_region_coverage_gain  * new_cov
                + self.w_region_frontier_gain * new_fr
                + self.w_cross_region_unknown * new_xunk
                + self.w_cross_region_frontier * new_xfr
            )

        def _route_score(route: List[Candidate]) -> float:
            seen_cov: Set[Cell] = set()
            seen_fr:  Set[Cell] = set()
            seen_xunk: Set[Cell] = set()
            seen_xfr:  Set[Cell] = set()
            total = 0.0
            overlap_cells = 0
            prev: Optional[Candidate] = None
            for i, cand in enumerate(route):
                d = discount ** i
                ig = _incremental_gain(cand, seen_cov, seen_fr, seen_xunk, seen_xfr)
                total += d * ig
                # Path cost from previous node (or robot for the first hop).
                if prev is None:
                    hop_dist = self._distance_world((robot[0], robot[1]), (cand.x, cand.y))
                else:
                    hop_dist = self._distance_world((prev.x, prev.y), (cand.x, cand.y))
                total -= d * self.w_path_cost * hop_dist
                # Obstacle/clearance penalty already baked into the candidate's
                # stored costs; re-apply it at the appropriate discount level.
                total -= d * (cand.obstacle_cost + cand.clearance_risk_cost)
                # Same-region micro-step penalty: two adjacent hops of the same
                # type that are very close together (both in coverage-fill mode
                # within suppression radius) add almost no new information.
                if prev is not None:
                    close = self._distance_world((prev.x, prev.y), (cand.x, cand.y)) < suppression_r
                    same_type = (cand.candidate_type == prev.candidate_type == 'COVERAGE_FILL')
                    if close and same_type:
                        total -= same_region_pen
                # Track overlap: cells seen by more than one hop.
                cells = gain_cells.get(cand.key, (set(), set(), set(), set()))
                all_seen = seen_cov | seen_fr | seen_xunk | seen_xfr
                all_new  = cells[0] | cells[1] | cells[2] | cells[3]
                overlap_cells += len(all_new & all_seen)
                # Update seen sets.
                seen_cov  |= cells[0]
                seen_fr   |= cells[1]
                seen_xunk |= cells[2]
                seen_xfr  |= cells[3]
                prev = cand
            total -= overlap_pen * overlap_cells
            return total

        # --- Initialise beam with single-hop routes ---
        BeamEntry = Tuple[float, List[Candidate]]
        beams: List[BeamEntry] = []
        for cand in pool:
            hop_dist = self._distance_world((robot[0], robot[1]), (cand.x, cand.y))
            cells = gain_cells.get(cand.key, (set(), set(), set(), set()))
            ig = (
                self.w_region_coverage_gain  * len(cells[0])
                + self.w_region_frontier_gain * len(cells[1])
                + self.w_cross_region_unknown * len(cells[2])
                + self.w_cross_region_frontier * len(cells[3])
            )
            s = ig - self.w_path_cost * hop_dist - cand.obstacle_cost - cand.clearance_risk_cost
            beams.append((s, [cand]))
        beams.sort(key=lambda x: x[0], reverse=True)
        beams = beams[:beam_width]

        # --- Expand to full horizon ---
        for _depth in range(1, horizon):
            next_beams: List[BeamEntry] = []
            for _score, route in beams:
                last = route[-1]
                used_keys = {c.key for c in route}
                # Candidates that are spatially distinct from the last waypoint
                # and not already in the route.
                expansions: List[Tuple[float, List[Candidate]]] = []
                for nc in pool:
                    if nc.key in used_keys:
                        continue
                    if self._distance_world((last.x, last.y), (nc.x, nc.y)) < suppression_r:
                        continue
                    new_route = route + [nc]
                    expansions.append((_route_score(new_route), new_route))
                if not expansions:
                    # Can't extend; keep the shorter route as-is.
                    next_beams.append((_score, route))
                else:
                    expansions.sort(key=lambda x: x[0], reverse=True)
                    next_beams.extend(expansions[:beam_width])

            next_beams.sort(key=lambda x: x[0], reverse=True)
            beams = next_beams[:beam_width]

        if not beams:
            return [c for c in pool[:horizon]]
        best_route = beams[0][1]
        return best_route[:horizon]

    def _find_best_candidate_by_mode(self, robot: Pose2D, active: int, stats: Dict[int, RegionStats], mode: str) -> Optional[Candidate]:
        best: Optional[Candidate] = None
        for c in self._candidate_cells_for_region(robot, active):
            wx, wy = self._cell_to_world(c)
            if self._is_blacklisted(wx, wy):
                continue
            cand = self._score_candidate_mode(robot, c, active, mode=mode)
            if cand is not None and (best is None or cand.score > best.score):
                best = cand
        return best

    def _score_candidate_mode(self, robot: Pose2D, c: Cell, active: int, mode: str) -> Optional[Candidate]:
        if not self._in_bounds(c) or self._is_occupied(c):
            return None
        if not self._has_clearance(c, self.candidate_min_clearance_m):
            return None
        wx, wy = self._cell_to_world(c)
        dist = self._distance_world((robot[0], robot[1]), (wx, wy))
        if dist < 0.20:
            return None
        if self.candidate_max_goal_distance_m > 0.0 and dist > self.candidate_max_goal_distance_m:
            return None
        yaw_options = self._candidate_yaws(c, active)
        if not yaw_options:
            yaw_options = [robot[2] + 2.0 * math.pi * i / max(1, self.candidate_yaw_samples) for i in range(max(1, self.candidate_yaw_samples))]

        best: Optional[Candidate] = None
        for yaw in yaw_options[: max(1, self.candidate_yaw_samples)]:
            rgain, rfr, xunk, xfr = self._view_gain(c, yaw, active)
            clearance = self._clearance_m(c)
            penalty, distance_cost, obstacle_cost, unknown_cost, clearance_risk_cost = self._candidate_penalty(robot, c, wx, wy)
            if mode == 'map':
                # Map expansion explicitly prioritizes region-local frontiers and
                # unknown visible through/gateway from the active region.
                raw_gain = rfr + xunk + xfr
                if raw_gain <= 0:
                    continue
                score = self._compose_score('map', rgain, rfr, xunk, xfr, clearance, dist, penalty)
                reason = 'REGION_MAP_EXPAND_CANDIDATE'
                score_mode = 'map'
            else:
                # Coverage fill uses only known free-space that is still uncovered
                # inside the active region. Unknown cells are not counted as covered.
                if rgain <= 0:
                    continue
                score = self._compose_score('coverage', rgain, rfr, xunk, xfr, clearance, dist, penalty)
                reason = 'REGION_COVERAGE_FILL_CANDIDATE'
                score_mode = 'coverage'
            cand = Candidate(
                cell=c,
                x=wx,
                y=wy,
                yaw=self._angle_wrap(yaw),
                score=score,
                region_cov_gain=rgain,
                region_frontier_gain=rfr,
                cross_unknown_gain=xunk,
                cross_frontier_gain=xfr,
                distance=dist,
                reason=reason,
                distance_cost=distance_cost,
                obstacle_cost=obstacle_cost,
                unknown_cost=unknown_cost,
                clearance_risk_cost=clearance_risk_cost,
                score_mode=score_mode,
                cached_clearance=clearance,
            )
            if mode == 'map':
                cand = self._push_map_expand_candidate_if_lidar_clear(robot, cand, active)
            if best is None or cand.score > best.score:
                best = cand
        return best

    def _push_map_expand_candidate_if_lidar_clear(self, robot: Pose2D, cand: Candidate, active: int) -> Candidate:
        """Push a map-expansion goal farther forward when the LiDAR corridor is clear.

        Frontier goals placed exactly on the free/unknown boundary tend to create
        stop-go behavior and only small map updates.  This routine keeps the same
        semantic yaw but advances the goal along that yaw through known-free cells;
        optionally it may enter a short LiDAR-clear unknown segment.  The latter
        is intentionally bounded so we do not blindly command deep unknown goals.
        """
        if not self.map_expand_lidar_push_enabled:
            return cand
        if self.scan_msg is None:
            return cand
        if (cand.region_frontier_gain + cand.cross_unknown_gain + cand.cross_frontier_gain) <= 0:
            return cand

        step = max(0.03, self.map_expand_goal_push_step_m)
        max_push = max(0.0, self.map_expand_goal_push_max_m)
        max_unknown_push = max(0.0, self.map_expand_unknown_push_max_m)
        if max_push < step:
            return cand

        start_x, start_y = cand.x, cand.y
        best_cell = cand.cell
        best_x, best_y = cand.x, cand.y
        best_push = 0.0
        unknown_push = 0.0

        nsteps = max(1, int(math.floor(max_push / step)))
        for i in range(1, nsteps + 1):
            push_d = i * step
            wx = start_x + push_d * math.cos(cand.yaw)
            wy = start_y + push_d * math.sin(cand.yaw)
            pc = self._world_to_cell(wx, wy)
            if pc is None:
                break
            if self._is_occupied(pc):
                break

            mv = self._map_value(pc)
            is_known_free = 0 <= mv < self.free_threshold
            is_unknown = mv < 0

            if is_unknown:
                if not self.map_expand_push_into_lidar_clear_unknown:
                    break
                unknown_push += step
                if unknown_push > max_unknown_push:
                    break
                # Unknown goal is allowed only when the current LiDAR ray says
                # the whole segment from robot to this point is clear.
                if not self._lidar_clear_to_world_point(robot, wx, wy, self.map_expand_goal_push_lidar_margin_m):
                    break
            elif is_known_free:
                unknown_push = 0.0
                if not self._has_clearance(pc, self.candidate_min_clearance_m):
                    break
                if not self._lidar_clear_to_world_point(robot, wx, wy, self.map_expand_goal_push_lidar_margin_m):
                    break
            else:
                break

            best_cell = pc
            best_x, best_y = wx, wy
            best_push = push_d

        if best_push < self.map_expand_goal_push_min_m:
            return cand

        # Re-score from the pushed pose.  If the pushed pose is in a short
        # LiDAR-clear unknown segment, _view_gain will still see nearby unknown;
        # if it is in known-free space, it gets a normal candidate score.
        rgain, rfr, xunk, xfr = self._view_gain(best_cell, cand.yaw, active)
        clearance = self._clearance_m(best_cell)
        dist = self._distance_world((robot[0], robot[1]), (best_x, best_y))
        penalty, distance_cost, obstacle_cost, unknown_cost, clearance_risk_cost = self._candidate_penalty(robot, best_cell, best_x, best_y)
        pushed_score = (
            self.w_region_frontier_gain * rfr
            + self.w_cross_region_unknown * xunk
            + self.w_cross_region_frontier * xfr
            + self.w_clearance_bonus * clearance
            - self.w_path_cost * dist
            - penalty
            + self.map_expand_goal_push_score_bonus_per_m * best_push
        )

        # Do not make the pushed goal worse unless the original candidate was
        # almost at the boundary and the push itself is the useful exploration act.
        if pushed_score + self.map_expand_goal_push_score_bonus_per_m * best_push < cand.score:
            return cand

        pushed = Candidate(
            cell=best_cell,
            x=best_x,
            y=best_y,
            yaw=cand.yaw,
            score=max(cand.score, pushed_score),
            region_cov_gain=rgain,
            region_frontier_gain=rfr,
            cross_unknown_gain=xunk,
            cross_frontier_gain=xfr,
            distance=dist,
            reason=f'{cand.reason}_LIDAR_PUSH_{best_push:.2f}m',
            distance_cost=distance_cost,
            obstacle_cost=obstacle_cost,
            unknown_cost=unknown_cost,
            clearance_risk_cost=clearance_risk_cost,
            score_mode='map',
            cached_clearance=clearance,
        )
        self._log_lidar_push(cand, pushed, best_push, unknown_push > 0.0)
        return pushed

    def _log_lidar_push(self, old: Candidate, pushed: Candidate, pushed_distance: float, entered_unknown: bool) -> None:
        self.lidar_push_since_log += 1
        now = self._now_sec()
        if self._shutting_down or not rclpy.ok():
            return
        if now - self.last_lidar_push_log_time < max(0.05, self.lidar_push_log_throttle_sec):
            return
        count = self.lidar_push_since_log
        self.lidar_push_since_log = 0
        self.last_lidar_push_log_time = now
        self.get_logger().info(
            f'LIDAR_PUSH | count={count} from=({old.x:.2f},{old.y:.2f}) '
            f'to=({pushed.x:.2f},{pushed.y:.2f}) pushed={pushed_distance:.2f}m '
            f'entered_unknown={entered_unknown} score={pushed.score:.1f}'
        )

    def _scan_range_at_relative_angle(self, rel_angle: float) -> Optional[float]:
        if self.scan_msg is None or not self.scan_msg.ranges:
            return None
        rel_angle = self._angle_wrap(rel_angle)
        amin = float(self.scan_msg.angle_min)
        amax = float(self.scan_msg.angle_max)
        inc = float(self.scan_msg.angle_increment)
        if inc <= 0.0 or rel_angle < amin or rel_angle > amax:
            return None
        idx = int(round((rel_angle - amin) / inc))
        if idx < 0 or idx >= len(self.scan_msg.ranges):
            return None
        # Use a small angular window and the minimum finite range to stay
        # conservative near doorframes/walls.
        win = 2
        vals: List[float] = []
        for j in range(max(0, idx - win), min(len(self.scan_msg.ranges), idx + win + 1)):
            r = float(self.scan_msg.ranges[j])
            if math.isfinite(r) and r > float(self.scan_msg.range_min):
                vals.append(r)
        if vals:
            return min(vals)
        if self.scan_msg.range_max > 0.0:
            return float(self.scan_msg.range_max)
        return None

    def _lidar_clear_to_world_point(self, robot: Pose2D, x: float, y: float, margin_m: float) -> bool:
        rx, ry, ryaw = robot
        dist = self._distance_world((rx, ry), (x, y))
        if dist <= 0.05:
            return True
        rel = self._angle_wrap(math.atan2(y - ry, x - rx) - ryaw)
        r = self._scan_range_at_relative_angle(rel)
        if r is None:
            # If we cannot validate with LiDAR, do not push into a farther goal.
            return False
        return r >= dist + max(0.0, margin_m)

    def _find_best_candidate(self, robot: Pose2D, active: int, stats: Dict[int, RegionStats]) -> Optional[Candidate]:
        unique = self._candidate_cells_for_region(robot, active)
        best: Optional[Candidate] = None
        for c in unique:
            wx, wy = self._cell_to_world(c)
            if self._is_blacklisted(wx, wy):
                continue
            cand = self._score_candidate(robot, c, active)
            if cand is not None and (best is None or cand.score > best.score):
                best = cand
        return best

    def _sample_region_candidates(self, robot: Pose2D, active: int) -> List[Cell]:
        step = max(1, int(round(self.candidate_grid_step_m / max(self.geom.resolution, 1e-6))))
        out: List[Cell] = []
        for y in range(0, self.geom.height, step):
            for x in range(0, self.geom.width, step):
                c = (x, y)
                if self._region_value(c) != active:
                    continue
                if not self._is_free(c):
                    continue
                if not self._has_clearance(c, self.candidate_min_clearance_m):
                    continue
                out.append(c)
        return out

    def _sample_frontier_candidates(self, robot: Pose2D, active: int, max_count: int) -> List[Cell]:
        radius = max(1, int(round(self.candidate_frontier_ring_radius_m / max(self.geom.resolution, 1e-6))))
        raw_frontiers: List[Cell] = []
        for y in range(self.geom.height):
            for x in range(self.geom.width):
                c = (x, y)
                if self._region_value(c) != active:
                    continue
                if self._is_frontier_cell(c, active):
                    raw_frontiers.append(c)
        if not raw_frontiers:
            return []
        # Sort by distance to robot so nearby frontiers are not starved.
        raw_frontiers.sort(key=lambda c: self._distance_world(self._cell_to_world(c), (robot[0], robot[1])))
        out: List[Cell] = []
        seen: Set[Cell] = set()
        for fc in raw_frontiers[: max_count * 3]:
            fx, fy = fc
            for dy in range(-radius, radius + 1):
                for dx in range(-radius, radius + 1):
                    if dx * dx + dy * dy > radius * radius:
                        continue
                    c = (fx - dx, fy - dy)  # free side inward candidate
                    if c in seen or not self._in_bounds(c):
                        continue
                    if self._region_value(c) != active or not self._is_free(c):
                        continue
                    if not self._has_clearance(c, self.candidate_min_clearance_m):
                        continue
                    seen.add(c)
                    out.append(c)
                    if len(out) >= max_count:
                        return out
        return out

    def _score_candidate(self, robot: Pose2D, c: Cell, active: int) -> Optional[Candidate]:
        if not self._in_bounds(c) or self._is_occupied(c):
            return None
        if not self._has_clearance(c, self.candidate_min_clearance_m):
            return None
        wx, wy = self._cell_to_world(c)
        dist = self._distance_world((robot[0], robot[1]), (wx, wy))
        if dist < 0.20:
            return None
        if self.candidate_max_goal_distance_m > 0.0 and dist > self.candidate_max_goal_distance_m:
            return None
        yaw_options = self._candidate_yaws(c, active)
        if not yaw_options:
            yaw_options = [robot[2] + 2.0 * math.pi * i / max(1, self.candidate_yaw_samples) for i in range(max(1, self.candidate_yaw_samples))]

        best: Optional[Candidate] = None
        for yaw in yaw_options[: max(1, self.candidate_yaw_samples)]:
            rgain, rfr, xunk, xfr = self._view_gain(c, yaw, active)
            clearance = self._clearance_m(c)
            penalty, distance_cost, obstacle_cost, unknown_cost, clearance_risk_cost = self._candidate_penalty(robot, c, wx, wy)
            score = self._compose_score('generic', rgain, rfr, xunk, xfr, clearance, dist, penalty)
            cand = Candidate(
                cell=c,
                x=wx,
                y=wy,
                yaw=self._angle_wrap(yaw),
                score=score,
                region_cov_gain=rgain,
                region_frontier_gain=rfr,
                cross_unknown_gain=xunk,
                cross_frontier_gain=xfr,
                distance=dist,
                reason='REGION_CANDIDATE',
                distance_cost=distance_cost,
                obstacle_cost=obstacle_cost,
                unknown_cost=unknown_cost,
                clearance_risk_cost=clearance_risk_cost,
                score_mode='generic',
                cached_clearance=clearance,
            )
            if best is None or cand.score > best.score:
                best = cand
        return best

    def _candidate_yaws(self, c: Cell, active: int) -> List[float]:
        targets: List[Tuple[float, float]] = []
        cx, cy = c
        radius = max(2, int(round(self.view_max_range_m / max(self.geom.resolution, 1e-6))))
        for dy in range(-radius, radius + 1, 2):
            for dx in range(-radius, radius + 1, 2):
                nb = (cx + dx, cy + dy)
                if not self._in_bounds(nb):
                    continue
                if self._region_value(nb) == active and self._is_free(nb) and not self._covered(nb):
                    targets.append(self._cell_to_world(nb))
                elif self._region_value(nb) == active and self._is_frontier_cell(nb, active):
                    targets.append(self._cell_to_world(nb))
                elif self.allow_cross_region_view_gain and self._is_unknown(nb):
                    targets.append(self._cell_to_world(nb))
        if not targets:
            return []
        wx, wy = self._cell_to_world(c)
        # Use target centroid and a few offsets around it.
        tx = sum(t[0] for t in targets) / len(targets)
        ty = sum(t[1] for t in targets) / len(targets)
        base = math.atan2(ty - wy, tx - wx)
        offsets = [0.0, math.radians(20), -math.radians(20), math.radians(45), -math.radians(45)]
        return [self._angle_wrap(base + o) for o in offsets]

    def _view_gain(self, c: Cell, yaw: float, active: int) -> Tuple[int, int, int, int]:
        fov = math.radians(self.view_fov_deg)
        step_a = math.radians(max(1.0, self.view_angle_step_deg))
        n = max(1, int(math.ceil(fov / step_a)))
        wx, wy = self._cell_to_world(c)
        region_uncovered: Set[Cell] = set()
        region_frontier: Set[Cell] = set()
        cross_unknown: Set[Cell] = set()
        cross_frontier: Set[Cell] = set()
        for k in range(n + 1):
            rel = -0.5 * fov + k * (fov / max(1, n))
            a = yaw + rel
            steps = max(1, int(math.floor(self.view_max_range_m / max(self.view_ray_step_m, 1e-3))))
            for s in range(1, steps + 1):
                d = s * self.view_ray_step_m
                pc = self._world_to_cell(wx + d * math.cos(a), wy + d * math.sin(a))
                if pc is None:
                    break
                if self._is_occupied(pc):
                    break
                label = self._region_value(pc)
                if label == active and self._is_free(pc) and not self._covered(pc):
                    region_uncovered.add(pc)
                if label == active and self._is_frontier_cell(pc, active):
                    region_frontier.add(pc)
                if self.allow_cross_region_view_gain:
                    if label != active and self._is_unknown(pc):
                        cross_unknown.add(pc)
                    if label != active and self._is_frontier_cell(pc, None):
                        cross_frontier.add(pc)
                if self.view_stop_at_unknown and self._is_unknown(pc):
                    break
        return (len(region_uncovered), len(region_frontier), len(cross_unknown), len(cross_frontier))

    # ------------------------------------------------------------------
    # Next-region and fallback goals
    # ------------------------------------------------------------------
    def _select_next_region_goal(self, robot: Pose2D, active: int, stats: Dict[int, RegionStats]) -> Optional[Candidate]:
        best_region = None
        best_score = -1e9
        for label, st in stats.items():
            if label == active or label in self.completed_regions:
                continue
            dist = self._distance_world((robot[0], robot[1]), (st.centroid_x, st.centroid_y))
            centroid_cell = self._world_to_cell(st.centroid_x, st.centroid_y)
            region_penalty, _, _, _, _ = self._candidate_penalty(robot, centroid_cell, st.centroid_x, st.centroid_y)
            unvisited_bonus = (self.w_next_region_unvisited * 35.0) if (self.prefer_unvisited_region and label not in self.visited_regions) else 0.0
            score = (
                self.w_next_region_uncovered * (1.0 - st.coverage_ratio) * 100.0
                + self.w_next_region_frontier * st.frontier
                + unvisited_bonus
                - self.w_next_region_path_cost * dist
                - 0.60 * region_penalty
            )
            if score > best_score:
                best_score = score
                best_region = label
        if best_region is None:
            return None
        # Use the best candidate inside the next region. If no scored candidate,
        # send centroid as an entry-ish goal.
        cands = self._sample_region_candidates(robot, best_region)
        best = None
        for c in cands[: self.candidate_max_count]:
            wx, wy = self._cell_to_world(c)
            if self._is_blacklisted(wx, wy):
                continue
            cand = self._score_candidate(robot, c, best_region)
            if cand is not None and (best is None or cand.score > best.score):
                best = cand
        if best is not None:
            best.reason = 'NEXT_REGION_ENTRY_GOAL'
            best.target_region_id = best_region
            return best
        st = stats[best_region]
        cell = self._world_to_cell(st.centroid_x, st.centroid_y)
        if cell is None:
            return None
        yaw = math.atan2(st.centroid_y - robot[1], st.centroid_x - robot[0])
        return Candidate(
            cell,
            st.centroid_x,
            st.centroid_y,
            yaw,
            best_score,
            0,
            st.frontier,
            0,
            0,
            0.0,
            'NEXT_REGION_CENTROID',
            target_region_id=best_region,
            base_score=best_score,
            dynamic_score=best_score,
            score_mode='generic',
        )

    def _select_global_frontier_goal(self, robot: Pose2D, stats: Dict[int, RegionStats]) -> Optional[Candidate]:
        frontier_cells: List[Cell] = []
        for y in range(self.geom.height):
            for x in range(self.geom.width):
                c = (x, y)
                if self._is_frontier_cell(c, None) and self._has_clearance(c, self.candidate_min_clearance_m):
                    frontier_cells.append(c)
        if not frontier_cells:
            return None
        # Do not just take the nearest frontiers. In house maps that causes
        # repeated oscillation near the spawn/entry area. Keep a deterministic
        # spread across the map, then let the score pick high-gain frontier goals.
        frontier_cells.sort(key=lambda c: self._distance_world(self._cell_to_world(c), (robot[0], robot[1])))
        if len(frontier_cells) > self.frontier_candidate_max_count:
            stride = max(1, int(math.ceil(len(frontier_cells) / self.frontier_candidate_max_count)))
            search_cells = frontier_cells[::stride][: self.frontier_candidate_max_count]
            search_cells.extend(frontier_cells[: max(20, self.frontier_candidate_max_count // 5)])
        else:
            search_cells = frontier_cells
        best: Optional[Candidate] = None
        seen_search: Set[Cell] = set()
        for c in search_cells:
            if c in seen_search:
                continue
            seen_search.add(c)
            wx, wy = self._cell_to_world(c)
            if self._is_blacklisted(wx, wy):
                continue
            label = self._region_value(c)
            active = label if label > 0 else (self.active_region or 0)
            cand = self._score_candidate(robot, c, active)
            if cand is None:
                yaw = math.atan2(wy - robot[1], wx - robot[0])
                penalty, distance_cost, obstacle_cost, unknown_cost, clearance_risk_cost = self._candidate_penalty(robot, c, wx, wy)
                dist = self._distance_world((wx, wy), (robot[0], robot[1]))
                cand = Candidate(c, wx, wy, yaw, 1.0 - penalty, 0, 1, 0, 0, dist, 'GLOBAL_FRONTIER', distance_cost, obstacle_cost, unknown_cost, clearance_risk_cost, score_mode='generic')
            cand.reason = 'GLOBAL_FRONTIER'
            cand.candidate_type = 'GLOBAL_FRONTIER'
            if best is None or cand.score > best.score:
                best = cand
        return best

    def _make_scan_turn_goal(self, robot: Pose2D) -> Candidate:
        rx, ry, yaw = robot
        # Rotate via Nav2 at current location. Alternate direction each time.
        sign = 1.0 if (self.goal_seq % 2 == 0) else -1.0
        nyaw = self._angle_wrap(yaw + sign * math.radians(self.scan_turn_yaw_step_deg))
        c = self._world_to_cell(rx, ry) or (0, 0)
        return Candidate(
            c,
            rx,
            ry,
            nyaw,
            max(2.0, self.candidate_queue_min_score + 0.5),
            0,
            0,
            0,
            0,
            0.0,
            'SCAN_TURN',
            candidate_type='GLOBAL_FRONTIER',
            base_score=max(2.0, self.candidate_queue_min_score + 0.5),
            dynamic_score=max(2.0, self.candidate_queue_min_score + 0.5),
        )

    def _make_forward_probe_goal(self, robot: Pose2D) -> Optional[Candidate]:
        if not self.allow_nav2_forward_probe_goal:
            return None
        rx, ry, yaw = robot
        max_dist = max(0.0, self.forward_probe_distance_m)
        min_dist = max(0.05, self.forward_probe_min_distance_m)
        if max_dist < min_dist:
            return None

        # Try the requested distance first, then back off. This goal only exists
        # to break the initial "valid but already satisfied" Nav2 no-op state.
        steps = max(1, int(math.ceil((max_dist - min_dist) / 0.10)))
        distances = [max_dist - i * 0.10 for i in range(steps + 1)]
        if distances[-1] > min_dist:
            distances.append(min_dist)

        for dist in distances:
            dist = max(min_dist, dist)
            wx = rx + dist * math.cos(yaw)
            wy = ry + dist * math.sin(yaw)
            if not self._lidar_clear_to_world_point(robot, wx, wy, self.forward_probe_lidar_margin_m):
                continue
            c = self._world_to_cell(wx, wy)
            if c is None or self._is_occupied(c):
                continue
            if not self._has_clearance(c, min(self.candidate_min_clearance_m, 0.18)):
                continue
            if self._is_blacklisted(wx, wy):
                continue
            score = max(2.5, self.candidate_queue_min_score + 1.0)
            return Candidate(
                c,
                wx,
                wy,
                yaw,
                score,
                0,
                0,
                0,
                0,
                dist,
                'FORWARD_PROBE',
                candidate_type='GLOBAL_FRONTIER',
                base_score=score,
                dynamic_score=score,
                score_mode='generic',
                cached_clearance=self._clearance_m(c),
            )
        return None

    def _is_blacklisted(self, x: float, y: float) -> bool:
        now = self._now_sec()
        for bx, by, exp in self.blacklist:
            if exp < now:
                continue
            if self._distance_world((x, y), (bx, by)) <= self.goal_blacklist_radius_m:
                return True
        return False

    def _expire_blacklist(self, now: float) -> None:
        self.blacklist = [(x, y, exp) for (x, y, exp) in self.blacklist if exp >= now]

    # ------------------------------------------------------------------
    # Publication / markers
    # ------------------------------------------------------------------
    def _candidate_to_pose(self, cand: Candidate) -> PoseStamped:
        pose = PoseStamped()
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.header.frame_id = self.global_frame
        pose.pose.position.x = float(cand.x)
        pose.pose.position.y = float(cand.y)
        pose.pose.position.z = 0.0
        pose.pose.orientation = _quaternion_msg_from_yaw(cand.yaw)
        return pose

    def _publish_state(self, state: str, robot: Optional[Pose2D], extra: Optional[dict] = None) -> None:
        self.last_state = state
        st = self.last_stats.get(self.active_region) if self.active_region is not None else None
        payload = {
            'state': state,
            'does_not_publish_cmd_vel': True,
            'nav2_action': self.navigate_action_name,
            'active_goal': self.active_goal,
            'active_region': self.active_region,
            'completed_regions': sorted(list(self.completed_regions))[:20],
            'visited_regions': sorted(list(self.visited_regions))[:20],
            'blacklist_count': len(self.blacklist),
            'coverage_front_only': self.coverage_front_only,
            'coverage_fov_deg': self.coverage_fov_deg,
            'coverage_max_range_m': self.coverage_max_range_m,
            'coverage_only_known_free': self.coverage_only_known_free,
            'coverage_stop_at_unknown': self.coverage_stop_at_unknown,
            'coverage_obstacle_margin_m': self.coverage_obstacle_margin_m,
            'region_map_gain_threshold': self.region_map_gain_threshold,
            'region_coverage_gain_threshold': self.region_coverage_gain_threshold,
            'region_coverage_threshold': self.region_coverage_threshold,
            'region_frontier_threshold': self.region_frontier_threshold,
            'map_expand_lidar_push_enabled': self.map_expand_lidar_push_enabled,
            'map_expand_push_into_lidar_clear_unknown': self.map_expand_push_into_lidar_clear_unknown,
            'map_expand_goal_push_max_m': self.map_expand_goal_push_max_m,
            'map_expand_unknown_push_max_m': self.map_expand_unknown_push_max_m,
        }
        if robot is not None:
            payload['robot'] = {'x': robot[0], 'y': robot[1], 'yaw': robot[2]}
        if st is not None:
            payload['active_region_stats'] = {
                'label': st.label,
                'coverage_ratio': round(st.coverage_ratio, 4),
                'covered': st.covered,
                'total_free': st.total_free,
                'frontier': st.frontier,
            }
        if self.last_best is not None:
            payload['last_best'] = {
                'x': round(self.last_best.x, 3),
                'y': round(self.last_best.y, 3),
                'yaw': round(self.last_best.yaw, 3),
                'score': round(self.last_best.score, 2),
                'region_cov_gain': self.last_best.region_cov_gain,
                'region_frontier_gain': self.last_best.region_frontier_gain,
                'cross_unknown_gain': self.last_best.cross_unknown_gain,
                'cross_frontier_gain': self.last_best.cross_frontier_gain,
                'reason': self.last_best.reason,
            }
        if extra:
            payload.update(extra)
        msg = String()
        msg.data = json.dumps(payload, ensure_ascii=False)
        self.state_pub.publish(msg)

    def _publish_markers(self, robot: Optional[Pose2D], best: Optional[Candidate], stats: Dict[int, RegionStats]) -> None:
        arr = MarkerArray()
        now = self.get_clock().now().to_msg()
        delete = Marker()
        delete.header.frame_id = self.global_frame
        delete.header.stamp = now
        delete.ns = 'region_nav2_explorer'
        delete.action = Marker.DELETEALL
        arr.markers.append(delete)

        mid = 0
        # Use the beam-search route queue when available: G1/G2/G3 are the
        # ordered next-steps of the best exploration plan, not just the three
        # independently highest-scoring candidates.  Nav2 still receives only G1.
        if self.route_queue_enabled and self.route_queue:
            top_candidates = list(self.route_queue[:3])
        else:
            top_candidates = list(self.candidate_queue[:3])
            if best is not None and all(c.key != best.key for c in top_candidates):
                top_candidates = [best] + top_candidates
            top_candidates = top_candidates[:3]
        colors = [
            (0.1, 1.0, 0.25, 0.98),   # #1 green
            (0.0, 0.75, 1.0, 0.92),   # #2 cyan
            (1.0, 0.25, 0.95, 0.88),  # #3 magenta
        ]

        for rank, cand in enumerate(top_candidates, start=1):
            r, g, b, a = colors[rank - 1]
            m = Marker()
            m.header.frame_id = self.global_frame
            m.header.stamp = now
            m.ns = 'region_nav2_explorer'
            m.id = mid
            mid += 1
            m.type = Marker.SPHERE
            m.action = Marker.ADD
            m.pose.position.x = cand.x
            m.pose.position.y = cand.y
            m.pose.position.z = 0.16 + 0.04 * (rank - 1)
            size = 0.26 if rank == 1 else 0.20
            m.scale.x = size
            m.scale.y = size
            m.scale.z = size
            m.color.r = r
            m.color.g = g
            m.color.b = b
            m.color.a = a
            arr.markers.append(m)

            arrow = Marker()
            arrow.header.frame_id = self.global_frame
            arrow.header.stamp = now
            arrow.ns = 'region_nav2_explorer'
            arrow.id = mid
            mid += 1
            arrow.type = Marker.ARROW
            arrow.action = Marker.ADD
            arrow.pose.position.x = cand.x
            arrow.pose.position.y = cand.y
            arrow.pose.position.z = 0.22 + 0.04 * (rank - 1)
            arrow.pose.orientation = _quaternion_msg_from_yaw(cand.yaw)
            arrow.scale.x = 0.38 if rank == 1 else 0.28
            arrow.scale.y = 0.055
            arrow.scale.z = 0.055
            arrow.color.r = r
            arrow.color.g = g
            arrow.color.b = b
            arrow.color.a = a
            arr.markers.append(arrow)

            t = Marker()
            t.header.frame_id = self.global_frame
            t.header.stamp = now
            t.ns = 'region_nav2_explorer'
            t.id = mid
            mid += 1
            t.type = Marker.TEXT_VIEW_FACING
            t.action = Marker.ADD
            t.pose.position.x = cand.x
            t.pose.position.y = cand.y
            t.pose.position.z = 0.50 + 0.08 * (rank - 1)
            t.scale.z = 0.18 if rank == 1 else 0.15
            t.color.r = r
            t.color.g = g
            t.color.b = b
            t.color.a = 1.0
            t.text = (
                f'G{rank} {cand.candidate_type}\n'
                f'{cand.reason}\n'
                f'score={cand.score:.1f} d={cand.distance:.1f}\n'
                f'cov={cand.region_cov_gain} fr={cand.region_frontier_gain}'
            )
            arr.markers.append(t)

        # Text at active region centroid.
        if self.active_region is not None and self.active_region in stats:
            st = stats[self.active_region]
            t = Marker()
            t.header.frame_id = self.global_frame
            t.header.stamp = now
            t.ns = 'region_nav2_explorer'
            t.id = mid
            mid += 1
            t.type = Marker.TEXT_VIEW_FACING
            t.action = Marker.ADD
            t.pose.position.x = st.centroid_x
            t.pose.position.y = st.centroid_y
            t.pose.position.z = 0.62
            t.scale.z = 0.24
            t.color.r = 0.1
            t.color.g = 1.0
            t.color.b = 0.2
            t.color.a = 1.0
            t.text = f'ACTIVE R{self.active_region}\ncov={st.coverage_ratio:.2f}\nfr={st.frontier}'
            arr.markers.append(t)

        self.marker_pub.publish(arr)

    def _begin_shutdown(self) -> None:
        self._shutting_down = True
        for name in ('timer', 'coverage_timer', 'goal_dispatch_timer'):
            timer = getattr(self, name, None)
            if timer is None:
                continue
            try:
                timer.cancel()
            except Exception:
                pass


def main(args=None) -> None:
    rclpy.init(args=args)
    node = RegionNav2ExplorerNode()
    # At least two threads: one for the coverage group (10 Hz painter + sensor
    # callbacks), one for the planning group (candidate scoring / Nav2 goals).
    # Without this the single-threaded executor serializes everything and a long
    # planning callback freezes coverage, which is what made it look discrete.
    executor = MultiThreadedExecutor(num_threads=3)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node._begin_shutdown()
        try:
            executor.shutdown()
        except Exception:
            pass
        try:
            node.destroy_node()
        except Exception:
            pass
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
