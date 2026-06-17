#!/usr/bin/env python3

import heapq
import math
import time
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

import rclpy
from rclpy.action import ActionClient
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy

from action_msgs.msg import GoalStatus
from geometry_msgs.msg import PoseStamped, Twist, TwistStamped, Point
from nav2_msgs.action import ComputePathToPose, NavigateToPose
from nav_msgs.msg import OccupancyGrid, Path
from sensor_msgs.msg import LaserScan
from std_msgs.msg import String
from visualization_msgs.msg import Marker, MarkerArray

import tf2_ros


VERSION = 'v31_cartographer_external_slam'


@dataclass
class GridInfo:
    width: int
    height: int
    resolution: float
    origin_x: float
    origin_y: float


@dataclass
class Pose2D:
    x: float
    y: float
    yaw: float


@dataclass
class Candidate:
    x: float
    y: float
    nav_yaw: float
    view_yaw: float
    score: float
    unknown_gain: int
    visual_gain: int
    frontier_gain: int
    clearance: float
    distance: float
    source: str


class VisibilityExplorerNode(Node):
    def __init__(self):
        super().__init__('visibility_explorer')

        # Parameters
        self.declare_parameter('version_name', VERSION)
        self.declare_parameter('map_topic', '/map')
        self.declare_parameter('scan_topic', '/scan')
        self.declare_parameter('cmd_vel_topic', '/cmd_vel')
        self.declare_parameter('global_frame', 'map')
        self.declare_parameter('robot_frame', 'base_footprint')
        self.declare_parameter('cmd_vel_stamped', True)
        self.declare_parameter('path_topic', '/visibility_explorer/path')
        self.declare_parameter('plan_alias_topic', '/plan')
        self.declare_parameter('scan_viz_topic', '/scan_reliable')
        self.declare_parameter('publish_scan_reliable', True)
        self.declare_parameter('scan_sub_best_effort', True)

        self.declare_parameter('planning_period', 0.25)
        self.declare_parameter('map_stable_time', 0.10)
        self.declare_parameter('max_plan_trials', 20)

        self.declare_parameter('free_threshold', 20)
        self.declare_parameter('occupied_threshold', 65)
        self.declare_parameter('unknown_is_obstacle_for_clearance', False)

        self.declare_parameter('min_frontier_cluster_size', 5)
        self.declare_parameter('frontier_goal_offset', 0.55)
        self.declare_parameter('candidate_stride_cells', 3)
        self.declare_parameter('min_goal_distance', 0.25)
        self.declare_parameter('max_goal_distance', 5.0)
        self.declare_parameter('max_goal_bearing_deg', 160.0)

        self.declare_parameter('robot_radius', 0.23)
        self.declare_parameter('min_goal_clearance', 0.30)
        self.declare_parameter('min_path_clearance', 0.18)
        self.declare_parameter('front_stop_distance', 0.24)
        self.declare_parameter('front_slow_distance', 0.42)
        self.declare_parameter('side_stop_distance', 0.16)

        self.declare_parameter('view_fov_deg', 60.0)
        self.declare_parameter('view_ray_count', 31)
        self.declare_parameter('view_max_range', 3.5)
        self.declare_parameter('coverage_robot_radius', 0.28)
        self.declare_parameter('coverage_publish_period', 0.4)
        self.declare_parameter('enable_slam_warmup', False)
        self.declare_parameter('slam_warmup_min_duration', 8.0)
        self.declare_parameter('slam_warmup_max_duration', 18.0)
        self.declare_parameter('slam_warmup_min_known_cells', 350)
        self.declare_parameter('slam_warmup_min_free_cells', 160)
        self.declare_parameter('slam_warmup_angular_speed', 0.22)
        self.declare_parameter('map_health_log_period', 2.5)
        self.declare_parameter('visual_unchecked_threshold', 95)
        self.declare_parameter('coverage_candidate_max_checked', 95)

        # v27: make under-covered visible space a first-class objective.
        # Unknown space is still important, but visual coverage now dominates NBV scoring.
        self.declare_parameter('unknown_gain_weight', 12.0)
        self.declare_parameter('visual_gain_weight', 3.0)
        self.declare_parameter('frontier_gain_weight', 1.6)
        self.declare_parameter('clearance_weight', 0.55)
        self.declare_parameter('distance_weight', 0.22)
        self.declare_parameter('blacklist_weight', 5.0)
        self.declare_parameter('heading_weight', 0.12)
        self.declare_parameter('coverage_source_score_bonus', 22.0)

        self.declare_parameter('planner_id', '')
        self.declare_parameter('enable_internal_grid_planner', True)
        self.declare_parameter('prefer_internal_grid_planner', True)
        self.declare_parameter('internal_planner_max_expansions', 50000)
        self.declare_parameter('nav2_result_timeout', 75.0)
        self.declare_parameter('stuck_timeout', 10.0)
        self.declare_parameter('stuck_min_progress', 0.10)
        self.declare_parameter('goal_success_radius', 0.35)
        self.declare_parameter('blacklist_radius', 0.55)
        self.declare_parameter('blacklist_duration', 45.0)

        self.declare_parameter('enable_sector_scan', False)
        self.declare_parameter('sector_scan_angle_deg', 45.0)
        self.declare_parameter('sector_scan_angular_speed', 0.75)
        self.declare_parameter('enable_view_yaw_align', True)
        self.declare_parameter('view_align_tolerance_deg', 12.0)
        self.declare_parameter('view_align_timeout', 1.0)
        self.declare_parameter('view_align_angular_speed', 0.75)

        self.declare_parameter('enable_short_lidar_probe_fallback', True)
        self.declare_parameter('enable_unknown_first_probe', True)
        self.declare_parameter('enable_scan_open_space_probe', True)
        self.declare_parameter('allow_probe_during_map_stabilizing', True)
        self.declare_parameter('prefer_scan_probe_before_nav2', True)
        self.declare_parameter('coverage_priority_over_probe', True)
        self.declare_parameter('unmapped_priority_over_coverage', True)
        self.declare_parameter('unmapped_priority_probe_gain', 8)
        self.declare_parameter('unknown_first_min_gain', 2)
        self.declare_parameter('nav2_min_goal_distance', 0.0)
        self.declare_parameter('execute_with_nav2_navigator', False)
        self.declare_parameter('direct_follow_lookahead_distance', 0.35)
        self.declare_parameter('direct_follow_max_linear_speed', 0.24)
        self.declare_parameter('direct_follow_min_linear_speed', 0.045)
        self.declare_parameter('direct_follow_max_angular_speed', 1.35)
        self.declare_parameter('direct_follow_angular_gain', 1.8)
        self.declare_parameter('direct_follow_stuck_timeout', 3.5)
        self.declare_parameter('direct_follow_progress_epsilon', 0.04)
        self.declare_parameter('direct_goal_tolerance', 0.18)
        self.declare_parameter('probe_success_radius', 0.12)
        self.declare_parameter('probe_min_distance', 0.45)
        self.declare_parameter('probe_fov_deg', 120.0)
        self.declare_parameter('probe_distance', 0.75)
        self.declare_parameter('probe_min_front_clearance', 0.48)
        self.declare_parameter('probe_side_clearance', 0.22)
        self.declare_parameter('probe_linear_speed', 0.20)
        self.declare_parameter('probe_angular_gain', 1.8)
        self.declare_parameter('probe_timeout', 4.0)
        self.declare_parameter('probe_fail_cooldown', 2.0)
        self.declare_parameter('probe_safety_margin', 0.05)
        self.declare_parameter('probe_fail_blacklist_radius', 0.60)
        self.declare_parameter('planner_first_when_probe_blocked', True)
        self.declare_parameter('micro_rotate_on_total_failure', True)
        self.declare_parameter('micro_rotate_duration', 0.45)
        self.declare_parameter('micro_rotate_speed', 0.45)

        self.version_name = str(self.get_parameter('version_name').value)
        self.map_topic = str(self.get_parameter('map_topic').value)
        self.scan_topic = str(self.get_parameter('scan_topic').value)
        self.cmd_vel_topic = str(self.get_parameter('cmd_vel_topic').value)
        self.global_frame = str(self.get_parameter('global_frame').value)
        self.robot_frame = str(self.get_parameter('robot_frame').value)
        self.cmd_vel_stamped = bool(self.get_parameter('cmd_vel_stamped').value)
        self.path_topic = str(self.get_parameter('path_topic').value)
        self.plan_alias_topic = str(self.get_parameter('plan_alias_topic').value)
        self.scan_viz_topic = str(self.get_parameter('scan_viz_topic').value)
        self.publish_scan_reliable = bool(self.get_parameter('publish_scan_reliable').value)
        self.scan_sub_best_effort = bool(self.get_parameter('scan_sub_best_effort').value)

        self.planning_period = float(self.get_parameter('planning_period').value)
        self.map_stable_time = float(self.get_parameter('map_stable_time').value)
        self.max_plan_trials = int(self.get_parameter('max_plan_trials').value)

        self.free_threshold = int(self.get_parameter('free_threshold').value)
        self.occupied_threshold = int(self.get_parameter('occupied_threshold').value)
        self.unknown_is_obstacle_for_clearance = bool(self.get_parameter('unknown_is_obstacle_for_clearance').value)

        self.min_frontier_cluster_size = int(self.get_parameter('min_frontier_cluster_size').value)
        self.frontier_goal_offset = float(self.get_parameter('frontier_goal_offset').value)
        self.candidate_stride_cells = int(self.get_parameter('candidate_stride_cells').value)
        self.min_goal_distance = float(self.get_parameter('min_goal_distance').value)
        self.max_goal_distance = float(self.get_parameter('max_goal_distance').value)
        self.max_goal_bearing = math.radians(float(self.get_parameter('max_goal_bearing_deg').value))

        self.robot_radius = float(self.get_parameter('robot_radius').value)
        self.min_goal_clearance = float(self.get_parameter('min_goal_clearance').value)
        self.min_path_clearance = float(self.get_parameter('min_path_clearance').value)
        self.front_stop_distance = float(self.get_parameter('front_stop_distance').value)
        self.front_slow_distance = float(self.get_parameter('front_slow_distance').value)
        self.side_stop_distance = float(self.get_parameter('side_stop_distance').value)

        self.view_fov = math.radians(float(self.get_parameter('view_fov_deg').value))
        self.view_ray_count = int(self.get_parameter('view_ray_count').value)
        self.view_max_range = float(self.get_parameter('view_max_range').value)
        self.coverage_robot_radius = float(self.get_parameter('coverage_robot_radius').value)
        self.coverage_publish_period = float(self.get_parameter('coverage_publish_period').value)
        self.enable_slam_warmup = bool(self.get_parameter('enable_slam_warmup').value)
        self.slam_warmup_min_duration = float(self.get_parameter('slam_warmup_min_duration').value)
        self.slam_warmup_max_duration = float(self.get_parameter('slam_warmup_max_duration').value)
        self.slam_warmup_min_known_cells = int(self.get_parameter('slam_warmup_min_known_cells').value)
        self.slam_warmup_min_free_cells = int(self.get_parameter('slam_warmup_min_free_cells').value)
        self.slam_warmup_angular_speed = float(self.get_parameter('slam_warmup_angular_speed').value)
        self.map_health_log_period = float(self.get_parameter('map_health_log_period').value)
        self.visual_unchecked_threshold = int(self.get_parameter('visual_unchecked_threshold').value)
        self.coverage_candidate_max_checked = int(self.get_parameter('coverage_candidate_max_checked').value)

        self.w_unknown = float(self.get_parameter('unknown_gain_weight').value)
        self.w_visual = float(self.get_parameter('visual_gain_weight').value)
        self.w_frontier = float(self.get_parameter('frontier_gain_weight').value)
        self.w_clearance = float(self.get_parameter('clearance_weight').value)
        self.w_distance = float(self.get_parameter('distance_weight').value)
        self.w_blacklist = float(self.get_parameter('blacklist_weight').value)
        self.w_heading = float(self.get_parameter('heading_weight').value)
        self.coverage_source_score_bonus = float(self.get_parameter('coverage_source_score_bonus').value)

        self.planner_id = str(self.get_parameter('planner_id').value)
        self.enable_internal_grid_planner = bool(self.get_parameter('enable_internal_grid_planner').value)
        self.prefer_internal_grid_planner = bool(self.get_parameter('prefer_internal_grid_planner').value)
        self.internal_planner_max_expansions = int(self.get_parameter('internal_planner_max_expansions').value)
        self.nav2_result_timeout = float(self.get_parameter('nav2_result_timeout').value)
        self.stuck_timeout = float(self.get_parameter('stuck_timeout').value)
        self.stuck_min_progress = float(self.get_parameter('stuck_min_progress').value)
        self.goal_success_radius = float(self.get_parameter('goal_success_radius').value)
        self.blacklist_radius = float(self.get_parameter('blacklist_radius').value)
        self.blacklist_duration = float(self.get_parameter('blacklist_duration').value)

        self.enable_sector_scan = bool(self.get_parameter('enable_sector_scan').value)
        self.sector_scan_angle = math.radians(float(self.get_parameter('sector_scan_angle_deg').value))
        self.sector_scan_angular_speed = float(self.get_parameter('sector_scan_angular_speed').value)
        self.enable_view_yaw_align = bool(self.get_parameter('enable_view_yaw_align').value)
        self.view_align_tolerance = math.radians(float(self.get_parameter('view_align_tolerance_deg').value))
        self.view_align_timeout = float(self.get_parameter('view_align_timeout').value)
        self.view_align_angular_speed = float(self.get_parameter('view_align_angular_speed').value)

        self.enable_short_lidar_probe_fallback = bool(self.get_parameter('enable_short_lidar_probe_fallback').value)
        self.enable_unknown_first_probe = bool(self.get_parameter('enable_unknown_first_probe').value)
        self.enable_scan_open_space_probe = bool(self.get_parameter('enable_scan_open_space_probe').value)
        self.allow_probe_during_map_stabilizing = bool(self.get_parameter('allow_probe_during_map_stabilizing').value)
        self.prefer_scan_probe_before_nav2 = bool(self.get_parameter('prefer_scan_probe_before_nav2').value)
        self.coverage_priority_over_probe = bool(self.get_parameter('coverage_priority_over_probe').value)
        self.unmapped_priority_over_coverage = bool(self.get_parameter('unmapped_priority_over_coverage').value)
        self.unmapped_priority_probe_gain = int(self.get_parameter('unmapped_priority_probe_gain').value)
        self.unknown_first_min_gain = int(self.get_parameter('unknown_first_min_gain').value)
        self.nav2_min_goal_distance = float(self.get_parameter('nav2_min_goal_distance').value)
        self.execute_with_nav2_navigator = bool(self.get_parameter('execute_with_nav2_navigator').value)
        self.direct_follow_lookahead_distance = float(self.get_parameter('direct_follow_lookahead_distance').value)
        self.direct_follow_max_linear_speed = float(self.get_parameter('direct_follow_max_linear_speed').value)
        self.direct_follow_min_linear_speed = float(self.get_parameter('direct_follow_min_linear_speed').value)
        self.direct_follow_max_angular_speed = float(self.get_parameter('direct_follow_max_angular_speed').value)
        self.direct_follow_angular_gain = float(self.get_parameter('direct_follow_angular_gain').value)
        self.direct_follow_stuck_timeout = float(self.get_parameter('direct_follow_stuck_timeout').value)
        self.direct_follow_progress_epsilon = float(self.get_parameter('direct_follow_progress_epsilon').value)
        self.direct_goal_tolerance = float(self.get_parameter('direct_goal_tolerance').value)
        self.probe_success_radius = float(self.get_parameter('probe_success_radius').value)
        self.probe_min_distance = float(self.get_parameter('probe_min_distance').value)
        self.probe_fov = math.radians(float(self.get_parameter('probe_fov_deg').value))
        self.probe_distance = float(self.get_parameter('probe_distance').value)
        self.probe_min_front_clearance = float(self.get_parameter('probe_min_front_clearance').value)
        self.probe_side_clearance = float(self.get_parameter('probe_side_clearance').value)
        self.probe_linear_speed = float(self.get_parameter('probe_linear_speed').value)
        self.probe_angular_gain = float(self.get_parameter('probe_angular_gain').value)
        self.probe_timeout = float(self.get_parameter('probe_timeout').value)
        self.probe_fail_cooldown = float(self.get_parameter('probe_fail_cooldown').value)
        self.probe_safety_margin = float(self.get_parameter('probe_safety_margin').value)
        self.probe_fail_blacklist_radius = float(self.get_parameter('probe_fail_blacklist_radius').value)
        self.planner_first_when_probe_blocked = bool(self.get_parameter('planner_first_when_probe_blocked').value)
        self.micro_rotate_on_total_failure = bool(self.get_parameter('micro_rotate_on_total_failure').value)
        self.micro_rotate_duration = float(self.get_parameter('micro_rotate_duration').value)
        self.micro_rotate_speed = float(self.get_parameter('micro_rotate_speed').value)

        # ROS interfaces
        map_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self.map_sub = self.create_subscription(OccupancyGrid, self.map_topic, self._on_map, map_qos)
        if self.scan_sub_best_effort:
            scan_qos = QoSProfile(
                reliability=ReliabilityPolicy.BEST_EFFORT,
                durability=DurabilityPolicy.VOLATILE,
                history=HistoryPolicy.KEEP_LAST,
                depth=10,
            )
        else:
            scan_qos = QoSProfile(
                reliability=ReliabilityPolicy.RELIABLE,
                durability=DurabilityPolicy.VOLATILE,
                history=HistoryPolicy.KEEP_LAST,
                depth=10,
            )
        self.scan_sub = self.create_subscription(LaserScan, self.scan_topic, self._on_scan, scan_qos)

        self.coverage_pub = self.create_publisher(OccupancyGrid, '/visual_coverage_map', 1)
        scan_viz_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )
        self.scan_viz_pub = self.create_publisher(LaserScan, self.scan_viz_topic, scan_viz_qos)
        path_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self.path_pub = self.create_publisher(Path, self.path_topic, path_qos)
        self.plan_alias_pub = self.create_publisher(Path, self.plan_alias_topic, path_qos)
        self.status_pub = self.create_publisher(String, '/visibility_explorer/status', 10)
        self.marker_pub = self.create_publisher(MarkerArray, '/frontier_viewpoint_markers', 1)
        self.selected_goal_pub = self.create_publisher(PoseStamped, '/selected_viewpoint_goal_map', 1)

        if self.cmd_vel_stamped:
            self.cmd_vel_pub = self.create_publisher(TwistStamped, self.cmd_vel_topic, 10)
        else:
            self.cmd_vel_pub = self.create_publisher(Twist, self.cmd_vel_topic, 10)

        self.tf_buffer = tf2_ros.Buffer(cache_time=Duration(seconds=10.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.plan_client = ActionClient(self, ComputePathToPose, 'compute_path_to_pose')
        self.nav_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')

        self.timer = self.create_timer(0.10, self._on_timer)

        # State
        self.map_msg: Optional[OccupancyGrid] = None
        self.grid: Optional[GridInfo] = None
        self.last_map_geometry_change = time.time()
        self.scan_msg: Optional[LaserScan] = None
        self.coverage: Optional[List[int]] = None
        self.clearance: Optional[List[float]] = None

        self.state = 'WAIT_MAP'
        self.pending_candidates: List[Candidate] = []
        self.current_candidate: Optional[Candidate] = None
        self.current_goal_pose: Optional[PoseStamped] = None
        self.current_path: Optional[Path] = None
        self.direct_follow_active = False
        self.direct_path_points: List[Tuple[float, float]] = []
        self.direct_follow_start_time = 0.0
        self.direct_follow_last_progress_time = 0.0
        self.direct_follow_best_distance = float('inf')
        self.nav_goal_handle = None
        self.nav_result_future = None
        self.plan_goal_handle = None
        self.plan_result_future = None

        self.blacklist: List[Tuple[float, float, float]] = []
        self.last_plan_time = 0.0
        self.last_cov_pub_time = 0.0
        self.last_status_time = 0.0

        self.nav_start_time = 0.0
        self.last_progress_time = 0.0
        self.best_goal_distance = float('inf')

        self.scanning = False
        self.scan_start_yaw = 0.0
        self.scan_target_abs = 0.0
        self.scan_direction = 1.0

        self.view_align_active = False
        self.view_align_target_yaw = 0.0
        self.view_align_start_time = 0.0

        self.slam_warmup_done = False
        self.slam_warmup_active = False
        self.slam_warmup_start_time = 0.0
        self.last_map_health_log_time = 0.0

        self.probe_active = False
        self.probe_target: Optional[Tuple[float, float]] = None
        self.probe_start_time = 0.0
        self.probe_blocked_until = 0.0
        self.probe_fail_blacklist: List[Tuple[float, float, float]] = []

        self.micro_rotate_active = False
        self.micro_rotate_start_time = 0.0
        self.micro_rotate_direction = 1.0

        self.get_logger().info(f'VisibilityExplorerNode {self.version_name} started.')
        self.get_logger().info(f'PATH_TOPICS | rviz_plan_alias={self.plan_alias_topic} explorer_path={self.path_topic}')
        self.get_logger().info(f'SCAN_QOS | input={self.scan_topic} best_effort_sub={self.scan_sub_best_effort} rviz_reliable_alias={self.scan_viz_topic}')
        self.get_logger().info(
            f'UNMAPPED_PRIORITY | unknown_weight={self.w_unknown:.2f} visual_weight={self.w_visual:.2f} '
            f'unmapped_over_coverage={self.unmapped_priority_over_coverage} override_gain={self.unmapped_priority_probe_gain} '
            f'unchecked_threshold={self.visual_unchecked_threshold} candidate_max_checked={self.coverage_candidate_max_checked} '
            f'coverage_over_probe={self.coverage_priority_over_probe}'
        )
        self.get_logger().info(
            f'INTERNAL_PLANNER | enabled={self.enable_internal_grid_planner} prefer={self.prefer_internal_grid_planner} '
            f'max_expansions={self.internal_planner_max_expansions}'
        )
        self.get_logger().info(
            f'SLAM_WARMUP | enabled={self.enable_slam_warmup} min={self.slam_warmup_min_duration:.1f}s '
            f'max={self.slam_warmup_max_duration:.1f}s min_known={self.slam_warmup_min_known_cells} '
            f'min_free={self.slam_warmup_min_free_cells} wz={self.slam_warmup_angular_speed:.2f}'
        )

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------

    def _on_map(self, msg: OccupancyGrid):
        new_grid = GridInfo(
            width=msg.info.width,
            height=msg.info.height,
            resolution=msg.info.resolution,
            origin_x=msg.info.origin.position.x,
            origin_y=msg.info.origin.position.y,
        )
        if self.grid is None or self._grid_changed(self.grid, new_grid):
            self.last_map_geometry_change = time.time()
            self._resize_coverage(new_grid)
        self.grid = new_grid
        self.map_msg = msg
        self.clearance = None

    def _on_scan(self, msg: LaserScan):
        self.scan_msg = msg
        if self.publish_scan_reliable:
            try:
                self.scan_viz_pub.publish(msg)
            except Exception:
                pass

    def _on_timer(self):
        robot = self._lookup_robot_pose()
        if robot is None:
            self._status('WAIT_TF')
            return

        if self.map_msg is None or self.grid is None:
            self._status('WAIT_MAP')
            return

        self._expire_blacklist()
        self._expire_probe_fail_blacklist()
        self._update_visual_coverage(robot)
        self._publish_coverage_periodic()
        self._log_map_health_periodic(robot)

        if self._handle_slam_warmup(robot):
            return

        if self.probe_active:
            self._execute_probe(robot)
            return

        if self.direct_follow_active:
            self._execute_direct_follow(robot)
            return

        if self.micro_rotate_active:
            self._execute_micro_rotate(robot)
            return

        if self.view_align_active:
            self._execute_view_align(robot)
            return

        if self.scanning:
            self._execute_sector_scan(robot)
            return

        if self.state in ['PLAN_WAIT', 'NAV_SEND_WAIT']:
            return

        if self.state == 'NAVIGATING':
            self._monitor_nav(robot)
            return

        now = time.time()
        if now - self.last_plan_time < self.planning_period:
            return
        self.last_plan_time = now

        # On the real robot SLAM may resize/update the map slowly.  Do not freeze the
        # robot every time the map geometry changes; first try a short scan-only
        # probe into open space, then wait if no safe ray exists.
        if now - self.last_map_geometry_change < self.map_stable_time:
            if self.allow_probe_during_map_stabilizing and not self.probe_active and not self.direct_follow_active:
                if self._try_scan_open_space_probe(robot, reason='map_stabilizing'):
                    return
            self._status('MAP_STABILIZING')
            return

        self._plan_next(robot)

    # ------------------------------------------------------------------
    # SLAM startup / health diagnostics
    # ------------------------------------------------------------------

    def _map_known_stats(self) -> Tuple[int, int, int, int]:
        if self.map_msg is None:
            return 0, 0, 0, 0
        free = 0
        occupied = 0
        unknown = 0
        for v in self.map_msg.data:
            if self._is_unknown_value(v):
                unknown += 1
            elif self._is_occupied_value(v):
                occupied += 1
            elif self._is_free_value(v):
                free += 1
        return free + occupied, free, occupied, unknown

    def _log_map_health_periodic(self, robot: Pose2D):
        now = time.time()
        if now - self.last_map_health_log_time < self.map_health_log_period:
            return
        self.last_map_health_log_time = now
        known, free, occ, unknown = self._map_known_stats()
        cell = self._world_to_cell(robot.x, robot.y)
        robot_val = None
        if cell is not None and self.map_msg is not None:
            robot_val = self.map_msg.data[self._idx(*cell)]
        self.get_logger().info(
            f'MAP_HEALTH | known={known} free={free} occ={occ} unknown={unknown} '
            f'robot_cell={cell} robot_occ={robot_val} warmup_done={self.slam_warmup_done}'
        )

    def _handle_slam_warmup(self, robot: Pose2D) -> bool:
        """Hold exploration and rotate slowly until the initial SLAM map is usable.

        Real TurtleBot3 SLAM often drops initial LaserScan messages while odom/map TF
        are still appearing.  Starting exploration immediately can make the robot
        move before scan matching has a stable local map.  This warmup performs only
        a slow in-place yaw motion; no linear motion is commanded.
        """
        if not self.enable_slam_warmup or self.slam_warmup_done:
            return False
        if self.probe_active or self.direct_follow_active or self.micro_rotate_active or self.view_align_active or self.scanning:
            return False
        known, free, occ, unknown = self._map_known_stats()
        now = time.time()
        if not self.slam_warmup_active:
            self.slam_warmup_active = True
            self.slam_warmup_start_time = now
            self.get_logger().warn(
                f'SLAM_WARMUP_START | rotating in place before exploration known={known} free={free} '
                f'min_known={self.slam_warmup_min_known_cells} min_free={self.slam_warmup_min_free_cells}'
            )
        elapsed = now - self.slam_warmup_start_time
        enough_map = known >= self.slam_warmup_min_known_cells and free >= self.slam_warmup_min_free_cells
        if elapsed >= self.slam_warmup_min_duration and (enough_map or elapsed >= self.slam_warmup_max_duration):
            self.slam_warmup_done = True
            self.slam_warmup_active = False
            self._stop_cmd()
            self.get_logger().warn(
                f'SLAM_WARMUP_DONE | elapsed={elapsed:.1f}s known={known} free={free} occ={occ} reason=' +
                ('map_ready' if enough_map else 'timeout')
            )
            self.state = 'IDLE'
            return False
        self.state = 'SLAM_WARMUP'
        self._publish_cmd(0.0, self.slam_warmup_angular_speed)
        self._status(f'SLAM_WARMUP elapsed={elapsed:.1f}s known={known} free={free} occ={occ}')
        return True

    # ------------------------------------------------------------------
    # Planning
    # ------------------------------------------------------------------

    def _plan_next(self, robot: Pose2D):
        self.state = 'PLANNING'
        self.current_candidate = None
        self.current_goal_pose = None
        self.current_path = None
        self._publish_path(None)
        self._clear_markers()

        if self.clearance is None:
            self.clearance = self._compute_clearance()

        # v26: unknown/scan probes have priority, but safety-stopped probes must not monopolize
        # the state machine. If a probe was just safety-stopped, temporarily skip direct
        # probing and immediately try Nav2 candidates instead.
        probe_blocked = time.time() < self.probe_blocked_until

        # v30: if a LiDAR ray sees substantial unknown / out-of-map space, expand the map
        # before doing local coverage cleanup. This keeps v27/v28 coverage behavior, but
        # prevents the robot from repeatedly polishing already-mapped cells while a clear
        # unmapped area is visible.
        if self.unmapped_priority_over_coverage and self.enable_unknown_first_probe and not probe_blocked:
            a = self._best_open_probe_angle(require_unknown=True)
            if a is not None:
                ug, vg, fg = self._probe_ray_gain(a)
                if ug >= self.unmapped_priority_probe_gain:
                    self.get_logger().info(
                        f'UNMAPPED_PRIORITY_OVERRIDE | angle={math.degrees(a):.1f} '
                        f'gain[u/v/f]=[{ug}/{vg}/{fg}] threshold={self.unmapped_priority_probe_gain}; '
                        f'expanding map before coverage sweep'
                    )
                    if self._try_unknown_first_lidar_probe(robot):
                        return

        # v27: do not let unknown/open-space probes monopolize the planner.
        # Build and score NBV/coverage candidates first when coverage_priority_over_probe is true.
        if not self.coverage_priority_over_probe:
            if self.enable_unknown_first_probe and not probe_blocked and self._try_unknown_first_lidar_probe(robot):
                return
            if self.prefer_scan_probe_before_nav2 and not probe_blocked and self._try_scan_open_space_probe(robot, reason='scan_open_before_nav2'):
                return
        if probe_blocked:
            self.get_logger().info('PROBE_BLOCKED_COOLDOWN | skipping direct probe; trying Nav2 candidates')

        robot_cell = self._world_to_cell(robot.x, robot.y)
        if robot_cell is None:
            self._status('ROBOT_OUTSIDE_MAP')
            
            if not self._try_short_lidar_probe(robot, reason='robot_outside_map'):
                self._start_micro_rotate(robot, reason='robot_outside_map')
            return

        reachable = self._reachable_free(robot_cell)
        frontiers = self._detect_frontiers(reachable)
        clusters = self._cluster_frontiers(frontiers)

        candidates = []
        candidates.extend(self._frontier_candidates(clusters, robot))
        candidates.extend(self._coverage_sweep_candidates(reachable, frontiers, robot))

        if not candidates:
            self._status(f'NO_CANDIDATE reachable={len(reachable)} frontiers={len(frontiers)} clusters={len(clusters)}')
            
            if not self._try_short_lidar_probe(robot, reason='no_candidate'):
                self._start_micro_rotate(robot, reason='no_candidate')
            return

        candidates.sort(key=lambda c: c.score, reverse=True)
        self.pending_candidates = candidates[:self.max_plan_trials]
        self._try_next_candidate(robot)

    def _try_next_candidate(self, robot: Pose2D):
        if not self.pending_candidates:
            self._status('NO_VALID_PATH_AFTER_TRIALS')
            if not self._try_short_lidar_probe(robot, reason='no_valid_path'):
                self._start_micro_rotate(robot, reason='no_valid_path')
            return

        cand = self.pending_candidates.pop(0)
        if self._blacklist_cost(cand.x, cand.y) > 0.95:
            self._try_next_candidate(robot)
            return

        if cand.distance < self.nav2_min_goal_distance:
            self.get_logger().info(
                f'CANDIDATE_BELOW_NAV2_MIN_DISTANCE | dist={cand.distance:.2f}; using short lidar probe instead'
            )
            
            if not self._try_short_lidar_probe(robot, reason='candidate_close'):
                self._try_next_candidate(robot)
            return

        trial = self.max_plan_trials - len(self.pending_candidates)
        self.get_logger().info(
            f'PATH_CHECK | trial={trial}/{self.max_plan_trials} score={cand.score:.2f} '
            f'source={cand.source} goal=({cand.x:.2f},{cand.y:.2f}) '
            f'gain[u/v/f]=[{cand.unknown_gain}/{cand.visual_gain}/{cand.frontier_gain}] '
            f'clearance={cand.clearance:.2f} dist={cand.distance:.2f}'
        )

        # v31: Cartographer can provide /map quickly without Nav2.  If Nav2's
        # ComputePathToPose server is not running, or if explicitly preferred, use
        # a small internal grid planner over the Cartographer occupancy map.
        # This keeps the real-robot stack lightweight: robot_bringup + cartographer
        # + explorer is enough for exploration.
        use_internal = (
            self.enable_internal_grid_planner
            and (self.prefer_internal_grid_planner or not self.plan_client.server_is_ready())
        )
        if use_internal:
            path = self._compute_internal_grid_path(robot, cand)
            if path is None:
                self.get_logger().warn('INTERNAL_PATH_INVALID | trying next candidate')
                self.state = 'PLANNING'
                self._try_next_candidate(robot)
                return
            path_len = self._path_length(path)
            path_clear = self._path_min_clearance(path)
            if path_len < 0.05:
                self.get_logger().warn(f'INTERNAL_PATH_REJECT_SHORT | len={path_len:.2f}')
                self.state = 'PLANNING'
                self._try_next_candidate(robot)
                return
            if path_clear < self.min_path_clearance:
                self.get_logger().warn(f'INTERNAL_PATH_REJECT_CLEARANCE | min={path_clear:.2f}')
                self.state = 'PLANNING'
                self._try_next_candidate(robot)
                return
            self.get_logger().info(
                f'INTERNAL_PATH_VALID | len={path_len:.2f} min_clear={path_clear:.2f} '
                f'goal=({cand.x:.2f},{cand.y:.2f}) points={len(path.poses)}'
            )
            self._publish_path(path)
            self._start_direct_follow(cand, path)
            return

        if not self.plan_client.server_is_ready():
            self._status('WAIT_NAV2_PLANNER_TRY_PROBE')
            if not self._try_short_lidar_probe(robot, reason='planner_not_ready'):
                self._start_micro_rotate(robot, reason='planner_not_ready')
            return

        pose = self._candidate_to_pose(cand)
        goal = ComputePathToPose.Goal()
        goal.goal = pose
        goal.use_start = False
        goal.planner_id = self.planner_id

        self.current_candidate = cand
        self.state = 'PLAN_WAIT'
        send_future = self.plan_client.send_goal_async(goal)
        send_future.add_done_callback(lambda fut: self._on_plan_goal_response(fut, robot, cand))

    def _on_plan_goal_response(self, future, robot: Pose2D, cand: Candidate):
        try:
            handle = future.result()
        except Exception as exc:
            self.get_logger().warn(f'PLAN_SEND_FAILED | {exc}')
            self.state = 'PLANNING'
            self._try_next_candidate(robot)
            return
        if not handle.accepted:
            self.get_logger().warn('PLAN_REJECTED')
            self.state = 'PLANNING'
            self._try_next_candidate(robot)
            return
        self.plan_goal_handle = handle
        result_future = handle.get_result_async()
        result_future.add_done_callback(lambda fut: self._on_plan_result(fut, robot, cand))

    def _on_plan_result(self, future, robot: Pose2D, cand: Candidate):
        try:
            result = future.result().result
            path = result.path
        except Exception as exc:
            self.get_logger().warn(f'PLAN_RESULT_FAILED | {exc}')
            self.state = 'PLANNING'
            self._try_next_candidate(robot)
            return

        if path is None or len(path.poses) < 2:
            self.get_logger().warn('PATH_INVALID_EMPTY')
            self.state = 'PLANNING'
            self._try_next_candidate(robot)
            return

        path_len = self._path_length(path)
        path_clear = self._path_min_clearance(path)
        if path_len < 0.05:
            self.get_logger().warn(f'PATH_REJECT_SHORT | len={path_len:.2f}')
            self.state = 'PLANNING'
            self._try_next_candidate(robot)
            return
        if path_clear < self.min_path_clearance:
            self.get_logger().warn(f'PATH_REJECT_CLEARANCE | min={path_clear:.2f}')
            self.state = 'PLANNING'
            self._try_next_candidate(robot)
            return

        self.get_logger().info(f'PATH_VALID | len={path_len:.2f} min_clear={path_clear:.2f} goal=({cand.x:.2f},{cand.y:.2f})')
        self._publish_path(path)
        if self.execute_with_nav2_navigator:
            self._send_nav_goal(cand)
        else:
            self._start_direct_follow(cand, path)

    def _send_nav_goal(self, cand: Candidate):
        if not self.nav_client.server_is_ready():
            self.get_logger().warn('NAV2_NAVIGATE_NOT_READY')
            
            robot = self._lookup_robot_pose()
            if not self._try_short_lidar_probe(robot, reason='navigator_not_ready'):
                self._start_micro_rotate(robot, reason='navigator_not_ready')
            return

        pose = self._candidate_to_pose(cand)
        goal = NavigateToPose.Goal()
        goal.pose = pose
        self.current_goal_pose = pose
        self.current_candidate = cand
        self.selected_goal_pub.publish(pose)

        self.state = 'NAV_SEND_WAIT'
        future = self.nav_client.send_goal_async(goal)
        future.add_done_callback(self._on_nav_goal_response)
        self._status(f'NAV_SEND goal=({cand.x:.2f},{cand.y:.2f}) source={cand.source}')

    def _on_nav_goal_response(self, future):
        try:
            handle = future.result()
        except Exception as exc:
            self.get_logger().warn(f'NAV_SEND_FAILED | {exc}')
            self._fail_current_goal('nav_send_failed')
            return
        if not handle.accepted:
            self.get_logger().warn('NAV_REJECTED')
            self._fail_current_goal('nav_rejected')
            return
        self.nav_goal_handle = handle
        self.nav_result_future = handle.get_result_async()
        self.nav_result_future.add_done_callback(self._on_nav_result)
        robot = self._lookup_robot_pose()
        self.nav_start_time = time.time()
        self.last_progress_time = self.nav_start_time
        if robot and self.current_candidate:
            self.best_goal_distance = math.hypot(self.current_candidate.x - robot.x, self.current_candidate.y - robot.y)
        else:
            self.best_goal_distance = float('inf')
        self.state = 'NAVIGATING'

    def _on_nav_result(self, future):
        try:
            wrapped = future.result()
            status = wrapped.status
        except Exception as exc:
            self.get_logger().warn(f'NAV_RESULT_FAILED | {exc}')
            self._fail_current_goal('nav_result_failed')
            return
        cand = self.current_candidate
        if status == GoalStatus.STATUS_SUCCEEDED:
            self.get_logger().info('NAV_SUCCESS')
            self._start_post_goal_observation()
        else:
            self.get_logger().warn(f'NAV_FAILED | status={status}')
            if cand:
                self._add_blacklist(cand.x, cand.y)
            self.state = 'IDLE'
            self.current_candidate = None
            self._stop_cmd()

    def _monitor_nav(self, robot: Pose2D):
        cand = self.current_candidate
        if cand is None:
            self.state = 'IDLE'
            return
        dist = math.hypot(cand.x - robot.x, cand.y - robot.y)
        if dist < self.goal_success_radius:
            self.get_logger().info(f'NAV_LOCAL_SUCCESS | dist={dist:.2f}; canceling Nav2 goal and observing')
            self._cancel_nav()
            self._start_post_goal_observation()
            return
        if dist < self.best_goal_distance - self.stuck_min_progress:
            self.best_goal_distance = dist
            self.last_progress_time = time.time()
        if time.time() - self.last_progress_time > self.stuck_timeout:
            self.get_logger().warn(f'NAV_STUCK | dist={dist:.2f} best={self.best_goal_distance:.2f}')
            self._cancel_nav()
            self._add_blacklist(cand.x, cand.y)
            self.state = 'IDLE'
            self.current_candidate = None

    # ------------------------------------------------------------------
    # Direct path follower: Nav2 planner path, local cmd_vel execution.
    # ------------------------------------------------------------------

    def _start_direct_follow(self, cand: Candidate, path: Path):
        self.current_candidate = cand
        self.current_path = path
        self.direct_path_points = [(ps.pose.position.x, ps.pose.position.y) for ps in path.poses]
        if len(self.direct_path_points) < 2:
            self.get_logger().warn('DIRECT_FOLLOW_REJECT_EMPTY_PATH')
            self.state = 'PLANNING'
            robot = self._lookup_robot_pose()
            if robot is not None:
                self._try_next_candidate(robot)
            return
        self.direct_follow_active = True
        self.state = 'DIRECT_FOLLOW'
        self.direct_follow_start_time = time.time()
        self.direct_follow_last_progress_time = self.direct_follow_start_time
        robot = self._lookup_robot_pose()
        if robot is not None:
            self.direct_follow_best_distance = math.hypot(cand.x - robot.x, cand.y - robot.y)
        else:
            self.direct_follow_best_distance = float('inf')
        pose = self._candidate_to_pose(cand)
        self.current_goal_pose = pose
        self.selected_goal_pub.publish(pose)
        self.get_logger().info(
            f'DIRECT_FOLLOW_START | Nav2 ComputePath path accepted; executing with cmd_vel '
            f'goal=({cand.x:.2f},{cand.y:.2f}) source={cand.source} points={len(self.direct_path_points)} '
            f'vmax={self.direct_follow_max_linear_speed:.2f}'
        )

    def _execute_direct_follow(self, robot: Pose2D):
        cand = self.current_candidate
        if cand is None or not self.direct_path_points:
            self.direct_follow_active = False
            self.state = 'IDLE'
            self._stop_cmd()
            return

        goal_dist = math.hypot(cand.x - robot.x, cand.y - robot.y)
        if goal_dist <= self.direct_goal_tolerance:
            self.get_logger().info(f'DIRECT_FOLLOW_SUCCESS | dist={goal_dist:.2f}')
            self.direct_follow_active = False
            self._stop_cmd()
            self._start_post_goal_observation()
            return

        if goal_dist < self.direct_follow_best_distance - self.direct_follow_progress_epsilon:
            self.direct_follow_best_distance = goal_dist
            self.direct_follow_last_progress_time = time.time()

        if time.time() - self.direct_follow_last_progress_time > self.direct_follow_stuck_timeout:
            self.get_logger().warn(
                f'DIRECT_FOLLOW_STUCK | dist={goal_dist:.2f} best={self.direct_follow_best_distance:.2f}; '
                f'blacklist and replan'
            )
            self._add_blacklist(cand.x, cand.y)
            self.direct_follow_active = False
            self.state = 'IDLE'
            self._stop_cmd()
            self._start_micro_rotate(robot, reason='direct_follow_stuck')
            return

        front = self._scan_min_sector(0.0, math.radians(25.0))
        side = min(self._scan_min_sector(math.radians(75.0), math.radians(22.0)),
                   self._scan_min_sector(math.radians(-75.0), math.radians(22.0)))
        if front < self.front_stop_distance or side < self.side_stop_distance:
            self.get_logger().warn(f'DIRECT_STOP_SAFETY | front={front:.2f} side={side:.2f}; blacklist and replan')
            self._add_blacklist(cand.x, cand.y)
            self.direct_follow_active = False
            self.state = 'IDLE'
            self._stop_cmd()
            self._start_micro_rotate(robot, reason='direct_safety_stop')
            return

        tx, ty = self._direct_lookahead_point(robot)
        dx = tx - robot.x
        dy = ty - robot.y
        target_dist = math.hypot(dx, dy)
        if target_dist < 1e-3:
            self._publish_cmd(0.0, 0.0)
            return
        target_yaw = math.atan2(dy, dx)
        yaw_err = self._wrap(target_yaw - robot.yaw)

        # Forward-only: rotate in place until the path target enters the forward cone.
        v = self.direct_follow_max_linear_speed
        abs_yaw = abs(yaw_err)
        if abs_yaw > math.radians(70.0):
            v = 0.0
        elif abs_yaw > math.radians(45.0):
            v *= 0.35
        elif abs_yaw > math.radians(25.0):
            v *= 0.65

        if v > 0.0:
            v = max(self.direct_follow_min_linear_speed, v)

        if front < self.front_slow_distance and v > 0.0:
            scale = (front - self.front_stop_distance) / max(1e-3, self.front_slow_distance - self.front_stop_distance)
            v *= max(0.25, min(1.0, scale))

        w = max(-self.direct_follow_max_angular_speed,
                min(self.direct_follow_max_angular_speed, self.direct_follow_angular_gain * yaw_err))
        self._publish_cmd(v, w)

    def _direct_lookahead_point(self, robot: Pose2D) -> Tuple[float, float]:
        pts = self.direct_path_points
        if not pts:
            return robot.x, robot.y
        nearest_i = 0
        nearest_d = float('inf')
        for i, (x, y) in enumerate(pts):
            d = math.hypot(x - robot.x, y - robot.y)
            if d < nearest_d:
                nearest_d = d
                nearest_i = i
        lookahead = max(0.15, self.direct_follow_lookahead_distance)
        accum = 0.0
        prev = (robot.x, robot.y)
        for i in range(nearest_i, len(pts)):
            cur = pts[i]
            accum += math.hypot(cur[0] - prev[0], cur[1] - prev[1])
            if accum >= lookahead:
                return cur
            prev = cur
        return pts[-1]

    # ------------------------------------------------------------------
    # Short probe fallback: only forward, short, and guarded by LiDAR.
    # ------------------------------------------------------------------

    def _probe_safety_blocked_now(self) -> Tuple[bool, float, float]:
        front = self._scan_min_sector(0.0, math.radians(28.0))
        side = min(self._scan_min_sector(math.radians(70.0), math.radians(25.0)),
                   self._scan_min_sector(math.radians(-70.0), math.radians(25.0)))
        blocked = (front < self.front_stop_distance + self.probe_safety_margin or
                   side < self.side_stop_distance + self.probe_safety_margin)
        return blocked, front, side

    def _probe_fail_blacklist_cost(self, x: float, y: float) -> float:
        now = time.time()
        for bx, by, t in self.probe_fail_blacklist:
            if now - t > self.probe_fail_cooldown:
                continue
            if math.hypot(x - bx, y - by) <= self.probe_fail_blacklist_radius:
                return 1.0
        return 0.0

    def _expire_probe_fail_blacklist(self):
        now = time.time()
        self.probe_fail_blacklist = [(x, y, t) for x, y, t in self.probe_fail_blacklist
                                     if now - t <= self.probe_fail_cooldown]

    def _mark_probe_failed(self, x: float, y: float, front: float, side: float):
        self.probe_fail_blacklist.append((x, y, time.time()))
        self.probe_blocked_until = time.time() + self.probe_fail_cooldown
        self.get_logger().warn(
            f'PROBE_FAIL_BLACKLIST | target=({x:.2f},{y:.2f}) cooldown={self.probe_fail_cooldown:.1f}s '
            f'front={front:.2f} side={side:.2f}; next cycle will use Nav2 candidates'
        )

    def _try_scan_open_space_probe(self, robot: Optional[Pose2D], reason: str) -> bool:
        """Short scan-only step for real robot operation when SLAM/map is lagging.

        This does not require an unknown-gain estimate from /map.  If /scan says a
        forward/side-front ray is open and local side clearance is acceptable, move
        a short distance.  Safety stops still blacklist the probe and fail over to
        Nav2 candidates / micro-rotate.
        """
        if not self.enable_scan_open_space_probe or robot is None or self.scan_msg is None:
            return False
        if time.time() < self.probe_blocked_until:
            return False
        blocked, front, side = self._probe_safety_blocked_now()
        if blocked:
            self.probe_blocked_until = time.time() + self.probe_fail_cooldown
            self.get_logger().warn(
                f'SCAN_OPEN_PROBE_SUPPRESSED_BY_SAFETY | reason={reason} front={front:.2f} side={side:.2f}; '
                f'cooldown={self.probe_fail_cooldown:.1f}s'
            )
            return False
        angle = self._best_open_probe_angle(require_unknown=False)
        if angle is None:
            return False
        r = self._scan_range_at_angle(angle)
        if not math.isfinite(r):
            r = self.scan_msg.range_max if self.scan_msg is not None else self.probe_distance
        target_dist = min(self.probe_distance, max(0.25, r - 0.35))
        if target_dist < self.probe_min_distance:
            return False
        tx = robot.x + target_dist * math.cos(robot.yaw + angle)
        ty = robot.y + target_dist * math.sin(robot.yaw + angle)
        if self._probe_fail_blacklist_cost(tx, ty) > 0.5:
            return False
        self.probe_target = (tx, ty)
        self.probe_start_time = time.time()
        self.probe_active = True
        self.state = 'PROBE'
        self._publish_straight_path(robot, tx, ty)
        ug, vg, fg = self._probe_ray_gain(angle)
        self.get_logger().info(
            f'SCAN_OPEN_SPACE_PROBE_START | reason={reason} angle={math.degrees(angle):.1f} '
            f'dist={target_dist:.2f} scan_range={r:.2f} gain[u/v/f]=[{ug}/{vg}/{fg}] target=({tx:.2f},{ty:.2f})'
        )
        return True

    def _try_unknown_first_lidar_probe(self, robot: Pose2D) -> bool:
        """Prefer short LiDAR probes into unmapped/unknown space, but never loop on unsafe probes."""
        if not self.enable_short_lidar_probe_fallback or self.scan_msg is None:
            return False
        if time.time() < self.probe_blocked_until:
            return False
        blocked, front, side = self._probe_safety_blocked_now()
        if blocked:
            self.probe_blocked_until = time.time() + self.probe_fail_cooldown
            self.get_logger().warn(
                f'UNKNOWN_FIRST_PROBE_SUPPRESSED_BY_SAFETY | front={front:.2f} side={side:.2f}; '
                f'trying Nav2 for {self.probe_fail_cooldown:.1f}s'
            )
            return False
        angle = self._best_open_probe_angle(require_unknown=True)
        if angle is None:
            return False
        ug, vg, fg = self._probe_ray_gain(angle)
        if ug < self.unknown_first_min_gain:
            return False
        r = self._scan_range_at_angle(angle)
        if not math.isfinite(r):
            r = self.scan_msg.range_max if self.scan_msg is not None else self.probe_distance
        target_dist = min(self.probe_distance, max(0.25, r - 0.35))
        if target_dist < self.probe_min_distance:
            self.get_logger().info(
                f'UNKNOWN_FIRST_PROBE_SKIPPED_SHORT | angle={math.degrees(angle):.1f} dist={target_dist:.2f} '
                f'min={self.probe_min_distance:.2f}'
            )
            return False
        tx = robot.x + target_dist * math.cos(robot.yaw + angle)
        ty = robot.y + target_dist * math.sin(robot.yaw + angle)
        if self._probe_fail_blacklist_cost(tx, ty) > 0.5:
            self.get_logger().info(f'UNKNOWN_FIRST_PROBE_SKIPPED_BLACKLIST | target=({tx:.2f},{ty:.2f})')
            return False
        self.probe_target = (tx, ty)
        self.probe_start_time = time.time()
        self.probe_active = True
        self.state = 'PROBE'
        self._publish_straight_path(robot, tx, ty)
        self.get_logger().info(
            f'UNKNOWN_FIRST_PROBE_START | angle={math.degrees(angle):.1f} '
            f'dist={target_dist:.2f} gain[u/v/f]=[{ug}/{vg}/{fg}] target=({tx:.2f},{ty:.2f})'
        )
        return True

    def _try_short_lidar_probe(self, robot: Optional[Pose2D], reason: str) -> bool:
        self.state = 'IDLE'
        if time.time() < self.probe_blocked_until:
            self._status(f'{reason} | PROBE_COOLDOWN_TRY_NAV2_OR_ROTATE')
            return False
        if not self.enable_short_lidar_probe_fallback or robot is None or self.scan_msg is None:
            self._status(f'{reason} | NO_PROBE_AVAILABLE')
            return False
        blocked, front, side = self._probe_safety_blocked_now()
        if blocked:
            self.probe_blocked_until = time.time() + self.probe_fail_cooldown
            self.get_logger().warn(
                f'SHORT_PROBE_SUPPRESSED_BY_SAFETY | reason={reason} front={front:.2f} side={side:.2f}; '
                f'cooldown={self.probe_fail_cooldown:.1f}s'
            )
            return False
        angle = self._best_open_probe_angle()
        if angle is None:
            self._status(f'{reason} | NO_SAFE_FRONT_PROBE')
            return False
        r = self._scan_range_at_angle(angle)
        if not math.isfinite(r):
            r = self.scan_msg.range_max if self.scan_msg is not None else self.probe_distance
        target_dist = min(self.probe_distance, max(0.25, r - 0.35))
        if target_dist < self.probe_min_distance:
            self._status(f'{reason} | PROBE_TOO_SHORT dist={target_dist:.2f}')
            return False
        tx = robot.x + target_dist * math.cos(robot.yaw + angle)
        ty = robot.y + target_dist * math.sin(robot.yaw + angle)
        if self._probe_fail_blacklist_cost(tx, ty) > 0.5:
            self._status(f'{reason} | PROBE_TARGET_BLACKLISTED')
            return False
        self.probe_target = (tx, ty)
        self.probe_start_time = time.time()
        self.probe_active = True
        self.state = 'PROBE'
        self._publish_straight_path(robot, tx, ty)
        self.get_logger().info(f'SHORT_LIDAR_PROBE_START | reason={reason} angle={math.degrees(angle):.1f} target=({tx:.2f},{ty:.2f})')
        return True

    def _execute_probe(self, robot: Pose2D):
        if self.probe_target is None:
            self.probe_active = False
            self.state = 'IDLE'
            self._stop_cmd()
            return
        front = self._scan_min_sector(0.0, math.radians(28.0))
        side = min(self._scan_min_sector(math.radians(70.0), math.radians(25.0)),
                   self._scan_min_sector(math.radians(-70.0), math.radians(25.0)))
        tx, ty = self.probe_target
        if front < self.front_stop_distance or side < self.side_stop_distance:
            self.get_logger().warn(f'PROBE_STOP_SAFETY | front={front:.2f} side={side:.2f}')
            self._mark_probe_failed(tx, ty, front, side)
            self.probe_active = False
            self.state = 'IDLE'
            self._stop_cmd()
            return
        dx = tx - robot.x
        dy = ty - robot.y
        dist = math.hypot(dx, dy)
        if dist < self.probe_success_radius or time.time() - self.probe_start_time > self.probe_timeout:
            self.get_logger().info(f'PROBE_DONE | dist={dist:.2f}')
            self.probe_active = False
            self.state = 'IDLE'
            self._stop_cmd()
            return
        target_yaw = math.atan2(dy, dx)
        yaw_err = self._wrap(target_yaw - robot.yaw)
        v = self.probe_linear_speed
        if abs(yaw_err) > math.radians(55.0):
            v = 0.0
        elif abs(yaw_err) > math.radians(30.0):
            v *= 0.45
        if front < self.front_slow_distance:
            v *= max(0.25, (front - self.front_stop_distance) / max(1e-3, self.front_slow_distance - self.front_stop_distance))
        w = max(-0.9, min(0.9, self.probe_angular_gain * yaw_err))
        self._publish_cmd(v, w)

    def _best_open_probe_angle(self, require_unknown: bool = False) -> Optional[float]:
        scan = self.scan_msg
        if scan is None:
            return None
        half = self.probe_fov / 2.0
        best_angle = None
        best_score = -1e9
        count = len(scan.ranges)
        if count == 0:
            return None
        for i, raw in enumerate(scan.ranges):
            if not math.isfinite(raw):
                rng = scan.range_max
            else:
                rng = raw
            a = scan.angle_min + i * scan.angle_increment
            if abs(a) > half:
                continue
            if rng < self.probe_min_front_clearance:
                continue
            side_left = self._scan_min_sector(a + math.radians(55.0), math.radians(18.0))
            side_right = self._scan_min_sector(a - math.radians(55.0), math.radians(18.0))
            if min(side_left, side_right) < self.probe_side_clearance:
                continue
            unknown_gain, visual_gain, frontier_gain = self._probe_ray_gain(a)
            if require_unknown and unknown_gain < self.unknown_first_min_gain:
                continue
            open_dist = min(rng, self.probe_distance)
            forward_bonus = max(0.0, math.cos(a))
            score = (
                14.0 * unknown_gain
                + 0.08 * visual_gain
                + 1.6 * frontier_gain
                + 2.5 * open_dist
                + 0.9 * forward_bonus
                - 0.75 * abs(a)
            )
            if score > best_score:
                best_score = score
                best_angle = a
        if best_angle is None or best_score < 0.5:
            return None
        return best_angle

    def _start_micro_rotate(self, robot: Optional[Pose2D], reason: str) -> bool:
        if not self.micro_rotate_on_total_failure or robot is None:
            self._status(f'{reason} | NO_RECOVERY')
            return False
        left = self._scan_min_sector(math.radians(80.0), math.radians(35.0))
        right = self._scan_min_sector(math.radians(-80.0), math.radians(35.0))
        self.micro_rotate_direction = 1.0 if left >= right else -1.0
        self.micro_rotate_start_time = time.time()
        self.micro_rotate_active = True
        self.state = 'MICRO_ROTATE'
        self.get_logger().warn(
            f'MICRO_ROTATE_START | reason={reason} dir={self.micro_rotate_direction:+.0f} '
            f'left={left:.2f} right={right:.2f} duration={self.micro_rotate_duration:.2f}s'
        )
        return True

    def _execute_micro_rotate(self, robot: Pose2D):
        if time.time() - self.micro_rotate_start_time >= self.micro_rotate_duration:
            self.micro_rotate_active = False
            self.state = 'IDLE'
            self._stop_cmd()
            self.get_logger().warn('MICRO_ROTATE_DONE | replanning')
            return
        self._publish_cmd(0.0, self.micro_rotate_direction * self.micro_rotate_speed)

    # ------------------------------------------------------------------
    # Candidate generation
    # ------------------------------------------------------------------

    def _frontier_candidates(self, clusters: List[List[Tuple[int, int]]], robot: Pose2D) -> List[Candidate]:
        out: List[Candidate] = []
        for cluster in clusters:
            cand = self._candidate_from_frontier_cluster(cluster, robot)
            if cand:
                out.append(cand)
        return out

    def _candidate_from_frontier_cluster(self, cluster: List[Tuple[int, int]], robot: Pose2D) -> Optional[Candidate]:
        if self.grid is None:
            return None
        mx = sum(c[0] for c in cluster) / len(cluster)
        my = sum(c[1] for c in cluster) / len(cluster)
        ux, uy, n = 0.0, 0.0, 0
        for cx, cy in cluster:
            for nx, ny in self._neighbors8(cx, cy):
                if self._is_unknown_cell(nx, ny):
                    ux += nx - cx
                    uy += ny - cy
                    n += 1
        if n == 0:
            return None
        norm = math.hypot(ux, uy)
        if norm < 1e-6:
            return None
        ux /= norm
        uy /= norm
        off = self.frontier_goal_offset / self.grid.resolution
        gx = int(round(mx - ux * off))
        gy = int(round(my - uy * off))
        safe = self._nearest_safe_cell(gx, gy, max_radius_m=0.9)
        if safe is None:
            return None
        wx, wy = self._cell_to_world(*safe)
        view_yaw = math.atan2(uy, ux)
        return self._make_candidate(wx, wy, view_yaw, len(cluster), robot, source='frontier')

    def _coverage_sweep_candidates(self, reachable: Set[Tuple[int, int]], frontiers: Set[Tuple[int, int]], robot: Pose2D) -> List[Candidate]:
        out: List[Candidate] = []
        if self.coverage is None or self.grid is None:
            return out
        stride = max(1, self.candidate_stride_cells)
        for cx, cy in reachable:
            if cx % stride != 0 or cy % stride != 0:
                continue
            idx = self._idx(cx, cy)
            if self.coverage[idx] >= self.coverage_candidate_max_checked and not self._near_frontier(cx, cy, frontiers, 3):
                continue
            wx, wy = self._cell_to_world(cx, cy)
            dist = math.hypot(wx - robot.x, wy - robot.y)
            if dist < self.min_goal_distance or dist > self.max_goal_distance:
                continue
            yaw_to_goal = math.atan2(wy - robot.y, wx - robot.x)
            bearing = abs(self._wrap(yaw_to_goal - robot.yaw))
            if bearing > self.max_goal_bearing:
                continue
            if self._clearance_at_cell(cx, cy) < self.min_goal_clearance:
                continue
            # Evaluate a small set of view directions; keep best.
            best: Optional[Candidate] = None
            for dyaw in [0.0, math.radians(30), math.radians(-30), math.radians(60), math.radians(-60), math.radians(90), math.radians(-90), math.radians(135), math.radians(-135), math.pi]:
                view_yaw = self._wrap(yaw_to_goal + dyaw)
                cand = self._make_candidate(wx, wy, view_yaw, 0, robot, source='coverage')
                if cand is not None and (best is None or cand.score > best.score):
                    best = cand
            if best:
                out.append(best)
        return out

    def _make_candidate(self, x: float, y: float, view_yaw: float, frontier_size: int, robot: Pose2D, source: str) -> Optional[Candidate]:
        cell = self._world_to_cell(x, y)
        if cell is None:
            return None
        if not self._is_free_cell(*cell):
            return None
        clearance = self._clearance_at_cell(*cell)
        if clearance < self.min_goal_clearance:
            return None
        dist = math.hypot(x - robot.x, y - robot.y)
        if dist < self.min_goal_distance or dist > self.max_goal_distance:
            return None
        motion_yaw = math.atan2(y - robot.y, x - robot.x)
        bearing = abs(self._wrap(motion_yaw - robot.yaw))
        if bearing > self.max_goal_bearing:
            return None
        unknown_gain, visual_gain, frontier_gain = self._estimate_view_gain(x, y, view_yaw)
        blacklist = self._blacklist_cost(x, y)
        if blacklist > 0.98:
            return None
        cell_unchecked = 0.0
        if self.coverage is not None:
            cell_unchecked = max(0.0, min(1.0, (self.visual_unchecked_threshold - self.coverage[cell[0] + cell[1] * self.grid.width]) / max(1.0, float(self.visual_unchecked_threshold))))
        coverage_bonus = self.coverage_source_score_bonus * cell_unchecked if source == 'coverage' else 0.0
        score = (
            self.w_unknown * unknown_gain
            + self.w_visual * visual_gain
            + self.w_frontier * (frontier_gain + 0.15 * frontier_size)
            + self.w_clearance * clearance
            + coverage_bonus
            - self.w_distance * dist
            - self.w_blacklist * blacklist
            - self.w_heading * bearing
        )
        return Candidate(x, y, motion_yaw, view_yaw, score, unknown_gain, visual_gain, frontier_gain, clearance, dist, source)

    # ------------------------------------------------------------------
    # Coverage and gains
    # ------------------------------------------------------------------

    def _update_visual_coverage(self, robot: Pose2D):
        if self.coverage is None or self.grid is None or self.map_msg is None:
            return
        newly = 0
        # robot disk
        rc = self._world_to_cell(robot.x, robot.y)
        if rc:
            rad = int(math.ceil(self.coverage_robot_radius / self.grid.resolution))
            for dy in range(-rad, rad + 1):
                for dx in range(-rad, rad + 1):
                    if math.hypot(dx, dy) * self.grid.resolution > self.coverage_robot_radius:
                        continue
                    cx, cy = rc[0] + dx, rc[1] + dy
                    if self._in_bounds(cx, cy):
                        idx = self._idx(cx, cy)
                        if self.coverage[idx] < 100:
                            self.coverage[idx] = 100
                            newly += 1
        # camera-like sector
        ray_count = max(3, self.view_ray_count)
        for i in range(ray_count):
            a = robot.yaw - self.view_fov / 2.0 + self.view_fov * i / max(1, ray_count - 1)
            r = 0.0
            while r <= self.view_max_range:
                wx = robot.x + r * math.cos(a)
                wy = robot.y + r * math.sin(a)
                cell = self._world_to_cell(wx, wy)
                if cell is None:
                    break
                cx, cy = cell
                idx = self._idx(cx, cy)
                val = self.map_msg.data[idx]
                if self._is_occupied_value(val):
                    break
                if self.coverage[idx] < 100:
                    self.coverage[idx] = 100
                    newly += 1
                if self._is_unknown_value(val):
                    break
                r += self.grid.resolution
        if newly > 0:
            total = sum(1 for v in self.coverage if v >= 80)
            self.get_logger().info(f'COVERAGE_UPDATE | newly={newly} total_checked={total} rays={self.view_ray_count} fov={math.degrees(self.view_fov):.1f}deg')

    def _estimate_view_gain(self, x: float, y: float, yaw: float) -> Tuple[int, int, int]:
        if self.coverage is None or self.grid is None or self.map_msg is None:
            return 0, 0, 0
        unknown: Set[int] = set()
        visual: Set[int] = set()
        frontier: Set[int] = set()
        ray_count = max(3, self.view_ray_count)
        for i in range(ray_count):
            a = yaw - self.view_fov / 2.0 + self.view_fov * i / max(1, ray_count - 1)
            r = 0.0
            while r <= self.view_max_range:
                wx = x + r * math.cos(a)
                wy = y + r * math.sin(a)
                cell = self._world_to_cell(wx, wy)
                if cell is None:
                    unknown.add(-1000000 - i)
                    break
                cx, cy = cell
                idx = self._idx(cx, cy)
                val = self.map_msg.data[idx]
                if self._is_occupied_value(val):
                    break
                if self._is_unknown_value(val):
                    unknown.add(idx)
                    break
                if self._is_free_value(val):
                    if self.coverage[idx] < self.visual_unchecked_threshold:
                        visual.add(idx)
                    if self._is_frontier_cell(cx, cy):
                        frontier.add(idx)
                r += self.grid.resolution
        return len(unknown), len(visual), len(frontier)

    def _probe_ray_gain(self, rel_angle: float) -> Tuple[int, int, int]:
        robot = self._lookup_robot_pose()
        if robot is None:
            return 0, 0, 0
        return self._estimate_view_gain(robot.x, robot.y, robot.yaw + rel_angle)

    def _publish_coverage_periodic(self):
        now = time.time()
        if now - self.last_cov_pub_time < self.coverage_publish_period:
            return
        self.last_cov_pub_time = now
        if self.map_msg is None or self.coverage is None:
            return
        msg = OccupancyGrid()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.global_frame
        msg.info = self.map_msg.info
        data = []
        for i, occ in enumerate(self.map_msg.data):
            if self._is_occupied_value(occ):
                data.append(-1)
            else:
                data.append(int(self.coverage[i]))
        msg.data = data
        self.coverage_pub.publish(msg)

    # ------------------------------------------------------------------
    # Frontier / grid algorithms
    # ------------------------------------------------------------------

    def _reachable_free(self, start: Tuple[int, int]) -> Set[Tuple[int, int]]:
        if not self._is_free_cell(*start):
            # Permit the immediate robot area if SLAM has not labeled it as free yet.
            return {start} if self._in_bounds(*start) else set()
        q = deque([start])
        seen: Set[Tuple[int, int]] = set()
        while q:
            c = q.popleft()
            if c in seen:
                continue
            seen.add(c)
            for n in self._neighbors4(*c):
                if n not in seen and self._is_free_cell(*n):
                    q.append(n)
        return seen

    def _detect_frontiers(self, reachable: Set[Tuple[int, int]]) -> Set[Tuple[int, int]]:
        out = set()
        for cx, cy in reachable:
            if self._is_frontier_cell(cx, cy):
                out.add((cx, cy))
        return out

    def _cluster_frontiers(self, frontiers: Set[Tuple[int, int]]) -> List[List[Tuple[int, int]]]:
        clusters = []
        visited = set()
        for cell in frontiers:
            if cell in visited:
                continue
            q = deque([cell])
            visited.add(cell)
            cluster = []
            while q:
                c = q.popleft()
                cluster.append(c)
                for n in self._neighbors8(*c):
                    if n in frontiers and n not in visited:
                        visited.add(n)
                        q.append(n)
            if len(cluster) >= self.min_frontier_cluster_size:
                clusters.append(cluster)
        return clusters

    def _compute_clearance(self) -> List[float]:
        assert self.grid is not None and self.map_msg is not None
        n = self.grid.width * self.grid.height
        inf = 1e9
        dist = [inf] * n
        q = deque()
        for y in range(self.grid.height):
            for x in range(self.grid.width):
                idx = self._idx(x, y)
                val = self.map_msg.data[idx]
                if self._is_occupied_value(val) or (self.unknown_is_obstacle_for_clearance and self._is_unknown_value(val)):
                    dist[idx] = 0.0
                    q.append((x, y))
        if not q:
            return [10.0] * n
        while q:
            x, y = q.popleft()
            base = dist[self._idx(x, y)]
            for nx, ny in self._neighbors8(x, y):
                step = math.hypot(nx - x, ny - y) * self.grid.resolution
                ni = self._idx(nx, ny)
                nd = base + step
                if nd < dist[ni]:
                    dist[ni] = nd
                    q.append((nx, ny))
        return dist

    def _nearest_safe_cell(self, cx: int, cy: int, max_radius_m: float) -> Optional[Tuple[int, int]]:
        if self.grid is None:
            return None
        max_r = int(math.ceil(max_radius_m / self.grid.resolution))
        best = None
        best_d = 1e9
        for r in range(0, max_r + 1):
            for dy in range(-r, r + 1):
                for dx in range(-r, r + 1):
                    if r > 0 and abs(dx) != r and abs(dy) != r:
                        continue
                    nx, ny = cx + dx, cy + dy
                    if not self._in_bounds(nx, ny):
                        continue
                    if not self._is_free_cell(nx, ny):
                        continue
                    clear = self._clearance_at_cell(nx, ny)
                    if clear < self.min_goal_clearance:
                        continue
                    d = dx * dx + dy * dy
                    if d < best_d:
                        best_d = d
                        best = (nx, ny)
            if best is not None:
                return best
        return None

    # ------------------------------------------------------------------
    # Post-goal observation: face the chosen NBV yaw so low-coverage cells
    # actually enter the 60 degree visual coverage cone.
    # ------------------------------------------------------------------

    def _start_post_goal_observation(self):
        cand = self.current_candidate
        if cand is not None and self.enable_view_yaw_align:
            self._start_view_align(cand)
            return
        if self.enable_sector_scan:
            self._start_sector_scan()
            return
        self.state = 'IDLE'
        self.current_candidate = None

    def _start_view_align(self, cand: Candidate):
        self.view_align_target_yaw = cand.view_yaw
        self.view_align_start_time = time.time()
        self.view_align_active = True
        self.state = 'VIEW_ALIGN'
        self._stop_cmd()
        self.get_logger().info(
            f'VIEW_ALIGN_START | target_yaw={math.degrees(cand.view_yaw):.1f}deg '
            f'visual_gain={cand.visual_gain} unknown_gain={cand.unknown_gain} source={cand.source}'
        )

    def _execute_view_align(self, robot: Pose2D):
        yaw_err = self._wrap(self.view_align_target_yaw - robot.yaw)
        if abs(yaw_err) <= self.view_align_tolerance or time.time() - self.view_align_start_time >= self.view_align_timeout:
            self.view_align_active = False
            self._stop_cmd()
            self.get_logger().info(f'VIEW_ALIGN_DONE | yaw_err={math.degrees(yaw_err):.1f}deg')
            if self.enable_sector_scan:
                self._start_sector_scan()
            else:
                self.state = 'IDLE'
                self.current_candidate = None
            return
        front = self._scan_min_sector(0.0, math.radians(24.0))
        if front < self.front_stop_distance:
            self.view_align_active = False
            self.state = 'IDLE'
            self.current_candidate = None
            self._stop_cmd()
            self.get_logger().warn(f'VIEW_ALIGN_ABORT_FRONT | front={front:.2f}')
            return
        w = max(-self.view_align_angular_speed, min(self.view_align_angular_speed, 1.8 * yaw_err))
        self._publish_cmd(0.0, w)

    # ------------------------------------------------------------------
    # Sector scan after goal success
    # ------------------------------------------------------------------

    def _start_sector_scan(self):
        robot = self._lookup_robot_pose()
        if robot is None:
            self.state = 'IDLE'
            return
        self.scanning = True
        self.state = 'SCANNING'
        self.scan_start_yaw = robot.yaw
        self.scan_target_abs = abs(self.sector_scan_angle)
        self.scan_direction = 1.0 if self.sector_scan_angle >= 0 else -1.0
        self._stop_cmd()
        self.get_logger().info(f'SECTOR_SCAN_START | angle={math.degrees(self.sector_scan_angle):.1f}deg')

    def _execute_sector_scan(self, robot: Pose2D):
        delta = abs(self._wrap(robot.yaw - self.scan_start_yaw))
        if delta >= self.scan_target_abs:
            self.scanning = False
            self.state = 'IDLE'
            self.current_candidate = None
            self._stop_cmd()
            self.get_logger().info('SECTOR_SCAN_DONE')
            return
        front = self._scan_min_sector(0.0, math.radians(24.0))
        if front < self.front_stop_distance:
            self.scanning = False
            self.state = 'IDLE'
            self._stop_cmd()
            self.get_logger().warn(f'SECTOR_SCAN_ABORT_FRONT | front={front:.2f}')
            return
        self._publish_cmd(0.0, self.scan_direction * self.sector_scan_angular_speed)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _lookup_robot_pose(self) -> Optional[Pose2D]:
        try:
            tf = self.tf_buffer.lookup_transform(
                self.global_frame,
                self.robot_frame,
                rclpy.time.Time(),
                timeout=Duration(seconds=0.03),
            )
        except Exception:
            return None
        t = tf.transform.translation
        q = tf.transform.rotation
        yaw = self._yaw_from_quat(q.x, q.y, q.z, q.w)
        return Pose2D(t.x, t.y, yaw)

    def _candidate_to_pose(self, cand: Candidate) -> PoseStamped:
        msg = PoseStamped()
        # Use time=0 to ask TF/Nav2 for the latest available transform.
        # This avoids small sim-time extrapolation errors such as requested 108.220 while latest is 108.200.
        msg.header.stamp = rclpy.time.Time().to_msg()
        msg.header.frame_id = self.global_frame
        msg.pose.position.x = cand.x
        msg.pose.position.y = cand.y
        msg.pose.position.z = 0.0
        qx, qy, qz, qw = self._quat_from_yaw(cand.nav_yaw)
        msg.pose.orientation.x = qx
        msg.pose.orientation.y = qy
        msg.pose.orientation.z = qz
        msg.pose.orientation.w = qw
        return msg

    def _path_length(self, path: Path) -> float:
        if len(path.poses) < 2:
            return 0.0
        length = 0.0
        prev = path.poses[0].pose.position
        for ps in path.poses[1:]:
            cur = ps.pose.position
            length += math.hypot(cur.x - prev.x, cur.y - prev.y)
            prev = cur
        return length

    def _path_min_clearance(self, path: Path) -> float:
        if self.clearance is None:
            self.clearance = self._compute_clearance()
        best = 10.0
        for ps in path.poses:
            c = self._world_to_cell(ps.pose.position.x, ps.pose.position.y)
            if c is None:
                best = min(best, 0.0)
            else:
                best = min(best, self._clearance_at_cell(*c))
        return best

    def _compute_internal_grid_path(self, robot: Pose2D, cand: Candidate) -> Optional[Path]:
        """A lightweight A* planner over the current /map.

        This is intentionally conservative and only traverses cells already marked
        as free in the SLAM map. Unknown-space expansion is handled separately by
        LiDAR probe. The purpose is to remove the dependency on Nav2 planner when
        Cartographer is used as the external SLAM provider.
        """
        if self.grid is None or self.map_msg is None:
            return None
        if self.clearance is None:
            self.clearance = self._compute_clearance()

        start = self._world_to_cell(robot.x, robot.y)
        goal = self._world_to_cell(cand.x, cand.y)
        if start is None or goal is None:
            return None

        if not self._is_free_cell(*goal):
            safe_goal = self._nearest_safe_cell(goal[0], goal[1], max_radius_m=0.55)
            if safe_goal is None:
                return None
            goal = safe_goal

        if not self._is_free_cell(*start):
            safe_start = self._nearest_safe_cell(start[0], start[1], max_radius_m=0.45)
            if safe_start is None:
                return None
            start = safe_start

        def passable(c: Tuple[int, int]) -> bool:
            x, y = c
            if not self._in_bounds(x, y):
                return False
            if not self._is_free_cell(x, y):
                return False
            # Be slightly more permissive than candidate clearance to avoid
            # rejecting every path near initial sparse maps, but never below
            # robot_radius minus small tolerance.
            min_clear = max(0.12, self.min_path_clearance)
            return self._clearance_at_cell(x, y) >= min_clear

        if not passable(goal):
            safe_goal = self._nearest_safe_cell(goal[0], goal[1], max_radius_m=0.75)
            if safe_goal is None or not passable(safe_goal):
                return None
            goal = safe_goal

        open_heap: List[Tuple[float, float, Tuple[int, int]]] = []
        g_cost: Dict[Tuple[int, int], float] = {start: 0.0}
        parent: Dict[Tuple[int, int], Tuple[int, int]] = {}
        sx, sy = start
        gx, gy = goal
        h0 = math.hypot(gx - sx, gy - sy) * self.grid.resolution
        heapq.heappush(open_heap, (h0, 0.0, start))
        closed: Set[Tuple[int, int]] = set()
        expansions = 0

        while open_heap and expansions < self.internal_planner_max_expansions:
            _, gc, cur = heapq.heappop(open_heap)
            if cur in closed:
                continue
            closed.add(cur)
            expansions += 1
            if cur == goal:
                break
            cx, cy = cur
            for nx, ny in self._neighbors8(cx, cy):
                nxt = (nx, ny)
                if nxt in closed or not passable(nxt):
                    continue
                step = math.hypot(nx - cx, ny - cy) * self.grid.resolution
                # Small cost for low-clearance cells, so paths prefer center lines
                # when available but still work in narrow real-world corridors.
                clear = max(0.01, self._clearance_at_cell(nx, ny))
                clearance_cost = 0.02 / clear
                ng = gc + step + clearance_cost
                if ng < g_cost.get(nxt, float('inf')):
                    g_cost[nxt] = ng
                    parent[nxt] = cur
                    h = math.hypot(gx - nx, gy - ny) * self.grid.resolution
                    heapq.heappush(open_heap, (ng + h, ng, nxt))

        if goal not in parent and goal != start:
            self.get_logger().warn(
                f'INTERNAL_PATH_FAIL | expansions={expansions} start={start} goal={goal} '
                f'closed={len(closed)}'
            )
            return None

        cells = [goal]
        while cells[-1] != start:
            cells.append(parent[cells[-1]])
        cells.reverse()

        # Downsample nearly-straight grid path so the direct follower gets a stable,
        # not overly dense route. Always keep start and final goal.
        sampled: List[Tuple[int, int]] = []
        last_dir = None
        for i, c in enumerate(cells):
            if i == 0 or i == len(cells) - 1:
                sampled.append(c)
                if i > 0:
                    prev = cells[i - 1]
                    last_dir = (c[0] - prev[0], c[1] - prev[1])
                continue
            prev = cells[i - 1]
            nxt = cells[i + 1]
            cur_dir = (c[0] - prev[0], c[1] - prev[1])
            next_dir = (nxt[0] - c[0], nxt[1] - c[1])
            if cur_dir != next_dir:
                sampled.append(c)
                last_dir = next_dir
            elif len(sampled) == 0 or math.hypot(c[0]-sampled[-1][0], c[1]-sampled[-1][1]) * self.grid.resolution >= 0.25:
                sampled.append(c)

        msg = Path()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.global_frame
        for i, c in enumerate(sampled):
            wx, wy = self._cell_to_world(*c)
            ps = PoseStamped()
            ps.header = msg.header
            ps.pose.position.x = wx
            ps.pose.position.y = wy
            if i + 1 < len(sampled):
                nx, ny = self._cell_to_world(*sampled[i + 1])
                yaw = math.atan2(ny - wy, nx - wx)
            elif i > 0:
                px, py = self._cell_to_world(*sampled[i - 1])
                yaw = math.atan2(wy - py, wx - px)
            else:
                yaw = robot.yaw
            qx, qy, qz, qw = self._quat_from_yaw(yaw)
            ps.pose.orientation.x = qx
            ps.pose.orientation.y = qy
            ps.pose.orientation.z = qz
            ps.pose.orientation.w = qw
            msg.poses.append(ps)
        return msg

    def _clearance_at_cell(self, cx: int, cy: int) -> float:
        if self.clearance is None:
            self.clearance = self._compute_clearance()
        if not self._in_bounds(cx, cy):
            return 0.0
        return float(self.clearance[self._idx(cx, cy)])

    def _near_frontier(self, cx: int, cy: int, frontiers: Set[Tuple[int, int]], r: int) -> bool:
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                if (cx + dx, cy + dy) in frontiers:
                    return True
        return False

    def _publish_path_msg(self, msg: Path):
        # Publish the same path to both the package-specific topic and /plan.
        # RViz examples commonly use /plan, while the explorer internally used
        # /visibility_explorer/path. Keeping both removes display ambiguity.
        self.path_pub.publish(msg)
        self.plan_alias_pub.publish(msg)

    def _publish_path(self, path: Optional[Path]):
        if path is None:
            msg = Path()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = self.global_frame
            self._publish_path_msg(msg)
        else:
            path.header.stamp = self.get_clock().now().to_msg()
            path.header.frame_id = self.global_frame
            for ps in path.poses:
                ps.header.stamp = path.header.stamp
                ps.header.frame_id = path.header.frame_id
            self._publish_path_msg(path)

    def _publish_straight_path(self, robot: Pose2D, tx: float, ty: float):
        msg = Path()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.global_frame
        steps = 12
        for i in range(steps + 1):
            t = i / steps
            ps = PoseStamped()
            ps.header = msg.header
            ps.pose.position.x = robot.x * (1 - t) + tx * t
            ps.pose.position.y = robot.y * (1 - t) + ty * t
            yaw = math.atan2(ty - robot.y, tx - robot.x)
            qx, qy, qz, qw = self._quat_from_yaw(yaw)
            ps.pose.orientation.x = qx
            ps.pose.orientation.y = qy
            ps.pose.orientation.z = qz
            ps.pose.orientation.w = qw
            msg.poses.append(ps)
        self._publish_path_msg(msg)

    def _clear_markers(self):
        arr = MarkerArray()
        m = Marker()
        m.header.stamp = self.get_clock().now().to_msg()
        m.header.frame_id = self.global_frame
        m.action = Marker.DELETEALL
        arr.markers.append(m)
        self.marker_pub.publish(arr)

    def _status(self, text: str):
        now = time.time()
        if now - self.last_status_time < 1.5:
            return
        self.last_status_time = now
        msg = String()
        msg.data = f'{self.version_name} | state={self.state} | {text}'
        self.status_pub.publish(msg)
        self.get_logger().info(msg.data)

    def _fail_current_goal(self, reason: str):
        if self.current_candidate:
            self._add_blacklist(self.current_candidate.x, self.current_candidate.y)
        self._cancel_nav()
        self.state = 'IDLE'
        self.current_candidate = None
        self._stop_cmd()
        self.get_logger().warn(f'GOAL_FAIL | {reason}')

    def _cancel_nav(self):
        if self.nav_goal_handle is not None:
            try:
                self.nav_goal_handle.cancel_goal_async()
            except Exception:
                pass
        self.nav_goal_handle = None
        self.nav_result_future = None

    def _publish_cmd(self, v: float, w: float):
        if self.cmd_vel_stamped:
            msg = TwistStamped()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = self.robot_frame
            msg.twist.linear.x = float(v)
            msg.twist.angular.z = float(w)
            self.cmd_vel_pub.publish(msg)
        else:
            msg = Twist()
            msg.linear.x = float(v)
            msg.angular.z = float(w)
            self.cmd_vel_pub.publish(msg)

    def _stop_cmd(self):
        try:
            self._publish_cmd(0.0, 0.0)
        except Exception:
            pass

    def _scan_range_at_angle(self, angle: float) -> float:
        scan = self.scan_msg
        if scan is None or not scan.ranges:
            return float('inf')
        idx = int(round((angle - scan.angle_min) / scan.angle_increment))
        idx = max(0, min(len(scan.ranges) - 1, idx))
        r = scan.ranges[idx]
        if not math.isfinite(r):
            return scan.range_max
        return r

    def _scan_min_sector(self, center: float, half_width: float) -> float:
        scan = self.scan_msg
        if scan is None or len(scan.ranges) == 0:
            return float('inf')
        best = float('inf')
        for i, raw in enumerate(scan.ranges):
            a = scan.angle_min + i * scan.angle_increment
            if abs(self._wrap(a - center)) <= half_width:
                if math.isfinite(raw):
                    best = min(best, raw)
        return best

    def _add_blacklist(self, x: float, y: float):
        self.blacklist.append((x, y, time.time()))
        self.get_logger().warn(f'BLACKLIST_ADD | ({x:.2f},{y:.2f})')

    def _expire_blacklist(self):
        now = time.time()
        self.blacklist = [(x, y, t) for x, y, t in self.blacklist if now - t <= self.blacklist_duration]

    def _blacklist_cost(self, x: float, y: float) -> float:
        now = time.time()
        cost = 0.0
        for bx, by, t in self.blacklist:
            d = math.hypot(x - bx, y - by)
            if d <= self.blacklist_radius:
                cost = max(cost, math.exp(-(now - t) / max(1e-3, self.blacklist_duration)))
        return cost

    # grid basics
    def _resize_coverage(self, new_grid: GridInfo):
        if self.coverage is None or self.grid is None:
            self.coverage = [0 for _ in range(new_grid.width * new_grid.height)]
            return
        old_grid = self.grid
        old_cov = self.coverage
        new_cov = [0 for _ in range(new_grid.width * new_grid.height)]
        for ny in range(new_grid.height):
            for nx in range(new_grid.width):
                wx = new_grid.origin_x + (nx + 0.5) * new_grid.resolution
                wy = new_grid.origin_y + (ny + 0.5) * new_grid.resolution
                ox = int((wx - old_grid.origin_x) / old_grid.resolution)
                oy = int((wy - old_grid.origin_y) / old_grid.resolution)
                if 0 <= ox < old_grid.width and 0 <= oy < old_grid.height:
                    new_cov[ny * new_grid.width + nx] = old_cov[oy * old_grid.width + ox]
        self.coverage = new_cov

    def _grid_changed(self, a: GridInfo, b: GridInfo) -> bool:
        return a.width != b.width or a.height != b.height or abs(a.resolution - b.resolution) > 1e-9 or abs(a.origin_x - b.origin_x) > 1e-6 or abs(a.origin_y - b.origin_y) > 1e-6

    def _world_to_cell(self, x: float, y: float) -> Optional[Tuple[int, int]]:
        if self.grid is None:
            return None
        cx = int((x - self.grid.origin_x) / self.grid.resolution)
        cy = int((y - self.grid.origin_y) / self.grid.resolution)
        if not self._in_bounds(cx, cy):
            return None
        return cx, cy

    def _cell_to_world(self, cx: int, cy: int) -> Tuple[float, float]:
        assert self.grid is not None
        return self.grid.origin_x + (cx + 0.5) * self.grid.resolution, self.grid.origin_y + (cy + 0.5) * self.grid.resolution

    def _idx(self, cx: int, cy: int) -> int:
        assert self.grid is not None
        return cy * self.grid.width + cx

    def _in_bounds(self, cx: int, cy: int) -> bool:
        return self.grid is not None and 0 <= cx < self.grid.width and 0 <= cy < self.grid.height

    def _neighbors4(self, cx: int, cy: int):
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nx, ny = cx + dx, cy + dy
            if self._in_bounds(nx, ny):
                yield nx, ny

    def _neighbors8(self, cx: int, cy: int):
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                nx, ny = cx + dx, cy + dy
                if self._in_bounds(nx, ny):
                    yield nx, ny

    def _is_free_cell(self, cx: int, cy: int) -> bool:
        if self.map_msg is None or not self._in_bounds(cx, cy):
            return False
        return self._is_free_value(self.map_msg.data[self._idx(cx, cy)])

    def _is_unknown_cell(self, cx: int, cy: int) -> bool:
        if self.map_msg is None or not self._in_bounds(cx, cy):
            return False
        return self._is_unknown_value(self.map_msg.data[self._idx(cx, cy)])

    def _is_frontier_cell(self, cx: int, cy: int) -> bool:
        if not self._is_free_cell(cx, cy):
            return False
        return any(self._is_unknown_cell(nx, ny) for nx, ny in self._neighbors8(cx, cy))

    def _is_free_value(self, val: int) -> bool:
        return 0 <= val <= self.free_threshold

    def _is_unknown_value(self, val: int) -> bool:
        return val < 0

    def _is_occupied_value(self, val: int) -> bool:
        return val >= self.occupied_threshold

    @staticmethod
    def _wrap(a: float) -> float:
        while a > math.pi:
            a -= 2.0 * math.pi
        while a < -math.pi:
            a += 2.0 * math.pi
        return a

    @staticmethod
    def _yaw_from_quat(x: float, y: float, z: float, w: float) -> float:
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        return math.atan2(siny_cosp, cosy_cosp)

    @staticmethod
    def _quat_from_yaw(yaw: float) -> Tuple[float, float, float, float]:
        half = yaw * 0.5
        return 0.0, 0.0, math.sin(half), math.cos(half)


def main(args=None):
    rclpy.init(args=args)
    node = VisibilityExplorerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node._stop_cmd()
        except Exception:
            pass
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
