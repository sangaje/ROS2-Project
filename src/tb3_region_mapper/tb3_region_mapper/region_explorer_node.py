#!/usr/bin/env python3
"""Region-aware active SLAM coverage explorer for TurtleBot3.

This node is intentionally conservative:
- It does NOT replace Nav2/local control.
- It computes region-local next-best-view goals from /map and /slam_region_graph/region_map.
- It publishes a goal PoseStamped to /goal_pose only when enable_goal_publishing is true.
- It always publishes coverage/state/debug markers so the algorithm can be tested without motion.

Region labels:
The v9+ region graph publishes OccupancyGrid categorical values rather than raw ids:
    encoded = 1 + ((region_id * 37) % 98)
This node uses the encoded value as the primary region label and exposes decoded ids when possible.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy

from std_msgs.msg import String, ColorRGBA
from nav_msgs.msg import OccupancyGrid
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import PoseStamped, Point, Quaternion
from visualization_msgs.msg import Marker, MarkerArray

import tf2_ros
from tf2_ros import TransformException

Cell = Tuple[int, int]


def yaw_from_quaternion(q) -> float:
    # Standard planar yaw extraction from geometry_msgs/Quaternion.
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


def quaternion_from_yaw(yaw: float) -> Quaternion:
    return Quaternion(x=0.0, y=0.0, z=math.sin(yaw * 0.5), w=math.cos(yaw * 0.5))


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


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
        if other is None:
            return False
        return (
            self.width == other.width
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
    area_m2: float
    centroid: Tuple[float, float]
    covered: int
    total: int
    coverage_ratio: float
    frontier_count: int
    status: str


@dataclass
class Candidate:
    x: float
    y: float
    yaw: float
    score: float
    visible_unknown: int
    visible_uncovered: int
    visible_frontier: int
    path_cost: float
    turn_cost: float
    clearance_m: float


class RegionExplorerNode(Node):
    def __init__(self):
        super().__init__('region_explorer')

        # Topics / frames
        self.declare_parameter('map_topic', '/map')
        self.declare_parameter('region_map_topic', '/slam_region_graph/region_map')
        self.declare_parameter('scan_topic', '/scan')
        self.declare_parameter('coverage_map_topic', '/region_explorer/coverage_map')
        self.declare_parameter('state_topic', '/region_explorer/state')
        self.declare_parameter('marker_topic', '/region_explorer/markers')
        self.declare_parameter('goal_pose_topic', '/region_explorer/goal_pose')
        self.declare_parameter('nav2_goal_pose_topic', '/goal_pose')
        self.declare_parameter('global_frame', 'map')
        self.declare_parameter('robot_frame', 'base_footprint')

        # Scheduling
        self.declare_parameter('timer_period', 0.25)
        self.declare_parameter('planning_period', 1.0)
        self.declare_parameter('publish_debug_every_n', 1)

        # Occupancy interpretation
        self.declare_parameter('free_threshold', 65)
        self.declare_parameter('occupied_threshold', 70)
        self.declare_parameter('min_region_cells', 20)

        # Coverage update
        self.declare_parameter('coverage_use_lidar_scan', True)
        self.declare_parameter('coverage_max_range_m', 3.2)
        self.declare_parameter('coverage_ray_step_m', 0.05)
        self.declare_parameter('coverage_mark_unknown', True)
        self.declare_parameter('coverage_stop_on_map_obstacle', True)
        self.declare_parameter('coverage_downsample_angle_step_deg', 2.0)

        # Candidate / viewpoint sampling
        self.declare_parameter('candidate_grid_step_m', 0.25)
        self.declare_parameter('candidate_min_clearance_m', 0.22)
        self.declare_parameter('candidate_max_count', 400)
        self.declare_parameter('candidate_eval_angle_step_deg', 5.0)
        self.declare_parameter('view_fov_deg', 360.0)
        self.declare_parameter('view_max_range_m', 3.2)
        self.declare_parameter('yaw_samples', 12)
        self.declare_parameter('use_region_restriction_for_view_gain', True)

        # Utility weights
        self.declare_parameter('w_unknown', 3.0)
        self.declare_parameter('w_unseen', 2.0)
        self.declare_parameter('w_frontier', 2.5)
        self.declare_parameter('w_path', 0.8)
        self.declare_parameter('w_turn', 0.2)
        self.declare_parameter('w_clearance', 0.4)
        self.declare_parameter('w_outside_region', 5.0)

        # Region completion
        self.declare_parameter('region_coverage_threshold', 0.85)
        self.declare_parameter('region_frontier_threshold', 8)
        self.declare_parameter('map_growth_stall_time_sec', 6.0)
        self.declare_parameter('map_growth_epsilon_cells', 5)
        self.declare_parameter('completed_region_revisit_delay_sec', 15.0)

        # Goal publishing
        self.declare_parameter('enable_goal_publishing', False)
        self.declare_parameter('publish_to_nav2_goal_pose', False)
        self.declare_parameter('goal_min_interval_sec', 3.0)
        self.declare_parameter('goal_reached_radius_m', 0.25)
        self.declare_parameter('goal_timeout_sec', 22.0)
        self.declare_parameter('goal_z', 0.0)

        # State behavior
        self.declare_parameter('lock_active_region_until_complete', True)
        self.declare_parameter('switch_to_robot_region_if_outside', True)
        self.declare_parameter('select_next_region_policy', 'nearest_uncovered')  # nearest_uncovered | largest_uncovered

        # Read params
        self.map_topic = str(self.get_parameter('map_topic').value)
        self.region_map_topic = str(self.get_parameter('region_map_topic').value)
        self.scan_topic = str(self.get_parameter('scan_topic').value)
        self.coverage_map_topic = str(self.get_parameter('coverage_map_topic').value)
        self.state_topic = str(self.get_parameter('state_topic').value)
        self.marker_topic = str(self.get_parameter('marker_topic').value)
        self.goal_pose_topic = str(self.get_parameter('goal_pose_topic').value)
        self.nav2_goal_pose_topic = str(self.get_parameter('nav2_goal_pose_topic').value)
        self.global_frame = str(self.get_parameter('global_frame').value)
        self.robot_frame = str(self.get_parameter('robot_frame').value)

        self.timer_period = float(self.get_parameter('timer_period').value)
        self.planning_period = float(self.get_parameter('planning_period').value)
        self.publish_debug_every_n = max(1, int(self.get_parameter('publish_debug_every_n').value))

        self.free_threshold = int(self.get_parameter('free_threshold').value)
        self.occupied_threshold = int(self.get_parameter('occupied_threshold').value)
        self.min_region_cells = int(self.get_parameter('min_region_cells').value)

        self.coverage_use_lidar_scan = bool(self.get_parameter('coverage_use_lidar_scan').value)
        self.coverage_max_range_m = float(self.get_parameter('coverage_max_range_m').value)
        self.coverage_ray_step_m = float(self.get_parameter('coverage_ray_step_m').value)
        self.coverage_mark_unknown = bool(self.get_parameter('coverage_mark_unknown').value)
        self.coverage_stop_on_map_obstacle = bool(self.get_parameter('coverage_stop_on_map_obstacle').value)
        self.coverage_downsample_angle_step_deg = float(self.get_parameter('coverage_downsample_angle_step_deg').value)

        self.candidate_grid_step_m = float(self.get_parameter('candidate_grid_step_m').value)
        self.candidate_min_clearance_m = float(self.get_parameter('candidate_min_clearance_m').value)
        self.candidate_max_count = int(self.get_parameter('candidate_max_count').value)
        self.candidate_eval_angle_step_deg = float(self.get_parameter('candidate_eval_angle_step_deg').value)
        self.view_fov_deg = float(self.get_parameter('view_fov_deg').value)
        self.view_max_range_m = float(self.get_parameter('view_max_range_m').value)
        self.yaw_samples = int(self.get_parameter('yaw_samples').value)
        self.use_region_restriction_for_view_gain = bool(self.get_parameter('use_region_restriction_for_view_gain').value)

        self.w_unknown = float(self.get_parameter('w_unknown').value)
        self.w_unseen = float(self.get_parameter('w_unseen').value)
        self.w_frontier = float(self.get_parameter('w_frontier').value)
        self.w_path = float(self.get_parameter('w_path').value)
        self.w_turn = float(self.get_parameter('w_turn').value)
        self.w_clearance = float(self.get_parameter('w_clearance').value)
        self.w_outside_region = float(self.get_parameter('w_outside_region').value)

        self.region_coverage_threshold = float(self.get_parameter('region_coverage_threshold').value)
        self.region_frontier_threshold = int(self.get_parameter('region_frontier_threshold').value)
        self.map_growth_stall_time_sec = float(self.get_parameter('map_growth_stall_time_sec').value)
        self.map_growth_epsilon_cells = int(self.get_parameter('map_growth_epsilon_cells').value)
        self.completed_region_revisit_delay_sec = float(self.get_parameter('completed_region_revisit_delay_sec').value)

        self.enable_goal_publishing = bool(self.get_parameter('enable_goal_publishing').value)
        self.publish_to_nav2_goal_pose = bool(self.get_parameter('publish_to_nav2_goal_pose').value)
        self.goal_min_interval_sec = float(self.get_parameter('goal_min_interval_sec').value)
        self.goal_reached_radius_m = float(self.get_parameter('goal_reached_radius_m').value)
        self.goal_timeout_sec = float(self.get_parameter('goal_timeout_sec').value)
        self.goal_z = float(self.get_parameter('goal_z').value)

        self.lock_active_region_until_complete = bool(self.get_parameter('lock_active_region_until_complete').value)
        self.switch_to_robot_region_if_outside = bool(self.get_parameter('switch_to_robot_region_if_outside').value)
        self.select_next_region_policy = str(self.get_parameter('select_next_region_policy').value)

        # QoS
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
        pub_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        self.map_sub = self.create_subscription(OccupancyGrid, self.map_topic, self._on_map, map_qos)
        self.region_sub = self.create_subscription(OccupancyGrid, self.region_map_topic, self._on_region_map, map_qos)
        self.scan_sub = self.create_subscription(LaserScan, self.scan_topic, self._on_scan, sensor_qos)

        self.coverage_pub = self.create_publisher(OccupancyGrid, self.coverage_map_topic, pub_qos)
        self.state_pub = self.create_publisher(String, self.state_topic, 10)
        self.marker_pub = self.create_publisher(MarkerArray, self.marker_topic, 10)
        self.goal_pub = self.create_publisher(PoseStamped, self.goal_pose_topic, 10)
        self.nav_goal_pub = self.create_publisher(PoseStamped, self.nav2_goal_pose_topic, 10)

        self.tf_buffer = tf2_ros.Buffer(cache_time=Duration(seconds=10.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # State
        self.map_msg: Optional[OccupancyGrid] = None
        self.region_msg: Optional[OccupancyGrid] = None
        self.scan_msg: Optional[LaserScan] = None
        self.geom: Optional[GridGeom] = None
        self.coverage: List[bool] = []
        self.last_known_count = 0
        self.last_growth_time = time.time()
        self.active_region_label: Optional[int] = None
        self.completed_regions: Dict[int, float] = {}
        self.last_plan_time = 0.0
        self.last_goal_time = 0.0
        self.current_goal: Optional[PoseStamped] = None
        self.current_goal_sent_time = 0.0
        self.last_best: Optional[Candidate] = None
        self.last_candidates: List[Candidate] = []
        self.tick = 0

        self.timer = self.create_timer(self.timer_period, self._on_timer)
        self.get_logger().info(
            'REGION_EXPLORER_READY | '
            f'domain-aware externally | map={self.map_topic} region={self.region_map_topic} '
            f'enable_goal={self.enable_goal_publishing} nav2_topic={self.publish_to_nav2_goal_pose}'
        )

    # ---------------------------------------------------------------------
    # Callbacks
    # ---------------------------------------------------------------------
    def _on_map(self, msg: OccupancyGrid):
        new_geom = GridGeom.from_msg(msg)
        if not new_geom.same_geometry(self.geom):
            old_geom = self.geom
            old_cov = self.coverage
            self.geom = new_geom
            self.coverage = [False] * (new_geom.width * new_geom.height)
            if old_geom is not None and old_cov:
                self._copy_coverage_overlap(old_geom, old_cov, new_geom, self.coverage)
            self.get_logger().info(
                f'COVERAGE_GEOMETRY_RESET | size={new_geom.width}x{new_geom.height} '
                f'res={new_geom.resolution:.3f} origin=({new_geom.origin_x:.2f},{new_geom.origin_y:.2f})'
            )
        self.map_msg = msg

        known = sum(1 for v in msg.data if v >= 0)
        if known > self.last_known_count + self.map_growth_epsilon_cells:
            self.last_growth_time = time.time()
        self.last_known_count = known

    def _on_region_map(self, msg: OccupancyGrid):
        self.region_msg = msg

    def _on_scan(self, msg: LaserScan):
        self.scan_msg = msg

    # ---------------------------------------------------------------------
    # Timer
    # ---------------------------------------------------------------------
    def _on_timer(self):
        self.tick += 1
        robot = self._lookup_robot_pose()
        ready = self.map_msg is not None and self.region_msg is not None and self.geom is not None and robot is not None
        if not ready:
            self._publish_state({'state': 'WAIT_INPUT', 'has_map': self.map_msg is not None, 'has_region_map': self.region_msg is not None, 'has_tf': robot is not None})
            return

        assert robot is not None
        assert self.geom is not None
        self._update_coverage(robot)
        self._publish_coverage_map()

        now = time.time()
        if now - self.last_plan_time >= self.planning_period:
            self.last_plan_time = now
            self._planning_step(robot)
        elif self.tick % self.publish_debug_every_n == 0:
            stats = self._compute_region_stats()
            self._publish_markers(robot, stats, self.last_candidates, self.last_best)

    # ---------------------------------------------------------------------
    # Geometry helpers
    # ---------------------------------------------------------------------
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

    def _copy_coverage_overlap(self, old_geom: GridGeom, old_cov: List[bool], new_geom: GridGeom, new_cov: List[bool]):
        # Preserve coverage when Cartographer expands map bounds.
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
            tf = self.tf_buffer.lookup_transform(
                self.global_frame,
                self.robot_frame,
                rclpy.time.Time(),
                timeout=Duration(seconds=0.05),
            )
        except TransformException:
            return None
        t = tf.transform.translation
        q = tf.transform.rotation
        return RobotPose(float(t.x), float(t.y), yaw_from_quaternion(q))

    # ---------------------------------------------------------------------
    # Map interpretation
    # ---------------------------------------------------------------------
    def _map_value(self, c: Cell) -> int:
        assert self.map_msg is not None
        return int(self.map_msg.data[self._idx(c)])

    def _is_known(self, c: Cell) -> bool:
        return self._map_value(c) >= 0

    def _is_obstacle(self, c: Cell) -> bool:
        return self._map_value(c) >= self.occupied_threshold

    def _is_free_candidate(self, c: Cell) -> bool:
        v = self._map_value(c)
        return 0 <= v < self.occupied_threshold

    def _region_label_value(self, c: Cell) -> int:
        if self.region_msg is None:
            return -1
        i = self._idx(c)
        if i < 0 or i >= len(self.region_msg.data):
            return -1
        return int(self.region_msg.data[i])

    def _decode_region_id(self, label: int) -> int:
        # region_map v9+ uses v = 1 + ((id * 37) % 98). Inverse of 37 mod 98 is 53.
        if label <= 0:
            return -1
        rid = ((label - 1) * 53) % 98
        return 98 if rid == 0 else rid

    def _neighbors4(self, c: Cell) -> Iterable[Cell]:
        x, y = c
        for nx, ny in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
            if self._in_bounds(nx, ny):
                yield (nx, ny)

    # ---------------------------------------------------------------------
    # Coverage
    # ---------------------------------------------------------------------
    def _update_coverage(self, robot: RobotPose):
        if self.map_msg is None or self.geom is None:
            return
        if self.coverage_use_lidar_scan and self.scan_msg is not None:
            self._update_coverage_from_scan(robot, self.scan_msg)
        else:
            # Fallback fixed 360-degree raycast.
            angles = self._angle_sequence(-math.pi, math.pi, math.radians(self.coverage_downsample_angle_step_deg))
            for a in angles:
                self._mark_ray(robot.x, robot.y, robot.yaw + a, self.coverage_max_range_m, self.coverage)

    def _update_coverage_from_scan(self, robot: RobotPose, scan: LaserScan):
        angle_step_limit = math.radians(max(0.5, self.coverage_downsample_angle_step_deg))
        stride = max(1, int(math.ceil(angle_step_limit / max(abs(scan.angle_increment), 1e-6))))
        n = len(scan.ranges)
        for i in range(0, n, stride):
            r = float(scan.ranges[i])
            if math.isnan(r) or math.isinf(r) or r <= max(0.0, scan.range_min):
                r = self.coverage_max_range_m
            else:
                r = min(r, self.coverage_max_range_m, float(scan.range_max) if scan.range_max > 0 else self.coverage_max_range_m)
            a = robot.yaw + float(scan.angle_min) + i * float(scan.angle_increment)
            self._mark_ray(robot.x, robot.y, a, r, self.coverage)

    def _mark_ray(self, x0: float, y0: float, yaw: float, max_range: float, target: List[bool]) -> List[Cell]:
        assert self.geom is not None
        visited: List[Cell] = []
        step = max(self.coverage_ray_step_m, self.geom.resolution * 0.5)
        n = max(1, int(max_range / step))
        last: Optional[Cell] = None
        ca = math.cos(yaw)
        sa = math.sin(yaw)
        for k in range(n + 1):
            r = k * step
            c = self._world_to_cell(x0 + r * ca, y0 + r * sa)
            if c is None:
                break
            if c == last:
                continue
            last = c
            if not self.coverage_mark_unknown and not self._is_known(c):
                continue
            target[self._idx(c)] = True
            visited.append(c)
            if self.coverage_stop_on_map_obstacle and self._is_obstacle(c):
                break
        return visited

    def _angle_sequence(self, start: float, stop: float, step: float) -> List[float]:
        out = []
        a = start
        while a <= stop + 1e-9:
            out.append(a)
            a += step
        return out

    def _publish_coverage_map(self):
        if self.map_msg is None or self.geom is None or not self.coverage:
            return
        msg = OccupancyGrid()
        msg.header = self.map_msg.header
        msg.header.frame_id = self.global_frame
        msg.info = self.map_msg.info
        # Unknown where SLAM does not know anything, 0 for unobserved known, 100 for covered.
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

    # ---------------------------------------------------------------------
    # Region stats / completion
    # ---------------------------------------------------------------------
    def _compute_region_cells(self) -> Dict[int, List[Cell]]:
        if self.region_msg is None or self.geom is None:
            return {}
        out: Dict[int, List[Cell]] = {}
        for y in range(self.geom.height):
            for x in range(self.geom.width):
                c = (x, y)
                label = self._region_label_value(c)
                if label <= 0:
                    continue
                if not self._is_free_candidate(c):
                    # region_map may include cells that became stale; ignore hard obstacles.
                    continue
                out.setdefault(label, []).append(c)
        return {k: v for k, v in out.items() if len(v) >= self.min_region_cells}

    def _frontier_count_for_region(self, cells: Sequence[Cell]) -> int:
        cell_set = set(cells)
        count = 0
        for c in cells:
            # Frontier: region/free cell adjacent to unknown map cell.
            for nb in self._neighbors4(c):
                if nb not in cell_set and self._map_value(nb) < 0:
                    count += 1
                    break
        return count

    def _compute_region_stats(self) -> Dict[int, RegionStats]:
        if self.geom is None:
            return {}
        regions = self._compute_region_cells()
        stats: Dict[int, RegionStats] = {}
        now = time.time()
        for label, cells in regions.items():
            total = len(cells)
            covered = sum(1 for c in cells if self.coverage[self._idx(c)]) if self.coverage else 0
            ratio = covered / max(1, total)
            cx = cy = 0.0
            for c in cells:
                wx, wy = self._cell_to_world(c)
                cx += wx
                cy += wy
            cx /= max(1, total)
            cy /= max(1, total)
            frontier_count = self._frontier_count_for_region(cells)
            completed_at = self.completed_regions.get(label)
            if completed_at is not None and now - completed_at < self.completed_region_revisit_delay_sec:
                status = 'COVERED'
            elif ratio >= self.region_coverage_threshold and frontier_count <= self.region_frontier_threshold:
                status = 'READY_TO_COMPLETE'
            else:
                status = 'ACTIVE_OR_UNVISITED'
            stats[label] = RegionStats(
                label=label,
                decoded_id=self._decode_region_id(label),
                cells=list(cells),
                area_m2=total * self.geom.resolution * self.geom.resolution,
                centroid=(cx, cy),
                covered=covered,
                total=total,
                coverage_ratio=ratio,
                frontier_count=frontier_count,
                status=status,
            )
        return stats

    def _region_at_pose(self, robot: RobotPose) -> Optional[int]:
        c = self._world_to_cell(robot.x, robot.y)
        if c is None:
            return None
        label = self._region_label_value(c)
        if label > 0:
            return label
        # Fallback: nearest label in small radius.
        best = None
        best_d2 = 999999
        r = 4
        cx, cy = c
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
        return best

    def _region_complete(self, st: RegionStats) -> bool:
        stall = time.time() - self.last_growth_time >= self.map_growth_stall_time_sec
        # Strong completion if coverage/frontier thresholds are met.
        if st.coverage_ratio >= self.region_coverage_threshold and st.frontier_count <= self.region_frontier_threshold:
            return True
        # Soft completion if coverage is high and map growth has stalled.
        if st.coverage_ratio >= max(0.70, self.region_coverage_threshold - 0.08) and stall:
            return True
        return False

    # ---------------------------------------------------------------------
    # Planning
    # ---------------------------------------------------------------------
    def _planning_step(self, robot: RobotPose):
        stats = self._compute_region_stats()
        robot_region = self._region_at_pose(robot)

        if not stats:
            self._publish_state({'state': 'WAIT_REGION_STATS', 'region_count': 0})
            self._publish_markers(robot, stats, [], None)
            return

        if self.active_region_label is None:
            self.active_region_label = robot_region if robot_region in stats else self._select_initial_region(robot, stats)

        # If robot enters another region before active region is complete, optionally switch.
        if self.switch_to_robot_region_if_outside and robot_region in stats and self.active_region_label not in stats:
            self.active_region_label = robot_region
        elif self.switch_to_robot_region_if_outside and robot_region in stats and not self.lock_active_region_until_complete:
            self.active_region_label = robot_region

        active = stats.get(self.active_region_label) if self.active_region_label is not None else None
        if active is None:
            self.active_region_label = self._select_initial_region(robot, stats)
            active = stats.get(self.active_region_label) if self.active_region_label is not None else None
        if active is None:
            self._publish_state({'state': 'NO_ACTIVE_REGION'})
            return

        completed_now = False
        if self._region_complete(active):
            if active.label not in self.completed_regions:
                self.completed_regions[active.label] = time.time()
                completed_now = True
            nxt = self._select_next_region(robot, active, stats)
            if nxt is not None:
                self.active_region_label = nxt.label
                active = nxt
            else:
                # All regions look covered; keep current active region and publish DONE.
                self._publish_state(self._state_payload('DONE_OR_WAITING_FOR_NEW_REGION', robot, active, stats, None, completed_now))
                self._publish_markers(robot, stats, [], None)
                return

        candidates = self._sample_candidates(robot, active)
        best = self._select_best_candidate(robot, active, candidates)
        self.last_candidates = candidates[:80]
        self.last_best = best

        if best is not None:
            self._maybe_publish_goal(best)

        self._publish_state(self._state_payload('ACTIVE_REGION_COVERAGE', robot, active, stats, best, completed_now))
        self._publish_markers(robot, stats, self.last_candidates, best)

    def _select_initial_region(self, robot: RobotPose, stats: Dict[int, RegionStats]) -> Optional[int]:
        best_label = None
        best_d = 1e18
        for label, st in stats.items():
            d = math.hypot(st.centroid[0] - robot.x, st.centroid[1] - robot.y)
            if d < best_d:
                best_d = d
                best_label = label
        return best_label

    def _select_next_region(self, robot: RobotPose, current: RegionStats, stats: Dict[int, RegionStats]) -> Optional[RegionStats]:
        candidates = [s for s in stats.values() if s.label != current.label and s.label not in self.completed_regions]
        if not candidates:
            # Reopen partially covered regions if all were marked completed but new map expanded.
            candidates = [s for s in stats.values() if s.label != current.label and s.coverage_ratio < self.region_coverage_threshold]
        if not candidates:
            return None
        if self.select_next_region_policy == 'largest_uncovered':
            candidates.sort(key=lambda s: (s.total - s.covered, -math.hypot(s.centroid[0] - robot.x, s.centroid[1] - robot.y)), reverse=True)
            return candidates[0]
        # nearest_uncovered default.
        candidates.sort(key=lambda s: math.hypot(s.centroid[0] - robot.x, s.centroid[1] - robot.y) - 0.4 * (1.0 - s.coverage_ratio))
        return candidates[0]

    def _sample_candidates(self, robot: RobotPose, region: RegionStats) -> List[Candidate]:
        if self.geom is None:
            return []
        step_cells = max(1, int(round(self.candidate_grid_step_m / self.geom.resolution)))
        candidates_cells: List[Cell] = []
        for c in region.cells:
            if c[0] % step_cells != 0 or c[1] % step_cells != 0:
                continue
            if not self._is_free_candidate(c):
                continue
            clearance = self._approx_clearance(c, max_radius_m=max(0.6, self.candidate_min_clearance_m * 2.5))
            if clearance < self.candidate_min_clearance_m:
                continue
            candidates_cells.append(c)

        # Prefer uncovered/frontier-near cells first when too many.
        if len(candidates_cells) > self.candidate_max_count:
            candidates_cells.sort(key=lambda c: self._candidate_prefilter_score(c, region), reverse=True)
            candidates_cells = candidates_cells[:self.candidate_max_count]

        out: List[Candidate] = []
        full_fov = self.view_fov_deg >= 359.0
        yaw_list: List[float]
        if full_fov:
            yaw_list = [robot.yaw]
        else:
            yaw_list = [(-math.pi + 2.0 * math.pi * i / max(1, self.yaw_samples)) for i in range(max(1, self.yaw_samples))]

        for c in candidates_cells:
            wx, wy = self._cell_to_world(c)
            for yaw in yaw_list:
                cand = self._evaluate_candidate(robot, region, wx, wy, yaw)
                if cand is not None:
                    out.append(cand)
        out.sort(key=lambda c: c.score, reverse=True)
        return out[:self.candidate_max_count]

    def _candidate_prefilter_score(self, c: Cell, region: RegionStats) -> float:
        score = 0.0
        if not self.coverage[self._idx(c)]:
            score += 2.0
        for nb in self._neighbors4(c):
            if self._map_value(nb) < 0:
                score += 1.0
        wx, wy = self._cell_to_world(c)
        score -= 0.05 * math.hypot(wx - region.centroid[0], wy - region.centroid[1])
        return score

    def _approx_clearance(self, c: Cell, max_radius_m: float) -> float:
        assert self.geom is not None
        max_r = max(1, int(math.ceil(max_radius_m / self.geom.resolution)))
        best_cells = max_r
        cx, cy = c
        for dy in range(-max_r, max_r + 1):
            for dx in range(-max_r, max_r + 1):
                cc = (cx + dx, cy + dy)
                if not self._in_bounds(cc[0], cc[1]):
                    continue
                if self._is_obstacle(cc):
                    d2 = dx * dx + dy * dy
                    if d2 < best_cells * best_cells:
                        best_cells = max(0, int(math.sqrt(d2)))
        return best_cells * self.geom.resolution

    def _evaluate_candidate(self, robot: RobotPose, region: RegionStats, wx: float, wy: float, yaw: float) -> Optional[Candidate]:
        c = self._world_to_cell(wx, wy)
        if c is None:
            return None
        label = self._region_label_value(c)
        outside_region_penalty = 0.0 if label == region.label else self.w_outside_region

        clearance = self._approx_clearance(c, max_radius_m=0.8)
        if clearance < self.candidate_min_clearance_m:
            return None

        visible = self._visible_cells_from_pose(wx, wy, yaw, self.view_fov_deg, self.view_max_range_m)
        if not visible:
            return None

        visible_unknown = 0
        visible_uncovered = 0
        visible_frontier = 0
        region_cell_set = None  # avoid expensive set if not needed
        if self.use_region_restriction_for_view_gain:
            region_cell_set = set(region.cells)

        for vc in visible:
            if region_cell_set is not None and vc not in region_cell_set:
                continue
            mv = self._map_value(vc)
            if mv < 0:
                visible_unknown += 1
            else:
                if not self.coverage[self._idx(vc)]:
                    visible_uncovered += 1
                for nb in self._neighbors4(vc):
                    if self._map_value(nb) < 0:
                        visible_frontier += 1
                        break

        path_cost = math.hypot(wx - robot.x, wy - robot.y)
        turn_cost = abs(math.atan2(math.sin(yaw - robot.yaw), math.cos(yaw - robot.yaw)))
        score = (
            self.w_unknown * visible_unknown
            + self.w_unseen * visible_uncovered
            + self.w_frontier * visible_frontier
            + self.w_clearance * clearance
            - self.w_path * path_cost
            - self.w_turn * turn_cost
            - outside_region_penalty
        )
        return Candidate(wx, wy, yaw, score, visible_unknown, visible_uncovered, visible_frontier, path_cost, turn_cost, clearance)

    def _visible_cells_from_pose(self, x: float, y: float, yaw: float, fov_deg: float, max_range: float) -> Set[Cell]:
        visible: Set[Cell] = set()
        if fov_deg >= 359.0:
            start = -math.pi
            stop = math.pi
        else:
            half = math.radians(fov_deg) * 0.5
            start = -half
            stop = half
        step = math.radians(max(1.0, self.candidate_eval_angle_step_deg))
        for a in self._angle_sequence(start, stop, step):
            cells = self._trace_ray_cells(x, y, yaw + a, max_range)
            visible.update(cells)
        return visible

    def _trace_ray_cells(self, x0: float, y0: float, yaw: float, max_range: float) -> List[Cell]:
        assert self.geom is not None
        out: List[Cell] = []
        step = max(self.coverage_ray_step_m, self.geom.resolution * 0.5)
        n = max(1, int(max_range / step))
        ca = math.cos(yaw)
        sa = math.sin(yaw)
        last: Optional[Cell] = None
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

    def _select_best_candidate(self, robot: RobotPose, region: RegionStats, candidates: List[Candidate]) -> Optional[Candidate]:
        if not candidates:
            # Fallback to region centroid if no NBV sample survives.
            yaw = math.atan2(region.centroid[1] - robot.y, region.centroid[0] - robot.x)
            return Candidate(region.centroid[0], region.centroid[1], yaw, -999.0, 0, 0, 0, math.hypot(region.centroid[0] - robot.x, region.centroid[1] - robot.y), 0.0, 0.0)
        return candidates[0]

    def _maybe_publish_goal(self, best: Candidate):
        now = time.time()
        if not self.enable_goal_publishing:
            # Still publish debug goal on /region_explorer/goal_pose for visualization/testing.
            self.goal_pub.publish(self._make_goal_msg(best))
            return
        if self.current_goal is not None:
            gx = self.current_goal.pose.position.x
            gy = self.current_goal.pose.position.y
            if math.hypot(best.x - gx, best.y - gy) < 0.20 and now - self.last_goal_time < self.goal_timeout_sec:
                return
        if now - self.last_goal_time < self.goal_min_interval_sec:
            return
        msg = self._make_goal_msg(best)
        self.goal_pub.publish(msg)
        if self.publish_to_nav2_goal_pose:
            self.nav_goal_pub.publish(msg)
        self.current_goal = msg
        self.last_goal_time = now
        self.current_goal_sent_time = now
        self.get_logger().info(
            f'GOAL_PUBLISHED | x={best.x:.2f} y={best.y:.2f} yaw={math.degrees(best.yaw):.1f} '
            f'score={best.score:.1f} unk={best.visible_unknown} unseen={best.visible_uncovered} frontier={best.visible_frontier}'
        )

    def _make_goal_msg(self, best: Candidate) -> PoseStamped:
        msg = PoseStamped()
        msg.header.frame_id = self.global_frame
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.pose.position.x = float(best.x)
        msg.pose.position.y = float(best.y)
        msg.pose.position.z = float(self.goal_z)
        msg.pose.orientation = quaternion_from_yaw(best.yaw)
        return msg

    # ---------------------------------------------------------------------
    # State + markers
    # ---------------------------------------------------------------------
    def _state_payload(self, state: str, robot: RobotPose, active: RegionStats, stats: Dict[int, RegionStats], best: Optional[Candidate], completed_now: bool) -> Dict:
        covered_regions = len(self.completed_regions)
        payload = {
            'state': state,
            'active_region_label': active.label,
            'active_region_id': active.decoded_id,
            'robot_region_label': self._region_at_pose(robot),
            'region_count': len(stats),
            'completed_region_count': covered_regions,
            'completed_now': completed_now,
            'coverage_ratio': round(active.coverage_ratio, 4),
            'covered_cells': active.covered,
            'total_cells': active.total,
            'frontier_count': active.frontier_count,
            'area_m2': round(active.area_m2, 3),
            'map_growth_stalled': time.time() - self.last_growth_time >= self.map_growth_stall_time_sec,
            'enable_goal_publishing': self.enable_goal_publishing,
            'publish_to_nav2_goal_pose': self.publish_to_nav2_goal_pose,
        }
        if best is not None:
            payload['best_goal'] = {
                'x': round(best.x, 3),
                'y': round(best.y, 3),
                'yaw_deg': round(math.degrees(best.yaw), 1),
                'score': round(best.score, 2),
                'visible_unknown': best.visible_unknown,
                'visible_uncovered': best.visible_uncovered,
                'visible_frontier': best.visible_frontier,
                'path_cost': round(best.path_cost, 3),
                'clearance_m': round(best.clearance_m, 3),
            }
        return payload

    def _publish_state(self, payload: Dict):
        msg = String()
        msg.data = json.dumps(payload, ensure_ascii=False)
        self.state_pub.publish(msg)

    def _publish_markers(self, robot: RobotPose, stats: Dict[int, RegionStats], candidates: List[Candidate], best: Optional[Candidate]):
        ma = MarkerArray()
        now = self.get_clock().now().to_msg()
        clear = Marker()
        clear.action = Marker.DELETEALL
        ma.markers.append(clear)
        mid = 1

        # Active region centroid + label.
        for label, st in stats.items():
            if label != self.active_region_label and st.status != 'READY_TO_COMPLETE':
                continue
            text = Marker()
            text.header.frame_id = self.global_frame
            text.header.stamp = now
            text.ns = 'region_explorer_region_text'
            text.id = mid
            mid += 1
            text.type = Marker.TEXT_VIEW_FACING
            text.action = Marker.ADD
            text.pose.position.x = st.centroid[0]
            text.pose.position.y = st.centroid[1]
            text.pose.position.z = 0.55
            text.pose.orientation.w = 1.0
            text.scale.z = 0.18
            if label == self.active_region_label:
                text.color = ColorRGBA(r=1.0, g=1.0, b=1.0, a=1.0)
                prefix = 'ACTIVE'
            else:
                text.color = ColorRGBA(r=0.3, g=1.0, b=0.3, a=0.9)
                prefix = 'READY'
            text.text = f'{prefix} R{st.decoded_id}\ncov={st.coverage_ratio:.2f} fr={st.frontier_count}'
            ma.markers.append(text)

        # Candidate points.
        if candidates:
            pts = Marker()
            pts.header.frame_id = self.global_frame
            pts.header.stamp = now
            pts.ns = 'region_explorer_candidates'
            pts.id = mid
            mid += 1
            pts.type = Marker.POINTS
            pts.action = Marker.ADD
            pts.pose.orientation.w = 1.0
            pts.scale.x = 0.045
            pts.scale.y = 0.045
            pts.color = ColorRGBA(r=0.2, g=0.7, b=1.0, a=0.55)
            for c in candidates[:120]:
                pts.points.append(Point(x=c.x, y=c.y, z=0.12))
            ma.markers.append(pts)

        # Best viewpoint.
        if best is not None:
            sph = Marker()
            sph.header.frame_id = self.global_frame
            sph.header.stamp = now
            sph.ns = 'region_explorer_best_goal'
            sph.id = mid
            mid += 1
            sph.type = Marker.SPHERE
            sph.action = Marker.ADD
            sph.pose.position.x = best.x
            sph.pose.position.y = best.y
            sph.pose.position.z = 0.20
            sph.pose.orientation.w = 1.0
            sph.scale.x = 0.18
            sph.scale.y = 0.18
            sph.scale.z = 0.18
            sph.color = ColorRGBA(r=1.0, g=0.85, b=0.1, a=0.95)
            ma.markers.append(sph)

            arr = Marker()
            arr.header.frame_id = self.global_frame
            arr.header.stamp = now
            arr.ns = 'region_explorer_best_yaw'
            arr.id = mid
            mid += 1
            arr.type = Marker.ARROW
            arr.action = Marker.ADD
            arr.pose.position.x = best.x
            arr.pose.position.y = best.y
            arr.pose.position.z = 0.26
            arr.pose.orientation = quaternion_from_yaw(best.yaw)
            arr.scale.x = 0.35
            arr.scale.y = 0.045
            arr.scale.z = 0.045
            arr.color = ColorRGBA(r=1.0, g=0.65, b=0.0, a=0.95)
            ma.markers.append(arr)

        self.marker_pub.publish(ma)


def main(args=None):
    rclpy.init(args=args)
    node = RegionExplorerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
