#!/usr/bin/env python3

import math
import heapq
import zlib
from collections import deque
from typing import List, Optional, Tuple

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from rclpy.exceptions import ParameterAlreadyDeclaredException

from geometry_msgs.msg import PoseStamped, TwistStamped
from nav_msgs.msg import OccupancyGrid, Path
from sensor_msgs.msg import LaserScan


def safe_declare(node: Node, name: str, default):
    try:
        node.declare_parameter(name, default)
    except ParameterAlreadyDeclaredException:
        pass
    return node.get_parameter(name).value


def yaw_from_quat(q) -> float:
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


def norm_angle(a: float) -> float:
    while a > math.pi:
        a -= 2.0 * math.pi
    while a < -math.pi:
        a += 2.0 * math.pi
    return a


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


class AStarCmdVelFollower(Node):
    """V54 conservative A* + direct TwistStamped controller.

    This node deliberately bypasses Nav2 and publishes geometry_msgs/TwistStamped
    on /cmd_vel. It supports two stable modes:

      manual: use manual_goal_topic as target. Intended for Waffle RViz 2D Nav Goal.
      leader: follow leader_pose_topic. Intended for Burger following Waffle.

    Important V54 change vs V48/V49/V53:
      - Waffle and Burger use the same controller implementation, but leader goal
        computation is parametrized and isolated.
      - Burger default leader_goal_mode=line_between_robots. The target is a point
        on the current Burger--Waffle line at follow_distance behind the Waffle.
        This avoids using Waffle yaw, which was unstable when SLAM / local heading
        changed near walls.
      - If A* fails because the live SLAM map is still sparse, the controller may
        fall back to direct pure-pursuit toward the target when direct_fallback_if_no_path=true.
    """

    def __init__(self):
        super().__init__('astar_cmd_vel_follower')

        safe_declare(self, 'use_sim_time', True)
        self.robot_name = str(safe_declare(self, 'robot_name', 'robot'))
        self.map_topic = str(safe_declare(self, 'map_topic', '/map'))
        self.robot_pose_topic = str(safe_declare(self, 'robot_pose_topic', '/burger_pose'))
        self.leader_pose_topic = str(safe_declare(self, 'leader_pose_topic', '/leader_pose'))
        self.manual_goal_topic = str(safe_declare(self, 'manual_goal_topic', '/burger_goal_pose'))
        self.scan_topic = str(safe_declare(self, 'scan_topic', '/scan_nav'))
        self.cmd_vel_topic = str(safe_declare(self, 'cmd_vel_topic', '/cmd_vel'))
        self.path_topic = str(safe_declare(self, 'path_topic', '/astar_path'))
        self.frame_id = str(safe_declare(self, 'frame_id', 'map'))

        self.target_mode = str(safe_declare(self, 'target_mode', 'leader'))  # leader | manual
        self.leader_goal_mode = str(safe_declare(self, 'leader_goal_mode', 'line_between_robots'))  # line_between_robots | leader_yaw_behind | leader_pose
        self.follow_distance = float(safe_declare(self, 'follow_distance', 0.75))
        self.goal_tolerance = float(safe_declare(self, 'goal_tolerance', 0.22))
        self.replan_period_sec = float(safe_declare(self, 'replan_period_sec', 0.7))
        self.control_rate_hz = float(safe_declare(self, 'control_rate_hz', 10.0))
        self.lookahead_distance = float(safe_declare(self, 'lookahead_distance', 0.40))

        self.occupied_threshold = int(safe_declare(self, 'occupied_threshold', 55))
        self.treat_unknown_as_obstacle = bool(safe_declare(self, 'treat_unknown_as_obstacle', False))
        self.inflation_radius_m = float(safe_declare(self, 'inflation_radius_m', 0.20))
        # Conservative planning knobs. Hard inflation blocks cells near obstacles.
        # Soft inflation does not block, but increases traversal cost so A* prefers
        # the center of corridors. Unknown cells can be allowed for live SLAM, but
        # heavily penalized instead of being treated as normal free space.
        self.soft_inflation_radius_m = float(safe_declare(self, 'soft_inflation_radius_m', 0.35))
        self.clearance_cost_weight = float(safe_declare(self, 'clearance_cost_weight', 6.0))
        self.unknown_cost_weight = float(safe_declare(self, 'unknown_cost_weight', 2.0))
        self.diagonal_motion = bool(safe_declare(self, 'diagonal_motion', False))
        self.max_astar_cells = int(safe_declare(self, 'max_astar_cells', 120000))
        self.max_path_len_cells = int(safe_declare(self, 'max_path_len_cells', 4000))
        self.nearest_free_radius_cells = int(safe_declare(self, 'nearest_free_radius_cells', 24))
        self.direct_fallback_if_no_path = bool(safe_declare(self, 'direct_fallback_if_no_path', True))
        self.replan_on_goal_delta_m = float(safe_declare(self, 'replan_on_goal_delta_m', 0.12))
        self.replan_on_robot_delta_m = float(safe_declare(self, 'replan_on_robot_delta_m', 0.15))

        self.max_linear = float(safe_declare(self, 'max_linear', 0.12))
        self.min_linear = float(safe_declare(self, 'min_linear', 0.025))
        self.max_angular = float(safe_declare(self, 'max_angular', 0.70))
        self.k_linear = float(safe_declare(self, 'k_linear', 0.55))
        self.k_angular = float(safe_declare(self, 'k_angular', 1.65))
        self.slow_yaw_threshold = float(safe_declare(self, 'slow_yaw_threshold', 0.65))
        self.rotate_yaw_threshold = float(safe_declare(self, 'rotate_yaw_threshold', 1.15))
        self.goal_slow_radius = float(safe_declare(self, 'goal_slow_radius', 0.45))

        self.front_stop_distance = float(safe_declare(self, 'front_stop_distance', 0.24))
        self.front_slow_distance = float(safe_declare(self, 'front_slow_distance', 0.42))
        self.front_sector_deg = float(safe_declare(self, 'front_sector_deg', 35.0))
        self.safety_turn_bias = float(safe_declare(self, 'safety_turn_bias', 0.50))

        self.stale_pose_sec = float(safe_declare(self, 'stale_pose_sec', 2.5))
        self.stale_map_sec = float(safe_declare(self, 'stale_map_sec', 8.0))
        self.stale_goal_sec = float(safe_declare(self, 'stale_goal_sec', 0.0))
        self.log_period_sec = float(safe_declare(self, 'log_period_sec', 1.5))
        self.publish_empty_path_on_fail = bool(safe_declare(self, 'publish_empty_path_on_fail', True))

        self.map_msg: Optional[OccupancyGrid] = None
        self.map_stamp = None
        self.occ_cache = None
        self.cost_cache = None
        self.occ_cache_key = None
        self.robot_pose: Optional[PoseStamped] = None
        self.leader_pose: Optional[PoseStamped] = None
        self.manual_goal: Optional[PoseStamped] = None
        self.robot_stamp = None
        self.leader_stamp = None
        self.manual_goal_stamp = None
        self.front_min = float('inf')
        self.path_world: List[Tuple[float, float]] = []
        self.last_replan_time = None
        self.last_replan_robot_xy: Optional[Tuple[float, float]] = None
        self.last_replan_goal_xy: Optional[Tuple[float, float]] = None
        self.last_log_time = None
        self.last_status = ''

        qos = 10
        self.create_subscription(OccupancyGrid, self.map_topic, self.on_map, qos)
        self.create_subscription(PoseStamped, self.robot_pose_topic, self.on_robot_pose, qos)
        self.create_subscription(PoseStamped, self.leader_pose_topic, self.on_leader_pose, qos)
        self.create_subscription(PoseStamped, self.manual_goal_topic, self.on_manual_goal, qos)
        self.create_subscription(LaserScan, self.scan_topic, self.on_scan, qos)
        self.cmd_pub = self.create_publisher(TwistStamped, self.cmd_vel_topic, qos)
        self.path_pub = self.create_publisher(Path, self.path_topic, qos)

        self.timer = self.create_timer(1.0 / max(1.0, self.control_rate_hz), self.control_loop)

        self.get_logger().info(
            'V55_ASTAR_CMD_VEL_CONTROLLER_READY | TwistStamped direct controller | '
            f'robot={self.robot_name} mode={self.target_mode}/{self.leader_goal_mode} '
            f'map={self.map_topic} pose={self.robot_pose_topic} leader={self.leader_pose_topic} '
            f'manual={self.manual_goal_topic} cmd={self.cmd_vel_topic} path={self.path_topic} '
            f'follow={self.follow_distance:.2f} lookahead={self.lookahead_distance:.2f} '
            f'hard_infl={self.inflation_radius_m:.2f} soft_infl={self.soft_inflation_radius_m:.2f} '
            f'unknown_cost={self.unknown_cost_weight:.1f} clearance_w={self.clearance_cost_weight:.1f} '
            f'diagonal={self.diagonal_motion} direct_fallback={self.direct_fallback_if_no_path}'
        )

    def now(self):
        return self.get_clock().now()

    def elapsed_sec(self, stamp) -> Optional[float]:
        if stamp is None:
            return None
        try:
            return (self.now() - stamp).nanoseconds * 1e-9
        except Exception:
            # ROS sim time can jump from 0 or reset when /clock starts.
            # Treat this as stale for age checks and as immediately due for timers.
            return None

    def age_ok(self, stamp, limit_sec: float) -> bool:
        if stamp is None:
            return False
        if limit_sec <= 0.0:
            return True
        dt = self.elapsed_sec(stamp)
        if dt is None:
            return False
        return dt <= limit_sec

    def on_map(self, msg: OccupancyGrid):
        self.map_msg = msg
        self.map_stamp = self.now()
        key = self.occupancy_cache_key(msg)
        if key != self.occ_cache_key:
            self.occ_cache = None
            self.cost_cache = None
            self.occ_cache_key = key
            self.path_world = []

    def on_robot_pose(self, msg: PoseStamped):
        self.robot_pose = msg
        self.robot_stamp = self.now()

    def on_leader_pose(self, msg: PoseStamped):
        self.leader_pose = msg
        self.leader_stamp = self.now()

    def on_manual_goal(self, msg: PoseStamped):
        self.manual_goal = msg
        self.manual_goal_stamp = self.now()
        self.path_world = []
        self.last_replan_goal_xy = None
        if self.target_mode != 'manual':
            self.get_logger().info(f'MANUAL_GOAL_RECEIVED_BUT_TARGET_MODE_IS_{self.target_mode} | topic={self.manual_goal_topic}')
        else:
            self.get_logger().info(f'MANUAL_GOAL_ACCEPTED | robot={self.robot_name} xy=({msg.pose.position.x:.2f},{msg.pose.position.y:.2f})')

    def on_scan(self, msg: LaserScan):
        sector = math.radians(self.front_sector_deg)
        mn = float('inf')
        angle = msg.angle_min
        for r in msg.ranges:
            if abs(norm_angle(angle)) <= sector:
                if math.isfinite(r) and msg.range_min <= r <= msg.range_max:
                    mn = min(mn, float(r))
            angle += msg.angle_increment
        self.front_min = mn

    def stop(self):
        cmd = TwistStamped()
        cmd.header.stamp = self.now().to_msg()
        cmd.header.frame_id = 'base_footprint'
        self.cmd_pub.publish(cmd)

    def publish_path(self, pts: List[Tuple[float, float]]):
        path = Path()
        path.header.stamp = self.now().to_msg()
        path.header.frame_id = self.frame_id
        for x, y in pts:
            ps = PoseStamped()
            ps.header = path.header
            ps.pose.position.x = x
            ps.pose.position.y = y
            ps.pose.orientation.w = 1.0
            path.poses.append(ps)
        self.path_pub.publish(path)

    def target_pose_xy(self) -> Optional[Tuple[float, float]]:
        if self.target_mode == 'manual':
            if self.manual_goal is None or not self.age_ok(self.manual_goal_stamp, self.stale_goal_sec):
                return None
            return (float(self.manual_goal.pose.position.x), float(self.manual_goal.pose.position.y))

        if self.leader_pose is None or not self.age_ok(self.leader_stamp, self.stale_goal_sec):
            return None
        lx = float(self.leader_pose.pose.position.x)
        ly = float(self.leader_pose.pose.position.y)

        if self.leader_goal_mode == 'leader_pose':
            return (lx, ly)

        if self.leader_goal_mode == 'leader_yaw_behind':
            lyaw = yaw_from_quat(self.leader_pose.pose.orientation)
            return (lx - self.follow_distance * math.cos(lyaw), ly - self.follow_distance * math.sin(lyaw))

        # Default: line_between_robots. Desired goal lies on the current line
        # from Waffle to Burger, follow_distance behind Waffle. This is much
        # more stable than using Waffle yaw during SLAM.
        if self.robot_pose is None:
            return (lx, ly)
        rx = float(self.robot_pose.pose.position.x)
        ry = float(self.robot_pose.pose.position.y)
        vx = rx - lx
        vy = ry - ly
        d = math.hypot(vx, vy)
        # Already close enough to the desired formation spacing. Returning the
        # current robot position makes the outer controller stop instead of
        # trying to drive to a point behind itself.
        if d <= self.follow_distance + self.goal_tolerance:
            return (rx, ry)
        if d < 1e-3:
            lyaw = yaw_from_quat(self.leader_pose.pose.orientation)
            return (lx - self.follow_distance * math.cos(lyaw), ly - self.follow_distance * math.sin(lyaw))
        return (lx + self.follow_distance * vx / d, ly + self.follow_distance * vy / d)

    def world_to_grid(self, x: float, y: float) -> Optional[Tuple[int, int]]:
        m = self.map_msg
        if m is None:
            return None
        res = max(1e-9, m.info.resolution)
        gx = int(math.floor((x - m.info.origin.position.x) / res))
        gy = int(math.floor((y - m.info.origin.position.y) / res))
        if gx < 0 or gy < 0 or gx >= m.info.width or gy >= m.info.height:
            return None
        return gx, gy

    def grid_to_world(self, gx: int, gy: int) -> Tuple[float, float]:
        m = self.map_msg
        assert m is not None
        return (
            m.info.origin.position.x + (gx + 0.5) * m.info.resolution,
            m.info.origin.position.y + (gy + 0.5) * m.info.resolution,
        )

    def occupancy_cache_key(self, msg: OccupancyGrid):
        try:
            data_bytes = memoryview(msg.data).cast('B')
        except TypeError:
            data_bytes = bytes((int(v) & 0xff) for v in msg.data)
        data_crc = zlib.crc32(data_bytes)
        return (
            msg.info.width, msg.info.height, msg.info.resolution,
            msg.info.origin.position.x, msg.info.origin.position.y,
            len(msg.data), data_crc, self.occupied_threshold,
            self.treat_unknown_as_obstacle, self.inflation_radius_m,
            self.soft_inflation_radius_m, self.clearance_cost_weight,
            self.unknown_cost_weight, self.diagonal_motion,
        )

    def build_occupied(self):
        m = self.map_msg
        if m is None:
            return None
        key = self.occupancy_cache_key(m)
        if self.occ_cache is not None and self.occ_cache_key == key:
            return self.occ_cache

        w, h = m.info.width, m.info.height
        occ = bytearray(w * h)
        cost = [0.0] * (w * h)
        obstacles = []
        unknowns = []

        for i, v in enumerate(m.data):
            if v < 0:
                unknowns.append(i)
                if self.treat_unknown_as_obstacle:
                    occ[i] = 1
                    obstacles.append(i)
                else:
                    cost[i] += max(0.0, self.unknown_cost_weight)
                continue
            if v >= self.occupied_threshold:
                occ[i] = 1
                obstacles.append(i)

        # Hard safety inflation: cells in this radius are forbidden.
        hard_cells = int(math.ceil(self.inflation_radius_m / max(1e-6, m.info.resolution)))
        if hard_cells > 0 and obstacles:
            base = bytearray(occ)
            offsets = []
            r2 = hard_cells * hard_cells
            for dy in range(-hard_cells, hard_cells + 1):
                for dx in range(-hard_cells, hard_cells + 1):
                    if dx * dx + dy * dy <= r2:
                        offsets.append((dx, dy))
            for idx, val in enumerate(base):
                if not val:
                    continue
                x = idx % w
                y = idx // w
                for dx, dy in offsets:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < w and 0 <= ny < h:
                        occ[ny * w + nx] = 1

        # Soft safety inflation: cells further away are allowed but expensive.
        # This shifts A* toward open space rather than hugging walls.
        soft_cells = int(math.ceil(self.soft_inflation_radius_m / max(1e-6, m.info.resolution)))
        if soft_cells > 0 and obstacles and self.clearance_cost_weight > 0.0:
            offsets = []
            r = soft_cells
            for dy in range(-r, r + 1):
                for dx in range(-r, r + 1):
                    d = math.hypot(dx, dy)
                    if d <= r:
                        # Highest cost near obstacle, tapering to zero at soft radius.
                        c = self.clearance_cost_weight * (1.0 - d / max(1.0, float(r)))
                        if c > 0.0:
                            offsets.append((dx, dy, c))
            for idx in obstacles:
                x = idx % w
                y = idx // w
                for dx, dy, c in offsets:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < w and 0 <= ny < h:
                        ni = ny * w + nx
                        if not occ[ni]:
                            if c > cost[ni]:
                                cost[ni] = c

        self.occ_cache = occ
        self.cost_cache = cost
        self.occ_cache_key = key
        return occ

    def nearest_free(self, start: Tuple[int, int], occ: bytearray, max_r: Optional[int] = None) -> Optional[Tuple[int, int]]:
        m = self.map_msg
        assert m is not None
        w, h = m.info.width, m.info.height
        sx, sy = start
        if max_r is None:
            max_r = self.nearest_free_radius_cells
        if 0 <= sx < w and 0 <= sy < h and not occ[sy * w + sx]:
            return start
        for r in range(1, max_r + 1):
            for dy in range(-r, r + 1):
                for dx in range(-r, r + 1):
                    if abs(dx) != r and abs(dy) != r:
                        continue
                    nx, ny = sx + dx, sy + dy
                    if 0 <= nx < w and 0 <= ny < h and not occ[ny * w + nx]:
                        return (nx, ny)
        return None

    def astar(self, start: Tuple[int, int], goal: Tuple[int, int], occ: bytearray) -> Optional[List[Tuple[int, int]]]:
        m = self.map_msg
        assert m is not None
        w, h = m.info.width, m.info.height
        sx, sy = start
        gx, gy = goal

        def idx(x, y): return y * w + x
        def heuristic(x, y): return math.hypot(gx - x, gy - y)

        open_heap = []
        heapq.heappush(open_heap, (heuristic(sx, sy), 0.0, sx, sy))
        came = {}
        gscore = {idx(sx, sy): 0.0}
        closed = set()
        expanded = 0
        neigh = [(-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0)]
        if self.diagonal_motion:
            neigh += [(-1, -1, math.sqrt(2)), (-1, 1, math.sqrt(2)), (1, -1, math.sqrt(2)), (1, 1, math.sqrt(2))]
        cost_cache = self.cost_cache if self.cost_cache is not None else [0.0] * (w * h)

        while open_heap and expanded < self.max_astar_cells:
            _, gc, x, y = heapq.heappop(open_heap)
            cur = idx(x, y)
            if cur in closed:
                continue
            closed.add(cur)
            expanded += 1
            if (x, y) == (gx, gy):
                path = [(x, y)]
                while cur in came:
                    cur = came[cur]
                    path.append((cur % w, cur // w))
                    if len(path) > self.max_path_len_cells:
                        return None
                path.reverse()
                return path
            for dx, dy, cost in neigh:
                nx, ny = x + dx, y + dy
                if nx < 0 or ny < 0 or nx >= w or ny >= h:
                    continue
                ni = idx(nx, ny)
                if occ[ni] or ni in closed:
                    continue
                if dx != 0 and dy != 0:
                    if occ[idx(x + dx, y)] or occ[idx(x, y + dy)]:
                        continue
                # Conservative traversal cost: path length plus clearance/unknown penalty.
                ng = gc + cost * (1.0 + float(cost_cache[ni]))
                if ng < gscore.get(ni, float('inf')):
                    gscore[ni] = ng
                    came[ni] = cur
                    heapq.heappush(open_heap, (ng + heuristic(nx, ny), ng, nx, ny))
        return None

    def need_replan(self, goal_xy: Tuple[float, float], robot_xy: Tuple[float, float]) -> bool:
        if self.last_replan_time is None:
            return True
        dt = self.elapsed_sec(self.last_replan_time)
        if dt is None or dt >= self.replan_period_sec:
            return True
        if not self.path_world:
            return True
        if self.last_replan_goal_xy is not None:
            if math.hypot(goal_xy[0] - self.last_replan_goal_xy[0], goal_xy[1] - self.last_replan_goal_xy[1]) >= self.replan_on_goal_delta_m:
                return True
        if self.last_replan_robot_xy is not None:
            if math.hypot(robot_xy[0] - self.last_replan_robot_xy[0], robot_xy[1] - self.last_replan_robot_xy[1]) >= self.replan_on_robot_delta_m:
                return True
        return False

    def maybe_replan(self, goal_xy: Tuple[float, float], robot_xy: Tuple[float, float]) -> str:
        if self.map_msg is None or self.robot_pose is None:
            return 'NO_MAP_OR_POSE'
        if not self.need_replan(goal_xy, robot_xy):
            return 'REUSE_PATH'
        self.last_replan_time = self.now()
        self.last_replan_goal_xy = goal_xy
        self.last_replan_robot_xy = robot_xy

        start = self.world_to_grid(robot_xy[0], robot_xy[1])
        goal = self.world_to_grid(goal_xy[0], goal_xy[1])
        if start is None or goal is None:
            self.path_world = []
            if self.publish_empty_path_on_fail:
                self.publish_path([])
            return f'GRID_FAIL start={start} goal={goal}'
        occ = self.build_occupied()
        if occ is None:
            self.path_world = []
            return 'OCC_FAIL'
        start2 = self.nearest_free(start, occ)
        goal2 = self.nearest_free(goal, occ)
        if start2 is None or goal2 is None:
            self.path_world = []
            if self.publish_empty_path_on_fail:
                self.publish_path([])
            return f'NO_FREE start2={start2} goal2={goal2}'
        path_grid = self.astar(start2, goal2, occ)
        if not path_grid:
            self.path_world = []
            if self.publish_empty_path_on_fail:
                self.publish_path([])
            return 'NO_PATH'
        self.path_world = [self.grid_to_world(x, y) for x, y in path_grid]
        self.publish_path(self.path_world)
        return f'PATH_OK n={len(self.path_world)}'

    def pick_lookahead(self, rx: float, ry: float, fallback_goal: Tuple[float, float]) -> Tuple[float, float, str]:
        if not self.path_world:
            return fallback_goal[0], fallback_goal[1], 'DIRECT_FALLBACK'
        best_i = 0
        best_d = float('inf')
        for i, (x, y) in enumerate(self.path_world):
            d = (x - rx) ** 2 + (y - ry) ** 2
            if d < best_d:
                best_d = d
                best_i = i
        accum = 0.0
        prev = self.path_world[best_i]
        for j in range(best_i + 1, len(self.path_world)):
            cur = self.path_world[j]
            accum += math.hypot(cur[0] - prev[0], cur[1] - prev[1])
            prev = cur
            if accum >= self.lookahead_distance:
                return cur[0], cur[1], 'PATH_LOOKAHEAD'
        x, y = self.path_world[-1]
        return x, y, 'PATH_END'

    def set_status(self, status: str):
        self.last_status = status

    def control_loop(self):
        if self.robot_pose is None or not self.age_ok(self.robot_stamp, self.stale_pose_sec):
            self.set_status('STOP_STALE_ROBOT_POSE')
            self.stop()
            return
        if self.map_msg is None or not self.age_ok(self.map_stamp, self.stale_map_sec):
            self.set_status('STOP_STALE_MAP')
            self.stop()
            return
        goal_xy = self.target_pose_xy()
        if goal_xy is None:
            self.set_status('STOP_NO_TARGET')
            self.stop()
            return

        rx = float(self.robot_pose.pose.position.x)
        ry = float(self.robot_pose.pose.position.y)
        ryaw = yaw_from_quat(self.robot_pose.pose.orientation)
        robot_xy = (rx, ry)
        dist_goal = math.hypot(goal_xy[0] - rx, goal_xy[1] - ry)
        if dist_goal <= self.goal_tolerance:
            self.set_status('STOP_GOAL_TOLERANCE')
            self.stop()
            return

        replan_status = self.maybe_replan(goal_xy, robot_xy)
        if not self.path_world and not self.direct_fallback_if_no_path:
            self.set_status(f'STOP_{replan_status}')
            self.stop()
            return

        tx, ty, target_src = self.pick_lookahead(rx, ry, goal_xy)
        dx = tx - rx
        dy = ty - ry
        target_dist = math.hypot(dx, dy)
        target_yaw = math.atan2(dy, dx)
        yaw_err = norm_angle(target_yaw - ryaw)

        linear = clamp(self.k_linear * target_dist, self.min_linear, self.max_linear)
        if dist_goal < self.goal_slow_radius:
            linear *= clamp(dist_goal / max(self.goal_tolerance, self.goal_slow_radius), 0.25, 1.0)
        angular = clamp(self.k_angular * yaw_err, -self.max_angular, self.max_angular)

        if abs(yaw_err) > self.rotate_yaw_threshold:
            linear = 0.0
        elif abs(yaw_err) > self.slow_yaw_threshold:
            linear *= 0.35

        if self.front_min < self.front_stop_distance:
            linear = 0.0
            turn = self.safety_turn_bias if angular >= 0.0 else -self.safety_turn_bias
            angular = clamp(turn, -self.max_angular, self.max_angular)
            target_src += '+FRONT_STOP'
        elif self.front_min < self.front_slow_distance:
            linear *= 0.35
            target_src += '+FRONT_SLOW'

        cmd = TwistStamped()
        cmd.header.stamp = self.now().to_msg()
        cmd.header.frame_id = 'base_footprint'
        cmd.twist.linear.x = float(linear)
        cmd.twist.angular.z = float(angular)
        self.cmd_pub.publish(cmd)
        self.set_status(f'{replan_status}/{target_src}')

        log_dt = self.elapsed_sec(self.last_log_time)
        if self.last_log_time is None or log_dt is None or log_dt >= self.log_period_sec:
            self.last_log_time = self.now()
            self.get_logger().info(
                f'V55_ASTAR_CMD | robot={self.robot_name} mode={self.target_mode}/{self.leader_goal_mode} '
                f'status={self.last_status} goal=({goal_xy[0]:.2f},{goal_xy[1]:.2f}) '
                f'robot=({rx:.2f},{ry:.2f},{ryaw:.2f}) dist={dist_goal:.2f} path_n={len(self.path_world)} '
                f'target=({tx:.2f},{ty:.2f}) yaw_err={yaw_err:.2f} cmd=({linear:.2f},{angular:.2f}) front={self.front_min:.2f}'
            )


def main():
    rclpy.init()
    node = AStarCmdVelFollower()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.stop()
        except Exception:
            pass
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
