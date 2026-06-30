#!/usr/bin/env python3

import math
import heapq
from collections import deque
from typing import List, Optional, Tuple

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from rclpy.exceptions import ParameterAlreadyDeclaredException

from geometry_msgs.msg import PoseStamped, TwistStamped
from nav_msgs.msg import OccupancyGrid, Path
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Header


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
    """Minimal grid A* + pure-pursuit-ish cmd_vel controller.

    This node is intentionally independent from Nav2. It is used as a debugging
    control path for Waffle or Burger when live SLAM + Nav2 lifecycle/costmap
    behavior is too opaque.

    Inputs:
      /map: OccupancyGrid in map frame
      /leader_pose: Waffle pose in map frame
      robot_pose_topic: robot pose in map frame
      /scan_nav: robot scan for last-resort front safety
      manual_goal_topic: optional manual goal in map frame

    Output:
      /cmd_vel: geometry_msgs/TwistStamped for Jazzy-style Nav2-compatible pipeline
      /astar_path: nav_msgs/Path for RViz debug
    """

    def __init__(self):
        super().__init__('astar_cmd_vel_follower')

        safe_declare(self, 'use_sim_time', True)
        self.map_topic = str(safe_declare(self, 'map_topic', '/map'))
        self.robot_pose_topic = str(safe_declare(self, 'robot_pose_topic', '/burger_pose'))
        self.leader_pose_topic = str(safe_declare(self, 'leader_pose_topic', '/leader_pose'))
        self.manual_goal_topic = str(safe_declare(self, 'manual_goal_topic', '/burger_goal_pose'))
        self.scan_topic = str(safe_declare(self, 'scan_topic', '/scan_nav'))
        self.cmd_vel_topic = str(safe_declare(self, 'cmd_vel_topic', '/cmd_vel'))
        self.path_topic = str(safe_declare(self, 'path_topic', '/astar_path'))
        self.frame_id = str(safe_declare(self, 'frame_id', 'map'))

        self.target_mode = str(safe_declare(self, 'target_mode', 'leader'))  # leader | manual
        self.follow_distance = float(safe_declare(self, 'follow_distance', 1.05))
        self.goal_tolerance = float(safe_declare(self, 'goal_tolerance', 0.25))
        self.replan_period_sec = float(safe_declare(self, 'replan_period_sec', 0.8))
        self.control_rate_hz = float(safe_declare(self, 'control_rate_hz', 10.0))
        self.lookahead_distance = float(safe_declare(self, 'lookahead_distance', 0.45))

        self.occupied_threshold = int(safe_declare(self, 'occupied_threshold', 55))
        self.treat_unknown_as_obstacle = bool(safe_declare(self, 'treat_unknown_as_obstacle', False))
        self.inflation_radius_m = float(safe_declare(self, 'inflation_radius_m', 0.18))
        self.max_astar_cells = int(safe_declare(self, 'max_astar_cells', 70000))
        self.max_path_len_cells = int(safe_declare(self, 'max_path_len_cells', 3000))

        self.max_linear = float(safe_declare(self, 'max_linear', 0.16))
        self.min_linear = float(safe_declare(self, 'min_linear', 0.03))
        self.max_angular = float(safe_declare(self, 'max_angular', 0.75))
        self.k_linear = float(safe_declare(self, 'k_linear', 0.55))
        self.k_angular = float(safe_declare(self, 'k_angular', 1.6))
        self.slow_yaw_threshold = float(safe_declare(self, 'slow_yaw_threshold', 0.75))
        self.rotate_yaw_threshold = float(safe_declare(self, 'rotate_yaw_threshold', 1.25))

        self.front_stop_distance = float(safe_declare(self, 'front_stop_distance', 0.25))
        self.front_slow_distance = float(safe_declare(self, 'front_slow_distance', 0.45))
        self.front_sector_deg = float(safe_declare(self, 'front_sector_deg', 35.0))

        self.stale_pose_sec = float(safe_declare(self, 'stale_pose_sec', 2.0))
        self.stale_map_sec = float(safe_declare(self, 'stale_map_sec', 5.0))
        self.stale_goal_sec = float(safe_declare(self, 'stale_goal_sec', 5.0))
        self.log_period_sec = float(safe_declare(self, 'log_period_sec', 2.0))

        self.map_msg: Optional[OccupancyGrid] = None
        self.map_stamp = None
        self.occ_cache = None
        self.occ_cache_key = None
        self.robot_pose: Optional[PoseStamped] = None
        self.leader_pose: Optional[PoseStamped] = None
        self.manual_goal: Optional[PoseStamped] = None
        self.robot_stamp = None
        self.leader_stamp = None
        self.manual_goal_stamp = None
        self.front_min = float('inf')
        self.path_world: List[Tuple[float, float]] = []
        self.path_goal_key = None
        self.last_replan_time = self.get_clock().now() - Duration(seconds=999.0)
        self.last_log_time = self.get_clock().now() - Duration(seconds=999.0)

        qos = 10
        self.create_subscription(OccupancyGrid, self.map_topic, self.on_map, qos)
        self.create_subscription(PoseStamped, self.robot_pose_topic, self.on_robot_pose, qos)
        self.create_subscription(PoseStamped, self.leader_pose_topic, self.on_leader_pose, qos)
        self.create_subscription(PoseStamped, self.manual_goal_topic, self.on_manual_goal, qos)
        self.create_subscription(LaserScan, self.scan_topic, self.on_scan, qos)
        self.cmd_pub = self.create_publisher(TwistStamped, self.cmd_vel_topic, qos)
        self.path_pub = self.create_publisher(Path, self.path_topic, qos)

        period = 1.0 / max(1.0, self.control_rate_hz)
        self.timer = self.create_timer(period, self.control_loop)

        self.get_logger().info(
            'V48_ASTAR_CMD_VEL_CONTROLLER_READY | direct /cmd_vel TwistStamped controller | '
            f'mode={self.target_mode} map={self.map_topic} robot={self.robot_pose_topic} '
            f'leader={self.leader_pose_topic} manual={self.manual_goal_topic} cmd={self.cmd_vel_topic} '
            f'follow_distance={self.follow_distance:.2f} lookahead={self.lookahead_distance:.2f}'
        )

    def now(self):
        return self.get_clock().now()

    def age_ok(self, stamp, limit_sec: float) -> bool:
        if stamp is None:
            return False
        if limit_sec <= 0.0:
            return True
        try:
            return (self.now() - stamp).nanoseconds * 1e-9 <= limit_sec
        except Exception:
            return False

    def on_map(self, msg: OccupancyGrid):
        self.map_msg = msg
        self.map_stamp = self.now()
        key = (msg.info.width, msg.info.height, msg.info.resolution, msg.info.origin.position.x, msg.info.origin.position.y, len(msg.data))
        if key != self.occ_cache_key:
            self.occ_cache = None
            self.occ_cache_key = key

    def on_robot_pose(self, msg: PoseStamped):
        self.robot_pose = msg
        self.robot_stamp = self.now()

    def on_leader_pose(self, msg: PoseStamped):
        self.leader_pose = msg
        self.leader_stamp = self.now()

    def on_manual_goal(self, msg: PoseStamped):
        self.manual_goal = msg
        self.manual_goal_stamp = self.now()
        if self.target_mode != 'manual':
            self.get_logger().info('MANUAL_GOAL_RECEIVED_BUT_TARGET_MODE_IS_NOT_MANUAL | set target_mode:=manual to use /burger_goal_pose')

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

    def target_pose_xy(self) -> Optional[Tuple[float, float]]:
        if self.target_mode == 'manual':
            if self.manual_goal is None or not self.age_ok(self.manual_goal_stamp, self.stale_goal_sec):
                return None
            return (self.manual_goal.pose.position.x, self.manual_goal.pose.position.y)

        if self.leader_pose is None or not self.age_ok(self.leader_stamp, self.stale_goal_sec):
            return None
        lx = self.leader_pose.pose.position.x
        ly = self.leader_pose.pose.position.y
        lyaw = yaw_from_quat(self.leader_pose.pose.orientation)
        gx = lx - self.follow_distance * math.cos(lyaw)
        gy = ly - self.follow_distance * math.sin(lyaw)
        return (gx, gy)

    def world_to_grid(self, x: float, y: float) -> Optional[Tuple[int, int]]:
        m = self.map_msg
        if m is None:
            return None
        ox = m.info.origin.position.x
        oy = m.info.origin.position.y
        res = m.info.resolution
        gx = int(math.floor((x - ox) / res))
        gy = int(math.floor((y - oy) / res))
        if gx < 0 or gy < 0 or gx >= m.info.width or gy >= m.info.height:
            return None
        return gx, gy

    def grid_to_world(self, gx: int, gy: int) -> Tuple[float, float]:
        m = self.map_msg
        assert m is not None
        ox = m.info.origin.position.x
        oy = m.info.origin.position.y
        res = m.info.resolution
        return (ox + (gx + 0.5) * res, oy + (gy + 0.5) * res)

    def build_occupied(self):
        m = self.map_msg
        if m is None:
            return None
        key = (m.info.width, m.info.height, m.info.resolution, m.info.origin.position.x, m.info.origin.position.y, len(m.data), self.occupied_threshold, self.treat_unknown_as_obstacle, self.inflation_radius_m)
        if self.occ_cache is not None and self.occ_cache_key == key:
            return self.occ_cache

        w, h = m.info.width, m.info.height
        occ = bytearray(w * h)
        q = deque()
        for i, v in enumerate(m.data):
            blocked = (v >= self.occupied_threshold) or (v < 0 and self.treat_unknown_as_obstacle)
            if blocked:
                occ[i] = 1
                q.append(i)

        inflate_cells = int(math.ceil(self.inflation_radius_m / max(1e-6, m.info.resolution)))
        if inflate_cells > 0 and q:
            base = bytearray(occ)
            offsets = []
            r2 = inflate_cells * inflate_cells
            for dy in range(-inflate_cells, inflate_cells + 1):
                for dx in range(-inflate_cells, inflate_cells + 1):
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

        self.occ_cache = occ
        self.occ_cache_key = key
        return occ

    def nearest_free(self, start: Tuple[int, int], occ: bytearray, max_r: int = 12) -> Optional[Tuple[int, int]]:
        m = self.map_msg
        assert m is not None
        w, h = m.info.width, m.info.height
        sx, sy = start
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
        neigh = [(-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
                 (-1, -1, math.sqrt(2)), (-1, 1, math.sqrt(2)), (1, -1, math.sqrt(2)), (1, 1, math.sqrt(2))]

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
                # Prevent diagonal corner cutting through inflated obstacles.
                if dx != 0 and dy != 0:
                    if occ[idx(x + dx, y)] or occ[idx(x, y + dy)]:
                        continue
                ng = gc + cost
                if ng < gscore.get(ni, float('inf')):
                    gscore[ni] = ng
                    came[ni] = cur
                    heapq.heappush(open_heap, (ng + heuristic(nx, ny), ng, nx, ny))
        return None

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

    def maybe_replan(self, goal_xy: Tuple[float, float]):
        if self.map_msg is None or self.robot_pose is None:
            return
        now = self.now()
        if (now - self.last_replan_time).nanoseconds * 1e-9 < self.replan_period_sec:
            return
        self.last_replan_time = now

        rx = self.robot_pose.pose.position.x
        ry = self.robot_pose.pose.position.y
        start = self.world_to_grid(rx, ry)
        goal = self.world_to_grid(goal_xy[0], goal_xy[1])
        if start is None or goal is None:
            self.path_world = []
            return
        occ = self.build_occupied()
        if occ is None:
            self.path_world = []
            return
        start2 = self.nearest_free(start, occ)
        goal2 = self.nearest_free(goal, occ, max_r=20)
        if start2 is None or goal2 is None:
            self.path_world = []
            return
        path_grid = self.astar(start2, goal2, occ)
        if not path_grid:
            self.path_world = []
            return
        self.path_world = [self.grid_to_world(x, y) for x, y in path_grid]
        self.publish_path(self.path_world)

    def pick_lookahead(self, rx: float, ry: float, fallback_goal: Tuple[float, float]) -> Tuple[float, float]:
        if not self.path_world:
            return fallback_goal
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
                return cur
        return self.path_world[-1]

    def control_loop(self):
        if self.robot_pose is None or not self.age_ok(self.robot_stamp, self.stale_pose_sec):
            self.stop()
            return
        if self.map_msg is None or not self.age_ok(self.map_stamp, self.stale_map_sec):
            self.stop()
            return
        goal_xy = self.target_pose_xy()
        if goal_xy is None:
            self.stop()
            return

        rx = self.robot_pose.pose.position.x
        ry = self.robot_pose.pose.position.y
        ryaw = yaw_from_quat(self.robot_pose.pose.orientation)
        dist_goal = math.hypot(goal_xy[0] - rx, goal_xy[1] - ry)
        if dist_goal <= self.goal_tolerance:
            self.stop()
            return

        self.maybe_replan(goal_xy)
        tx, ty = self.pick_lookahead(rx, ry, goal_xy)
        dx = tx - rx
        dy = ty - ry
        target_dist = math.hypot(dx, dy)
        target_yaw = math.atan2(dy, dx)
        yaw_err = norm_angle(target_yaw - ryaw)

        linear = clamp(self.k_linear * target_dist, self.min_linear, self.max_linear)
        angular = clamp(self.k_angular * yaw_err, -self.max_angular, self.max_angular)

        if abs(yaw_err) > self.rotate_yaw_threshold:
            linear = 0.0
        elif abs(yaw_err) > self.slow_yaw_threshold:
            linear *= 0.35

        if self.front_min < self.front_stop_distance:
            linear = 0.0
            angular = clamp(0.55 if angular >= 0.0 else -0.55, -self.max_angular, self.max_angular)
        elif self.front_min < self.front_slow_distance:
            linear *= 0.35

        cmd = TwistStamped()
        cmd.header.stamp = self.now().to_msg()
        cmd.header.frame_id = 'base_footprint'
        cmd.twist.linear.x = float(linear)
        cmd.twist.angular.z = float(angular)
        self.cmd_pub.publish(cmd)

        if (self.now() - self.last_log_time).nanoseconds * 1e-9 >= self.log_period_sec:
            self.last_log_time = self.now()
            self.get_logger().info(
                f'ASTAR_CMD | mode={self.target_mode} goal=({goal_xy[0]:.2f},{goal_xy[1]:.2f}) '
                f'robot=({rx:.2f},{ry:.2f},{ryaw:.2f}) dist={dist_goal:.2f} '
                f'path_n={len(self.path_world)} target=({tx:.2f},{ty:.2f}) yaw_err={yaw_err:.2f} '
                f'cmd=({linear:.2f},{angular:.2f}) front={self.front_min:.2f}'
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
