#!/usr/bin/env python3
"""Fleet Goal Dispatcher — A* + collision state machine.

Path planning:
  - A* on inflated occupancy grid (walls + path_inflation_radius)
  - Peer robot injected as soft+hard obstacle in each other's A* grid
  - Full path divided into waypoints every waypoint_step_m
  - Published as nav_msgs/Path to NavigateThroughPoses proxies

Collision avoidance state machine (Burger = lower priority):
  MOVING  → if dist < peer_stop_dist  → WAITING  (hold in place)
          → if dist < peer_back_dist  → BACKING  (reverse waypoint)
  WAITING → if dist > peer_clear_dist → MOVING   (replan + resume)
  BACKING → backed up enough OR dist > peer_stop_dist → WAITING
"""

from __future__ import annotations

import heapq
import math
from collections import deque
from dataclasses import dataclass
from typing import List, Optional, Tuple

import rclpy
from rclpy.node import Node
from rclpy.exceptions import ParameterAlreadyDeclaredException
from geometry_msgs.msg import Point, PoseStamped
from nav_msgs.msg import OccupancyGrid, Path
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import Marker, MarkerArray


def _safe_declare(node, name, default):
    try:
        node.declare_parameter(name, default)
    except ParameterAlreadyDeclaredException:
        pass
    return node.get_parameter(name).value


def _quat_to_yaw(q) -> float:
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


@dataclass
class _Candidate:
    x: float
    y: float
    score: float
    clearance: float


# Collision avoidance states for Burger (lower-priority robot)
_MOVING  = 'MOVING'
_WAITING = 'WAITING'
_BACKING = 'BACKING'


class FleetGoalDispatcher(Node):
    """A*-based fleet dispatcher with inter-robot collision avoidance."""

    def __init__(self):
        super().__init__('fleet_goal_dispatcher')

        # --- Topics ---
        _safe_declare(self, 'use_sim_time', True)
        self.input_goal_topic   = str(_safe_declare(self, 'input_goal_topic',      '/fleet_goal_pose'))
        alias_raw               = str(_safe_declare(self, 'alias_topics',          '/goal_pose,/move_base_simple/goal'))
        self.alias_topics       = [t.strip() for t in alias_raw.split(',') if t.strip()]
        self.waffle_pose_topic  = str(_safe_declare(self, 'waffle_pose_topic',     '/leader_pose'))
        self.burger_pose_topic  = str(_safe_declare(self, 'burger_pose_topic',     '/burger_pose'))
        self.map_topic          = str(_safe_declare(self, 'map_topic',             '/map'))
        self.frame_id           = str(_safe_declare(self, 'frame_id',             'map'))
        self.waffle_waypoints_topic = str(_safe_declare(self, 'waffle_waypoints_topic', '/waffle_waypoints'))
        self.burger_waypoints_topic = str(_safe_declare(self, 'burger_waypoints_topic', '/burger_waypoints'))

        # --- Formation ---
        self.formation_separation_m  = float(_safe_declare(self, 'formation_separation_m',  1.20))
        self.min_pair_distance_m     = float(_safe_declare(self, 'min_pair_distance_m',     0.85))
        self.search_rings            = int  (_safe_declare(self, 'search_rings',             4))
        self.search_angles           = int  (_safe_declare(self, 'search_angles',            20))
        self.clearance_check_radius_m= float(_safe_declare(self, 'clearance_check_radius_m', 0.55))
        self.occupied_threshold      = int  (_safe_declare(self, 'occupied_threshold',       45))

        # --- A* / grid ---
        self.path_inflation_radius_m = float(_safe_declare(self, 'path_inflation_radius_m', 0.20))
        self.peer_hard_radius_m      = float(_safe_declare(self, 'peer_hard_radius_m',      0.45))
        self.peer_soft_radius_m      = float(_safe_declare(self, 'peer_soft_radius_m',      0.85))
        self.peer_soft_cost          = float(_safe_declare(self, 'peer_soft_cost',           8.0))

        # --- Waypoints ---
        self.waypoint_step_m         = float(_safe_declare(self, 'waypoint_step_m',         0.80))
        self.waypoint_final_snap_m   = float(_safe_declare(self, 'waypoint_final_snap_m',   0.80))

        # --- Update ---
        self.republish_period_sec    = float(_safe_declare(self, 'republish_period_sec',    0.75))
        self.change_threshold_m      = float(_safe_declare(self, 'change_threshold_m',      0.20))
        self.max_idle_sec            = float(_safe_declare(self, 'max_idle_sec',             4.0))

        # --- Collision avoidance ---
        self.peer_stop_dist   = float(_safe_declare(self, 'peer_stop_dist',   0.80))
        self.peer_back_dist   = float(_safe_declare(self, 'peer_back_dist',   0.55))
        self.peer_clear_dist  = float(_safe_declare(self, 'peer_clear_dist',  1.05))
        self.backup_dist_m    = float(_safe_declare(self, 'backup_dist_m',    0.40))

        # --- Debug ---
        self.debug_markers       = bool(_safe_declare(self, 'debug_markers',       True))
        self.debug_markers_topic = str (_safe_declare(self, 'debug_markers_topic', '/fleet_debug_markers'))

        # --- State ---
        self.map          : Optional[OccupancyGrid] = None
        self.waffle_pose  : Optional[PoseStamped]   = None
        self.burger_pose  : Optional[PoseStamped]   = None
        self.final_pair   : Optional[Tuple]         = None
        self.clicked_goal : Optional[PoseStamped]   = None
        self.last_w_chain : List = []
        self.last_b_chain : List = []
        self.last_publish_time = -999.0
        self._inflated_grid    = None
        self._inflated_map_id  = 0
        self._static_path_cache     = {}
        self._static_cache_map_id   = -1

        # Collision state machine
        self._burger_state   = _MOVING
        self._backup_start   : Optional[Tuple[float, float]] = None

        # --- Publishers ---
        self.pub_waffle  = self.create_publisher(Path,        self.waffle_waypoints_topic, 10)
        self.pub_burger  = self.create_publisher(Path,        self.burger_waypoints_topic, 10)
        self.pub_markers = self.create_publisher(MarkerArray, self.debug_markers_topic,    10)

        # --- Subscriptions ---
        self.create_subscription(OccupancyGrid, self.map_topic,         self._on_map,         10)
        self.create_subscription(PoseStamped,   self.waffle_pose_topic,  self._on_waffle_pose, 10)
        self.create_subscription(PoseStamped,   self.burger_pose_topic,  self._on_burger_pose, 10)
        self.create_subscription(PoseStamped,   self.input_goal_topic,   self._on_fleet_goal,  10)
        seen = {self.input_goal_topic}
        for t in self.alias_topics:
            if t and t not in seen:
                seen.add(t)
                self.create_subscription(PoseStamped, t, self._on_fleet_goal, 10)

        self.create_timer(self.republish_period_sec, self._timer_cb)
        self.get_logger().info(
            f'FLEET_DISPATCHER_READY | in={self.input_goal_topic} '
            f'sep={self.formation_separation_m:.2f}m step={self.waypoint_step_m:.2f}m '
            f'stop={self.peer_stop_dist:.2f}m back={self.peer_back_dist:.2f}m '
            f'clear={self.peer_clear_dist:.2f}m'
        )

    # =========================================================================
    # Callbacks
    # =========================================================================

    def _on_map(self, msg: OccupancyGrid) -> None:
        self.map = msg
        self._inflated_map_id += 1
        self._inflated_grid = None
        self._static_path_cache.clear()

    def _on_waffle_pose(self, msg: PoseStamped) -> None:
        self.waffle_pose = msg

    def _on_burger_pose(self, msg: PoseStamped) -> None:
        self.burger_pose = msg

    def _on_fleet_goal(self, msg: PoseStamped) -> None:
        x   = float(msg.pose.position.x)
        y   = float(msg.pose.position.y)
        yaw = _quat_to_yaw(msg.pose.orientation)
        if not math.isfinite(yaw):
            yaw = 0.0
        self.clicked_goal = msg
        self._burger_state = _MOVING   # reset on new goal
        self._backup_start = None
        waffle_xy, burger_xy = self._compute_final_pair(x, y, yaw)
        self.final_pair = (waffle_xy, burger_xy)
        self.get_logger().info(
            f'FLEET_GOAL | clicked=({x:.3f},{y:.3f}) '
            f'waffle=({waffle_xy[0]:.3f},{waffle_xy[1]:.3f}) '
            f'burger=({burger_xy[0]:.3f},{burger_xy[1]:.3f})'
        )
        stamp = self.get_clock().now().to_msg()
        self._publish_chains(stamp, force=True)

    # =========================================================================
    # Timer — main replan loop
    # =========================================================================

    def _timer_cb(self) -> None:
        if self.final_pair is None:
            return
        stamp    = self.get_clock().now().to_msg()
        now_sec  = self.get_clock().now().nanoseconds * 1e-9
        idle_sec = now_sec - self.last_publish_time

        # --- Collision state machine for Burger ---
        collision_handled = self._update_collision_state(stamp, now_sec)

        # --- Waffle always gets A* path (unaffected by burger state) ---
        w_peer = self._burger_xy()
        w_chain = self._build_waypoint_chain(self.waffle_pose, self.final_pair[0], peer_xy=w_peer)

        if collision_handled:
            # Only update Waffle; Burger handled by collision logic
            if self._chain_changed(w_chain, self.last_w_chain) or idle_sec >= self.max_idle_sec:
                self.pub_waffle.publish(self._chain_to_path(w_chain, stamp))
                self.last_w_chain = list(w_chain)
                if self.debug_markers:
                    self._publish_debug_markers(w_chain, self.last_b_chain, stamp)
            return

        # --- Normal mode: both robots get A* paths ---
        b_peer  = self._waffle_xy()
        b_chain = self._build_waypoint_chain(self.burger_pose, self.final_pair[1], peer_xy=b_peer)

        if self._chain_changed(w_chain, self.last_w_chain) or \
           self._chain_changed(b_chain, self.last_b_chain) or \
           idle_sec >= self.max_idle_sec:
            self._do_publish(w_chain, b_chain, stamp)
            self.last_publish_time = now_sec

    # =========================================================================
    # Collision State Machine
    # =========================================================================

    def _update_collision_state(self, stamp, now_sec: float) -> bool:
        """Update Burger state. Returns True if Burger is in WAITING/BACKING."""
        if self.waffle_pose is None or self.burger_pose is None:
            return False

        wx, wy = self._waffle_xy()
        bx, by = self._burger_xy()
        dist   = math.hypot(wx - bx, wy - by)
        prev   = self._burger_state

        # --- Transitions ---
        if self._burger_state == _MOVING:
            if dist < self.peer_back_dist:
                self._burger_state = _BACKING
                self._backup_start = (bx, by)
            elif dist < self.peer_stop_dist:
                self._burger_state = _WAITING

        elif self._burger_state == _WAITING:
            if dist >= self.peer_clear_dist:
                self._burger_state = _MOVING
                self._static_path_cache.clear()   # force fresh A* on resume
                self._backup_start = None

        elif self._burger_state == _BACKING:
            backed = 0.0
            if self._backup_start is not None:
                backed = math.hypot(bx - self._backup_start[0], by - self._backup_start[1])
            if backed >= self.backup_dist_m or dist >= self.peer_stop_dist:
                self._burger_state = _WAITING

        if prev != self._burger_state:
            self.get_logger().warn(
                f'BURGER_STATE | {prev} → {self._burger_state} | dist={dist:.2f}m'
            )

        # --- Actions ---
        if self._burger_state == _WAITING:
            # Hold in place: publish single-pose path at current position
            hold = self._chain_to_path([(bx, by)], stamp)
            self.pub_burger.publish(hold)
            self.last_b_chain = [(bx, by)]
            return True

        if self._burger_state == _BACKING:
            # Publish a waypoint directly behind the burger
            byaw   = _quat_to_yaw(self.burger_pose.pose.orientation)
            back_x = bx - self.backup_dist_m * math.cos(byaw)
            back_y = by - self.backup_dist_m * math.sin(byaw)
            backup = self._chain_to_path([(back_x, back_y)], stamp)
            self.pub_burger.publish(backup)
            self.last_b_chain = [(back_x, back_y)]
            self.get_logger().info(
                f'BURGER_BACKUP | from=({bx:.2f},{by:.2f}) to=({back_x:.2f},{back_y:.2f}) '
                f'dist_to_waffle={dist:.2f}m'
            )
            return True

        return False   # _MOVING, normal planning

    def _waffle_xy(self) -> Optional[Tuple[float, float]]:
        if self.waffle_pose is None:
            return None
        return (float(self.waffle_pose.pose.position.x),
                float(self.waffle_pose.pose.position.y))

    def _burger_xy(self) -> Optional[Tuple[float, float]]:
        if self.burger_pose is None:
            return None
        return (float(self.burger_pose.pose.position.x),
                float(self.burger_pose.pose.position.y))

    # =========================================================================
    # Final-slot computation
    # =========================================================================

    def _compute_final_pair(self, x, y, yaw):
        candidates = self._make_candidates(x, y, yaw)
        if len(candidates) < 2:
            return self._fallback_pair(x, y, yaw)
        scored = []
        for i, c1 in enumerate(candidates):
            for c2 in candidates[i + 1:]:
                d = math.hypot(c1.x - c2.x, c1.y - c2.y)
                if d < self.min_pair_distance_m:
                    continue
                center_err = math.hypot(0.5 * (c1.x + c2.x) - x,
                                        0.5 * (c1.y + c2.y) - y)
                sep_err = abs(d - self.formation_separation_m)
                sc = c1.score + c2.score - 1.3 * center_err - 0.8 * sep_err
                scored.append((sc, c1, c2))
        scored.sort(key=lambda t: t[0], reverse=True)
        for sc, c1, c2 in scored[:20]:
            if self._reachable(c1.x, c1.y, c2.x, c2.y):
                return self._assign_by_distance((c1.x, c1.y), (c2.x, c2.y))
        if scored:
            _, c1, c2 = scored[0]
            return self._assign_by_distance((c1.x, c1.y), (c2.x, c2.y))
        return self._fallback_pair(x, y, yaw)

    def _fallback_pair(self, x, y, yaw):
        r  = self.formation_separation_m * 0.5
        lx = -math.sin(yaw)
        ly =  math.cos(yaw)
        return self._assign_by_distance(
            (x + lx * r, y + ly * r),
            (x - lx * r, y - ly * r),
        )

    def _assign_by_distance(self, a, b):
        wp = self._waffle_xy()
        bp = self._burger_xy()
        if wp is None or bp is None:
            return a, b
        keep = math.hypot(wp[0]-a[0], wp[1]-a[1]) + math.hypot(bp[0]-b[0], bp[1]-b[1])
        swap = math.hypot(wp[0]-b[0], wp[1]-b[1]) + math.hypot(bp[0]-a[0], bp[1]-a[1])
        return (b, a) if swap + 0.05 < keep else (a, b)

    def _make_candidates(self, x, y, yaw):
        out = []
        rings  = max(1, self.search_rings)
        angles = max(8, self.search_angles)
        for ring in range(1, rings + 1):
            radius = self.formation_separation_m * (0.3 + 0.25 * ring)
            for j in range(angles):
                ang = yaw + 2.0 * math.pi * j / angles
                cx  = x + radius * math.cos(ang)
                cy  = y + radius * math.sin(ang)
                ev  = self._evaluate_candidate(cx, cy, x, y)
                if ev is not None:
                    out.append(ev)
        out.sort(key=lambda c: c.score, reverse=True)
        return out[:64]

    def _evaluate_candidate(self, cx, cy, gx, gy):
        grid = self.map
        if grid is None:
            return None
        res = float(grid.info.resolution)
        if res <= 0.0:
            return None
        mx, my = self._world_to_grid(cx, cy)
        if mx is None:
            return None
        w = int(grid.info.width)
        h = int(grid.info.height)
        if not (0 <= mx < w and 0 <= my < h):
            return None
        if int(grid.data[my * w + mx]) >= self.occupied_threshold:
            return None
        rad_cells = max(1, int(self.clearance_check_radius_m / res))
        occ = 0
        min_occ_d = self.clearance_check_radius_m + res
        for dy in range(-rad_cells, rad_cells + 1):
            yy = my + dy
            if not (0 <= yy < h):
                continue
            for dx in range(-rad_cells, rad_cells + 1):
                xx = mx + dx
                if not (0 <= xx < w):
                    continue
                d = math.hypot(dx * res, dy * res)
                if d > self.clearance_check_radius_m:
                    continue
                if int(grid.data[yy * w + xx]) >= self.occupied_threshold:
                    occ += 1
                    if d < min_occ_d:
                        min_occ_d = d
        if occ > 0 and min_occ_d < 0.18:
            return None
        if min_occ_d > self.clearance_check_radius_m:
            min_occ_d = self.clearance_check_radius_m
        score = 2.0 * min_occ_d - 0.6 * math.hypot(cx - gx, cy - gy) - 2.0 * occ / 100.0
        return _Candidate(cx, cy, score, min_occ_d)

    # =========================================================================
    # Inflated grid (cached)
    # =========================================================================

    def _ensure_inflated_grid(self):
        if self.map is None:
            return None
        if (self._inflated_grid is not None and
                self._static_cache_map_id == self._inflated_map_id):
            return self._inflated_grid
        grid = self.map
        w    = int(grid.info.width)
        h    = int(grid.info.height)
        res  = float(grid.info.resolution)
        if w <= 0 or h <= 0 or res <= 0:
            return None
        thr  = self.occupied_threshold
        base = bytearray(w * h)
        for i, v in enumerate(grid.data):
            if int(v) >= thr:
                base[i] = 1
        rad = max(1, int(math.ceil(self.path_inflation_radius_m / res)))
        inflated = bytearray(base)
        for cy in range(h):
            for cx in range(w):
                if not base[cy * w + cx]:
                    continue
                for dy in range(-rad, rad + 1):
                    ny = cy + dy
                    if not (0 <= ny < h):
                        continue
                    for dx in range(-rad, rad + 1):
                        nx = cx + dx
                        if not (0 <= nx < w):
                            continue
                        if math.hypot(dx * res, dy * res) <= self.path_inflation_radius_m:
                            inflated[ny * w + nx] = 1
        self._inflated_grid   = inflated
        self._static_cache_map_id = self._inflated_map_id
        self._static_path_cache.clear()
        return inflated

    # =========================================================================
    # A* path planner
    # =========================================================================

    def _astar(self, sx, sy, gx, gy, w, h, res, inflated,
               peer_gx=None, peer_gy=None, max_cells=120000):
        """A* on inflated grid. Peer robot injected as soft+hard obstacle."""
        peer_hard = 0
        peer_soft = 0
        if peer_gx is not None:
            peer_hard = max(1, int(math.ceil(self.peer_hard_radius_m / res)))
            peer_soft = max(1, int(math.ceil(self.peer_soft_radius_m / res)))

        def heuristic(x, y):
            return math.hypot(gx - x, gy - y)

        NEIGHBORS = [(-1,0,1.0),(1,0,1.0),(0,-1,1.0),(0,1,1.0),
                     (-1,-1,1.414),(-1,1,1.414),(1,-1,1.414),(1,1,1.414)]

        open_heap = []
        heapq.heappush(open_heap, (heuristic(sx, sy), 0.0, sx, sy))
        came: dict = {}
        gscore: dict = {(sx, sy): 0.0}
        expanded = 0

        while open_heap and expanded < max_cells:
            _, gc, cx, cy = heapq.heappop(open_heap)
            key = (cx, cy)
            if gscore.get(key, float('inf')) < gc - 1e-9:
                continue
            expanded += 1
            if cx == gx and cy == gy:
                path_grid = [key]
                while key in came:
                    key = came[key]
                    path_grid.append(key)
                path_grid.reverse()
                return path_grid
            for ddx, ddy, move_cost in NEIGHBORS:
                nx, ny = cx + ddx, cy + ddy
                nkey = (nx, ny)
                if not (0 <= nx < w and 0 <= ny < h):
                    continue
                if inflated[ny * w + nx]:
                    continue
                if ddx != 0 and ddy != 0:
                    if inflated[cy * w + (cx + ddx)] or inflated[(cy + ddy) * w + cx]:
                        continue
                extra = 0.0
                if peer_gx is not None:
                    pd = math.hypot(nx - peer_gx, ny - peer_gy)
                    if pd < peer_hard:
                        continue  # hard block around peer
                    if pd < peer_soft:
                        extra = self.peer_soft_cost * (1.0 - pd / peer_soft)
                ng = gc + move_cost + extra
                if ng < gscore.get(nkey, float('inf')):
                    gscore[nkey] = ng
                    came[nkey] = key
                    heapq.heappush(open_heap, (ng + heuristic(nx, ny), ng, nx, ny))
        return None

    def _find_grid_path(self, x1, y1, x2, y2,
                        peer_xy: Optional[Tuple[float, float]] = None):
        """Find A* path. peer_xy = position of the OTHER robot to avoid."""
        grid     = self.map
        inflated = self._ensure_inflated_grid()
        if grid is None or inflated is None:
            return None
        w   = int(grid.info.width)
        h   = int(grid.info.height)
        res = float(grid.info.resolution)
        sx, sy = self._world_to_grid(x1, y1)
        gx, gy = self._world_to_grid(x2, y2)
        if sx is None or gx is None:
            return None
        sx = max(0, min(w-1, sx)); sy = max(0, min(h-1, sy))
        gx = max(0, min(w-1, gx)); gy = max(0, min(h-1, gy))
        if sx == gx and sy == gy:
            return [(x1, y1), (x2, y2)]

        # Use cache only for static paths (no peer)
        cache_key = (sx, sy, gx, gy)
        if peer_xy is None and cache_key in self._static_path_cache:
            return self._static_path_cache[cache_key]

        peer_gx = peer_gy = None
        if peer_xy is not None:
            pgx, pgy = self._world_to_grid(peer_xy[0], peer_xy[1])
            if pgx is not None:
                peer_gx = max(0, min(w-1, pgx))
                peer_gy = max(0, min(h-1, pgy))

        path_grid = self._astar(sx, sy, gx, gy, w, h, res, inflated,
                                peer_gx=peer_gx, peer_gy=peer_gy)
        if not path_grid:
            if peer_xy is None:
                self._static_path_cache[cache_key] = None
            return None

        ox  = float(grid.info.origin.position.x)
        oy  = float(grid.info.origin.position.y)
        skip = max(1, int(0.10 / res))
        path = []
        for i in range(0, len(path_grid) - 1, skip):
            mx2, my2 = path_grid[i]
            path.append((ox + (mx2 + 0.5) * res, oy + (my2 + 0.5) * res))
        mx2, my2 = path_grid[-1]
        path.append((ox + (mx2 + 0.5) * res, oy + (my2 + 0.5) * res))

        if peer_xy is None:
            self._static_path_cache[cache_key] = path
        return path

    def _reachable(self, x1, y1, x2, y2, max_cells=60000) -> bool:
        inflated = self._ensure_inflated_grid()
        grid     = self.map
        if grid is None or inflated is None:
            return True
        w = int(grid.info.width); h = int(grid.info.height)
        sx, sy = self._world_to_grid(x1, y1)
        gx, gy = self._world_to_grid(x2, y2)
        if sx is None or gx is None:
            return True
        sx = max(0,min(w-1,sx)); sy = max(0,min(h-1,sy))
        gx = max(0,min(w-1,gx)); gy = max(0,min(h-1,gy))
        if sx == gx and sy == gy:
            return True
        visited = {(sx, sy)}
        queue   = deque([(sx, sy)])
        exp     = 0
        while queue and exp < max_cells:
            cx, cy = queue.popleft(); exp += 1
            for ddx, ddy in ((-1,0),(1,0),(0,-1),(0,1)):
                nx, ny = cx+ddx, cy+ddy
                if (nx, ny) in visited:                        continue
                if not (0 <= nx < w and 0 <= ny < h):         continue
                if inflated[ny * w + nx]:                      continue
                if nx == gx and ny == gy:                      return True
                visited.add((nx, ny)); queue.append((nx, ny))
        return False

    # =========================================================================
    # Waypoint chain
    # =========================================================================

    def _build_waypoint_chain(self, robot_pose, final_xy,
                              peer_xy: Optional[Tuple[float, float]] = None):
        if robot_pose is None:
            return [final_xy]
        px   = float(robot_pose.pose.position.x)
        py   = float(robot_pose.pose.position.y)
        dist = math.hypot(final_xy[0] - px, final_xy[1] - py)
        if dist <= self.waypoint_final_snap_m:
            return [final_xy]
        grid_path = self._find_grid_path(px, py, final_xy[0], final_xy[1], peer_xy=peer_xy)
        if not grid_path or len(grid_path) < 2:
            return [final_xy]
        step      = max(0.20, self.waypoint_step_m)
        waypoints = []
        accum     = 0.0
        next_mark = step
        for i in range(1, len(grid_path)):
            seg = math.hypot(grid_path[i][0] - grid_path[i-1][0],
                             grid_path[i][1] - grid_path[i-1][1])
            accum += seg
            if accum >= next_mark:
                waypoints.append(grid_path[i])
                next_mark += step
        if not waypoints or math.hypot(waypoints[-1][0] - final_xy[0],
                                       waypoints[-1][1] - final_xy[1]) > 0.10:
            waypoints.append(final_xy)
        return waypoints

    # =========================================================================
    # Publishing
    # =========================================================================

    def _chain_to_path(self, points, stamp) -> Path:
        path = Path()
        path.header.frame_id = self.frame_id
        path.header.stamp    = stamp
        for x, y in points:
            ps = PoseStamped()
            ps.header.frame_id    = self.frame_id
            ps.header.stamp       = stamp
            ps.pose.position.x    = float(x)
            ps.pose.position.y    = float(y)
            ps.pose.position.z    = 0.0
            ps.pose.orientation.w = 1.0
            path.poses.append(ps)
        return path

    def _publish_chains(self, stamp, force=False) -> bool:
        if self.final_pair is None:
            return False
        w_peer  = self._burger_xy()
        b_peer  = self._waffle_xy()
        w_chain = self._build_waypoint_chain(self.waffle_pose, self.final_pair[0], peer_xy=w_peer)
        b_chain = self._build_waypoint_chain(self.burger_pose, self.final_pair[1], peer_xy=b_peer)
        if force or self._chain_changed(w_chain, self.last_w_chain) \
                 or self._chain_changed(b_chain, self.last_b_chain):
            self._do_publish(w_chain, b_chain, stamp)
            self.last_publish_time = self.get_clock().now().nanoseconds * 1e-9
            return True
        return False

    def _do_publish(self, w_chain, b_chain, stamp) -> None:
        self.pub_waffle.publish(self._chain_to_path(w_chain, stamp))
        self.pub_burger.publish(self._chain_to_path(b_chain, stamp))
        self.last_w_chain = list(w_chain)
        self.last_b_chain = list(b_chain)
        if self.debug_markers and self.final_pair is not None:
            self._publish_debug_markers(w_chain, b_chain, stamp)

    def _chain_changed(self, new, old) -> bool:
        thr = self.change_threshold_m
        if abs(len(new) - len(old)) > 1:
            return True
        if not new or not old:
            return bool(new) != bool(old)
        indices = [0, len(new) - 1]
        if len(new) > 2:
            indices.append(len(new) // 2)
        for i in indices:
            j = min(i, len(old) - 1)
            if math.hypot(new[i][0] - old[j][0], new[i][1] - old[j][1]) > thr:
                return True
        return False

    # =========================================================================
    # Debug markers
    # =========================================================================

    def _publish_debug_markers(self, w_chain, b_chain, stamp) -> None:
        ma  = MarkerArray()
        mid = 0
        fid = self.frame_id

        def _pt(x, y, z=0.0):
            p = Point(); p.x = float(x); p.y = float(y); p.z = float(z); return p

        def _col(r, g, b, a=1.0):
            c = ColorRGBA(); c.r=float(r); c.g=float(g); c.b=float(b); c.a=float(a); return c

        def _sphere(sid, x, y, col, sz=0.12, ns='fleet_chain', z=0.05):
            m = Marker()
            m.header.frame_id = fid; m.header.stamp = stamp
            m.ns = ns; m.id = sid; m.type = Marker.SPHERE; m.action = Marker.ADD
            m.pose.position = _pt(x, y, z); m.pose.orientation.w = 1.0
            m.scale.x = m.scale.y = m.scale.z = sz; m.color = col; m.lifetime.sec = 4
            return m

        def _text(tid, x, y, txt, col, z=0.28, ns='fleet_labels'):
            m = Marker()
            m.header.frame_id = fid; m.header.stamp = stamp
            m.ns = ns; m.id = tid; m.type = Marker.TEXT_VIEW_FACING; m.action = Marker.ADD
            m.pose.position = _pt(x, y, z); m.pose.orientation.w = 1.0
            m.scale.z = 0.10; m.color = col; m.text = txt; m.lifetime.sec = 4
            return m

        def _line(lid, pts, col, width=0.022, ns='fleet_lines', z=0.02):
            m = Marker()
            m.header.frame_id = fid; m.header.stamp = stamp
            m.ns = ns; m.id = lid; m.type = Marker.LINE_STRIP; m.action = Marker.ADD
            m.pose.orientation.w = 1.0; m.scale.x = width; m.color = col; m.lifetime.sec = 4
            for x2, y2 in pts:
                m.points.append(_pt(x2, y2, z))
            return m

        BLUE     = _col(0.2, 0.5, 1.0, 0.90)
        BLUE_T   = _col(0.2, 0.5, 1.0, 0.45)
        ORANGE   = _col(1.0, 0.55, 0.0, 0.90)
        ORANGE_T = _col(1.0, 0.55, 0.0, 0.45)
        GREEN    = _col(0.1, 0.9, 0.2, 0.85)
        WHITE    = _col(1.0, 1.0, 1.0, 0.85)
        RED      = _col(1.0, 0.1, 0.1, 0.90)
        YELLOW   = _col(1.0, 0.9, 0.0, 0.90)

        # Burger state indicator
        state_col = {'MOVING': GREEN, 'WAITING': YELLOW, 'BACKING': RED}.get(
            self._burger_state, WHITE)
        bxy = self._burger_xy()
        if bxy:
            ma.markers.append(_sphere(mid, bxy[0], bxy[1], state_col, 0.20, 'fleet_state', 0.15)); mid += 1
            ma.markers.append(_text(mid, bxy[0], bxy[1], self._burger_state, state_col, 0.40, 'fleet_state')); mid += 1

        # Waffle chain
        for ci, (cx, cy) in enumerate(w_chain):
            ma.markers.append(_sphere(mid, cx, cy, BLUE, 0.10, 'fleet_chain')); mid += 1
            ma.markers.append(_text(mid, cx, cy, f'W{ci+1}', BLUE, 0.22, 'fleet_labels')); mid += 1

        # Burger chain
        for ci, (cx, cy) in enumerate(b_chain):
            ma.markers.append(_sphere(mid, cx, cy, ORANGE, 0.10, 'fleet_chain')); mid += 1
            ma.markers.append(_text(mid, cx, cy, f'B{ci+1}', ORANGE, 0.22, 'fleet_labels')); mid += 1

        # Final slots
        if self.final_pair is not None:
            fw, fb = self.final_pair
            ma.markers.append(_sphere(mid, fw[0], fw[1], BLUE_T,   0.30, 'fleet_finals', 0.02)); mid += 1
            ma.markers.append(_text(mid, fw[0], fw[1], 'W-final',  BLUE_T, 0.52, 'fleet_finals')); mid += 1
            ma.markers.append(_sphere(mid, fb[0], fb[1], ORANGE_T, 0.30, 'fleet_finals', 0.02)); mid += 1
            ma.markers.append(_text(mid, fb[0], fb[1], 'B-final',  ORANGE_T, 0.52, 'fleet_finals')); mid += 1

        # Clicked goal
        if self.clicked_goal is not None:
            gx = float(self.clicked_goal.pose.position.x)
            gy = float(self.clicked_goal.pose.position.y)
            ma.markers.append(_sphere(mid, gx, gy, GREEN, 0.15, 'fleet_clicked', 0.01)); mid += 1
            ma.markers.append(_text(mid, gx, gy, 'goal', GREEN, 0.50, 'fleet_clicked')); mid += 1

        # A* grid paths
        if self.waffle_pose is not None and self.final_pair is not None:
            wpx = float(self.waffle_pose.pose.position.x)
            wpy = float(self.waffle_pose.pose.position.y)
            gp = self._find_grid_path(wpx, wpy, self.final_pair[0][0], self.final_pair[0][1],
                                      peer_xy=self._burger_xy())
            if gp and len(gp) >= 2:
                sk  = max(1, len(gp) // 40)
                vis = [gp[i] for i in range(0, len(gp), sk)]
                if vis[-1] != gp[-1]: vis.append(gp[-1])
                ma.markers.append(_line(mid, vis, BLUE_T, 0.022, 'fleet_astar_path')); mid += 1

        if self.burger_pose is not None and self.final_pair is not None:
            bpx = float(self.burger_pose.pose.position.x)
            bpy = float(self.burger_pose.pose.position.y)
            gp = self._find_grid_path(bpx, bpy, self.final_pair[1][0], self.final_pair[1][1],
                                      peer_xy=self._waffle_xy())
            if gp and len(gp) >= 2:
                sk  = max(1, len(gp) // 40)
                vis = [gp[i] for i in range(0, len(gp), sk)]
                if vis[-1] != gp[-1]: vis.append(gp[-1])
                ma.markers.append(_line(mid, vis, ORANGE_T, 0.022, 'fleet_astar_path')); mid += 1

        # Separation between final slots
        if self.final_pair is not None:
            fw, fb = self.final_pair
            sep = math.hypot(fw[0]-fb[0], fw[1]-fb[1])
            mx2 = 0.5*(fw[0]+fb[0]); my2 = 0.5*(fw[1]+fb[1])
            ma.markers.append(_line(mid, [fw, fb], WHITE, 0.015, 'fleet_sep')); mid += 1
            ma.markers.append(_text(mid, mx2, my2, f'{sep:.2f}m', WHITE, 0.28, 'fleet_sep')); mid += 1

        # Robot distance (collision distance indicator)
        wxy = self._waffle_xy()
        bxy = self._burger_xy()
        if wxy and bxy:
            dist = math.hypot(wxy[0]-bxy[0], wxy[1]-bxy[1])
            dist_col = RED if dist < self.peer_stop_dist else (YELLOW if dist < self.peer_clear_dist else GREEN)
            mx2 = 0.5*(wxy[0]+bxy[0]); my2 = 0.5*(wxy[1]+bxy[1])
            ma.markers.append(_line(mid, [wxy, bxy], dist_col, 0.018, 'fleet_robot_dist')); mid += 1
            ma.markers.append(_text(mid, mx2, my2+0.15, f'd={dist:.2f}m', dist_col, 0.10, 'fleet_robot_dist')); mid += 1

        self.pub_markers.publish(ma)

    # =========================================================================
    # Coordinate helpers
    # =========================================================================

    def _world_to_grid(self, x, y):
        grid = self.map
        if grid is None: return None, None
        res  = float(grid.info.resolution)
        if res <= 0.0:   return None, None
        ox   = float(grid.info.origin.position.x)
        oy   = float(grid.info.origin.position.y)
        return int(math.floor((x - ox) / res)), int(math.floor((y - oy) / res))


def main():
    rclpy.init()
    node = FleetGoalDispatcher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
