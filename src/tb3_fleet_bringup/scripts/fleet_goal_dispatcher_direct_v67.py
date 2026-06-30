#!/usr/bin/env python3

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import rclpy
from rclpy.node import Node
from rclpy.exceptions import ParameterAlreadyDeclaredException
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import OccupancyGrid


def _safe_declare(node: Node, name: str, default):
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
class Candidate:
    x: float
    y: float
    score: float
    clearance: float
    occ: int
    unknown: int
    angle: float


class FleetGoalDispatcher(Node):
    """StarCraft-style group-move goal splitter for two robots.

    Input: RViz PoseStamped group goal (/fleet_goal_pose) plus compatibility aliases such as /goal_pose.
    Output: two PoseStamped Nav2 goals around the clicked point:
      - /waffle_goal_pose
      - /burger_goal_pose

    The dispatcher does not proxy Nav2 actions directly. Existing pose_to_nav2_action
    nodes convert those PoseStamped topics into NavigateToPose goals in each domain.
    """

    def __init__(self) -> None:
        super().__init__('fleet_goal_dispatcher')
        _safe_declare(self, 'use_sim_time', True)
        self.input_goal_topic = self._abs(str(_safe_declare(self, 'input_goal_topic', '/fleet_goal_pose')))
        alias_raw = str(_safe_declare(self, 'input_alias_topics', '/goal_pose')).strip()
        self.input_alias_topics = [self._abs(t.strip()) for t in alias_raw.split(',') if t.strip()]
        self.waffle_goal_topic = self._abs(str(_safe_declare(self, 'waffle_goal_topic', '/waffle_goal_pose')))
        self.burger_goal_topic = self._abs(str(_safe_declare(self, 'burger_goal_topic', '/burger_goal_pose')))
        self.map_topic = self._abs(str(_safe_declare(self, 'map_topic', '/map')))
        self.waffle_pose_topic = self._abs(str(_safe_declare(self, 'waffle_pose_topic', '/leader_pose')))
        self.burger_pose_topic = self._abs(str(_safe_declare(self, 'burger_pose_topic', '/burger_pose')))
        self.frame_id = str(_safe_declare(self, 'frame_id', 'map'))
        self.formation_separation_m = float(_safe_declare(self, 'formation_separation_m', 0.75))
        self.min_pair_distance_m = float(_safe_declare(self, 'min_pair_distance_m', 0.55))
        self.search_rings = int(_safe_declare(self, 'search_rings', 3))
        self.search_angles = int(_safe_declare(self, 'search_angles', 16))
        self.clearance_check_radius_m = float(_safe_declare(self, 'clearance_check_radius_m', 0.65))
        self.occupied_threshold = int(_safe_declare(self, 'occupied_threshold', 45))
        self.unknown_penalty = float(_safe_declare(self, 'unknown_penalty', 0.25))
        self.occupied_penalty = float(_safe_declare(self, 'occupied_penalty', 1000.0))
        self.assignment_prefers_shorter_total_distance = bool(_safe_declare(self, 'assignment_prefers_shorter_total_distance', True))
        self.republish_count = int(_safe_declare(self, 'republish_count', 180))
        self.republish_period_sec = float(_safe_declare(self, 'republish_period_sec', 0.50))

        self.map: Optional[OccupancyGrid] = None
        self.waffle_pose: Optional[PoseStamped] = None
        self.burger_pose: Optional[PoseStamped] = None
        self.pending_pair: Optional[Tuple[PoseStamped, PoseStamped]] = None
        self.pending_left = 0

        self.pub_waffle = self.create_publisher(PoseStamped, self.waffle_goal_topic, 10)
        self.pub_burger = self.create_publisher(PoseStamped, self.burger_goal_topic, 10)
        self.create_subscription(PoseStamped, self.input_goal_topic, lambda msg: self._on_fleet_goal(msg, self.input_goal_topic), 10)
        self._alias_subs = []
        seen_alias = {self.input_goal_topic}
        for topic in self.input_alias_topics:
            if topic in seen_alias:
                continue
            seen_alias.add(topic)
            self._alias_subs.append(self.create_subscription(PoseStamped, topic, lambda msg, src=topic: self._on_fleet_goal(msg, src), 10))
        self.create_subscription(OccupancyGrid, self.map_topic, self._on_map, 10)
        self.create_subscription(PoseStamped, self.waffle_pose_topic, self._on_waffle_pose, 10)
        self.create_subscription(PoseStamped, self.burger_pose_topic, self._on_burger_pose, 10)
        self.timer = self.create_timer(max(0.05, self.republish_period_sec), self._republish_pending)

        self.goal_count = 0
        self.get_logger().info(
            'V67_FLEET_GOAL_DISPATCHER_READY | '
            f'in={self.input_goal_topic} aliases={self.input_alias_topics} waffle_out={self.waffle_goal_topic} burger_out={self.burger_goal_topic} '
            f'map={self.map_topic} sep={self.formation_separation_m:.2f}m frame={self.frame_id}'
        )

    @staticmethod
    def _abs(topic: str) -> str:
        topic = topic.strip()
        return topic if topic.startswith('/') else '/' + topic

    def _on_map(self, msg: OccupancyGrid) -> None:
        self.map = msg

    def _on_waffle_pose(self, msg: PoseStamped) -> None:
        self.waffle_pose = msg

    def _on_burger_pose(self, msg: PoseStamped) -> None:
        self.burger_pose = msg

    def _on_fleet_goal(self, msg: PoseStamped, source_topic: str = '/fleet_goal_pose') -> None:
        self.goal_count += 1
        x = float(msg.pose.position.x)
        y = float(msg.pose.position.y)
        yaw = _quat_to_yaw(msg.pose.orientation)
        if not math.isfinite(yaw):
            yaw = 0.0

        waffle_xy, burger_xy, detail = self._compute_pair(x, y, yaw)
        if self.assignment_prefers_shorter_total_distance:
            waffle_xy, burger_xy, detail = self._assign_by_distance(waffle_xy, burger_xy, detail)

        wmsg = self._make_goal(msg, waffle_xy[0], waffle_xy[1])
        bmsg = self._make_goal(msg, burger_xy[0], burger_xy[1])
        self.pending_pair = (wmsg, bmsg)
        self.pending_left = max(1, self.republish_count)
        self._publish_pair(wmsg, bmsg)
        self.pending_left -= 1

        self.get_logger().info(
            f'V67_FLEET_GOAL_DISPATCH | n={self.goal_count} src={source_topic} clicked=({x:.3f},{y:.3f}) '
            f'waffle=({wmsg.pose.position.x:.3f},{wmsg.pose.position.y:.3f}) '
            f'burger=({bmsg.pose.position.x:.3f},{bmsg.pose.position.y:.3f}) {detail}'
        )

    def _make_goal(self, src: PoseStamped, x: float, y: float) -> PoseStamped:
        out = PoseStamped()
        out.header = src.header
        out.header.frame_id = out.header.frame_id or self.frame_id
        out.header.stamp = self.get_clock().now().to_msg()
        out.pose = src.pose
        out.pose.position.x = float(x)
        out.pose.position.y = float(y)
        out.pose.position.z = 0.0
        return out

    def _republish_pending(self) -> None:
        if self.pending_pair is None or self.pending_left <= 0:
            return
        self._publish_pair(self.pending_pair[0], self.pending_pair[1])
        self.pending_left -= 1

    def _publish_pair(self, wmsg: PoseStamped, bmsg: PoseStamped) -> None:
        # Stamp again on repeated publishes so Nav2 receives fresh goals.
        now = self.get_clock().now().to_msg()
        wmsg.header.stamp = now
        bmsg.header.stamp = now
        self.pub_waffle.publish(wmsg)
        self.pub_burger.publish(bmsg)

    def _assign_by_distance(self, a: Tuple[float, float], b: Tuple[float, float], detail: str):
        if self.waffle_pose is None or self.burger_pose is None:
            return a, b, detail + ' assign=fixed_no_pose'
        wx = self.waffle_pose.pose.position.x
        wy = self.waffle_pose.pose.position.y
        bx = self.burger_pose.pose.position.x
        by = self.burger_pose.pose.position.y
        keep = math.hypot(wx - a[0], wy - a[1]) + math.hypot(bx - b[0], by - b[1])
        swap = math.hypot(wx - b[0], wy - b[1]) + math.hypot(bx - a[0], by - a[1])
        if swap + 0.05 < keep:
            return b, a, detail + f' assign=swap keep={keep:.2f} swap={swap:.2f}'
        return a, b, detail + f' assign=keep keep={keep:.2f} swap={swap:.2f}'

    def _compute_pair(self, x: float, y: float, yaw: float):
        if self.map is None or self.map.info.width <= 0 or self.map.info.height <= 0:
            a, b = self._fallback_pair(x, y, yaw)
            return a, b, 'mode=no_map_lateral_slots'

        candidates = self._make_candidates(x, y, yaw)
        if len(candidates) < 2:
            a, b = self._fallback_pair(x, y, yaw)
            return a, b, 'mode=no_candidates_lateral_slots'

        best_pair = None
        best_score = -1e18
        for i, c1 in enumerate(candidates):
            for c2 in candidates[i + 1:]:
                dist = math.hypot(c1.x - c2.x, c1.y - c2.y)
                if dist < self.min_pair_distance_m:
                    continue
                center_err = math.hypot(((c1.x + c2.x) * 0.5) - x, ((c1.y + c2.y) * 0.5) - y)
                sep_err = abs(dist - self.formation_separation_m)
                score = c1.score + c2.score - 1.3 * center_err - 0.8 * sep_err
                if score > best_score:
                    best_score = score
                    best_pair = (c1, c2, dist, center_err, sep_err)

        if best_pair is None:
            a, b = self._fallback_pair(x, y, yaw)
            return a, b, 'mode=no_pair_lateral_slots'

        c1, c2, dist, center_err, sep_err = best_pair
        detail = (
            f'mode=map_safe_pair score={best_score:.2f} pair_dist={dist:.2f} '
            f'center_err={center_err:.2f} sep_err={sep_err:.2f} '
            f'clear=({c1.clearance:.2f},{c2.clearance:.2f}) occ=({c1.occ},{c2.occ}) unk=({c1.unknown},{c2.unknown})'
        )
        return (c1.x, c1.y), (c2.x, c2.y), detail

    def _fallback_pair(self, x: float, y: float, yaw: float):
        # Goal yaw's left/right lateral vector. If yaw is 0, slots become +/-Y.
        r = self.formation_separation_m * 0.5
        lx = -math.sin(yaw)
        ly = math.cos(yaw)
        return (x + lx * r, y + ly * r), (x - lx * r, y - ly * r)

    def _make_candidates(self, x: float, y: float, yaw: float) -> List[Candidate]:
        out: List[Candidate] = []
        # Include center-near lateral slots and several rings around the clicked goal.
        radii = [max(0.25, self.formation_separation_m * 0.5)]
        for k in range(1, max(1, self.search_rings)):
            radii.append(max(0.25, self.formation_separation_m * (0.35 + 0.25 * k)))
        angles = max(8, self.search_angles)
        for radius in radii:
            for j in range(angles):
                ang = yaw + (2.0 * math.pi * j / angles)
                cx = x + radius * math.cos(ang)
                cy = y + radius * math.sin(ang)
                ev = self._evaluate_candidate(cx, cy, x, y, ang)
                if ev is not None:
                    out.append(ev)
        out.sort(key=lambda c: c.score, reverse=True)
        return out[:64]

    def _evaluate_candidate(self, cx: float, cy: float, gx: float, gy: float, ang: float) -> Optional[Candidate]:
        grid = self.map
        if grid is None:
            return None
        res = float(grid.info.resolution)
        if res <= 0.0:
            return None
        mx, my = self._world_to_map(cx, cy)
        if mx is None:
            return None
        w = int(grid.info.width)
        h = int(grid.info.height)
        if mx < 0 or my < 0 or mx >= w or my >= h:
            return None
        idx = my * w + mx
        val = int(grid.data[idx])
        if val >= self.occupied_threshold:
            return None

        rad_cells = max(1, int(self.clearance_check_radius_m / res))
        occ = 0
        unk = 0
        min_occ_d = self.clearance_check_radius_m + res
        for dy in range(-rad_cells, rad_cells + 1):
            yy = my + dy
            if yy < 0 or yy >= h:
                continue
            for dx in range(-rad_cells, rad_cells + 1):
                xx = mx + dx
                if xx < 0 or xx >= w:
                    continue
                d = math.hypot(dx * res, dy * res)
                if d > self.clearance_check_radius_m:
                    continue
                v = int(grid.data[yy * w + xx])
                if v < 0:
                    unk += 1
                elif v >= self.occupied_threshold:
                    occ += 1
                    if d < min_occ_d:
                        min_occ_d = d
        if occ > 0 and min_occ_d < 0.20:
            return None
        if min_occ_d > self.clearance_check_radius_m:
            min_occ_d = self.clearance_check_radius_m
        goal_dist = math.hypot(cx - gx, cy - gy)
        score = (2.0 * min_occ_d) - (0.8 * goal_dist) - (self.unknown_penalty * unk / 50.0) - (self.occupied_penalty * occ / 500.0)
        return Candidate(cx, cy, score, min_occ_d, occ, unk, ang)

    def _world_to_map(self, x: float, y: float):
        grid = self.map
        if grid is None:
            return None, None
        ox = float(grid.info.origin.position.x)
        oy = float(grid.info.origin.position.y)
        res = float(grid.info.resolution)
        if res <= 0.0:
            return None, None
        mx = int(math.floor((x - ox) / res))
        my = int(math.floor((y - oy) / res))
        return mx, my


def main() -> None:
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
