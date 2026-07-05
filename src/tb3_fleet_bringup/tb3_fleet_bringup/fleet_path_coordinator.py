#!/usr/bin/env python3

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import rclpy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import OccupancyGrid, Path
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import Bool, String
from visualization_msgs.msg import Marker, MarkerArray


Point2 = Tuple[float, float]


def yaw_from_quaternion(q) -> float:
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


def quaternion_from_yaw(yaw: float):
    half = 0.5 * yaw
    return 0.0, 0.0, math.sin(half), math.cos(half)


def distance(a: Point2, b: Point2) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


@dataclass
class PathSample:
    x: float
    y: float
    tx: float
    ty: float
    along: float


@dataclass
class Conflict:
    x: float
    y: float
    leader_tangent: Point2
    follower_tangent: Point2
    path_distance: float
    leader_eta: float
    follower_eta: float


class FleetPathCoordinator(Node):
    """Central path reservation and two-robot collision coordinator.

    The current fleet transports both plans and both poses to the leader domain:
      leader:   /leader_pose, /plan
      follower: /burger_pose, /burger_plan

    When future path occupancy overlaps in space and time, this node reserves the
    conflict region for the leader, moves the robots to opposite side spots,
    lets the leader pass, and then resumes the follower. The pairwise conflict
    detector is independent of robot names and can be reused as more bridged
    robot tracks are added.
    """

    IDLE = 'IDLE'
    PAUSING = 'PAUSING'
    MOVE_ASIDE = 'MOVE_ASIDE'
    LEADER_PASS = 'LEADER_PASS'
    NO_SAFE_HOLD = 'NO_SAFE_HOLD'
    COOLDOWN = 'COOLDOWN'

    def __init__(self) -> None:
        super().__init__('fleet_path_coordinator')

        self.declare_parameter('leader_pose_topic', '/leader_pose')
        self.declare_parameter('leader_path_topic', '/plan')
        self.declare_parameter('follower_pose_topic', '/burger_pose')
        self.declare_parameter('follower_path_topic', '/burger_plan')
        self.declare_parameter('map_topic', '/map')
        self.declare_parameter('leader_user_goal_topic', '/goal_pose')
        self.declare_parameter('leader_named_goal_topic', '/waffle_goal_pose')
        self.declare_parameter('leader_coord_goal_topic', '/fleet/leader_coord_goal')
        self.declare_parameter('follower_coord_goal_topic', '/burger_goal_pose')
        self.declare_parameter('follower_command_topic', '/fleet/follow_command')
        self.declare_parameter('follower_status_topic', '/fleet/follow_enabled')
        self.declare_parameter('status_topic', '/fleet/coordination_status')
        self.declare_parameter('marker_topic', '/fleet/coordination_markers')

        self.declare_parameter('check_period_sec', 0.40)
        self.declare_parameter('path_stale_sec', 4.0)
        self.declare_parameter('lookahead_distance_m', 3.5)
        self.declare_parameter('sample_step_m', 0.12)
        self.declare_parameter('conflict_distance_m', 0.48)
        self.declare_parameter('conflict_time_window_sec', 4.0)
        self.declare_parameter('same_direction_cosine', 0.70)
        self.declare_parameter('leader_nominal_speed_mps', 0.18)
        self.declare_parameter('follower_nominal_speed_mps', 0.20)
        self.declare_parameter('minimum_robot_separation_m', 0.50)

        self.declare_parameter('side_offset_m', 0.48)
        self.declare_parameter('side_offset_max_m', 0.72)
        self.declare_parameter('side_offset_step_m', 0.08)
        self.declare_parameter('yield_backoff_m', 0.30)
        self.declare_parameter('map_clearance_m', 0.27)
        self.declare_parameter('occupied_threshold', 50)
        self.declare_parameter('allow_unknown_yield', False)
        self.declare_parameter('yield_reached_radius_m', 0.24)
        self.declare_parameter('move_aside_timeout_sec', 8.0)
        self.declare_parameter('leader_pass_distance_m', 0.70)
        self.declare_parameter('leader_pass_timeout_sec', 14.0)
        self.declare_parameter('cooldown_sec', 5.0)

        gp = self.get_parameter
        self.leader_pose_topic = str(gp('leader_pose_topic').value)
        self.leader_path_topic = str(gp('leader_path_topic').value)
        self.follower_pose_topic = str(gp('follower_pose_topic').value)
        self.follower_path_topic = str(gp('follower_path_topic').value)
        self.map_topic = str(gp('map_topic').value)
        self.leader_user_goal_topic = str(gp('leader_user_goal_topic').value)
        self.leader_named_goal_topic = str(gp('leader_named_goal_topic').value)
        self.leader_coord_goal_topic = str(gp('leader_coord_goal_topic').value)
        self.follower_coord_goal_topic = str(gp('follower_coord_goal_topic').value)
        self.follower_command_topic = str(gp('follower_command_topic').value)
        self.follower_status_topic = str(gp('follower_status_topic').value)
        self.status_topic = str(gp('status_topic').value)
        self.marker_topic = str(gp('marker_topic').value)

        self.check_period = max(0.1, float(gp('check_period_sec').value))
        self.path_stale = max(0.5, float(gp('path_stale_sec').value))
        self.lookahead = max(0.8, float(gp('lookahead_distance_m').value))
        self.sample_step = max(0.05, float(gp('sample_step_m').value))
        self.conflict_distance = max(0.25, float(gp('conflict_distance_m').value))
        self.conflict_time_window = max(0.5, float(gp('conflict_time_window_sec').value))
        self.same_direction_cosine = float(gp('same_direction_cosine').value)
        self.leader_speed = max(0.05, float(gp('leader_nominal_speed_mps').value))
        self.follower_speed = max(0.05, float(gp('follower_nominal_speed_mps').value))
        self.minimum_separation = max(
            0.35, float(gp('minimum_robot_separation_m').value)
        )

        self.side_offset = max(0.30, float(gp('side_offset_m').value))
        self.side_offset_max = max(
            self.side_offset, float(gp('side_offset_max_m').value)
        )
        self.side_offset_step = max(0.04, float(gp('side_offset_step_m').value))
        self.yield_backoff = max(0.0, float(gp('yield_backoff_m').value))
        self.map_clearance = max(0.18, float(gp('map_clearance_m').value))
        self.occupied_threshold = int(gp('occupied_threshold').value)
        self.allow_unknown_yield = bool(gp('allow_unknown_yield').value)
        self.yield_reached_radius = max(
            0.12, float(gp('yield_reached_radius_m').value)
        )
        self.move_aside_timeout = max(
            2.0, float(gp('move_aside_timeout_sec').value)
        )
        self.leader_pass_distance = max(
            0.35, float(gp('leader_pass_distance_m').value)
        )
        self.leader_pass_timeout = max(
            4.0, float(gp('leader_pass_timeout_sec').value)
        )
        self.cooldown_sec = max(1.0, float(gp('cooldown_sec').value))

        self.leader_pose: Optional[PoseStamped] = None
        self.follower_pose: Optional[PoseStamped] = None
        self.leader_path: Optional[Path] = None
        self.follower_path: Optional[Path] = None
        self.map_msg: Optional[OccupancyGrid] = None
        self.leader_path_time = -1.0e9
        self.follower_path_time = -1.0e9
        self.leader_resume_goal: Optional[PoseStamped] = None

        self.state = self.IDLE
        self.state_since = self._now()
        self.cooldown_until = -1.0e9
        self.active_conflict: Optional[Conflict] = None
        self.leader_yield: Optional[PoseStamped] = None
        self.follower_yield: Optional[PoseStamped] = None
        self.last_goal_publish = -1.0e9
        self.last_command_publish = -1.0e9
        self.last_status = ''
        self.follower_follow_enabled: Optional[bool] = None

        map_qos = QoSProfile(
            depth=1,
            history=HistoryPolicy.KEEP_LAST,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.create_subscription(PoseStamped, self.leader_pose_topic, self._leader_pose_cb, 20)
        self.create_subscription(PoseStamped, self.follower_pose_topic, self._follower_pose_cb, 20)
        self.create_subscription(Path, self.leader_path_topic, self._leader_path_cb, 10)
        self.create_subscription(Path, self.follower_path_topic, self._follower_path_cb, 10)
        self.create_subscription(OccupancyGrid, self.map_topic, self._map_cb, map_qos)
        self.create_subscription(PoseStamped, self.leader_user_goal_topic, self._user_goal_cb, 10)
        self.create_subscription(PoseStamped, self.leader_named_goal_topic, self._user_goal_cb, 10)
        coordination_qos = QoSProfile(
            depth=1,
            history=HistoryPolicy.KEEP_LAST,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.create_subscription(
            Bool,
            self.follower_status_topic,
            self._follower_status_cb,
            coordination_qos,
        )

        self.leader_goal_pub = self.create_publisher(
            PoseStamped, self.leader_coord_goal_topic, 10
        )
        self.follower_goal_pub = self.create_publisher(
            PoseStamped, self.follower_coord_goal_topic, 10
        )
        self.command_pub = self.create_publisher(
            String, self.follower_command_topic, coordination_qos
        )
        self.status_pub = self.create_publisher(
            String, self.status_topic, coordination_qos
        )
        self.marker_pub = self.create_publisher(MarkerArray, self.marker_topic, 10)
        self.create_timer(self.check_period, self._tick)

        self.get_logger().info(
            'FLEET_COORDINATOR_READY | '
            f'leader=({self.leader_pose_topic},{self.leader_path_topic}) '
            f'follower=({self.follower_pose_topic},{self.follower_path_topic}) '
            f'conflict={self.conflict_distance:.2f}m time={self.conflict_time_window:.1f}s '
            f'side={self.side_offset:.2f}-{self.side_offset_max:.2f}m'
        )

    def _now(self) -> float:
        return self.get_clock().now().nanoseconds * 1.0e-9

    def _leader_pose_cb(self, msg: PoseStamped) -> None:
        self.leader_pose = msg

    def _follower_pose_cb(self, msg: PoseStamped) -> None:
        self.follower_pose = msg

    def _leader_path_cb(self, msg: Path) -> None:
        if len(msg.poses) >= 2:
            self.leader_path = msg
            self.leader_path_time = self._now()

    def _follower_path_cb(self, msg: Path) -> None:
        if len(msg.poses) >= 2:
            self.follower_path = msg
            self.follower_path_time = self._now()

    def _map_cb(self, msg: OccupancyGrid) -> None:
        self.map_msg = msg

    def _user_goal_cb(self, msg: PoseStamped) -> None:
        self.leader_resume_goal = self._copy_goal(msg)

    def _follower_status_cb(self, msg: Bool) -> None:
        self.follower_follow_enabled = bool(msg.data)

    @staticmethod
    def _xy(msg: PoseStamped) -> Point2:
        return msg.pose.position.x, msg.pose.position.y

    @staticmethod
    def _copy_goal(msg: PoseStamped) -> PoseStamped:
        out = PoseStamped()
        out.header.frame_id = msg.header.frame_id or 'map'
        out.pose = msg.pose
        return out

    @staticmethod
    def _goal(x: float, y: float, yaw: float) -> PoseStamped:
        out = PoseStamped()
        out.header.frame_id = 'map'
        out.pose.position.x = x
        out.pose.position.y = y
        qx, qy, qz, qw = quaternion_from_yaw(yaw)
        out.pose.orientation.x = qx
        out.pose.orientation.y = qy
        out.pose.orientation.z = qz
        out.pose.orientation.w = qw
        return out

    def _path_endpoint_goal(self, path: Optional[Path]) -> Optional[PoseStamped]:
        if path is None or not path.poses:
            return None
        return self._copy_goal(path.poses[-1])

    def _samples(self, path: Path, pose: PoseStamped) -> List[PathSample]:
        raw = [(p.pose.position.x, p.pose.position.y) for p in path.poses]
        if len(raw) < 2:
            return []
        robot = self._xy(pose)
        start = min(range(len(raw)), key=lambda i: distance(raw[i], robot))
        samples: List[PathSample] = []
        along = 0.0
        next_sample = 0.0
        for i in range(start, len(raw)):
            if i > start:
                along += distance(raw[i - 1], raw[i])
            if along > self.lookahead:
                break
            if along + 1.0e-9 < next_sample and i < len(raw) - 1:
                continue
            prev_i = max(start, i - 1)
            next_i = min(len(raw) - 1, i + 1)
            tx = raw[next_i][0] - raw[prev_i][0]
            ty = raw[next_i][1] - raw[prev_i][1]
            norm = math.hypot(tx, ty)
            if norm < 1.0e-6:
                continue
            samples.append(PathSample(
                raw[i][0], raw[i][1], tx / norm, ty / norm, along
            ))
            next_sample = along + self.sample_step
        return samples

    def _find_conflict(self) -> Optional[Conflict]:
        if (
            self.leader_pose is None
            or self.follower_pose is None
            or self.leader_path is None
            or self.follower_path is None
        ):
            return None
        now = self._now()
        if (
            now - self.leader_path_time > self.path_stale
            or now - self.follower_path_time > self.path_stale
        ):
            return None

        leader_samples = self._samples(self.leader_path, self.leader_pose)
        follower_samples = self._samples(self.follower_path, self.follower_pose)
        best = None
        for a in leader_samples:
            for b in follower_samples:
                separation = math.hypot(a.x - b.x, a.y - b.y)
                if separation > self.conflict_distance:
                    continue
                direction_dot = a.tx * b.tx + a.ty * b.ty
                # Shared same-direction route is expected during following.
                if direction_dot >= self.same_direction_cosine:
                    continue
                leader_eta = a.along / self.leader_speed
                follower_eta = b.along / self.follower_speed
                eta_delta = abs(leader_eta - follower_eta)
                if eta_delta > self.conflict_time_window:
                    continue
                score = separation + 0.08 * eta_delta + 0.01 * (a.along + b.along)
                item = (score, a, b, separation, leader_eta, follower_eta)
                if best is None or item[0] < best[0]:
                    best = item
        if best is None:
            return None
        _, a, b, path_distance, leader_eta, follower_eta = best
        return Conflict(
            x=0.5 * (a.x + b.x),
            y=0.5 * (a.y + b.y),
            leader_tangent=(a.tx, a.ty),
            follower_tangent=(b.tx, b.ty),
            path_distance=path_distance,
            leader_eta=leader_eta,
            follower_eta=follower_eta,
        )

    def _world_to_map(self, x: float, y: float) -> Optional[Tuple[int, int]]:
        if self.map_msg is None:
            return None
        info = self.map_msg.info
        resolution = float(info.resolution)
        if resolution <= 0.0:
            return None
        origin = info.origin
        yaw = yaw_from_quaternion(origin.orientation)
        dx = x - origin.position.x
        dy = y - origin.position.y
        local_x = math.cos(yaw) * dx + math.sin(yaw) * dy
        local_y = -math.sin(yaw) * dx + math.cos(yaw) * dy
        mx = int(math.floor(local_x / resolution))
        my = int(math.floor(local_y / resolution))
        if mx < 0 or my < 0 or mx >= info.width or my >= info.height:
            return None
        return mx, my

    def _candidate_is_free(self, x: float, y: float) -> bool:
        if self.map_msg is None:
            return False
        center = self._world_to_map(x, y)
        if center is None:
            return False
        info = self.map_msg.info
        radius = max(1, int(math.ceil(self.map_clearance / info.resolution)))
        cx, cy = center
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                if dx * dx + dy * dy > radius * radius:
                    continue
                mx, my = cx + dx, cy + dy
                if mx < 0 or my < 0 or mx >= info.width or my >= info.height:
                    return False
                value = int(self.map_msg.data[my * info.width + mx])
                if value < 0 and not self.allow_unknown_yield:
                    return False
                if value >= self.occupied_threshold:
                    return False
        return True

    def _select_side_goals(
        self, conflict: Conflict
    ) -> Tuple[Optional[PoseStamped], Optional[PoseStamped]]:
        if self.leader_pose is None or self.follower_pose is None:
            return None, None
        tx, ty = conflict.leader_tangent
        nx, ny = -ty, tx
        ftx, fty = conflict.follower_tangent
        leader_yaw = math.atan2(ty, tx)
        follower_yaw = math.atan2(fty, ftx)
        leader_now = self._xy(self.leader_pose)
        follower_now = self._xy(self.follower_pose)
        valid = []

        offset = self.side_offset
        while offset <= self.side_offset_max + 1.0e-6:
            # Preferred assignment: leader left, follower right in the common
            # leader-path coordinate system. Swapped assignment is a fallback
            # when walls make the preferred side unavailable.
            for leader_side in (1.0, -1.0):
                follower_side = -leader_side
                lx = conflict.x - self.yield_backoff * tx + leader_side * offset * nx
                ly = conflict.y - self.yield_backoff * ty + leader_side * offset * ny
                bx = conflict.x - self.yield_backoff * ftx + follower_side * offset * nx
                by = conflict.y - self.yield_backoff * fty + follower_side * offset * ny
                if not self._candidate_is_free(lx, ly):
                    continue
                if not self._candidate_is_free(bx, by):
                    continue
                if math.hypot(lx - bx, ly - by) < self.minimum_separation + 0.18:
                    continue
                movement = distance(leader_now, (lx, ly)) + distance(follower_now, (bx, by))
                fallback_penalty = 0.35 if leader_side < 0.0 else 0.0
                valid.append((
                    movement + fallback_penalty,
                    self._goal(lx, ly, leader_yaw),
                    self._goal(bx, by, follower_yaw),
                ))
            offset += self.side_offset_step
        if not valid:
            return None, None
        _, leader_goal, follower_goal = min(valid, key=lambda item: item[0])
        return leader_goal, follower_goal

    def _publish_follow_command(self, command: str, force: bool = False) -> None:
        now = self._now()
        if not force and now - self.last_command_publish < 1.5:
            return
        msg = String()
        msg.data = command
        self.command_pub.publish(msg)
        self.last_command_publish = now

    def _publish_goal(self, pub, goal: Optional[PoseStamped]) -> None:
        if goal is None:
            return
        goal.header.stamp = self.get_clock().now().to_msg()
        pub.publish(goal)

    def _publish_side_goals(self, force: bool = False) -> None:
        now = self._now()
        if not force and now - self.last_goal_publish < 1.5:
            return
        self._publish_goal(self.leader_goal_pub, self.leader_yield)
        self._publish_goal(self.follower_goal_pub, self.follower_yield)
        self.last_goal_publish = now

    def _set_state(self, state: str, detail: str) -> None:
        self.state = state
        self.state_since = self._now()
        text = f'{state} | {detail}'
        if text != self.last_status:
            msg = String()
            msg.data = text
            self.status_pub.publish(msg)
            self.get_logger().warn(f'FLEET_COORDINATION | {text}')
            self.last_status = text

    def _start_resolution(self, conflict: Conflict) -> None:
        self.active_conflict = conflict
        self.leader_resume_goal = (
            self.leader_resume_goal or self._path_endpoint_goal(self.leader_path)
        )
        self.leader_yield, self.follower_yield = self._select_side_goals(conflict)
        self._publish_follow_command('PAUSE', force=True)
        if self.leader_yield is None or self.follower_yield is None:
            # Stop the leader at its current pose as a fail-safe. The follower
            # follow action is cancelled by PAUSE.
            if self.leader_pose is not None:
                yaw = yaw_from_quaternion(self.leader_pose.pose.orientation)
                x, y = self._xy(self.leader_pose)
                self.leader_yield = self._goal(x, y, yaw)
                self._publish_goal(self.leader_goal_pub, self.leader_yield)
            self._set_state(
                self.NO_SAFE_HOLD,
                f'conflict=({conflict.x:.2f},{conflict.y:.2f}); no free side pair',
            )
            return
        self._set_state(
            self.PAUSING,
            f'conflict=({conflict.x:.2f},{conflict.y:.2f}) '
            f'eta=({conflict.leader_eta:.1f},{conflict.follower_eta:.1f}) '
            'waiting for follower PAUSE acknowledgement',
        )

    def _goal_reached(
        self, pose: Optional[PoseStamped], goal: Optional[PoseStamped]
    ) -> bool:
        if pose is None or goal is None:
            return goal is None
        return distance(self._xy(pose), self._xy(goal)) <= self.yield_reached_radius

    def _leader_passed(self) -> bool:
        if self.leader_pose is None or self.active_conflict is None:
            return False
        tx, ty = self.active_conflict.leader_tangent
        x, y = self._xy(self.leader_pose)
        dx = x - self.active_conflict.x
        dy = y - self.active_conflict.y
        return dx * tx + dy * ty >= self.leader_pass_distance

    def _restore_leader(self) -> None:
        goal = self.leader_resume_goal or self._path_endpoint_goal(self.leader_path)
        self._publish_goal(self.leader_goal_pub, goal)

    def _clear_resolution(self) -> None:
        self._publish_follow_command('RESUME', force=True)
        self.active_conflict = None
        self.leader_yield = None
        self.follower_yield = None
        self.leader_resume_goal = None
        self.cooldown_until = self._now() + self.cooldown_sec
        self._set_state(self.COOLDOWN, f'reservation released for {self.cooldown_sec:.1f}s')

    def _publish_markers(self) -> None:
        markers = MarkerArray()
        clear = Marker()
        clear.header.frame_id = 'map'
        clear.header.stamp = self.get_clock().now().to_msg()
        clear.action = Marker.DELETEALL
        markers.markers.append(clear)
        if self.active_conflict is not None:
            marker = Marker()
            marker.header.frame_id = 'map'
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = 'fleet_conflict'
            marker.id = 0
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            marker.pose.position.x = self.active_conflict.x
            marker.pose.position.y = self.active_conflict.y
            marker.pose.orientation.w = 1.0
            marker.scale.x = marker.scale.y = 2.0 * self.conflict_distance
            marker.scale.z = 0.08
            marker.color.r = 1.0
            marker.color.g = 0.1
            marker.color.b = 0.1
            marker.color.a = 0.65
            markers.markers.append(marker)
        for marker_id, (goal, color) in enumerate((
            (self.leader_yield, (0.1, 0.4, 1.0)),
            (self.follower_yield, (1.0, 0.4, 0.1)),
        ), start=1):
            if goal is None:
                continue
            marker = Marker()
            marker.header.frame_id = 'map'
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = 'fleet_yield'
            marker.id = marker_id
            marker.type = Marker.CYLINDER
            marker.action = Marker.ADD
            marker.pose = goal.pose
            marker.scale.x = marker.scale.y = 0.28
            marker.scale.z = 0.08
            marker.color.r, marker.color.g, marker.color.b = color
            marker.color.a = 0.9
            markers.markers.append(marker)
        self.marker_pub.publish(markers)

    def _tick(self) -> None:
        now = self._now()
        if self.leader_pose is None or self.follower_pose is None:
            return

        robot_separation = distance(self._xy(self.leader_pose), self._xy(self.follower_pose))

        if self.state == self.COOLDOWN:
            if now >= self.cooldown_until:
                self._set_state(self.IDLE, 'ready')
            self._publish_markers()
            return

        if self.state == self.IDLE:
            conflict = self._find_conflict()
            if conflict is not None:
                self._start_resolution(conflict)
            elif robot_separation < self.minimum_separation:
                self._publish_follow_command('PAUSE', force=True)
                self._set_state(
                    self.NO_SAFE_HOLD,
                    f'emergency separation={robot_separation:.2f}m',
                )

        elif self.state == self.NO_SAFE_HOLD:
            self._publish_follow_command('PAUSE')
            conflict = self._find_conflict()
            if conflict is None and robot_separation >= self.minimum_separation + 0.12:
                self._clear_resolution()
            elif conflict is not None:
                leader_goal, follower_goal = self._select_side_goals(conflict)
                if leader_goal is not None and follower_goal is not None:
                    self.active_conflict = conflict
                    self.leader_yield = leader_goal
                    self.follower_yield = follower_goal
                    self._publish_side_goals(force=True)
                    self._set_state(self.MOVE_ASIDE, 'free side pair became available')

        elif self.state == self.PAUSING:
            self._publish_follow_command('PAUSE')
            pause_acknowledged = self.follower_follow_enabled is False
            if pause_acknowledged or now - self.state_since >= 1.5:
                self._publish_side_goals(force=True)
                pause_result = (
                    'PAUSE acknowledged'
                    if pause_acknowledged
                    else 'PAUSE acknowledgement timeout; goal override fallback'
                )
                self._set_state(
                    self.MOVE_ASIDE,
                    f'{pause_result}; leader=left follower=right',
                )

        elif self.state == self.MOVE_ASIDE:
            self._publish_follow_command('PAUSE')
            self._publish_side_goals()
            leader_ready = self._goal_reached(self.leader_pose, self.leader_yield)
            follower_ready = self._goal_reached(self.follower_pose, self.follower_yield)
            if (
                (leader_ready and follower_ready and now - self.state_since >= 1.0)
                or now - self.state_since >= self.move_aside_timeout
            ):
                self._restore_leader()
                self._set_state(
                    self.LEADER_PASS,
                    f'aside=({leader_ready},{follower_ready}); leader has reservation',
                )

        elif self.state == self.LEADER_PASS:
            self._publish_follow_command('PAUSE')
            if (
                self._leader_passed()
                or now - self.state_since >= self.leader_pass_timeout
            ):
                self._clear_resolution()

        self._publish_markers()


def main() -> None:
    rclpy.init()
    node = FleetPathCoordinator()
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
