from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import rclpy
from geometry_msgs.msg import PoseArray, PoseStamped
from nav_msgs.msg import OccupancyGrid, Path
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)
from std_msgs.msg import Bool, String
from visualization_msgs.msg import Marker, MarkerArray


Point2 = Tuple[float, float]


def distance(first: Point2, second: Point2) -> float:
    return math.hypot(first[0] - second[0], first[1] - second[1])


def normalize(vector: Point2) -> Point2:
    length = math.hypot(vector[0], vector[1])
    if length < 1.0e-9:
        return 1.0, 0.0
    return vector[0] / length, vector[1] / length


def rotate(vector: Point2, angle: float) -> Point2:
    cosine = math.cos(angle)
    sine = math.sin(angle)
    return (
        cosine * vector[0] - sine * vector[1],
        sine * vector[0] + cosine * vector[1],
    )


def yaw_from_quaternion(quaternion) -> float:
    siny_cosp = 2.0 * (
        quaternion.w * quaternion.z + quaternion.x * quaternion.y
    )
    cosy_cosp = 1.0 - 2.0 * (
        quaternion.y * quaternion.y + quaternion.z * quaternion.z
    )
    return math.atan2(siny_cosp, cosy_cosp)


def quaternion_from_yaw(yaw: float):
    half = 0.5 * yaw
    return 0.0, 0.0, math.sin(half), math.cos(half)


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
    """Coordinate reciprocal avoidance for the Waffle and Burger.

    Both robots first receive short, opposite evasive goals. The robot with
    right-of-way then resumes its original Nav2 goal while the other remains
    clear. Right-of-way is dynamic when Burger is independently navigating and
    remains with Waffle only while Burger is in follow mode.
    """

    IDLE = 'IDLE'
    CLEARING = 'CLEARING'
    PRIORITY_PASS = 'PRIORITY_PASS'
    ESCAPING = 'ESCAPING'
    BLOCKED = 'BLOCKED'
    COOLDOWN = 'COOLDOWN'

    LEADER = 'leader'
    FOLLOWER = 'follower'

    def __init__(self) -> None:
        super().__init__('fleet_path_coordinator')

        self.declare_parameter('leader_pose_topic', '/leader_pose')
        self.declare_parameter('leader_path_topic', '/plan')
        self.declare_parameter('follower_pose_topic', '/burger_pose')
        self.declare_parameter('follower_path_topic', '/burger_plan')
        self.declare_parameter('map_topic', '/map')
        self.declare_parameter('leader_user_goal_topic', '/goal_pose')
        self.declare_parameter('leader_named_goal_topic', '/waffle_goal_pose')
        self.declare_parameter(
            'leader_coord_goal_topic', '/fleet/leader_coord_goal'
        )
        self.declare_parameter(
            'follower_coord_goal_topic', '/burger_goal_pose'
        )
        self.declare_parameter(
            'follower_command_topic', '/fleet/follow_command'
        )
        self.declare_parameter(
            'follower_status_topic', '/fleet/follow_enabled'
        )
        self.declare_parameter('status_topic', '/fleet/coordination_status')
        self.declare_parameter('marker_topic', '/fleet/coordination_markers')

        self.declare_parameter('check_period_sec', 0.30)
        self.declare_parameter('path_stale_sec', 4.0)
        self.declare_parameter('lookahead_distance_m', 3.5)
        self.declare_parameter('sample_step_m', 0.10)
        self.declare_parameter('conflict_distance_m', 0.50)
        self.declare_parameter('conflict_time_window_sec', 4.0)
        self.declare_parameter('same_direction_cosine', 0.72)
        self.declare_parameter('leader_nominal_speed_mps', 0.18)
        self.declare_parameter('follower_nominal_speed_mps', 0.18)
        self.declare_parameter('minimum_robot_separation_m', 0.48)

        self.declare_parameter('evasion_offset_m', 0.42)
        self.declare_parameter('evasion_offset_max_m', 0.72)
        self.declare_parameter('evasion_backoff_m', 0.12)
        self.declare_parameter('map_clearance_m', 0.18)
        self.declare_parameter('occupied_threshold', 50)
        self.declare_parameter('allow_unknown_evasion', False)
        self.declare_parameter('evasion_reached_radius_m', 0.23)
        self.declare_parameter('clearing_timeout_sec', 7.0)
        self.declare_parameter('pass_distance_m', 0.65)
        self.declare_parameter('pass_timeout_sec', 14.0)
        self.declare_parameter('escape_timeout_sec', 7.0)
        self.declare_parameter('cooldown_sec', 4.0)
        self.declare_parameter('priority_eta_margin_sec', 0.8)
        self.declare_parameter('motion_speed_threshold_mps', 0.04)
        self.declare_parameter('motion_trigger_distance_m', 0.90)
        self.declare_parameter('motion_prediction_horizon_sec', 2.5)
        self.declare_parameter('motion_predicted_clearance_m', 0.62)

        get = self.get_parameter
        self.leader_pose_topic = str(get('leader_pose_topic').value)
        self.leader_path_topic = str(get('leader_path_topic').value)
        self.follower_pose_topic = str(get('follower_pose_topic').value)
        self.follower_path_topic = str(get('follower_path_topic').value)
        self.map_topic = str(get('map_topic').value)
        self.leader_user_goal_topic = str(get('leader_user_goal_topic').value)
        self.leader_named_goal_topic = str(
            get('leader_named_goal_topic').value
        )
        self.leader_coord_goal_topic = str(
            get('leader_coord_goal_topic').value
        )
        self.follower_coord_goal_topic = str(
            get('follower_coord_goal_topic').value
        )
        self.follower_command_topic = str(
            get('follower_command_topic').value
        )
        self.follower_status_topic = str(
            get('follower_status_topic').value
        )
        self.status_topic = str(get('status_topic').value)
        self.marker_topic = str(get('marker_topic').value)

        self.check_period = max(0.1, float(get('check_period_sec').value))
        self.path_stale = max(0.5, float(get('path_stale_sec').value))
        self.lookahead = max(0.8, float(get('lookahead_distance_m').value))
        self.sample_step = max(0.05, float(get('sample_step_m').value))
        self.conflict_distance = max(
            0.30, float(get('conflict_distance_m').value)
        )
        self.conflict_time_window = max(
            0.5, float(get('conflict_time_window_sec').value)
        )
        self.same_direction_cosine = float(
            get('same_direction_cosine').value
        )
        self.leader_speed = max(
            0.05, float(get('leader_nominal_speed_mps').value)
        )
        self.follower_speed = max(
            0.05, float(get('follower_nominal_speed_mps').value)
        )
        self.minimum_separation = max(
            0.35, float(get('minimum_robot_separation_m').value)
        )
        self.evasion_offset = max(
            0.28, float(get('evasion_offset_m').value)
        )
        self.evasion_offset_max = max(
            self.evasion_offset, float(get('evasion_offset_max_m').value)
        )
        self.evasion_backoff = max(
            0.0, float(get('evasion_backoff_m').value)
        )
        self.map_clearance = max(
            0.12, float(get('map_clearance_m').value)
        )
        self.occupied_threshold = int(get('occupied_threshold').value)
        self.allow_unknown_evasion = bool(
            get('allow_unknown_evasion').value
        )
        self.evasion_reached_radius = max(
            0.12, float(get('evasion_reached_radius_m').value)
        )
        self.clearing_timeout = max(
            2.0, float(get('clearing_timeout_sec').value)
        )
        self.pass_distance = max(0.35, float(get('pass_distance_m').value))
        self.pass_timeout = max(4.0, float(get('pass_timeout_sec').value))
        self.escape_timeout = max(
            2.0, float(get('escape_timeout_sec').value)
        )
        self.cooldown_sec = max(1.0, float(get('cooldown_sec').value))
        self.priority_eta_margin = max(
            0.0, float(get('priority_eta_margin_sec').value)
        )
        self.motion_speed_threshold = max(
            0.01, float(get('motion_speed_threshold_mps').value)
        )
        self.motion_trigger_distance = max(
            self.minimum_separation + 0.10,
            float(get('motion_trigger_distance_m').value),
        )
        self.motion_prediction_horizon = max(
            0.5, float(get('motion_prediction_horizon_sec').value)
        )
        self.motion_predicted_clearance = max(
            self.minimum_separation,
            float(get('motion_predicted_clearance_m').value),
        )

        self.leader_pose: Optional[PoseStamped] = None
        self.follower_pose: Optional[PoseStamped] = None
        self.leader_path: Optional[Path] = None
        self.follower_path: Optional[Path] = None
        self.map_msg: Optional[OccupancyGrid] = None
        self.leader_path_time = -1.0e9
        self.follower_path_time = -1.0e9
        self.follower_follow_enabled: Optional[bool] = None
        self.leader_velocity: Point2 = (0.0, 0.0)
        self.follower_velocity: Point2 = (0.0, 0.0)
        self.leader_motion_sample: Optional[Tuple[float, Point2]] = None
        self.follower_motion_sample: Optional[Tuple[float, Point2]] = None
        self.collision_warning = False

        self.state = self.IDLE
        self.state_since = self._now()
        self.cooldown_until = -1.0e9
        self.last_status = ''
        self.last_command_time = -1.0e9
        self.last_evasion_publish = -1.0e9
        self.last_priority = self.FOLLOWER

        self.active_conflict: Optional[Conflict] = None
        self.priority_robot: Optional[str] = None
        self.leader_evasion_goal: Optional[PoseStamped] = None
        self.follower_evasion_goal: Optional[PoseStamped] = None
        self.leader_resume_goal: Optional[PoseStamped] = None
        self.follower_resume_goal: Optional[PoseStamped] = None
        self.follower_was_following = True
        self.release_separation = self.motion_trigger_distance + 0.25

        map_qos = QoSProfile(
            depth=1,
            history=HistoryPolicy.KEEP_LAST,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        coordination_qos = QoSProfile(
            depth=1,
            history=HistoryPolicy.KEEP_LAST,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.create_subscription(
            PoseStamped, self.leader_pose_topic, self._leader_pose_cb, 20
        )
        self.create_subscription(
            PoseStamped, self.follower_pose_topic, self._follower_pose_cb, 20
        )
        self.create_subscription(
            Path, self.leader_path_topic, self._leader_path_cb, 10
        )
        self.create_subscription(
            Path, self.follower_path_topic, self._follower_path_cb, 10
        )
        self.create_subscription(
            OccupancyGrid, self.map_topic, self._map_cb, map_qos
        )
        self.create_subscription(
            PoseStamped,
            self.leader_user_goal_topic,
            self._leader_goal_cb,
            10,
        )
        self.create_subscription(
            PoseStamped,
            self.leader_named_goal_topic,
            self._leader_goal_cb,
            10,
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
        self.marker_pub = self.create_publisher(
            MarkerArray, self.marker_topic, 10
        )
        self.robot_poses_pub = self.create_publisher(
            PoseArray, '/fleet/robot_poses', 10
        )
        self.warning_pub = self.create_publisher(
            Bool, '/fleet/collision_warning', coordination_qos
        )
        self.hazard_pose_pub = self.create_publisher(
            PoseStamped, '/fleet/hazard_pose', 10
        )
        self.create_timer(self.check_period, self._tick)

        self.get_logger().info(
            'RECIPROCAL_FLEET_COORDINATOR_READY | '
            f'leader_pose={self.leader_pose_topic} '
            f'follower_pose={self.follower_pose_topic} '
            f'motion_trigger={self.motion_trigger_distance:.2f}m '
            f'prediction={self.motion_prediction_horizon:.1f}s '
            f'evasion={self.evasion_offset:.2f}-{self.evasion_offset_max:.2f}m'
        )

    def _now(self) -> float:
        return self.get_clock().now().nanoseconds * 1.0e-9

    @staticmethod
    def _xy(message: PoseStamped) -> Point2:
        return message.pose.position.x, message.pose.position.y

    @staticmethod
    def _copy_goal(message: PoseStamped) -> PoseStamped:
        goal = PoseStamped()
        goal.header.frame_id = message.header.frame_id or 'map'
        goal.pose = message.pose
        return goal

    @staticmethod
    def _goal(x: float, y: float, yaw: float) -> PoseStamped:
        goal = PoseStamped()
        goal.header.frame_id = 'map'
        goal.pose.position.x = x
        goal.pose.position.y = y
        qx, qy, qz, qw = quaternion_from_yaw(yaw)
        goal.pose.orientation.x = qx
        goal.pose.orientation.y = qy
        goal.pose.orientation.z = qz
        goal.pose.orientation.w = qw
        return goal

    def _leader_pose_cb(self, message: PoseStamped) -> None:
        self.leader_pose = message
        self.leader_velocity, self.leader_motion_sample = self._update_velocity(
            self.leader_velocity,
            self.leader_motion_sample,
            self._xy(message),
        )

    def _follower_pose_cb(self, message: PoseStamped) -> None:
        self.follower_pose = message
        (
            self.follower_velocity,
            self.follower_motion_sample,
        ) = self._update_velocity(
            self.follower_velocity,
            self.follower_motion_sample,
            self._xy(message),
        )

    def _update_velocity(
        self,
        previous_velocity: Point2,
        previous_sample: Optional[Tuple[float, Point2]],
        position: Point2,
    ) -> Tuple[Point2, Tuple[float, Point2]]:
        now = self._now()
        if previous_sample is None:
            return previous_velocity, (now, position)
        elapsed = now - previous_sample[0]
        if elapsed <= 0.02:
            return previous_velocity, (now, position)
        if elapsed > 1.5:
            return (0.0, 0.0), (now, position)
        measured = (
            (position[0] - previous_sample[1][0]) / elapsed,
            (position[1] - previous_sample[1][1]) / elapsed,
        )
        # Low-pass filtering rejects AMCL/SLAM pose jitter while retaining
        # teleoperation motion quickly enough for a 2.5 second prediction.
        alpha = 0.45
        velocity = (
            alpha * measured[0] + (1.0 - alpha) * previous_velocity[0],
            alpha * measured[1] + (1.0 - alpha) * previous_velocity[1],
        )
        if math.hypot(*velocity) < 0.015:
            velocity = (0.0, 0.0)
        return velocity, (now, position)

    def _leader_path_cb(self, message: Path) -> None:
        if len(message.poses) >= 2:
            self.leader_path = message
            self.leader_path_time = self._now()

    def _follower_path_cb(self, message: Path) -> None:
        if len(message.poses) >= 2:
            self.follower_path = message
            self.follower_path_time = self._now()

    def _map_cb(self, message: OccupancyGrid) -> None:
        self.map_msg = message

    def _leader_goal_cb(self, message: PoseStamped) -> None:
        if self.state in (self.IDLE, self.COOLDOWN):
            self.leader_resume_goal = self._copy_goal(message)

    def _follower_status_cb(self, message: Bool) -> None:
        self.follower_follow_enabled = bool(message.data)

    def _path_endpoint(self, path: Optional[Path]) -> Optional[PoseStamped]:
        if path is None or not path.poses:
            return None
        return self._copy_goal(path.poses[-1])

    def _samples(self, path: Path, pose: PoseStamped) -> List[PathSample]:
        raw = [(item.pose.position.x, item.pose.position.y) for item in path.poses]
        if len(raw) < 2:
            return []
        robot = self._xy(pose)
        start = min(
            range(len(raw)),
            key=lambda index: distance(raw[index], robot),
        )
        samples: List[PathSample] = []
        along = 0.0
        next_sample = 0.0
        for index in range(start, len(raw)):
            if index > start:
                along += distance(raw[index - 1], raw[index])
            if along > self.lookahead:
                break
            if along + 1.0e-9 < next_sample and index < len(raw) - 1:
                continue
            previous = max(start, index - 1)
            following = min(len(raw) - 1, index + 1)
            tangent = normalize((
                raw[following][0] - raw[previous][0],
                raw[following][1] - raw[previous][1],
            ))
            samples.append(PathSample(
                raw[index][0],
                raw[index][1],
                tangent[0],
                tangent[1],
                along,
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

        best = None
        leader_samples = self._samples(self.leader_path, self.leader_pose)
        follower_samples = self._samples(
            self.follower_path, self.follower_pose
        )
        for leader in leader_samples:
            for follower in follower_samples:
                spatial = math.hypot(
                    leader.x - follower.x,
                    leader.y - follower.y,
                )
                if spatial > self.conflict_distance:
                    continue
                direction_dot = (
                    leader.tx * follower.tx + leader.ty * follower.ty
                )
                if direction_dot >= self.same_direction_cosine:
                    continue
                leader_eta = leader.along / self.leader_speed
                follower_eta = follower.along / self.follower_speed
                eta_delta = abs(leader_eta - follower_eta)
                if eta_delta > self.conflict_time_window:
                    continue
                score = (
                    spatial
                    + 0.08 * eta_delta
                    + 0.01 * (leader.along + follower.along)
                )
                item = (
                    score,
                    leader,
                    follower,
                    spatial,
                    leader_eta,
                    follower_eta,
                )
                if best is None or score < best[0]:
                    best = item
        if best is None:
            return None
        _, leader, follower, spatial, leader_eta, follower_eta = best
        return Conflict(
            x=0.5 * (leader.x + follower.x),
            y=0.5 * (leader.y + follower.y),
            leader_tangent=(leader.tx, leader.ty),
            follower_tangent=(follower.tx, follower.ty),
            path_distance=spatial,
            leader_eta=leader_eta,
            follower_eta=follower_eta,
        )

    def _world_to_map(self, x: float, y: float) -> Optional[Tuple[int, int]]:
        if self.map_msg is None:
            return None
        info = self.map_msg.info
        if info.resolution <= 0.0:
            return None
        origin_yaw = yaw_from_quaternion(info.origin.orientation)
        dx = x - info.origin.position.x
        dy = y - info.origin.position.y
        local_x = math.cos(origin_yaw) * dx + math.sin(origin_yaw) * dy
        local_y = -math.sin(origin_yaw) * dx + math.cos(origin_yaw) * dy
        mx = int(math.floor(local_x / info.resolution))
        my = int(math.floor(local_y / info.resolution))
        if mx < 0 or my < 0 or mx >= info.width or my >= info.height:
            return None
        return mx, my

    def _candidate_is_free(
        self,
        x: float,
        y: float,
        *,
        relaxed: bool = False,
    ) -> bool:
        # Before the first map arrives, Nav2 remains the final collision check.
        if self.map_msg is None:
            return True
        center = self._world_to_map(x, y)
        if center is None:
            return False
        info = self.map_msg.info
        clearance = self.map_clearance * (0.65 if relaxed else 1.0)
        radius = max(1, int(math.ceil(clearance / info.resolution)))
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                if dx * dx + dy * dy > radius * radius:
                    continue
                mx = center[0] + dx
                my = center[1] + dy
                if mx < 0 or my < 0 or mx >= info.width or my >= info.height:
                    return False
                value = int(self.map_msg.data[my * info.width + mx])
                if value >= self.occupied_threshold:
                    return False
                if (
                    value < 0
                    and not relaxed
                    and not self.allow_unknown_evasion
                ):
                    return False
        return True

    @staticmethod
    def _swept_separation(
        first_start: Point2,
        first_goal: Point2,
        second_start: Point2,
        second_goal: Point2,
    ) -> float:
        return min(
            distance(
                (
                    first_start[0] + ratio * (first_goal[0] - first_start[0]),
                    first_start[1] + ratio * (first_goal[1] - first_start[1]),
                ),
                (
                    second_start[0] + ratio * (second_goal[0] - second_start[0]),
                    second_start[1] + ratio * (second_goal[1] - second_start[1]),
                ),
            )
            for ratio in (0.0, 0.25, 0.5, 0.75, 1.0)
        )

    def _reciprocal_goals(
        self,
        leader_tangent: Point2,
        follower_tangent: Point2,
    ) -> Tuple[Optional[PoseStamped], Optional[PoseStamped]]:
        if self.leader_pose is None or self.follower_pose is None:
            return None, None
        leader_now = self._xy(self.leader_pose)
        follower_now = self._xy(self.follower_pose)
        leader_tangent = normalize(leader_tangent)
        follower_tangent = normalize(follower_tangent)
        leader_right = (leader_tangent[1], -leader_tangent[0])
        follower_right = (follower_tangent[1], -follower_tangent[0])

        side_pairs = (
            (1.0, 1.0, 0.0, 'both-right'),
            (-1.0, -1.0, 0.20, 'both-left'),
            (1.0, -1.0, 0.35, 'leader-right/follower-left'),
            (-1.0, 1.0, 0.35, 'leader-left/follower-right'),
        )
        candidates = []
        offset = self.evasion_offset
        while offset <= self.evasion_offset_max + 1.0e-6:
            for leader_side, follower_side, penalty, label in side_pairs:
                leader_goal = (
                    leader_now[0]
                    + leader_side * offset * leader_right[0]
                    - self.evasion_backoff * leader_tangent[0],
                    leader_now[1]
                    + leader_side * offset * leader_right[1]
                    - self.evasion_backoff * leader_tangent[1],
                )
                follower_goal = (
                    follower_now[0]
                    + follower_side * offset * follower_right[0]
                    - self.evasion_backoff * follower_tangent[0],
                    follower_now[1]
                    + follower_side * offset * follower_right[1]
                    - self.evasion_backoff * follower_tangent[1],
                )
                final_separation = distance(leader_goal, follower_goal)
                swept = self._swept_separation(
                    leader_now,
                    leader_goal,
                    follower_now,
                    follower_goal,
                )
                if final_separation < self.minimum_separation + 0.10:
                    continue
                if final_separation < (
                    distance(leader_now, follower_now) + 0.15
                ):
                    continue
                if swept < min(
                    distance(leader_now, follower_now),
                    self.minimum_separation * 0.75,
                ):
                    continue
                strict = (
                    self._candidate_is_free(*leader_goal)
                    and self._candidate_is_free(*follower_goal)
                )
                relaxed = (
                    self._candidate_is_free(*leader_goal, relaxed=True)
                    and self._candidate_is_free(*follower_goal, relaxed=True)
                )
                if not strict and not relaxed:
                    continue
                relaxed_penalty = 0.6 if not strict else 0.0
                score = (
                    penalty
                    + relaxed_penalty
                    + 0.2 * offset
                    - 0.4 * final_separation
                )
                candidates.append((
                    score,
                    label,
                    self._goal(
                        *leader_goal,
                        math.atan2(leader_tangent[1], leader_tangent[0]),
                    ),
                    self._goal(
                        *follower_goal,
                        math.atan2(follower_tangent[1], follower_tangent[0]),
                    ),
                ))
            offset += 0.10
        if not candidates:
            return self._escape_goals()
        _, label, leader_goal, follower_goal = min(
            candidates, key=lambda item: item[0]
        )
        self.get_logger().info(f'EVASION_PAIR_SELECTED | {label}')
        return leader_goal, follower_goal

    def _escape_goals(
        self,
    ) -> Tuple[Optional[PoseStamped], Optional[PoseStamped]]:
        if self.leader_pose is None or self.follower_pose is None:
            return None, None
        leader_now = self._xy(self.leader_pose)
        follower_now = self._xy(self.follower_pose)
        away = normalize((
            leader_now[0] - follower_now[0],
            leader_now[1] - follower_now[1],
        ))
        leader_yaw = yaw_from_quaternion(self.leader_pose.pose.orientation)
        follower_yaw = yaw_from_quaternion(
            self.follower_pose.pose.orientation
        )
        candidates = []
        for distance_m in (0.38, 0.48, 0.60, 0.72):
            for angle_degrees in (0, 30, -30, 60, -60, 90, -90):
                direction = rotate(away, math.radians(angle_degrees))
                leader_goal = (
                    leader_now[0] + distance_m * direction[0],
                    leader_now[1] + distance_m * direction[1],
                )
                follower_goal = (
                    follower_now[0] - distance_m * direction[0],
                    follower_now[1] - distance_m * direction[1],
                )
                strict = (
                    self._candidate_is_free(*leader_goal)
                    and self._candidate_is_free(*follower_goal)
                )
                relaxed = (
                    self._candidate_is_free(*leader_goal, relaxed=True)
                    and self._candidate_is_free(*follower_goal, relaxed=True)
                )
                if not strict and not relaxed:
                    continue
                final_separation = distance(leader_goal, follower_goal)
                score = (
                    (0.5 if not strict else 0.0)
                    + abs(angle_degrees) / 180.0
                    - final_separation
                )
                candidates.append((
                    score,
                    self._goal(*leader_goal, leader_yaw),
                    self._goal(*follower_goal, follower_yaw),
                ))
        if not candidates:
            return None, None
        _, leader_goal, follower_goal = min(
            candidates, key=lambda item: item[0]
        )
        return leader_goal, follower_goal

    def _choose_priority(self, conflict: Conflict) -> str:
        # Follow mode is intentionally leader-centric. Independent Burger
        # navigation is symmetric and uses ETA with alternating tie-breaks.
        if self.follower_was_following:
            priority = self.LEADER
        elif (
            conflict.leader_eta + self.priority_eta_margin
            < conflict.follower_eta
        ):
            priority = self.LEADER
        elif (
            conflict.follower_eta + self.priority_eta_margin
            < conflict.leader_eta
        ):
            priority = self.FOLLOWER
        else:
            priority = (
                self.FOLLOWER
                if self.last_priority == self.LEADER
                else self.LEADER
            )
        self.last_priority = priority
        return priority

    def _choose_motion_priority(self) -> str:
        if self.follower_was_following:
            priority = self.LEADER
        else:
            leader_speed = math.hypot(*self._effective_velocity(
                self.leader_velocity, self.leader_motion_sample
            ))
            follower_speed = math.hypot(*self._effective_velocity(
                self.follower_velocity, self.follower_motion_sample
            ))
            if leader_speed > follower_speed + 0.03:
                priority = self.LEADER
            elif follower_speed > leader_speed + 0.03:
                priority = self.FOLLOWER
            else:
                priority = (
                    self.FOLLOWER
                    if self.last_priority == self.LEADER
                    else self.LEADER
                )
        self.last_priority = priority
        return priority

    def _motion_tangent(
        self,
        pose: PoseStamped,
        velocity: Point2,
        sample: Optional[Tuple[float, Point2]],
    ) -> Point2:
        effective = self._effective_velocity(velocity, sample)
        if math.hypot(*effective) >= self.motion_speed_threshold:
            return normalize(effective)
        yaw = yaw_from_quaternion(pose.pose.orientation)
        return math.cos(yaw), math.sin(yaw)

    def _effective_velocity(
        self,
        velocity: Point2,
        sample: Optional[Tuple[float, Point2]],
    ) -> Point2:
        if sample is None or self._now() - sample[0] > 0.8:
            return 0.0, 0.0
        return velocity

    def _motion_risk(self) -> Tuple[bool, float, float]:
        """Return risk, predicted closest distance and time to that distance."""
        if self.leader_pose is None or self.follower_pose is None:
            return False, float('inf'), 0.0
        leader_velocity = self._effective_velocity(
            self.leader_velocity, self.leader_motion_sample
        )
        follower_velocity = self._effective_velocity(
            self.follower_velocity, self.follower_motion_sample
        )
        leader_speed = math.hypot(*leader_velocity)
        follower_speed = math.hypot(*follower_velocity)
        if (
            leader_speed < self.motion_speed_threshold
            and follower_speed < self.motion_speed_threshold
        ):
            return False, distance(
                self._xy(self.leader_pose),
                self._xy(self.follower_pose),
            ), 0.0

        relative_position = (
            self.follower_pose.pose.position.x
            - self.leader_pose.pose.position.x,
            self.follower_pose.pose.position.y
            - self.leader_pose.pose.position.y,
        )
        relative_velocity = (
            follower_velocity[0] - leader_velocity[0],
            follower_velocity[1] - leader_velocity[1],
        )
        current_distance = math.hypot(*relative_position)
        relative_speed_squared = (
            relative_velocity[0] ** 2 + relative_velocity[1] ** 2
        )
        if relative_speed_squared < 1.0e-6:
            closest_time = 0.0
        else:
            closest_time = max(
                0.0,
                min(
                    self.motion_prediction_horizon,
                    -(
                        relative_position[0] * relative_velocity[0]
                        + relative_position[1] * relative_velocity[1]
                    )
                    / relative_speed_squared,
                ),
            )
        closest_vector = (
            relative_position[0] + closest_time * relative_velocity[0],
            relative_position[1] + closest_time * relative_velocity[1],
        )
        closest_distance = math.hypot(*closest_vector)

        # Normal leader-following at the configured 0.70 m spacing must not
        # look like a teleoperation hazard.
        same_direction_following = False
        if (
            self.follower_follow_enabled is not False
            and leader_speed >= self.motion_speed_threshold
            and follower_speed >= self.motion_speed_threshold
        ):
            direction_dot = (
                leader_velocity[0] * follower_velocity[0]
                + leader_velocity[1] * follower_velocity[1]
            ) / max(leader_speed * follower_speed, 1.0e-6)
            same_direction_following = (
                direction_dot > 0.75
                and math.hypot(*relative_velocity) < 0.10
                and current_distance >= self.minimum_separation
            )
        if same_direction_following:
            return False, closest_distance, closest_time

        immediate = current_distance < self.motion_trigger_distance
        predicted = (
            closest_time > 0.05
            and closest_distance < self.motion_predicted_clearance
        )
        return immediate or predicted, closest_distance, closest_time

    def _publish_follow_command(
        self,
        command: str,
        *,
        force: bool = False,
    ) -> None:
        now = self._now()
        if not force and now - self.last_command_time < 1.5:
            return
        message = String()
        message.data = command
        self.command_pub.publish(message)
        self.last_command_time = now

    def _publish_goal(self, publisher, goal: Optional[PoseStamped]) -> None:
        if goal is None:
            return
        goal.header.stamp = rclpy.time.Time().to_msg()
        publisher.publish(goal)

    def _publish_evasion_goals(self, *, force: bool = False) -> None:
        now = self._now()
        if not force and now - self.last_evasion_publish < 2.5:
            return
        self._publish_goal(self.leader_goal_pub, self.leader_evasion_goal)
        self._publish_goal(self.follower_goal_pub, self.follower_evasion_goal)
        self.last_evasion_publish = now

    def _set_state(self, state: str, detail: str) -> None:
        self.state = state
        self.state_since = self._now()
        text = f'{state} | {detail}'
        if text == self.last_status:
            return
        message = String()
        message.data = text
        self.status_pub.publish(message)
        self.get_logger().warning(f'FLEET_COORDINATION | {text}')
        self.last_status = text

    def _capture_resume_goals(self) -> None:
        self.leader_resume_goal = (
            self.leader_resume_goal
            or self._path_endpoint(self.leader_path)
        )
        self.follower_resume_goal = self._path_endpoint(self.follower_path)
        self.follower_was_following = (
            self.follower_follow_enabled is not False
        )

    def _begin_conflict(self, conflict: Conflict) -> None:
        self.active_conflict = conflict
        self._capture_resume_goals()
        self.priority_robot = self._choose_priority(conflict)
        (
            self.leader_evasion_goal,
            self.follower_evasion_goal,
        ) = self._reciprocal_goals(
            conflict.leader_tangent,
            conflict.follower_tangent,
        )
        self._publish_follow_command('PAUSE', force=True)
        if (
            self.leader_evasion_goal is None
            or self.follower_evasion_goal is None
        ):
            self._set_state(
                self.BLOCKED,
                'no unoccupied reciprocal maneuver is available',
            )
            return
        self._publish_evasion_goals(force=True)
        self._set_state(
            self.CLEARING,
            f'priority={self.priority_robot} '
            f'conflict=({conflict.x:.2f},{conflict.y:.2f}) '
            f'eta=({conflict.leader_eta:.1f},{conflict.follower_eta:.1f})',
        )

    def _begin_escape(self, reason: str = 'proximity') -> None:
        self.active_conflict = None
        self._capture_resume_goals()
        self.priority_robot = self._choose_motion_priority()
        current_separation = distance(
            self._xy(self.leader_pose),
            self._xy(self.follower_pose),
        )
        self.release_separation = max(
            self.motion_trigger_distance + 0.20,
            current_separation + 0.30,
        )
        leader_tangent = self._motion_tangent(
            self.leader_pose,
            self.leader_velocity,
            self.leader_motion_sample,
        )
        follower_tangent = self._motion_tangent(
            self.follower_pose,
            self.follower_velocity,
            self.follower_motion_sample,
        )
        (
            self.leader_evasion_goal,
            self.follower_evasion_goal,
        ) = self._reciprocal_goals(leader_tangent, follower_tangent)
        self._publish_follow_command('PAUSE', force=True)
        if (
            self.leader_evasion_goal is None
            or self.follower_evasion_goal is None
        ):
            self._set_state(
                self.BLOCKED,
                'robots are too close and no free escape pair exists',
            )
            return
        self._publish_evasion_goals(force=True)
        self._set_state(
            self.CLEARING,
            f'{reason}; priority={self.priority_robot}; '
            'position-based reciprocal evasion',
        )

    def _goal_reached(
        self,
        pose: Optional[PoseStamped],
        goal: Optional[PoseStamped],
    ) -> bool:
        return (
            pose is not None
            and goal is not None
            and distance(self._xy(pose), self._xy(goal))
            <= self.evasion_reached_radius
        )

    def _priority_passed(self) -> bool:
        if self.leader_pose is None or self.follower_pose is None:
            return False
        if self.active_conflict is None:
            separation = distance(
                self._xy(self.leader_pose),
                self._xy(self.follower_pose),
            )
            motion_risk, _, _ = self._motion_risk()
            return separation >= self.release_separation and not motion_risk
        if self.priority_robot == self.LEADER:
            pose = self.leader_pose
            tangent = self.active_conflict.leader_tangent
        else:
            pose = self.follower_pose
            tangent = self.active_conflict.follower_tangent
        if pose is None:
            return False
        position = self._xy(pose)
        displacement = (
            position[0] - self.active_conflict.x,
            position[1] - self.active_conflict.y,
        )
        return (
            displacement[0] * tangent[0]
            + displacement[1] * tangent[1]
            >= self.pass_distance
        )

    def _release_priority(self) -> None:
        if self.priority_robot == self.LEADER:
            self._publish_goal(
                self.leader_goal_pub,
                self.leader_resume_goal,
            )
        else:
            self._publish_goal(
                self.follower_goal_pub,
                self.follower_resume_goal,
            )
        self._set_state(
            self.PRIORITY_PASS,
            f'{self.priority_robot} has right-of-way',
        )

    def _restore_after_maneuver(self) -> None:
        if self.priority_robot == self.LEADER:
            if self.follower_was_following:
                self._publish_follow_command('RESUME', force=True)
            else:
                self._publish_goal(
                    self.follower_goal_pub,
                    self.follower_resume_goal,
                )
        elif self.priority_robot == self.FOLLOWER:
            self._publish_goal(
                self.leader_goal_pub,
                self.leader_resume_goal,
            )
        else:
            self._publish_goal(
                self.leader_goal_pub,
                self.leader_resume_goal,
            )
            if self.follower_was_following:
                self._publish_follow_command('RESUME', force=True)
            else:
                self._publish_goal(
                    self.follower_goal_pub,
                    self.follower_resume_goal,
                )

        self.active_conflict = None
        self.priority_robot = None
        self.leader_evasion_goal = None
        self.follower_evasion_goal = None
        self.leader_resume_goal = None
        self.follower_resume_goal = None
        self.cooldown_until = self._now() + self.cooldown_sec
        self._set_state(
            self.COOLDOWN,
            f'reciprocal maneuver complete; {self.cooldown_sec:.1f}s cooldown',
        )

    def _publish_markers(self) -> None:
        message = MarkerArray()
        clear = Marker()
        clear.header.frame_id = 'map'
        clear.header.stamp = self.get_clock().now().to_msg()
        clear.action = Marker.DELETEALL
        message.markers.append(clear)

        if self.active_conflict is not None:
            conflict = Marker()
            conflict.header.frame_id = 'map'
            conflict.header.stamp = self.get_clock().now().to_msg()
            conflict.ns = 'fleet_conflict'
            conflict.id = 0
            conflict.type = Marker.SPHERE
            conflict.action = Marker.ADD
            conflict.pose.position.x = self.active_conflict.x
            conflict.pose.position.y = self.active_conflict.y
            conflict.pose.orientation.w = 1.0
            conflict.scale.x = conflict.scale.y = (
                2.0 * self.conflict_distance
            )
            conflict.scale.z = 0.08
            conflict.color.r = 1.0
            conflict.color.g = 0.1
            conflict.color.b = 0.1
            conflict.color.a = 0.55
            message.markers.append(conflict)

        goals = (
            (self.leader_evasion_goal, (0.1, 0.4, 1.0)),
            (self.follower_evasion_goal, (1.0, 0.4, 0.1)),
        )
        for marker_id, (goal, color) in enumerate(goals, start=1):
            if goal is None:
                continue
            marker = Marker()
            marker.header.frame_id = 'map'
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = 'fleet_evasion'
            marker.id = marker_id
            marker.type = Marker.CYLINDER
            marker.action = Marker.ADD
            marker.pose = goal.pose
            marker.scale.x = marker.scale.y = 0.30
            marker.scale.z = 0.08
            marker.color.r, marker.color.g, marker.color.b = color
            marker.color.a = 0.9
            message.markers.append(marker)
        self.marker_pub.publish(message)

    def _publish_safety_state(self, warning: bool) -> None:
        if self.leader_pose is None or self.follower_pose is None:
            return
        poses = PoseArray()
        poses.header.frame_id = 'map'
        poses.header.stamp = self.get_clock().now().to_msg()
        # Stable ordering: index 0 is Waffle/leader, index 1 is Burger/follower.
        poses.poses = [self.leader_pose.pose, self.follower_pose.pose]
        self.robot_poses_pub.publish(poses)

        warning_message = Bool()
        warning_message.data = warning
        self.warning_pub.publish(warning_message)
        self.collision_warning = warning

        if warning:
            hazard = PoseStamped()
            hazard.header.frame_id = 'map'
            hazard.header.stamp = poses.header.stamp
            if self.active_conflict is not None:
                hazard.pose.position.x = self.active_conflict.x
                hazard.pose.position.y = self.active_conflict.y
            else:
                hazard.pose.position.x = 0.5 * (
                    self.leader_pose.pose.position.x
                    + self.follower_pose.pose.position.x
                )
                hazard.pose.position.y = 0.5 * (
                    self.leader_pose.pose.position.y
                    + self.follower_pose.pose.position.y
                )
            hazard.pose.orientation.w = 1.0
            self.hazard_pose_pub.publish(hazard)

    def _tick(self) -> None:
        if self.leader_pose is None or self.follower_pose is None:
            return
        now = self._now()
        separation = distance(
            self._xy(self.leader_pose),
            self._xy(self.follower_pose),
        )

        if self.state == self.COOLDOWN:
            if now >= self.cooldown_until:
                self._set_state(self.IDLE, 'ready')
            self._publish_safety_state(False)
            self._publish_markers()
            return

        if self.state == self.IDLE:
            motion_risk, closest_distance, closest_time = self._motion_risk()
            if motion_risk:
                self._begin_escape(
                    'motion risk '
                    f'closest={closest_distance:.2f}m '
                    f'in={closest_time:.1f}s'
                )
            elif separation < self.minimum_separation:
                self._begin_escape('minimum separation violation')

        elif self.state == self.CLEARING:
            self._publish_follow_command('PAUSE')
            self._publish_evasion_goals()
            leader_ready = self._goal_reached(
                self.leader_pose, self.leader_evasion_goal
            )
            follower_ready = self._goal_reached(
                self.follower_pose, self.follower_evasion_goal
            )
            if (
                (leader_ready and follower_ready)
                or now - self.state_since >= self.clearing_timeout
            ):
                self._release_priority()

        elif self.state == self.PRIORITY_PASS:
            self._publish_follow_command('PAUSE')
            if (
                self._priority_passed()
                or now - self.state_since >= self.pass_timeout
            ):
                self._restore_after_maneuver()

        elif self.state == self.ESCAPING:
            self._publish_follow_command('PAUSE')
            self._publish_evasion_goals()
            both_ready = (
                self._goal_reached(
                    self.leader_pose, self.leader_evasion_goal
                )
                and self._goal_reached(
                    self.follower_pose, self.follower_evasion_goal
                )
            )
            if (
                both_ready
                or separation >= self.minimum_separation + 0.18
                or now - self.state_since >= self.escape_timeout
            ):
                self._restore_after_maneuver()

        elif self.state == self.BLOCKED:
            self._publish_follow_command('PAUSE')
            if separation >= self.minimum_separation + 0.18:
                self._restore_after_maneuver()
            else:
                leader_goal, follower_goal = self._escape_goals()
                if leader_goal is not None and follower_goal is not None:
                    self.leader_evasion_goal = leader_goal
                    self.follower_evasion_goal = follower_goal
                    if self.priority_robot is None:
                        self.priority_robot = self._choose_motion_priority()
                    self._publish_evasion_goals(force=True)
                    self._set_state(
                        self.CLEARING,
                        'a free reciprocal escape pair became available',
                    )

        warning = self.state not in (self.IDLE, self.COOLDOWN)
        self._publish_safety_state(warning)
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
