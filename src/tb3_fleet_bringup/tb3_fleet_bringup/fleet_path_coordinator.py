from __future__ import annotations

import math
from typing import Optional, Tuple

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
        self.declare_parameter('follower_user_goal_topic', '/burger_user_goal')
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
        self.declare_parameter('pose_stale_sec', 1.5)
        self.declare_parameter('minimum_robot_separation_m', 0.48)

        self.declare_parameter('evasion_offset_m', 0.42)
        self.declare_parameter('evasion_offset_max_m', 0.72)
        self.declare_parameter('evasion_backoff_m', 0.12)
        self.declare_parameter('map_clearance_m', 0.18)
        self.declare_parameter('occupied_threshold', 50)
        self.declare_parameter('allow_unknown_evasion', False)
        self.declare_parameter('evasion_reached_radius_m', 0.23)
        self.declare_parameter('clearing_timeout_sec', 7.0)
        self.declare_parameter('pass_timeout_sec', 14.0)
        self.declare_parameter('cooldown_sec', 4.0)
        self.declare_parameter('motion_speed_threshold_mps', 0.055)
        self.declare_parameter('motion_trigger_distance_m', 0.90)
        self.declare_parameter('motion_prediction_horizon_sec', 2.5)
        self.declare_parameter('motion_predicted_clearance_m', 0.62)
        self.declare_parameter('risk_confirmation_cycles', 2)

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
        self.follower_user_goal_topic = str(
            get('follower_user_goal_topic').value
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
        self.pose_stale = max(0.5, float(get('pose_stale_sec').value))
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
        self.pass_timeout = max(4.0, float(get('pass_timeout_sec').value))
        self.cooldown_sec = max(1.0, float(get('cooldown_sec').value))
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
        self.risk_confirmation_cycles = max(
            1, int(get('risk_confirmation_cycles').value)
        )

        self.leader_pose: Optional[PoseStamped] = None
        self.follower_pose: Optional[PoseStamped] = None
        self.leader_pose_time = -1.0e9
        self.follower_pose_time = -1.0e9
        self.leader_path: Optional[Path] = None
        self.follower_path: Optional[Path] = None
        self.map_msg: Optional[OccupancyGrid] = None
        self.follower_follow_enabled: Optional[bool] = None
        self.leader_velocity: Point2 = (0.0, 0.0)
        self.follower_velocity: Point2 = (0.0, 0.0)
        self.leader_motion_sample: Optional[Tuple[float, Point2]] = None
        self.follower_motion_sample: Optional[Tuple[float, Point2]] = None
        self.collision_warning = False
        self.risk_observation_count = 0

        self.state = self.IDLE
        self.state_since = self._now()
        self.cooldown_until = -1.0e9
        self.last_status = ''
        self.last_command_time = -1.0e9
        self.last_evasion_publish = -1.0e9
        self.last_pose_warning_log = -1.0e9
        self.last_priority = self.FOLLOWER
        self.leader_goal_subscription_count = 0
        self.follower_goal_subscription_count = 0

        self.priority_robot: Optional[str] = None
        self.leader_evasion_goal: Optional[PoseStamped] = None
        self.follower_evasion_goal: Optional[PoseStamped] = None
        self.leader_resume_goal: Optional[PoseStamped] = None
        self.follower_resume_goal: Optional[PoseStamped] = None
        self.leader_user_goal: Optional[PoseStamped] = None
        self.follower_user_goal: Optional[PoseStamped] = None
        self.follower_was_following = True
        self.follower_desired_following = True
        self.pose_hold_active = False
        self.leader_hold_sent = False
        self.follower_hold_sent = False
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
            PoseStamped,
            self.follower_user_goal_topic,
            self._follower_goal_cb,
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
        self.leader_pose_time = self._now()
        self.leader_velocity, self.leader_motion_sample = self._update_velocity(
            self.leader_velocity,
            self.leader_motion_sample,
            self._xy(message),
        )

    def _follower_pose_cb(self, message: PoseStamped) -> None:
        self.follower_pose = message
        self.follower_pose_time = self._now()
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
        alpha = 0.30
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

    def _follower_path_cb(self, message: Path) -> None:
        if len(message.poses) >= 2:
            self.follower_path = message

    def _map_cb(self, message: OccupancyGrid) -> None:
        self.map_msg = message

    def _leader_goal_cb(self, message: PoseStamped) -> None:
        self.leader_user_goal = self._copy_goal(message)
        if self.state in (self.IDLE, self.COOLDOWN):
            self._publish_goal(self.leader_goal_pub, self.leader_user_goal)
        else:
            self.leader_resume_goal = self._copy_goal(message)

    def _follower_goal_cb(self, message: PoseStamped) -> None:
        self.follower_user_goal = self._copy_goal(message)
        self.follower_desired_following = False
        if self.state in (self.IDLE, self.COOLDOWN):
            self.follower_follow_enabled = False
            self._publish_follow_command('PAUSE', force=True)
            self._publish_goal(self.follower_goal_pub, self.follower_user_goal)
        else:
            self.follower_was_following = False
            self.follower_resume_goal = self._copy_goal(message)

    def _follower_status_cb(self, message: Bool) -> None:
        self.follower_follow_enabled = bool(message.data)
        if self.state == self.IDLE and not self.pose_hold_active:
            self.follower_desired_following = bool(message.data)
            # PAUSE and a named Burger goal cross DDS as different topics, so
            # ordering is not guaranteed. Reassert the independent goal only
            # after the follower confirms that follow-mode actions are stopped.
            if (
                not message.data
                and self.follower_user_goal is not None
            ):
                self._publish_goal(
                    self.follower_goal_pub,
                    self.follower_user_goal,
                )

    def _path_endpoint(self, path: Optional[Path]) -> Optional[PoseStamped]:
        if path is None or not path.poses:
            return None
        return self._copy_goal(path.poses[-1])

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
        closing_speed = -(
            relative_position[0] * relative_velocity[0]
            + relative_position[1] * relative_velocity[1]
        ) / max(current_distance, 1.0e-6)

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

        emergency = current_distance < self.minimum_separation + 0.08
        near_motion = current_distance < max(
            self.minimum_separation + 0.20, 0.72
        )
        approaching = (
            current_distance < self.motion_trigger_distance
            and closing_speed > 0.02
        )
        predicted = (
            closest_time > 0.05
            and closest_distance < self.motion_predicted_clearance
        )
        return (
            emergency or near_motion or approaching or predicted,
            closest_distance,
            closest_time,
        )

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

    def _republish_goals_to_new_consumers(self) -> None:
        leader_count = self.leader_goal_pub.get_subscription_count()
        follower_count = self.follower_goal_pub.get_subscription_count()
        if (
            leader_count > 0
            and self.leader_goal_subscription_count == 0
            and self.state in (self.IDLE, self.COOLDOWN)
        ):
            self._publish_goal(self.leader_goal_pub, self.leader_user_goal)
        if (
            follower_count > 0
            and self.follower_goal_subscription_count == 0
            and self.state in (self.IDLE, self.COOLDOWN)
            and not self.follower_desired_following
        ):
            self._publish_goal(self.follower_goal_pub, self.follower_user_goal)
        self.leader_goal_subscription_count = leader_count
        self.follower_goal_subscription_count = follower_count

    def _hold_for_missing_pose(self) -> None:
        self._publish_follow_command('PAUSE')
        if not self.pose_hold_active:
            self.pose_hold_active = True
            self.leader_hold_sent = False
            self.follower_hold_sent = False
        # A goal at the last known pose preempts outstanding Nav2 motion while
        # retaining the real user goal in leader/follower_user_goal.
        if self.leader_pose is not None and not self.leader_hold_sent:
            self._publish_goal(self.leader_goal_pub, self.leader_pose)
            self.leader_hold_sent = True
        if self.follower_pose is not None and not self.follower_hold_sent:
            self._publish_goal(self.follower_goal_pub, self.follower_pose)
            self.follower_hold_sent = True

    def _resume_after_pose_hold(self) -> None:
        if not self.pose_hold_active:
            return
        self.pose_hold_active = False
        self.leader_hold_sent = False
        self.follower_hold_sent = False
        if self.state in (self.IDLE, self.COOLDOWN):
            self._publish_goal(self.leader_goal_pub, self.leader_user_goal)
            if self.follower_desired_following:
                self._publish_follow_command('RESUME', force=True)
            else:
                self._publish_goal(
                    self.follower_goal_pub,
                    self.follower_user_goal,
                )
            return

        if self.state in (self.CLEARING, self.BLOCKED):
            self._publish_evasion_goals(force=True)
            return

        if self.state == self.PRIORITY_PASS:
            self._publish_evasion_goals(force=True)
            if self.priority_robot == self.LEADER:
                self._publish_goal(
                    self.leader_goal_pub,
                    self.leader_resume_goal,
                )
            elif self.priority_robot == self.FOLLOWER:
                self._publish_goal(
                    self.follower_goal_pub,
                    self.follower_resume_goal,
                )

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
            self._copy_goal(self.leader_user_goal)
            if self.leader_user_goal is not None
            else self._path_endpoint(self.leader_path)
        )
        self.follower_resume_goal = (
            self._copy_goal(self.follower_user_goal)
            if self.follower_user_goal is not None
            else self._path_endpoint(self.follower_path)
        )
        self.follower_was_following = (
            self.follower_desired_following
        )

    def _begin_escape(self, reason: str = 'proximity') -> None:
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
        separation = distance(
            self._xy(self.leader_pose),
            self._xy(self.follower_pose),
        )
        motion_risk, _, _ = self._motion_risk()
        return separation >= self.release_separation and not motion_risk

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
        warning_message = Bool()
        warning_message.data = warning
        self.warning_pub.publish(warning_message)
        self.collision_warning = warning

        if self.leader_pose is None or self.follower_pose is None:
            return
        poses = PoseArray()
        poses.header.frame_id = 'map'
        poses.header.stamp = self.get_clock().now().to_msg()
        # Stable ordering: index 0 is Waffle/leader, index 1 is Burger/follower.
        poses.poses = [self.leader_pose.pose, self.follower_pose.pose]
        self.robot_poses_pub.publish(poses)

        if warning:
            hazard = PoseStamped()
            hazard.header.frame_id = 'map'
            hazard.header.stamp = poses.header.stamp
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
            self._hold_for_missing_pose()
            self._publish_safety_state(True)
            return
        now = self._now()
        if (
            now - self.leader_pose_time > self.pose_stale
            or now - self.follower_pose_time > self.pose_stale
        ):
            self._hold_for_missing_pose()
            self._publish_safety_state(True)
            if now - self.last_pose_warning_log >= 5.0:
                self.get_logger().error(
                    'FLEET_SAFETY_POSE_STALE | motion held until both '
                    'robot poses are fresh'
                )
                self.last_pose_warning_log = now
            return

        self._resume_after_pose_hold()
        separation = distance(
            self._xy(self.leader_pose),
            self._xy(self.follower_pose),
        )
        self._republish_goals_to_new_consumers()

        if self.state == self.COOLDOWN:
            motion_risk, closest_distance, closest_time = self._motion_risk()
            if motion_risk:
                self.risk_observation_count += 1
            else:
                self.risk_observation_count = 0
            emergency = separation < self.minimum_separation
            if emergency or (
                motion_risk
                and self.risk_observation_count
                >= self.risk_confirmation_cycles
            ):
                self.risk_observation_count = 0
                self._begin_escape(
                    'risk returned during cooldown '
                    f'closest={closest_distance:.2f}m '
                    f'in={closest_time:.1f}s'
                )
            elif now >= self.cooldown_until:
                self._set_state(self.IDLE, 'ready')
            else:
                self._publish_safety_state(False)
                self._publish_markers()
                return

        if self.state == self.IDLE:
            motion_risk, closest_distance, closest_time = self._motion_risk()
            if motion_risk:
                self.risk_observation_count += 1
            else:
                self.risk_observation_count = 0
            emergency = separation < self.minimum_separation
            if emergency or (
                motion_risk
                and self.risk_observation_count
                >= self.risk_confirmation_cycles
            ):
                self.risk_observation_count = 0
                self._begin_escape(
                    'motion risk '
                    f'closest={closest_distance:.2f}m '
                    f'in={closest_time:.1f}s'
                )

        elif self.state == self.CLEARING:
            self._publish_follow_command('PAUSE')
            self._publish_evasion_goals()
            leader_ready = self._goal_reached(
                self.leader_pose, self.leader_evasion_goal
            )
            follower_ready = self._goal_reached(
                self.follower_pose, self.follower_evasion_goal
            )
            if leader_ready and follower_ready:
                self._release_priority()
            elif now - self.state_since >= self.clearing_timeout:
                motion_risk, _, _ = self._motion_risk()
                if (
                    separation >= self.minimum_separation + 0.20
                    and not motion_risk
                ):
                    self._release_priority()
                else:
                    self._set_state(
                        self.BLOCKED,
                        'clearing timeout while separation is still unsafe',
                    )

        elif self.state == self.PRIORITY_PASS:
            self._publish_follow_command('PAUSE')
            if self._priority_passed():
                self._restore_after_maneuver()
            elif now - self.state_since >= self.pass_timeout:
                self._set_state(
                    self.BLOCKED,
                    'priority pass timeout while separation is still unsafe',
                )

        elif self.state == self.BLOCKED:
            self._publish_follow_command('PAUSE')
            motion_risk, _, _ = self._motion_risk()
            if (
                separation >= self.minimum_separation + 0.20
                and not motion_risk
            ):
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
