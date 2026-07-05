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
    """Coordinate priority-preserving avoidance for Waffle and Burger.

    The priority robot keeps its original Nav2 goal. Only the yielding robot
    receives a short move-aside goal and later resumes its saved destination.
    A sole moving robot has priority; Waffle has priority when both move.
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
        self.declare_parameter('guard_pose_topic', '/guard_pose')
        self.declare_parameter('guard_coord_goal_topic', '/guard_goal_pose')

        self.declare_parameter('check_period_sec', 0.30)
        self.declare_parameter('pose_stale_sec', 1.5)
        self.declare_parameter('minimum_robot_separation_m', 0.48)

        self.declare_parameter('evasion_offset_m', 0.42)
        self.declare_parameter('evasion_offset_max_m', 0.72)
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
        self.guard_pose_topic = str(get('guard_pose_topic').value)
        self.guard_coord_goal_topic = str(
            get('guard_coord_goal_topic').value
        )

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

        # Guard: a third robot that never leads or follows. It only reports
        # its pose and receives a short yield goal whenever the leader or
        # follower gets close, then returns to where it was standing. It
        # runs its own small state machine independent of the leader/
        # follower one above so an unused guard topic is a total no-op.
        self.guard_pose: Optional[PoseStamped] = None
        self.guard_pose_time = -1.0e9
        self.guard_velocity: Point2 = (0.0, 0.0)
        self.guard_motion_sample: Optional[Tuple[float, Point2]] = None
        self.guard_state = self.IDLE
        self.guard_state_since = self._now()
        self.guard_cooldown_until = -1.0e9
        self.guard_resume_pose: Optional[PoseStamped] = None
        self.guard_evasion_goal: Optional[PoseStamped] = None
        self.guard_release_separation = self.motion_trigger_distance + 0.25
        self.guard_last_evasion_publish = -1.0e9
        self.guard_last_status = ''

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
        self.create_subscription(
            PoseStamped, self.guard_pose_topic, self._guard_pose_cb, 20
        )

        self.guard_goal_pub = self.create_publisher(
            PoseStamped, self.guard_coord_goal_topic, 10
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
            'PRIORITY_FLEET_COORDINATOR_READY | '
            f'leader_pose={self.leader_pose_topic} '
            f'follower_pose={self.follower_pose_topic} '
            f'guard_pose={self.guard_pose_topic} '
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

    def _guard_pose_cb(self, message: PoseStamped) -> None:
        self.guard_pose = message
        self.guard_pose_time = self._now()
        self.guard_velocity, self.guard_motion_sample = self._update_velocity(
            self.guard_velocity,
            self.guard_motion_sample,
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
            if self.priority_robot == self.LEADER:
                self._publish_goal(
                    self.leader_goal_pub,
                    self.leader_user_goal,
                )

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
            if self.priority_robot == self.FOLLOWER:
                self._publish_goal(
                    self.follower_goal_pub,
                    self.follower_user_goal,
                )

    def _follower_status_cb(self, message: Bool) -> None:
        self.follower_follow_enabled = bool(message.data)
        if (
            not message.data
            and self.priority_robot == self.LEADER
            and self.state in (self.CLEARING, self.PRIORITY_PASS, self.BLOCKED)
            and self.follower_evasion_goal is not None
        ):
            self._publish_goal(
                self.follower_goal_pub,
                self.follower_evasion_goal,
            )
            return
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

    def _search_yield_candidate(
        self,
        priority_now: Point2,
        tangent: Point2,
        yielding_now: Point2,
    ) -> Optional[Tuple[Point2, float]]:
        """Find a short, map-free sidestep point next to `yielding_now`,
        biased away from the priority robot's position and heading."""
        right = (tangent[1], -tangent[0])
        relative = (
            yielding_now[0] - priority_now[0],
            yielding_now[1] - priority_now[1],
        )
        away = normalize(relative)
        side = 1.0 if relative[0] * right[0] + relative[1] * right[1] >= 0 else -1.0
        preferred_side = (side * right[0], side * right[1])
        directions = (
            normalize((
                preferred_side[0] + 0.65 * away[0],
                preferred_side[1] + 0.65 * away[1],
            )),
            preferred_side,
            away,
            (-preferred_side[0], -preferred_side[1]),
        )

        current_separation = distance(priority_now, yielding_now)
        candidates = []
        offset = max(0.30, self.evasion_offset - 0.10)
        while offset <= self.evasion_offset_max + 1.0e-6:
            for direction_index, direction in enumerate(directions):
                candidate = (
                    yielding_now[0] + offset * direction[0],
                    yielding_now[1] + offset * direction[1],
                )
                final_relative = (
                    candidate[0] - priority_now[0],
                    candidate[1] - priority_now[1],
                )
                final_separation = math.hypot(*final_relative)
                lateral_clearance = abs(
                    final_relative[0] * right[0]
                    + final_relative[1] * right[1]
                )
                if final_separation < self.minimum_separation + 0.10:
                    continue
                if (
                    current_separation < self.motion_trigger_distance
                    and final_separation < current_separation + 0.08
                ):
                    continue
                strict = self._candidate_is_free(*candidate)
                relaxed = self._candidate_is_free(
                    *candidate,
                    relaxed=True,
                )
                if not strict and not relaxed:
                    continue
                score = (
                    offset
                    + 0.12 * direction_index
                    + (0.50 if not strict else 0.0)
                    - 0.30 * lateral_clearance
                    - 0.15 * final_separation
                )
                candidates.append((
                    score,
                    candidate,
                    final_separation,
                ))
            offset += 0.10

        if not candidates:
            return None
        _, candidate, final_separation = min(
            candidates,
            key=lambda item: item[0],
        )
        return candidate, final_separation

    def _yield_goal(self) -> Optional[PoseStamped]:
        """Select one short goal for only the robot that must give way."""
        if (
            self.leader_pose is None
            or self.follower_pose is None
            or self.priority_robot is None
        ):
            return None

        if self.priority_robot == self.LEADER:
            priority_pose = self.leader_pose
            yielding_pose = self.follower_pose
            priority_velocity = self.leader_velocity
            priority_sample = self.leader_motion_sample
        else:
            priority_pose = self.follower_pose
            yielding_pose = self.leader_pose
            priority_velocity = self.follower_velocity
            priority_sample = self.follower_motion_sample

        priority_now = self._xy(priority_pose)
        yielding_now = self._xy(yielding_pose)
        tangent = self._motion_tangent(
            priority_pose,
            priority_velocity,
            priority_sample,
        )
        result = self._search_yield_candidate(priority_now, tangent, yielding_now)
        if result is None:
            return None
        candidate, final_separation = result
        yielding_yaw = yaw_from_quaternion(yielding_pose.pose.orientation)
        yielding_robot = (
            self.FOLLOWER
            if self.priority_robot == self.LEADER
            else self.LEADER
        )
        self.get_logger().info(
            f'YIELD_GOAL_SELECTED | robot={yielding_robot} '
            f'distance={distance(yielding_now, candidate):.2f}m '
            f'final_separation={final_separation:.2f}m'
        )
        return self._goal(*candidate, yielding_yaw)

    def _guard_yield_goal(
        self,
        threat_pose: PoseStamped,
        threat_velocity: Point2,
        threat_sample: Optional[Tuple[float, Point2]],
    ) -> Optional[PoseStamped]:
        """Same sidestep search as `_yield_goal`, but for the guard robot
        yielding to whichever of leader/follower is threatening it."""
        if self.guard_pose is None:
            return None
        threat_now = self._xy(threat_pose)
        guard_now = self._xy(self.guard_pose)
        tangent = self._motion_tangent(
            threat_pose, threat_velocity, threat_sample
        )
        result = self._search_yield_candidate(threat_now, tangent, guard_now)
        if result is None:
            return None
        candidate, final_separation = result
        guard_yaw = yaw_from_quaternion(self.guard_pose.pose.orientation)
        self.get_logger().info(
            'GUARD_YIELD_GOAL_SELECTED | '
            f'distance={distance(guard_now, candidate):.2f}m '
            f'final_separation={final_separation:.2f}m'
        )
        return self._goal(*candidate, guard_yaw)

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
            leader_moving = leader_speed >= self.motion_speed_threshold
            follower_moving = follower_speed >= self.motion_speed_threshold
            if leader_moving:
                # Waffle keeps fleet right-of-way whenever both robots move.
                priority = self.LEADER
            elif follower_moving:
                priority = self.FOLLOWER
            else:
                priority = self.LEADER
        return priority

    def _priority_is_moving(self) -> bool:
        if self.priority_robot == self.LEADER:
            velocity = self._effective_velocity(
                self.leader_velocity,
                self.leader_motion_sample,
            )
        elif self.priority_robot == self.FOLLOWER:
            velocity = self._effective_velocity(
                self.follower_velocity,
                self.follower_motion_sample,
            )
        else:
            return False
        return math.hypot(*velocity) >= self.motion_speed_threshold

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
        near_motion = (
            current_distance
            < max(self.minimum_separation + 0.20, 0.72)
            and closing_speed > -0.01
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

    def _guard_conflict(
        self,
    ) -> Tuple[bool, Optional[PoseStamped], Point2, Optional[Tuple[float, Point2]]]:
        """Return whether guard is currently threatened, and by which of
        leader/follower. Guard itself is treated as stationary here since
        it never moves on its own initiative except to yield and return."""
        if self.guard_pose is None:
            return False, None, (0.0, 0.0), None
        guard_now = self._xy(self.guard_pose)
        best: Optional[Tuple[float, PoseStamped, Point2, Optional[Tuple[float, Point2]]]] = None
        for other_pose, other_velocity, other_sample in (
            (self.leader_pose, self.leader_velocity, self.leader_motion_sample),
            (self.follower_pose, self.follower_velocity, self.follower_motion_sample),
        ):
            if other_pose is None:
                continue
            other_now = self._xy(other_pose)
            current_distance = distance(guard_now, other_now)
            effective_velocity = self._effective_velocity(
                other_velocity, other_sample
            )
            other_speed = math.hypot(*effective_velocity)

            closest_distance = current_distance
            closest_time = 0.0
            if other_speed >= self.motion_speed_threshold:
                relative_position = (
                    other_now[0] - guard_now[0],
                    other_now[1] - guard_now[1],
                )
                speed_squared = (
                    effective_velocity[0] ** 2 + effective_velocity[1] ** 2
                )
                if speed_squared > 1.0e-6:
                    closest_time = max(0.0, min(
                        self.motion_prediction_horizon,
                        -(
                            relative_position[0] * effective_velocity[0]
                            + relative_position[1] * effective_velocity[1]
                        ) / speed_squared,
                    ))
                    closest_vector = (
                        relative_position[0]
                        + closest_time * effective_velocity[0],
                        relative_position[1]
                        + closest_time * effective_velocity[1],
                    )
                    closest_distance = math.hypot(*closest_vector)

            emergency = current_distance < self.minimum_separation + 0.08
            approaching = (
                current_distance < self.motion_trigger_distance
                and other_speed >= self.motion_speed_threshold
            )
            predicted = (
                closest_time > 0.05
                and closest_distance < self.motion_predicted_clearance
            )
            if emergency or approaching or predicted:
                if best is None or current_distance < best[0]:
                    best = (
                        current_distance, other_pose, other_velocity,
                        other_sample,
                    )

        if best is None:
            return False, None, (0.0, 0.0), None
        _, threat_pose, threat_velocity, threat_sample = best
        return True, threat_pose, threat_velocity, threat_sample

    def _set_guard_state(self, state: str, detail: str) -> None:
        self.guard_state = state
        self.guard_state_since = self._now()
        text = f'GUARD_{state} | {detail}'
        if text == self.guard_last_status:
            return
        message = String()
        message.data = text
        self.status_pub.publish(message)
        self.get_logger().warning(f'FLEET_COORDINATION | {text}')
        self.guard_last_status = text

    def _begin_guard_escape(
        self,
        threat_pose: PoseStamped,
        threat_velocity: Point2,
        threat_sample: Optional[Tuple[float, Point2]],
    ) -> None:
        self.guard_resume_pose = self._copy_goal(self.guard_pose)
        current_separation = distance(
            self._xy(self.guard_pose), self._xy(threat_pose)
        )
        self.guard_release_separation = max(
            self.motion_trigger_distance + 0.20,
            current_separation + 0.30,
        )
        goal = self._guard_yield_goal(
            threat_pose, threat_velocity, threat_sample
        )
        if goal is None:
            self._set_guard_state(
                self.BLOCKED,
                'no short free goal is available for guard',
            )
            return
        self.guard_evasion_goal = goal
        self._publish_goal(self.guard_goal_pub, goal)
        self.guard_last_evasion_publish = self._now()
        self._set_guard_state(
            self.CLEARING, 'guard yielding to nearby robot motion'
        )

    def _tick_guard(self) -> None:
        if self.guard_pose is None:
            return
        now = self._now()
        if now - self.guard_pose_time > self.pose_stale:
            return

        if self.guard_state == self.COOLDOWN:
            if now < self.guard_cooldown_until:
                return
            self._set_guard_state(self.IDLE, 'ready')

        conflict, threat_pose, threat_velocity, threat_sample = (
            self._guard_conflict()
        )

        if self.guard_state == self.IDLE:
            if conflict:
                self._begin_guard_escape(
                    threat_pose, threat_velocity, threat_sample
                )
            return

        if self.guard_state not in (self.CLEARING, self.BLOCKED):
            return

        threat_now = self._xy(threat_pose) if threat_pose is not None else None
        separated = (
            threat_now is None
            or distance(self._xy(self.guard_pose), threat_now)
            >= self.guard_release_separation
        )
        if not conflict and separated:
            self._publish_goal(self.guard_goal_pub, self.guard_resume_pose)
            self.guard_evasion_goal = None
            self.guard_resume_pose = None
            self.guard_cooldown_until = now + self.cooldown_sec
            self._set_guard_state(
                self.COOLDOWN,
                f'guard resumed; {self.cooldown_sec:.1f}s cooldown',
            )
            return

        if self.guard_state == self.CLEARING:
            if now - self.guard_last_evasion_publish >= 2.5:
                self._publish_goal(
                    self.guard_goal_pub, self.guard_evasion_goal
                )
                self.guard_last_evasion_publish = now
            if now - self.guard_state_since >= self.clearing_timeout:
                if not conflict:
                    self._publish_goal(
                        self.guard_goal_pub, self.guard_resume_pose
                    )
                    self.guard_cooldown_until = now + self.cooldown_sec
                    self._set_guard_state(
                        self.COOLDOWN, 'clearing timeout; resuming'
                    )
                else:
                    self._set_guard_state(
                        self.BLOCKED,
                        'clearing timeout while a robot is still close',
                    )
        elif self.guard_state == self.BLOCKED and conflict:
            goal = self._guard_yield_goal(
                threat_pose, threat_velocity, threat_sample
            )
            if goal is not None:
                self.guard_evasion_goal = goal
                self._publish_goal(self.guard_goal_pub, goal)
                self.guard_last_evasion_publish = now
                self._set_guard_state(
                    self.CLEARING,
                    'a short free yield goal became available',
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
            self._publish_priority_resume_goal()
            self._publish_evasion_goals(force=True)
            return

        if self.state == self.PRIORITY_PASS:
            self._publish_priority_resume_goal()
            self._publish_evasion_goals(force=True)

    def _publish_priority_resume_goal(self) -> None:
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

    def _keep_yielder_paused(self, *, force: bool = False) -> None:
        if self.priority_robot == self.LEADER:
            self._publish_follow_command('PAUSE', force=force)

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
        yield_goal = self._yield_goal()
        if self.priority_robot == self.LEADER:
            self.leader_evasion_goal = None
            self.follower_evasion_goal = yield_goal
        else:
            self.leader_evasion_goal = yield_goal
            self.follower_evasion_goal = None
        self._keep_yielder_paused(force=True)
        if yield_goal is None:
            self._set_state(
                self.BLOCKED,
                'no short free goal is available for the yielding robot',
            )
            return
        self._publish_evasion_goals(force=True)
        self._set_state(
            self.CLEARING,
            f'{reason}; priority={self.priority_robot}; '
            'priority goal preserved; yielding robot moves aside',
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
        if motion_risk:
            return False
        if not self._priority_is_moving():
            return separation >= self.minimum_separation + 0.20
        return separation >= self.release_separation

    def _release_priority(self) -> None:
        self._set_state(
            self.PRIORITY_PASS,
            f'{self.priority_robot} keeps its original Nav2 goal',
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
            f'yield maneuver complete; {self.cooldown_sec:.1f}s cooldown',
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
        self._tick_guard()
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
            self._keep_yielder_paused()
            self._publish_evasion_goals()
            if self.priority_robot == self.LEADER:
                yielding_ready = self._goal_reached(
                    self.follower_pose,
                    self.follower_evasion_goal,
                )
            else:
                yielding_ready = self._goal_reached(
                    self.leader_pose,
                    self.leader_evasion_goal,
                )
            if yielding_ready:
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
            self._keep_yielder_paused()
            if self._priority_passed():
                self._restore_after_maneuver()
            elif now - self.state_since >= self.pass_timeout:
                self._set_state(
                    self.BLOCKED,
                    'priority pass timeout while separation is still unsafe',
                )

        elif self.state == self.BLOCKED:
            self._keep_yielder_paused()
            motion_risk, _, _ = self._motion_risk()
            if (
                separation >= self.minimum_separation + 0.20
                and not motion_risk
            ):
                self._restore_after_maneuver()
            else:
                if self.priority_robot is None:
                    self.priority_robot = self._choose_motion_priority()
                yield_goal = self._yield_goal()
                if yield_goal is not None:
                    if self.priority_robot == self.LEADER:
                        self.leader_evasion_goal = None
                        self.follower_evasion_goal = yield_goal
                    else:
                        self.leader_evasion_goal = yield_goal
                        self.follower_evasion_goal = None
                    self._publish_evasion_goals(force=True)
                    self._set_state(
                        self.CLEARING,
                        'a short free yield goal became available',
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
