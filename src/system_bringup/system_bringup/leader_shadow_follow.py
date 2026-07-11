#!/usr/bin/env python3
"""Low-speed leader shadow follow for the active scout.

The leader remains a leader: it does not copy the scout pose and it stops
issuing shadow goals as soon as failover owns recovery.  During normal
operation it estimates the active scout's movement heading, creates a rear
standoff target, validates it against the shared map, and publishes rate-limited
leader Nav2 goals.
"""

from __future__ import annotations

import json
import math
from enum import Enum
from typing import Optional, Tuple

import rclpy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import OccupancyGrid
from rcl_interfaces.msg import Parameter, ParameterType, ParameterValue
from rcl_interfaces.srv import SetParameters
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Bool, String


Point2 = Tuple[float, float]


class LeaderMode(Enum):
    IDLE = 'LEADER_IDLE'
    SHADOW_FOLLOW = 'LEADER_SHADOW_FOLLOW'
    SCOUT_SUSPECTED_DEAD = 'LEADER_SCOUT_SUSPECTED_DEAD'
    RECOVERY_NAVIGATING = 'LEADER_RECOVERY_NAVIGATING'
    RECOVERY_POSITION_REACHED = 'LEADER_RECOVERY_POSITION_REACHED'
    WAIT_NEW_SCOUT = 'LEADER_WAIT_NEW_SCOUT'
    SHADOW_NEW_SCOUT = 'LEADER_SHADOW_NEW_SCOUT'


def yaw_from_quaternion(q) -> float:
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


def quaternion_from_yaw(yaw: float) -> tuple[float, float, float, float]:
    return (0.0, 0.0, math.sin(0.5 * yaw), math.cos(0.5 * yaw))


def angle_lerp(current: float, target: float, alpha: float) -> float:
    diff = math.atan2(math.sin(target - current), math.cos(target - current))
    return current + alpha * diff


class LeaderShadowFollow(Node):
    def __init__(self) -> None:
        super().__init__('leader_shadow_follow')

        self.declare_parameter('enable_leader_shadow_follow', True)
        self.declare_parameter('leader_pose_topic', '/leader_pose')
        self.declare_parameter('active_scout_pose_topic', '/member_pose')
        self.declare_parameter('follower_scout_pose_topic', '/burger_pose')
        self.declare_parameter('leader_goal_topic', '/fleet/leader_coord_goal')
        self.declare_parameter('leader_cancel_topic', '/fleet/leader_nav_cancel')
        self.declare_parameter(
            'controller_set_parameters_service',
            '/controller_server/set_parameters',
        )
        self.declare_parameter('map_topic', '/map')
        self.declare_parameter('failover_state_topic', '/failover/state')
        self.declare_parameter('active_scout_id_topic', '/failover/active_scout_id')
        self.declare_parameter('active_scout_robot_name', 'scout22')
        self.declare_parameter('follower_robot_name', 'follower21')
        self.declare_parameter('require_localization_ready', True)
        self.declare_parameter('localization_ready_topic', '/localization_ready')
        self.declare_parameter('scout_pose_timeout_sec', 2.5)
        self.declare_parameter('startup_grace_sec', 8.0)

        self.declare_parameter('leader_shadow_follow_distance_m', 2.8)
        self.declare_parameter('leader_shadow_stop_distance_m', 2.2)
        self.declare_parameter('leader_shadow_resume_distance_m', 3.0)
        self.declare_parameter('leader_shadow_far_distance_m', 4.5)
        self.declare_parameter('leader_shadow_max_linear_vel', 0.28)
        self.declare_parameter('leader_shadow_catchup_max_linear_vel', 0.40)
        self.declare_parameter('leader_shadow_max_angular_vel', 0.70)
        self.declare_parameter('leader_restore_max_linear_vel', 0.45)
        self.declare_parameter('leader_restore_max_angular_vel', 0.90)
        self.declare_parameter('leader_shadow_goal_update_period_sec', 1.0)
        self.declare_parameter('leader_shadow_goal_min_change_m', 0.5)
        self.declare_parameter('leader_shadow_heading_min_motion_m', 0.15)
        self.declare_parameter('leader_shadow_heading_alpha', 0.35)
        self.declare_parameter('leader_shadow_map_clearance_m', 0.22)
        self.declare_parameter('leader_shadow_target_search_radius_m', 1.2)
        self.declare_parameter('leader_shadow_target_search_step_m', 0.15)
        self.declare_parameter('occupied_threshold', 50)
        self.declare_parameter('allow_unknown_shadow_target', False)

        self.declare_parameter('enable_leader_continuous_scan', True)
        self.declare_parameter('leader_scan_topic', '/scan')
        self.declare_parameter('leader_scan_fov_deg', 60.0)
        self.declare_parameter('leader_scan_update_rate_hz', 10.0)
        self.declare_parameter('leader_scan_timeout_sec', 1.0)

        get = self.get_parameter
        self.enabled = bool(get('enable_leader_shadow_follow').value)
        self.leader_pose_topic = str(get('leader_pose_topic').value)
        self.active_pose_topic = str(get('active_scout_pose_topic').value)
        self.follower_pose_topic = str(get('follower_scout_pose_topic').value)
        self.leader_goal_topic = str(get('leader_goal_topic').value)
        self.leader_cancel_topic = str(get('leader_cancel_topic').value)
        self.controller_set_parameters_service = str(
            get('controller_set_parameters_service').value
        )
        self.map_topic = str(get('map_topic').value)
        self.failover_state_topic = str(get('failover_state_topic').value)
        self.active_scout_id_topic = str(get('active_scout_id_topic').value)
        self.original_scout_id = str(get('active_scout_robot_name').value)
        self.follower_robot_name = str(get('follower_robot_name').value)
        self.require_localization_ready = bool(get('require_localization_ready').value)
        self.localization_ready_topic = str(get('localization_ready_topic').value)
        self.scout_pose_timeout = max(0.2, float(get('scout_pose_timeout_sec').value))
        self.startup_grace = max(0.0, float(get('startup_grace_sec').value))

        self.follow_distance = max(0.5, float(get('leader_shadow_follow_distance_m').value))
        self.stop_distance = max(0.2, float(get('leader_shadow_stop_distance_m').value))
        self.resume_distance = max(self.stop_distance, float(get('leader_shadow_resume_distance_m').value))
        self.far_distance = max(self.resume_distance, float(get('leader_shadow_far_distance_m').value))
        self.shadow_linear_vel = max(0.03, float(get('leader_shadow_max_linear_vel').value))
        self.catchup_linear_vel = max(
            self.shadow_linear_vel,
            float(get('leader_shadow_catchup_max_linear_vel').value),
        )
        self.shadow_angular_vel = max(0.05, float(get('leader_shadow_max_angular_vel').value))
        self.restore_linear_vel = max(self.shadow_linear_vel, float(get('leader_restore_max_linear_vel').value))
        self.restore_angular_vel = max(self.shadow_angular_vel, float(get('leader_restore_max_angular_vel').value))
        self.goal_period = max(0.3, float(get('leader_shadow_goal_update_period_sec').value))
        self.goal_min_change = max(0.05, float(get('leader_shadow_goal_min_change_m').value))
        self.heading_min_motion = max(0.02, float(get('leader_shadow_heading_min_motion_m').value))
        self.heading_alpha = min(1.0, max(0.01, float(get('leader_shadow_heading_alpha').value)))
        self.map_clearance = max(0.05, float(get('leader_shadow_map_clearance_m').value))
        self.search_radius = max(0.0, float(get('leader_shadow_target_search_radius_m').value))
        self.search_step = max(0.05, float(get('leader_shadow_target_search_step_m').value))
        self.occupied_threshold = int(get('occupied_threshold').value)
        self.allow_unknown = bool(get('allow_unknown_shadow_target').value)

        self.scan_enabled = bool(get('enable_leader_continuous_scan').value)
        self.scan_topic = str(get('leader_scan_topic').value)
        self.scan_fov_deg = max(1.0, min(180.0, float(get('leader_scan_fov_deg').value)))
        self.scan_rate = max(0.5, float(get('leader_scan_update_rate_hz').value))
        self.scan_timeout = max(0.1, float(get('leader_scan_timeout_sec').value))

        latched_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
        )
        map_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
        )

        self.goal_pub = self.create_publisher(PoseStamped, self.leader_goal_topic, 10)
        self.cancel_pub = self.create_publisher(Bool, self.leader_cancel_topic, latched_qos)
        self.state_pub = self.create_publisher(String, '/leader_shadow/state', latched_qos)
        self.goal_debug_pub = self.create_publisher(PoseStamped, '/leader_shadow/goal', 10)
        self.scan_state_pub = self.create_publisher(String, '/leader_scan/state', latched_qos)

        self.create_subscription(PoseStamped, self.leader_pose_topic, self._on_leader_pose, 10)
        self.create_subscription(PoseStamped, self.active_pose_topic, self._on_original_scout_pose, 20)
        self.create_subscription(PoseStamped, self.follower_pose_topic, self._on_follower_scout_pose, 20)
        self.create_subscription(OccupancyGrid, self.map_topic, self._on_map, map_qos)
        self.create_subscription(String, self.failover_state_topic, self._on_failover_state, latched_qos)
        self.create_subscription(String, self.active_scout_id_topic, self._on_active_scout_id, latched_qos)
        if self.require_localization_ready:
            self.create_subscription(
                Bool,
                self.localization_ready_topic,
                self._on_localization_ready,
                latched_qos,
            )
        if self.scan_enabled:
            self.create_subscription(LaserScan, self.scan_topic, self._on_scan, 10)

        self.controller_client = self.create_client(
            SetParameters, self.controller_set_parameters_service
        )

        self.mode = LeaderMode.IDLE
        self.start_wall = self._now()
        self.active_scout_id = self.original_scout_id
        self.failover_state = 'NORMAL_OPERATION'
        self.localization_ready = not self.require_localization_ready
        self.leader_pose: Optional[PoseStamped] = None
        self.original_scout_pose: Optional[PoseStamped] = None
        self.follower_scout_pose: Optional[PoseStamped] = None
        self.original_scout_wall = -1.0e9
        self.follower_scout_wall = -1.0e9
        self.map_msg: Optional[OccupancyGrid] = None
        self.last_scan_wall = -1.0e9
        self.last_scan_stamp = -1.0

        self.heading: Optional[float] = None
        self.previous_scout_sample: Optional[Tuple[float, Point2]] = None
        # True from the start: the leader should shadow the scout right
        # away, not only once it has already wandered resume_distance_m
        # away. With this False, "shadow_active" could only ever flip on
        # via the resume-distance branch below, which never fires if the
        # scout starts anywhere closer than that (the common case indoors)
        # -- the leader would never begin following at all.
        self.shadow_active = True
        self.last_goal: Optional[PoseStamped] = None
        self.shadow_goal_active = False
        self.last_goal_wall = -1.0e9
        self.last_nominal_target: Optional[Point2] = None
        self.speed_profile: Optional[str] = None
        self.speed_limit_pending = False

        self.create_timer(0.2, self._tick)
        self.create_timer(1.0 / self.scan_rate, self._scan_tick)
        self._publish_cancel(False)
        self._publish_state('startup')
        self.get_logger().warning(
            '[LEADER_SHADOW] READY | '
            f'enabled={self.enabled} scout={self.original_scout_id}:{self.active_pose_topic} '
            f'follower_scout={self.follower_robot_name}:{self.follower_pose_topic} '
            f'distance={self.follow_distance:.2f}m fov={self.scan_fov_deg:.1f}deg '
            f'controller_service={self.controller_set_parameters_service} '
            f'localization_gate={self.require_localization_ready}:{self.localization_ready_topic}'
        )

    def _now(self) -> float:
        return self.get_clock().now().nanoseconds * 1.0e-9

    def _on_leader_pose(self, msg: PoseStamped) -> None:
        self.leader_pose = msg

    def _on_original_scout_pose(self, msg: PoseStamped) -> None:
        self.original_scout_pose = msg
        self.original_scout_wall = self._now()
        if self.active_scout_id == self.original_scout_id:
            self._update_heading_from_pose(msg)

    def _on_follower_scout_pose(self, msg: PoseStamped) -> None:
        self.follower_scout_pose = msg
        self.follower_scout_wall = self._now()
        if self.active_scout_id == self.follower_robot_name:
            self._update_heading_from_pose(msg)

    def _on_map(self, msg: OccupancyGrid) -> None:
        self.map_msg = msg

    def _on_failover_state(self, msg: String) -> None:
        previous = self.failover_state
        self.failover_state = msg.data.strip() or 'UNKNOWN'
        if previous != self.failover_state and not self._failover_allows_shadow():
            self._cancel_shadow_goal(f'failover_{self.failover_state}')

    def _on_active_scout_id(self, msg: String) -> None:
        scout_id = msg.data.strip()
        if not scout_id or scout_id == self.active_scout_id:
            return
        self._cancel_shadow_goal('active_scout_changed')
        self.active_scout_id = scout_id
        self.previous_scout_sample = None
        self.heading = None
        self.shadow_active = False
        self.last_goal = None
        self.mode = LeaderMode.SHADOW_NEW_SCOUT
        self.get_logger().warning(
            f'[LEADER_SHADOW] ACTIVE_SCOUT_CHANGED | active_scout={scout_id}'
        )

    def _on_localization_ready(self, msg: Bool) -> None:
        previous = self.localization_ready
        self.localization_ready = bool(msg.data)
        if previous and not self.localization_ready:
            self._cancel_shadow_goal('localization_not_ready')
        if self.localization_ready and not previous:
            self.get_logger().warning(
                f'[LEADER_SHADOW] LOCALIZATION_READY | topic={self.localization_ready_topic}'
            )

    def _on_scan(self, msg: LaserScan) -> None:
        self.last_scan_wall = self._now()
        stamp = msg.header.stamp
        self.last_scan_stamp = float(stamp.sec) + float(stamp.nanosec) * 1.0e-9

    def _tick(self) -> None:
        if not self.enabled:
            self._cancel_shadow_goal('disabled')
            self._set_controller_speed_limit(False)
            self.mode = LeaderMode.IDLE
            self._publish_state('disabled')
            return
        if self._now() - self.start_wall < self.startup_grace:
            self._publish_state('startup_grace')
            return
        if self.require_localization_ready and not self.localization_ready:
            self._cancel_shadow_goal('waiting_localization_ready')
            self.mode = LeaderMode.IDLE
            self._set_controller_speed_limit(False)
            self._publish_state('waiting_localization_ready')
            return
        if not self._failover_allows_shadow():
            self._stop_shadow_for_failover()
            self._publish_state('failover_owns_leader_goal')
            return

        scout_pose, scout_wall = self._active_scout_pose()
        if self.leader_pose is None or scout_pose is None:
            self._cancel_shadow_goal('waiting_pose')
            self.mode = LeaderMode.IDLE
            self._set_controller_speed_limit(False)
            self._publish_state('waiting_pose')
            return
        scout_age = self._now() - scout_wall
        if scout_age > self.scout_pose_timeout:
            self._cancel_shadow_goal('scout_pose_stale')
            self.mode = LeaderMode.SCOUT_SUSPECTED_DEAD
            self._set_controller_speed_limit(False)
            self._publish_state(f'scout_pose_stale_{scout_age:.2f}s')
            return

        distance_to_scout = self._distance_pose(self.leader_pose, scout_pose)
        if self.shadow_active:
            if distance_to_scout <= self.stop_distance:
                self.shadow_active = False
        elif distance_to_scout >= self.resume_distance:
            self.shadow_active = True

        if not self.shadow_active:
            self._cancel_shadow_goal('inside_standoff_hysteresis')
            self.mode = LeaderMode.IDLE
            self._set_controller_speed_limit(False)
            self._publish_state('inside_standoff_hysteresis')
            return

        self.mode = (
            LeaderMode.SHADOW_NEW_SCOUT
            if self.active_scout_id != self.original_scout_id
            else LeaderMode.SHADOW_FOLLOW
        )
        catchup = distance_to_scout >= self.far_distance
        self._set_controller_speed_limit(True, catchup=catchup)

        goal = self._build_shadow_goal(scout_pose)
        if goal is None:
            self._cancel_shadow_goal('no_feasible_shadow_target')
            self._publish_state('no_feasible_shadow_target')
            return
        if not self._should_publish_goal(goal):
            self._publish_state('goal_rate_limited')
            return

        self.goal_pub.publish(goal)
        self.goal_debug_pub.publish(goal)
        self.last_goal = goal
        self.shadow_goal_active = True
        self.last_goal_wall = self._now()
        self._publish_state('goal_sent', goal=goal, distance_to_scout=distance_to_scout)
        self.get_logger().warning(
            '[LEADER_SHADOW] GOAL_SENT | '
            f'active_scout={self.active_scout_id} '
            f'x={goal.pose.position.x:.3f} y={goal.pose.position.y:.3f} '
            f'D={distance_to_scout:.2f} catchup={catchup}'
        )

    def _scan_tick(self) -> None:
        if not self.scan_enabled:
            self._publish_scan_state('SCAN_DISABLED')
            return
        now = self._now()
        age = now - self.last_scan_wall
        scan_state = 'SCAN_ACTIVE' if age <= self.scan_timeout else 'SCAN_STALE'
        self._publish_scan_state(scan_state, age=age)

    def _failover_allows_shadow(self) -> bool:
        return self.failover_state in (
            '',
            'NORMAL_OPERATION',
            'NEW_SCOUT_EXPLORING',
        )

    def _stop_shadow_for_failover(self) -> None:
        self._cancel_shadow_goal(f'failover_{self.failover_state}')
        self.shadow_active = False
        self.last_goal = None
        self._set_controller_speed_limit(False)
        if self.failover_state in (
            'SCOUT_SUSPECTED_DEAD',
            'SCOUT_DEAD_CONFIRMED',
            'FAILOVER_TRIGGERED',
            'RECOVERY_NAVIGATING',
        ):
            self.mode = LeaderMode.RECOVERY_NAVIGATING
        elif self.failover_state == 'FOLLOWER_SCOUT_TAKEOVER':
            self.mode = LeaderMode.WAIT_NEW_SCOUT
        else:
            self.mode = LeaderMode.IDLE

    def _active_scout_pose(self) -> Tuple[Optional[PoseStamped], float]:
        if self.active_scout_id == self.follower_robot_name:
            return self.follower_scout_pose, self.follower_scout_wall
        return self.original_scout_pose, self.original_scout_wall

    def _update_heading_from_pose(self, pose: PoseStamped) -> None:
        now = self._now()
        point = (pose.pose.position.x, pose.pose.position.y)
        if self.previous_scout_sample is None:
            self.previous_scout_sample = (now, point)
            if self.heading is None:
                self.heading = yaw_from_quaternion(pose.pose.orientation)
            return
        _, previous = self.previous_scout_sample
        dx = point[0] - previous[0]
        dy = point[1] - previous[1]
        moved = math.hypot(dx, dy)
        if moved < self.heading_min_motion:
            return
        measured = math.atan2(dy, dx)
        self.heading = measured if self.heading is None else angle_lerp(
            self.heading, measured, self.heading_alpha
        )
        self.previous_scout_sample = (now, point)

    def _build_shadow_goal(self, scout_pose: PoseStamped) -> Optional[PoseStamped]:
        heading = self.heading
        if heading is None:
            heading = yaw_from_quaternion(scout_pose.pose.orientation)
        nominal = (
            scout_pose.pose.position.x - self.follow_distance * math.cos(heading),
            scout_pose.pose.position.y - self.follow_distance * math.sin(heading),
        )
        self.last_nominal_target = nominal
        feasible = self._nearest_feasible(nominal)
        if feasible is None:
            return None
        goal = PoseStamped()
        goal.header.frame_id = 'map'
        goal.header.stamp = self.get_clock().now().to_msg()
        goal.pose.position.x = feasible[0]
        goal.pose.position.y = feasible[1]
        goal.pose.position.z = 0.0
        qx, qy, qz, qw = quaternion_from_yaw(heading)
        goal.pose.orientation.x = qx
        goal.pose.orientation.y = qy
        goal.pose.orientation.z = qz
        goal.pose.orientation.w = qw
        return goal

    def _nearest_feasible(self, nominal: Point2) -> Optional[Point2]:
        if self._candidate_is_free(nominal[0], nominal[1]):
            return nominal
        best: Optional[Tuple[float, Point2]] = None
        rings = int(math.ceil(self.search_radius / self.search_step))
        for ring in range(1, rings + 1):
            radius = ring * self.search_step
            samples = max(12, int(math.ceil(2.0 * math.pi * radius / self.search_step)))
            for index in range(samples):
                theta = 2.0 * math.pi * index / samples
                candidate = (
                    nominal[0] + radius * math.cos(theta),
                    nominal[1] + radius * math.sin(theta),
                )
                if not self._candidate_is_free(candidate[0], candidate[1]):
                    continue
                score = math.hypot(candidate[0] - nominal[0], candidate[1] - nominal[1])
                if best is None or score < best[0]:
                    best = (score, candidate)
            if best is not None:
                return best[1]
        return None

    def _candidate_is_free(self, x: float, y: float) -> bool:
        if self.map_msg is None:
            return True
        center = self._world_to_map(x, y)
        if center is None:
            return False
        info = self.map_msg.info
        radius = max(1, int(math.ceil(self.map_clearance / info.resolution)))
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
                if value < 0 and not self.allow_unknown:
                    return False
        return True

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

    def _should_publish_goal(self, goal: PoseStamped) -> bool:
        now = self._now()
        if now - self.last_goal_wall < self.goal_period:
            return False
        if self.last_goal is None:
            return True
        distance = math.hypot(
            goal.pose.position.x - self.last_goal.pose.position.x,
            goal.pose.position.y - self.last_goal.pose.position.y,
        )
        return distance >= self.goal_min_change

    def _set_controller_speed_limit(self, limited: bool, *, catchup: bool = False) -> None:
        profile = 'restore'
        if limited:
            profile = 'catchup' if catchup else 'shadow'
        if self.speed_limit_pending:
            return
        if self.speed_profile == profile:
            return
        if not self.controller_client.service_is_ready():
            self.get_logger().info(
                '[LEADER_SHADOW] CONTROLLER_PARAM_SERVICE_WAIT | '
                'shadow speed limit will be retried',
                throttle_duration_sec=5.0,
            )
            return
        max_linear = self.catchup_linear_vel if limited and catchup else self.shadow_linear_vel
        max_angular = self.shadow_angular_vel if limited else self.restore_angular_vel
        if not limited:
            max_linear = self.restore_linear_vel
        request = SetParameters.Request()
        request.parameters = [
            self._double_parameter('FollowPath.max_vel_x', max_linear),
            self._double_parameter('FollowPath.max_speed_xy', max_linear),
            self._double_parameter('FollowPath.max_vel_theta', max_angular),
        ]
        self.speed_limit_pending = True
        future = self.controller_client.call_async(request)
        future.add_done_callback(
            lambda fut: self._on_speed_limit_result(fut, profile)
        )

    def _on_speed_limit_result(self, future, profile: str) -> None:
        self.speed_limit_pending = False
        try:
            results = future.result().results
        except Exception as exc:  # noqa: BLE001
            self.get_logger().warning(f'[LEADER_SHADOW] SPEED_LIMIT_FAILED | {exc}')
            return
        ok = all(bool(result.successful) for result in results)
        if ok:
            self.speed_profile = profile
            self.get_logger().warning(
                '[LEADER_SHADOW] SPEED_LIMIT_SET | '
                f'profile={profile} linear={self.shadow_linear_vel:.2f}/{self.restore_linear_vel:.2f}'
            )
        else:
            reason = '; '.join(str(result.reason) for result in results if result.reason)
            self.get_logger().warning(
                f'[LEADER_SHADOW] SPEED_LIMIT_REJECTED | {reason or "unknown"}'
            )

    @staticmethod
    def _double_parameter(name: str, value: float) -> Parameter:
        param = Parameter()
        param.name = name
        param.value = ParameterValue(
            type=ParameterType.PARAMETER_DOUBLE,
            double_value=float(value),
        )
        return param

    def _publish_cancel(self, value: bool) -> None:
        msg = Bool()
        msg.data = bool(value)
        self.cancel_pub.publish(msg)

    def _cancel_shadow_goal(self, reason: str) -> None:
        """Cancel the previous shadow goal once when shadow loses authority."""
        if not self.shadow_goal_active:
            return
        self._publish_cancel(True)
        self.shadow_goal_active = False
        self.last_goal = None
        self.get_logger().warning(
            f'[LEADER_SHADOW] GOAL_CANCELLED | reason={reason}'
        )

    def _publish_state(
        self,
        reason: str,
        *,
        goal: Optional[PoseStamped] = None,
        distance_to_scout: Optional[float] = None,
    ) -> None:
        data = {
            'mode': self.mode.value,
            'reason': reason,
            'active_scout_id': self.active_scout_id,
            'failover_state': self.failover_state,
            'shadow_active': self.shadow_active,
            'scan_fov_deg': self.scan_fov_deg,
            'scan_heading_reference': 'leader_current_heading',
        }
        if distance_to_scout is not None:
            data['distance_to_scout_m'] = round(float(distance_to_scout), 3)
        if self.heading is not None:
            data['movement_heading_rad'] = round(float(self.heading), 4)
        if self.last_nominal_target is not None:
            data['nominal_target'] = {
                'x': round(float(self.last_nominal_target[0]), 3),
                'y': round(float(self.last_nominal_target[1]), 3),
            }
        if goal is not None:
            data['goal'] = {
                'x': round(float(goal.pose.position.x), 3),
                'y': round(float(goal.pose.position.y), 3),
            }
        msg = String()
        msg.data = json.dumps(data, sort_keys=True)
        self.state_pub.publish(msg)

    def _publish_scan_state(self, state: str, *, age: Optional[float] = None) -> None:
        data = {
            'state': state,
            'enabled': self.scan_enabled,
            'scan_topic': self.scan_topic,
            'fov_deg': self.scan_fov_deg,
            'relative_bearing_accept_rad': round(math.radians(self.scan_fov_deg) * 0.5, 4),
            'heading_reference': 'leader_current_heading',
            'risk_scan_only': True,
            'nav2_obstacle_lidar_unchanged': True,
        }
        if age is not None:
            data['age_sec'] = round(float(age), 3)
        if self.last_scan_stamp >= 0.0:
            data['last_stamp_sec'] = round(float(self.last_scan_stamp), 3)
        msg = String()
        msg.data = json.dumps(data, sort_keys=True)
        self.scan_state_pub.publish(msg)

    @staticmethod
    def _distance_pose(first: PoseStamped, second: PoseStamped) -> float:
        return math.hypot(
            first.pose.position.x - second.pose.position.x,
            first.pose.position.y - second.pose.position.y,
        )


def main(args=None) -> None:
    rclpy.init(args=args)
    node = LeaderShadowFollow()
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
