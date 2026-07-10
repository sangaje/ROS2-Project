#!/usr/bin/env python3
"""Unified runtime for non-leader field robots.

Follower and scout are the same robot runtime with different role-selected
goal sources:

* FOLLOWER -> follow leader pose with Nav2.
* ACTIVE_SCOUT -> publish active-scout heartbeat and run in-process SAC inference.
* RECOVERY_NAVIGATING -> cancel the previous role and navigate to failure pose.
* LOCALIZATION_SPIN -> own cmd_vel for a verified in-place rotation.
* LOCALIZATION_SETTLE -> wait briefly so AMCL can settle before checking.
"""

from __future__ import annotations

import json
import math
from copy import deepcopy
from enum import Enum
from functools import partial
from typing import Optional

import rclpy
from action_msgs.msg import GoalStatus
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped, Twist, TwistStamped
from nav2_msgs.action import NavigateToPose
from nav_msgs.msg import Odometry
from rclpy.action import ActionClient
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import Bool, String

from .scout_rl_runtime import ActiveScoutRLRuntime


class Role(Enum):
    IDLE = 'IDLE'
    FOLLOWER = 'FOLLOWER'
    ACTIVE_SCOUT = 'ACTIVE_SCOUT'
    RECOVERY_NAVIGATING = 'RECOVERY_NAVIGATING'
    ARRIVED_AT_FAILURE_POSE = 'ARRIVED_AT_FAILURE_POSE'
    LOCALIZATION_CHECK = 'LOCALIZATION_CHECK'
    LOCALIZATION_SPIN = 'LOCALIZATION_SPIN'
    LOCALIZATION_SETTLE = 'LOCALIZATION_SETTLE'
    FAILED = 'FAILED'


class MotionAuthority(Enum):
    """The one non-safety command source allowed to own this robot."""

    NONE = 'NONE'
    LOCALIZATION_SPIN = 'LOCALIZATION_SPIN'
    FAILOVER_RECOVERY_NAV = 'FAILOVER_RECOVERY_NAV'
    ACTIVE_SCOUT_RL = 'ACTIVE_SCOUT_RL'
    NORMAL_FOLLOW = 'NORMAL_FOLLOW'


def yaw_from_quaternion(q) -> float:
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


def quaternion_from_yaw(yaw: float) -> tuple[float, float, float, float]:
    return (0.0, 0.0, math.sin(0.5 * yaw), math.cos(0.5 * yaw))


def wrap_angle(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def normalize_role(raw: str) -> Role:
    key = raw.strip().upper()
    aliases = {
        'FOLLOWING': 'FOLLOWER',
        'SCOUT': 'ACTIVE_SCOUT',
        'ACTIVE_SCOUT_EXPLORING': 'ACTIVE_SCOUT',
        'RECOVERY': 'RECOVERY_NAVIGATING',
    }
    key = aliases.get(key, key)
    return Role[key] if key in Role.__members__ else Role.IDLE


def parse_epoch(value) -> Optional[int]:
    """Parse an ownership epoch without truncating floats or accepting bools."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if value >= 0 else None
    if isinstance(value, str) and value.strip().isdigit():
        return int(value.strip())
    return None


class UnifiedFieldRobot(Node):
    def __init__(self) -> None:
        super().__init__('unified_field_robot')

        self.declare_parameter('robot_name', 'field_robot')
        self.declare_parameter('initial_role', 'IDLE')
        self.declare_parameter('enable_follow_mode', True)
        self.declare_parameter('enable_scout_mode', True)
        self.declare_parameter('enable_recovery_mode', True)
        self.declare_parameter('enable_localization_spin', True)
        self.declare_parameter('enable_exploration', True)
        self.declare_parameter('leader_pose_topic', '/leader_pose')
        self.declare_parameter('self_pose_topic', '/burger_pose')
        self.declare_parameter('localization_ready_topic', '/localization_ready')
        self.declare_parameter('require_localization_ready', True)
        self.declare_parameter('role_command_topic', '/fleet/field_robot_role_cmd')
        self.declare_parameter('fleet_role_topic', '/fleet/scout_role')
        self.declare_parameter('status_topic', '/fleet/field_robot_status')
        self.declare_parameter('legacy_takeover_status_topic', '/fleet/scout_takeover_status')
        self.declare_parameter('active_scout_heartbeat_topic', '/scout/signal')
        self.declare_parameter('navigate_action', '/navigate_to_pose')
        self.declare_parameter('cmd_vel_topic', '/cmd_vel')
        self.declare_parameter('use_stamped_cmd_vel', True)
        self.declare_parameter('amcl_pose_topic', '/amcl_pose')
        self.declare_parameter('odom_topic', '/odom')
        self.declare_parameter('follow_distance_m', 0.70)
        self.declare_parameter('follow_goal_period_sec', 1.0)
        self.declare_parameter('follow_goal_update_distance_m', 0.25)
        self.declare_parameter('recovery_arrival_tolerance_m', 0.40)
        self.declare_parameter('self_pose_timeout_sec', 2.0)
        self.declare_parameter('movement_start_distance_m', 0.03)
        self.declare_parameter('movement_start_samples', 3)
        self.declare_parameter('max_recovery_nav_retries', 3)
        self.declare_parameter('recovery_nav_retry_sec', 1.0)
        self.declare_parameter('max_xy_covariance', 0.22)
        self.declare_parameter('max_yaw_covariance', 0.16)
        self.declare_parameter('max_amcl_pose_age_sec', 3.0)
        self.declare_parameter('spin_speed_rad_s', 0.40)
        self.declare_parameter('spin_target_angle_rad', 7.10)
        self.declare_parameter('spin_timeout_sec', 42.0)
        self.declare_parameter('settle_duration_sec', 2.0)
        self.declare_parameter('max_spin_retries', 3)
        self.declare_parameter('heartbeat_period_sec', 0.5)

        get = self.get_parameter
        self.robot_name = str(get('robot_name').value)
        self.role = normalize_role(str(get('initial_role').value))
        self.enable_follow = bool(get('enable_follow_mode').value)
        self.enable_scout = bool(get('enable_scout_mode').value)
        self.enable_recovery = bool(get('enable_recovery_mode').value)
        self.enable_spin = bool(get('enable_localization_spin').value)
        self.enable_exploration = bool(get('enable_exploration').value)
        self.leader_pose_topic = str(get('leader_pose_topic').value)
        self.self_pose_topic = str(get('self_pose_topic').value)
        self.localization_ready_topic = str(get('localization_ready_topic').value)
        self.require_localization_ready = bool(get('require_localization_ready').value)
        self.role_command_topic = str(get('role_command_topic').value)
        self.fleet_role_topic = str(get('fleet_role_topic').value)
        self.status_topic = str(get('status_topic').value)
        self.legacy_status_topic = str(get('legacy_takeover_status_topic').value)
        self.heartbeat_topic = str(get('active_scout_heartbeat_topic').value)
        self.navigate_action = str(get('navigate_action').value)
        self.cmd_vel_topic = str(get('cmd_vel_topic').value)
        self.use_stamped = bool(get('use_stamped_cmd_vel').value)
        self.amcl_pose_topic = str(get('amcl_pose_topic').value)
        self.odom_topic = str(get('odom_topic').value)
        self.follow_distance = max(0.1, float(get('follow_distance_m').value))
        self.follow_goal_period = max(0.2, float(get('follow_goal_period_sec').value))
        self.follow_update_distance = max(0.05, float(get('follow_goal_update_distance_m').value))
        self.arrival_tolerance = max(0.05, float(get('recovery_arrival_tolerance_m').value))
        self.self_pose_timeout = max(0.1, float(get('self_pose_timeout_sec').value))
        self.movement_start_distance = max(
            0.005, float(get('movement_start_distance_m').value)
        )
        self.movement_start_samples = max(
            1, int(get('movement_start_samples').value)
        )
        self.max_recovery_nav_retries = max(
            0, int(get('max_recovery_nav_retries').value)
        )
        self.recovery_nav_retry_sec = max(
            0.1, float(get('recovery_nav_retry_sec').value)
        )
        self.max_xy_cov = max(0.0, float(get('max_xy_covariance').value))
        self.max_yaw_cov = max(0.0, float(get('max_yaw_covariance').value))
        self.max_amcl_age = max(0.1, float(get('max_amcl_pose_age_sec').value))
        self.spin_speed = abs(float(get('spin_speed_rad_s').value))
        self.spin_target = max(0.0, float(get('spin_target_angle_rad').value))
        self.spin_timeout = max(1.0, float(get('spin_timeout_sec').value))
        self.settle_duration = max(0.0, float(get('settle_duration_sec').value))
        self.max_spin_retries = max(0, int(get('max_spin_retries').value))
        self.heartbeat_period = max(0.1, float(get('heartbeat_period_sec').value))
        self.scout_rl_enabled = bool(self.enable_scout and self.enable_exploration)

        latched_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
        )
        self.status_pub = self.create_publisher(String, self.status_topic, latched_qos)
        self.legacy_status_pub = self.create_publisher(String, self.legacy_status_topic, latched_qos)
        self.role_pub = self.create_publisher(String, f'/{self.robot_name}/role', latched_qos)
        self.heartbeat_pub = self.create_publisher(String, self.heartbeat_topic, 10)
        if self.use_stamped:
            self.cmd_pub = self.create_publisher(TwistStamped, self.cmd_vel_topic, 10)
        else:
            self.cmd_pub = self.create_publisher(Twist, self.cmd_vel_topic, 10)

        self.nav_client = ActionClient(self, NavigateToPose, self.navigate_action)
        self.create_subscription(String, self.role_command_topic, self._on_role_command, latched_qos)
        self.create_subscription(String, self.fleet_role_topic, self._on_fleet_role, latched_qos)
        self.create_subscription(PoseStamped, self.leader_pose_topic, self._on_leader_pose, 10)
        self.create_subscription(PoseStamped, self.self_pose_topic, self._on_self_pose, 10)
        self.create_subscription(PoseWithCovarianceStamped, self.amcl_pose_topic, self._on_amcl, 10)
        self.create_subscription(Odometry, self.odom_topic, self._on_odom, 10)
        self.create_subscription(
            Bool, self.localization_ready_topic, self._on_localization_ready, latched_qos
        )

        self.epoch = 0
        self.goal_epoch = 0
        self.active_goal_handle = None
        self.active_goal_source: Optional[str] = None
        self.inflight_goal_ids = set()
        self.cancel_requests = 0
        self.pending_nav_goal: Optional[PoseStamped] = None
        self.pending_nav_source: Optional[str] = None
        self.nav_retry_not_before = -1.0e9
        self.recovery_nav_failures = 0
        self.leader_pose: Optional[PoseStamped] = None
        self.self_pose: Optional[PoseStamped] = None
        self.self_pose_wall: Optional[float] = None
        self.localization_ready = False
        self.last_follow_goal_xy: Optional[tuple[float, float]] = None
        self.last_follow_goal_wall = -1.0e9
        self.recovery_target: Optional[PoseStamped] = None
        self.last_amcl_wall: Optional[float] = None
        self.xy_cov = float('inf')
        self.yaw_cov = float('inf')
        self.last_odom_yaw: Optional[float] = None
        self.last_odom_xy: Optional[tuple[float, float]] = None
        self.nav_start_odom_xy: Optional[tuple[float, float]] = None
        self.movement_started = False
        self.movement_sample_count = 0
        self.accumulated_yaw = 0.0
        self.spin_start_wall = 0.0
        self.spin_command_started = False
        self.spin_motion_detected = False
        self.spin_last_attempt_completed = False
        self.settle_start_wall = 0.0
        self.spin_attempt = 0
        self.spin_direction = 1.0
        self.heartbeat_seq = 0
        self.motion_authority = MotionAuthority.NONE
        self.rl_runtime: Optional[ActiveScoutRLRuntime] = None
        if self.scout_rl_enabled:
            self.rl_runtime = ActiveScoutRLRuntime(
                self,
                self._publish_rl_command,
                on_stop=self._on_rl_stopped,
            )

        self.create_timer(0.1, self._tick)
        self.create_timer(self.heartbeat_period, self._publish_heartbeat)
        self._enter_role(self.role, reason='startup')
        self.get_logger().warning(
            'UNIFIED_FIELD_ROBOT_READY | '
            f'robot={self.robot_name} role={self.role.value} '
            f'nav={self.navigate_action} cmd_vel={self.cmd_vel_topic}'
        )
        if self.scout_rl_enabled:
            self.get_logger().warning(
                '[SCOUT_RL] ROLE_GATED=true DETERMINISTIC=true SDE=false '
                'backend=in_process'
            )

    def _now(self) -> float:
        return self.get_clock().now().nanoseconds * 1.0e-9

    def _on_leader_pose(self, msg: PoseStamped) -> None:
        self.leader_pose = msg

    def _on_self_pose(self, msg: PoseStamped) -> None:
        self.self_pose = msg
        self.self_pose_wall = self._now()

    def _on_localization_ready(self, msg: Bool) -> None:
        previous = self.localization_ready
        self.localization_ready = bool(msg.data)
        if previous and not self.localization_ready:
            if self.role == Role.ACTIVE_SCOUT:
                self._deactivate_rl('localization_not_ready')
                self._set_authority(MotionAuthority.NONE, 'localization_not_ready')
            if self.active_goal_handle is not None or self.inflight_goal_ids:
                self._invalidate_nav_goal(
                    'localization_not_ready', clear_pending=True
                )
        elif self.localization_ready and not previous:
            self.get_logger().warning(
                'FIELD_LOCALIZATION_READY | '
                f'topic={self.localization_ready_topic}'
            )
            if self.role == Role.ACTIVE_SCOUT:
                self._activate_rl()

    def _on_amcl(self, msg: PoseWithCovarianceStamped) -> None:
        cov = msg.pose.covariance
        self.xy_cov = max(abs(float(cov[0])), abs(float(cov[7])))
        self.yaw_cov = abs(float(cov[35]))
        self.last_amcl_wall = self._now()

    def _on_odom(self, msg: Odometry) -> None:
        yaw = yaw_from_quaternion(msg.pose.pose.orientation)
        if self.role == Role.LOCALIZATION_SPIN and self.last_odom_yaw is not None:
            delta = wrap_angle(yaw - self.last_odom_yaw)
            self.accumulated_yaw += self.spin_direction * delta
            if abs(delta) >= 0.005:
                self.spin_motion_detected = True
        self.last_odom_yaw = yaw
        self.last_odom_xy = (
            float(msg.pose.pose.position.x),
            float(msg.pose.pose.position.y),
        )
        if (
            self.motion_authority in (
                MotionAuthority.NORMAL_FOLLOW,
                MotionAuthority.FAILOVER_RECOVERY_NAV,
            )
            and self.nav_start_odom_xy is not None
        ):
            distance = math.hypot(
                self.last_odom_xy[0] - self.nav_start_odom_xy[0],
                self.last_odom_xy[1] - self.nav_start_odom_xy[1],
            )
            if distance >= self.movement_start_distance:
                self.movement_sample_count += 1
                if self.movement_sample_count >= self.movement_start_samples:
                    self.movement_started = True
            else:
                self.movement_sample_count = 0

    def _on_fleet_role(self, msg: String) -> None:
        try:
            data = json.loads(msg.data)
        except json.JSONDecodeError:
            return
        active = str(data.get('active_scout_id', ''))
        epoch = parse_epoch(data.get('epoch'))
        if epoch is None:
            self.get_logger().warning('FLEET_ROLE_BAD_EPOCH')
            return
        if epoch < self.epoch:
            return
        if active != self.robot_name and epoch >= self.epoch and self.role == Role.ACTIVE_SCOUT:
            self.epoch = epoch
            self._enter_role(Role.IDLE, reason='higher_epoch_not_active')

    def _on_role_command(self, msg: String) -> None:
        try:
            data = json.loads(msg.data)
        except json.JSONDecodeError:
            self.get_logger().warning(f'ROLE_COMMAND_BAD_JSON | {msg.data!r}')
            return
        target = str(data.get('robot', ''))
        if target and target != self.robot_name:
            return
        epoch = parse_epoch(data.get('epoch', self.epoch))
        if epoch is None:
            self.get_logger().warning('ROLE_COMMAND_BAD_EPOCH')
            return
        if epoch < self.epoch:
            self.get_logger().warning(
                f'ROLE_COMMAND_OLD_EPOCH | got={epoch} current={self.epoch}'
            )
            return
        role = normalize_role(str(data.get('role', data.get('command', 'IDLE'))))
        self.epoch = epoch
        if role == Role.RECOVERY_NAVIGATING:
            target_pose = self._pose_from_json(data.get('target_pose') or data.get('failure_pose'))
            if target_pose is None:
                self._enter_role(Role.FAILED, reason='recovery_without_target')
                return
            self.recovery_target = target_pose
            self.recovery_nav_failures = 0
            self.nav_retry_not_before = -1.0e9
            self.spin_attempt = 0
            self.spin_last_attempt_completed = False
        self._enter_role(role, reason='role_command')

    def _tick(self) -> None:
        if self.role == Role.FOLLOWER:
            self._tick_follow()
        elif self.role == Role.RECOVERY_NAVIGATING:
            self._tick_recovery()
        elif self.role == Role.LOCALIZATION_CHECK:
            self._tick_localization_check()
        elif self.role == Role.LOCALIZATION_SPIN:
            self._tick_spin()
        elif self.role == Role.LOCALIZATION_SETTLE:
            self._tick_localization_settle()
        self._dispatch_pending_nav_goal()
        self._publish_status()

    def _tick_follow(self) -> None:
        if not self.enable_follow or self.leader_pose is None:
            return
        if self.require_localization_ready and not self.localization_ready:
            # Without this, FOLLOWER starts sending Nav2 goals the instant
            # it has a leader pose, before this robot's own AMCL has
            # converged (global_localize_kickstart's spin state machine
            # hasn't finished/published /localization_ready yet). Nav2
            # then acts on a bad pose estimate, and the reported position
            # "teleports" once AMCL catches up or re-converges elsewhere.
            return
        now = self._now()
        if now - self.last_follow_goal_wall < self.follow_goal_period:
            return
        yaw = yaw_from_quaternion(self.leader_pose.pose.orientation)
        goal = self._copy_pose(self.leader_pose)
        goal.pose.position.x -= self.follow_distance * math.cos(yaw)
        goal.pose.position.y -= self.follow_distance * math.sin(yaw)
        xy = (goal.pose.position.x, goal.pose.position.y)
        if self.last_follow_goal_xy is not None:
            if math.hypot(xy[0] - self.last_follow_goal_xy[0], xy[1] - self.last_follow_goal_xy[1]) < self.follow_update_distance:
                return
        self._queue_nav_goal(goal, source='FOLLOW')
        self.last_follow_goal_xy = xy
        self.last_follow_goal_wall = now

    def _tick_recovery(self) -> None:
        if self.recovery_target is None:
            self._enter_role(Role.FAILED, reason='missing_recovery_target')
            return
        if self._at_pose(self.recovery_target):
            self._enter_role(Role.ARRIVED_AT_FAILURE_POSE, reason='arrival_distance')
            self._enter_role(Role.LOCALIZATION_CHECK, reason='arrived')
            return
        if self._now() < self.nav_retry_not_before:
            return
        if (
            self.active_goal_handle is None
            and not self.inflight_goal_ids
            and self.pending_nav_goal is None
            and self.cancel_requests == 0
        ):
            self._queue_nav_goal(self.recovery_target, source='RECOVERY')

    def _tick_localization_check(self) -> None:
        self.get_logger().warning(
            'FIELD_LOCALIZATION_CHECK | '
            f'xy_cov={self.xy_cov:.4f} yaw_cov={self.yaw_cov:.4f}',
            throttle_duration_sec=2.0,
        )
        if self.spin_attempt > 0 and not self.spin_last_attempt_completed:
            if self.spin_attempt > self.max_spin_retries:
                self._enter_role(Role.FAILED, reason='spin_retries_exhausted')
            else:
                self._start_spin()
            return
        if self._localization_ok():
            self._enter_role(Role.ACTIVE_SCOUT, reason='localization_ok')
            return
        if not self.enable_spin or self.spin_attempt > self.max_spin_retries:
            self._enter_role(Role.FAILED, reason='localization_failed')
            return
        self._start_spin()

    def _start_spin(self) -> None:
        self.spin_attempt += 1
        self.spin_direction = 1.0 if self.spin_attempt % 2 == 1 else -1.0
        self.accumulated_yaw = 0.0
        self.spin_start_wall = 0.0
        self.spin_command_started = False
        self.spin_motion_detected = False
        self.spin_last_attempt_completed = False
        self._enter_role(Role.LOCALIZATION_SPIN, reason='amcl_covariance')

    def _tick_spin(self) -> None:
        if not self._non_rl_motion_quiesced():
            return
        if not self.spin_command_started:
            self.spin_command_started = True
            self.spin_start_wall = self._now()
            self.accumulated_yaw = 0.0
            self.spin_motion_detected = False
            self._set_authority(
                MotionAuthority.LOCALIZATION_SPIN, 'spin_command_started'
            )
        elapsed = self._now() - self.spin_start_wall
        if self.accumulated_yaw >= self.spin_target:
            self._publish_twist(0.0)
            self._set_authority(MotionAuthority.NONE, 'spin_complete')
            self.spin_last_attempt_completed = True
            self.settle_start_wall = self._now()
            self._enter_role(Role.LOCALIZATION_SETTLE, reason='spin_complete')
            return
        if elapsed >= self.spin_timeout:
            self._publish_twist(0.0)
            self._set_authority(MotionAuthority.NONE, 'spin_timeout')
            if not self.spin_motion_detected:
                self.get_logger().error(
                    'SPIN_FAILED_NO_MOTION | '
                    f'cmd_vel={self.cmd_vel_topic} elapsed={elapsed:.2f}s'
                )
            else:
                self.get_logger().error(
                    'SPIN_TIMEOUT | '
                    f'rotated={self.accumulated_yaw:.3f} '
                    f'target={self.spin_target:.3f}'
                )
            self.settle_start_wall = self._now()
            self._enter_role(Role.LOCALIZATION_SETTLE, reason='spin_timeout')
            return
        self._publish_twist(self.spin_direction * self.spin_speed)

    def _tick_localization_settle(self) -> None:
        self._publish_twist(0.0)
        if self._now() - self.settle_start_wall < self.settle_duration:
            return
        self._enter_role(Role.LOCALIZATION_CHECK, reason='settled')

    def _enter_role(self, role: Role, reason: str) -> None:
        if role == self.role and reason != 'startup':
            return
        old = self.role
        if role != Role.FOLLOWER:
            self.last_follow_goal_xy = None
        if old != role:
            self._set_authority(MotionAuthority.NONE, f'release_{old.value}')
            self._invalidate_nav_goal(
                f'role_change_{old.value}_to_{role.value}',
                clear_pending=True,
            )
        if old == Role.ACTIVE_SCOUT and role != Role.ACTIVE_SCOUT:
            self._deactivate_rl(f'role_change_to_{role.value}')
            if self.scout_rl_enabled:
                self.get_logger().warning(f'[SCOUT_RL] DEACTIVATED role={role.value}')
        self.role = role
        if role == Role.ACTIVE_SCOUT:
            if self.scout_rl_enabled:
                self.get_logger().warning('[SCOUT_RL] ACTIVATED role=ACTIVE_SCOUT')
            self._activate_rl()
        self.get_logger().warning(
            f'FIELD_ROLE_TRANSITION | robot={self.robot_name} '
            f'{old.value}->{role.value} reason={reason} epoch={self.epoch}'
        )
        self._publish_status()

    def _queue_nav_goal(self, pose: PoseStamped, source: str) -> None:
        """Keep only the newest goal and release the previous Nav2 owner."""
        self.pending_nav_goal = self._copy_pose(pose)
        self.pending_nav_source = source
        if self.active_goal_handle is not None or self.inflight_goal_ids:
            self._invalidate_nav_goal(
                f'new_{source}_goal', clear_pending=False
            )

    def _dispatch_pending_nav_goal(self) -> None:
        if self.pending_nav_goal is None or self.pending_nav_source is None:
            return
        source = self.pending_nav_source
        if source == 'FOLLOW' and self.role != Role.FOLLOWER:
            self.pending_nav_goal = None
            self.pending_nav_source = None
            return
        if source == 'RECOVERY' and self.role != Role.RECOVERY_NAVIGATING:
            self.pending_nav_goal = None
            self.pending_nav_source = None
            return
        if not self._non_rl_motion_quiesced():
            return
        if source == 'FOLLOW' and self.require_localization_ready and not self.localization_ready:
            return
        if not self.nav_client.server_is_ready():
            self.get_logger().warning(
                f'FIELD_NAV2_WAIT | source={source} action={self.navigate_action}',
                throttle_duration_sec=5.0,
            )
            return
        pose = self.pending_nav_goal
        self.pending_nav_goal = None
        self.pending_nav_source = None
        goal = NavigateToPose.Goal()
        goal.pose = deepcopy(pose)
        goal.pose.header.frame_id = goal.pose.header.frame_id or 'map'
        goal.pose.header.stamp = self.get_clock().now().to_msg()
        self.goal_epoch += 1
        goal_id = self.goal_epoch
        try:
            future = self.nav_client.send_goal_async(goal)
        except Exception as exc:  # noqa: BLE001
            self.get_logger().error(
                f'FIELD_NAV_GOAL_SEND_ERROR | source={source} {exc}'
            )
            self._handle_nav_failure(source, 'send_exception')
            return
        self.inflight_goal_ids.add(goal_id)
        self.nav_start_odom_xy = self.last_odom_xy
        self.movement_started = False
        self.movement_sample_count = 0
        authority = (
            MotionAuthority.NORMAL_FOLLOW
            if source == 'FOLLOW'
            else MotionAuthority.FAILOVER_RECOVERY_NAV
        )
        self._set_authority(authority, f'{source.lower()}_goal_sent')
        future.add_done_callback(partial(self._goal_response_cb, goal_id=goal_id, source=source))
        self.get_logger().warning(
            f'FIELD_NAV_GOAL_SENT | source={source} '
            f'x={goal.pose.pose.position.x:.3f} y={goal.pose.pose.position.y:.3f}'
        )

    def _goal_response_cb(self, future, goal_id: int, source: str) -> None:
        self.inflight_goal_ids.discard(goal_id)
        try:
            handle = future.result()
        except Exception as exc:  # noqa: BLE001
            self.get_logger().warning(f'FIELD_NAV_GOAL_ERROR | source={source} {exc}')
            if goal_id == self.goal_epoch:
                self._set_authority(MotionAuthority.NONE, 'goal_response_error')
                self._handle_nav_failure(source, 'response_exception')
            return
        if goal_id != self.goal_epoch:
            if handle.accepted:
                self._request_cancel(handle, goal_id, 'stale_goal_response')
            return
        if not handle.accepted:
            self.get_logger().warning(f'FIELD_NAV_GOAL_REJECTED | source={source}')
            self._set_authority(MotionAuthority.NONE, 'goal_rejected')
            self._handle_nav_failure(source, 'rejected')
            return
        self.active_goal_handle = handle
        self.active_goal_source = source
        result_future = handle.get_result_async()
        result_future.add_done_callback(partial(self._goal_result_cb, goal_id=goal_id, source=source))

    def _goal_result_cb(self, future, goal_id: int, source: str) -> None:
        try:
            result = future.result()
            status = result.status
        except Exception as exc:  # noqa: BLE001
            status = None
            error = str(exc)
        else:
            error = ''
        if goal_id != self.goal_epoch:
            self.get_logger().info(
                f'STALE_FIELD_NAV_RESULT_IGNORED | source={source} '
                f'goal_id={goal_id} current={self.goal_epoch}'
            )
            return
        self.active_goal_handle = None
        self.active_goal_source = None
        self._set_authority(MotionAuthority.NONE, 'goal_result')
        self.get_logger().warning(f'FIELD_NAV_RESULT | source={source} status={status}')
        if source != 'RECOVERY' or self.role != Role.RECOVERY_NAVIGATING:
            return
        if (
            status == GoalStatus.STATUS_SUCCEEDED
            and self.recovery_target is not None
            and self._at_pose(self.recovery_target)
        ):
            self._enter_role(Role.ARRIVED_AT_FAILURE_POSE, reason='nav2_succeeded_at_pose')
            self._enter_role(Role.LOCALIZATION_CHECK, reason='recovery_arrival_verified')
            return
        reason = error or f'status_{status}'
        if status == GoalStatus.STATUS_SUCCEEDED:
            reason = 'success_without_verified_arrival'
        self._handle_nav_failure(source, reason)

    def _invalidate_nav_goal(self, reason: str, *, clear_pending: bool) -> None:
        self.goal_epoch += 1
        if clear_pending:
            self.pending_nav_goal = None
            self.pending_nav_source = None
        handle = self.active_goal_handle
        self.active_goal_handle = None
        self.active_goal_source = None
        if self.motion_authority in (
            MotionAuthority.NORMAL_FOLLOW,
            MotionAuthority.FAILOVER_RECOVERY_NAV,
        ):
            self._set_authority(MotionAuthority.NONE, reason)
        if handle is None:
            return
        self._request_cancel(handle, self.goal_epoch - 1, reason)

    def _request_cancel(self, handle, goal_id: int, reason: str) -> None:
        try:
            future = handle.cancel_goal_async()
            self.get_logger().warning(f'FIELD_NAV_CANCEL | reason={reason}')
        except Exception as exc:  # noqa: BLE001
            self.get_logger().warning(f'FIELD_NAV_CANCEL_ERROR | reason={reason} {exc}')
            return
        self.cancel_requests += 1
        future.add_done_callback(
            partial(self._cancel_response_cb, goal_id=goal_id, reason=reason)
        )

    def _cancel_response_cb(self, future, goal_id: int, reason: str) -> None:
        self.cancel_requests = max(0, self.cancel_requests - 1)
        try:
            response = future.result()
            count = len(getattr(response, 'goals_canceling', []) or [])
        except Exception as exc:  # noqa: BLE001
            self.get_logger().warning(
                f'FIELD_NAV_CANCEL_ACK_ERROR | goal_id={goal_id} '
                f'reason={reason} {exc}'
            )
            return
        self.get_logger().info(
            f'FIELD_NAV_CANCEL_ACK | goal_id={goal_id} '
            f'reason={reason} goals_canceling={count}'
        )

    def _handle_nav_failure(self, source: str, reason: str) -> None:
        if source != 'RECOVERY' or self.role != Role.RECOVERY_NAVIGATING:
            return
        if self.recovery_nav_failures >= self.max_recovery_nav_retries:
            self._enter_role(Role.FAILED, reason=f'recovery_nav_failed_{reason}')
            return
        self.recovery_nav_failures += 1
        delay = self.recovery_nav_retry_sec * (2 ** (self.recovery_nav_failures - 1))
        self.nav_retry_not_before = self._now() + delay
        self.get_logger().warning(
            'FIELD_RECOVERY_RETRY_SCHEDULED | '
            f'reason={reason} retry={self.recovery_nav_failures}/'
            f'{self.max_recovery_nav_retries} in={delay:.1f}s'
        )

    def _nav_motion_quiesced(self) -> bool:
        return (
            self.active_goal_handle is None
            and not self.inflight_goal_ids
            and self.cancel_requests == 0
        )

    def _non_rl_motion_quiesced(self) -> bool:
        return self._nav_motion_quiesced()

    def _localization_ok(self) -> bool:
        if self.last_amcl_wall is None:
            return False
        if self._now() - self.last_amcl_wall > self.max_amcl_age:
            return False
        return self.xy_cov <= self.max_xy_cov and self.yaw_cov <= self.max_yaw_cov

    def _at_pose(self, target: PoseStamped) -> bool:
        if self.self_pose is None or self.self_pose_wall is None:
            return False
        if self._now() - self.self_pose_wall > self.self_pose_timeout:
            return False
        frame = str(self.self_pose.header.frame_id or '').strip().lstrip('/')
        target_frame = str(target.header.frame_id or 'map').strip().lstrip('/')
        if frame != target_frame:
            return False
        dx = self.self_pose.pose.position.x - target.pose.position.x
        dy = self.self_pose.pose.position.y - target.pose.position.y
        return (
            math.isfinite(dx)
            and math.isfinite(dy)
            and math.hypot(dx, dy) <= self.arrival_tolerance
        )

    def _activate_rl(self) -> None:
        if self.rl_runtime is None or self.role != Role.ACTIVE_SCOUT:
            return
        if self.require_localization_ready and not self.localization_ready:
            return
        if not self._nav_motion_quiesced():
            return
        self.rl_runtime.activate()
        if self.rl_runtime.active:
            self._set_authority(MotionAuthority.ACTIVE_SCOUT_RL, 'rl_activated')

    def _deactivate_rl(self, reason: str) -> None:
        if self.rl_runtime is not None:
            self.rl_runtime.deactivate(reason)

    def _on_rl_stopped(self, reason: str) -> None:
        if self.motion_authority == MotionAuthority.ACTIVE_SCOUT_RL:
            self._set_authority(MotionAuthority.NONE, f'rl_{reason}')

    def _set_authority(self, authority: MotionAuthority, reason: str) -> None:
        if authority == self.motion_authority:
            return
        old = self.motion_authority
        self.motion_authority = authority
        self.get_logger().warning(
            'FIELD_MOTION_AUTHORITY | '
            f'{old.value}->{authority.value} reason={reason} epoch={self.epoch}'
        )

    def _publish_heartbeat(self) -> None:
        if self.role != Role.ACTIVE_SCOUT:
            return
        if self.require_localization_ready and not self.localization_ready:
            return
        if self.scout_rl_enabled and (
            self.motion_authority != MotionAuthority.ACTIVE_SCOUT_RL
            or self.rl_runtime is None
            or not self.rl_runtime.active
        ):
            return
        self.heartbeat_seq += 1
        msg = String()
        msg.data = json.dumps({
            'robot': self.robot_name,
            'role': self.role.value,
            'epoch': self.epoch,
            'seq': self.heartbeat_seq,
            'stamp_sec': self._now(),
        }, sort_keys=True)
        self.heartbeat_pub.publish(msg)

    def _publish_status(self) -> None:
        data = {
            'robot': self.robot_name,
            'epoch': self.epoch,
            'role': self.role.value,
            'status': self.role.value,
            'motion_authority': self.motion_authority.value,
            'cmd_source': self.motion_authority.value,
            'goal_generation': self.goal_epoch,
            'pending_goal_count': 1 if self.pending_nav_goal is not None else 0,
            'movement_started': self.movement_started,
            'xy_cov': None if math.isinf(self.xy_cov) else self.xy_cov,
            'yaw_cov': None if math.isinf(self.yaw_cov) else self.yaw_cov,
        }
        msg = String()
        msg.data = json.dumps(data, sort_keys=True)
        self.status_pub.publish(msg)
        self.legacy_status_pub.publish(msg)
        role_msg = String()
        role_msg.data = self.role.value
        self.role_pub.publish(role_msg)

    def _publish_rl_command(self, linear_x: float, angular_z: float) -> None:
        """The RL runtime may command only the ACTIVE_SCOUT authority.

        A zero command is always permitted so a stale callback cannot leave a
        prior velocity latched while a role transition is in flight.
        """
        if (
            self.role != Role.ACTIVE_SCOUT
            or self.motion_authority != MotionAuthority.ACTIVE_SCOUT_RL
        ):
            if linear_x == 0.0 and angular_z == 0.0:
                self._publish_command(0.0, 0.0)
            return
        self._publish_command(linear_x, angular_z)

    def _publish_twist(self, angular_z: float) -> None:
        self._publish_command(0.0, angular_z)

    def _publish_command(self, linear_x: float, angular_z: float) -> None:
        if self.use_stamped:
            msg = TwistStamped()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = 'base_footprint'
            msg.twist.linear.x = float(linear_x)
            msg.twist.angular.z = angular_z
            self.cmd_pub.publish(msg)
        else:
            msg = Twist()
            msg.linear.x = float(linear_x)
            msg.angular.z = angular_z
            self.cmd_pub.publish(msg)

    def _copy_pose(self, pose: PoseStamped) -> PoseStamped:
        msg = PoseStamped()
        msg.header.frame_id = pose.header.frame_id or 'map'
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.pose = deepcopy(pose.pose)
        return msg

    def _pose_from_json(self, data) -> Optional[PoseStamped]:
        if not isinstance(data, dict):
            return None
        try:
            x = float(data['x'])
            y = float(data['y'])
            yaw = float(data.get('yaw', 0.0))
        except (KeyError, TypeError, ValueError):
            return None
        msg = PoseStamped()
        msg.header.frame_id = str(data.get('frame_id', 'map'))
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.pose.position.x = x
        msg.pose.position.y = y
        qx, qy, qz, qw = quaternion_from_yaw(yaw)
        msg.pose.orientation.x = qx
        msg.pose.orientation.y = qy
        msg.pose.orientation.z = qz
        msg.pose.orientation.w = qw
        return msg

    def destroy_node(self) -> None:
        try:
            self._publish_twist(0.0)
            self._deactivate_rl('node_shutdown')
            self._invalidate_nav_goal('node_shutdown', clear_pending=True)
        except Exception:  # noqa: BLE001
            pass
        super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = UnifiedFieldRobot()
    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
