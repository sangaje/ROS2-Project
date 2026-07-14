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

from .motion_authority import (
    NAV_AUTHORITIES,
    MotionAuthority,
    authority_allows_nonzero,
)
from .nav_goal_manager import NavGoalManager
from .rl_activation_gate import BackendGateInputs, evaluate_backend_activation
from .role_contract import Role, normalize_role, parse_epoch, parse_role_message
from .scout_rl_runtime import ActiveScoutRLRuntime

def yaw_from_quaternion(q) -> float:
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


def quaternion_from_yaw(yaw: float) -> tuple[float, float, float, float]:
    return (0.0, 0.0, math.sin(0.5 * yaw), math.cos(0.5 * yaw))


def wrap_angle(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


class UnifiedFieldRobot(Node):
    def __init__(self) -> None:
        super().__init__('unified_field_robot')

        self.declare_parameter('robot_name', 'field_robot')
        self.declare_parameter('fleet_role', 'member')
        self.declare_parameter('active_scout_robot_name', 'scout22')
        self.declare_parameter('initial_role', 'IDLE')
        self.declare_parameter('enable_follow_mode', True)
        self.declare_parameter('enable_scout_mode', True)
        self.declare_parameter('enable_recovery_mode', True)
        self.declare_parameter('enable_localization_spin', True)
        self.declare_parameter('enable_exploration', True)
        self.declare_parameter('rl_backend', 'external_worker')
        self.declare_parameter('leader_pose_topic', '/leader_pose')
        self.declare_parameter('self_pose_topic', '/burger_pose')
        self.declare_parameter('localization_ready_topic', '/localization_ready')
        self.declare_parameter('require_localization_ready', True)
        self.declare_parameter('require_follow_localization_ready', True)
        self.declare_parameter('system_ready_topic', '/system/ready')
        self.declare_parameter('require_system_ready', False)
        self.declare_parameter('start_motion_topic', '/fleet/start_motion')
        self.declare_parameter('require_start_motion', True)
        self.declare_parameter('role_command_topic', '/fleet/field_robot_role_cmd')
        self.declare_parameter('fleet_role_topic', '/fleet/scout_role')
        self.declare_parameter('status_topic', '/fleet/field_robot_status')
        self.declare_parameter('legacy_takeover_status_topic', '/fleet/scout_takeover_status')
        self.declare_parameter('active_scout_heartbeat_topic', '/scout/signal')
        self.declare_parameter('navigate_action', '/navigate_to_pose')
        self.declare_parameter('cmd_vel_topic', '/cmd_vel')
        self.declare_parameter('external_rl_cmd_topic', '/fleet/active_scout_rl_cmd')
        self.declare_parameter('use_stamped_cmd_vel', True)
        self.declare_parameter('amcl_pose_topic', '/amcl_pose')
        self.declare_parameter('odom_topic', '/odom')
        self.declare_parameter('follow_distance_m', 0.50)
        self.declare_parameter('follow_stop_distance_m', 0.35)
        self.declare_parameter('follow_resume_distance_m', 0.55)
        self.declare_parameter('follow_goal_period_sec', 0.5)
        self.declare_parameter('follow_goal_update_distance_m', 0.10)
        self.declare_parameter('follow_startup_leader_motion_m', 0.30)
        self.declare_parameter('follow_startup_close_distance_m', 0.35)
        self.declare_parameter('follow_startup_timeout_sec', 4.0)
        self.declare_parameter('recovery_arrival_tolerance_m', 0.40)
        self.declare_parameter('self_pose_timeout_sec', 2.0)
        self.declare_parameter('movement_start_distance_m', 0.03)
        self.declare_parameter('movement_start_samples', 3)
        self.declare_parameter('max_recovery_nav_retries', 3)
        self.declare_parameter('recovery_nav_retry_sec', 1.0)
        self.declare_parameter('recovery_stop_linear_epsilon_mps', 0.03)
        self.declare_parameter('recovery_stop_angular_epsilon_radps', 0.08)
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
        self.fleet_role = str(get('fleet_role').value).strip().lower()
        self.active_scout_robot_name = str(get('active_scout_robot_name').value).strip()
        self.role = normalize_role(str(get('initial_role').value))
        self.enable_follow = bool(get('enable_follow_mode').value)
        self.enable_scout = bool(get('enable_scout_mode').value)
        self.enable_recovery = bool(get('enable_recovery_mode').value)
        self.enable_spin = bool(get('enable_localization_spin').value)
        self.enable_exploration = bool(get('enable_exploration').value)
        self.rl_backend = str(get('rl_backend').value).strip().lower()
        if self.rl_backend not in ('disabled', 'in_process', 'external_worker'):
            raise RuntimeError(
                'rl_backend must be one of disabled, in_process, external_worker; '
                f'got {self.rl_backend!r}'
            )
        self.leader_pose_topic = str(get('leader_pose_topic').value)
        self.self_pose_topic = str(get('self_pose_topic').value)
        self.localization_ready_topic = str(get('localization_ready_topic').value)
        self.require_localization_ready = bool(get('require_localization_ready').value)
        self.require_follow_localization_ready = bool(
            get('require_follow_localization_ready').value
        )
        self.system_ready_topic = str(get('system_ready_topic').value)
        self.require_system_ready = bool(get('require_system_ready').value)
        self.start_motion_topic = str(get('start_motion_topic').value)
        self.require_start_motion = bool(get('require_start_motion').value)
        self.role_command_topic = str(get('role_command_topic').value)
        self.fleet_role_topic = str(get('fleet_role_topic').value)
        self.status_topic = str(get('status_topic').value)
        self.legacy_status_topic = str(get('legacy_takeover_status_topic').value)
        self.heartbeat_topic = str(get('active_scout_heartbeat_topic').value)
        self.navigate_action = str(get('navigate_action').value)
        self.cmd_vel_topic = str(get('cmd_vel_topic').value)
        self.external_rl_cmd_topic = str(get('external_rl_cmd_topic').value)
        self.use_stamped = bool(get('use_stamped_cmd_vel').value)
        self.amcl_pose_topic = str(get('amcl_pose_topic').value)
        self.odom_topic = str(get('odom_topic').value)
        self.follow_distance = max(0.1, float(get('follow_distance_m').value))
        self.follow_stop_distance = max(0.05, float(get('follow_stop_distance_m').value))
        self.follow_resume_distance = max(
            self.follow_stop_distance,
            float(get('follow_resume_distance_m').value),
        )
        self.follow_goal_period = max(0.2, float(get('follow_goal_period_sec').value))
        self.follow_update_distance = max(0.05, float(get('follow_goal_update_distance_m').value))
        self.follow_startup_leader_motion = max(
            0.0, float(get('follow_startup_leader_motion_m').value)
        )
        self.follow_startup_close_distance = max(
            0.0, float(get('follow_startup_close_distance_m').value)
        )
        self.follow_startup_timeout = max(
            0.0, float(get('follow_startup_timeout_sec').value)
        )
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
        self.recovery_stop_linear_epsilon = max(
            0.0, float(get('recovery_stop_linear_epsilon_mps').value)
        )
        self.recovery_stop_angular_epsilon = max(
            0.0, float(get('recovery_stop_angular_epsilon_radps').value)
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
        # Capability and execution backend are intentionally separate.  The
        # normal robot stack owns roles/Nav2 here; external_worker owns SAC.
        self.scout_rl_enabled = bool(self.enable_scout and self.enable_exploration)
        self.in_process_rl_enabled = bool(
            self.scout_rl_enabled and self.rl_backend == 'in_process'
        )

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
        if self.scout_rl_enabled and self.rl_backend == 'external_worker':
            self.create_subscription(
                TwistStamped,
                self.external_rl_cmd_topic,
                self._on_external_rl_cmd,
                10,
            )

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
        if self.require_system_ready:
            self.create_subscription(
                Bool, self.system_ready_topic, self._on_system_ready, latched_qos
            )
        if self.require_start_motion:
            self.create_subscription(
                Bool, self.start_motion_topic, self._on_start_motion, latched_qos
            )

        self.epoch = 0
        self._last_role_tuple = (self.role.value, '', self.epoch)
        self.nav_retry_not_before = -1.0e9
        self.recovery_nav_failures = 0
        self.recovery_nav_succeeded = False
        self.recovery_arrived = False
        self.leader_pose: Optional[PoseStamped] = None
        self.leader_pose_wall: Optional[float] = None
        self.self_pose: Optional[PoseStamped] = None
        self.self_pose_wall: Optional[float] = None
        self.localization_ready = False
        self.system_ready = not self.require_system_ready
        self.start_motion = not self.require_start_motion
        self.last_follow_goal_xy: Optional[tuple[float, float]] = None
        self.last_follow_goal_wall = -1.0e9
        self.pending_follow_goal_xy: Optional[tuple[float, float]] = None
        self.pending_follow_goal_wall = -1.0e9
        self.follow_goal_pending = False
        self.follow_goal_epoch = 0
        self.follow_goal_handle = None
        self.first_follow_leader_xy: Optional[tuple[float, float]] = None
        self.first_follow_wall: Optional[float] = None
        self.follow_startup_released = False
        self.recovery_target: Optional[PoseStamped] = None
        self.last_amcl_wall: Optional[float] = None
        self.xy_cov = float('inf')
        self.yaw_cov = float('inf')
        self.last_odom_yaw: Optional[float] = None
        self.last_odom_xy: Optional[tuple[float, float]] = None
        self.last_linear_speed = 0.0
        self.last_angular_speed = 0.0
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
        self.nav = NavGoalManager(
            self.nav_client,
            now_stamp=lambda: self.get_clock().now().to_msg(),
            copy_pose=self._copy_pose,
            log=self.get_logger(),
            set_authority=self._set_authority,
            current_authority=lambda: self.motion_authority,
            on_goal_sent=self._on_nav_goal_sent,
            on_failure=self._handle_nav_failure,
            on_result=self._on_nav_result,
        )
        self.rl_runtime: Optional[ActiveScoutRLRuntime] = None
        if self.in_process_rl_enabled:
            self.rl_runtime = ActiveScoutRLRuntime(
                self,
                self._publish_rl_command,
                on_stop=self._on_rl_stopped,
                on_ready=self._on_rl_ready,
            )

        self.create_timer(0.1, self._tick)
        self.create_timer(self.heartbeat_period, self._publish_heartbeat)
        self._enter_role(self.role, reason='startup')
        self.get_logger().warning(
            'UNIFIED_FIELD_ROBOT_READY | '
            f'robot={self.robot_name} role={self.role.value} '
            f'nav={self.navigate_action} cmd_vel={self.cmd_vel_topic} '
            f'rl_backend={self.rl_backend} '
            f'system_gate={self.require_system_ready}:{self.system_ready_topic} '
            f'start_motion_gate={self.require_start_motion}:{self.start_motion_topic}'
        )
        if self.in_process_rl_enabled:
            self.get_logger().warning(
                '[SCOUT_RL] ROLE_GATED=true DETERMINISTIC=true SDE=false '
                'backend=in_process'
            )
        elif self.scout_rl_enabled and self.rl_backend == 'external_worker':
            self.get_logger().warning(
                '[SCOUT_RL] ROLE_GATED=true backend=external_worker '
                'runtime=not_created_in_unified_field_robot'
            )

    def _now(self) -> float:
        return self.get_clock().now().nanoseconds * 1.0e-9

    def _on_leader_pose(self, msg: PoseStamped) -> None:
        self.leader_pose = msg
        self.leader_pose_wall = self._now()
        self._log_pose_pipeline('leader', self.leader_pose_topic, msg, self.leader_pose_wall)

    def _on_self_pose(self, msg: PoseStamped) -> None:
        self.self_pose = msg
        self.self_pose_wall = self._now()
        self._log_pose_pipeline('self', self.self_pose_topic, msg, self.self_pose_wall)

    def _stamp_age_ms(self, msg: PoseStamped) -> float:
        stamp = msg.header.stamp
        stamp_sec = float(stamp.sec) + float(stamp.nanosec) * 1.0e-9
        if stamp_sec <= 0.0:
            return -1.0
        return max(0.0, (self._now() - stamp_sec) * 1000.0)

    def _log_pose_pipeline(
        self,
        name: str,
        topic: str,
        msg: PoseStamped,
        received_wall: float,
    ) -> None:
        receive_age_ms = max(0.0, (self._now() - received_wall) * 1000.0)
        self.get_logger().warning(
            'POSE_PIPELINE | '
            f'node=unified_field_robot robot={self.robot_name} '
            f'name={name} topic={topic} '
            f'frame_id={msg.header.frame_id or "(empty)"} '
            f'source_stamp_age_ms={self._stamp_age_ms(msg):.0f} '
            f'receive_age_ms={receive_age_ms:.0f}',
            throttle_duration_sec=3.0,
        )
        leader_age_ms = (
            -1.0 if self.leader_pose_wall is None
            else max(0.0, (self._now() - self.leader_pose_wall) * 1000.0)
        )
        self_age_ms = (
            -1.0 if self.self_pose_wall is None
            else max(0.0, (self._now() - self.self_pose_wall) * 1000.0)
        )
        self.get_logger().warning(
            'FOLLOWER_POSE_INPUT | '
            f'leader_pose_rx={self.leader_pose is not None} '
            f'leader_pose_age_ms={leader_age_ms:.0f} '
            f'leader_pose_frame={self.leader_pose.header.frame_id if self.leader_pose else ""} '
            f'self_pose_rx={self.self_pose is not None} '
            f'self_pose_age_ms={self_age_ms:.0f} '
            f'self_pose_frame={self.self_pose.header.frame_id if self.self_pose else ""} '
            'blocking_reason=none',
            throttle_duration_sec=1.0,
        )

    def _on_localization_ready(self, msg: Bool) -> None:
        previous = self.localization_ready
        self.localization_ready = bool(msg.data)
        if previous and not self.localization_ready:
            if self.role == Role.ACTIVE_SCOUT:
                self._deactivate_rl('localization_not_ready')
                self._set_authority(MotionAuthority.NONE, 'localization_not_ready')
            if self.nav.active_goal_count:
                self._invalidate_nav_goal(
                    'localization_not_ready', clear_pending=True
                )
        elif self.localization_ready and not previous:
            self.get_logger().warning(
                'FIELD_LOCALIZATION_READY | '
                f'topic={self.localization_ready_topic}'
            )
            self.get_logger().warning(
                f'SCOUT_LOCALIZATION_READY | robot={self.robot_name}'
            )
            if self.role == Role.ACTIVE_SCOUT:
                self._activate_rl()

    def _on_system_ready(self, msg: Bool) -> None:
        previous = self.system_ready
        self.system_ready = bool(msg.data)
        if previous and not self.system_ready:
            self._deactivate_rl('system_not_ready')
            self._invalidate_nav_goal('system_not_ready', clear_pending=True)
            self._set_authority(MotionAuthority.NONE, 'system_not_ready')
            self._publish_command(0.0, 0.0)
        if self.system_ready != previous:
            self.get_logger().warning(
                'FIELD_SYSTEM_READY | '
                f'robot={self.robot_name} ready={self.system_ready} '
                f'topic={self.system_ready_topic}'
            )

    def _on_start_motion(self, msg: Bool) -> None:
        previous = self.start_motion
        self.start_motion = bool(msg.data)
        if previous and not self.start_motion:
            self._deactivate_rl('start_motion_false')
            self._invalidate_nav_goal('start_motion_false', clear_pending=True)
            self._set_authority(MotionAuthority.NONE, 'start_motion_false')
            self._reset_follow_goal_memory()
            self._publish_command(0.0, 0.0)
        elif self.start_motion and not previous:
            self._reset_follow_goal_memory()
            if self.role == Role.ACTIVE_SCOUT:
                self._activate_rl()
        if self.start_motion != previous:
            self.get_logger().warning(
                'FIELD_START_MOTION | '
                f'robot={self.robot_name} ready={self.start_motion} '
                f'topic={self.start_motion_topic}'
            )

    def _on_amcl(self, msg: PoseWithCovarianceStamped) -> None:
        cov = msg.pose.covariance
        self.xy_cov = max(abs(float(cov[0])), abs(float(cov[7])))
        self.yaw_cov = abs(float(cov[35]))
        self.last_amcl_wall = self._now()

    def _on_odom(self, msg: Odometry) -> None:
        yaw = yaw_from_quaternion(msg.pose.pose.orientation)
        self.last_linear_speed = abs(float(msg.twist.twist.linear.x))
        self.last_angular_speed = abs(float(msg.twist.twist.angular.z))
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
            self.motion_authority in NAV_AUTHORITIES
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

    def _on_external_rl_cmd(self, msg: TwistStamped) -> None:
        self._publish_rl_command(
            float(msg.twist.linear.x),
            float(msg.twist.angular.z),
        )

    def _on_fleet_role(self, msg: String) -> None:
        update = parse_role_message(msg.data, self.robot_name)
        if update is None:
            return
        active = update.active_scout_id or ''
        epoch = update.epoch
        if epoch is None:
            self.get_logger().warning('FLEET_ROLE_BAD_EPOCH')
            return
        if epoch < self.epoch:
            return
        new_role = Role.ACTIVE_SCOUT if active == self.robot_name else Role.IDLE
        next_tuple = (new_role.value, active, epoch)
        if next_tuple == self._last_role_tuple:
            return
        self.get_logger().info(
            'ROLE_STATE | '
            f'old={self.role.value} new='
            f'{new_role.value} '
            f'reason=fleet_role_update robot_name={self.robot_name} '
            f'active_scout_robot_name={active or self.active_scout_robot_name} '
            f'fleet_role={self.fleet_role}',
            throttle_duration_sec=2.0,
        )
        self._last_role_tuple = next_tuple
        if active != self.robot_name and epoch >= self.epoch and self.role == Role.ACTIVE_SCOUT:
            self.epoch = epoch
            self._enter_role(Role.IDLE, reason='higher_epoch_not_active')

    def _on_role_command(self, msg: String) -> None:
        if not str(msg.data).strip().startswith('{'):
            self.get_logger().warning(f'ROLE_COMMAND_BAD_JSON | {msg.data!r}')
            return
        update = parse_role_message(msg.data, self.robot_name)
        if update is None:
            self.get_logger().warning(f'ROLE_COMMAND_BAD_JSON | {msg.data!r}')
            return
        if update.robot and update.robot != self.robot_name:
            return
        epoch = self.epoch if update.epoch is None else update.epoch
        if epoch is None:
            self.get_logger().warning('ROLE_COMMAND_BAD_EPOCH')
            return
        if epoch < self.epoch:
            self.get_logger().warning(
                f'ROLE_COMMAND_OLD_EPOCH | got={epoch} current={self.epoch}'
            )
            return
        role = update.role
        next_tuple = (role.value, update.active_scout_id or '', epoch)
        if next_tuple == self._last_role_tuple:
            return
        self._last_role_tuple = next_tuple
        self.epoch = epoch
        if role == Role.RECOVERY_NAVIGATING:
            self._cancel_follow_goal('recovery_role_command')
            self.nav.invalidate('recovery_role_command', clear_pending=True)
            target_pose = self._pose_from_json(
                update.payload.get('target_pose') or update.payload.get('failure_pose')
            )
            if target_pose is None:
                self._enter_role(Role.FAILED, reason='recovery_without_target')
                return
            self.recovery_target = target_pose
            self.recovery_nav_failures = 0
            self.recovery_nav_succeeded = False
            self.recovery_arrived = False
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
        elif self.role == Role.ACTIVE_SCOUT:
            self._activate_rl()
        self._dispatch_pending_nav_goal()
        self._publish_status()

    def _tick_follow(self) -> None:
        if not self.enable_follow:
            return
        now = self._now()
        if getattr(self, 'require_system_ready', False) and not getattr(
            self, 'system_ready', True
        ):
            self._log_follow_gate('system_not_ready')
            return
        if self.leader_pose is None:
            self._log_follow_gate('leader_pose_missing')
            return
        if self.leader_pose_wall is None or now - self.leader_pose_wall > self.self_pose_timeout:
            self._log_follow_gate('leader_pose_stale')
            return
        leader_frame = str(self.leader_pose.header.frame_id or '').strip().lstrip('/')
        if leader_frame != 'map':
            self._log_follow_gate('leader_pose_stale')
            return
        if self.self_pose is None or self.self_pose_wall is None:
            self._log_follow_gate('self_pose_missing')
            return
        if now - self.self_pose_wall > self.self_pose_timeout:
            self._log_follow_gate('self_pose_stale')
            return
        self_frame = str(self.self_pose.header.frame_id or '').strip().lstrip('/')
        if self_frame != 'map':
            self._log_follow_gate('self_pose_stale')
            return
        if self.require_follow_localization_ready and not self.localization_ready:
            self._log_follow_gate('localization_not_ready')
            return
        if not self.nav_client.server_is_ready():
            self._log_follow_gate('nav_server_unavailable')
            return
        if now - self.last_follow_goal_wall < self.follow_goal_period:
            return
        leader_distance = math.hypot(
            self.leader_pose.pose.position.x - self.self_pose.pose.position.x,
            self.leader_pose.pose.position.y - self.self_pose.pose.position.y,
        )
        if leader_distance <= self.follow_stop_distance:
            self._log_follow_gate('close_to_leader', leader_distance=leader_distance)
            return
        yaw = yaw_from_quaternion(self.leader_pose.pose.orientation)
        goal = self._copy_pose(self.leader_pose)
        goal.pose.position.x -= self.follow_distance * math.cos(yaw)
        goal.pose.position.y -= self.follow_distance * math.sin(yaw)
        xy = (goal.pose.position.x, goal.pose.position.y)
        if self.last_follow_goal_xy is not None:
            if math.hypot(xy[0] - self.last_follow_goal_xy[0], xy[1] - self.last_follow_goal_xy[1]) < self.follow_update_distance:
                self._log_follow_debug(
                    'goal_not_changed',
                    leader_distance=leader_distance,
                    goal_required=False,
                    goal_sent=False,
                    target_xy=xy,
                )
                return
        self._log_follow_debug(
            'goal_sent',
            leader_distance=leader_distance,
            goal_required=True,
            goal_sent=True,
            target_xy=xy,
        )
        self._send_follow_nav2_goal(goal, xy)

    def _log_follow_gate(
        self,
        reason: str,
        *,
        leader_distance: float = float('nan'),
        leader_moved: float = float('nan'),
        startup_elapsed: float = float('nan'),
    ) -> None:
        self.get_logger().warning(
            self._follow_debug_text(
                reason,
                leader_distance=leader_distance,
                leader_moved=leader_moved,
                startup_elapsed=startup_elapsed,
                goal_required=False,
                goal_sent=False,
            ),
            throttle_duration_sec=3.0,
        )

    def _log_follow_debug(
        self,
        reason: str,
        *,
        leader_distance: float = float('nan'),
        goal_required: bool,
        goal_sent: bool,
        target_xy: Optional[tuple[float, float]] = None,
    ) -> None:
        self.get_logger().warning(
            self._follow_debug_text(
                reason,
                leader_distance=leader_distance,
                goal_required=goal_required,
                goal_sent=goal_sent,
                target_xy=target_xy,
            ),
            throttle_duration_sec=1.0,
        )

    def _follow_debug_text(
        self,
        reason: str,
        *,
        leader_distance: float = float('nan'),
        leader_moved: float = float('nan'),
        startup_elapsed: float = float('nan'),
        goal_required: bool,
        goal_sent: bool,
        target_xy: Optional[tuple[float, float]] = None,
    ) -> str:
        leader_age = (
            -1.0 if self.leader_pose is None
            else max(0.0, self._now() - getattr(self, 'leader_pose_wall', self._now()))
        )
        self_age = (
            -1.0 if self.self_pose_wall is None
            else max(0.0, self._now() - self.self_pose_wall)
        )
        if math.isnan(leader_distance) and self.leader_pose is not None and self.self_pose is not None:
            leader_distance = math.hypot(
                self.leader_pose.pose.position.x - self.self_pose.pose.position.x,
                self.leader_pose.pose.position.y - self.self_pose.pose.position.y,
            )
        cmd_age = -1.0
        if getattr(self, 'movement_started', False):
            cmd_age = 0.0
        target_x = float('nan')
        target_y = float('nan')
        if target_xy is not None:
            target_x, target_y = target_xy
        return (
            'FOLLOWER_FOLLOW_DEBUG | FOLLOW_DEBUG | '
            f'robot={self.robot_name} role={self.role.value} '
            f'start_motion_ignored={getattr(self, "require_start_motion", False)} '
            f'leader_pose_rx={self.leader_pose is not None} '
            f'leader_pose_age_ms={leader_age * 1000.0:.0f} '
            f'self_pose_rx={self.self_pose is not None} '
            f'self_pose_age_ms={self_age * 1000.0:.0f} '
            f'localization_ready={self.localization_ready} '
            f'system_ready={getattr(self, "system_ready", True)} '
            f'nav_server_ready={self.nav_client.server_is_ready()} '
            f'distance_to_leader={leader_distance:.3f} '
            f'target_x={target_x:.3f} '
            f'target_y={target_y:.3f} '
            f'leader_moved={leader_moved:.3f} '
            f'startup_elapsed={startup_elapsed:.2f} '
            f'goal_required={goal_required} goal_sent={goal_sent} '
            f'goal_accepted={self.follow_goal_handle is not None} '
            'path_received=unknown '
            f'goal_pending={self.follow_goal_pending} '
            f'active_goals={1 if self.follow_goal_handle is not None else 0} '
            f'controller_cmd_age_ms={cmd_age * 1000.0 if cmd_age >= 0.0 else -1.0:.0f} '
            f'odom_motion={self.movement_started} '
            f'blocking_reason={reason}'
        )

    def _tick_recovery(self) -> None:
        if not self._start_motion_allowed():
            return
        if getattr(self, 'require_system_ready', False) and not getattr(
            self, 'system_ready', True
        ):
            return
        if self.recovery_target is None:
            self._enter_role(Role.FAILED, reason='missing_recovery_target')
            return
        if self._recovery_arrival_verified(require_nav_result=True):
            self._mark_recovery_arrived('nav_result_and_pose')
            return
        if self._now() < self.nav_retry_not_before:
            return
        if (
            self.nav.is_idle
            and not self.nav.has_pending_goal
        ):
            self._queue_nav_goal(self.recovery_target, source='RECOVERY')

    def _tick_localization_check(self) -> None:
        self.get_logger().warning(
            'FIELD_LOCALIZATION_CHECK | '
            f'localization_ready={self.localization_ready} '
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
            if not self._nav_motion_quiesced():
                self.get_logger().warning(
                    'SCOUT_WAIT_MOTION_RELEASE | source=localization_check',
                    throttle_duration_sec=2.0,
                )
                return
            self._publish_command(0.0, 0.0)
            self._set_authority(MotionAuthority.NONE, 'active_scout_takeover_ready')
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
        if not self._start_motion_allowed():
            self._publish_twist(0.0)
            return
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
            self._cancel_follow_goal(f'role_change_to_{role.value}')
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
        self._last_role_tuple = (role.value, self.active_scout_robot_name, self.epoch)
        if role == Role.ACTIVE_SCOUT:
            if self.scout_rl_enabled:
                self.get_logger().warning('[SCOUT_RL] ACTIVATED role=ACTIVE_SCOUT')
            self._activate_rl()
        self.get_logger().warning(
            'ROLE_STATE | '
            f'old={old.value} new={role.value} reason={reason} '
            f'robot_name={self.robot_name} '
            f'active_scout_robot_name={self.active_scout_robot_name} '
            f'fleet_role={self.fleet_role} epoch={self.epoch}'
        )
        self._publish_status()

    def _queue_nav_goal(self, pose: PoseStamped, source: str) -> None:
        if not self._start_motion_allowed():
            return
        self.nav.request_goal(pose, source)

    def _send_follow_nav2_goal(
        self,
        pose: PoseStamped,
        xy: tuple[float, float],
    ) -> None:
        goal = NavigateToPose.Goal()
        goal.pose = self._copy_pose(pose)
        goal.pose.header.frame_id = goal.pose.header.frame_id or 'map'
        goal.pose.header.stamp = self.get_clock().now().to_msg()
        self.follow_goal_epoch += 1
        goal_id = self.follow_goal_epoch
        self.follow_goal_pending = True
        self.pending_follow_goal_xy = xy
        self.pending_follow_goal_wall = self._now()
        self.last_follow_goal_xy = xy
        self.last_follow_goal_wall = self.pending_follow_goal_wall
        self._set_authority(MotionAuthority.NORMAL_FOLLOW, 'follow_goal_sent')
        try:
            future = self.nav_client.send_goal_async(goal)
        except Exception as exc:  # noqa: BLE001
            self.follow_goal_pending = False
            self._set_authority(MotionAuthority.NONE, 'follow_goal_send_error')
            self.get_logger().error(
                f'FOLLOW_NAV2_DIRECT_SEND_ERROR | action={self.navigate_action} error={exc}'
            )
            return
        future.add_done_callback(
            lambda fut, goal_id=goal_id: self._on_follow_goal_response(fut, goal_id)
        )
        self.get_logger().warning(
            'FOLLOW_NAV2_DIRECT_GOAL_SENT | '
            f'robot={self.robot_name} action={self.navigate_action} '
            f'goal_id={goal_id} x={xy[0]:.3f} y={xy[1]:.3f}'
        )

    def _on_follow_goal_response(self, future, goal_id: int) -> None:
        if goal_id == self.follow_goal_epoch:
            self.follow_goal_pending = False
        try:
            handle = future.result()
        except Exception as exc:  # noqa: BLE001
            if goal_id == self.follow_goal_epoch:
                self._set_authority(MotionAuthority.NONE, 'follow_goal_response_error')
                self._reset_follow_goal_memory()
            self.get_logger().warning(
                f'FOLLOW_NAV2_DIRECT_GOAL_ERROR | action={self.navigate_action} error={exc}'
            )
            return
        if goal_id != self.follow_goal_epoch or self.role != Role.FOLLOWER:
            if handle.accepted:
                try:
                    handle.cancel_goal_async()
                except Exception as exc:  # noqa: BLE001
                    self.get_logger().warning(
                        f'FOLLOW_NAV2_DIRECT_STALE_CANCEL_ERROR | error={exc}'
                    )
            return
        if not handle.accepted:
            self._set_authority(MotionAuthority.NONE, 'follow_goal_rejected')
            self._reset_follow_goal_memory()
            self.get_logger().warning(
                f'FOLLOW_NAV2_DIRECT_GOAL_REJECTED | action={self.navigate_action}'
            )
            return
        self.follow_goal_handle = handle
        self.get_logger().warning(
            f'FOLLOW_NAV2_DIRECT_GOAL_ACCEPTED | robot={self.robot_name} goal_id={goal_id}'
        )
        handle.get_result_async().add_done_callback(
            lambda fut, goal_id=goal_id: self._on_follow_goal_result(fut, goal_id)
        )

    def _on_follow_goal_result(self, future, goal_id: int) -> None:
        status = None
        error = ''
        try:
            result = future.result()
            status = result.status
        except Exception as exc:  # noqa: BLE001
            error = str(exc)
        if goal_id != self.follow_goal_epoch:
            return
        self.follow_goal_handle = None
        if status == GoalStatus.STATUS_SUCCEEDED:
            self._set_authority(MotionAuthority.NONE, 'follow_goal_succeeded')
            return
        self._set_authority(MotionAuthority.NONE, 'follow_goal_finished')
        self._reset_follow_goal_memory()
        self.get_logger().warning(
            'FOLLOW_NAV2_DIRECT_RESULT | '
            f'robot={self.robot_name} status={status} error={error}'
        )

    def _cancel_follow_goal(self, reason: str) -> None:
        self.follow_goal_epoch += 1
        self.follow_goal_pending = False
        handle = self.follow_goal_handle
        self.follow_goal_handle = None
        if handle is not None:
            try:
                handle.cancel_goal_async()
            except Exception as exc:  # noqa: BLE001
                self.get_logger().warning(
                    f'FOLLOW_NAV2_DIRECT_CANCEL_ERROR | reason={reason} error={exc}'
                )
        self._reset_follow_goal_memory()

    def _dispatch_pending_nav_goal(self) -> None:
        self.nav.dispatch(
            source_allowed=self._nav_source_allowed,
            can_send=self._non_rl_motion_quiesced,
            action_name=self.navigate_action,
        )

    def _nav_source_allowed(self, source: str) -> bool:
        if not self._start_motion_allowed():
            return False
        if getattr(self, 'require_system_ready', False) and not getattr(
            self, 'system_ready', True
        ):
            return False
        if source == 'FOLLOW':
            return self.role == Role.FOLLOWER and (
                not self.require_follow_localization_ready or self.localization_ready
            )
        return source == 'RECOVERY' and self.role == Role.RECOVERY_NAVIGATING

    def _on_nav_goal_sent(self, source: str) -> None:
        self.nav_start_odom_xy = self.last_odom_xy
        self.movement_started = False
        self.movement_sample_count = 0
        if source == 'FOLLOW' and self.pending_follow_goal_xy is not None:
            self._log_follow_debug(
                'goal_sent',
                goal_required=True,
                goal_sent=True,
            )
            self.last_follow_goal_xy = self.pending_follow_goal_xy
            self.last_follow_goal_wall = self.pending_follow_goal_wall
            self.pending_follow_goal_xy = None

    def _on_nav_result(self, source: str, status, error: str) -> None:
        if source == 'FOLLOW':
            if status != GoalStatus.STATUS_SUCCEEDED:
                self._reset_follow_goal_memory()
            return
        if source != 'RECOVERY' or self.role != Role.RECOVERY_NAVIGATING:
            return
        if status == GoalStatus.STATUS_SUCCEEDED:
            self.recovery_nav_succeeded = True
        if (
            status == GoalStatus.STATUS_SUCCEEDED
            and self.recovery_target is not None
            and self._recovery_arrival_verified(require_nav_result=True)
        ):
            self._mark_recovery_arrived('recovery_arrival_verified')
            return
        reason = error or f'status_{status}'
        if status == GoalStatus.STATUS_SUCCEEDED:
            reason = 'success_without_verified_arrival'
        self._handle_nav_failure(source, reason)

    def _invalidate_nav_goal(self, reason: str, *, clear_pending: bool) -> None:
        self.nav.invalidate(reason, clear_pending=clear_pending)

    def _handle_nav_failure(self, source: str, reason: str) -> None:
        if source == 'FOLLOW':
            self._reset_follow_goal_memory()
            self.get_logger().warning(
                f'FOLLOW_GOAL_RETRY_READY | reason={reason}'
            )
            return
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
        self.nav.recover_timeouts()
        return self.nav.is_idle

    def _non_rl_motion_quiesced(self) -> bool:
        return self._nav_motion_quiesced()

    def _localization_ok(self) -> bool:
        if not self.require_localization_ready:
            return True
        if not self.localization_ready:
            return False
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

    def _recovery_motion_stopped(self) -> bool:
        return (
            self.last_linear_speed <= self.recovery_stop_linear_epsilon
            and self.last_angular_speed <= self.recovery_stop_angular_epsilon
        )

    def _recovery_arrival_verified(self, *, require_nav_result: bool) -> bool:
        if self.recovery_target is None:
            return False
        if require_nav_result and not self.recovery_nav_succeeded:
            return False
        return (
            self._at_pose(self.recovery_target)
            and self._nav_motion_quiesced()
            and self.movement_started
            and self._recovery_motion_stopped()
        )

    def _mark_recovery_arrived(self, reason: str) -> None:
        if self.recovery_arrived:
            return
        self.recovery_arrived = True
        distance = float('nan')
        if self.self_pose is not None and self.recovery_target is not None:
            dx = self.self_pose.pose.position.x - self.recovery_target.pose.position.x
            dy = self.self_pose.pose.position.y - self.recovery_target.pose.position.y
            distance = math.hypot(dx, dy)
        self.get_logger().warning(
            'SCOUT_RECOVERY_ARRIVED | '
            f'robot={self.robot_name} distance={distance:.3f} reason={reason}'
        )
        self._enter_role(Role.ARRIVED_AT_FAILURE_POSE, reason=reason)
        self._enter_role(Role.LOCALIZATION_CHECK, reason='recovery_arrival_verified')

    def _activate_rl(self) -> None:
        if not self._start_motion_allowed():
            self._deactivate_rl('start_motion_false')
            return
        allowed, reason = evaluate_backend_activation(BackendGateInputs(
            role=self.role,
            scout_enabled=self.scout_rl_enabled,
            require_localization_ready=self.require_localization_ready,
            localization_ready=self.localization_ready and (
                getattr(self, 'system_ready', True)
                or not getattr(self, 'require_system_ready', False)
            ),
            nav_idle=self._nav_motion_quiesced(),
        ))
        if not allowed:
            if reason != 'localization_not_ready':
                return
            self.get_logger().warning(
                'SCOUT_RL_WAIT_LOCALIZATION | '
                f'topic={self.localization_ready_topic}',
                throttle_duration_sec=5.0,
            )
            return
        if self.rl_backend == 'external_worker' and self.scout_rl_enabled:
            self._set_authority(
                MotionAuthority.ACTIVE_SCOUT_RL,
                'external_worker_motion_release',
            )
            return
        if self.rl_runtime is None:
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

    def _on_rl_ready(self) -> None:
        """Grant command authority only after asynchronous model loading completes."""
        if (
            self.role == Role.ACTIVE_SCOUT
            and self.rl_runtime is not None
            and self.rl_runtime.active
            and (not self.require_localization_ready or self.localization_ready)
            and self._nav_motion_quiesced()
        ):
            self._set_authority(MotionAuthority.ACTIVE_SCOUT_RL, 'rl_model_ready')

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
        if self.in_process_rl_enabled and (
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

    def _recovery_complete_for_role(self) -> bool:
        return bool(
            self.recovery_arrived
            or (
                self.role == Role.ACTIVE_SCOUT
                and self.recovery_target is None
                and self.epoch == 0
            )
        )

    def _active_scout_ready_status(self) -> bool:
        return bool(
            self.role == Role.ACTIVE_SCOUT
            and self._localization_ok()
            and self._nav_motion_quiesced()
            and self._recovery_complete_for_role()
            and self.motion_authority in (
                MotionAuthority.NONE,
                MotionAuthority.ACTIVE_SCOUT_RL,
            )
        )

    def _publish_status(self) -> None:
        runtime = self.rl_runtime
        active_goal_count = self.nav.active_goal_count
        nav_goal_active = bool(
            active_goal_count > 0
            or self.nav.has_pending_goal
            or self.nav.cancel_requests > 0
        )
        recovery_complete = self._recovery_complete_for_role()
        active_scout_ready = self._active_scout_ready_status()
        status = 'ACTIVE_SCOUT_READY' if active_scout_ready else self.role.value
        data = {
            'robot': self.robot_name,
            'epoch': self.epoch,
            'role': self.role.value,
            'status': status,
            'active_scout_id': (
                self.robot_name if self.role == Role.ACTIVE_SCOUT else ''
            ),
            'motion_authority': self.motion_authority.value,
            'cmd_source': self.motion_authority.value,
            'goal_generation': self.nav.goal_epoch,
            'pending_goal_count': 1 if self.nav.has_pending_goal else 0,
            'active_goal_count': active_goal_count,
            'nav_goal_active': nav_goal_active,
            'movement_started': self.movement_started,
            'localization_ready': self.localization_ready,
            'localization_ok': self._localization_ok(),
            'requires_localization_ready': self.require_localization_ready,
            'system_ready': getattr(self, 'system_ready', True),
            'start_motion': getattr(self, 'start_motion', True),
            'nav_server_ready': self.nav_client.server_is_ready(),
            'recovery_arrived': self.recovery_arrived,
            'recovery_complete': recovery_complete,
            'active_scout_ready': active_scout_ready,
            'xy_cov': None if math.isinf(self.xy_cov) else self.xy_cov,
            'yaw_cov': None if math.isinf(self.yaw_cov) else self.yaw_cov,
            'rl_enabled': self.scout_rl_enabled,
            'rl_backend': self.rl_backend,
            'rl_active': bool(
                (runtime is not None and runtime.active)
                or (
                    self.rl_backend == 'external_worker'
                    and self.motion_authority == MotionAuthority.ACTIVE_SCOUT_RL
                )
            ),
            'rl_stop_reason': (
                runtime.last_stop_reason if runtime is not None else self.rl_backend
            ),
        }
        msg = String()
        msg.data = json.dumps(data, sort_keys=True)
        self.status_pub.publish(msg)
        self.legacy_status_pub.publish(msg)
        role_msg = String()
        role_msg.data = json.dumps({
            'robot': self.robot_name,
            'epoch': self.epoch,
            'role': self.role.value,
            'active_scout_id': (
                self.robot_name if self.role == Role.ACTIVE_SCOUT else ''
            ),
            'localization_ready': self.localization_ready,
            'localization_ok': self._localization_ok(),
            'requires_localization_ready': self.require_localization_ready,
            'recovery_complete': recovery_complete,
            'active_scout_ready': active_scout_ready,
            'motion_authority': self.motion_authority.value,
            'start_motion': getattr(self, 'start_motion', True),
        }, sort_keys=True)
        self.role_pub.publish(role_msg)

    def _publish_rl_command(self, linear_x: float, angular_z: float) -> None:
        """The RL runtime may command only the ACTIVE_SCOUT authority.

        A zero command is always permitted so a stale callback cannot leave a
        prior velocity latched while a role transition is in flight.
        """
        if (
            self.role != Role.ACTIVE_SCOUT
            or not self._start_motion_allowed()
            or not authority_allows_nonzero(
                self.motion_authority, MotionAuthority.ACTIVE_SCOUT_RL
            )
        ):
            if linear_x == 0.0 and angular_z == 0.0:
                self._publish_command(0.0, 0.0)
            return
        self._publish_command(linear_x, angular_z)

    def _publish_twist(self, angular_z: float) -> None:
        self._publish_command(0.0, angular_z)

    def _publish_command(self, linear_x: float, angular_z: float) -> None:
        if not self._start_motion_allowed() and (
            abs(float(linear_x)) > 1.0e-9 or abs(float(angular_z)) > 1.0e-9
        ):
            linear_x = 0.0
            angular_z = 0.0
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

    def _start_motion_allowed(self) -> bool:
        return bool(
            not getattr(self, 'require_start_motion', False)
            or getattr(self, 'start_motion', True)
        )

    def _reset_follow_goal_memory(self) -> None:
        self.last_follow_goal_xy = None
        self.pending_follow_goal_xy = None
        self.pending_follow_goal_wall = -1.0e9
        self.first_follow_leader_xy = None
        self.first_follow_wall = None
        self.follow_startup_released = False

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
