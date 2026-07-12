#!/usr/bin/env python3
"""Standalone ACTIVE_SCOUT policy process with failover activation gates."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import json
from typing import Optional

import rclpy
from geometry_msgs.msg import Twist, TwistStamped
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import Bool, String

from .scout_rl_runtime import ActiveScoutRLRuntime


class RLWorkerState(Enum):
    STANDBY = 'STANDBY'
    RECOVERY_NAVIGATING = 'RECOVERY_NAVIGATING'
    WAIT_LOCALIZATION = 'WAIT_LOCALIZATION'
    WAIT_MOTION_RELEASE = 'WAIT_MOTION_RELEASE'
    WAIT_SENSOR_READY = 'WAIT_SENSOR_READY'
    ACTIVE = 'ACTIVE'
    FAILED = 'FAILED'


@dataclass(frozen=True)
class RoleUpdate:
    role: str
    robot: str
    epoch: Optional[int]
    localization_ready: Optional[bool]
    recovery_complete: Optional[bool]


@dataclass(frozen=True)
class GateInputs:
    role: str
    role_robot_matches: bool
    role_epoch: int
    failover_epoch: int
    active_scout_matches: bool
    failover_state: str
    localization_ready: bool
    recovery_complete: bool
    nav_goal_inactive: bool
    motion_authority: str
    model_ready: bool
    sensor_ready: bool
    tf_ready: bool
    require_failover_activation: bool


def parse_epoch(value) -> Optional[int]:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if value >= 0 else None
    if isinstance(value, str) and value.strip().isdigit():
        return int(value.strip())
    return None


def _optional_bool(value) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        key = value.strip().lower()
        if key in ('true', '1', 'yes', 'y'):
            return True
        if key in ('false', '0', 'no', 'n'):
            return False
    return None


def parse_role_update(raw: str, robot_name: str) -> Optional[RoleUpdate]:
    text = str(raw or '').strip()
    if not text:
        return None
    if not text.startswith('{'):
        return RoleUpdate(
            role=text.upper(),
            robot=robot_name,
            epoch=None,
            localization_ready=None,
            recovery_complete=None,
        )
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    role = str(payload.get('role', payload.get('status', 'IDLE'))).strip().upper()
    robot = str(payload.get('robot', robot_name)).strip() or robot_name
    return RoleUpdate(
        role=role,
        robot=robot,
        epoch=parse_epoch(payload.get('epoch')),
        localization_ready=_optional_bool(payload.get('localization_ready')),
        recovery_complete=_optional_bool(payload.get('recovery_complete')),
    )


def evaluate_activation_gate(gate: GateInputs) -> tuple[RLWorkerState, str]:
    role = gate.role.strip().upper()
    failover_state = gate.failover_state.strip().upper()
    motion_authority = gate.motion_authority.strip().upper()
    if role in ('FAILED',):
        return RLWorkerState.FAILED, 'role_failed'
    if (
        role in ('RECOVERY_NAVIGATING', 'FOLLOWER', 'IDLE')
        or failover_state in ('RECOVERY_NAVIGATING', 'FAILOVER_TRIGGERED')
        or motion_authority == 'FAILOVER_RECOVERY_NAV'
    ):
        return RLWorkerState.RECOVERY_NAVIGATING, 'recovery_or_non_scout_role'
    if role != 'ACTIVE_SCOUT' or not gate.role_robot_matches:
        return RLWorkerState.STANDBY, 'role_not_active_scout'
    if gate.role_epoch < gate.failover_epoch:
        return RLWorkerState.STANDBY, 'stale_epoch'
    if gate.require_failover_activation and not gate.active_scout_matches:
        return RLWorkerState.STANDBY, 'active_scout_id_mismatch'
    if not gate.localization_ready:
        return RLWorkerState.WAIT_LOCALIZATION, 'localization_not_ready'
    if gate.require_failover_activation and not gate.recovery_complete:
        return RLWorkerState.WAIT_LOCALIZATION, 'recovery_not_complete'
    if not gate.nav_goal_inactive or motion_authority not in ('NONE', 'ACTIVE_SCOUT_RL', ''):
        return RLWorkerState.WAIT_MOTION_RELEASE, 'motion_authority_busy'
    if not gate.model_ready or not gate.sensor_ready or not gate.tf_ready:
        return RLWorkerState.WAIT_SENSOR_READY, 'runtime_inputs_not_ready'
    return RLWorkerState.ACTIVE, 'activation_gate_passed'


class ScoutRLPolicyWorker(Node):
    """Run deterministic RL only after failover ownership is fully settled."""

    def __init__(self) -> None:
        super().__init__('scout_rl_policy_worker')
        self.declare_parameter('robot_name', 'scout22')
        self.declare_parameter('role_topic', '')
        self.declare_parameter('initial_role_active', False)
        self.declare_parameter('failover_state_topic', '/failover/state')
        self.declare_parameter('active_scout_id_topic', '/failover/active_scout_id')
        self.declare_parameter('scout_epoch_topic', '/failover/scout_epoch')
        self.declare_parameter('localization_ready_topic', '/localization_ready')
        self.declare_parameter('field_robot_status_topic', '/fleet/field_robot_status')
        self.declare_parameter('require_failover_activation', True)
        self.declare_parameter('cmd_vel_topic', '/cmd_vel')
        self.declare_parameter('use_stamped_cmd_vel', True)
        self.declare_parameter('enable_velocity_safety_filter', True)

        get = self.get_parameter
        self.robot_name = str(get('robot_name').value).strip()
        self.role_topic = str(get('role_topic').value).strip() or f'/{self.robot_name}/role'
        initial_active = bool(get('initial_role_active').value)
        self.failover_state_topic = str(get('failover_state_topic').value)
        self.active_scout_id_topic = str(get('active_scout_id_topic').value)
        self.scout_epoch_topic = str(get('scout_epoch_topic').value)
        self.localization_ready_topic = str(get('localization_ready_topic').value)
        self.field_robot_status_topic = str(get('field_robot_status_topic').value)
        self.require_failover_activation = bool(get('require_failover_activation').value)
        self.cmd_vel_topic = str(get('cmd_vel_topic').value)
        self.use_stamped = bool(get('use_stamped_cmd_vel').value)
        self.enable_velocity_safety_filter = bool(
            get('enable_velocity_safety_filter').value
        )

        self.desired_role = 'ACTIVE_SCOUT' if initial_active else 'IDLE'
        self.role_epoch = 0
        self.failover_epoch = 0
        self.active_scout_id = self.robot_name if initial_active else ''
        self.failover_state = 'NORMAL_OPERATION'
        self.localization_ready = bool(initial_active)
        self.role_localization_ready: Optional[bool] = None
        self.role_recovery_complete: Optional[bool] = True if initial_active else None
        self.status_recovery_complete = bool(initial_active)
        self.nav_goal_inactive = bool(initial_active)
        self.motion_authority = 'NONE'
        self.worker_state = RLWorkerState.STANDBY
        self.runtime_active = False
        self.last_gate_reason = 'startup'

        if self.use_stamped:
            self.cmd_pub = self.create_publisher(TwistStamped, self.cmd_vel_topic, 10)
        else:
            self.cmd_pub = self.create_publisher(Twist, self.cmd_vel_topic, 10)

        latched_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
        )
        self.create_subscription(String, self.role_topic, self._on_role, latched_qos)
        self.create_subscription(String, self.failover_state_topic, self._on_failover_state, latched_qos)
        self.create_subscription(String, self.active_scout_id_topic, self._on_active_scout_id, latched_qos)
        self.create_subscription(String, self.scout_epoch_topic, self._on_scout_epoch, latched_qos)
        self.create_subscription(Bool, self.localization_ready_topic, self._on_localization_ready, latched_qos)
        self.create_subscription(String, self.field_robot_status_topic, self._on_field_status, 10)

        self.runtime = ActiveScoutRLRuntime(
            self,
            self._publish_command,
            enable_velocity_safety_filter=self.enable_velocity_safety_filter,
        )
        self.create_timer(0.1, self._evaluate_gate)
        self.get_logger().warning(
            'SCOUT_RL_STANDBY | '
            f'robot={self.robot_name} epoch={self.role_epoch} '
            f'role_topic={self.role_topic} cmd_vel={self.cmd_vel_topic} '
            f'initial_active={initial_active} '
            f'require_failover_activation={self.require_failover_activation}'
        )

    def _on_role(self, msg: String) -> None:
        update = parse_role_update(msg.data, self.robot_name)
        if update is None:
            self.get_logger().warning('SCOUT_RL_ROLE_IGNORED | malformed')
            return
        if update.robot and update.robot != self.robot_name:
            return
        epoch = self.role_epoch if update.epoch is None else update.epoch
        if epoch < self.failover_epoch:
            self.get_logger().warning(
                'SCOUT_RL_ROLE_IGNORED | '
                f'stale_epoch={epoch} failover_epoch={self.failover_epoch}'
            )
            return
        if epoch < self.role_epoch:
            return
        self.role_epoch = epoch
        self.desired_role = update.role
        self.role_localization_ready = update.localization_ready
        self.role_recovery_complete = update.recovery_complete
        self._evaluate_gate()

    def _on_failover_state(self, msg: String) -> None:
        self.failover_state = str(msg.data).strip().upper() or 'NORMAL_OPERATION'
        self._evaluate_gate()

    def _on_active_scout_id(self, msg: String) -> None:
        self.active_scout_id = str(msg.data).strip()
        self._evaluate_gate()

    def _on_scout_epoch(self, msg: String) -> None:
        epoch = parse_epoch(str(msg.data).strip())
        if epoch is None:
            return
        if epoch > self.failover_epoch:
            self.failover_epoch = epoch
        self._evaluate_gate()

    def _on_localization_ready(self, msg: Bool) -> None:
        self.localization_ready = bool(msg.data)
        self._evaluate_gate()

    def _on_field_status(self, msg: String) -> None:
        try:
            data = json.loads(msg.data)
        except (json.JSONDecodeError, TypeError):
            return
        if not isinstance(data, dict):
            return
        if str(data.get('robot', '')).strip() != self.robot_name:
            return
        epoch = parse_epoch(data.get('epoch'))
        if epoch is not None and epoch < self.failover_epoch:
            return
        if epoch is not None and epoch > self.role_epoch:
            self.role_epoch = epoch
        status = str(data.get('status', data.get('role', ''))).strip().upper()
        self.motion_authority = str(data.get('motion_authority', 'NONE')).strip().upper()
        self.status_recovery_complete = bool(
            data.get('recovery_complete', False)
            or data.get('active_scout_ready', False)
            or status == 'ACTIVE_SCOUT_READY'
        )
        active_goal_count = int(data.get('active_goal_count', 0) or 0)
        pending_goal_count = int(data.get('pending_goal_count', 0) or 0)
        nav_goal_active = bool(data.get('nav_goal_active', False))
        self.nav_goal_inactive = (
            active_goal_count == 0
            and pending_goal_count == 0
            and not nav_goal_active
            and self.motion_authority not in (
                'FAILOVER_RECOVERY_NAV',
                'NORMAL_FOLLOW',
                'LOCALIZATION_SPIN',
            )
        )
        status_localization = _optional_bool(data.get('localization_ready'))
        if status_localization is not None:
            self.localization_ready = status_localization
        self._evaluate_gate()

    def _build_gate_inputs(self) -> GateInputs:
        active_scout_matches = self.active_scout_id == self.robot_name
        if not self.active_scout_id and self.failover_epoch == 0:
            active_scout_matches = True
        recovery_complete = bool(
            self.role_recovery_complete
            or self.status_recovery_complete
            or (
                self.failover_epoch == 0
                and self.failover_state in ('', 'NORMAL_OPERATION')
            )
        )
        localization_ready = bool(
            self.localization_ready
            or (self.role_localization_ready is True)
        )
        return GateInputs(
            role=self.desired_role,
            role_robot_matches=True,
            role_epoch=self.role_epoch,
            failover_epoch=self.failover_epoch,
            active_scout_matches=active_scout_matches,
            failover_state=self.failover_state,
            localization_ready=localization_ready,
            recovery_complete=recovery_complete,
            nav_goal_inactive=self.nav_goal_inactive,
            motion_authority=self.motion_authority,
            model_ready=self.runtime.ready,
            sensor_ready=self.runtime.sensor_ready(),
            tf_ready=self.runtime.tf_ready(),
            require_failover_activation=self.require_failover_activation,
        )

    def _evaluate_gate(self) -> None:
        if not hasattr(self, 'runtime'):
            return
        gate = self._build_gate_inputs()
        state, reason = evaluate_activation_gate(gate)
        self.last_gate_reason = reason
        if state != self.worker_state:
            self.worker_state = state
            self._log_state_transition(state, reason)
        should_activate = state == RLWorkerState.ACTIVE
        if should_activate and not self.runtime_active:
            self.get_logger().warning(
                'SCOUT_RL_ACTIVATING | '
                f'robot={self.robot_name} epoch={self.role_epoch}'
            )
            self.runtime_active = True
            self.runtime.activate()
            self.get_logger().warning(
                'SCOUT_RL_ACTIVE | '
                f'robot={self.robot_name} epoch={self.role_epoch}'
            )
        elif not should_activate and self.runtime_active:
            self.runtime_active = False
            self.runtime.deactivate(reason)

    def _log_state_transition(self, state: RLWorkerState, reason: str) -> None:
        if state == RLWorkerState.RECOVERY_NAVIGATING:
            self.get_logger().warning(
                f'SCOUT_RECOVERY_NAV_ACTIVE | robot={self.robot_name}'
            )
        elif state == RLWorkerState.WAIT_LOCALIZATION:
            self.get_logger().warning(
                f'SCOUT_WAIT_LOCALIZATION | robot={self.robot_name} reason={reason}'
            )
        elif state == RLWorkerState.WAIT_MOTION_RELEASE:
            self.get_logger().warning(
                f'SCOUT_WAIT_MOTION_RELEASE | robot={self.robot_name} reason={reason}'
            )
        elif state == RLWorkerState.WAIT_SENSOR_READY:
            self.get_logger().warning(
                f'SCOUT_WAIT_SENSOR_READY | robot={self.robot_name} reason={reason}'
            )
        elif state == RLWorkerState.ACTIVE:
            self.get_logger().warning(
                f'SCOUT_NAV_AUTHORITY_RELEASED | robot={self.robot_name}'
            )
        elif state == RLWorkerState.STANDBY:
            self.get_logger().warning(
                f'SCOUT_RL_STANDBY | robot={self.robot_name} reason={reason}'
            )
        elif state == RLWorkerState.FAILED:
            self.get_logger().error(
                f'SCOUT_RL_FAILED | robot={self.robot_name} reason={reason}'
            )

    def _publish_command(self, linear_x: float, angular_z: float) -> None:
        if not self.runtime_active and (linear_x != 0.0 or angular_z != 0.0):
            return
        if self.use_stamped:
            msg = TwistStamped()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = 'base_footprint'
            msg.twist.linear.x = float(linear_x)
            msg.twist.angular.z = float(angular_z)
        else:
            msg = Twist()
            msg.linear.x = float(linear_x)
            msg.angular.z = float(angular_z)
        self.cmd_pub.publish(msg)

    def destroy_node(self) -> None:
        try:
            self.runtime.shutdown()
        finally:
            super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = ScoutRLPolicyWorker()
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
