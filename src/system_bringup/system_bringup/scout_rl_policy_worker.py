#!/usr/bin/env python3
"""Standalone ACTIVE_SCOUT policy process with failover activation gates."""

from __future__ import annotations

import json
import os
from typing import Optional

import rclpy
from geometry_msgs.msg import Twist, TwistStamped
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import Bool, String

from .rl_activation_gate import GateInputs, RLWorkerState, evaluate_activation_gate
from .role_contract import RoleMessage, parse_epoch, parse_role_message
from .scout_rl_runtime import ActiveScoutRLRuntime


# Compatibility import surface for existing direct callers.  The canonical
# parser now lives next to the field-robot role contract.
RoleUpdate = RoleMessage
parse_role_update = parse_role_message


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
        self.declare_parameter('require_localization_ready', True)
        self.declare_parameter('require_system_ready', False)
        self.declare_parameter('system_ready_topic', '/system/ready')
        self.declare_parameter('require_start_motion', True)
        self.declare_parameter('start_motion_topic', '/fleet/start_motion')
        # Backward-compatible names accepted by old launch files. They no
        # longer control the final motion barrier.
        self.declare_parameter('require_video_ready', True)
        self.declare_parameter('video_ready_topic', '/fleet/start_motion')
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
        self.require_localization_ready = bool(get('require_localization_ready').value)
        self.require_system_ready = bool(get('require_system_ready').value)
        self.system_ready_topic = str(get('system_ready_topic').value)
        requested_start_motion = bool(get('require_start_motion').value)
        self.require_start_motion = True
        self.start_motion_topic = str(get('start_motion_topic').value).strip()
        legacy_topic = str(get('video_ready_topic').value).strip()
        if not self.start_motion_topic:
            self.start_motion_topic = legacy_topic or '/fleet/start_motion'
        self.require_video_ready = self.require_start_motion
        self.video_ready_topic = self.start_motion_topic
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
        self.localization_ready = False
        self.system_ready = not self.require_system_ready
        self.role_localization_ready: Optional[bool] = None
        self.role_recovery_complete: Optional[bool] = True if initial_active else None
        self.status_recovery_complete = bool(initial_active)
        self.nav_goal_inactive = bool(initial_active)
        self.start_motion = False
        self.video_ready = self.start_motion
        self.motion_authority = 'NONE'
        self.worker_state = RLWorkerState.STANDBY
        self.runtime_active = False
        self.last_gate_reason = 'startup'
        self.last_debug_wall = -1.0e9
        self.startup_released = False

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
        if self.require_system_ready:
            self.create_subscription(Bool, self.system_ready_topic, self._on_system_ready, latched_qos)
        self.create_subscription(Bool, self.start_motion_topic, self._on_start_motion, latched_qos)
        self.create_subscription(String, self.field_robot_status_topic, self._on_field_status, 10)

        self.runtime = ActiveScoutRLRuntime(
            self,
            self._publish_command,
            enable_velocity_safety_filter=self.enable_velocity_safety_filter,
        )
        # A short but non-aggressive gate rate leaves executor capacity for
        # scan/map callbacks on the hardware inference Jetson.
        self.create_timer(0.25, self._evaluate_gate)
        self.get_logger().warning(
            'RL_WORKER_READY | '
            f'robot={self.robot_name} domain={os.environ.get("ROS_DOMAIN_ID", "")} '
            'backend=external_worker standby=true '
            f'cmd_topic={self.cmd_vel_topic}'
        )
        self.get_logger().warning(
            'SCOUT_RL_STANDBY | '
            f'robot={self.robot_name} epoch={self.role_epoch} '
            f'role_topic={self.role_topic} cmd_vel={self.cmd_vel_topic} '
            f'initial_active={initial_active} '
            f'require_failover_activation={self.require_failover_activation} '
            f'require_system_ready={self.require_system_ready}:{self.system_ready_topic} '
            f'require_start_motion={self.require_start_motion}:{self.start_motion_topic} '
            f'requested_start_motion_gate={requested_start_motion}'
        )

    def _now(self) -> float:
        return self.get_clock().now().nanoseconds * 1.0e-9

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
        self.desired_role = update.role.value
        self.role_localization_ready = update.localization_ready
        self.role_recovery_complete = update.recovery_complete
        if update.active_scout_id is not None:
            self.active_scout_id = update.active_scout_id
        self.get_logger().info(
            'RL_ROLE_UPDATE | '
            f'robot={self.robot_name} role={self.desired_role} '
            f'epoch={self.role_epoch} active_scout={self.active_scout_id or "(unset)"}'
        )
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

    def _on_system_ready(self, msg: Bool) -> None:
        previous = self.system_ready
        self.system_ready = bool(msg.data)
        if previous and not self.system_ready and self.runtime_active:
            self.runtime_active = False
            self.runtime.deactivate('system_not_ready')
            self._publish_zero()
        if self.system_ready != previous:
            self.get_logger().warning(
                'SCOUT_SYSTEM_READY | '
                f'robot={self.robot_name} ready={self.system_ready} '
                f'topic={self.system_ready_topic}'
            )
        self._evaluate_gate()

    def _on_start_motion(self, msg: Bool) -> None:
        previous = self.start_motion
        self.start_motion = bool(msg.data)
        self.video_ready = self.start_motion
        if previous and not self.start_motion and self.runtime_active:
            self.runtime_active = False
            self.runtime.deactivate('start_motion_false')
            self._publish_zero()
        if self.start_motion != previous:
            self.get_logger().warning(
                'SCOUT_START_MOTION | '
                f'robot={self.robot_name} ready={self.start_motion} '
                f'topic={self.start_motion_topic}'
            )
        if self.start_motion and not previous:
            self.startup_released = True
            self.get_logger().warning(
                'SCOUT_RL_RESUME_REQUEST | '
                f'robot={self.robot_name} reason=start_motion_true '
                'stale_action_dropped=true latest_sensors_retained=true'
            )
        self._evaluate_gate()

    def _on_video_ready(self, msg: Bool) -> None:
        self._on_start_motion(msg)

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
        status_localization = (
            data.get('localization_ready')
            if isinstance(data.get('localization_ready'), bool) else None
        )
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
        sensor_ready = self.runtime.sensor_ready()
        if not self.start_motion:
            sensor_ready = False
        if self.require_system_ready and not self.system_ready:
            sensor_ready = False
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
            sensor_ready=sensor_ready,
            tf_ready=self.runtime.tf_ready(),
            require_failover_activation=self.require_failover_activation,
            require_localization_ready=self.require_localization_ready,
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
        elif (
            not should_activate
            and self.runtime_active
            and state == RLWorkerState.WAIT_SENSOR_READY
            and self.start_motion
            and (self.system_ready or not self.require_system_ready)
        ):
            self.runtime.hold(reason)
        elif not should_activate and self.runtime_active:
            self.runtime_active = False
            self.runtime.deactivate(reason)
            self._publish_zero()
        self._log_rl_debug(gate, state, reason)

    def _debug_blocking_reason(
        self,
        gate: GateInputs,
        state: RLWorkerState,
        reason: str,
        runtime: dict[str, object],
    ) -> str:
        role = self.desired_role.strip().upper()
        if role != 'ACTIVE_SCOUT':
            return 'role_inactive'
        if self.require_failover_activation and not gate.active_scout_matches:
            return 'lease_expired'
        if self.require_start_motion and not self.start_motion:
            if not self.startup_released:
                return 'startup_not_released'
            return 'start_motion_false'
        if self.require_system_ready and not self.system_ready:
            return 'system_not_ready'
        if state == RLWorkerState.WAIT_LOCALIZATION:
            return 'localization_not_ready'
        if state == RLWorkerState.WAIT_MOTION_RELEASE:
            return 'cmd_vel_authority_lost'
        if state == RLWorkerState.WAIT_SENSOR_READY:
            scan_age = float(runtime.get('scan_age_ms', -1.0))
            map_age = float(runtime.get('map_age_ms', -1.0))
            max_scan_ms = self.runtime.config.max_scan_age_sec * 1000.0
            max_map_ms = self.runtime.config.max_map_age_sec * 1000.0
            if scan_age < 0.0 or scan_age > max_scan_ms:
                return 'scan_stale'
            if map_age < 0.0 or map_age > max_map_ms:
                return 'map_stale'
            if not bool(runtime.get('policy_worker_alive', False)):
                return 'policy_worker_dead'
            return 'observation_not_ready'
        if str(runtime.get('last_stop_reason', '')) == 'inference_timeout':
            return 'inference_timeout'
        if not bool(runtime.get('safety_allowed', True)):
            return 'safety_stop'
        if state == RLWorkerState.ACTIVE:
            return 'none'
        return reason

    def _log_rl_debug(
        self,
        gate: GateInputs,
        state: RLWorkerState,
        reason: str,
    ) -> None:
        now = self._now()
        if now - self.last_debug_wall < 1.0:
            return
        self.last_debug_wall = now
        runtime = self.runtime.debug_snapshot()
        blocking = self._debug_blocking_reason(gate, state, reason, runtime)
        role_active = self.desired_role.strip().upper() == 'ACTIVE_SCOUT'
        raw_nonzero = (
            abs(float(runtime['raw_cmd_linear'])) > 1.0e-4
            or abs(float(runtime['raw_cmd_angular'])) > 1.0e-4
        )
        final_nonzero = (
            abs(float(runtime['final_cmd_linear'])) > 1.0e-4
            or abs(float(runtime['final_cmd_angular'])) > 1.0e-4
        )
        hardware_publish_allowed = bool(
            role_active
            and gate.active_scout_matches
            and self.start_motion
            and state == RLWorkerState.ACTIVE
        )
        self.get_logger().warning(
            'SCOUT_RL_DEBUG | '
            f'robot={self.robot_name} '
            f'role={self.desired_role} '
            f'role_active={role_active} '
            f'active_scout_id={self.active_scout_id or "(unset)"} '
            f'epoch={self.role_epoch} '
            f'lease_valid={gate.active_scout_matches} '
            f'start_motion={self.start_motion} '
            f'system_ready={self.system_ready} '
            f'dashboard_ready={self.video_ready} '
            f'scan_age_ms={float(runtime["scan_age_ms"]):.0f} '
            f'map_age_ms={float(runtime["map_age_ms"]):.0f} '
            f'odom_age_ms=-1 '
            f'observation_ready={runtime["observation_ready"]} '
            f'policy_worker_alive={runtime["policy_worker_alive"]} '
            f'inference_age_ms={float(runtime["inference_age_ms"]):.0f} '
            f'raw_action_linear={float(runtime["raw_cmd_linear"]):.3f} '
            f'raw_action_angular={float(runtime["raw_cmd_angular"]):.3f} '
            f'safety_allowed={runtime["safety_allowed"]} '
            f'final_cmd_linear={float(runtime["final_cmd_linear"]):.3f} '
            f'final_cmd_angular={float(runtime["final_cmd_angular"]):.3f} '
            f'cmd_vel_published={runtime["cmd_vel_published"]} '
            f'gate_state={state.value} gate_reason={reason} '
            f'blocking_reason={blocking}'
        )
        self.get_logger().warning(
            'SCOUT_RL_GATE | '
            f'role_active={role_active} '
            f'start_motion={self.start_motion} '
            f'raw_action_nonzero={raw_nonzero} '
            f'final_command_nonzero={final_nonzero} '
            f'hardware_publish_allowed={hardware_publish_allowed} '
            f'blocking_reason={blocking}'
        )

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
                f'SCOUT_WAIT_SENSOR_READY | robot={self.robot_name} reason={reason} '
                f'inputs={self.runtime.readiness_summary()} '
                f'system_ready={self.system_ready}/{self.require_system_ready} '
                f'start_motion={self.start_motion}/{self.require_start_motion}'
            )
        elif state == RLWorkerState.ACTIVE:
            self.get_logger().warning(
                'RL_ACTIVATION_GATE | '
                f'robot={self.robot_name} role={self.desired_role} '
                f'epoch={self.role_epoch} model=true scan=true map=true tf=true '
                f'nav_idle={self.nav_goal_inactive} system=true start_motion=true'
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
        if not self.start_motion and (linear_x != 0.0 or angular_z != 0.0):
            self._publish_zero()
            return
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

    def _publish_zero(self) -> None:
        if self.use_stamped:
            msg = TwistStamped()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = 'base_footprint'
            msg.twist.linear.x = 0.0
            msg.twist.angular.z = 0.0
        else:
            msg = Twist()
            msg.linear.x = 0.0
            msg.angular.z = 0.0
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
