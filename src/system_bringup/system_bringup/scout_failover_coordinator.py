#!/usr/bin/env python3
"""Leader-domain active-scout failover coordinator.

This node keeps system_bringup as the orchestration root.  It watches the
current active scout's bridged heartbeat and map-frame pose, freezes the last
fresh pose on failure, sends leader/follower recovery goals through the
existing fleet goal topics, pauses follower follow mode, and publishes an
epoch-based recovery role command to the unified field robot runtime.
"""

from __future__ import annotations

import json
import math
from enum import Enum
from typing import Optional

import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import Bool, String


class FailoverState(Enum):
    NORMAL_OPERATION = 'NORMAL_OPERATION'
    SCOUT_SUSPECTED_DEAD = 'SCOUT_SUSPECTED_DEAD'
    SCOUT_DEAD_CONFIRMED = 'SCOUT_DEAD_CONFIRMED'
    FAILOVER_TRIGGERED = 'FAILOVER_TRIGGERED'
    RECOVERY_NAVIGATING = 'RECOVERY_NAVIGATING'
    FOLLOWER_SCOUT_TAKEOVER = 'FOLLOWER_SCOUT_TAKEOVER'
    NEW_SCOUT_EXPLORING = 'NEW_SCOUT_EXPLORING'
    FAILOVER_FAILED = 'FAILOVER_FAILED'


def yaw_from_quaternion(q) -> float:
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


def quaternion_from_yaw(yaw: float) -> tuple[float, float, float, float]:
    return (0.0, 0.0, math.sin(0.5 * yaw), math.cos(0.5 * yaw))


class ScoutFailoverCoordinator(Node):
    def __init__(self) -> None:
        super().__init__('scout_failover_coordinator')

        self.declare_parameter('enable_scout_failover', True)
        self.declare_parameter('leader_robot_name', 'leader')
        self.declare_parameter('active_scout_robot_name', 'scout22')
        self.declare_parameter('follower_robot_name', 'follower21')
        self.declare_parameter('scout_liveness_topic', '/scout/signal')
        self.declare_parameter('scout_pose_topic', '/member_pose')
        self.declare_parameter('leader_pose_topic', '/leader_pose')
        self.declare_parameter('follower_pose_topic', '/burger_pose')
        self.declare_parameter('leader_goal_topic', '/fleet/leader_coord_goal')
        self.declare_parameter('leader_cancel_topic', '/fleet/leader_nav_cancel')
        self.declare_parameter('follow_command_topic', '/fleet/follow_command')
        self.declare_parameter('role_command_topic', '/fleet/field_robot_role_cmd')
        self.declare_parameter('field_robot_status_topic', '/fleet/field_robot_status')
        self.declare_parameter('role_topic', '/fleet/scout_role')
        self.declare_parameter('require_bootstrap_complete', True)
        self.declare_parameter('bootstrap_ready_topic', '/localization_ready')
        self.declare_parameter('scout_liveness_timeout_sec', 2.0)
        self.declare_parameter('scout_failure_confirm_sec', 0.5)
        self.declare_parameter('scout_pose_timeout_sec', 5.0)
        self.declare_parameter('startup_grace_sec', 5.0)
        self.declare_parameter('leader_recovery_standoff_m', 0.70)
        self.declare_parameter('leader_failure_arrival_tolerance_m', 0.80)
        self.declare_parameter('follower_recovery_standoff_m', 0.15)
        self.declare_parameter('scout_takeover_arrival_tolerance_m', 0.40)
        self.declare_parameter('recovery_goal_republish_sec', 2.0)
        self.declare_parameter('max_recovery_goal_republishes', 5)

        get = self.get_parameter
        self.enabled = bool(get('enable_scout_failover').value)
        self.leader_name = str(get('leader_robot_name').value)
        self.active_scout_id = str(get('active_scout_robot_name').value)
        self.original_scout_id = self.active_scout_id
        self.follower_name = str(get('follower_robot_name').value)
        self.scout_liveness_topic = str(get('scout_liveness_topic').value)
        self.scout_pose_topic = str(get('scout_pose_topic').value)
        self.leader_pose_topic = str(get('leader_pose_topic').value)
        self.follower_pose_topic = str(get('follower_pose_topic').value)
        self.leader_goal_topic = str(get('leader_goal_topic').value)
        self.leader_cancel_topic = str(get('leader_cancel_topic').value)
        self.follow_command_topic = str(get('follow_command_topic').value)
        self.role_command_topic = str(get('role_command_topic').value)
        self.field_robot_status_topic = str(get('field_robot_status_topic').value)
        self.role_topic = str(get('role_topic').value)
        self.require_bootstrap_complete = bool(get('require_bootstrap_complete').value)
        self.bootstrap_ready_topic = str(get('bootstrap_ready_topic').value)
        self.liveness_timeout = max(0.2, float(get('scout_liveness_timeout_sec').value))
        self.confirm_sec = max(0.0, float(get('scout_failure_confirm_sec').value))
        self.pose_timeout = max(0.2, float(get('scout_pose_timeout_sec').value))
        self.startup_grace = max(0.0, float(get('startup_grace_sec').value))
        self.leader_standoff = max(0.0, float(get('leader_recovery_standoff_m').value))
        self.leader_arrival_tolerance = max(
            0.05, float(get('leader_failure_arrival_tolerance_m').value)
        )
        self.follower_standoff = max(0.0, float(get('follower_recovery_standoff_m').value))
        self.arrival_tolerance = max(
            0.05, float(get('scout_takeover_arrival_tolerance_m').value)
        )
        self.goal_republish_sec = max(0.5, float(get('recovery_goal_republish_sec').value))
        self.max_goal_republishes = max(
            1, int(get('max_recovery_goal_republishes').value)
        )

        latched_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
        )

        self.leader_goal_pub = self.create_publisher(PoseStamped, self.leader_goal_topic, 10)
        self.leader_cancel_pub = self.create_publisher(Bool, self.leader_cancel_topic, latched_qos)
        self.follow_command_pub = self.create_publisher(String, self.follow_command_topic, latched_qos)
        self.role_command_pub = self.create_publisher(String, self.role_command_topic, latched_qos)
        self.role_pub = self.create_publisher(String, self.role_topic, latched_qos)
        self.state_pub = self.create_publisher(String, '/failover/state', latched_qos)
        self.active_scout_pub = self.create_publisher(String, '/failover/active_scout_id', latched_qos)
        self.epoch_pub = self.create_publisher(String, '/failover/scout_epoch', latched_qos)
        self.scout_alive_pub = self.create_publisher(Bool, '/failover/scout_alive', latched_qos)
        self.last_pose_pub = self.create_publisher(PoseStamped, '/failover/last_scout_pose', latched_qos)
        self.failure_pose_pub = self.create_publisher(PoseStamped, '/failover/failure_pose', latched_qos)

        self.create_subscription(String, self.scout_liveness_topic, self._on_liveness, 10)
        self.create_subscription(PoseStamped, self.scout_pose_topic, self._on_scout_pose, 10)
        self.create_subscription(PoseStamped, self.leader_pose_topic, self._on_leader_pose, 10)
        self.create_subscription(PoseStamped, self.follower_pose_topic, self._on_follower_pose, 10)
        self.create_subscription(String, self.field_robot_status_topic, self._on_field_status, 10)
        if self.require_bootstrap_complete:
            self.create_subscription(
                Bool,
                self.bootstrap_ready_topic,
                self._on_bootstrap_ready,
                latched_qos,
            )

        self.state = FailoverState.NORMAL_OPERATION
        self.start_wall = self._now()
        self.bootstrap_ready = not self.require_bootstrap_complete
        self.last_liveness_wall: Optional[float] = None
        self.last_scout_pose_wall: Optional[float] = None
        self.last_scout_pose: Optional[PoseStamped] = None
        self.failure_pose: Optional[PoseStamped] = None
        self.leader_pose: Optional[PoseStamped] = None
        self.follower_pose: Optional[PoseStamped] = None
        self.suspected_since: Optional[float] = None
        self.scout_epoch = 0
        self.recovery_goal_publish_count = 0
        self.last_recovery_goal_wall = -1.0e9
        self.leader_goal: Optional[PoseStamped] = None
        self.follower_goal: Optional[PoseStamped] = None
        self.leader_recovery_position_reached = False

        self.create_timer(0.2, self._tick)
        self._publish_state()
        self.get_logger().warning(
            '[FAILOVER] READY | '
            f'enabled={self.enabled} liveness={self.scout_liveness_topic} '
            f'scout_pose={self.scout_pose_topic} follower_pose={self.follower_pose_topic} '
            f'bootstrap_gate={self.require_bootstrap_complete}:{self.bootstrap_ready_topic}'
        )

    def _now(self) -> float:
        return self.get_clock().now().nanoseconds * 1.0e-9

    def _on_liveness(self, _msg: String) -> None:
        if self.state in (
            FailoverState.FOLLOWER_SCOUT_TAKEOVER,
            FailoverState.NEW_SCOUT_EXPLORING,
        ):
            return
        self.last_liveness_wall = self._now()
        if self.state == FailoverState.SCOUT_SUSPECTED_DEAD:
            self._transition(FailoverState.NORMAL_OPERATION)
            self.suspected_since = None

    def _on_scout_pose(self, msg: PoseStamped) -> None:
        if (msg.header.frame_id or 'map') != 'map':
            self.get_logger().warning(
                f'[FAILOVER] SCOUT_POSE_IGNORED frame={msg.header.frame_id!r}',
                throttle_duration_sec=5.0,
            )
            return
        if self.failure_pose is not None:
            return
        self.last_scout_pose = msg
        self.last_scout_pose_wall = self._now()
        self.last_pose_pub.publish(msg)

    def _on_leader_pose(self, msg: PoseStamped) -> None:
        self.leader_pose = msg

    def _on_follower_pose(self, msg: PoseStamped) -> None:
        self.follower_pose = msg

    def _on_bootstrap_ready(self, msg: Bool) -> None:
        previous = self.bootstrap_ready
        self.bootstrap_ready = bool(msg.data)
        if self.bootstrap_ready and not previous:
            self.get_logger().warning(
                f'[FAILOVER] BOOTSTRAP_READY | topic={self.bootstrap_ready_topic}'
            )

    def _on_field_status(self, msg: String) -> None:
        try:
            data = json.loads(msg.data)
        except json.JSONDecodeError:
            return
        if int(data.get('epoch', -1)) != self.scout_epoch:
            return
        if self.require_bootstrap_complete and not self.bootstrap_ready:
            return
        status = str(data.get('status', '')).upper()
        if status == 'ACTIVE_SCOUT':
            robot = str(data.get('robot', self.follower_name))
            # unified_field_robot republishes its status at 10 Hz, so this
            # callback fires continuously while a scout stays ACTIVE_SCOUT --
            # only log/transition on the actual edge into that role, not on
            # every repeat message, or this spams the log forever.
            already_resumed = (
                self.state == FailoverState.NEW_SCOUT_EXPLORING
                and self.active_scout_id == robot
            )
            self.active_scout_id = robot
            self._publish_role()
            if already_resumed:
                return
            self._transition(FailoverState.NEW_SCOUT_EXPLORING)
            self.get_logger().warning(
                '[FAILOVER] EXPLORATION_RESUMED | '
                f'active_scout={self.active_scout_id} epoch={self.scout_epoch}'
            )

    def _tick(self) -> None:
        self._publish_state()
        self._publish_role()
        if not self.enabled:
            return
        if self._now() - self.start_wall < self.startup_grace:
            return
        if self.require_bootstrap_complete and not self.bootstrap_ready:
            self.get_logger().info(
                '[FAILOVER] BOOTSTRAP_HOLD | '
                f'waiting for {self.bootstrap_ready_topic}=true',
                throttle_duration_sec=5.0,
            )
            return

        if self.state == FailoverState.NORMAL_OPERATION:
            self._check_liveness()
        elif self.state == FailoverState.SCOUT_SUSPECTED_DEAD:
            self._confirm_or_recover()
        elif self.state in (
            FailoverState.FAILOVER_TRIGGERED,
            FailoverState.RECOVERY_NAVIGATING,
        ):
            self._recovery_loop()
    def _check_liveness(self) -> None:
        if self.last_liveness_wall is None:
            self.get_logger().info(
                '[FAILOVER] WAIT_SCOUT_HEARTBEAT',
                throttle_duration_sec=5.0,
            )
            return
        age = self._now() - self.last_liveness_wall
        if age <= self.liveness_timeout:
            return
        self.suspected_since = self._now()
        self.get_logger().warning(
            '[FAILOVER] SCOUT_HEARTBEAT_LOST | '
            f'age={age:.2f}s timeout={self.liveness_timeout:.2f}s'
        )
        self._transition(FailoverState.SCOUT_SUSPECTED_DEAD)

    def _confirm_or_recover(self) -> None:
        if self.last_liveness_wall is not None:
            age = self._now() - self.last_liveness_wall
            if age <= self.liveness_timeout:
                self.suspected_since = None
                self._transition(FailoverState.NORMAL_OPERATION)
                return
        if self.suspected_since is None:
            self.suspected_since = self._now()
            return
        if self._now() - self.suspected_since < self.confirm_sec:
            return
        self._confirm_dead()

    def _confirm_dead(self) -> None:
        if self.last_scout_pose is None or self.last_scout_pose_wall is None:
            self.get_logger().error('[FAILOVER] FAILED | no scout pose cached')
            self._transition(FailoverState.FAILOVER_FAILED)
            return
        pose_age = self._now() - self.last_scout_pose_wall
        if pose_age > self.pose_timeout:
            self.get_logger().error(
                '[FAILOVER] FAILED | stale scout pose '
                f'age={pose_age:.2f}s max={self.pose_timeout:.2f}s'
            )
            self._transition(FailoverState.FAILOVER_FAILED)
            return

        self.failure_pose = self._copy_pose(self.last_scout_pose)
        self.failure_pose_pub.publish(self.failure_pose)
        self.scout_epoch += 1
        self.leader_goal = self._offset_pose(self.failure_pose, self.leader_standoff)
        self.follower_goal = self._offset_pose(self.failure_pose, self.follower_standoff)
        self.leader_recovery_position_reached = self._leader_already_near_failure()
        self.recovery_goal_publish_count = 0
        self.last_recovery_goal_wall = -1.0e9
        self.get_logger().warning(
            '[FAILOVER] SCOUT_DEAD_CONFIRMED | '
            f'epoch={self.scout_epoch} pose_age={pose_age:.2f}s'
        )
        self.get_logger().warning(
            '[FAILOVER] LAST_POSE_FROZEN | '
            f'x={self.failure_pose.pose.position.x:.3f} '
            f'y={self.failure_pose.pose.position.y:.3f}'
        )
        self._transition(FailoverState.SCOUT_DEAD_CONFIRMED)
        self._trigger_failover()

    def _trigger_failover(self) -> None:
        self._cancel_leader_goal()
        command = String()
        command.data = 'PAUSE'
        self.follow_command_pub.publish(command)
        self.get_logger().warning('[FAILOVER] FOLLOWER_FOLLOW_CANCEL | command=PAUSE')
        self._transition(FailoverState.FAILOVER_TRIGGERED)
        self._recovery_loop(force=True)

    def _recovery_loop(self, force: bool = False) -> None:
        if self.failure_pose is None or self.leader_goal is None or self.follower_goal is None:
            self._transition(FailoverState.FAILOVER_FAILED)
            return
        now = self._now()
        should_publish = force or (
            now - self.last_recovery_goal_wall >= self.goal_republish_sec
            and self.recovery_goal_publish_count < self.max_goal_republishes
        )
        if should_publish:
            if self.leader_recovery_position_reached:
                self.get_logger().warning(
                    '[FAILOVER] LEADER_RECOVERY_POSITION_REACHED | '
                    'already inside failure tolerance; recovery goal skipped'
                )
            else:
                self.leader_goal_pub.publish(self.leader_goal)
            self._publish_recovery_role_command()
            self.recovery_goal_publish_count += 1
            self.last_recovery_goal_wall = now
            if not self.leader_recovery_position_reached:
                self.get_logger().warning(
                    '[FAILOVER] LEADER_RECOVERY_GOAL_SENT | '
                    f'x={self.leader_goal.pose.position.x:.3f} '
                    f'y={self.leader_goal.pose.position.y:.3f}'
                )
            self.get_logger().warning(
                '[FAILOVER] FOLLOWER_RECOVERY_ROLE_SENT | '
                f'x={self.follower_goal.pose.position.x:.3f} '
                f'y={self.follower_goal.pose.position.y:.3f}'
            )
            self._transition(FailoverState.RECOVERY_NAVIGATING)

        if self._follower_arrived():
            self.get_logger().warning('[FAILOVER] FOLLOWER_ARRIVED')
            self._transition(FailoverState.FOLLOWER_SCOUT_TAKEOVER)
    def _follower_arrived(self) -> bool:
        if self.follower_pose is None or self.failure_pose is None:
            return False
        dx = self.follower_pose.pose.position.x - self.failure_pose.pose.position.x
        dy = self.follower_pose.pose.position.y - self.failure_pose.pose.position.y
        return math.hypot(dx, dy) <= self.arrival_tolerance

    def _leader_already_near_failure(self) -> bool:
        if self.leader_pose is None or self.failure_pose is None:
            return False
        dx = self.leader_pose.pose.position.x - self.failure_pose.pose.position.x
        dy = self.leader_pose.pose.position.y - self.failure_pose.pose.position.y
        distance = math.hypot(dx, dy)
        return distance <= self.leader_arrival_tolerance

    def _cancel_leader_goal(self) -> None:
        msg = Bool()
        msg.data = True
        self.leader_cancel_pub.publish(msg)
        self.get_logger().warning(
            f'[FAILOVER] LEADER_NAV_CANCEL | topic={self.leader_cancel_topic}'
        )

    def _publish_recovery_role_command(self) -> None:
        if self.follower_goal is None or self.failure_pose is None:
            return
        data = {
            'role': 'RECOVERY_NAVIGATING',
            'epoch': self.scout_epoch,
            'robot': self.follower_name,
            'previous_scout': self.original_scout_id,
            'target_pose': {
                'frame_id': 'map',
                'x': self.follower_goal.pose.position.x,
                'y': self.follower_goal.pose.position.y,
                'yaw': yaw_from_quaternion(self.follower_goal.pose.orientation),
            },
            'failure_pose': {
                'frame_id': 'map',
                'x': self.failure_pose.pose.position.x,
                'y': self.failure_pose.pose.position.y,
                'yaw': yaw_from_quaternion(self.failure_pose.pose.orientation),
            },
        }
        msg = String()
        msg.data = json.dumps(data, sort_keys=True)
        self.role_command_pub.publish(msg)
        self.get_logger().warning(
            '[FAILOVER] FIELD_ROLE_COMMAND | '
            f'role=RECOVERY_NAVIGATING epoch={self.scout_epoch} robot={self.follower_name}'
        )

    def _transition(self, new_state: FailoverState) -> None:
        if self.state == new_state:
            return
        old = self.state
        self.state = new_state
        self.get_logger().warning(f'[FAILOVER] STATE | {old.value} -> {new_state.value}')
        self._publish_state()

    def _publish_state(self) -> None:
        state = String()
        state.data = self.state.value
        self.state_pub.publish(state)
        active = String()
        active.data = self.active_scout_id
        self.active_scout_pub.publish(active)
        epoch = String()
        epoch.data = str(self.scout_epoch)
        self.epoch_pub.publish(epoch)
        alive = Bool()
        alive.data = self.state in (
            FailoverState.NORMAL_OPERATION,
            FailoverState.SCOUT_SUSPECTED_DEAD,
        )
        self.scout_alive_pub.publish(alive)

    def _publish_role(self) -> None:
        data = {
            'epoch': self.scout_epoch,
            'active_scout_id': self.active_scout_id,
            'previous_scout_id': self.original_scout_id,
            'state': self.state.value,
        }
        msg = String()
        msg.data = json.dumps(data, sort_keys=True)
        self.role_pub.publish(msg)

    def _copy_pose(self, source: PoseStamped) -> PoseStamped:
        msg = PoseStamped()
        msg.header.frame_id = source.header.frame_id or 'map'
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.pose = source.pose
        return msg

    def _offset_pose(self, failure: PoseStamped, standoff: float) -> PoseStamped:
        yaw = yaw_from_quaternion(failure.pose.orientation)
        msg = PoseStamped()
        msg.header.frame_id = 'map'
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.pose.position.x = failure.pose.position.x - standoff * math.cos(yaw)
        msg.pose.position.y = failure.pose.position.y - standoff * math.sin(yaw)
        msg.pose.position.z = 0.0
        qx, qy, qz, qw = quaternion_from_yaw(yaw)
        msg.pose.orientation.x = qx
        msg.pose.orientation.y = qy
        msg.pose.orientation.z = qz
        msg.pose.orientation.w = qw
        return msg


def main(args=None) -> None:
    rclpy.init(args=args)
    node = ScoutFailoverCoordinator()
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
