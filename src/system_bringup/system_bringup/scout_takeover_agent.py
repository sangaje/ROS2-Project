#!/usr/bin/env python3
"""Follower-side scout takeover agent.

The leader-domain coordinator publishes an epoch-based takeover command after
the follower reaches the failed scout's last pose.  This local agent owns the
post-arrival sequence: verify AMCL quality, perform a bounded in-place spin if
needed, then start the configured exploration process in this robot's domain.
"""

from __future__ import annotations

import json
import math
import shlex
import signal
import subprocess
from enum import Enum
from typing import Optional

import rclpy
from geometry_msgs.msg import PoseWithCovarianceStamped, Twist, TwistStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import String


class AgentState(Enum):
    STANDBY = 'STANDBY'
    CHECK_LOCALIZATION = 'CHECK_LOCALIZATION'
    LOCALIZATION_SPIN = 'LOCALIZATION_SPIN'
    SETTLING = 'SETTLING'
    ACTIVE_SCOUT = 'ACTIVE_SCOUT'
    FAILED = 'FAILED'


def wrap_angle(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def yaw_from_quaternion(q) -> float:
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


class ScoutTakeoverAgent(Node):
    def __init__(self) -> None:
        super().__init__('scout_takeover_agent')

        self.declare_parameter('robot_name', 'follower21')
        self.declare_parameter('takeover_topic', '/fleet/scout_takeover')
        self.declare_parameter('role_topic', '/fleet/scout_role')
        self.declare_parameter('status_topic', '/fleet/scout_takeover_status')
        self.declare_parameter('amcl_pose_topic', '/amcl_pose')
        self.declare_parameter('odom_topic', '/odom')
        self.declare_parameter('cmd_vel_topic', '/cmd_vel')
        self.declare_parameter('use_stamped_cmd_vel', True)
        self.declare_parameter('require_amcl_quality', True)
        self.declare_parameter('max_xy_covariance', 0.35)
        self.declare_parameter('max_yaw_covariance', 0.25)
        self.declare_parameter('max_amcl_pose_age_sec', 3.0)
        self.declare_parameter('enable_localization_spin_on_takeover', True)
        self.declare_parameter('spin_speed_rad_s', 0.35)
        self.declare_parameter('spin_target_angle_rad', 6.45)
        self.declare_parameter('spin_timeout_sec', 30.0)
        self.declare_parameter('settle_duration_sec', 2.0)
        self.declare_parameter('max_spin_retries', 2)
        self.declare_parameter('enable_exploration_after_takeover', True)
        self.declare_parameter(
            'exploration_command',
            'ros2 run turtlebot3_rl_training eval_policy '
            '--model rl_models/sac_turtlebot3_burger.zip '
            '--real-robot --disable-slam-map',
        )
        self.declare_parameter('stop_exploration_on_standby', True)

        get = self.get_parameter
        self.robot_name = str(get('robot_name').value)
        self.takeover_topic = str(get('takeover_topic').value)
        self.role_topic = str(get('role_topic').value)
        self.status_topic = str(get('status_topic').value)
        self.amcl_pose_topic = str(get('amcl_pose_topic').value)
        self.odom_topic = str(get('odom_topic').value)
        self.cmd_vel_topic = str(get('cmd_vel_topic').value)
        self.use_stamped = bool(get('use_stamped_cmd_vel').value)
        self.require_amcl = bool(get('require_amcl_quality').value)
        self.max_xy_cov = max(0.0, float(get('max_xy_covariance').value))
        self.max_yaw_cov = max(0.0, float(get('max_yaw_covariance').value))
        self.max_amcl_age = max(0.1, float(get('max_amcl_pose_age_sec').value))
        self.spin_enabled = bool(get('enable_localization_spin_on_takeover').value)
        self.spin_speed = abs(float(get('spin_speed_rad_s').value))
        self.spin_target = max(0.0, float(get('spin_target_angle_rad').value))
        self.spin_timeout = max(1.0, float(get('spin_timeout_sec').value))
        self.settle_duration = max(0.0, float(get('settle_duration_sec').value))
        self.max_spin_retries = max(0, int(get('max_spin_retries').value))
        self.exploration_enabled = bool(get('enable_exploration_after_takeover').value)
        self.exploration_command = str(get('exploration_command').value).strip()
        self.stop_exploration_on_standby = bool(get('stop_exploration_on_standby').value)

        latched_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
        )
        self.status_pub = self.create_publisher(String, self.status_topic, latched_qos)
        if self.use_stamped:
            self.cmd_pub = self.create_publisher(TwistStamped, self.cmd_vel_topic, 10)
        else:
            self.cmd_pub = self.create_publisher(Twist, self.cmd_vel_topic, 10)

        self.create_subscription(String, self.takeover_topic, self._on_takeover, latched_qos)
        self.create_subscription(String, self.role_topic, self._on_role, latched_qos)
        self.create_subscription(PoseWithCovarianceStamped, self.amcl_pose_topic, self._on_amcl, 10)
        self.create_subscription(Odometry, self.odom_topic, self._on_odom, 10)

        self.state = AgentState.STANDBY
        self.epoch = 0
        self.last_amcl_wall: Optional[float] = None
        self.xy_cov = float('inf')
        self.yaw_cov = float('inf')
        self.last_odom_yaw: Optional[float] = None
        self.accumulated_yaw = 0.0
        self.spin_start_wall = 0.0
        self.settle_start_wall = 0.0
        self.spin_attempt = 0
        self.spin_direction = 1.0
        self.exploration_process: Optional[subprocess.Popen] = None

        self.create_timer(0.1, self._tick)
        self._publish_status('STANDBY')
        self.get_logger().warning(
            'SCOUT_TAKEOVER_AGENT_READY | '
            f'robot={self.robot_name} takeover={self.takeover_topic} '
            f'cmd_vel={self.cmd_vel_topic} type={"TwistStamped" if self.use_stamped else "Twist"}'
        )

    def _now(self) -> float:
        return self.get_clock().now().nanoseconds * 1.0e-9

    def _on_takeover(self, msg: String) -> None:
        try:
            data = json.loads(msg.data)
        except json.JSONDecodeError:
            self.get_logger().warning(f'TAKEOVER_IGNORED_BAD_JSON | {msg.data!r}')
            return
        if str(data.get('command', '')).upper() != 'TAKEOVER_SCOUT':
            return
        target_robot = str(data.get('robot', ''))
        if target_robot != self.robot_name:
            return
        epoch = int(data.get('epoch', 0))
        if epoch < self.epoch:
            self.get_logger().warning(
                f'TAKEOVER_IGNORED_OLD_EPOCH | got={epoch} current={self.epoch}'
            )
            return
        if self.state == AgentState.ACTIVE_SCOUT and epoch == self.epoch:
            return
        self.epoch = epoch
        self.spin_attempt = 0
        self.get_logger().warning(
            f'[FAILOVER] TAKEOVER_RECEIVED | robot={self.robot_name} epoch={self.epoch}'
        )
        self._transition(AgentState.CHECK_LOCALIZATION)

    def _on_role(self, msg: String) -> None:
        try:
            data = json.loads(msg.data)
        except json.JSONDecodeError:
            return
        active = str(data.get('active_scout_id', ''))
        epoch = int(data.get('epoch', 0))
        if active == self.robot_name:
            return
        if epoch >= self.epoch and self.stop_exploration_on_standby:
            self._stop_exploration('role_epoch_standby')
            if self.state == AgentState.ACTIVE_SCOUT:
                self._transition(AgentState.STANDBY)

    def _on_amcl(self, msg: PoseWithCovarianceStamped) -> None:
        cov = msg.pose.covariance
        self.xy_cov = max(abs(float(cov[0])), abs(float(cov[7])))
        self.yaw_cov = abs(float(cov[35]))
        self.last_amcl_wall = self._now()

    def _on_odom(self, msg: Odometry) -> None:
        yaw = yaw_from_quaternion(msg.pose.pose.orientation)
        if self.state == AgentState.LOCALIZATION_SPIN and self.last_odom_yaw is not None:
            self.accumulated_yaw += abs(wrap_angle(yaw - self.last_odom_yaw))
        self.last_odom_yaw = yaw

    def _tick(self) -> None:
        if self.state == AgentState.CHECK_LOCALIZATION:
            self._tick_check_localization()
        elif self.state == AgentState.LOCALIZATION_SPIN:
            self._tick_spin()
        elif self.state == AgentState.SETTLING:
            self._tick_settle()

    def _tick_check_localization(self) -> None:
        self.get_logger().warning(
            '[FAILOVER] LOCALIZATION_CHECK | '
            f'xy_cov={self.xy_cov:.4f} yaw_cov={self.yaw_cov:.4f}',
            throttle_duration_sec=2.0,
        )
        if self._localization_ok():
            self._activate_scout()
            return
        if not self.spin_enabled:
            self._transition(AgentState.FAILED)
            self._publish_status('LOCALIZATION_FAILED_NO_SPIN')
            return
        if self.spin_attempt > self.max_spin_retries:
            self._transition(AgentState.FAILED)
            self._publish_status('LOCALIZATION_FAILED')
            return
        self._start_spin()

    def _localization_ok(self) -> bool:
        if not self.require_amcl:
            return True
        if self.last_amcl_wall is None:
            return False
        if self._now() - self.last_amcl_wall > self.max_amcl_age:
            return False
        return self.xy_cov <= self.max_xy_cov and self.yaw_cov <= self.max_yaw_cov

    def _start_spin(self) -> None:
        self.spin_attempt += 1
        self.spin_direction = 1.0 if self.spin_attempt % 2 == 1 else -1.0
        self.accumulated_yaw = 0.0
        self.spin_start_wall = self._now()
        self.get_logger().warning(
            '[FAILOVER] LOCALIZATION_SPIN_REQUIRED | '
            f'attempt={self.spin_attempt}/{self.max_spin_retries + 1}'
        )
        self._transition(AgentState.LOCALIZATION_SPIN)

    def _tick_spin(self) -> None:
        elapsed = self._now() - self.spin_start_wall
        if self.accumulated_yaw >= self.spin_target:
            self._publish_twist(0.0)
            self.settle_start_wall = self._now()
            self._transition(AgentState.SETTLING)
            return
        if elapsed >= self.spin_timeout:
            self._publish_twist(0.0)
            self.get_logger().warning(
                '[FAILOVER] LOCALIZATION_SPIN_TIMEOUT | '
                f'rotated={math.degrees(self.accumulated_yaw):.0f}deg'
            )
            self._transition(AgentState.CHECK_LOCALIZATION)
            return
        self._publish_twist(self.spin_direction * self.spin_speed)

    def _tick_settle(self) -> None:
        self._publish_twist(0.0)
        if self._now() - self.settle_start_wall >= self.settle_duration:
            self._transition(AgentState.CHECK_LOCALIZATION)

    def _activate_scout(self) -> None:
        self._transition(AgentState.ACTIVE_SCOUT)
        self._start_exploration()
        self._publish_status('ACTIVE_SCOUT')
        self.get_logger().warning(
            f'[FAILOVER] EXPLORATION_RESUMED | robot={self.robot_name} epoch={self.epoch}'
        )

    def _start_exploration(self) -> None:
        if not self.exploration_enabled:
            return
        if self.exploration_process is not None and self.exploration_process.poll() is None:
            return
        if not self.exploration_command:
            self.get_logger().warning('EXPLORATION_COMMAND_EMPTY | logical takeover only')
            return
        argv = shlex.split(self.exploration_command)
        self.exploration_process = subprocess.Popen(argv)
        self.get_logger().warning(
            'EXPLORATION_PROCESS_STARTED | ' + self.exploration_command
        )

    def _stop_exploration(self, reason: str) -> None:
        process = self.exploration_process
        if process is None or process.poll() is not None:
            return
        self.get_logger().warning(f'EXPLORATION_PROCESS_STOP | reason={reason}')
        process.send_signal(signal.SIGINT)
        self.exploration_process = None

    def _publish_twist(self, angular_z: float) -> None:
        if self.use_stamped:
            msg = TwistStamped()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = 'base_footprint'
            msg.twist.angular.z = angular_z
            self.cmd_pub.publish(msg)
        else:
            msg = Twist()
            msg.angular.z = angular_z
            self.cmd_pub.publish(msg)

    def _transition(self, new_state: AgentState) -> None:
        if self.state == new_state:
            return
        old = self.state
        self.state = new_state
        self.get_logger().warning(f'TAKEOVER_AGENT_STATE | {old.value} -> {new_state.value}')
        self._publish_status(new_state.value)

    def _publish_status(self, status: str) -> None:
        data = {
            'robot': self.robot_name,
            'epoch': self.epoch,
            'status': status,
            'state': self.state.value,
            'xy_cov': None if math.isinf(self.xy_cov) else self.xy_cov,
            'yaw_cov': None if math.isinf(self.yaw_cov) else self.yaw_cov,
        }
        msg = String()
        msg.data = json.dumps(data, sort_keys=True)
        self.status_pub.publish(msg)

    def destroy_node(self) -> None:
        try:
            self._publish_twist(0.0)
            self._stop_exploration('node_shutdown')
        except Exception:  # noqa: BLE001
            pass
        super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = ScoutTakeoverAgent()
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
