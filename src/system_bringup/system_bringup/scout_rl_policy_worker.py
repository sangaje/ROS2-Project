#!/usr/bin/env python3
"""Standalone ACTIVE_SCOUT policy process for PC/robot integration tests.

This worker deliberately owns the learned-policy ``/cmd_vel`` publisher.  Run
the normal system bringup with ``enable_exploration:=false`` so there is exactly
one RL command owner.  It subscribes to the latched role topic published by
``UnifiedFieldRobot`` and activates only for ``ACTIVE_SCOUT``.
"""

from __future__ import annotations

import rclpy
from geometry_msgs.msg import Twist, TwistStamped
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import String

from .scout_rl_runtime import ActiveScoutRLRuntime


class ScoutRLPolicyWorker(Node):
    """Run deterministic RL outside the field-role orchestration process."""

    def __init__(self) -> None:
        super().__init__('scout_rl_policy_worker')
        self.declare_parameter('robot_name', 'scout22')
        self.declare_parameter('role_topic', '')
        self.declare_parameter('initial_role_active', True)
        self.declare_parameter('cmd_vel_topic', '/cmd_vel')
        self.declare_parameter('use_stamped_cmd_vel', True)
        self.declare_parameter('enable_velocity_safety_filter', True)

        get = self.get_parameter
        self.robot_name = str(get('robot_name').value).strip()
        self.role_topic = str(get('role_topic').value).strip() or f'/{self.robot_name}/role'
        self.role_active = bool(get('initial_role_active').value)
        self.cmd_vel_topic = str(get('cmd_vel_topic').value)
        self.use_stamped = bool(get('use_stamped_cmd_vel').value)
        self.enable_velocity_safety_filter = bool(
            get('enable_velocity_safety_filter').value
        )

        if self.use_stamped:
            self.cmd_pub = self.create_publisher(TwistStamped, self.cmd_vel_topic, 10)
        else:
            self.cmd_pub = self.create_publisher(Twist, self.cmd_vel_topic, 10)
        role_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
        )
        self.create_subscription(String, self.role_topic, self._on_role, role_qos)
        self.runtime = ActiveScoutRLRuntime(
            self,
            self._publish_command,
            enable_velocity_safety_filter=self.enable_velocity_safety_filter,
        )
        if self.role_active:
            self.runtime.activate()
        self.get_logger().warning(
            'SCOUT_RL_POLICY_WORKER_READY | '
            f'robot={self.robot_name} role_topic={self.role_topic} '
            f'cmd_vel={self.cmd_vel_topic} initial_active={self.role_active} '
            f'safety_filter={self.enable_velocity_safety_filter}'
        )

    def _on_role(self, msg: String) -> None:
        active = str(msg.data).strip().upper() == 'ACTIVE_SCOUT'
        if active == self.role_active:
            return
        self.role_active = active
        if active:
            self.runtime.activate()
            self.get_logger().warning('SCOUT_RL_POLICY_WORKER_ROLE | ACTIVE_SCOUT')
        else:
            self.runtime.deactivate('role_not_active_scout')
            self.get_logger().warning(
                f'SCOUT_RL_POLICY_WORKER_ROLE | inactive={msg.data!r}'
            )

    def _publish_command(self, linear_x: float, angular_z: float) -> None:
        if not self.role_active and (linear_x != 0.0 or angular_z != 0.0):
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
