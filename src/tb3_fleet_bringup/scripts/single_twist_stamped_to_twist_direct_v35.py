#!/usr/bin/env python3

from __future__ import annotations

from typing import List, Optional

import rclpy
from rclpy.node import Node
from rclpy.time import Time
from geometry_msgs.msg import Twist, TwistStamped


def _split_csv(raw: str) -> List[str]:
    return [x.strip() for x in str(raw).split(',') if x.strip()]


def _copy_twist(src: Twist) -> Twist:
    out = Twist()
    out.linear.x = src.linear.x
    out.linear.y = src.linear.y
    out.linear.z = src.linear.z
    out.angular.x = src.angular.x
    out.angular.y = src.angular.y
    out.angular.z = src.angular.z
    return out


class SingleTwistStampedToTwistBridge(Node):
    """No-namespace TwistStamped -> Twist converter for one robot in one ROS domain.

    Public ROS API in each domain:
      /cmd_vel geometry_msgs/msg/TwistStamped

    Internal ROS topics for ros_gz_bridge:
      /gz_cmd_vel_unstamped
      /gz_cmd_vel_model_unstamped

    v36 real-Nav2 mode: by default, publish only when a TwistStamped arrives.
    Nav2 controller/velocity_smoother already publishes continuously while active,
    so this bridge must not generate its own persistent commands.
    """

    def __init__(self) -> None:
        super().__init__('single_twist_stamped_to_twist_bridge')

        self.declare_parameter('robot_name', 'robot')
        self.declare_parameter('cmd_vel_topic', '/cmd_vel')
        self.declare_parameter('internal_cmd_vel_topics', '/gz_cmd_vel_unstamped,/gz_cmd_vel_model_unstamped')
        self.declare_parameter('cmd_republish_rate_hz', 0.0)
        self.declare_parameter('watchdog_timeout_sec', 0.0)
        self.declare_parameter('watchdog_rate_hz', 10.0)
        self.declare_parameter('log_every_n_republish', 100)

        self.robot_name = str(self.get_parameter('robot_name').value)
        self.cmd_vel_topic = str(self.get_parameter('cmd_vel_topic').value)
        internal_topics = _split_csv(self.get_parameter('internal_cmd_vel_topics').value)
        self.cmd_republish_rate_hz = float(self.get_parameter('cmd_republish_rate_hz').value)
        self.watchdog_timeout_sec = float(self.get_parameter('watchdog_timeout_sec').value)
        watchdog_rate_hz = max(1.0, float(self.get_parameter('watchdog_rate_hz').value))
        self.log_every_n_republish = max(0, int(self.get_parameter('log_every_n_republish').value))

        if not self.cmd_vel_topic.startswith('/'):
            self.cmd_vel_topic = '/' + self.cmd_vel_topic
        if not internal_topics:
            raise RuntimeError('internal_cmd_vel_topics is empty')

        self._pubs = []
        self._out_topic_names = []
        for topic in internal_topics:
            if not topic.startswith('/'):
                topic = '/' + topic
            self._pubs.append(self.create_publisher(Twist, topic, 10))
            self._out_topic_names.append(topic)

        self._latest_twist = Twist()
        self._have_command = False
        self._last_msg_time: Optional[Time] = None
        self._stopped_by_watchdog = True
        self._republish_count = 0

        self._sub = self.create_subscription(TwistStamped, self.cmd_vel_topic, self._on_cmd, 10)
        
        if self.cmd_republish_rate_hz > 0.0:
            self.create_timer(1.0 / self.cmd_republish_rate_hz, self._republish_tick)


        self.get_logger().info(
            'V36_REAL_NAV2_TWIST_STAMPED_TO_TWIST_READY | '
            f'robot={self.robot_name} | in={self.cmd_vel_topic} TwistStamped | '
            f'outs={self._out_topic_names} Twist | republish={self.cmd_republish_rate_hz:.1f}Hz | mode=nav2_passthrough'
        )

        if self.watchdog_timeout_sec > 0.0:
            self.create_timer(1.0 / watchdog_rate_hz, self._watchdog_tick)
            self.get_logger().info(
                f'SINGLE_TWIST_STAMPED_WATCHDOG_ENABLED | robot={self.robot_name} timeout={self.watchdog_timeout_sec:.3f}s'
            )
        else:
            self.get_logger().info(
                f'V36_TWIST_BRIDGE_WATCHDOG_DISABLED | robot={self.robot_name} | no synthetic republish when cmd_republish_rate_hz=0'
            )

    def _publish_all(self, twist: Twist) -> None:
        for pub in self._pubs:
            pub.publish(twist)

    def _on_cmd(self, msg: TwistStamped) -> None:
        self._latest_twist = _copy_twist(msg.twist)
        self._have_command = True
        self._last_msg_time = self.get_clock().now()
        self._stopped_by_watchdog = False
        self._republish_count = 0
        self._publish_all(self._latest_twist)
        self.get_logger().info(
            f'V36_TWIST_BRIDGE_CMD_RX | robot={self.robot_name} | vx={msg.twist.linear.x:.3f} wz={msg.twist.angular.z:.3f}'
        )

    def _republish_tick(self) -> None:
        if not self._have_command:
            return
        self._publish_all(self._latest_twist)
        self._republish_count += 1
        if self.log_every_n_republish > 0 and self._republish_count % self.log_every_n_republish == 0:
            t = self._latest_twist
            self.get_logger().info(
                f'SINGLE_TWIST_STAMPED_CMDVEL_REPUBLISH | robot={self.robot_name} | n={self._republish_count} | vx={t.linear.x:.3f} wz={t.angular.z:.3f}'
            )

    def _watchdog_tick(self) -> None:
        if self._last_msg_time is None or self._stopped_by_watchdog:
            return
        age = (self.get_clock().now() - self._last_msg_time).nanoseconds * 1e-9
        if age > self.watchdog_timeout_sec:
            self._latest_twist = Twist()
            self._have_command = True
            self._publish_all(self._latest_twist)
            self._stopped_by_watchdog = True
            self.get_logger().warn(
                f'SINGLE_TWIST_STAMPED_CMDVEL_TIMEOUT_STOP | robot={self.robot_name} age={age:.3f}s timeout={self.watchdog_timeout_sec:.3f}s'
            )


def main() -> None:
    rclpy.init()
    node = SingleTwistStampedToTwistBridge()
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
