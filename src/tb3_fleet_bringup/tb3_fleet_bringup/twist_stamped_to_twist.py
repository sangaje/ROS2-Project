#!/usr/bin/env python3

from __future__ import annotations

from typing import Dict, List, Optional

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


class TwistStampedToTwistBridge(Node):
    """Convert public TwistStamped cmd_vel to internal un-stamped Twist topics.

    v16 command path intentionally publishes to two internal Twist routes per robot:

      /robot1/cmd_vel TwistStamped
        -> /robot1/gz_cmd_vel_unstamped       Twist -> GZ /robot1/cmd_vel
        -> /robot1/gz_cmd_vel_model_unstamped Twist -> GZ /model/robot1/cmd_vel

    The patched SDF listens to /robot1/cmd_vel or /robot2/cmd_vel inside GZ.
    The /model/<robot>/cmd_vel route is kept as a fallback for SDFs that scope
    DiffDrive command topics automatically by model name.
    """

    def __init__(self) -> None:
        super().__init__('twist_stamped_to_twist_bridge')

        self.declare_parameter('robot_names', 'robot1,robot2')
        self.declare_parameter('cmd_vel_topic', 'cmd_vel')
        self.declare_parameter('internal_cmd_vel_topics', 'gz_cmd_vel_unstamped,gz_cmd_vel_model_unstamped')
        self.declare_parameter('cmd_republish_rate_hz', 20.0)
        self.declare_parameter('watchdog_timeout_sec', 0.0)
        self.declare_parameter('watchdog_rate_hz', 10.0)
        self.declare_parameter('log_every_n_republish', 100)

        robot_names = _split_csv(self.get_parameter('robot_names').value)
        cmd_vel_topic = str(self.get_parameter('cmd_vel_topic').value).strip('/')
        internal_topics = [x.strip('/') for x in _split_csv(self.get_parameter('internal_cmd_vel_topics').value)]
        self.cmd_republish_rate_hz = max(1.0, float(self.get_parameter('cmd_republish_rate_hz').value))
        self.watchdog_timeout_sec = float(self.get_parameter('watchdog_timeout_sec').value)
        watchdog_rate_hz = max(1.0, float(self.get_parameter('watchdog_rate_hz').value))
        self.log_every_n_republish = max(0, int(self.get_parameter('log_every_n_republish').value))

        if not robot_names:
            raise RuntimeError('robot_names parameter is empty')
        if not internal_topics:
            raise RuntimeError('internal_cmd_vel_topics parameter is empty')

        self._cmd_pub_map: Dict[str, List[object]] = {}
        self._last_msg_time: Dict[str, Optional[Time]] = {}
        self._latest_twist: Dict[str, Twist] = {}
        self._have_command: Dict[str, bool] = {}
        self._stopped_by_watchdog: Dict[str, bool] = {}
        self._republish_count: Dict[str, int] = {}
        self._subs = []

        for robot in robot_names:
            in_topic = f'/{robot}/{cmd_vel_topic}'
            pubs: List[object] = []
            out_topic_names: List[str] = []
            for internal_topic in internal_topics:
                out_topic = f'/{robot}/{internal_topic}'
                pubs.append(self.create_publisher(Twist, out_topic, 10))
                out_topic_names.append(out_topic)

            sub = self.create_subscription(
                TwistStamped,
                in_topic,
                lambda msg, r=robot: self._on_cmd(r, msg),
                10,
            )
            self._cmd_pub_map[robot] = pubs
            self._subs.append(sub)
            self._last_msg_time[robot] = None
            self._latest_twist[robot] = Twist()
            self._have_command[robot] = False
            self._stopped_by_watchdog[robot] = True
            self._republish_count[robot] = 0
            self.get_logger().info(
                'TWIST_STAMPED_CMDVEL_READY | '
                f'robot={robot} | in={in_topic} TwistStamped | '
                f'outs={out_topic_names} Twist | republish={self.cmd_republish_rate_hz:.1f}Hz'
            )

        self.create_timer(1.0 / self.cmd_republish_rate_hz, self._republish_tick)

        if self.watchdog_timeout_sec > 0.0:
            self.create_timer(1.0 / watchdog_rate_hz, self._watchdog_tick)
            self.get_logger().info(
                f'TWIST_STAMPED_WATCHDOG_ENABLED | timeout={self.watchdog_timeout_sec:.3f}s rate={watchdog_rate_hz:.1f}Hz'
            )
        else:
            self.get_logger().info('TWIST_STAMPED_WATCHDOG_DISABLED | command persists until explicit zero command')

    def _publish_all(self, robot: str, twist: Twist) -> None:
        for pub in self._cmd_pub_map[robot]:
            pub.publish(twist)

    def _on_cmd(self, robot: str, msg: TwistStamped) -> None:
        self._last_msg_time[robot] = self.get_clock().now()
        self._stopped_by_watchdog[robot] = False
        self._latest_twist[robot] = _copy_twist(msg.twist)
        self._have_command[robot] = True
        self._republish_count[robot] = 0
        self._publish_all(robot, self._latest_twist[robot])
        self.get_logger().info(
            f'TWIST_STAMPED_CMDVEL_RX | robot={robot} | vx={msg.twist.linear.x:.3f} wz={msg.twist.angular.z:.3f} | published_to_all_internal_routes=1'
        )

    def _republish_tick(self) -> None:
        for robot, have_command in self._have_command.items():
            if not have_command:
                continue
            self._publish_all(robot, self._latest_twist[robot])
            self._republish_count[robot] += 1
            if self.log_every_n_republish > 0 and self._republish_count[robot] % self.log_every_n_republish == 0:
                t = self._latest_twist[robot]
                self.get_logger().info(
                    f'TWIST_STAMPED_CMDVEL_REPUBLISH | robot={robot} | n={self._republish_count[robot]} | vx={t.linear.x:.3f} wz={t.angular.z:.3f}'
                )

    def _watchdog_tick(self) -> None:
        now = self.get_clock().now()
        zero = Twist()
        for robot, last_time in self._last_msg_time.items():
            if last_time is None:
                continue
            if self._stopped_by_watchdog[robot]:
                continue
            age = (now - last_time).nanoseconds * 1e-9
            if age > self.watchdog_timeout_sec:
                self._latest_twist[robot] = zero
                self._have_command[robot] = True
                self._publish_all(robot, zero)
                self._stopped_by_watchdog[robot] = True
                self.get_logger().warn(
                    f'TWIST_STAMPED_CMDVEL_TIMEOUT_STOP | robot={robot} age={age:.3f}s timeout={self.watchdog_timeout_sec:.3f}s'
                )


def main() -> None:
    rclpy.init()
    node = TwistStampedToTwistBridge()
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
