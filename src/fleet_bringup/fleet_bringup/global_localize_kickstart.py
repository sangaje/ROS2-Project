#!/usr/bin/env python3
"""Kick AMCL into global localization instead of trusting a fixed seed.

A hardcoded initial pose only works when it matches where the robot was
actually placed; when it doesn't, AMCL still converges confidently, just to
the wrong spot near that seed. Calling AMCL's own
`/reinitialize_global_localization` service spreads particles across the
whole map instead, and a short in-place spin gives scan matching more than
one viewpoint to disambiguate against.
"""
from __future__ import annotations

import rclpy
from geometry_msgs.msg import Twist, TwistStamped
from nav_msgs.msg import OccupancyGrid
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from std_srvs.srv import Empty


class GlobalLocalizeKickstart(Node):

    def __init__(self) -> None:
        super().__init__('global_localize_kickstart')
        self.declare_parameter(
            'reinit_service', '/reinitialize_global_localization'
        )
        self.declare_parameter('map_topic', '/map')
        self.declare_parameter('spin_enabled', True)
        self.declare_parameter('spin_duration_sec', 8.0)
        self.declare_parameter('spin_speed_rad_s', 0.6)
        self.declare_parameter('cmd_vel_topic', '/cmd_vel')
        self.declare_parameter('use_stamped_cmd_vel', True)

        get = self.get_parameter
        self.reinit_service_name = str(get('reinit_service').value)
        self.map_topic = str(get('map_topic').value)
        self.spin_enabled = bool(get('spin_enabled').value)
        self.spin_duration = max(0.0, float(get('spin_duration_sec').value))
        self.spin_speed = float(get('spin_speed_rad_s').value)
        self.cmd_vel_topic = str(get('cmd_vel_topic').value)
        self.use_stamped = bool(get('use_stamped_cmd_vel').value)

        self.client = self.create_client(Empty, self.reinit_service_name)
        if self.use_stamped:
            self.cmd_pub = self.create_publisher(
                TwistStamped, self.cmd_vel_topic, 10
            )
        else:
            self.cmd_pub = self.create_publisher(
                Twist, self.cmd_vel_topic, 10
            )

        # AMCL advertises /reinitialize_global_localization as soon as it
        # configures, well before it has actually received a map -- calling
        # it that early just spreads particles over an empty/nonexistent
        # map and wastes the one-shot spin. Wait for a real map first (this
        # is what has to cross the domain_bridge -> map_relay pipeline when
        # a member/follower owns SLAM instead of this robot).
        map_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
        )
        self.map_received = False
        self.map_sub = self.create_subscription(
            OccupancyGrid, self.map_topic, self._on_map, map_qos
        )

        self.spin_timer = None
        self.spin_deadline = 0.0
        self.done = False
        self.wait_timer = self.create_timer(0.5, self._try_reinitialize)

        self.get_logger().info(
            'GLOBAL_LOCALIZE_KICKSTART_READY | '
            f'service={self.reinit_service_name} map_topic={self.map_topic} '
            f'spin={self.spin_enabled} duration={self.spin_duration:.1f}s'
        )

    def _now(self) -> float:
        return self.get_clock().now().nanoseconds * 1.0e-9

    def _on_map(self, _msg: OccupancyGrid) -> None:
        self.map_received = True

    def _try_reinitialize(self) -> None:
        if self.done:
            return
        if not self.map_received:
            self.get_logger().info(
                f'GLOBAL_LOCALIZE_WAIT | no map yet on {self.map_topic}',
                throttle_duration_sec=5.0,
            )
            return
        if not self.client.service_is_ready():
            self.get_logger().info(
                'GLOBAL_LOCALIZE_WAIT | '
                f'service not ready: {self.reinit_service_name}',
                throttle_duration_sec=5.0,
            )
            return
        self.wait_timer.cancel()
        future = self.client.call_async(Empty.Request())
        future.add_done_callback(self._on_reinitialized)

    def _on_reinitialized(self, future) -> None:
        try:
            future.result()
        except Exception as error:  # noqa: BLE001
            self.get_logger().error(
                f'GLOBAL_LOCALIZE_REINIT_FAILED | {error}'
            )
        else:
            self.get_logger().warning('GLOBAL_LOCALIZE_REINITIALIZED')

        if not self.spin_enabled or self.spin_duration <= 0.0:
            self.done = True
            return
        self.spin_deadline = self._now() + self.spin_duration
        self.spin_timer = self.create_timer(0.1, self._spin_tick)

    def _spin_tick(self) -> None:
        if self._now() >= self.spin_deadline:
            self._publish_twist(0.0)
            self.spin_timer.cancel()
            self.done = True
            self.get_logger().warning('GLOBAL_LOCALIZE_SPIN_COMPLETE')
            return
        self._publish_twist(self.spin_speed)

    def _publish_twist(self, angular_z: float) -> None:
        if self.use_stamped:
            message = TwistStamped()
            message.header.stamp = self.get_clock().now().to_msg()
            message.header.frame_id = 'base_footprint'
            message.twist.angular.z = angular_z
            self.cmd_pub.publish(message)
        else:
            message = Twist()
            message.angular.z = angular_z
            self.cmd_pub.publish(message)


def main() -> None:
    rclpy.init()
    node = GlobalLocalizeKickstart()
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
