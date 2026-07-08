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

import math
import time

import rclpy
from geometry_msgs.msg import PoseWithCovarianceStamped, Twist, TwistStamped
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
        self.declare_parameter('retry_enabled', True)
        self.declare_parameter('retry_timeout_sec', 86400.0)
        self.declare_parameter('retry_interval_sec', 8.0)
        self.declare_parameter('stop_when_localized', True)
        self.declare_parameter('amcl_pose_topic', '/amcl_pose')
        self.declare_parameter('localized_xy_cov_threshold', 0.35)
        self.declare_parameter('localized_yaw_cov_threshold', 0.25)
        self.declare_parameter('localized_required_samples', 5)

        get = self.get_parameter
        self.reinit_service_name = str(get('reinit_service').value)
        self.map_topic = str(get('map_topic').value)
        self.spin_enabled = bool(get('spin_enabled').value)
        self.spin_duration = max(0.0, float(get('spin_duration_sec').value))
        self.spin_speed = float(get('spin_speed_rad_s').value)
        self.cmd_vel_topic = str(get('cmd_vel_topic').value)
        self.use_stamped = bool(get('use_stamped_cmd_vel').value)
        self.retry_enabled = bool(get('retry_enabled').value)
        self.retry_timeout_sec = max(0.0, float(get('retry_timeout_sec').value))
        self.retry_interval_sec = max(0.0, float(get('retry_interval_sec').value))
        self.stop_when_localized = bool(get('stop_when_localized').value)
        self.amcl_pose_topic = str(get('amcl_pose_topic').value)
        self.localized_xy_cov_threshold = max(
            0.0, float(get('localized_xy_cov_threshold').value)
        )
        self.localized_yaw_cov_threshold = max(
            0.0, float(get('localized_yaw_cov_threshold').value)
        )
        self.localized_required_samples = max(
            1, int(get('localized_required_samples').value)
        )

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
        self.pose_sub = self.create_subscription(
            PoseWithCovarianceStamped,
            self.amcl_pose_topic,
            self._on_amcl_pose,
            10,
        )

        self.spin_timer = None
        self.spin_deadline = 0.0
        self.done = False
        self.reinit_in_flight = False
        self.localized = False
        self.localized_samples = 0
        self.last_pose_cov = None
        self.start_wall_sec = time.monotonic()
        self.next_reinit_wall_sec = self.start_wall_sec
        self.attempt = 0
        self.wait_timer = self.create_timer(0.5, self._tick)

        self.get_logger().info(
            'GLOBAL_LOCALIZE_KICKSTART_READY | '
            f'service={self.reinit_service_name} map_topic={self.map_topic} '
            f'spin={self.spin_enabled} duration={self.spin_duration:.1f}s '
            f'retry={self.retry_enabled} timeout={self.retry_timeout_sec:.1f}s '
            f'interval={self.retry_interval_sec:.1f}s '
            f'stop_when_localized={self.stop_when_localized}'
        )

    def _now(self) -> float:
        return self.get_clock().now().nanoseconds * 1.0e-9

    def _on_map(self, _msg: OccupancyGrid) -> None:
        self.map_received = True

    def _on_amcl_pose(self, msg: PoseWithCovarianceStamped) -> None:
        cov = msg.pose.covariance
        xy_cov = max(float(cov[0]), float(cov[7]))
        yaw_cov = float(cov[35])
        if not all(math.isfinite(value) for value in (xy_cov, yaw_cov)):
            self.localized_samples = 0
            return
        self.last_pose_cov = {
            'xy_cov': xy_cov,
            'yaw_cov': yaw_cov,
            'x': float(msg.pose.pose.position.x),
            'y': float(msg.pose.pose.position.y),
        }
        if (
            xy_cov <= self.localized_xy_cov_threshold
            and yaw_cov <= self.localized_yaw_cov_threshold
        ):
            self.localized_samples += 1
        else:
            self.localized_samples = 0

        if (
            not self.localized
            and self.attempt > 0
            and self.localized_samples >= self.localized_required_samples
        ):
            self.localized = True
            self.get_logger().warning(
                'GLOBAL_LOCALIZE_LOCALIZED | '
                f'x={self.last_pose_cov["x"]:+.3f} '
                f'y={self.last_pose_cov["y"]:+.3f} '
                f'xy_cov={xy_cov:.4f} yaw_cov={yaw_cov:.4f} '
                f'samples={self.localized_samples}'
            )

    def _tick(self) -> None:
        if self.done:
            return
        if self.stop_when_localized and self.localized:
            self._finish('localized')
            return
        elapsed = time.monotonic() - self.start_wall_sec
        if self.retry_timeout_sec > 0.0 and elapsed >= self.retry_timeout_sec:
            self._finish(f'timeout_after_{elapsed:.1f}s')
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
        if self.reinit_in_flight or self.spin_timer is not None:
            return
        if time.monotonic() < self.next_reinit_wall_sec:
            return

        self.reinit_in_flight = True
        self.attempt += 1
        self.get_logger().warning(
            'GLOBAL_LOCALIZE_ATTEMPT | '
            f'attempt={self.attempt} elapsed={elapsed:.1f}s'
        )
        future = self.client.call_async(Empty.Request())
        future.add_done_callback(self._on_reinitialized)

    def _on_reinitialized(self, future) -> None:
        self.reinit_in_flight = False
        try:
            future.result()
        except Exception as error:  # noqa: BLE001
            self.get_logger().error(
                f'GLOBAL_LOCALIZE_REINIT_FAILED | {error}'
            )
            self._schedule_next_attempt()
            return
        else:
            self.get_logger().warning(
                f'GLOBAL_LOCALIZE_REINITIALIZED | attempt={self.attempt}'
            )

        if not self.spin_enabled or self.spin_duration <= 0.0:
            if self.retry_enabled:
                self._schedule_next_attempt()
            else:
                self._finish('single_reinitialize_complete')
            return
        self.spin_deadline = self._now() + self.spin_duration
        self.spin_timer = self.create_timer(0.1, self._spin_tick)

    def _spin_tick(self) -> None:
        if self.stop_when_localized and self.localized:
            self._publish_twist(0.0)
            self.spin_timer.cancel()
            self.spin_timer = None
            self._finish('localized_during_spin')
            return
        if self._now() >= self.spin_deadline:
            self._publish_twist(0.0)
            self.spin_timer.cancel()
            self.spin_timer = None
            self.get_logger().warning(
                f'GLOBAL_LOCALIZE_SPIN_COMPLETE | attempt={self.attempt}'
            )
            if self.retry_enabled:
                self._schedule_next_attempt()
            else:
                self._finish('single_spin_complete')
            return
        self._publish_twist(self.spin_speed)

    def _schedule_next_attempt(self) -> None:
        if not self.retry_enabled:
            self._finish('retry_disabled')
            return
        self.next_reinit_wall_sec = time.monotonic() + self.retry_interval_sec
        cov_text = 'no_amcl_pose'
        if self.last_pose_cov is not None:
            cov_text = (
                f'xy_cov={self.last_pose_cov["xy_cov"]:.4f} '
                f'yaw_cov={self.last_pose_cov["yaw_cov"]:.4f}'
            )
        self.get_logger().warning(
            'GLOBAL_LOCALIZE_RETRY_SCHEDULED | '
            f'next_in={self.retry_interval_sec:.1f}s {cov_text}'
        )

    def _finish(self, reason: str) -> None:
        self._publish_twist(0.0)
        if self.spin_timer is not None:
            self.spin_timer.cancel()
            self.spin_timer = None
        self.done = True
        self.get_logger().warning(
            f'GLOBAL_LOCALIZE_DONE | reason={reason} attempts={self.attempt}'
        )

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

    def destroy_node(self) -> None:
        try:
            self._publish_twist(0.0)
        except Exception:  # noqa: BLE001
            pass
        super().destroy_node()


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
