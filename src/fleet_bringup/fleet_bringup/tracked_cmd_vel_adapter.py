#!/usr/bin/env python3
"""Tracked Waffle cmd_vel adapter.

This node is a fallback for the real tracked Waffle before OpenCR firmware is
rebuilt with the tracked chassis kinematics.  It keeps hardware /cmd_vel owned
by one publisher:

    Nav2/shadow/localization -> /cmd_vel_nav -> this node -> /cmd_vel

If OpenCR has already been corrected for r=0.040 m and L=0.447 m, do not run
this adapter with compensation gains greater than 1.0.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import rclpy
from geometry_msgs.msg import Twist, TwistStamped
from rclpy.exceptions import ParameterAlreadyDeclaredException
from rclpy.node import Node
from std_msgs.msg import String


def _safe_declare(node: Node, name: str, default):
    try:
        node.declare_parameter(name, default)
    except ParameterAlreadyDeclaredException:
        pass
    return node.get_parameter(name).value


def _clamp(value: float, limit: float) -> float:
    limit = abs(float(limit))
    return max(-limit, min(limit, float(value)))


def _slew(current: float, target: float, limit: float, dt: float) -> float:
    step = max(0.0, float(limit)) * max(0.0, float(dt))
    delta = target - current
    if abs(delta) <= step:
        return target
    return current + step * (1.0 if delta > 0.0 else -1.0)


@dataclass
class Velocity:
    linear_x: float = 0.0
    angular_z: float = 0.0

    def is_zero(self) -> bool:
        return abs(self.linear_x) < 1.0e-6 and abs(self.angular_z) < 1.0e-6


class TrackedCmdVelAdapter(Node):
    """Scale and rate-limit tracked Waffle velocity commands."""

    def __init__(self) -> None:
        super().__init__('tracked_cmd_vel_adapter')
        _safe_declare(self, 'use_sim_time', False)
        self.enabled = bool(_safe_declare(self, 'enabled', True))
        self.input_topic = str(_safe_declare(self, 'input_topic', '/cmd_vel_nav'))
        self.output_topic = str(_safe_declare(self, 'output_topic', '/cmd_vel'))
        self.use_stamped = bool(_safe_declare(self, 'enable_stamped_cmd_vel', True))
        self.linear_gain = float(_safe_declare(self, 'linear_gain', 0.825))
        self.angular_gain = float(_safe_declare(self, 'angular_gain', 1.286))
        self.max_linear_velocity = abs(float(
            _safe_declare(self, 'max_linear_velocity', 0.22)
        ))
        self.max_angular_velocity = abs(float(
            _safe_declare(self, 'max_angular_velocity', 1.5)
        ))
        self.max_linear_acceleration = abs(float(
            _safe_declare(self, 'max_linear_acceleration', 0.30)
        ))
        self.max_angular_acceleration = abs(float(
            _safe_declare(self, 'max_angular_acceleration', 1.5)
        ))
        self.command_timeout_sec = max(
            0.05, float(_safe_declare(self, 'command_timeout_sec', 0.5))
        )
        self.control_rate_hz = max(
            1.0, float(_safe_declare(self, 'control_rate_hz', 30.0))
        )
        self.diagnostic_topic = str(
            _safe_declare(self, 'diagnostic_topic', '/tracked_cmd_vel_adapter/status')
        )
        self.diagnostic_period_sec = max(
            0.1, float(_safe_declare(self, 'diagnostic_period_sec', 1.0))
        )

        if self.input_topic == self.output_topic:
            raise RuntimeError(
                'tracked_cmd_vel_adapter input_topic and output_topic must differ '
                f'to avoid a feedback loop: {self.input_topic}'
            )

        self._target = Velocity()
        self._output = Velocity()
        self._last_input_time: Optional[float] = None
        self._last_tick_time: Optional[float] = None
        self._last_diag_time = 0.0
        self._last_status = ''
        self._published_zero_after_timeout = False

        self._diag_pub = self.create_publisher(String, self.diagnostic_topic, 10)
        if self.use_stamped:
            self._pub = self.create_publisher(TwistStamped, self.output_topic, 10)
            self.create_subscription(
                TwistStamped, self.input_topic, self._on_twist_stamped, 10
            )
        else:
            self._pub = self.create_publisher(Twist, self.output_topic, 10)
            self.create_subscription(Twist, self.input_topic, self._on_twist, 10)

        self.create_timer(1.0 / self.control_rate_hz, self._tick)
        self.get_logger().info(
            'TRACKED_CMD_VEL_ADAPTER_READY | '
            f'enabled={self.enabled} in={self.input_topic} out={self.output_topic} '
            f'type={"TwistStamped" if self.use_stamped else "Twist"} '
            f'linear_gain={self.linear_gain:.3f} angular_gain={self.angular_gain:.3f}'
        )

    def _now(self) -> float:
        return self.get_clock().now().nanoseconds * 1.0e-9

    def _on_twist_stamped(self, msg: TwistStamped) -> None:
        self._accept(msg.twist)

    def _on_twist(self, msg: Twist) -> None:
        self._accept(msg)

    def _accept(self, twist: Twist) -> None:
        self._last_input_time = self._now()
        self._published_zero_after_timeout = False
        if self.enabled:
            linear = self.linear_gain * float(twist.linear.x)
            angular = self.angular_gain * float(twist.angular.z)
        else:
            linear = float(twist.linear.x)
            angular = float(twist.angular.z)
        self._target = Velocity(
            linear_x=_clamp(linear, self.max_linear_velocity),
            angular_z=_clamp(angular, self.max_angular_velocity),
        )

    def _fresh(self, now: float) -> bool:
        return (
            self._last_input_time is not None
            and now - self._last_input_time <= self.command_timeout_sec
        )

    def _make_msg(self, vel: Velocity):
        if self.use_stamped:
            msg = TwistStamped()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = 'base_link'
            msg.twist.linear.x = vel.linear_x
            msg.twist.angular.z = vel.angular_z
            return msg
        msg = Twist()
        msg.linear.x = vel.linear_x
        msg.angular.z = vel.angular_z
        return msg

    def _publish(self, vel: Velocity) -> None:
        self._pub.publish(self._make_msg(vel))

    def _publish_zero(self) -> None:
        self._output = Velocity()
        self._publish(self._output)

    def _tick(self) -> None:
        now = self._now()
        last = self._last_tick_time if self._last_tick_time is not None else now
        self._last_tick_time = now
        dt = max(0.0, now - last)

        stale = not self._fresh(now)
        target = Velocity() if stale else self._target
        self._output = Velocity(
            linear_x=_slew(
                self._output.linear_x,
                target.linear_x,
                self.max_linear_acceleration,
                dt,
            ),
            angular_z=_slew(
                self._output.angular_z,
                target.angular_z,
                self.max_angular_acceleration,
                dt,
            ),
        )

        if stale and self._output.is_zero():
            if not self._published_zero_after_timeout:
                self._publish_zero()
                self._published_zero_after_timeout = True
        else:
            self._publish(self._output)

        self._diagnose(now, stale)

    def _diagnose(self, now: float, stale: bool) -> None:
        status = (
            f'enabled={self.enabled} stale={stale} '
            f'in={self.input_topic} out={self.output_topic} '
            f'target=({self._target.linear_x:.3f},{self._target.angular_z:.3f}) '
            f'out=({self._output.linear_x:.3f},{self._output.angular_z:.3f})'
        )
        if (
            status == self._last_status
            and now - self._last_diag_time < self.diagnostic_period_sec
        ):
            return
        self._last_status = status
        self._last_diag_time = now
        self._diag_pub.publish(String(data=status))

    def destroy_node(self) -> bool:
        try:
            self._publish_zero()
        except Exception as exc:  # pragma: no cover - shutdown best effort
            self.get_logger().warn(f'failed to publish shutdown zero cmd_vel: {exc}')
        return super().destroy_node()


def main() -> None:
    rclpy.init()
    node = TrackedCmdVelAdapter()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.destroy_node()
        finally:
            if rclpy.ok():
                rclpy.shutdown()


if __name__ == '__main__':
    main()
