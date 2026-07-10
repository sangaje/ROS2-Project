#!/usr/bin/env python3
"""Publish an RViz-visible preview of the commanded /cmd_vel.

Draws a short unicycle-model trajectory (curves with angular.z, lengthens
with linear.x) plus a text readout of the raw values, so a nonzero-but-tiny
command is visually obvious even when the robot barely moves. Goes gray and
says STALE the moment cmd_vel stops arriving, so "is anything being
commanded at all" is answerable at a glance without reading topic echoes.
"""

from __future__ import annotations

import math
from typing import Optional

import rclpy
from rclpy.node import Node
from rclpy.exceptions import ParameterAlreadyDeclaredException
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from geometry_msgs.msg import Point, Twist, TwistStamped
from visualization_msgs.msg import Marker, MarkerArray


def _safe_declare(node: Node, name: str, default):
    try:
        node.declare_parameter(name, default)
    except ParameterAlreadyDeclaredException:
        pass
    return node.get_parameter(name).value


class CmdVelMarker(Node):
    """Subscribe to one robot's cmd_vel and publish a debug MarkerArray."""

    def __init__(self) -> None:
        super().__init__('cmd_vel_marker')
        _safe_declare(self, 'use_sim_time', False)
        self.cmd_vel_topic = str(_safe_declare(self, 'cmd_vel_topic', '/cmd_vel'))
        self.use_stamped_cmd_vel = bool(
            _safe_declare(self, 'use_stamped_cmd_vel', True)
        )
        self.base_frame_id = str(_safe_declare(self, 'base_frame_id', 'base_footprint'))
        self.marker_topic = str(
            _safe_declare(self, 'marker_topic', '/cmd_vel_debug_markers')
        )
        # Multiplies both v and w before the preview rollout below (shape is
        # unaffected, only size) so a small-but-real command -- e.g. an RL
        # policy near its deadband -- is still visible instead of collapsing
        # to a few centimeters over preview_seconds.
        self.preview_scale = max(
            0.1, float(_safe_declare(self, 'preview_scale', 3.0))
        )
        self.preview_seconds = max(
            0.1, float(_safe_declare(self, 'preview_seconds', 1.5))
        )
        self.preview_samples = max(
            2, int(_safe_declare(self, 'preview_samples', 12))
        )
        self.publish_rate_hz = max(
            1.0, float(_safe_declare(self, 'publish_rate_hz', 10.0))
        )
        self.stale_timeout_sec = max(
            0.1, float(_safe_declare(self, 'stale_timeout_sec', 0.5))
        )
        self.zero_linear_deadband = max(
            0.0, float(_safe_declare(self, 'zero_linear_deadband', 0.005))
        )
        self.zero_angular_deadband = max(
            0.0, float(_safe_declare(self, 'zero_angular_deadband', 0.02))
        )

        self.last_linear_x = 0.0
        self.last_angular_z = 0.0
        self.last_msg_wall: Optional[float] = None

        marker_qos = QoSProfile(
            depth=1,
            history=HistoryPolicy.KEEP_LAST,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )
        self.pub = self.create_publisher(MarkerArray, self.marker_topic, marker_qos)

        if self.use_stamped_cmd_vel:
            self.create_subscription(
                TwistStamped, self.cmd_vel_topic, self._on_twist_stamped, 10
            )
        else:
            self.create_subscription(Twist, self.cmd_vel_topic, self._on_twist, 10)

        self.create_timer(1.0 / self.publish_rate_hz, self._tick)

        self.get_logger().info(
            'CMD_VEL_MARKER_READY | '
            f'in={self.cmd_vel_topic} '
            f'type={"TwistStamped" if self.use_stamped_cmd_vel else "Twist"} '
            f'frame={self.base_frame_id} out={self.marker_topic}'
        )

    def _now(self) -> float:
        return self.get_clock().now().nanoseconds * 1.0e-9

    def _on_twist_stamped(self, msg: TwistStamped) -> None:
        self._on_velocity(msg.twist)

    def _on_twist(self, msg: Twist) -> None:
        self._on_velocity(msg)

    def _on_velocity(self, twist: Twist) -> None:
        self.last_linear_x = float(twist.linear.x)
        self.last_angular_z = float(twist.angular.z)
        self.last_msg_wall = self._now()

    def _is_fresh(self) -> bool:
        if self.last_msg_wall is None:
            return False
        return self._now() - self.last_msg_wall <= self.stale_timeout_sec

    def _is_moving(self) -> bool:
        return (
            abs(self.last_linear_x) > self.zero_linear_deadband
            or abs(self.last_angular_z) > self.zero_angular_deadband
        )

    def _preview_points(self) -> list[Point]:
        """Sample a short unicycle-model rollout of the current command."""
        v = self.last_linear_x * self.preview_scale
        w = self.last_angular_z * self.preview_scale
        points = []
        for i in range(self.preview_samples + 1):
            t = self.preview_seconds * i / self.preview_samples
            yaw = w * t
            if abs(w) > 1.0e-6:
                x = (v / w) * math.sin(yaw)
                y = (v / w) * (1.0 - math.cos(yaw))
            else:
                x = v * t
                y = 0.0
            points.append(Point(x=x, y=y, z=0.05))
        return points

    def _tick(self) -> None:
        stamp = self.get_clock().now().to_msg()
        fresh = self._is_fresh()
        moving = fresh and self._is_moving()

        if moving:
            color = (0.15, 0.95, 0.25, 0.95)
        elif fresh:
            color = (0.85, 0.85, 0.85, 0.65)
        else:
            color = (0.55, 0.55, 0.55, 0.35)

        path = Marker()
        path.header.stamp = stamp
        path.header.frame_id = self.base_frame_id
        path.ns = 'cmd_vel_preview'
        path.id = 1
        path.type = Marker.LINE_STRIP
        path.action = Marker.ADD
        path.pose.orientation.w = 1.0
        path.scale.x = 0.045
        path.color.r, path.color.g, path.color.b, path.color.a = color
        path.points = (
            self._preview_points() if moving else [Point(x=0.0, y=0.0, z=0.05)]
        )

        text = Marker()
        text.header.stamp = stamp
        text.header.frame_id = self.base_frame_id
        text.ns = 'cmd_vel_text'
        text.id = 2
        text.type = Marker.TEXT_VIEW_FACING
        text.action = Marker.ADD
        text.pose.position.z = 0.45
        text.pose.orientation.w = 1.0
        text.scale.z = 0.16
        text.color.r, text.color.g, text.color.b, text.color.a = color
        if not fresh:
            age = '' if self.last_msg_wall is None else f' ({self._now() - self.last_msg_wall:.1f}s ago)'
            text.text = f'cmd_vel STALE{age}'
        else:
            text.text = (
                f'lin={self.last_linear_x:+.3f} m/s  ang={self.last_angular_z:+.3f} rad/s'
            )

        arr = MarkerArray()
        arr.markers.append(path)
        arr.markers.append(text)
        self.pub.publish(arr)


def main() -> None:
    rclpy.init()
    node = CmdVelMarker()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.destroy_node()
        except KeyboardInterrupt:
            pass
        if rclpy.ok():
            try:
                rclpy.shutdown()
            except KeyboardInterrupt:
                pass


if __name__ == '__main__':
    main()
