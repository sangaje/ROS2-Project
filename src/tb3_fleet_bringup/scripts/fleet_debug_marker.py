#!/usr/bin/env python3

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import rclpy
from rclpy.node import Node
from rclpy.exceptions import ParameterAlreadyDeclaredException
from geometry_msgs.msg import PoseStamped, Point
from visualization_msgs.msg import Marker, MarkerArray


def _safe_declare(node: Node, name: str, default):
    try:
        node.declare_parameter(name, default)
    except ParameterAlreadyDeclaredException:
        pass
    return node.get_parameter(name).value


def _quat_to_yaw(q) -> float:
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


class FleetDebugMarker(Node):
    """Publish simple RViz markers for Waffle and Burger poses in the shared map frame.

    This avoids bridging /tf across domains. /tf would collide because both Nav2 stacks
    intentionally use map->odom->base_footprint inside their own domain.
    """

    def __init__(self) -> None:
        super().__init__('fleet_debug_marker')
        _safe_declare(self, 'use_sim_time', True)
        self.waffle_pose_topic = self._abs(str(_safe_declare(self, 'waffle_pose_topic', '/leader_pose')))
        self.burger_pose_topic = self._abs(str(_safe_declare(self, 'burger_pose_topic', '/burger_pose')))
        self.marker_topic = self._abs(str(_safe_declare(self, 'marker_topic', '/fleet_debug_markers')))
        self.frame_id = str(_safe_declare(self, 'frame_id', 'map'))
        self.publish_rate_hz = max(1.0, float(_safe_declare(self, 'publish_rate_hz', 10.0)))
        self.stale_timeout_sec = max(0.5, float(_safe_declare(self, 'stale_timeout_sec', 5.0)))

        self.poses: Dict[str, Optional[PoseStamped]] = {'waffle': None, 'burger': None}
        self.last_seen: Dict[str, Optional[rclpy.time.Time]] = {'waffle': None, 'burger': None}

        self.sub_w = self.create_subscription(PoseStamped, self.waffle_pose_topic, lambda m: self._on_pose('waffle', m), 10)
        self.sub_b = self.create_subscription(PoseStamped, self.burger_pose_topic, lambda m: self._on_pose('burger', m), 10)
        self.pub = self.create_publisher(MarkerArray, self.marker_topic, 10)
        self.create_timer(1.0 / self.publish_rate_hz, self._tick)

        self.get_logger().info(
            'V41_FLEET_DEBUG_MARKER_READY | '
            f'waffle={self.waffle_pose_topic} burger={self.burger_pose_topic} out={self.marker_topic}'
        )

    @staticmethod
    def _abs(topic: str) -> str:
        topic = topic.strip()
        return topic if topic.startswith('/') else '/' + topic

    def _on_pose(self, name: str, msg: PoseStamped) -> None:
        self.poses[name] = msg
        self.last_seen[name] = self.get_clock().now()

    def _fresh(self, name: str) -> bool:
        t = self.last_seen.get(name)
        if t is None:
            return False
        age = (self.get_clock().now() - t).nanoseconds * 1e-9
        return age <= self.stale_timeout_sec

    def _make_body(self, ns: str, mid: int, pose: PoseStamped, scale: Tuple[float, float, float]) -> Marker:
        m = Marker()
        m.header.stamp = self.get_clock().now().to_msg()
        m.header.frame_id = self.frame_id
        m.ns = ns
        m.id = mid
        m.type = Marker.CYLINDER
        m.action = Marker.ADD
        m.pose.position.x = pose.pose.position.x
        m.pose.position.y = pose.pose.position.y
        m.pose.position.z = 0.12
        m.pose.orientation = pose.pose.orientation
        m.scale.x, m.scale.y, m.scale.z = scale
        # Do not rely on color semantics for correctness; use different alpha/intensity only for visual separation.
        if ns == 'waffle':
            m.color.r, m.color.g, m.color.b, m.color.a = 0.1, 0.4, 1.0, 0.85
        else:
            m.color.r, m.color.g, m.color.b, m.color.a = 1.0, 0.35, 0.05, 0.85
        return m

    def _make_arrow(self, ns: str, mid: int, pose: PoseStamped, length: float) -> Marker:
        yaw = _quat_to_yaw(pose.pose.orientation)
        x = pose.pose.position.x
        y = pose.pose.position.y
        m = Marker()
        m.header.stamp = self.get_clock().now().to_msg()
        m.header.frame_id = self.frame_id
        m.ns = ns + '_heading'
        m.id = mid
        m.type = Marker.ARROW
        m.action = Marker.ADD
        m.points = [Point(x=x, y=y, z=0.22), Point(x=x + length * math.cos(yaw), y=y + length * math.sin(yaw), z=0.22)]
        m.scale.x = 0.035
        m.scale.y = 0.08
        m.scale.z = 0.08
        if ns == 'waffle':
            m.color.r, m.color.g, m.color.b, m.color.a = 0.1, 0.4, 1.0, 0.95
        else:
            m.color.r, m.color.g, m.color.b, m.color.a = 1.0, 0.35, 0.05, 0.95
        return m

    def _make_text(self, ns: str, mid: int, pose: PoseStamped, text: str) -> Marker:
        m = Marker()
        m.header.stamp = self.get_clock().now().to_msg()
        m.header.frame_id = self.frame_id
        m.ns = ns + '_label'
        m.id = mid
        m.type = Marker.TEXT_VIEW_FACING
        m.action = Marker.ADD
        m.pose.position.x = pose.pose.position.x
        m.pose.position.y = pose.pose.position.y
        m.pose.position.z = 0.55
        m.pose.orientation.w = 1.0
        m.scale.z = 0.20
        m.text = text
        m.color.r = 1.0
        m.color.g = 1.0
        m.color.b = 1.0
        m.color.a = 0.95
        return m

    def _delete_ns(self, ns: str, ids) -> MarkerArray:
        arr = MarkerArray()
        for mid in ids:
            m = Marker()
            m.header.stamp = self.get_clock().now().to_msg()
            m.header.frame_id = self.frame_id
            m.ns = ns
            m.id = mid
            m.action = Marker.DELETE
            arr.markers.append(m)
        return arr

    def _tick(self) -> None:
        arr = MarkerArray()
        if self._fresh('waffle') and self.poses['waffle'] is not None:
            p = self.poses['waffle']
            arr.markers.append(self._make_body('waffle', 1, p, (0.38, 0.38, 0.18)))
            arr.markers.append(self._make_arrow('waffle', 2, p, 0.55))
            arr.markers.append(self._make_text('waffle', 3, p, 'waffle / domain25'))
        if self._fresh('burger') and self.poses['burger'] is not None:
            p = self.poses['burger']
            arr.markers.append(self._make_body('burger', 11, p, (0.30, 0.30, 0.16)))
            arr.markers.append(self._make_arrow('burger', 12, p, 0.45))
            arr.markers.append(self._make_text('burger', 13, p, 'burger / domain24'))
        if arr.markers:
            self.pub.publish(arr)


def main() -> None:
    rclpy.init()
    node = FleetDebugMarker()
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
