#!/usr/bin/env python3

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import rclpy
from rclpy.node import Node
from rclpy.exceptions import ParameterAlreadyDeclaredException
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
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
    """Publish simple RViz markers for fleet poses/goals in the shared map frame.

    This avoids bridging /tf across domains. /tf would collide because both Nav2 stacks
    intentionally use map->odom->base_footprint inside their own domain.
    """

    def __init__(self) -> None:
        super().__init__('fleet_debug_marker')
        _safe_declare(self, 'use_sim_time', True)
        self.leader_pose_topic = self._abs(str(_safe_declare(self, 'leader_pose_topic', '/leader_pose')))
        self.burger_pose_topic = self._abs(str(_safe_declare(self, 'burger_pose_topic', '/burger_pose')))
        self.member_pose_topic = self._abs(str(_safe_declare(self, 'member_pose_topic', '/member_pose')))
        self.leader_goal_topic = self._abs(str(_safe_declare(self, 'leader_goal_topic', '/goal_pose')))
        self.leader_coord_goal_topic = self._abs(str(_safe_declare(self, 'leader_coord_goal_topic', '/fleet/leader_coord_goal')))
        self.burger_goal_topic = self._abs(str(_safe_declare(self, 'burger_goal_topic', '/burger_goal_pose')))
        self.member_goal_topic = self._abs(str(_safe_declare(self, 'member_goal_topic', '/member_goal_pose')))
        self.marker_topic = self._abs(str(_safe_declare(self, 'marker_topic', '/fleet_debug_markers')))
        self.frame_id = str(_safe_declare(self, 'frame_id', 'map'))
        self.publish_rate_hz = max(1.0, float(_safe_declare(self, 'publish_rate_hz', 10.0)))
        self.stale_timeout_sec = max(0.5, float(_safe_declare(self, 'stale_timeout_sec', 5.0)))
        self.goal_stale_timeout_sec = max(1.0, float(_safe_declare(self, 'goal_stale_timeout_sec', 120.0)))
        self.leader_domain_id = str(_safe_declare(self, 'leader_domain_id', '')).strip()
        self.burger_domain_id = str(_safe_declare(self, 'burger_domain_id', '')).strip()
        self.member_domain_id = str(_safe_declare(self, 'member_domain_id', '')).strip()

        self.poses: Dict[str, Optional[PoseStamped]] = {
            'leader': None,
            'burger': None,
            'member': None,
        }
        self.goals: Dict[str, Optional[PoseStamped]] = {
            'leader_user_goal': None,
            'leader_coord_goal': None,
            'burger_goal': None,
            'member_goal': None,
        }
        self.last_seen: Dict[str, Optional[rclpy.time.Time]] = {
            'leader': None,
            'burger': None,
            'member': None,
            'leader_user_goal': None,
            'leader_coord_goal': None,
            'burger_goal': None,
            'member_goal': None,
        }

        self.sub_w = self.create_subscription(PoseStamped, self.leader_pose_topic, lambda m: self._on_pose('leader', m), 10)
        self.sub_b = self.create_subscription(PoseStamped, self.burger_pose_topic, lambda m: self._on_pose('burger', m), 10)
        self.sub_m = self.create_subscription(PoseStamped, self.member_pose_topic, lambda m: self._on_pose('member', m), 10)
        self.sub_goal_leader = self.create_subscription(PoseStamped, self.leader_goal_topic, lambda m: self._on_goal('leader_user_goal', m), 10)
        self.sub_goal_leader_coord = self.create_subscription(PoseStamped, self.leader_coord_goal_topic, lambda m: self._on_goal('leader_coord_goal', m), 10)
        self.sub_goal_burger = self.create_subscription(PoseStamped, self.burger_goal_topic, lambda m: self._on_goal('burger_goal', m), 10)
        self.sub_goal_member = self.create_subscription(PoseStamped, self.member_goal_topic, lambda m: self._on_goal('member_goal', m), 10)
        marker_qos = QoSProfile(
            depth=1,
            history=HistoryPolicy.KEEP_LAST,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.pub = self.create_publisher(
            MarkerArray, self.marker_topic, marker_qos
        )
        self.create_timer(1.0 / self.publish_rate_hz, self._tick)

        self.get_logger().info(
            'V41_FLEET_DEBUG_MARKER_READY | '
            f'leader={self.leader_pose_topic} burger={self.burger_pose_topic} '
            f'member={self.member_pose_topic} goals={self.leader_goal_topic},'
            f'{self.member_goal_topic},{self.burger_goal_topic} '
            f'out={self.marker_topic}'
        )

    @staticmethod
    def _abs(topic: str) -> str:
        topic = topic.strip()
        return topic if topic.startswith('/') else '/' + topic

    def _on_pose(self, name: str, msg: PoseStamped) -> None:
        self.poses[name] = msg
        self.last_seen[name] = self.get_clock().now()

    def _on_goal(self, name: str, msg: PoseStamped) -> None:
        self.goals[name] = msg
        self.last_seen[name] = self.get_clock().now()

    def _fresh(self, name: str, timeout: Optional[float] = None) -> bool:
        t = self.last_seen.get(name)
        if t is None:
            return False
        age = (self.get_clock().now() - t).nanoseconds * 1e-9
        return age <= (timeout if timeout is not None else self.stale_timeout_sec)

    def _domain_label(self, ns: str) -> str:
        domain = {
            'leader': self.leader_domain_id,
            'burger': self.burger_domain_id,
            'member': self.member_domain_id,
        }.get(ns, '')
        return f' d{domain}' if domain else ''

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
        if ns == 'leader':
            m.color.r, m.color.g, m.color.b, m.color.a = 0.1, 0.4, 1.0, 0.85
        elif ns == 'member':
            m.color.r, m.color.g, m.color.b, m.color.a = 0.0, 0.85, 0.35, 0.85
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
        if ns == 'leader':
            m.color.r, m.color.g, m.color.b, m.color.a = 0.1, 0.4, 1.0, 0.95
        elif ns == 'member':
            m.color.r, m.color.g, m.color.b, m.color.a = 0.0, 0.85, 0.35, 0.95
        else:
            m.color.r, m.color.g, m.color.b, m.color.a = 1.0, 0.35, 0.05, 0.95
        return m

    def _make_goal(self, ns: str, mid: int, pose: PoseStamped, text: str, color: Tuple[float, float, float]) -> Tuple[Marker, Marker]:
        disc = Marker()
        disc.header.stamp = self.get_clock().now().to_msg()
        disc.header.frame_id = self.frame_id
        disc.ns = ns
        disc.id = mid
        disc.type = Marker.SPHERE
        disc.action = Marker.ADD
        disc.pose.position.x = pose.pose.position.x
        disc.pose.position.y = pose.pose.position.y
        disc.pose.position.z = 0.10
        disc.pose.orientation.w = 1.0
        disc.scale.x = 0.24
        disc.scale.y = 0.24
        disc.scale.z = 0.08
        disc.color.r, disc.color.g, disc.color.b = color
        disc.color.a = 0.90

        label = self._make_text(ns, mid + 1, pose, text)
        label.pose.position.z = 0.42
        label.scale.z = 0.17
        label.color.r, label.color.g, label.color.b = color
        return disc, label

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

    def _make_delete(self, ns: str, mid: int) -> Marker:
        m = Marker()
        m.header.stamp = self.get_clock().now().to_msg()
        m.header.frame_id = self.frame_id
        m.ns = ns
        m.id = mid
        m.action = Marker.DELETE
        return m

    def _tick(self) -> None:
        arr = MarkerArray()
        if self._fresh('leader') and self.poses['leader'] is not None:
            p = self.poses['leader']
            arr.markers.append(self._make_body('leader', 1, p, (0.38, 0.38, 0.18)))
            arr.markers.append(self._make_arrow('leader', 2, p, 0.55))
            arr.markers.append(self._make_text('leader', 3, p, 'leader' + self._domain_label('leader')))
        elif self.last_seen['leader'] is not None:
            arr.markers.append(self._make_delete('leader', 1))
            arr.markers.append(self._make_delete('leader_heading', 2))
            arr.markers.append(self._make_delete('leader_label', 3))
        if self._fresh('burger') and self.poses['burger'] is not None:
            p = self.poses['burger']
            arr.markers.append(self._make_body('burger', 11, p, (0.30, 0.30, 0.16)))
            arr.markers.append(self._make_arrow('burger', 12, p, 0.45))
            arr.markers.append(self._make_text('burger', 13, p, 'follower / burger' + self._domain_label('burger')))
        elif self.last_seen['burger'] is not None:
            arr.markers.append(self._make_delete('burger', 11))
            arr.markers.append(self._make_delete('burger_heading', 12))
            arr.markers.append(self._make_delete('burger_label', 13))
        if self._fresh('member') and self.poses['member'] is not None:
            p = self.poses['member']
            arr.markers.append(self._make_body('member', 21, p, (0.32, 0.32, 0.16)))
            arr.markers.append(self._make_arrow('member', 22, p, 0.48))
            arr.markers.append(self._make_text('member', 23, p, 'scout / member' + self._domain_label('member')))
        elif self.last_seen['member'] is not None:
            arr.markers.append(self._make_delete('member', 21))
            arr.markers.append(self._make_delete('member_heading', 22))
            arr.markers.append(self._make_delete('member_label', 23))
        goal_specs = (
            ('leader_user_goal', 101, 'clicked goal', (1.0, 0.9, 0.05)),
            ('leader_coord_goal', 111, 'leader nav goal', (0.1, 0.45, 1.0)),
            ('burger_goal', 121, 'burger goal', (1.0, 0.35, 0.05)),
            ('member_goal', 131, 'member goal', (0.0, 0.9, 0.4)),
        )
        for name, mid, text, color in goal_specs:
            if self._fresh(name, self.goal_stale_timeout_sec) and self.goals[name] is not None:
                markers = self._make_goal(name, mid, self.goals[name], text, color)
                arr.markers.extend(markers)
            elif self.last_seen[name] is not None:
                arr.markers.append(self._make_delete(name, mid))
                arr.markers.append(self._make_delete(name + '_label', mid + 1))
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
