#!/usr/bin/env python3
from __future__ import annotations

import math
from typing import Optional, Tuple

import rclpy
from rclpy.node import Node
from rclpy.exceptions import ParameterAlreadyDeclaredException
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import LaserScan


def _norm_angle(a: float) -> float:
    return math.atan2(math.sin(a), math.cos(a))


def _quat_to_yaw(q) -> float:
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


class PeerLaserFilter(Node):
    """Mask only the other robot's LiDAR returns.

    The node does NOT remove a whole sector. It removes ranges near the predicted
    peer bearing AND near the predicted peer range. This keeps walls behind the
    peer visible while preventing Nav2/Cartographer from treating the peer robot
    as a wall.
    """

    def __init__(self) -> None:
        super().__init__('peer_laser_filter')
        try:
            self.declare_parameter('use_sim_time', True)
        except ParameterAlreadyDeclaredException:
            pass
        self.declare_parameter('robot_name', 'robot')
        self.declare_parameter('scan_in', '/scan_nav_raw')
        self.declare_parameter('scan_out', '/scan_nav')
        self.declare_parameter('own_pose_topic', '/own_pose')
        self.declare_parameter('own_pose_backup_topic', '')
        self.declare_parameter('peer_pose_topic', '/peer_pose')
        self.declare_parameter('peer_pose_backup_topic', '')
        self.declare_parameter('peer_radius_m', 0.32)
        self.declare_parameter('angular_padding_deg', 12.0)
        self.declare_parameter('range_padding_m', 0.20)
        self.declare_parameter('max_peer_distance_m', 3.0)
        self.declare_parameter('stale_pose_sec', 1.5)
        self.declare_parameter('set_to_range_max', True)
        self.declare_parameter('log_every_n', 100)

        self.robot_name = str(self.get_parameter('robot_name').value)
        self.scan_in = self._abs(str(self.get_parameter('scan_in').value))
        self.scan_out = self._abs(str(self.get_parameter('scan_out').value))
        self.own_pose_topic = self._abs(str(self.get_parameter('own_pose_topic').value))
        self.own_pose_backup_topic = str(self.get_parameter('own_pose_backup_topic').value).strip()
        self.peer_pose_topic = self._abs(str(self.get_parameter('peer_pose_topic').value))
        self.peer_pose_backup_topic = str(self.get_parameter('peer_pose_backup_topic').value).strip()
        self.peer_radius = float(self.get_parameter('peer_radius_m').value)
        self.angular_padding = math.radians(float(self.get_parameter('angular_padding_deg').value))
        self.range_padding = float(self.get_parameter('range_padding_m').value)
        self.max_peer_distance = float(self.get_parameter('max_peer_distance_m').value)
        self.stale_pose_sec = float(self.get_parameter('stale_pose_sec').value)
        self.set_to_range_max = bool(self.get_parameter('set_to_range_max').value)
        self.log_every_n = max(0, int(self.get_parameter('log_every_n').value))

        self.own_pose: Optional[Tuple[float, float, float, object]] = None
        self.own_pose_backup: Optional[Tuple[float, float, float, object]] = None
        self.peer_pose: Optional[Tuple[float, float, float, object]] = None
        self.peer_pose_backup: Optional[Tuple[float, float, float, object]] = None
        self.count = 0
        self.masked_total = 0

        self.pub = self.create_publisher(LaserScan, self.scan_out, 10)
        self.create_subscription(LaserScan, self.scan_in, self._on_scan, 20)
        self.create_subscription(PoseStamped, self.own_pose_topic, self._on_own_pose, 10)
        if self.own_pose_backup_topic:
            self.own_pose_backup_topic = self._abs(self.own_pose_backup_topic)
            self.create_subscription(PoseStamped, self.own_pose_backup_topic, self._on_own_pose_backup, 10)
        self.create_subscription(PoseStamped, self.peer_pose_topic, self._on_peer_pose, 10)
        if self.peer_pose_backup_topic:
            self.peer_pose_backup_topic = self._abs(self.peer_pose_backup_topic)
            self.create_subscription(PoseStamped, self.peer_pose_backup_topic, self._on_peer_pose_backup, 10)

        self.get_logger().info(
            'V68_PEER_LASER_FILTER_READY | '
            f'robot={self.robot_name} scan={self.scan_in}->{self.scan_out} '
            f'own={self.own_pose_topic} own_backup={self.own_pose_backup_topic or "none"} '
            f'peer={self.peer_pose_topic} peer_backup={self.peer_pose_backup_topic or "none"} '
            f'radius={self.peer_radius:.2f} pad_deg={math.degrees(self.angular_padding):.1f} range_pad={self.range_padding:.2f}'
        )

    @staticmethod
    def _abs(topic: str) -> str:
        topic = topic.strip()
        return topic if topic.startswith('/') else '/' + topic

    def _pose_tuple(self, msg: PoseStamped):
        p = msg.pose.position
        return (float(p.x), float(p.y), _quat_to_yaw(msg.pose.orientation), self.get_clock().now())

    def _on_own_pose(self, msg: PoseStamped) -> None:
        self.own_pose = self._pose_tuple(msg)

    def _on_own_pose_backup(self, msg: PoseStamped) -> None:
        self.own_pose_backup = self._pose_tuple(msg)

    def _on_peer_pose(self, msg: PoseStamped) -> None:
        self.peer_pose = self._pose_tuple(msg)

    def _on_peer_pose_backup(self, msg: PoseStamped) -> None:
        self.peer_pose_backup = self._pose_tuple(msg)

    def _fresh(self, pose) -> bool:
        if pose is None:
            return False
        try:
            age = (self.get_clock().now() - pose[3]).nanoseconds * 1e-9
            return age <= self.stale_pose_sec
        except Exception:
            return True

    def _select_pose(self, main, backup):
        if self._fresh(main):
            return main
        if self._fresh(backup):
            return backup
        return None

    def _on_scan(self, msg: LaserScan) -> None:
        own = self._select_pose(self.own_pose, self.own_pose_backup)
        peer = self._select_pose(self.peer_pose, self.peer_pose_backup)
        out = LaserScan()
        out.header = msg.header
        out.angle_min = msg.angle_min
        out.angle_max = msg.angle_max
        out.angle_increment = msg.angle_increment
        out.time_increment = msg.time_increment
        out.scan_time = msg.scan_time
        out.range_min = msg.range_min
        out.range_max = msg.range_max
        out.ranges = list(msg.ranges)
        out.intensities = msg.intensities

        masked = 0
        peer_dist = float('nan')
        peer_bearing = float('nan')

        if own is not None and peer is not None:
            ox, oy, oyaw, _ = own
            px, py, _, _ = peer
            dx = px - ox
            dy = py - oy
            peer_dist = math.hypot(dx, dy)
            if 0.05 < peer_dist <= self.max_peer_distance:
                peer_bearing = _norm_angle(math.atan2(dy, dx) - oyaw)
                half_width = math.asin(min(0.95, self.peer_radius / max(peer_dist, 0.05))) + self.angular_padding
                r_min = max(msg.range_min, peer_dist - self.peer_radius - self.range_padding)
                r_max = min(msg.range_max, peer_dist + self.peer_radius + self.range_padding)
                replacement = msg.range_max if self.set_to_range_max else float('inf')
                for i, r in enumerate(out.ranges):
                    if not math.isfinite(r):
                        continue
                    if r < r_min or r > r_max:
                        continue
                    angle = msg.angle_min + i * msg.angle_increment
                    if abs(_norm_angle(angle - peer_bearing)) <= half_width:
                        out.ranges[i] = replacement
                        masked += 1

        self.pub.publish(out)
        self.count += 1
        self.masked_total += masked
        if self.log_every_n and self.count % self.log_every_n == 0:
            self.get_logger().info(
                'V68_PEER_LASER_FILTER | '
                f'robot={self.robot_name} n={self.count} masked={masked} total={self.masked_total} '
                f'peer_dist={peer_dist:.2f} peer_bearing_deg={math.degrees(peer_bearing) if math.isfinite(peer_bearing) else float("nan"):.1f}'
            )


def main() -> None:
    rclpy.init()
    node = PeerLaserFilter()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
