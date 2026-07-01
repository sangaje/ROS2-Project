#!/usr/bin/env python3

from __future__ import annotations

import math
from typing import Optional, Tuple

import rclpy
from rclpy.node import Node
from rclpy.exceptions import ParameterAlreadyDeclaredException
from geometry_msgs.msg import PoseWithCovarianceStamped, TransformStamped, Quaternion
from nav_msgs.msg import Odometry
from tf2_ros import TransformBroadcaster


def _safe_declare(node: Node, name: str, default):
    try:
        node.declare_parameter(name, default)
    except ParameterAlreadyDeclaredException:
        pass
    return node.get_parameter(name).value


def _yaw_to_quat(yaw: float) -> Quaternion:
    q = Quaternion()
    half = yaw * 0.5
    q.x = 0.0
    q.y = 0.0
    q.z = math.sin(half)
    q.w = math.cos(half)
    return q


def _quat_to_yaw(q: Quaternion) -> float:
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


def _norm(a: float) -> float:
    return math.atan2(math.sin(a), math.cos(a))


class MapOdomLocalization(Node):
    """Deterministic simulation localization for Nav2.

    This node owns map->odom. /odom_nav is already origin-normalized by
    single_domain_nav2_frame_tools_v40, so map->odom is simply the known spawn pose.
    """

    def __init__(self) -> None:
        super().__init__('map_odom_localization')
        _safe_declare(self, 'use_sim_time', True)
        self.robot_name = str(_safe_declare(self, 'robot_name', 'robot'))
        self.map_frame = str(_safe_declare(self, 'map_frame', 'map'))
        self.odom_frame = str(_safe_declare(self, 'odom_frame', 'odom'))
        self.base_frame = str(_safe_declare(self, 'base_frame', 'base_footprint'))
        self.odom_topic = self._abs(str(_safe_declare(self, 'odom_topic', '/odom_nav')))
        self.amcl_pose_topic = self._abs(str(_safe_declare(self, 'amcl_pose_topic', '/amcl_pose')))
        self.initial_x = float(_safe_declare(self, 'initial_x', 0.0))
        self.initial_y = float(_safe_declare(self, 'initial_y', 0.0))
        self.initial_yaw = float(_safe_declare(self, 'initial_yaw', 0.0))
        self.publish_rate_hz = max(1.0, float(_safe_declare(self, 'publish_rate_hz', 30.0)))
        self.publish_amcl_pose = bool(_safe_declare(self, 'publish_amcl_pose', True))
        self.log_every_n = max(0, int(_safe_declare(self, 'log_every_n', 300)))

        self.tf_broadcaster = TransformBroadcaster(self)
        self.amcl_pose_pub = self.create_publisher(PoseWithCovarianceStamped, self.amcl_pose_topic, 10)
        self.odom_sub = self.create_subscription(Odometry, self.odom_topic, self._on_odom, 30)
        self.last_odom: Optional[Odometry] = None
        self.tick_count = 0

        self.create_timer(1.0 / self.publish_rate_hz, self._tick)
        self.get_logger().info(
            'V40_MAP_ODOM_LOCALIZATION_READY | '
            f'robot={self.robot_name} owns_tf={self.map_frame}->{self.odom_frame} odom={self.odom_topic} '
            f'initial=({self.initial_x:.2f},{self.initial_y:.2f},{self.initial_yaw:.2f})'
        )

    @staticmethod
    def _abs(topic: str) -> str:
        topic = topic.strip()
        return topic if topic.startswith('/') else '/' + topic

    def _on_odom(self, msg: Odometry) -> None:
        self.last_odom = msg

    def _stamp(self):
        # Always use node clock to guarantee monotonically increasing TF timestamps.
        return self.get_clock().now().to_msg()

    def _map_pose_from_odom(self) -> Tuple[float, float, float]:
        if self.last_odom is None:
            return self.initial_x, self.initial_y, self.initial_yaw
        p = self.last_odom.pose.pose.position
        yaw_odom = _quat_to_yaw(self.last_odom.pose.pose.orientation)
        c = math.cos(self.initial_yaw)
        s = math.sin(self.initial_yaw)
        mx = self.initial_x + c * p.x - s * p.y
        my = self.initial_y + s * p.x + c * p.y
        myaw = _norm(self.initial_yaw + yaw_odom)
        return mx, my, myaw

    def _tick(self) -> None:
        stamp = self._stamp()

        tf = TransformStamped()
        tf.header.stamp = stamp
        tf.header.frame_id = self.map_frame
        tf.child_frame_id = self.odom_frame
        tf.transform.translation.x = self.initial_x
        tf.transform.translation.y = self.initial_y
        tf.transform.translation.z = 0.0
        tf.transform.rotation = _yaw_to_quat(self.initial_yaw)
        self.tf_broadcaster.sendTransform(tf)

        if self.publish_amcl_pose:
            x, y, yaw = self._map_pose_from_odom()
            msg = PoseWithCovarianceStamped()
            msg.header.stamp = stamp
            msg.header.frame_id = self.map_frame
            msg.pose.pose.position.x = x
            msg.pose.pose.position.y = y
            msg.pose.pose.position.z = 0.0
            msg.pose.pose.orientation = _yaw_to_quat(yaw)
            msg.pose.covariance[0] = 0.02
            msg.pose.covariance[7] = 0.02
            msg.pose.covariance[35] = 0.02
            self.amcl_pose_pub.publish(msg)

        self.tick_count += 1
        if self.log_every_n and self.tick_count % self.log_every_n == 0:
            x, y, yaw = self._map_pose_from_odom()
            self.get_logger().info(
                f'V40_MAP_ODOM_TICK | robot={self.robot_name} n={self.tick_count} map_pose=({x:.2f},{y:.2f},{yaw:.2f}) odom_seen={self.last_odom is not None}'
            )


def main() -> None:
    rclpy.init()
    node = MapOdomLocalization()
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
