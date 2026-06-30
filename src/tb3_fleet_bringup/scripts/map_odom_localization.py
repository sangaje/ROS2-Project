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

    v44 adds relative-origin mode for live SLAM sharing. When Domain25 uses
    Cartographer, the map frame starts at the Waffle initial pose, not at the
    Gazebo world origin. Burger therefore needs map->odom from its spawn pose
    relative to Waffle's spawn pose.
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
        self.initialpose_topic = self._abs(str(_safe_declare(self, 'initialpose_topic', '/initialpose')))
        self.initial_x_raw = float(_safe_declare(self, 'initial_x', 0.0))
        self.initial_y_raw = float(_safe_declare(self, 'initial_y', 0.0))
        self.initial_yaw_raw = float(_safe_declare(self, 'initial_yaw', 0.0))
        self.relative_to_world_origin = bool(_safe_declare(self, 'relative_to_world_origin', False))
        self.world_origin_x = float(_safe_declare(self, 'world_origin_x', 0.0))
        self.world_origin_y = float(_safe_declare(self, 'world_origin_y', 0.0))
        self.world_origin_yaw = float(_safe_declare(self, 'world_origin_yaw', 0.0))

        if self.relative_to_world_origin:
            dx = self.initial_x_raw - self.world_origin_x
            dy = self.initial_y_raw - self.world_origin_y
            c = math.cos(-self.world_origin_yaw)
            ss = math.sin(-self.world_origin_yaw)
            self.initial_x = c * dx - ss * dy
            self.initial_y = ss * dx + c * dy
            self.initial_yaw = _norm(self.initial_yaw_raw - self.world_origin_yaw)
        else:
            self.initial_x = self.initial_x_raw
            self.initial_y = self.initial_y_raw
            self.initial_yaw = self.initial_yaw_raw
        self.publish_rate_hz = max(1.0, float(_safe_declare(self, 'publish_rate_hz', 30.0)))
        self.publish_amcl_pose = bool(_safe_declare(self, 'publish_amcl_pose', True))
        self.log_every_n = max(0, int(_safe_declare(self, 'log_every_n', 300)))

        self.tf_broadcaster = TransformBroadcaster(self)
        self.amcl_pose_pub = self.create_publisher(PoseWithCovarianceStamped, self.amcl_pose_topic, 10)
        self.odom_sub = self.create_subscription(Odometry, self.odom_topic, self._on_odom, 30)
        self.initialpose_sub = self.create_subscription(
            PoseWithCovarianceStamped,
            self.initialpose_topic,
            self._on_initialpose,
            10,
        )
        self.last_odom: Optional[Odometry] = None
        self.tick_count = 0

        self.create_timer(1.0 / self.publish_rate_hz, self._tick)
        self.get_logger().info(
            'V44_MAP_ODOM_LOCALIZATION_READY | '
            f'robot={self.robot_name} owns_tf={self.map_frame}->{self.odom_frame} odom={self.odom_topic} '
            f'initial=({self.initial_x:.2f},{self.initial_y:.2f},{self.initial_yaw:.2f}) raw=({self.initial_x_raw:.2f},{self.initial_y_raw:.2f},{self.initial_yaw_raw:.2f}) relative={self.relative_to_world_origin} origin=({self.world_origin_x:.2f},{self.world_origin_y:.2f},{self.world_origin_yaw:.2f})'
        )

    @staticmethod
    def _abs(topic: str) -> str:
        topic = topic.strip()
        return topic if topic.startswith('/') else '/' + topic

    def _on_odom(self, msg: Odometry) -> None:
        self.last_odom = msg

    def _on_initialpose(self, msg: PoseWithCovarianceStamped) -> None:
        """Move map->odom so the robot appears at the RViz 2D Pose Estimate."""
        desired_x = float(msg.pose.pose.position.x)
        desired_y = float(msg.pose.pose.position.y)
        desired_yaw = _quat_to_yaw(msg.pose.pose.orientation)

        if self.last_odom is None:
            self.initial_x = desired_x
            self.initial_y = desired_y
            self.initial_yaw = desired_yaw
        else:
            odom_pose = self.last_odom.pose.pose
            odom_x = float(odom_pose.position.x)
            odom_y = float(odom_pose.position.y)
            odom_yaw = _quat_to_yaw(odom_pose.orientation)

            map_odom_yaw = _norm(desired_yaw - odom_yaw)
            c = math.cos(map_odom_yaw)
            s = math.sin(map_odom_yaw)
            self.initial_x = desired_x - (c * odom_x - s * odom_y)
            self.initial_y = desired_y - (s * odom_x + c * odom_y)
            self.initial_yaw = map_odom_yaw

        self.get_logger().warn(
            'RVIZ_INITIALPOSE_APPLIED | '
            f'topic={self.initialpose_topic} desired_map_pose='
            f'({desired_x:.2f},{desired_y:.2f},{desired_yaw:.2f}) '
            f'new_map_odom=({self.initial_x:.2f},{self.initial_y:.2f},{self.initial_yaw:.2f})'
        )

    def _stamp(self):
        if self.last_odom is not None:
            st = self.last_odom.header.stamp
            if not (st.sec == 0 and st.nanosec == 0):
                return st
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
                f'V44_MAP_ODOM_TICK | robot={self.robot_name} n={self.tick_count} map_pose=({x:.2f},{y:.2f},{yaw:.2f}) odom_seen={self.last_odom is not None}'
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
