#!/usr/bin/env python3

from __future__ import annotations

import math
from typing import Optional, Tuple

import rclpy
from rclpy.node import Node
from rclpy.exceptions import ParameterAlreadyDeclaredException
from geometry_msgs.msg import PoseWithCovarianceStamped, TransformStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from tf2_ros import TransformBroadcaster, StaticTransformBroadcaster


def _yaw_to_quat(yaw: float):
    half = yaw * 0.5
    return (0.0, 0.0, math.sin(half), math.cos(half))


def _quat_to_yaw(q) -> float:
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


def _norm_angle(a: float) -> float:
    return math.atan2(math.sin(a), math.cos(a))


class SingleDomainNav2FrameTools(Node):
    """Reframe one Gazebo TB3 into a no-namespace Nav2 API inside one ROS_DOMAIN_ID.

    Input from ros_gz_bridge:
      /odom, /scan

    Output for Nav2:
      /odom_nav, /scan_nav, /initialpose, TF odom->base_footprint, static base TFs.

    v40 important change:
      Gazebo diff-drive odometry may start either at zero or at the spawned world pose
      depending on model/plugin version. This node captures the first odom sample as
      the local odom origin and republishes /odom_nav relative to that origin. Then
      map_odom_localization can always publish map->odom as the known spawn pose.
    """

    def __init__(self) -> None:
        super().__init__('single_domain_nav2_frame_tools')

        self.declare_parameter('robot_name', 'robot')
        try:
            self.declare_parameter('use_sim_time', True)
        except ParameterAlreadyDeclaredException:
            pass
        self.declare_parameter('odom_in', '/odom')
        self.declare_parameter('scan_in', '/scan')
        self.declare_parameter('odom_out', '/odom_nav')
        self.declare_parameter('scan_out', '/scan_nav')
        self.declare_parameter('initialpose_topic', '/initialpose')
        self.declare_parameter('map_frame', 'map')
        self.declare_parameter('odom_frame', 'odom')
        self.declare_parameter('base_frame', 'base_footprint')
        self.declare_parameter('base_link_frame', 'base_link')
        self.declare_parameter('scan_frame', 'base_scan')
        self.declare_parameter('scan_z', 0.172)
        self.declare_parameter('initial_x', 0.0)
        self.declare_parameter('initial_y', 0.0)
        self.declare_parameter('initial_yaw', 0.0)
        self.declare_parameter('reset_odom_origin_on_first_msg', True)
        self.declare_parameter('initial_pose_repeat_count', 80)
        self.declare_parameter('initial_pose_period_sec', 0.25)
        self.declare_parameter('log_every_n_odom', 100)
        self.declare_parameter('log_every_n_scan', 200)
        self.declare_parameter('drop_non_monotonic_stamps', True)
        self.declare_parameter('max_stamp_jump_sec', 5.0)

        self.robot_name = str(self.get_parameter('robot_name').value)
        self.odom_in = self._abs(str(self.get_parameter('odom_in').value))
        self.scan_in = self._abs(str(self.get_parameter('scan_in').value))
        self.odom_out = self._abs(str(self.get_parameter('odom_out').value))
        self.scan_out = self._abs(str(self.get_parameter('scan_out').value))
        self.initialpose_topic = self._abs(str(self.get_parameter('initialpose_topic').value))
        self.map_frame = str(self.get_parameter('map_frame').value)
        self.odom_frame = str(self.get_parameter('odom_frame').value)
        self.base_frame = str(self.get_parameter('base_frame').value)
        self.base_link_frame = str(self.get_parameter('base_link_frame').value)
        self.scan_frame = str(self.get_parameter('scan_frame').value)
        self.scan_z = float(self.get_parameter('scan_z').value)
        self.initial_x = float(self.get_parameter('initial_x').value)
        self.initial_y = float(self.get_parameter('initial_y').value)
        self.initial_yaw = float(self.get_parameter('initial_yaw').value)
        self.reset_origin = bool(self.get_parameter('reset_odom_origin_on_first_msg').value)
        self.initial_pose_repeat_count = int(self.get_parameter('initial_pose_repeat_count').value)
        self.initial_pose_period_sec = float(self.get_parameter('initial_pose_period_sec').value)
        self.log_every_n_odom = max(0, int(self.get_parameter('log_every_n_odom').value))
        self.log_every_n_scan = max(0, int(self.get_parameter('log_every_n_scan').value))
        self.drop_non_monotonic_stamps = bool(self.get_parameter('drop_non_monotonic_stamps').value)
        self.max_stamp_jump_sec = max(0.0, float(self.get_parameter('max_stamp_jump_sec').value))

        self.tf_broadcaster = TransformBroadcaster(self)
        self.static_tf_broadcaster = StaticTransformBroadcaster(self)
        self.odom_pub = self.create_publisher(Odometry, self.odom_out, 10)
        self.scan_pub = self.create_publisher(LaserScan, self.scan_out, 10)
        self.initial_pub = self.create_publisher(PoseWithCovarianceStamped, self.initialpose_topic, 10)
        self.odom_sub = self.create_subscription(Odometry, self.odom_in, self._on_odom, 20)
        self.scan_sub = self.create_subscription(LaserScan, self.scan_in, self._on_scan, 20)

        self.odom_count = 0
        self.scan_count = 0
        self.initial_count = 0
        self.last_sim_stamp = None
        self.last_odom_stamp_sec: Optional[float] = None
        self.last_scan_stamp_sec: Optional[float] = None
        self.dropped_odom_stamps = 0
        self.dropped_scan_stamps = 0
        self.origin: Optional[Tuple[float, float, float]] = None

        now = self.get_clock().now().to_msg()
        t1 = TransformStamped()
        t1.header.stamp = now
        t1.header.frame_id = self.base_frame
        t1.child_frame_id = self.base_link_frame
        t1.transform.rotation.w = 1.0

        t2 = TransformStamped()
        t2.header.stamp = now
        t2.header.frame_id = self.base_frame
        t2.child_frame_id = self.scan_frame
        t2.transform.translation.z = self.scan_z
        t2.transform.rotation.w = 1.0
        self.static_tf_broadcaster.sendTransform([t1, t2])

        self.create_timer(self.initial_pose_period_sec, self._initial_pose_tick)
        self.get_logger().info(
            'V40_FRAME_TOOLS_READY | '
            f'robot={self.robot_name} odom_in={self.odom_in} odom_out={self.odom_out} '
            f'scan_in={self.scan_in} scan_out={self.scan_out} '
            f'reset_odom_origin={self.reset_origin} drop_bad_stamps={self.drop_non_monotonic_stamps} '
            f'max_stamp_jump_sec={self.max_stamp_jump_sec:.1f} '
            f'frames={self.map_frame}->{self.odom_frame}->{self.base_frame}->{self.scan_frame}'
        )

    @staticmethod
    def _abs(topic: str) -> str:
        topic = topic.strip()
        return topic if topic.startswith('/') else '/' + topic

    def _relative_pose(self, x: float, y: float, yaw: float) -> Tuple[float, float, float]:
        if self.origin is None:
            self.origin = (x, y, yaw) if self.reset_origin else (0.0, 0.0, 0.0)
            ox, oy, oyaw = self.origin
            self.get_logger().info(
                f'V40_ODOM_ORIGIN_CAPTURED | robot={self.robot_name} raw_origin=({ox:.3f},{oy:.3f},{oyaw:.3f}) reset={self.reset_origin}'
            )
        ox, oy, oyaw = self.origin
        dx = x - ox
        dy = y - oy
        c = math.cos(-oyaw)
        s = math.sin(-oyaw)
        rx = c * dx - s * dy
        ry = s * dx + c * dy
        ryaw = _norm_angle(yaw - oyaw)
        return rx, ry, ryaw

    @staticmethod
    def _stamp_to_sec(stamp) -> float:
        return float(stamp.sec) + float(stamp.nanosec) * 1e-9

    def _stamp_is_usable(self, stream: str, stamp) -> bool:
        if not self.drop_non_monotonic_stamps:
            return True

        current = self._stamp_to_sec(stamp)
        if stream == 'odom':
            previous = self.last_odom_stamp_sec
        else:
            previous = self.last_scan_stamp_sec

        if previous is None:
            if stream == 'odom':
                self.last_odom_stamp_sec = current
            else:
                self.last_scan_stamp_sec = current
            return True

        delta = current - previous
        bad_backwards = delta <= 0.0
        bad_jump = self.max_stamp_jump_sec > 0.0 and abs(delta) > self.max_stamp_jump_sec
        if bad_backwards or bad_jump:
            if stream == 'odom':
                self.dropped_odom_stamps += 1
                dropped = self.dropped_odom_stamps
            else:
                self.dropped_scan_stamps += 1
                dropped = self.dropped_scan_stamps

            if dropped <= 5 or dropped % 100 == 0:
                self.get_logger().warn(
                    f'V40_DROP_BAD_{stream.upper()}_STAMP | robot={self.robot_name} '
                    f'dropped={dropped} previous={previous:.3f} current={current:.3f} delta={delta:.3f}'
                )
            return False

        if stream == 'odom':
            self.last_odom_stamp_sec = current
        else:
            self.last_scan_stamp_sec = current
        return True

    def _on_odom(self, msg: Odometry) -> None:
        stamp = msg.header.stamp
        if stamp.sec == 0 and stamp.nanosec == 0:
            stamp = self.get_clock().now().to_msg()
        if not self._stamp_is_usable('odom', stamp):
            return

        p = msg.pose.pose.position
        raw_yaw = _quat_to_yaw(msg.pose.pose.orientation)
        rel_x, rel_y, rel_yaw = self._relative_pose(p.x, p.y, raw_yaw)
        qx, qy, qz, qw = _yaw_to_quat(rel_yaw)

        out = Odometry()
        out.header = msg.header
        out.header.stamp = stamp
        out.header.frame_id = self.odom_frame
        out.child_frame_id = self.base_frame
        out.pose = msg.pose
        out.pose.pose.position.x = rel_x
        out.pose.pose.position.y = rel_y
        out.pose.pose.position.z = p.z
        out.pose.pose.orientation.x = qx
        out.pose.pose.orientation.y = qy
        out.pose.pose.orientation.z = qz
        out.pose.pose.orientation.w = qw
        out.twist = msg.twist
        self.odom_pub.publish(out)
        self.last_sim_stamp = stamp

        tf = TransformStamped()
        tf.header.stamp = stamp
        tf.header.frame_id = self.odom_frame
        tf.child_frame_id = self.base_frame
        tf.transform.translation.x = rel_x
        tf.transform.translation.y = rel_y
        tf.transform.translation.z = p.z
        tf.transform.rotation = out.pose.pose.orientation
        self.tf_broadcaster.sendTransform(tf)

        self.odom_count += 1
        if self.log_every_n_odom and self.odom_count % self.log_every_n_odom == 0:
            self.get_logger().info(
                f'V40_ODOM_NAV | robot={self.robot_name} n={self.odom_count} rel_xy=({rel_x:.2f},{rel_y:.2f}) raw_xy=({p.x:.2f},{p.y:.2f})'
            )

    def _on_scan(self, msg: LaserScan) -> None:
        stamp = msg.header.stamp
        if stamp.sec == 0 and stamp.nanosec == 0:
            stamp = self.get_clock().now().to_msg()
        if not self._stamp_is_usable('scan', stamp):
            return

        out = LaserScan()
        out.header = msg.header
        out.header.stamp = stamp
        out.header.frame_id = self.scan_frame
        out.angle_min = msg.angle_min
        out.angle_max = msg.angle_max
        out.angle_increment = msg.angle_increment
        out.time_increment = msg.time_increment
        out.scan_time = msg.scan_time
        out.range_min = msg.range_min
        out.range_max = msg.range_max
        out.ranges = msg.ranges
        out.intensities = msg.intensities
        self.scan_pub.publish(out)
        self.last_sim_stamp = out.header.stamp

        self.scan_count += 1
        if self.log_every_n_scan and self.scan_count % self.log_every_n_scan == 0:
            self.get_logger().info(f'V40_SCAN_NAV | robot={self.robot_name} n={self.scan_count} frame={self.scan_frame}')

    def _initial_pose_tick(self) -> None:
        if self.initial_count >= self.initial_pose_repeat_count:
            return
        qx, qy, qz, qw = _yaw_to_quat(self.initial_yaw)
        msg = PoseWithCovarianceStamped()
        if self.last_sim_stamp is not None and not (self.last_sim_stamp.sec == 0 and self.last_sim_stamp.nanosec == 0):
            msg.header.stamp = self.last_sim_stamp
        else:
            msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.map_frame
        msg.pose.pose.position.x = self.initial_x
        msg.pose.pose.position.y = self.initial_y
        msg.pose.pose.orientation.x = qx
        msg.pose.pose.orientation.y = qy
        msg.pose.pose.orientation.z = qz
        msg.pose.pose.orientation.w = qw
        msg.pose.covariance[0] = 0.25
        msg.pose.covariance[7] = 0.25
        msg.pose.covariance[35] = 0.0685
        self.initial_pub.publish(msg)
        self.initial_count += 1
        if self.initial_count <= 5 or self.initial_count == self.initial_pose_repeat_count:
            self.get_logger().info(
                f'V40_INITIALPOSE_PUB | robot={self.robot_name} count={self.initial_count}/{self.initial_pose_repeat_count} '
                f'xy=({self.initial_x:.2f},{self.initial_y:.2f}) yaw={self.initial_yaw:.2f}'
            )


def main() -> None:
    rclpy.init()
    node = SingleDomainNav2FrameTools()
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
