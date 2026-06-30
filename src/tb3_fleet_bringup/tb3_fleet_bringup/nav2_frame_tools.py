#!/usr/bin/env python3

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional

import rclpy
from rclpy.node import Node
from rclpy.time import Time
from builtin_interfaces.msg import Time as TimeMsg
from geometry_msgs.msg import PoseWithCovarianceStamped, TransformStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from tf2_ros import TransformBroadcaster, StaticTransformBroadcaster


def _split_csv(raw: str) -> List[str]:
    return [x.strip() for x in str(raw).split(',') if x.strip()]


def _yaw_to_quat(yaw: float):
    half = yaw * 0.5
    return (0.0, 0.0, math.sin(half), math.cos(half))


@dataclass
class RobotFrames:
    name: str
    odom_frame: str
    base_frame: str
    base_link_frame: str
    scan_frame: str


class Nav2FrameTools(Node):
    """Prepare Gazebo bridge topics for namespaced Nav2.

    For each robot name, this node:
      - reads /<robot>/odom and republishes /<robot>/odom_nav with clean frame IDs
      - publishes TF <robot>/odom -> <robot>/base_footprint from odometry
      - reads /<robot>/scan and republishes /<robot>/scan_nav with clean frame ID
      - publishes static TF <robot>/base_footprint -> <robot>/base_link
      - publishes static TF <robot>/base_footprint -> <robot>/base_scan
      - repeats /<robot>/initialpose for AMCL startup
    """

    def __init__(self) -> None:
        super().__init__('nav2_frame_tools')

        self.declare_parameter('robot_names', 'burger,waffle')
        self.declare_parameter('initial_xs', '-2.9,-1.8')
        self.declare_parameter('initial_ys', '0.5,0.5')
        self.declare_parameter('initial_yaws', '0.0,0.0')
        self.declare_parameter('initial_pose_repeat_count', 30)
        self.declare_parameter('initial_pose_period_sec', 0.5)
        self.declare_parameter('scan_z', 0.172)
        self.declare_parameter('log_every_n_odom', 100)
        self.declare_parameter('log_every_n_scan', 200)

        self.robot_names = _split_csv(self.get_parameter('robot_names').value)
        if not self.robot_names:
            raise RuntimeError('robot_names is empty')

        xs = [float(x) for x in _split_csv(self.get_parameter('initial_xs').value)]
        ys = [float(x) for x in _split_csv(self.get_parameter('initial_ys').value)]
        yaws = [float(x) for x in _split_csv(self.get_parameter('initial_yaws').value)]
        while len(xs) < len(self.robot_names):
            xs.append(0.0)
        while len(ys) < len(self.robot_names):
            ys.append(0.0)
        while len(yaws) < len(self.robot_names):
            yaws.append(0.0)

        self.initial_pose_repeat_count = int(self.get_parameter('initial_pose_repeat_count').value)
        self.initial_pose_period_sec = float(self.get_parameter('initial_pose_period_sec').value)
        self.scan_z = float(self.get_parameter('scan_z').value)
        self.log_every_n_odom = max(0, int(self.get_parameter('log_every_n_odom').value))
        self.log_every_n_scan = max(0, int(self.get_parameter('log_every_n_scan').value))

        self.tf_broadcaster = TransformBroadcaster(self)
        self.static_tf_broadcaster = StaticTransformBroadcaster(self)

        self.frames: Dict[str, RobotFrames] = {}
        self.odom_counts: Dict[str, int] = {}
        self.scan_counts: Dict[str, int] = {}
        self.initial_counts: Dict[str, int] = {}
        self.initial_poses: Dict[str, tuple] = {}
        self.odom_pubs = {}
        self.scan_pubs = {}
        self.initial_pubs = {}
        self.subs = []

        static_transforms: List[TransformStamped] = []

        for i, robot in enumerate(self.robot_names):
            frames = RobotFrames(
                name=robot,
                odom_frame=f'{robot}/odom',
                base_frame=f'{robot}/base_footprint',
                base_link_frame=f'{robot}/base_link',
                scan_frame=f'{robot}/base_scan',
            )
            self.frames[robot] = frames
            self.odom_counts[robot] = 0
            self.scan_counts[robot] = 0
            self.initial_counts[robot] = 0
            self.initial_poses[robot] = (xs[i], ys[i], yaws[i])

            self.odom_pubs[robot] = self.create_publisher(Odometry, f'/{robot}/odom_nav', 10)
            self.scan_pubs[robot] = self.create_publisher(LaserScan, f'/{robot}/scan_nav', 10)
            self.initial_pubs[robot] = self.create_publisher(PoseWithCovarianceStamped, f'/{robot}/initialpose', 10)

            self.subs.append(self.create_subscription(Odometry, f'/{robot}/odom', lambda msg, r=robot: self._on_odom(r, msg), 20))
            self.subs.append(self.create_subscription(LaserScan, f'/{robot}/scan', lambda msg, r=robot: self._on_scan(r, msg), 20))

            # base_footprint -> base_link identity
            t1 = TransformStamped()
            t1.header.frame_id = frames.base_frame
            t1.child_frame_id = frames.base_link_frame
            t1.transform.rotation.w = 1.0
            static_transforms.append(t1)

            # base_footprint -> base_scan approximate TB3 laser pose
            t2 = TransformStamped()
            t2.header.frame_id = frames.base_frame
            t2.child_frame_id = frames.scan_frame
            t2.transform.translation.z = self.scan_z
            t2.transform.rotation.w = 1.0
            static_transforms.append(t2)

            self.get_logger().info(
                'NAV2_FRAME_TOOLS_READY | '
                f'robot={robot} | odom_in=/{robot}/odom odom_out=/{robot}/odom_nav | '
                f'scan_in=/{robot}/scan scan_out=/{robot}/scan_nav | '
                f'frames=map->{frames.odom_frame}->{frames.base_frame}->{frames.scan_frame}'
            )

        now = self.get_clock().now().to_msg()
        for t in static_transforms:
            t.header.stamp = now
        self.static_tf_broadcaster.sendTransform(static_transforms)
        self.create_timer(self.initial_pose_period_sec, self._initial_pose_tick)

    def _on_odom(self, robot: str, msg: Odometry) -> None:
        frames = self.frames[robot]
        stamp = msg.header.stamp
        if stamp.sec == 0 and stamp.nanosec == 0:
            stamp = self.get_clock().now().to_msg()

        out = Odometry()
        out.header = msg.header
        out.header.stamp = stamp
        out.header.frame_id = frames.odom_frame
        out.child_frame_id = frames.base_frame
        out.pose = msg.pose
        out.twist = msg.twist
        self.odom_pubs[robot].publish(out)

        tf = TransformStamped()
        tf.header.stamp = stamp
        tf.header.frame_id = frames.odom_frame
        tf.child_frame_id = frames.base_frame
        tf.transform.translation.x = msg.pose.pose.position.x
        tf.transform.translation.y = msg.pose.pose.position.y
        tf.transform.translation.z = msg.pose.pose.position.z
        tf.transform.rotation = msg.pose.pose.orientation
        self.tf_broadcaster.sendTransform(tf)

        self.odom_counts[robot] += 1
        if self.log_every_n_odom and self.odom_counts[robot] % self.log_every_n_odom == 0:
            p = msg.pose.pose.position
            self.get_logger().info(f'NAV2_ODOM_REFRAME | robot={robot} n={self.odom_counts[robot]} xy=({p.x:.2f},{p.y:.2f})')

    def _on_scan(self, robot: str, msg: LaserScan) -> None:
        frames = self.frames[robot]
        out = LaserScan()
        out.header = msg.header
        out.header.frame_id = frames.scan_frame
        out.angle_min = msg.angle_min
        out.angle_max = msg.angle_max
        out.angle_increment = msg.angle_increment
        out.time_increment = msg.time_increment
        out.scan_time = msg.scan_time
        out.range_min = msg.range_min
        out.range_max = msg.range_max
        out.ranges = msg.ranges
        out.intensities = msg.intensities
        self.scan_pubs[robot].publish(out)

        self.scan_counts[robot] += 1
        if self.log_every_n_scan and self.scan_counts[robot] % self.log_every_n_scan == 0:
            self.get_logger().info(f'NAV2_SCAN_REFRAME | robot={robot} n={self.scan_counts[robot]} frame={frames.scan_frame}')

    def _initial_pose_tick(self) -> None:
        for robot in self.robot_names:
            if self.initial_counts[robot] >= self.initial_pose_repeat_count:
                continue
            x, y, yaw = self.initial_poses[robot]
            qx, qy, qz, qw = _yaw_to_quat(yaw)
            msg = PoseWithCovarianceStamped()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = 'map'
            msg.pose.pose.position.x = x
            msg.pose.pose.position.y = y
            msg.pose.pose.orientation.x = qx
            msg.pose.pose.orientation.y = qy
            msg.pose.pose.orientation.z = qz
            msg.pose.pose.orientation.w = qw
            msg.pose.covariance[0] = 0.25
            msg.pose.covariance[7] = 0.25
            msg.pose.covariance[35] = 0.0685
            self.initial_pubs[robot].publish(msg)
            self.initial_counts[robot] += 1
            self.get_logger().info(
                f'NAV2_INITIALPOSE_PUB | robot={robot} count={self.initial_counts[robot]}/{self.initial_pose_repeat_count} '
                f'xy=({x:.2f},{y:.2f}) yaw={yaw:.2f}'
            )


def main() -> None:
    rclpy.init()
    node = Nav2FrameTools()
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
