#!/usr/bin/env python3
"""Auto Initial Pose - map 수신 후 자동으로 AMCL 에 initialpose 발행.

와플의 시작 위치와 방향이 정해져 있다고 가정 (config 또는 파라미터).
map 토픽이 처음 도착하면 그때 /initialpose 발행.
(AMCL 이 map 없이 initialpose 받으면 무시되므로 순서 보장 필요)

토픽:
    Sub: /map (또는 파라미터로 변경 가능)  OccupancyGrid
    Pub: /initialpose  PoseWithCovarianceStamped

파라미터:
    map_topic:           /map        트리거 토픽 (map_relay 출력)
    initialpose_topic:   /initialpose
    initial_x:           0.0         map frame 기준 x
    initial_y:           0.0         map frame 기준 y
    initial_yaw_deg:     0.0         heading (도)
    cov_xy:              0.25        x/y 분산 (m²)
    cov_yaw:             0.0685      yaw 분산 (rad²) ≈ (15°)²
    delay_sec:           1.0         map 수신 후 발행 대기 (AMCL 준비)
    republish_count:     3           안전 위해 N회 반복 발행
    republish_interval:  0.5         반복 간격
"""

from __future__ import annotations

import math

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy

from geometry_msgs.msg import PoseWithCovarianceStamped, Quaternion
from nav_msgs.msg import OccupancyGrid


def yaw_to_quaternion(yaw_rad: float) -> Quaternion:
    q = Quaternion()
    q.x = 0.0
    q.y = 0.0
    q.z = math.sin(yaw_rad / 2.0)
    q.w = math.cos(yaw_rad / 2.0)
    return q


class AutoInitialPose(Node):
    def __init__(self):
        super().__init__('auto_initialpose')

        self.declare_parameter('map_topic', '/map')
        self.declare_parameter('initialpose_topic', '/initialpose')
        self.declare_parameter('initial_x', 0.0)
        self.declare_parameter('initial_y', 0.0)
        self.declare_parameter('initial_yaw_deg', 0.0)
        self.declare_parameter('cov_xy', 0.25)
        self.declare_parameter('cov_yaw', 0.0685)
        self.declare_parameter('delay_sec', 1.0)
        self.declare_parameter('republish_count', 3)
        self.declare_parameter('republish_interval', 0.5)

        self.map_topic = self.get_parameter('map_topic').value
        self.pose_topic = self.get_parameter('initialpose_topic').value
        self.init_x = float(self.get_parameter('initial_x').value)
        self.init_y = float(self.get_parameter('initial_y').value)
        self.init_yaw_rad = math.radians(float(
            self.get_parameter('initial_yaw_deg').value))
        self.cov_xy = float(self.get_parameter('cov_xy').value)
        self.cov_yaw = float(self.get_parameter('cov_yaw').value)
        self.delay_sec = float(self.get_parameter('delay_sec').value)
        self.republish_count = int(self.get_parameter('republish_count').value)
        self.republish_interval = float(
            self.get_parameter('republish_interval').value)

        # State
        self.map_received = False
        self.publish_done_count = 0
        self.delay_timer = None
        self.repub_timer = None
        self.map_frame = 'map'

        # QoS - map 은 latched
        latched_qos = QoSProfile(
            depth=1,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
        )

        # Pub/Sub
        self.create_subscription(
            OccupancyGrid, self.map_topic, self.on_map, latched_qos)
        # initialpose 는 transient_local 로 보내면 AMCL 이 늦게 켜져도 받음
        self.pub_pose = self.create_publisher(
            PoseWithCovarianceStamped, self.pose_topic, 10)

        self.get_logger().info("=" * 50)
        self.get_logger().info("Auto Initial Pose")
        self.get_logger().info("=" * 50)
        self.get_logger().info(
            f"Trigger: {self.map_topic} (첫 수신 후 {self.delay_sec}s 뒤 발행)")
        self.get_logger().info(f"Publish: {self.pose_topic}")
        self.get_logger().info(
            f"Initial pose: x={self.init_x:.2f}, y={self.init_y:.2f}, "
            f"yaw={math.degrees(self.init_yaw_rad):.1f}°")
        self.get_logger().info(
            f"Covariance: xy={self.cov_xy:.3f} m², "
            f"yaw={self.cov_yaw:.4f} rad² "
            f"({math.degrees(math.sqrt(self.cov_yaw)):.1f}° 1σ)")
        self.get_logger().info(
            f"Republish: {self.republish_count}회 x {self.republish_interval}s")
        self.get_logger().info("=== Ready ===")

    def on_map(self, msg: OccupancyGrid):
        if self.map_received:
            return
        self.map_received = True
        self.map_frame = msg.header.frame_id or 'map'
        info = msg.info
        self.get_logger().info(
            f"첫 map 수신: {info.width}x{info.height} @ "
            f"{info.resolution:.3f}m/cell, frame={self.map_frame}, "
            f"origin=({info.origin.position.x:+.2f}, "
            f"{info.origin.position.y:+.2f})")
        self.get_logger().info(
            f"{self.delay_sec}s 후 initialpose 발행 시작...")
        # delay 후 첫 발행
        self.delay_timer = self.create_timer(
            self.delay_sec, self._first_publish)

    def _first_publish(self):
        # one-shot
        if self.delay_timer is not None:
            self.delay_timer.cancel()
            self.delay_timer = None
        self._publish_once()
        # 추가 발행 예약
        if self.republish_count > 1:
            self.repub_timer = self.create_timer(
                self.republish_interval, self._republish_callback)

    def _republish_callback(self):
        if self.publish_done_count >= self.republish_count:
            self.repub_timer.cancel()
            self.repub_timer = None
            self.get_logger().info(
                f"initialpose 발행 완료 ({self.publish_done_count}회). 종료 대기.")
            return
        self._publish_once()

    def _publish_once(self):
        msg = PoseWithCovarianceStamped()
        msg.header.frame_id = self.map_frame
        msg.header.stamp = self.get_clock().now().to_msg()

        msg.pose.pose.position.x = self.init_x
        msg.pose.pose.position.y = self.init_y
        msg.pose.pose.position.z = 0.0
        msg.pose.pose.orientation = yaw_to_quaternion(self.init_yaw_rad)

        # 6x6 covariance: [x, y, z, roll, pitch, yaw]
        cov = [0.0] * 36
        cov[0]  = self.cov_xy    # x-x
        cov[7]  = self.cov_xy    # y-y
        cov[14] = 0.0            # z-z (2D 이므로)
        cov[21] = 0.0            # roll
        cov[28] = 0.0            # pitch
        cov[35] = self.cov_yaw   # yaw-yaw
        msg.pose.covariance = cov

        self.pub_pose.publish(msg)
        self.publish_done_count += 1
        self.get_logger().info(
            f"initialpose 발행 #{self.publish_done_count}: "
            f"({self.init_x:+.2f}, {self.init_y:+.2f}, "
            f"{math.degrees(self.init_yaw_rad):+.1f}°)")


def main():
    rclpy.init()
    node = None
    try:
        node = AutoInitialPose()
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("\n중단됨.")
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()