#!/usr/bin/env python3

import math

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseWithCovarianceStamped


def yaw_to_quat(yaw: float):
    qz = math.sin(yaw * 0.5)
    qw = math.cos(yaw * 0.5)
    return 0.0, 0.0, qz, qw


class InitialPosePublisher(Node):
    """
    Repeatedly publishes /initialpose for follower AMCL.

    This is intentionally separated from map_server: in the shared-map test,
    the master domain publishes the saved /map and domain_bridge forwards it to
    each follower domain. Each follower still needs its own AMCL initial pose in
    that shared map frame.
    """

    def __init__(self):
        super().__init__('initial_pose_publisher')

        self.declare_parameter('frame_id', 'map')
        self.declare_parameter('x', 0.0)
        self.declare_parameter('y', 0.0)
        self.declare_parameter('yaw', 0.0)
        self.declare_parameter('cov_xy', 0.25)
        self.declare_parameter('cov_yaw', 0.0685)
        self.declare_parameter('period_sec', 1.0)
        self.declare_parameter('repeat_count', 20)

        self.frame_id = str(self.get_parameter('frame_id').value)
        self.x = float(self.get_parameter('x').value)
        self.y = float(self.get_parameter('y').value)
        self.yaw = float(self.get_parameter('yaw').value)
        self.cov_xy = float(self.get_parameter('cov_xy').value)
        self.cov_yaw = float(self.get_parameter('cov_yaw').value)
        self.repeat_count = int(self.get_parameter('repeat_count').value)
        period_sec = float(self.get_parameter('period_sec').value)

        self.pub = self.create_publisher(PoseWithCovarianceStamped, '/initialpose', 10)
        self.sent = 0
        self.timer = self.create_timer(period_sec, self.on_timer)

        self.get_logger().info(
            f'INITIAL_POSE_PUBLISHER_READY | frame={self.frame_id} '
            f'pose=({self.x:.3f},{self.y:.3f},yaw={self.yaw:.3f}) '
            f'repeat_count={self.repeat_count} period={period_sec:.2f}s'
        )

    def build_msg(self) -> PoseWithCovarianceStamped:
        msg = PoseWithCovarianceStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.frame_id
        msg.pose.pose.position.x = self.x
        msg.pose.pose.position.y = self.y
        msg.pose.pose.position.z = 0.0
        qx, qy, qz, qw = yaw_to_quat(self.yaw)
        msg.pose.pose.orientation.x = qx
        msg.pose.pose.orientation.y = qy
        msg.pose.pose.orientation.z = qz
        msg.pose.pose.orientation.w = qw

        cov = [0.0] * 36
        cov[0] = self.cov_xy
        cov[7] = self.cov_xy
        cov[35] = self.cov_yaw
        msg.pose.covariance = cov
        return msg

    def on_timer(self):
        if self.repeat_count >= 0 and self.sent >= self.repeat_count:
            self.get_logger().info('INITIAL_POSE_PUBLISH_DONE')
            self.timer.cancel()
            return

        msg = self.build_msg()
        self.pub.publish(msg)
        self.sent += 1
        self.get_logger().info(
            f'INITIAL_POSE_PUBLISHED | {self.sent}/{self.repeat_count} '
            f'pose=({self.x:.3f},{self.y:.3f},yaw={self.yaw:.3f})'
        )


def main():
    rclpy.init()
    node = InitialPosePublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    try:
        if rclpy.ok():
            rclpy.shutdown()
    except Exception:
        pass


if __name__ == '__main__':
    main()
