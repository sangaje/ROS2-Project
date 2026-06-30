#!/usr/bin/env python3
"""
Follower-side pose reporter.

Runs inside one robot's ROS_DOMAIN_ID. It reads local TF map->base frame and
publishes /fleet/<robot>/pose for the master domain via domain_bridge.
"""

import rclpy
from rclpy.node import Node
from rclpy.time import Time
from geometry_msgs.msg import PoseStamped
from tf2_ros import Buffer, TransformListener, TransformException


class RobotPoseReporter(Node):
    def __init__(self):
        super().__init__('robot_pose_reporter')

        self.declare_parameter('robot_name', 'robot1')
        self.declare_parameter('map_frame', 'map')
        self.declare_parameter('base_frame', 'base_footprint')
        self.declare_parameter('pose_topic', '')
        self.declare_parameter('publish_period_sec', 0.10)
        self.declare_parameter('log_warn_period_sec', 2.0)

        self.robot_name = str(self.get_parameter('robot_name').value).strip().strip('/') or 'robot1'
        self.map_frame = str(self.get_parameter('map_frame').value).strip() or 'map'
        self.base_frame = str(self.get_parameter('base_frame').value).strip() or 'base_footprint'
        self.pose_topic = str(self.get_parameter('pose_topic').value).strip() or f'/fleet/{self.robot_name}/pose'
        self.period = max(0.02, float(self.get_parameter('publish_period_sec').value))
        self.log_warn_period_sec = max(0.5, float(self.get_parameter('log_warn_period_sec').value))

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.pose_pub = self.create_publisher(PoseStamped, self.pose_topic, 10)
        self.create_timer(self.period, self.on_timer)

        self.last_warn_time = 0.0
        self.get_logger().info(
            'ROBOT_POSE_REPORTER_READY | '
            f'robot={self.robot_name} {self.map_frame}->{self.base_frame} topic={self.pose_topic} period={self.period:.3f}s'
        )

    def on_timer(self):
        try:
            tf = self.tf_buffer.lookup_transform(self.map_frame, self.base_frame, Time())
        except TransformException as exc:
            now = self.get_clock().now().nanoseconds * 1e-9
            if now - self.last_warn_time >= self.log_warn_period_sec:
                self.get_logger().warn(f'TF_LOOKUP_FAILED | {self.map_frame}->{self.base_frame} | {exc}')
                self.last_warn_time = now
            return

        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.map_frame
        msg.pose.position.x = tf.transform.translation.x
        msg.pose.position.y = tf.transform.translation.y
        msg.pose.position.z = tf.transform.translation.z
        msg.pose.orientation = tf.transform.rotation
        self.pose_pub.publish(msg)


def main():
    rclpy.init()
    node = RobotPoseReporter()
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
