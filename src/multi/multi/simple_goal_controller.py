#!/usr/bin/env python3
"""
A small point-to-point controller for Gazebo multi TurtleBot3 tests.

The Gazebo DiffDrive odometry pose may start from 0,0 even when the model is
spawned at a non-zero world pose.  This node therefore calibrates the first
odom sample to the spawn pose supplied by the launch file and performs all goal
logic in the RViz /map frame.
"""

import math
from typing import Optional, Tuple

from geometry_msgs.msg import PointStamped, PoseStamped, Twist
from nav_msgs.msg import Odometry
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from visualization_msgs.msg import Marker, MarkerArray

Pose2D = Tuple[float, float, float]


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def normalize_angle(angle: float) -> float:
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle


def yaw_from_quaternion(x: float, y: float, z: float, w: float) -> float:
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


def relative_pose(reference: Pose2D, current: Pose2D) -> Pose2D:
    """Return current pose expressed relative to reference pose."""
    rx, ry, ryaw = reference
    cx, cy, cyaw = current
    dx = cx - rx
    dy = cy - ry
    c = math.cos(ryaw)
    s = math.sin(ryaw)
    rel_x = c * dx + s * dy
    rel_y = -s * dx + c * dy
    rel_yaw = normalize_angle(cyaw - ryaw)
    return rel_x, rel_y, rel_yaw


def compose_pose(origin: Pose2D, delta: Pose2D) -> Pose2D:
    """Return a map pose by composing an origin pose with a relative delta."""
    ox, oy, oyaw = origin
    dx, dy, dyaw = delta
    c = math.cos(oyaw)
    s = math.sin(oyaw)
    x = ox + c * dx - s * dy
    y = oy + s * dx + c * dy
    yaw = normalize_angle(oyaw + dyaw)
    return x, y, yaw


class SimpleGoalController(Node):
    """Drive one robot toward normal or rescue point goals."""

    def __init__(self) -> None:
        super().__init__('simple_goal_controller')
        self.declare_parameter('robot_name', 'burger1')
        self.declare_parameter('odom_topic', '/burger1/odom')
        self.declare_parameter('cmd_vel_topic', '/burger1/cmd_vel')
        self.declare_parameter('goal_topic', '/burger1/goal_point')
        self.declare_parameter('initial_x', 0.0)
        self.declare_parameter('initial_y', 0.0)
        self.declare_parameter('initial_yaw', 0.0)
        self.declare_parameter('goal_tolerance', 0.12)
        self.declare_parameter('max_linear_speed', 0.22)
        self.declare_parameter('max_angular_speed', 1.4)
        self.declare_parameter('linear_gain', 0.8)
        self.declare_parameter('angular_gain', 2.5)
        self.declare_parameter('rescue_goal_topic', '')
        # RViz flickering was caused by multiple controller nodes publishing to
        # the same MarkerArray topic.  By default, only auto_patrol_rescue
        # publishes robot markers centrally.
        self.declare_parameter('publish_visual_markers', False)

        self.robot_name = str(self.get_parameter('robot_name').value)
        odom_topic = str(self.get_parameter('odom_topic').value)
        cmd_vel_topic = str(self.get_parameter('cmd_vel_topic').value)
        goal_topic = str(self.get_parameter('goal_topic').value)
        rescue_goal_topic = str(self.get_parameter('rescue_goal_topic').value)
        if not rescue_goal_topic:
            rescue_goal_topic = f'/{self.robot_name}/rescue_goal'

        self.initial_pose: Pose2D = (
            float(self.get_parameter('initial_x').value),
            float(self.get_parameter('initial_y').value),
            float(self.get_parameter('initial_yaw').value),
        )
        self.goal_tolerance = float(self.get_parameter('goal_tolerance').value)
        self.max_linear_speed = float(self.get_parameter('max_linear_speed').value)
        self.max_angular_speed = float(self.get_parameter('max_angular_speed').value)
        self.linear_gain = float(self.get_parameter('linear_gain').value)
        self.angular_gain = float(self.get_parameter('angular_gain').value)
        self.publish_visual_markers = bool(
            self.get_parameter('publish_visual_markers').value
        )

        self.odom_zero_pose: Optional[Pose2D] = None
        self.map_pose: Optional[Pose2D] = None
        self.goal: Optional[Tuple[float, float]] = None
        self.was_moving = False
        self.signal_lost = False
        self.rescue_mode = False
        self.waiting_for_rescue = False

        self.cmd_pub = self.create_publisher(Twist, cmd_vel_topic, 10)
        self.map_pose_pub = self.create_publisher(
            PoseStamped,
            f'/{self.robot_name}/map_pose',
            20,
        )
        self.marker_pub = None
        if self.publish_visual_markers:
            self.marker_pub = self.create_publisher(
                MarkerArray,
                '/multi/robot_markers',
                10,
            )
        self.failure_report_pub = self.create_publisher(
            String,
            '/multi/robot_failure_report',
            10,
        )
        self.create_subscription(Odometry, odom_topic, self.on_odom, 20)
        self.create_subscription(PointStamped, goal_topic, self.on_goal, 10)
        self.create_subscription(PointStamped, rescue_goal_topic, self.on_rescue_goal, 10)
        self.create_subscription(String, '/multi/fail_robot', self.on_fail_robot, 10)
        self.create_subscription(String, '/multi/recover_robot', self.on_recover_robot, 10)
        self.create_timer(0.05, self.control_loop)
        if self.publish_visual_markers:
            self.create_timer(0.2, self.publish_markers)
        self.create_timer(0.2, self.publish_failure_report_if_needed)

        self.get_logger().info(
            f'{self.robot_name} controller started: odom={odom_topic}, '
            f'cmd_vel={cmd_vel_topic}, goal={goal_topic}, '
            f'rescue_goal={rescue_goal_topic}, '
            f'initial_map_pose=({self.initial_pose[0]:.2f}, '
            f'{self.initial_pose[1]:.2f}, {self.initial_pose[2]:.2f})'
        )

    def on_odom(self, msg: Odometry) -> None:
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        odom_pose: Pose2D = (p.x, p.y, yaw_from_quaternion(q.x, q.y, q.z, q.w))

        if self.odom_zero_pose is None:
            self.odom_zero_pose = odom_pose
            self.get_logger().info(
                f'First odom calibrated to RViz/map start pose. '
                f'first_odom=({odom_pose[0]:.2f}, {odom_pose[1]:.2f}, {odom_pose[2]:.2f})'
            )

        odom_delta = relative_pose(self.odom_zero_pose, odom_pose)
        self.map_pose = compose_pose(self.initial_pose, odom_delta)
        self.publish_map_pose()

    def publish_map_pose(self) -> None:
        if self.map_pose is None:
            return
        x, y, yaw = self.map_pose
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'map'
        msg.pose.position.x = x
        msg.pose.position.y = y
        msg.pose.position.z = 0.0
        msg.pose.orientation.z = math.sin(yaw / 2.0)
        msg.pose.orientation.w = math.cos(yaw / 2.0)
        self.map_pose_pub.publish(msg)

    def on_goal(self, msg: PointStamped) -> None:
        if self.signal_lost:
            self.publish_stop()
            self.get_logger().warn(
                'Ignoring normal goal because this robot is in SIGNAL_LOST '
                'stop mode.'
            )
            return
        if self.rescue_mode:
            self.get_logger().info(
                'Ignoring normal patrol/manual goal because this robot is in '
                'RESCUE mode.'
            )
            return
        self.goal = (msg.point.x, msg.point.y)
        self.was_moving = True
        self.get_logger().info(
            f'New normal goal in map frame: x={msg.point.x:.2f}, '
            f'y={msg.point.y:.2f}'
        )

    def on_rescue_goal(self, msg: PointStamped) -> None:
        if self.signal_lost:
            self.publish_stop()
            self.get_logger().warn(
                'Ignoring rescue goal because this robot itself is SIGNAL_LOST.'
            )
            return
        self.rescue_mode = True
        self.waiting_for_rescue = False
        self.goal = (msg.point.x, msg.point.y)
        self.was_moving = True
        self.get_logger().warn(
            f'High-priority RESCUE goal accepted: x={msg.point.x:.2f}, '
            f'y={msg.point.y:.2f}'
        )

    def on_fail_robot(self, msg: String) -> None:
        target = msg.data.strip()
        if target.lower() in ('reset', 'clear', 'none'):
            self.signal_lost = False
            self.rescue_mode = False
            self.waiting_for_rescue = False
            self.goal = None
            self.publish_stop()
            return
        if target == self.robot_name:
            self.signal_lost = True
            self.rescue_mode = False
            self.waiting_for_rescue = False
            self.goal = None
            self.was_moving = False
            self.publish_stop()
            self.publish_failure_report_once()
            self.get_logger().warn(
                f'{self.robot_name} entered SIGNAL_LOST stop mode. '
                'Publishing zero cmd_vel and reporting failure.'
            )
        else:
            # A different robot failed. Stop following patrol goals immediately and
            # wait for a high-priority rescue_goal from the central manager.
            if not self.signal_lost:
                self.rescue_mode = True
                self.waiting_for_rescue = True
                self.goal = None
                self.was_moving = False
                self.publish_stop()
                self.get_logger().warn(
                    f'{self.robot_name} stopping patrol and waiting for '
                    f'rescue goal because {target} failed.'
                )

    def on_recover_robot(self, msg: String) -> None:
        target = msg.data.strip()
        if target.lower() in ('all', 'reset', 'clear') or target == self.robot_name:
            if self.signal_lost:
                self.get_logger().info(
                    f'{self.robot_name} recovered from SIGNAL_LOST stop mode.'
                )
            self.signal_lost = False
            self.rescue_mode = False
            self.waiting_for_rescue = False
            self.goal = None
            self.was_moving = False
            self.publish_stop()

    def publish_stop(self) -> None:
        self.cmd_pub.publish(Twist())

    def publish_failure_report_once(self) -> None:
        msg = String()
        msg.data = self.robot_name
        self.failure_report_pub.publish(msg)

    def publish_failure_report_if_needed(self) -> None:
        # The manual ros2 topic pub --once command may reach only one subscriber
        # before it exits.  A failed robot therefore republishes its own failure
        # so the central rescue manager can reliably react.
        if self.signal_lost:
            self.publish_failure_report_once()

    def control_loop(self) -> None:
        if self.signal_lost:
            self.publish_stop()
            return
        if self.waiting_for_rescue:
            self.publish_stop()
            return
        if self.map_pose is None or self.goal is None:
            return
        x, y, yaw = self.map_pose
        gx, gy = self.goal
        dx = gx - x
        dy = gy - y
        dist = math.hypot(dx, dy)

        cmd = Twist()
        if dist <= self.goal_tolerance:
            if self.was_moving:
                self.publish_stop()
                mode = 'RESCUE' if self.rescue_mode else 'normal'
                self.get_logger().info(f'{mode} goal reached: x={gx:.2f}, y={gy:.2f}')
            self.was_moving = False
            if not self.rescue_mode:
                self.goal = None
            return

        target_yaw = math.atan2(dy, dx)
        yaw_error = normalize_angle(target_yaw - yaw)

        cmd.angular.z = clamp(
            self.angular_gain * yaw_error,
            -self.max_angular_speed,
            self.max_angular_speed,
        )

        if abs(yaw_error) < 0.45:
            cmd.linear.x = clamp(self.linear_gain * dist, 0.0, self.max_linear_speed)
        else:
            cmd.linear.x = 0.0

        self.cmd_pub.publish(cmd)
        self.was_moving = True

    def publish_markers(self) -> None:
        if self.marker_pub is None or self.map_pose is None:
            return
        x, y, yaw = self.map_pose
        marker_array = MarkerArray()

        color = self.color_for_robot()
        body = Marker()
        body.header.frame_id = 'map'
        body.header.stamp = self.get_clock().now().to_msg()
        body.ns = f'{self.robot_name}_body'
        body.id = 0
        body.type = Marker.ARROW
        body.action = Marker.ADD
        body.pose.position.x = x
        body.pose.position.y = y
        body.pose.position.z = 0.08
        body.pose.orientation.z = math.sin(yaw / 2.0)
        body.pose.orientation.w = math.cos(yaw / 2.0)
        body.scale.x = 0.45
        body.scale.y = 0.12
        body.scale.z = 0.12
        body.color.a = 1.0
        body.color.r, body.color.g, body.color.b = color
        marker_array.markers.append(body)

        text = Marker()
        text.header.frame_id = 'map'
        text.header.stamp = body.header.stamp
        text.ns = f'{self.robot_name}_label'
        text.id = 1
        text.type = Marker.TEXT_VIEW_FACING
        text.action = Marker.ADD
        text.pose.position.x = x
        text.pose.position.y = y
        text.pose.position.z = 0.45
        text.pose.orientation.w = 1.0
        text.scale.z = 0.22
        text.color.a = 1.0
        text.color.r = 1.0
        text.color.g = 1.0
        text.color.b = 1.0
        text.text = self.robot_name
        marker_array.markers.append(text)

        if self.goal is not None:
            goal = Marker()
            goal.header.frame_id = 'map'
            goal.header.stamp = body.header.stamp
            goal.ns = f'{self.robot_name}_active_goal'
            goal.id = 2
            goal.type = Marker.CYLINDER
            goal.action = Marker.ADD
            goal.pose.position.x = self.goal[0]
            goal.pose.position.y = self.goal[1]
            goal.pose.position.z = 0.03
            goal.pose.orientation.w = 1.0
            goal.scale.x = 0.28
            goal.scale.y = 0.28
            goal.scale.z = 0.06
            goal.color.a = 0.8
            goal.color.r, goal.color.g, goal.color.b = color
            marker_array.markers.append(goal)

        self.marker_pub.publish(marker_array)

    def color_for_robot(self) -> Tuple[float, float, float]:
        if self.robot_name == 'burger1':
            return (1.0, 0.1, 0.1)
        return (0.1, 0.3, 1.0)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = SimpleGoalController()
    try:
        rclpy.spin(node)
    finally:
        node.publish_stop()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
