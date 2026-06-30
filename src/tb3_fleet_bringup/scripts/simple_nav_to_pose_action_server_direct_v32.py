#!/usr/bin/env python3
from __future__ import annotations

import math
import time
from typing import Optional, Tuple

import rclpy
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.node import Node
from builtin_interfaces.msg import Duration
from geometry_msgs.msg import PoseStamped, TwistStamped
from nav_msgs.msg import Odometry
from nav2_msgs.action import NavigateToPose


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def wrap_pi(a: float) -> float:
    while a > math.pi:
        a -= 2.0 * math.pi
    while a < -math.pi:
        a += 2.0 * math.pi
    return a


def yaw_from_quat(q) -> float:
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


class SimpleNavigateToPoseActionServer(Node):
    """A small NavigateToPose-compatible fallback action server.

    This is intentionally not a full global planner. It is a deterministic test
    controller for the fleet/domain-bridge setup when Nav2's bt_navigator action
    server is not created because planner_server remains inactive.

    Input:
      /odom   nav_msgs/Odometry, robot-local odom from Gazebo bridge
    Output:
      /cmd_vel geometry_msgs/TwistStamped, consumed by the existing converter
    Action:
      /navigate_to_pose nav2_msgs/action/NavigateToPose

    Current map pose is estimated as:
      map_x = initial_x + odom_x rotated by initial_yaw
      map_y = initial_y + odom_y rotated by initial_yaw
      map_yaw = initial_yaw + odom_yaw
    """

    def __init__(self) -> None:
        super().__init__('simple_nav_to_pose_action_server')
        self.declare_parameter('action_name', '/navigate_to_pose')
        self.declare_parameter('odom_topic', '/odom')
        self.declare_parameter('cmd_vel_topic', '/cmd_vel')
        self.declare_parameter('robot_name', 'robot')
        self.declare_parameter('initial_x', 0.0)
        self.declare_parameter('initial_y', 0.0)
        self.declare_parameter('initial_yaw', 0.0)
        self.declare_parameter('control_hz', 10.0)
        self.declare_parameter('xy_goal_tolerance', 0.18)
        self.declare_parameter('yaw_goal_tolerance', 0.35)
        self.declare_parameter('max_linear_vel', 0.20)
        self.declare_parameter('max_angular_vel', 0.80)
        self.declare_parameter('k_linear', 0.55)
        self.declare_parameter('k_angular', 1.80)
        self.declare_parameter('heading_only_threshold', 0.75)
        self.declare_parameter('goal_timeout_sec', 120.0)
        self.declare_parameter('odom_timeout_sec', 2.0)
        self.declare_parameter('log_period_sec', 2.0)

        self.action_name = self._abs(str(self.get_parameter('action_name').value))
        self.odom_topic = self._abs(str(self.get_parameter('odom_topic').value))
        self.cmd_vel_topic = self._abs(str(self.get_parameter('cmd_vel_topic').value))
        self.robot_name = str(self.get_parameter('robot_name').value)
        self.initial_x = float(self.get_parameter('initial_x').value)
        self.initial_y = float(self.get_parameter('initial_y').value)
        self.initial_yaw = float(self.get_parameter('initial_yaw').value)
        self.control_hz = float(self.get_parameter('control_hz').value)
        self.xy_tol = float(self.get_parameter('xy_goal_tolerance').value)
        self.yaw_tol = float(self.get_parameter('yaw_goal_tolerance').value)
        self.max_v = float(self.get_parameter('max_linear_vel').value)
        self.max_w = float(self.get_parameter('max_angular_vel').value)
        self.k_v = float(self.get_parameter('k_linear').value)
        self.k_w = float(self.get_parameter('k_angular').value)
        self.heading_only_threshold = float(self.get_parameter('heading_only_threshold').value)
        self.goal_timeout_sec = float(self.get_parameter('goal_timeout_sec').value)
        self.odom_timeout_sec = float(self.get_parameter('odom_timeout_sec').value)
        self.log_period_sec = float(self.get_parameter('log_period_sec').value)

        self._last_odom: Optional[Odometry] = None
        self._last_odom_wall_time: float = 0.0
        self._odom_sub = self.create_subscription(Odometry, self.odom_topic, self._on_odom, 50)
        self._cmd_pub = self.create_publisher(TwistStamped, self.cmd_vel_topic, 10)
        self._server = ActionServer(
            self,
            NavigateToPose,
            self.action_name,
            execute_callback=self._execute,
            goal_callback=self._goal_callback,
            cancel_callback=self._cancel_callback,
        )
        self.get_logger().warn(
            'V32_SIMPLE_NAV_TO_POSE_FALLBACK_READY | '
            f'robot={self.robot_name} action={self.action_name} odom={self.odom_topic} cmd={self.cmd_vel_topic} '
            f'initial=({self.initial_x:.2f},{self.initial_y:.2f},{self.initial_yaw:.2f}) '
            'NOTE=this is a direct controller fallback, not a global planner'
        )

    @staticmethod
    def _abs(topic: str) -> str:
        return topic if topic.startswith('/') else '/' + topic

    def _on_odom(self, msg: Odometry) -> None:
        self._last_odom = msg
        self._last_odom_wall_time = time.monotonic()

    def _goal_callback(self, goal_request) -> GoalResponse:
        p = goal_request.pose.pose.position
        self.get_logger().info(f'V32_SIMPLE_NAV_GOAL_ACCEPT | robot={self.robot_name} target=({p.x:.2f},{p.y:.2f})')
        return GoalResponse.ACCEPT

    def _cancel_callback(self, goal_handle) -> CancelResponse:
        self.get_logger().warn(f'V32_SIMPLE_NAV_CANCEL | robot={self.robot_name}')
        self._publish_zero()
        return CancelResponse.ACCEPT

    def _current_pose_map(self) -> Optional[Tuple[float, float, float]]:
        if self._last_odom is None:
            return None
        if time.monotonic() - self._last_odom_wall_time > self.odom_timeout_sec:
            return None
        p = self._last_odom.pose.pose.position
        yaw_odom = yaw_from_quat(self._last_odom.pose.pose.orientation)
        c = math.cos(self.initial_yaw)
        s = math.sin(self.initial_yaw)
        x = self.initial_x + c * p.x - s * p.y
        y = self.initial_y + s * p.x + c * p.y
        yaw = wrap_pi(self.initial_yaw + yaw_odom)
        return x, y, yaw

    def _publish_cmd(self, v: float, w: float) -> None:
        msg = TwistStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_footprint'
        msg.twist.linear.x = float(v)
        msg.twist.angular.z = float(w)
        self._cmd_pub.publish(msg)

    def _publish_zero(self) -> None:
        self._publish_cmd(0.0, 0.0)

    def _make_result(self, error_code: int = 0, error_msg: str = ''):
        result = NavigateToPose.Result()
        if hasattr(result, 'error_code'):
            result.error_code = int(error_code)
        if hasattr(result, 'error_msg'):
            result.error_msg = str(error_msg)
        return result

    def _make_feedback(self, goal_pose: PoseStamped, dist: float, start_wall: float):
        fb = NavigateToPose.Feedback()
        cur = self._current_pose_map()
        if hasattr(fb, 'current_pose') and cur is not None:
            x, y, yaw = cur
            fb.current_pose.header.stamp = self.get_clock().now().to_msg()
            fb.current_pose.header.frame_id = 'map'
            fb.current_pose.pose.position.x = x
            fb.current_pose.pose.position.y = y
            fb.current_pose.pose.orientation.z = math.sin(yaw * 0.5)
            fb.current_pose.pose.orientation.w = math.cos(yaw * 0.5)
        if hasattr(fb, 'distance_remaining'):
            fb.distance_remaining = float(dist)
        if hasattr(fb, 'number_of_recoveries'):
            fb.number_of_recoveries = 0
        if hasattr(fb, 'navigation_time'):
            elapsed = max(0.0, time.monotonic() - start_wall)
            fb.navigation_time = Duration(sec=int(elapsed), nanosec=int((elapsed % 1.0) * 1e9))
        return fb

    def _execute(self, goal_handle):
        goal = goal_handle.request.pose
        gx = float(goal.pose.position.x)
        gy = float(goal.pose.position.y)
        gyaw = yaw_from_quat(goal.pose.orientation)
        period = 1.0 / max(1.0, self.control_hz)
        start_wall = time.monotonic()
        last_log = 0.0

        self.get_logger().info(f'V32_SIMPLE_NAV_EXECUTE | robot={self.robot_name} target=({gx:.2f},{gy:.2f},{gyaw:.2f})')

        while rclpy.ok():
            if goal_handle.is_cancel_requested:
                self._publish_zero()
                goal_handle.canceled()
                return self._make_result(1, 'canceled')

            if time.monotonic() - start_wall > self.goal_timeout_sec:
                self._publish_zero()
                goal_handle.abort()
                self.get_logger().error(f'V32_SIMPLE_NAV_TIMEOUT | robot={self.robot_name}')
                return self._make_result(2, 'timeout')

            cur = self._current_pose_map()
            if cur is None:
                self._publish_zero()
                if time.monotonic() - last_log > self.log_period_sec:
                    self.get_logger().warn(f'V32_SIMPLE_NAV_WAIT_ODOM | robot={self.robot_name} odom={self.odom_topic}')
                    last_log = time.monotonic()
                time.sleep(period)
                continue

            x, y, yaw = cur
            dx = gx - x
            dy = gy - y
            dist = math.hypot(dx, dy)
            target_heading = math.atan2(dy, dx)
            heading_err = wrap_pi(target_heading - yaw)
            final_yaw_err = wrap_pi(gyaw - yaw)

            goal_handle.publish_feedback(self._make_feedback(goal, dist, start_wall))

            if dist <= self.xy_tol:
                # Align roughly to requested final yaw, but keep this permissive for tests.
                if abs(final_yaw_err) <= self.yaw_tol:
                    self._publish_zero()
                    goal_handle.succeed()
                    self.get_logger().info(f'V32_SIMPLE_NAV_SUCCEEDED | robot={self.robot_name} dist={dist:.3f}')
                    return self._make_result(0, '')
                v = 0.0
                w = clamp(self.k_w * final_yaw_err, -self.max_w, self.max_w)
            else:
                if abs(heading_err) > self.heading_only_threshold:
                    v = 0.0
                else:
                    v = clamp(self.k_v * dist, 0.04, self.max_v)
                w = clamp(self.k_w * heading_err, -self.max_w, self.max_w)

            self._publish_cmd(v, w)
            if time.monotonic() - last_log > self.log_period_sec:
                self.get_logger().info(
                    f'V32_SIMPLE_NAV_CONTROL | robot={self.robot_name} pose=({x:.2f},{y:.2f},{yaw:.2f}) '
                    f'target=({gx:.2f},{gy:.2f}) dist={dist:.2f} herr={math.degrees(heading_err):.1f} cmd=({v:.2f},{w:.2f})'
                )
                last_log = time.monotonic()
            time.sleep(period)

        self._publish_zero()
        goal_handle.abort()
        return self._make_result(3, 'rclpy shutdown')


def main() -> None:
    rclpy.init()
    node = SimpleNavigateToPoseActionServer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node._publish_zero()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
