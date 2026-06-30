#!/usr/bin/env python3

from __future__ import annotations

import math
from typing import Optional

import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from nav2_msgs.action import NavigateToPose
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped, Quaternion


def _yaw_from_quat(q: Quaternion) -> float:
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


def _quat_from_yaw(yaw: float) -> Quaternion:
    q = Quaternion()
    q.z = math.sin(yaw * 0.5)
    q.w = math.cos(yaw * 0.5)
    return q


class WaffleBurgerNav2Follower(Node):
    """Send Nav2 goals to burger so it follows behind waffle.

    Waffle is commanded by /waffle/navigate_to_pose.
    Burger is commanded by this node through /burger/navigate_to_pose.
    """

    def __init__(self) -> None:
        super().__init__('waffle_burger_nav2_follower')
        self.declare_parameter('leader_name', 'waffle')
        self.declare_parameter('follower_name', 'burger')
        self.declare_parameter('follow_distance', 1.05)
        self.declare_parameter('min_distance', 0.55)
        self.declare_parameter('goal_period_sec', 1.5)
        self.declare_parameter('goal_update_distance', 0.25)
        self.declare_parameter('goal_update_yaw_deg', 12.0)
        self.declare_parameter('wait_for_action_sec', 60.0)
        self.declare_parameter('log_period_sec', 2.0)

        self.leader = str(self.get_parameter('leader_name').value)
        self.follower = str(self.get_parameter('follower_name').value)
        self.follow_distance = float(self.get_parameter('follow_distance').value)
        self.min_distance = float(self.get_parameter('min_distance').value)
        self.goal_period_sec = float(self.get_parameter('goal_period_sec').value)
        self.goal_update_distance = float(self.get_parameter('goal_update_distance').value)
        self.goal_update_yaw = math.radians(float(self.get_parameter('goal_update_yaw_deg').value))
        self.wait_for_action_sec = float(self.get_parameter('wait_for_action_sec').value)
        self.log_period_sec = float(self.get_parameter('log_period_sec').value)

        self.leader_odom: Optional[Odometry] = None
        self.follower_odom: Optional[Odometry] = None
        self.last_goal_xy: Optional[tuple] = None
        self.last_goal_yaw: Optional[float] = None
        self.current_goal_handle = None
        self.action_ready = False

        self.create_subscription(Odometry, f'/{self.leader}/odom_nav', self._on_leader_odom, 20)
        self.create_subscription(Odometry, f'/{self.follower}/odom_nav', self._on_follower_odom, 20)
        self.client = ActionClient(self, NavigateToPose, f'/{self.follower}/navigate_to_pose')

        self.create_timer(self.goal_period_sec, self._tick)
        self.create_timer(self.log_period_sec, self._log_tick)

        self.get_logger().info(
            'V21_NAV2_FOLLOWER_READY | '
            f'leader=/{self.leader}/navigate_to_pose external | follower=/{self.follower}/navigate_to_pose automatic | '
            f'follow_distance={self.follow_distance:.2f} min_distance={self.min_distance:.2f}'
        )

    def _on_leader_odom(self, msg: Odometry) -> None:
        self.leader_odom = msg

    def _on_follower_odom(self, msg: Odometry) -> None:
        self.follower_odom = msg

    def _distance_between_robots(self) -> Optional[float]:
        if self.leader_odom is None or self.follower_odom is None:
            return None
        lp = self.leader_odom.pose.pose.position
        fp = self.follower_odom.pose.pose.position
        return math.hypot(lp.x - fp.x, lp.y - fp.y)

    def _cancel_current_goal(self, reason: str) -> None:
        if self.current_goal_handle is not None:
            try:
                self.current_goal_handle.cancel_goal_async()
            except Exception as exc:
                self.get_logger().warn(f'NAV2_FOLLOW_CANCEL_FAILED | reason={reason} err={exc}')
            self.current_goal_handle = None
            self.get_logger().warn(f'NAV2_FOLLOW_CANCEL | reason={reason}')

    def _tick(self) -> None:
        if self.leader_odom is None or self.follower_odom is None:
            self.get_logger().warn('NAV2_FOLLOW_WAIT_ODOM | missing leader or follower odom_nav')
            return

        if not self.action_ready:
            self.action_ready = self.client.wait_for_server(timeout_sec=0.01)
            if not self.action_ready:
                self.get_logger().warn(f'NAV2_FOLLOW_WAIT_ACTION | action=/{self.follower}/navigate_to_pose')
                return

        dist = self._distance_between_robots()
        if dist is not None and dist < self.min_distance:
            self._cancel_current_goal(f'too_close dist={dist:.2f}')
            return

        lp = self.leader_odom.pose.pose.position
        lq = self.leader_odom.pose.pose.orientation
        yaw = _yaw_from_quat(lq)
        target_x = lp.x - self.follow_distance * math.cos(yaw)
        target_y = lp.y - self.follow_distance * math.sin(yaw)
        target_yaw = yaw

        if self.last_goal_xy is not None:
            moved = math.hypot(target_x - self.last_goal_xy[0], target_y - self.last_goal_xy[1])
            dyaw = abs(math.atan2(math.sin(target_yaw - self.last_goal_yaw), math.cos(target_yaw - self.last_goal_yaw))) if self.last_goal_yaw is not None else 999.0
            if moved < self.goal_update_distance and dyaw < self.goal_update_yaw:
                return

        goal = NavigateToPose.Goal()
        goal.pose = PoseStamped()
        goal.pose.header.stamp = self.get_clock().now().to_msg()
        goal.pose.header.frame_id = 'map'
        goal.pose.pose.position.x = target_x
        goal.pose.pose.position.y = target_y
        goal.pose.pose.orientation = _quat_from_yaw(target_yaw)

        future = self.client.send_goal_async(goal)
        future.add_done_callback(self._on_goal_response)
        self.last_goal_xy = (target_x, target_y)
        self.last_goal_yaw = target_yaw
        self.get_logger().info(
            f'NAV2_FOLLOW_GOAL_SENT | follower={self.follower} target=({target_x:.2f},{target_y:.2f}) yaw={target_yaw:.2f} dist={dist if dist is not None else -1:.2f}'
        )

    def _on_goal_response(self, future) -> None:
        try:
            goal_handle = future.result()
        except Exception as exc:
            self.get_logger().error(f'NAV2_FOLLOW_GOAL_RESPONSE_ERROR | {exc}')
            return
        if not goal_handle.accepted:
            self.get_logger().warn('NAV2_FOLLOW_GOAL_REJECTED')
            return
        self.current_goal_handle = goal_handle
        self.get_logger().info('NAV2_FOLLOW_GOAL_ACCEPTED')

    def _log_tick(self) -> None:
        dist = self._distance_between_robots()
        self.get_logger().info(
            f'NAV2_FOLLOW_STATUS | leader_odom={self.leader_odom is not None} follower_odom={self.follower_odom is not None} '
            f'action_ready={self.action_ready} dist={dist if dist is not None else -1:.2f}'
        )


def main() -> None:
    rclpy.init()
    node = WaffleBurgerNav2Follower()
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
