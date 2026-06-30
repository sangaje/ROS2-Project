#!/usr/bin/env python3
"""Follower-side Nav2 goal proxy.

The node receives /fleet/<robot>/goal_pose and calls that robot's Nav2
NavigateToPose action. It works in a single ROS domain and in split-domain
setups where domain_bridge forwards only the lightweight fleet topics.
"""

import math

import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node

from geometry_msgs.msg import PoseStamped
from nav2_msgs.action import NavigateToPose
from std_msgs.msg import Bool, String


class RobotGoalProxy(Node):
    def __init__(self):
        super().__init__('robot_goal_proxy')

        self.declare_parameter('robot_name', 'robot1')
        self.declare_parameter('goal_topic', '')
        self.declare_parameter('hold_topic', '')
        self.declare_parameter('cancel_topic', '')
        self.declare_parameter('status_topic', '')
        self.declare_parameter('navigate_action', '')
        self.declare_parameter('cancel_previous_goal', True)
        self.declare_parameter('ignore_duplicate_goals', True)
        self.declare_parameter('same_goal_xy_tolerance_m', 0.05)
        self.declare_parameter('same_goal_yaw_tolerance_rad', 0.08)
        self.declare_parameter('min_resend_period_sec', 1.0)
        self.declare_parameter('wait_for_server_sec', 10.0)

        self.robot_name = str(self.get_parameter('robot_name').value).strip().strip('/') or 'robot1'
        self.goal_topic = str(self.get_parameter('goal_topic').value).strip() or f'/fleet/{self.robot_name}/goal_pose'
        self.hold_topic = str(self.get_parameter('hold_topic').value).strip() or f'/fleet/{self.robot_name}/hold'
        self.cancel_topic = str(self.get_parameter('cancel_topic').value).strip() or f'/fleet/{self.robot_name}/cancel'
        self.status_topic = str(self.get_parameter('status_topic').value).strip() or f'/fleet/{self.robot_name}/status'
        self.navigate_action = str(self.get_parameter('navigate_action').value).strip() or f'/{self.robot_name}/navigate_to_pose'
        self.cancel_previous_goal = bool(self.get_parameter('cancel_previous_goal').value)
        self.ignore_duplicate_goals = bool(self.get_parameter('ignore_duplicate_goals').value)
        self.same_goal_xy_tolerance_m = float(self.get_parameter('same_goal_xy_tolerance_m').value)
        self.same_goal_yaw_tolerance_rad = float(self.get_parameter('same_goal_yaw_tolerance_rad').value)
        self.min_resend_period_sec = float(self.get_parameter('min_resend_period_sec').value)
        self.wait_for_server_sec = float(self.get_parameter('wait_for_server_sec').value)

        self.nav_client = ActionClient(self, NavigateToPose, self.navigate_action)
        self.status_pub = self.create_publisher(String, self.status_topic, 10)

        self.create_subscription(PoseStamped, self.goal_topic, self.on_goal_pose, 10)
        self.create_subscription(Bool, self.hold_topic, self.on_hold, 10)
        self.create_subscription(Bool, self.cancel_topic, self.on_cancel, 10)

        self.current_goal_handle = None
        self.is_holding = False
        self.goal_seq = 0
        self.last_sent_pose = None
        self.last_sent_time_sec = -1.0

        self._publish_status('BOOTING')
        self.get_logger().info(
            'ROBOT_GOAL_PROXY_READY | '
            f'robot={self.robot_name} goal_topic={self.goal_topic} '
            f'hold_topic={self.hold_topic} cancel_topic={self.cancel_topic} '
            f'status_topic={self.status_topic} action={self.navigate_action} '
            f'ignore_duplicate_goals={self.ignore_duplicate_goals}'
        )

        self.create_timer(1.0, self._wait_for_nav2_once)
        self._nav2_ready_reported = False

    def _publish_status(self, text: str):
        msg = String()
        msg.data = text
        self.status_pub.publish(msg)

    def _wait_for_nav2_once(self):
        if self._nav2_ready_reported:
            return
        if self.nav_client.server_is_ready():
            self._nav2_ready_reported = True
            self._publish_status('IDLE_NAV2_READY')
            self.get_logger().info(f'NAV2_ACTION_READY | action={self.navigate_action}')

    def on_hold(self, msg: Bool):
        self.is_holding = bool(msg.data)
        if self.is_holding:
            self.get_logger().warn('HOLD_RECEIVED | cancel current goal')
            self._publish_status('HOLD')
            self._cancel_current_goal()
        else:
            self.get_logger().info('HOLD_RELEASED')
            self._publish_status('IDLE_HOLD_RELEASED')

    def on_cancel(self, msg: Bool):
        if not bool(msg.data):
            return
        self.get_logger().warn('CANCEL_RECEIVED | cancel current goal')
        self._publish_status('CANCEL_REQUESTED')
        self._cancel_current_goal()

    def _cancel_current_goal(self):
        if self.current_goal_handle is None:
            self.get_logger().info('CANCEL_SKIPPED | no active goal handle')
            return
        try:
            cancel_future = self.current_goal_handle.cancel_goal_async()
            cancel_future.add_done_callback(self._on_cancel_done)
        except Exception as exc:
            self.get_logger().error(f'CANCEL_FAILED | error={exc}')

    def _on_cancel_done(self, fut):
        try:
            result = fut.result()
            self.get_logger().warn(f'CANCEL_DONE | goals_canceling={len(result.goals_canceling)}')
            self._publish_status('CANCELED')
        except Exception as exc:
            self.get_logger().error(f'CANCEL_RESULT_FAILED | error={exc}')

    @staticmethod
    def _yaw_from_pose(pose: PoseStamped) -> float:
        q = pose.pose.orientation
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        return math.atan2(siny_cosp, cosy_cosp)

    def _is_duplicate_goal(self, pose: PoseStamped) -> bool:
        if self.last_sent_pose is None:
            return False

        now_sec = self.get_clock().now().nanoseconds * 1e-9
        if self.min_resend_period_sec > 0.0 and now_sec - self.last_sent_time_sec >= self.min_resend_period_sec:
            return False

        dx = pose.pose.position.x - self.last_sent_pose.pose.position.x
        dy = pose.pose.position.y - self.last_sent_pose.pose.position.y
        dist = math.hypot(dx, dy)
        yaw = self._yaw_from_pose(pose)
        last_yaw = self._yaw_from_pose(self.last_sent_pose)
        dyaw = abs(math.atan2(math.sin(yaw - last_yaw), math.cos(yaw - last_yaw)))
        return dist <= self.same_goal_xy_tolerance_m and dyaw <= self.same_goal_yaw_tolerance_rad

    def on_goal_pose(self, pose: PoseStamped):
        if self.is_holding:
            self.get_logger().warn('GOAL_IGNORED_DURING_HOLD')
            self._publish_status('GOAL_IGNORED_HOLD')
            return

        self.goal_seq += 1
        seq = self.goal_seq

        if not self.nav_client.server_is_ready():
            self.get_logger().info(f'WAIT_NAV2_ACTION | action={self.navigate_action}')
            ok = self.nav_client.wait_for_server(timeout_sec=self.wait_for_server_sec)
            if not ok:
                self.get_logger().error(f'NAV2_ACTION_UNAVAILABLE | action={self.navigate_action}')
                self._publish_status('ERROR_NAV2_ACTION_UNAVAILABLE')
                return

        if self.ignore_duplicate_goals and self._is_duplicate_goal(pose):
            self.get_logger().info('DUPLICATE_GOAL_IGNORED')
            return

        if self.cancel_previous_goal and self.current_goal_handle is not None:
            self.get_logger().warn('NEW_GOAL_CANCEL_PREVIOUS')
            self._cancel_current_goal()

        goal = NavigateToPose.Goal()

        # Re-stamp the incoming fleet goal inside the follower robot domain.
        # The fleet master may use wall time while Gazebo/Nav2 uses sim time.
        # Nav2 should receive a goal stamped with this node's local clock.
        local_pose = PoseStamped()
        local_pose.header.frame_id = pose.header.frame_id or 'map'
        local_pose.header.stamp = self.get_clock().now().to_msg()
        local_pose.pose = pose.pose

        goal.pose = local_pose
        goal.behavior_tree = ''
        self.last_sent_pose = local_pose
        self.last_sent_time_sec = self.get_clock().now().nanoseconds * 1e-9

        self.get_logger().info(
            f'SEND_NAV2_GOAL | seq={seq} robot={self.robot_name} '
            f'goal=({pose.pose.position.x:.3f},{pose.pose.position.y:.3f}) '
            f'frame={pose.header.frame_id}'
        )
        self._publish_status(f'SENDING_GOAL_{seq}')

        send_future = self.nav_client.send_goal_async(goal, feedback_callback=lambda fb: self._feedback_cb(seq, fb))
        send_future.add_done_callback(lambda fut: self._goal_response_cb(seq, fut))

    def _goal_response_cb(self, seq: int, fut):
        try:
            goal_handle = fut.result()
        except Exception as exc:
            self.get_logger().error(f'GOAL_RESPONSE_EXCEPTION | seq={seq} error={exc}')
            self._publish_status(f'ERROR_GOAL_RESPONSE_{seq}')
            return

        if goal_handle is None:
            self.get_logger().error(f'GOAL_HANDLE_NONE | seq={seq}')
            self._publish_status(f'ERROR_GOAL_HANDLE_NONE_{seq}')
            return

        if not goal_handle.accepted:
            self.get_logger().error(f'GOAL_REJECTED | seq={seq}')
            self._publish_status(f'GOAL_REJECTED_{seq}')
            return

        self.current_goal_handle = goal_handle
        self.get_logger().info(f'GOAL_ACCEPTED | seq={seq}')
        self._publish_status(f'MOVING_{seq}')

        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(lambda rfut: self._result_cb(seq, rfut))

    def _feedback_cb(self, seq: int, feedback_msg):
        # Keep feedback logging sparse. Nav2 publishes frequent feedback.
        pass

    def _result_cb(self, seq: int, fut):
        try:
            result = fut.result()
            status = int(result.status)
            self.get_logger().info(f'GOAL_RESULT | seq={seq} status={status}')
            if status == 4:  # STATUS_SUCCEEDED in action_msgs/GoalStatus
                self._publish_status(f'SUCCEEDED_{seq}')
            else:
                self._publish_status(f'FINISHED_{seq}_STATUS_{status}')
        except Exception as exc:
            self.get_logger().error(f'GOAL_RESULT_EXCEPTION | seq={seq} error={exc}')
            self._publish_status(f'ERROR_RESULT_{seq}')
        finally:
            self.current_goal_handle = None


def main():
    rclpy.init()
    node = RobotGoalProxy()
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
