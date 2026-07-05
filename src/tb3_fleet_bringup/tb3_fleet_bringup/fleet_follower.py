from __future__ import annotations

import math
from typing import Optional, Tuple

import rclpy
from geometry_msgs.msg import PoseStamped
from nav2_msgs.action import NavigateToPose
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)
from std_msgs.msg import Bool, String


Point2 = Tuple[float, float]


def yaw_from_quaternion(quaternion) -> float:
    siny_cosp = 2.0 * (
        quaternion.w * quaternion.z + quaternion.x * quaternion.y
    )
    cosy_cosp = 1.0 - 2.0 * (
        quaternion.y * quaternion.y + quaternion.z * quaternion.z
    )
    return math.atan2(siny_cosp, cosy_cosp)


def quaternion_from_yaw(yaw: float):
    half = 0.5 * yaw
    return 0.0, 0.0, math.sin(half), math.cos(half)


class FleetFollower(Node):
    """Send Nav2 goals behind the leader and obey central PAUSE/RESUME commands."""

    def __init__(self) -> None:
        super().__init__('fleet_follower')

        self.declare_parameter('leader_pose_topic', '/leader_pose')
        self.declare_parameter('follower_pose_topic', '/burger_pose')
        self.declare_parameter('navigate_action', '/navigate_to_pose')
        self.declare_parameter('follow_command_topic', '/fleet/follow_command')
        self.declare_parameter('follow_status_topic', '/fleet/follow_enabled')
        self.declare_parameter('follow_distance', 0.70)
        self.declare_parameter('goal_period_sec', 1.0)
        self.declare_parameter('goal_update_distance', 0.20)
        self.declare_parameter('goal_refresh_sec', 3.0)
        self.declare_parameter('action_wait_timeout_sec', 1.0)
        self.declare_parameter('cancel_previous_goal', False)
        self.declare_parameter('start_following', True)

        parameter = self.get_parameter
        self.leader_pose_topic = self._absolute(
            str(parameter('leader_pose_topic').value)
        )
        self.follower_pose_topic = self._absolute(
            str(parameter('follower_pose_topic').value)
        )
        self.navigate_action = self._absolute(
            str(parameter('navigate_action').value)
        )
        self.follow_command_topic = self._absolute(
            str(parameter('follow_command_topic').value)
        )
        self.follow_status_topic = self._absolute(
            str(parameter('follow_status_topic').value)
        )
        self.follow_distance = max(0.25, float(parameter('follow_distance').value))
        self.goal_period = max(0.2, float(parameter('goal_period_sec').value))
        self.goal_update_distance = max(
            0.05, float(parameter('goal_update_distance').value)
        )
        self.goal_refresh = max(1.0, float(parameter('goal_refresh_sec').value))
        self.action_wait_timeout = max(
            0.0, float(parameter('action_wait_timeout_sec').value)
        )
        self.cancel_previous_goal = bool(
            parameter('cancel_previous_goal').value
        )
        self.follow_enabled = bool(parameter('start_following').value)

        self.leader_pose: Optional[PoseStamped] = None
        self.follower_pose: Optional[PoseStamped] = None
        self.active_goal_handle = None
        self.last_goal_xy: Optional[Point2] = None
        self.last_goal_time = -1.0e9
        self.wait_log_time = -1.0e9
        self.goal_count = 0

        coordination_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.create_subscription(
            PoseStamped,
            self.leader_pose_topic,
            self._leader_pose_callback,
            20,
        )
        self.create_subscription(
            PoseStamped,
            self.follower_pose_topic,
            self._follower_pose_callback,
            20,
        )
        self.create_subscription(
            String,
            self.follow_command_topic,
            self._command_callback,
            coordination_qos,
        )
        self.status_publisher = self.create_publisher(
            Bool,
            self.follow_status_topic,
            coordination_qos,
        )
        self.navigation_client = ActionClient(
            self,
            NavigateToPose,
            self.navigate_action,
        )
        self.create_timer(self.goal_period, self._tick)
        self._publish_status()

        self.get_logger().info(
            'FLEET_FOLLOWER_READY | '
            f'leader={self.leader_pose_topic} '
            f'follower={self.follower_pose_topic} '
            f'action={self.navigate_action} '
            f'distance={self.follow_distance:.2f}m '
            f'enabled={self.follow_enabled}'
        )

    @staticmethod
    def _absolute(topic: str) -> str:
        return topic if topic.startswith('/') else '/' + topic

    def _now(self) -> float:
        return self.get_clock().now().nanoseconds * 1.0e-9

    def _leader_pose_callback(self, message: PoseStamped) -> None:
        self.leader_pose = message

    def _follower_pose_callback(self, message: PoseStamped) -> None:
        self.follower_pose = message

    def _publish_status(self) -> None:
        message = Bool()
        message.data = self.follow_enabled
        self.status_publisher.publish(message)

    def _cancel_goal(self, reason: str) -> None:
        handle = self.active_goal_handle
        self.active_goal_handle = None
        if handle is None:
            return
        try:
            handle.cancel_goal_async()
            self.get_logger().info(f'FOLLOW_GOAL_CANCEL_REQUESTED | reason={reason}')
        except Exception as error:
            self.get_logger().warning(
                f'FOLLOW_GOAL_CANCEL_ERROR | reason={reason} error={error}'
            )

    def _set_following(self, enabled: bool, command: str) -> None:
        changed = enabled != self.follow_enabled
        self.follow_enabled = enabled
        self._publish_status()
        if not enabled:
            self._cancel_goal(command)
        self.last_goal_xy = None
        self.last_goal_time = -1.0e9
        if changed:
            state = 'FOLLOWING' if enabled else 'PAUSED'
            self.get_logger().warning(
                f'FOLLOW_STATE | state={state} command={command}'
            )

    def _command_callback(self, message: String) -> None:
        command = message.data.strip().upper()
        if command in ('FOLLOW', 'RESUME', 'START', 'ON', '1', 'TRUE'):
            self._set_following(True, command)
        elif command in ('PAUSE', 'STOP', 'HOLD', 'OFF', '0', 'FALSE'):
            self._set_following(False, command)
        elif command == 'TOGGLE':
            self._set_following(not self.follow_enabled, command)
        else:
            self.get_logger().warning(
                f'FOLLOW_COMMAND_IGNORED | command={message.data!r}'
            )

    def _target_behind_leader(self) -> PoseStamped:
        assert self.leader_pose is not None
        leader = self.leader_pose.pose
        yaw = yaw_from_quaternion(leader.orientation)
        qx, qy, qz, qw = quaternion_from_yaw(yaw)

        target = PoseStamped()
        target.header.frame_id = 'map'
        target.header.stamp = rclpy.time.Time().to_msg()
        target.pose.position.x = (
            leader.position.x - self.follow_distance * math.cos(yaw)
        )
        target.pose.position.y = (
            leader.position.y - self.follow_distance * math.sin(yaw)
        )
        target.pose.orientation.x = qx
        target.pose.orientation.y = qy
        target.pose.orientation.z = qz
        target.pose.orientation.w = qw
        return target

    def _should_send(self, target: PoseStamped) -> bool:
        current = (target.pose.position.x, target.pose.position.y)
        if self.last_goal_xy is None:
            return True
        moved = math.hypot(
            current[0] - self.last_goal_xy[0],
            current[1] - self.last_goal_xy[1],
        )
        return (
            moved >= self.goal_update_distance
            or self._now() - self.last_goal_time >= self.goal_refresh
        )

    def _log_wait(self, reason: str) -> None:
        now = self._now()
        if now - self.wait_log_time >= 5.0:
            self.get_logger().warning(f'FOLLOW_WAIT | {reason}')
            self.wait_log_time = now

    def _tick(self) -> None:
        if not self.follow_enabled:
            return
        if self.leader_pose is None:
            self._log_wait(f'no {self.leader_pose_topic}')
            return
        if self.follower_pose is None:
            self._log_wait('follower localization is not ready')
            return
        if not self.navigation_client.wait_for_server(
            timeout_sec=self.action_wait_timeout
        ):
            self._log_wait(f'action server is not ready: {self.navigate_action}')
            return

        target = self._target_behind_leader()
        if not self._should_send(target):
            return
        if self.cancel_previous_goal:
            self._cancel_goal('new follow target')

        goal = NavigateToPose.Goal()
        goal.pose = target
        self.last_goal_xy = (
            target.pose.position.x,
            target.pose.position.y,
        )
        self.last_goal_time = self._now()
        self.goal_count += 1
        future = self.navigation_client.send_goal_async(goal)
        future.add_done_callback(self._goal_response_callback)

    def _goal_response_callback(self, future) -> None:
        try:
            handle = future.result()
        except Exception as error:
            self.get_logger().error(f'FOLLOW_GOAL_ERROR | {error}')
            return
        if not handle.accepted:
            self.get_logger().warning('FOLLOW_GOAL_REJECTED')
            return
        if not self.follow_enabled:
            handle.cancel_goal_async()
            return
        self.active_goal_handle = handle
        self.get_logger().info(
            f'FOLLOW_GOAL_ACCEPTED | count={self.goal_count}'
        )


def main() -> None:
    rclpy.init()
    node = FleetFollower()
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
