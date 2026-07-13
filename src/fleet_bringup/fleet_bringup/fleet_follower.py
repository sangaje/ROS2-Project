from __future__ import annotations

from copy import deepcopy
from functools import partial
import math
from typing import Optional, Tuple

import rclpy
from action_msgs.msg import GoalStatus
from geometry_msgs.msg import PoseStamped
from nav2_msgs.action import NavigateToPose
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
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
    """Minimal Nav2 follower: keep this robot behind /leader_pose."""

    def __init__(self) -> None:
        super().__init__('fleet_follower')

        self.declare_parameter('leader_pose_topic', '/leader_pose')
        self.declare_parameter('follower_pose_topic', '/burger_pose')
        self.declare_parameter('navigate_action', '/navigate_to_pose')
        self.declare_parameter('follow_command_topic', '/fleet/follow_command')
        self.declare_parameter('follow_status_topic', '/fleet/follow_enabled')
        self.declare_parameter('localization_ready_topic', '/localization_ready')
        self.declare_parameter('require_localization_ready', False)
        self.declare_parameter('follow_distance', 0.50)
        self.declare_parameter('stop_distance_m', 0.35)
        self.declare_parameter('goal_period_sec', 0.5)
        self.declare_parameter('goal_update_distance', 0.10)
        self.declare_parameter('pose_timeout_sec', 0.8)
        self.declare_parameter('start_following', True)

        parameter = self.get_parameter
        self.leader_pose_topic = self._absolute(str(parameter('leader_pose_topic').value))
        self.follower_pose_topic = self._absolute(str(parameter('follower_pose_topic').value))
        self.navigate_action = self._absolute(str(parameter('navigate_action').value))
        self.follow_command_topic = self._absolute(str(parameter('follow_command_topic').value))
        self.follow_status_topic = self._absolute(str(parameter('follow_status_topic').value))
        self.localization_ready_topic = self._absolute(
            str(parameter('localization_ready_topic').value)
        )
        self.require_localization_ready = bool(
            parameter('require_localization_ready').value
        )
        self.follow_distance = max(0.1, float(parameter('follow_distance').value))
        self.stop_distance = max(0.05, float(parameter('stop_distance_m').value))
        self.goal_period = max(0.2, float(parameter('goal_period_sec').value))
        self.goal_update_distance = max(
            0.03, float(parameter('goal_update_distance').value)
        )
        self.pose_timeout = max(0.1, float(parameter('pose_timeout_sec').value))
        self.follow_enabled = bool(parameter('start_following').value)

        self.leader_pose: Optional[PoseStamped] = None
        self.leader_pose_wall: Optional[float] = None
        self.follower_pose: Optional[PoseStamped] = None
        self.follower_pose_wall: Optional[float] = None
        self.localization_ready = False
        self.goal_count = 0
        self.active_goal_handle = None
        self.active_goal_id = 0
        self.goal_pending = False
        self.last_goal_xy: Optional[Point2] = None
        self.last_goal_time = -1.0e9
        self.last_goal_outcome = 'none'

        latched_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
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
            latched_qos,
        )
        self.create_subscription(
            Bool,
            self.localization_ready_topic,
            self._localization_ready_callback,
            latched_qos,
        )
        self.status_publisher = self.create_publisher(
            Bool,
            self.follow_status_topic,
            latched_qos,
        )
        self.navigation_client = ActionClient(
            self,
            NavigateToPose,
            self.navigate_action,
        )
        self.create_timer(self.goal_period, self._tick)
        self._publish_status()
        self.get_logger().warning(
            'FLEET_FOLLOWER_MINIMAL_READY | '
            f'leader={self.leader_pose_topic} self={self.follower_pose_topic} '
            f'action={self.navigate_action} distance={self.follow_distance:.2f}'
        )

    @staticmethod
    def _absolute(topic: str) -> str:
        return topic if topic.startswith('/') else '/' + topic

    def _now(self) -> float:
        return self.get_clock().now().nanoseconds * 1.0e-9

    def _leader_pose_callback(self, message: PoseStamped) -> None:
        self.leader_pose = message
        self.leader_pose_wall = self._now()

    def _follower_pose_callback(self, message: PoseStamped) -> None:
        self.follower_pose = message
        self.follower_pose_wall = self._now()

    def _localization_ready_callback(self, message: Bool) -> None:
        self.localization_ready = bool(message.data)

    def _publish_status(self) -> None:
        message = Bool()
        message.data = self.follow_enabled
        self.status_publisher.publish(message)

    def _set_following(self, enabled: bool, command: str) -> None:
        changed = enabled != self.follow_enabled
        self.follow_enabled = enabled
        self._publish_status()
        self.last_goal_xy = None
        self.last_goal_time = -1.0e9
        self.last_goal_outcome = 'none'
        if not enabled:
            self._cancel_goal(command)
        if changed:
            state = 'FOLLOWING' if enabled else 'PAUSED'
            self.get_logger().warning(f'FOLLOW_STATE | state={state} command={command}')

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

    def _make_goal_pose(self, x: float, y: float, yaw: float) -> PoseStamped:
        qx, qy, qz, qw = quaternion_from_yaw(yaw)
        target = PoseStamped()
        target.header.frame_id = 'map'
        target.header.stamp = self.get_clock().now().to_msg()
        target.pose.position.x = float(x)
        target.pose.position.y = float(y)
        target.pose.orientation.x = qx
        target.pose.orientation.y = qy
        target.pose.orientation.z = qz
        target.pose.orientation.w = qw
        return target

    def _target_behind_leader(self) -> PoseStamped:
        assert self.leader_pose is not None
        leader = self.leader_pose.pose
        leader_yaw = yaw_from_quaternion(leader.orientation)
        return self._make_goal_pose(
            leader.position.x - self.follow_distance * math.cos(leader_yaw),
            leader.position.y - self.follow_distance * math.sin(leader_yaw),
            leader_yaw,
        )

    def _should_send(self, target: PoseStamped) -> bool:
        now = self._now()
        current = (target.pose.position.x, target.pose.position.y)
        if self.last_goal_xy is None:
            return True
        moved = math.hypot(
            current[0] - self.last_goal_xy[0],
            current[1] - self.last_goal_xy[1],
        )
        if moved >= self.goal_update_distance:
            return True
        if self.last_goal_outcome == 'failed':
            return now - self.last_goal_time >= self.goal_period
        return False

    def _tick(self) -> None:
        if not self.follow_enabled:
            return
        now = self._now()
        reason = self._blocking_reason(now)
        if reason is not None:
            self._log_follow_debug(reason)
            return
        distance_to_leader = math.hypot(
            self.leader_pose.pose.position.x - self.follower_pose.pose.position.x,
            self.leader_pose.pose.position.y - self.follower_pose.pose.position.y,
        )
        if distance_to_leader <= self.stop_distance:
            self._log_follow_debug('close_to_leader', distance_to_leader=distance_to_leader)
            return
        target = self._target_behind_leader()
        if not self._should_send(target):
            self._log_follow_debug(
                'goal_not_changed',
                distance_to_leader=distance_to_leader,
                target=target,
            )
            return
        self._send_goal(target, distance_to_leader)

    def _blocking_reason(self, now: float) -> Optional[str]:
        if self.leader_pose is None:
            return 'leader_pose_missing'
        if self.leader_pose_wall is None or now - self.leader_pose_wall > self.pose_timeout:
            return 'leader_pose_stale'
        if str(self.leader_pose.header.frame_id or '').strip().lstrip('/') != 'map':
            return 'leader_pose_not_map'
        if self.follower_pose is None:
            return 'self_pose_missing'
        if self.follower_pose_wall is None or now - self.follower_pose_wall > self.pose_timeout:
            return 'self_pose_stale'
        if str(self.follower_pose.header.frame_id or '').strip().lstrip('/') != 'map':
            return 'self_pose_not_map'
        if self.require_localization_ready and not self.localization_ready:
            return 'localization_not_ready'
        if not self.navigation_client.server_is_ready():
            return 'nav_server_unavailable'
        return None

    def _send_goal(self, target: PoseStamped, distance_to_leader: float) -> None:
        goal = NavigateToPose.Goal()
        goal.pose = deepcopy(target)
        goal.pose.header.frame_id = goal.pose.header.frame_id or 'map'
        goal.pose.header.stamp = self.get_clock().now().to_msg()
        self.goal_count += 1
        goal_id = self.goal_count
        self.goal_pending = True
        self.last_goal_xy = (target.pose.position.x, target.pose.position.y)
        self.last_goal_time = self._now()
        self.last_goal_outcome = 'pending'
        future = self.navigation_client.send_goal_async(goal)
        future.add_done_callback(
            partial(self._goal_response_callback, goal_id=goal_id)
        )
        self._log_follow_debug(
            'goal_sent',
            distance_to_leader=distance_to_leader,
            target=target,
            goal_sent=True,
        )
        self.get_logger().warning(
            'FOLLOW_NAV2_DIRECT_GOAL_SENT | '
            f'action={self.navigate_action} goal_id={goal_id} '
            f'x={target.pose.position.x:.3f} y={target.pose.position.y:.3f} '
            f'distance_to_leader={distance_to_leader:.3f}'
        )

    def _goal_response_callback(self, future, goal_id: int) -> None:
        if goal_id == self.goal_count:
            self.goal_pending = False
        try:
            handle = future.result()
        except Exception as error:
            self.get_logger().error(f'FOLLOW_NAV2_DIRECT_GOAL_ERROR | {error}')
            if goal_id == self.goal_count:
                self.last_goal_outcome = 'failed'
            return
        if goal_id != self.goal_count:
            if handle.accepted:
                handle.cancel_goal_async()
            return
        if not handle.accepted:
            self.get_logger().warning('FOLLOW_NAV2_DIRECT_GOAL_REJECTED')
            self.last_goal_outcome = 'failed'
            return
        self.active_goal_handle = handle
        self.active_goal_id = goal_id
        handle.get_result_async().add_done_callback(
            partial(self._goal_result_callback, goal_id=goal_id)
        )
        self.get_logger().warning(
            f'FOLLOW_NAV2_DIRECT_GOAL_ACCEPTED | goal_id={goal_id}'
        )

    def _goal_result_callback(self, future, goal_id: int) -> None:
        succeeded = False
        try:
            result = future.result()
            succeeded = result.status == GoalStatus.STATUS_SUCCEEDED
        except Exception as error:
            self.get_logger().warning(f'FOLLOW_NAV2_DIRECT_RESULT_ERROR | {error}')
        if goal_id == self.goal_count:
            self.last_goal_outcome = 'succeeded' if succeeded else 'failed'
        if goal_id == self.active_goal_id:
            self.active_goal_handle = None
            self.active_goal_id = 0

    def _cancel_goal(self, reason: str) -> None:
        handle = self.active_goal_handle
        self.active_goal_handle = None
        self.active_goal_id = 0
        if handle is None:
            return
        try:
            handle.cancel_goal_async()
            self.get_logger().warning(f'FOLLOW_NAV2_DIRECT_CANCEL | reason={reason}')
        except Exception as error:
            self.get_logger().warning(
                f'FOLLOW_NAV2_DIRECT_CANCEL_ERROR | reason={reason} error={error}'
            )

    def _log_follow_debug(
        self,
        reason: str,
        *,
        distance_to_leader: float = float('nan'),
        target: Optional[PoseStamped] = None,
        goal_sent: bool = False,
    ) -> None:
        now = self._now()
        leader_age_ms = (
            -1.0 if self.leader_pose_wall is None
            else max(0.0, (now - self.leader_pose_wall) * 1000.0)
        )
        self_age_ms = (
            -1.0 if self.follower_pose_wall is None
            else max(0.0, (now - self.follower_pose_wall) * 1000.0)
        )
        target_x = float('nan') if target is None else target.pose.position.x
        target_y = float('nan') if target is None else target.pose.position.y
        self.get_logger().warning(
            'FOLLOWER_FOLLOW_DEBUG | '
            'role=FOLLOWER '
            f'leader_pose_age_ms={leader_age_ms:.0f} '
            f'self_pose_age_ms={self_age_ms:.0f} '
            f'nav_server_ready={self.navigation_client.server_is_ready()} '
            f'distance_to_leader={distance_to_leader:.3f} '
            f'target_x={target_x:.3f} target_y={target_y:.3f} '
            f'goal_sent={goal_sent} '
            f'goal_accepted={self.active_goal_handle is not None} '
            f'blocking_reason={reason}',
            throttle_duration_sec=1.0,
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
