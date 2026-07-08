from __future__ import annotations

import math
from functools import partial
from typing import Optional, Tuple

import rclpy
from geometry_msgs.msg import PoseArray, PoseStamped
from nav2_msgs.action import NavigateToPose
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)
from sensor_msgs.msg import LaserScan
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


def wrap_angle(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


class FleetFollower(Node):
    """Send Nav2 goals behind the leader and obey central PAUSE/RESUME commands."""

    def __init__(self) -> None:
        super().__init__('fleet_follower')

        self.declare_parameter('leader_pose_topic', '/leader_pose')
        self.declare_parameter('follower_pose_topic', '/burger_pose')
        self.declare_parameter('navigate_action', '/navigate_to_pose')
        self.declare_parameter('follow_command_topic', '/fleet/follow_command')
        self.declare_parameter('follow_status_topic', '/fleet/follow_enabled')
        self.declare_parameter(
            'collision_warning_topic', '/fleet/collision_warning'
        )
        self.declare_parameter('fleet_poses_topic', '/fleet/robot_poses')
        self.declare_parameter('scan_topic', '/scan')
        self.declare_parameter('follow_distance', 0.70)
        self.declare_parameter('goal_period_sec', 1.0)
        self.declare_parameter('goal_update_distance', 0.20)
        self.declare_parameter('goal_refresh_sec', 3.0)
        self.declare_parameter('action_wait_timeout_sec', 1.0)
        self.declare_parameter('cancel_previous_goal', False)
        self.declare_parameter('start_following', True)
        self.declare_parameter('use_scan_space_selection', True)
        self.declare_parameter(
            'formation_candidate_angles_deg',
            [180.0, 135.0, -135.0, 90.0, -90.0, 45.0, -45.0, 0.0],
        )
        self.declare_parameter('scan_sector_half_width_deg', 28.0)
        self.declare_parameter('min_slot_clearance_m', 0.45)
        self.declare_parameter('avoidance_trigger_range_m', 0.42)
        self.declare_parameter('avoidance_goal_distance_m', 0.55)
        self.declare_parameter('peer_separation_weight', 0.35)
        self.declare_parameter('localization_ready_topic', '/localization_ready')
        self.declare_parameter('require_localization_ready', True)

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
        self.collision_warning_topic = self._absolute(
            str(parameter('collision_warning_topic').value)
        )
        self.fleet_poses_topic = self._absolute(
            str(parameter('fleet_poses_topic').value)
        )
        self.scan_topic = self._absolute(str(parameter('scan_topic').value))
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
        self.use_scan_space_selection = bool(
            parameter('use_scan_space_selection').value
        )
        self.formation_candidate_angles = [
            math.radians(float(v))
            for v in list(parameter('formation_candidate_angles_deg').value)
        ]
        self.scan_sector_half_width = math.radians(
            max(1.0, float(parameter('scan_sector_half_width_deg').value))
        )
        self.min_slot_clearance = max(
            0.05, float(parameter('min_slot_clearance_m').value)
        )
        self.avoidance_trigger_range = max(
            0.05, float(parameter('avoidance_trigger_range_m').value)
        )
        self.avoidance_goal_distance = max(
            0.10, float(parameter('avoidance_goal_distance_m').value)
        )
        self.peer_separation_weight = max(
            0.0, float(parameter('peer_separation_weight').value)
        )
        self.localization_ready_topic = self._absolute(
            str(parameter('localization_ready_topic').value)
        )
        self.require_localization_ready = bool(
            parameter('require_localization_ready').value
        )

        self.leader_pose: Optional[PoseStamped] = None
        self.follower_pose: Optional[PoseStamped] = None
        self.active_goal_handle = None
        self.active_goal_id = 0
        self.last_goal_xy: Optional[Point2] = None
        self.last_goal_time = -1.0e9
        self.wait_log_time = -1.0e9
        self.goal_count = 0
        self.collision_warning = False
        self.fleet_poses: Optional[PoseArray] = None
        self.latest_scan: Optional[LaserScan] = None
        self.localization_ready = False

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
        self.create_subscription(
            Bool,
            self.collision_warning_topic,
            self._warning_callback,
            coordination_qos,
        )
        self.create_subscription(
            PoseArray,
            self.fleet_poses_topic,
            self._fleet_poses_callback,
            10,
        )
        self.create_subscription(
            LaserScan,
            self.scan_topic,
            self._scan_callback,
            10,
        )
        self.create_subscription(
            Bool,
            self.localization_ready_topic,
            self._localization_ready_callback,
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
            f'scan={self.scan_topic} '
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

    def _warning_callback(self, message: Bool) -> None:
        changed = message.data != self.collision_warning
        self.collision_warning = bool(message.data)
        if changed:
            self.get_logger().warning(
                'FLEET_SAFETY_WARNING | '
                f'active={self.collision_warning} '
                'robot_poses_order=[leader,follower]'
            )

    def _fleet_poses_callback(self, message: PoseArray) -> None:
        self.fleet_poses = message

    def _scan_callback(self, message: LaserScan) -> None:
        self.latest_scan = message

    def _localization_ready_callback(self, message: Bool) -> None:
        self.localization_ready = bool(message.data)

    def _cancel_goal(self, reason: str) -> None:
        handle = self.active_goal_handle
        self.active_goal_handle = None
        self.active_goal_id = 0
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

    def _make_goal_pose(self, x: float, y: float, yaw: float) -> PoseStamped:
        qx, qy, qz, qw = quaternion_from_yaw(yaw)
        target = PoseStamped()
        target.header.frame_id = 'map'
        target.header.stamp = rclpy.time.Time().to_msg()
        target.pose.position.x = x
        target.pose.position.y = y
        target.pose.orientation.x = qx
        target.pose.orientation.y = qy
        target.pose.orientation.z = qz
        target.pose.orientation.w = qw
        return target

    def _scan_clearance_for_world_direction(self, world_direction: float) -> float:
        if self.latest_scan is None or self.follower_pose is None:
            return float('inf')
        follower_yaw = yaw_from_quaternion(self.follower_pose.pose.orientation)
        target_bearing = wrap_angle(world_direction - follower_yaw)
        scan = self.latest_scan
        samples = []
        for index, value in enumerate(scan.ranges):
            if not math.isfinite(value):
                value = scan.range_max
            value = min(max(float(value), scan.range_min), scan.range_max)
            bearing = scan.angle_min + index * scan.angle_increment
            if abs(wrap_angle(bearing - target_bearing)) <= self.scan_sector_half_width:
                samples.append(value)
        if not samples:
            return 0.0
        samples.sort()
        return samples[max(0, int(0.35 * (len(samples) - 1)))]

    def _nearest_obstacle_world_bearing(self) -> Optional[Tuple[float, float]]:
        if self.latest_scan is None or self.follower_pose is None:
            return None
        scan = self.latest_scan
        best_range = float('inf')
        best_bearing = 0.0
        for index, value in enumerate(scan.ranges):
            if not math.isfinite(value):
                continue
            value = float(value)
            if value < scan.range_min or value > scan.range_max:
                continue
            if value < best_range:
                best_range = value
                best_bearing = scan.angle_min + index * scan.angle_increment
        if not math.isfinite(best_range):
            return None
        follower_yaw = yaw_from_quaternion(self.follower_pose.pose.orientation)
        return best_range, wrap_angle(follower_yaw + best_bearing)

    def _avoidance_target_if_needed(self) -> Optional[PoseStamped]:
        nearest = self._nearest_obstacle_world_bearing()
        if nearest is None or self.follower_pose is None or self.leader_pose is None:
            return None
        nearest_range, obstacle_direction = nearest
        if nearest_range > self.avoidance_trigger_range:
            return None

        left = wrap_angle(obstacle_direction + math.pi / 2.0)
        right = wrap_angle(obstacle_direction - math.pi / 2.0)
        left_clear = self._scan_clearance_for_world_direction(left)
        right_clear = self._scan_clearance_for_world_direction(right)
        escape = left if left_clear >= right_clear else right
        follower = self.follower_pose.pose.position
        leader = self.leader_pose.pose.position
        yaw_to_leader = math.atan2(leader.y - follower.y, leader.x - follower.x)
        target = self._make_goal_pose(
            follower.x + self.avoidance_goal_distance * math.cos(escape),
            follower.y + self.avoidance_goal_distance * math.sin(escape),
            yaw_to_leader,
        )
        self.get_logger().warning(
            'FOLLOW_AVOIDANCE_TARGET | '
            f'obstacle_range={nearest_range:.2f} '
            f'escape_clearance={max(left_clear, right_clear):.2f}'
        )
        return target

    def _peer_clearance_score(self, x: float, y: float) -> float:
        if self.fleet_poses is None:
            return 0.0
        best = float('inf')
        for pose in self.fleet_poses.poses:
            d = math.hypot(x - pose.position.x, y - pose.position.y)
            if d > 1e-3:
                best = min(best, d)
        if not math.isfinite(best):
            return 0.0
        return min(2.0, best)

    def _target_open_slot_around_leader(self) -> PoseStamped:
        assert self.leader_pose is not None
        leader = self.leader_pose.pose
        leader_yaw = yaw_from_quaternion(leader.orientation)
        if (
            not self.use_scan_space_selection
            or self.latest_scan is None
            or self.follower_pose is None
        ):
            return self._make_goal_pose(
                leader.position.x - self.follow_distance * math.cos(leader_yaw),
                leader.position.y - self.follow_distance * math.sin(leader_yaw),
                leader_yaw,
            )

        follower = self.follower_pose.pose.position
        best_target = None
        best_score = -float('inf')
        for offset in self.formation_candidate_angles:
            slot_angle = leader_yaw + offset
            x = leader.position.x + self.follow_distance * math.cos(slot_angle)
            y = leader.position.y + self.follow_distance * math.sin(slot_angle)
            direction_from_follower = math.atan2(y - follower.y, x - follower.x)
            clearance = self._scan_clearance_for_world_direction(
                direction_from_follower
            )
            if clearance < self.min_slot_clearance:
                continue
            travel = math.hypot(x - follower.x, y - follower.y)
            peer_score = self._peer_clearance_score(x, y)
            behind_bonus = 0.25 if abs(wrap_angle(offset - math.pi)) < 0.01 else 0.0
            score = (
                clearance
                + self.peer_separation_weight * peer_score
                + behind_bonus
                - 0.20 * travel
            )
            if score > best_score:
                best_score = score
                best_target = self._make_goal_pose(x, y, leader_yaw)

        if best_target is not None:
            return best_target

        return self._make_goal_pose(
            leader.position.x - self.follow_distance * math.cos(leader_yaw),
            leader.position.y - self.follow_distance * math.sin(leader_yaw),
            leader_yaw,
        )

    def _target_behind_leader(self) -> PoseStamped:
        assert self.leader_pose is not None
        avoidance = self._avoidance_target_if_needed()
        if avoidance is not None:
            return avoidance
        return self._target_open_slot_around_leader()

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
        if self.require_localization_ready and not self.localization_ready:
            self._log_wait('waiting for the localization spin to finish')
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
        goal_id = self.goal_count
        future = self.navigation_client.send_goal_async(goal)
        future.add_done_callback(
            partial(self._goal_response_callback, goal_id=goal_id)
        )

    def _goal_response_callback(self, future, goal_id: int) -> None:
        try:
            handle = future.result()
        except Exception as error:
            self.get_logger().error(f'FOLLOW_GOAL_ERROR | {error}')
            return
        if not handle.accepted:
            self.get_logger().warning('FOLLOW_GOAL_REJECTED')
            return
        if not self.follow_enabled or goal_id != self.goal_count:
            handle.cancel_goal_async()
            return
        self.active_goal_handle = handle
        self.active_goal_id = goal_id
        handle.get_result_async().add_done_callback(
            partial(self._goal_result_callback, goal_id=goal_id)
        )
        self.get_logger().info(
            f'FOLLOW_GOAL_ACCEPTED | count={self.goal_count}'
        )

    def _goal_result_callback(self, future, goal_id: int) -> None:
        try:
            future.result()
        except Exception as error:
            self.get_logger().warning(f'FOLLOW_GOAL_RESULT_ERROR | {error}')
        if goal_id == self.active_goal_id:
            self.active_goal_handle = None
            self.active_goal_id = 0


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
