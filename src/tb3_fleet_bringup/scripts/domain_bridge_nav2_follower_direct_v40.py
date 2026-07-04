#!/usr/bin/env python3

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import OccupancyGrid, Path
from nav2_msgs.action import NavigateToPose
from std_msgs.msg import Bool, String


Point2 = Tuple[float, float]
BlockInfo = Tuple[float, float, float, float, float]


def _yaw_from_quat(q) -> float:
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


def _quat_from_yaw(yaw: float):
    half = yaw * 0.5
    return (0.0, 0.0, math.sin(half), math.cos(half))


class DomainBridgeNav2Follower(Node):
    """Follow Waffle with Nav2 and yield when Burger blocks Waffle's path.

    Follower-domain inputs:
      /leader_pose: bridged Waffle pose
      /waffle_plan: bridged and remapped Waffle Nav2 global path
      /burger_pose: Burger pose in the shared map frame
      /map: Burger Cartographer map (or a bridged static map)

    Output:
      /navigate_to_pose goals for Burger Nav2
    """

    def __init__(self) -> None:
        super().__init__('domain_bridge_nav2_follower')

        self.declare_parameter('leader_pose_topic', '/leader_pose')
        self.declare_parameter('leader_path_topic', '/waffle_plan')
        self.declare_parameter('follower_pose_topic', '/burger_pose')
        self.declare_parameter('map_topic', '/map')
        self.declare_parameter('navigate_action', '/navigate_to_pose')
        self.declare_parameter('follow_distance', 1.05)
        self.declare_parameter('goal_period_sec', 1.0)
        self.declare_parameter('goal_update_distance', 0.20)
        self.declare_parameter('follow_goal_refresh_sec', 3.0)
        self.declare_parameter('wait_for_action_timeout_sec', 1.0)
        self.declare_parameter('cancel_previous_goal', False)
        self.declare_parameter('log_wait_every_n', 10)
        self.declare_parameter('follow_command_topic', '/fleet/follow_command')
        self.declare_parameter('follow_status_topic', '/fleet/follow_enabled')
        self.declare_parameter('start_following', True)

        self.declare_parameter('enable_path_yield', True)
        self.declare_parameter('path_stale_sec', 3.0)
        self.declare_parameter('path_block_distance', 0.55)
        self.declare_parameter('path_lookahead_min', 0.30)
        self.declare_parameter('path_lookahead_max', 2.50)
        self.declare_parameter('yield_lateral_distance', 0.75)
        self.declare_parameter('yield_release_distance', 0.80)
        self.declare_parameter('yield_map_clearance', 0.28)
        self.declare_parameter('yield_occupied_threshold', 50)
        self.declare_parameter('yield_allow_unknown', False)
        self.declare_parameter('yield_min_hold_sec', 4.0)
        self.declare_parameter('yield_max_hold_sec', 12.0)
        self.declare_parameter('yield_pass_distance', 0.35)
        self.declare_parameter('yield_goal_refresh_sec', 3.0)

        self.leader_pose_topic = self._abs(str(self.get_parameter('leader_pose_topic').value))
        self.leader_path_topic = self._abs(str(self.get_parameter('leader_path_topic').value))
        self.follower_pose_topic = self._abs(str(self.get_parameter('follower_pose_topic').value))
        self.map_topic = self._abs(str(self.get_parameter('map_topic').value))
        self.navigate_action = self._abs(str(self.get_parameter('navigate_action').value))
        self.follow_distance = float(self.get_parameter('follow_distance').value)
        self.goal_period_sec = max(0.2, float(self.get_parameter('goal_period_sec').value))
        self.goal_update_distance = max(0.05, float(self.get_parameter('goal_update_distance').value))
        self.follow_goal_refresh_sec = max(
            1.0,
            float(self.get_parameter('follow_goal_refresh_sec').value),
        )
        self.wait_timeout = max(0.0, float(self.get_parameter('wait_for_action_timeout_sec').value))
        self.cancel_previous_goal = bool(self.get_parameter('cancel_previous_goal').value)
        self.log_wait_every_n = max(1, int(self.get_parameter('log_wait_every_n').value))
        self.follow_command_topic = self._abs(str(self.get_parameter('follow_command_topic').value))
        self.follow_status_topic = self._abs(str(self.get_parameter('follow_status_topic').value))
        self.follow_enabled = bool(self.get_parameter('start_following').value)

        self.enable_path_yield = bool(self.get_parameter('enable_path_yield').value)
        self.path_stale_sec = max(0.5, float(self.get_parameter('path_stale_sec').value))
        self.path_block_distance = max(0.15, float(self.get_parameter('path_block_distance').value))
        self.path_lookahead_min = max(0.0, float(self.get_parameter('path_lookahead_min').value))
        self.path_lookahead_max = max(
            self.path_lookahead_min + 0.1,
            float(self.get_parameter('path_lookahead_max').value),
        )
        self.yield_lateral_distance = max(0.35, float(self.get_parameter('yield_lateral_distance').value))
        self.yield_release_distance = max(
            self.path_block_distance + 0.1,
            float(self.get_parameter('yield_release_distance').value),
        )
        self.yield_map_clearance = max(0.15, float(self.get_parameter('yield_map_clearance').value))
        self.yield_occupied_threshold = int(self.get_parameter('yield_occupied_threshold').value)
        self.yield_allow_unknown = bool(self.get_parameter('yield_allow_unknown').value)
        self.yield_min_hold_sec = max(0.0, float(self.get_parameter('yield_min_hold_sec').value))
        self.yield_max_hold_sec = max(
            self.yield_min_hold_sec + 1.0,
            float(self.get_parameter('yield_max_hold_sec').value),
        )
        self.yield_pass_distance = max(0.0, float(self.get_parameter('yield_pass_distance').value))
        self.yield_goal_refresh_sec = max(1.0, float(self.get_parameter('yield_goal_refresh_sec').value))

        self.leader_pose: Optional[PoseStamped] = None
        self.follower_pose: Optional[PoseStamped] = None
        self.leader_path: Optional[Path] = None
        self.map_msg: Optional[OccupancyGrid] = None
        self.path_received_sec: Optional[float] = None

        self.last_goal_xy: Optional[Point2] = None
        self.last_goal_mode = ''
        self.last_goal_sent_sec = -1.0e9
        self.active_goal_handle = None
        self.goal_count = 0
        self.wait_count = 0

        self.yield_active = False
        self.yield_target: Optional[Tuple[float, float, float]] = None
        self.yield_started_sec = 0.0
        self.yield_block_point: Optional[Point2] = None
        self.yield_tangent: Optional[Point2] = None
        self.no_safe_yield_hold = False
        self.last_no_yield_log_sec = -1.0e9

        map_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.create_subscription(PoseStamped, self.leader_pose_topic, self._on_leader_pose, 20)
        self.create_subscription(PoseStamped, self.follower_pose_topic, self._on_follower_pose, 20)
        self.create_subscription(Path, self.leader_path_topic, self._on_leader_path, 10)
        self.create_subscription(OccupancyGrid, self.map_topic, self._on_map, map_qos)
        self.create_subscription(String, self.follow_command_topic, self._on_follow_command, 10)
        status_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.follow_status_pub = self.create_publisher(Bool, self.follow_status_topic, status_qos)
        self.client = ActionClient(self, NavigateToPose, self.navigate_action)
        self.create_timer(self.goal_period_sec, self._tick)
        self._publish_follow_status()

        self.get_logger().info(
            'V57_NAV2_FOLLOWER_SIGNAL_READY | '
            f'leader={self.leader_pose_topic} path={self.leader_path_topic} '
            f'follower={self.follower_pose_topic} action={self.navigate_action} '
            f'follow={self.follow_distance:.2f}m block={self.path_block_distance:.2f}m '
            f'yield={self.yield_lateral_distance:.2f}m command={self.follow_command_topic} '
            f'enabled={self.follow_enabled}'
        )

    @staticmethod
    def _abs(topic: str) -> str:
        return topic if topic.startswith('/') else '/' + topic

    def _now_sec(self) -> float:
        return self.get_clock().now().nanoseconds * 1.0e-9

    def _on_leader_pose(self, msg: PoseStamped) -> None:
        self.leader_pose = msg

    def _on_follower_pose(self, msg: PoseStamped) -> None:
        self.follower_pose = msg

    def _on_leader_path(self, msg: Path) -> None:
        if len(msg.poses) >= 2:
            self.leader_path = msg
            self.path_received_sec = self._now_sec()

    def _on_map(self, msg: OccupancyGrid) -> None:
        self.map_msg = msg

    def _publish_follow_status(self) -> None:
        msg = Bool()
        msg.data = self.follow_enabled
        self.follow_status_pub.publish(msg)

    def _cancel_active_goal(self, reason: str) -> None:
        handle = self.active_goal_handle
        self.active_goal_handle = None
        if handle is None:
            return
        try:
            handle.cancel_goal_async()
            self.get_logger().info(f'V57_FOLLOW_GOAL_CANCEL_REQUESTED | reason={reason}')
        except Exception as exc:
            self.get_logger().warn(f'V57_FOLLOW_GOAL_CANCEL_ERROR | reason={reason} error={exc}')

    def _set_follow_enabled(self, enabled: bool, command: str) -> None:
        changed = enabled != self.follow_enabled
        self.follow_enabled = enabled
        self._publish_follow_status()
        if not enabled:
            self._cancel_active_goal(command)
            self.yield_active = False
            self.yield_target = None
            self.yield_block_point = None
            self.yield_tangent = None
            self.no_safe_yield_hold = False
        self.last_goal_xy = None
        self.last_goal_mode = ''
        if changed:
            state = 'FOLLOWING' if enabled else 'PAUSED'
            self.get_logger().warn(f'V57_FOLLOW_STATE | state={state} command={command}')
        else:
            self.get_logger().info(
                f'V57_FOLLOW_STATE_UNCHANGED | enabled={enabled} command={command}'
            )

    def _on_follow_command(self, msg: String) -> None:
        command = msg.data.strip().upper()
        if command in ('FOLLOW', 'RESUME', 'START', 'ON', '1', 'TRUE'):
            self._set_follow_enabled(True, command)
        elif command in ('PAUSE', 'STOP', 'HOLD', 'OFF', '0', 'FALSE'):
            self._set_follow_enabled(False, command)
        elif command == 'TOGGLE':
            self._set_follow_enabled(not self.follow_enabled, command)
        else:
            self.get_logger().warn(
                f'V57_FOLLOW_COMMAND_IGNORED | command={msg.data!r} '
                'valid=FOLLOW|RESUME|PAUSE|STOP|TOGGLE'
            )

    def _make_goal(self, x: float, y: float, yaw: float) -> PoseStamped:
        qx, qy, qz, qw = _quat_from_yaw(yaw)
        goal = PoseStamped()
        # A zero stamp asks TF/Nav2 for the latest transform. This avoids stale
        # cross-machine timestamps when a goal waits for AMCL to become ready.
        goal.header.stamp = rclpy.time.Time().to_msg()
        goal.header.frame_id = 'map'
        goal.pose.position.x = x
        goal.pose.position.y = y
        goal.pose.orientation.x = qx
        goal.pose.orientation.y = qy
        goal.pose.orientation.z = qz
        goal.pose.orientation.w = qw
        return goal

    def _target_from_leader(self) -> PoseStamped:
        assert self.leader_pose is not None
        p = self.leader_pose.pose.position
        yaw = _yaw_from_quat(self.leader_pose.pose.orientation)
        return self._make_goal(
            p.x - self.follow_distance * math.cos(yaw),
            p.y - self.follow_distance * math.sin(yaw),
            yaw,
        )

    def _path_points(self) -> List[Point2]:
        if self.leader_path is None:
            return []
        return [(p.pose.position.x, p.pose.position.y) for p in self.leader_path.poses]

    def _upcoming_path_nearest(self) -> Optional[BlockInfo]:
        if (
            not self.enable_path_yield
            or self.leader_pose is None
            or self.follower_pose is None
            or self.leader_path is None
            or self.path_received_sec is None
            or self._now_sec() - self.path_received_sec > self.path_stale_sec
        ):
            return None

        points = self._path_points()
        if len(points) < 2:
            return None

        lx = self.leader_pose.pose.position.x
        ly = self.leader_pose.pose.position.y
        bx = self.follower_pose.pose.position.x
        by = self.follower_pose.pose.position.y
        leader_idx = min(
            range(len(points)),
            key=lambda i: (points[i][0] - lx) ** 2 + (points[i][1] - ly) ** 2,
        )

        along = 0.0
        best = None
        for i in range(leader_idx, len(points)):
            if i > leader_idx:
                along += math.hypot(
                    points[i][0] - points[i - 1][0],
                    points[i][1] - points[i - 1][1],
                )
            if along < self.path_lookahead_min:
                continue
            if along > self.path_lookahead_max:
                break
            distance = math.hypot(points[i][0] - bx, points[i][1] - by)
            if best is None or distance < best[0]:
                best = (distance, i)

        if best is None:
            return None

        _, idx = best
        prev_idx = max(0, idx - 1)
        next_idx = min(len(points) - 1, idx + 1)
        tx = points[next_idx][0] - points[prev_idx][0]
        ty = points[next_idx][1] - points[prev_idx][1]
        norm = math.hypot(tx, ty)
        if norm < 1.0e-6:
            return None
        return best[0], points[idx][0], points[idx][1], tx / norm, ty / norm

    def _path_block_info(self) -> Optional[BlockInfo]:
        nearest = self._upcoming_path_nearest()
        if nearest is None or nearest[0] > self.path_block_distance:
            return None
        return nearest

    def _distance_to_upcoming_path(self) -> float:
        nearest = self._upcoming_path_nearest()
        return nearest[0] if nearest is not None else float('inf')

    def _world_to_map(self, x: float, y: float) -> Optional[Tuple[int, int]]:
        if self.map_msg is None:
            return None
        info = self.map_msg.info
        resolution = float(info.resolution)
        if resolution <= 0.0:
            return None
        origin = info.origin
        yaw = _yaw_from_quat(origin.orientation)
        dx = x - origin.position.x
        dy = y - origin.position.y
        local_x = math.cos(yaw) * dx + math.sin(yaw) * dy
        local_y = -math.sin(yaw) * dx + math.cos(yaw) * dy
        mx = int(math.floor(local_x / resolution))
        my = int(math.floor(local_y / resolution))
        if mx < 0 or my < 0 or mx >= info.width or my >= info.height:
            return None
        return mx, my

    def _candidate_is_free(self, x: float, y: float) -> bool:
        if self.map_msg is None:
            return False
        center = self._world_to_map(x, y)
        if center is None:
            return False
        info = self.map_msg.info
        radius_cells = max(1, int(math.ceil(self.yield_map_clearance / info.resolution)))
        cx, cy = center
        for dy in range(-radius_cells, radius_cells + 1):
            for dx in range(-radius_cells, radius_cells + 1):
                if dx * dx + dy * dy > radius_cells * radius_cells:
                    continue
                mx = cx + dx
                my = cy + dy
                if mx < 0 or my < 0 or mx >= info.width or my >= info.height:
                    return False
                value = int(self.map_msg.data[my * info.width + mx])
                if value < 0 and not self.yield_allow_unknown:
                    return False
                if value >= self.yield_occupied_threshold:
                    return False
        return True

    def _point_distance_to_path(self, x: float, y: float) -> float:
        points = self._path_points()
        if not points:
            return float('inf')
        return min(math.hypot(px - x, py - y) for px, py in points)

    def _select_yield_target(self, block: BlockInfo) -> Optional[Tuple[float, float, float]]:
        if self.leader_pose is None or self.follower_pose is None:
            return None
        _, px, py, tx, ty = block
        nx, ny = -ty, tx
        yaw = math.atan2(ty, tx)
        lx = self.leader_pose.pose.position.x
        ly = self.leader_pose.pose.position.y
        bx = self.follower_pose.pose.position.x
        by = self.follower_pose.pose.position.y

        candidates: List[Tuple[float, float, float]] = []
        for lateral in (self.yield_lateral_distance, self.yield_lateral_distance + 0.20):
            for side in (1.0, -1.0):
                candidates.append((
                    px + side * lateral * nx - 0.15 * tx,
                    py + side * lateral * ny - 0.15 * ty,
                    yaw,
                ))

        away_x = bx - lx
        away_y = by - ly
        away_norm = math.hypot(away_x, away_y)
        if away_norm > 1.0e-6:
            candidates.append((
                bx + self.yield_lateral_distance * away_x / away_norm,
                by + self.yield_lateral_distance * away_y / away_norm,
                yaw,
            ))

        valid = []
        minimum_path_clearance = max(
            self.path_block_distance + 0.10,
            self.yield_lateral_distance * 0.80,
        )
        for cx, cy, cyaw in candidates:
            if not self._candidate_is_free(cx, cy):
                continue
            path_clearance = self._point_distance_to_path(cx, cy)
            if path_clearance < minimum_path_clearance:
                continue
            leader_distance = math.hypot(cx - lx, cy - ly)
            move_distance = math.hypot(cx - bx, cy - by)
            if leader_distance < self.yield_release_distance:
                continue
            score = path_clearance + 0.50 * leader_distance - 0.30 * move_distance
            valid.append((score, cx, cy, cyaw))
        if not valid:
            return None
        _, x, y, target_yaw = max(valid, key=lambda item: item[0])
        return x, y, target_yaw

    def _leader_passed_yield_point(self) -> bool:
        if (
            self.leader_pose is None
            or self.yield_block_point is None
            or self.yield_tangent is None
        ):
            return False
        lx = self.leader_pose.pose.position.x
        ly = self.leader_pose.pose.position.y
        dx = lx - self.yield_block_point[0]
        dy = ly - self.yield_block_point[1]
        return dx * self.yield_tangent[0] + dy * self.yield_tangent[1] >= self.yield_pass_distance

    def _update_yield_state(self) -> Optional[PoseStamped]:
        now = self._now_sec()
        block = self._path_block_info()

        if not self.yield_active and block is None and self.no_safe_yield_hold:
            self.no_safe_yield_hold = False
            self.last_goal_xy = None
            self.last_goal_mode = ''
            self.get_logger().info('V56_YIELD_HOLD_EXIT | Waffle path no longer blocked; resume follow')

        if not self.yield_active and block is not None:
            target = self._select_yield_target(block)
            if target is None:
                if not self.no_safe_yield_hold and self.active_goal_handle is not None:
                    try:
                        self.active_goal_handle.cancel_goal_async()
                    except Exception:
                        pass
                self.no_safe_yield_hold = True
                self.last_goal_xy = None
                self.last_goal_mode = ''
                if now - self.last_no_yield_log_sec >= 2.0:
                    self.get_logger().warn(
                        'V56_YIELD_NO_SAFE_SPOT | cancel follow and hold; no mapped free side spot exists'
                    )
                    self.last_no_yield_log_sec = now
                return None
            self.no_safe_yield_hold = False
            self.yield_active = True
            self.yield_target = target
            self.yield_started_sec = now
            self.yield_block_point = (block[1], block[2])
            self.yield_tangent = (block[3], block[4])
            self.last_goal_xy = None
            self.last_goal_mode = ''
            self.get_logger().warn(
                f'V56_YIELD_ENTER | path_distance={block[0]:.2f}m '
                f'target=({target[0]:.2f},{target[1]:.2f})'
            )

        if not self.yield_active or self.yield_target is None:
            return None

        elapsed = now - self.yield_started_sec
        far_from_path = self._distance_to_upcoming_path() >= self.yield_release_distance
        leader_passed = self._leader_passed_yield_point()
        if (
            elapsed >= self.yield_min_hold_sec
            and far_from_path
            and (leader_passed or elapsed >= self.yield_max_hold_sec)
        ):
            self.get_logger().info(
                f'V56_YIELD_EXIT | elapsed={elapsed:.1f}s leader_passed={leader_passed}; resume follow'
            )
            self.yield_active = False
            self.yield_target = None
            self.yield_block_point = None
            self.yield_tangent = None
            self.last_goal_xy = None
            self.last_goal_mode = ''
            return None

        return self._make_goal(*self.yield_target)

    def _send_goal(self, goal_pose: PoseStamped, mode: str) -> None:
        gx = goal_pose.pose.position.x
        gy = goal_pose.pose.position.y
        now = self._now_sec()
        if self.last_goal_xy is not None and self.last_goal_mode == mode:
            distance = math.hypot(gx - self.last_goal_xy[0], gy - self.last_goal_xy[1])
            refresh_period = (
                self.yield_goal_refresh_sec
                if mode == 'yield'
                else self.follow_goal_refresh_sec
            )
            refresh_due = now - self.last_goal_sent_sec >= refresh_period
            if distance < self.goal_update_distance and not refresh_due:
                return

        if self.cancel_previous_goal and self.active_goal_handle is not None:
            try:
                self.active_goal_handle.cancel_goal_async()
            except Exception:
                pass

        goal_msg = NavigateToPose.Goal()
        goal_msg.pose = goal_pose
        goal_msg.pose.header.stamp = rclpy.time.Time().to_msg()
        self.last_goal_xy = (gx, gy)
        self.last_goal_mode = mode
        self.last_goal_sent_sec = now
        self.goal_count += 1
        self.get_logger().info(
            f'V56_{mode.upper()}_GOAL_SENT | n={self.goal_count} target=({gx:.2f},{gy:.2f})'
        )
        future = self.client.send_goal_async(goal_msg)
        future.add_done_callback(lambda f, sent_mode=mode: self._on_goal_response(f, sent_mode))

    def _tick(self) -> None:
        if not self.follow_enabled:
            return
        if self.leader_pose is None:
            self.wait_count += 1
            if self.wait_count % self.log_wait_every_n == 1:
                self.get_logger().warn('V56_FOLLOW_WAIT | no /leader_pose yet')
            return
        if self.follower_pose is None:
            self.wait_count += 1
            if self.wait_count % self.log_wait_every_n == 1:
                self.get_logger().warn(
                    'V58_FOLLOW_WAIT | follower localization not ready '
                    '(waiting for map -> base_footprint)'
                )
            return

        if not self.client.wait_for_server(timeout_sec=self.wait_timeout):
            self.wait_count += 1
            if self.wait_count % self.log_wait_every_n == 1:
                self.get_logger().warn(f'V56_FOLLOW_WAIT | action server not ready: {self.navigate_action}')
            return

        yield_goal = self._update_yield_state()
        if yield_goal is not None:
            self._send_goal(yield_goal, 'yield')
            return
        if self.no_safe_yield_hold:
            return
        self._send_goal(self._target_from_leader(), 'follow')

    def _on_goal_response(self, future, mode: str) -> None:
        try:
            handle = future.result()
        except Exception as exc:
            self.get_logger().error(f'V56_{mode.upper()}_GOAL_ERROR | {exc}')
            return
        if not handle.accepted:
            self.get_logger().warn(f'V56_{mode.upper()}_GOAL_REJECTED')
            return
        if not self.follow_enabled:
            try:
                handle.cancel_goal_async()
            except Exception:
                pass
            self.get_logger().warn(
                f'V57_{mode.upper()}_GOAL_CANCEL_ON_PAUSE | goal was accepted after pause'
            )
            return
        self.active_goal_handle = handle
        self.get_logger().info(f'V56_{mode.upper()}_GOAL_ACCEPTED')


def main() -> None:
    rclpy.init()
    node = DomainBridgeNav2Follower()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.destroy_node()
        except KeyboardInterrupt:
            pass
        if rclpy.ok():
            try:
                rclpy.shutdown()
            except KeyboardInterrupt:
                pass


if __name__ == '__main__':
    main()
