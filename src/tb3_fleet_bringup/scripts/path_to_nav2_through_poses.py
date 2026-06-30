#!/usr/bin/env python3
"""Nav2 NavigateThroughPoses proxy — conservative rewrite.

Subscribes to nav_msgs/Path (waypoint chain from fleet_goal_dispatcher).
  - Multi-pose path  → NavigateThroughPoses  (normal navigation)
  - Single-pose path → treated as PAUSE/BACKUP command:
      cancel current goal, then send single NavigateThroughPoses goal
  - Empty path       → cancel immediately, do not resend

Conservative parameters:
  - min_resend_sec: 0.4  (fast response to collision state changes)
  - change_threshold_m: 0.20
  - Immediate cancel when path length changes (pause/resume detection)
"""

from __future__ import annotations

import math
from typing import List, Optional

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.exceptions import ParameterAlreadyDeclaredException

from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
from nav2_msgs.action import NavigateThroughPoses


def _decl(node, name, default):
    try:
        node.declare_parameter(name, default)
    except ParameterAlreadyDeclaredException:
        pass
    return node.get_parameter(name).value


class PathToNav2ThroughPoses(Node):

    def __init__(self):
        super().__init__('path_to_nav2_through_poses')
        _decl(self, 'use_sim_time', True)
        self.path_topic         = str  (_decl(self, 'path_topic',         '/waffle_waypoints'))
        self.action_name        = str  (_decl(self, 'action_name',        '/navigate_through_poses'))
        self.default_frame_id   = str  (_decl(self, 'default_frame_id',   'map'))
        self.change_threshold_m = float(_decl(self, 'change_threshold_m', 0.20))
        # Conservative: respond quickly to collision-state changes
        self.min_resend_sec     = float(_decl(self, 'min_resend_sec',     0.40))

        self.client = ActionClient(self, NavigateThroughPoses, self.action_name)
        self.create_subscription(Path, self.path_topic, self._on_path, 10)
        self.create_timer(0.3, self._tick)

        self.pending_poses  : Optional[List[PoseStamped]] = None
        self.active_poses   : Optional[List[PoseStamped]] = None
        self.active_handle                                 = None
        self.cancel_in_progress                            = False
        self.last_send_sec                                 = -1.0
        self.goal_seq                                      = 0
        self.wait_logged                                   = False
        self._is_paused                                    = False  # single-pose hold state

        self.get_logger().info(
            f'PATH_TO_NAV2_THROUGH_POSES | '
            f'path={self.path_topic} action={self.action_name} '
            f'min_resend={self.min_resend_sec:.2f}s threshold={self.change_threshold_m:.2f}m'
        )

    # -------------------------------------------------------------------------
    def _now_sec(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    # -------------------------------------------------------------------------
    def _on_path(self, msg: Path) -> None:
        now = self.get_clock().now().to_msg()

        # --- Empty path: immediate cancel, no resend -------------------------
        if not msg.poses:
            if self.active_handle is not None and not self.cancel_in_progress:
                self.cancel_in_progress = True
                self.get_logger().info('CANCEL | empty path received')
                self.active_handle.cancel_goal_async().add_done_callback(
                    self._on_cancel_done)
            self.pending_poses = None
            self._is_paused = True
            return

        # Restamp poses
        poses = []
        for ps in msg.poses:
            out = PoseStamped()
            out.header.frame_id = ps.header.frame_id or self.default_frame_id
            out.header.stamp    = now
            out.pose            = ps.pose
            poses.append(out)

        single_pose = len(poses) == 1
        was_paused  = self._is_paused
        self._is_paused = single_pose

        # --- Detect path change ---------------------------------------------
        if not self._path_changed(poses) and not (was_paused and not single_pose):
            return  # nothing meaningful changed

        # --- Rate limiting (bypass for pause/resume transitions) ------------
        dt = self._now_sec() - self.last_send_sec
        bypass_rate_limit = was_paused and not single_pose  # resuming from pause
        if not bypass_rate_limit and self.last_send_sec > 0 and dt < self.min_resend_sec:
            self.pending_poses = poses
            return

        self.pending_poses = poses

        # Cancel active goal before sending new one
        if self.active_handle is not None and not self.cancel_in_progress:
            self.cancel_in_progress = True
            self.get_logger().info(
                f'CANCEL_FOR_REPLAN | poses={len(poses)} '
                f'{"(single=pause/backup)" if single_pose else ""}'
            )
            self.active_handle.cancel_goal_async().add_done_callback(
                self._on_cancel_done)
            return
        self._tick()

    # -------------------------------------------------------------------------
    def _path_changed(self, new_poses: List[PoseStamped]) -> bool:
        old = self.active_poses

        # Any length difference (including to/from single-pose) = change
        if old is None:
            return True
        if len(new_poses) != len(old):
            return True

        # Check first, mid, last positions
        indices = {0, len(new_poses) - 1}
        if len(new_poses) > 2:
            indices.add(len(new_poses) // 2)
        for i in sorted(indices):
            j = min(i, len(old) - 1)
            dx = new_poses[i].pose.position.x - old[j].pose.position.x
            dy = new_poses[i].pose.position.y - old[j].pose.position.y
            if math.hypot(dx, dy) > self.change_threshold_m:
                return True
        return False

    # -------------------------------------------------------------------------
    def _tick(self) -> None:
        if self.pending_poses is None:
            return
        if self.active_handle is not None or self.cancel_in_progress:
            return
        if not self.client.wait_for_server(timeout_sec=0.01):
            if not self.wait_logged:
                self.get_logger().warn(f'NAV2_ACTION_NOT_READY | {self.action_name}')
                self.wait_logged = True
            return
        self.wait_logged = False
        self._send()

    # -------------------------------------------------------------------------
    def _send(self) -> None:
        poses = self.pending_poses
        if not poses:
            self.pending_poses = None
            return
        self.pending_poses = None
        self.goal_seq += 1
        seq = self.goal_seq

        goal = NavigateThroughPoses.Goal()
        goal.poses = poses

        self.active_poses   = poses
        self.last_send_sec  = self._now_sec()

        summary = ' -> '.join(
            f'({p.pose.position.x:.2f},{p.pose.position.y:.2f})' for p in poses[:4])
        if len(poses) > 4:
            summary += f' ...+{len(poses)-4}'
        self.get_logger().info(
            f'SEND | seq={seq} n={len(poses)} '
            f'{"[PAUSE/BACKUP]" if len(poses)==1 else "[NAVIGATE]"} {summary}'
        )

        fut = self.client.send_goal_async(goal)
        fut.add_done_callback(lambda f: self._on_response(seq, f))

    # -------------------------------------------------------------------------
    def _on_cancel_done(self, fut) -> None:
        try:
            fut.result()
        except Exception as e:
            self.get_logger().warn(f'CANCEL_RESULT | {e}')
        self.cancel_in_progress = False
        self.active_handle      = None
        self.active_poses       = None
        self._tick()

    def _on_response(self, seq: int, fut) -> None:
        try:
            handle = fut.result()
        except Exception as e:
            self.get_logger().error(f'GOAL_RESPONSE_ERROR | seq={seq} {e}')
            self.active_handle = None; self.active_poses = None
            return
        if not handle.accepted:
            self.get_logger().warn(f'GOAL_REJECTED | seq={seq}')
            self.active_handle = None; self.active_poses = None
            return
        self.active_handle = handle
        self.get_logger().info(f'GOAL_ACCEPTED | seq={seq} n={len(self.active_poses or [])}')
        handle.get_result_async().add_done_callback(lambda f: self._on_result(seq, f))

    def _on_result(self, seq: int, fut) -> None:
        try:
            result = fut.result()
            self.get_logger().info(f'GOAL_RESULT | seq={seq} status={int(result.status)}')
        except Exception as e:
            self.get_logger().error(f'GOAL_RESULT_ERROR | seq={seq} {e}')
        self.active_handle = None
        self.active_poses  = None
        if self.pending_poses:
            self._tick()


def main():
    rclpy.init()
    node = PathToNav2ThroughPoses()
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
