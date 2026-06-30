#!/usr/bin/env python3
"""
Fleet commander node.

Input:
  /fleet/group_goal              geometry_msgs/PoseStamped

Outputs:
  /fleet/<robot>/goal_pose       geometry_msgs/PoseStamped
  /fleet/<robot>/hold            std_msgs/Bool
  /fleet/<robot>/cancel          std_msgs/Bool

This node does not call Nav2 directly. It only expands one group goal into
per-robot formation goals. The per-robot goal topics are meant to be bridged
into each robot's ROS_DOMAIN_ID, where robot_goal_proxy calls local Nav2.
"""

import math
from typing import Dict, List, Tuple

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Bool, String


def quat_to_yaw(q) -> float:
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


def yaw_to_quat(yaw: float):
    qz = math.sin(yaw * 0.5)
    qw = math.cos(yaw * 0.5)
    return 0.0, 0.0, qz, qw


def rotate(dx: float, dy: float, yaw: float) -> Tuple[float, float]:
    c = math.cos(yaw)
    s = math.sin(yaw)
    return c * dx - s * dy, s * dx + c * dy


def parse_robot_names(raw) -> List[str]:
    """Accept either a ROS string array parameter or comma-separated string."""
    if isinstance(raw, (list, tuple)):
        names = [str(x).strip() for x in raw if str(x).strip()]
    else:
        names = [x.strip() for x in str(raw).split(',') if x.strip()]
    return [n.strip('/') for n in names]


def make_formation_offsets(n: int, spacing: float, formation_type: str) -> List[Tuple[float, float]]:
    """
    Create target-local formation slots.

    Local axes:
      +x: target heading direction
      +y: left side of target heading

    Formations:
      wedge: center/front, then rows behind target
      line:  side-by-side line centered at target
      column: one-behind-another column
    """
    if n <= 0:
        return []

    formation_type = formation_type.lower().strip()

    if formation_type == 'line':
        start = -0.5 * (n - 1)
        return [(0.0, (start + i) * spacing) for i in range(n)]

    if formation_type == 'column':
        return [(-i * spacing, 0.0) for i in range(n)]

    # default: wedge/grid-like rows behind the target
    offsets: List[Tuple[float, float]] = [(0.0, 0.0)]
    row = 0
    while len(offsets) < n:
        row += 1
        back = -spacing * row
        # row 1: -1, 0, +1; row 2: -2, -1, 0, +1, +2
        for col in range(-row, row + 1):
            if len(offsets) >= n:
                break
            offsets.append((back, spacing * col))
    return offsets[:n]


class FleetCommanderNode(Node):
    def __init__(self):
        super().__init__('fleet_commander_node')

        self.declare_parameter('robot_names', 'robot1,robot2')
        self.declare_parameter('spacing', 0.60)
        self.declare_parameter('frame_id', 'map')
        self.declare_parameter('formation_type', 'wedge')
        self.declare_parameter('republish_count', 3)
        self.declare_parameter('republish_period_sec', 0.10)

        raw_names = self.get_parameter('robot_names').value
        self.robot_names = parse_robot_names(raw_names)
        if not self.robot_names:
            self.robot_names = ['robot1', 'robot2']

        self.spacing = float(self.get_parameter('spacing').value)
        self.frame_id = str(self.get_parameter('frame_id').value)
        self.formation_type = str(self.get_parameter('formation_type').value)
        self.republish_count = max(1, int(self.get_parameter('republish_count').value))
        self.republish_period_sec = max(0.01, float(self.get_parameter('republish_period_sec').value))

        self.goal_pubs: Dict[str, object] = {}
        self.hold_pubs: Dict[str, object] = {}
        self.cancel_pubs: Dict[str, object] = {}

        for name in self.robot_names:
            self.goal_pubs[name] = self.create_publisher(PoseStamped, f'/fleet/{name}/goal_pose', 10)
            self.hold_pubs[name] = self.create_publisher(Bool, f'/fleet/{name}/hold', 10)
            self.cancel_pubs[name] = self.create_publisher(Bool, f'/fleet/{name}/cancel', 10)
            self.create_subscription(PoseStamped, f'/fleet/{name}/pose', self._make_pose_cb(name), 10)
            self.create_subscription(String, f'/fleet/{name}/status', self._make_status_cb(name), 10)

        self.robot_pose: Dict[str, PoseStamped] = {}
        self.robot_status: Dict[str, str] = {}
        self.pending_msgs: List[Tuple[str, PoseStamped]] = []
        self.pending_republish_left = 0
        self.republish_timer = None

        self.create_subscription(PoseStamped, '/fleet/group_goal', self.on_group_goal, 10)
        self.create_subscription(Bool, '/fleet/group_hold', self.on_group_hold, 10)
        self.create_subscription(Bool, '/fleet/group_cancel', self.on_group_cancel, 10)

        self.get_logger().info(
            'FLEET_COMMANDER_READY | '
            f'robots={self.robot_names} spacing={self.spacing:.2f} '
            f'formation={self.formation_type} frame_id={self.frame_id} '
            f'republish_count={self.republish_count}'
        )

    def _make_pose_cb(self, robot_name: str):
        def cb(msg: PoseStamped):
            self.robot_pose[robot_name] = msg
        return cb

    def _make_status_cb(self, robot_name: str):
        def cb(msg: String):
            self.robot_status[robot_name] = msg.data
        return cb

    def on_group_hold(self, msg: Bool):
        for name, pub in self.hold_pubs.items():
            pub.publish(msg)
            self.get_logger().warn(f'GROUP_HOLD_FORWARDED | robot={name} hold={msg.data}')

    def on_group_cancel(self, msg: Bool):
        if not msg.data:
            return
        for name, pub in self.cancel_pubs.items():
            pub.publish(msg)
            self.get_logger().warn(f'GROUP_CANCEL_FORWARDED | robot={name}')

    def on_group_goal(self, msg: PoseStamped):
        tx = float(msg.pose.position.x)
        ty = float(msg.pose.position.y)
        yaw = quat_to_yaw(msg.pose.orientation)

        offsets = make_formation_offsets(len(self.robot_names), self.spacing, self.formation_type)
        self.pending_msgs = []

        self.get_logger().info(
            f'GROUP_GOAL_RECEIVED | target=({tx:.3f},{ty:.3f},yaw={yaw:.3f}) '
            f'robots={self.robot_names} formation={self.formation_type}'
        )

        for robot_name, (dx_local, dy_local) in zip(self.robot_names, offsets):
            dx_map, dy_map = rotate(dx_local, dy_local, yaw)
            gx = tx + dx_map
            gy = ty + dy_map

            out = PoseStamped()
            out.header.stamp = self.get_clock().now().to_msg()
            out.header.frame_id = self.frame_id
            out.pose.position.x = gx
            out.pose.position.y = gy
            out.pose.position.z = 0.0
            qx, qy, qz, qw = yaw_to_quat(yaw)
            out.pose.orientation.x = qx
            out.pose.orientation.y = qy
            out.pose.orientation.z = qz
            out.pose.orientation.w = qw

            self.pending_msgs.append((robot_name, out))
            self.get_logger().info(
                f'ROBOT_GOAL_PREPARED | robot={robot_name} '
                f'goal=({gx:.3f},{gy:.3f},yaw={yaw:.3f}) '
                f'offset_local=({dx_local:.3f},{dy_local:.3f})'
            )

        self.pending_republish_left = self.republish_count
        self._publish_pending_goals()

        if self.republish_timer is not None:
            self.republish_timer.cancel()
        if self.republish_count > 1:
            self.republish_timer = self.create_timer(self.republish_period_sec, self._republish_timer_cb)

    def _republish_timer_cb(self):
        if self.pending_republish_left <= 0:
            if self.republish_timer is not None:
                self.republish_timer.cancel()
                self.republish_timer = None
            return
        self._publish_pending_goals()

    def _publish_pending_goals(self):
        if self.pending_republish_left <= 0:
            return
        for robot_name, msg in self.pending_msgs:
            msg.header.stamp = self.get_clock().now().to_msg()
            self.goal_pubs[robot_name].publish(msg)
            self.get_logger().info(
                f'ROBOT_GOAL_PUBLISHED | robot={robot_name} '
                f'goal=({msg.pose.position.x:.3f},{msg.pose.position.y:.3f}) '
                f'republish_left={self.pending_republish_left}'
            )
        self.pending_republish_left -= 1


def main():
    rclpy.init()
    node = FleetCommanderNode()
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
