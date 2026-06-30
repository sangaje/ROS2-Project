#!/usr/bin/env python3
"""Simple fleet state monitor for Domain 25."""

import math
from typing import Dict, List

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import String


def parse_robot_names(raw) -> List[str]:
    if isinstance(raw, (list, tuple)):
        return [str(x).strip().strip('/') for x in raw if str(x).strip()]
    return [x.strip().strip('/') for x in str(raw).split(',') if x.strip()]


class FleetStateEcho(Node):
    def __init__(self):
        super().__init__('fleet_state_echo')
        self.declare_parameter('robot_names', 'robot1,robot2')
        self.declare_parameter('print_period_sec', 1.0)
        self.robot_names = parse_robot_names(self.get_parameter('robot_names').value) or ['robot1', 'robot2']
        self.poses: Dict[str, PoseStamped] = {}
        self.status: Dict[str, str] = {}
        for name in self.robot_names:
            self.create_subscription(PoseStamped, f'/fleet/{name}/pose', self._pose_cb(name), 10)
            self.create_subscription(String, f'/fleet/{name}/status', self._status_cb(name), 10)
        self.create_timer(float(self.get_parameter('print_period_sec').value), self.on_timer)
        self.get_logger().info(f'FLEET_STATE_ECHO_READY | robots={self.robot_names}')

    def _pose_cb(self, name):
        def cb(msg):
            self.poses[name] = msg
        return cb

    def _status_cb(self, name):
        def cb(msg):
            self.status[name] = msg.data
        return cb

    def on_timer(self):
        lines = []
        for name in self.robot_names:
            p = self.poses.get(name)
            st = self.status.get(name, 'NO_STATUS')
            if p is None:
                lines.append(f'{name}: pose=NO_POSE status={st}')
            else:
                lines.append(f'{name}: pose=({p.pose.position.x:.2f},{p.pose.position.y:.2f}) status={st}')
        self.get_logger().info(' | '.join(lines))


def main():
    rclpy.init()
    node = FleetStateEcho()
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
