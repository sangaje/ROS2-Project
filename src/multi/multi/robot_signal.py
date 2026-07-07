#!/usr/bin/env python3
"""Publish and monitor lightweight robot heartbeat signals."""

import json
from typing import Dict, List

import rclpy
from rclpy.node import Node
from rclpy.time import Time
from std_msgs.msg import String


def as_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in ('1', 'true', 'yes', 'on')
    return bool(value)


class RobotSignal(Node):
    """Send this robot's heartbeat and report peer heartbeats."""

    def __init__(self) -> None:
        super().__init__('robot_signal')
        self.declare_parameter('robot_name', 'burger1')
        self.declare_parameter('peer_robots', ['waffle1'])
        self.declare_parameter('publish_signal', True)
        self.declare_parameter('signal_period_sec', 0.5)
        self.declare_parameter('signal_timeout_sec', 2.0)
        self.declare_parameter('status_period_sec', 1.0)

        self.robot_name = str(self.get_parameter('robot_name').value).strip('/')
        self.peer_robots: List[str] = list(self.get_parameter('peer_robots').value)
        self.publish_signal = as_bool(self.get_parameter('publish_signal').value)
        self.signal_timeout_sec = float(
            self.get_parameter('signal_timeout_sec').value
        )
        self.sequence = 0
        self.last_signal_time: Dict[str, Time] = {}
        self.last_signal_seq: Dict[str, int] = {}
        self.last_signal_state: Dict[str, str] = {}

        self.signal_pub = None
        if self.publish_signal:
            self.signal_pub = self.create_publisher(
                String,
                f'/{self.robot_name}/signal',
                10,
            )
            self.create_timer(
                float(self.get_parameter('signal_period_sec').value),
                self.publish_heartbeat,
            )

        for peer in self.peer_robots:
            peer_name = str(peer).strip('/')
            if peer_name and peer_name != self.robot_name:
                self.create_subscription(
                    String,
                    f'/{peer_name}/signal',
                    lambda msg, name=peer_name: self.on_signal(name, msg),
                    10,
                )

        self.create_timer(
            float(self.get_parameter('status_period_sec').value),
            self.report_peer_status,
        )
        self.get_logger().info(
            f'RobotSignal started. robot={self.robot_name}, '
            f'publish_signal={self.publish_signal}, '
            f'peers={self.peer_robots}'
        )

    def publish_heartbeat(self) -> None:
        if self.signal_pub is None:
            return
        self.sequence += 1
        msg = String()
        msg.data = json.dumps({
            'robot': self.robot_name,
            'seq': self.sequence,
            'stamp_sec': self.get_clock().now().nanoseconds / 1e9,
        })
        self.signal_pub.publish(msg)

    def on_signal(self, peer: str, msg: String) -> None:
        seq = -1
        try:
            data = json.loads(msg.data)
            seq = int(data.get('seq', -1))
        except Exception:
            pass
        self.last_signal_time[peer] = self.get_clock().now()
        self.last_signal_seq[peer] = seq
        if self.last_signal_state.get(peer) != 'OK':
            self.get_logger().info(f'{peer} signal received.')
        self.last_signal_state[peer] = 'OK'

    def report_peer_status(self) -> None:
        now = self.get_clock().now()
        for peer in self.peer_robots:
            peer_name = str(peer).strip('/')
            if not peer_name or peer_name == self.robot_name:
                continue
            last_time = self.last_signal_time.get(peer_name)
            if last_time is None:
                state = 'WAITING'
            else:
                age = (now - last_time).nanoseconds / 1e9
                state = 'OK' if age <= self.signal_timeout_sec else 'LOST'
            if self.last_signal_state.get(peer_name) == state:
                continue
            seq = self.last_signal_seq.get(peer_name, -1)
            self.last_signal_state[peer_name] = state
            self.get_logger().warn(
                f'{peer_name} signal state={state}, last_seq={seq}'
            )


def main(args=None) -> None:
    rclpy.init(args=args)
    node = RobotSignal()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
