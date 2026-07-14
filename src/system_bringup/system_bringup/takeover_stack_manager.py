#!/usr/bin/env python3
"""Start follower-owned SLAM/Risk stack after ACTIVE_SCOUT takeover."""

from __future__ import annotations

import os
import signal
import subprocess
import time
from typing import Optional

import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import String

from .role_contract import Role, parse_role_message


class TakeoverStackManager(Node):
    """Late-start map authority for a follower that becomes ACTIVE_SCOUT."""

    def __init__(self) -> None:
        super().__init__('takeover_stack_manager')

        self.declare_parameter('robot_name', 'follower21')
        self.declare_parameter('role_topic', '/follower21/role')
        self.declare_parameter('enabled', True)
        self.declare_parameter('start_cartographer', True)
        self.declare_parameter('start_risk_map', True)
        self.declare_parameter('cartographer_configuration_basename', 'turtlebot3_lds_2d_risk_safe_no_odom.lua')
        self.declare_parameter('detection_source', 'flask_topic')
        self.declare_parameter('external_detection_topic', '/field/follower21/risk_observation')
        self.declare_parameter('enable_yolo', False)
        self.declare_parameter('start_camera', False)
        self.declare_parameter('start_rviz', False)
        self.declare_parameter('pre_shutdown_lifecycle_nodes', ['/amcl'])
        self.declare_parameter('lifecycle_command_timeout_sec', 3.0)
        self.declare_parameter('startup_cooldown_sec', 2.0)

        get = self.get_parameter
        self.robot_name = str(get('robot_name').value).strip() or 'follower21'
        self.role_topic = str(get('role_topic').value).strip() or f'/{self.robot_name}/role'
        self.enabled = bool(get('enabled').value)
        self.start_cartographer = bool(get('start_cartographer').value)
        self.start_risk_map = bool(get('start_risk_map').value)
        self.cartographer_configuration = str(
            get('cartographer_configuration_basename').value
        ).strip()
        self.detection_source = str(get('detection_source').value).strip()
        self.external_detection_topic = str(get('external_detection_topic').value).strip()
        self.enable_yolo = bool(get('enable_yolo').value)
        self.start_camera = bool(get('start_camera').value)
        self.start_rviz = bool(get('start_rviz').value)
        self.lifecycle_nodes = [
            str(name).strip()
            for name in get('pre_shutdown_lifecycle_nodes').value
            if str(name).strip()
        ]
        self.lifecycle_timeout = max(
            0.5, float(get('lifecycle_command_timeout_sec').value)
        )
        self.startup_cooldown = max(0.0, float(get('startup_cooldown_sec').value))

        latched_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
        )
        self.create_subscription(String, self.role_topic, self._on_role, latched_qos)

        self.process: Optional[subprocess.Popen] = None
        self.started = False
        self.last_start_attempt_wall = -1.0e9
        self.get_logger().warning(
            'TAKEOVER_STACK_MANAGER_READY | '
            f'enabled={self.enabled} robot={self.robot_name} role_topic={self.role_topic} '
            f'cartographer={self.start_cartographer} risk_map={self.start_risk_map}'
        )

    def _on_role(self, msg: String) -> None:
        update = parse_role_message(msg.data, self.robot_name)
        if update is None:
            return
        if update.robot and update.robot != self.robot_name:
            return
        if update.role != Role.ACTIVE_SCOUT:
            return
        self._start_takeover_stack()

    def _start_takeover_stack(self) -> None:
        if not self.enabled or self.started:
            return
        now = time.monotonic()
        if now - self.last_start_attempt_wall < self.startup_cooldown:
            return
        self.last_start_attempt_wall = now

        self._shutdown_previous_localization()
        cmd = self._launch_command()
        self.get_logger().warning(
            'TAKEOVER_STACK_START | ' + ' '.join(cmd)
        )
        try:
            self.process = subprocess.Popen(cmd, env=os.environ.copy())
        except OSError as exc:
            self.get_logger().error(f'TAKEOVER_STACK_START_FAILED | {exc}')
            return
        self.started = True

    def _shutdown_previous_localization(self) -> None:
        for node_name in self.lifecycle_nodes:
            for transition in ('deactivate', 'shutdown'):
                cmd = ['ros2', 'lifecycle', 'set', node_name, transition]
                try:
                    result = subprocess.run(
                        cmd,
                        env=os.environ.copy(),
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        text=True,
                        timeout=self.lifecycle_timeout,
                        check=False,
                    )
                except (OSError, subprocess.TimeoutExpired) as exc:
                    self.get_logger().warning(
                        'TAKEOVER_LIFECYCLE_COMMAND_FAILED | '
                        f'node={node_name} transition={transition} error={exc}'
                    )
                    continue
                self.get_logger().warning(
                    'TAKEOVER_LIFECYCLE_COMMAND | '
                    f'node={node_name} transition={transition} '
                    f'code={result.returncode} output={result.stdout.strip()}'
                )

    def _launch_command(self) -> list[str]:
        return [
            'ros2',
            'launch',
            'bayesian_risk_map',
            'real_robot_risk_slam.launch.py',
            'start_robot_bringup:=false',
            f'start_cartographer:={str(self.start_cartographer).lower()}',
            f'start_risk_map:={str(self.start_risk_map).lower()}',
            f'cartographer_configuration_basename:={self.cartographer_configuration}',
            f'start_camera:={str(self.start_camera).lower()}',
            f'enable_yolo:={str(self.enable_yolo).lower()}',
            f'detection_source:={self.detection_source}',
            f'external_detection_topic:={self.external_detection_topic}',
            f'start_rviz:={str(self.start_rviz).lower()}',
        ]

    def destroy_node(self) -> bool:
        if self.process is not None and self.process.poll() is None:
            try:
                self.process.send_signal(signal.SIGINT)
                self.process.wait(timeout=3.0)
            except Exception:
                self.process.terminate()
        return super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = TakeoverStackManager()
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
