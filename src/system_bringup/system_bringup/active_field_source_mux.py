#!/usr/bin/env python3
"""Select the current active field robot source by robot id and epoch."""

from __future__ import annotations

import json
from typing import Any

import rclpy
from nav_msgs.msg import OccupancyGrid
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import String
from geometry_msgs.msg import PoseStamped

from .fleet_registry import build_legacy_registry, normalize_registry
from .role_contract import parse_epoch


def _latched_qos(depth: int = 1) -> QoSProfile:
    return QoSProfile(
        depth=depth,
        reliability=ReliabilityPolicy.RELIABLE,
        durability=DurabilityPolicy.TRANSIENT_LOCAL,
        history=HistoryPolicy.KEEP_LAST,
    )


class ActiveFieldSourceMux(Node):
    def __init__(self) -> None:
        super().__init__('active_field_source_mux')
        self.declare_parameter('fleet_registry_json', '')
        self.declare_parameter('active_scout_robot_name', 'scout22')
        self.declare_parameter('risk_domain_id', '')
        self.declare_parameter('follower_robot_name', 'follower21')
        self.declare_parameter('follower_domain_id', '')
        self.declare_parameter('active_scout_id_topic', '/failover/active_scout_id')
        self.declare_parameter('scout_epoch_topic', '/failover/scout_epoch')

        get = self.get_parameter
        registry = normalize_registry(str(get('fleet_registry_json').value))
        if not registry:
            registry = build_legacy_registry(
                active_scout_robot_name=str(get('active_scout_robot_name').value),
                risk_domain_id=str(get('risk_domain_id').value),
                follower_robot_name=str(get('follower_robot_name').value),
                follower_domain_id=str(get('follower_domain_id').value),
            )
        if not registry:
            registry = build_legacy_registry(
                active_scout_robot_name='scout22',
                risk_domain_id='22',
                follower_robot_name='follower21',
                follower_domain_id='21',
            )

        self.robot_names = [robot.robot_name for robot in registry]
        initial_active = next(
            (robot.robot_name for robot in registry if robot.initial_role == 'ACTIVE_SCOUT'),
            self.robot_names[0],
        )
        self.active_scout_id = initial_active
        self.epoch = 0

        latched = _latched_qos()
        best_effort = QoSProfile(depth=10)
        map_qos = _latched_qos(depth=1)
        self.map_pub = self.create_publisher(OccupancyGrid, '/active_scout/map', map_qos)
        self.pose_pub = self.create_publisher(PoseStamped, '/active_scout/pose', best_effort)
        self.heartbeat_pub = self.create_publisher(String, '/active_scout/heartbeat', best_effort)
        self.risk_pub = self.create_publisher(String, '/active_scout/risk_observation', best_effort)
        self.state_pub = self.create_publisher(String, '/active_scout/source_state', latched)

        self.latest: dict[str, dict[str, Any]] = {
            robot: {'map': None, 'pose': None, 'heartbeat': None} for robot in self.robot_names
        }
        for robot in self.robot_names:
            self.create_subscription(
                OccupancyGrid,
                f'/field/{robot}/map',
                lambda msg, robot=robot: self._on_map(robot, msg),
                map_qos,
            )
            self.create_subscription(
                PoseStamped,
                f'/field/{robot}/pose',
                lambda msg, robot=robot: self._on_pose(robot, msg),
                best_effort,
            )
            self.create_subscription(
                String,
                f'/field/{robot}/heartbeat',
                lambda msg, robot=robot: self._on_heartbeat(robot, msg),
                best_effort,
            )

        if self.robot_names:
            self.create_subscription(
                PoseStamped,
                '/member_pose',
                lambda msg: self._on_legacy_pose(msg, preferred_role='ACTIVE_SCOUT'),
                best_effort,
            )
            self.create_subscription(
                PoseStamped,
                '/burger_pose',
                lambda msg: self._on_legacy_pose(msg, preferred_role='FOLLOWER'),
                best_effort,
            )
            self.create_subscription(
                String,
                '/scout/signal',
                lambda msg: self._on_heartbeat(self.active_scout_id, msg),
                best_effort,
            )
            self.create_subscription(
                String,
                f'/field/{robot}/risk_observation',
                lambda msg, robot=robot: self._on_risk(robot, msg),
                best_effort,
            )

        self.create_subscription(
            String, str(get('active_scout_id_topic').value), self._on_active_scout, latched
        )
        self.create_subscription(
            String, str(get('scout_epoch_topic').value), self._on_epoch, latched
        )
        self.create_timer(1.0, self._publish_state)
        self._publish_state()
        self.get_logger().warning(
            'ACTIVE_FIELD_SOURCE_MUX_READY | '
            f'robots={self.robot_names} active={self.active_scout_id} epoch={self.epoch}'
        )

    def _on_active_scout(self, msg: String) -> None:
        robot = str(msg.data).strip()
        if not robot or robot == self.active_scout_id:
            return
        if robot not in self.latest:
            self.get_logger().error(
                f'ACTIVE_FIELD_SOURCE_UNKNOWN | robot={robot} known={self.robot_names}'
            )
            return
        self.active_scout_id = robot
        self.get_logger().warning(
            f'ACTIVE_FIELD_SOURCE_CHANGED | active={robot} epoch={self.epoch}'
        )
        self._republish_cached()
        self._publish_state()

    def _on_epoch(self, msg: String) -> None:
        epoch = parse_epoch(str(msg.data).strip())
        if epoch is None or epoch < self.epoch:
            return
        if epoch != self.epoch:
            self.epoch = epoch
            self._publish_state()

    def _on_map(self, robot: str, msg: OccupancyGrid) -> None:
        self.latest[robot]['map'] = msg
        if robot == self.active_scout_id:
            self.map_pub.publish(msg)

    def _on_pose(self, robot: str, msg: PoseStamped) -> None:
        self.latest[robot]['pose'] = msg
        if robot == self.active_scout_id:
            self.pose_pub.publish(msg)

    def _on_legacy_pose(self, msg: PoseStamped, *, preferred_role: str) -> None:
        robot = self.active_scout_id
        if preferred_role == 'FOLLOWER':
            follower = next((name for name in self.robot_names if name != self.active_scout_id), '')
            robot = follower or self.active_scout_id
        self._on_pose(robot, msg)

    def _on_heartbeat(self, robot: str, msg: String) -> None:
        if not self._message_matches(robot, msg.data, allow_missing_epoch=True):
            return
        self.latest[robot]['heartbeat'] = msg
        if robot == self.active_scout_id:
            self.heartbeat_pub.publish(msg)

    def _on_risk(self, robot: str, msg: String) -> None:
        if not self._message_matches(robot, msg.data, allow_missing_epoch=False):
            return
        if robot == self.active_scout_id:
            self.risk_pub.publish(msg)

    def _message_matches(self, robot: str, raw: str, *, allow_missing_epoch: bool) -> bool:
        if robot != self.active_scout_id:
            return False
        try:
            payload = json.loads(raw)
        except (TypeError, json.JSONDecodeError):
            return allow_missing_epoch and self.epoch == 0
        if not isinstance(payload, dict):
            return False
        payload_robot = str(payload.get('robot', robot)).strip()
        if payload_robot and payload_robot != robot:
            return False
        epoch = parse_epoch(payload.get('epoch', payload.get('role_epoch')))
        if epoch is None:
            return allow_missing_epoch and self.epoch == 0
        return epoch == self.epoch

    def _republish_cached(self) -> None:
        cached = self.latest.get(self.active_scout_id, {})
        if cached.get('map') is not None:
            self.map_pub.publish(cached['map'])
        if cached.get('pose') is not None:
            self.pose_pub.publish(cached['pose'])
        if cached.get('heartbeat') is not None:
            self.heartbeat_pub.publish(cached['heartbeat'])

    def _publish_state(self) -> None:
        msg = String()
        msg.data = json.dumps({
            'active_scout_id': self.active_scout_id,
            'epoch': self.epoch,
            'robots': self.robot_names,
        }, sort_keys=True)
        self.state_pub.publish(msg)


def main() -> None:
    rclpy.init()
    node = ActiveFieldSourceMux()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.destroy_node()
        finally:
            if rclpy.ok():
                rclpy.shutdown()


if __name__ == '__main__':
    main()
