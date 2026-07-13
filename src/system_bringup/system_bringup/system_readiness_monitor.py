#!/usr/bin/env python3
"""Central dependency-driven startup barrier for the real fleet.

The leader domain owns the authoritative /system/ready latch.  Field robots
receive it through domain_bridge and every motion owner gates nonzero movement
on it.  This node observes real data-flow signals instead of sleeping for a
fixed launch delay.
"""

from __future__ import annotations

import json
import math
from enum import Enum
from typing import Optional

import rclpy
from geometry_msgs.msg import PoseStamped
from lifecycle_msgs.msg import State
from lifecycle_msgs.srv import GetState
from nav2_msgs.action import NavigateToPose
from nav_msgs.msg import OccupancyGrid
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import Bool, String
from tf2_ros import Buffer, ExtrapolationException, LookupException, TransformListener


class ReadinessStage(str, Enum):
    BOOTING = 'BOOTING'
    SENSOR_READY = 'SENSOR_READY'
    MAP_TF_READY = 'MAP_TF_READY'
    LOCALIZATION_READY = 'LOCALIZATION_READY'
    NAV2_READY = 'NAV2_READY'
    DASHBOARD_VIDEO_READY = 'DASHBOARD_VIDEO_READY'
    SYSTEM_READY = 'SYSTEM_READY'
    RUNNING = 'RUNNING'


def _stamp_to_float(stamp) -> float:
    return float(stamp.sec) + float(stamp.nanosec) * 1.0e-9


class SystemReadinessMonitor(Node):
    def __init__(self) -> None:
        super().__init__('system_readiness_monitor')
        self.declare_parameter('map_topic', '/map')
        self.declare_parameter('leader_pose_topic', '/leader_pose')
        self.declare_parameter('scout_pose_topic', '/member_pose')
        self.declare_parameter('follower_pose_topic', '/burger_pose')
        self.declare_parameter('field_robot_status_topic', '/fleet/field_robot_status')
        self.declare_parameter('leader_localization_ready_topic', '/localization_ready')
        self.declare_parameter('video_ready_topic', '/fleet/video_ready')
        self.declare_parameter('require_scout', True)
        self.declare_parameter('require_follower', True)
        self.declare_parameter('leader_bt_state_service', '/bt_navigator/get_state')
        self.declare_parameter('leader_navigate_action', '/navigate_to_pose')
        self.declare_parameter('global_frame', 'map')
        self.declare_parameter('base_frame', 'base_footprint')
        self.declare_parameter('map_min_known_cells', 100)
        self.declare_parameter('pose_timeout_sec', 3.0)
        self.declare_parameter('tf_timeout_sec', 0.2)
        self.declare_parameter('stable_duration_sec', 1.0)
        self.declare_parameter('check_period_sec', 0.25)
        self.declare_parameter('ready_topic', '/system/ready')
        self.declare_parameter('readiness_topic', '/system/readiness')
        self.declare_parameter('detail_topic', '/system/readiness_detail')

        get = self.get_parameter
        self.map_topic = str(get('map_topic').value)
        self.leader_pose_topic = str(get('leader_pose_topic').value)
        self.scout_pose_topic = str(get('scout_pose_topic').value)
        self.follower_pose_topic = str(get('follower_pose_topic').value)
        self.field_status_topic = str(get('field_robot_status_topic').value)
        self.leader_localization_topic = str(get('leader_localization_ready_topic').value)
        self.video_ready_topic = str(get('video_ready_topic').value)
        self.require_scout = bool(get('require_scout').value)
        self.require_follower = bool(get('require_follower').value)
        self.leader_bt_state_service = str(get('leader_bt_state_service').value)
        self.leader_navigate_action = str(get('leader_navigate_action').value)
        self.global_frame = str(get('global_frame').value).strip().lstrip('/')
        self.base_frame = str(get('base_frame').value).strip().lstrip('/')
        self.map_min_known_cells = max(1, int(get('map_min_known_cells').value))
        self.pose_timeout = max(0.2, float(get('pose_timeout_sec').value))
        self.tf_timeout = max(0.05, float(get('tf_timeout_sec').value))
        self.stable_duration = max(0.0, float(get('stable_duration_sec').value))
        self.check_period = max(0.1, float(get('check_period_sec').value))
        self.ready_topic = str(get('ready_topic').value)
        self.readiness_topic = str(get('readiness_topic').value)
        self.detail_topic = str(get('detail_topic').value)

        latched_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
        )
        volatile_qos = QoSProfile(depth=10)
        map_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
        )
        self.ready_pub = self.create_publisher(Bool, self.ready_topic, latched_qos)
        self.stage_pub = self.create_publisher(String, self.readiness_topic, latched_qos)
        self.detail_pub = self.create_publisher(String, self.detail_topic, latched_qos)

        self.create_subscription(OccupancyGrid, self.map_topic, self._on_map, map_qos)
        self.create_subscription(PoseStamped, self.leader_pose_topic, self._on_leader_pose, volatile_qos)
        self.create_subscription(PoseStamped, self.scout_pose_topic, self._on_scout_pose, volatile_qos)
        self.create_subscription(PoseStamped, self.follower_pose_topic, self._on_follower_pose, volatile_qos)
        self.create_subscription(String, self.field_status_topic, self._on_field_status, volatile_qos)
        self.create_subscription(Bool, self.leader_localization_topic, self._on_leader_localization, latched_qos)
        self.create_subscription(Bool, self.video_ready_topic, self._on_video_ready, latched_qos)

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.leader_nav_client = ActionClient(self, NavigateToPose, self.leader_navigate_action)
        self.leader_state_client = self.create_client(GetState, self.leader_bt_state_service)

        self.map_known_cells = 0
        self.map_stamp_sec: Optional[float] = None
        self.leader_pose_wall: Optional[float] = None
        self.scout_pose_wall: Optional[float] = None
        self.follower_pose_wall: Optional[float] = None
        self.leader_localization_ready = False
        self.video_ready = False
        self.field_status_by_robot: dict[str, dict] = {}
        self.leader_nav_active_cached = False
        self.leader_state_request_pending = False
        self.system_good_since: Optional[float] = None
        self.ready = False
        self.stage = ReadinessStage.BOOTING
        self.last_detail = ''

        self._publish(False, ReadinessStage.BOOTING, ['startup'])
        self.create_timer(self.check_period, self._tick)
        self.get_logger().warning(
            'SYSTEM_READINESS_MONITOR_READY | '
            f'ready={self.ready_topic} detail={self.detail_topic} '
            f'map={self.map_topic} scout={self.scout_pose_topic} '
            f'leader={self.leader_pose_topic} follower={self.follower_pose_topic} '
            f'video={self.video_ready_topic}'
        )

    def _now(self) -> float:
        return self.get_clock().now().nanoseconds * 1.0e-9

    def _on_map(self, msg: OccupancyGrid) -> None:
        self.map_known_cells = sum(1 for cell in msg.data if cell >= 0)
        self.map_stamp_sec = _stamp_to_float(msg.header.stamp)

    def _on_leader_pose(self, msg: PoseStamped) -> None:  # noqa: ARG002
        self.leader_pose_wall = self._now()

    def _on_scout_pose(self, msg: PoseStamped) -> None:  # noqa: ARG002
        self.scout_pose_wall = self._now()

    def _on_follower_pose(self, msg: PoseStamped) -> None:  # noqa: ARG002
        self.follower_pose_wall = self._now()

    def _on_leader_localization(self, msg: Bool) -> None:
        self.leader_localization_ready = bool(msg.data)

    def _on_video_ready(self, msg: Bool) -> None:
        self.video_ready = bool(msg.data)

    def _on_field_status(self, msg: String) -> None:
        try:
            data = json.loads(msg.data)
        except (TypeError, json.JSONDecodeError):
            return
        if not isinstance(data, dict):
            return
        robot = str(data.get('robot', '')).strip()
        if not robot:
            return
        data['_received_wall'] = self._now()
        self.field_status_by_robot[robot] = data

    def _fresh(self, wall: Optional[float]) -> bool:
        return wall is not None and self._now() - wall <= self.pose_timeout

    def _tf_ok(self) -> bool:
        try:
            transform = self.tf_buffer.lookup_transform(
                self.global_frame,
                self.base_frame,
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=self.tf_timeout),
            )
        except (LookupException, ExtrapolationException, Exception):  # noqa: BLE001
            return False
        stamp = _stamp_to_float(transform.header.stamp)
        if stamp <= 0.0:
            return True
        return math.isfinite(stamp)

    def _field_robot_ready(self, robot: str) -> bool:
        data = self.field_status_by_robot.get(robot)
        if not data:
            return False
        received = data.get('_received_wall')
        if not isinstance(received, (int, float)) or not self._fresh(float(received)):
            return False
        return bool(
            data.get('localization_ready')
            or data.get('active_scout_ready')
            or str(data.get('role', '')).upper() == 'ACTIVE_SCOUT'
        )

    def _field_robot_nav_ready(self, robot: str) -> bool:
        data = self.field_status_by_robot.get(robot)
        if not data:
            return False
        received = data.get('_received_wall')
        if not isinstance(received, (int, float)) or not self._fresh(float(received)):
            return False
        return bool(data.get('nav_server_ready'))

    def _leader_nav2_ready(self) -> bool:
        if not self.leader_nav_client.server_is_ready():
            return False
        if not self.leader_state_request_pending and self.leader_state_client.service_is_ready():
            self.leader_state_request_pending = True
            future = self.leader_state_client.call_async(GetState.Request())
            future.add_done_callback(self._on_leader_nav_state)
        return self.leader_nav_active_cached

    def _on_leader_nav_state(self, future) -> None:
        self.leader_state_request_pending = False
        try:
            response = future.result()
        except Exception:  # noqa: BLE001
            self.leader_nav_active_cached = False
            return
        self.leader_nav_active_cached = int(response.current_state.id) == State.PRIMARY_STATE_ACTIVE

    def _evaluate(self) -> tuple[ReadinessStage, dict, list[str]]:
        map_ok = self.map_known_cells >= self.map_min_known_cells
        scout_pose_ok = self._fresh(self.scout_pose_wall) if self.require_scout else True
        leader_pose_ok = self._fresh(self.leader_pose_wall)
        follower_pose_ok = (
            self._fresh(self.follower_pose_wall) if self.require_follower else True
        )
        map_tf = map_ok and leader_pose_ok and self._tf_ok()
        scout_localization = (
            scout_pose_ok and self._field_robot_ready('scout22')
            if self.require_scout else True
        )
        follower_localization = (
            follower_pose_ok and self._field_robot_ready('follower21')
            if self.require_follower else True
        )
        leader_nav2 = self._leader_nav2_ready()
        follower_nav2 = (
            self._field_robot_nav_ready('follower21')
            if self.require_follower else True
        )
        domain_bridges = scout_pose_ok and follower_pose_ok
        dashboard = self.video_ready
        detail = {
            'scout_sensor': scout_pose_ok,
            'scout_localization': scout_localization,
            'leader_localization': self.leader_localization_ready,
            'follower_localization': follower_localization,
            'leader_nav2': leader_nav2,
            'follower_nav2': follower_nav2,
            'map_tf': map_tf,
            'domain_bridges': domain_bridges,
            'dashboard': dashboard,
            'scout_video': dashboard,
            'yolo_video': dashboard,
        }
        checks = {
            'sensor': scout_pose_ok and leader_pose_ok and follower_pose_ok,
            'map_tf': map_tf,
            'localization': (
                scout_localization
                and self.leader_localization_ready
                and follower_localization
            ),
            'nav2': leader_nav2 and follower_nav2,
            'dashboard': dashboard,
            'domain_bridges': domain_bridges,
        }
        reasons = [name for name, ok in checks.items() if not ok]
        if not checks['sensor']:
            stage = ReadinessStage.BOOTING
        elif not checks['map_tf']:
            stage = ReadinessStage.SENSOR_READY
        elif not checks['localization']:
            stage = ReadinessStage.MAP_TF_READY
        elif not checks['nav2']:
            stage = ReadinessStage.LOCALIZATION_READY
        elif not checks['dashboard']:
            stage = ReadinessStage.NAV2_READY
        else:
            stage = ReadinessStage.DASHBOARD_VIDEO_READY
        detail['system_ready'] = not reasons
        detail['blocking_reasons'] = reasons
        detail['stage'] = stage.value
        detail['map_known_cells'] = self.map_known_cells
        return stage, detail, reasons

    def _publish(self, ready: bool, stage: ReadinessStage, reasons: list[str]) -> None:
        detail = {
            'scout_sensor': False,
            'scout_localization': False,
            'leader_localization': False,
            'follower_localization': False,
            'leader_nav2': False,
            'follower_nav2': False,
            'map_tf': False,
            'domain_bridges': False,
            'dashboard': False,
            'scout_video': False,
            'yolo_video': False,
            'system_ready': bool(ready),
            'stage': stage.value,
            'blocking_reasons': reasons,
        }
        self._publish_detail(ready, stage, detail)

    def _publish_detail(self, ready: bool, stage: ReadinessStage, detail: dict) -> None:
        self.ready_pub.publish(Bool(data=bool(ready)))
        self.stage_pub.publish(String(data=stage.value))
        text = json.dumps(detail, sort_keys=True)
        self.detail_pub.publish(String(data=text))
        if text != self.last_detail:
            self.last_detail = text
            self.get_logger().warning(
                'SYSTEM_READINESS | '
                f'stage={stage.value} ready={ready} '
                f'blocking={detail.get("blocking_reasons", [])}'
            )

    def _tick(self) -> None:
        stage, detail, reasons = self._evaluate()
        now = self._now()
        if reasons:
            self.system_good_since = None
            ready = False
        else:
            if self.system_good_since is None:
                self.system_good_since = now
            ready = now - self.system_good_since >= self.stable_duration
            if ready:
                stage = ReadinessStage.RUNNING if self.ready else ReadinessStage.SYSTEM_READY
                detail['stage'] = stage.value
        if ready != self.ready or stage != self.stage or detail.get('blocking_reasons'):
            self.ready = ready
            self.stage = stage
            self._publish_detail(ready, stage, detail)


def main() -> None:
    rclpy.init()
    node = SystemReadinessMonitor()
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
