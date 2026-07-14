#!/usr/bin/env python3
"""Leader-owned Flask dashboard for fleet, risk map, and OMX debug status."""

from __future__ import annotations

import logging
import math
import threading
import time
import json
import urllib.error
import urllib.request
from collections import deque
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import rclpy
from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import (
    Point,
    PointStamped,
    PoseArray,
    PoseStamped,
    Twist,
    TwistStamped,
)
from nav_msgs.msg import OccupancyGrid, Path as NavPath
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import Bool, Empty, Float32, Int32, String


def _declare(node: Node, name: str, default: Any) -> Any:
    node.declare_parameter(name, default)
    return node.get_parameter(name).value


def _quaternion_to_yaw(q: Any) -> float:
    return math.atan2(
        2.0 * (q.w * q.z + q.x * q.y),
        1.0 - 2.0 * (q.y * q.y + q.z * q.z),
    )


def _stamp_to_float(stamp: Any) -> float:
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9


def _grid_signature(msg: OccupancyGrid) -> Dict[str, Any]:
    info = msg.info
    origin = info.origin
    return {
        'frame_id': msg.header.frame_id,
        'width': int(info.width),
        'height': int(info.height),
        'resolution': float(info.resolution),
        'origin': {
            'x': float(origin.position.x),
            'y': float(origin.position.y),
            'z': float(origin.position.z),
            'qx': float(origin.orientation.x),
            'qy': float(origin.orientation.y),
            'qz': float(origin.orientation.z),
            'qw': float(origin.orientation.w),
            'yaw': _quaternion_to_yaw(origin.orientation),
        },
    }


def _metadata_match(a: Optional[Dict[str, Any]], b: Optional[Dict[str, Any]]) -> bool:
    if not a or not b:
        return False
    if a['frame_id'] != b['frame_id']:
        return False
    if a['width'] != b['width'] or a['height'] != b['height']:
        return False
    if abs(a['resolution'] - b['resolution']) > 1e-6:
        return False
    ao = a['origin']
    bo = b['origin']
    for key in ('x', 'y', 'z', 'qx', 'qy', 'qz', 'qw'):
        if abs(float(ao[key]) - float(bo[key])) > 1e-5:
            return False
    return True


def _grid_stats(msg: OccupancyGrid) -> Dict[str, Any]:
    data = list(msg.data)
    valid = [int(value) for value in data if int(value) >= 0]
    positives = [value for value in valid if value > 0]
    return {
        'data_length': len(data),
        'unknown_count': sum(1 for value in data if int(value) < 0),
        'zero_count': sum(1 for value in valid if value == 0),
        'positive_count': len(positives),
        'min_value': min(data) if data else 0,
        'max_value': max(data) if data else 0,
        'mean_positive': (
            sum(positives) / float(len(positives)) if positives else 0.0
        ),
    }


class CachedMjpegStream:
    """One upstream MJPEG reader, many dashboard browser clients.

    The dashboard used to proxy upstream streams per browser connection and
    later polled latest JPEG endpoints.  Both modes create avoidable socket
    churn.  This cache keeps one persistent upstream reader per video source,
    stores the latest encoded JPEG bytes, and fans those bytes out to all
    browser clients without re-encoding.
    """

    def __init__(
        self,
        *,
        name: str,
        url: Callable[[], str],
        logger: Any,
        reconnect_min_sec: float = 0.5,
        reconnect_max_sec: float = 8.0,
    ) -> None:
        self.name = name
        self._url = url
        self._logger = logger
        self._reconnect_min_sec = max(0.1, float(reconnect_min_sec))
        self._reconnect_max_sec = max(self._reconnect_min_sec, float(reconnect_max_sec))
        self._condition = threading.Condition()
        self._thread = threading.Thread(
            target=self._run,
            name=f'dashboard_cached_mjpeg_{name}',
            daemon=True,
        )
        self._started = False
        self._stop = False
        self._jpeg: Optional[bytes] = None
        self._version = 0
        self._last_frame_wall = 0.0
        self._last_error = ''
        self._upstream_connected = False
        self._client_count = 0
        self._client_disconnects = 0
        self._upstream_connect_count = 0
        self._drop_count = 0
        self._frame_times: deque[float] = deque(maxlen=180)
        self._display_times: deque[float] = deque(maxlen=300)
        self._upstream_bytes: deque[tuple[float, int]] = deque(maxlen=300)
        self._display_bytes: deque[tuple[float, int]] = deque(maxlen=600)

    def start(self) -> None:
        if self._started:
            return
        self._started = True
        self._thread.start()

    def stop(self) -> None:
        self._stop = True
        with self._condition:
            self._condition.notify_all()

    def latest_frame(self) -> tuple[int, Optional[bytes]]:
        self.start()
        with self._condition:
            return self._version, self._jpeg

    def metrics(self) -> Dict[str, Any]:
        now = time.time()
        with self._condition:
            jpeg_size = len(self._jpeg) if self._jpeg is not None else 0
            frame_age_ms = (
                (now - self._last_frame_wall) * 1000.0
                if self._last_frame_wall > 0.0 else -1.0
            )
            return {
                'name': self.name,
                'version': int(self._version),
                'capture_fps': self._rate(self._frame_times),
                'encode_fps': 0.0,
                'display_fps': self._rate(self._display_times),
                'jpeg_size_kb': jpeg_size / 1024.0,
                'bitrate_mbps': self._mbps(self._display_bytes, now),
                'upstream_mbps': self._mbps(self._upstream_bytes, now),
                'encode_ms': 0.0,
                'frame_age_ms': frame_age_ms,
                'client_count': int(self._client_count),
                'upstream_connection_count': 1 if self._upstream_connected else 0,
                'upstream_connect_count_total': int(self._upstream_connect_count),
                'drop_count': int(self._drop_count),
                'last_error': self._last_error,
            }

    def generator(self):
        self.start()
        boundary = b'--frame\r\nContent-Type: image/jpeg\r\nCache-Control: no-cache\r\n\r\n'
        version = -1
        with self._condition:
            self._client_count += 1
        try:
            while not self._stop:
                with self._condition:
                    self._condition.wait_for(
                        lambda: self._stop or (
                            self._jpeg is not None and self._version != version
                        ),
                        timeout=2.0,
                    )
                    if self._stop:
                        break
                    if self._jpeg is None or self._version == version:
                        continue
                    frame = self._jpeg
                    version = self._version
                    self._display_times.append(time.time())
                    self._display_bytes.append((time.time(), len(frame)))
                yield boundary + frame + b'\r\n'
        finally:
            with self._condition:
                self._client_count = max(0, self._client_count - 1)
                self._client_disconnects += 1

    def _run(self) -> None:
        backoff = self._reconnect_min_sec
        while not self._stop:
            url = self._url()
            try:
                self._read_upstream(url)
                backoff = self._reconnect_min_sec
            except Exception as exc:  # noqa: BLE001
                with self._condition:
                    self._upstream_connected = False
                    self._last_error = f'{type(exc).__name__}: {exc}'
                    self._condition.notify_all()
                self._logger.warning(
                    'DASHBOARD_VIDEO_UPSTREAM_RECONNECT | '
                    f'name={self.name} url={url} error={exc}',
                    throttle_duration_sec=5.0,
                )
                time.sleep(backoff)
                backoff = min(self._reconnect_max_sec, backoff * 1.7)

    def _read_upstream(self, url: str) -> None:
        with urllib.request.urlopen(url, timeout=3.0) as upstream:
            with self._condition:
                self._upstream_connected = True
                self._upstream_connect_count += 1
                self._last_error = ''
                self._condition.notify_all()
            buffer = b''
            while not self._stop:
                chunk = upstream.read(8192)
                if not chunk:
                    raise RuntimeError('upstream closed')
                buffer += chunk
                frames, buffer = self._extract_jpegs(buffer)
                for frame in frames:
                    self._update_frame(frame)
                if len(buffer) > 2 * 1024 * 1024:
                    self._drop_count += 1
                    buffer = buffer[-256 * 1024:]

    def _update_frame(self, frame: bytes) -> None:
        now = time.time()
        with self._condition:
            if self._jpeg is not None:
                self._drop_count += 1
            self._jpeg = frame
            self._version += 1
            self._last_frame_wall = now
            self._frame_times.append(now)
            self._upstream_bytes.append((now, len(frame)))
            self._condition.notify_all()

    @staticmethod
    def _extract_jpegs(buffer: bytes) -> tuple[list[bytes], bytes]:
        frames: list[bytes] = []
        cursor = 0
        while True:
            start = buffer.find(b'\xff\xd8', cursor)
            if start < 0:
                return frames, buffer[-2:]
            end = buffer.find(b'\xff\xd9', start + 2)
            if end < 0:
                return frames, buffer[start:]
            end += 2
            frames.append(buffer[start:end])
            cursor = end

    @staticmethod
    def _rate(samples: deque[float]) -> float:
        if len(samples) < 2:
            return 0.0
        span = max(1.0e-6, samples[-1] - samples[0])
        return (len(samples) - 1) / span

    @staticmethod
    def _mbps(samples: deque[tuple[float, int]], now: float) -> float:
        cutoff = now - 5.0
        recent = [item for item in samples if item[0] >= cutoff]
        if not recent:
            return 0.0
        span = max(1.0, now - recent[0][0])
        return sum(item[1] for item in recent) * 8.0 / (span * 1_000_000.0)


class LeaderUnifiedDashboard(Node):
    def __init__(self) -> None:
        super().__init__('leader_unified_dashboard')

        self.host = str(_declare(self, 'host', '0.0.0.0'))
        self.port = int(_declare(self, 'port', 8091))
        self.map_topic = str(_declare(self, 'map_topic', '/map'))
        self.risk_topic = str(_declare(self, 'risk_topic', '/risk/risk_map'))
        self.leader_pose_topic = str(_declare(self, 'leader_pose_topic', '/leader_pose'))
        self.follower_pose_topic = str(_declare(self, 'follower_pose_topic', '/burger_pose'))
        self.follower_name = str(_declare(self, 'follower_name', 'follower21'))
        self.member_pose_topic = str(_declare(self, 'member_pose_topic', '/member_pose'))
        self.second_follower_pose_topic = str(
            _declare(self, 'second_follower_pose_topic', self.member_pose_topic)
        )
        self.second_follower_name = str(
            _declare(self, 'second_follower_name', 'scout22')
        )
        self.second_follower_role = str(
            _declare(self, 'second_follower_role', 'scout')
        )
        self.fleet_poses_topic = str(_declare(self, 'fleet_poses_topic', '/fleet/robot_poses'))
        self.fleet_status_topic = str(_declare(self, 'fleet_status_topic', '/fleet/coordination_status'))
        self.collision_warning_topic = str(_declare(self, 'collision_warning_topic', '/fleet/collision_warning'))
        self.leader_nav_path_topic = str(_declare(self, 'leader_nav_path_topic', '/plan'))
        self.leader_bridged_nav_path_topic = str(
            _declare(self, 'leader_bridged_nav_path_topic', '/leader_plan')
        )
        self.follower_nav_path_topic = str(_declare(self, 'follower_nav_path_topic', '/burger_plan'))
        self.member_nav_path_topic = str(_declare(self, 'member_nav_path_topic', '/member_plan'))
        self.omx_waypoint_route_topic = str(
            _declare(self, 'omx_waypoint_route_topic', '/omx/waypoint_route')
        )
        self.omx_debug_port = int(_declare(self, 'omx_debug_port', 8080))
        self.omx_stream_path = str(_declare(self, 'omx_stream_path', '/stream.mjpg'))
        self.omx_state_path = str(_declare(self, 'omx_state_path', '/state.json'))
        self.yolo_server_port = int(_declare(self, 'yolo_server_port', 5005))
        self.yolo_overlay_stream_path = str(
            _declare(self, 'yolo_overlay_stream_path', '/stream/yolo.mjpg')
        )
        self.yolo_status_path = str(_declare(self, 'yolo_status_path', '/api/status'))
        self.video_ready_topic = str(_declare(self, 'video_ready_topic', '/fleet/video_ready'))
        self.start_motion_topic = str(_declare(self, 'start_motion_topic', '/fleet/start_motion'))
        self.start_motion_detail_topic = str(
            _declare(self, 'start_motion_detail_topic', '/fleet/start_motion_detail')
        )
        self.scout_motion_ready_detail_topic = str(
            _declare(
                self,
                'scout_motion_ready_detail_topic',
                '/fleet/scout_motion_ready_detail',
            )
        )
        self.readiness_detail_topic = str(
            _declare(self, 'readiness_detail_topic', '/fleet/readiness_detail')
        )
        self.system_ready_topic = str(_declare(self, 'system_ready_topic', '/system/ready'))
        self.system_readiness_detail_topic = str(
            _declare(self, 'system_readiness_detail_topic', '/system/readiness_detail')
        )
        self.require_scout_video_ready = bool(_declare(self, 'require_scout_video_ready', True))
        self.require_omx_video_ready = bool(_declare(self, 'require_omx_video_ready', True))
        self.video_ready_poll_period_sec = float(
            _declare(self, 'video_ready_poll_period_sec', 1.0)
        )
        self.video_ready_max_age_sec = float(
            _declare(self, 'video_ready_max_age_sec', 5.0)
        )
        self.dashboard_session_timeout_sec = float(
            _declare(self, 'dashboard_session_timeout_sec', 6.0)
        )
        self.dashboard_stable_duration_sec = float(
            _declare(self, 'dashboard_stable_duration_sec', 1.0)
        )
        # Once motion has actually been released, a single missed poll (one
        # late YOLO frame over flaky Wi-Fi, one browser heartbeat gap) must
        # not immediately yank motion authority and stop the robot -- that
        # produced exactly the observed "works, then doesn't, then works
        # again" oscillation. Require readiness to stay down for this long
        # before actually dropping start_motion; a transient blip that
        # recovers within the window never interrupts motion at all.
        self.motion_drop_grace_sec = float(
            _declare(self, 'motion_drop_grace_sec', 4.0)
        )
        self.fleet_registry_json = str(_declare(self, 'fleet_registry_json', ''))
        self.robot_stale_timeout_sec = float(_declare(self, 'robot_stale_timeout_sec', 3.0))
        self.map_stale_timeout_sec = float(_declare(self, 'map_stale_timeout_sec', 30.0))
        self.risk_stale_timeout_sec = float(_declare(self, 'risk_stale_timeout_sec', 10.0))
        self.cmd_vel_topic = str(_declare(self, 'cmd_vel_topic', '/cmd_vel'))
        self.cmd_vel_topic_type = str(
            _declare(self, 'cmd_vel_topic_type', 'geometry_msgs/msg/TwistStamped')
        ).strip()
        self.cmd_vel_nav_topic = str(_declare(self, 'cmd_vel_nav_topic', '/cmd_vel_nav'))
        self.cmd_vel_nav_topic_type = str(
            _declare(self, 'cmd_vel_nav_topic_type', 'geometry_msgs/msg/TwistStamped')
        ).strip()

        self._lock = threading.RLock()
        self._grids: Dict[str, Dict[str, Any]] = {
            'map': self._empty_grid_state(),
            'risk': self._empty_grid_state(),
        }
        self._grid_render_condition = threading.Condition()
        self._grid_render_stop = False
        self._grid_render_thread = threading.Thread(
            target=self._grid_render_loop,
            name='dashboard_grid_png_cache_worker',
            daemon=True,
        )
        self._grid_render_thread.start()
        self._robots: Dict[str, Dict[str, Any]] = {
            'leader': self._empty_robot('leader', 'leader', self.leader_pose_topic),
            self.follower_name: self._empty_robot(
                self.follower_name, 'follower', self.follower_pose_topic
            ),
            self.second_follower_name: self._empty_robot(
                self.second_follower_name,
                self.second_follower_role,
                self.second_follower_pose_topic,
            ),
        }
        self._robot_order = (
            'leader',
            self.follower_name,
            self.second_follower_name,
        )
        self._topic_state: Dict[str, Dict[str, Any]] = {}
        self._nav_paths: Dict[str, Dict[str, Any]] = {
            'leader': self._empty_path_state('leader', self.leader_nav_path_topic),
            'leader_bridge': self._empty_path_state(
                'leader_bridge', self.leader_bridged_nav_path_topic
            ),
            'follower': self._empty_path_state('follower', self.follower_nav_path_topic),
            'member': self._empty_path_state('member', self.member_nav_path_topic),
            'omx_route': self._empty_path_state(
                'omx_route', self.omx_waypoint_route_topic
            ),
        }
        self._omx_state: Dict[str, Any] = {
            'state': None,
            'status': None,
            'target_detected': None,
            'aim_progress': None,
            'queue_size': None,
            'waffle_status': None,
            'waffle_nav_result': None,
            'fire_status': None,
            'fire_disabled': None,
            'aim_error_norm': None,
            'leader_cmd_vel': None,
            'leader_cmd_vel_nav': None,
            'camera_ready': None,
            'observation_status': None,
        }
        self._video_ready = False
        self._start_motion = False
        self._system_ready = False
        self._system_readiness_detail: Dict[str, Any] = {}
        self._scout_motion_ready_detail: Dict[str, Any] = {}
        self._scout_motion_ready_detail_wall_sec: Optional[float] = None
        self._video_ready_detail: Dict[str, Any] = {
            'ready': False,
            'start_motion': False,
            'scout_yolo_ready': False,
            'scout_inference_ready': False,
            'omx_frame_ready': False,
            'published_count': 0,
        }
        self._dashboard_manifest: Dict[str, Any] = {}
        self._dashboard_manifest_wall_sec: Optional[float] = None
        self._dashboard_ui_good_since: Optional[float] = None
        self._motion_not_ready_since: Optional[float] = None
        self._dashboard_ui_ready = False
        self._startup_motion_released = False
        self._events: Dict[str, Dict[str, Any]] = {
            'fire': self._empty_event('/omx/fire', 'std_msgs/msg/Empty'),
            'target_processed': self._empty_event(
                '/omx/target_processed', 'geometry_msgs/msg/PointStamped'
            ),
            'target_lost': self._empty_event(
                '/omx/target_lost', 'geometry_msgs/msg/PointStamped'
            ),
            'target_blocked': self._empty_event(
                '/omx/target_blocked', 'geometry_msgs/msg/PointStamped'
            ),
            'target_not_found': self._empty_event(
                '/omx/target_not_found', 'geometry_msgs/msg/PointStamped'
            ),
            'nav_goal': self._empty_event(
                '/omx/nav_goal', 'geometry_msgs/msg/PoseStamped'
            ),
            'nav_cancel': self._empty_event('/omx/nav_cancel', 'std_msgs/msg/Empty'),
            'patrol_complete': self._empty_event(
                '/omx/patrol_complete', 'std_msgs/msg/Empty'
            ),
        }
        self._video_streams = {
            'scout_yolo': CachedMjpegStream(
                name='scout_yolo',
                url=lambda: f'http://127.0.0.1:{self.yolo_server_port}{self.yolo_overlay_stream_path}',
                logger=self.get_logger(),
            ),
            'omx': CachedMjpegStream(
                name='omx',
                url=lambda: f'http://127.0.0.1:{self.omx_debug_port}{self.omx_stream_path}',
                logger=self.get_logger(),
            ),
        }
        self.get_logger().info(
            'DASHBOARD_VIDEO_LAYOUT | '
            'required_panels=scout_yolo,leader_omx '
            'scout_' + 'raw_enabled=false '
            'browser_video_count=2 '
            'upstream_video_count=2'
        )

        # /map is published by map_relay with both TRANSIENT_LOCAL and
        # VOLATILE writers.  Request VOLATILE here: it is compatible with
        # both writers, while a TRANSIENT_LOCAL subscription silently misses
        # the live volatile Cartographer/map-relay updates.
        grid_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )
        latest_best_effort_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )
        self.create_subscription(OccupancyGrid, self.map_topic, lambda msg: self._on_grid('map', msg), grid_qos)
        self.create_subscription(OccupancyGrid, self.risk_topic, lambda msg: self._on_grid('risk', msg), grid_qos)
        self.create_subscription(PoseStamped, self.leader_pose_topic, lambda msg: self._on_pose('leader', msg, self.leader_pose_topic), 10)
        self.create_subscription(PoseStamped, self.follower_pose_topic, lambda msg: self._on_pose(self.follower_name, msg, self.follower_pose_topic), 10)
        self.create_subscription(PoseStamped, self.second_follower_pose_topic, lambda msg: self._on_pose(self.second_follower_name, msg, self.second_follower_pose_topic), 10)
        self.create_subscription(PoseArray, self.fleet_poses_topic, self._on_fleet_poses, 10)
        self.create_subscription(String, self.fleet_status_topic, self._on_fleet_status, 10)
        self.create_subscription(Bool, self.collision_warning_topic, self._on_collision_warning, 10)
        self.create_subscription(NavPath, self.leader_nav_path_topic, lambda msg: self._on_nav_path('leader', msg), 10)
        self.create_subscription(NavPath, self.leader_bridged_nav_path_topic, lambda msg: self._on_nav_path('leader_bridge', msg), 10)
        self.create_subscription(NavPath, self.follower_nav_path_topic, lambda msg: self._on_nav_path('follower', msg), 10)
        self.create_subscription(NavPath, self.member_nav_path_topic, lambda msg: self._on_nav_path('member', msg), 10)
        self.create_subscription(NavPath, self.omx_waypoint_route_topic, lambda msg: self._on_nav_path('omx_route', msg), 10)
        self.create_subscription(
            String,
            '/risk/yolo_detections',
            lambda msg: self._set_topic_value('/risk/yolo_detections', msg.data, 'std_msgs/msg/String'),
            latest_best_effort_qos,
        )
        self.create_subscription(String, '/omx/state', lambda msg: self._set_omx('state', msg.data, '/omx/state', 'std_msgs/msg/String'), 10)
        self.create_subscription(String, '/omx/status', lambda msg: self._set_omx('status', msg.data, '/omx/status', 'std_msgs/msg/String'), 10)
        self.create_subscription(Bool, '/omx/target_detected', lambda msg: self._set_omx('target_detected', bool(msg.data), '/omx/target_detected', 'std_msgs/msg/Bool'), latest_best_effort_qos)
        self.create_subscription(Bool, '/omx/camera_ready', self._on_omx_camera_ready, 10)
        self.create_subscription(String, '/omx/observation_status', self._on_omx_observation_status, latest_best_effort_qos)
        self.create_subscription(Float32, '/omx/aim_progress', lambda msg: self._set_omx('aim_progress', float(msg.data), '/omx/aim_progress', 'std_msgs/msg/Float32'), 10)
        self.create_subscription(Int32, '/omx/queue_size', lambda msg: self._set_omx('queue_size', int(msg.data), '/omx/queue_size', 'std_msgs/msg/Int32'), 10)
        self.create_subscription(String, '/waffle/status', lambda msg: self._set_omx('waffle_status', msg.data, '/waffle/status', 'std_msgs/msg/String'), 10)
        self.create_subscription(String, '/waffle/nav_result', lambda msg: self._set_omx('waffle_nav_result', msg.data, '/waffle/nav_result', 'std_msgs/msg/String'), 10)
        self.create_subscription(String, '/omx/fire_status', lambda msg: self._set_omx('fire_status', msg.data, '/omx/fire_status', 'std_msgs/msg/String'), 10)
        self.create_subscription(Bool, '/omx/fire_disable', lambda msg: self._set_omx('fire_disabled', bool(msg.data), '/omx/fire_disable', 'std_msgs/msg/Bool'), 10)
        self.create_subscription(Point, '/omx/error_norm', self._on_aim_error, 10)
        if self.cmd_vel_topic_type == 'geometry_msgs/msg/Twist':
            self.create_subscription(Twist, self.cmd_vel_topic, self._on_cmd_vel, 10)
        else:
            self.create_subscription(
                TwistStamped, self.cmd_vel_topic, self._on_cmd_vel_stamped, 10
            )
        if self.cmd_vel_nav_topic_type == 'geometry_msgs/msg/Twist':
            self.create_subscription(Twist, self.cmd_vel_nav_topic, self._on_cmd_vel_nav, 10)
        else:
            self.create_subscription(
                TwistStamped, self.cmd_vel_nav_topic, self._on_cmd_vel_nav_stamped, 10
            )
        self.create_subscription(Empty, '/omx/fire', lambda msg: self._on_empty_event('fire'), 10)
        self.create_subscription(Empty, '/omx/nav_cancel', lambda msg: self._on_empty_event('nav_cancel'), 10)
        self.create_subscription(Empty, '/omx/patrol_complete', lambda msg: self._on_empty_event('patrol_complete'), 10)
        self.create_subscription(PointStamped, '/omx/target_processed', lambda msg: self._on_point_event('target_processed', msg), 10)
        self.create_subscription(PointStamped, '/omx/target_lost', lambda msg: self._on_point_event('target_lost', msg), 10)
        self.create_subscription(PointStamped, '/omx/target_blocked', lambda msg: self._on_point_event('target_blocked', msg), 10)
        self.create_subscription(PointStamped, '/omx/target_not_found', lambda msg: self._on_point_event('target_not_found', msg), 10)
        self.create_subscription(PoseStamped, '/omx/nav_goal', self._on_nav_goal, 10)

        video_ready_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.create_subscription(
            Bool, self.system_ready_topic, self._on_system_ready, video_ready_qos
        )
        self.create_subscription(
            String,
            self.system_readiness_detail_topic,
            self._on_system_readiness_detail,
            video_ready_qos,
        )
        self.create_subscription(
            String,
            self.scout_motion_ready_detail_topic,
            self._on_scout_motion_ready_detail,
            video_ready_qos,
        )
        self.video_ready_pub = self.create_publisher(
            Bool, self.video_ready_topic, video_ready_qos
        )
        self.start_motion_pub = self.create_publisher(
            Bool, self.start_motion_topic, video_ready_qos
        )
        self.readiness_detail_pub = self.create_publisher(
            String, self.readiness_detail_topic, video_ready_qos
        )
        self.start_motion_detail_pub = self.create_publisher(
            String, self.start_motion_detail_topic, video_ready_qos
        )
        self.create_timer(
            max(0.2, self.video_ready_poll_period_sec),
            self._evaluate_video_ready,
        )
        self._publish_video_ready(False, 'startup')
        self._publish_start_motion(False, {'reason': 'startup', 'blocking_reasons': ['startup']})

        self._app = self._build_app()
        self._server_thread = threading.Thread(target=self._run_flask, daemon=True)
        self._server_thread.start()
        self.get_logger().info(f'UNIFIED_DASHBOARD_START | host={self.host} | port={self.port}')

    @staticmethod
    def _empty_grid_state() -> Dict[str, Any]:
        return {
            'msg': None,
            'seq': 0,
            'received_wall_sec': None,
            'png_seq': -1,
            'png': None,
            'metadata': None,
            'stats': None,
            'render_pending_seq': -1,
            'render_in_progress': False,
            'render_error': None,
        }

    @staticmethod
    def _empty_robot(name: str, role: str, topic: str) -> Dict[str, Any]:
        return {
            'name': name,
            'role': role,
            'topic': topic,
            'pose': None,
            'received_wall_sec': None,
            'source': None,
        }

    @staticmethod
    def _empty_event(topic: str, msg_type: str) -> Dict[str, Any]:
        return {
            'topic': topic,
            'type': msg_type,
            'count': 0,
            'received_wall_sec': None,
            'last': None,
        }

    @staticmethod
    def _empty_path_state(name: str, topic: str) -> Dict[str, Any]:
        return {
            'name': name,
            'topic': topic,
            'msg': None,
            'received_wall_sec': None,
            'seq': 0,
        }

    def _resource_paths(self) -> tuple[str, str]:
        try:
            share = Path(get_package_share_directory('system_bringup'))
        except Exception:
            share = Path(__file__).resolve().parents[1]
        return str(share / 'templates'), str(share / 'static')

    def _build_app(self):
        try:
            from flask import Flask, Response, jsonify, render_template, request, stream_with_context
        except ImportError as exc:
            raise RuntimeError('Flask is required for leader_unified_dashboard') from exc

        template_folder, static_folder = self._resource_paths()
        app = Flask(
            __name__,
            template_folder=template_folder,
            static_folder=static_folder,
            static_url_path='/static',
        )
        app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
        asset_version = str(int(time.time()))

        @app.after_request
        def add_no_cache_headers(response):
            if request.path == '/' or request.path.startswith('/static/'):
                response.headers['Cache-Control'] = (
                    'no-store, no-cache, must-revalidate, max-age=0'
                )
                response.headers['Pragma'] = 'no-cache'
                response.headers['Expires'] = '0'
            return response

        @app.get('/')
        def index():
            return render_template('dashboard.html', asset_version=asset_version)

        @app.get('/health')
        def health():
            return jsonify({'status': 'ok'})

        @app.get('/api/status')
        @app.get('/api/state')
        def status():
            response = jsonify(self.snapshot())
            response.headers['Cache-Control'] = 'no-store'
            return response

        @app.post('/api/dashboard_readiness')
        def dashboard_readiness():
            payload = request.get_json(silent=True) or {}
            self._on_dashboard_manifest(payload)
            response = jsonify({'ok': True, 'ui_ready': self._dashboard_ui_ready})
            response.headers['Cache-Control'] = 'no-store'
            return response

        @app.get('/api/robots')
        def robots():
            response = jsonify({'robots': self.snapshot()['robots']})
            response.headers['Cache-Control'] = 'no-store'
            return response

        @app.get('/api/yolo_status')
        def yolo_status():
            response = jsonify(self.yolo_status())
            response.headers['Cache-Control'] = 'no-store'
            return response

        @app.get('/api/video_metrics')
        def video_metrics():
            response = jsonify(self.video_metrics())
            response.headers['Cache-Control'] = 'no-store'
            return response

        @app.get('/api/map.png')
        @app.get('/map.png')
        def map_png():
            return self._grid_response(Response, request, 'map')

        @app.get('/api/risk.png')
        @app.get('/risk.png')
        def risk_png():
            return self._grid_response(Response, request, 'risk')

        @app.get('/api/yolo_stream/<kind>.mjpg')
        def yolo_stream(kind):
            if kind not in ('yolo', 'overlay'):
                return jsonify({'ok': False, 'error': 'kind must be yolo'}), 404

            response = Response(
                stream_with_context(self._video_streams['scout_yolo'].generator()),
                mimetype='multipart/x-mixed-replace; boundary=frame',
            )
            response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
            response.headers['X-Accel-Buffering'] = 'no'
            return response

        @app.get('/api/yolo_frame/<kind>.jpg')
        def yolo_frame(kind):
            if kind in ('yolo', 'overlay'):
                version, frame = self._video_streams['scout_yolo'].latest_frame()
                if frame is None:
                    return jsonify({'ok': False, 'error': 'waiting for first yolo frame'}), 503
                response = Response(frame, mimetype='image/jpeg')
                response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
                response.headers['X-Frame-Version'] = str(version)
                return response
            else:
                return jsonify({'ok': False, 'error': 'kind must be yolo'}), 404

        @app.get('/api/omx_frame.jpg')
        def omx_frame():
            version, frame = self._video_streams['omx'].latest_frame()
            if frame is None:
                return jsonify({'ok': False, 'error': 'waiting for first OMX frame'}), 503
            response = Response(frame, mimetype='image/jpeg')
            response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
            response.headers['X-Frame-Version'] = str(version)
            return response

        @app.get('/api/omx_stream.mjpg')
        def omx_stream():
            response = Response(
                stream_with_context(self._video_streams['omx'].generator()),
                mimetype='multipart/x-mixed-replace; boundary=frame',
            )
            response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
            response.headers['X-Accel-Buffering'] = 'no'
            return response

        return app

    def _grid_response(self, response_class: Any, request_obj: Any, kind: str) -> Any:
        try:
            png = self.grid_png(kind)
        except Exception as exc:
            self.get_logger().error(f'UNIFIED_DASHBOARD_ERROR | route=/{kind}.png | error={exc}')
            return response_class(f'{kind} render error\n', status=500, mimetype='text/plain')
        if png is None:
            return response_class(f'{kind} not available\n', status=404, mimetype='text/plain')
        with self._lock:
            seq = int(self._grids[kind]['png_seq'])
            stats = dict(self._grids[kind].get('stats') or {})
        etag = f'"{kind}-{seq}"'
        if request_obj.headers.get('If-None-Match') == etag:
            response = response_class(status=304)
            response.headers['ETag'] = etag
            response.headers['Cache-Control'] = 'no-cache'
            response.headers['X-Grid-Seq'] = str(seq)
            return response
        response = response_class(png, mimetype='image/png')
        response.headers['Cache-Control'] = 'no-cache'
        response.headers['ETag'] = etag
        response.headers['X-Grid-Seq'] = str(seq)
        response.headers['X-Grid-Positive-Count'] = str(int(stats.get('positive_count', 0) or 0))
        response.headers['X-Grid-Max-Value'] = str(int(stats.get('max_value', 0) or 0))
        response.headers['X-Grid-Bytes'] = str(len(png))
        return response

    def _run_flask(self) -> None:
        logging.getLogger('werkzeug').setLevel(logging.WARNING)
        try:
            self.get_logger().info(
                f'UNIFIED_DASHBOARD_FLASK_BIND | host={self.host} '
                f'| port={self.port} | threaded=true | reloader=false'
            )
            self._app.run(
                host=self.host,
                port=self.port,
                threaded=True,
                use_reloader=False,
                debug=False,
            )
        except Exception as exc:
            self.get_logger().error(f'UNIFIED_DASHBOARD_ERROR | host={self.host} | port={self.port} | error={exc}')

    def _on_grid(self, kind: str, msg: OccupancyGrid) -> None:
        now = time.time()
        with self._lock:
            state = self._grids[kind]
            state['msg'] = msg
            state['seq'] += 1
            state['received_wall_sec'] = now
            state['metadata'] = _grid_signature(msg)
            state['stats'] = _grid_stats(msg)
            state['render_pending_seq'] = int(state['seq'])
            state['render_error'] = None
            state['png'] = None
            state['png_seq'] = -1
            topic = self.map_topic if kind == 'map' else self.risk_topic
            self._touch_topic(topic, now, 'nav_msgs/msg/OccupancyGrid')
        with self._grid_render_condition:
            self._grid_render_condition.notify_all()
        if kind == 'risk':
            stats = _grid_stats(msg)
            source_stamp = _stamp_to_float(msg.header.stamp)
            source_age_ms = (
                max(0.0, (time.time() - source_stamp) * 1000.0)
                if source_stamp > 0.0 else -1.0
            )
            try:
                resolved = self.resolve_topic_name(self.risk_topic)
            except Exception:  # noqa: BLE001
                resolved = self.risk_topic
            self.get_logger().warning(
                'DASHBOARD_RISK_SUBSCRIBER | '
                f'configured_topic={self.risk_topic} '
                f'resolved_topic={resolved} '
                f'publisher_count={self.count_publishers(self.risk_topic)} '
                f'subscription_count={self.count_subscribers(self.risk_topic)} '
                'qos_compatible=true '
                f'callback_count={int(self._grids["risk"]["seq"])} '
                'receive_age_ms=0 '
                f'source_stamp_age_ms={source_age_ms:.1f} '
                f'frame_id={msg.header.frame_id} '
                f'width={int(msg.info.width)} '
                f'height={int(msg.info.height)} '
                f'resolution={float(msg.info.resolution):.6f} '
                f'data_length={int(stats["data_length"])} '
                f'positive_count={int(stats["positive_count"])} '
                f'max_value={int(stats["max_value"])} '
                f'grid_seq={int(self._grids["risk"]["seq"])}',
                throttle_duration_sec=1.0,
            )

    def _on_pose(self, name: str, msg: PoseStamped, source: str) -> None:
        now = time.time()
        with self._lock:
            robot = self._robots[name]
            first_sample = robot['received_wall_sec'] is None
            role = robot['role']
            robot['pose'] = msg
            robot['received_wall_sec'] = now
            robot['source'] = source
            self._touch_topic(source, now, 'geometry_msgs/msg/PoseStamped')
        if first_sample:
            tag = (
                'LEADER_DASHBOARD_POSE_FIRST_RX'
                if name == 'leader'
                else 'DASHBOARD_ROBOT_POSE_FIRST_RX'
            )
            self.get_logger().info(
                f'{tag} | robot={name} role={role} '
                f'topic={source} frame_id={msg.header.frame_id}'
            )

    def _on_fleet_poses(self, msg: PoseArray) -> None:
        now = time.time()
        names = ('leader', self.follower_name)
        with self._lock:
            for index, pose in enumerate(msg.poses[: len(names)]):
                robot = self._robots[names[index]]
                if robot['received_wall_sec'] is not None and now - robot['received_wall_sec'] < 1.0:
                    continue
                pose_msg = PoseStamped()
                pose_msg.header = msg.header
                pose_msg.pose = pose
                robot['pose'] = pose_msg
                robot['received_wall_sec'] = now
                robot['source'] = self.fleet_poses_topic
            self._touch_topic(self.fleet_poses_topic, now, 'geometry_msgs/msg/PoseArray')

    def _on_fleet_status(self, msg: String) -> None:
        now = time.time()
        with self._lock:
            self._topic_state[self.fleet_status_topic] = {
                'value': msg.data,
                'received_wall_sec': now,
                'type': 'std_msgs/msg/String',
            }

    def _on_collision_warning(self, msg: Bool) -> None:
        now = time.time()
        with self._lock:
            self._topic_state[self.collision_warning_topic] = {
                'value': bool(msg.data),
                'received_wall_sec': now,
                'type': 'std_msgs/msg/Bool',
            }

    def _on_omx_camera_ready(self, msg: Bool) -> None:
        self._set_omx(
            'camera_ready',
            bool(msg.data),
            '/omx/camera_ready',
            'std_msgs/msg/Bool',
        )

    def _on_omx_observation_status(self, msg: String) -> None:
        try:
            payload = json.loads(msg.data)
        except (json.JSONDecodeError, TypeError):
            payload = {'raw': msg.data}
        self._set_omx(
            'observation_status',
            payload,
            '/omx/observation_status',
            'std_msgs/msg/String',
        )

    def _publish_video_ready(self, ready: bool, reason: str) -> None:
        msg = Bool()
        msg.data = bool(ready)
        self.video_ready_pub.publish(msg)
        with self._lock:
            self._video_ready_detail['published_count'] = int(
                self._video_ready_detail.get('published_count', 0)
            ) + 1
            self._topic_state[self.video_ready_topic] = {
                'value': bool(ready),
                'received_wall_sec': time.time(),
                'type': 'std_msgs/msg/Bool',
                'reason': reason,
            }

    def _publish_start_motion(self, ready: bool, detail: Dict[str, Any]) -> None:
        msg = Bool()
        msg.data = bool(ready)
        self.start_motion_pub.publish(msg)
        payload = dict(detail)
        payload['start_motion'] = bool(ready)
        payload['topic'] = self.start_motion_topic
        payload.setdefault(
            'phase',
            'RUNNING' if ready else (
                'RUNTIME_STOPPED'
                if self._startup_motion_released
                else 'STARTUP_NOT_RELEASED'
            ),
        )
        payload['authoritative_publisher'] = 'leader_unified_dashboard_startup_coordinator'
        payload['publisher_count'] = self.count_publishers(self.start_motion_topic)
        detail_msg = String(data=json.dumps(payload, sort_keys=True))
        self.readiness_detail_pub.publish(detail_msg)
        self.start_motion_detail_pub.publish(detail_msg)
        with self._lock:
            self._topic_state[self.start_motion_topic] = {
                'value': bool(ready),
                'received_wall_sec': time.time(),
                'type': 'std_msgs/msg/Bool',
                'reason': payload.get('reason', ''),
            }
            self._topic_state[self.readiness_detail_topic] = {
                'value': payload,
                'received_wall_sec': time.time(),
                'type': 'std_msgs/msg/String',
            }
            self._topic_state[self.start_motion_detail_topic] = {
                'value': payload,
                'received_wall_sec': time.time(),
                'type': 'std_msgs/msg/String',
            }

    def _on_system_ready(self, msg: Bool) -> None:
        with self._lock:
            self._system_ready = bool(msg.data)
            self._topic_state[self.system_ready_topic] = {
                'value': self._system_ready,
                'received_wall_sec': time.time(),
                'type': 'std_msgs/msg/Bool',
            }

    def _on_system_readiness_detail(self, msg: String) -> None:
        try:
            payload = json.loads(msg.data)
        except (json.JSONDecodeError, TypeError):
            payload = {'raw': msg.data}
        if isinstance(payload, dict):
            with self._lock:
                self._system_readiness_detail = payload
                self._topic_state[self.system_readiness_detail_topic] = {
                    'value': payload,
                    'received_wall_sec': time.time(),
                    'type': 'std_msgs/msg/String',
                }

    def _on_scout_motion_ready_detail(self, msg: String) -> None:
        try:
            payload = json.loads(msg.data)
        except (json.JSONDecodeError, TypeError):
            payload = {'raw': msg.data}
        if not isinstance(payload, dict):
            return
        now = time.time()
        with self._lock:
            self._scout_motion_ready_detail = payload
            self._scout_motion_ready_detail_wall_sec = now
            self._topic_state[self.scout_motion_ready_detail_topic] = {
                'value': payload,
                'received_wall_sec': now,
                'type': 'std_msgs/msg/String',
            }
        self._evaluate_motion_release(now, dashboard_ready=self._video_ready)

    def _publish_readiness_detail(self, detail: Dict[str, Any]) -> None:
        msg = String(data=json.dumps(detail, sort_keys=True))
        self.readiness_detail_pub.publish(msg)
        self.start_motion_detail_pub.publish(msg)
        with self._lock:
            now = time.time()
            self._topic_state[self.readiness_detail_topic] = {
                'value': detail,
                'received_wall_sec': now,
                'type': 'std_msgs/msg/String',
            }
            self._topic_state[self.start_motion_detail_topic] = {
                'value': detail,
                'received_wall_sec': now,
                'type': 'std_msgs/msg/String',
            }

    @staticmethod
    def _false_reasons(prefix: str, values: Dict[str, Any]) -> list[str]:
        reasons = []
        for key, value in values.items():
            if value is False:
                reasons.append(f'{prefix}_{key}_not_ready')
        return reasons

    @staticmethod
    def _panel_false_reasons(panels: Dict[str, Any]) -> list[str]:
        reasons = []
        for name, panel in panels.items():
            if isinstance(panel, dict) and panel.get('ready') is False:
                reasons.append(f'dashboard_panel_{name}_not_ready')
        return reasons

    def _on_dashboard_manifest(self, payload: Dict[str, Any]) -> None:
        if not isinstance(payload, dict):
            return
        panels = payload.get('panels')
        if not isinstance(panels, dict):
            return
        with self._lock:
            self._dashboard_manifest = payload
            self._dashboard_manifest_wall_sec = time.time()

    def _dashboard_ui_panels_ready(self, now: float) -> tuple[bool, Dict[str, Any]]:
        required = ['map', 'risk_map', 'fleet_state']
        if self.require_scout_video_ready:
            required.append('scout_yolo')
        if self.require_omx_video_ready:
            required.append('leader_omx')
        with self._lock:
            manifest = dict(self._dashboard_manifest)
            manifest_wall = self._dashboard_manifest_wall_sec
        panels = manifest.get('panels') if isinstance(manifest, dict) else {}
        if not isinstance(panels, dict):
            panels = {}
        session_fresh = (
            isinstance(manifest_wall, (int, float))
            and now - float(manifest_wall) <= self.dashboard_session_timeout_sec
        )
        panel_details: Dict[str, Any] = {}
        all_ready = session_fresh
        for name in required:
            panel = panels.get(name, {})
            loaded = bool(isinstance(panel, dict) and panel.get('loaded'))
            rendered = bool(
                isinstance(panel, dict) and panel.get('rendered', loaded)
            )
            placeholder = bool(isinstance(panel, dict) and panel.get('placeholder'))
            version = panel.get('version') if isinstance(panel, dict) else None
            if name == 'fleet_state':
                version_ok = bool(isinstance(panel, dict) and panel.get('robots'))
            elif name == 'risk_map':
                backend_seq = panel.get('backend_seq') if isinstance(panel, dict) else None
                png_seq = panel.get('png_seq') if isinstance(panel, dict) else None
                png_bytes = panel.get('png_bytes') if isinstance(panel, dict) else None
                status = str(panel.get('status', 'NO DATA')) if isinstance(panel, dict) else 'NO DATA'
                grid_received = bool(isinstance(panel, dict) and panel.get('grid_received'))
                version_ok = (
                    isinstance(version, (int, float))
                    and isinstance(backend_seq, (int, float))
                    and isinstance(png_seq, (int, float))
                    and int(version) == int(backend_seq)
                    and int(png_seq) == int(backend_seq)
                    and int(version) > 0
                    and isinstance(png_bytes, (int, float))
                    and int(png_bytes) > 0
                    and grid_received
                    and status in ('EMPTY_RISK_MAP', 'ACTIVE_RISK_MAP')
                )
            else:
                version_ok = isinstance(version, (int, float)) and float(version) > 0
            ok = bool(loaded and rendered and not placeholder and version_ok)
            panel_details[name] = {
                'loaded': loaded,
                'rendered': rendered,
                'placeholder': placeholder,
                'version': version,
                'backend_seq': panel.get('backend_seq') if isinstance(panel, dict) else None,
                'png_seq': panel.get('png_seq') if isinstance(panel, dict) else None,
                'grid_received': panel.get('grid_received') if isinstance(panel, dict) else None,
                'png_bytes': panel.get('png_bytes') if isinstance(panel, dict) else None,
                'positive_count': panel.get('positive_count') if isinstance(panel, dict) else None,
                'max_value': panel.get('max_value') if isinstance(panel, dict) else None,
                'status': panel.get('status') if isinstance(panel, dict) else None,
                'render_error': panel.get('render_error') if isinstance(panel, dict) else None,
                'ready': ok,
            }
            all_ready = all_ready and ok
        if all_ready:
            if self._dashboard_ui_good_since is None:
                self._dashboard_ui_good_since = now
            stable = now - self._dashboard_ui_good_since >= self.dashboard_stable_duration_sec
        else:
            self._dashboard_ui_good_since = None
            stable = False
        detail = {
            'session_id': manifest.get('session_id'),
            'session_fresh': session_fresh,
            'required_panels': required,
            'panels': panel_details,
            'stable_sec': (
                0.0 if self._dashboard_ui_good_since is None
                else now - self._dashboard_ui_good_since
            ),
            'stable_required_sec': self.dashboard_stable_duration_sec,
        }
        return bool(all_ready and stable), detail

    def _scout_motion_release_snapshot(self, now: float) -> tuple[bool, Dict[str, Any]]:
        with self._lock:
            detail = dict(self._scout_motion_ready_detail)
            received = self._scout_motion_ready_detail_wall_sec
        fresh = (
            isinstance(received, (int, float))
            and now - float(received) <= 2.0
        )
        release_ready = bool(fresh and detail.get('ready'))
        blocking = str(detail.get('blocking_reason') or '').strip()
        if not fresh:
            blocking = 'scout_motion_ready_detail_stale'
        elif not blocking:
            blocking = 'none' if release_ready else 'scout_motion_not_ready'
        detail['detail_fresh'] = fresh
        detail['release_ready'] = release_ready
        detail['blocking_reason'] = blocking
        detail['received_age_ms'] = (
            int((now - float(received)) * 1000.0)
            if isinstance(received, (int, float)) else -1
        )
        return release_ready, detail

    def _evaluate_motion_release(
        self,
        now: float,
        *,
        dashboard_ready: bool,
        base_detail: Optional[Dict[str, Any]] = None,
    ) -> bool:
        release_ready, scout_detail = self._scout_motion_release_snapshot(now)
        with self._lock:
            previous_start_motion = self._start_motion
            start_motion = bool(previous_start_motion or release_ready)
            if start_motion:
                self._startup_motion_released = True
            phase = (
                'RUNNING'
                if start_motion else (
                    'RUNTIME_STOPPED'
                    if self._startup_motion_released
                    else 'STARTUP_NOT_RELEASED'
                )
            )
            self._start_motion = start_motion
        blocking_reason = (
            'none' if start_motion else scout_detail.get('blocking_reason', 'unknown')
        )
        stable_elapsed_ms = int(scout_detail.get('stable_elapsed_ms', 0) or 0)
        release_detail = {
            'reason': (
                'minimum_scout_runtime_ready'
                if start_motion else 'minimum_scout_runtime_not_ready'
            ),
            'phase': phase,
            'start_motion': start_motion,
            'scout_motion_ready': release_ready,
            'scout_motion_ready_detail': scout_detail,
            'stable_elapsed_ms': stable_elapsed_ms,
            'blocking_reason': blocking_reason,
            'dashboard_ready': bool(dashboard_ready),
            'system_ready': self._system_ready,
        }
        if base_detail is not None:
            base_detail.update(release_detail)
        self.get_logger().warning(
            'MOTION_RELEASE_DEBUG | '
            f'state={phase} '
            f'start_motion={start_motion} '
            f'scan_ready={scout_detail.get("scan_ready")} '
            f'odom_ready={scout_detail.get("odom_ready")} '
            f'map_ready={scout_detail.get("map_ready")} '
            f'tf_ready={scout_detail.get("tf_ready")} '
            f'model_ready={scout_detail.get("model_ready")} '
            f'observation_ready={scout_detail.get("observation_ready")} '
            f'dashboard_ready={bool(dashboard_ready)} '
            f'leader_ready={self._system_readiness_detail.get("leader_localization")} '
            f'follower_ready={self._system_readiness_detail.get("follower_localization")} '
            f'stable_elapsed_ms={stable_elapsed_ms} '
            f'blocking_reason={blocking_reason}',
            throttle_duration_sec=1.0,
        )
        if start_motion and not previous_start_motion:
            self._publish_start_motion(start_motion, release_detail)
            self.get_logger().warning(
                'MOTION_RELEASED | '
                'reason=minimum_scout_runtime_ready '
                f'stable_elapsed_ms={stable_elapsed_ms}'
            )
        return start_motion

    def _evaluate_video_ready(self) -> None:
        yolo = self.yolo_status()
        data = yolo.get('data') if isinstance(yolo, dict) else None
        if not isinstance(data, dict):
            data = {}
        yolo_age = float(data.get('yolo_frame_age_sec', -1.0) or -1.0)
        inference_age = float(data.get('inference_frame_age_sec', -1.0) or -1.0)
        scout_yolo_ready = (
            int(data.get('yolo_frames', 0) or 0) > 0
            and 0.0 <= yolo_age <= self.video_ready_max_age_sec
        )
        scout_inference_ready = (
            int(data.get('inference_frames', 0) or 0) > 0
            and 0.0 <= inference_age <= self.video_ready_max_age_sec
        )
        with self._lock:
            observation = self._omx_state.get('observation_status')
            camera_ready = bool(self._omx_state.get('camera_ready'))
            observation_wall = self._omx_state.get(
                'observation_status_received_wall_sec'
            )
        now = time.time()
        observation_fresh = (
            isinstance(observation_wall, (int, float))
            and now - float(observation_wall) <= self.video_ready_max_age_sec
        )
        omx_frame_ready = False
        if isinstance(observation, dict):
            omx_frame_ready = bool(
                observation.get('camera_ready')
                and observation.get('frame_valid')
                and observation.get('inference_ran')
                and observation_fresh
            )
        elif not self.require_omx_video_ready:
            omx_frame_ready = True
        if self.require_omx_video_ready and not omx_frame_ready:
            omx_frame_ready = camera_ready and bool(observation) and observation_fresh
        stream_metrics = self.video_metrics()
        scout_stream = stream_metrics.get('scout_yolo', {})
        omx_stream = stream_metrics.get('omx', {})
        scout_stream_ready = (
            scout_stream.get('upstream_connection_count') == 1
            and 0.0 <= float(scout_stream.get('frame_age_ms', -1.0)) <= self.video_ready_max_age_sec * 1000.0
        )
        omx_stream_ready = (
            omx_stream.get('upstream_connection_count') == 1
            and 0.0 <= float(omx_stream.get('frame_age_ms', -1.0)) <= self.video_ready_max_age_sec * 1000.0
        )
        scout_ready = (
            scout_yolo_ready and scout_inference_ready and scout_stream_ready
            if self.require_scout_video_ready else True
        )
        omx_ready = (omx_frame_ready and omx_stream_ready) if self.require_omx_video_ready else True
        backend_ready = bool(scout_ready and omx_ready)
        ui_ready, ui_detail = self._dashboard_ui_panels_ready(now)
        ready = bool(backend_ready and ui_ready)
        dashboard_status = {
            'backend_ready': backend_ready,
            'ui_ready': ui_ready,
            'session_fresh': bool(ui_detail.get('session_fresh')),
            'stable_sec': float(ui_detail.get('stable_sec', 0.0) or 0.0),
            'stable_required_sec': self.dashboard_stable_duration_sec,
            'panels': ui_detail.get('panels', {}),
        }
        with self._lock:
            system_detail = dict(self._system_readiness_detail)
            system_ready_snapshot = self._system_ready
        infrastructure = {
            'system_ready': system_ready_snapshot,
            'leader_localization': system_detail.get('leader_localization'),
            'follower_localization': system_detail.get('follower_localization'),
            'leader_nav2': system_detail.get('leader_nav2'),
            'follower_nav2': system_detail.get('follower_nav2'),
            'map_tf': system_detail.get('map_tf'),
        }
        blocking_reasons = []
        blocking_reasons.extend(self._false_reasons('infrastructure', infrastructure))
        blocking_reasons.extend(self._false_reasons('dashboard', {
            'backend': backend_ready,
            'ui': ui_ready,
            'session': bool(ui_detail.get('session_fresh')),
        }))
        panels = ui_detail.get('panels', {})
        if isinstance(panels, dict):
            blocking_reasons.extend(self._panel_false_reasons(panels))
        if isinstance(system_detail.get('blocking_reasons'), list):
            blocking_reasons.extend(
                f'infrastructure_{reason}'
                for reason in system_detail['blocking_reasons']
            )
        detail = {
            'ready': ready,
            'backend_ready': backend_ready,
            'dashboard_ui_ready': ui_ready,
            'dashboard': dashboard_status,
            'dashboard_ui': ui_detail,
            'infrastructure': infrastructure,
            'system_readiness_detail': system_detail,
            'blocking_reasons': sorted(set(str(item) for item in blocking_reasons)),
            'topic': self.video_ready_topic,
            'scout_required': self.require_scout_video_ready,
            'omx_required': self.require_omx_video_ready,
            'scout_yolo_ready': scout_yolo_ready,
            'scout_inference_ready': scout_inference_ready,
            'scout_stream_ready': scout_stream_ready,
            'omx_frame_ready': omx_frame_ready,
            'omx_stream_ready': omx_stream_ready,
            'yolo_frame_age_sec': yolo_age,
            'inference_frame_age_sec': inference_age,
            'omx_observation_fresh': observation_fresh,
            'video_streams': stream_metrics,
            'video_ready_max_age_sec': self.video_ready_max_age_sec,
            'yolo_status': yolo.get('status'),
            'yolo_error': yolo.get('error'),
        }
        with self._lock:
            previous = self._video_ready
            self._video_ready = ready
            published_count = int(self._video_ready_detail.get('published_count', 0))
            detail['published_count'] = published_count
            detail['system_ready'] = system_ready_snapshot
            detail['system_ready_topic'] = self.system_ready_topic
            detail['start_motion_topic'] = self.start_motion_topic
            detail['authoritative_publisher'] = (
                'leader_unified_dashboard_startup_coordinator'
            )
            detail['publisher_count'] = self.count_publishers(self.start_motion_topic)
            self._video_ready_detail = detail
            self._dashboard_ui_ready = ui_ready
        start_motion = self._evaluate_motion_release(
            now,
            dashboard_ready=ready,
            base_detail=detail,
        )
        if (
            not start_motion
            and detail.get('blocking_reason')
            and detail.get('blocking_reason') != 'none'
        ):
            reasons = list(detail.get('blocking_reasons', []))
            reasons.append(str(detail['blocking_reason']))
            detail['blocking_reasons'] = sorted(set(reasons))
        phase = str(detail.get('phase', 'RUNNING' if start_motion else 'STARTUP_NOT_RELEASED'))
        self._publish_readiness_detail(detail)
        panel_state = detail.get('dashboard_ui', {}).get('panels', {})
        blocking_panels = []
        if isinstance(panel_state, dict):
            blocking_panels = [
                name
                for name, panel in panel_state.items()
                if isinstance(panel, dict) and not bool(panel.get('ready'))
            ]
        self.get_logger().warning(
            'STARTUP_COORDINATOR | '
            f'phase={phase} '
            f'scout_yolo_rendered={panel_state.get("scout_yolo", {}).get("rendered") if isinstance(panel_state.get("scout_yolo"), dict) else None} '
            f'leader_omx_rendered={panel_state.get("leader_omx", {}).get("rendered") if isinstance(panel_state.get("leader_omx"), dict) else None} '
            f'map_rendered={panel_state.get("map", {}).get("rendered") if isinstance(panel_state.get("map"), dict) else None} '
            f'risk_map_rendered={panel_state.get("risk_map", {}).get("rendered") if isinstance(panel_state.get("risk_map"), dict) else None} '
            f'risk_map_has_positive_evidence={int(panel_state.get("risk_map", {}).get("positive_count") or 0) > 0 if isinstance(panel_state.get("risk_map"), dict) else None} '
            f'risk_map_status={panel_state.get("risk_map", {}).get("status") if isinstance(panel_state.get("risk_map"), dict) else None} '
            f'fleet_state_rendered={panel_state.get("fleet_state", {}).get("rendered") if isinstance(panel_state.get("fleet_state"), dict) else None} '
            f'browser_heartbeat={bool(ui_detail.get("session_fresh"))} '
            f'leader_localization_ready={infrastructure.get("leader_localization")} '
            f'stable_ready_sec={float(ui_detail.get("stable_sec", 0.0) or 0.0):.2f} '
            f'start_motion={start_motion} '
            f'publisher_count={self.count_publishers(self.start_motion_topic)} '
            f'blocking_reasons={detail.get("blocking_reasons", [])}',
            throttle_duration_sec=1.0,
        )
        if ready != previous:
            self._publish_video_ready(ready, 'all_video_ready' if ready else 'video_not_ready')
            self.get_logger().warning(
                'FLEET_VIDEO_READY | '
                f'ready={ready} scout_yolo={scout_yolo_ready} '
                f'scout_inference={scout_inference_ready} '
                f'scout_stream={scout_stream_ready} '
                f'omx_frame={omx_frame_ready} omx_stream={omx_stream_ready} ui={ui_ready}'
            )
        if not ready:
            self.get_logger().warning(
                'DASHBOARD_READINESS_DETAIL | '
                f'scout_yolo={scout_ready} '
                f'leader_omx={omx_ready} '
                f'map={panel_state.get("map", {}).get("ready") if isinstance(panel_state.get("map"), dict) else None} '
                f'risk_map={panel_state.get("risk_map", {}).get("ready") if isinstance(panel_state.get("risk_map"), dict) else None} '
                f'fleet_state={panel_state.get("fleet_state", {}).get("ready") if isinstance(panel_state.get("fleet_state"), dict) else None} '
                f'browser_heartbeat={bool(ui_detail.get("session_fresh"))} '
                f'blocking_panels={blocking_panels}',
                throttle_duration_sec=2.0,
            )
    def _on_nav_path(self, name: str, msg: NavPath) -> None:
        now = time.time()
        with self._lock:
            path = self._nav_paths[name]
            first_sample = path['received_wall_sec'] is None
            path['msg'] = msg
            path['received_wall_sec'] = now
            path['seq'] += 1
            self._touch_topic(path['topic'], now, 'nav_msgs/msg/Path')
        if first_sample:
            self.get_logger().info(
                'DASHBOARD_NAV_PATH_FIRST_RX | '
                f'name={name} topic={self._nav_paths[name]["topic"]} '
                f'frame_id={msg.header.frame_id} poses={len(msg.poses)}'
            )

    def _set_omx(
        self,
        key: str,
        value: Any,
        topic_name: Optional[str] = None,
        msg_type: Optional[str] = None,
    ) -> None:
        now = time.time()
        with self._lock:
            self._omx_state[key] = value
            self._omx_state[f'{key}_received_wall_sec'] = now
            if topic_name and msg_type:
                self._topic_state[topic_name] = {
                    'value': value,
                    'received_wall_sec': now,
                    'type': msg_type,
                }

    def _on_aim_error(self, msg: Point) -> None:
        value = {
            'x': float(msg.x),
            'y': float(msg.y),
            'magnitude': math.hypot(float(msg.x), float(msg.y)),
        }
        self._set_omx(
            'aim_error_norm',
            value,
            '/omx/error_norm',
            'geometry_msgs/msg/Point',
        )

    def _cmd_vel_value(self, msg: Twist) -> Dict[str, float]:
        return {
            'linear_x': float(msg.linear.x),
            'linear_y': float(msg.linear.y),
            'angular_z': float(msg.angular.z),
        }

    def _on_cmd_vel(self, msg: Twist) -> None:
        value = self._cmd_vel_value(msg)
        self._set_omx('leader_cmd_vel', value, '/cmd_vel', 'geometry_msgs/msg/Twist')

    def _on_cmd_vel_stamped(self, msg: TwistStamped) -> None:
        value = self._cmd_vel_value(msg.twist)
        self._set_omx(
            'leader_cmd_vel',
            value,
            '/cmd_vel',
            'geometry_msgs/msg/TwistStamped',
        )

    def _on_cmd_vel_nav_stamped(self, msg: TwistStamped) -> None:
        value = self._cmd_vel_value(msg.twist)
        self._set_omx(
            'leader_cmd_vel_nav',
            value,
            '/cmd_vel_nav',
            'geometry_msgs/msg/TwistStamped',
        )

    def _on_cmd_vel_nav(self, msg: Twist) -> None:
        value = self._cmd_vel_value(msg)
        self._set_omx(
            'leader_cmd_vel_nav',
            value,
            self.cmd_vel_nav_topic,
            'geometry_msgs/msg/Twist',
        )

    def _on_empty_event(self, key: str) -> None:
        now = time.time()
        with self._lock:
            event = self._events[key]
            event['count'] += 1
            event['received_wall_sec'] = now
            self._touch_topic(event['topic'], now, event['type'])

    def _on_point_event(self, key: str, msg: PointStamped) -> None:
        now = time.time()
        value = {
            'frame_id': msg.header.frame_id,
            'stamp_sec': _stamp_to_float(msg.header.stamp),
            'x': float(msg.point.x),
            'y': float(msg.point.y),
            'z': float(msg.point.z),
        }
        with self._lock:
            event = self._events[key]
            event['count'] += 1
            event['received_wall_sec'] = now
            event['last'] = value
            self._topic_state[event['topic']] = {
                'value': value,
                'received_wall_sec': now,
                'type': event['type'],
            }

    def _on_nav_goal(self, msg: PoseStamped) -> None:
        now = time.time()
        yaw = _quaternion_to_yaw(msg.pose.orientation)
        value = {
            'frame_id': msg.header.frame_id,
            'stamp_sec': _stamp_to_float(msg.header.stamp),
            'x': float(msg.pose.position.x),
            'y': float(msg.pose.position.y),
            'z': float(msg.pose.position.z),
            'yaw_rad': yaw,
            'yaw_deg': math.degrees(yaw),
        }
        with self._lock:
            event = self._events['nav_goal']
            event['count'] += 1
            event['received_wall_sec'] = now
            event['last'] = value
            self._topic_state[event['topic']] = {
                'value': value,
                'received_wall_sec': now,
                'type': event['type'],
            }

    def _touch_topic(self, topic_name: str, now: float, msg_type: str) -> None:
        self._topic_state[topic_name] = {
            'received_wall_sec': now,
            'type': msg_type,
        }

    def _set_topic_value(self, topic_name: str, value: Any, msg_type: str) -> None:
        now = time.time()
        with self._lock:
            self._topic_state[topic_name] = {
                'value': value,
                'received_wall_sec': now,
                'type': msg_type,
            }

    def snapshot(self) -> Dict[str, Any]:
        now = time.time()
        with self._lock:
            map_meta = self._grids['map']['metadata']
            risk_meta = self._grids['risk']['metadata']
            map_seq = int(self._grids['map']['seq'])
            risk_seq = int(self._grids['risk']['seq'])
            metadata_matches = _metadata_match(map_meta, risk_meta)
        if map_meta and risk_meta:
            origin_delta_x = float(risk_meta['origin']['x']) - float(map_meta['origin']['x'])
            origin_delta_y = float(risk_meta['origin']['y']) - float(map_meta['origin']['y'])
            origin_delta_yaw = float(risk_meta['origin']['yaw']) - float(map_meta['origin']['yaw'])
            self.get_logger().warning(
                'RISK_MAP_ALIGNMENT | '
                f'map_seq={map_seq} '
                f'risk_seq={risk_seq} '
                f'map_frame={map_meta["frame_id"]} '
                f'risk_frame={risk_meta["frame_id"]} '
                f'map_size={map_meta["width"]}x{map_meta["height"]} '
                f'risk_size={risk_meta["width"]}x{risk_meta["height"]} '
                f'map_resolution={float(map_meta["resolution"]):.6f} '
                f'risk_resolution={float(risk_meta["resolution"]):.6f} '
                f'origin_delta_x={origin_delta_x:.6f} '
                f'origin_delta_y={origin_delta_y:.6f} '
                f'origin_delta_yaw={origin_delta_yaw:.6f} '
                f'metadata_match={metadata_matches} '
                f'action={"direct_overlay" if metadata_matches else "wait"}',
                throttle_duration_sec=2.0,
            )
        with self._lock:
            return {
                'server_time_sec': now,
                'omx_debug': {
                    'port': self.omx_debug_port,
                    'stream_path': self.omx_stream_path,
                    'state_path': self.omx_state_path,
                },
                'yolo_server': {
                    'port': self.yolo_server_port,
                    'overlay_stream_path': self.yolo_overlay_stream_path,
                    'overlay_proxy_path': '/api/yolo_stream/yolo.mjpg',
                    'status_path': self.yolo_status_path,
                },
                'video_streams': self.video_metrics(),
                'omx': dict(self._omx_state),
                'events': self._events_summary(now),
                'map': self._grid_summary('map', now, self.map_stale_timeout_sec),
                'risk': {
                    **self._grid_summary('risk', now, self.risk_stale_timeout_sec),
                    'metadata_matches_map': metadata_matches,
                },
                'robots': [
                    self._robot_summary(name, now)
                    for name in self._robot_order
                ],
                'fleet': {
                    'coordination_status': self._topic_value(self.fleet_status_topic, now),
                    'collision_warning': self._topic_value(self.collision_warning_topic, now),
                    'video_ready': dict(self._video_ready_detail),
                    'start_motion': bool(self._start_motion),
                    'system_ready': bool(self._system_ready),
                },
                'nav2_paths': [
                    self._nav_path_summary(name, now)
                    for name in (
                        'leader', 'leader_bridge', 'follower', 'member',
                        'omx_route',
                    )
                ],
                'topics': self._topics_summary(now),
            }

    def video_metrics(self) -> Dict[str, Any]:
        return {
            name: stream.metrics()
            for name, stream in self._video_streams.items()
        }

    def yolo_status(self) -> Dict[str, Any]:
        url = f'http://127.0.0.1:{self.yolo_server_port}{self.yolo_status_path}'
        started = time.time()
        try:
            with urllib.request.urlopen(url, timeout=0.35) as response:
                raw = response.read(256 * 1024)
            payload = json.loads(raw.decode('utf-8'))
            return {
                'status': 'OK',
                'url': url,
                'latency_ms': (time.time() - started) * 1000.0,
                'data': payload,
            }
        except (OSError, urllib.error.URLError, json.JSONDecodeError) as exc:
            return {
                'status': 'NO DATA',
                'url': url,
                'latency_ms': (time.time() - started) * 1000.0,
                'error': str(exc),
                'data': None,
            }

    def _grid_summary(self, kind: str, now: float, stale_timeout: float) -> Dict[str, Any]:
        state = self._grids[kind]
        age = None
        status = 'NO DATA'
        if state['received_wall_sec'] is not None:
            age = max(0.0, now - state['received_wall_sec'])
            status = 'STALE' if age > stale_timeout else 'OK'
        msg = state['msg']
        topic = self.map_topic if kind == 'map' else self.risk_topic
        stats = state.get('stats') or {}
        publisher_count = self.count_publishers(topic)
        if kind == 'risk':
            if publisher_count <= 0 and msg is None:
                status = 'NO_TOPIC'
            elif msg is None:
                status = 'WAITING_FIRST_GRID'
            elif age is not None and age > stale_timeout:
                status = 'STALE_RISK_MAP'
            elif int(stats.get('positive_count', 0) or 0) > 0:
                status = 'ACTIVE_RISK_MAP'
            else:
                status = 'EMPTY_RISK_MAP'
        png = state.get('png')
        return {
            'topic': topic,
            'status': status,
            'age_sec': age,
            'seq': int(state['seq']),
            'metadata': state['metadata'],
            'stamp_sec': _stamp_to_float(msg.header.stamp) if msg is not None else None,
            'grid_received': msg is not None,
            'callback_count': int(state['seq']),
            'publisher_count': publisher_count,
            'subscriber_count': self.count_subscribers(topic),
            'png_seq': int(state.get('png_seq', -1)),
            'png_bytes': len(png) if isinstance(png, (bytes, bytearray)) else 0,
            'render_pending_seq': int(state.get('render_pending_seq', -1)),
            'render_in_progress': bool(state.get('render_in_progress', False)),
            'render_error': state.get('render_error'),
            **stats,
        }

    def _robot_summary(self, name: str, now: float) -> Dict[str, Any]:
        robot = self._robots[name]
        pose = robot['pose']
        age = None
        status = 'NO DATA'
        if robot['received_wall_sec'] is not None:
            age = max(0.0, now - robot['received_wall_sec'])
            status = 'STALE' if age > self.robot_stale_timeout_sec else 'ONLINE'
        yaw = None
        position = None
        frame_id = None
        if pose is not None:
            yaw = _quaternion_to_yaw(pose.pose.orientation)
            position = {
                'x': float(pose.pose.position.x),
                'y': float(pose.pose.position.y),
                'z': float(pose.pose.position.z),
            }
            frame_id = pose.header.frame_id
        return {
            'name': robot['name'],
            'role': robot['role'],
            'topic': robot['topic'],
            'source': robot['source'] or robot['topic'],
            'status': status,
            'age_sec': age,
            'frame_id': frame_id,
            'position': position,
            'yaw_rad': yaw,
            'yaw_deg': math.degrees(yaw) if yaw is not None else None,
        }

    def _topic_value(self, topic_name: str, now: float) -> Dict[str, Any]:
        state = self._topic_state.get(topic_name)
        if state is None:
            return {'topic': topic_name, 'status': 'NO DATA', 'age_sec': None, 'value': None}
        age = max(0.0, now - state['received_wall_sec'])
        return {
            'topic': topic_name,
            'status': 'STALE' if age > self.robot_stale_timeout_sec else 'OK',
            'age_sec': age,
            'value': state.get('value'),
        }

    def _nav_path_summary(self, name: str, now: float) -> Dict[str, Any]:
        path = self._nav_paths[name]
        msg = path['msg']
        age = None
        status = 'NO DATA'
        if path['received_wall_sec'] is not None:
            age = max(0.0, now - path['received_wall_sec'])
            status = 'STALE' if age > 5.0 else 'OK'
        points = []
        frame_id = None
        stamp_sec = None
        if msg is not None:
            frame_id = msg.header.frame_id
            stamp_sec = _stamp_to_float(msg.header.stamp)
            stride = max(1, len(msg.poses) // 240)
            for pose in msg.poses[::stride]:
                points.append({
                    'x': float(pose.pose.position.x),
                    'y': float(pose.pose.position.y),
                    'z': float(pose.pose.position.z),
                })
            if msg.poses and points[-1] != {
                'x': float(msg.poses[-1].pose.position.x),
                'y': float(msg.poses[-1].pose.position.y),
                'z': float(msg.poses[-1].pose.position.z),
            }:
                points.append({
                    'x': float(msg.poses[-1].pose.position.x),
                    'y': float(msg.poses[-1].pose.position.y),
                    'z': float(msg.poses[-1].pose.position.z),
                })
        return {
            'name': path['name'],
            'topic': path['topic'],
            'status': status,
            'age_sec': age,
            'seq': int(path['seq']),
            'frame_id': frame_id,
            'stamp_sec': stamp_sec,
            'pose_count': len(msg.poses) if msg is not None else 0,
            'points': points,
            'start': points[0] if points else None,
            'end': points[-1] if points else None,
        }

    def _events_summary(self, now: float) -> Dict[str, Dict[str, Any]]:
        events = {}
        for name, event in self._events.items():
            age = None
            status = 'NO DATA'
            if event['received_wall_sec'] is not None:
                age = max(0.0, now - event['received_wall_sec'])
                status = 'STALE' if age > 10.0 else 'OK'
            events[name] = {
                'topic': event['topic'],
                'type': event['type'],
                'status': status,
                'age_sec': age,
                'count': int(event['count']),
                'last': event['last'],
            }
        return events

    def _topics_summary(self, now: float) -> Dict[str, Dict[str, Any]]:
        topics = {}
        for topic_name, state in self._topic_state.items():
            age = max(0.0, now - state['received_wall_sec'])
            topics[topic_name] = {
                'type': state.get('type'),
                'age_sec': age,
                'status': 'STALE' if age > 5.0 else 'OK',
                'value': state.get('value'),
            }
        return topics

    def grid_png(self, kind: str) -> Optional[bytes]:
        with self._lock:
            state = self._grids[kind]
            if state['msg'] is None:
                return None
            if state['png'] is None or int(state['png_seq']) != int(state['seq']):
                state['render_pending_seq'] = int(state['seq'])
                with self._grid_render_condition:
                    self._grid_render_condition.notify_all()
            return state['png']

    def _grid_render_loop(self) -> None:
        while not getattr(self, '_grid_render_stop', False):
            job = None
            with self._lock:
                for kind, state in self._grids.items():
                    pending = int(state.get('render_pending_seq', -1))
                    if (
                        state.get('msg') is not None
                        and pending > int(state.get('png_seq', -1))
                        and not state.get('render_in_progress', False)
                    ):
                        msg = state['msg']
                        state['render_in_progress'] = True
                        job = (
                            kind,
                            pending,
                            int(msg.info.width),
                            int(msg.info.height),
                            list(msg.data),
                        )
                        break
            if job is None:
                with self._grid_render_condition:
                    self._grid_render_condition.wait(timeout=0.5)
                continue
            kind, seq, width, height, data = job
            started = time.monotonic()
            try:
                png = _encode_grid_png(data, width, height, kind)
                elapsed_ms = (time.monotonic() - started) * 1000.0
                with self._lock:
                    state = self._grids[kind]
                    if int(state.get('seq', -1)) == seq:
                        state['png'] = png
                        state['png_seq'] = seq
                        state['render_pending_seq'] = -1
                        state['render_error'] = None
                    state['render_in_progress'] = False
                log_name = (
                    'DASHBOARD_RISK_RENDER_BACKEND'
                    if kind == 'risk' else 'DASHBOARD_GRID_PNG_CACHE'
                )
                self.get_logger().warning(
                    f'{log_name} | '
                    f'kind={kind} '
                    f'grid_seq={seq} '
                    f'png_seq={seq} '
                    'render_pending_seq=-1 '
                    'render_in_progress=false '
                    'grid_received_age_ms=0 '
                    f'render_duration_ms={elapsed_ms:.1f} '
                    f'png_bytes={len(png)} '
                    'success=true '
                    'error=none',
                    throttle_duration_sec=1.0 if kind == 'risk' else 5.0,
                )
            except Exception as exc:  # noqa: BLE001
                with self._lock:
                    self._grids[kind]['render_in_progress'] = False
                    self._grids[kind]['render_error'] = str(exc)
                self.get_logger().warning(
                    f'DASHBOARD_RISK_RENDER_BACKEND | kind={kind} '
                    f'grid_seq={seq} png_seq=-1 render_pending_seq={seq} '
                    'render_in_progress=false grid_received_age_ms=0 '
                    'render_duration_ms=0.0 png_bytes=0 success=false '
                    f'error={exc}',
                    throttle_duration_sec=5.0,
                )

    def destroy_node(self) -> None:
        self._grid_render_stop = True
        with self._grid_render_condition:
            self._grid_render_condition.notify_all()
        grid_thread = getattr(self, '_grid_render_thread', None)
        if grid_thread is not None and grid_thread.is_alive():
            grid_thread.join(timeout=1.0)
        for stream in getattr(self, '_video_streams', {}).values():
            stream.stop()
        super().destroy_node()


def _encode_grid_png(data: list, width: int, height: int, kind: str) -> bytes:
    import cv2
    import numpy as np

    if width <= 0 or height <= 0 or len(data) < width * height:
        raise ValueError(f'invalid {kind} grid dimensions/data')

    grid = np.asarray(data[: width * height], dtype=np.int16).reshape((height, width))
    # Cartographer grids grow with explored area.  Encoding their full native
    # resolution on every dashboard refresh can take longer than the next map
    # update, so a browser never receives a completed PNG.  Downsample only
    # the presentation image; world metadata remains full-resolution and the
    # canvas scales this image back over the correct map extent.
    max_render_dimension = 1600
    stride = max(1, int(math.ceil(max(width, height) / max_render_dimension)))
    if stride > 1:
        grid = grid[::stride, ::stride]
        height, width = grid.shape
    grid = np.flipud(grid)

    if kind == 'risk':
        clipped = np.clip(grid, 0, 100).astype(np.float32)
        t = clipped / 100.0
        img = np.zeros((height, width, 4), dtype=np.uint8)
        img[..., 0] = 0
        img[..., 1] = np.clip(255.0 * (1.0 - t), 0, 255).astype(np.uint8)
        img[..., 2] = 255
        img[..., 3] = np.where(clipped > 0, np.clip(225.0 * t, 0, 225), 0).astype(np.uint8)
    else:
        img = np.zeros((height, width, 3), dtype=np.uint8)
        unknown = grid < 0
        occupied = grid >= 65
        known = ~unknown & ~occupied
        shade = np.clip(245 - (np.clip(grid, 0, 100) * 1.8), 35, 245).astype(np.uint8)
        # Keep unexplored space visibly distinct from the dashboard's black
        # canvas, otherwise a valid all-unknown fresh map looks like no map.
        img[unknown] = (96, 91, 82)
        img[known] = np.stack([shade[known], shade[known], shade[known]], axis=-1)
        img[occupied] = (38, 38, 38)

    ok, encoded = cv2.imencode('.png', img)
    if not ok:
        raise RuntimeError(f'failed to encode {kind} grid')
    return encoded.tobytes()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = LeaderUnifiedDashboard()
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
