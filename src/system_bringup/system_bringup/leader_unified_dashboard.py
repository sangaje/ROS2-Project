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
from pathlib import Path
from typing import Any, Dict, Optional

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
        self.yolo_raw_stream_path = str(
            _declare(self, 'yolo_raw_stream_path', '/stream/raw.mjpg')
        )
        self.yolo_overlay_stream_path = str(
            _declare(self, 'yolo_overlay_stream_path', '/stream/yolo.mjpg')
        )
        self.yolo_status_path = str(_declare(self, 'yolo_status_path', '/api/status'))
        self.video_ready_topic = str(_declare(self, 'video_ready_topic', '/fleet/video_ready'))
        self.start_motion_topic = str(_declare(self, 'start_motion_topic', '/fleet/start_motion'))
        self.start_motion_detail_topic = str(
            _declare(self, 'start_motion_detail_topic', '/fleet/start_motion_detail')
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
            _declare(self, 'video_ready_max_age_sec', 3.0)
        )
        self.dashboard_session_timeout_sec = float(
            _declare(self, 'dashboard_session_timeout_sec', 3.0)
        )
        self.dashboard_stable_duration_sec = float(
            _declare(self, 'dashboard_stable_duration_sec', 2.0)
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
        self._video_ready_detail: Dict[str, Any] = {
            'ready': False,
            'start_motion': False,
            'scout_raw_ready': False,
            'scout_yolo_ready': False,
            'scout_inference_ready': False,
            'omx_frame_ready': False,
            'published_count': 0,
        }
        self._dashboard_manifest: Dict[str, Any] = {}
        self._dashboard_manifest_wall_sec: Optional[float] = None
        self._dashboard_ui_good_since: Optional[float] = None
        self._dashboard_ui_ready = False
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

        @app.get('/')
        def index():
            return render_template('dashboard.html')

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

        @app.get('/api/map.png')
        @app.get('/map.png')
        def map_png():
            return self._grid_response(Response, 'map')

        @app.get('/api/risk.png')
        @app.get('/risk.png')
        def risk_png():
            return self._grid_response(Response, 'risk')

        @app.get('/api/yolo_stream/<kind>.mjpg')
        def yolo_stream(kind):
            if kind == 'raw':
                path = self.yolo_raw_stream_path
            elif kind in ('yolo', 'overlay'):
                path = self.yolo_overlay_stream_path
            else:
                return jsonify({'ok': False, 'error': 'kind must be raw or yolo'}), 404

            url = f'http://127.0.0.1:{self.yolo_server_port}{path}'

            def generate():
                try:
                    with urllib.request.urlopen(url, timeout=2.0) as upstream:
                        while True:
                            # MJPEG frames are commonly smaller than 64 KiB.
                            # Waiting for a whole 64 KiB block kept the browser
                            # panel black despite a healthy upstream stream.
                            chunk = upstream.read(8192)
                            if not chunk:
                                break
                            yield chunk
                except Exception as exc:  # noqa: BLE001
                    self.get_logger().warn(
                        'UNIFIED_DASHBOARD_YOLO_STREAM_PROXY_ERROR | '
                        f'kind={kind} url={url} error={exc}',
                        throttle_duration_sec=5.0,
                    )

            response = Response(
                stream_with_context(generate()),
                mimetype='multipart/x-mixed-replace; boundary=frame',
            )
            response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
            response.headers['X-Accel-Buffering'] = 'no'
            return response

        @app.get('/api/yolo_frame/<kind>.jpg')
        def yolo_frame(kind):
            if kind == 'raw':
                path = '/frame/raw.jpg'
            elif kind in ('yolo', 'overlay'):
                path = '/frame/yolo.jpg'
            else:
                return jsonify({'ok': False, 'error': 'kind must be raw or yolo'}), 404
            return self._single_jpeg_proxy(
                Response,
                f'http://127.0.0.1:{self.yolo_server_port}{path}',
                f'Scout {kind} frame waiting',
            )

        @app.get('/api/omx_frame.jpg')
        def omx_frame():
            return self._single_jpeg_proxy(
                Response,
                f'http://127.0.0.1:{self.omx_debug_port}/frame.jpg',
                'OMX frame waiting',
            )

        @app.get('/api/omx_stream.mjpg')
        def omx_stream():
            """Proxy the local OMX debug stream through the dashboard port.

            The dashboard is normally opened from another machine.  Pointing
            the browser directly at ``<leader-ip>:8080`` made the video panel
            depend on that extra port being reachable from the client, even
            though the dashboard itself on 8091 was reachable.  Keep the
            upstream strictly local and expose it through the already-open
            dashboard HTTP endpoint instead.
            """
            url = f'http://127.0.0.1:{self.omx_debug_port}{self.omx_stream_path}'

            def placeholder_frame(message: str) -> bytes:
                import cv2
                import numpy as np

                image = np.zeros((360, 640, 3), dtype=np.uint8)
                cv2.putText(
                    image, 'OMX VIDEO WAITING', (155, 160),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 190, 255), 2,
                )
                cv2.putText(
                    image, message[:72], (48, 205),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (220, 220, 220), 1,
                )
                ok, encoded = cv2.imencode('.jpg', image)
                if not ok:
                    return b''
                return (
                    b'--frame\r\nContent-Type: image/jpeg\r\n\r\n'
                    + encoded.tobytes() + b'\r\n'
                )

            def generate():
                # Keep this response alive while OMX is still loading or
                # restarting.  A normal proxy emitted no bytes in that state,
                # leaving a browser image permanently black until a reload.
                while True:
                    try:
                        with urllib.request.urlopen(url, timeout=1.0) as upstream:
                            while True:
                                chunk = upstream.read(8192)
                                if not chunk:
                                    break
                                yield chunk
                    except Exception as exc:  # noqa: BLE001
                        self.get_logger().warning(
                            'UNIFIED_DASHBOARD_OMX_STREAM_PROXY_ERROR | '
                            f'url={url} error={exc}',
                            throttle_duration_sec=5.0,
                        )
                        frame = placeholder_frame(f'upstream reconnecting: {type(exc).__name__}')
                        if frame:
                            yield frame
                        time.sleep(1.0)

            response = Response(
                stream_with_context(generate()),
                mimetype='multipart/x-mixed-replace; boundary=frame',
            )
            response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
            response.headers['X-Accel-Buffering'] = 'no'
            return response

        return app

    def _single_jpeg_proxy(self, response_class: Any, url: str, waiting_label: str) -> Any:
        """Serve one current frame, never a long-lived MJPEG connection.

        Browser reloads repeatedly interrupted the three concurrent MJPEG
        proxies, producing panels that appeared only after a lucky F5.  A
        short single-image request is independent, cache-busted by the UI,
        and cannot retain a stale stream connection.
        """
        try:
            with urllib.request.urlopen(url, timeout=0.8) as upstream:
                payload = upstream.read()
            response = response_class(payload, mimetype='image/jpeg')
        except Exception as exc:  # noqa: BLE001
            self.get_logger().warning(
                f'UNIFIED_DASHBOARD_FRAME_PROXY_WAIT | url={url} error={exc}',
                throttle_duration_sec=5.0,
            )
            response = response_class(
                _placeholder_jpeg(waiting_label), mimetype='image/jpeg'
            )
        response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
        return response

    def _grid_response(self, response_class: Any, kind: str) -> Any:
        try:
            png = self.grid_png(kind)
        except Exception as exc:
            self.get_logger().error(f'UNIFIED_DASHBOARD_ERROR | route=/{kind}.png | error={exc}')
            return response_class(f'{kind} render error\n', status=500, mimetype='text/plain')
        if png is None:
            return response_class(f'{kind} not available\n', status=404, mimetype='text/plain')
        response = response_class(png, mimetype='image/png')
        response.headers['Cache-Control'] = 'no-store'
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
            state['png'] = None
            state['png_seq'] = -1
            state['metadata'] = _grid_signature(msg)
            topic = self.map_topic if kind == 'map' else self.risk_topic
            self._touch_topic(topic, now, 'nav_msgs/msg/OccupancyGrid')

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
            required.extend(['scout_raw', 'scout_yolo'])
        if self.require_omx_video_ready:
            required.append('omx_camera')
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
            else:
                version_ok = isinstance(version, (int, float)) and float(version) > 0
            ok = bool(loaded and rendered and not placeholder and version_ok)
            panel_details[name] = {
                'loaded': loaded,
                'rendered': rendered,
                'placeholder': placeholder,
                'version': version,
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

    def _evaluate_video_ready(self) -> None:
        yolo = self.yolo_status()
        data = yolo.get('data') if isinstance(yolo, dict) else None
        if not isinstance(data, dict):
            data = {}
        raw_age = float(data.get('raw_frame_age_sec', -1.0) or -1.0)
        yolo_age = float(data.get('yolo_frame_age_sec', -1.0) or -1.0)
        inference_age = float(data.get('inference_frame_age_sec', -1.0) or -1.0)
        scout_raw_ready = (
            int(data.get('raw_frames', 0) or 0) > 0
            and 0.0 <= raw_age <= self.video_ready_max_age_sec
        )
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
        scout_ready = (
            scout_raw_ready and scout_yolo_ready and scout_inference_ready
            if self.require_scout_video_ready else True
        )
        omx_ready = omx_frame_ready if self.require_omx_video_ready else True
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
            'scout_raw_ready': scout_raw_ready,
            'scout_yolo_ready': scout_yolo_ready,
            'scout_inference_ready': scout_inference_ready,
            'omx_frame_ready': omx_frame_ready,
            'raw_frame_age_sec': raw_age,
            'yolo_frame_age_sec': yolo_age,
            'inference_frame_age_sec': inference_age,
            'omx_observation_fresh': observation_fresh,
            'video_ready_max_age_sec': self.video_ready_max_age_sec,
            'yolo_status': yolo.get('status'),
            'yolo_error': yolo.get('error'),
        }
        with self._lock:
            previous = self._video_ready
            previous_start_motion = self._start_motion
            self._video_ready = ready
            system_ready = system_ready_snapshot
            start_motion = bool(ready and system_ready)
            self._start_motion = start_motion
            published_count = int(self._video_ready_detail.get('published_count', 0))
            detail['published_count'] = published_count
            detail['start_motion'] = start_motion
            detail['system_ready'] = system_ready
            detail['system_ready_topic'] = self.system_ready_topic
            detail['start_motion_topic'] = self.start_motion_topic
            self._video_ready_detail = detail
            self._dashboard_ui_ready = ui_ready
        self._publish_readiness_detail(detail)
        if ready != previous:
            self._publish_video_ready(ready, 'all_video_ready' if ready else 'video_not_ready')
            self.get_logger().warning(
                'FLEET_VIDEO_READY | '
                f'ready={ready} scout_raw={scout_raw_ready} '
                f'scout_yolo={scout_yolo_ready} '
                f'scout_inference={scout_inference_ready} '
                f'omx_frame={omx_frame_ready} ui={ui_ready}'
            )
        if start_motion != previous_start_motion:
            reason = 'dashboard_and_system_ready' if start_motion else 'motion_barrier_not_ready'
            self._publish_start_motion(start_motion, {**detail, 'reason': reason})
            self.get_logger().warning(
                'FLEET_START_MOTION | '
                f'ready={start_motion} video_ready={ready} system_ready={system_ready} '
                f'ui={ui_ready} backend={backend_ready}'
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
            return {
                'server_time_sec': now,
                'omx_debug': {
                    'port': self.omx_debug_port,
                    'stream_path': self.omx_stream_path,
                    'state_path': self.omx_state_path,
                },
                'yolo_server': {
                    'port': self.yolo_server_port,
                    'raw_stream_path': self.yolo_raw_stream_path,
                    'overlay_stream_path': self.yolo_overlay_stream_path,
                    'raw_proxy_path': '/api/yolo_stream/raw.mjpg',
                    'overlay_proxy_path': '/api/yolo_stream/yolo.mjpg',
                    'status_path': self.yolo_status_path,
                },
                'omx': dict(self._omx_state),
                'events': self._events_summary(now),
                'map': self._grid_summary('map', now, self.map_stale_timeout_sec),
                'risk': {
                    **self._grid_summary('risk', now, self.risk_stale_timeout_sec),
                    'metadata_matches_map': _metadata_match(map_meta, risk_meta),
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
        return {
            'topic': self.map_topic if kind == 'map' else self.risk_topic,
            'status': status,
            'age_sec': age,
            'seq': int(state['seq']),
            'metadata': state['metadata'],
            'stamp_sec': _stamp_to_float(msg.header.stamp) if msg is not None else None,
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
            msg = state['msg']
            seq = int(state['seq'])
            if msg is None:
                return None
            if state['png'] is not None and state['png_seq'] == seq:
                return state['png']
            width = int(msg.info.width)
            height = int(msg.info.height)
            data = list(msg.data)

        png = _encode_grid_png(data, width, height, kind)
        with self._lock:
            state = self._grids[kind]
            if int(state['seq']) == seq:
                state['png'] = png
                state['png_seq'] = seq
        return png


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


def _placeholder_jpeg(label: str) -> bytes:
    import cv2
    import numpy as np

    image = np.zeros((360, 640, 3), dtype=np.uint8)
    cv2.putText(
        image, label[:62], (72, 185), cv2.FONT_HERSHEY_SIMPLEX,
        0.72, (0, 190, 255), 2,
    )
    ok, encoded = cv2.imencode('.jpg', image)
    if not ok:
        raise RuntimeError('failed to create dashboard placeholder')
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
