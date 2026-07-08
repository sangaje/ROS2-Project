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
from geometry_msgs.msg import Point, PointStamped, PoseArray, PoseStamped, Twist
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
        self.robot_stale_timeout_sec = float(_declare(self, 'robot_stale_timeout_sec', 3.0))
        self.map_stale_timeout_sec = float(_declare(self, 'map_stale_timeout_sec', 30.0))
        self.risk_stale_timeout_sec = float(_declare(self, 'risk_stale_timeout_sec', 10.0))

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
        }
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

        grid_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
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
        self.create_subscription(String, '/risk/yolo_detections', lambda msg: self._set_topic_value('/risk/yolo_detections', msg.data, 'std_msgs/msg/String'), 10)
        self.create_subscription(String, '/omx/state', lambda msg: self._set_omx('state', msg.data, '/omx/state', 'std_msgs/msg/String'), 10)
        self.create_subscription(String, '/omx/status', lambda msg: self._set_omx('status', msg.data, '/omx/status', 'std_msgs/msg/String'), 10)
        self.create_subscription(Bool, '/omx/target_detected', lambda msg: self._set_omx('target_detected', bool(msg.data), '/omx/target_detected', 'std_msgs/msg/Bool'), 10)
        self.create_subscription(Float32, '/omx/aim_progress', lambda msg: self._set_omx('aim_progress', float(msg.data), '/omx/aim_progress', 'std_msgs/msg/Float32'), 10)
        self.create_subscription(Int32, '/omx/queue_size', lambda msg: self._set_omx('queue_size', int(msg.data), '/omx/queue_size', 'std_msgs/msg/Int32'), 10)
        self.create_subscription(String, '/waffle/status', lambda msg: self._set_omx('waffle_status', msg.data, '/waffle/status', 'std_msgs/msg/String'), 10)
        self.create_subscription(String, '/waffle/nav_result', lambda msg: self._set_omx('waffle_nav_result', msg.data, '/waffle/nav_result', 'std_msgs/msg/String'), 10)
        self.create_subscription(String, '/omx/fire_status', lambda msg: self._set_omx('fire_status', msg.data, '/omx/fire_status', 'std_msgs/msg/String'), 10)
        self.create_subscription(Bool, '/omx/fire_disable', lambda msg: self._set_omx('fire_disabled', bool(msg.data), '/omx/fire_disable', 'std_msgs/msg/Bool'), 10)
        self.create_subscription(Point, '/omx/error_norm', self._on_aim_error, 10)
        self.create_subscription(Twist, '/cmd_vel', self._on_cmd_vel, 10)
        self.create_subscription(Empty, '/omx/fire', lambda msg: self._on_empty_event('fire'), 10)
        self.create_subscription(Empty, '/omx/nav_cancel', lambda msg: self._on_empty_event('nav_cancel'), 10)
        self.create_subscription(Empty, '/omx/patrol_complete', lambda msg: self._on_empty_event('patrol_complete'), 10)
        self.create_subscription(PointStamped, '/omx/target_processed', lambda msg: self._on_point_event('target_processed', msg), 10)
        self.create_subscription(PointStamped, '/omx/target_lost', lambda msg: self._on_point_event('target_lost', msg), 10)
        self.create_subscription(PointStamped, '/omx/target_blocked', lambda msg: self._on_point_event('target_blocked', msg), 10)
        self.create_subscription(PointStamped, '/omx/target_not_found', lambda msg: self._on_point_event('target_not_found', msg), 10)
        self.create_subscription(PoseStamped, '/omx/nav_goal', self._on_nav_goal, 10)

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
            from flask import Flask, Response, jsonify, render_template, stream_with_context
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
                            chunk = upstream.read(65536)
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
            return response

        return app

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

    def _on_cmd_vel(self, msg: Twist) -> None:
        value = {
            'linear_x': float(msg.linear.x),
            'linear_y': float(msg.linear.y),
            'angular_z': float(msg.angular.z),
        }
        self._set_omx('leader_cmd_vel', value, '/cmd_vel', 'geometry_msgs/msg/Twist')

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
                },
                'nav2_paths': [
                    self._nav_path_summary(name, now)
                    for name in ('leader', 'leader_bridge', 'follower', 'member')
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
        img[unknown] = (58, 55, 50)
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
