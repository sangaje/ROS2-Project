#!/usr/bin/env python3
"""Leader-hosted fleet dashboard for OMX, map/risk layers, and robot poses."""

from __future__ import annotations

import logging
import math
import threading
import time
from typing import Any, Dict, Optional, Tuple

import rclpy
from geometry_msgs.msg import PoseArray, PoseStamped
from nav_msgs.msg import OccupancyGrid
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import Bool, Float32, Int32, String


def _declare(node: Node, name: str, default: Any) -> Any:
    node.declare_parameter(name, default)
    return node.get_parameter(name).value


def quaternion_to_yaw(q: Any) -> float:
    return math.atan2(
        2.0 * (q.w * q.z + q.x * q.y),
        1.0 - 2.0 * (q.y * q.y + q.z * q.z),
    )


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
            'yaw': quaternion_to_yaw(origin.orientation),
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


def _stamp_to_float(stamp: Any) -> float:
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9


class UnifiedDashboardNode(Node):
    def __init__(self) -> None:
        super().__init__('leader_unified_dashboard')

        self.host = str(_declare(self, 'host', '0.0.0.0'))
        self.port = int(_declare(self, 'port', 8091))
        self.map_topic = str(_declare(self, 'map_topic', '/map'))
        self.risk_topic = str(_declare(self, 'risk_topic', '/risk/risk_map'))
        self.leader_pose_topic = str(_declare(self, 'leader_pose_topic', '/leader_pose'))
        self.follower_pose_topic = str(_declare(self, 'follower_pose_topic', '/burger_pose'))
        self.member_pose_topic = str(_declare(self, 'member_pose_topic', '/member_pose'))
        self.fleet_poses_topic = str(_declare(self, 'fleet_poses_topic', '/fleet/robot_poses'))
        self.fleet_status_topic = str(_declare(self, 'fleet_status_topic', '/fleet/coordination_status'))
        self.collision_warning_topic = str(_declare(self, 'collision_warning_topic', '/fleet/collision_warning'))
        self.omx_debug_port = int(_declare(self, 'omx_debug_port', 8080))
        self.omx_stream_path = str(_declare(self, 'omx_stream_path', '/stream.mjpg'))
        self.omx_state_path = str(_declare(self, 'omx_state_path', '/state.json'))
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
            'burger': self._empty_robot('burger', 'follower', self.follower_pose_topic),
            'member': self._empty_robot('member', 'member', self.member_pose_topic),
        }
        self._topic_state: Dict[str, Dict[str, Any]] = {}
        self._omx_state: Dict[str, Any] = {
            'state': None,
            'status': None,
            'target_detected': None,
            'aim_progress': None,
            'queue_size': None,
            'waffle_status': None,
        }

        latched_grid_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.create_subscription(
            OccupancyGrid, self.map_topic, lambda msg: self._on_grid('map', msg), latched_grid_qos
        )
        self.create_subscription(
            OccupancyGrid, self.risk_topic, lambda msg: self._on_grid('risk', msg), latched_grid_qos
        )
        self.create_subscription(
            PoseStamped,
            self.leader_pose_topic,
            lambda msg: self._on_pose('leader', msg, self.leader_pose_topic),
            10,
        )
        self.create_subscription(
            PoseStamped,
            self.follower_pose_topic,
            lambda msg: self._on_pose('burger', msg, self.follower_pose_topic),
            10,
        )
        self.create_subscription(
            PoseStamped,
            self.member_pose_topic,
            lambda msg: self._on_pose('member', msg, self.member_pose_topic),
            10,
        )
        self.create_subscription(PoseArray, self.fleet_poses_topic, self._on_fleet_poses, 10)
        self.create_subscription(String, self.fleet_status_topic, self._on_fleet_status, 10)
        self.create_subscription(Bool, self.collision_warning_topic, self._on_collision_warning, 10)
        self.create_subscription(String, '/omx/state', lambda msg: self._set_omx('state', msg.data), 10)
        self.create_subscription(String, '/omx/status', lambda msg: self._set_omx('status', msg.data), 10)
        self.create_subscription(Bool, '/omx/target_detected', lambda msg: self._set_omx('target_detected', bool(msg.data)), 10)
        self.create_subscription(Float32, '/omx/aim_progress', lambda msg: self._set_omx('aim_progress', float(msg.data)), 10)
        self.create_subscription(Int32, '/omx/queue_size', lambda msg: self._set_omx('queue_size', int(msg.data)), 10)
        self.create_subscription(String, '/waffle/status', lambda msg: self._set_omx('waffle_status', msg.data), 10)

        self._app = self._build_app()
        self._server_thread = threading.Thread(target=self._serve, daemon=True)
        self._server_thread.start()

        self.get_logger().info(
            'LEADER_UNIFIED_DASHBOARD_READY | '
            f'http://0.0.0.0:{self.port}/ | map={self.map_topic} '
            f'risk={self.risk_topic} poses={self.leader_pose_topic},'
            f'{self.follower_pose_topic},{self.member_pose_topic}'
        )

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

    def _serve(self) -> None:
        logging.getLogger('werkzeug').setLevel(logging.WARNING)
        self._app.run(
            host=self.host,
            port=self.port,
            debug=False,
            use_reloader=False,
            threaded=True,
        )

    def _build_app(self):
        try:
            from flask import Flask, Response, jsonify
        except ImportError as exc:
            raise RuntimeError('Flask is required for unified_dashboard') from exc

        app = Flask(__name__)

        @app.get('/')
        def index():
            return DASHBOARD_HTML

        @app.get('/api/state')
        def state():
            response = jsonify(self.snapshot())
            response.headers['Cache-Control'] = 'no-store'
            return response

        @app.get('/map.png')
        def map_png():
            png = self.grid_png('map')
            if png is None:
                return Response('map not available\n', status=404, mimetype='text/plain')
            response = Response(png, mimetype='image/png')
            response.headers['Cache-Control'] = 'no-store'
            return response

        @app.get('/risk.png')
        def risk_png():
            png = self.grid_png('risk')
            if png is None:
                return Response('risk map not available\n', status=404, mimetype='text/plain')
            response = Response(png, mimetype='image/png')
            response.headers['Cache-Control'] = 'no-store'
            return response

        return app

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
            robot['pose'] = msg
            robot['received_wall_sec'] = now
            robot['source'] = source
            self._touch_topic(source, now, 'geometry_msgs/msg/PoseStamped')

    def _on_fleet_poses(self, msg: PoseArray) -> None:
        now = time.time()
        names = ('leader', 'burger')
        with self._lock:
            for index, pose in enumerate(msg.poses[: len(names)]):
                name = names[index]
                robot = self._robots[name]
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

    def _set_omx(self, key: str, value: Any) -> None:
        now = time.time()
        with self._lock:
            self._omx_state[key] = value
            self._omx_state[f'{key}_received_wall_sec'] = now

    def _touch_topic(self, topic: str, now: float, msg_type: str) -> None:
        self._topic_state[topic] = {
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
                'omx': dict(self._omx_state),
                'map': self._grid_summary('map', now, self.map_stale_timeout_sec),
                'risk': {
                    **self._grid_summary('risk', now, self.risk_stale_timeout_sec),
                    'metadata_matches_map': _metadata_match(map_meta, risk_meta),
                },
                'robots': [self._robot_summary(name, now) for name in ('leader', 'burger', 'member')],
                'fleet': {
                    'coordination_status': self._topic_value(self.fleet_status_topic, now),
                    'collision_warning': self._topic_value(self.collision_warning_topic, now),
                },
                'topics': self._topics_summary(now),
            }

    def _grid_summary(self, kind: str, now: float, stale_timeout: float) -> Dict[str, Any]:
        state = self._grids[kind]
        age = None
        status = 'NO DATA'
        if state['received_wall_sec'] is not None:
            age = max(0.0, now - state['received_wall_sec'])
            status = 'STALE' if age > stale_timeout else 'OK'
        return {
            'topic': self.map_topic if kind == 'map' else self.risk_topic,
            'status': status,
            'age_sec': age,
            'seq': int(state['seq']),
            'metadata': state['metadata'],
            'stamp_sec': _stamp_to_float(state['msg'].header.stamp) if state['msg'] is not None else None,
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
            yaw = quaternion_to_yaw(pose.pose.orientation)
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

    def _topic_value(self, topic: str, now: float) -> Dict[str, Any]:
        state = self._topic_state.get(topic)
        if state is None:
            return {'topic': topic, 'status': 'NO DATA', 'age_sec': None, 'value': None}
        age = max(0.0, now - state['received_wall_sec'])
        return {
            'topic': topic,
            'status': 'STALE' if age > self.robot_stale_timeout_sec else 'OK',
            'age_sec': age,
            'value': state.get('value'),
        }

    def _topics_summary(self, now: float) -> Dict[str, Dict[str, Any]]:
        topics = {}
        for topic, state in self._topic_state.items():
            age = max(0.0, now - state['received_wall_sec'])
            topics[topic] = {
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


DASHBOARD_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Leader Unified Dashboard</title>
  <style>
    :root {
      color-scheme: dark;
      --bg: #101214;
      --panel: #181c20;
      --panel2: #20262b;
      --line: #303840;
      --text: #eef2f4;
      --muted: #9aa8b1;
      --good: #5bd08a;
      --warn: #f4c95d;
      --bad: #ff6f61;
      --leader: #58a6ff;
      --follower: #63d297;
      --member: #f2cc60;
      font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    * { box-sizing: border-box; }
    body { margin: 0; background: var(--bg); color: var(--text); }
    header {
      display: grid;
      grid-template-columns: minmax(0, 1fr) auto;
      gap: 12px;
      align-items: center;
      padding: 12px 16px;
      border-bottom: 1px solid var(--line);
      background: #111519;
    }
    h1 { margin: 0; font-size: 19px; font-weight: 700; }
    .top-status { display: flex; gap: 8px; flex-wrap: wrap; justify-content: flex-end; }
    .pill {
      display: inline-flex;
      align-items: center;
      gap: 7px;
      padding: 6px 9px;
      border: 1px solid var(--line);
      border-radius: 4px;
      background: var(--panel);
      color: var(--muted);
      font: 12px ui-monospace, SFMono-Regular, Menlo, monospace;
      white-space: nowrap;
    }
    .dot { width: 7px; height: 7px; border-radius: 50%; background: var(--muted); }
    .ok .dot, .online .dot { background: var(--good); }
    .stale .dot { background: var(--warn); }
    .bad .dot, .no-data .dot { background: var(--bad); }
    main {
      display: grid;
      grid-template-columns: minmax(300px, 0.95fr) minmax(440px, 1.35fr);
      gap: 12px;
      padding: 12px;
    }
    section {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 6px;
      overflow: hidden;
      min-width: 0;
    }
    section h2 {
      margin: 0;
      padding: 9px 11px;
      background: var(--panel2);
      border-bottom: 1px solid var(--line);
      color: #d7e0e6;
      font-size: 13px;
      font-weight: 650;
    }
    .stream-wrap { background: #030405; min-height: 280px; display: grid; place-items: center; }
    .stream-wrap img { display: block; width: 100%; height: auto; max-height: 58vh; object-fit: contain; }
    .map-panel { display: grid; grid-template-rows: auto minmax(380px, 1fr); }
    .map-tools {
      display: flex;
      gap: 12px;
      align-items: center;
      flex-wrap: wrap;
      padding: 8px 10px;
      border-bottom: 1px solid var(--line);
      color: var(--muted);
      font-size: 13px;
    }
    label { display: inline-flex; align-items: center; gap: 6px; white-space: nowrap; }
    input[type="range"] { width: 130px; }
    #mapCanvas { width: 100%; height: 100%; min-height: 380px; display: block; background: #08090a; }
    .lower {
      grid-column: 1 / -1;
      display: grid;
      grid-template-columns: minmax(0, 1.2fr) minmax(320px, 0.8fr);
      gap: 12px;
    }
    table { width: 100%; border-collapse: collapse; font-size: 13px; }
    th, td { padding: 8px 10px; text-align: left; border-bottom: 1px solid #283038; }
    th { color: var(--muted); font-size: 11px; text-transform: uppercase; letter-spacing: 0.04em; background: #15191d; }
    td { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }
    .role { font-family: inherit; font-weight: 700; }
    .role.leader { color: var(--leader); }
    .role.follower { color: var(--follower); }
    .role.member { color: var(--member); }
    .status-online, .status-ok { color: var(--good); font-weight: 700; }
    .status-stale { color: var(--warn); font-weight: 700; }
    .status-no-data { color: var(--bad); font-weight: 700; }
    .topic-list { padding: 8px 10px; display: grid; gap: 7px; font: 12px ui-monospace, SFMono-Regular, Menlo, monospace; }
    .topic-row { display: grid; grid-template-columns: minmax(150px, 1fr) auto; gap: 8px; color: var(--muted); }
    .warning { color: var(--warn); padding: 0 10px 8px; font-size: 12px; min-height: 20px; }
    .legend { margin-left: auto; display: inline-flex; gap: 10px; align-items: center; }
    .swatch { width: 10px; height: 10px; display: inline-block; border-radius: 2px; }
    @media (max-width: 980px) {
      header, main, .lower { grid-template-columns: 1fr; }
      .top-status { justify-content: flex-start; }
    }
  </style>
</head>
<body>
  <header>
    <div><h1>Leader Unified Dashboard</h1></div>
    <div class="top-status">
      <span id="leaderPill" class="pill no-data"><span class="dot"></span>Leader NO DATA</span>
      <span id="mapPill" class="pill no-data"><span class="dot"></span>Map NO DATA</span>
      <span id="riskPill" class="pill no-data"><span class="dot"></span>Risk NO DATA</span>
      <span id="robotPill" class="pill no-data"><span class="dot"></span>Robots 0/3</span>
    </div>
  </header>
  <main>
    <section>
      <h2>OMX AIM Debug Stream</h2>
      <div class="stream-wrap"><img id="omxStream" alt="OMX MJPEG stream"></div>
    </section>
    <section class="map-panel">
      <h2>Shared Map + Risk Overlay</h2>
      <div class="map-tools">
        <label><input id="layerMap" type="checkbox" checked> Map</label>
        <label><input id="layerRisk" type="checkbox" checked> Risk</label>
        <label><input id="layerRobots" type="checkbox" checked> Robots</label>
        <label>Risk opacity <input id="riskOpacity" type="range" min="0" max="1" step="0.05" value="0.55"></label>
        <span class="legend">
          <span><span class="swatch" style="background: var(--leader)"></span> leader</span>
          <span><span class="swatch" style="background: var(--follower)"></span> follower</span>
          <span><span class="swatch" style="background: var(--member)"></span> member</span>
        </span>
      </div>
      <canvas id="mapCanvas"></canvas>
      <div id="mapWarning" class="warning"></div>
    </section>
    <div class="lower">
      <section>
        <h2>Robot Status</h2>
        <table>
          <thead><tr><th>Robot</th><th>Role</th><th>Status</th><th>X</th><th>Y</th><th>Yaw</th><th>Age</th></tr></thead>
          <tbody id="robotRows"></tbody>
        </table>
      </section>
      <section>
        <h2>Topic / Data Status</h2>
        <div id="topicRows" class="topic-list"></div>
      </section>
    </div>
  </main>
  <script>
    const stateUrl = '/api/state';
    const mapImg = new Image();
    const riskImg = new Image();
    let latest = null;
    let mapSeq = -1;
    let riskSeq = -1;
    let mapReady = false;
    let riskReady = false;
    const canvas = document.getElementById('mapCanvas');
    const ctx = canvas.getContext('2d');
    const roleColors = {leader: '#58a6ff', follower: '#63d297', member: '#f2cc60'};

    function statusClass(status) {
      return String(status || 'NO DATA').toLowerCase().replaceAll(' ', '-');
    }
    function fmt(value, digits = 2) {
      return Number.isFinite(value) ? value.toFixed(digits) : '--';
    }
    function ageText(value) {
      return Number.isFinite(value) ? `${value.toFixed(1)}s` : '--';
    }
    function yawDeg(rad) {
      return Number.isFinite(rad) ? `${(rad * 180.0 / Math.PI).toFixed(0)} deg` : '--';
    }
    function setPill(id, label, status) {
      const el = document.getElementById(id);
      el.className = `pill ${statusClass(status)}`;
      el.innerHTML = `<span class="dot"></span>${label} ${status || 'NO DATA'}`;
    }
    function configureStream(s) {
      const img = document.getElementById('omxStream');
      if (img.dataset.configured === '1') return;
      const port = s.omx_debug.port;
      const path = s.omx_debug.stream_path || '/stream.mjpg';
      img.src = `${location.protocol}//${location.hostname}:${port}${path}`;
      img.dataset.configured = '1';
    }
    function updateImages(s) {
      if (s.map.seq !== mapSeq && s.map.status !== 'NO DATA') {
        mapSeq = s.map.seq;
        mapReady = false;
        mapImg.src = `/map.png?v=${mapSeq}&t=${Date.now()}`;
      }
      if (s.risk.seq !== riskSeq && s.risk.status !== 'NO DATA') {
        riskSeq = s.risk.seq;
        riskReady = false;
        riskImg.src = `/risk.png?v=${riskSeq}&t=${Date.now()}`;
      }
    }
    mapImg.onload = () => { mapReady = true; draw(); };
    riskImg.onload = () => { riskReady = true; draw(); };
    mapImg.onerror = () => { mapReady = false; draw(); };
    riskImg.onerror = () => { riskReady = false; draw(); };

    function resizeCanvas() {
      const rect = canvas.getBoundingClientRect();
      const ratio = window.devicePixelRatio || 1;
      const w = Math.max(320, Math.floor(rect.width * ratio));
      const h = Math.max(360, Math.floor(rect.height * ratio));
      if (canvas.width !== w || canvas.height !== h) {
        canvas.width = w;
        canvas.height = h;
      }
    }
    function mapViewport(meta) {
      const w = meta.width;
      const h = meta.height;
      const scale = Math.min(canvas.width / w, canvas.height / h);
      return {
        scale,
        x: (canvas.width - w * scale) * 0.5,
        y: (canvas.height - h * scale) * 0.5,
        w: w * scale,
        h: h * scale,
      };
    }
    function worldToCell(meta, x, y) {
      const o = meta.origin;
      const dx = x - o.x;
      const dy = y - o.y;
      const c = Math.cos(o.yaw || 0.0);
      const s = Math.sin(o.yaw || 0.0);
      const gx = (c * dx + s * dy) / meta.resolution;
      const gy = (-s * dx + c * dy) / meta.resolution;
      return {x: gx, y: gy};
    }
    function cellToCanvas(meta, vp, cell) {
      return {
        x: vp.x + cell.x * vp.scale,
        y: vp.y + (meta.height - cell.y) * vp.scale,
      };
    }
    function drawRobot(meta, vp, robot) {
      if (!robot.position || !Number.isFinite(robot.yaw_rad)) return;
      const cell = worldToCell(meta, robot.position.x, robot.position.y);
      if (cell.x < -1 || cell.x > meta.width + 1 || cell.y < -1 || cell.y > meta.height + 1) return;
      const p = cellToCanvas(meta, vp, cell);
      const color = roleColors[robot.role] || '#d0d7de';
      const stale = robot.status !== 'ONLINE';
      const radius = Math.max(5, Math.min(11, vp.scale * 0.18));
      const yawGrid = robot.yaw_rad - (meta.origin.yaw || 0.0);
      ctx.save();
      ctx.globalAlpha = stale ? 0.48 : 1.0;
      ctx.strokeStyle = color;
      ctx.fillStyle = color;
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.arc(p.x, p.y, radius, 0, Math.PI * 2);
      ctx.fill();
      ctx.beginPath();
      ctx.moveTo(p.x, p.y);
      ctx.lineTo(p.x + Math.cos(yawGrid) * radius * 2.4, p.y - Math.sin(yawGrid) * radius * 2.4);
      ctx.stroke();
      ctx.font = '12px ui-monospace, SFMono-Regular, Menlo, monospace';
      ctx.fillStyle = '#f6f8fa';
      ctx.strokeStyle = '#000';
      ctx.lineWidth = 3;
      const label = `${robot.name} (${robot.role})`;
      ctx.strokeText(label, p.x + radius + 5, p.y - radius - 5);
      ctx.fillText(label, p.x + radius + 5, p.y - radius - 5);
      ctx.restore();
    }
    function draw() {
      resizeCanvas();
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.fillStyle = '#08090a';
      ctx.fillRect(0, 0, canvas.width, canvas.height);
      if (!latest || !latest.map.metadata) {
        ctx.fillStyle = '#9aa8b1';
        ctx.font = '15px ui-monospace, SFMono-Regular, Menlo, monospace';
        ctx.fillText('Waiting for /map', 18, 32);
        return;
      }
      const meta = latest.map.metadata;
      const vp = mapViewport(meta);
      if (document.getElementById('layerMap').checked && mapReady) {
        ctx.drawImage(mapImg, vp.x, vp.y, vp.w, vp.h);
      }
      if (document.getElementById('layerRisk').checked && riskReady && latest.risk.metadata_matches_map) {
        ctx.save();
        ctx.globalAlpha = Number(document.getElementById('riskOpacity').value);
        ctx.drawImage(riskImg, vp.x, vp.y, vp.w, vp.h);
        ctx.restore();
      }
      if (document.getElementById('layerRobots').checked) {
        latest.robots.forEach(robot => drawRobot(meta, vp, robot));
      }
      ctx.strokeStyle = '#4b5560';
      ctx.lineWidth = 1;
      ctx.strokeRect(vp.x, vp.y, vp.w, vp.h);
    }
    function updateTables(s) {
      const robotRows = document.getElementById('robotRows');
      robotRows.innerHTML = s.robots.map(r => {
        const pos = r.position || {};
        const sc = statusClass(r.status);
        return `<tr>
          <td>${r.name}</td>
          <td class="role ${r.role}">${r.role}</td>
          <td class="status-${sc}">${r.status}</td>
          <td>${fmt(pos.x)}</td>
          <td>${fmt(pos.y)}</td>
          <td>${yawDeg(r.yaw_rad)}</td>
          <td>${ageText(r.age_sec)}</td>
        </tr>`;
      }).join('');
      const rows = [
        ['/map', s.map.status, s.map.age_sec],
        ['/risk/risk_map', s.risk.status, s.risk.age_sec],
        [s.fleet.coordination_status.topic, s.fleet.coordination_status.status, s.fleet.coordination_status.age_sec],
        [s.fleet.collision_warning.topic, s.fleet.collision_warning.status, s.fleet.collision_warning.age_sec],
        ['/omx/state', s.omx.state ? 'OK' : 'NO DATA', s.omx.state_received_wall_sec ? s.server_time_sec - s.omx.state_received_wall_sec : null],
        ['/omx/status', s.omx.status ? 'OK' : 'NO DATA', s.omx.status_received_wall_sec ? s.server_time_sec - s.omx.status_received_wall_sec : null],
      ];
      document.getElementById('topicRows').innerHTML = rows.map(row => {
        const cls = statusClass(row[1]);
        return `<div class="topic-row"><span>${row[0]}</span><span class="status-${cls}">${row[1]} ${ageText(row[2])}</span></div>`;
      }).join('');
    }
    function updateTop(s) {
      const leader = s.robots.find(r => r.name === 'leader') || {};
      const online = s.robots.filter(r => r.status === 'ONLINE').length;
      setPill('leaderPill', 'Leader', leader.status);
      setPill('mapPill', 'Map', s.map.status);
      setPill('riskPill', 'Risk', s.risk.status);
      const rp = document.getElementById('robotPill');
      rp.className = `pill ${online ? 'online' : 'no-data'}`;
      rp.innerHTML = `<span class="dot"></span>Robots ${online}/${s.robots.length}`;
      const warning = document.getElementById('mapWarning');
      warning.textContent = (!s.risk.metadata_matches_map && s.risk.status !== 'NO DATA')
        ? 'Risk overlay metadata does not match /map, so overlay rendering is suppressed.'
        : '';
    }
    async function refresh() {
      try {
        const s = await (await fetch(stateUrl, {cache: 'no-store'})).json();
        latest = s;
        configureStream(s);
        updateImages(s);
        updateTop(s);
        updateTables(s);
        draw();
      } catch (err) {
        console.warn('dashboard refresh failed', err);
      }
    }
    ['layerMap', 'layerRisk', 'layerRobots', 'riskOpacity'].forEach(id => {
      document.getElementById(id).addEventListener('input', draw);
      document.getElementById(id).addEventListener('change', draw);
    });
    window.addEventListener('resize', draw);
    refresh();
    setInterval(refresh, 500);
  </script>
</body>
</html>
"""


def main(args=None) -> None:
    rclpy.init(args=args)
    node = UnifiedDashboardNode()
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
