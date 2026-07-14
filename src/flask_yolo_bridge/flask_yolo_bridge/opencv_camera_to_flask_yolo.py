#!/usr/bin/env python3

from __future__ import annotations

import json
import math
import os
import statistics
import threading
import time
from collections import deque

import requests

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy, HistoryPolicy, ReliabilityPolicy
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import String

from flask_yolo_bridge.observation_contract import (
    PoseSample,
    build_observation_metadata,
    closest_pose_sample,
    make_boot_id,
    parse_role_payload,
)
from flask_yolo_bridge.ros_param_helpers import FlexibleParameterNodeMixin


class OpenCVCameraToFlaskYolo(FlexibleParameterNodeMixin, Node):
    """
    Direct robot-camera -> HTTP JPEG -> PC Flask YOLO -> compact ROS detection JSON.

    This intentionally does not publish ROS image topics. The only ROS output is
    /risk/yolo_detections, which keeps DDS/network load tiny on TurtleBot3/Pi4.
    """

    def __init__(self):
        super().__init__('opencv_camera_to_flask_yolo')

        self.device = self.declare_parameter('device', '/dev/video0').value
        self.fallback_devices = str(
            self.declare_parameter(
                'fallback_devices',
                '/dev/video1,/dev/video0,/dev/video2,/dev/video3',
            ).value
        )
        self.frame_id = str(self.declare_parameter('frame_id', 'camera_link').value)
        self.width = int(self.declare_parameter('width', 640).value)
        self.height = int(self.declare_parameter('height', 480).value)
        self.send_width = int(self.declare_parameter('send_width', self.width).value)
        self.send_height = int(self.declare_parameter('send_height', self.height).value)
        self.camera_fps = float(self.declare_parameter('camera_fps', 5.0).value)
        self.buffer_size = int(self.declare_parameter('buffer_size', 1).value)
        self.fourcc = str(self.declare_parameter('fourcc', 'MJPG').value).strip()

        self.server_url = str(
            self.declare_parameter('server_url', 'http://127.0.0.1:5005/detect').value
        ).strip()
        self.output_topic = self.declare_parameter('output_topic', '/risk/yolo_detections').value
        self.max_rate_hz = float(self.declare_parameter('max_rate_hz', 5.0).value)
        self.active_max_rate_hz = float(
            self.declare_parameter('active_max_rate_hz', self.max_rate_hz).value
        )
        self.standby_max_rate_hz = float(
            self.declare_parameter('standby_max_rate_hz', min(1.0, self.max_rate_hz)).value
        )
        self.active_max_upload_mbps = float(
            self.declare_parameter('active_max_upload_mbps', 2.5).value
        )
        self.standby_max_upload_mbps = float(
            self.declare_parameter('standby_max_upload_mbps', 0.8).value
        )
        requested_workers = int(self.declare_parameter('http_worker_count', 1).value)
        self.http_worker_count = 1
        if requested_workers != 1:
            self.get_logger().warning(
                'CAMERA_HTTP_WORKER_CLAMPED | requested='
                f'{requested_workers} using=1 reason=one_inflight_request'
            )
        self.jpeg_quality = int(self.declare_parameter('jpeg_quality', 65).value)
        self.letterbox_color = int(
            self.declare_parameter('letterbox_color', 0).value
        )
        self.timeout_sec = float(self.declare_parameter('timeout_sec', 1.0).value)
        self.connect_timeout_sec = float(
            self.declare_parameter('connect_timeout_sec', min(0.3, self.timeout_sec)).value
        )
        self.read_timeout_sec = float(
            self.declare_parameter('read_timeout_sec', self.timeout_sec).value
        )
        self.max_http_roundtrip_sec = float(
            self.declare_parameter('max_http_roundtrip_sec', 1.0).value
        )
        self.max_frame_age_sec = float(
            self.declare_parameter('max_frame_age_sec', 1.2).value
        )
        self.retry_open_period_sec = float(
            self.declare_parameter('retry_open_period_sec', 1.0).value
        )
        self.log_period_sec = float(self.declare_parameter('log_period_sec', 2.0).value)
        self.publish_empty_detections = self.declare_bool_parameter('publish_empty_detections', True)
        self.enable_role_gating = self.declare_bool_parameter('enable_role_gating', False)
        self.robot_name = str(self.declare_parameter('robot_name', '').value).strip()
        self.boot_id = str(
            self.declare_parameter('boot_id', make_boot_id()).value
        ).strip() or make_boot_id()
        self.initial_role = str(
            self.declare_parameter('initial_role', 'ACTIVE_SCOUT').value
        ).strip().upper() or 'ACTIVE_SCOUT'
        self.current_role = self.initial_role
        self.current_role_epoch = 0
        self.role_topic = str(self.declare_parameter('role_topic', '').value).strip()
        self.active_scout_id_topic = str(
            self.declare_parameter('active_scout_id_topic', '/failover/active_scout_id').value
        ).strip()
        self.pose_topic = str(
            self.declare_parameter('pose_topic', '/member_pose').value
        ).strip()
        self.require_capture_pose = self.declare_bool_parameter(
            'require_capture_pose',
            True,
        )
        self.pose_history_duration_sec = float(
            self.declare_parameter('pose_history_duration_sec', 8.0).value
        )
        self.pose_history_max_samples = max(
            2,
            int(self.declare_parameter('pose_history_max_samples', 240).value),
        )
        self.max_pose_interpolation_error_sec = float(
            self.declare_parameter('max_pose_interpolation_error_sec', 0.35).value
        )
        self.camera_hfov_deg = float(
            self.declare_parameter('camera_hfov_deg', 62.0).value
        )
        self.camera_calibration_id = str(
            self.declare_parameter('camera_calibration_id', '').value
        ).strip()
        self.active_roles = {
            item.strip().upper()
            for item in str(
                self.declare_parameter(
                    'active_roles',
                    'ACTIVE_SCOUT,SCOUT,RECOVERING',
                ).value
            ).split(',')
            if item.strip()
        }
        # Informational only: these role names never grant camera/upload/
        # publish capability by themselves. They used to be OR'd into the
        # activation check below (a bug -- FOLLOWER sat in both this set and
        # publish_roles, so a follower's camera stayed open and kept
        # uploading to the same YOLO server as the real active scout). Kept
        # as a declared parameter only so existing launch files that still
        # pass it don't error, and so role logs can label a role as a known
        # standby state.
        self.standby_roles = {
            item.strip().upper()
            for item in str(
                self.declare_parameter(
                    'standby_roles',
                    'FOLLOWER,IDLE,TAKEOVER_PENDING',
                ).value
            ).split(',')
            if item.strip()
        }
        self.publish_roles = {
            item.strip().upper()
            for item in str(
                self.declare_parameter(
                    'publish_roles',
                    'ACTIVE_SCOUT,SCOUT,RECOVERING',
                ).value
            ).split(',')
            if item.strip()
        }
        initial_active = bool(
            self.declare_bool_parameter('initial_role_active', True)
        )
        # Three independently-named gates. They happen to share one source
        # condition today (role membership in active_roles/publish_roles),
        # but are tracked/logged separately so a FOLLOWER (or any role
        # outside active_roles) never opens the camera, never uploads a
        # frame, and never publishes a risk observation -- not even at a
        # throttled "standby" rate.
        self.role_topic_camera_enabled = initial_active
        self.role_topic_publish_enabled = (
            initial_active and self.initial_role in self.publish_roles
        )
        self.active_scout_id_enabled = None
        self.camera_process_enabled = initial_active
        self.camera_upload_enabled = initial_active
        self.risk_observation_publish_enabled = (
            initial_active and self.initial_role in self.publish_roles
        )
        if not self.enable_role_gating:
            self.camera_process_enabled = True
            self.camera_upload_enabled = True
            self.risk_observation_publish_enabled = True
        if self.enable_role_gating and not self.role_topic:
            if self.robot_name:
                self.role_topic = f'/{self.robot_name}/role'
            else:
                self.get_logger().warn(
                    'OPENCV_CAMERA_ROLE_GATE_DISABLED | '
                    'robot_name/role_topic missing'
                )
                self.enable_role_gating = False
                self.camera_process_enabled = True
                self.camera_upload_enabled = True
                self.risk_observation_publish_enabled = True
        # Back-compat alias: capture_loop/http_worker_loop/wait_latest_frame
        # read camera_process_enabled/camera_upload_enabled directly; this
        # name is kept only for the startup log line below.
        self.camera_active = self.camera_process_enabled

        self.pub = self.create_publisher(
            String,
            self.output_topic,
            QoSProfile(
                reliability=ReliabilityPolicy.BEST_EFFORT,
                durability=DurabilityPolicy.VOLATILE,
                history=HistoryPolicy.KEEP_LAST,
                depth=1,
            ),
        )
        if self.role_topic:
            role_qos = QoSProfile(
                reliability=ReliabilityPolicy.RELIABLE,
                durability=DurabilityPolicy.TRANSIENT_LOCAL,
                history=HistoryPolicy.KEEP_LAST,
                depth=1,
            )
            self.create_subscription(String, self.role_topic, self.on_role, role_qos)
            if self.active_scout_id_topic:
                self.create_subscription(
                    String,
                    self.active_scout_id_topic,
                    self.on_active_scout_id,
                    role_qos,
                )
        self.pose_history = deque(maxlen=self.pose_history_max_samples)
        self.pose_lock = threading.Lock()
        self.create_subscription(PoseStamped, self.pose_topic, self.on_pose, 10)
        self.http = requests.Session()
        adapter = requests.adapters.HTTPAdapter(
            pool_connections=1,
            pool_maxsize=max(1, self.http_worker_count),
            max_retries=0,
            pool_block=True,
        )
        self.http.mount('http://', adapter)
        self.http.mount('https://', adapter)
        self.http.headers.update({'Connection': 'keep-alive'})

        self.frame_condition = threading.Condition()
        self.latest_frame = None
        self.latest_capture_sec = 0.0
        self.latest_capture_mono_sec = 0.0
        self.latest_observation_meta = {}
        self.latest_seq = 0
        self.sent_seq = 0
        self.publish_lock = threading.Lock()
        self.published_seq = 0
        self.inflight_requests = 0
        self.stop_threads = False

        self.rx_count = 0
        self.sent_count = 0
        self.ok_count = 0
        self.fail_count = 0
        self.drop_count = 0
        self.missing_pose_drop_count = 0
        self.last_log_wall_sec = 0.0
        self.ok_timestamps = deque(maxlen=120)
        self.resize_ms_samples = deque(maxlen=120)
        self.encode_ms_samples = deque(maxlen=120)
        self.rtt_ms_samples = deque(maxlen=120)
        self.capture_age_ms_samples = deque(maxlen=120)
        self.jpeg_size_samples = deque(maxlen=120)
        self.tx_bytes_window = deque(maxlen=240)
        self.tx_budget_tokens = 0.0
        self.tx_budget_last_mono_sec = time.monotonic()
        self.active_device = ''
        self.next_open_attempt_mono_sec = 0.0
        self.read_fail_streak = 0

        import cv2

        self.cv2 = cv2
        self.cap = None
        if self.camera_process_enabled:
            self.open_camera(log_success=False)

        self.capture_thread = threading.Thread(
            target=self.capture_loop,
            name='opencv_http_yolo_capture_latest_frame',
            daemon=True,
        )
        self.worker_threads = [
            threading.Thread(
                target=self.http_worker_loop,
                name=f'opencv_http_yolo_latest_frame_sender_{index + 1}',
                daemon=True,
            )
            for index in range(max(1, self.http_worker_count))
        ]
        self.capture_thread.start()
        for thread in self.worker_threads:
            thread.start()

        self.get_logger().info(
            f'OPENCV_CAMERA_TO_FLASK_YOLO_READY | device={self.device} '
            f'active_device={self.active_device or "none"} opened={self.is_camera_open()} '
            f'capture_request={self.width}x{self.height} send={self.send_width}x{self.send_height} '
            f'camera_fps={self.camera_fps:.1f} fourcc={self.fourcc or "default"} '
            f'server={self.server_url} out={self.output_topic} rate={self.max_rate_hz:.2f}Hz '
            f'active_rate={self.active_max_rate_hz:.2f}Hz standby_rate={self.standby_max_rate_hz:.2f}Hz '
            f'active_budget={self.active_max_upload_mbps:.2f}Mbps '
            f'standby_budget={self.standby_max_upload_mbps:.2f}Mbps '
            f'http_workers={max(1, self.http_worker_count)} '
            f'jpeg_quality={self.jpeg_quality} timeout={self.timeout_sec:.2f}s '
            f'letterbox=preserve_full_fov color={self.letterbox_color} '
            f'connect_timeout={self.connect_timeout_sec:.2f}s read_timeout={self.read_timeout_sec:.2f}s '
            f'max_http_roundtrip={self.max_http_roundtrip_sec:.2f}s '
            f'max_frame_age={self.max_frame_age_sec:.2f}s '
            f'role_gating={self.enable_role_gating} role_topic={self.role_topic or "none"} '
            f'active_scout_id_topic={self.active_scout_id_topic or "none"} '
            f'pose_topic={self.pose_topic} require_capture_pose={self.require_capture_pose} '
            f'robot_id={self.robot_name or "unknown"} boot_id={self.boot_id} '
            f'camera_process_enabled={self.camera_process_enabled} '
            f'camera_upload_enabled={self.camera_upload_enabled} '
            f'risk_observation_publish_enabled={self.risk_observation_publish_enabled} '
            f'active_roles={sorted(self.active_roles)} '
            f'standby_roles={sorted(self.standby_roles)} '
            f'publish_roles={sorted(self.publish_roles)} '
            f'latest_frame_only=true ros_image_publish=false'
        )

    def _apply_role_gate(self, source: str, value: str):
        if self.active_scout_id_enabled is None:
            camera_process_enabled = self.role_topic_camera_enabled
            risk_observation_publish_enabled = self.role_topic_publish_enabled
        else:
            camera_process_enabled = bool(self.active_scout_id_enabled)
            risk_observation_publish_enabled = bool(self.active_scout_id_enabled)
        camera_upload_enabled = camera_process_enabled
        changed = (
            camera_process_enabled != self.camera_process_enabled
            or camera_upload_enabled != self.camera_upload_enabled
            or risk_observation_publish_enabled != self.risk_observation_publish_enabled
        )
        if not changed:
            return
        self.camera_process_enabled = camera_process_enabled
        self.camera_upload_enabled = camera_upload_enabled
        self.risk_observation_publish_enabled = risk_observation_publish_enabled
        self.camera_active = camera_process_enabled
        if not camera_process_enabled:
            self.close_camera()
            with self.frame_condition:
                self.latest_frame = None
                self.sent_seq = self.latest_seq
                self.frame_condition.notify_all()
        else:
            self.next_open_attempt_mono_sec = 0.0
            with self.frame_condition:
                self.frame_condition.notify_all()
        self.get_logger().warning(
            'OPENCV_CAMERA_ROLE_GATE | '
            f'source={source} value={value} role={self.current_role} '
            f'camera_process_enabled={self.camera_process_enabled} '
            f'camera_upload_enabled={self.camera_upload_enabled} '
            f'risk_observation_publish_enabled={self.risk_observation_publish_enabled} '
            f'robot={self.robot_name or "unknown"}'
        )

    def on_role(self, msg: String):
        raw = msg.data.strip()
        role = raw
        if raw.startswith('{'):
            role, epoch = parse_role_payload(raw, self.current_role)
            self.current_role = role
            self.current_role_epoch = epoch
        else:
            self.current_role = raw.strip().upper()
        role_name = role.strip().upper()
        # Only active_roles (ACTIVE_SCOUT/SCOUT/RECOVERING by default) may
        # open the camera, upload to the YOLO server, or publish a risk
        # observation. FOLLOWER/IDLE/TAKEOVER_PENDING/RECOVERY_NAVIGATING
        # get none of the three, even at a throttled rate -- a follower has
        # no scout authority until it actually takes over.
        self.role_topic_camera_enabled = role_name in self.active_roles
        self.role_topic_publish_enabled = role_name in self.publish_roles
        self._apply_role_gate('role_topic', role_name)

    def on_active_scout_id(self, msg: String):
        active_scout_id = str(msg.data or '').strip()
        if not active_scout_id:
            return
        active = active_scout_id == self.robot_name
        self.active_scout_id_enabled = active
        if active:
            self.current_role = 'ACTIVE_SCOUT'
        self._apply_role_gate('active_scout_id', active_scout_id)

    @staticmethod
    def stamp_to_sec(stamp) -> float:
        try:
            return float(stamp.sec) + float(stamp.nanosec) * 1e-9
        except Exception:
            return 0.0

    @staticmethod
    def yaw_from_quaternion(q) -> float:
        import math
        return math.atan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y * q.y + q.z * q.z),
        )

    def on_pose(self, msg: PoseStamped) -> None:
        stamp_sec = self.stamp_to_sec(msg.header.stamp)
        if stamp_sec <= 0.0:
            stamp_sec = self.get_clock().now().nanoseconds * 1e-9
        p = msg.pose.position
        q = msg.pose.orientation
        sample = PoseSample(
            stamp_sec=stamp_sec,
            x=float(p.x),
            y=float(p.y),
            yaw=self.yaw_from_quaternion(q),
        )
        cutoff = stamp_sec - max(1.0, self.pose_history_duration_sec)
        with self.pose_lock:
            self.pose_history.append(sample)
            while len(self.pose_history) > 2 and self.pose_history[0].stamp_sec < cutoff:
                self.pose_history.popleft()

    def lookup_capture_pose(self, capture_sec: float):
        with self.pose_lock:
            samples = list(self.pose_history)
        return closest_pose_sample(
            samples,
            capture_sec,
            self.max_pose_interpolation_error_sec,
        )

    def camera_candidates(self):
        candidates = []

        def add(value):
            value = str(value).strip()
            if not value:
                return
            if value.lower() == 'auto':
                for index in range(8):
                    add(f'/dev/video{index}')
                return
            if value not in candidates:
                candidates.append(value)

        add(self.device)
        for item in self.fallback_devices.split(','):
            add(item)
        return candidates

    def configure_capture(self, cap):
        cv2 = self.cv2
        if self.buffer_size > 0:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, int(self.buffer_size))
        if self.fourcc:
            code = self.fourcc.upper()[:4]
            if len(code) == 4:
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*code))
        if self.width > 0:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(self.width))
        if self.height > 0:
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(self.height))
        if self.camera_fps > 0.0:
            cap.set(cv2.CAP_PROP_FPS, float(self.camera_fps))

    def close_camera(self):
        cap = self.cap
        self.cap = None
        self.active_device = ''
        self.read_fail_streak = 0
        if cap is not None:
            try:
                cap.release()
            except Exception:
                pass

    def is_camera_open(self):
        return self.cap is not None and self.cap.isOpened()

    def open_camera(self, log_success=True):
        self.close_camera()
        cv2 = self.cv2
        for candidate in self.camera_candidates():
            if (
                not str(candidate).isdigit()
                and str(candidate).startswith('/dev/')
                and not os.path.exists(str(candidate))
            ):
                continue
            device_arg = int(candidate) if str(candidate).isdigit() else candidate
            cap = cv2.VideoCapture(device_arg, cv2.CAP_V4L2)
            self.configure_capture(cap)
            if not cap.isOpened():
                cap.release()
                continue

            frame_ok = False
            for _ in range(3):
                ok, frame = cap.read()
                if ok and frame is not None:
                    frame_ok = True
                    break
                time.sleep(0.03)
            if not frame_ok:
                cap.release()
                continue

            self.cap = cap
            self.active_device = str(candidate)
            self.read_fail_streak = 0
            if log_success:
                self.get_logger().info(
                    f'OPENCV_CAMERA_OPENED | active_device={self.active_device} '
                    f'requested_device={self.device}'
                )
            return True

        self.next_open_attempt_mono_sec = time.monotonic() + max(0.1, self.retry_open_period_sec)
        self.get_logger().warn(
            f'OPENCV_CAMERA_OPEN_FAILED | requested_device={self.device} '
            f'fallback_devices={self.fallback_devices}',
            throttle_duration_sec=2.0,
        )
        return False

    def capture_loop(self):
        while not self.stop_threads:
            if not self.camera_process_enabled:
                if self.is_camera_open():
                    self.close_camera()
                time.sleep(0.1)
                continue
            if not self.is_camera_open():
                self.fail_count += 1
                now = time.monotonic()
                if now >= self.next_open_attempt_mono_sec:
                    self.open_camera()
                time.sleep(0.05)
                continue
            ok, frame = self.cap.read()
            if not ok or frame is None:
                self.fail_count += 1
                self.read_fail_streak += 1
                max_streak = max(5, int(self.camera_fps))
                if self.read_fail_streak >= max_streak:
                    self.get_logger().warn(
                        f'OPENCV_CAMERA_READ_FAILED | active_device={self.active_device} '
                        f'fail_streak={self.read_fail_streak}; reopening',
                        throttle_duration_sec=2.0,
                    )
                    self.close_camera()
                    self.next_open_attempt_mono_sec = 0.0
                time.sleep(0.02)
                continue
            self.read_fail_streak = 0
            capture_sec = self.get_clock().now().nanoseconds * 1e-9
            capture_mono_sec = time.monotonic()
            pose, pose_error_sec = self.lookup_capture_pose(capture_sec)
            if pose is None and self.require_capture_pose:
                self.drop_count += 1
                self.missing_pose_drop_count += 1
                self.get_logger().warn(
                    'OBSERVATION_MISSING_POSE_DROPPED | '
                    f'robot_id={self.robot_name or "unknown"} '
                    f'seq_candidate={self.latest_seq + 1} '
                    f'capture_ros_sec={capture_sec:.6f} '
                    f'pose_topic={self.pose_topic} '
                    f'max_error_sec={self.max_pose_interpolation_error_sec:.3f}',
                    throttle_duration_sec=2.0,
                )
                continue
            if pose is None:
                pose = PoseSample(capture_sec, 0.0, 0.0, 0.0)
                pose_error_sec = float('inf')
            with self.frame_condition:
                if self.latest_frame is not None and self.latest_seq != self.sent_seq:
                    self.drop_count += 1
                self.latest_frame = frame
                self.latest_capture_sec = capture_sec
                self.latest_capture_mono_sec = capture_mono_sec
                self.latest_seq += 1
                self.latest_observation_meta = build_observation_metadata(
                    robot_id=self.robot_name or f'robot_domain_{os.environ.get("ROS_DOMAIN_ID", "unknown")}',
                    boot_id=self.boot_id,
                    sequence=self.latest_seq,
                    role=self.current_role,
                    role_epoch=self.current_role_epoch,
                    frame_id=self.frame_id,
                    camera_hfov_deg=self.camera_hfov_deg,
                    capture_ros_sec=capture_sec,
                    capture_wall_sec=time.time(),
                    capture_mono_sec=capture_mono_sec,
                    pose=pose,
                    pose_time_error_sec=pose_error_sec,
                    image_width=0,
                    image_height=0,
                    calibration_id=self.camera_calibration_id,
                )
                self.rx_count += 1
                self.frame_condition.notify()

    def wait_latest_frame(self, next_allowed: float):
        with self.frame_condition:
            while not self.stop_threads:
                if not self.camera_upload_enabled:
                    self.frame_condition.wait(timeout=0.1)
                    continue
                now = time.monotonic()
                if (
                    self.latest_frame is not None
                    and self.latest_seq != self.sent_seq
                    and now >= next_allowed
                ):
                    self.sent_seq = self.latest_seq
                    return (
                        self.latest_frame.copy(),
                        self.latest_capture_sec,
                        self.latest_capture_mono_sec,
                        self.latest_seq,
                        dict(getattr(self, 'latest_observation_meta', {}) or {}),
                    )
                timeout = 0.1
                if self.latest_frame is not None and now < next_allowed:
                    timeout = max(0.001, min(0.1, next_allowed - now))
                self.frame_condition.wait(timeout=timeout)
        return None, 0.0, 0.0, 0, {}

    def prepare_frame_for_upload(self, frame):
        import cv2
        import numpy as np

        source_h, source_w = frame.shape[:2]
        target_w = int(self.send_width)
        target_h = int(self.send_height)
        metadata = {
            'source_width': int(source_w),
            'source_height': int(source_h),
            'display_width': int(source_w),
            'display_height': int(source_h),
            'letterbox_scale': 1.0,
            'letterbox_pad_x': 0,
            'letterbox_pad_y': 0,
            'letterbox_content_width': int(source_w),
            'letterbox_content_height': int(source_h),
            'letterbox_mode': 'native',
        }
        if target_w <= 0 or target_h <= 0:
            return frame, metadata
        if source_w == target_w and source_h == target_h:
            metadata.update({
                'display_width': target_w,
                'display_height': target_h,
                'letterbox_mode': 'native',
            })
            return frame, metadata

        source_ratio = source_w / max(1.0, float(source_h))
        target_ratio = target_w / max(1.0, float(target_h))
        if math.isclose(source_ratio, target_ratio, rel_tol=1.0e-3, abs_tol=1.0e-3):
            interpolation = (
                cv2.INTER_AREA
                if target_w < source_w or target_h < source_h
                else cv2.INTER_LINEAR
            )
            resized = cv2.resize(frame, (target_w, target_h), interpolation=interpolation)
            metadata.update({
                'display_width': target_w,
                'display_height': target_h,
                'letterbox_scale': target_w / max(1.0, float(source_w)),
                'letterbox_content_width': target_w,
                'letterbox_content_height': target_h,
                'letterbox_mode': 'resize_full_frame',
            })
            return resized, metadata

        scale = min(target_w / float(source_w), target_h / float(source_h))
        content_w = max(1, int(round(source_w * scale)))
        content_h = max(1, int(round(source_h * scale)))
        interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
        resized = cv2.resize(frame, (content_w, content_h), interpolation=interpolation)
        pad_x = max(0, (target_w - content_w) // 2)
        pad_y = max(0, (target_h - content_h) // 2)
        right = max(0, target_w - content_w - pad_x)
        bottom = max(0, target_h - content_h - pad_y)
        color = int(max(0, min(255, self.letterbox_color)))
        letterboxed = cv2.copyMakeBorder(
            resized,
            pad_y,
            bottom,
            pad_x,
            right,
            cv2.BORDER_CONSTANT,
            value=(color, color, color),
        )
        if letterboxed.shape[1] != target_w or letterboxed.shape[0] != target_h:
            canvas = np.full((target_h, target_w, 3), color, dtype=frame.dtype)
            canvas[pad_y:pad_y + content_h, pad_x:pad_x + content_w] = resized
            letterboxed = canvas
        metadata.update({
            'display_width': target_w,
            'display_height': target_h,
            'letterbox_scale': float(scale),
            'letterbox_pad_x': int(pad_x),
            'letterbox_pad_y': int(pad_y),
            'letterbox_content_width': int(content_w),
            'letterbox_content_height': int(content_h),
            'letterbox_mode': 'letterbox_full_frame',
        })
        return letterboxed, metadata

    def encode_jpeg(self, frame) -> bytes:
        import cv2

        resize_start = time.monotonic()
        frame, resize_meta = self.prepare_frame_for_upload(frame)
        resize_ms = (time.monotonic() - resize_start) * 1000.0
        quality = max(1, min(100, int(self.jpeg_quality)))
        encode_start = time.monotonic()
        ok, buf = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        if not ok:
            raise RuntimeError('cv2.imencode(.jpg) failed')
        encode_ms = (time.monotonic() - encode_start) * 1000.0
        return bytes(buf), frame.shape[1], frame.shape[0], resize_ms, encode_ms, resize_meta

    def http_worker_loop(self):
        next_allowed = 0.0
        while not self.stop_threads:
            if not self.camera_upload_enabled:
                time.sleep(0.1)
                next_allowed = 0.0
                continue
            rate = self._current_upload_rate_hz()
            period = 1.0 / rate if rate > 0.0 else 0.0
            frame, capture_sec, capture_mono_sec, seq, meta = self.wait_latest_frame(next_allowed)
            if frame is None:
                continue
            next_allowed = time.monotonic() + period
            try:
                self.post_frame(frame, capture_sec, capture_mono_sec, seq, meta)
            except Exception as exc:
                self.fail_count += 1
                self.reset_http_session()
                self.get_logger().warn(
                    f'HTTP YOLO send failed: {exc}',
                    throttle_duration_sec=2.0,
                )
                self.log_periodic()

    def reset_http_session(self):
        try:
            self.http.close()
        except Exception:
            pass
        self.http = requests.Session()
        adapter = requests.adapters.HTTPAdapter(
            pool_connections=1,
            pool_maxsize=max(1, self.http_worker_count),
            max_retries=0,
            pool_block=True,
        )
        self.http.mount('http://', adapter)
        self.http.mount('https://', adapter)
        self.http.headers.update({'Connection': 'keep-alive'})

    def post_frame(self, frame, capture_sec: float, capture_mono_sec: float, seq: int, meta: dict):
        frame_age_before_send = time.monotonic() - capture_mono_sec if capture_mono_sec > 0.0 else 0.0
        if self.max_frame_age_sec > 0.0 and frame_age_before_send > self.max_frame_age_sec:
            self.drop_count += 1
            self.get_logger().warn(
                f'dropped stale frame before send: age={frame_age_before_send:.3f}s '
                f'limit={self.max_frame_age_sec:.3f}s seq={seq}',
                throttle_duration_sec=2.0,
            )
            self.log_periodic()
            return

        jpeg, w, h, resize_ms, encode_ms, resize_meta = self.encode_jpeg(frame)
        self.resize_ms_samples.append(resize_ms)
        self.encode_ms_samples.append(encode_ms)
        self.jpeg_size_samples.append(float(len(jpeg)))
        if not self._consume_upload_budget(len(jpeg)):
            self.drop_count += 1
            self.get_logger().warn(
                'OPENCV_HTTP_YOLO_TX_BUDGET_DROP | '
                f'robot={self.robot_name or "unknown"} role={self.current_role} '
                f'jpeg_bytes={len(jpeg)} budget_mbps={self._current_upload_budget_mbps():.2f} '
                f'seq={seq}',
                throttle_duration_sec=2.0,
            )
            self.log_periodic()
            return
        files = {'image': ('frame.jpg', jpeg, 'image/jpeg')}
        data = dict(meta or {})
        data.update({
            'frame_id': self.frame_id,
            'capture_ros_sec': f'{capture_sec:.9f}' if capture_sec > 0.0 else '',
            'robot_frame_age_ms_at_send': f'{frame_age_before_send * 1000.0:.3f}',
            'image_width': str(int(w)),
            'image_height': str(int(h)),
        })
        data.update({
            key: f'{value:.9f}' if isinstance(value, float) else str(value)
            for key, value in resize_meta.items()
        })
        request_start = time.monotonic()
        self.inflight_requests += 1
        try:
            resp = self.http.post(
                self.server_url,
                files=files,
                data=data,
                timeout=(self.connect_timeout_sec, self.read_timeout_sec),
            )
        finally:
            self.inflight_requests = max(0, self.inflight_requests - 1)
        roundtrip_sec = time.monotonic() - request_start
        self.sent_count += 1
        self.tx_bytes_window.append((time.monotonic(), int(len(jpeg))))
        self.rtt_ms_samples.append(roundtrip_sec * 1000.0)
        resp.raise_for_status()
        payload = resp.json()
        if not payload.get('ok', True) or payload.get('stale', False):
            self.drop_count += 1
            self.get_logger().warn(
                f'dropped server-rejected YOLO frame: stale={payload.get("stale", False)} '
                f'error={payload.get("error", "unknown")} seq={seq}',
                throttle_duration_sec=2.0,
            )
            self.log_periodic()
            return

        total_frame_age_sec = time.monotonic() - capture_mono_sec if capture_mono_sec > 0.0 else roundtrip_sec
        self.capture_age_ms_samples.append(total_frame_age_sec * 1000.0)
        if self.max_http_roundtrip_sec > 0.0 and roundtrip_sec > self.max_http_roundtrip_sec:
            self.drop_count += 1
            self.get_logger().warn(
                f'dropped stale YOLO response: roundtrip={roundtrip_sec:.3f}s '
                f'limit={self.max_http_roundtrip_sec:.3f}s seq={seq}',
                throttle_duration_sec=2.0,
            )
            self.log_periodic()
            return
        if self.max_frame_age_sec > 0.0 and total_frame_age_sec > self.max_frame_age_sec:
            self.drop_count += 1
            self.get_logger().warn(
                f'dropped stale YOLO result: frame_age={total_frame_age_sec:.3f}s '
                f'limit={self.max_frame_age_sec:.3f}s seq={seq}',
                throttle_duration_sec=2.0,
            )
            self.log_periodic()
            return

        detections = payload.get('detections', [])
        with self.publish_lock:
            if seq <= self.published_seq:
                self.drop_count += 1
                self.get_logger().warn(
                    f'dropped out-of-order YOLO result: seq={seq} '
                    f'latest_published_seq={self.published_seq}',
                    throttle_duration_sec=2.0,
                )
                self.log_periodic()
                return
            self.published_seq = int(seq)

        if detections or self.publish_empty_detections:
            payload['image_width'] = int(payload.get('image_width') or w)
            payload['image_height'] = int(payload.get('image_height') or h)
            payload['ros_frame_id'] = self.frame_id
            payload['capture_ros_sec'] = capture_sec
            payload.setdefault('robot_id', data.get('robot_id', self.robot_name))
            payload.setdefault('boot_id', data.get('boot_id', self.boot_id))
            payload.setdefault('sequence', int(seq))
            payload.setdefault('role', data.get('role', self.current_role))
            payload.setdefault('role_epoch', int(data.get('role_epoch', self.current_role_epoch) or 0))
            payload.setdefault('capture_pose', {
                'x': float(data.get('capture_pose_x', 0.0)),
                'y': float(data.get('capture_pose_y', 0.0)),
                'yaw': float(data.get('capture_pose_yaw', 0.0)),
                'stamp_sec': float(data.get('capture_pose_stamp_sec', capture_sec or 0.0)),
            })
            payload.setdefault('pose_time_error_ms', float(data.get('pose_time_error_ms', -1.0)))
            payload.setdefault('capture_to_send_delay_ms', float(data.get('capture_to_send_delay_ms', -1.0)))
            payload.setdefault('camera_hfov_deg', float(data.get('camera_hfov_deg', self.camera_hfov_deg)))
            payload['robot_bridge_wall_sec'] = time.time()
            payload['robot_frame_age_ms'] = total_frame_age_sec * 1000.0
            payload['http_roundtrip_ms'] = roundtrip_sec * 1000.0
            payload['camera_encode_ms'] = encode_ms
            payload['camera_resize_ms'] = resize_ms
            payload['letterbox'] = resize_meta
            payload['robot_frame_seq'] = int(seq)

            if self._current_role_allows_publish():
                msg = String()
                msg.data = json.dumps(payload, separators=(',', ':'))
                self.pub.publish(msg)

        self.ok_count += 1
        self.ok_timestamps.append(time.monotonic())
        self.log_periodic()

    def _current_upload_rate_hz(self) -> float:
        role = str(self.current_role or '').strip().upper()
        if role in self.active_roles:
            return max(0.1, float(self.active_max_rate_hz))
        return max(0.1, float(self.standby_max_rate_hz))

    def _current_upload_budget_mbps(self) -> float:
        role = str(self.current_role or '').strip().upper()
        if role in self.active_roles:
            return max(0.0, float(self.active_max_upload_mbps))
        return max(0.0, float(self.standby_max_upload_mbps))

    def _consume_upload_budget(self, byte_count: int) -> bool:
        budget_mbps = self._current_upload_budget_mbps()
        if budget_mbps <= 0.0:
            return True
        now = time.monotonic()
        elapsed = max(0.0, now - self.tx_budget_last_mono_sec)
        self.tx_budget_last_mono_sec = now
        bytes_per_sec = budget_mbps * 125000.0
        burst_capacity = max(65536.0, bytes_per_sec * 2.0)
        self.tx_budget_tokens = min(
            burst_capacity,
            self.tx_budget_tokens + elapsed * bytes_per_sec,
        )
        if self.tx_budget_tokens < float(byte_count):
            return False
        self.tx_budget_tokens -= float(byte_count)
        return True

    def _current_role_allows_publish(self) -> bool:
        return bool(self.risk_observation_publish_enabled)

    def log_periodic(self):
        now = time.time()
        if now - self.last_log_wall_sec < self.log_period_sec:
            return
        self.last_log_wall_sec = now
        output_fps = 0.0
        if len(self.ok_timestamps) >= 2:
            dt = self.ok_timestamps[-1] - self.ok_timestamps[0]
            output_fps = (len(self.ok_timestamps) - 1) / dt if dt > 1e-6 else 0.0
        resize = self._sample_summary(self.resize_ms_samples)
        encode = self._sample_summary(self.encode_ms_samples)
        rtt = self._sample_summary(self.rtt_ms_samples)
        age = self._sample_summary(self.capture_age_ms_samples)
        jpeg = self._sample_summary(self.jpeg_size_samples)
        tx_mbps = self._recent_tx_mbps(now=time.monotonic())
        queue_depth = 1 if self.latest_frame is not None and self.latest_seq != self.sent_seq else 0
        self.get_logger().info(
            'CAMERA_NETWORK_STATUS | '
            f'robot={self.robot_name or "unknown"} '
            f'role={self.current_role} '
            f'capture_resolution={self.width}x{self.height} '
            f'send_resolution={self.send_width}x{self.send_height} '
            f'jpeg_quality={self.jpeg_quality} '
            f'capture_fps={self.camera_fps:.2f} '
            f'upload_fps={output_fps:.2f} '
            f'bitrate_mbps={tx_mbps:.3f} '
            f'queue_depth={queue_depth} '
            f'inflight_requests={self.inflight_requests} '
            f'dropped_frames={self.drop_count} '
            'http_session_reused=true',
        )
        self.get_logger().info(
            f'OPENCV_HTTP_YOLO_STATUS | captured={self.rx_count} sent={self.sent_count} '
            f'ok={self.ok_count} fail={self.fail_count} replaced={self.drop_count} '
            f'missing_pose={self.missing_pose_drop_count} '
            f'role={self.current_role} upload_rate={self._current_upload_rate_hz():.2f} '
            f'publish_enabled={self._current_role_allows_publish()} '
            f'output_fps={output_fps:.2f} '
            f'tx_mbps={tx_mbps:.3f} jpeg_bytes_p50={jpeg[0]:.0f} '
            f'p95={jpeg[1]:.0f} max={jpeg[2]:.0f} '
            f'camera_resize_ms_p50={resize[0]:.1f} p95={resize[1]:.1f} max={resize[2]:.1f} '
            f'camera_encode_ms_p50={encode[0]:.1f} p95={encode[1]:.1f} max={encode[2]:.1f} '
            f'network_rtt_ms_p50={rtt[0]:.1f} p95={rtt[1]:.1f} max={rtt[2]:.1f} '
            f'end_to_end_frame_age_ms_p50={age[0]:.1f} p95={age[1]:.1f} max={age[2]:.1f} '
            f'out={self.output_topic}'
        )

    @staticmethod
    def _sample_summary(samples):
        values = list(samples)
        if not values:
            return -1.0, -1.0, -1.0
        ordered = sorted(float(v) for v in values)
        p50 = statistics.median(ordered)
        p95_index = min(len(ordered) - 1, int(round(0.95 * (len(ordered) - 1))))
        return float(p50), float(ordered[p95_index]), float(max(ordered))

    def _recent_tx_mbps(self, *, now: float) -> float:
        cutoff = now - 10.0
        while self.tx_bytes_window and self.tx_bytes_window[0][0] < cutoff:
            self.tx_bytes_window.popleft()
        if not self.tx_bytes_window:
            return 0.0
        span = max(1.0, now - self.tx_bytes_window[0][0])
        total_bytes = sum(item[1] for item in self.tx_bytes_window)
        return (float(total_bytes) * 8.0) / (span * 1_000_000.0)

    def destroy_node(self):
        self.stop_threads = True
        with self.frame_condition:
            self.frame_condition.notify_all()
        threads = [getattr(self, 'capture_thread', None)]
        threads.extend(getattr(self, 'worker_threads', []))
        for thread in threads:
            if thread is not None and thread.is_alive():
                thread.join(timeout=max(2.0, self.timeout_sec + 0.5))
        try:
            self.http.close()
        except Exception:
            pass
        try:
            self.close_camera()
        except Exception:
            pass
        super().destroy_node()


def main():
    rclpy.init()
    node = OpenCVCameraToFlaskYolo()
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
