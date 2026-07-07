#!/usr/bin/env python3

from __future__ import annotations

import json
import os
import threading
import time
from collections import deque

import requests

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy, HistoryPolicy, ReliabilityPolicy
from std_msgs.msg import String

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
        self.width = int(self.declare_parameter('width', 320).value)
        self.height = int(self.declare_parameter('height', 240).value)
        self.send_width = int(self.declare_parameter('send_width', self.width).value)
        self.send_height = int(self.declare_parameter('send_height', self.height).value)
        self.camera_fps = float(self.declare_parameter('camera_fps', 15.0).value)
        self.buffer_size = int(self.declare_parameter('buffer_size', 1).value)
        self.fourcc = str(self.declare_parameter('fourcc', 'MJPG').value).strip()

        self.server_url = str(
            self.declare_parameter('server_url', 'http://127.0.0.1:5005/detect').value
        ).strip()
        self.output_topic = self.declare_parameter('output_topic', '/risk/yolo_detections').value
        self.max_rate_hz = float(self.declare_parameter('max_rate_hz', 5.0).value)
        self.http_worker_count = int(self.declare_parameter('http_worker_count', 1).value)
        self.jpeg_quality = int(self.declare_parameter('jpeg_quality', 60).value)
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
        self.http = requests.Session()

        self.frame_condition = threading.Condition()
        self.latest_frame = None
        self.latest_capture_sec = 0.0
        self.latest_capture_mono_sec = 0.0
        self.latest_seq = 0
        self.sent_seq = 0
        self.publish_lock = threading.Lock()
        self.published_seq = 0
        self.stop_threads = False

        self.rx_count = 0
        self.sent_count = 0
        self.ok_count = 0
        self.fail_count = 0
        self.drop_count = 0
        self.last_log_wall_sec = 0.0
        self.ok_timestamps = deque(maxlen=120)
        self.active_device = ''
        self.next_open_attempt_mono_sec = 0.0
        self.read_fail_streak = 0

        import cv2

        self.cv2 = cv2
        self.cap = None
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
            f'http_workers={max(1, self.http_worker_count)} '
            f'jpeg_quality={self.jpeg_quality} timeout={self.timeout_sec:.2f}s '
            f'connect_timeout={self.connect_timeout_sec:.2f}s read_timeout={self.read_timeout_sec:.2f}s '
            f'max_http_roundtrip={self.max_http_roundtrip_sec:.2f}s '
            f'max_frame_age={self.max_frame_age_sec:.2f}s '
            f'latest_frame_only=true ros_image_publish=false'
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
            with self.frame_condition:
                if self.latest_frame is not None and self.latest_seq != self.sent_seq:
                    self.drop_count += 1
                self.latest_frame = frame
                self.latest_capture_sec = capture_sec
                self.latest_capture_mono_sec = capture_mono_sec
                self.latest_seq += 1
                self.rx_count += 1
                self.frame_condition.notify()

    def wait_latest_frame(self, next_allowed: float):
        with self.frame_condition:
            while not self.stop_threads:
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
                    )
                timeout = 0.1
                if self.latest_frame is not None and now < next_allowed:
                    timeout = max(0.001, min(0.1, next_allowed - now))
                self.frame_condition.wait(timeout=timeout)
        return None, 0.0, 0.0, 0

    def encode_jpeg(self, frame) -> bytes:
        import cv2

        if self.send_width > 0 and self.send_height > 0:
            h, w = frame.shape[:2]
            if w != self.send_width or h != self.send_height:
                frame = cv2.resize(
                    frame,
                    (int(self.send_width), int(self.send_height)),
                    interpolation=cv2.INTER_AREA if self.send_width < w or self.send_height < h else cv2.INTER_LINEAR,
                )
        quality = max(1, min(100, int(self.jpeg_quality)))
        ok, buf = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        if not ok:
            raise RuntimeError('cv2.imencode(.jpg) failed')
        return bytes(buf), frame.shape[1], frame.shape[0]

    def http_worker_loop(self):
        period = 1.0 / self.max_rate_hz if self.max_rate_hz > 0.0 else 0.0
        next_allowed = 0.0
        while not self.stop_threads:
            frame, capture_sec, capture_mono_sec, seq = self.wait_latest_frame(next_allowed)
            if frame is None:
                continue
            next_allowed = time.monotonic() + period
            try:
                self.post_frame(frame, capture_sec, capture_mono_sec, seq)
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

    def post_frame(self, frame, capture_sec: float, capture_mono_sec: float, seq: int):
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

        jpeg, w, h = self.encode_jpeg(frame)
        self.sent_count += 1
        files = {'image': ('frame.jpg', jpeg, 'image/jpeg')}
        data = {
            'frame_id': self.frame_id,
            'capture_ros_sec': f'{capture_sec:.9f}' if capture_sec > 0.0 else '',
            'capture_wall_sec': f'{time.time() - frame_age_before_send:.9f}',
            'robot_frame_age_ms_at_send': f'{frame_age_before_send * 1000.0:.3f}',
        }
        request_start = time.monotonic()
        resp = self.http.post(
            self.server_url,
            files=files,
            data=data,
            timeout=(self.connect_timeout_sec, self.read_timeout_sec),
        )
        roundtrip_sec = time.monotonic() - request_start
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
            payload['robot_bridge_wall_sec'] = time.time()
            payload['robot_frame_age_ms'] = total_frame_age_sec * 1000.0
            payload['http_roundtrip_ms'] = roundtrip_sec * 1000.0
            payload['robot_frame_seq'] = int(seq)

            msg = String()
            msg.data = json.dumps(payload, separators=(',', ':'))
            self.pub.publish(msg)

        self.ok_count += 1
        self.ok_timestamps.append(time.monotonic())
        self.log_periodic()

    def log_periodic(self):
        now = time.time()
        if now - self.last_log_wall_sec < self.log_period_sec:
            return
        self.last_log_wall_sec = now
        output_fps = 0.0
        if len(self.ok_timestamps) >= 2:
            dt = self.ok_timestamps[-1] - self.ok_timestamps[0]
            output_fps = (len(self.ok_timestamps) - 1) / dt if dt > 1e-6 else 0.0
        self.get_logger().info(
            f'OPENCV_HTTP_YOLO_STATUS | captured={self.rx_count} sent={self.sent_count} '
            f'ok={self.ok_count} fail={self.fail_count} replaced={self.drop_count} '
            f'output_fps={output_fps:.2f} out={self.output_topic}'
        )

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
