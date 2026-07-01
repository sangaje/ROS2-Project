#!/usr/bin/env python3

from __future__ import annotations

import json
import threading
import time
from collections import deque

import requests

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

from tb3_flask_yolo_bridge.ros_param_helpers import FlexibleParameterNodeMixin


class OpenCVCameraToFlaskYolo(FlexibleParameterNodeMixin, Node):
    """
    Direct robot-camera -> HTTP JPEG -> PC Flask YOLO -> compact ROS detection JSON.

    This intentionally does not publish ROS image topics. The only ROS output is
    /risk/yolo_detections, which keeps DDS/network load tiny on TurtleBot3/Pi4.
    """

    def __init__(self):
        super().__init__('opencv_camera_to_flask_yolo')

        self.device = self.declare_parameter('device', '/dev/video0').value
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
        self.jpeg_quality = int(self.declare_parameter('jpeg_quality', 60).value)
        self.timeout_sec = float(self.declare_parameter('timeout_sec', 1.0).value)
        self.log_period_sec = float(self.declare_parameter('log_period_sec', 2.0).value)
        self.publish_empty_detections = self.declare_bool_parameter('publish_empty_detections', True)

        self.pub = self.create_publisher(String, self.output_topic, 10)
        self.http = requests.Session()

        self.frame_condition = threading.Condition()
        self.latest_frame = None
        self.latest_capture_sec = 0.0
        self.latest_seq = 0
        self.sent_seq = 0
        self.stop_threads = False

        self.rx_count = 0
        self.sent_count = 0
        self.ok_count = 0
        self.fail_count = 0
        self.drop_count = 0
        self.last_log_wall_sec = 0.0
        self.ok_timestamps = deque(maxlen=120)

        import cv2

        device_arg = int(self.device) if str(self.device).isdigit() else self.device
        self.cap = cv2.VideoCapture(device_arg)
        if self.buffer_size > 0:
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, int(self.buffer_size))
        if self.fourcc:
            code = self.fourcc.upper()[:4]
            if len(code) == 4:
                self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*code))
        if self.width > 0:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(self.width))
        if self.height > 0:
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(self.height))
        if self.camera_fps > 0.0:
            self.cap.set(cv2.CAP_PROP_FPS, float(self.camera_fps))

        self.capture_thread = threading.Thread(
            target=self.capture_loop,
            name='opencv_http_yolo_capture_latest_frame',
            daemon=True,
        )
        self.worker_thread = threading.Thread(
            target=self.http_worker_loop,
            name='opencv_http_yolo_latest_frame_sender',
            daemon=True,
        )
        self.capture_thread.start()
        self.worker_thread.start()

        self.get_logger().info(
            f'OPENCV_CAMERA_TO_FLASK_YOLO_READY | device={self.device} opened={self.cap.isOpened()} '
            f'capture_request={self.width}x{self.height} send={self.send_width}x{self.send_height} '
            f'camera_fps={self.camera_fps:.1f} fourcc={self.fourcc or "default"} '
            f'server={self.server_url} out={self.output_topic} rate={self.max_rate_hz:.2f}Hz '
            f'jpeg_quality={self.jpeg_quality} latest_frame_only=true ros_image_publish=false'
        )

    def capture_loop(self):
        while not self.stop_threads:
            if not self.cap.isOpened():
                self.fail_count += 1
                time.sleep(0.05)
                continue
            ok, frame = self.cap.read()
            if not ok or frame is None:
                self.fail_count += 1
                time.sleep(0.02)
                continue
            capture_sec = self.get_clock().now().nanoseconds * 1e-9
            with self.frame_condition:
                if self.latest_frame is not None and self.latest_seq != self.sent_seq:
                    self.drop_count += 1
                self.latest_frame = frame
                self.latest_capture_sec = capture_sec
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
                    return self.latest_frame.copy(), self.latest_capture_sec, self.latest_seq
                timeout = 0.5
                if self.latest_frame is not None and now < next_allowed:
                    timeout = max(0.001, min(0.5, next_allowed - now))
                self.frame_condition.wait(timeout=timeout)
        return None, 0.0, 0

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
            frame, capture_sec, seq = self.wait_latest_frame(next_allowed)
            if frame is None:
                continue
            next_allowed = time.monotonic() + period
            try:
                self.post_frame(frame, capture_sec, seq)
            except Exception as exc:
                self.fail_count += 1
                self.get_logger().warn(
                    f'HTTP YOLO send failed: {exc}',
                    throttle_duration_sec=2.0,
                )
                self.log_periodic()

    def post_frame(self, frame, capture_sec: float, seq: int):
        jpeg, w, h = self.encode_jpeg(frame)
        self.sent_count += 1
        files = {'image': ('frame.jpg', jpeg, 'image/jpeg')}
        data = {
            'frame_id': self.frame_id,
            'capture_ros_sec': f'{capture_sec:.9f}' if capture_sec > 0.0 else '',
        }
        resp = self.http.post(self.server_url, files=files, data=data, timeout=self.timeout_sec)
        resp.raise_for_status()
        payload = resp.json()

        detections = payload.get('detections', [])
        if detections or self.publish_empty_detections:
            payload['image_width'] = int(payload.get('image_width') or w)
            payload['image_height'] = int(payload.get('image_height') or h)
            payload['ros_frame_id'] = self.frame_id
            payload['capture_ros_sec'] = capture_sec
            payload['robot_bridge_wall_sec'] = time.time()
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
        for thread in (getattr(self, 'capture_thread', None), getattr(self, 'worker_thread', None)):
            if thread is not None and thread.is_alive():
                thread.join(timeout=max(2.0, self.timeout_sec + 0.5))
        try:
            self.http.close()
        except Exception:
            pass
        try:
            self.cap.release()
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
