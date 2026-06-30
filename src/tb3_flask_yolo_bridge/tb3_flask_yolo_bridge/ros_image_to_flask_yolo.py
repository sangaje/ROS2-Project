#!/usr/bin/env python3

from __future__ import annotations

import json
import threading
import time
from collections import deque

import numpy as np
import requests

import rclpy
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from sensor_msgs.msg import CompressedImage, Image
from std_msgs.msg import String

from tb3_flask_yolo_bridge.ros_param_helpers import FlexibleParameterNodeMixin


class RosImageToFlaskYolo(FlexibleParameterNodeMixin, Node):
    def __init__(self):
        super().__init__('ros_image_to_flask_yolo')
        self.image_topic = self.declare_parameter('image_topic', '/camera/image_raw').value
        self.input_type = str(self.declare_parameter('input_type', 'raw').value).strip().lower()
        self.server_url = self.declare_parameter('server_url', 'http://127.0.0.1:5005/detect').value
        self.output_topic = self.declare_parameter('output_topic', '/risk/yolo_detections').value
        self.max_rate_hz = float(self.declare_parameter('max_rate_hz', 10.0).value)
        self.jpeg_quality = int(self.declare_parameter('jpeg_quality', 75).value)
        self.timeout_sec = float(self.declare_parameter('timeout_sec', 1.0).value)
        self.frame_id = self.declare_parameter('frame_id', '').value
        self.publish_debug_image = self.declare_bool_parameter('publish_debug_image', True)
        self.debug_image_topic = self.declare_parameter('debug_image_topic', '/risk/debug_yolo_image').value
        self.log_period_sec = float(self.declare_parameter('log_period_sec', 2.0).value)

        self.last_log_wall_sec = 0.0
        self.rx_count = 0
        self.sent_count = 0
        self.ok_count = 0
        self.fail_count = 0
        self.dropped_count = 0
        self.ok_timestamps = deque(maxlen=120)
        self.http = requests.Session()
        self.worker_condition = threading.Condition()
        self.pending_frame = None
        self.worker_stop = False

        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self.pub = self.create_publisher(String, self.output_topic, 10)
        self.debug_pub = self.create_publisher(Image, self.debug_image_topic, 10)
        if self.input_type in ('compressed', 'jpeg', 'jpg'):
            self.sub = self.create_subscription(CompressedImage, self.image_topic, self.on_compressed, sensor_qos)
        else:
            self.sub = self.create_subscription(Image, self.image_topic, self.on_raw, sensor_qos)

        self.worker_thread = threading.Thread(
            target=self.worker_loop,
            name='flask_yolo_latest_frame_worker',
            daemon=True,
        )
        self.worker_thread.start()

        self.get_logger().info(
            f'ROS_IMAGE_TO_FLASK_YOLO_MULTITHREADED_READY | input={self.image_topic} type={self.input_type} '
            f'server={self.server_url} out={self.output_topic} rate={self.max_rate_hz:.2f}Hz jpeg_quality={self.jpeg_quality} '
            f'latest_frame_only=true debug_image={self.publish_debug_image} debug_topic={self.debug_image_topic}'
        )

    def enqueue_latest(self, kind, msg):
        self.rx_count += 1
        with self.worker_condition:
            if self.pending_frame is not None:
                self.dropped_count += 1
            self.pending_frame = (kind, msg)
            self.worker_condition.notify()

    def on_raw(self, msg: Image):
        self.enqueue_latest('raw', msg)

    def on_compressed(self, msg: CompressedImage):
        self.enqueue_latest('compressed', msg)

    def worker_loop(self):
        period = 1.0 / self.max_rate_hz if self.max_rate_hz > 0.0 else 0.0
        next_allowed = 0.0
        while True:
            item = None
            with self.worker_condition:
                while not self.worker_stop:
                    now = time.monotonic()
                    if self.pending_frame is not None and now >= next_allowed:
                        item = self.pending_frame
                        self.pending_frame = None
                        next_allowed = now + period
                        break
                    timeout = 0.5
                    if self.pending_frame is not None and period > 0.0:
                        timeout = max(0.001, min(0.5, next_allowed - now))
                    self.worker_condition.wait(timeout=timeout)
                if self.worker_stop:
                    return

            if item is None:
                continue
            kind, msg = item
            try:
                if kind == 'compressed':
                    self.process_compressed(msg)
                else:
                    self.process_raw(msg)
            except Exception as exc:
                self.fail_count += 1
                self.get_logger().warn(
                    f'{kind} image worker failed: {exc}',
                    throttle_duration_sec=2.0,
                )
                self.log_periodic()

    def process_raw(self, msg: Image):
        frame = self.image_msg_to_bgr8(msg)
        jpeg = self.encode_jpeg(frame)
        self.post_and_publish(
            jpeg, msg.header.frame_id, int(msg.width), int(msg.height), frame,
            self.stamp_to_sec(msg.header.stamp),
        )

    def process_compressed(self, msg: CompressedImage):
        try:
            frame = self.decode_jpeg(bytes(msg.data)) if self.publish_debug_image else None
            height = int(frame.shape[0]) if frame is not None else 0
            width = int(frame.shape[1]) if frame is not None else 0
            self.post_and_publish(
                bytes(msg.data), msg.header.frame_id, width, height, frame,
                self.stamp_to_sec(msg.header.stamp),
            )
        except Exception as exc:
            raise RuntimeError(f'compressed image send failed: {exc}') from exc

    def image_msg_to_bgr8(self, msg: Image):
        enc = msg.encoding.lower()
        h = int(msg.height)
        w = int(msg.width)
        step = int(msg.step)
        raw = np.frombuffer(msg.data, dtype=np.uint8)
        if raw.size < h * step:
            raise ValueError(f'buffer too small raw={raw.size} expected={h * step}')
        rows = raw[:h * step].reshape((h, step))
        if enc in ('bgr8', '8uc3'):
            return rows[:, :w * 3].reshape((h, w, 3)).copy()
        if enc == 'rgb8':
            return rows[:, :w * 3].reshape((h, w, 3))[:, :, ::-1].copy()
        if enc == 'bgra8':
            return rows[:, :w * 4].reshape((h, w, 4))[:, :, :3].copy()
        if enc == 'rgba8':
            return rows[:, :w * 4].reshape((h, w, 4))[:, :, [2, 1, 0]].copy()
        if enc in ('mono8', '8uc1'):
            gray = rows[:, :w].reshape((h, w))
            return np.repeat(gray[:, :, None], 3, axis=2).copy()
        raise ValueError(f'unsupported image encoding: {msg.encoding}')

    def encode_jpeg(self, frame) -> bytes:
        import cv2

        ok, buf = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), int(self.jpeg_quality)])
        if not ok:
            raise ValueError('cv2.imencode(.jpg) failed')
        return bytes(buf)

    def decode_jpeg(self, jpeg: bytes):
        import cv2

        arr = np.frombuffer(jpeg, dtype=np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError('cv2.imdecode failed')
        return frame

    def bgr8_to_image_msg(self, img, frame_id: str = '') -> Image:
        msg = Image()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = frame_id
        msg.height = int(img.shape[0])
        msg.width = int(img.shape[1])
        msg.encoding = 'bgr8'
        msg.is_bigendian = 0
        msg.step = int(img.shape[1] * 3)
        msg.data = img.astype(np.uint8, copy=False).tobytes()
        return msg

    @staticmethod
    def stamp_to_sec(stamp):
        return float(stamp.sec) + float(stamp.nanosec) * 1e-9

    def post_and_publish(
        self, jpeg: bytes, frame_id: str, width_hint: int, height_hint: int,
        frame=None, capture_wall_sec: float = 0.0,
    ):
        self.sent_count += 1
        files = {'image': ('frame.jpg', jpeg, 'image/jpeg')}
        data = {
            'frame_id': self.frame_id or frame_id or '',
            'capture_ros_sec': f'{capture_wall_sec:.9f}' if capture_wall_sec > 0.0 else '',
        }
        resp = self.http.post(self.server_url, files=files, data=data, timeout=self.timeout_sec)
        resp.raise_for_status()
        payload = resp.json()
        if width_hint > 0 and not payload.get('image_width'):
            payload['image_width'] = int(width_hint)
        if height_hint > 0 and not payload.get('image_height'):
            payload['image_height'] = int(height_hint)
        payload['ros_frame_id'] = self.frame_id or frame_id or ''
        payload['bridge_stamp_wall_sec'] = time.time()

        out = String()
        out.data = json.dumps(payload, separators=(',', ':'))
        self.pub.publish(out)
        if self.publish_debug_image and frame is not None:
            self.debug_pub.publish(self.bgr8_to_image_msg(self.make_debug_overlay(frame, payload), frame_id))
        self.ok_count += 1
        self.ok_timestamps.append(time.monotonic())
        self.log_periodic()

    def make_debug_overlay(self, frame, payload):
        try:
            import cv2

            img = frame.copy()
            detections = payload.get('detections', [])
            latency = float(payload.get('latency_ms', 0.0))
            ok = bool(payload.get('ok', True))
            color = (0, 255, 0) if ok else (0, 0, 255)
            cv2.putText(
                img,
                f'Flask YOLO det={len(detections)} latency={latency:.1f}ms sent={self.sent_count} ok={self.ok_count}',
                (10, 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.62,
                color,
                2,
            )
            for det in detections:
                bbox = det.get('bbox', det.get('xyxy', None))
                if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
                    continue
                x1, y1, x2, y2 = [int(round(float(v))) for v in bbox]
                conf = float(det.get('conf', det.get('confidence', 0.0)))
                label = str(det.get('label', det.get('name', 'person')))
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    img,
                    f'{label} {conf:.2f}',
                    (x1, max(18, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 255, 0),
                    2,
                )
            if not detections:
                cv2.putText(img, 'NO PERSON DETECTED', (10, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 180, 255), 2)
            return img
        except Exception:
            return frame

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
            f'FLASK_YOLO_BRIDGE_STATUS | rx={self.rx_count} sent={self.sent_count} '
            f'ok={self.ok_count} fail={self.fail_count} dropped={self.dropped_count} '
            f'output_fps={output_fps:.2f} pending={self.pending_frame is not None} '
            f'out={self.output_topic} debug={self.debug_image_topic}'
        )

    def destroy_node(self):
        with self.worker_condition:
            self.worker_stop = True
            self.pending_frame = None
            self.worker_condition.notify_all()
        worker = getattr(self, 'worker_thread', None)
        if worker is not None and worker.is_alive():
            worker.join(timeout=max(2.0, self.timeout_sec + 0.5))
        try:
            self.http.close()
        except Exception:
            pass
        super().destroy_node()


def main():
    rclpy.init()
    node = RosImageToFlaskYolo()
    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        executor.remove_node(node)
        executor.shutdown(timeout_sec=1.0)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
