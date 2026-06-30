#!/usr/bin/env python3

from __future__ import annotations

import time
import threading

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image

from tb3_flask_yolo_bridge.ros_param_helpers import FlexibleParameterNodeMixin


class OpenCVCameraPublisher(FlexibleParameterNodeMixin, Node):
    def __init__(self):
        super().__init__('opencv_camera_publisher')
        self.device = self.declare_parameter('device', '/dev/video0').value
        self.image_topic = self.declare_parameter('image_topic', '/camera/image_raw').value
        self.frame_id = self.declare_parameter('frame_id', 'camera_link').value
        self.width = int(self.declare_parameter('width', 640).value)
        self.height = int(self.declare_parameter('height', 480).value)
        self.fps = float(self.declare_parameter('fps', 15.0).value)
        self.buffer_size = int(self.declare_parameter('buffer_size', 1).value)
        self.fourcc = str(self.declare_parameter('fourcc', 'MJPG').value).strip()
        self.async_capture = self.declare_bool_parameter('async_capture', True)
        self.show_preview = self.declare_bool_parameter('show_preview', False)
        self.window_name = self.declare_parameter('window_name', 'OpenCV Camera Source').value
        self.log_period_sec = float(self.declare_parameter('log_period_sec', 2.0).value)

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
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        if self.height > 0:
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        if self.fps > 0:
            self.cap.set(cv2.CAP_PROP_FPS, self.fps)

        camera_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self.pub = self.create_publisher(Image, self.image_topic, camera_qos)
        self.count = 0
        self.fail_count = 0
        self.last_log = 0.0
        self.frame_lock = threading.Lock()
        self.latest_frame = None
        self.latest_seq = 0
        self.published_seq = 0
        self.capture_stop = False
        self.capture_thread = None
        if self.async_capture:
            self.capture_thread = threading.Thread(
                target=self.capture_loop,
                name='opencv_camera_latest_frame_worker',
                daemon=True,
            )
            self.capture_thread.start()
        self.create_timer(1.0 / max(0.5, self.fps), self.on_timer)
        self.get_logger().info(
            f'OPENCV_CAMERA_PUBLISHER_READY | device={self.device} opened={self.cap.isOpened()} '
            f'out={self.image_topic} size={self.width}x{self.height} fps={self.fps:.1f} '
            f'fourcc={self.fourcc or "default"} async_capture={self.async_capture}'
        )

    def bgr8_to_image_msg(self, img) -> Image:
        msg = Image()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.frame_id
        msg.height = int(img.shape[0])
        msg.width = int(img.shape[1])
        msg.encoding = 'bgr8'
        msg.is_bigendian = 0
        msg.step = int(img.shape[1] * 3)
        msg.data = img.astype(np.uint8, copy=False).tobytes()
        return msg

    def capture_loop(self):
        while not self.capture_stop:
            if not self.cap.isOpened():
                self.fail_count += 1
                time.sleep(0.05)
                continue
            ok, frame = self.cap.read()
            if not ok or frame is None:
                self.fail_count += 1
                time.sleep(0.02)
                continue
            with self.frame_lock:
                self.latest_frame = frame
                self.latest_seq += 1

    def get_latest_frame(self):
        with self.frame_lock:
            if self.latest_frame is None:
                return None, 0
            return self.latest_frame.copy(), self.latest_seq

    def read_frame(self):
        if not self.cap.isOpened():
            self.fail_count += 1
            return None
        ok, frame = self.cap.read()
        if not ok or frame is None:
            self.fail_count += 1
            return None
        return frame

    def on_timer(self):
        if self.async_capture:
            frame, seq = self.get_latest_frame()
            if frame is None:
                self.log_status()
                return
            if seq == self.published_seq:
                return
            self.published_seq = seq
        else:
            frame = self.read_frame()
            if frame is None:
                self.log_status()
                return
        self.pub.publish(self.bgr8_to_image_msg(frame))
        self.count += 1
        if self.show_preview:
            try:
                import cv2
                cv2.imshow(self.window_name, frame)
                cv2.waitKey(1)
            except Exception:
                self.show_preview = False
        self.log_status()

    def log_status(self):
        now = time.time()
        if now - self.last_log < self.log_period_sec:
            return
        self.last_log = now
        self.get_logger().info(
            f'OPENCV_CAMERA_STATUS | frames={self.count} fail={self.fail_count} '
            f'opened={self.cap.isOpened()} out={self.image_topic}'
        )

    def destroy_node(self):
        self.capture_stop = True
        if self.capture_thread is not None:
            self.capture_thread.join(timeout=2.0)
        try:
            self.cap.release()
        except Exception:
            pass
        if self.show_preview:
            try:
                import cv2
                cv2.destroyAllWindows()
            except Exception:
                pass
        super().destroy_node()


def main():
    rclpy.init()
    node = OpenCVCameraPublisher()
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
