#!/usr/bin/env python3

from __future__ import annotations

import time

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image


class OpenCVCameraPublisher(Node):
    def __init__(self):
        super().__init__('opencv_camera_publisher')
        self.device = self.declare_parameter('device', '/dev/video0').value
        self.image_topic = self.declare_parameter('image_topic', '/camera/image_raw').value
        self.frame_id = self.declare_parameter('frame_id', 'camera_link').value
        self.width = int(self.declare_parameter('width', 640).value)
        self.height = int(self.declare_parameter('height', 480).value)
        self.fps = float(self.declare_parameter('fps', 15.0).value)
        self.show_preview = bool(self.declare_parameter('show_preview', False).value)
        self.window_name = self.declare_parameter('window_name', 'OpenCV Camera Source').value
        self.log_period_sec = float(self.declare_parameter('log_period_sec', 2.0).value)

        import cv2

        device_arg = int(self.device) if str(self.device).isdigit() else self.device
        self.cap = cv2.VideoCapture(device_arg)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
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
        self.create_timer(1.0 / max(0.5, self.fps), self.on_timer)
        self.get_logger().info(
            f'OPENCV_CAMERA_PUBLISHER_READY | device={self.device} opened={self.cap.isOpened()} '
            f'out={self.image_topic} size={self.width}x{self.height} fps={self.fps:.1f}'
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

    def on_timer(self):
        if not self.cap.isOpened():
            self.fail_count += 1
            self.log_status()
            return
        ok, frame = self.cap.read()
        if not ok or frame is None:
            self.fail_count += 1
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
