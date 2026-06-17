
import time
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image


class OpenCVYoloViewer(Node):
    def __init__(self):
        super().__init__('opencv_yolo_viewer_node')

        self.image_topic = self.declare_parameter('image_topic', '/risk/debug_yolo_image').value
        self.window_name = self.declare_parameter('window_name', 'YOLO Debug /risk/debug_yolo_image').value
        self.resize_width = int(self.declare_parameter('resize_width', 960).value)
        self.log_rate_sec = float(self.declare_parameter('log_rate_sec', 2.0).value)

        self.count = 0
        self.last_log = 0.0
        self.last_shape = ''

        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
        )
        self.sub = self.create_subscription(Image, self.image_topic, self.on_image, qos)
        self.get_logger().info(f'OpenCV YOLO viewer started | topic={self.image_topic}')

        try:
            import cv2
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        except Exception as e:
            self.get_logger().error(f'cv2.namedWindow failed: {e}')
            self.get_logger().error('Check DISPLAY/X11. If SSH, use: ssh -X or run this on the PC desktop terminal.')

    def image_msg_to_bgr8(self, msg: Image):
        enc = msg.encoding.lower()
        h = int(msg.height)
        w = int(msg.width)
        step = int(msg.step)

        raw = np.frombuffer(msg.data, dtype=np.uint8)
        if raw.size < h * step:
            raise ValueError(f'buffer too small: raw={raw.size}, expected={h * step}, enc={msg.encoding}')

        rows = raw[:h * step].reshape((h, step))

        if enc in ('bgr8', '8uc3'):
            return rows[:, :w * 3].reshape((h, w, 3)).copy()
        if enc == 'rgb8':
            return rows[:, :w * 3].reshape((h, w, 3))[:, :, ::-1].copy()
        if enc in ('mono8', '8uc1'):
            gray = rows[:, :w].reshape((h, w))
            return np.repeat(gray[:, :, None], 3, axis=2).copy()
        if enc == 'bgra8':
            return rows[:, :w * 4].reshape((h, w, 4))[:, :, :3].copy()
        if enc == 'rgba8':
            return rows[:, :w * 4].reshape((h, w, 4))[:, :, [2, 1, 0]].copy()

        raise ValueError(f'unsupported encoding: {msg.encoding}')

    def on_image(self, msg: Image):
        self.count += 1
        try:
            import cv2
            img = self.image_msg_to_bgr8(msg)

            h, w = img.shape[:2]
            self.last_shape = f'{w}x{h} {msg.encoding}'

            if self.resize_width > 0 and w > 0 and w != self.resize_width:
                scale = self.resize_width / float(w)
                img = cv2.resize(img, (self.resize_width, int(h * scale)))

            cv2.imshow(self.window_name, img)
            cv2.waitKey(1)

            now = time.time()
            if now - self.last_log >= self.log_rate_sec:
                self.last_log = now
                self.get_logger().info(
                    f'OPENCV_YOLO_VIEW | frames={self.count} | last_shape={self.last_shape} | topic={self.image_topic}'
                )
        except Exception as e:
            self.get_logger().warn(f'OpenCV YOLO viewer failed: {e}', throttle_duration_sec=2.0)


def main(args=None):
    rclpy.init(args=args)
    node = OpenCVYoloViewer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            import cv2
            cv2.destroyAllWindows()
        except Exception:
            pass
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
