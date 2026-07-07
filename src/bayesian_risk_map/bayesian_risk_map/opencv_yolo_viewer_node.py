
import time
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from sensor_msgs.msg import CompressedImage, Image
from nav_msgs.msg import OccupancyGrid

from bayesian_risk_map.ros_param_helpers import FlexibleParameterNodeMixin


class OpenCVYoloViewer(FlexibleParameterNodeMixin, Node):
    def __init__(self):
        super().__init__('opencv_yolo_viewer_node')

        self.image_topic = self.declare_parameter('image_topic', '/risk/debug_yolo_image').value
        self.image_type = str(self.declare_parameter('image_type', 'auto').value).strip().lower()
        self.window_name = self.declare_parameter('window_name', 'YOLO Debug /risk/debug_yolo_image').value
        self.resize_width = int(self.declare_parameter('resize_width', 960).value)
        self.log_rate_sec = float(self.declare_parameter('log_rate_sec', 2.0).value)
        self.enable_image_view = self.declare_bool_parameter('enable_image_view', True)
        self.grid_topics_csv = self.declare_parameter(
            'grid_topics',
            ''
        ).value

        self.count = 0
        self.last_log = 0.0
        self.last_grid_log = 0.0
        self.last_shape = ''
        self.grid_stats = {}
        self.window_ready = False

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
        )
        self.sub = None
        if self.enable_image_view:
            image_type = self.image_type
            if image_type == 'auto':
                image_type = 'compressed' if str(self.image_topic).endswith('/compressed') else 'raw'
            if image_type in ('compressed', 'jpeg', 'jpg'):
                self.sub = self.create_subscription(
                    CompressedImage,
                    self.image_topic,
                    self.on_compressed_image,
                    qos,
                )
            else:
                self.sub = self.create_subscription(Image, self.image_topic, self.on_image, qos)
        self.grid_subs = []
        grid_topics = [t.strip() for t in str(self.grid_topics_csv).split(',') if t.strip()]
        for topic in grid_topics:
            self.grid_subs.append(self.create_subscription(
                OccupancyGrid, topic, lambda msg, topic=topic: self.on_grid(topic, msg), qos
            ))

        self.get_logger().info(
            f'OpenCV YOLO/region viewer started | image_topic={self.image_topic} | '
            f'image_type={self.image_type} | enable_image_view={self.enable_image_view} | grid_topics={grid_topics}'
        )

        if self.enable_image_view:
            try:
                import cv2
                cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
                self.window_ready = True
            except Exception as e:
                self.get_logger().error(f'cv2.namedWindow failed: {e}')
                self.get_logger().error('Check DISPLAY/X11. If SSH, use: ssh -X or run this on the PC desktop terminal.')
                self.enable_image_view = False

    def image_msg_to_bgr8(self, msg: Image):
        enc = msg.encoding.lower()
        h = int(msg.height)
        w = int(msg.width)
        step = int(msg.step)

        if h <= 0 or w <= 0 or step <= 0:
            raise ValueError(f'invalid image h={h}, w={w}, step={step}, enc={msg.encoding}')

        raw = np.frombuffer(msg.data, dtype=np.uint8)
        if raw.size < h * step:
            raise ValueError(f'buffer too small: raw={raw.size}, expected={h * step}, enc={msg.encoding}')

        rows = raw[:h * step].reshape((h, step))

        if enc in ('bgr8', '8uc3'):
            if step < w * 3:
                raise ValueError(f'invalid step={step} for {enc} width={w}, need>={w * 3}')
            return rows[:, :w * 3].reshape((h, w, 3)).copy()
        if enc == 'rgb8':
            if step < w * 3:
                raise ValueError(f'invalid step={step} for {enc} width={w}, need>={w * 3}')
            return rows[:, :w * 3].reshape((h, w, 3))[:, :, ::-1].copy()
        if enc in ('mono8', '8uc1'):
            if step < w:
                raise ValueError(f'invalid step={step} for {enc} width={w}, need>={w}')
            gray = rows[:, :w].reshape((h, w))
            return np.repeat(gray[:, :, None], 3, axis=2).copy()
        if enc == 'bgra8':
            if step < w * 4:
                raise ValueError(f'invalid step={step} for {enc} width={w}, need>={w * 4}')
            return rows[:, :w * 4].reshape((h, w, 4))[:, :, :3].copy()
        if enc == 'rgba8':
            if step < w * 4:
                raise ValueError(f'invalid step={step} for {enc} width={w}, need>={w * 4}')
            return rows[:, :w * 4].reshape((h, w, 4))[:, :, [2, 1, 0]].copy()

        raise ValueError(f'unsupported encoding: {msg.encoding}')

    def compressed_msg_to_bgr8(self, msg: CompressedImage):
        try:
            import cv2
            raw = np.frombuffer(msg.data, dtype=np.uint8)
            img = cv2.imdecode(raw, cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError(f'cv2.imdecode failed format={msg.format}')
            return img
        except Exception as e:
            raise ValueError(f'compressed image decode failed: {e}') from e

    def on_grid(self, topic: str, msg: OccupancyGrid):
        try:
            data = np.array(msg.data, dtype=np.int16)
            known = data[data >= 0]
            nonzero = int(np.sum(data > 0))
            maxv = int(np.max(known)) if known.size else 0
            meanv = float(np.mean(known)) if known.size else 0.0
            unique_pos = int(len(np.unique(data[data > 0]))) if np.any(data > 0) else 0
            self.grid_stats[topic] = (
                int(msg.info.width), int(msg.info.height), nonzero, maxv, meanv, unique_pos
            )

            now = time.time()
            if now - self.last_grid_log >= self.log_rate_sec:
                self.last_grid_log = now
                parts = []
                for name, (w, h, nz, mx, mean, upos) in sorted(self.grid_stats.items()):
                    parts.append(f'{name}: {w}x{h} nz={nz} max={mx} mean={mean:.1f} uniq_pos={upos}')
                self.get_logger().info('GRID_VIEW | ' + ' | '.join(parts))
        except Exception as e:
            self.get_logger().warn(f'grid parse failed topic={topic}: {e}', throttle_duration_sec=2.0)

    def on_image(self, msg: Image):
        if not self.enable_image_view or not self.window_ready:
            return
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

    def on_compressed_image(self, msg: CompressedImage):
        if not self.enable_image_view or not self.window_ready:
            return
        self.count += 1
        try:
            import cv2
            img = self.compressed_msg_to_bgr8(msg)

            h, w = img.shape[:2]
            self.last_shape = f'{w}x{h} {msg.format or "compressed"}'

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
            self.get_logger().warn(f'OpenCV YOLO compressed viewer failed: {e}', throttle_duration_sec=2.0)


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
