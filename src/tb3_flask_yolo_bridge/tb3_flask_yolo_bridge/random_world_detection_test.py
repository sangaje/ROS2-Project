#!/usr/bin/env python3

from __future__ import annotations

import json
import math
import random
import time
from typing import Optional, Tuple

import numpy as np

import rclpy
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import String
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import Point
from visualization_msgs.msg import Marker, MarkerArray
import tf2_ros

from tb3_flask_yolo_bridge.ros_param_helpers import FlexibleParameterNodeMixin


def yaw_from_quaternion(q) -> float:
    return math.atan2(
        2.0 * (q.w * q.z + q.x * q.y),
        1.0 - 2.0 * (q.y * q.y + q.z * q.z),
    )


def wrap_angle(a: float) -> float:
    return math.atan2(math.sin(a), math.cos(a))


class RandomWorldDetectionTest(FlexibleParameterNodeMixin, Node):
    """Publish compact person detections only when a random world target is in camera FOV."""

    def __init__(self):
        super().__init__('random_world_detection_test')
        self.output_topic = self.declare_parameter('output_topic', '/risk/yolo_detections').value
        self.map_topic = self.declare_parameter('map_topic', '/map').value
        self.map_frame = self.declare_parameter('map_frame', 'map').value
        self.base_frame = self.declare_parameter('base_frame', 'base_link').value
        self.publish_rate_hz = float(self.declare_parameter('publish_rate_hz', 5.0).value)
        self.detection_rate_hz = float(self.declare_parameter('detection_rate_hz', 2.0).value)
        self.retarget_after_detection_sec = float(self.declare_parameter('retarget_after_detection_sec', 4.0).value)
        self.retarget_period_sec = float(self.declare_parameter('retarget_period_sec', 30.0).value)
        self.camera_hfov_deg = float(self.declare_parameter('camera_hfov_deg', 62.0).value)
        self.min_range_m = float(self.declare_parameter('min_range_m', 0.5).value)
        self.max_range_m = float(self.declare_parameter('max_range_m', 5.0).value)
        self.confidence = float(self.declare_parameter('confidence', 0.92).value)
        self.image_width = int(self.declare_parameter('image_width', 640).value)
        self.image_height = int(self.declare_parameter('image_height', 480).value)
        self.x_min = float(self.declare_parameter('x_min', -2.0).value)
        self.x_max = float(self.declare_parameter('x_max', 2.0).value)
        self.y_min = float(self.declare_parameter('y_min', -2.0).value)
        self.y_max = float(self.declare_parameter('y_max', 2.0).value)
        self.use_map_free_cells = self.declare_bool_parameter('use_map_free_cells', True)
        self.free_threshold = int(self.declare_parameter('free_threshold', 30).value)

        map_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self.map_sub = self.create_subscription(OccupancyGrid, self.map_topic, self.on_map, map_qos)
        self.pub = self.create_publisher(String, self.output_topic, 10)
        self.marker_pub = self.create_publisher(MarkerArray, '/risk/random_detection_test_markers', 10)
        self.tf_buffer = tf2_ros.Buffer(cache_time=Duration(seconds=30.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.map_msg: Optional[OccupancyGrid] = None
        self.target_xy: Optional[Tuple[float, float]] = None
        self.last_target_wall_sec = 0.0
        self.last_detection_wall_sec = 0.0
        self.last_publish_wall_sec = 0.0
        self.timer = self.create_timer(1.0 / max(0.1, self.publish_rate_hz), self.on_timer)
        self.choose_new_target('startup')
        self.get_logger().info(
            f'RANDOM_WORLD_DETECTION_TEST_READY | out={self.output_topic} map={self.map_topic} '
            f'fov={self.camera_hfov_deg:.1f} range=[{self.min_range_m:.1f},{self.max_range_m:.1f}] '
            f'bounds=({self.x_min:.1f},{self.y_min:.1f})-({self.x_max:.1f},{self.y_max:.1f})'
        )

    def on_map(self, msg: OccupancyGrid):
        self.map_msg = msg

    def get_robot_pose(self) -> Optional[Tuple[float, float, float]]:
        candidates = [self.base_frame]
        for frame in ('base_footprint', 'base_link'):
            if frame not in candidates:
                candidates.append(frame)
        for base in candidates:
            try:
                tf = self.tf_buffer.lookup_transform(
                    self.map_frame,
                    base,
                    rclpy.time.Time(seconds=0, nanoseconds=0),
                    timeout=Duration(seconds=0.05),
                )
                t = tf.transform.translation
                return float(t.x), float(t.y), yaw_from_quaternion(tf.transform.rotation)
            except Exception:
                continue
        return None

    def choose_new_target(self, reason: str):
        target = self.sample_free_map_target() if self.use_map_free_cells else None
        if target is None:
            target = (
                random.uniform(self.x_min, self.x_max),
                random.uniform(self.y_min, self.y_max),
            )
        self.target_xy = target
        self.last_target_wall_sec = time.time()
        self.get_logger().info(f'RANDOM_TEST_TARGET | reason={reason} xy=({target[0]:.2f},{target[1]:.2f})')

    def sample_free_map_target(self) -> Optional[Tuple[float, float]]:
        msg = self.map_msg
        if msg is None or msg.info.width <= 0 or msg.info.height <= 0:
            return None
        h = int(msg.info.height)
        w = int(msg.info.width)
        data = np.array(msg.data, dtype=np.int16).reshape((h, w))
        free = np.argwhere((data >= 0) & (data <= self.free_threshold))
        if free.size == 0:
            return None
        random.shuffle(free)
        for gy, gx in free[:500]:
            x, y = self.grid_to_world(int(gx), int(gy), msg)
            if self.x_min <= x <= self.x_max and self.y_min <= y <= self.y_max:
                return x, y
        return None

    def grid_to_world(self, gx: int, gy: int, msg: OccupancyGrid) -> Tuple[float, float]:
        res = float(msg.info.resolution)
        ox = float(msg.info.origin.position.x)
        oy = float(msg.info.origin.position.y)
        oyaw = yaw_from_quaternion(msg.info.origin.orientation)
        lx = (gx + 0.5) * res
        ly = (gy + 0.5) * res
        c = math.cos(oyaw)
        s = math.sin(oyaw)
        return ox + c * lx - s * ly, oy + s * lx + c * ly

    def on_timer(self):
        now = time.time()
        if self.target_xy is None or now - self.last_target_wall_sec >= self.retarget_period_sec:
            self.choose_new_target('period')
        pose = self.get_robot_pose()
        if pose is None or self.target_xy is None:
            self.publish_markers(None, None, False)
            return

        rx, ry, ryaw = pose
        tx, ty = self.target_xy
        dx = tx - rx
        dy = ty - ry
        rng = math.hypot(dx, dy)
        bearing = wrap_angle(math.atan2(dy, dx) - ryaw)
        visible = self.min_range_m <= rng <= self.max_range_m and abs(bearing) <= math.radians(self.camera_hfov_deg) * 0.5
        self.publish_markers(pose, (bearing, rng), visible)
        if not visible:
            return
        if self.detection_rate_hz > 0.0 and now - self.last_publish_wall_sec < 1.0 / self.detection_rate_hz:
            return

        self.last_publish_wall_sec = now
        self.last_detection_wall_sec = now
        self.pub.publish(String(data=json.dumps(self.make_detection_payload(bearing, rng), separators=(',', ':'))))
        if now - self.last_target_wall_sec >= self.retarget_after_detection_sec:
            self.choose_new_target('detected')

    def make_detection_payload(self, bearing: float, rng: float) -> dict:
        half_fov = math.radians(self.camera_hfov_deg) * 0.5
        cx = 0.5 * self.image_width + (bearing / max(half_fov, 1e-6)) * 0.35 * self.image_width
        box_h = max(40.0, min(220.0, 420.0 / max(rng, 0.5)))
        box_w = 0.42 * box_h
        x1 = max(0.0, cx - 0.5 * box_w)
        x2 = min(float(self.image_width - 1), cx + 0.5 * box_w)
        y2 = min(float(self.image_height - 1), 0.78 * self.image_height)
        y1 = max(0.0, y2 - box_h)
        return {
            'ok': True,
            'source': 'random_world_detection_test',
            'stamp_wall_sec': time.time(),
            'image_width': self.image_width,
            'image_height': self.image_height,
            'detections': [{
                'bbox': [x1, y1, x2, y2],
                'conf': self.confidence,
                'class_id': 0,
                'label': 'person',
                'bearing_rad': bearing,
                'range_hat_m': rng,
            }],
        }

    def publish_markers(self, pose, bearing_range, visible: bool):
        arr = MarkerArray()
        clear = Marker()
        clear.action = Marker.DELETEALL
        arr.markers.append(clear)
        if self.target_xy is not None:
            target = Marker()
            target.header.frame_id = self.map_frame
            target.header.stamp = self.get_clock().now().to_msg()
            target.ns = 'random_detection_target'
            target.id = 1
            target.type = Marker.SPHERE
            target.action = Marker.ADD
            target.pose.position.x = float(self.target_xy[0])
            target.pose.position.y = float(self.target_xy[1])
            target.pose.position.z = 0.18
            target.scale.x = 0.25
            target.scale.y = 0.25
            target.scale.z = 0.35
            target.color.r = 1.0
            target.color.g = 0.2 if visible else 0.7
            target.color.b = 0.1
            target.color.a = 0.95
            arr.markers.append(target)
        if pose is not None and bearing_range is not None:
            rx, ry, ryaw = pose
            bearing, rng = bearing_range
            ray = Marker()
            ray.header.frame_id = self.map_frame
            ray.header.stamp = self.get_clock().now().to_msg()
            ray.ns = 'random_detection_ray'
            ray.id = 2
            ray.type = Marker.LINE_STRIP
            ray.action = Marker.ADD
            ray.scale.x = 0.035
            ray.color.r = 0.1
            ray.color.g = 1.0 if visible else 0.4
            ray.color.b = 0.1
            ray.color.a = 0.9
            ray.points.append(Point(x=float(rx), y=float(ry), z=0.1))
            ray.points.append(Point(x=float(rx + rng * math.cos(ryaw + bearing)), y=float(ry + rng * math.sin(ryaw + bearing)), z=0.1))
            arr.markers.append(ray)
        self.marker_pub.publish(arr)


def main():
    rclpy.init()
    node = RandomWorldDetectionTest()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.destroy_node()
        except KeyboardInterrupt:
            pass
        if rclpy.ok():
            try:
                rclpy.shutdown()
            except KeyboardInterrupt:
                pass


if __name__ == '__main__':
    main()
