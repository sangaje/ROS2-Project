
import json
import math
import os
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy, HistoryPolicy

from std_msgs.msg import Bool, String
from sensor_msgs.msg import CompressedImage, Image
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import PointStamped, Point, PoseStamped
from visualization_msgs.msg import Marker, MarkerArray

import tf2_ros

from tb3_bayesian_risk_map.ros_param_helpers import FlexibleParameterNodeMixin


@dataclass
class Detection2D:
    bbox: Tuple[float, float, float, float]
    conf: float
    bearing_rad: float
    range_hat_m: float


@dataclass
class EvidencePoint:
    evidence_id: int
    x: float
    y: float
    confidence: float
    stamp_sec: float


@dataclass
class PoseSample:
    stamp_sec: float
    x: float
    y: float
    yaw: float


@dataclass
class RegionState:
    region_id: int
    area_cells: int
    centroid_x: float
    centroid_y: float
    coverage_ratio: float
    frontier_ratio: float
    obstacle_density: float
    structural_risk: float
    person_risk: float
    priority: float
    checked: bool
    last_seen_sec: float


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


def yaw_from_quaternion(q):
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


def wrap_angle(a):
    return math.atan2(math.sin(a), math.cos(a))


class RoomAwareRiskMapNode(FlexibleParameterNodeMixin, Node):
    """
    v2 design:
    - Positive evidence creates candidate probability from YOLO bearing/range + map line-of-sight.
    - Risk is produced from positive evidence only.
    - Empty observation is stored separately as /risk/observed_empty_map.
      It does NOT reduce /risk/risk_map.
    - Risk halo is local and geodesic in free-space, so it does not cross walls and does not spread globally.
    - Room probability is a diagnostic layer based on connected free-space regions.
    """

    def __init__(self):
        super().__init__('bayesian_risk_map_node')

        # Core parameters
        self.map_topic = self.declare_parameter('map_topic', '/map').value
        self.map_qos_durability = str(
            self.declare_parameter('map_qos_durability', 'volatile').value
        ).strip().lower()
        self.image_topic = self.declare_parameter('image_topic', '/camera/image_raw').value
        self.map_frame = self.declare_parameter('map_frame', 'map').value
        self.base_frame = self.declare_parameter('base_frame', 'base_link').value
        self.pose_source = str(self.declare_parameter('pose_source', 'tf').value).strip().lower()
        self.pose_topic = self.declare_parameter('pose_topic', '/leader_pose').value
        self.pose_topic_stale_sec = float(self.declare_parameter('pose_topic_stale_sec', 2.5).value)
        self.update_rate_hz = float(self.declare_parameter('update_rate_hz', 2.0).value)
        self.tf_timeout_sec = float(self.declare_parameter('tf_timeout_sec', 0.25).value)
        self.pose_history_duration_sec = float(
            self.declare_parameter('pose_history_duration_sec', 5.0).value
        )
        self.pose_history_max_error_sec = float(
            self.declare_parameter('pose_history_max_error_sec', 0.75).value
        )

        # YOLO
        self.detection_source = str(self.declare_parameter('detection_source', 'local_yolo').value).strip().lower()
        self.external_detection_topic = self.declare_parameter('external_detection_topic', '/risk/yolo_detections').value
        self.external_person_only = self.declare_bool_parameter('external_person_only', True)
        self.debug_image_topic = self.declare_parameter('debug_image_topic', '/risk/debug_yolo_image').value
        self.enable_yolo = self.declare_bool_parameter('enable_yolo', True)
        self.model_path = self.declare_parameter('model_path', 'yolo11n.pt').value
        self.device = self.declare_parameter('device', 'cpu').value
        self.conf_threshold = float(self.declare_parameter('conf_threshold', 0.20).value)
        self.yolo_imgsz = int(self.declare_parameter('yolo_imgsz', 640).value)
        self.yolo_max_rate_hz = float(self.declare_parameter('yolo_max_rate_hz', 3.0).value)
        self.yolo_async = self.declare_bool_parameter('yolo_async', True)
        self.detection_timeout_sec = float(self.declare_parameter('detection_timeout_sec', 0.8).value)
        self.detection_reuse_max_distance_m = float(
            self.declare_parameter('detection_reuse_max_distance_m', 0.50).value
        )
        self.external_detection_max_count = int(
            self.declare_parameter('external_detection_max_count', 64).value
        )
        self.opencv_camera_device = self.declare_parameter('opencv_camera_device', '/dev/video0').value
        self.opencv_camera_width = int(self.declare_parameter('opencv_camera_width', 640).value)
        self.opencv_camera_height = int(self.declare_parameter('opencv_camera_height', 480).value)
        self.opencv_camera_fps = float(self.declare_parameter('opencv_camera_fps', 15.0).value)
        self.opencv_camera_buffer_size = int(
            self.declare_parameter('opencv_camera_buffer_size', 1).value
        )
        self.opencv_async_capture = self.declare_bool_parameter('opencv_async_capture', True)
        self.opencv_reopen_after_failures = int(
            self.declare_parameter('opencv_reopen_after_failures', 5).value
        )
        # Direct OpenCV camera mode only. Empty string disables explicit FOURCC.
        # Useful on real TurtleBot3 USB cameras where MJPG avoids high USB/CPU load.
        self.opencv_camera_fourcc = str(
            self.declare_parameter('opencv_camera_fourcc', '').value
        ).strip()

        # Fake
        self.enable_fake_detection = self.declare_bool_parameter('enable_fake_detection', False)
        self.fake_detection_interval_sec = float(self.declare_parameter('fake_detection_interval_sec', 2.0).value)
        self.fake_bearing_deg = float(self.declare_parameter('fake_bearing_deg', 0.0).value)
        self.fake_range_m = float(self.declare_parameter('fake_range_m', 2.0).value)
        self.fake_confidence = float(self.declare_parameter('fake_confidence', 0.9).value)

        # Camera prior
        self.camera_hfov_deg = float(self.declare_parameter('camera_hfov_deg', 62.0).value)
        self.camera_vfov_deg = float(self.declare_parameter('camera_vfov_deg', 49.5).value)
        self.real_person_height_m = float(self.declare_parameter('real_person_height_m', 1.70).value)
        self.min_range_m = float(self.declare_parameter('min_range_m', 0.5).value)
        self.max_range_m = float(self.declare_parameter('max_range_m', 5.0).value)

        # Positive model
        self.bearing_sigma_deg = float(self.declare_parameter('bearing_sigma_deg', 8.0).value)
        self.angular_sample_step_deg = float(self.declare_parameter('angular_sample_step_deg', 1.0).value)
        self.range_sigma_m = float(self.declare_parameter('range_sigma_m', 0.75).value)
        self.use_bbox_range_prior = self.declare_bool_parameter('use_bbox_range_prior', True)
        self.source_min_value = float(self.declare_parameter('source_min_value', 0.03).value)
        self.positive_memory_alpha = float(self.declare_parameter('positive_memory_alpha', 0.85).value)

        # Halo
        self.source_halo_radius_m = float(self.declare_parameter('source_halo_radius_m', 0.75).value)
        self.source_halo_sigma_m = float(self.declare_parameter('source_halo_sigma_m', 0.35).value)
        self.source_halo_seed_threshold = float(self.declare_parameter('source_halo_seed_threshold', 0.12).value)
        self.source_halo_top_k = int(self.declare_parameter('source_halo_top_k', 24).value)
        self.source_halo_seed_separation_m = float(
            self.declare_parameter('source_halo_seed_separation_m', 0.20).value
        )
        self.risk_source_mode = str(
            self.declare_parameter('risk_source_mode', 'evidence_points').value
        ).strip().lower()
        self.evidence_source_gain = float(self.declare_parameter('evidence_source_gain', 0.65).value)
        self.evidence_distribution_radius_m = float(
            self.declare_parameter('evidence_distribution_radius_m', 0.45).value
        )
        self.evidence_distribution_sigma_m = float(
            self.declare_parameter('evidence_distribution_sigma_m', 0.22).value
        )

        # Room / region
        self.enable_room_probability = self.declare_bool_parameter('enable_room_probability', True)
        self.room_top_k = int(self.declare_parameter('room_top_k', 3).value)
        self.room_min_score = float(self.declare_parameter('room_min_score', 0.02).value)

        # Region segmentation / priority for live teleop SLAM.
        # Internal name is region, not room, because a partial SLAM map can split/merge rooms while mapping.
        self.enable_region_segmentation = self.declare_bool_parameter('enable_region_segmentation', True)
        self.region_update_period_sec = float(self.declare_parameter('region_update_period_sec', 1.0).value)
        self.region_core_clearance_m = float(self.declare_parameter('region_core_clearance_m', 0.38).value)
        self.region_expand_clearance_m = float(self.declare_parameter('region_expand_clearance_m', 0.22).value)
        self.min_region_area_m2 = float(self.declare_parameter('min_region_area_m2', 0.30).value)
        self.region_iou_match_threshold = float(self.declare_parameter('region_iou_match_threshold', 0.20).value)
        self.region_checked_coverage_ratio = float(self.declare_parameter('region_checked_coverage_ratio', 0.70).value)
        self.region_frontier_gain_scale = float(self.declare_parameter('region_frontier_gain_scale', 18.0).value)
        self.region_obstacle_gain_scale = float(self.declare_parameter('region_obstacle_gain_scale', 6.0).value)
        self.region_debug_log_period_sec = float(self.declare_parameter('region_debug_log_period_sec', 2.0).value)
        self.diagnostic_publish_rate_hz = float(
            self.declare_parameter('diagnostic_publish_rate_hz', 1.0).value
        )

        # Teleop / live mapping optimization.
        # This keeps the risk layer responsive while avoiding unnecessary CPU churn
        # during manual exploration with Cartographer.
        self.teleop_mode = self.declare_bool_parameter('teleop_mode', False)
        self.risk_publish_rate_hz = float(self.declare_parameter('risk_publish_rate_hz', 5.0).value)
        if self.teleop_mode:
            self.region_update_period_sec = max(self.region_update_period_sec, 1.5)
            self.diagnostic_publish_rate_hz = min(self.diagnostic_publish_rate_hz, 0.5)
            self.risk_publish_rate_hz = min(self.risk_publish_rate_hz, 5.0)

        # Empty observation
        self.enable_empty_observation_map = self.declare_bool_parameter('enable_empty_observation_map', True)
        self.enable_visibility_tracking = self.declare_bool_parameter('enable_visibility_tracking', True)
        self.visibility_num_rays = int(self.declare_parameter('visibility_num_rays', 96).value)
        if self.teleop_mode:
            self.visibility_num_rays = min(self.visibility_num_rays, 48)
        self.observed_empty_alpha = float(self.declare_parameter('observed_empty_alpha', 0.20).value)

        # Occupancy policy
        self.allow_unknown = self.declare_bool_parameter('allow_unknown', False)
        self.risk_persist_in_unknown = self.declare_bool_parameter('risk_persist_in_unknown', True)
        self.free_threshold = int(self.declare_parameter('free_threshold', 30).value)
        self.occupied_threshold = int(self.declare_parameter('occupied_threshold', 65).value)

        # Clear
        self.clear_radius_m = float(self.declare_parameter('clear_radius_m', 0.6).value)

        # Debug image
        self.publish_overlay = self.declare_bool_parameter('publish_overlay', True)
        self.publish_debug_image = self.declare_bool_parameter('publish_debug_image', True)
        self.publish_debug_compressed_image = self.declare_bool_parameter(
            'publish_debug_compressed_image',
            False,
        )
        self.debug_compressed_image_topic = str(
            self.declare_parameter('debug_compressed_image_topic', '/risk/debug_yolo_image/compressed').value
        ).strip()
        self.debug_compressed_jpeg_quality = int(
            self.declare_parameter('debug_compressed_jpeg_quality', 70).value
        )
        self.debug_compressed_resize_width = int(
            self.declare_parameter('debug_compressed_resize_width', 480).value
        )
        self.debug_compressed_publish_rate_hz = float(
            self.declare_parameter('debug_compressed_publish_rate_hz', 3.0).value
        )
        self.debug_show_opencv = self.declare_bool_parameter('debug_show_opencv', False)
        self.debug_save_images = self.declare_bool_parameter('debug_save_images', False)
        self.debug_image_dir = self.declare_parameter('debug_image_dir', '/tmp/tb3_bayesian_risk_map_debug').value
        self.debug_image_rate_hz = float(self.declare_parameter('debug_image_rate_hz', 1.0).value)
        self.debug_log_image_status = self.declare_bool_parameter('debug_log_image_status', True)
        self.publish_diagnostic_maps = self.declare_bool_parameter('publish_diagnostic_maps', False)

        # Persistence. Critical for Cartographer: map geometry can grow/change while exploring.
        # If true, positive/risk layers are reprojected in world coordinates instead of reset.
        self.preserve_risk_on_map_resize = self.declare_bool_parameter('preserve_risk_on_map_resize', True)
        self.publish_yolo_debug_even_without_detection = self.declare_bool_parameter(
            'publish_yolo_debug_even_without_detection',
            True,
        )

        # State
        self.latest_map_msg: Optional[OccupancyGrid] = None
        self.occ_grid: Optional[np.ndarray] = None
        self.map_signature = None
        self.map_resolution = None
        self.map_origin_x = None
        self.map_origin_y = None
        self.map_origin_yaw = 0.0
        self.prev_map_geometry = None

        self.detection_candidate_map = None
        self.positive_memory_map = None
        self.risk_map = None
        self.risk_dirty = True
        self.observed_empty_map = None
        self.visibility_map = None
        self.room_probability_map = None

        self.visual_seen_map = None
        self.region_id_map = None
        self.region_priority_map = None
        self.region_checked_map = None
        self.region_states: Dict[int, RegionState] = {}
        self.next_region_id = 1
        self.last_region_update_wall_sec = 0.0
        self.last_region_debug_wall_sec = 0.0
        self.last_diagnostic_publish_ros_ns = 0
        self.last_risk_publish_ros_ns = 0

        self.latest_detections: List[Detection2D] = []
        self.last_yolo_wall_sec = 0.0
        self.last_yolo_ros_sec = None
        self.last_fake_wall_sec = 0.0
        self.last_debug_save_wall_sec = 0.0
        self.last_debug_compressed_publish_wall_sec = 0.0
        self.pose_history = deque()
        self.latest_detection_pose = None
        self.latest_detection_capture_sec = None
        self.latest_detection_delay_ms = -1.0
        self.detection_lock = threading.Lock()
        self.pose_lock = threading.Lock()
        self.yolo_condition = threading.Condition()
        self.yolo_pending_frame = None
        self.yolo_worker_stop = False
        self.yolo_worker_thread = None
        self.yolo_drop_count = 0

        self.external_detection_rx_count = 0
        self.image_rx_count = 0
        self.yolo_frame_count = 0
        self.yolo_det_count = 0
        self.last_image_encoding = ''
        self.last_image_shape = ''
        self.opencv_cap = None
        self.opencv_camera_timer = None
        self.opencv_camera_warned = False
        self.opencv_read_fail_count = 0
        self.opencv_frame_lock = threading.Lock()
        self.opencv_latest_frame = None
        self.opencv_latest_capture_sec = 0.0
        self.opencv_latest_seq = 0
        self.opencv_consumed_seq = 0
        self.opencv_capture_thread = None
        self.opencv_capture_stop = False

        self.evidence_points: List[EvidencePoint] = []
        self.next_evidence_id = 1
        self.topic_pose: Optional[PoseStamped] = None
        self.topic_pose_stamp = None

        # TF
        self.tf_buffer = tf2_ros.Buffer(cache_time=Duration(seconds=120.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # QoS
        map_durability = (
            DurabilityPolicy.TRANSIENT_LOCAL
            if self.map_qos_durability in ('transient_local', 'transient-local', 'transientlocal')
            else DurabilityPolicy.VOLATILE
        )
        self.qos_map_sub = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=map_durability,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self.qos_sensor_sub = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self.qos_grid_pub = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self.qos_marker_pub = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self.qos_image_pub = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        # IO
        self.map_sub = self.create_subscription(OccupancyGrid, self.map_topic, self.on_map, self.qos_map_sub)
        self.pose_sub = None
        if self.pose_source in ('topic', 'pose_topic', 'pose'):
            self.pose_sub = self.create_subscription(PoseStamped, self.pose_topic, self.on_pose_topic, 10)
        self.image_sub = None
        self.external_detection_sub = None
        self.use_opencv_camera = False
        if self.detection_source in ('local_yolo', 'ros_image', 'image'):
            self.image_sub = self.create_subscription(Image, self.image_topic, self.on_image, self.qos_sensor_sub)
        elif self.detection_source in ('opencv_camera', 'direct_camera', 'cv2_camera', 'cv2'):
            self.use_opencv_camera = True
        elif self.detection_source in ('flask_topic', 'external', 'json'):
            self.external_detection_sub = self.create_subscription(
                String,
                self.external_detection_topic,
                self.on_external_detections,
                10,
            )
            self.enable_yolo = False
        elif self.detection_source in ('none', 'fake'):
            self.enable_yolo = False
        else:
            self.get_logger().warn(
                f'unknown detection_source={self.detection_source}; falling back to flask_topic on {self.external_detection_topic}'
            )
            self.external_detection_sub = self.create_subscription(
                String,
                self.external_detection_topic,
                self.on_external_detections,
                10,
            )
            self.enable_yolo = False
        self.clear_point_sub = self.create_subscription(PointStamped, '/risk/clear_point', self.on_clear_point, 10)
        self.clear_all_sub = self.create_subscription(Bool, '/risk/clear_all', self.on_clear_all, 10)

        self.pub_risk = self.create_publisher(OccupancyGrid, '/risk/risk_map', self.qos_grid_pub)
        self.pub_detection_candidate = None
        self.pub_positive_memory = None
        self.pub_visibility = None
        self.pub_observed_empty = None
        self.pub_room_probability = None
        self.pub_visual_seen = None
        self.pub_region_id = None
        self.pub_region_priority = None
        self.pub_region_checked = None
        self.pub_combined_priority = None
        if self.publish_diagnostic_maps:
            self.pub_detection_candidate = self.create_publisher(OccupancyGrid, '/risk/detection_candidate_map', self.qos_grid_pub)
            self.pub_positive_memory = self.create_publisher(OccupancyGrid, '/risk/positive_memory_map', self.qos_grid_pub)
            self.pub_visibility = self.create_publisher(OccupancyGrid, '/risk/visibility_map', self.qos_grid_pub)
            self.pub_observed_empty = self.create_publisher(OccupancyGrid, '/risk/observed_empty_map', self.qos_grid_pub)
            self.pub_room_probability = self.create_publisher(OccupancyGrid, '/risk/room_probability_map', self.qos_grid_pub)
            self.pub_visual_seen = self.create_publisher(OccupancyGrid, '/risk/visual_seen_map', self.qos_grid_pub)
            self.pub_region_id = self.create_publisher(OccupancyGrid, '/risk/region_id_map', self.qos_grid_pub)
            self.pub_region_priority = self.create_publisher(OccupancyGrid, '/risk/region_priority_map', self.qos_grid_pub)
            self.pub_region_checked = self.create_publisher(OccupancyGrid, '/risk/region_checked_map', self.qos_grid_pub)
            self.pub_combined_priority = self.create_publisher(OccupancyGrid, '/risk/combined_priority_map', self.qos_grid_pub)
        self.pub_markers = self.create_publisher(MarkerArray, '/risk/evidence_markers', self.qos_marker_pub)
        self.pub_overlay = self.create_publisher(Image, '/risk/overlay_image', self.qos_image_pub)
        self.pub_debug_image = self.create_publisher(Image, self.debug_image_topic, self.qos_image_pub)
        self.pub_debug_compressed_image = self.create_publisher(
            CompressedImage,
            self.debug_compressed_image_topic,
            self.qos_image_pub,
        )

        if self.debug_save_images:
            os.makedirs(self.debug_image_dir, exist_ok=True)

        # YOLO
        self.yolo = None
        if self.enable_yolo and (self.image_sub is not None or self.use_opencv_camera):
            try:
                from ultralytics import YOLO
                self.yolo = YOLO(self.model_path)
                self.get_logger().info(f'YOLO loaded: {self.model_path}')
            except Exception as e:
                self.get_logger().error(f'YOLO load failed: {e}')
                self.enable_yolo = False
        if self.enable_yolo and self.yolo is not None and self.yolo_async:
            self.start_yolo_worker()
        # Start camera capture independently of YOLO.
        # Frames are always captured; YOLO inference runs on top when available.
        if self.use_opencv_camera:
            self.open_opencv_camera()
            self.opencv_camera_timer = self.create_timer(
                1.0 / max(0.1, self.yolo_max_rate_hz),
                self.on_opencv_camera_timer,
            )

        self.timer = self.create_timer(1.0 / max(0.1, self.update_rate_hz), self.on_timer)
        if self.debug_log_image_status:
            self.debug_timer = self.create_timer(2.0, self.on_debug_timer)

        self.get_logger().info(
            'PERSISTENT_ROOM_RISK_V4_REGION_PRIORITY_TELEOP started | '
            'risk persists across Cartographer map resize; negative observations DO NOT reduce /risk/risk_map; '
            'they are published separately as /risk/observed_empty_map; region_id/priority maps are live SLAM diagnostics; '
            f'detection_source={self.detection_source} external_detection_topic={self.external_detection_topic} '
            f'pose_source={self.pose_source} pose_topic={self.pose_topic} map_qos_durability={self.map_qos_durability} '
            f'teleop_mode={self.teleop_mode} risk_publish_rate_hz={self.risk_publish_rate_hz:.2f} '
            f'risk_source_mode={self.risk_source_mode} evidence_radius={self.evidence_distribution_radius_m:.2f} '
            f'publish_diagnostic_maps={self.publish_diagnostic_maps} '
            f'debug_raw={self.publish_debug_image}:{self.debug_image_topic} '
            f'debug_compressed={self.publish_debug_compressed_image}:{self.debug_compressed_image_topic} '
            f'yolo_async={self.yolo_async}'
        )

    # ---------------- Image conversion without cv_bridge ----------------

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
        if enc == 'bgra8':
            if step < w * 4:
                raise ValueError(f'invalid step={step} for {enc} width={w}, need>={w * 4}')
            return rows[:, :w * 4].reshape((h, w, 4))[:, :, :3].copy()
        if enc == 'rgba8':
            if step < w * 4:
                raise ValueError(f'invalid step={step} for {enc} width={w}, need>={w * 4}')
            return rows[:, :w * 4].reshape((h, w, 4))[:, :, [2, 1, 0]].copy()
        if enc in ('mono8', '8uc1'):
            if step < w:
                raise ValueError(f'invalid step={step} for {enc} width={w}, need>={w}')
            gray = rows[:, :w].reshape((h, w))
            return np.repeat(gray[:, :, None], 3, axis=2).copy()

        raise ValueError(f'unsupported image encoding: {msg.encoding}')

    def bgr8_to_image_msg(self, img, header=None):
        msg = Image()
        if header is not None:
            msg.header = header
        msg.height = int(img.shape[0])
        msg.width = int(img.shape[1])
        msg.encoding = 'bgr8'
        msg.is_bigendian = 0
        msg.step = int(img.shape[1] * 3)
        msg.data = img.astype(np.uint8, copy=False).tobytes()
        return msg

    def bgr8_to_compressed_image_msg(self, img, header=None):
        msg = CompressedImage()
        if header is not None:
            msg.header = header
        msg.format = 'jpeg'
        try:
            import cv2
            quality = clamp(int(self.debug_compressed_jpeg_quality), 1, 100)
            encode_img = img
            resize_width = int(self.debug_compressed_resize_width)
            if resize_width > 0:
                h, w = img.shape[:2]
                if w > 0 and w != resize_width:
                    scale = resize_width / float(w)
                    encode_img = cv2.resize(img, (resize_width, max(1, int(h * scale))))
            ok, encoded = cv2.imencode(
                '.jpg',
                encode_img,
                [int(cv2.IMWRITE_JPEG_QUALITY), quality],
            )
            if not ok:
                raise ValueError('cv2.imencode returned false')
            msg.data = encoded.tobytes()
            return msg
        except Exception as e:
            self.get_logger().warn(f'debug JPEG encode failed: {e}', throttle_duration_sec=2.0)
            return None

    def publish_debug_frame(self, img, header=None):
        if self.publish_debug_image:
            self.pub_debug_image.publish(self.bgr8_to_image_msg(img, header))
        if self.publish_debug_compressed_image:
            now = time.time()
            if self.debug_compressed_publish_rate_hz > 0.0:
                min_period = 1.0 / max(0.1, self.debug_compressed_publish_rate_hz)
                if now - self.last_debug_compressed_publish_wall_sec < min_period:
                    return
            self.last_debug_compressed_publish_wall_sec = now
            msg = self.bgr8_to_compressed_image_msg(img, header)
            if msg is not None:
                self.pub_debug_compressed_image.publish(msg)

    # ---------------- Map / frame helpers ----------------

    def on_map(self, msg: OccupancyGrid):
        h = int(msg.info.height)
        w = int(msg.info.width)
        if h <= 0 or w <= 0:
            return

        data = np.array(msg.data, dtype=np.int16).reshape((h, w))
        res = float(msg.info.resolution)
        ox = float(msg.info.origin.position.x)
        oy = float(msg.info.origin.position.y)
        oyaw = yaw_from_quaternion(msg.info.origin.orientation)

        sig = (h, w, round(res, 6), round(ox, 4), round(oy, 4), round(oyaw, 4))

        self.latest_map_msg = msg
        self.occ_grid = data
        self.map_resolution = res
        self.map_origin_x = ox
        self.map_origin_y = oy
        self.map_origin_yaw = oyaw

        if sig != self.map_signature:
            old_geometry = self.prev_map_geometry
            new_geometry = {
                'height': h,
                'width': w,
                'resolution': res,
                'origin_x': ox,
                'origin_y': oy,
                'origin_yaw': oyaw,
            }

            if (
                self.preserve_risk_on_map_resize
                and old_geometry is not None
                and self.positive_memory_map is not None
            ):
                old_positive = self.positive_memory_map
                old_empty = self.observed_empty_map
                old_detection = self.detection_candidate_map
                old_seen = self.visual_seen_map
                old_region = self.region_id_map

                self.detection_candidate_map = self.reproject_layer_to_new_map(
                    old_detection, old_geometry, new_geometry
                )
                self.positive_memory_map = self.reproject_layer_to_new_map(
                    old_positive, old_geometry, new_geometry
                )
                self.observed_empty_map = self.reproject_layer_to_new_map(
                    old_empty, old_geometry, new_geometry
                )
                self.visibility_map = np.zeros((h, w), dtype=np.float32)
                self.room_probability_map = np.zeros((h, w), dtype=np.float32)
                self.visual_seen_map = self.reproject_layer_to_new_map(
                    old_seen, old_geometry, new_geometry
                )
                self.region_id_map = self.reproject_region_ids_to_new_map(
                    old_region, old_geometry, new_geometry
                )
                self.region_priority_map = np.zeros((h, w), dtype=np.float32)
                self.region_checked_map = np.zeros((h, w), dtype=np.float32)
                self.risk_map = np.zeros((h, w), dtype=np.float32)

                free = self.valid_free_mask()
                memory_mask = self.risk_memory_mask()
                self.detection_candidate_map[~free] = 0.0
                self.positive_memory_map[~memory_mask] = 0.0
                self.observed_empty_map[~free] = 0.0
                self.visual_seen_map[~free] = 0.0

                self.get_logger().warn(
                    f'map geometry changed: {sig}; persistent risk layers reprojected, not reset'
                )
            else:
                self.detection_candidate_map = np.zeros((h, w), dtype=np.float32)
                self.positive_memory_map = np.zeros((h, w), dtype=np.float32)
                self.risk_map = np.zeros((h, w), dtype=np.float32)
                self.observed_empty_map = np.zeros((h, w), dtype=np.float32)
                self.visibility_map = np.zeros((h, w), dtype=np.float32)
                self.room_probability_map = np.zeros((h, w), dtype=np.float32)
                self.visual_seen_map = np.zeros((h, w), dtype=np.float32)
                self.region_id_map = None
                self.region_priority_map = np.zeros((h, w), dtype=np.float32)
                self.region_checked_map = np.zeros((h, w), dtype=np.float32)
                self.region_states.clear()
                self.evidence_points.clear()
                self.get_logger().warn(f'map geometry initialized/changed: {sig}; internal maps initialized')

            self.map_signature = sig
            self.prev_map_geometry = new_geometry
            self.risk_dirty = True

    def grid_to_world_with_geometry(self, gx: int, gy: int, geom) -> Tuple[float, float]:
        lx = (gx + 0.5) * float(geom['resolution'])
        ly = (gy + 0.5) * float(geom['resolution'])
        yaw = float(geom.get('origin_yaw', 0.0))
        c = math.cos(yaw)
        s = math.sin(yaw)
        return (
            float(geom['origin_x']) + c * lx - s * ly,
            float(geom['origin_y']) + s * lx + c * ly,
        )

    def world_to_grid_with_geometry(self, x: float, y: float, geom) -> Optional[Tuple[int, int]]:
        w = int(geom['width'])
        h = int(geom['height'])
        dx = x - float(geom['origin_x'])
        dy = y - float(geom['origin_y'])
        yaw = float(geom.get('origin_yaw', 0.0))
        c = math.cos(-yaw)
        s = math.sin(-yaw)
        lx = c * dx - s * dy
        ly = s * dx + c * dy
        gx = int(math.floor(lx / float(geom['resolution'])))
        gy = int(math.floor(ly / float(geom['resolution'])))
        if gx < 0 or gx >= w or gy < 0 or gy >= h:
            return None
        return gx, gy

    def reproject_layer_to_new_map(self, old_arr, old_geom, new_geom):
        if old_arr is None:
            return np.zeros((int(new_geom['height']), int(new_geom['width'])), dtype=np.float32)

        new_arr = np.zeros((int(new_geom['height']), int(new_geom['width'])), dtype=np.float32)
        old_h, old_w = old_arr.shape
        ys, xs = np.where(old_arr > 1e-6)

        for y, x in zip(ys, xs):
            wx, wy = self.grid_to_world_with_geometry(int(x), int(y), old_geom)
            ng = self.world_to_grid_with_geometry(wx, wy, new_geom)
            if ng is None:
                continue
            nx, ny = ng
            val = float(old_arr[y, x])
            if val > new_arr[ny, nx]:
                new_arr[ny, nx] = val

        return new_arr

    def reproject_region_ids_to_new_map(self, old_arr, old_geom, new_geom):
        new_arr = np.zeros((int(new_geom['height']), int(new_geom['width'])), dtype=np.int32)
        if old_arr is None:
            return new_arr
        ys, xs = np.where(old_arr > 0)
        for y, x in zip(ys, xs):
            wx, wy = self.grid_to_world_with_geometry(int(x), int(y), old_geom)
            ng = self.world_to_grid_with_geometry(wx, wy, new_geom)
            if ng is None:
                continue
            nx, ny = ng
            new_arr[ny, nx] = int(old_arr[y, x])
        return new_arr

    def world_to_grid(self, x: float, y: float) -> Optional[Tuple[int, int]]:
        if self.occ_grid is None:
            return None
        h, w = self.occ_grid.shape
        dx = x - self.map_origin_x
        dy = y - self.map_origin_y
        c = math.cos(-self.map_origin_yaw)
        s = math.sin(-self.map_origin_yaw)
        lx = c * dx - s * dy
        ly = s * dx + c * dy
        gx = int(math.floor(lx / self.map_resolution))
        gy = int(math.floor(ly / self.map_resolution))
        if gx < 0 or gx >= w or gy < 0 or gy >= h:
            return None
        return gx, gy

    def grid_to_world(self, gx: int, gy: int) -> Tuple[float, float]:
        lx = (gx + 0.5) * self.map_resolution
        ly = (gy + 0.5) * self.map_resolution
        c = math.cos(self.map_origin_yaw)
        s = math.sin(self.map_origin_yaw)
        return (
            self.map_origin_x + c * lx - s * ly,
            self.map_origin_y + s * lx + c * ly,
        )

    def valid_free_mask(self):
        occ = self.occ_grid
        if self.allow_unknown:
            return occ < self.occupied_threshold
        return (occ >= 0) & (occ <= self.free_threshold)

    def risk_memory_mask(self):
        occ = self.occ_grid
        if self.risk_persist_in_unknown:
            return occ < self.occupied_threshold
        return self.valid_free_mask()

    def traversable(self, gy: int, gx: int):
        v = int(self.occ_grid[gy, gx])
        if v == -1:
            return self.allow_unknown
        if v >= self.occupied_threshold:
            return False
        return v <= self.free_threshold

    def get_robot_pose(self) -> Optional[Tuple[float, float, float]]:
        if self.pose_source in ('topic', 'pose_topic', 'pose'):
            if self.topic_pose is None or self.topic_pose_stamp is None:
                self.get_logger().warn(
                    f'pose topic wait: no pose received on {self.pose_topic}',
                    throttle_duration_sec=2.0,
                )
                return None
            if self.pose_topic_stale_sec > 0.0:
                try:
                    age = (self.get_clock().now() - self.topic_pose_stamp).nanoseconds * 1e-9
                except Exception:
                    age = float('inf')
                if age > self.pose_topic_stale_sec:
                    self.get_logger().warn(
                        f'pose topic stale: topic={self.pose_topic} age={age:.2f}s limit={self.pose_topic_stale_sec:.2f}s',
                        throttle_duration_sec=2.0,
                    )
                    return None
            p = self.topic_pose.pose.position
            q = self.topic_pose.pose.orientation
            return float(p.x), float(p.y), yaw_from_quaternion(q)

        # Always request latest available TF. Cartographer can lag slightly, and querying
        # a stamped time often causes "extrapolation into the past" during live mapping.
        candidate_frames = [self.base_frame]
        if self.base_frame != 'base_footprint':
            candidate_frames.append('base_footprint')
        if self.base_frame != 'base_link':
            candidate_frames.append('base_link')

        last_error = None
        for base in candidate_frames:
            try:
                tf = self.tf_buffer.lookup_transform(
                    self.map_frame,
                    base,
                    rclpy.time.Time(seconds=0, nanoseconds=0),
                    timeout=Duration(seconds=self.tf_timeout_sec),
                )
                t = tf.transform.translation
                q = tf.transform.rotation
                if base != self.base_frame:
                    self.get_logger().warn(
                        f'TF fallback used: {self.map_frame}->{base} instead of {self.base_frame}',
                        throttle_duration_sec=5.0
                    )
                return float(t.x), float(t.y), yaw_from_quaternion(q)
            except Exception as e:
                last_error = e

        self.get_logger().warn(
            f'TF lookup failed for candidates {candidate_frames}: {last_error}',
            throttle_duration_sec=2.0
        )
        return None

    def on_pose_topic(self, msg: PoseStamped):
        self.topic_pose = msg
        self.topic_pose_stamp = self.get_clock().now()
        pose_sec = self.header_to_sec(msg.header)
        if pose_sec is None:
            pose_sec = self.topic_pose_stamp.nanoseconds * 1e-9
        p = msg.pose.position
        q = msg.pose.orientation
        self.record_pose_sample(pose_sec, (float(p.x), float(p.y), yaw_from_quaternion(q)))

    def record_pose_sample(self, stamp_sec: float, pose):
        sample = PoseSample(
            stamp_sec=float(stamp_sec),
            x=float(pose[0]),
            y=float(pose[1]),
            yaw=float(pose[2]),
        )
        with self.pose_lock:
            self.pose_history.append(sample)
            cutoff = sample.stamp_sec - max(1.0, self.pose_history_duration_sec)
            while len(self.pose_history) > 2 and self.pose_history[0].stamp_sec < cutoff:
                self.pose_history.popleft()

    def lookup_pose_at(self, stamp_sec: float):
        if stamp_sec <= 0.0:
            return None
        with self.pose_lock:
            samples = list(self.pose_history)
        if not samples:
            return None
        oldest = samples[0]
        newest = samples[-1]
        max_error = max(0.0, self.pose_history_max_error_sec)

        if stamp_sec <= oldest.stamp_sec:
            if oldest.stamp_sec - stamp_sec <= max_error:
                return oldest.x, oldest.y, oldest.yaw
            return None
        if stamp_sec >= newest.stamp_sec:
            if stamp_sec - newest.stamp_sec <= max_error:
                return newest.x, newest.y, newest.yaw
            return None

        previous = oldest
        for current in list(samples)[1:]:
            if current.stamp_sec >= stamp_sec:
                dt = current.stamp_sec - previous.stamp_sec
                ratio = 0.0 if dt <= 1e-9 else clamp(
                    (stamp_sec - previous.stamp_sec) / dt, 0.0, 1.0
                )
                yaw_delta = wrap_angle(current.yaw - previous.yaw)
                return (
                    previous.x + ratio * (current.x - previous.x),
                    previous.y + ratio * (current.y - previous.y),
                    wrap_angle(previous.yaw + ratio * yaw_delta),
                )
            previous = current
        return newest.x, newest.y, newest.yaw

    # ---------------- Detection ----------------

    def start_yolo_worker(self):
        if self.yolo_worker_thread is not None:
            return
        self.yolo_worker_stop = False
        self.yolo_worker_thread = threading.Thread(
            target=self.yolo_worker_loop,
            name='risk_map_latest_yolo_worker',
            daemon=True,
        )
        self.yolo_worker_thread.start()
        self.get_logger().info('YOLO worker thread started | latest-frame-only=true')

    def stop_yolo_worker(self):
        thread = self.yolo_worker_thread
        if thread is None:
            return
        with self.yolo_condition:
            self.yolo_worker_stop = True
            self.yolo_condition.notify_all()
        thread.join(timeout=2.0)
        self.yolo_worker_thread = None

    def yolo_worker_loop(self):
        while True:
            with self.yolo_condition:
                while self.yolo_pending_frame is None and not self.yolo_worker_stop:
                    self.yolo_condition.wait(timeout=0.5)
                if self.yolo_worker_stop:
                    return
                item = self.yolo_pending_frame
                self.yolo_pending_frame = None
            if item is None:
                continue
            frame, encoding, header, capture_sec = item
            self.process_yolo_frame(frame, encoding=encoding, header=header, capture_sec=capture_sec)

    def enqueue_yolo_frame(self, frame, encoding='bgr8', header=None, capture_sec=None):
        if not self.enable_yolo or self.yolo is None:
            return False
        now_wall = time.time()
        if (
            self.yolo_max_rate_hz > 0.0
            and now_wall - self.last_yolo_wall_sec < 1.0 / self.yolo_max_rate_hz
        ):
            return False
        self.last_yolo_wall_sec = now_wall
        if not self.yolo_async:
            self.process_yolo_frame(frame, encoding=encoding, header=header, capture_sec=capture_sec)
            return True
        with self.yolo_condition:
            if self.yolo_pending_frame is not None:
                self.yolo_drop_count += 1
            self.yolo_pending_frame = (frame, encoding, header, capture_sec)
            self.yolo_condition.notify()
        return True

    def bbox_center_to_bearing(self, bbox, image_w):
        x1, y1, x2, y2 = bbox
        cx = 0.5 * (x1 + x2)
        fx = (image_w / 2.0) / math.tan(math.radians(self.camera_hfov_deg) / 2.0)
        return math.atan2(cx - image_w / 2.0, fx)

    def bbox_height_to_range(self, bbox, image_h):
        x1, y1, x2, y2 = bbox
        bbox_h = max(1.0, y2 - y1)
        fy = (image_h / 2.0) / math.tan(math.radians(self.camera_vfov_deg) / 2.0)
        return clamp(fy * self.real_person_height_m / bbox_h, self.min_range_m, self.max_range_m)

    def opencv_device_arg(self):
        raw = self.opencv_camera_device
        if isinstance(raw, int):
            return raw
        text = str(raw).strip()
        if text.isdigit():
            return int(text)
        return text

    def open_opencv_camera(self):
        try:
            import cv2
            self.opencv_cap = cv2.VideoCapture(self.opencv_device_arg())
            if self.opencv_camera_buffer_size > 0:
                self.opencv_cap.set(cv2.CAP_PROP_BUFFERSIZE, int(self.opencv_camera_buffer_size))
            if self.opencv_camera_fourcc:
                fourcc = self.opencv_camera_fourcc.upper()[:4]
                if len(fourcc) == 4:
                    self.opencv_cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
            if self.opencv_camera_width > 0:
                self.opencv_cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(self.opencv_camera_width))
            if self.opencv_camera_height > 0:
                self.opencv_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(self.opencv_camera_height))
            if self.opencv_camera_fps > 0.0:
                self.opencv_cap.set(cv2.CAP_PROP_FPS, float(self.opencv_camera_fps))
            if not self.opencv_cap.isOpened():
                self.get_logger().error(
                    f'OpenCV camera open failed: device={self.opencv_camera_device}'
                )
                return False
            self.get_logger().info(
                f'OpenCV camera opened directly: device={self.opencv_camera_device} '
                f'size={self.opencv_camera_width}x{self.opencv_camera_height} '
                f'fps={self.opencv_camera_fps:.1f} fourcc={self.opencv_camera_fourcc or "default"} '
                f'async_capture={self.opencv_async_capture}'
            )
            self.opencv_read_fail_count = 0
            self.opencv_camera_warned = False
            if self.opencv_async_capture:
                self.start_opencv_capture_worker()
            return True
        except Exception as e:
            self.get_logger().error(f'OpenCV camera setup failed: {e}')
            self.opencv_cap = None
            return False

    def reset_opencv_camera(self):
        self.stop_opencv_capture_worker()
        cap = self.opencv_cap
        self.opencv_cap = None
        if cap is not None:
            try:
                cap.release()
            except Exception:
                pass

    def start_opencv_capture_worker(self):
        if self.opencv_capture_thread is not None:
            return
        self.opencv_capture_stop = False
        self.opencv_capture_thread = threading.Thread(
            target=self.opencv_capture_loop,
            name='risk_map_latest_camera_worker',
            daemon=True,
        )
        self.opencv_capture_thread.start()
        self.get_logger().info('OpenCV camera worker started | latest-frame-only=true')

    def stop_opencv_capture_worker(self):
        thread = self.opencv_capture_thread
        if thread is None:
            return
        self.opencv_capture_stop = True
        thread.join(timeout=2.0)
        self.opencv_capture_thread = None

    def opencv_capture_loop(self):
        while not self.opencv_capture_stop:
            cap = self.opencv_cap
            if cap is None or not cap.isOpened():
                time.sleep(0.05)
                continue
            ok, frame = cap.read()
            if not ok or frame is None:
                self.opencv_read_fail_count += 1
                if self.opencv_read_fail_count >= max(1, int(self.opencv_reopen_after_failures)):
                    self.get_logger().warn(
                        f'OpenCV camera worker read failed {self.opencv_read_fail_count} times',
                        throttle_duration_sec=2.0,
                    )
                time.sleep(0.02)
                continue
            capture_sec = self.get_clock().now().nanoseconds * 1e-9
            with self.opencv_frame_lock:
                self.opencv_latest_frame = frame
                self.opencv_latest_capture_sec = capture_sec
                self.opencv_latest_seq += 1
            self.opencv_read_fail_count = 0

    def latest_opencv_frame(self):
        with self.opencv_frame_lock:
            if self.opencv_latest_frame is None:
                return None, 0.0, 0
            return self.opencv_latest_frame.copy(), self.opencv_latest_capture_sec, self.opencv_latest_seq

    def on_opencv_camera_timer(self):
        if not self.use_opencv_camera:
            return
        if self.opencv_cap is None or not self.opencv_cap.isOpened():
            if not self.opencv_camera_warned:
                self.opencv_camera_warned = True
                self.get_logger().warn(
                    f'OpenCV camera is not open; retrying device={self.opencv_camera_device}'
                )
            self.open_opencv_camera()
            return

        capture_sec = None
        if self.opencv_async_capture:
            frame, capture_sec, seq = self.latest_opencv_frame()
            if frame is None:
                self.get_logger().warn('waiting for latest OpenCV camera frame...', throttle_duration_sec=2.0)
                return
            if seq == self.opencv_consumed_seq:
                return
            self.opencv_consumed_seq = seq
        else:
            ok, frame = self.opencv_cap.read()
            if not ok or frame is None:
                self.opencv_read_fail_count += 1
                self.get_logger().warn('OpenCV camera frame read failed', throttle_duration_sec=2.0)
                if self.opencv_read_fail_count >= max(1, int(self.opencv_reopen_after_failures)):
                    self.get_logger().warn(
                        f'OpenCV camera read failed {self.opencv_read_fail_count} times; reopening camera',
                        throttle_duration_sec=2.0,
                    )
                    self.reset_opencv_camera()
                    self.open_opencv_camera()
                return
            self.opencv_read_fail_count = 0
            capture_sec = self.get_clock().now().nanoseconds * 1e-9

        # Always run YOLO if available; otherwise publish raw frame as debug image.
        if self.enable_yolo and self.yolo is not None:
            self.enqueue_yolo_frame(frame, encoding='opencv_bgr8', header=None, capture_sec=capture_sec)
        else:
            self.image_rx_count += 1
            self.last_image_encoding = 'opencv_bgr8_raw'
            h, w = frame.shape[:2]
            self.last_image_shape = f'{w}x{h} opencv_bgr8_raw'
            self.publish_debug_frame(frame, None)

    def on_image(self, msg: Image):
        if not self.enable_yolo or self.yolo is None:
            return

        try:
            frame = self.image_msg_to_bgr8(msg)
        except Exception as e:
            self.get_logger().warn(f'image conversion failed: {e}', throttle_duration_sec=2.0)
            return

        self.enqueue_yolo_frame(frame, encoding=msg.encoding, header=msg.header)

    def header_to_sec(self, header):
        if header is None:
            return None
        try:
            sec = float(header.stamp.sec) + float(header.stamp.nanosec) * 1e-9
            return sec if sec > 0.0 else None
        except Exception:
            return None

    def update_detection_capture_pose(self, header=None, capture_sec=None):
        now_ros_sec = self.get_clock().now().nanoseconds * 1e-9
        if capture_sec is None:
            capture_sec = self.header_to_sec(header)
        if capture_sec is None:
            capture_sec = now_ros_sec
        self.latest_detection_capture_sec = capture_sec
        self.latest_detection_pose = self.lookup_pose_at(capture_sec)
        if self.latest_detection_pose is None:
            # Fallback for startup/direct-camera cases where the pose history is still sparse.
            self.latest_detection_pose = self.get_robot_pose()
        self.latest_detection_delay_ms = max(0.0, (now_ros_sec - capture_sec) * 1000.0)

    def process_yolo_frame(self, frame, encoding='bgr8', header=None, capture_sec=None):
        self.image_rx_count += 1
        self.last_image_encoding = encoding

        h, w = frame.shape[:2]
        self.last_image_shape = f'{w}x{h} {encoding}'

        detections = []
        try:
            results = self.yolo.predict(
                source=frame,
                imgsz=self.yolo_imgsz,
                conf=self.conf_threshold,
                classes=[0],
                device=self.device,
                verbose=False,
            )
            if results and results[0].boxes is not None:
                xyxy = results[0].boxes.xyxy.cpu().numpy()
                confs = results[0].boxes.conf.cpu().numpy()
                for box, conf in zip(xyxy, confs):
                    conf = float(conf)
                    if conf < self.conf_threshold:
                        continue
                    bbox = tuple(float(v) for v in box)
                    detections.append(Detection2D(
                        bbox=bbox,
                        conf=conf,
                        bearing_rad=self.bbox_center_to_bearing(bbox, w),
                        range_hat_m=self.bbox_height_to_range(bbox, h),
                    ))
        except Exception as e:
            self.get_logger().warn(f'YOLO failed: {e}', throttle_duration_sec=2.0)
            return

        with self.detection_lock:
            self.yolo_frame_count += 1
            self.yolo_det_count += len(detections)
            self.latest_detections = detections
            self.last_yolo_ros_sec = self.get_clock().now().nanoseconds * 1e-9
            self.update_detection_capture_pose(header, capture_sec)

        overlay = self.make_overlay(frame, detections)
        if overlay is None:
            # Fallback: publish raw frame even when overlay failed or YOLO is off
            if self.publish_yolo_debug_even_without_detection:
                self.publish_debug_frame(frame, header)
        else:
            if self.publish_overlay:
                self.pub_overlay.publish(self.bgr8_to_image_msg(overlay, header))
            self.publish_debug_frame(overlay, header)
            self.debug_output_image(overlay)

    def on_external_detections(self, msg: String):
        self.external_detection_rx_count += 1
        try:
            payload = json.loads(msg.data)
        except Exception as e:
            self.get_logger().warn(f'external detection JSON parse failed: {e}', throttle_duration_sec=2.0)
            return

        image_w = int(payload.get('image_width') or payload.get('width') or 0)
        image_h = int(payload.get('image_height') or payload.get('height') or 0)
        if image_w <= 0 or image_h <= 0:
            meta = payload.get('image') if isinstance(payload.get('image'), dict) else {}
            image_w = int(meta.get('width') or 0)
            image_h = int(meta.get('height') or 0)
        if image_w <= 0 or image_h <= 0:
            self.get_logger().warn('external detection ignored: missing image_width/image_height', throttle_duration_sec=2.0)
            return

        raw_dets = payload.get('detections', [])
        if not isinstance(raw_dets, list):
            self.get_logger().warn('external detection ignored: detections is not a list', throttle_duration_sec=2.0)
            return
        max_count = max(1, int(self.external_detection_max_count))
        if len(raw_dets) > max_count:
            self.get_logger().warn(
                f'external detections truncated: {len(raw_dets)} -> {max_count}',
                throttle_duration_sec=2.0,
            )
            raw_dets = raw_dets[:max_count]

        detections: List[Detection2D] = []
        for item in raw_dets:
            if not isinstance(item, dict):
                continue
            conf = float(item.get('conf', item.get('confidence', 0.0)))
            if conf < self.conf_threshold:
                continue
            label = str(item.get('label', item.get('name', ''))).lower()
            cls = item.get('class_id', item.get('class', item.get('cls', None)))
            is_person = label == 'person' or cls == 0 or str(cls) == '0'
            if self.external_person_only and not is_person:
                continue

            bbox_raw = item.get('bbox', item.get('xyxy', None))
            if bbox_raw is None and all(k in item for k in ('x1', 'y1', 'x2', 'y2')):
                bbox_raw = [item['x1'], item['y1'], item['x2'], item['y2']]
            if not isinstance(bbox_raw, (list, tuple)) or len(bbox_raw) != 4:
                continue
            bbox = tuple(float(v) for v in bbox_raw)
            bearing = item.get('bearing_rad', None)
            range_hat = item.get('range_hat_m', item.get('range_m', None))
            detections.append(Detection2D(
                bbox=bbox,
                conf=conf,
                bearing_rad=float(bearing) if bearing is not None else self.bbox_center_to_bearing(bbox, image_w),
                range_hat_m=float(range_hat) if range_hat is not None else self.bbox_height_to_range(bbox, image_h),
            ))

        capture_sec = float(
            payload.get('capture_ros_sec')
            or payload.get('capture_wall_sec')
            or 0.0
        )
        with self.detection_lock:
            self.yolo_frame_count += 1
            self.yolo_det_count += len(detections)
            self.latest_detections = detections
            self.last_yolo_ros_sec = self.get_clock().now().nanoseconds * 1e-9
            self.latest_detection_capture_sec = capture_sec if capture_sec > 0.0 else None
            self.latest_detection_pose = self.lookup_pose_at(capture_sec)
            if self.latest_detection_pose is None:
                self.latest_detection_pose = self.get_robot_pose()
            if capture_sec > 0.0:
                self.latest_detection_delay_ms = max(
                    0.0, (self.last_yolo_ros_sec - capture_sec) * 1000.0
                )
            else:
                self.latest_detection_delay_ms = -1.0
        self.last_image_shape = f'{image_w}x{image_h} flask_json'
        self.last_image_encoding = 'external_json'

    def maybe_make_fake_detection(self):
        if not self.enable_fake_detection:
            return []
        now = time.time()
        if now - self.last_fake_wall_sec < self.fake_detection_interval_sec:
            return []
        self.last_fake_wall_sec = now
        return [Detection2D(
            bbox=(0.0, 0.0, 1.0, 1.0),
            conf=self.fake_confidence,
            bearing_rad=math.radians(self.fake_bearing_deg),
            range_hat_m=self.fake_range_m,
        )]

    def make_overlay(self, frame, detections):
        try:
            import cv2
            img = frame.copy()
            h, w = img.shape[:2]
            cv2.putText(img, f'ROOM_RISK_V2 det={len(detections)} frame={w}x{h}',
                        (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
            for det in detections:
                x1, y1, x2, y2 = [int(v) for v in det.bbox]
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f'person {det.conf:.2f} r~{det.range_hat_m:.1f}m b={math.degrees(det.bearing_rad):.1f}'
                cv2.putText(img, label, (x1, max(15, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            return img
        except Exception:
            return None

    def debug_output_image(self, img):
        now = time.time()
        if self.debug_save_images:
            min_dt = 1.0 / max(0.1, self.debug_image_rate_hz)
            if now - self.last_debug_save_wall_sec >= min_dt:
                self.last_debug_save_wall_sec = now
                try:
                    import cv2
                    os.makedirs(self.debug_image_dir, exist_ok=True)
                    path = os.path.join(self.debug_image_dir, f'room_risk_v2_{int(now * 1000)}.jpg')
                    cv2.imwrite(path, img)
                except Exception:
                    pass

        if self.debug_show_opencv:
            try:
                import cv2
                cv2.imshow('tb3_bayesian_risk_map ROOM_RISK_V2', img)
                cv2.waitKey(1)
            except Exception:
                self.debug_show_opencv = False

    def on_debug_timer(self):
        with self.detection_lock:
            yolo_frame_count = self.yolo_frame_count
            yolo_det_count = self.yolo_det_count
            current_det_count = len(self.latest_detections)
            yolo_drop_count = self.yolo_drop_count
        self.get_logger().info(
            f'YOLO_DEBUG | image_rx={self.image_rx_count} | external_rx={self.external_detection_rx_count} | yolo_frames={yolo_frame_count} | '
            f'total_dets={yolo_det_count} | current_dets={current_det_count} | yolo_dropped={yolo_drop_count} | '
            f'positive_max={float(np.max(self.positive_memory_map)) if self.positive_memory_map is not None else 0.0:.3f} | '
            f'risk_max={float(np.max(self.risk_map)) if self.risk_map is not None else 0.0:.3f} | '
            f'last_shape={self.last_image_shape} | enc={self.last_image_encoding} | '
            f'capture_delay_ms={self.latest_detection_delay_ms:.1f} | '
            f'history_pose={self.latest_detection_pose is not None} | '
            f'source={self.detection_source} image_topic={self.image_topic} external_topic={self.external_detection_topic}',
            throttle_duration_sec=2.0
        )

    # ---------------- Positive candidate / empty observation ----------------

    def build_detection_candidate_map(self, robot_pose, detections):
        h, w = self.occ_grid.shape
        out = np.zeros((h, w), dtype=np.float32)
        if not detections:
            return out

        rx, ry, ryaw = robot_pose
        bearing_sigma = math.radians(self.bearing_sigma_deg)
        angle_step = math.radians(max(0.25, self.angular_sample_step_deg))
        r_step = max(self.map_resolution, 0.03)

        for det in detections:
            theta0 = ryaw + det.bearing_rad
            width = max(3.0 * bearing_sigma, angle_step)
            n = max(1, int(math.ceil((2.0 * width) / angle_step)))

            max_cell = None
            max_val = 0.0

            for i in range(n + 1):
                theta = theta0 - width + i * (2.0 * width / n)
                aw = math.exp(-0.5 * (wrap_angle(theta - theta0) / max(bearing_sigma, 1e-6)) ** 2)

                r = self.min_range_m
                while r <= self.max_range_m + 1e-6:
                    x = rx + r * math.cos(theta)
                    y = ry + r * math.sin(theta)
                    g = self.world_to_grid(x, y)
                    if g is None:
                        break
                    gx, gy = g
                    if not self.traversable(gy, gx):
                        break

                    if self.use_bbox_range_prior:
                        rw = math.exp(-0.5 * ((r - det.range_hat_m) / max(self.range_sigma_m, 1e-6)) ** 2)
                    else:
                        rw = 1.0

                    val = det.conf * aw * rw
                    if val >= self.source_min_value:
                        if val > out[gy, gx]:
                            out[gy, gx] = val
                        if val > max_val:
                            max_val = val
                            max_cell = (gx, gy)
                    r += r_step

            if max_cell is not None:
                wx, wy = self.grid_to_world(max_cell[0], max_cell[1])
                self.evidence_points.append(EvidencePoint(
                    evidence_id=self.next_evidence_id,
                    x=wx,
                    y=wy,
                    confidence=max_val,
                    stamp_sec=self.get_clock().now().nanoseconds * 1e-9,
                ))
                self.next_evidence_id += 1
                if len(self.evidence_points) > 200:
                    self.evidence_points = self.evidence_points[-200:]

        out[~self.valid_free_mask()] = 0.0
        return np.clip(out, 0.0, 1.0)

    def update_positive_memory(self, candidate):
        if np.max(candidate) <= 1e-6:
            return False
        # Persistent positive evidence. No negative subtraction.
        alpha = clamp(self.positive_memory_alpha, 0.0, 1.0)
        fused = 1.0 - (1.0 - self.positive_memory_map) * (1.0 - alpha * candidate)
        self.positive_memory_map = np.maximum(self.positive_memory_map, fused)
        self.positive_memory_map[~self.risk_memory_mask()] = 0.0
        self.risk_dirty = True
        return True

    def compute_visibility_map(self, robot_pose):
        h, w = self.occ_grid.shape
        vis = np.zeros((h, w), dtype=np.float32)

        rx, ry, ryaw = robot_pose
        hfov = math.radians(self.camera_hfov_deg)
        r_step = max(self.map_resolution, 0.03)
        n = max(3, self.visibility_num_rays)

        for i in range(n):
            b = -0.5 * hfov + i * hfov / (n - 1)
            th = ryaw + b
            r = self.min_range_m
            while r <= self.max_range_m + 1e-6:
                x = rx + r * math.cos(th)
                y = ry + r * math.sin(th)
                g = self.world_to_grid(x, y)
                if g is None:
                    break
                gx, gy = g
                if not self.traversable(gy, gx):
                    break
                vis[gy, gx] = 1.0
                r += r_step

        return vis

    def update_observed_empty(self, visibility, had_detection):
        if not self.enable_empty_observation_map:
            return
        # This is only a separate "we looked here and did not get a detection" map.
        # It is not subtracted from risk.
        if had_detection:
            return
        a = clamp(self.observed_empty_alpha, 0.0, 1.0)
        self.observed_empty_map = np.maximum(self.observed_empty_map, a * visibility)
        self.observed_empty_map[~self.valid_free_mask()] = 0.0

    # ---------------- Live SLAM region segmentation / priority ----------------

    def connected_components(self, free_mask):
        import cv2

        count, raw_labels, stats, _ = cv2.connectedComponentsWithStats(
            free_mask.astype(np.uint8), connectivity=4
        )
        labels = raw_labels.astype(np.int32) - 1
        labels[~free_mask] = -1
        sizes = stats[1:count, cv2.CC_STAT_AREA].astype(np.int32)
        return labels, sizes

    def obstacle_unknown_blocked_mask(self):
        occ = self.occ_grid
        # For segmentation, unknown must act as blocked. Otherwise a partial SLAM map
        # leaks regions through not-yet-observed space.
        return (occ < 0) | (occ >= self.occupied_threshold)

    def known_free_mask(self):
        return (self.occ_grid >= 0) & (self.occ_grid <= self.free_threshold)

    def compute_clearance_to_blocked(self, blocked):
        import cv2

        free_image = (~blocked).astype(np.uint8)
        return cv2.distanceTransform(free_image, cv2.DIST_L2, 3) * float(self.map_resolution)

    def frontier_boundary_mask(self, free):
        import cv2

        unknown = self.occ_grid < 0
        kernel = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8)
        near_unknown = cv2.dilate(unknown.astype(np.uint8), kernel, iterations=1).astype(bool)
        return free & near_unknown

    def obstacle_neighbor_mask(self, free):
        import cv2

        occ = self.occ_grid >= self.occupied_threshold
        kernel = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8)
        near_obstacle = cv2.dilate(occ.astype(np.uint8), kernel, iterations=1).astype(bool)
        return free & near_obstacle

    def allocate_region_id_for_component(self, comp_mask):
        if self.region_id_map is None or self.region_id_map.shape != comp_mask.shape:
            rid = self.next_region_id
            self.next_region_id += 1
            return rid

        old_ids, counts = np.unique(self.region_id_map[comp_mask], return_counts=True)
        candidates = [(int(r), int(c)) for r, c in zip(old_ids, counts) if int(r) > 0]
        if not candidates:
            rid = self.next_region_id
            self.next_region_id += 1
            return rid

        best_id, best_overlap = max(candidates, key=lambda rc: rc[1])
        old_area = int(np.sum(self.region_id_map == best_id))
        new_area = int(np.sum(comp_mask))
        union = max(1, old_area + new_area - best_overlap)
        iou = best_overlap / float(union)
        if iou >= self.region_iou_match_threshold or best_overlap >= max(25, 0.35 * new_area):
            return best_id

        rid = self.next_region_id
        self.next_region_id += 1
        return rid

    def expand_region_labels(self, seed_labels, free, clearance):
        h, w = seed_labels.shape
        labels = seed_labels.copy()
        q = deque()
        ys, xs = np.where(labels > 0)
        for y, x in zip(ys, xs):
            q.append((int(x), int(y), int(labels[y, x])))

        min_clear = max(0.0, float(self.region_expand_clearance_m))
        while q:
            x, y, rid = q.popleft()
            for nx, ny in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
                if nx < 0 or nx >= w or ny < 0 or ny >= h:
                    continue
                if labels[ny, nx] != 0:
                    continue
                if not free[ny, nx]:
                    continue
                if clearance[ny, nx] < min_clear:
                    continue
                labels[ny, nx] = rid
                q.append((nx, ny, rid))
        return labels

    def update_visual_seen_map(self, visibility):
        if visibility is None:
            return
        if self.visual_seen_map is None or self.visual_seen_map.shape != visibility.shape:
            self.visual_seen_map = np.zeros_like(visibility, dtype=np.float32)
        self.visual_seen_map = np.maximum(self.visual_seen_map, visibility.astype(np.float32))
        self.visual_seen_map[~self.valid_free_mask()] = 0.0

    def update_region_segmentation(self, force=False):
        if not self.enable_region_segmentation:
            return
        if self.occ_grid is None or self.map_resolution is None:
            return
        now = time.time()
        if not force and now - self.last_region_update_wall_sec < self.region_update_period_sec:
            return
        self.last_region_update_wall_sec = now

        h, w = self.occ_grid.shape
        free = self.known_free_mask()
        blocked = self.obstacle_unknown_blocked_mask()
        clearance = self.compute_clearance_to_blocked(blocked)

        core_clear = max(float(self.region_core_clearance_m), float(self.map_resolution))
        core = free & (clearance >= core_clear)

        comp_labels, comp_sizes = self.connected_components(core)
        min_cells = max(8, int(math.ceil(self.min_region_area_m2 / max(self.map_resolution ** 2, 1e-9))))

        labels = np.zeros((h, w), dtype=np.int32)
        labels[self.occ_grid >= self.occupied_threshold] = -1
        labels[self.occ_grid < 0] = -2

        for cid, size in enumerate(comp_sizes):
            if int(size) < min_cells:
                continue
            comp_mask = comp_labels == cid
            rid = self.allocate_region_id_for_component(comp_mask)
            labels[comp_mask] = rid

        labels = self.expand_region_labels(labels, free, clearance)
        labels[self.occ_grid >= self.occupied_threshold] = -1
        labels[self.occ_grid < 0] = -2
        self.region_id_map = labels

        self.update_region_states()
        self.build_region_priority_map()
        self.log_region_debug_periodic()

    def update_region_states(self):
        if self.region_id_map is None:
            return
        states: Dict[int, RegionState] = {}
        free = self.known_free_mask()
        frontier = self.frontier_boundary_mask(free)
        obstacle_near = self.obstacle_neighbor_mask(free)
        seen = self.visual_seen_map if self.visual_seen_map is not None else np.zeros_like(self.occ_grid, dtype=np.float32)
        positive = self.positive_memory_map if self.positive_memory_map is not None else np.zeros_like(self.occ_grid, dtype=np.float32)
        now_ros = self.get_clock().now().nanoseconds * 1e-9

        for rid in sorted(int(r) for r in np.unique(self.region_id_map) if int(r) > 0):
            mask = self.region_id_map == rid
            area = int(np.sum(mask))
            if area <= 0:
                continue
            ys, xs = np.where(mask)
            mean_gx = float(np.mean(xs))
            mean_gy = float(np.mean(ys))
            centroid_x, centroid_y = self.grid_to_world(mean_gx, mean_gy)
            coverage_ratio = float(np.mean(seen[mask] > 0.5)) if area > 0 else 0.0
            frontier_ratio = float(np.sum(frontier[mask])) / float(max(1, area))
            obstacle_density = float(np.sum(obstacle_near[mask])) / float(max(1, area))
            person_risk = float(np.max(positive[mask])) if area > 0 else 0.0

            frontier_score = clamp(frontier_ratio * self.region_frontier_gain_scale, 0.0, 1.0)
            obstacle_score = clamp(obstacle_density * self.region_obstacle_gain_scale, 0.0, 1.0)
            unchecked = clamp(1.0 - coverage_ratio, 0.0, 1.0)
            structural_risk = clamp(0.55 * frontier_score + 0.45 * obstacle_score, 0.0, 1.0)
            checked = coverage_ratio >= self.region_checked_coverage_ratio and person_risk < 0.05
            priority = 0.0 if checked else 100.0 * clamp(
                0.45 * unchecked + 0.30 * structural_risk + 0.25 * person_risk,
                0.0,
                1.0,
            )

            states[rid] = RegionState(
                region_id=rid,
                area_cells=area,
                centroid_x=centroid_x,
                centroid_y=centroid_y,
                coverage_ratio=coverage_ratio,
                frontier_ratio=frontier_ratio,
                obstacle_density=obstacle_density,
                structural_risk=structural_risk,
                person_risk=person_risk,
                priority=priority,
                checked=checked,
                last_seen_sec=now_ros if coverage_ratio > 0.0 else 0.0,
            )
        self.region_states = states

    def build_region_priority_map(self):
        if self.region_id_map is None:
            return
        h, w = self.region_id_map.shape
        pri = np.zeros((h, w), dtype=np.float32)
        chk = np.zeros((h, w), dtype=np.float32)
        for rid, st in self.region_states.items():
            mask = self.region_id_map == rid
            pri[mask] = float(st.priority) / 100.0
            chk[mask] = 1.0 if st.checked else float(st.coverage_ratio)
        pri[~self.valid_free_mask()] = 0.0
        chk[~self.valid_free_mask()] = 0.0
        self.region_priority_map = np.clip(pri, 0.0, 1.0)
        self.region_checked_map = np.clip(chk, 0.0, 1.0)

    def log_region_debug_periodic(self):
        now = time.time()
        if now - self.last_region_debug_wall_sec < self.region_debug_log_period_sec:
            return
        self.last_region_debug_wall_sec = now
        if not self.region_states:
            self.get_logger().info('REGION_DEBUG | no stable regions yet')
            return
        top = sorted(self.region_states.values(), key=lambda st: st.priority, reverse=True)[:5]
        text = '; '.join(
            f'id={st.region_id} pri={st.priority:.1f} cov={st.coverage_ratio:.2f} '
            f'front={st.frontier_ratio:.3f} obs={st.obstacle_density:.3f} area={st.area_cells}'
            for st in top
        )
        self.get_logger().info(f'REGION_DEBUG | n={len(self.region_states)} | {text}')

    def build_room_probability_map(self):
        out = np.zeros_like(self.positive_memory_map, dtype=np.float32)
        if not self.enable_room_probability:
            return out

        # Prefer persistent live regions when available. The older connected-component
        # diagnostic is too broad in connected houses/corridors.
        if self.region_id_map is not None and self.region_states:
            scores = {rid: float(np.sum(self.positive_memory_map[self.region_id_map == rid]))
                      for rid in self.region_states.keys()}
            total = float(sum(scores.values()))
            if total <= 1e-6:
                return out
            ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:max(1, self.room_top_k)]
            for rid, score in ranked:
                if score <= 0.0:
                    continue
                out[self.region_id_map == rid] = score / total
            out[~self.valid_free_mask()] = 0.0
            return np.clip(out, 0.0, 1.0)

        free = self.valid_free_mask()
        labels, sizes = self.connected_components(free)
        if len(sizes) == 0:
            return out

        scores = np.zeros(len(sizes), dtype=np.float32)
        src = self.positive_memory_map
        ys, xs = np.where(src > self.room_min_score)
        for y, x in zip(ys, xs):
            cid = labels[y, x]
            if cid >= 0:
                scores[cid] += float(src[y, x])

        total = float(np.sum(scores))
        if total <= 1e-6:
            return out

        probs = scores / total
        keep = np.argsort(-probs)[:max(1, self.room_top_k)]
        for cid in keep:
            if probs[cid] <= 0.0:
                continue
            out[labels == cid] = float(probs[cid])

        out[~free] = 0.0
        return np.clip(out, 0.0, 1.0)

    # ---------------- Bounded geodesic halo ----------------

    def select_source_seeds(self, source):
        flat = source.ravel()
        idx = np.where(flat >= self.source_halo_seed_threshold)[0]
        if idx.size == 0:
            return []
        vals = flat[idx]
        order = np.argsort(-vals)
        seeds = []
        _, w = source.shape
        min_sep_cells = max(
            1.0,
            float(self.source_halo_seed_separation_m) / max(float(self.map_resolution), 1e-6),
        )
        min_sep_sq = min_sep_cells * min_sep_cells
        for i in idx[order]:
            y = int(i // w)
            x = int(i % w)
            if any((x - sx) ** 2 + (y - sy) ** 2 < min_sep_sq for sx, sy, _ in seeds):
                continue
            seeds.append((x, y, float(source[y, x])))
            if len(seeds) >= max(1, self.source_halo_top_k):
                break
        return seeds

    def build_bounded_geodesic_halo(self, source):
        h, w = source.shape
        halo = np.zeros((h, w), dtype=np.float32)
        free = self.valid_free_mask()
        memory_mask = self.risk_memory_mask()
        seeds = self.select_source_seeds(source)
        if not seeds:
            return halo

        max_cells = max(1, int(math.ceil(self.source_halo_radius_m / self.map_resolution)))
        sigma = max(self.source_halo_sigma_m, self.map_resolution)

        for sx, sy, sval in seeds:
            if not memory_mask[sy, sx]:
                continue
            halo[sy, sx] = max(float(halo[sy, sx]), float(sval))
            if not free[sy, sx]:
                continue
            x0 = max(0, sx - max_cells)
            x1 = min(w, sx + max_cells + 1)
            y0 = max(0, sy - max_cells)
            y1 = min(h, sy + max_cells + 1)
            local_free = free[y0:y1, x0:x1]
            local_dist = np.full(local_free.shape, np.inf, dtype=np.float32)
            lsx, lsy = sx - x0, sy - y0
            local_dist[lsy, lsx] = 0.0
            q = deque([(lsx, lsy)])
            while q:
                x, y = q.popleft()
                d = float(local_dist[y, x])
                if d > self.source_halo_radius_m:
                    continue

                gain = math.exp(-0.5 * (d / sigma) ** 2)
                val = sval * gain
                gy = y + y0
                gx = x + x0
                if val > halo[gy, gx]:
                    halo[gy, gx] = val

                for nx, ny in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
                    if nx < 0 or nx >= local_free.shape[1] or ny < 0 or ny >= local_free.shape[0]:
                        continue
                    if not local_free[ny, nx]:
                        continue
                    nd = d + self.map_resolution
                    if nd <= self.source_halo_radius_m and nd < local_dist[ny, nx]:
                        local_dist[ny, nx] = nd
                        q.append((nx, ny))

        halo[~memory_mask] = 0.0
        return np.clip(halo, 0.0, 1.0)

    def build_evidence_source_map(self):
        out = np.zeros_like(self.positive_memory_map, dtype=np.float32)
        if not self.evidence_points:
            return out

        h, w = out.shape
        free = self.valid_free_mask()
        memory_mask = self.risk_memory_mask()
        gain = max(0.0, float(self.evidence_source_gain))
        radius_m = max(self.map_resolution, float(self.evidence_distribution_radius_m))
        sigma_m = max(self.map_resolution, float(self.evidence_distribution_sigma_m))
        for ev in self.evidence_points:
            g = self.world_to_grid(float(ev.x), float(ev.y))
            if g is None:
                continue
            sx, sy = g
            if not memory_mask[sy, sx]:
                continue

            max_cells = max(1, int(math.ceil(radius_m / self.map_resolution)))
            x0 = max(0, sx - max_cells)
            x1 = min(w, sx + max_cells + 1)
            y0 = max(0, sy - max_cells)
            y1 = min(h, sy + max_cells + 1)

            local_dist = np.full((y1 - y0, x1 - x0), np.inf, dtype=np.float32)
            local_dist[sy - y0, sx - x0] = 0.0
            q = deque([(sx - x0, sy - y0)])
            while q:
                x, y = q.popleft()
                d = float(local_dist[y, x])
                if d > radius_m:
                    continue

                gy = y + y0
                gx = x + x0
                if memory_mask[gy, gx]:
                    kernel = math.exp(-0.5 * (d / sigma_m) ** 2)
                    val = clamp(float(ev.confidence) * gain * kernel, 0.0, 1.0)
                    if val >= self.source_min_value:
                        # Probabilistic union makes overlapping marker kernels stronger,
                        # so clustered evidence darkens and nearby pairs reinforce the middle.
                        out[gy, gx] = 1.0 - (1.0 - out[gy, gx]) * (1.0 - val)

                for nx, ny in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
                    if nx < 0 or nx >= local_dist.shape[1] or ny < 0 or ny >= local_dist.shape[0]:
                        continue
                    ngx = nx + x0
                    ngy = ny + y0
                    if not free[ngy, ngx] and not (ngx == sx and ngy == sy):
                        continue
                    nd = d + self.map_resolution
                    if nd <= radius_m and nd < local_dist[ny, nx]:
                        local_dist[ny, nx] = nd
                        q.append((nx, ny))

        out[~memory_mask] = 0.0
        return np.clip(out, 0.0, 1.0)

    def build_risk_source_map(self):
        if self.risk_source_mode in ('evidence', 'evidence_points', 'markers', 'marker_distribution'):
            evidence_source = self.build_evidence_source_map()
            if float(np.max(evidence_source)) > 1e-6:
                return evidence_source
            return self.positive_memory_map
        return self.positive_memory_map

    # ---------------- Main update ----------------

    def on_timer(self):
        if self.occ_grid is None or self.latest_map_msg is None:
            self.get_logger().warn('waiting for /map...', throttle_duration_sec=3.0)
            return

        robot_pose = self.get_robot_pose()
        if robot_pose is None:
            return

        now_ros_sec = self.get_clock().now().nanoseconds * 1e-9
        if self.pose_source not in ('topic', 'pose_topic', 'pose'):
            self.record_pose_sample(now_ros_sec, robot_pose)

        detections = []
        detections.extend(self.maybe_make_fake_detection())
        with self.detection_lock:
            last_yolo_ros_sec = self.last_yolo_ros_sec
            latest_detection_pose = self.latest_detection_pose
            latest_detections = list(self.latest_detections)
        can_reuse_detection = (
            last_yolo_ros_sec is not None
            and now_ros_sec - last_yolo_ros_sec <= self.detection_timeout_sec
        )
        if can_reuse_detection and latest_detection_pose is not None:
            dx = float(robot_pose[0]) - float(latest_detection_pose[0])
            dy = float(robot_pose[1]) - float(latest_detection_pose[1])
            if math.hypot(dx, dy) > max(0.0, self.detection_reuse_max_distance_m):
                can_reuse_detection = False
        if can_reuse_detection:
            detections.extend(latest_detections)

        projection_pose = (
            latest_detection_pose
            if detections and latest_detection_pose is not None
            else robot_pose
        )
        self.detection_candidate_map = self.build_detection_candidate_map(projection_pose, detections)
        had_detection = float(np.max(self.detection_candidate_map)) > 1e-6

        if had_detection:
            self.update_positive_memory(self.detection_candidate_map)

        if self.enable_visibility_tracking:
            self.visibility_map = self.compute_visibility_map(robot_pose)
            self.update_visual_seen_map(self.visibility_map)
            self.update_observed_empty(self.visibility_map, had_detection)

        if self.enable_region_segmentation:
            self.update_region_segmentation()
        if self.enable_room_probability:
            self.room_probability_map = self.build_room_probability_map()

        # Risk is positive-only bounded halo. No negative observation subtraction.
        if self.risk_dirty:
            self.risk_map = self.build_bounded_geodesic_halo(self.build_risk_source_map())
            self.risk_dirty = False

        self.publish_all_maps()
        self.publish_markers(robot_pose)

    # ---------------- Clear / publish ----------------

    def on_clear_all(self, msg):
        if not msg.data:
            return
        for arr in (
            self.detection_candidate_map,
            self.positive_memory_map,
            self.risk_map,
            self.observed_empty_map,
            self.visibility_map,
            self.room_probability_map,
            self.visual_seen_map,
            self.region_priority_map,
            self.region_checked_map,
        ):
            if arr is not None:
                arr.fill(0.0)
        if self.region_id_map is not None:
            self.region_id_map.fill(0)
        self.region_states.clear()
        self.evidence_points.clear()
        self.risk_dirty = True
        self.get_logger().warn('cleared all room-aware risk/region maps')

    def on_clear_point(self, msg):
        if self.occ_grid is None:
            return
        x = float(msg.point.x)
        y = float(msg.point.y)
        g = self.world_to_grid(x, y)
        if g is None:
            return

        gx0, gy0 = g
        r_cells = max(1, int(math.ceil(self.clear_radius_m / self.map_resolution)))
        r_cells_sq = float(r_cells * r_cells)
        h, w = self.occ_grid.shape

        for gy in range(max(0, gy0 - r_cells), min(h - 1, gy0 + r_cells) + 1):
            for gx in range(max(0, gx0 - r_cells), min(w - 1, gx0 + r_cells) + 1):
                d2 = float((gx - gx0) * (gx - gx0) + (gy - gy0) * (gy - gy0))
                if d2 <= r_cells_sq:
                    self.detection_candidate_map[gy, gx] = 0.0
                    self.positive_memory_map[gy, gx] = 0.0
                    self.risk_map[gy, gx] = 0.0
                    self.room_probability_map[gy, gx] = 0.0

        self.evidence_points = [
            ev for ev in self.evidence_points
            if math.hypot(ev.x - x, ev.y - y) > self.clear_radius_m
        ]
        self.risk_dirty = True
        self.get_logger().warn(f'clear positive risk around ({x:.2f},{y:.2f}) r={self.clear_radius_m:.2f}m')

    def array_to_occgrid(self, arr, stamp):
        msg = OccupancyGrid()
        msg.header.frame_id = self.map_frame
        msg.header.stamp = stamp
        msg.info = self.latest_map_msg.info
        if arr is None:
            arr = np.zeros_like(self.occ_grid, dtype=np.float32)
        msg.data = np.rint(np.clip(arr, 0.0, 1.0) * 100.0).astype(np.int8).flatten().tolist()
        return msg

    def region_id_to_occgrid(self, stamp):
        msg = OccupancyGrid()
        msg.header.frame_id = self.map_frame
        msg.header.stamp = stamp
        msg.info = self.latest_map_msg.info
        if self.region_id_map is None:
            data = np.zeros_like(self.occ_grid, dtype=np.int8)
        else:
            rid = self.region_id_map
            data = np.zeros(rid.shape, dtype=np.int16)
            data[rid == -1] = -1
            data[rid == -2] = 0
            pos = rid > 0
            data[pos] = ((rid[pos] * 17) % 97) + 3
            data = np.clip(data, -1, 100).astype(np.int8)
        msg.data = data.flatten().tolist()
        return msg

    def combined_priority_map(self):
        base = self.risk_map if self.risk_map is not None else np.zeros_like(self.occ_grid, dtype=np.float32)
        pri = self.region_priority_map if self.region_priority_map is not None else np.zeros_like(base, dtype=np.float32)
        return np.maximum(base, pri)

    def publish_all_maps(self):
        now_ros = self.get_clock().now()
        stamp = now_ros.to_msg()
        now_ros_ns = int(now_ros.nanoseconds)
        # The risk layer is latency-sensitive, but it still benefits from a small
        # publish throttle while teleop driving causes the map to resize frequently.
        risk_period_ns = (
            int(1e9 / self.risk_publish_rate_hz)
            if self.risk_publish_rate_hz > 0.0 else 0
        )
        if risk_period_ns <= 0 or now_ros_ns - self.last_risk_publish_ros_ns >= risk_period_ns:
            self.pub_risk.publish(self.array_to_occgrid(self.risk_map, stamp))
            self.last_risk_publish_ros_ns = now_ros_ns
        diagnostic_period_ns = (
            int(1e9 / self.diagnostic_publish_rate_hz)
            if self.diagnostic_publish_rate_hz > 0.0 else 0
        )
        if (
            diagnostic_period_ns > 0
            and now_ros_ns - self.last_diagnostic_publish_ros_ns < diagnostic_period_ns
        ):
            return
        self.last_diagnostic_publish_ros_ns = now_ros_ns

        # Heavy full-map diagnostic layers are deliberately rate-limited.
        if not self.publish_diagnostic_maps:
            return

        self.pub_detection_candidate.publish(self.array_to_occgrid(self.detection_candidate_map, stamp))
        self.pub_positive_memory.publish(self.array_to_occgrid(self.positive_memory_map, stamp))
        self.pub_visibility.publish(self.array_to_occgrid(self.visibility_map, stamp))
        self.pub_observed_empty.publish(self.array_to_occgrid(self.observed_empty_map, stamp))
        self.pub_room_probability.publish(self.array_to_occgrid(self.room_probability_map, stamp))
        self.pub_visual_seen.publish(self.array_to_occgrid(self.visual_seen_map, stamp))
        self.pub_region_id.publish(self.region_id_to_occgrid(stamp))
        self.pub_region_priority.publish(self.array_to_occgrid(self.region_priority_map, stamp))
        self.pub_region_checked.publish(self.array_to_occgrid(self.region_checked_map, stamp))
        self.pub_combined_priority.publish(self.array_to_occgrid(self.combined_priority_map(), stamp))

    def publish_markers(self, robot_pose):
        stamp = self.get_clock().now().to_msg()
        ma = MarkerArray()

        delete_all = Marker()
        delete_all.action = Marker.DELETEALL
        ma.markers.append(delete_all)

        rx, ry, ryaw = robot_pose
        fov = Marker()
        fov.header.frame_id = self.map_frame
        fov.header.stamp = stamp
        fov.ns = 'risk_fov'
        fov.id = 1
        fov.type = Marker.LINE_LIST
        fov.action = Marker.ADD
        fov.scale.x = 0.03
        fov.color.r = 0.2
        fov.color.g = 0.8
        fov.color.b = 1.0
        fov.color.a = 0.85

        for b in (-0.5 * math.radians(self.camera_hfov_deg), 0.5 * math.radians(self.camera_hfov_deg)):
            fov.points.append(Point(x=float(rx), y=float(ry), z=0.05))
            fov.points.append(Point(
                x=float(rx + self.max_range_m * math.cos(ryaw + b)),
                y=float(ry + self.max_range_m * math.sin(ryaw + b)),
                z=0.05,
            ))
        ma.markers.append(fov)

        for i, ev in enumerate(self.evidence_points[-80:]):
            m = Marker()
            m.header.frame_id = self.map_frame
            m.header.stamp = stamp
            m.ns = 'positive_evidence'
            m.id = 1000 + i
            m.type = Marker.SPHERE
            m.action = Marker.ADD
            m.pose.position.x = ev.x
            m.pose.position.y = ev.y
            m.pose.position.z = 0.08
            m.pose.orientation.w = 1.0
            s = 0.14 + 0.22 * clamp(ev.confidence, 0.0, 1.0)
            m.scale.x = s
            m.scale.y = s
            m.scale.z = 0.08
            m.color.r = 1.0
            m.color.g = 0.15
            m.color.b = 0.0
            m.color.a = 0.85
            ma.markers.append(m)

        self.pub_markers.publish(ma)

    def destroy_node(self):
        self.stop_yolo_worker()
        self.stop_opencv_capture_worker()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = RoomAwareRiskMapNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        cap = getattr(node, 'opencv_cap', None)
        if cap is not None:
            try:
                cap.release()
            except Exception:
                pass
        if getattr(node, 'debug_show_opencv', False):
            try:
                import cv2
                cv2.destroyAllWindows()
            except Exception:
                pass
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
