#!/usr/bin/env python3

import math
from typing import Dict, Optional, Tuple

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from nav_msgs.msg import OccupancyGrid, MapMetaData
from geometry_msgs.msg import PoseStamped


class RobotFootprintMapFreeSpaceFilter(Node):
    """Post-process an occupancy grid to erase known robot footprints.

    This is intentionally a *map-level* cleaner, not a laser filter.
    Cartographer receives the normal scan and builds the map normally. If the peer
    robot is inserted as an occupied blob, this node clears only the small circular
    footprint around the current known robot poses in the published map.
    """

    def __init__(self) -> None:
        super().__init__('robot_footprint_map_free_space_filter')

        self.declare_parameter('map_in_topic', '/map_raw')
        self.declare_parameter('map_out_topic', '/map')
        self.declare_parameter('map_metadata_out_topic', '/map_metadata')
        self.declare_parameter('pose_topics', '/leader_pose,/burger_pose')
        self.declare_parameter('fallback_pose_topics', '')
        self.declare_parameter('clear_radius_m', 0.26)
        self.declare_parameter('occupied_threshold', 35)
        self.declare_parameter('clear_unknown', False)
        self.declare_parameter('stale_pose_sec', 2.0)
        self.declare_parameter('publish_metadata', True)
        self.declare_parameter('log_every_n', 50)

        self.map_in_topic = str(self.get_parameter('map_in_topic').value)
        self.map_out_topic = str(self.get_parameter('map_out_topic').value)
        self.meta_out_topic = str(self.get_parameter('map_metadata_out_topic').value)
        pose_topics = self._parse_topics(str(self.get_parameter('pose_topics').value))
        fallback_topics = self._parse_topics(str(self.get_parameter('fallback_pose_topics').value))
        self.clear_radius_m = float(self.get_parameter('clear_radius_m').value)
        self.occupied_threshold = int(self.get_parameter('occupied_threshold').value)
        self.clear_unknown = bool(self.get_parameter('clear_unknown').value)
        self.stale_pose_sec = float(self.get_parameter('stale_pose_sec').value)
        self.publish_metadata = bool(self.get_parameter('publish_metadata').value)
        self.log_every_n = max(1, int(self.get_parameter('log_every_n').value))

        self.poses: Dict[str, Tuple[float, float, rclpy.time.Time]] = {}
        self.pose_priority: Dict[str, int] = {}
        self.publish_count = 0

        sub_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )
        pub_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )

        self.map_pub = self.create_publisher(OccupancyGrid, self.map_out_topic, pub_qos)
        self.meta_pub = self.create_publisher(MapMetaData, self.meta_out_topic, pub_qos)
        self.map_sub = self.create_subscription(OccupancyGrid, self.map_in_topic, self._on_map, sub_qos)

        for i, topic in enumerate(pose_topics):
            self.pose_priority[topic] = 0
            self.create_subscription(PoseStamped, topic, lambda msg, t=topic: self._on_pose(t, msg), 10)
        for i, topic in enumerate(fallback_topics):
            self.pose_priority[topic] = 1
            self.create_subscription(PoseStamped, topic, lambda msg, t=topic: self._on_pose(t, msg), 10)

        self.get_logger().info(
            f'MAP_CLEANER_READY | '
            f'map={self.map_in_topic}->{self.map_out_topic} | poses={pose_topics} fallback={fallback_topics} | '
            f'clear_radius={self.clear_radius_m:.2f}m clear_unknown={self.clear_unknown} | '
            f'NOTE: laser scan is not filtered; Cartographer sees normal scan; only final /map is cleaned.'
        )

    def _parse_topics(self, value: str):
        return [s.strip() for s in value.split(',') if s.strip()]

    def _on_pose(self, topic: str, msg: PoseStamped) -> None:
        # Keep poses only if they are in map-ish frame. We still accept empty frame
        # because some helper publishers may leave it empty when using map by convention.
        frame = (msg.header.frame_id or '').strip()
        if frame and frame != 'map':
            # Do not transform here; raw TF is intentionally not bridged across domains.
            return
        self.poses[topic] = (float(msg.pose.position.x), float(msg.pose.position.y), self.get_clock().now())

    def _fresh_pose_points(self):
        now = self.get_clock().now()
        pts = []
        # Deduplicate near-identical fallback/main poses.
        for topic, (x, y, stamp) in list(self.poses.items()):
            age = (now - stamp).nanoseconds * 1e-9
            if age > self.stale_pose_sec:
                continue
            duplicate = False
            for px, py, _ in pts:
                if math.hypot(x - px, y - py) < 0.10:
                    duplicate = True
                    break
            if not duplicate:
                pts.append((x, y, topic))
        return pts

    def _world_to_map(self, msg: OccupancyGrid, x: float, y: float) -> Optional[Tuple[int, int]]:
        info = msg.info
        res = float(info.resolution)
        if res <= 0:
            return None
        ox = float(info.origin.position.x)
        oy = float(info.origin.position.y)
        # Most occupancy grids are axis-aligned. Handle yaw approximately for safety.
        q = info.origin.orientation
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        yaw = math.atan2(siny_cosp, cosy_cosp)
        dx = x - ox
        dy = y - oy
        if abs(yaw) > 1e-6:
            c = math.cos(-yaw)
            s = math.sin(-yaw)
            mx_f = (c * dx - s * dy) / res
            my_f = (s * dx + c * dy) / res
        else:
            mx_f = dx / res
            my_f = dy / res
        mx = int(math.floor(mx_f))
        my = int(math.floor(my_f))
        if mx < 0 or my < 0 or mx >= info.width or my >= info.height:
            return None
        return mx, my

    def _on_map(self, msg: OccupancyGrid) -> None:
        out = OccupancyGrid()
        out.header = msg.header
        out.header.stamp = self.get_clock().now().to_msg()
        out.info = msg.info
        data = list(msg.data)

        res = float(msg.info.resolution)
        if res <= 0.0:
            self.map_pub.publish(out)
            return

        radius_cells = max(1, int(math.ceil(self.clear_radius_m / res)))
        cleared = 0
        used = 0

        for x, y, topic in self._fresh_pose_points():
            center = self._world_to_map(msg, x, y)
            if center is None:
                continue
            used += 1
            cx, cy = center
            for dy in range(-radius_cells, radius_cells + 1):
                yy = cy + dy
                if yy < 0 or yy >= msg.info.height:
                    continue
                for dx in range(-radius_cells, radius_cells + 1):
                    if dx * dx + dy * dy > radius_cells * radius_cells:
                        continue
                    xx = cx + dx
                    if xx < 0 or xx >= msg.info.width:
                        continue
                    idx = yy * msg.info.width + xx
                    v = data[idx]
                    if v >= self.occupied_threshold or (self.clear_unknown and v < 0):
                        data[idx] = 0
                        cleared += 1

        out.data = data
        self.map_pub.publish(out)
        if self.publish_metadata:
            meta = MapMetaData()
            meta.map_load_time = out.info.map_load_time
            meta.resolution = out.info.resolution
            meta.width = out.info.width
            meta.height = out.info.height
            meta.origin = out.info.origin
            self.meta_pub.publish(meta)

        self.publish_count += 1
        if self.publish_count == 1:
            self.get_logger().info(
                f'MAP_CLEANER_FIRST_MAP | passthrough_ok=1 poses_used={used} cleared_cells={cleared} '
                f'size={msg.info.width}x{msg.info.height} res={res:.3f} out={self.map_out_topic}'
            )
        if self.publish_count % self.log_every_n == 0:
            self.get_logger().info(
                f'MAP_CLEANER_PUBLISH | poses_used={used} cleared_cells={cleared} '
                f'radius={self.clear_radius_m:.2f}m size={msg.info.width}x{msg.info.height} res={res:.3f}'
            )


def main(args=None):
    rclpy.init(args=args)
    node = RobotFootprintMapFreeSpaceFilter()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
