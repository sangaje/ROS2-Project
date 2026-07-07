"""Detect free-space regions from /map and publish a Nav2 goal pose."""

from __future__ import annotations

from collections import deque
import math
from typing import Any, Dict, List, Optional, Tuple

from geometry_msgs.msg import Point, PointStamped, PoseArray, PoseStamped
from nav_msgs.msg import OccupancyGrid
import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from visualization_msgs.msg import Marker, MarkerArray


class Region:
    """Simple connected free-space region in an OccupancyGrid."""

    def __init__(self, index: int, cells: List[int], width: int) -> None:
        self.index = index
        self.cells = cells
        self.cell_count = len(cells)
        xs = [cell % width for cell in cells]
        ys = [cell // width for cell in cells]
        self.min_x = min(xs)
        self.max_x = max(xs)
        self.min_y = min(ys)
        self.max_y = max(ys)
        self.center_x = sum(xs) / self.cell_count
        self.center_y = sum(ys) / self.cell_count


class RegionNav2Goal(Node):
    """Turn map regions into RViz markers and optional Nav2 goal poses."""

    def __init__(self) -> None:
        super().__init__('region_nav2_goal')

        self.declare_parameter('map_topic', '/map')
        self.declare_parameter('marker_topic', '/tb3_multi/region_markers')
        self.declare_parameter('region_centers_topic', '/tb3_multi/region_centers')
        self.declare_parameter('selected_region_topic', '/tb3_multi/selected_region')
        self.declare_parameter('nav2_goal_topic', '/goal_pose')
        self.declare_parameter('free_threshold', 25)
        self.declare_parameter('min_region_cells', 60)
        self.declare_parameter('max_regions', 12)
        self.declare_parameter('publish_nav2_goal', False)
        self.declare_parameter('goal_region_index', 1)
        self.declare_parameter('analysis_period_sec', 1.0)
        self.declare_parameter('republish_goal_distance_m', 0.35)

        map_topic = str(self.get_parameter('map_topic').value)
        marker_topic = str(self.get_parameter('marker_topic').value)
        centers_topic = str(self.get_parameter('region_centers_topic').value)
        selected_topic = str(self.get_parameter('selected_region_topic').value)
        self.nav2_goal_topic = str(self.get_parameter('nav2_goal_topic').value)
        self.free_threshold = self.as_int(self.get_parameter('free_threshold').value)
        self.min_region_cells = self.as_int(
            self.get_parameter('min_region_cells').value
        )
        self.max_regions = self.as_int(self.get_parameter('max_regions').value)
        self.publish_nav2_goal = self.as_bool(
            self.get_parameter('publish_nav2_goal').value
        )
        self.goal_region_index = self.as_int(
            self.get_parameter('goal_region_index').value
        )
        analysis_period = self.as_float(
            self.get_parameter('analysis_period_sec').value
        )
        self.republish_goal_distance = self.as_float(
            self.get_parameter('republish_goal_distance_m').value
        )

        map_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.create_subscription(OccupancyGrid, map_topic, self.on_map, map_qos)
        self.marker_pub = self.create_publisher(MarkerArray, marker_topic, 10)
        self.centers_pub = self.create_publisher(PoseArray, centers_topic, 10)
        self.selected_pub = self.create_publisher(PointStamped, selected_topic, 10)
        self.nav2_goal_pub = self.create_publisher(PoseStamped, self.nav2_goal_topic, 10)

        self.latest_map: Optional[OccupancyGrid] = None
        self.last_goal: Optional[Tuple[float, float]] = None
        self.create_timer(max(0.2, analysis_period), self.process_latest_map)

        self.get_logger().info(
            'Region mapper ready: map=%s, markers=%s, nav2_goal=%s, '
            'publish_nav2_goal=%s'
            % (map_topic, marker_topic, self.nav2_goal_topic, self.publish_nav2_goal)
        )

    def as_bool(self, value: Any) -> bool:
        if isinstance(value, str):
            return value.strip().lower() in ('1', 'true', 'yes', 'on')
        return bool(value)

    def as_int(self, value: Any) -> int:
        return int(str(value).strip())

    def as_float(self, value: Any) -> float:
        return float(str(value).strip())

    def on_map(self, msg: OccupancyGrid) -> None:
        self.latest_map = msg

    def process_latest_map(self) -> None:
        msg = self.latest_map
        if msg is None:
            return

        now = self.get_clock().now()

        regions = self.detect_regions(msg)
        self.publish_region_outputs(msg, regions)

        selected = self.select_region(regions)
        if selected is None:
            return

        x, y = self.region_world_center(msg, selected)
        selected_msg = PointStamped()
        selected_msg.header.stamp = now.to_msg()
        selected_msg.header.frame_id = msg.header.frame_id or 'map'
        selected_msg.point.x = x
        selected_msg.point.y = y
        selected_msg.point.z = 0.0
        self.selected_pub.publish(selected_msg)

        if self.publish_nav2_goal and self.should_publish_goal(x, y):
            self.publish_goal_pose(selected_msg)
            self.last_goal = (x, y)
            self.get_logger().info(
                f'Nav2 region goal published to {self.nav2_goal_topic}: '
                f'region=room_{selected.index}, x={x:.2f}, y={y:.2f}'
            )

    def detect_regions(self, msg: OccupancyGrid) -> List[Region]:
        width = msg.info.width
        height = msg.info.height
        data = msg.data
        if width == 0 or height == 0 or len(data) != width * height:
            return []

        visited = bytearray(width * height)
        regions: List[Region] = []

        for index, value in enumerate(data):
            if visited[index] or not self.is_free(value):
                continue
            cells = self.flood_fill(index, data, visited, width, height)
            if len(cells) >= self.min_region_cells:
                regions.append(Region(len(regions) + 1, cells, width))

        regions.sort(key=lambda region: region.cell_count, reverse=True)
        for index, region in enumerate(regions[: self.max_regions], start=1):
            region.index = index
        return regions[: self.max_regions]

    def flood_fill(
        self,
        start: int,
        data: Tuple[int, ...],
        visited: bytearray,
        width: int,
        height: int,
    ) -> List[int]:
        queue = deque([start])
        visited[start] = 1
        cells: List[int] = []

        while queue:
            index = queue.popleft()
            cells.append(index)
            x = index % width
            y = index // width
            for nx, ny in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
                if nx < 0 or ny < 0 or nx >= width or ny >= height:
                    continue
                nindex = ny * width + nx
                if visited[nindex] or not self.is_free(data[nindex]):
                    continue
                visited[nindex] = 1
                queue.append(nindex)

        return cells

    def is_free(self, value: int) -> bool:
        return 0 <= value <= self.free_threshold

    def select_region(self, regions: List[Region]) -> Optional[Region]:
        if not regions:
            return None
        index = max(1, self.goal_region_index) - 1
        if index >= len(regions):
            index = 0
        return regions[index]

    def region_world_center(
        self,
        msg: OccupancyGrid,
        region: Region,
    ) -> Tuple[float, float]:
        return self.grid_to_world(msg, region.center_x, region.center_y)

    def grid_to_world(
        self,
        msg: OccupancyGrid,
        grid_x: float,
        grid_y: float,
    ) -> Tuple[float, float]:
        resolution = msg.info.resolution
        origin = msg.info.origin.position
        yaw = self.origin_yaw(msg.info.origin.orientation)
        local_x = (grid_x + 0.5) * resolution
        local_y = (grid_y + 0.5) * resolution
        cos_yaw = math.cos(yaw)
        sin_yaw = math.sin(yaw)
        world_x = origin.x + local_x * cos_yaw - local_y * sin_yaw
        world_y = origin.y + local_x * sin_yaw + local_y * cos_yaw
        return world_x, world_y

    def grid_corner_to_world(
        self,
        msg: OccupancyGrid,
        grid_x: float,
        grid_y: float,
    ) -> Tuple[float, float]:
        resolution = msg.info.resolution
        origin = msg.info.origin.position
        yaw = self.origin_yaw(msg.info.origin.orientation)
        local_x = grid_x * resolution
        local_y = grid_y * resolution
        cos_yaw = math.cos(yaw)
        sin_yaw = math.sin(yaw)
        world_x = origin.x + local_x * cos_yaw - local_y * sin_yaw
        world_y = origin.y + local_x * sin_yaw + local_y * cos_yaw
        return world_x, world_y

    def make_point(self, x: float, y: float, z: float = 0.0) -> Point:
        point = Point()
        point.x = x
        point.y = y
        point.z = z
        return point

    def origin_yaw(self, orientation) -> float:
        siny_cosp = 2.0 * (
            orientation.w * orientation.z + orientation.x * orientation.y
        )
        cosy_cosp = 1.0 - 2.0 * (
            orientation.y * orientation.y + orientation.z * orientation.z
        )
        return math.atan2(siny_cosp, cosy_cosp)

    def should_publish_goal(self, x: float, y: float) -> bool:
        if self.last_goal is None:
            return True
        return math.hypot(x - self.last_goal[0], y - self.last_goal[1]) >= (
            self.republish_goal_distance
        )

    def publish_goal_pose(self, selected_msg: PointStamped) -> None:
        goal = PoseStamped()
        goal.header = selected_msg.header
        goal.pose.position.x = selected_msg.point.x
        goal.pose.position.y = selected_msg.point.y
        goal.pose.position.z = 0.0
        goal.pose.orientation.w = 1.0
        self.nav2_goal_pub.publish(goal)

    def publish_region_outputs(
        self,
        msg: OccupancyGrid,
        regions: List[Region],
    ) -> None:
        now_msg = self.get_clock().now().to_msg()
        frame_id = msg.header.frame_id or 'map'
        marker_array = MarkerArray()
        centers = PoseArray()
        centers.header.stamp = now_msg
        centers.header.frame_id = frame_id

        delete_all = Marker()
        delete_all.header.stamp = now_msg
        delete_all.header.frame_id = frame_id
        delete_all.ns = 'tb3_multi_regions'
        delete_all.action = Marker.DELETEALL
        marker_array.markers.append(delete_all)

        for region in regions:
            x, y = self.region_world_center(msg, region)
            color = self.color_for_region(region.index)

            fill = Marker()
            fill.header.stamp = now_msg
            fill.header.frame_id = frame_id
            fill.ns = 'tb3_multi_region_cells'
            fill.id = region.index
            fill.type = Marker.CUBE_LIST
            fill.action = Marker.ADD
            fill.pose.orientation.w = 1.0
            fill.scale.x = msg.info.resolution
            fill.scale.y = msg.info.resolution
            fill.scale.z = 0.02
            fill.color.a = 0.16
            fill.color.r, fill.color.g, fill.color.b = color

            for cell in region.cells:
                cell_x = cell % msg.info.width
                cell_y = cell // msg.info.width
                world_x, world_y = self.grid_to_world(msg, cell_x, cell_y)
                fill.points.append(self.make_point(world_x, world_y, 0.02))
            marker_array.markers.append(fill)

            outline = Marker()
            outline.header.stamp = now_msg
            outline.header.frame_id = frame_id
            outline.ns = 'tb3_multi_region_outlines'
            outline.id = region.index
            outline.type = Marker.LINE_LIST
            outline.action = Marker.ADD
            outline.pose.orientation.w = 1.0
            outline.scale.x = max(0.02, msg.info.resolution * 0.45)
            outline.color.a = 0.95
            outline.color.r, outline.color.g, outline.color.b = color
            outline.points = self.region_outline_points(msg, region)
            marker_array.markers.append(outline)

            center = Marker()
            center.header.stamp = now_msg
            center.header.frame_id = frame_id
            center.ns = 'tb3_multi_region_centers'
            center.id = region.index
            center.type = Marker.SPHERE
            center.action = Marker.ADD
            center.pose.position.x = x
            center.pose.position.y = y
            center.pose.position.z = 0.08
            center.pose.orientation.w = 1.0
            center.scale.x = 0.18
            center.scale.y = 0.18
            center.scale.z = 0.18
            center.color.a = 0.95
            center.color.r, center.color.g, center.color.b = color
            marker_array.markers.append(center)

            text = Marker()
            text.header.stamp = now_msg
            text.header.frame_id = frame_id
            text.ns = 'tb3_multi_region_labels'
            text.id = region.index
            text.type = Marker.TEXT_VIEW_FACING
            text.action = Marker.ADD
            text.pose.position.x = x
            text.pose.position.y = y
            text.pose.position.z = 0.35
            text.pose.orientation.w = 1.0
            text.scale.z = 0.22
            text.color.a = 1.0
            text.color.r = 1.0
            text.color.g = 1.0
            text.color.b = 1.0
            text.text = f'room_{region.index}'
            marker_array.markers.append(text)

            pose = PoseStamped().pose
            pose.position.x = x
            pose.position.y = y
            pose.orientation.w = 1.0
            centers.poses.append(pose)

        self.marker_pub.publish(marker_array)
        self.centers_pub.publish(centers)

    def region_outline_points(
        self,
        msg: OccupancyGrid,
        region: Region,
    ) -> List[Point]:
        width = msg.info.width
        height = msg.info.height
        region_cells = set(region.cells)
        points: List[Point] = []

        def add_edge(start_x: int, start_y: int, end_x: int, end_y: int) -> None:
            x1, y1 = self.grid_corner_to_world(msg, start_x, start_y)
            x2, y2 = self.grid_corner_to_world(msg, end_x, end_y)
            points.append(self.make_point(x1, y1, 0.05))
            points.append(self.make_point(x2, y2, 0.05))

        for cell in region.cells:
            cell_x = cell % width
            cell_y = cell // width
            neighbors = (
                (cell_x, cell_y - 1, cell_x, cell_y, cell_x + 1, cell_y),
                (
                    cell_x + 1,
                    cell_y,
                    cell_x + 1,
                    cell_y,
                    cell_x + 1,
                    cell_y + 1,
                ),
                (
                    cell_x,
                    cell_y + 1,
                    cell_x,
                    cell_y + 1,
                    cell_x + 1,
                    cell_y + 1,
                ),
                (cell_x - 1, cell_y, cell_x, cell_y, cell_x, cell_y + 1),
            )
            for neighbor_x, neighbor_y, sx, sy, ex, ey in neighbors:
                if neighbor_x < 0 or neighbor_y < 0:
                    add_edge(sx, sy, ex, ey)
                    continue
                if neighbor_x >= width or neighbor_y >= height:
                    add_edge(sx, sy, ex, ey)
                    continue
                neighbor_index = neighbor_y * width + neighbor_x
                if neighbor_index not in region_cells:
                    add_edge(sx, sy, ex, ey)

        return points

    def color_for_region(self, index: int) -> Tuple[float, float, float]:
        palette: Dict[int, Tuple[float, float, float]] = {
            1: (0.10, 0.58, 0.95),
            2: (0.10, 0.78, 0.36),
            3: (0.95, 0.66, 0.12),
            4: (0.92, 0.23, 0.32),
            5: (0.70, 0.37, 0.92),
            6: (0.00, 0.75, 0.75),
        }
        return palette.get(((index - 1) % len(palette)) + 1, (0.8, 0.8, 0.8))


def main(args=None) -> None:
    rclpy.init(args=args)
    node = RegionNav2Goal()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
