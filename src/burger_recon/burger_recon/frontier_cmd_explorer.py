#!/usr/bin/env python3

import math
from collections import deque
from typing import List, Optional, Tuple

import numpy as np
import rclpy
from geometry_msgs.msg import TwistStamped
from nav_msgs.msg import OccupancyGrid
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from rclpy.time import Time
from sensor_msgs.msg import LaserScan
from tf2_ros import Buffer, TransformException, TransformListener

GridCell = Tuple[int, int]  # (y, x)
WorldPoint = Tuple[float, float]


def normalize_angle(angle: float) -> float:
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle


def quaternion_to_yaw(q) -> float:
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


def as_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() in ("true", "1", "yes", "y", "on")
    return bool(value)


class FrontierCmdExplorer(Node):
    """
    Direct frontier exploration node.

    This node bypasses Nav2 and directly publishes /cmd_vel as TwistStamped.

    Inputs:
      - /map
      - /scan
      - TF: map -> base_footprint

    Output:
      - /cmd_vel, geometry_msgs/msg/TwistStamped

    Main behavior:
      - Detect frontier cells from OccupancyGrid.
      - Select an exploration target.
      - Optionally project target into unknown space.
      - Continuously publish cmd_vel.
      - Do not stop discretely at every frontier goal.
    """

    def __init__(self):
        super().__init__("frontier_cmd_explorer")

        # ------------------------------------------------------------------
        # Frames / topics
        # ------------------------------------------------------------------
        self.declare_parameter("map_topic", "/map")
        self.declare_parameter("scan_topic", "/scan")
        self.declare_parameter("cmd_vel_topic", "/cmd_vel")
        self.declare_parameter("map_frame", "map")
        self.declare_parameter("base_frame", "base_footprint")

        # ------------------------------------------------------------------
        # Map / frontier parameters
        # ------------------------------------------------------------------
        self.declare_parameter("free_threshold", 20)
        self.declare_parameter("min_frontier_size", 1)

        # ------------------------------------------------------------------
        # Control parameters
        # ------------------------------------------------------------------
        self.declare_parameter("control_period", 0.1)
        self.declare_parameter("linear_speed", 0.12)
        self.declare_parameter("angular_kp", 0.30)
        self.declare_parameter("max_angular_speed", 0.08)
        self.declare_parameter("angle_tolerance", 0.95)

        # This is kept for compatibility, but continuous mode mainly uses
        # goal_switch_distance.
        self.declare_parameter("goal_tolerance", 0.05)

        # ------------------------------------------------------------------
        # Continuous motion parameters
        # ------------------------------------------------------------------
        self.declare_parameter("continuous_motion", True)
        self.declare_parameter("goal_switch_distance", 0.25)
        self.declare_parameter("slowdown_distance", 0.8)
        self.declare_parameter("min_linear_speed", 0.025)

        self.declare_parameter("wander_when_no_goal", True)
        self.declare_parameter("wander_linear_speed", 0.04)
        self.declare_parameter("wander_turn_speed", 0.10)

        # ------------------------------------------------------------------
        # Obstacle avoidance
        # ------------------------------------------------------------------
        self.declare_parameter("safe_front_distance", 0.18)
        self.declare_parameter("front_angle_deg", 20.0)
        self.declare_parameter("avoid_turn_speed", 0.08)

        # ------------------------------------------------------------------
        # Goal selection
        # ------------------------------------------------------------------
        self.declare_parameter("min_goal_distance", 0.10)
        self.declare_parameter("max_goal_distance", 30.0)
        self.declare_parameter("information_gain_weight", 0.20)
        self.declare_parameter("goal_recompute_period", 5.0)

        # ------------------------------------------------------------------
        # Unknown-space target projection
        # ------------------------------------------------------------------
        self.declare_parameter("target_unknown_space", True)
        self.declare_parameter("unknown_target_offset", 3.0)
        self.declare_parameter("fallback_to_frontier_goal", True)

        # ------------------------------------------------------------------
        # Debug
        # ------------------------------------------------------------------
        self.declare_parameter("log_frontier_stats", True)
        self.declare_parameter("log_goal_details", True)
        self.declare_parameter("log_tracking", False)

        # ------------------------------------------------------------------
        # Read parameters
        # ------------------------------------------------------------------
        self.map_topic = self.get_parameter("map_topic").value
        self.scan_topic = self.get_parameter("scan_topic").value
        self.cmd_vel_topic = self.get_parameter("cmd_vel_topic").value
        self.map_frame = self.get_parameter("map_frame").value
        self.base_frame = self.get_parameter("base_frame").value

        self.free_threshold = int(self.get_parameter("free_threshold").value)
        self.min_frontier_size = int(self.get_parameter("min_frontier_size").value)

        self.control_period = float(self.get_parameter("control_period").value)
        self.linear_speed = float(self.get_parameter("linear_speed").value)
        self.angular_kp = float(self.get_parameter("angular_kp").value)
        self.max_angular_speed = float(self.get_parameter("max_angular_speed").value)
        self.angle_tolerance = float(self.get_parameter("angle_tolerance").value)
        self.goal_tolerance = float(self.get_parameter("goal_tolerance").value)

        self.continuous_motion = as_bool(self.get_parameter("continuous_motion").value)
        self.goal_switch_distance = float(
            self.get_parameter("goal_switch_distance").value
        )
        self.slowdown_distance = float(self.get_parameter("slowdown_distance").value)
        self.min_linear_speed = float(self.get_parameter("min_linear_speed").value)

        self.wander_when_no_goal = as_bool(
            self.get_parameter("wander_when_no_goal").value
        )
        self.wander_linear_speed = float(
            self.get_parameter("wander_linear_speed").value
        )
        self.wander_turn_speed = float(self.get_parameter("wander_turn_speed").value)

        self.safe_front_distance = float(
            self.get_parameter("safe_front_distance").value
        )
        self.front_angle_deg = float(self.get_parameter("front_angle_deg").value)
        self.avoid_turn_speed = float(self.get_parameter("avoid_turn_speed").value)

        self.min_goal_distance = float(self.get_parameter("min_goal_distance").value)
        self.max_goal_distance = float(self.get_parameter("max_goal_distance").value)
        self.information_gain_weight = float(
            self.get_parameter("information_gain_weight").value
        )
        self.goal_recompute_period = float(
            self.get_parameter("goal_recompute_period").value
        )

        self.target_unknown_space = as_bool(
            self.get_parameter("target_unknown_space").value
        )
        self.unknown_target_offset = float(
            self.get_parameter("unknown_target_offset").value
        )
        self.fallback_to_frontier_goal = as_bool(
            self.get_parameter("fallback_to_frontier_goal").value
        )

        self.log_frontier_stats = as_bool(
            self.get_parameter("log_frontier_stats").value
        )
        self.log_goal_details = as_bool(self.get_parameter("log_goal_details").value)
        self.log_tracking = as_bool(self.get_parameter("log_tracking").value)

        # ------------------------------------------------------------------
        # Runtime state
        # ------------------------------------------------------------------
        self.latest_map: Optional[OccupancyGrid] = None
        self.latest_scan: Optional[LaserScan] = None
        self.current_goal: Optional[WorldPoint] = None
        self.last_goal_time = self.get_clock().now()

        # ------------------------------------------------------------------
        # TF
        # ------------------------------------------------------------------
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # ------------------------------------------------------------------
        # QoS
        # ------------------------------------------------------------------
        map_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        # ------------------------------------------------------------------
        # ROS interfaces
        # ------------------------------------------------------------------
        self.map_sub = self.create_subscription(
            OccupancyGrid,
            self.map_topic,
            self.on_map,
            map_qos,
        )

        self.scan_sub = self.create_subscription(
            LaserScan,
            self.scan_topic,
            self.on_scan,
            10,
        )

        self.cmd_pub = self.create_publisher(
            TwistStamped,
            self.cmd_vel_topic,
            10,
        )

        self.timer = self.create_timer(self.control_period, self.control_loop)

        self.get_logger().info(
            f"frontier_cmd_explorer started. "
            f"map={self.map_topic}, scan={self.scan_topic}, "
            f"cmd_vel={self.cmd_vel_topic}, base_frame={self.base_frame}, "
            f"continuous_motion={self.continuous_motion}, "
            f"target_unknown_space={self.target_unknown_space}, "
            f"unknown_target_offset={self.unknown_target_offset:.2f}"
        )

    # ----------------------------------------------------------------------
    # ROS callbacks
    # ----------------------------------------------------------------------
    def on_map(self, msg: OccupancyGrid) -> None:
        self.latest_map = msg

    def on_scan(self, msg: LaserScan) -> None:
        self.latest_scan = msg

    # ----------------------------------------------------------------------
    # Main control loop
    # ----------------------------------------------------------------------
    def control_loop(self) -> None:
        if self.latest_map is None:
            self.get_logger().info("waiting for /map...")
            self.publish_stop()
            return

        if self.latest_scan is None:
            self.get_logger().info("waiting for /scan...")
            self.publish_stop()
            return

        robot_pose = self.lookup_robot_pose()
        if robot_pose is None:
            self.get_logger().warn(
                f"failed to lookup pose: {self.map_frame} -> {self.base_frame}"
            )
            self.publish_stop()
            return

        rx, ry, ryaw = robot_pose

        # Goal is updated periodically, but we do not stop between goals.
        if self.current_goal is None or self.should_recompute_goal():
            new_goal = self.select_frontier_goal((rx, ry), self.latest_map)
            self.last_goal_time = self.get_clock().now()

            if new_goal is not None:
                self.current_goal = new_goal
                gx, gy = self.current_goal
                self.get_logger().info(f"new frontier goal: x={gx:.2f}, y={gy:.2f}")

        if self.current_goal is None:
            self.publish_wander_or_stop()
            return

        gx, gy = self.current_goal
        dx = gx - rx
        dy = gy - ry
        distance = math.hypot(dx, dy)

        if self.log_tracking:
            self.get_logger().info(
                f"tracking: robot=({rx:.2f}, {ry:.2f}), "
                f"goal=({gx:.2f}, {gy:.2f}), distance={distance:.3f}"
            )

        # Continuous mode:
        # When we are near the current goal, switch to another goal or wander,
        # but do not publish a full stop.
        switch_distance = self.goal_switch_distance
        if not self.continuous_motion:
            switch_distance = self.goal_tolerance

        if distance < switch_distance:
            self.get_logger().info(
                f"switching goal without stop: "
                f"goal=({gx:.2f}, {gy:.2f}), distance={distance:.3f}"
            )
            self.current_goal = None
            self.publish_wander_or_stop()
            return

        self.drive_toward_goal(rx, ry, ryaw, gx, gy, distance)

    def drive_toward_goal(
        self,
        rx: float,
        ry: float,
        ryaw: float,
        gx: float,
        gy: float,
        distance: float,
    ) -> None:
        if self.front_obstacle_too_close():
            self.publish_cmd(0.0, self.avoid_turn_speed)
            return

        dx = gx - rx
        dy = gy - ry

        target_yaw = math.atan2(dy, dx)
        yaw_error = normalize_angle(target_yaw - ryaw)

        angular_z = self.angular_kp * yaw_error
        angular_z = max(-self.max_angular_speed, min(self.max_angular_speed, angular_z))

        # Smooth forward motion:
        # - If the robot is almost facing opposite direction, rotate in place.
        # - If moderately misaligned, still move slowly forward.
        # - If aligned, move faster.
        heading_scale = max(0.0, math.cos(yaw_error))
        distance_scale = min(1.0, distance / max(self.slowdown_distance, 1e-6))

        if abs(yaw_error) > 1.5:
            linear_x = 0.0
        elif abs(yaw_error) > self.angle_tolerance:
            linear_x = self.min_linear_speed * 0.5
        else:
            linear_x = self.linear_speed * max(0.25, heading_scale) * distance_scale
            linear_x = max(self.min_linear_speed, linear_x)

        self.publish_cmd(linear_x, angular_z)

    def publish_wander_or_stop(self) -> None:
        if not self.wander_when_no_goal:
            self.publish_stop()
            return

        if self.front_obstacle_too_close():
            self.publish_cmd(0.0, self.wander_turn_speed)
        else:
            self.publish_cmd(self.wander_linear_speed, 0.0)

    def should_recompute_goal(self) -> bool:
        now = self.get_clock().now()
        elapsed = (now - self.last_goal_time).nanoseconds / 1e9
        return elapsed >= self.goal_recompute_period

    # ----------------------------------------------------------------------
    # TF
    # ----------------------------------------------------------------------
    def lookup_robot_pose(self) -> Optional[Tuple[float, float, float]]:
        try:
            transform = self.tf_buffer.lookup_transform(
                self.map_frame,
                self.base_frame,
                Time(),
                timeout=Duration(seconds=0.2),
            )
        except TransformException:
            return None

        t = transform.transform.translation
        q = transform.transform.rotation

        x = t.x
        y = t.y
        yaw = quaternion_to_yaw(q)

        return (x, y, yaw)

    # ----------------------------------------------------------------------
    # Frontier goal selection
    # ----------------------------------------------------------------------
    def select_frontier_goal(
        self,
        robot_xy: WorldPoint,
        msg: OccupancyGrid,
    ) -> Optional[WorldPoint]:
        clusters = self.detect_frontier_clusters(msg)

        if self.log_frontier_stats:
            sizes = sorted([len(c) for c in clusters], reverse=True)[:5]
            self.get_logger().info(
                f"frontier clusters={len(clusters)}, top_sizes={sizes}"
            )

        if not clusters:
            return None

        rx, ry = robot_xy
        best_goal: Optional[WorldPoint] = None
        best_score = float("inf")

        fallback_goal: Optional[WorldPoint] = None
        fallback_score = float("inf")

        for cluster in clusters:
            frontier_cell = self.cluster_representative(cluster)
            frontier_world = self.grid_to_world(frontier_cell[1], frontier_cell[0], msg)

            goal_world = self.make_goal_from_frontier_cluster(
                cluster=cluster,
                frontier_cell=frontier_cell,
                msg=msg,
            )

            if goal_world is None:
                if self.fallback_to_frontier_goal:
                    goal_world = frontier_world
                else:
                    continue

            gx, gy = goal_world
            distance = math.hypot(gx - rx, gy - ry)

            info_gain = math.sqrt(len(cluster)) * msg.info.resolution
            score = distance - self.information_gain_weight * info_gain

            if score < fallback_score:
                fallback_score = score
                fallback_goal = goal_world

            if distance < self.min_goal_distance:
                continue

            if distance > self.max_goal_distance:
                continue

            if score < best_score:
                best_score = score
                best_goal = goal_world

        if best_goal is not None:
            if self.log_goal_details:
                self.get_logger().info(
                    f"selected exploration goal: "
                    f"x={best_goal[0]:.2f}, y={best_goal[1]:.2f}, "
                    f"score={best_score:.3f}"
                )
            return best_goal

        # Fallback:
        # In continuous exploration, returning some directional attractor is
        # often better than freezing.
        if self.fallback_to_frontier_goal and fallback_goal is not None:
            if self.log_goal_details:
                self.get_logger().warn(
                    f"using fallback goal despite distance filters: "
                    f"x={fallback_goal[0]:.2f}, y={fallback_goal[1]:.2f}, "
                    f"score={fallback_score:.3f}"
                )
            return fallback_goal

        return None

    def make_goal_from_frontier_cluster(
        self,
        cluster: List[GridCell],
        frontier_cell: GridCell,
        msg: OccupancyGrid,
    ) -> Optional[WorldPoint]:
        if not self.target_unknown_space:
            return self.grid_to_world(frontier_cell[1], frontier_cell[0], msg)

        unknown_dir = self.estimate_unknown_direction(cluster, msg)

        if unknown_dir is None:
            if self.fallback_to_frontier_goal:
                return self.grid_to_world(frontier_cell[1], frontier_cell[0], msg)
            return None

        return self.project_goal_toward_unknown(frontier_cell, unknown_dir, msg)

    def estimate_unknown_direction(
        self,
        cluster: List[GridCell],
        msg: OccupancyGrid,
    ) -> Optional[Tuple[float, float]]:
        width = msg.info.width
        height = msg.info.height
        grid = np.array(msg.data, dtype=np.int16).reshape((height, width))

        frontier_points = []
        unknown_points = []

        neighbors = [
            (-1, -1),
            (-1, 0),
            (-1, 1),
            (0, -1),
            (0, 1),
            (1, -1),
            (1, 0),
            (1, 1),
        ]

        for y, x in cluster:
            frontier_points.append((float(x), float(y)))

            for dy, dx in neighbors:
                ny = y + dy
                nx = x + dx

                if ny < 0 or ny >= height or nx < 0 or nx >= width:
                    continue

                if grid[ny, nx] == -1:
                    unknown_points.append((float(nx), float(ny)))

        if not frontier_points or not unknown_points:
            return None

        fx = sum(p[0] for p in frontier_points) / len(frontier_points)
        fy = sum(p[1] for p in frontier_points) / len(frontier_points)

        ux_mean = sum(p[0] for p in unknown_points) / len(unknown_points)
        uy_mean = sum(p[1] for p in unknown_points) / len(unknown_points)

        vx = ux_mean - fx
        vy = uy_mean - fy

        norm = math.hypot(vx, vy)

        if norm < 1e-6:
            return None

        return (vx / norm, vy / norm)

    def project_goal_toward_unknown(
        self,
        frontier_cell: GridCell,
        unknown_dir: Tuple[float, float],
        msg: OccupancyGrid,
    ) -> WorldPoint:
        y, x = frontier_cell
        ux, uy = unknown_dir

        resolution = msg.info.resolution
        offset_cells = self.unknown_target_offset / max(resolution, 1e-9)

        target_x = float(x) + ux * offset_cells
        target_y = float(y) + uy * offset_cells

        return self.grid_float_to_world(target_x, target_y, msg)

    # ----------------------------------------------------------------------
    # Frontier extraction
    # ----------------------------------------------------------------------
    def detect_frontier_clusters(self, msg: OccupancyGrid) -> List[List[GridCell]]:
        width = msg.info.width
        height = msg.info.height

        if width == 0 or height == 0:
            return []

        grid = np.array(msg.data, dtype=np.int16).reshape((height, width))

        free = (grid >= 0) & (grid <= self.free_threshold)
        unknown = grid == -1

        frontier = np.zeros((height, width), dtype=bool)

        for y in range(1, height - 1):
            for x in range(1, width - 1):
                if not free[y, x]:
                    continue

                if np.any(unknown[y - 1 : y + 2, x - 1 : x + 2]):
                    frontier[y, x] = True

        clusters = self.cluster_frontiers(frontier)

        return [
            cluster for cluster in clusters if len(cluster) >= self.min_frontier_size
        ]

    def cluster_frontiers(self, frontier: np.ndarray) -> List[List[GridCell]]:
        height, width = frontier.shape
        visited = np.zeros((height, width), dtype=bool)
        clusters: List[List[GridCell]] = []

        neighbors = [
            (-1, -1),
            (-1, 0),
            (-1, 1),
            (0, -1),
            (0, 1),
            (1, -1),
            (1, 0),
            (1, 1),
        ]

        for sy in range(height):
            for sx in range(width):
                if not frontier[sy, sx] or visited[sy, sx]:
                    continue

                cluster: List[GridCell] = []
                queue = deque([(sy, sx)])
                visited[sy, sx] = True

                while queue:
                    y, x = queue.popleft()
                    cluster.append((y, x))

                    for dy, dx in neighbors:
                        ny = y + dy
                        nx = x + dx

                        if ny < 0 or ny >= height or nx < 0 or nx >= width:
                            continue

                        if visited[ny, nx]:
                            continue

                        if not frontier[ny, nx]:
                            continue

                        visited[ny, nx] = True
                        queue.append((ny, nx))

                clusters.append(cluster)

        return clusters

    def cluster_representative(self, cluster: List[GridCell]) -> GridCell:
        ys = [cell[0] for cell in cluster]
        xs = [cell[1] for cell in cluster]

        cy = sum(ys) / len(ys)
        cx = sum(xs) / len(xs)

        return min(
            cluster,
            key=lambda cell: (cell[0] - cy) ** 2 + (cell[1] - cx) ** 2,
        )

    # ----------------------------------------------------------------------
    # Coordinate conversion
    # ----------------------------------------------------------------------
    def grid_to_world(self, x: int, y: int, msg: OccupancyGrid) -> WorldPoint:
        return self.grid_float_to_world(float(x), float(y), msg)

    def grid_float_to_world(
        self,
        x: float,
        y: float,
        msg: OccupancyGrid,
    ) -> WorldPoint:
        resolution = msg.info.resolution
        origin = msg.info.origin

        local_x = (x + 0.5) * resolution
        local_y = (y + 0.5) * resolution

        yaw = quaternion_to_yaw(origin.orientation)

        cos_yaw = math.cos(yaw)
        sin_yaw = math.sin(yaw)

        world_x = origin.position.x + cos_yaw * local_x - sin_yaw * local_y
        world_y = origin.position.y + sin_yaw * local_x + cos_yaw * local_y

        return (world_x, world_y)

    # ----------------------------------------------------------------------
    # Obstacle avoidance
    # ----------------------------------------------------------------------
    def front_obstacle_too_close(self) -> bool:
        if self.latest_scan is None:
            return False

        msg = self.latest_scan
        n = len(msg.ranges)

        if n == 0:
            return False

        half_width = max(1, int((self.front_angle_deg / 360.0) * n))
        front_indices = list(range(0, half_width)) + list(range(n - half_width, n))

        front_values = []
        for idx in front_indices:
            r = msg.ranges[idx]
            if math.isfinite(r):
                front_values.append(r)

        if not front_values:
            return False

        return min(front_values) < self.safe_front_distance

    # ----------------------------------------------------------------------
    # Command publishing
    # ----------------------------------------------------------------------
    def publish_cmd(self, linear_x: float, angular_z: float) -> None:
        msg = TwistStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.base_frame

        msg.twist.linear.x = float(linear_x)
        msg.twist.angular.z = float(angular_z)

        self.cmd_pub.publish(msg)

    def publish_stop(self) -> None:
        self.publish_cmd(0.0, 0.0)


def main(args=None):
    rclpy.init(args=args)
    node = FrontierCmdExplorer()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.publish_stop()
            node.destroy_node()
        except Exception:
            pass

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
