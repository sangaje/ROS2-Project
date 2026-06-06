#!/usr/bin/env python3

import math
from collections import deque
from typing import Dict, List, Optional, Tuple

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped, Quaternion
from nav2_msgs.action import NavigateToPose
from nav_msgs.msg import OccupancyGrid
from rclpy.action import ActionClient
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from rclpy.time import Time
from tf2_ros import Buffer, TransformException, TransformListener

GridCell = Tuple[int, int]  # (y, x)
WorldPoint = Tuple[float, float]


def yaw_to_quaternion(yaw: float) -> Quaternion:
    q = Quaternion()
    q.z = math.sin(yaw / 2.0)
    q.w = math.cos(yaw / 2.0)
    return q


def quaternion_to_yaw(q: Quaternion) -> float:
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


class FrontierExplorer(Node):
    """
    Frontier-based exploration node.

    Modes:
      - raw:
          Uses the frontier representative point directly.
          Almost no safety constraints.
          Useful for debugging whether Nav2 accepts goals.

      - relaxed:
          Applies backoff from frontier toward robot.
          Only checks that the resulting goal cell itself is free.

      - safe:
          Applies backoff + clearance + obstacle + unknown-ratio constraints.
          Intended for more stable operation.
    """

    def __init__(self):
        super().__init__("frontier_explorer")

        # ---------------------------------------------------------------------
        # Frame / topic parameters
        # ---------------------------------------------------------------------
        self.declare_parameter("map_topic", "/map")
        self.declare_parameter("map_frame", "map")
        self.declare_parameter("base_frame", "base_footprint")

        # ---------------------------------------------------------------------
        # OccupancyGrid thresholds
        # ---------------------------------------------------------------------
        self.declare_parameter("free_threshold", 20)
        self.declare_parameter("occupied_threshold", 65)

        # ---------------------------------------------------------------------
        # Frontier extraction
        # ---------------------------------------------------------------------
        self.declare_parameter("min_frontier_size", 8)

        # ---------------------------------------------------------------------
        # Exploration loop
        # ---------------------------------------------------------------------
        self.declare_parameter("timer_period", 2.0)
        self.declare_parameter("min_goal_distance", 0.45)
        self.declare_parameter("max_goal_distance", 8.0)
        self.declare_parameter("information_gain_weight", 0.20)

        # ---------------------------------------------------------------------
        # Goal generation mode
        # ---------------------------------------------------------------------
        self.declare_parameter("goal_mode", "relaxed")
        self.declare_parameter("use_distance_filter", True)
        self.declare_parameter("use_rejected_filter", True)

        # ---------------------------------------------------------------------
        # Safe / relaxed goal correction
        # ---------------------------------------------------------------------
        self.declare_parameter("goal_backoff_distance", 0.35)
        self.declare_parameter("goal_clearance_cells", 2)

        # Unknown handling in safety window:
        #   strict : reject if any unknown is in the clearance window
        #   ratio  : reject only if unknown ratio is too high
        #   ignore : ignore unknown cells around goal
        self.declare_parameter("unknown_policy", "ratio")
        self.declare_parameter("max_unknown_ratio", 0.60)

        # ---------------------------------------------------------------------
        # Rejection handling
        # ---------------------------------------------------------------------
        self.declare_parameter("rejected_goal_radius", 0.50)
        self.declare_parameter("max_rejected_goals", 40)

        # ---------------------------------------------------------------------
        # Debug / logging
        # ---------------------------------------------------------------------
        self.declare_parameter("log_frontier_stats", True)
        self.declare_parameter("log_goal_debug", True)

        # ---------------------------------------------------------------------
        # Read parameters
        # ---------------------------------------------------------------------
        self.map_topic = (
            self.get_parameter("map_topic").get_parameter_value().string_value
        )
        self.map_frame = (
            self.get_parameter("map_frame").get_parameter_value().string_value
        )
        self.base_frame = (
            self.get_parameter("base_frame").get_parameter_value().string_value
        )

        self.free_threshold = (
            self.get_parameter("free_threshold").get_parameter_value().integer_value
        )
        self.occupied_threshold = (
            self.get_parameter("occupied_threshold").get_parameter_value().integer_value
        )
        self.min_frontier_size = (
            self.get_parameter("min_frontier_size").get_parameter_value().integer_value
        )

        self.timer_period = (
            self.get_parameter("timer_period").get_parameter_value().double_value
        )
        self.min_goal_distance = (
            self.get_parameter("min_goal_distance").get_parameter_value().double_value
        )
        self.max_goal_distance = (
            self.get_parameter("max_goal_distance").get_parameter_value().double_value
        )
        self.information_gain_weight = (
            self.get_parameter("information_gain_weight")
            .get_parameter_value()
            .double_value
        )

        self.goal_mode = (
            self.get_parameter("goal_mode").get_parameter_value().string_value
        )
        self.use_distance_filter = (
            self.get_parameter("use_distance_filter").get_parameter_value().bool_value
        )
        self.use_rejected_filter = (
            self.get_parameter("use_rejected_filter").get_parameter_value().bool_value
        )

        self.goal_backoff_distance = (
            self.get_parameter("goal_backoff_distance")
            .get_parameter_value()
            .double_value
        )
        self.goal_clearance_cells = (
            self.get_parameter("goal_clearance_cells")
            .get_parameter_value()
            .integer_value
        )

        self.unknown_policy = (
            self.get_parameter("unknown_policy").get_parameter_value().string_value
        )
        self.max_unknown_ratio = (
            self.get_parameter("max_unknown_ratio").get_parameter_value().double_value
        )

        self.rejected_goal_radius = (
            self.get_parameter("rejected_goal_radius")
            .get_parameter_value()
            .double_value
        )
        self.max_rejected_goals = (
            self.get_parameter("max_rejected_goals").get_parameter_value().integer_value
        )

        self.log_frontier_stats = (
            self.get_parameter("log_frontier_stats").get_parameter_value().bool_value
        )
        self.log_goal_debug = (
            self.get_parameter("log_goal_debug").get_parameter_value().bool_value
        )

        # ---------------------------------------------------------------------
        # Runtime state
        # ---------------------------------------------------------------------
        self.latest_map: Optional[OccupancyGrid] = None
        self.navigating = False
        self.last_goal: Optional[WorldPoint] = None
        self.rejected_goals: List[WorldPoint] = []

        # ---------------------------------------------------------------------
        # TF
        # ---------------------------------------------------------------------
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # ---------------------------------------------------------------------
        # Subscriptions
        # ---------------------------------------------------------------------
        # SLAM Toolbox /map generally uses transient local durability.
        map_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        self.map_sub = self.create_subscription(
            OccupancyGrid,
            self.map_topic,
            self.on_map,
            map_qos,
        )

        # ---------------------------------------------------------------------
        # Nav2 action client
        # ---------------------------------------------------------------------
        self.nav_client = ActionClient(
            self,
            NavigateToPose,
            "navigate_to_pose",
        )

        # For RViz visualization/debugging
        self.goal_pub = self.create_publisher(
            PoseStamped,
            "/frontier_goal",
            10,
        )

        # ---------------------------------------------------------------------
        # Timer
        # ---------------------------------------------------------------------
        self.timer = self.create_timer(self.timer_period, self.exploration_step)

        self.get_logger().info(
            f"frontier_explorer started. "
            f"map_topic={self.map_topic}, "
            f"map_frame={self.map_frame}, "
            f"base_frame={self.base_frame}, "
            f"goal_mode={self.goal_mode}"
        )

    # -------------------------------------------------------------------------
    # ROS callbacks
    # -------------------------------------------------------------------------
    def on_map(self, msg: OccupancyGrid) -> None:
        self.latest_map = msg

    def exploration_step(self) -> None:
        if self.latest_map is None:
            self.get_logger().info("waiting for /map...")
            return

        if self.navigating:
            return

        robot_pose = self.lookup_robot_pose()
        if robot_pose is None:
            self.get_logger().warn(
                f"failed to lookup robot pose: {self.map_frame} -> {self.base_frame}"
            )
            return

        frontiers = self.detect_frontier_clusters(self.latest_map)

        if self.log_frontier_stats:
            sizes = sorted([len(c) for c in frontiers], reverse=True)[:5]
            self.get_logger().info(
                f"frontier clusters={len(frontiers)}, top_sizes={sizes}"
            )

        if not frontiers:
            self.get_logger().info("no frontier found. exploration may be complete.")
            return

        goal = self.select_goal(frontiers, robot_pose, self.latest_map)

        if goal is None:
            self.get_logger().info("no valid frontier goal selected")
            return

        self.send_nav_goal(goal, robot_pose)

    # -------------------------------------------------------------------------
    # TF / pose
    # -------------------------------------------------------------------------
    def lookup_robot_pose(self) -> Optional[WorldPoint]:
        try:
            transform = self.tf_buffer.lookup_transform(
                self.map_frame,
                self.base_frame,
                Time(),
                timeout=Duration(seconds=0.5),
            )
        except TransformException as exc:
            self.get_logger().debug(f"TF lookup failed: {exc}")
            return None

        x = transform.transform.translation.x
        y = transform.transform.translation.y
        return (x, y)

    # -------------------------------------------------------------------------
    # Frontier extraction
    # -------------------------------------------------------------------------
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

                # A frontier cell is a free cell adjacent to unknown space.
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

    # -------------------------------------------------------------------------
    # Goal selection
    # -------------------------------------------------------------------------
    def select_goal(
        self,
        clusters: List[List[GridCell]],
        robot_pose: WorldPoint,
        msg: OccupancyGrid,
    ) -> Optional[WorldPoint]:
        best_goal: Optional[WorldPoint] = None
        best_score = float("inf")

        rx, ry = robot_pose

        debug_counts: Dict[str, int] = {
            "total_clusters": 0,
            "candidate_none": 0,
            "near_rejected": 0,
            "too_close": 0,
            "too_far": 0,
            "accepted_candidate": 0,
        }

        for cluster in clusters:
            debug_counts["total_clusters"] += 1

            candidate_cell = self.cluster_representative(cluster)
            frontier_world = self.grid_to_world(
                candidate_cell[1], candidate_cell[0], msg
            )

            goal_candidate = self.make_goal_candidate(
                frontier_world=frontier_world,
                robot_pose=robot_pose,
                msg=msg,
            )

            if goal_candidate is None:
                debug_counts["candidate_none"] += 1
                continue

            if self.goal_mode != "raw" and self.use_rejected_filter:
                if self.is_near_rejected_goal(goal_candidate):
                    debug_counts["near_rejected"] += 1
                    continue

            wx, wy = goal_candidate
            distance = math.hypot(wx - rx, wy - ry)

            if self.goal_mode != "raw" and self.use_distance_filter:
                if distance < self.min_goal_distance:
                    debug_counts["too_close"] += 1
                    continue

                if distance > self.max_goal_distance:
                    debug_counts["too_far"] += 1
                    continue

            debug_counts["accepted_candidate"] += 1

            info_gain = math.sqrt(len(cluster)) * msg.info.resolution

            # Lower score is better.
            # Prefer closer frontiers while rewarding larger frontier clusters.
            score = distance - self.information_gain_weight * info_gain

            # Mildly penalize repeatedly selecting almost the same goal.
            if self.last_goal is not None:
                lx, ly = self.last_goal
                if math.hypot(wx - lx, wy - ly) < 0.35:
                    score += 0.5

            if score < best_score:
                best_score = score
                best_goal = (wx, wy)

        if self.log_goal_debug:
            self.get_logger().info(
                f"goal_mode={self.goal_mode}, goal selection debug={debug_counts}"
            )

        if best_goal is not None:
            self.get_logger().info(
                f"selected goal: x={best_goal[0]:.2f}, y={best_goal[1]:.2f}, "
                f"score={best_score:.3f}"
            )

        return best_goal

    def make_goal_candidate(
        self,
        frontier_world: WorldPoint,
        robot_pose: WorldPoint,
        msg: OccupancyGrid,
    ) -> Optional[WorldPoint]:
        """
        Create a goal candidate according to goal_mode.

        raw:
            p_goal = p_frontier
            Almost unconstrained. Debug only.

        relaxed:
            p_goal = backoff(frontier, robot)
            Only checks that the resulting cell itself is free.

        safe:
            p_goal = backoff(frontier, robot)
            Checks free cell, obstacle clearance, unknown policy, etc.
        """

        if self.goal_mode == "raw":
            return frontier_world

        if self.goal_mode == "relaxed":
            return self.backoff_goal_relaxed(
                frontier_world=frontier_world,
                robot_pose=robot_pose,
                msg=msg,
                backoff_distance=self.goal_backoff_distance,
            )

        if self.goal_mode == "safe":
            return self.backoff_goal_from_frontier(
                frontier_world=frontier_world,
                robot_pose=robot_pose,
                msg=msg,
                backoff_distance=self.goal_backoff_distance,
            )

        self.get_logger().warn(
            f"unknown goal_mode={self.goal_mode}. fallback to safe mode."
        )

        return self.backoff_goal_from_frontier(
            frontier_world=frontier_world,
            robot_pose=robot_pose,
            msg=msg,
            backoff_distance=self.goal_backoff_distance,
        )

    def cluster_representative(self, cluster: List[GridCell]) -> GridCell:
        ys = [cell[0] for cell in cluster]
        xs = [cell[1] for cell in cluster]

        cy = sum(ys) / len(ys)
        cx = sum(xs) / len(xs)

        # Choose the actual frontier cell closest to the centroid.
        return min(
            cluster,
            key=lambda cell: (cell[0] - cy) ** 2 + (cell[1] - cx) ** 2,
        )

    # -------------------------------------------------------------------------
    # Coordinate conversion
    # -------------------------------------------------------------------------
    def grid_to_world(self, x: int, y: int, msg: OccupancyGrid) -> WorldPoint:
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

    def world_to_grid(
        self,
        wx: float,
        wy: float,
        msg: OccupancyGrid,
    ) -> Optional[GridCell]:
        resolution = msg.info.resolution
        origin = msg.info.origin

        dx = wx - origin.position.x
        dy = wy - origin.position.y

        yaw = quaternion_to_yaw(origin.orientation)

        cos_yaw = math.cos(-yaw)
        sin_yaw = math.sin(-yaw)

        local_x = cos_yaw * dx - sin_yaw * dy
        local_y = sin_yaw * dx + cos_yaw * dy

        x = int(local_x / resolution)
        y = int(local_y / resolution)

        if x < 0 or y < 0 or x >= msg.info.width or y >= msg.info.height:
            return None

        return (y, x)

    # -------------------------------------------------------------------------
    # Goal correction / constraints
    # -------------------------------------------------------------------------
    def backoff_goal_from_frontier(
        self,
        frontier_world: WorldPoint,
        robot_pose: WorldPoint,
        msg: OccupancyGrid,
        backoff_distance: float,
    ) -> Optional[WorldPoint]:
        """
        Safe mode:
        Try several points between frontier and robot.
        Accept the first point that passes is_safe_goal_cell().
        """

        fx, fy = frontier_world
        rx, ry = robot_pose

        vx = fx - rx
        vy = fy - ry
        norm = math.hypot(vx, vy)

        if norm < 1e-6:
            return None

        ux = vx / norm
        uy = vy / norm

        candidate_distances = self.unique_backoff_distances(backoff_distance)

        for d in candidate_distances:
            gx = fx - d * ux
            gy = fy - d * uy

            cell = self.world_to_grid(gx, gy, msg)
            if cell is None:
                continue

            y, x = cell

            if self.is_safe_goal_cell(y, x, msg):
                return (gx, gy)

        return None

    def backoff_goal_relaxed(
        self,
        frontier_world: WorldPoint,
        robot_pose: WorldPoint,
        msg: OccupancyGrid,
        backoff_distance: float,
    ) -> Optional[WorldPoint]:
        """
        Relaxed mode:
        Try several backoff distances.
        Accept the first point whose cell itself is free.
        """

        fx, fy = frontier_world
        rx, ry = robot_pose

        vx = fx - rx
        vy = fy - ry
        norm = math.hypot(vx, vy)

        if norm < 1e-6:
            return None

        ux = vx / norm
        uy = vy / norm

        candidate_distances = self.unique_backoff_distances(backoff_distance)

        for d in candidate_distances:
            gx = fx - d * ux
            gy = fy - d * uy

            cell = self.world_to_grid(gx, gy, msg)
            if cell is None:
                continue

            y, x = cell

            if self.is_basic_free_cell(y, x, msg):
                return (gx, gy)

        return None

    def unique_backoff_distances(self, preferred: float) -> List[float]:
        values = [
            0.00,
            0.15,
            0.25,
            preferred,
            0.35,
            0.50,
            0.75,
        ]

        unique = sorted(set(round(v, 3) for v in values))
        return unique

    def is_basic_free_cell(
        self,
        y: int,
        x: int,
        msg: OccupancyGrid,
    ) -> bool:
        width = msg.info.width
        height = msg.info.height

        if y < 0 or y >= height or x < 0 or x >= width:
            return False

        grid = np.array(msg.data, dtype=np.int16).reshape((height, width))
        value = grid[y, x]

        return 0 <= value <= self.free_threshold

    def is_safe_goal_cell(
        self,
        y: int,
        x: int,
        msg: OccupancyGrid,
    ) -> bool:
        width = msg.info.width
        height = msg.info.height

        if y < 0 or y >= height or x < 0 or x >= width:
            return False

        grid = np.array(msg.data, dtype=np.int16).reshape((height, width))

        # Goal cell itself must be free.
        if grid[y, x] < 0:
            return False

        if grid[y, x] > self.free_threshold:
            return False

        r = self.goal_clearance_cells

        y0 = max(0, y - r)
        y1 = min(height, y + r + 1)
        x0 = max(0, x - r)
        x1 = min(width, x + r + 1)

        window = grid[y0:y1, x0:x1]

        # Reject if obstacle is nearby.
        if np.any(window >= self.occupied_threshold):
            return False

        unknown_count = int(np.sum(window == -1))
        total_count = int(window.size)
        unknown_ratio = unknown_count / max(total_count, 1)

        if self.unknown_policy == "strict":
            if unknown_count > 0:
                return False

        elif self.unknown_policy == "ratio":
            if unknown_ratio > self.max_unknown_ratio:
                return False

        elif self.unknown_policy == "ignore":
            pass

        else:
            self.get_logger().warn(
                f"unknown unknown_policy={self.unknown_policy}. fallback to ratio."
            )
            if unknown_ratio > self.max_unknown_ratio:
                return False

        return True

    # -------------------------------------------------------------------------
    # Rejected-goal memory
    # -------------------------------------------------------------------------
    def is_near_rejected_goal(self, goal: WorldPoint) -> bool:
        gx, gy = goal

        for rx, ry in self.rejected_goals:
            if math.hypot(gx - rx, gy - ry) < self.rejected_goal_radius:
                return True

        return False

    def remember_rejected_goal(self, goal: Optional[WorldPoint]) -> None:
        if goal is None:
            return

        self.rejected_goals.append(goal)

        if len(self.rejected_goals) > self.max_rejected_goals:
            self.rejected_goals.pop(0)

    # -------------------------------------------------------------------------
    # Nav2 action
    # -------------------------------------------------------------------------
    def send_nav_goal(self, goal: WorldPoint, robot_pose: WorldPoint) -> None:
        if not self.nav_client.wait_for_server(timeout_sec=1.0):
            self.get_logger().warn("NavigateToPose action server not available")
            return

        gx, gy = goal
        rx, ry = robot_pose

        yaw = math.atan2(gy - ry, gx - rx)

        pose = PoseStamped()
        pose.header.frame_id = self.map_frame
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.pose.position.x = gx
        pose.pose.position.y = gy
        pose.pose.position.z = 0.0
        pose.pose.orientation = yaw_to_quaternion(yaw)

        goal_msg = NavigateToPose.Goal()
        goal_msg.pose = pose

        self.get_logger().info(
            f"sending frontier goal: x={gx:.2f}, y={gy:.2f}, yaw={yaw:.2f}"
        )

        self.goal_pub.publish(pose)

        self.navigating = True
        self.last_goal = goal

        send_future = self.nav_client.send_goal_async(
            goal_msg,
            feedback_callback=self.feedback_callback,
        )
        send_future.add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future) -> None:
        try:
            goal_handle = future.result()
        except Exception as exc:
            self.get_logger().error(f"failed to send goal: {exc}")
            self.remember_rejected_goal(self.last_goal)
            self.navigating = False
            return

        if not goal_handle.accepted:
            self.get_logger().warn("frontier goal rejected")
            self.remember_rejected_goal(self.last_goal)
            self.navigating = False
            return

        self.get_logger().info("frontier goal accepted")

        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self.result_callback)

    def feedback_callback(self, feedback_msg) -> None:
        _ = feedback_msg.feedback

    def result_callback(self, future) -> None:
        try:
            wrapped_result = future.result()
        except Exception as exc:
            self.get_logger().error(f"failed to get navigation result: {exc}")
            self.navigating = False
            return

        status = wrapped_result.status
        result = wrapped_result.result

        self.get_logger().info(f"frontier goal finished with status={status}")

        if hasattr(result, "error_code"):
            self.get_logger().info(f"nav2 error_code={result.error_code}")

        if hasattr(result, "error_msg"):
            self.get_logger().info(f"nav2 error_msg={result.error_msg}")

        # action_msgs/GoalStatus: STATUS_SUCCEEDED = 4
        if status != 4:
            self.remember_rejected_goal(self.last_goal)

        self.navigating = False


def main(args=None):
    rclpy.init(args=args)
    node = FrontierExplorer()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.destroy_node()
        except Exception:
            pass

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
