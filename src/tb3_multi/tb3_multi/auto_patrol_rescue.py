#!/usr/bin/env python3
"""
Automatic patrol and rescue manager for the Burger/Waffle pair.

This node is intentionally centralized.
The remaining robots do not directly "see" the failed robot.  Instead, this
manager watches every robot's odom/map pose, remembers the last known position
of a failed Burger, and repeatedly sends high-priority rescue goals to the
remaining robots.
"""

import json
import math
from typing import Dict, List, Optional, Tuple

from geometry_msgs.msg import PointStamped, PoseStamped
from nav_msgs.msg import Odometry
import rclpy
from rclpy.node import Node
from rclpy.time import Time
from std_msgs.msg import String
from visualization_msgs.msg import Marker, MarkerArray

Pose2D = Tuple[float, float, float]
Point2D = Tuple[float, float]


DEFAULT_WAYPOINTS: Dict[str, List[Point2D]] = {
    'burger1': [(-2.25, 0.00), (-2.25, 0.75), (-2.25, -0.75), (-2.25, 0.00)],
    'waffle1': [(-2.85, 0.65)],
}

DEFAULT_INITIAL_POSES: Dict[str, Pose2D] = {
    'burger1': (-2.25, 0.00, 0.0),
    'waffle1': (-2.85, 0.65, 0.0),
}


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def normalize_angle(angle: float) -> float:
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle


def yaw_from_quaternion(x: float, y: float, z: float, w: float) -> float:
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


def relative_pose(reference: Pose2D, current: Pose2D) -> Pose2D:
    rx, ry, ryaw = reference
    cx, cy, cyaw = current
    dx = cx - rx
    dy = cy - ry
    c = math.cos(ryaw)
    s = math.sin(ryaw)
    return (
        c * dx + s * dy,
        -s * dx + c * dy,
        normalize_angle(cyaw - ryaw),
    )


def compose_pose(origin: Pose2D, delta: Pose2D) -> Pose2D:
    ox, oy, oyaw = origin
    dx, dy, dyaw = delta
    c = math.cos(oyaw)
    s = math.sin(oyaw)
    return (
        ox + c * dx - s * dy,
        oy + s * dx + c * dy,
        normalize_angle(oyaw + dyaw),
    )


def distance(a: Point2D, b: Point2D) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def parse_waypoints(raw: str, fallback: List[Point2D]) -> List[Point2D]:
    try:
        data = json.loads(raw)
        points: List[Point2D] = []
        for item in data:
            if isinstance(item, list) and len(item) >= 2:
                points.append((float(item[0]), float(item[1])))
        return points if points else fallback
    except Exception:
        return fallback


def as_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in ('1', 'true', 'yes', 'on')
    return bool(value)


class AutoPatrolRescue(Node):
    """Central manager for patrol goals, failure state, and rescue assignments."""

    def __init__(self) -> None:
        super().__init__('auto_patrol_rescue')

        self.declare_parameter('robots', ['burger1', 'waffle1'])
        self.declare_parameter('burger_robots', ['burger1'])
        self.declare_parameter('goal_tolerance', 0.25)
        self.declare_parameter('enable_patrol', False)
        # Robots that should move during normal operation.
        # Only burger1 patrols; waffle1 stays idle until burger1 fails.
        self.declare_parameter('patrol_robots', ['burger1'])
        self.declare_parameter('timer_period', 0.5)
        self.declare_parameter('patrol_republish_period', 1.0)
        self.declare_parameter('rescue_republish_period', 0.2)
        self.declare_parameter('signal_timeout_sec', 3.0)
        self.declare_parameter('enable_timeout_detection', False)
        self.declare_parameter('startup_grace_sec', 8.0)
        self.declare_parameter('rescue_offset_m', 0.45)
        self.declare_parameter('burger1_waypoints', json.dumps(DEFAULT_WAYPOINTS['burger1']))
        self.declare_parameter('waffle1_waypoints', json.dumps(DEFAULT_WAYPOINTS['waffle1']))
        for robot, pose in DEFAULT_INITIAL_POSES.items():
            self.declare_parameter(f'{robot}_initial_x', pose[0])
            self.declare_parameter(f'{robot}_initial_y', pose[1])
            self.declare_parameter(f'{robot}_initial_yaw', pose[2])

        self.robots: List[str] = list(self.get_parameter('robots').value)
        self.burger_robots = set(self.get_parameter('burger_robots').value)
        self.goal_tolerance = float(self.get_parameter('goal_tolerance').value)
        self.enable_patrol = as_bool(self.get_parameter('enable_patrol').value)
        self.patrol_robots = set(self.get_parameter('patrol_robots').value)
        self.patrol_republish_period = float(self.get_parameter('patrol_republish_period').value)
        self.rescue_republish_period = float(self.get_parameter('rescue_republish_period').value)
        self.signal_timeout_sec = float(self.get_parameter('signal_timeout_sec').value)
        self.enable_timeout_detection = as_bool(
            self.get_parameter('enable_timeout_detection').value
        )
        self.startup_grace_sec = float(self.get_parameter('startup_grace_sec').value)
        self.rescue_offset_m = float(self.get_parameter('rescue_offset_m').value)

        self.initial_pose: Dict[str, Pose2D] = {}
        self.odom_zero_pose: Dict[str, Pose2D] = {}
        for robot in self.robots:
            fallback = DEFAULT_INITIAL_POSES.get(robot, (0.0, 0.0, 0.0))
            self.initial_pose[robot] = self.initial_pose_for(robot, fallback)

        self.waypoints: Dict[str, List[Point2D]] = {}
        for robot in self.robots:
            fallback = DEFAULT_WAYPOINTS.get(robot, [(0.0, 0.0)])
            raw = self.waypoints_parameter_for(robot, fallback)
            self.waypoints[robot] = parse_waypoints(raw, fallback)

        self.goal_pubs = {
            robot: self.create_publisher(PointStamped, f'/{robot}/goal_point', 10)
            for robot in self.robots
        }
        self.rescue_goal_pubs = {
            robot: self.create_publisher(PointStamped, f'/{robot}/rescue_goal', 10)
            for robot in self.robots
        }
        self.mode_pub = self.create_publisher(String, '/tb3_multi/mode', 10)
        self.event_pub = self.create_publisher(String, '/tb3_multi/rescue_event', 10)
        self.status_pub = self.create_publisher(String, '/tb3_multi/robot_status', 10)
        self.marker_pub = self.create_publisher(MarkerArray, '/tb3_multi/rescue_markers', 10)
        # Single centralized publisher for all robot markers.  This avoids RViz
        # flickering caused by three controller nodes publishing partial marker
        # arrays to the same topic.
        self.robot_marker_pub = self.create_publisher(MarkerArray, '/tb3_multi/robot_markers', 10)

        self.pose: Dict[str, Pose2D] = {}
        self.pose_source: Dict[str, str] = {}
        self.last_pose_time: Dict[str, Time] = {}
        self.last_goal: Dict[str, Optional[Point2D]] = {robot: None for robot in self.robots}
        self.last_goal_publish_time: Dict[str, Optional[Time]] = {
            robot: None for robot in self.robots
        }
        self.waypoint_index: Dict[str, int] = {robot: -1 for robot in self.robots}
        self.failed_robots = set()
        self.failure_position: Dict[str, Point2D] = {}
        self.failure_reason: Dict[str, str] = {}
        self.rescue_assignments: Dict[str, Point2D] = {}
        self.start_time = self.get_clock().now()

        for robot in self.robots:
            self.create_subscription(
                Odometry,
                f'/{robot}/odom',
                lambda msg, r=robot: self.on_odom(r, msg),
                20,
            )
            self.create_subscription(
                PoseStamped,
                f'/{robot}/map_pose',
                lambda msg, r=robot: self.on_map_pose(r, msg),
                20,
            )

        self.create_subscription(String, '/tb3_multi/fail_robot', self.on_fail_robot, 10)
        self.create_subscription(
            String,
            '/tb3_multi/robot_failure_report',
            self.on_failure_report,
            10,
        )
        self.create_subscription(String, '/tb3_multi/recover_robot', self.on_recover_robot, 10)
        self.create_timer(float(self.get_parameter('timer_period').value), self.on_timer)

        self.get_logger().info(
            'AutoPatrolRescue started. '
            f'enable_patrol={self.enable_patrol}, '
            f'patrol_robots={sorted(self.patrol_robots)}, '
            f'timeout_detection={self.enable_timeout_detection}. '
            'Manual test: ros2 topic pub --once /tb3_multi/fail_robot '
            'std_msgs/msg/String "{data: \'burger1\'}"'
        )

    def initial_pose_for(self, robot: str, fallback: Pose2D) -> Pose2D:
        return (
            float(self.get_parameter(f'{robot}_initial_x').value)
            if self.has_parameter(f'{robot}_initial_x') else fallback[0],
            float(self.get_parameter(f'{robot}_initial_y').value)
            if self.has_parameter(f'{robot}_initial_y') else fallback[1],
            float(self.get_parameter(f'{robot}_initial_yaw').value)
            if self.has_parameter(f'{robot}_initial_yaw') else fallback[2],
        )

    def waypoints_parameter_for(self, robot: str, fallback: List[Point2D]) -> str:
        parameter_name = f'{robot}_waypoints'
        if self.has_parameter(parameter_name):
            return str(self.get_parameter(parameter_name).value)
        return json.dumps(fallback)

    def now(self) -> Time:
        return self.get_clock().now()

    def seconds_since(self, stamp: Time) -> float:
        return (self.now() - stamp).nanoseconds / 1e9

    def update_pose(self, robot: str, pose: Pose2D, source: str) -> None:
        self.pose[robot] = pose
        self.pose_source[robot] = source
        self.last_pose_time[robot] = self.now()

    def on_odom(self, robot: str, msg: Odometry) -> None:
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        odom_pose: Pose2D = (p.x, p.y, yaw_from_quaternion(q.x, q.y, q.z, q.w))
        if robot not in self.odom_zero_pose:
            self.odom_zero_pose[robot] = odom_pose
            self.get_logger().info(
                f'{robot}: odom calibrated for manager. '
                f'first_odom=({odom_pose[0]:.2f}, {odom_pose[1]:.2f}) '
                f'initial_map=({self.initial_pose[robot][0]:.2f}, '
                f'{self.initial_pose[robot][1]:.2f})'
            )
        delta = relative_pose(self.odom_zero_pose[robot], odom_pose)
        self.update_pose(robot, compose_pose(self.initial_pose[robot], delta), 'odom')

    def on_map_pose(self, robot: str, msg: PoseStamped) -> None:
        # Keep this as an additional source, but odom is enough for rescue.
        q = msg.pose.orientation
        self.update_pose(
            robot,
            (
                msg.pose.position.x,
                msg.pose.position.y,
                yaw_from_quaternion(q.x, q.y, q.z, q.w),
            ),
            'map_pose',
        )

    def on_fail_robot(self, msg: String) -> None:
        robot = msg.data.strip()
        if robot.lower() in ('reset', 'clear', 'none'):
            self.clear_failures('manual reset')
            return
        if robot not in self.robots:
            self.log_unknown_robot(robot)
            return
        self.mark_failed(robot, 'manual signal lost command')

    def on_failure_report(self, msg: String) -> None:
        robot = msg.data.strip()
        if robot in self.robots and robot not in self.failed_robots:
            self.mark_failed(robot, 'controller failure report')

    def on_recover_robot(self, msg: String) -> None:
        robot = msg.data.strip()
        if robot.lower() in ('all', 'reset', 'clear'):
            self.clear_failures('manual recover all')
            return
        if robot not in self.robots:
            self.log_unknown_robot(robot)
            return
        if robot in self.failed_robots:
            self.failed_robots.remove(robot)
            self.failure_position.pop(robot, None)
            self.failure_reason.pop(robot, None)
            self.rescue_assignments.clear()
            self.last_goal[robot] = None
            self.publish_event(f'{robot} recovered. Patrol mode will resume.')

    def clear_failures(self, reason: str) -> None:
        self.failed_robots.clear()
        self.failure_position.clear()
        self.failure_reason.clear()
        self.rescue_assignments.clear()
        for robot in self.robots:
            self.last_goal[robot] = None
        self.publish_event(f'All failures cleared: {reason}')

    def mark_failed(self, robot: str, reason: str) -> None:
        if robot in self.failed_robots:
            return
        if robot in self.pose:
            fail_point = (self.pose[robot][0], self.pose[robot][1])
            source = self.pose_source.get(robot, 'pose')
        else:
            fail_point = self.initial_pose.get(robot, (0.0, 0.0, 0.0))[:2]
            source = 'initial_fallback'

        self.failed_robots.add(robot)
        self.failure_position[robot] = fail_point
        self.failure_reason[robot] = reason
        self.rescue_assignments.clear()
        self.last_goal[robot] = fail_point
        self.publish_goal(robot, fail_point, rescue=False, force=True)
        self.publish_event(
            f'{robot} SIGNAL LOST ({reason}). Last position from {source}: '
            f'x={fail_point[0]:.2f}, y={fail_point[1]:.2f}. '
            'Remaining robots are switching to RESCUE.'
        )

    def log_unknown_robot(self, robot: str) -> None:
        self.get_logger().warn(
            f'Unknown robot "{robot}". Valid robots: {", ".join(self.robots)}'
        )

    def publish_event(self, text: str) -> None:
        msg = String()
        msg.data = text
        self.event_pub.publish(msg)
        self.get_logger().warn(text)

    def on_timer(self) -> None:
        self.check_timeouts()
        if self.failed_robots:
            self.publish_mode('RESCUE')
            self.rescue_loop()
        elif self.enable_patrol:
            self.publish_mode('PATROL')
            self.patrol_loop()
        else:
            self.publish_mode('MANUAL_READY')
        self.publish_status()
        self.publish_robot_markers()
        self.publish_markers()

    def publish_mode(self, mode: str) -> None:
        msg = String()
        msg.data = mode
        self.mode_pub.publish(msg)

    def check_timeouts(self) -> None:
        if not self.enable_timeout_detection:
            return
        if self.seconds_since(self.start_time) < self.startup_grace_sec:
            return
        for robot in self.burger_robots:
            if robot in self.failed_robots:
                continue
            if robot not in self.last_pose_time:
                continue
            if self.seconds_since(self.last_pose_time[robot]) > self.signal_timeout_sec:
                self.mark_failed(
                    robot,
                    f'pose timeout > {self.signal_timeout_sec:.1f}s',
                )

    def patrol_loop(self) -> None:
        # Only patrol_robots receive normal waypoint goals.
        # Non-patrol robots remain stopped and wait for a rescue_goal.
        for robot in self.robots:
            if robot not in self.patrol_robots:
                self.last_goal[robot] = None
                continue
            if robot not in self.pose:
                continue
            current = (self.pose[robot][0], self.pose[robot][1])
            goal = self.last_goal.get(robot)
            should_choose_next = (
                goal is None or distance(current, goal) <= self.goal_tolerance
            )
            if should_choose_next:
                self.waypoint_index[robot] = (
                    self.waypoint_index[robot] + 1
                ) % len(self.waypoints[robot])
                goal = self.waypoints[robot][self.waypoint_index[robot]]
                self.publish_goal(robot, goal, rescue=False, force=True)
            else:
                self.publish_goal(robot, goal, rescue=False, force=False)

    def rescue_loop(self) -> None:
        failed_robot = sorted(self.failed_robots)[0]
        failed_point = self.failure_position.get(failed_robot)
        if failed_point is None:
            return

        active_robots = [robot for robot in self.robots if robot not in self.failed_robots]
        for index, robot in enumerate(active_robots):
            goal = self.rescue_goal_for_robot(failed_point, index, len(active_robots))
            self.rescue_assignments[robot] = goal
            # High-priority rescue goal. Publish forcibly and repeatedly so it is not missed.
            self.publish_goal(robot, goal, rescue=True, force=True)

        for robot in self.failed_robots:
            stop_point = self.failure_position.get(robot)
            if stop_point is not None:
                self.publish_goal(robot, stop_point, rescue=False, force=False)

    def rescue_goal_for_robot(
        self,
        failed_point: Point2D,
        index: int,
        count: int,
    ) -> Point2D:
        if index == 0 or self.rescue_offset_m <= 0.0 or count <= 1:
            return failed_point
        angle = (2.0 * math.pi * (index - 1)) / max(1, count - 1)
        return (
            failed_point[0] + self.rescue_offset_m * math.cos(angle),
            failed_point[1] + self.rescue_offset_m * math.sin(angle),
        )

    def publish_goal(
        self,
        robot: str,
        point: Point2D,
        rescue: bool,
        force: bool = False,
    ) -> None:
        now = self.now()
        period = self.rescue_republish_period if rescue else self.patrol_republish_period
        last_time = self.last_goal_publish_time.get(robot)
        if not force and last_time is not None and self.seconds_since(last_time) < period:
            return

        msg = PointStamped()
        msg.header.stamp = now.to_msg()
        msg.header.frame_id = 'map'
        msg.point.x = float(point[0])
        msg.point.y = float(point[1])
        msg.point.z = 0.0
        if rescue:
            # Publish to both the dedicated rescue topic and the normal goal topic.
            # The dedicated topic puts the controller into RESCUE mode. The normal
            # topic is a compatibility fallback for an old controller.
            self.rescue_goal_pubs[robot].publish(msg)
            self.goal_pubs[robot].publish(msg)
        else:
            self.goal_pubs[robot].publish(msg)
        self.last_goal[robot] = point
        self.last_goal_publish_time[robot] = now

    def publish_status(self) -> None:
        data = {}
        for robot in self.robots:
            pose = self.pose.get(robot)
            data[robot] = {
                'state': self.state_for(robot),
                'x': None if pose is None else round(pose[0], 3),
                'y': None if pose is None else round(pose[1], 3),
                'source': self.pose_source.get(robot, 'none'),
                'last_goal': self.last_goal.get(robot),
            }
        msg = String()
        msg.data = json.dumps(data, ensure_ascii=False)
        self.status_pub.publish(msg)

    def state_for(self, robot: str) -> str:
        if robot in self.failed_robots:
            return 'SIGNAL_LOST'
        if self.failed_robots:
            return 'RESCUE'
        if self.enable_patrol and robot in self.patrol_robots:
            return 'PATROL'
        return 'IDLE'

    def publish_robot_markers(self) -> None:
        marker_array = MarkerArray()
        stamp = self.now().to_msg()
        for idx, robot in enumerate(self.robots):
            pose = self.pose.get(
                robot,
                self.initial_pose.get(robot, (0.0, 0.0, 0.0)),
            )
            x, y, yaw = pose
            state = self.marker_state_for(robot)
            color = self.color_for_robot(robot)

            body = Marker()
            body.header.frame_id = 'map'
            body.header.stamp = stamp
            body.ns = f'{robot}_body'
            body.id = 10 * idx
            body.type = Marker.ARROW
            body.action = Marker.ADD
            body.pose.position.x = x
            body.pose.position.y = y
            body.pose.position.z = 0.10
            body.pose.orientation.z = math.sin(yaw / 2.0)
            body.pose.orientation.w = math.cos(yaw / 2.0)
            body.scale.x = 0.42
            body.scale.y = 0.11
            body.scale.z = 0.11
            body.color.a = 1.0
            body.color.r, body.color.g, body.color.b = color
            marker_array.markers.append(body)

            text = Marker()
            text.header.frame_id = 'map'
            text.header.stamp = stamp
            text.ns = f'{robot}_label'
            text.id = 10 * idx + 1
            text.type = Marker.TEXT_VIEW_FACING
            text.action = Marker.ADD
            text.pose.position.x = x
            text.pose.position.y = y
            text.pose.position.z = 0.48
            text.pose.orientation.w = 1.0
            text.scale.z = 0.20
            text.color.a = 1.0
            text.color.r = 1.0
            text.color.g = 1.0
            text.color.b = 1.0
            text.text = f'{robot}\n{state}'
            marker_array.markers.append(text)

        self.robot_marker_pub.publish(marker_array)

    def marker_state_for(self, robot: str) -> str:
        if robot in self.failed_robots:
            return 'SIGNAL_LOST'
        if robot in self.rescue_assignments:
            return 'RESCUE'
        if self.enable_patrol and robot in self.patrol_robots:
            return 'PATROL'
        return 'IDLE'

    def publish_markers(self) -> None:
        marker_array = MarkerArray()
        stamp = self.now().to_msg()
        marker_id = 0

        for robot, points in self.waypoints.items():
            if robot not in self.patrol_robots:
                continue
            for point in points:
                marker = Marker()
                marker.header.frame_id = 'map'
                marker.header.stamp = stamp
                marker.ns = f'{robot}_patrol_waypoints'
                marker.id = marker_id
                marker_id += 1
                marker.type = Marker.CUBE
                marker.action = Marker.ADD
                marker.pose.position.x = point[0]
                marker.pose.position.y = point[1]
                marker.pose.position.z = 0.04
                marker.pose.orientation.w = 1.0
                marker.scale.x = 0.12
                marker.scale.y = 0.12
                marker.scale.z = 0.08
                marker.color.a = 0.45
                marker.color.r, marker.color.g, marker.color.b = self.color_for_robot(robot)
                marker_array.markers.append(marker)

        for robot, point in self.failure_position.items():
            marker = Marker()
            marker.header.frame_id = 'map'
            marker.header.stamp = stamp
            marker.ns = 'signal_lost_position'
            marker.id = 100 + self.robots.index(robot) if robot in self.robots else 199
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            marker.pose.position.x = point[0]
            marker.pose.position.y = point[1]
            marker.pose.position.z = 0.16
            marker.pose.orientation.w = 1.0
            marker.scale.x = 0.38
            marker.scale.y = 0.38
            marker.scale.z = 0.38
            marker.color.a = 0.95
            marker.color.r = 1.0
            marker.color.g = 0.0
            marker.color.b = 0.0
            marker_array.markers.append(marker)

            text = Marker()
            text.header.frame_id = 'map'
            text.header.stamp = stamp
            text.ns = 'signal_lost_text'
            text.id = 200 + self.robots.index(robot) if robot in self.robots else 299
            text.type = Marker.TEXT_VIEW_FACING
            text.action = Marker.ADD
            text.pose.position.x = point[0]
            text.pose.position.y = point[1]
            text.pose.position.z = 0.65
            text.pose.orientation.w = 1.0
            text.scale.z = 0.25
            text.color.a = 1.0
            text.color.r = 1.0
            text.color.g = 0.1
            text.color.b = 0.1
            text.text = f'{robot} SIGNAL LOST'
            marker_array.markers.append(text)

        for robot, point in self.rescue_assignments.items():
            marker = Marker()
            marker.header.frame_id = 'map'
            marker.header.stamp = stamp
            marker.ns = f'{robot}_rescue_goal'
            marker.id = 300 + self.robots.index(robot) if robot in self.robots else 399
            marker.type = Marker.CYLINDER
            marker.action = Marker.ADD
            marker.pose.position.x = point[0]
            marker.pose.position.y = point[1]
            marker.pose.position.z = 0.08
            marker.pose.orientation.w = 1.0
            marker.scale.x = 0.26
            marker.scale.y = 0.26
            marker.scale.z = 0.10
            marker.color.a = 0.85
            marker.color.r, marker.color.g, marker.color.b = self.color_for_robot(robot)
            marker_array.markers.append(marker)

        self.marker_pub.publish(marker_array)

    def color_for_robot(self, robot: str) -> Tuple[float, float, float]:
        if robot == 'burger1':
            return (1.0, 0.1, 0.1)
        return (0.1, 0.3, 1.0)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = AutoPatrolRescue()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
