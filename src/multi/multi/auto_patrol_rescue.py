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
from typing import Dict, List, Optional, Set, Tuple

from geometry_msgs.msg import PointStamped, PoseStamped
from nav_msgs.msg import OccupancyGrid, Odometry
import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
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
        self.declare_parameter('enable_scout_failover', True)
        self.declare_parameter('scout_robot', 'burger1')
        self.declare_parameter('failover_successor_robot', 'waffle1')
        self.declare_parameter('scout_liveness_required_sources', ['signal', 'map_pose', 'odom'])
        self.declare_parameter('scout_signal_timeout_sec', 2.5)
        self.declare_parameter('scout_map_pose_timeout_sec', 3.5)
        self.declare_parameter('scout_odom_timeout_sec', 3.5)
        self.declare_parameter('scout_down_grace_sec', 2.0)
        self.declare_parameter('last_scout_pose_max_age_sec', 8.0)
        self.declare_parameter('failover_standoff_m', 0.45)
        self.declare_parameter('max_recovery_goal_retries', 3)
        self.declare_parameter('recovery_goal_retry_period_sec', 1.5)
        self.declare_parameter('recovery_goal_timeout_sec', 45.0)
        self.declare_parameter('map_topic', '/map')
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
        self.initial_patrol_robots = set(self.patrol_robots)
        self.patrol_republish_period = float(self.get_parameter('patrol_republish_period').value)
        self.rescue_republish_period = float(self.get_parameter('rescue_republish_period').value)
        self.signal_timeout_sec = float(self.get_parameter('signal_timeout_sec').value)
        self.enable_timeout_detection = as_bool(
            self.get_parameter('enable_timeout_detection').value
        )
        self.startup_grace_sec = float(self.get_parameter('startup_grace_sec').value)
        self.rescue_offset_m = float(self.get_parameter('rescue_offset_m').value)
        self.enable_scout_failover = as_bool(
            self.get_parameter('enable_scout_failover').value
        )
        self.scout_robot = str(self.get_parameter('scout_robot').value).strip('/')
        self.failover_successor_robot = str(
            self.get_parameter('failover_successor_robot').value
        ).strip('/')
        self.required_liveness_sources: Set[str] = {
            str(source).strip()
            for source in self.get_parameter('scout_liveness_required_sources').value
            if str(source).strip()
        }
        self.liveness_timeout_sec = {
            'signal': float(self.get_parameter('scout_signal_timeout_sec').value),
            'map_pose': float(self.get_parameter('scout_map_pose_timeout_sec').value),
            'odom': float(self.get_parameter('scout_odom_timeout_sec').value),
        }
        self.scout_down_grace_sec = float(
            self.get_parameter('scout_down_grace_sec').value
        )
        self.last_scout_pose_max_age_sec = float(
            self.get_parameter('last_scout_pose_max_age_sec').value
        )
        self.failover_standoff_m = max(
            0.0, float(self.get_parameter('failover_standoff_m').value)
        )
        self.max_recovery_goal_retries = max(
            1, int(self.get_parameter('max_recovery_goal_retries').value)
        )
        self.recovery_goal_retry_period_sec = max(
            0.1, float(self.get_parameter('recovery_goal_retry_period_sec').value)
        )
        self.recovery_goal_timeout_sec = max(
            1.0, float(self.get_parameter('recovery_goal_timeout_sec').value)
        )
        self.map_topic = str(self.get_parameter('map_topic').value)

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
        self.mode_pub = self.create_publisher(String, '/multi/mode', 10)
        self.event_pub = self.create_publisher(String, '/multi/rescue_event', 10)
        self.status_pub = self.create_publisher(String, '/multi/robot_status', 10)
        self.marker_pub = self.create_publisher(MarkerArray, '/multi/rescue_markers', 10)
        # Single centralized publisher for all robot markers.  This avoids RViz
        # flickering caused by three controller nodes publishing partial marker
        # arrays to the same topic.
        self.robot_marker_pub = self.create_publisher(MarkerArray, '/multi/robot_markers', 10)

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
        self.scout_liveness_time: Dict[str, Time] = {}
        self.scout_last_global_pose: Optional[Pose2D] = None
        self.scout_last_global_pose_time: Optional[Time] = None
        self.failover_state = 'FOLLOWING'
        self.scout_suspected_since: Optional[Time] = None
        self.death_pose: Optional[Pose2D] = None
        self.recovery_target: Optional[Point2D] = None
        self.recovery_goal_retries = 0
        self.recovery_goal_started_time: Optional[Time] = None
        self.last_recovery_goal_time: Optional[Time] = None
        self.promoted_scout: Optional[str] = None
        self.last_valid_map_time: Optional[Time] = None
        self.last_valid_map_size: Optional[Tuple[int, int]] = None

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
            self.create_subscription(
                String,
                f'/{robot}/signal',
                lambda msg, r=robot: self.on_robot_signal(r, msg),
                10,
            )

        map_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.create_subscription(OccupancyGrid, self.map_topic, self.on_map, map_qos)

        self.create_subscription(String, '/multi/fail_robot', self.on_fail_robot, 10)
        self.create_subscription(
            String,
            '/multi/robot_failure_report',
            self.on_failure_report,
            10,
        )
        self.create_subscription(String, '/multi/recover_robot', self.on_recover_robot, 10)
        self.create_timer(float(self.get_parameter('timer_period').value), self.on_timer)

        self.get_logger().info(
            'AutoPatrolRescue started. '
            f'enable_patrol={self.enable_patrol}, '
            f'patrol_robots={sorted(self.patrol_robots)}, '
            f'timeout_detection={self.enable_timeout_detection}, '
            f'scout_failover={self.enable_scout_failover}, '
            f'scout={self.scout_robot}, successor={self.failover_successor_robot}, '
            f'required_liveness={sorted(self.required_liveness_sources)}. '
            'Manual test: ros2 topic pub --once /multi/fail_robot '
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
        if robot == self.scout_robot and source == 'odom':
            self.update_scout_liveness('odom')

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
        frame_id = msg.header.frame_id or 'map'
        pose = (
            msg.pose.position.x,
            msg.pose.position.y,
            yaw_from_quaternion(q.x, q.y, q.z, q.w),
        )
        self.update_pose(
            robot,
            pose,
            'map_pose',
        )
        if robot == self.scout_robot:
            self.update_scout_liveness('map_pose')
            if frame_id == 'map':
                self.scout_last_global_pose = pose
                self.scout_last_global_pose_time = self.now()
            else:
                self.get_logger().warn(
                    f'Ignoring scout pose for failover cache: expected frame=map, '
                    f'got frame={frame_id}'
                )

    def on_robot_signal(self, robot: str, msg: String) -> None:
        del msg
        if robot == self.scout_robot:
            self.update_scout_liveness('signal')

    def on_map(self, msg: OccupancyGrid) -> None:
        if not self.valid_map(msg):
            self.get_logger().warn(
                'Ignoring invalid map sample for failover diagnostics '
                f'width={msg.info.width} height={msg.info.height} '
                f'resolution={msg.info.resolution:.3f} data={len(msg.data)}'
            )
            return
        self.last_valid_map_time = self.now()
        self.last_valid_map_size = (int(msg.info.width), int(msg.info.height))

    def valid_map(self, msg: OccupancyGrid) -> bool:
        width = int(msg.info.width)
        height = int(msg.info.height)
        return (
            width > 0
            and height > 0
            and float(msg.info.resolution) > 0.0
            and len(msg.data) == width * height
        )

    def update_scout_liveness(self, source: str) -> None:
        self.scout_liveness_time[source] = self.now()

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
            if robot == self.scout_robot:
                self.reset_failover_state()
            self.publish_event(f'{robot} recovered. Patrol mode will resume.')

    def clear_failures(self, reason: str) -> None:
        self.failed_robots.clear()
        self.failure_position.clear()
        self.failure_reason.clear()
        self.rescue_assignments.clear()
        for robot in self.robots:
            self.last_goal[robot] = None
        self.reset_failover_state()
        self.publish_event(f'All failures cleared: {reason}')

    def reset_failover_state(self) -> None:
        self.failover_state = 'FOLLOWING'
        self.scout_suspected_since = None
        self.death_pose = None
        self.recovery_target = None
        self.recovery_goal_retries = 0
        self.recovery_goal_started_time = None
        self.last_recovery_goal_time = None
        self.promoted_scout = None
        self.patrol_robots = set(self.initial_patrol_robots)

    def mark_failed(self, robot: str, reason: str) -> None:
        if robot in self.failed_robots:
            return
        if robot == self.scout_robot and self.death_pose is not None:
            fail_point = (self.death_pose[0], self.death_pose[1])
            source = 'frozen_scout_map_pose'
        elif robot == self.scout_robot and self.scout_last_global_pose is not None:
            fail_point = (self.scout_last_global_pose[0], self.scout_last_global_pose[1])
            source = 'last_scout_map_pose'
        elif robot in self.pose:
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
        if robot != self.scout_robot or self.failover_state not in ('FAILOVER_COMMITTED', 'NAVIGATING_TO_LAST_SCOUT_POSE'):
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
        self.update_scout_failover()
        if self.failover_state == 'SCOUT_ACTIVE':
            self.publish_mode('SCOUT_ACTIVE')
            self.patrol_loop()
        elif self.failover_state in (
            'FAILOVER_COMMITTED',
            'NAVIGATING_TO_LAST_SCOUT_POSE',
            'FAILOVER_RECOVERY',
        ):
            self.publish_mode(self.failover_state)
            self.failover_navigation_loop()
        elif self.failed_robots:
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

    def update_scout_failover(self) -> None:
        if not self.enable_scout_failover:
            return
        if self.failover_state in (
            'FAILOVER_COMMITTED',
            'NAVIGATING_TO_LAST_SCOUT_POSE',
            'PROMOTING_TO_SCOUT',
            'SCOUT_ACTIVE',
            'FAILOVER_RECOVERY',
        ):
            return
        if self.seconds_since(self.start_time) < self.startup_grace_sec:
            return
        if self.scout_robot not in self.robots:
            return
        if self.failover_successor_robot not in self.robots:
            return

        all_stale, ages = self.all_required_scout_topics_stale()
        if not all_stale:
            if self.failover_state == 'SCOUT_SUSPECTED_DOWN':
                self.publish_event(
                    'SCOUT_HEALTH | HEALTHY again before grace expired '
                    f'ages={self.format_ages(ages)}'
                )
            self.failover_state = 'FOLLOWING'
            self.scout_suspected_since = None
            return

        now = self.now()
        if self.failover_state != 'SCOUT_SUSPECTED_DOWN':
            self.failover_state = 'SCOUT_SUSPECTED_DOWN'
            self.scout_suspected_since = now
            self.publish_event(
                'SCOUT_HEALTH | SUSPECTED_DOWN '
                f'scout={self.scout_robot} ages={self.format_ages(ages)}'
            )
            return

        assert self.scout_suspected_since is not None
        if self.seconds_since(self.scout_suspected_since) < self.scout_down_grace_sec:
            return

        self.commit_scout_failover(ages)

    def all_required_scout_topics_stale(self) -> Tuple[bool, Dict[str, Optional[float]]]:
        ages: Dict[str, Optional[float]] = {}
        if not self.required_liveness_sources:
            return False, ages
        for source in sorted(self.required_liveness_sources):
            stamp = self.scout_liveness_time.get(source)
            if stamp is None:
                ages[source] = None
                continue
            ages[source] = self.seconds_since(stamp)
        all_stale = True
        for source in self.required_liveness_sources:
            age = ages.get(source)
            timeout = self.liveness_timeout_sec.get(source, self.signal_timeout_sec)
            if age is not None and age <= timeout:
                all_stale = False
                break
        return all_stale, ages

    def format_ages(self, ages: Dict[str, Optional[float]]) -> str:
        parts = []
        for source in sorted(self.required_liveness_sources):
            age = ages.get(source)
            text = 'never' if age is None else f'{age:.1f}s'
            timeout = self.liveness_timeout_sec.get(source, self.signal_timeout_sec)
            parts.append(f'{source}={text}/{timeout:.1f}s')
        return ','.join(parts)

    def commit_scout_failover(self, ages: Dict[str, Optional[float]]) -> None:
        if self.failover_state in (
            'FAILOVER_COMMITTED',
            'NAVIGATING_TO_LAST_SCOUT_POSE',
            'SCOUT_ACTIVE',
        ):
            return
        if self.scout_last_global_pose is None or self.scout_last_global_pose_time is None:
            self.failover_state = 'FAILOVER_RECOVERY'
            self.publish_event(
                'SCOUT_FAILOVER | ABORTED no valid scout global map pose cached '
                f'ages={self.format_ages(ages)}'
            )
            return
        pose_age = self.seconds_since(self.scout_last_global_pose_time)
        if pose_age > self.last_scout_pose_max_age_sec:
            self.failover_state = 'FAILOVER_RECOVERY'
            self.publish_event(
                'SCOUT_FAILOVER | ABORTED stale scout global pose '
                f'pose_age={pose_age:.1f}s max={self.last_scout_pose_max_age_sec:.1f}s'
            )
            return

        self.death_pose = self.scout_last_global_pose
        self.recovery_target = self.safe_recovery_target(self.death_pose)
        self.failover_state = 'FAILOVER_COMMITTED'
        self.recovery_goal_retries = 0
        self.recovery_goal_started_time = None
        self.last_recovery_goal_time = None
        self.rescue_assignments.clear()
        self.mark_failed(self.scout_robot, 'aggregate liveness timeout')
        self.publish_event(
            'SCOUT_FAILOVER | CONFIRMED_DOWN '
            f'scout={self.scout_robot} successor={self.failover_successor_robot} '
            f'death=({self.death_pose[0]:.2f},{self.death_pose[1]:.2f},yaw={self.death_pose[2]:.2f}) '
            f'target=({self.recovery_target[0]:.2f},{self.recovery_target[1]:.2f}) '
            f'pose_age={pose_age:.1f}s ages={self.format_ages(ages)}'
        )

    def safe_recovery_target(self, death_pose: Pose2D) -> Point2D:
        x, y, yaw = death_pose
        if self.failover_standoff_m <= 0.0:
            return (x, y)
        return (
            x - self.failover_standoff_m * math.cos(yaw),
            y - self.failover_standoff_m * math.sin(yaw),
        )

    def patrol_loop(self) -> None:
        # Only patrol_robots receive normal waypoint goals.
        # Non-patrol robots remain stopped and wait for a rescue_goal.
        for robot in self.robots:
            if robot in self.failed_robots:
                continue
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

    def failover_navigation_loop(self) -> None:
        if self.failover_state == 'FAILOVER_RECOVERY':
            return
        if self.recovery_target is None:
            self.failover_state = 'FAILOVER_RECOVERY'
            self.publish_event('SCOUT_FAILOVER | RECOVERY_FAILED missing recovery target')
            return
        successor = self.failover_successor_robot
        if successor not in self.pose:
            return

        current = (self.pose[successor][0], self.pose[successor][1])
        if distance(current, self.recovery_target) <= self.goal_tolerance:
            self.promote_successor_to_scout()
            return

        now = self.now()
        should_send = self.last_recovery_goal_time is None
        if self.last_recovery_goal_time is not None:
            should_send = (
                self.seconds_since(self.last_recovery_goal_time)
                >= self.recovery_goal_retry_period_sec
            )
        if not should_send:
            return
        if self.recovery_goal_retries >= self.max_recovery_goal_retries:
            if (
                self.recovery_goal_started_time is not None
                and self.seconds_since(self.recovery_goal_started_time)
                < self.recovery_goal_timeout_sec
            ):
                return
            self.failover_state = 'FAILOVER_RECOVERY'
            self.publish_event(
                'SCOUT_FAILOVER | RECOVERY_FAILED recovery goal timeout '
                f'successor={successor} target=({self.recovery_target[0]:.2f},'
                f'{self.recovery_target[1]:.2f}) retries={self.recovery_goal_retries} '
                f'timeout={self.recovery_goal_timeout_sec:.1f}s'
            )
            return

        self.failover_state = 'NAVIGATING_TO_LAST_SCOUT_POSE'
        self.rescue_assignments[successor] = self.recovery_target
        self.publish_goal(successor, self.recovery_target, rescue=True, force=True)
        self.recovery_goal_retries += 1
        if self.recovery_goal_started_time is None:
            self.recovery_goal_started_time = now
        self.last_recovery_goal_time = now
        self.publish_event(
            'SCOUT_FAILOVER | RECOVERY_GOAL_SENT '
            f'successor={successor} target=({self.recovery_target[0]:.2f},'
            f'{self.recovery_target[1]:.2f}) attempt={self.recovery_goal_retries}/'
            f'{self.max_recovery_goal_retries}'
        )

    def promote_successor_to_scout(self) -> None:
        successor = self.failover_successor_robot
        if self.promoted_scout == successor:
            return
        self.failover_state = 'PROMOTING_TO_SCOUT'
        self.promoted_scout = successor
        self.patrol_robots.discard(self.scout_robot)
        self.patrol_robots.add(successor)
        self.rescue_assignments.pop(successor, None)
        self.last_goal[successor] = None
        self.last_goal_publish_time[successor] = None
        self.waypoint_index[successor] = -1
        self.failover_state = 'SCOUT_ACTIVE'
        self.publish_event(
            'SCOUT_FAILOVER | ROLE_PROMOTED '
            f'{successor} logical_role=scout domain_unchanged=true '
            'follow_disabled=true scout_behavior=patrol'
        )

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
        data['_failover'] = {
            'state': self.failover_state,
            'scout': self.scout_robot,
            'successor': self.failover_successor_robot,
            'promoted_scout': self.promoted_scout,
            'death_pose': self.death_pose,
            'recovery_target': self.recovery_target,
            'last_valid_map_size': self.last_valid_map_size,
        }
        msg = String()
        msg.data = json.dumps(data, ensure_ascii=False)
        self.status_pub.publish(msg)

    def state_for(self, robot: str) -> str:
        if self.promoted_scout == robot and self.failover_state == 'SCOUT_ACTIVE':
            return 'SCOUT_ACTIVE'
        if (
            robot == self.failover_successor_robot
            and self.failover_state == 'NAVIGATING_TO_LAST_SCOUT_POSE'
        ):
            return 'TAKING_OVER_SCOUT'
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
        if self.promoted_scout == robot and self.failover_state == 'SCOUT_ACTIVE':
            return 'SCOUT_ACTIVE'
        if (
            robot == self.failover_successor_robot
            and self.failover_state == 'NAVIGATING_TO_LAST_SCOUT_POSE'
        ):
            return 'TAKEOVER'
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
