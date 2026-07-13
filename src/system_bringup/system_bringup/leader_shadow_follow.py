#!/usr/bin/env python3
"""Low-speed leader shadow follow for the active scout.

The leader remains a leader: it does not copy the scout pose and it stops
issuing shadow goals as soon as failover owns recovery.  During normal
operation it estimates the active scout's movement heading, creates a rear
standoff target, validates it against the shared map, and publishes rate-limited
leader Nav2 goals.
"""

from __future__ import annotations

from copy import deepcopy
import json
import math
from enum import Enum
from typing import Optional, Tuple

import rclpy
from action_msgs.msg import GoalStatus
from geometry_msgs.msg import PointStamped, PoseStamped, Twist, TwistStamped
from nav2_msgs.action import NavigateToPose
from nav_msgs.msg import OccupancyGrid, Odometry, Path
from rcl_interfaces.msg import Parameter, ParameterType, ParameterValue
from rcl_interfaces.srv import SetParameters
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Bool, String


Point2 = Tuple[float, float]


class LeaderMode(Enum):
    IDLE = 'LEADER_IDLE'
    SHADOW_FOLLOW = 'LEADER_SHADOW_FOLLOW'
    SCOUT_SUSPECTED_DEAD = 'LEADER_SCOUT_SUSPECTED_DEAD'
    RECOVERY_NAVIGATING = 'LEADER_RECOVERY_NAVIGATING'
    RECOVERY_POSITION_REACHED = 'LEADER_RECOVERY_POSITION_REACHED'
    WAIT_NEW_SCOUT = 'LEADER_WAIT_NEW_SCOUT'
    SHADOW_NEW_SCOUT = 'LEADER_SHADOW_NEW_SCOUT'


def yaw_from_quaternion(q) -> float:
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


def quaternion_from_yaw(yaw: float) -> tuple[float, float, float, float]:
    return (0.0, 0.0, math.sin(0.5 * yaw), math.cos(0.5 * yaw))


def angle_lerp(current: float, target: float, alpha: float) -> float:
    diff = math.atan2(math.sin(target - current), math.cos(target - current))
    return current + alpha * diff


class LeaderShadowFollow(Node):
    def __init__(self) -> None:
        super().__init__('leader_shadow_follow')

        self.declare_parameter('enable_leader_shadow_follow', True)
        self.declare_parameter('leader_pose_topic', '/leader_pose')
        self.declare_parameter('active_scout_pose_topic', '/member_pose')
        self.declare_parameter('follower_scout_pose_topic', '/burger_pose')
        self.declare_parameter('leader_goal_topic', '/fleet/leader_coord_goal')
        self.declare_parameter('navigate_action', '/navigate_to_pose')
        self.declare_parameter('leader_cancel_topic', '/fleet/leader_nav_cancel')
        self.declare_parameter('cmd_vel_topic', '/cmd_vel')
        self.declare_parameter('use_stamped_cmd_vel', True)
        self.declare_parameter('direct_shadow_cmd_vel', False)
        self.declare_parameter('leader_follow_backend', 'nav2')
        self.declare_parameter('leader_path_topic', '/plan')
        self.declare_parameter('odom_topic', '/odom')
        self.declare_parameter(
            'controller_set_parameters_service',
            '/controller_server/set_parameters',
        )
        self.declare_parameter('map_topic', '/map')
        self.declare_parameter('risk_topic', '/risk/risk_map')
        self.declare_parameter('enable_risk_priority_follow', True)
        self.declare_parameter('risk_min_value', 1)
        self.declare_parameter('risk_pose_timeout_sec', 10.0)
        self.declare_parameter('failover_state_topic', '/failover/state')
        self.declare_parameter('active_scout_id_topic', '/failover/active_scout_id')
        self.declare_parameter('active_scout_robot_name', 'scout22')
        self.declare_parameter('follower_robot_name', 'follower21')
        self.declare_parameter('require_localization_ready', True)
        self.declare_parameter('localization_ready_topic', '/localization_ready')
        self.declare_parameter('require_system_ready', False)
        self.declare_parameter('system_ready_topic', '/system/ready')
        self.declare_parameter('require_video_ready', True)
        self.declare_parameter('video_ready_topic', '/fleet/video_ready')
        self.declare_parameter('target_detected_topic', '/omx/target_detected')
        self.declare_parameter('target_detected_stop_hold_sec', 3.0)
        self.declare_parameter('target_detected_cancel_period_sec', 0.25)
        self.declare_parameter('target_memory_hold_sec', 3.0)
        self.declare_parameter('target_reacquire_publish_period_sec', 0.5)
        self.declare_parameter('target_processed_topic', '/omx/target_processed')
        self.declare_parameter('target_lost_topic', '/omx/target_lost')
        self.declare_parameter('target_reacquire_topic', '/omx/target_in_map')
        self.declare_parameter('pause_on_raw_target_detection', True)
        self.declare_parameter('omx_state_topic', '/omx/state')
        self.declare_parameter('pause_on_omx_aiming', True)
        self.declare_parameter('scout_pose_timeout_sec', 0.5)
        self.declare_parameter('startup_grace_sec', 8.0)

        self.declare_parameter('leader_shadow_follow_distance_m', 0.40)
        self.declare_parameter('leader_shadow_stop_distance_m', 0.30)
        self.declare_parameter('leader_shadow_resume_distance_m', 0.46)
        self.declare_parameter('leader_shadow_far_distance_m', 0.80)
        self.declare_parameter('leader_shadow_max_linear_vel', 0.26)
        self.declare_parameter('leader_shadow_catchup_max_linear_vel', 0.26)
        self.declare_parameter('leader_shadow_max_angular_vel', 1.00)
        self.declare_parameter('leader_restore_max_linear_vel', 0.26)
        self.declare_parameter('leader_restore_max_angular_vel', 1.00)
        self.declare_parameter('leader_shadow_goal_update_period_sec', 0.5)
        self.declare_parameter('leader_shadow_goal_min_change_m', 0.12)
        self.declare_parameter('leader_shadow_nav_execution_timeout_sec', 2.0)
        self.declare_parameter('leader_shadow_cmd_goal_tolerance_m', 0.16)
        self.declare_parameter('leader_shadow_cmd_linear_scale', 1.0)
        self.declare_parameter('leader_shadow_cmd_angular_scale', 1.0)
        self.declare_parameter('leader_shadow_cmd_max_linear_vel', 0.26)
        self.declare_parameter('leader_shadow_cmd_max_angular_vel', 1.00)
        self.declare_parameter('leader_shadow_linear_kp', 0.70)
        self.declare_parameter('leader_shadow_angular_kp', 1.40)
        self.declare_parameter('leader_shadow_heading_slowdown_rad', 0.75)
        self.declare_parameter('leader_shadow_heading_min_motion_m', 0.15)
        self.declare_parameter('leader_shadow_heading_alpha', 0.35)
        self.declare_parameter('leader_shadow_map_clearance_m', 0.22)
        self.declare_parameter('leader_shadow_target_search_radius_m', 1.2)
        self.declare_parameter('leader_shadow_target_search_step_m', 0.15)
        self.declare_parameter('occupied_threshold', 50)
        self.declare_parameter('allow_unknown_shadow_target', False)

        self.declare_parameter('enable_leader_continuous_scan', True)
        self.declare_parameter('leader_scan_topic', '/scan')
        self.declare_parameter('leader_scan_fov_deg', 60.0)
        self.declare_parameter('leader_scan_update_rate_hz', 10.0)
        self.declare_parameter('leader_scan_timeout_sec', 1.0)

        get = self.get_parameter
        self.enabled = bool(get('enable_leader_shadow_follow').value)
        self.leader_pose_topic = str(get('leader_pose_topic').value)
        self.active_pose_topic = str(get('active_scout_pose_topic').value)
        self.follower_pose_topic = str(get('follower_scout_pose_topic').value)
        self.leader_goal_topic = str(get('leader_goal_topic').value)
        self.navigate_action = str(get('navigate_action').value).strip() or '/navigate_to_pose'
        self.leader_cancel_topic = str(get('leader_cancel_topic').value)
        self.cmd_vel_topic = str(get('cmd_vel_topic').value)
        self.use_stamped_cmd_vel = bool(get('use_stamped_cmd_vel').value)
        self.direct_shadow_cmd_vel = bool(get('direct_shadow_cmd_vel').value)
        self.follow_backend = str(get('leader_follow_backend').value).strip().lower()
        if self.follow_backend not in ('nav2', 'direct'):
            self.get_logger().warning(
                f'LEADER_FOLLOW_BACKEND_INVALID | backend={self.follow_backend!r}; using nav2'
            )
            self.follow_backend = 'nav2'
        self.direct_shadow_cmd_vel = self.follow_backend == 'direct'
        self.leader_path_topic = str(get('leader_path_topic').value)
        self.odom_topic = str(get('odom_topic').value)
        self.controller_set_parameters_service = str(
            get('controller_set_parameters_service').value
        )
        self.map_topic = str(get('map_topic').value)
        self.risk_topic = str(get('risk_topic').value)
        self.enable_risk_priority = bool(get('enable_risk_priority_follow').value)
        self.risk_min_value = max(0, min(100, int(get('risk_min_value').value)))
        self.risk_pose_timeout = max(0.2, float(get('risk_pose_timeout_sec').value))
        self.failover_state_topic = str(get('failover_state_topic').value)
        self.active_scout_id_topic = str(get('active_scout_id_topic').value)
        self.original_scout_id = str(get('active_scout_robot_name').value)
        self.follower_robot_name = str(get('follower_robot_name').value)
        self.require_localization_ready = bool(get('require_localization_ready').value)
        self.localization_ready_topic = str(get('localization_ready_topic').value)
        self.require_system_ready = bool(get('require_system_ready').value)
        self.system_ready_topic = str(get('system_ready_topic').value)
        self.require_video_ready = bool(get('require_video_ready').value)
        self.video_ready_topic = str(get('video_ready_topic').value)
        self.target_detected_topic = str(get('target_detected_topic').value)
        self.target_stop_hold = max(
            0.1, float(get('target_detected_stop_hold_sec').value)
        )
        self.target_cancel_period = max(
            0.05, float(get('target_detected_cancel_period_sec').value)
        )
        self.target_memory_hold = max(0.0, float(get('target_memory_hold_sec').value))
        self.target_reacquire_period = max(
            0.1, float(get('target_reacquire_publish_period_sec').value)
        )
        self.target_processed_topic = str(get('target_processed_topic').value)
        self.target_lost_topic = str(get('target_lost_topic').value)
        self.target_reacquire_topic = str(get('target_reacquire_topic').value)
        self.pause_on_raw_target_detection = bool(
            get('pause_on_raw_target_detection').value
        )
        self.omx_state_topic = str(get('omx_state_topic').value)
        self.pause_on_omx_aiming = bool(get('pause_on_omx_aiming').value)
        self.scout_pose_timeout = max(0.2, float(get('scout_pose_timeout_sec').value))
        self.startup_grace = max(0.0, float(get('startup_grace_sec').value))

        self.follow_distance = max(0.1, float(get('leader_shadow_follow_distance_m').value))
        self.stop_distance = max(0.2, float(get('leader_shadow_stop_distance_m').value))
        self.resume_distance = max(self.stop_distance, float(get('leader_shadow_resume_distance_m').value))
        self.far_distance = max(self.resume_distance, float(get('leader_shadow_far_distance_m').value))
        self.shadow_linear_vel = max(0.03, float(get('leader_shadow_max_linear_vel').value))
        self.catchup_linear_vel = max(
            self.shadow_linear_vel,
            float(get('leader_shadow_catchup_max_linear_vel').value),
        )
        self.shadow_angular_vel = max(0.05, float(get('leader_shadow_max_angular_vel').value))
        self.restore_linear_vel = max(self.shadow_linear_vel, float(get('leader_restore_max_linear_vel').value))
        self.restore_angular_vel = max(self.shadow_angular_vel, float(get('leader_restore_max_angular_vel').value))
        self.goal_period = max(0.3, float(get('leader_shadow_goal_update_period_sec').value))
        self.goal_min_change = max(0.05, float(get('leader_shadow_goal_min_change_m').value))
        self.nav_execution_timeout = max(
            0.5, float(get('leader_shadow_nav_execution_timeout_sec').value)
        )
        self.cmd_goal_tolerance = max(0.03, float(get('leader_shadow_cmd_goal_tolerance_m').value))
        self.cmd_linear_scale = max(0.1, float(get('leader_shadow_cmd_linear_scale').value))
        self.cmd_angular_scale = max(0.1, float(get('leader_shadow_cmd_angular_scale').value))
        self.cmd_max_linear_vel = max(self.shadow_linear_vel, float(
            get('leader_shadow_cmd_max_linear_vel').value
        ))
        self.cmd_max_angular_vel = max(self.shadow_angular_vel, float(
            get('leader_shadow_cmd_max_angular_vel').value
        ))
        self.linear_kp = max(0.01, float(get('leader_shadow_linear_kp').value))
        self.angular_kp = max(0.01, float(get('leader_shadow_angular_kp').value))
        self.heading_slowdown_rad = max(
            0.05, float(get('leader_shadow_heading_slowdown_rad').value)
        )
        self.heading_min_motion = max(0.02, float(get('leader_shadow_heading_min_motion_m').value))
        self.heading_alpha = min(1.0, max(0.01, float(get('leader_shadow_heading_alpha').value)))
        self.map_clearance = max(0.05, float(get('leader_shadow_map_clearance_m').value))
        self.search_radius = max(0.0, float(get('leader_shadow_target_search_radius_m').value))
        self.search_step = max(0.05, float(get('leader_shadow_target_search_step_m').value))
        self.occupied_threshold = int(get('occupied_threshold').value)
        self.allow_unknown = bool(get('allow_unknown_shadow_target').value)

        self.scan_enabled = bool(get('enable_leader_continuous_scan').value)
        self.scan_topic = str(get('leader_scan_topic').value)
        self.scan_fov_deg = max(1.0, min(180.0, float(get('leader_scan_fov_deg').value)))
        self.scan_rate = max(0.5, float(get('leader_scan_update_rate_hz').value))
        self.scan_timeout = max(0.1, float(get('leader_scan_timeout_sec').value))

        latched_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
        )
        map_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
        )

        self.cancel_pub = self.create_publisher(Bool, self.leader_cancel_topic, latched_qos)
        self.nav_client = ActionClient(self, NavigateToPose, self.navigate_action)
        if self.use_stamped_cmd_vel:
            self.cmd_pub = self.create_publisher(TwistStamped, self.cmd_vel_topic, 10)
        else:
            self.cmd_pub = self.create_publisher(Twist, self.cmd_vel_topic, 10)
        self.state_pub = self.create_publisher(String, '/leader_shadow/state', latched_qos)
        self.goal_debug_pub = self.create_publisher(PoseStamped, '/leader_shadow/goal', 10)
        self.scan_state_pub = self.create_publisher(String, '/leader_scan/state', latched_qos)
        self.target_reacquire_pub = self.create_publisher(
            PointStamped,
            self.target_reacquire_topic,
            10,
        )

        self.create_subscription(PoseStamped, self.leader_pose_topic, self._on_leader_pose, 10)
        self.create_subscription(PoseStamped, self.active_pose_topic, self._on_original_scout_pose, 20)
        self.create_subscription(PoseStamped, self.follower_pose_topic, self._on_follower_scout_pose, 20)
        self.create_subscription(Path, self.leader_path_topic, self._on_path, 10)
        self.create_subscription(Odometry, self.odom_topic, self._on_odom, 10)
        if self.use_stamped_cmd_vel:
            self.create_subscription(TwistStamped, self.cmd_vel_topic, self._on_cmd_vel_stamped, 10)
        else:
            self.create_subscription(Twist, self.cmd_vel_topic, self._on_cmd_vel, 10)
        self.create_subscription(OccupancyGrid, self.map_topic, self._on_map, map_qos)
        self.create_subscription(OccupancyGrid, self.risk_topic, self._on_risk_map, map_qos)
        self.create_subscription(String, self.failover_state_topic, self._on_failover_state, latched_qos)
        self.create_subscription(String, self.active_scout_id_topic, self._on_active_scout_id, latched_qos)
        if self.require_localization_ready:
            self.create_subscription(
                Bool,
                self.localization_ready_topic,
                self._on_localization_ready,
                latched_qos,
            )
        if self.require_system_ready:
            self.create_subscription(
                Bool,
                self.system_ready_topic,
                self._on_system_ready,
                latched_qos,
            )
        if self.require_video_ready:
            self.create_subscription(
                Bool,
                self.video_ready_topic,
                self._on_video_ready,
                latched_qos,
            )
        detected_qos = QoSProfile(
            depth=5,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
        )
        self.create_subscription(
            Bool,
            self.target_detected_topic,
            self._on_target_detected,
            detected_qos,
        )
        self.create_subscription(
            PointStamped,
            self.target_processed_topic,
            self._on_target_point,
            10,
        )
        self.create_subscription(
            PointStamped,
            self.target_lost_topic,
            self._on_target_lost,
            10,
        )
        self.create_subscription(String, self.omx_state_topic, self._on_omx_state, 10)
        if self.scan_enabled:
            scan_qos = QoSProfile(
                depth=5,
                reliability=ReliabilityPolicy.BEST_EFFORT,
                durability=DurabilityPolicy.VOLATILE,
                history=HistoryPolicy.KEEP_LAST,
            )
            self.create_subscription(LaserScan, self.scan_topic, self._on_scan, scan_qos)

        self.controller_client = self.create_client(
            SetParameters, self.controller_set_parameters_service
        )

        self.mode = LeaderMode.IDLE
        self.start_wall = self._now()
        self.active_scout_id = self.original_scout_id
        self.failover_state = 'NORMAL_OPERATION'
        self.localization_ready = not self.require_localization_ready
        self.system_ready = not self.require_system_ready
        self.video_ready = not self.require_video_ready
        self.target_detected = False
        self.target_detected_wall = -1.0e9
        self.target_last_seen_wall = -1.0e9
        self.last_target_point: Optional[PointStamped] = None
        self.last_target_point_wall = -1.0e9
        self.last_target_reacquire_wall = -1.0e9
        self.target_hold_active = False
        self.last_target_cancel_wall = -1.0e9
        self.omx_state = ''
        self.leader_pose: Optional[PoseStamped] = None
        self.leader_pose_wall = -1.0e9
        self.original_scout_pose: Optional[PoseStamped] = None
        self.follower_scout_pose: Optional[PoseStamped] = None
        self.original_scout_wall = -1.0e9
        self.follower_scout_wall = -1.0e9
        self.map_msg: Optional[OccupancyGrid] = None
        self.risk_msg: Optional[OccupancyGrid] = None
        self.risk_wall = -1.0e9
        self.last_risk_target: Optional[Point2] = None
        self.last_risk_value = -1
        self.last_scan_wall = -1.0e9
        self.last_scan_stamp = -1.0
        self.last_path_wall = -1.0e9
        self.last_cmd_vel_wall = -1.0e9
        self.last_nonzero_cmd_wall = -1.0e9
        self.last_odom_wall = -1.0e9
        self.odom_motion = False
        self.last_odom_xy: Optional[Point2] = None
        self.odom_delta_m = 0.0
        self.previous_distance_to_scout: Optional[float] = None
        self.distance_decreased = False

        self.heading: Optional[float] = None
        self.previous_scout_sample: Optional[Tuple[float, Point2]] = None
        # True from the start: the leader should shadow the scout right
        # away, not only once it has already wandered resume_distance_m
        # away. With this False, "shadow_active" could only ever flip on
        # via the resume-distance branch below, which never fires if the
        # scout starts anywhere closer than that (the common case indoors)
        # -- the leader would never begin following at all.
        self.shadow_active = True
        self.last_goal: Optional[PoseStamped] = None
        self.shadow_goal_active = False
        self.direct_cmd_active = False
        self.nav_goal_pending = False
        self.nav_goal_handle = None
        self.nav_goal_reason = ''
        self.last_goal_wall = -1.0e9
        self.last_nominal_target: Optional[Point2] = None
        self.last_target_mode = 'none'
        self.last_target_behind_scout = False
        self.last_target_free = False
        self.speed_profile: Optional[str] = None
        self.speed_limit_pending = False

        self.create_timer(0.2, self._tick)
        self.create_timer(1.0 / self.scan_rate, self._scan_tick)
        self._publish_cancel(False)
        self._publish_state('startup')
        self.get_logger().warning(
            '[LEADER_SHADOW] READY | '
            f'enabled={self.enabled} scout={self.original_scout_id}:{self.active_pose_topic} '
            f'follower_scout={self.follower_robot_name}:{self.follower_pose_topic} '
            f'distance={self.follow_distance:.2f}m fov={self.scan_fov_deg:.1f}deg '
            f'backend={self.follow_backend} direct_cmd={self.direct_shadow_cmd_vel}:{self.cmd_vel_topic} '
            f'navigate_action={self.navigate_action} '
            f'cmd_scale=lin{self.cmd_linear_scale:.2f}/ang{self.cmd_angular_scale:.2f} '
            f'cmd_cap=lin{self.cmd_max_linear_vel:.2f}/ang{self.cmd_max_angular_vel:.2f} '
            f'controller_service={self.controller_set_parameters_service} '
            f'localization_gate={self.require_localization_ready}:{self.localization_ready_topic} '
            f'system_gate={self.require_system_ready}:{self.system_ready_topic} '
            f'video_gate={self.require_video_ready}:{self.video_ready_topic}'
        )

    def _now(self) -> float:
        return self.get_clock().now().nanoseconds * 1.0e-9

    def _on_leader_pose(self, msg: PoseStamped) -> None:
        self.leader_pose = msg
        self.leader_pose_wall = self._now()
        self._log_pose_pipeline('leader', self.leader_pose_topic, msg, self.leader_pose_wall)

    def _on_original_scout_pose(self, msg: PoseStamped) -> None:
        self.original_scout_pose = msg
        self.original_scout_wall = self._now()
        self._log_pose_pipeline('scout', self.active_pose_topic, msg, self.original_scout_wall)
        if self.active_scout_id == self.original_scout_id:
            self._update_heading_from_pose(msg)

    def _on_follower_scout_pose(self, msg: PoseStamped) -> None:
        self.follower_scout_pose = msg
        self.follower_scout_wall = self._now()
        self._log_pose_pipeline('follower_scout', self.follower_pose_topic, msg, self.follower_scout_wall)
        if self.active_scout_id == self.follower_robot_name:
            self._update_heading_from_pose(msg)

    def _on_map(self, msg: OccupancyGrid) -> None:
        self.map_msg = msg

    def _on_risk_map(self, msg: OccupancyGrid) -> None:
        if int(msg.info.width) <= 0 or int(msg.info.height) <= 0:
            return
        if len(msg.data) != int(msg.info.width) * int(msg.info.height):
            return
        self.risk_msg = msg
        self.risk_wall = self._now()

    def _on_path(self, msg: Path) -> None:
        if msg.poses:
            self.last_path_wall = self._now()

    def _on_cmd_vel_stamped(self, msg: TwistStamped) -> None:
        self._record_cmd_vel(msg.twist)

    def _on_cmd_vel(self, msg: Twist) -> None:
        self._record_cmd_vel(msg)

    def _record_cmd_vel(self, twist: Twist) -> None:
        now = self._now()
        self.last_cmd_vel_wall = now
        if abs(float(twist.linear.x)) > 1.0e-3 or abs(float(twist.angular.z)) > 1.0e-3:
            self.last_nonzero_cmd_wall = now

    def _on_odom(self, msg: Odometry) -> None:
        self.last_odom_wall = self._now()
        xy = (
            float(msg.pose.pose.position.x),
            float(msg.pose.pose.position.y),
        )
        if self.last_odom_xy is not None:
            self.odom_delta_m = math.hypot(
                xy[0] - self.last_odom_xy[0],
                xy[1] - self.last_odom_xy[1],
            )
        self.last_odom_xy = xy
        self.odom_motion = bool(
            abs(float(msg.twist.twist.linear.x)) > 0.01
            or abs(float(msg.twist.twist.angular.z)) > 0.03
        )

    def _on_failover_state(self, msg: String) -> None:
        previous = self.failover_state
        self.failover_state = msg.data.strip() or 'UNKNOWN'
        if previous != self.failover_state and not self._failover_allows_shadow():
            self._cancel_shadow_goal(f'failover_{self.failover_state}')

    def _on_active_scout_id(self, msg: String) -> None:
        scout_id = msg.data.strip()
        if not scout_id or scout_id == self.active_scout_id:
            return
        self._cancel_shadow_goal('active_scout_changed')
        self.active_scout_id = scout_id
        self.previous_scout_sample = None
        self.heading = None
        self.shadow_active = False
        self.last_goal = None
        self.mode = LeaderMode.SHADOW_NEW_SCOUT
        self.get_logger().warning(
            f'[LEADER_SHADOW] ACTIVE_SCOUT_CHANGED | active_scout={scout_id}'
        )

    def _on_localization_ready(self, msg: Bool) -> None:
        previous = self.localization_ready
        self.localization_ready = bool(msg.data)
        if previous and not self.localization_ready:
            self._cancel_shadow_goal('localization_not_ready')
        if self.localization_ready and not previous:
            self.get_logger().warning(
                f'[LEADER_SHADOW] LOCALIZATION_READY | topic={self.localization_ready_topic}'
            )
            self.last_goal = None
            self.last_goal_wall = -1.0e9
            self._tick()

    def _on_system_ready(self, msg: Bool) -> None:
        previous = self.system_ready
        self.system_ready = bool(msg.data)
        if previous and not self.system_ready:
            self._cancel_shadow_goal('system_not_ready')
            self._stop_direct_cmd('system_not_ready')
        if self.system_ready != previous:
            self.get_logger().warning(
                f'[LEADER_SHADOW] SYSTEM_READY | ready={self.system_ready} topic={self.system_ready_topic}'
            )

    def _on_video_ready(self, msg: Bool) -> None:
        previous = self.video_ready
        self.video_ready = bool(msg.data)
        if previous and not self.video_ready:
            self._cancel_shadow_goal('video_not_ready')
            self._stop_direct_cmd('video_not_ready')
        if self.video_ready != previous:
            self.get_logger().warning(
                f'[LEADER_SHADOW] VIDEO_READY | ready={self.video_ready} topic={self.video_ready_topic}'
            )
        if self.video_ready and not previous:
            self.last_goal = None
            self.last_goal_wall = -1.0e9
            self._tick()

    def _on_target_detected(self, msg: Bool) -> None:
        self.target_detected = bool(msg.data)
        if self.target_detected:
            now = self._now()
            self.target_detected_wall = now
            self.target_last_seen_wall = now
            if self.pause_on_raw_target_detection:
                self._hold_for_omx_target('target_detected_signal')

    def _on_omx_state(self, msg: String) -> None:
        self.omx_state = str(msg.data).strip().upper()
        if self._is_omx_aiming(self.omx_state):
            self.target_last_seen_wall = self._now()

    def _on_target_point(self, msg: PointStamped) -> None:
        self.last_target_point = deepcopy(msg)
        self.last_target_point_wall = self._now()
        self.target_last_seen_wall = self.last_target_point_wall

    def _on_target_lost(self, msg: PointStamped) -> None:
        self.last_target_point = deepcopy(msg)
        self.last_target_point_wall = self._now()

    @staticmethod
    def _is_omx_aiming(state: str) -> bool:
        """OMX may keep aiming/firing; only the robot base must stay stopped."""
        return str(state).strip().upper() in (
            'TRACKING',
            'CONFIRMING',
            'FIRING',
            'COOLDOWN',
        )

    def _target_hold_reason(self) -> Optional[str]:
        now = self._now()
        if self.pause_on_raw_target_detection and self.target_detected:
            return 'target_detected'
        if (
            self.pause_on_raw_target_detection
            and now - self.target_detected_wall <= self.target_stop_hold
        ):
            return 'target_detected_hold'
        if self.pause_on_omx_aiming and self._is_omx_aiming(self.omx_state):
            return f'omx_{self.omx_state.lower()}'
        if (
            self.last_target_point is not None
            and self.target_memory_hold > 0.0
            and now - self.target_last_seen_wall <= self.target_memory_hold
        ):
            return 'target_reacquire_memory'
        return None

    def _hold_for_omx_target(self, reason: str) -> None:
        """Stop Nav2/base motion while leaving OMX PD tracking free to run."""
        self.target_hold_active = True
        self._cancel_shadow_goal(reason)
        self._stop_direct_cmd(reason)
        self._set_controller_speed_limit(False)
        self.shadow_active = False
        self.mode = LeaderMode.IDLE
        self._publish_remembered_target(reason)
        self._publish_state(reason)
        self._log_follow_debug(reason)
        self._publish_twist(0.0, 0.0)
        self.get_logger().warning(
            'LEADER_OMX_TARGET_HOLD | '
            'base_motion_stopped=true omx_pd_allowed=true '
            f'reason={reason} detected={self.target_detected} '
            f'omx_state={self.omx_state or "(empty)"} '
            f'has_memory={self.last_target_point is not None}',
            throttle_duration_sec=1.0,
        )

    def _publish_remembered_target(self, reason: str) -> None:
        if self.last_target_point is None:
            return
        now = self._now()
        if now - self.last_target_reacquire_wall < self.target_reacquire_period:
            return
        msg = deepcopy(self.last_target_point)
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = msg.header.frame_id or 'map'
        self.target_reacquire_pub.publish(msg)
        self.last_target_reacquire_wall = now
        self.get_logger().warning(
            'LEADER_OMX_REACQUIRE_TARGET | '
            f'reason={reason} topic={self.target_reacquire_topic} '
            f'x={msg.point.x:.3f} y={msg.point.y:.3f} z={msg.point.z:.3f}',
            throttle_duration_sec=1.0,
        )

    def _on_scan(self, msg: LaserScan) -> None:
        self.last_scan_wall = self._now()
        stamp = msg.header.stamp
        self.last_scan_stamp = float(stamp.sec) + float(stamp.nanosec) * 1.0e-9

    def _tick(self) -> None:
        if not self.enabled:
            self._cancel_shadow_goal('disabled')
            self._stop_direct_cmd('disabled')
            self._set_controller_speed_limit(False)
            self.mode = LeaderMode.IDLE
            self._publish_state('disabled')
            self._log_follow_debug('disabled')
            return
        target_reason = self._target_hold_reason()
        if target_reason is not None:
            self._hold_for_omx_target(target_reason)
            return
        if self.target_hold_active:
            self.target_hold_active = False
            self.last_goal = None
            self.last_goal_wall = -1.0e9
            self.get_logger().warning('LEADER_OMX_TARGET_RELEASE | nav2_resume=true')
        risk_goal = self._build_risk_goal()
        if risk_goal is not None:
            self.mode = LeaderMode.SHADOW_FOLLOW
            self._publish_nav2_goal(risk_goal, 'risk_goal_sent', catchup=True)
            return

        scout_pose, scout_wall = self._active_scout_pose()
        if scout_pose is None:
            self._cancel_shadow_goal('waiting_pose')
            self._stop_direct_cmd('waiting_pose')
            self.mode = LeaderMode.IDLE
            self._set_controller_speed_limit(False)
            self._publish_state('waiting_pose')
            self._log_follow_debug('waiting_pose')
            return
        pose_invalid_reason = self._pose_invalid_reason(scout_pose)
        if not pose_invalid_reason and self.leader_pose is not None:
            pose_invalid_reason = self._pose_pair_invalid_reason(self.leader_pose, scout_pose)
        if pose_invalid_reason:
            self._cancel_shadow_goal(f'pose_invalid_{pose_invalid_reason}')
            self._stop_direct_cmd(f'pose_invalid_{pose_invalid_reason}')
            self.mode = LeaderMode.IDLE
            self._set_controller_speed_limit(False)
            self._publish_state(f'pose_invalid_{pose_invalid_reason}')
            self._log_follow_debug(f'pose_invalid_{pose_invalid_reason}')
            return
        scout_age = self._now() - scout_wall
        if scout_age > self.scout_pose_timeout:
            self._cancel_shadow_goal('scout_pose_stale')
            self._stop_direct_cmd('scout_pose_stale')
            self.mode = LeaderMode.SCOUT_SUSPECTED_DEAD
            self._set_controller_speed_limit(False)
            self._publish_state(f'scout_pose_stale_{scout_age:.2f}s')
            self._log_follow_debug('scout_pose_stale')
            return

        self.mode = (
            LeaderMode.SHADOW_NEW_SCOUT
            if self.active_scout_id != self.original_scout_id
            else LeaderMode.SHADOW_FOLLOW
        )
        distance_to_scout = (
            self._distance_pose(self.leader_pose, scout_pose)
            if self.leader_pose is not None else float('inf')
        )
        previous_distance = getattr(self, 'previous_distance_to_scout', None)
        self.distance_decreased = (
            previous_distance is not None
            and distance_to_scout < previous_distance - 0.02
        )
        self.previous_distance_to_scout = distance_to_scout
        if self.leader_pose is not None and distance_to_scout <= self.stop_distance:
            self._cancel_shadow_goal('stopped_close_to_scout')
            self._stop_direct_cmd('stopped_close_to_scout')
            self.shadow_active = False
            self.mode = LeaderMode.IDLE
            self._set_controller_speed_limit(False)
            self._publish_state(
                'stopped_close_to_scout',
                distance_to_scout=distance_to_scout,
            )
            self._log_follow_debug(
                'stopped_close_to_scout',
                distance_to_scout=distance_to_scout,
            )
            return
        goal = self._build_shadow_goal(scout_pose)
        if goal is None:
            self._cancel_shadow_goal('no_feasible_shadow_target')
            self._stop_direct_cmd('no_feasible_shadow_target')
            self._publish_state('no_feasible_shadow_target')
            self._log_follow_debug('no_feasible_shadow_target')
            return
        distance_to_shadow_goal = (
            self._distance_pose(self.leader_pose, goal)
            if self.leader_pose is not None else float('inf')
        )
        self.shadow_active = distance_to_shadow_goal > self.cmd_goal_tolerance
        if not self.shadow_active:
            self._cancel_shadow_goal('at_shadow_target')
            self._stop_direct_cmd('at_shadow_target')
            self.mode = LeaderMode.IDLE
            self._set_controller_speed_limit(False)
            self._publish_state(
                'at_shadow_target',
                goal=goal,
                distance_to_scout=distance_to_scout,
            )
            self._log_follow_debug('at_shadow_target', goal=goal, distance_to_scout=distance_to_scout)
            return
        catchup = distance_to_scout >= self.far_distance
        self._publish_nav2_goal(
            goal,
            'scout_follow_goal_sent',
            catchup=catchup,
            distance_to_scout=distance_to_scout,
        )

    def _publish_nav2_goal(
        self,
        goal: PoseStamped,
        reason: str,
        *,
        catchup: bool,
        distance_to_scout: Optional[float] = None,
    ) -> None:
        target_reason = self._target_hold_reason()
        if target_reason is not None:
            self._hold_for_omx_target(target_reason)
            return
        self._stop_direct_cmd('nav2_goal_mode')
        self._publish_cancel(False)
        self._set_controller_speed_limit(True, catchup=catchup)
        if not self._should_publish_goal(goal):
            self._publish_state('goal_rate_limited', goal=goal, distance_to_scout=distance_to_scout)
            self._log_follow_debug('goal_rate_limited', goal=goal, distance_to_scout=distance_to_scout)
            return
        if not self.nav_client.server_is_ready():
            self._publish_state('nav2_action_wait', goal=goal, distance_to_scout=distance_to_scout)
            self._log_follow_debug('nav2_action_wait', goal=goal, distance_to_scout=distance_to_scout)
            self.get_logger().warning(
                'LEADER_NAV2_DIRECT_WAIT | '
                f'action={self.navigate_action} reason={reason} '
                f'x={goal.pose.position.x:.3f} y={goal.pose.position.y:.3f}',
                throttle_duration_sec=1.0,
            )
            return
        action_goal = NavigateToPose.Goal()
        action_goal.pose = deepcopy(goal)
        action_goal.pose.header.frame_id = action_goal.pose.header.frame_id or 'map'
        action_goal.pose.header.stamp = self.get_clock().now().to_msg()
        try:
            future = self.nav_client.send_goal_async(action_goal)
        except Exception as exc:  # noqa: BLE001
            self.get_logger().error(
                f'LEADER_NAV2_DIRECT_SEND_ERROR | action={self.navigate_action} error={exc}'
            )
            return
        self.nav_goal_pending = True
        self.nav_goal_reason = reason
        self.goal_debug_pub.publish(goal)
        self.last_goal = goal
        self.shadow_goal_active = True
        self.last_goal_wall = self._now()
        self._publish_state(reason, goal=goal, distance_to_scout=distance_to_scout)
        self._log_follow_debug(reason, goal=goal, distance_to_scout=distance_to_scout)
        future.add_done_callback(
            lambda fut, goal=deepcopy(goal), reason=reason: self._on_nav_goal_response(
                fut,
                goal,
                reason,
            )
        )
        self.get_logger().warning(
            'LEADER_NAV2_DIRECT_GOAL_SENT | '
            f'action={self.navigate_action} reason={reason} '
            f'x={goal.pose.position.x:.3f} y={goal.pose.position.y:.3f} catchup={catchup} '
            f'risk_value={self.last_risk_value}'
        )

    def _on_nav_goal_response(self, future, goal: PoseStamped, reason: str) -> None:
        self.nav_goal_pending = False
        try:
            handle = future.result()
        except Exception as exc:  # noqa: BLE001
            self.shadow_goal_active = False
            self.get_logger().warning(
                f'LEADER_NAV2_DIRECT_GOAL_ERROR | action={self.navigate_action} error={exc}'
            )
            return
        if self._target_hold_reason() is not None:
            if handle.accepted:
                try:
                    handle.cancel_goal_async()
                except Exception as exc:  # noqa: BLE001
                    self.get_logger().warning(
                        f'LEADER_NAV2_DIRECT_CANCEL_ERROR | reason=target_hold_response error={exc}'
                    )
            self.shadow_goal_active = False
            self.nav_goal_handle = None
            self.get_logger().warning(
                'LEADER_NAV2_DIRECT_GOAL_CANCELLED | '
                f'reason=target_hold_response action={self.navigate_action}'
            )
            return
        if not handle.accepted:
            self.shadow_goal_active = False
            self.nav_goal_handle = None
            self.get_logger().warning(
                'LEADER_NAV2_DIRECT_GOAL_REJECTED | '
                f'action={self.navigate_action} reason={reason} '
                f'x={goal.pose.position.x:.3f} y={goal.pose.position.y:.3f}'
            )
            return
        self.nav_goal_handle = handle
        self.get_logger().warning(
            'LEADER_NAV2_DIRECT_GOAL_ACCEPTED | '
            f'action={self.navigate_action} reason={reason} '
            f'x={goal.pose.position.x:.3f} y={goal.pose.position.y:.3f}'
        )
        handle.get_result_async().add_done_callback(
            lambda fut, reason=reason: self._on_nav_goal_result(fut, reason)
        )

    def _on_nav_goal_result(self, future, reason: str) -> None:
        self.nav_goal_handle = None
        self.shadow_goal_active = False
        try:
            result = future.result()
            status = int(getattr(result, 'status', GoalStatus.STATUS_UNKNOWN))
        except Exception as exc:  # noqa: BLE001
            self.get_logger().warning(
                f'LEADER_NAV2_DIRECT_RESULT_ERROR | action={self.navigate_action} error={exc}'
            )
            return
        self.get_logger().warning(
            'LEADER_NAV2_DIRECT_RESULT | '
            f'action={self.navigate_action} reason={reason} status={status}'
        )

    def _build_risk_goal(self) -> Optional[PoseStamped]:
        if not self.enable_risk_priority or self.risk_msg is None:
            return None
        if self._now() - self.risk_wall > self.risk_pose_timeout:
            return None
        risk = self.risk_msg
        width = int(risk.info.width)
        height = int(risk.info.height)
        if width <= 0 or height <= 0 or len(risk.data) != width * height:
            return None
        best_value = -1
        best_index = -1
        for index, raw in enumerate(risk.data):
            value = int(raw)
            if value > best_value:
                best_value = value
                best_index = index
        if best_index < 0 or best_value < self.risk_min_value:
            self.last_risk_target = None
            self.last_risk_value = best_value
            return None
        mx = best_index % width
        my = best_index // width
        target = self._map_to_world(risk, mx, my)
        target = self._nearest_free_point(target)
        self.last_risk_target = target
        self.last_risk_value = best_value
        yaw = 0.0
        if self.leader_pose is not None:
            yaw = math.atan2(
                target[1] - self.leader_pose.pose.position.y,
                target[0] - self.leader_pose.pose.position.x,
            )
        return self._pose_from_xy_yaw(target[0], target[1], yaw)

    def _map_to_world(self, grid: OccupancyGrid, mx: int, my: int) -> Point2:
        info = grid.info
        resolution = float(info.resolution)
        local_x = (float(mx) + 0.5) * resolution
        local_y = (float(my) + 0.5) * resolution
        yaw = yaw_from_quaternion(info.origin.orientation)
        world_x = (
            float(info.origin.position.x)
            + math.cos(yaw) * local_x
            - math.sin(yaw) * local_y
        )
        world_y = (
            float(info.origin.position.y)
            + math.sin(yaw) * local_x
            + math.cos(yaw) * local_y
        )
        return (world_x, world_y)

    def _nearest_free_point(self, target: Point2) -> Point2:
        if self._candidate_is_free(target[0], target[1]):
            return target
        if self.map_msg is None:
            return target
        best: Optional[tuple[float, Point2]] = None
        max_radius = max(self.search_radius, self.follow_distance, 1.0)
        radius_steps = max(1, int(math.ceil(max_radius / self.search_step)))
        for radius_index in range(1, radius_steps + 1):
            radius = radius_index * self.search_step
            angle_steps = max(12, int(math.ceil(2.0 * math.pi * radius / self.search_step)))
            for angle_index in range(angle_steps):
                theta = 2.0 * math.pi * float(angle_index) / float(angle_steps)
                candidate = (
                    target[0] + radius * math.cos(theta),
                    target[1] + radius * math.sin(theta),
                )
                if not self._candidate_is_free(candidate[0], candidate[1]):
                    continue
                score = math.hypot(candidate[0] - target[0], candidate[1] - target[1])
                if best is None or score < best[0]:
                    best = (score, candidate)
            if best is not None:
                return best[1]
        return target

    def _pose_from_xy_yaw(self, x: float, y: float, yaw: float) -> PoseStamped:
        goal = PoseStamped()
        goal.header.frame_id = 'map'
        goal.header.stamp = self.get_clock().now().to_msg()
        goal.pose.position.x = float(x)
        goal.pose.position.y = float(y)
        goal.pose.position.z = 0.0
        qx, qy, qz, qw = quaternion_from_yaw(yaw)
        goal.pose.orientation.x = qx
        goal.pose.orientation.y = qy
        goal.pose.orientation.z = qz
        goal.pose.orientation.w = qw
        return goal

    def _scan_tick(self) -> None:
        if not self.scan_enabled:
            self._publish_scan_state('SCAN_DISABLED')
            return
        now = self._now()
        age = now - self.last_scan_wall
        scan_state = 'SCAN_ACTIVE' if age <= self.scan_timeout else 'SCAN_STALE'
        self._publish_scan_state(scan_state, age=age)

    def _failover_allows_shadow(self) -> bool:
        return self.failover_state in (
            '',
            'NORMAL_OPERATION',
            'NEW_SCOUT_EXPLORING',
        )

    def _stop_shadow_for_failover(self) -> None:
        self._cancel_shadow_goal(f'failover_{self.failover_state}')
        self._stop_direct_cmd(f'failover_{self.failover_state}')
        self.shadow_active = False
        self.last_goal = None
        self._set_controller_speed_limit(False)
        if self.failover_state in (
            'SCOUT_SUSPECTED_DEAD',
            'SCOUT_DEAD_CONFIRMED',
            'FAILOVER_TRIGGERED',
            'RECOVERY_NAVIGATING',
        ):
            self.mode = LeaderMode.RECOVERY_NAVIGATING
        elif self.failover_state == 'FOLLOWER_SCOUT_TAKEOVER':
            self.mode = LeaderMode.WAIT_NEW_SCOUT
        else:
            self.mode = LeaderMode.IDLE

    def _force_leader_stop_for_target(self, reason: str) -> None:
        """Hard stop only the robot base while OMX owns target tracking."""
        now = self._now()
        if now - self.last_target_cancel_wall >= self.target_cancel_period:
            self._cancel_shadow_goal(reason)
            self.last_target_cancel_wall = now
            self.shadow_goal_active = False
            self.last_goal = None
            self.get_logger().warning(
                f'[LEADER_SHADOW] TARGET_HARD_STOP_CANCEL | reason={reason}'
            )
        self._stop_direct_cmd(reason)
        self._publish_twist(0.0, 0.0)

    def _active_scout_pose(self) -> Tuple[Optional[PoseStamped], float]:
        if self.active_scout_id == self.follower_robot_name:
            return self.follower_scout_pose, self.follower_scout_wall
        return self.original_scout_pose, self.original_scout_wall

    def _update_heading_from_pose(self, pose: PoseStamped) -> None:
        now = self._now()
        point = (pose.pose.position.x, pose.pose.position.y)
        if self.previous_scout_sample is None:
            self.previous_scout_sample = (now, point)
            if self.heading is None:
                self.heading = yaw_from_quaternion(pose.pose.orientation)
            return
        _, previous = self.previous_scout_sample
        dx = point[0] - previous[0]
        dy = point[1] - previous[1]
        moved = math.hypot(dx, dy)
        if moved < self.heading_min_motion:
            return
        measured = math.atan2(dy, dx)
        self.heading = measured if self.heading is None else angle_lerp(
            self.heading, measured, self.heading_alpha
        )
        self.previous_scout_sample = (now, point)

    def _build_shadow_goal(self, scout_pose: PoseStamped) -> Optional[PoseStamped]:
        heading = yaw_from_quaternion(scout_pose.pose.orientation)
        nominal = (
            scout_pose.pose.position.x - self.follow_distance * math.cos(heading),
            scout_pose.pose.position.y - self.follow_distance * math.sin(heading),
        )
        self.last_nominal_target = nominal
        feasible, mode = self._nearest_rear_feasible(scout_pose, heading, nominal)
        self.last_target_mode = mode
        if feasible is None:
            self.last_target_behind_scout = False
            self.last_target_free = False
            return None
        self.last_target_behind_scout = self._target_behind_scout(
            feasible,
            scout_pose,
            heading,
        )
        self.last_target_free = self._candidate_is_free(feasible[0], feasible[1])
        if not self.last_target_behind_scout:
            self.last_target_mode = 'hold_target_not_behind_scout'
            self.last_target_free = False
            return None
        goal = PoseStamped()
        goal.header.frame_id = 'map'
        goal.header.stamp = self.get_clock().now().to_msg()
        goal.pose.position.x = feasible[0]
        goal.pose.position.y = feasible[1]
        goal.pose.position.z = 0.0
        qx, qy, qz, qw = quaternion_from_yaw(heading)
        goal.pose.orientation.x = qx
        goal.pose.orientation.y = qy
        goal.pose.orientation.z = qz
        goal.pose.orientation.w = qw
        return goal

    def _nearest_rear_feasible(
        self,
        scout_pose: PoseStamped,
        scout_yaw: float,
        nominal: Point2,
    ) -> tuple[Optional[Point2], str]:
        if self._candidate_is_free(nominal[0], nominal[1]):
            return nominal, 'exact_rear'

        scout_x = float(scout_pose.pose.position.x)
        scout_y = float(scout_pose.pose.position.y)
        rear_angle = scout_yaw + math.pi
        sector_rad = math.radians(30.0)
        max_distance = max(self.follow_distance, self.search_radius, self.resume_distance)
        min_distance = max(self.stop_distance, self.search_step)
        distance_steps = max(1, int(math.ceil((max_distance - min_distance) / self.search_step)))
        angle_steps = 7
        best: Optional[Tuple[float, Point2]] = None
        for step in range(distance_steps + 1):
            distance = min_distance + step * self.search_step
            for angle_index in range(angle_steps):
                if angle_steps == 1:
                    delta = 0.0
                else:
                    delta = -sector_rad + (2.0 * sector_rad * angle_index / (angle_steps - 1))
                theta = rear_angle + delta
                candidate = (
                    scout_x + distance * math.cos(theta),
                    scout_y + distance * math.sin(theta),
                )
                if not self._target_behind_scout(candidate, scout_pose, scout_yaw):
                    continue
                if not self._candidate_is_free(candidate[0], candidate[1]):
                    continue
                score = abs(distance - self.follow_distance) + abs(delta)
                if best is None or score < best[0]:
                    best = (score, candidate)
        if best is not None:
            return best[1], 'adjusted_rear'
        return None, 'hold_no_safe_rear_goal'

    @staticmethod
    def _target_behind_scout(
        target: Point2,
        scout_pose: PoseStamped,
        scout_yaw: float,
    ) -> bool:
        dx = target[0] - float(scout_pose.pose.position.x)
        dy = target[1] - float(scout_pose.pose.position.y)
        forward_x = math.cos(scout_yaw)
        forward_y = math.sin(scout_yaw)
        return (dx * forward_x + dy * forward_y) < -1.0e-6

    def _candidate_is_free(self, x: float, y: float) -> bool:
        if self.map_msg is None:
            return True
        center = self._world_to_map(x, y)
        if center is None:
            return False
        info = self.map_msg.info
        radius = max(1, int(math.ceil(self.map_clearance / info.resolution)))
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                if dx * dx + dy * dy > radius * radius:
                    continue
                mx = center[0] + dx
                my = center[1] + dy
                if mx < 0 or my < 0 or mx >= info.width or my >= info.height:
                    return False
                value = int(self.map_msg.data[my * info.width + mx])
                if value >= self.occupied_threshold:
                    return False
                if value < 0 and not self.allow_unknown:
                    return False
        return True

    def _world_to_map(self, x: float, y: float) -> Optional[Tuple[int, int]]:
        if self.map_msg is None:
            return None
        info = self.map_msg.info
        if info.resolution <= 0.0:
            return None
        origin_yaw = yaw_from_quaternion(info.origin.orientation)
        dx = x - info.origin.position.x
        dy = y - info.origin.position.y
        local_x = math.cos(origin_yaw) * dx + math.sin(origin_yaw) * dy
        local_y = -math.sin(origin_yaw) * dx + math.cos(origin_yaw) * dy
        mx = int(math.floor(local_x / info.resolution))
        my = int(math.floor(local_y / info.resolution))
        if mx < 0 or my < 0 or mx >= info.width or my >= info.height:
            return None
        return mx, my

    def _should_publish_goal(self, goal: PoseStamped) -> bool:
        now = self._now()
        if now - self.last_goal_wall < self.goal_period:
            return False
        if self.last_goal is None:
            return True
        if self._nav_execution_stalled(now):
            self.shadow_goal_active = False
            return True
        distance = math.hypot(
            goal.pose.position.x - self.last_goal.pose.position.x,
            goal.pose.position.y - self.last_goal.pose.position.y,
        )
        return distance >= self.goal_min_change

    def _nav_execution_stalled(self, now: float) -> bool:
        if not self.shadow_goal_active or self.last_goal_wall < 0.0:
            return False
        if now - self.last_goal_wall < self.nav_execution_timeout:
            return False
        plan_after_goal = self.last_path_wall >= self.last_goal_wall
        cmd_after_goal = self.last_cmd_vel_wall >= self.last_goal_wall
        return not plan_after_goal and not cmd_after_goal

    def _publish_direct_shadow_cmd(self, goal: PoseStamped, *, catchup: bool) -> None:
        leader = self.leader_pose
        if leader is None:
            self._stop_direct_cmd('missing_leader_pose')
            return
        dx = goal.pose.position.x - leader.pose.position.x
        dy = goal.pose.position.y - leader.pose.position.y
        distance = math.hypot(dx, dy)
        target_heading = math.atan2(dy, dx) if distance > 1e-6 else yaw_from_quaternion(
            leader.pose.orientation
        )
        heading_error = math.atan2(
            math.sin(target_heading - yaw_from_quaternion(leader.pose.orientation)),
            math.cos(target_heading - yaw_from_quaternion(leader.pose.orientation)),
        )
        max_linear = self.catchup_linear_vel if catchup else self.shadow_linear_vel
        linear = 0.0
        if distance > self.cmd_goal_tolerance:
            linear = min(max_linear, self.linear_kp * (distance - self.cmd_goal_tolerance))
            if abs(heading_error) > self.heading_slowdown_rad:
                linear *= max(0.0, 1.0 - min(abs(heading_error), math.pi) / math.pi)
            if abs(heading_error) > 1.35:
                linear = 0.0
        angular = max(
            -self.shadow_angular_vel,
            min(self.shadow_angular_vel, self.angular_kp * heading_error),
        )
        linear = min(self.cmd_max_linear_vel, linear * self.cmd_linear_scale)
        angular = max(
            -self.cmd_max_angular_vel,
            min(self.cmd_max_angular_vel, angular * self.cmd_angular_scale),
        )
        self._publish_twist(linear, angular)
        self.direct_cmd_active = True

    def _publish_twist(self, linear_x: float, angular_z: float) -> None:
        if self.cmd_pub is None:
            return
        if self.use_stamped_cmd_vel:
            msg = TwistStamped()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = 'base_footprint'
            msg.twist.linear.x = float(linear_x)
            msg.twist.angular.z = float(angular_z)
        else:
            msg = Twist()
            msg.linear.x = float(linear_x)
            msg.angular.z = float(angular_z)
        self.cmd_pub.publish(msg)

    def _stop_direct_cmd(self, reason: str) -> None:
        if not self.direct_cmd_active:
            return
        self._publish_twist(0.0, 0.0)
        self.direct_cmd_active = False
        self.get_logger().warning(
            f'[LEADER_SHADOW] DIRECT_CMD_STOP | reason={reason}'
        )

    def _set_controller_speed_limit(self, limited: bool, *, catchup: bool = False) -> None:
        profile = 'restore'
        if limited:
            profile = 'catchup' if catchup else 'shadow'
        if self.speed_limit_pending:
            return
        if self.speed_profile == profile:
            return
        if not self.controller_client.service_is_ready():
            self.get_logger().info(
                '[LEADER_SHADOW] CONTROLLER_PARAM_SERVICE_WAIT | '
                'shadow speed limit will be retried',
                throttle_duration_sec=5.0,
            )
            return
        max_linear = self.catchup_linear_vel if limited and catchup else self.shadow_linear_vel
        max_angular = self.shadow_angular_vel if limited else self.restore_angular_vel
        if not limited:
            max_linear = self.restore_linear_vel
        request = SetParameters.Request()
        request.parameters = [
            self._double_parameter('FollowPath.max_vel_x', max_linear),
            self._double_parameter('FollowPath.max_speed_xy', max_linear),
            self._double_parameter('FollowPath.max_vel_theta', max_angular),
        ]
        self.speed_limit_pending = True
        future = self.controller_client.call_async(request)
        future.add_done_callback(
            lambda fut: self._on_speed_limit_result(fut, profile)
        )

    def _on_speed_limit_result(self, future, profile: str) -> None:
        self.speed_limit_pending = False
        try:
            results = future.result().results
        except Exception as exc:  # noqa: BLE001
            self.get_logger().warning(f'[LEADER_SHADOW] SPEED_LIMIT_FAILED | {exc}')
            return
        ok = all(bool(result.successful) for result in results)
        if ok:
            self.speed_profile = profile
            self.get_logger().warning(
                '[LEADER_SHADOW] SPEED_LIMIT_SET | '
                f'profile={profile} linear={self.shadow_linear_vel:.2f}/{self.restore_linear_vel:.2f}'
            )
        else:
            reason = '; '.join(str(result.reason) for result in results if result.reason)
            self.get_logger().warning(
                f'[LEADER_SHADOW] SPEED_LIMIT_REJECTED | {reason or "unknown"}'
            )

    @staticmethod
    def _double_parameter(name: str, value: float) -> Parameter:
        param = Parameter()
        param.name = name
        param.value = ParameterValue(
            type=ParameterType.PARAMETER_DOUBLE,
            double_value=float(value),
        )
        return param

    def _publish_cancel(self, value: bool) -> None:
        msg = Bool()
        msg.data = bool(value)
        self.cancel_pub.publish(msg)

    def _pulse_cancel(self) -> None:
        self._publish_cancel(True)
        self._publish_cancel(False)

    def _cancel_shadow_goal(self, reason: str) -> None:
        """Cancel the previous shadow goal once when shadow loses authority."""
        if not self.shadow_goal_active and self.nav_goal_handle is None and not self.nav_goal_pending:
            return
        self._pulse_cancel()
        if self.nav_goal_handle is not None:
            try:
                self.nav_goal_handle.cancel_goal_async()
            except Exception as exc:  # noqa: BLE001
                self.get_logger().warning(
                    f'LEADER_NAV2_DIRECT_CANCEL_ERROR | reason={reason} error={exc}'
                )
        self.nav_goal_handle = None
        self.nav_goal_pending = False
        self.shadow_goal_active = False
        self.last_goal = None
        self.get_logger().warning(
            f'LEADER_NAV2_DIRECT_GOAL_CANCELLED | reason={reason}'
        )

    def _publish_state(
        self,
        reason: str,
        *,
        goal: Optional[PoseStamped] = None,
        distance_to_scout: Optional[float] = None,
    ) -> None:
        data = {
            'mode': self.mode.value,
            'reason': reason,
            'active_scout_id': self.active_scout_id,
            'failover_state': self.failover_state,
            'shadow_active': self.shadow_active,
            'omx_state': self.omx_state,
            'target_detected': bool(self.target_detected),
            'target_hold_active': bool(self.target_hold_active),
            'scan_fov_deg': self.scan_fov_deg,
            'scan_heading_reference': 'leader_current_heading',
        }
        if distance_to_scout is not None:
            data['distance_to_scout_m'] = round(float(distance_to_scout), 3)
        if self.heading is not None:
            data['movement_heading_rad'] = round(float(self.heading), 4)
        if self.last_nominal_target is not None:
            data['nominal_target'] = {
                'x': round(float(self.last_nominal_target[0]), 3),
                'y': round(float(self.last_nominal_target[1]), 3),
            }
        if self.last_risk_target is not None:
            data['risk_target'] = {
                'x': round(float(self.last_risk_target[0]), 3),
                'y': round(float(self.last_risk_target[1]), 3),
                'value': int(self.last_risk_value),
            }
        if self.last_target_point is not None:
            data['remembered_target'] = {
                'x': round(float(self.last_target_point.point.x), 3),
                'y': round(float(self.last_target_point.point.y), 3),
                'z': round(float(self.last_target_point.point.z), 3),
                'age_ms': int(self._age_ms(self.last_target_point_wall)),
            }
        data['target_mode'] = getattr(self, 'last_target_mode', 'none')
        data['target_behind_scout'] = bool(
            getattr(self, 'last_target_behind_scout', False)
        )
        data['target_free'] = bool(getattr(self, 'last_target_free', False))
        if goal is not None:
            data['goal'] = {
                'x': round(float(goal.pose.position.x), 3),
                'y': round(float(goal.pose.position.y), 3),
            }
        msg = String()
        msg.data = json.dumps(data, sort_keys=True)
        self.state_pub.publish(msg)

    def _age_ms(self, wall: float) -> float:
        if wall < 0.0:
            return -1.0
        return max(0.0, (self._now() - wall) * 1000.0)

    def _stamp_age_ms(self, msg: PoseStamped) -> float:
        stamp = msg.header.stamp
        stamp_sec = float(stamp.sec) + float(stamp.nanosec) * 1.0e-9
        if stamp_sec <= 0.0:
            return -1.0
        return max(0.0, (self._now() - stamp_sec) * 1000.0)

    def _log_pose_pipeline(
        self,
        name: str,
        topic: str,
        msg: PoseStamped,
        received_wall: float,
    ) -> None:
        self.get_logger().warning(
            'POSE_PIPELINE | '
            f'node=leader_shadow_follow name={name} topic={topic} '
            f'frame_id={msg.header.frame_id or "(empty)"} '
            f'source_stamp_age_ms={self._stamp_age_ms(msg):.0f} '
            f'receive_age_ms={self._age_ms(received_wall):.0f}',
            throttle_duration_sec=3.0,
        )

    def _log_follow_debug(
        self,
        reason: str,
        *,
        goal: Optional[PoseStamped] = None,
        distance_to_scout: Optional[float] = None,
    ) -> None:
        scout_pose, scout_wall = self._active_scout_pose()
        distance_invalid_reason = ''
        if self.leader_pose is not None and scout_pose is not None:
            distance_invalid_reason = self._pose_pair_invalid_reason(
                self.leader_pose, scout_pose
            )
            if distance_to_scout is None and not distance_invalid_reason:
                distance_to_scout = self._distance_pose(self.leader_pose, scout_pose)
        goal_required = goal is not None and reason not in (
            'at_shadow_target',
            'waiting_pose',
            'waiting_system_ready',
            'waiting_video_ready',
            'waiting_localization_ready',
            'stopped_close_to_scout',
            'hold_resume_hysteresis',
        )
        scout_x = float('nan')
        scout_y = float('nan')
        scout_yaw = float('nan')
        leader_x = float('nan')
        leader_y = float('nan')
        target_x = float('nan')
        target_y = float('nan')
        target_yaw = float('nan')
        if scout_pose is not None:
            scout_x = float(scout_pose.pose.position.x)
            scout_y = float(scout_pose.pose.position.y)
            scout_yaw = yaw_from_quaternion(scout_pose.pose.orientation)
        if self.leader_pose is not None:
            leader_x = float(self.leader_pose.pose.position.x)
            leader_y = float(self.leader_pose.pose.position.y)
        if goal is not None:
            target_x = float(goal.pose.position.x)
            target_y = float(goal.pose.position.y)
            target_yaw = yaw_from_quaternion(goal.pose.orientation)
        self.get_logger().warning(
            'LEADER_FOLLOW_DEBUG | '
            f'backend={getattr(self, "follow_backend", "nav2")} '
            f'start_motion={getattr(self, "video_ready", True)} '
            f'scout_pose_rx={scout_pose is not None} '
            f'scout_pose_age_ms={self._age_ms(scout_wall):.0f} '
            f'leader_pose_rx={self.leader_pose is not None} '
            f'leader_pose_age_ms={self._age_ms(getattr(self, "leader_pose_wall", -1.0e9)):.0f} '
            f'system_ready={getattr(self, "system_ready", True)} '
            f'dashboard_ready={getattr(self, "video_ready", True)} '
            f'localization_ready={self.localization_ready} '
            f'active_scout_id={self.active_scout_id} '
            f'scout_x={scout_x:.3f} scout_y={scout_y:.3f} '
            f'scout_yaw_deg={math.degrees(scout_yaw) if math.isfinite(scout_yaw) else float("nan"):.1f} '
            f'leader_x={leader_x:.3f} leader_y={leader_y:.3f} '
            f'nav_server_mode={getattr(self, "follow_backend", "nav2") == "nav2"} '
            f'desired_follow_distance_m={self.follow_distance:.2f} '
            f'target_x={target_x:.3f} target_y={target_y:.3f} '
            f'target_yaw_deg={math.degrees(target_yaw) if math.isfinite(target_yaw) else float("nan"):.1f} '
            f'target_mode={getattr(self, "last_target_mode", "none")} '
            f'target_behind_scout={getattr(self, "last_target_behind_scout", False)} '
            f'target_free={getattr(self, "last_target_free", False)} '
            f'goal_required={goal_required} '
            f'goal_sent={reason.endswith("goal_sent") or reason == "goal_sent"} '
            f'goal_accepted={getattr(self, "shadow_goal_active", False)} '
            f'goal_status={reason} '
            f'path_received={getattr(self, "last_path_wall", -1.0e9) >= 0.0} '
            f'path_age_ms={self._age_ms(getattr(self, "last_path_wall", -1.0e9)):.0f} '
            f'cmd_vel_age_ms={self._age_ms(getattr(self, "last_cmd_vel_wall", -1.0e9)):.0f} '
            f'nonzero_cmd_age_ms={self._age_ms(getattr(self, "last_nonzero_cmd_wall", -1.0e9)):.0f} '
            f'odom_age_ms={self._age_ms(getattr(self, "last_odom_wall", -1.0e9)):.0f} '
            f'odom_motion={getattr(self, "odom_motion", False)} '
            f'distance_to_scout={distance_to_scout if distance_to_scout is not None else float("nan"):.3f} '
            f'risk_enabled={getattr(self, "enable_risk_priority", False)} '
            f'risk_rx={getattr(self, "risk_msg", None) is not None} '
            f'risk_age_ms={self._age_ms(getattr(self, "risk_wall", -1.0e9)):.0f} '
            f'risk_value={getattr(self, "last_risk_value", -1)} '
            f'distance_invalid_reason={distance_invalid_reason or "none"} '
            f'blocking_reason={reason}',
            throttle_duration_sec=1.0,
        )
        self._log_nav2_pipeline(
            reason,
            goal=goal,
            distance_to_scout=distance_to_scout,
        )

    def _log_nav2_pipeline(
        self,
        reason: str,
        *,
        goal: Optional[PoseStamped],
        distance_to_scout: Optional[float],
    ) -> None:
        if goal is None and self.last_goal is not None:
            goal = self.last_goal
        goal_sent = reason.endswith('goal_sent') or reason == 'goal_sent'
        goal_accepted = bool(
            getattr(self, 'shadow_goal_active', False)
            and getattr(self, 'last_goal_wall', -1.0e9) >= 0.0
            and (
                getattr(self, 'last_path_wall', -1.0e9) >= self.last_goal_wall
                or getattr(self, 'last_cmd_vel_wall', -1.0e9) >= self.last_goal_wall
                or getattr(self, 'last_nonzero_cmd_wall', -1.0e9) >= self.last_goal_wall
            )
        )
        target_x = float('nan')
        target_y = float('nan')
        target_yaw = float('nan')
        if goal is not None:
            target_x = float(goal.pose.position.x)
            target_y = float(goal.pose.position.y)
            target_yaw = yaw_from_quaternion(goal.pose.orientation)
        self.get_logger().warning(
            'LEADER_NAV2_PIPELINE | '
            f'target_x={target_x:.3f} '
            f'target_y={target_y:.3f} '
            f'target_yaw={target_yaw:.3f} '
            f'goal_sent={goal_sent} '
            f'goal_accepted={goal_accepted} '
            f'goal_status={reason} '
            f'path_age_ms={self._age_ms(getattr(self, "last_path_wall", -1.0e9)):.0f} '
            f'controller_cmd_age_ms={self._age_ms(getattr(self, "last_cmd_vel_wall", -1.0e9)):.0f} '
            f'hardware_cmd_age_ms={self._age_ms(getattr(self, "last_cmd_vel_wall", -1.0e9)):.0f} '
            f'odom_delta_m={getattr(self, "odom_delta_m", 0.0):.3f} '
            f'distance_to_scout={distance_to_scout if distance_to_scout is not None else float("nan"):.3f} '
            f'distance_decreased={getattr(self, "distance_decreased", False)} '
            f'blocking_reason={reason}',
            throttle_duration_sec=1.0,
        )

    def _publish_scan_state(self, state: str, *, age: Optional[float] = None) -> None:
        data = {
            'state': state,
            'enabled': self.scan_enabled,
            'scan_topic': self.scan_topic,
            'fov_deg': self.scan_fov_deg,
            'relative_bearing_accept_rad': round(math.radians(self.scan_fov_deg) * 0.5, 4),
            'heading_reference': 'leader_current_heading',
            'risk_scan_only': True,
            'nav2_obstacle_lidar_unchanged': True,
        }
        if age is not None:
            data['age_sec'] = round(float(age), 3)
        if self.last_scan_stamp >= 0.0:
            data['last_stamp_sec'] = round(float(self.last_scan_stamp), 3)
        msg = String()
        msg.data = json.dumps(data, sort_keys=True)
        self.scan_state_pub.publish(msg)

    @staticmethod
    def _distance_pose(first: PoseStamped, second: PoseStamped) -> float:
        return math.hypot(
            first.pose.position.x - second.pose.position.x,
            first.pose.position.y - second.pose.position.y,
        )

    @staticmethod
    def _pose_invalid_reason(msg: PoseStamped) -> str:
        pose = msg.pose
        quat = pose.orientation
        values = (
            pose.position.x,
            pose.position.y,
            pose.position.z,
            quat.x,
            quat.y,
            quat.z,
            quat.w,
        )
        if not all(math.isfinite(float(value)) for value in values):
            return 'nonfinite'
        norm = math.sqrt(
            quat.x * quat.x + quat.y * quat.y + quat.z * quat.z + quat.w * quat.w
        )
        if not math.isfinite(norm) or norm < 0.5 or norm > 1.5:
            return 'bad_quaternion'
        if not str(msg.header.frame_id).strip():
            return 'empty_frame'
        return ''

    def _pose_pair_invalid_reason(
        self,
        leader: PoseStamped,
        scout: PoseStamped,
    ) -> str:
        leader_reason = self._pose_invalid_reason(leader)
        if leader_reason:
            return f'leader_{leader_reason}'
        scout_reason = self._pose_invalid_reason(scout)
        if scout_reason:
            return f'scout_{scout_reason}'
        leader_frame = str(leader.header.frame_id).strip().lstrip('/')
        scout_frame = str(scout.header.frame_id).strip().lstrip('/')
        if leader_frame and scout_frame and leader_frame != scout_frame:
            return f'frame_mismatch_{leader_frame}_vs_{scout_frame}'
        return ''


def main(args=None) -> None:
    rclpy.init(args=args)
    node = LeaderShadowFollow()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
