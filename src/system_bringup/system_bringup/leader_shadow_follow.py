#!/usr/bin/env python3
"""Low-speed leader shadow follow for the active scout.

The leader remains a leader: it does not copy the scout pose and it stops
issuing shadow goals as soon as failover owns recovery.  During normal
operation it estimates the active scout's movement heading, creates a rear
standoff target, validates it against the shared map, and publishes rate-limited
leader Nav2 goals.
"""

from __future__ import annotations

import json
import math
from enum import Enum
from typing import Optional, Tuple

import rclpy
from geometry_msgs.msg import PoseStamped, Twist, TwistStamped
from nav_msgs.msg import OccupancyGrid, Odometry, Path
from rcl_interfaces.msg import Parameter, ParameterType, ParameterValue
from rcl_interfaces.srv import SetParameters
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
        self.declare_parameter('pause_on_raw_target_detection', True)
        self.declare_parameter('omx_state_topic', '/omx/state')
        self.declare_parameter('pause_on_omx_aiming', True)
        self.declare_parameter('scout_pose_timeout_sec', 2.5)
        self.declare_parameter('startup_grace_sec', 8.0)

        self.declare_parameter('leader_shadow_follow_distance_m', 1.2)
        self.declare_parameter('leader_shadow_stop_distance_m', 0.8)
        self.declare_parameter('leader_shadow_resume_distance_m', 1.3)
        self.declare_parameter('leader_shadow_far_distance_m', 2.4)
        self.declare_parameter('leader_shadow_max_linear_vel', 0.26)
        self.declare_parameter('leader_shadow_catchup_max_linear_vel', 0.26)
        self.declare_parameter('leader_shadow_max_angular_vel', 1.00)
        self.declare_parameter('leader_restore_max_linear_vel', 0.26)
        self.declare_parameter('leader_restore_max_angular_vel', 1.00)
        self.declare_parameter('leader_shadow_goal_update_period_sec', 1.0)
        self.declare_parameter('leader_shadow_goal_min_change_m', 0.35)
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
        self.pause_on_raw_target_detection = bool(
            get('pause_on_raw_target_detection').value
        )
        self.omx_state_topic = str(get('omx_state_topic').value)
        self.pause_on_omx_aiming = bool(get('pause_on_omx_aiming').value)
        self.scout_pose_timeout = max(0.2, float(get('scout_pose_timeout_sec').value))
        self.startup_grace = max(0.0, float(get('startup_grace_sec').value))

        self.follow_distance = max(0.5, float(get('leader_shadow_follow_distance_m').value))
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
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
        )

        self.goal_pub = self.create_publisher(PoseStamped, self.leader_goal_topic, 10)
        self.cancel_pub = self.create_publisher(Bool, self.leader_cancel_topic, latched_qos)
        self.cmd_pub = None
        if self.direct_shadow_cmd_vel:
            if self.use_stamped_cmd_vel:
                self.cmd_pub = self.create_publisher(TwistStamped, self.cmd_vel_topic, 10)
            else:
                self.cmd_pub = self.create_publisher(Twist, self.cmd_vel_topic, 10)
        self.state_pub = self.create_publisher(String, '/leader_shadow/state', latched_qos)
        self.goal_debug_pub = self.create_publisher(PoseStamped, '/leader_shadow/goal', 10)
        self.scan_state_pub = self.create_publisher(String, '/leader_scan/state', latched_qos)

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
        self.create_subscription(String, self.omx_state_topic, self._on_omx_state, 10)
        if self.scan_enabled:
            self.create_subscription(LaserScan, self.scan_topic, self._on_scan, 10)

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
        self.last_target_cancel_wall = -1.0e9
        self.omx_state = ''
        self.leader_pose: Optional[PoseStamped] = None
        self.leader_pose_wall = -1.0e9
        self.original_scout_pose: Optional[PoseStamped] = None
        self.follower_scout_pose: Optional[PoseStamped] = None
        self.original_scout_wall = -1.0e9
        self.follower_scout_wall = -1.0e9
        self.map_msg: Optional[OccupancyGrid] = None
        self.last_scan_wall = -1.0e9
        self.last_scan_stamp = -1.0
        self.last_path_wall = -1.0e9
        self.last_cmd_vel_wall = -1.0e9
        self.last_nonzero_cmd_wall = -1.0e9
        self.last_odom_wall = -1.0e9
        self.odom_motion = False

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
        self.last_goal_wall = -1.0e9
        self.last_nominal_target: Optional[Point2] = None
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
            self.last_goal_wall = -1.0e9
            self._tick()

    def _on_target_detected(self, msg: Bool) -> None:
        self.target_detected = bool(msg.data)
        if self.target_detected:
            self.target_detected_wall = self._now()
            if self.pause_on_raw_target_detection:
                self._force_leader_stop_for_target('target_detected_signal')

    def _on_omx_state(self, msg: String) -> None:
        self.omx_state = str(msg.data).strip().upper()

    @staticmethod
    def _is_omx_aiming(state: str) -> bool:
        """Target lock states that must hold leader shadow motion."""
        return str(state).strip().upper() in ('TRACKING', 'CONFIRMING', 'FIRING')

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
        if self._now() - self.start_wall < self.startup_grace:
            self._publish_state('startup_grace')
            self._log_follow_debug('startup_grace')
            return
        if self.require_localization_ready and not self.localization_ready:
            self._cancel_shadow_goal('waiting_localization_ready')
            self._stop_direct_cmd('waiting_localization_ready')
            self.mode = LeaderMode.IDLE
            self._set_controller_speed_limit(False)
            self._publish_state('waiting_localization_ready')
            self._log_follow_debug('waiting_localization_ready')
            return
        if getattr(self, 'require_system_ready', False) and not getattr(
            self, 'system_ready', True
        ):
            self._cancel_shadow_goal('waiting_system_ready')
            self._stop_direct_cmd('waiting_system_ready')
            self.mode = LeaderMode.IDLE
            self._set_controller_speed_limit(False)
            self._publish_state('waiting_system_ready')
            self._log_follow_debug('waiting_system_ready')
            return
        if self.require_video_ready and not self.video_ready:
            self._cancel_shadow_goal('waiting_video_ready')
            self._stop_direct_cmd('waiting_video_ready')
            self.mode = LeaderMode.IDLE
            self._set_controller_speed_limit(False)
            self._publish_state('waiting_video_ready')
            self._log_follow_debug('waiting_video_ready')
            return
        if self.pause_on_omx_aiming and self._is_omx_aiming(self.omx_state):
            self._cancel_shadow_goal('omx_aiming')
            self._stop_direct_cmd('omx_aiming')
            self.mode = LeaderMode.IDLE
            self._set_controller_speed_limit(False)
            self._publish_state('omx_aiming_hold')
            self._log_follow_debug('omx_aiming_hold')
            return
        if (
            self.pause_on_raw_target_detection
            and self._now() - self.target_detected_wall <= self.target_stop_hold
        ):
            self._force_leader_stop_for_target('target_detected_hold')
            self.shadow_active = False
            self.mode = LeaderMode.IDLE
            self._set_controller_speed_limit(False)
            self._publish_state('target_detected_hold')
            self._log_follow_debug('target_detected_hold')
            return
        if not self._failover_allows_shadow():
            self._stop_shadow_for_failover()
            self._publish_state('failover_owns_leader_goal')
            self._log_follow_debug('failover_owns_leader_goal')
            return

        scout_pose, scout_wall = self._active_scout_pose()
        if self.leader_pose is None or scout_pose is None:
            self._cancel_shadow_goal('waiting_pose')
            self._stop_direct_cmd('waiting_pose')
            self.mode = LeaderMode.IDLE
            self._set_controller_speed_limit(False)
            self._publish_state('waiting_pose')
            self._log_follow_debug('waiting_pose')
            return
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
        distance_to_scout = self._distance_pose(self.leader_pose, scout_pose)
        goal = self._build_shadow_goal(scout_pose)
        if goal is None:
            self._cancel_shadow_goal('no_feasible_shadow_target')
            self._stop_direct_cmd('no_feasible_shadow_target')
            self._publish_state('no_feasible_shadow_target')
            self._log_follow_debug('no_feasible_shadow_target')
            return
        distance_to_shadow_goal = self._distance_pose(self.leader_pose, goal)
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
        if self.direct_shadow_cmd_vel:
            self._cancel_shadow_goal('direct_shadow_cmd_vel')
            if not self.direct_cmd_active:
                self._pulse_cancel()
            self._set_controller_speed_limit(False)
            self._publish_direct_shadow_cmd(goal, catchup=catchup)
            self.goal_debug_pub.publish(goal)
            self.last_goal = goal
            self.last_goal_wall = self._now()
            self._publish_state('direct_cmd', goal=goal, distance_to_scout=distance_to_scout)
            self._log_follow_debug('direct_cmd', goal=goal, distance_to_scout=distance_to_scout)
            return

        self._stop_direct_cmd('nav2_shadow_goal_mode')
        self._publish_cancel(False)
        self._set_controller_speed_limit(True, catchup=catchup)
        if not self._should_publish_goal(goal):
            self._publish_state('goal_rate_limited')
            self._log_follow_debug('goal_rate_limited', goal=goal, distance_to_scout=distance_to_scout)
            return

        self.goal_pub.publish(goal)
        self.goal_debug_pub.publish(goal)
        self.last_goal = goal
        self.shadow_goal_active = True
        self.last_goal_wall = self._now()
        self._publish_state('goal_sent', goal=goal, distance_to_scout=distance_to_scout)
        self._log_follow_debug('goal_sent', goal=goal, distance_to_scout=distance_to_scout)
        self.get_logger().warning(
            '[LEADER_SHADOW] GOAL_SENT | '
            f'active_scout={self.active_scout_id} '
            f'x={goal.pose.position.x:.3f} y={goal.pose.position.y:.3f} '
            f'D={distance_to_scout:.2f} catchup={catchup}'
        )

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
        """Hard stop the leader while OMX owns target tracking."""
        now = self._now()
        if now - self.last_target_cancel_wall >= self.target_cancel_period:
            self._pulse_cancel()
            self.last_target_cancel_wall = now
            self.shadow_goal_active = False
            self.last_goal = None
            self.get_logger().warning(
                f'[LEADER_SHADOW] TARGET_HARD_STOP_CANCEL | reason={reason}'
            )
        self._stop_direct_cmd(reason)
        if self.direct_shadow_cmd_vel:
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
        heading = self.heading
        if heading is None:
            heading = yaw_from_quaternion(scout_pose.pose.orientation)
        nominal = (
            scout_pose.pose.position.x - self.follow_distance * math.cos(heading),
            scout_pose.pose.position.y - self.follow_distance * math.sin(heading),
        )
        self.last_nominal_target = nominal
        feasible = self._nearest_feasible(nominal)
        if feasible is None:
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

    def _nearest_feasible(self, nominal: Point2) -> Optional[Point2]:
        if self._candidate_is_free(nominal[0], nominal[1]):
            return nominal
        best: Optional[Tuple[float, Point2]] = None
        rings = int(math.ceil(self.search_radius / self.search_step))
        for ring in range(1, rings + 1):
            radius = ring * self.search_step
            samples = max(12, int(math.ceil(2.0 * math.pi * radius / self.search_step)))
            for index in range(samples):
                theta = 2.0 * math.pi * index / samples
                candidate = (
                    nominal[0] + radius * math.cos(theta),
                    nominal[1] + radius * math.sin(theta),
                )
                if not self._candidate_is_free(candidate[0], candidate[1]):
                    continue
                score = math.hypot(candidate[0] - nominal[0], candidate[1] - nominal[1])
                if best is None or score < best[0]:
                    best = (score, candidate)
            if best is not None:
                return best[1]
        return None

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
        distance = math.hypot(
            goal.pose.position.x - self.last_goal.pose.position.x,
            goal.pose.position.y - self.last_goal.pose.position.y,
        )
        return distance >= self.goal_min_change

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
        if not self.shadow_goal_active:
            return
        self._pulse_cancel()
        self.shadow_goal_active = False
        self.last_goal = None
        self.get_logger().warning(
            f'[LEADER_SHADOW] GOAL_CANCELLED | reason={reason}'
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
        )
        self.get_logger().warning(
            'LEADER_FOLLOW_DEBUG | '
            f'backend={getattr(self, "follow_backend", "nav2")} '
            f'scout_pose_rx={scout_pose is not None} '
            f'scout_pose_age_ms={self._age_ms(scout_wall):.0f} '
            f'leader_pose_rx={self.leader_pose is not None} '
            f'leader_pose_age_ms={self._age_ms(getattr(self, "leader_pose_wall", -1.0e9)):.0f} '
            f'system_ready={getattr(self, "system_ready", True)} '
            f'dashboard_ready={getattr(self, "video_ready", True)} '
            f'localization_ready={self.localization_ready} '
            f'nav_server_mode={getattr(self, "follow_backend", "nav2") == "nav2"} '
            f'goal_required={goal_required} '
            f'goal_sent={reason == "goal_sent"} '
            f'path_received={getattr(self, "last_path_wall", -1.0e9) >= 0.0} '
            f'path_age_ms={self._age_ms(getattr(self, "last_path_wall", -1.0e9)):.0f} '
            f'cmd_vel_age_ms={self._age_ms(getattr(self, "last_cmd_vel_wall", -1.0e9)):.0f} '
            f'nonzero_cmd_age_ms={self._age_ms(getattr(self, "last_nonzero_cmd_wall", -1.0e9)):.0f} '
            f'odom_age_ms={self._age_ms(getattr(self, "last_odom_wall", -1.0e9)):.0f} '
            f'odom_motion={getattr(self, "odom_motion", False)} '
            f'distance_to_scout={distance_to_scout if distance_to_scout is not None else float("nan"):.3f} '
            f'distance_invalid_reason={distance_invalid_reason or "none"} '
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
