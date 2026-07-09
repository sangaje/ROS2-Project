#!/usr/bin/env python3
"""AMCL initial-localization spin state machine.

A hardcoded initial pose only works when it matches where the robot was
actually placed; when it doesn't, AMCL still converges confidently, just to
the wrong spot near that seed. This node calls AMCL's own
`/reinitialize_global_localization` service to spread particles across the
whole map instead, then drives a verified in-place rotation so scan matching
gets more than one viewpoint to disambiguate against, and only declares
"ready" once `/amcl_pose` covariance has genuinely settled.

State machine:

    WAIT_MAP -> WAIT_AMCL_ACTIVE -> WAIT_SCAN -> WAIT_TF
      -> CHECK_LOCALIZATION_QUALITY (skips the spin only on a re-entry that
         is already converged -- never on a true first pass, since AMCL's
         seeded set_initial_pose covariance can look artificially good
         before the robot has moved at all)
      -> SPIN (odometry-yaw-integrated, not time-based) -> SETTLE
      -> CHECK_LOCALIZATION -> READY_FOR_NAV
                             -> RETRY_SPIN (bounded) -> FAIL_SAFE

Once READY_FOR_NAV is reached the `ready` topic latches true permanently --
this node never re-triggers motion afterward. Re-spinning while Nav2 might
already be driving the robot would be a hazard, not a fix; downstream
consumers (fleet_path_coordinator, fleet_follower) are expected to gate their
own goal-sending on this flag instead.
"""
from __future__ import annotations

import math
from enum import Enum, auto
from typing import Optional

import rclpy
from geometry_msgs.msg import PoseWithCovarianceStamped, Twist, TwistStamped
from lifecycle_msgs.msg import State as LifecycleState
from lifecycle_msgs.srv import GetState
from nav_msgs.msg import Odometry, OccupancyGrid
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from rclpy.time import Time
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Bool
from std_srvs.srv import Empty
from tf2_ros import Buffer, TransformListener


def wrap_angle(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def yaw_from_quaternion(q) -> float:
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


class State(Enum):
    WAIT_MAP = auto()
    WAIT_AMCL_ACTIVE = auto()
    WAIT_SCAN = auto()
    WAIT_ODOM = auto()
    WAIT_TF = auto()
    CHECK_LOCALIZATION_QUALITY = auto()
    SPIN = auto()
    SETTLE = auto()
    CHECK_LOCALIZATION = auto()
    RETRY_SPIN = auto()
    READY_FOR_NAV = auto()
    FAIL_SAFE = auto()


class GlobalLocalizeKickstart(Node):

    def __init__(self) -> None:
        super().__init__('global_localize_kickstart')

        self.declare_parameter('reinit_service', '/reinitialize_global_localization')
        self.declare_parameter('map_topic', '/map')
        self.declare_parameter('odom_topic', '/odom')
        self.declare_parameter('scan_topic', '/scan')
        self.declare_parameter('cmd_vel_topic', '/cmd_vel')
        self.declare_parameter('use_stamped_cmd_vel', True)
        self.declare_parameter('amcl_pose_topic', '/amcl_pose')
        self.declare_parameter('amcl_lifecycle_service', '/amcl/get_state')
        self.declare_parameter('base_frame', 'base_footprint')
        self.declare_parameter('global_frame', 'map')
        self.declare_parameter('ready_topic', 'localization_ready')

        self.declare_parameter('spin_enabled', True)
        self.declare_parameter('require_valid_map', True)
        self.declare_parameter('min_known_map_cells', 100)
        self.declare_parameter('require_scan_before_spin', True)
        self.declare_parameter('require_odom_before_spin', True)
        self.declare_parameter('require_amcl_before_spin', True)
        self.declare_parameter('spin_speed_rad_s', 0.35)
        self.declare_parameter('spin_target_angle_rad', 6.45)
        self.declare_parameter('spin_margin_rad', 0.0)
        self.declare_parameter('spin_timeout_sec', 30.0)
        self.declare_parameter('settle_duration_sec', 2.0)
        # 좌우 바퀴 응답이 비대칭인 로봇은 angular.z 만 명령해도 제자리
        # 회전이 아니라 호를 그리며 실제로 이동한다 -- "제자리" 스핀이라는
        # 전제가 깨지므로, odom 상 시작 위치에서 이 이상 벗어나면 spin을
        # 중단하고 재시도(반복해도 계속 드리프트하면 결국 FAIL_SAFE).
        self.declare_parameter('spin_max_drift_m', 0.35)

        self.declare_parameter('localization_cov_xy_threshold', 0.35)
        self.declare_parameter('localization_cov_yaw_threshold', 0.25)
        self.declare_parameter('localization_stable_duration_sec', 1.5)
        self.declare_parameter('localization_check_timeout_sec', 7.0)

        self.declare_parameter('max_spin_retries', 2)
        self.declare_parameter('amcl_active_poll_period_sec', 1.0)
        self.declare_parameter('max_scan_age_sec', 1.0)
        self.declare_parameter('max_odom_age_sec', 1.0)
        self.declare_parameter('tick_period_sec', 0.1)
        # Master guarantee: no matter which WAIT_* precondition is stuck
        # (map/AMCL-active/scan/odom/TF), the spin gets attempted anyway
        # once this much time has passed since this node started. Getting
        # the preconditions right is *preferred* -- spinning against a
        # real map with fresh sensors converges better -- but this node's
        # one non-negotiable job is to eventually spin and let Nav2 through,
        # not to wait forever for a perfect precondition that never arrives.
        self.declare_parameter('force_spin_after_sec', 20.0)

        get = self.get_parameter
        self.reinit_service_name = str(get('reinit_service').value)
        self.map_topic = str(get('map_topic').value)
        self.odom_topic = str(get('odom_topic').value)
        self.scan_topic = str(get('scan_topic').value)
        self.cmd_vel_topic = str(get('cmd_vel_topic').value)
        self.use_stamped = bool(get('use_stamped_cmd_vel').value)
        self.amcl_pose_topic = str(get('amcl_pose_topic').value)
        self.amcl_lifecycle_service_name = str(get('amcl_lifecycle_service').value)
        self.base_frame = str(get('base_frame').value)
        self.global_frame = str(get('global_frame').value)
        self.ready_topic = str(get('ready_topic').value)

        self.spin_enabled = bool(get('spin_enabled').value)
        self.require_valid_map = bool(get('require_valid_map').value)
        self.min_known_map_cells = max(0, int(get('min_known_map_cells').value))
        self.require_scan_before_spin = bool(get('require_scan_before_spin').value)
        self.require_odom_before_spin = bool(get('require_odom_before_spin').value)
        self.require_amcl_before_spin = bool(get('require_amcl_before_spin').value)
        self.spin_speed = abs(float(get('spin_speed_rad_s').value))
        self.spin_target_angle = max(0.0, float(get('spin_target_angle_rad').value))
        self.spin_margin = max(0.0, float(get('spin_margin_rad').value))
        self.spin_timeout_sec = max(1.0, float(get('spin_timeout_sec').value))
        self.settle_duration_sec = max(0.0, float(get('settle_duration_sec').value))
        self.spin_max_drift_m = max(0.05, float(get('spin_max_drift_m').value))

        self.cov_xy_threshold = max(0.0, float(get('localization_cov_xy_threshold').value))
        self.cov_yaw_threshold = max(0.0, float(get('localization_cov_yaw_threshold').value))
        self.stable_duration_sec = max(
            0.0, float(get('localization_stable_duration_sec').value)
        )
        self.check_timeout_sec = max(
            0.0, float(get('localization_check_timeout_sec').value)
        )

        self.max_spin_retries = max(0, int(get('max_spin_retries').value))
        self.amcl_active_poll_period_sec = max(
            0.1, float(get('amcl_active_poll_period_sec').value)
        )
        self.max_scan_age_sec = max(0.05, float(get('max_scan_age_sec').value))
        self.max_odom_age_sec = max(0.05, float(get('max_odom_age_sec').value))
        self.tick_period_sec = max(0.02, float(get('tick_period_sec').value))
        self.force_spin_after_sec = max(
            1.0, float(get('force_spin_after_sec').value)
        )

        latched_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
        )
        scan_qos = QoSProfile(
            depth=5,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
        )

        self.ready_pub = self.create_publisher(Bool, self.ready_topic, latched_qos)
        if self.use_stamped:
            self.cmd_pub = self.create_publisher(TwistStamped, self.cmd_vel_topic, 10)
        else:
            self.cmd_pub = self.create_publisher(Twist, self.cmd_vel_topic, 10)

        self.reinit_client = self.create_client(Empty, self.reinit_service_name)
        self.amcl_state_client = self.create_client(
            GetState, self.amcl_lifecycle_service_name
        )

        self.map_received = False
        self.create_subscription(OccupancyGrid, self.map_topic, self._on_map, latched_qos)
        self.create_subscription(
            PoseWithCovarianceStamped, self.amcl_pose_topic, self._on_amcl_pose, 10
        )
        self.create_subscription(Odometry, self.odom_topic, self._on_odom, 10)
        self.create_subscription(LaserScan, self.scan_topic, self._on_scan, scan_qos)

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.state = State.WAIT_MAP
        self.total_attempts = 0
        self.retry_count = 0
        self.reinit_in_flight = False
        self._amcl_state_call_in_flight = False
        self._last_amcl_poll_wall = 0.0
        self.node_start_wall = self._now()
        self._forced_spin = False

        self.last_scan_wall: Optional[float] = None
        self.last_odom_wall: Optional[float] = None
        self.last_odom_yaw: Optional[float] = None
        self.last_odom_xy: Optional[tuple] = None
        self.spin_start_xy: Optional[tuple] = None
        self.spin_direction = 1.0
        self.accumulated_yaw = 0.0
        self._last_progress_octant = -1
        self.spin_start_wall = 0.0
        self.settle_start_wall = 0.0
        self.check_start_wall = 0.0
        self.good_since_wall: Optional[float] = None
        self.last_pose_cov = None
        self.done = False

        self._publish_ready(False)
        self.timer = self.create_timer(self.tick_period_sec, self._tick)

        self.get_logger().info(
            'GLOBAL_LOCALIZE_READY | '
            f'map_topic={self.map_topic} scan_topic={self.scan_topic} '
            f'odom_topic={self.odom_topic} '
            f'spin_enabled={self.spin_enabled} spin_speed={self.spin_speed:.2f}rad/s '
            f'target={self.spin_target_angle + self.spin_margin:.2f}rad '
            f'max_spin_retries={self.max_spin_retries} '
            f'require_valid_map={self.require_valid_map} '
            f'min_known_map_cells={self.min_known_map_cells} '
            f'cmd_vel_type={"TwistStamped" if self.use_stamped else "Twist"}'
        )

    def _now(self) -> float:
        return self.get_clock().now().nanoseconds * 1.0e-9

    # -- subscriptions ----------------------------------------------------

    def _on_map(self, msg: OccupancyGrid) -> None:
        width = int(msg.info.width)
        height = int(msg.info.height)
        cell_count = width * height
        valid_shape = width > 0 and height > 0 and len(msg.data) == cell_count
        known_cells = sum(1 for cell in msg.data if int(cell) != -1) if valid_shape else 0
        valid = valid_shape and (
            not self.require_valid_map
            or known_cells >= self.min_known_map_cells
        )
        if not valid:
            self.get_logger().info(
                'GLOBAL_LOCALIZE_WAIT_MAP | invalid/empty map '
                f'width={width} height={height} data={len(msg.data)} '
                f'known={known_cells}/{self.min_known_map_cells}',
                throttle_duration_sec=5.0,
            )
            return
        if not self.map_received:
            self.get_logger().warning(
                'GLOBAL_LOCALIZE_VALID_MAP_RECEIVED | '
                f'width={width} height={height} known={known_cells}'
            )
        self.map_received = True

    def _on_scan(self, _msg: LaserScan) -> None:
        self.last_scan_wall = self._now()

    def _on_odom(self, msg: Odometry) -> None:
        yaw = yaw_from_quaternion(msg.pose.pose.orientation)
        if self.state == State.SPIN and self.last_odom_yaw is not None:
            delta = wrap_angle(yaw - self.last_odom_yaw)
            self.accumulated_yaw += abs(delta)
        self.last_odom_yaw = yaw
        self.last_odom_xy = (
            msg.pose.pose.position.x, msg.pose.pose.position.y,
        )
        self.last_odom_wall = self._now()

    def _on_amcl_pose(self, msg: PoseWithCovarianceStamped) -> None:
        cov = msg.pose.covariance
        xy_cov = max(float(cov[0]), float(cov[7]))
        yaw_cov = float(cov[35])
        if not all(math.isfinite(value) for value in (xy_cov, yaw_cov)):
            return
        self.last_pose_cov = {
            'xy_cov': xy_cov,
            'yaw_cov': yaw_cov,
            'x': float(msg.pose.pose.position.x),
            'y': float(msg.pose.pose.position.y),
        }

    # -- shared helpers -----------------------------------------------------

    def _scan_fresh(self) -> bool:
        if not self.require_scan_before_spin:
            return True
        if self.last_scan_wall is None:
            return False
        return (self._now() - self.last_scan_wall) <= self.max_scan_age_sec

    def _odom_fresh(self) -> bool:
        if not self.require_odom_before_spin:
            return True
        if self.last_odom_wall is None or self.last_odom_yaw is None:
            return False
        return (self._now() - self.last_odom_wall) <= self.max_odom_age_sec

    def _tf_ok(self) -> bool:
        try:
            return self.tf_buffer.can_transform(
                self.global_frame, self.base_frame, Time()
            )
        except Exception:  # noqa: BLE001
            return False

    def _covariance_ok(self) -> bool:
        if self.last_pose_cov is None:
            return False
        return (
            self.last_pose_cov['xy_cov'] <= self.cov_xy_threshold
            and self.last_pose_cov['yaw_cov'] <= self.cov_yaw_threshold
        )

    def _update_stability_tracking(self) -> bool:
        """True once covariance has been continuously good for
        stable_duration_sec (resets on any bad sample)."""
        now = self._now()
        if self._covariance_ok():
            if self.good_since_wall is None:
                self.good_since_wall = now
            return (now - self.good_since_wall) >= self.stable_duration_sec
        self.good_since_wall = None
        return False

    def _publish_ready(self, ready: bool) -> None:
        self.ready_pub.publish(Bool(data=ready))

    def _publish_twist(self, angular_z: float) -> None:
        if self.use_stamped:
            message = TwistStamped()
            message.header.stamp = self.get_clock().now().to_msg()
            message.header.frame_id = self.base_frame
            message.twist.angular.z = angular_z
            self.cmd_pub.publish(message)
        else:
            message = Twist()
            message.angular.z = angular_z
            self.cmd_pub.publish(message)

    def _transition(self, new_state: State) -> None:
        if new_state == self.state:
            return
        self.get_logger().warning(
            f'GLOBAL_LOCALIZE_STATE | from={self.state.name} to={new_state.name}'
        )
        self.state = new_state
        hook = getattr(self, f'_on_enter_{new_state.name.lower()}', None)
        if hook is not None:
            hook()

    # -- state entry hooks ----------------------------------------------

    def _on_enter_ready_for_nav(self) -> None:
        self._publish_twist(0.0)
        self._publish_ready(True)
        self.done = True
        self.get_logger().warning('GLOBAL_LOCALIZE_READY_FOR_NAV')

    def _on_enter_fail_safe(self) -> None:
        self._publish_twist(0.0)
        self._publish_ready(False)
        self.good_since_wall = None
        self.get_logger().error(
            'GLOBAL_LOCALIZE_FAIL_SAFE | automatic retries exhausted, '
            'still watching /amcl_pose passively (will not spin again)'
        )

    # -- main dispatch ----------------------------------------------------

    _PRE_SPIN_STATES = (
        State.WAIT_MAP,
        State.WAIT_AMCL_ACTIVE,
        State.WAIT_SCAN,
        State.WAIT_ODOM,
        State.WAIT_TF,
        State.CHECK_LOCALIZATION_QUALITY,
    )

    def _tick(self) -> None:
        if self.done:
            return
        if (
            not self._forced_spin
            and self.state in self._PRE_SPIN_STATES
            and self._now() - self.node_start_wall >= self.force_spin_after_sec
        ):
            self._forced_spin = True
            self.get_logger().error(
                'GLOBAL_LOCALIZE_FORCE_SPIN | '
                f'stuck in {self.state.name} for '
                f'{self.force_spin_after_sec:.0f}s+ since startup -- forcing '
                'the spin attempt anyway. Blocking condition snapshot: '
                f'map_received={self.map_received} '
                f'scan_fresh={self._scan_fresh()} '
                f'odom_fresh={self._odom_fresh()} tf_ok={self._tf_ok()}'
            )
            self._transition(State.CHECK_LOCALIZATION_QUALITY)
        getattr(self, f'_tick_{self.state.name.lower()}')()

    def _tick_wait_map(self) -> None:
        if self.map_received:
            self._transition(State.WAIT_AMCL_ACTIVE)
            return
        self.get_logger().info(
            f'GLOBAL_LOCALIZE_WAIT_MAP | no valid map yet on {self.map_topic}',
            throttle_duration_sec=5.0,
        )

    def _tick_wait_amcl_active(self) -> None:
        if not self.require_amcl_before_spin:
            self._transition(State.WAIT_SCAN)
            return
        now = self._now()
        if now - self._last_amcl_poll_wall < self.amcl_active_poll_period_sec:
            return
        self._last_amcl_poll_wall = now
        if self._amcl_state_call_in_flight:
            return
        if not self.amcl_state_client.service_is_ready():
            self.get_logger().info(
                'GLOBAL_LOCALIZE_WAIT_AMCL | service not ready: '
                f'{self.amcl_lifecycle_service_name}',
                throttle_duration_sec=5.0,
            )
            return
        self._amcl_state_call_in_flight = True
        future = self.amcl_state_client.call_async(GetState.Request())
        future.add_done_callback(self._on_amcl_get_state)

    def _on_amcl_get_state(self, future) -> None:
        self._amcl_state_call_in_flight = False
        try:
            result = future.result()
        except Exception as error:  # noqa: BLE001
            self.get_logger().warning(f'GLOBAL_LOCALIZE_AMCL_STATE_ERROR | {error}')
            return
        if result.current_state.id == LifecycleState.PRIMARY_STATE_ACTIVE:
            self._transition(State.WAIT_SCAN)

    def _tick_wait_scan(self) -> None:
        if self._scan_fresh():
            self._transition(State.WAIT_ODOM)
            return
        self.get_logger().info(
            f'GLOBAL_LOCALIZE_WAIT_SCAN | no fresh scan on {self.scan_topic}',
            throttle_duration_sec=5.0,
        )

    def _tick_wait_odom(self) -> None:
        if self._odom_fresh():
            self._transition(State.WAIT_TF)
            return
        self.get_logger().info(
            f'GLOBAL_LOCALIZE_WAIT_ODOM | no fresh odom/yaw on {self.odom_topic}',
            throttle_duration_sec=5.0,
        )

    def _tick_wait_tf(self) -> None:
        if self._tf_ok():
            self._transition(State.CHECK_LOCALIZATION_QUALITY)
            return
        self.get_logger().info(
            f'GLOBAL_LOCALIZE_WAIT_TF | {self.global_frame}->{self.base_frame} '
            'not available yet',
            throttle_duration_sec=5.0,
        )

    def _tick_check_localization_quality(self) -> None:
        # Never skip the spin on a genuine first pass -- AMCL's seeded
        # set_initial_pose covariance can look artificially good before the
        # robot has moved at all. Only a process that has already completed
        # a real reinit+spin+converge cycle may skip a re-spin here (e.g.
        # this node respawned while AMCL was already localized).
        if self.total_attempts > 0 and self._update_stability_tracking():
            self.get_logger().warning(
                'GLOBAL_LOCALIZE_ALREADY_CONVERGED | skipping spin'
            )
            self._transition(State.READY_FOR_NAV)
            return
        self.good_since_wall = None
        self._start_spin()

    def _start_spin(self) -> None:
        if self.reinit_in_flight:
            return
        if not self.reinit_client.service_is_ready():
            self.get_logger().info(
                'GLOBAL_LOCALIZE_WAIT_REINIT_SERVICE | '
                f'{self.reinit_service_name} not ready',
                throttle_duration_sec=5.0,
            )
            return
        self.total_attempts += 1
        self.reinit_in_flight = True
        self.get_logger().warning(
            f'GLOBAL_LOCALIZE_ATTEMPT | attempt={self.total_attempts} '
            f'retry={self.retry_count}/{self.max_spin_retries}'
        )
        future = self.reinit_client.call_async(Empty.Request())
        future.add_done_callback(self._on_reinitialized)

    def _on_reinitialized(self, future) -> None:
        self.reinit_in_flight = False
        try:
            future.result()
        except Exception as error:  # noqa: BLE001
            self.get_logger().error(f'GLOBAL_LOCALIZE_REINIT_FAILED | {error}')
            return
        self.get_logger().warning(
            f'GLOBAL_LOCALIZE_REINITIALIZED | attempt={self.total_attempts}'
        )
        self.accumulated_yaw = 0.0
        self.spin_start_xy = self.last_odom_xy
        self.spin_direction = 1.0 if self.total_attempts % 2 == 1 else -1.0
        self._last_progress_octant = -1
        self.spin_start_wall = self._now()
        if not self.spin_enabled:
            self.settle_start_wall = self._now()
            self._transition(State.SETTLE)
            return
        self.get_logger().warning(
            'GLOBAL_LOCALIZE_SPIN_START | '
            f'attempt={self.total_attempts} '
            f'direction={"ccw" if self.spin_direction > 0.0 else "cw"} '
            f'cmd_vel={self.cmd_vel_topic} '
            f'type={"TwistStamped" if self.use_stamped else "Twist"} '
            f'target={math.degrees(self.spin_target_angle + self.spin_margin):.0f}deg'
        )
        self._transition(State.SPIN)

    def _tick_spin(self) -> None:
        if not self._scan_fresh() or not self._odom_fresh() or not self._tf_ok():
            self._publish_twist(0.0)
            if not self._scan_fresh():
                fallback = State.WAIT_SCAN
            elif not self._odom_fresh():
                fallback = State.WAIT_ODOM
            else:
                fallback = State.WAIT_TF
            self.get_logger().warning(
                'GLOBAL_LOCALIZE_SPIN_PAUSED | scan/odom/TF dropped out mid-spin, '
                f'falling back to {fallback.name} (not counted as a failed attempt)'
            )
            self._transition(fallback)
            return

        if self.spin_start_xy is not None and self.last_odom_xy is not None:
            drift = math.hypot(
                self.last_odom_xy[0] - self.spin_start_xy[0],
                self.last_odom_xy[1] - self.spin_start_xy[1],
            )
            if drift > self.spin_max_drift_m:
                self._publish_twist(0.0)
                self.get_logger().error(
                    'GLOBAL_LOCALIZE_SPIN_DRIFT | '
                    f'attempt={self.total_attempts} drift={drift:.2f}m > '
                    f'{self.spin_max_drift_m:.2f}m -- left/right wheel '
                    'response is asymmetric enough that "spin in place" is '
                    'actually arcing away from the seeded position'
                )
                self.retry_count += 1
                self._transition(State.RETRY_SPIN)
                return

        target = self.spin_target_angle + self.spin_margin
        elapsed = self._now() - self.spin_start_wall

        if self.accumulated_yaw >= target:
            self._publish_twist(0.0)
            self.get_logger().warning(
                f'GLOBAL_LOCALIZE_SPIN_COMPLETE | attempt={self.total_attempts} '
                f'rotated={math.degrees(self.accumulated_yaw):.0f}deg '
                f'elapsed={elapsed:.1f}s'
            )
            self.settle_start_wall = self._now()
            self._transition(State.SETTLE)
            return

        if elapsed >= self.spin_timeout_sec:
            self._publish_twist(0.0)
            self.get_logger().error(
                f'GLOBAL_LOCALIZE_SPIN_TIMEOUT | attempt={self.total_attempts} '
                f'rotated={math.degrees(self.accumulated_yaw):.0f}deg of '
                f'{math.degrees(target):.0f}deg'
            )
            self.retry_count += 1
            self._transition(State.RETRY_SPIN)
            return

        octant = int(self.accumulated_yaw / (math.pi / 4.0))
        if octant != self._last_progress_octant:
            self._last_progress_octant = octant
            self.get_logger().info(
                'GLOBAL_LOCALIZE_SPIN_PROGRESS | '
                f'{math.degrees(self.accumulated_yaw):.0f}/'
                f'{math.degrees(target):.0f}deg'
            )
        self._publish_twist(self.spin_direction * self.spin_speed)

    def _tick_settle(self) -> None:
        self._publish_twist(0.0)
        if self._now() - self.settle_start_wall >= self.settle_duration_sec:
            self.get_logger().info('GLOBAL_LOCALIZE_SETTLING_DONE')
            self.good_since_wall = None
            self.check_start_wall = self._now()
            self._transition(State.CHECK_LOCALIZATION)

    def _tick_check_localization(self) -> None:
        if self._update_stability_tracking():
            self.get_logger().warning(
                'GLOBAL_LOCALIZE_LOCALIZED | '
                f'xy_cov={self.last_pose_cov["xy_cov"]:.4f} '
                f'yaw_cov={self.last_pose_cov["yaw_cov"]:.4f}'
            )
            self._transition(State.READY_FOR_NAV)
            return
        if self._now() - self.check_start_wall >= self.check_timeout_sec:
            cov_text = 'no_amcl_pose'
            if self.last_pose_cov is not None:
                cov_text = (
                    f'xy_cov={self.last_pose_cov["xy_cov"]:.4f} '
                    f'yaw_cov={self.last_pose_cov["yaw_cov"]:.4f}'
                )
            self.get_logger().warning(
                f'GLOBAL_LOCALIZE_NOT_CONVERGED | attempt={self.total_attempts} '
                f'{cov_text}'
            )
            self.retry_count += 1
            self._transition(State.RETRY_SPIN)

    def _tick_retry_spin(self) -> None:
        if self.retry_count > self.max_spin_retries:
            self._transition(State.FAIL_SAFE)
            return
        self._start_spin()

    def _tick_ready_for_nav(self) -> None:
        pass

    def _tick_fail_safe(self) -> None:
        if self._update_stability_tracking():
            self.get_logger().warning(
                'GLOBAL_LOCALIZE_RECOVERED_AFTER_FAILSAFE | covariance became '
                'stable without another automatic spin'
            )
            self._transition(State.READY_FOR_NAV)

    def destroy_node(self) -> None:
        try:
            self._publish_twist(0.0)
        except Exception:  # noqa: BLE001
            pass
        super().destroy_node()


def main() -> None:
    rclpy.init()
    node = GlobalLocalizeKickstart()
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
