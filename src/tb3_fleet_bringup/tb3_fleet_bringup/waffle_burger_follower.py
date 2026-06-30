#!/usr/bin/env python3

from __future__ import annotations

import math
from typing import Optional, Tuple

import rclpy
from rclpy.node import Node
from rclpy.time import Time
from geometry_msgs.msg import TwistStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan


def _yaw_from_quat(q) -> float:
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


def _wrap_pi(a: float) -> float:
    while a > math.pi:
        a -= 2.0 * math.pi
    while a < -math.pi:
        a += 2.0 * math.pi
    return a


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _finite_min(values, default=float('inf')) -> float:
    best = default
    for v in values:
        if math.isfinite(v):
            best = min(best, float(v))
    return best


class WaffleBurgerFollower(Node):
    """Hybrid follower for the current Gazebo test.

    v19 changes the previous odom-only follower into a robust hybrid controller:

      1. Normal path: use /waffle/odom and /burger/odom to drive burger toward a
         formation point behind waffle.
      2. Fallback path: if odometry or scan is missing/stale, mirror /waffle/cmd_vel
         to /burger/cmd_vel with a scale factor. This guarantees that moving the
         waffle also makes the burger move during the basic test.
      3. Safety path: if /burger/scan exists and front is blocked, stop/turn.
         If leader odom exists and the robots are too close, stop/back up.

    Public command topics remain TwistStamped:
      /waffle/cmd_vel -> user command
      /burger/cmd_vel -> follower command
    """

    def __init__(self) -> None:
        super().__init__('waffle_burger_follower')

        self.declare_parameter('leader_name', 'waffle')
        self.declare_parameter('follower_name', 'burger')
        self.declare_parameter('leader_cmd_topic', '/waffle/cmd_vel')
        self.declare_parameter('leader_odom_topic', '/waffle/odom')
        self.declare_parameter('follower_odom_topic', '/burger/odom')
        self.declare_parameter('follower_scan_topic', '/burger/scan')
        self.declare_parameter('follower_cmd_topic', '/burger/cmd_vel')
        self.declare_parameter('cmd_frame_id', 'base_link')
        self.declare_parameter('control_rate_hz', 20.0)

        self.declare_parameter('follow_distance', 0.95)
        self.declare_parameter('goal_tolerance', 0.10)
        self.declare_parameter('min_leader_distance', 0.60)
        self.declare_parameter('hard_min_leader_distance', 0.42)

        self.declare_parameter('max_linear', 0.22)
        self.declare_parameter('max_angular', 1.30)
        self.declare_parameter('kp_dist', 0.95)
        self.declare_parameter('kp_yaw', 2.40)

        self.declare_parameter('front_stop_distance', 0.36)
        self.declare_parameter('front_slow_distance', 0.70)
        self.declare_parameter('front_sector_deg', 42.0)
        self.declare_parameter('side_sector_deg', 75.0)
        self.declare_parameter('avoid_turn_speed', 0.80)

        self.declare_parameter('stale_timeout_sec', 1.50)
        self.declare_parameter('cmd_relay_scale_linear', 0.90)
        self.declare_parameter('cmd_relay_scale_angular', 0.90)
        self.declare_parameter('cmd_relay_timeout_sec', 2.00)
        self.declare_parameter('use_cmd_relay_fallback', True)
        self.declare_parameter('allow_follow_without_scan', True)
        self.declare_parameter('log_every_n', 20)

        self.leader_name = str(self.get_parameter('leader_name').value)
        self.follower_name = str(self.get_parameter('follower_name').value)
        self.cmd_frame_id = str(self.get_parameter('cmd_frame_id').value)

        self.follow_distance = float(self.get_parameter('follow_distance').value)
        self.goal_tolerance = float(self.get_parameter('goal_tolerance').value)
        self.min_leader_distance = float(self.get_parameter('min_leader_distance').value)
        self.hard_min_leader_distance = float(self.get_parameter('hard_min_leader_distance').value)
        self.max_linear = abs(float(self.get_parameter('max_linear').value))
        self.max_angular = abs(float(self.get_parameter('max_angular').value))
        self.kp_dist = float(self.get_parameter('kp_dist').value)
        self.kp_yaw = float(self.get_parameter('kp_yaw').value)
        self.front_stop_distance = float(self.get_parameter('front_stop_distance').value)
        self.front_slow_distance = float(self.get_parameter('front_slow_distance').value)
        self.front_sector_deg = float(self.get_parameter('front_sector_deg').value)
        self.side_sector_deg = float(self.get_parameter('side_sector_deg').value)
        self.avoid_turn_speed = abs(float(self.get_parameter('avoid_turn_speed').value))
        self.stale_timeout_sec = float(self.get_parameter('stale_timeout_sec').value)
        self.cmd_relay_scale_linear = float(self.get_parameter('cmd_relay_scale_linear').value)
        self.cmd_relay_scale_angular = float(self.get_parameter('cmd_relay_scale_angular').value)
        self.cmd_relay_timeout_sec = float(self.get_parameter('cmd_relay_timeout_sec').value)
        self.use_cmd_relay_fallback = bool(self.get_parameter('use_cmd_relay_fallback').value)
        self.allow_follow_without_scan = bool(self.get_parameter('allow_follow_without_scan').value)
        self.log_every_n = max(0, int(self.get_parameter('log_every_n').value))

        control_rate_hz = max(1.0, float(self.get_parameter('control_rate_hz').value))

        leader_cmd_topic = str(self.get_parameter('leader_cmd_topic').value)
        leader_odom_topic = str(self.get_parameter('leader_odom_topic').value)
        follower_odom_topic = str(self.get_parameter('follower_odom_topic').value)
        follower_scan_topic = str(self.get_parameter('follower_scan_topic').value)
        follower_cmd_topic = str(self.get_parameter('follower_cmd_topic').value)

        self.leader_pose: Optional[Tuple[float, float, float]] = None
        self.follower_pose: Optional[Tuple[float, float, float]] = None
        self.scan: Optional[LaserScan] = None
        self.leader_stamp: Optional[Time] = None
        self.follower_stamp: Optional[Time] = None
        self.scan_stamp: Optional[Time] = None
        self.leader_cmd: Optional[TwistStamped] = None
        self.leader_cmd_stamp: Optional[Time] = None

        self.tick_count = 0
        self.last_mode = 'INIT'

        self.create_subscription(TwistStamped, leader_cmd_topic, self._on_leader_cmd, 20)
        self.create_subscription(Odometry, leader_odom_topic, self._on_leader_odom, 20)
        self.create_subscription(Odometry, follower_odom_topic, self._on_follower_odom, 20)
        self.create_subscription(LaserScan, follower_scan_topic, self._on_scan, 10)
        self.cmd_pub = self.create_publisher(TwistStamped, follower_cmd_topic, 10)

        self.create_timer(1.0 / control_rate_hz, self._control_tick)

        self.get_logger().info(
            'V19_HYBRID_WAFFLE_LEADER_BURGER_FOLLOWER_READY | '
            f'leader_cmd={leader_cmd_topic} leader_odom={leader_odom_topic} | '
            f'follower_odom={follower_odom_topic} scan={follower_scan_topic} cmd={follower_cmd_topic} | '
            f'follow_distance={self.follow_distance:.2f} cmd_relay_fallback={int(self.use_cmd_relay_fallback)} '
            f'allow_without_scan={int(self.allow_follow_without_scan)}'
        )

    def _on_leader_cmd(self, msg: TwistStamped) -> None:
        self.leader_cmd = msg
        self.leader_cmd_stamp = self.get_clock().now()

    def _on_leader_odom(self, msg: Odometry) -> None:
        p = msg.pose.pose.position
        yaw = _yaw_from_quat(msg.pose.pose.orientation)
        self.leader_pose = (p.x, p.y, yaw)
        self.leader_stamp = self.get_clock().now()

    def _on_follower_odom(self, msg: Odometry) -> None:
        p = msg.pose.pose.position
        yaw = _yaw_from_quat(msg.pose.pose.orientation)
        self.follower_pose = (p.x, p.y, yaw)
        self.follower_stamp = self.get_clock().now()

    def _on_scan(self, msg: LaserScan) -> None:
        self.scan = msg
        self.scan_stamp = self.get_clock().now()

    def _age(self, stamp: Optional[Time], now: Time) -> float:
        if stamp is None:
            return float('inf')
        return (now - stamp).nanoseconds * 1e-9

    def _fresh(self, stamp: Optional[Time], now: Time, timeout: float) -> bool:
        return self._age(stamp, now) <= timeout

    def _scan_min_sector(self, center_rad: float, half_width_rad: float) -> float:
        if self.scan is None:
            return float('inf')
        best = float('inf')
        a = self.scan.angle_min
        inc = self.scan.angle_increment
        rmin = self.scan.range_min if self.scan.range_min > 0.0 else 0.0
        rmax = self.scan.range_max if self.scan.range_max > 0.0 else float('inf')
        for r in self.scan.ranges:
            if math.isfinite(r):
                da = _wrap_pi(a - center_rad)
                if abs(da) <= half_width_rad and rmin <= r <= rmax:
                    best = min(best, float(r))
            a += inc
        return best

    def _make_cmd(self, vx: float, wz: float) -> TwistStamped:
        msg = TwistStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.cmd_frame_id
        msg.twist.linear.x = float(_clamp(vx, -self.max_linear, self.max_linear))
        msg.twist.angular.z = float(_clamp(wz, -self.max_angular, self.max_angular))
        return msg

    def _publish(self, vx: float, wz: float, mode: str, extra: str = '') -> None:
        self.cmd_pub.publish(self._make_cmd(vx, wz))
        self._log_mode(mode, vx, wz, extra)

    def _publish_zero(self, mode: str, extra: str = '') -> None:
        self._publish(0.0, 0.0, mode, 'zero=1' + ((' | ' + extra) if extra else ''))

    def _log_mode(self, mode: str, vx: float, wz: float, extra: str = '') -> None:
        self.tick_count += 1
        changed = mode != self.last_mode
        if changed:
            self.last_mode = mode
        if changed or (self.log_every_n > 0 and self.tick_count % self.log_every_n == 0):
            msg = f'FOLLOW_CTRL | mode={mode} | vx={vx:.3f} wz={wz:.3f}'
            if extra:
                msg += ' | ' + extra
            if mode in ('FRONT_BLOCKED', 'TOO_CLOSE_BACKUP', 'WAIT_INPUT'):
                self.get_logger().warn(msg)
            else:
                self.get_logger().info(msg)

    def _safety_filter(self, vx: float, wz: float, now: Time) -> Tuple[float, float, str]:
        """Apply scan/front safety when available. Return vx,wz,extra."""
        scan_fresh = self._fresh(self.scan_stamp, now, self.stale_timeout_sec)
        if not scan_fresh:
            if self.allow_follow_without_scan:
                return vx, wz, f'scan_stale_age={self._age(self.scan_stamp, now):.2f}s ignored=1'
            return 0.0, 0.0, f'scan_stale_age={self._age(self.scan_stamp, now):.2f}s stop=1'

        front_half = math.radians(self.front_sector_deg) * 0.5
        side_half = math.radians(self.side_sector_deg) * 0.5
        front_min = self._scan_min_sector(0.0, front_half)
        left_min = self._scan_min_sector(math.pi / 2.0, side_half)
        right_min = self._scan_min_sector(-math.pi / 2.0, side_half)

        if front_min < self.front_stop_distance and vx > 0.0:
            turn_sign = -1.0 if left_min > right_min else 1.0
            return 0.0, turn_sign * self.avoid_turn_speed, f'front={front_min:.2f} left={left_min:.2f} right={right_min:.2f}'

        if front_min < self.front_slow_distance and vx > 0.0:
            scale = _clamp(
                (front_min - self.front_stop_distance) / max(1e-6, self.front_slow_distance - self.front_stop_distance),
                0.20,
                1.0,
            )
            vx *= scale
            return vx, wz, f'front={front_min:.2f} slow_scale={scale:.2f}'

        return vx, wz, f'front={front_min:.2f}'

    def _relay_leader_cmd(self, now: Time, reason: str) -> bool:
        if not self.use_cmd_relay_fallback:
            return False
        if self.leader_cmd is None or not self._fresh(self.leader_cmd_stamp, now, self.cmd_relay_timeout_sec):
            return False

        src = self.leader_cmd.twist
        vx = src.linear.x * self.cmd_relay_scale_linear
        wz = src.angular.z * self.cmd_relay_scale_angular
        vx, wz, extra = self._safety_filter(vx, wz, now)

        # If scan forced a turn, label it clearly.
        mode = 'CMD_RELAY_FOLLOW'
        if extra.startswith('front=') and 'left=' in extra and vx == 0.0:
            mode = 'FRONT_BLOCKED'

        self._publish(vx, wz, mode, f'reason={reason} | relay_from=/waffle/cmd_vel | {extra}')
        return True

    def _control_tick(self) -> None:
        now = self.get_clock().now()
        leader_fresh = self._fresh(self.leader_stamp, now, self.stale_timeout_sec)
        follower_fresh = self._fresh(self.follower_stamp, now, self.stale_timeout_sec)

        if not leader_fresh or not follower_fresh:
            if self._relay_leader_cmd(now, reason=f'odom_stale leader_age={self._age(self.leader_stamp, now):.2f}s follower_age={self._age(self.follower_stamp, now):.2f}s'):
                return
            self._publish_zero(
                'WAIT_INPUT',
                f'leader_odom_age={self._age(self.leader_stamp, now):.2f}s follower_odom_age={self._age(self.follower_stamp, now):.2f}s'
            )
            return

        assert self.leader_pose is not None
        assert self.follower_pose is not None
        lx, ly, lyaw = self.leader_pose
        fx, fy, fyaw = self.follower_pose

        target_x = lx - self.follow_distance * math.cos(lyaw)
        target_y = ly - self.follow_distance * math.sin(lyaw)

        dx = target_x - fx
        dy = target_y - fy
        goal_dist = math.hypot(dx, dy)
        leader_dist = math.hypot(lx - fx, ly - fy)
        target_heading = math.atan2(dy, dx)
        yaw_err = _wrap_pi(target_heading - fyaw)

        if leader_dist < self.hard_min_leader_distance:
            self._publish(-0.05, 0.0, 'TOO_CLOSE_BACKUP', f'leader_dist={leader_dist:.2f}')
            return
        if leader_dist < self.min_leader_distance:
            self._publish_zero('TOO_CLOSE_STOP', f'leader_dist={leader_dist:.2f}')
            return

        if goal_dist < self.goal_tolerance:
            # If leader is still commanding motion, don't freeze forever; relay lightly.
            if self._relay_leader_cmd(now, reason=f'at_slot goal_dist={goal_dist:.2f}'):
                return
            self._publish_zero('HOLD_DISTANCE', f'goal_dist={goal_dist:.2f} leader_dist={leader_dist:.2f}')
            return

        vx = self.kp_dist * goal_dist
        vx *= max(0.10, math.cos(yaw_err))
        if abs(yaw_err) > math.radians(70.0):
            vx = min(vx, 0.05)
        vx = _clamp(vx, 0.0, self.max_linear)
        wz = _clamp(self.kp_yaw * yaw_err, -self.max_angular, self.max_angular)

        vx, wz, safety_extra = self._safety_filter(vx, wz, now)
        mode = 'FOLLOWING'
        if safety_extra.startswith('front=') and 'left=' in safety_extra and vx == 0.0:
            mode = 'FRONT_BLOCKED'

        self._publish(
            vx,
            wz,
            mode,
            f'goal_dist={goal_dist:.2f} leader_dist={leader_dist:.2f} yaw_err_deg={math.degrees(yaw_err):.1f} | {safety_extra}'
        )


def main() -> None:
    rclpy.init()
    node = WaffleBurgerFollower()
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
