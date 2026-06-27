#!/usr/bin/env python3
"""
check_lidar60_freshness.py
==========================
Diagnostic: subscribe /scan and /rl_policy_scan_60, print stamp delta and
front/left/rear/right sector values.

Warns if the debug topic is stale compared to /scan.

Key design note:
  /rl_policy_scan_60 is a VISUALIZATION topic published every N steps.
  It is NOT the source of truth for the policy observation.  The actual
  observation is built from ros_interface.scan directly inside _get_obs().
  A stale /rl_policy_scan_60 only means the debug topic hasn't been published
  yet -- it does NOT mean the policy used a stale scan.

  If `scan_age_sec` in metrics_compact.csv is consistently > 0.35 s, THAT
  is a real problem (actual observation stale).

Usage:
  ros2 run turtlebot3_rl_training check_lidar60_freshness
  or:
  python3 scripts/check_lidar60_freshness.py

Env:
  ROS_DOMAIN_ID            -- match training domain (default 22)
  TB3_RL_SCAN_TOPIC        -- raw scan (default /scan)
  TB3_RL_POLICY_SCAN_60    -- policy debug topic (default /rl_policy_scan_60)
  TB3_RL_FRESHNESS_WARN_SEC -- warn threshold in seconds (default 0.40)
"""

import math
import os
import sys
import time

try:
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import LaserScan
except ImportError:
    sys.exit("rclpy not found. Source ROS2 environment first.")


def _sector_min(ranges, angles, center_rad, half_rad=math.radians(30.0)):
    import numpy as np
    r = np.asarray(ranges, dtype=float)
    a = np.asarray(angles, dtype=float)
    diff = np.abs(np.arctan2(np.sin(a - center_rad), np.cos(a - center_rad)))
    mask = diff <= half_rad
    valid = np.isfinite(r[mask]) & (r[mask] > 0.0)
    if not np.any(valid):
        return 999.0
    return float(np.min(r[mask][valid]))


def _stamp_sec(msg):
    try:
        s = msg.header.stamp
        return float(s.sec) + float(s.nanosec) * 1e-9
    except Exception:
        return -1.0


class LidarFreshnessChecker(Node):
    def __init__(self):
        super().__init__("lidar60_freshness_checker")
        self._raw_scan = None
        self._policy_scan = None
        self._raw_recv = 0
        self._policy_recv = 0
        self._warn_sec = float(os.environ.get("TB3_RL_FRESHNESS_WARN_SEC", "0.40"))
        raw_topic = os.environ.get("TB3_RL_SCAN_TOPIC", "/scan")
        policy_topic = os.environ.get("TB3_RL_POLICY_SCAN_60", "/rl_policy_scan_60")

        from rclpy.qos import QoSProfile, HistoryPolicy, ReliabilityPolicy, DurabilityPolicy
        sensor_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )
        self.create_subscription(LaserScan, raw_topic, self._raw_cb, sensor_qos)
        self.create_subscription(LaserScan, policy_topic, self._policy_cb, sensor_qos)
        self.create_timer(1.0, self._print_status)
        self.get_logger().info(f"Checking: raw={raw_topic}  policy={policy_topic}  warn>{self._warn_sec:.2f}s")

    def _raw_cb(self, msg):
        self._raw_scan = msg
        self._raw_recv += 1

    def _policy_cb(self, msg):
        self._policy_scan = msg
        self._policy_recv += 1

    def _print_status(self):
        import numpy as np
        now = time.time()

        if self._raw_scan is None:
            print("[FRESH] raw /scan: NOT RECEIVED yet", flush=True)
        else:
            raw = self._raw_scan
            raw_stamp = _stamp_sec(raw)
            raw_age = now - raw_stamp if raw_stamp > 0 else -1.0
            ranges = list(raw.ranges or [])
            r = np.array(ranges, dtype=float)
            r_max = float(raw.range_max or 3.5)
            r_min = float(raw.range_min or 0.12)
            r = np.clip(np.where(np.isfinite(r), r, r_max), r_min, r_max)
            a_min = float(raw.angle_min or 0.0)
            a_inc = float(raw.angle_increment or 0.0)
            angles = a_min + np.arange(r.size) * a_inc if r.size > 0 else np.array([])

            raw_front = _sector_min(r, angles, 0.0)
            raw_left = _sector_min(r, angles, math.pi / 2)
            raw_rear = _sector_min(r, angles, math.pi)
            raw_right = _sector_min(r, angles, -math.pi / 2)

            stale_marker = " *** STALE ***" if raw_age > self._warn_sec else ""
            print(
                f"[RAW ] age={raw_age:6.3f}s  beams={r.size:4d}"
                f"  F={raw_front:.3f}  L={raw_left:.3f}  R_={raw_rear:.3f}  Rt={raw_right:.3f}"
                f"{stale_marker}",
                flush=True,
            )

        if self._policy_scan is None:
            print(
                "[PLCY] /rl_policy_scan_60: NOT RECEIVED yet  "
                "(Normal if no training running, or publish is throttled.)",
                flush=True,
            )
        else:
            pol = self._policy_scan
            pol_stamp = _stamp_sec(pol)
            pol_age = now - pol_stamp if pol_stamp > 0 else -1.0
            p_ranges = list(pol.ranges or [])
            p_r = np.array(p_ranges, dtype=float)
            p_max = float(pol.range_max or 3.5)
            p_min = float(pol.range_min or 0.12)
            p_r = np.clip(np.where(np.isfinite(p_r), p_r, p_max), p_min, p_max)
            p_a_min = float(pol.angle_min or 0.0)
            p_a_inc = float(pol.angle_increment or 0.0)
            p_angles = p_a_min + np.arange(p_r.size) * p_a_inc if p_r.size > 0 else np.array([])
            p_front = _sector_min(p_r, p_angles, 0.0)
            p_left = _sector_min(p_r, p_angles, math.pi / 2)
            p_rear = _sector_min(p_r, p_angles, math.pi)
            p_right = _sector_min(p_r, p_angles, -math.pi / 2)
            stale_marker = " *** STALE (debug topic only) ***" if pol_age > self._warn_sec else ""
            print(
                f"[PLCY] age={pol_age:6.3f}s  beams={p_r.size:4d}"
                f"  F={p_front:.3f}  L={p_left:.3f}  R_={p_rear:.3f}  Rt={p_right:.3f}"
                f"{stale_marker}",
                flush=True,
            )

        if self._raw_scan is not None and self._policy_scan is not None:
            raw_stamp = _stamp_sec(self._raw_scan)
            pol_stamp = _stamp_sec(self._policy_scan)
            delta = abs(raw_stamp - pol_stamp)
            warn = " *** stamp delta large ***" if delta > self._warn_sec else ""
            print(f"[DIFF] stamp_delta={delta:.3f}s  recv: raw={self._raw_recv} policy={self._policy_recv}{warn}", flush=True)
            print(
                "NOTE: /rl_policy_scan_60 staleness is separate from actual obs freshness.\n"
                "      Check metrics_compact.csv column 'scan_age_sec' to diagnose real obs stale.",
                flush=True,
            )
        print("---", flush=True)


def main():
    rclpy.init()
    node = LidarFreshnessChecker()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
