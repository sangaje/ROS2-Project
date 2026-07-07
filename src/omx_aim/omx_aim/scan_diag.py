#!/usr/bin/env python3
"""LaserScan 자기 구조물 진단 도구.

사용:
    python3 apps/scan_diag.py                       # 기본 (20cm, 5 scan)
    python3 apps/scan_diag.py --max-range 0.25      # 25cm 미만 의심
    python3 apps/scan_diag.py --samples 20          # 더 많은 scan 으로 평균
    python3 apps/scan_diag.py --topic /scan_raw     # 다른 토픽

출력:
    - 인덱스 / 각도 / 거리 별 의심 구간
    - scan_processor 에 바로 붙여 쓸 수 있는 mask_ranges_deg
"""
import argparse
import math
import sys

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from rclpy.qos import qos_profile_sensor_data


class ScanDiag(Node):
    def __init__(self, topic: str, max_range: float, samples: int,
                 margin_deg: float):
        super().__init__('scan_diag')
        self.topic = topic
        self.max_range = max_range
        self.samples = samples
        self.margin = margin_deg
        self.create_subscription(
            LaserScan, topic, self.cb, qos_profile_sensor_data)
        self.count = 0
        self.persistent = None
        self.meta = None
        self.get_logger().info(
            f"수집 시작: topic={topic}, max_range={max_range}m, "
            f"samples={samples}")

    def cb(self, m: LaserScan):
        if self.meta is None:
            self.meta = (m.angle_min, m.angle_increment, len(m.ranges))
            self.persistent = [0] * len(m.ranges)

        for i, r in enumerate(m.ranges):
            if math.isnan(r) or r == 0.0 or (0 < r < self.max_range):
                self.persistent[i] += 1

        self.count += 1
        if self.count >= self.samples:
            self.report()
            rclpy.shutdown()

    def report(self):
        a_min, a_inc, n = self.meta
        print(f"\n=== /scan 진단 ({self.count} scan 누적) ===")
        print(f"토픽: {self.topic}")
        print(f"총 {n} step, angle_min={math.degrees(a_min):.1f}°, "
              f"inc={math.degrees(a_inc):.2f}°/step\n")

        threshold = self.count // 2 + 1
        print(f"자기 구조물 의심 (range < {self.max_range}m 또는 NaN/0, "
              f"{self.count}회 중 {threshold}회 이상 검출):\n")

        suspect = []
        for i, c in enumerate(self.persistent):
            if c >= threshold:
                ang = math.degrees(a_min + i * a_inc)
                ang = (ang + 180) % 360 - 180
                suspect.append((i, ang, c))

        if not suspect:
            print("  없음 — 자기 구조물 안 잡힘 ✓")
            return

        groups = []
        gs, prev = suspect[0], suspect[0]
        for cur in suspect[1:] + [(None,)*3]:
            if cur[0] is not None and cur[0] - prev[0] <= 2:
                prev = cur; continue
            groups.append((gs, prev))
            if cur[0] is not None:
                gs = cur; prev = cur

        for (i0, a0, _), (i1, a1, _) in groups:
            width = abs(i1 - i0) + 1
            print(f"  idx {i0:3d}~{i1:3d} ({width:2d}step)  "
                  f"angle {a0:+7.1f}° ~ {a1:+7.1f}°")

        print(f"\n→ scan_processor 마스킹 후보 (margin={self.margin}°):")
        flat = []
        for (_, a0, _), (_, a1, _) in groups:
            lo, hi = min(a0, a1) - self.margin, max(a0, a1) + self.margin
            flat.extend([f"{lo:.1f}", f"{hi:.1f}"])
        print(f"  mask_ranges_deg:=[{', '.join(flat)}]")
        print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--topic', default='/scan')
    ap.add_argument('--max-range', type=float, default=0.20,
                    help='이 거리 미만은 자기 구조물 후보 (m)')
    ap.add_argument('--samples', type=int, default=5,
                    help='몇 번의 scan 을 누적할지')
    ap.add_argument('--margin-deg', type=float, default=1.5,
                    help='마스킹 범위에 추가할 안전 마진 (°)')
    args, ros_args = ap.parse_known_args()

    rclpy.init(args=ros_args)
    try:
        node = ScanDiag(args.topic, args.max_range,
                        args.samples, args.margin_deg)
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()