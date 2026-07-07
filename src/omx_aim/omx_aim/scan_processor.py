#!/usr/bin/env python3
"""scan_processor — LaserScan 전처리: 180° flip + 자기 구조물 마스킹.

OMX Auto-Aim 프로젝트의 lidar 전처리 노드. 와플의 자기 구조물
(OMX 베이스, 카메라 마운트 등) 이 lidar 시야를 가려서 costmap 에
허상으로 들어가는 문제를 해결.

기능:
    1. 180° 회전 (flip_180) - lidar 가 뒤집혀 부착됐을 때 보정
    2. 각도 범위 마스킹 (mask_ranges_deg) - 특정 각도 무시
    3. 최소/최대 거리 필터 - 자기 구조물 거리 컷오프 / 멀리 잡음 컷
    4. 5초마다 마스킹 통계 로그

토픽:
    Sub: /scan           (sensor_qos = BEST_EFFORT)
    Pub: /scan_filtered  (sensor_qos = BEST_EFFORT)

사용 (가장 단순):
    python3 apps/scan_processor.py

    → 아래 DEFAULTS 의 운영 권장값으로 실행됨.

사용 (튜닝 시 override):
    python3 apps/scan_processor.py --ros-args \\
        -p 'mask_ranges_deg:=[30.0, 42.0, -42.0, -30.0]' \\
        -p min_valid_range:=0.20

마스킹 끄기 (sentinel [0.0]):
    python3 apps/scan_processor.py --ros-args \\
        -p 'mask_ranges_deg:=[0.0]'

파라미터:
    flip_180          (bool)     기본 false. true 면 ranges 배열 절반 swap.
    mask_ranges_deg   (double[]) [lo1, hi1, lo2, hi2, ...]. 단위 degree.
                                 wrap-around 지원 (예: [170, -170] = ±180°).
                                 sentinel [0.0] 또는 짝수 길이 아니면 비활성.
    min_valid_range   (double)   이 미만 거리는 inf. 0 이면 비활성.
    max_valid_range   (double)   이 초과 거리는 inf. 0 이면 비활성.
    log_period_sec    (double)   통계 로그 주기. 0 이면 끔.
"""
from __future__ import annotations

import math
from typing import List, Tuple

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import LaserScan


# ============================================================
# 운영 권장값 — 명령행에서 안 줘도 이 값으로 동작
# 와플의 자기 구조물 (좌우 +35°, -35°) 마스킹 + 18cm 이내 거리 컷
# ============================================================
DEFAULTS = {
    'flip_180': False,                                       # 라이다 정방향 부착됨
    'mask_ranges_deg': [32.7, 40.2, 145.0, 156.0, -157.0, -147.0, -40.2, -32.7],           # 좌우 자기 구조물
    'min_valid_range': 0.22,                                 # 22cm 이내 무시
    'max_valid_range': 0.0,                                  # 비활성
    'log_period_sec': 5.0,                                   # 5초마다 통계
}


def _normalize_pi(angle: float) -> float:
    """각도를 -π ~ π 로 정규화."""
    return math.atan2(math.sin(angle), math.cos(angle))


class ScanProcessor(Node):
    def __init__(self):
        super().__init__('scan_processor')

        # ===== 파라미터 선언 (default 가 명확한 타입이라 BYTE_ARRAY 문제 없음) =====
        self.declare_parameter('flip_180', DEFAULTS['flip_180'])
        self.declare_parameter('mask_ranges_deg', DEFAULTS['mask_ranges_deg'])
        self.declare_parameter('min_valid_range', DEFAULTS['min_valid_range'])
        self.declare_parameter('max_valid_range', DEFAULTS['max_valid_range'])
        self.declare_parameter('log_period_sec', DEFAULTS['log_period_sec'])

        # ===== 파라미터 로드 =====
        self.flip = bool(self.get_parameter('flip_180').value)
        self.min_valid = float(self.get_parameter('min_valid_range').value)
        self.max_valid = float(self.get_parameter('max_valid_range').value)
        log_period = float(self.get_parameter('log_period_sec').value)

        # 마스킹 각도 파싱 (deg → rad, 정규화)
        mask_raw = list(self.get_parameter('mask_ranges_deg').value or [])
        # sentinel [0.0] = "마스킹 끄기"
        if mask_raw == [0.0]:
            mask_raw = []
        self.mask: List[Tuple[float, float]] = []
        if mask_raw:
            if len(mask_raw) % 2 != 0:
                self.get_logger().error(
                    f"mask_ranges_deg 개수가 홀수: {len(mask_raw)} "
                    "(쌍으로 줘야 함). 마스킹 비활성.")
            else:
                for i in range(0, len(mask_raw), 2):
                    lo = _normalize_pi(math.radians(float(mask_raw[i])))
                    hi = _normalize_pi(math.radians(float(mask_raw[i + 1])))
                    self.mask.append((lo, hi))

        # ===== 통계 카운터 =====
        self.n_scans = 0
        self.n_points_total = 0
        self.n_points_masked_angle = 0
        self.n_points_masked_range = 0

        # ===== ROS I/O =====
        self.sub = self.create_subscription(
            LaserScan, '/scan', self.on_scan,
            qos_profile_sensor_data)
        self.pub = self.create_publisher(
            LaserScan, '/scan_filtered', qos_profile_sensor_data)

        # 주기 로그
        if log_period > 0:
            self.create_timer(log_period, self._log_stats)

        # ===== 시작 로그 =====
        mask_str = (
            ', '.join(
                f"[{math.degrees(a):+.1f}°,{math.degrees(b):+.1f}°]"
                for a, b in self.mask)
            if self.mask else '(비활성)'
        )
        self.get_logger().info(
            f"scan_processor 시작: /scan → /scan_filtered\n"
            f"  flip_180        : {self.flip}\n"
            f"  mask_ranges_deg : {mask_str}\n"
            f"  min_valid_range : {self.min_valid} m\n"
            f"  max_valid_range : {self.max_valid} m\n"
            f"  log_period      : {log_period} s")

    # ----- 마스킹 판정 -----

    def _in_mask(self, angle: float) -> bool:
        """angle(rad) 가 마스킹 구간 안에 있나? wrap-around 지원.

        예: mask = (+170°, -170°) → ±180° 구간 (후방 통과) 을 마스킹.
        """
        a = _normalize_pi(angle)
        for lo, hi in self.mask:
            if lo <= hi:
                if lo <= a <= hi:
                    return True
            else:  # wrap-around
                if a >= lo or a <= hi:
                    return True
        return False

    # ----- 메인 콜백 -----

    def on_scan(self, msg: LaserScan):
        n = len(msg.ranges)
        ranges = list(msg.ranges)
        intensities = list(msg.intensities) if msg.intensities else []

        # 1) 180° flip
        if self.flip:
            half = n // 2
            ranges = ranges[half:] + ranges[:half]
            if intensities:
                intensities = intensities[half:] + intensities[:half]

        # 2) 마스킹 + 거리 필터
        inf = float('inf')
        n_angle = 0
        n_range = 0
        for i in range(n):
            # 각도 마스킹 우선
            if self.mask:
                ang = msg.angle_min + i * msg.angle_increment
                if self._in_mask(ang):
                    ranges[i] = inf
                    n_angle += 1
                    continue

            # 거리 필터
            r = ranges[i]
            if math.isnan(r) or math.isinf(r):
                continue
            if self.min_valid > 0 and 0 < r < self.min_valid:
                ranges[i] = inf
                n_range += 1
            elif self.max_valid > 0 and r > self.max_valid:
                ranges[i] = inf
                n_range += 1

        # 3) 통계 누적
        self.n_scans += 1
        self.n_points_total += n
        self.n_points_masked_angle += n_angle
        self.n_points_masked_range += n_range

        # 4) 발행 — header(timestamp) 그대로 보존
        out = LaserScan()
        out.header = msg.header
        out.angle_min = msg.angle_min
        out.angle_max = msg.angle_max
        out.angle_increment = msg.angle_increment
        out.time_increment = msg.time_increment
        out.scan_time = msg.scan_time
        out.range_min = msg.range_min
        out.range_max = msg.range_max
        out.ranges = ranges
        out.intensities = intensities
        self.pub.publish(out)

    # ----- 주기 통계 로그 -----

    def _log_stats(self):
        if self.n_scans == 0:
            self.get_logger().warn(
                "/scan 메시지 수신 없음 - bringup 또는 lidar 노드 확인")
            return
        n = self.n_points_total
        pct_a = 100.0 * self.n_points_masked_angle / n if n else 0.0
        pct_r = 100.0 * self.n_points_masked_range / n if n else 0.0
        self.get_logger().info(
            f"통계 ({self.n_scans} scan): "
            f"angle 마스킹 {pct_a:.2f}%, "
            f"range 마스킹 {pct_r:.2f}%, "
            f"FPS≈{self.n_scans / 5.0:.1f}")
        # 리셋
        self.n_scans = 0
        self.n_points_total = 0
        self.n_points_masked_angle = 0
        self.n_points_masked_range = 0


def main():
    rclpy.init()
    node = ScanProcessor()
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