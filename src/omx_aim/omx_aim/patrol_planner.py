#!/usr/bin/env python3
"""Patrol Planner — risk_map 기반 PATROL 좌표 자동 생성.

Burger 가 발행한 /risk/risk_map (0~100 위험도) 위에서 NMS 로 hotspot 추출,
주기적으로 /omx/patrol_in_map 으로 발행.

알고리즘:
    1. risk >= min_risk 인 셀 모두 후보
    2. risk 내림차순 정렬
    3. NMS: 이미 선택된 후보들과 min_distance_m 이상 떨어진 것만 채택
    4. top max_candidates 선택 → 발행

토픽:
    Sub:  /risk/risk_map         nav_msgs/OccupancyGrid
    Pub:  /omx/patrol_in_map      geometry_msgs/PointStamped
    Pub:  /patrol_planner/markers visualization_msgs/MarkerArray (RViz)

파라미터:
    risk_topic                /risk/risk_map
    patrol_topic              /omx/patrol_in_map
    min_risk                  40       # 이 이상만 후보
    min_distance_m            1.0      # NMS: 후보 간 최소 거리
    max_candidates_per_cycle  3        # 한 주기에 발행할 좌표 수
    publish_period_sec        10.0     # 주기 발행 간격
    map_frame                 map
    patrol_z                  0.0      # 발행 좌표의 z (2D 운용)
    marker_lifetime_sec       12.0     # RViz 마커 수명
"""

from __future__ import annotations

import math
from typing import Optional

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy

from geometry_msgs.msg import PointStamped, Point
from nav_msgs.msg import OccupancyGrid
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import ColorRGBA


class PatrolPlanner(Node):
    def __init__(self):
        super().__init__('patrol_planner')

        # ---------- Parameters ----------
        self.declare_parameter('risk_topic', '/risk/risk_map')
        self.declare_parameter('patrol_topic', '/omx/patrol_in_map')
        self.declare_parameter('min_risk', 40)
        self.declare_parameter('min_distance_m', 1.0)
        self.declare_parameter('max_candidates_per_cycle', 3)
        self.declare_parameter('publish_period_sec', 10.0)
        self.declare_parameter('map_frame', 'map')
        self.declare_parameter('patrol_z', 0.0)
        self.declare_parameter('marker_lifetime_sec', 12.0)

        self.risk_topic = self.get_parameter('risk_topic').value
        self.patrol_topic = self.get_parameter('patrol_topic').value
        self.min_risk = int(self.get_parameter('min_risk').value)
        self.min_distance_m = float(self.get_parameter('min_distance_m').value)
        self.max_candidates = int(self.get_parameter(
            'max_candidates_per_cycle').value)
        self.publish_period_sec = float(self.get_parameter(
            'publish_period_sec').value)
        self.map_frame = self.get_parameter('map_frame').value
        self.patrol_z = float(self.get_parameter('patrol_z').value)
        self.marker_lifetime_sec = float(self.get_parameter(
            'marker_lifetime_sec').value)

        # ---------- State ----------
        self.risk_map: Optional[OccupancyGrid] = None
        self.cycle_count = 0
        self.total_published = 0

        # ---------- QoS ----------
        # risk_map 은 latched 가능 (bridge 가 transient_local 로 보냄)
        latched_qos = QoSProfile(
            depth=1,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
        )

        # ---------- Pub/Sub ----------
        self.create_subscription(
            OccupancyGrid, self.risk_topic, self.on_risk_map, latched_qos)
        self.pub_patrol = self.create_publisher(
            PointStamped, self.patrol_topic, 10)
        self.pub_markers = self.create_publisher(
            MarkerArray, '/patrol_planner/markers', 10)

        # ---------- Timer ----------
        self.create_timer(self.publish_period_sec, self.on_publish_cycle)

        self.get_logger().info("=" * 50)
        self.get_logger().info("Patrol Planner")
        self.get_logger().info("=" * 50)
        self.get_logger().info(
            f"Risk map: {self.risk_topic}  (latched/transient_local)")
        self.get_logger().info(f"Patrol out: {self.patrol_topic}")
        self.get_logger().info(
            f"필터: risk>={self.min_risk}, NMS 간격={self.min_distance_m}m, "
            f"주기당 최대 {self.max_candidates}개")
        self.get_logger().info(
            f"주기: {self.publish_period_sec}s")
        self.get_logger().info("=== Patrol Planner ready ===")

    # ----- Subscribers -----

    def on_risk_map(self, msg: OccupancyGrid):
        is_first = self.risk_map is None
        self.risk_map = msg
        if is_first:
            info = msg.info
            self.get_logger().info(
                f"첫 risk_map 수신: {info.width}x{info.height} "
                f"@ {info.resolution:.3f}m/cell, "
                f"origin=({info.origin.position.x:+.2f}, "
                f"{info.origin.position.y:+.2f}), "
                f"frame={msg.header.frame_id}")

    # ----- 주기 발행 -----

    def on_publish_cycle(self):
        if self.risk_map is None:
            self.get_logger().warn(
                f"risk_map 아직 미수신, {self.risk_topic} 발행자 확인 필요")
            return

        self.cycle_count += 1
        candidates = self._find_candidates(self.risk_map)

        if not candidates:
            self.get_logger().info(
                f"[cycle #{self.cycle_count}] 후보 없음 "
                f"(risk_map 전체 risk < {self.min_risk})")
            self._publish_markers([])
            return

        # 발행
        now = self.get_clock().now().to_msg()
        for x, y, risk_val in candidates:
            ps = PointStamped()
            ps.header.frame_id = self.map_frame
            ps.header.stamp = now
            ps.point.x = x
            ps.point.y = y
            ps.point.z = self.patrol_z
            self.pub_patrol.publish(ps)
            self.total_published += 1

        coords_str = ", ".join(
            f"({x:+.2f},{y:+.2f},r={r})" for x, y, r in candidates)
        self.get_logger().info(
            f"[cycle #{self.cycle_count}] {len(candidates)} PATROL 발행: "
            f"{coords_str}  (total={self.total_published})")

        self._publish_markers(candidates)

    # ----- 알고리즘 -----

    def _find_candidates(self, risk_map: OccupancyGrid):
        """NMS 로 risk hotspot 추출.

        Returns: [(x_world, y_world, risk_val), ...]
        """
        info = risk_map.info
        w = info.width
        h = info.height
        res = info.resolution
        ox = info.origin.position.x
        oy = info.origin.position.y
        data = risk_map.data

        # 1. risk >= min_risk 셀 모두 추출 (list of (risk, gx, gy))
        threshold = self.min_risk
        cells = []
        for gy in range(h):
            row_offset = gy * w
            for gx in range(w):
                v = data[row_offset + gx]
                if v >= threshold:   # -1 (unknown) 자동 제외
                    cells.append((v, gx, gy))

        if not cells:
            return []

        # 2. risk 내림차순 정렬
        cells.sort(key=lambda c: c[0], reverse=True)

        # 3. NMS: world 좌표로 변환 후 거리 검사
        # cell 중심 좌표 = origin + (gx + 0.5) * res
        selected = []  # [(x, y, risk)]
        min_d_sq = self.min_distance_m * self.min_distance_m

        for risk_val, gx, gy in cells:
            x = ox + (gx + 0.5) * res
            y = oy + (gy + 0.5) * res

            ok = True
            for sx, sy, _ in selected:
                dx = x - sx
                dy = y - sy
                if dx*dx + dy*dy < min_d_sq:
                    ok = False
                    break

            if ok:
                selected.append((x, y, int(risk_val)))
                if len(selected) >= self.max_candidates:
                    break

        return selected

    # ----- 시각화 -----

    def _publish_markers(self, candidates):
        """RViz 마커: 발행한 후보 위치를 색깔 sphere 로."""
        ma = MarkerArray()

        # 0번: 전체 clear (이전 마커 지우기)
        clear = Marker()
        clear.header.frame_id = self.map_frame
        clear.header.stamp = self.get_clock().now().to_msg()
        clear.ns = "patrol_candidates"
        clear.action = Marker.DELETEALL
        ma.markers.append(clear)

        for i, (x, y, risk_val) in enumerate(candidates):
            m = Marker()
            m.header.frame_id = self.map_frame
            m.header.stamp = self.get_clock().now().to_msg()
            m.ns = "patrol_candidates"
            m.id = i + 1   # 0 은 clear 용
            m.type = Marker.SPHERE
            m.action = Marker.ADD
            m.pose.position.x = x
            m.pose.position.y = y
            m.pose.position.z = 0.1
            m.pose.orientation.w = 1.0
            m.scale.x = 0.4
            m.scale.y = 0.4
            m.scale.z = 0.4

            # risk 에 따른 색: 노랑(40) → 빨강(100)
            t = (risk_val - self.min_risk) / max(1.0, 100.0 - self.min_risk)
            t = max(0.0, min(1.0, t))
            m.color = ColorRGBA(r=1.0, g=1.0 - t, b=0.0, a=0.8)

            m.lifetime.sec = int(self.marker_lifetime_sec)

            ma.markers.append(m)

            # 텍스트 마커 (risk 값 표시)
            t_marker = Marker()
            t_marker.header.frame_id = self.map_frame
            t_marker.header.stamp = m.header.stamp
            t_marker.ns = "patrol_candidates_text"
            t_marker.id = i + 1
            t_marker.type = Marker.TEXT_VIEW_FACING
            t_marker.action = Marker.ADD
            t_marker.pose.position.x = x
            t_marker.pose.position.y = y
            t_marker.pose.position.z = 0.6
            t_marker.pose.orientation.w = 1.0
            t_marker.scale.z = 0.25
            t_marker.color = ColorRGBA(r=1.0, g=1.0, b=1.0, a=1.0)
            t_marker.text = f"r={risk_val}"
            t_marker.lifetime.sec = int(self.marker_lifetime_sec)
            ma.markers.append(t_marker)

        self.pub_markers.publish(ma)


def main():
    rclpy.init()
    node = None
    try:
        node = PatrolPlanner()
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("\n중단됨.")
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()