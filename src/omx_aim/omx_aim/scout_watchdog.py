#!/usr/bin/env python3
"""Scout Watchdog — Burger 죽음 감지 + 주변 수색 TARGET 발행.

Heartbeat + pose 구독. ALIVE 상태에서 timeout 되면 DEAD 로 판정.
마지막 pose 주변 ring 에서 costmap-free 한 N개 지점을 TARGET 으로
순차 발행. /omx/fire 들리면 잔여 폐기, /omx/target_not_found 들리면
다음 후보 시도.

상태:
    INIT   heartbeat 한 번도 안 받음 (와플 단독 운용 케이스). 아무것도 안 함.
    ALIVE  heartbeat 정상 수신 중.
    DEAD   timeout 감지. 후보 발행 중 또는 완료.

토픽:
    Sub:  /scout/heartbeat         std_msgs/Empty
    Sub:  /scout/pose              geometry_msgs/PoseStamped
    Sub:  /global_costmap/costmap  nav_msgs/OccupancyGrid
    Sub:  /omx/fire                std_msgs/Empty
    Sub:  /omx/target_not_found    geometry_msgs/PointStamped
    Pub:  /omx/target_in_map       geometry_msgs/PointStamped
    Pub:  /scout_watchdog/markers  visualization_msgs/MarkerArray

scout_bridge.yaml 추가 필요 (Burger ↔ Desktop):
    /scout/heartbeat:
      type: std_msgs/msg/Empty
      qos: { reliability: best_effort, durability: volatile }
    /scout/pose:
      type: geometry_msgs/msg/PoseStamped
      qos: { reliability: reliable, durability: volatile }

Burger 측에서 발행 필요:
    /scout/heartbeat  1~2 Hz
    /scout/pose       1~10 Hz  (AMCL 의 amcl_pose 를 remap 해도 됨)

실행:
    python3 apps/scout_watchdog.py
"""

from __future__ import annotations

import math
import time
from enum import Enum
from typing import Optional, List, Tuple

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy

from std_msgs.msg import Empty
from geometry_msgs.msg import PointStamped, PoseStamped
from nav_msgs.msg import OccupancyGrid
from visualization_msgs.msg import Marker, MarkerArray


class WatchState(Enum):
    INIT = "INIT"
    ALIVE = "ALIVE"
    DEAD = "DEAD"


DEFAULTS = {
    # Topics
    'heartbeat_topic': '/scout/heartbeat',
    'pose_topic': '/scout/pose',
    'costmap_topic': '/global_costmap/costmap',
    'target_topic': '/omx/target_in_map',
    'fire_topic': '/omx/fire',
    'target_not_found_topic': '/omx/target_not_found',
    'map_frame': 'map',

    # Death detection
    'dead_timeout_sec': 5.0,         # heartbeat 끊긴 후 죽음 판정 시간
    'check_period_sec': 1.0,          # death check + 상태 평가 주기
    'status_log_period_sec': 10.0,    # 평상시 상태 로그 주기

    # Candidate generation
    'ring_radius_m': 1.2,            # ring 반경 (Burger 주변)
    'num_angular_candidates': 12,    # 각도 샘플 수 (30° 간격)
    'num_targets': 3,                # 최종 발행 TARGET 수
    'costmap_threshold': 80,          # 이 이상 cost 면 막힌 셀
    'target_z': 0.0,                 # 발행 z (2D 운용)

    # Matching
    'target_match_tolerance_m': 0.4, # /omx/target_not_found 매칭 허용 오차
}


class ScoutWatchdog(Node):
    def __init__(self):
        super().__init__('scout_watchdog')

        # ---------- Parameters ----------
        for key, value in DEFAULTS.items():
            self.declare_parameter(key, value)

        self.heartbeat_topic = self.get_parameter('heartbeat_topic').value
        self.pose_topic = self.get_parameter('pose_topic').value
        self.costmap_topic = self.get_parameter('costmap_topic').value
        self.target_topic = self.get_parameter('target_topic').value
        self.fire_topic = self.get_parameter('fire_topic').value
        self.target_not_found_topic = self.get_parameter(
            'target_not_found_topic').value
        self.map_frame = self.get_parameter('map_frame').value

        self.dead_timeout_sec = float(
            self.get_parameter('dead_timeout_sec').value)
        self.check_period_sec = float(
            self.get_parameter('check_period_sec').value)
        self.status_log_period_sec = float(
            self.get_parameter('status_log_period_sec').value)

        self.ring_radius_m = float(
            self.get_parameter('ring_radius_m').value)
        self.num_angular_candidates = int(
            self.get_parameter('num_angular_candidates').value)
        self.num_targets = int(self.get_parameter('num_targets').value)
        self.costmap_threshold = int(
            self.get_parameter('costmap_threshold').value)
        self.target_z = float(self.get_parameter('target_z').value)

        self.target_match_tolerance_m = float(
            self.get_parameter('target_match_tolerance_m').value)

        # ---------- State ----------
        self.state = WatchState.INIT
        self.last_hb_t: float = 0.0
        self.last_pose: Optional[Tuple[float, float]] = None
        self.costmap: Optional[OccupancyGrid] = None

        # Probe queue
        self.pending_targets: List[Tuple[float, float]] = []
        self.published_targets: List[Tuple[float, float]] = []
        self.current_probe: Optional[Tuple[float, float]] = None
        self.last_status_log_t: float = 0.0

        # ---------- QoS ----------
        latched_qos = QoSProfile(
            depth=1,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
        )

        # ---------- Subscribers ----------
        self.create_subscription(
            Empty, self.heartbeat_topic, self.on_heartbeat, 10)
        self.create_subscription(
            PoseStamped, self.pose_topic, self.on_pose, 10)
        self.create_subscription(
            OccupancyGrid, self.costmap_topic, self.on_costmap, latched_qos)
        self.create_subscription(
            Empty, self.fire_topic, self.on_fire, 10)
        self.create_subscription(
            PointStamped, self.target_not_found_topic,
            self.on_target_not_found, 10)

        # ---------- Publishers ----------
        self.pub_target = self.create_publisher(
            PointStamped, self.target_topic, 10)
        self.pub_markers = self.create_publisher(
            MarkerArray, '/scout_watchdog/markers', 10)

        # ---------- Timer ----------
        self.create_timer(self.check_period_sec, self.on_check_timer)

        # ---------- Banner ----------
        self.get_logger().info("=" * 50)
        self.get_logger().info("Scout Watchdog")
        self.get_logger().info("=" * 50)
        self.get_logger().info(
            f"Heartbeat: {self.heartbeat_topic}  "
            f"(timeout={self.dead_timeout_sec:.1f}s)")
        self.get_logger().info(f"Pose:      {self.pose_topic}")
        self.get_logger().info(
            f"Costmap:   {self.costmap_topic}  "
            f"(threshold={self.costmap_threshold})")
        self.get_logger().info(
            f"Target:    {self.target_topic}  (z={self.target_z})")
        self.get_logger().info(
            f"Ring:      r={self.ring_radius_m}m, "
            f"{self.num_angular_candidates} candidates → "
            f"{self.num_targets} targets")
        self.get_logger().info("=" * 50)
        self.get_logger().info(
            "=== Scout Watchdog ready (INIT, heartbeat 대기) ===")

    # ----- Subscriber callbacks -----

    def on_heartbeat(self, msg: Empty):
        self.last_hb_t = time.time()
        if self.state == WatchState.INIT:
            self.state = WatchState.ALIVE
            self.get_logger().info("Burger ALIVE (첫 heartbeat 수신)")
        elif self.state == WatchState.DEAD:
            # 부활: 이미 발행한 TARGET 은 회수 못 함, 잔여도 그대로 진행
            self.state = WatchState.ALIVE
            n_pending = len(self.pending_targets)
            self.get_logger().warn(
                f"Burger 부활 — 이미 발행한 "
                f"{len(self.published_targets)}개 TARGET 은 그대로 처리됨, "
                f"잔여 {n_pending}개도 그대로 진행")

    def on_pose(self, msg: PoseStamped):
        self.last_pose = (msg.pose.position.x, msg.pose.position.y)

    def on_costmap(self, msg: OccupancyGrid):
        is_first = self.costmap is None
        self.costmap = msg
        if is_first:
            info = msg.info
            self.get_logger().info(
                f"Costmap 첫 수신: {info.width}x{info.height} "
                f"@ {info.resolution:.3f}m/cell")

    def on_fire(self, msg: Empty):
        if self.state != WatchState.DEAD:
            return
        if not self.pending_targets:
            return
        n = len(self.pending_targets)
        self.get_logger().info(
            f"/omx/fire 수신 → 적 사살 확인, 잔여 후보 {n}개 폐기")
        self.pending_targets.clear()
        self.current_probe = None
        self._publish_markers()

    def on_target_not_found(self, msg: PointStamped):
        if self.state != WatchState.DEAD:
            return
        if self.current_probe is None:
            return
        # 좌표 매칭: 내가 발행한 probe 와 일치하는지 확인
        # (다른 PATROL/TARGET 의 miss 와 구분)
        p = (msg.point.x, msg.point.y)
        d = math.hypot(
            p[0] - self.current_probe[0], p[1] - self.current_probe[1])
        if d > self.target_match_tolerance_m:
            return
        self.get_logger().info(
            f"내 probe miss 확인 "
            f"({self.current_probe[0]:+.2f}, {self.current_probe[1]:+.2f}) "
            f"→ 다음 후보 시도")
        self.current_probe = None
        self.publish_next_target()

    # ----- Timer -----

    def on_check_timer(self):
        now = time.time()

        # 평상시 상태 로그
        if now - self.last_status_log_t > self.status_log_period_sec:
            self._log_status(now)
            self.last_status_log_t = now

        if self.state != WatchState.ALIVE:
            return
        elapsed = now - self.last_hb_t
        if elapsed > self.dead_timeout_sec:
            self.trigger_death()

    def _log_status(self, now: float):
        if self.state == WatchState.INIT:
            self.get_logger().info(
                f"[INIT] heartbeat 대기 중 ({self.heartbeat_topic})")
        elif self.state == WatchState.ALIVE:
            elapsed = now - self.last_hb_t
            pose_str = (f"({self.last_pose[0]:+.2f}, "
                        f"{self.last_pose[1]:+.2f})"
                        if self.last_pose else "N/A")
            self.get_logger().info(
                f"[ALIVE] last hb {elapsed:.1f}s ago, "
                f"pose={pose_str}, "
                f"costmap={'OK' if self.costmap else 'N/A'}")
        elif self.state == WatchState.DEAD:
            self.get_logger().info(
                f"[DEAD] published={len(self.published_targets)}, "
                f"pending={len(self.pending_targets)}")

    # ----- Death handling -----

    def trigger_death(self):
        self.state = WatchState.DEAD
        self.get_logger().error(
            f"Burger DEAD ({self.dead_timeout_sec:.1f}s heartbeat 없음)")

        # 이전 death event 흔적 정리 (재발 시)
        self.pending_targets = []
        self.published_targets = []
        self.current_probe = None

        if self.last_pose is None:
            self.get_logger().error(
                "마지막 pose 없음 → TARGET 발행 불가 "
                "(Burger pose 토픽 확인 필요)")
            return

        center = self.last_pose
        self.get_logger().info(
            f"마지막 pose: ({center[0]:+.2f}, {center[1]:+.2f})")

        candidates = self.generate_candidates(center)
        if not candidates:
            self.get_logger().warn(
                "Ring 후보 모두 막힘 → 중심 좌표 1개 fallback")
            self.pending_targets = [center]
        else:
            self.pending_targets = candidates
            self.get_logger().info(
                f"생성된 TARGET 후보 {len(candidates)}개")

        self._publish_markers(center=center)
        self.publish_next_target()

    def publish_next_target(self):
        if not self.pending_targets:
            self.current_probe = None
            self.get_logger().info("모든 TARGET 후보 처리 완료")
            return

        xy = self.pending_targets.pop(0)
        self.current_probe = xy
        self.published_targets.append(xy)

        msg = PointStamped()
        msg.header.frame_id = self.map_frame
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.point.x = xy[0]
        msg.point.y = xy[1]
        msg.point.z = self.target_z
        self.pub_target.publish(msg)

        total = len(self.published_targets) + len(self.pending_targets)
        self.get_logger().info(
            f"TARGET 발행 [{len(self.published_targets)}/{total}]: "
            f"({xy[0]:+.2f}, {xy[1]:+.2f})")
        self._publish_markers()

    # ----- Candidate generation -----

    def generate_candidates(
            self, center: Tuple[float, float]) -> List[Tuple[float, float]]:
        """Ring 후보 N개 → costmap-free 필터 → 각도 균등 분산 num_targets 개."""
        cx, cy = center
        radius = self.ring_radius_m
        n_ang = self.num_angular_candidates

        # 1. ring 후보 생성 (등각 분포)
        all_candidates = []
        for i in range(n_ang):
            theta = 2.0 * math.pi * i / n_ang
            x = cx + radius * math.cos(theta)
            y = cy + radius * math.sin(theta)
            all_candidates.append((x, y, theta))

        # 2. costmap-free 필터
        if self.costmap is None:
            self.get_logger().warn(
                "Costmap 미수신 → 필터 skip, 모든 후보 사용")
            free = all_candidates
        else:
            free = []
            blocked = 0
            for x, y, theta in all_candidates:
                val = self.costmap_value_at(x, y)
                if val is None:
                    blocked += 1
                    continue
                if val < self.costmap_threshold:
                    free.append((x, y, theta))
                else:
                    blocked += 1
            self.get_logger().info(
                f"Ring 후보 {n_ang}개 중 free={len(free)}, "
                f"막힘/맵밖={blocked}")

        if not free:
            return []

        # 3. 균등 분산 선택 (각거리 최대화 greedy)
        return self.pick_spread(free)

    def pick_spread(
            self, free: List[Tuple[float, float, float]]
            ) -> List[Tuple[float, float]]:
        """각도 균등 분산 num_targets 개 선택 (greedy max-min angular dist).

        free 가 num_targets 이하면 그대로 반환.
        그 외에는 첫 후보부터 시작, 매번 selected 와의 최소 각거리가
        가장 큰 후보를 추가.
        """
        if not free:
            return []
        if len(free) <= self.num_targets:
            return [(x, y) for x, y, _ in free]

        selected = [free[0]]
        while len(selected) < self.num_targets:
            best = None
            best_min_dist = -1.0
            for cand in free:
                if cand in selected:
                    continue
                min_d = float('inf')
                for s in selected:
                    d = abs(cand[2] - s[2])
                    d = min(d, 2.0 * math.pi - d)  # 원형 거리
                    if d < min_d:
                        min_d = d
                if min_d > best_min_dist:
                    best_min_dist = min_d
                    best = cand
            if best is None:
                break
            selected.append(best)

        return [(x, y) for x, y, _ in selected]

    def costmap_value_at(self, x: float, y: float) -> Optional[int]:
        """World coords → costmap cell value (or None if out of map)."""
        if self.costmap is None:
            return None
        info = self.costmap.info
        gx = int((x - info.origin.position.x) / info.resolution)
        gy = int((y - info.origin.position.y) / info.resolution)
        if not (0 <= gx < info.width and 0 <= gy < info.height):
            return None
        return self.costmap.data[gy * info.width + gx]

    # ----- Visualization -----

    def _publish_markers(
            self, center: Optional[Tuple[float, float]] = None):
        marker_array = MarkerArray()
        now = self.get_clock().now().to_msg()

        # DELETEALL 먼저 (이전 마커 정리)
        delete_marker = Marker()
        delete_marker.header.frame_id = self.map_frame
        delete_marker.action = Marker.DELETEALL
        marker_array.markers.append(delete_marker)

        # 중심 (Burger 죽은 자리) - 회색 실린더
        c = center if center is not None else self.last_pose
        if c is not None:
            m = Marker()
            m.header.frame_id = self.map_frame
            m.header.stamp = now
            m.ns = 'death_center'
            m.id = 0
            m.type = Marker.CYLINDER
            m.action = Marker.ADD
            m.pose.position.x = c[0]
            m.pose.position.y = c[1]
            m.pose.position.z = 0.05
            m.pose.orientation.w = 1.0
            m.scale.x = m.scale.y = 0.4
            m.scale.z = 0.1
            m.color.r = 0.5
            m.color.g = 0.5
            m.color.b = 0.5
            m.color.a = 0.7
            marker_array.markers.append(m)

        # 발행 완료 TARGET - 빨강
        for i, xy in enumerate(self.published_targets):
            m = Marker()
            m.header.frame_id = self.map_frame
            m.header.stamp = now
            m.ns = 'published'
            m.id = i
            m.type = Marker.SPHERE
            m.action = Marker.ADD
            m.pose.position.x = xy[0]
            m.pose.position.y = xy[1]
            m.pose.position.z = self.target_z
            m.pose.orientation.w = 1.0
            m.scale.x = m.scale.y = m.scale.z = 0.3
            m.color.r = 1.0
            m.color.g = 0.2
            m.color.b = 0.2
            m.color.a = 0.9
            marker_array.markers.append(m)

        # 대기 후보 - 주황 반투명
        for i, xy in enumerate(self.pending_targets):
            m = Marker()
            m.header.frame_id = self.map_frame
            m.header.stamp = now
            m.ns = 'pending'
            m.id = i
            m.type = Marker.SPHERE
            m.action = Marker.ADD
            m.pose.position.x = xy[0]
            m.pose.position.y = xy[1]
            m.pose.position.z = self.target_z
            m.pose.orientation.w = 1.0
            m.scale.x = m.scale.y = m.scale.z = 0.25
            m.color.r = 1.0
            m.color.g = 0.6
            m.color.b = 0.0
            m.color.a = 0.5
            marker_array.markers.append(m)

        self.pub_markers.publish(marker_array)


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = ScoutWatchdog()
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