#!/usr/bin/env python3
"""OMX Auto-Aim 메인 ROS 노드 — Stage H4 + R6 (모듈 분리 완료).

이 파일은 OmxYoloNode 의 ROS 인터페이스 (subscribers, publishers,
TF, costmap, 시각화, main loop) 만 담는다. 핵심 로직은 omx/ 패키지로 분리:

    omx/types.py          State, TargetType, LOSResult, TargetEntry
    omx/state_machine.py  StateMachine (큐 + 상태 머신)
    omx/boundary_gen.py   BoundaryGenerator (사주 경계 sweep)
    omx/yolo_detector.py  YoloDetector (cv2 + YOLO)
    omx/controller.py     OmxController (Dynamixel + IBVS)
    omx/config.py         dataclass + load_config
    omx/hardware.py       저수준 DXL bus

진화 단계:
    A/D/F/G  : 큐, LOS, 거리정렬, RViz 마커
    H1       : waffle_node.py 분리 (Nav2 클라이언트)
    H2       : CHECK_VIEW + VIEW_POSE v1 + WAITING_NAV + 큐 분리
    H3       : TARGET preempt (PATROL 폐기/큐 복귀)
    H4       : BoundaryGenerator 통합 (자동 sweep + 토글 토픽)
    R1~R6    : 코드 모듈 분리

토픽/state/큐 정책 상세는 INTERFACE_v3.md 참조.
"""

from __future__ import annotations

import sys
import math
import time
from typing import Optional

import cv2
import rclpy
from rclpy.node import Node
from rclpy.duration import Duration

from std_msgs.msg import String, Bool, Float32, Empty, Int32
from geometry_msgs.msg import Point, PointStamped, PoseStamped, Quaternion
from sensor_msgs.msg import JointState
from nav_msgs.msg import OccupancyGrid
from visualization_msgs.msg import Marker, MarkerArray

from tf2_ros import Buffer, TransformListener, TransformException

try:
    from tf2_geometry_msgs import do_transform_point
except ImportError:
    print()
    print("ERROR: tf2_geometry_msgs 패키지가 없습니다.")
    print("  sudo apt install ros-jazzy-tf2-geometry-msgs")
    sys.exit(1)

from omx.config import load_config, Config
from omx.types import State, TargetType, LOSResult
from omx.yolo_detector import YoloDetector
from omx.controller import OmxController
from omx.boundary_gen import BoundaryGenerator
from omx.state_machine import StateMachine


# ===========================================================
# Bresenham (LOS 셀 순회)
# ===========================================================

def bresenham_line(x0: int, y0: int, x1: int, y1: int):
    cells = []
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy

    while True:
        cells.append((x0, y0))
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x0 += sx
        if e2 < dx:
            err += dx
            y0 += sy
    return cells



# ===========================================================
# OmxYoloNode
# ===========================================================

class OmxYoloNode(Node):
    """OMX Auto-Aim 메인 ROS 노드.

    책임:
        - YOLO 영상 검출 (omx.yolo_detector.YoloDetector)
        - OMX 모터 제어 (omx.controller.OmxController)
        - state machine + 큐 (omx.state_machine.StateMachine)
        - BOUNDARY 자동 생성 (omx.boundary_gen.BoundaryGenerator)
        - ROS topic pub/sub, TF, costmap, 시각화

    StateMachine 에 콜백 주입으로 ROS 의존성 분리:
        los_check_fn, waffle_pos_fn, check_view_fn,
        compute_view_pose_fn, nav_cancel_fn

    토픽/state/큐 정책 상세는 INTERFACE_v3.md 참조.
    """

    def __init__(self, dry_run: bool = False, no_display: bool = False,
                debug_stream: bool = False,
                debug_port: int = 8080,
                debug_fps: int = 15,
                debug_quality: int = 70):
        super().__init__('omx_yolo_node')

        self.cfg = load_config()
        self.dry_run = dry_run
        self.no_display = no_display
        self.get_logger().info(f"Config loaded. port={self.cfg.motor.port}")

        if self.cfg.fire is None:
            raise RuntimeError("config.yaml 에 fire 섹션 필요")
        if self.cfg.yolo is None:
            raise RuntimeError("config.yaml 에 yolo 섹션 필요")
        if self.cfg.autotrack is None:
            raise RuntimeError("config.yaml 에 autotrack 섹션 필요")
        if self.cfg.patrol is None:
            raise RuntimeError("config.yaml 에 patrol 섹션 필요")
        if self.cfg.view_pose is None:
            raise RuntimeError("config.yaml 에 view_pose 섹션 필요")

        self.get_logger().info(
            f"VIEW_POSE: yaw_limit={self.cfg.view_pose.omx_yaw_limit_deg}°, "
            f"dist=[{self.cfg.view_pose.min_distance_m}, "
            f"{self.cfg.view_pose.max_distance_m}]m, "
            f"stand_off={self.cfg.view_pose.stand_off_distance}m")

        # TF
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Arm base offset
        self.declare_parameter('arm_base_x', 0.10)
        self.declare_parameter('arm_base_y', 0.00)
        self.declare_parameter('arm_base_z', 0.00)
        self.arm_offset = (
            self.get_parameter('arm_base_x').value,
            self.get_parameter('arm_base_y').value,
            self.get_parameter('arm_base_z').value,
        )
        self.get_logger().info(
            f"Arm base offset: x={self.arm_offset[0]}, "
            f"y={self.arm_offset[1]}, z={self.arm_offset[2]} m")

        # Costmap
        self.costmap: Optional[OccupancyGrid] = None
        self._costmap_logged = False

        # 내부 모듈
        self.detector = YoloDetector(self.cfg, logger=self.get_logger())
        self.ctrl = OmxController(self.cfg, dry_run=dry_run,
                                    logger=self.get_logger())
        self.sm = StateMachine(self.cfg, logger=self.get_logger())

        # 콜백 주입
        self.sm.los_check_fn = self.check_line_of_sight
        self.sm.waffle_pos_fn = self.get_waffle_xy
        self.sm.check_view_fn = self.check_view              # H2
        self.sm.compute_view_pose_fn = self.compute_view_pose  # H2
        self.sm.nav_cancel_fn = self.publish_nav_cancel        # H3

        # H4: BoundaryGenerator (사주 경계 자동 sweep)
        if self.cfg.boundary is None:
            raise RuntimeError("config.yaml 에 boundary 섹션 필요")
        self.boundary_gen = BoundaryGenerator(
            cfg=self.cfg.boundary,
            waffle_pose_fn=self.get_waffle_xy_yaw,
            logger=self.get_logger(),
        )
        self.get_logger().info(
            f"BoundaryGenerator: T={self.boundary_gen.enabled_target} "
            f"P={self.boundary_gen.enabled_patrol}, "
            f"sweep={self.boundary_gen.sweep_angles_deg} deg, "
            f"period={self.cfg.boundary.period_sec}s, "
            f"ttl={self.cfg.boundary.ttl_sec}s")

        self.ctrl.connect()
        self.ctrl.go_home()

        self.paused = False
        self.control_period = 1.0 / self.cfg.ibvs.control_hz

        self.fps_t = time.time()
        self.fps_n = 0
        self.fps_disp = 0.0

        # Publishers
        self.pub_status = self.create_publisher(String, '/omx/status', 10)
        self.pub_state = self.create_publisher(String, '/omx/state', 10)
        self.pub_detected = self.create_publisher(Bool, '/omx/target_detected', 10)
        self.pub_error = self.create_publisher(Point, '/omx/error_norm', 10)
        self.pub_joint = self.create_publisher(JointState, '/omx/joint_state', 10)
        self.pub_fire = self.create_publisher(Empty, '/omx/fire', 10)
        self.pub_processed = self.create_publisher(PointStamped, '/omx/target_processed', 10)
        self.pub_target_lost = self.create_publisher(PointStamped, '/omx/target_lost', 10)
        self.pub_target_blocked = self.create_publisher(PointStamped, '/omx/target_blocked', 10)
        self.pub_progress = self.create_publisher(Float32, '/omx/aim_progress', 10)
        self.pub_queue_size = self.create_publisher(Int32, '/omx/queue_size', 10)
        self.pub_patrol_complete = self.create_publisher(Empty, '/omx/patrol_complete', 10)
        self.pub_queue_markers = self.create_publisher(
            MarkerArray, '/omx/queue_markers', 10)
        # H2 신규
        self.pub_nav_goal = self.create_publisher(
            PoseStamped, '/omx/nav_goal', 10)
        # H3 신규
        self.pub_nav_cancel = self.create_publisher(
            Empty, '/omx/nav_cancel', 10)
        self.pub_target_not_found = self.create_publisher(
            PointStamped, '/omx/target_not_found', 10)

        # Subscribers
        self.create_subscription(String, '/omx/control_mode',
                                 self.on_control_mode, 10)
        self.create_subscription(PointStamped, '/omx/target_in_map',
                                 self.on_target_in_map, 10)
        self.create_subscription(PointStamped, '/omx/boundary_in_map',
                                 self.on_boundary_in_map, 10)
        self.create_subscription(PointStamped, '/omx/patrol_in_map',
                                 self.on_patrol_in_map, 10)
        self.create_subscription(Bool, '/omx/arm_enable',
                                 self.on_arm_enable, 10)
        self.create_subscription(Empty, '/omx/abort',
                                 self.on_abort, 10)
        self.create_subscription(
            OccupancyGrid, self.cfg.patrol.costmap_topic,
            self.on_costmap, 1)
        # H2 신규
        self.create_subscription(String, '/waffle/nav_result',
                                 self.on_nav_result, 10)
        # H4 신규
        self.create_subscription(String, '/omx/boundary_enable',
                                 self.on_boundary_enable, 10)
        self.debug_stream = None
        if debug_stream:
            from omx.debug_stream import DebugStream
            self.debug_stream = DebugStream(
                port=debug_port, fps=debug_fps, quality=debug_quality)
            self.debug_stream.start()
            self.get_logger().info(
                f"Debug stream ON: http://0.0.0.0:{debug_port}/ "
                f"(fps={debug_fps}, q={debug_quality})")
        self.timer = self.create_timer(self.control_period, self.loop)
        self.status_timer = self.create_timer(1.0, self.publish_periodic)

        self._last_state = self.sm.state

        self.get_logger().info(
            f"Timer: 메인 {self.cfg.ibvs.control_hz} Hz, 상태 1 Hz")
        self.get_logger().info(f"Initial armed: {self.sm.armed}")
        self.get_logger().info("=== Node ready (H4) ===")

    # ----- Costmap -----

    def on_costmap(self, msg: OccupancyGrid):
        self.costmap = msg
        if not self._costmap_logged:
            self.get_logger().info(
                f"Costmap 수신: {msg.info.width}x{msg.info.height} "
                f"cells @ {msg.info.resolution}m/cell")
            self._costmap_logged = True

    # ----- TF helpers -----

    def get_waffle_xy(self):
        try:
            tr = self.tf_buffer.lookup_transform(
                'map', 'base_link', rclpy.time.Time(),
                timeout=Duration(seconds=0.1))
            return tr.transform.translation.x, tr.transform.translation.y
        except TransformException:
            return None

    def get_waffle_xy_yaw(self):
        """H2: 와플 (x, y, yaw) in map frame. H4 BOUNDARY 자동 생성에 사용."""
        try:
            tr = self.tf_buffer.lookup_transform(
                'map', 'base_link', rclpy.time.Time(),
                timeout=Duration(seconds=0.1))
            q = tr.transform.rotation
            # quaternion -> yaw
            yaw = math.atan2(
                2.0 * (q.w * q.z + q.x * q.y),
                1.0 - 2.0 * (q.y * q.y + q.z * q.z))
            return (tr.transform.translation.x,
                    tr.transform.translation.y,
                    yaw)
        except TransformException:
            return None

    # ----- LOS -----

    def check_line_of_sight(self, target_map) -> LOSResult:
        """와플 → target 의 LOS."""
        waffle = self.get_waffle_xy()
        if waffle is None:
            return LOSResult.UNKNOWN
        return self._los_between(waffle, (target_map[0], target_map[1]))

    def _los_between(self, start_xy, end_xy) -> LOSResult:
        """H5: 임의 두 점 사이 LOS. costmap 위에서 Bresenham."""
        if self.costmap is None:
            return LOSResult.UNKNOWN

        info = self.costmap.info
        res = info.resolution
        ox = info.origin.position.x
        oy = info.origin.position.y

        sgx = int((start_xy[0] - ox) / res)
        sgy = int((start_xy[1] - oy) / res)
        egx = int((end_xy[0] - ox) / res)
        egy = int((end_xy[1] - oy) / res)

        cells = bresenham_line(sgx, sgy, egx, egy)

        threshold = self.cfg.patrol.los_cost_threshold
        width = info.width
        height = info.height
        data = self.costmap.data

        has_unknown = False
        for cx, cy in cells:
            if cx < 0 or cx >= width or cy < 0 or cy >= height:
                has_unknown = True
                continue
            idx = cy * width + cx
            cost = data[idx]
            if cost == -1:
                has_unknown = True
            elif cost >= threshold:
                return LOSResult.BLOCKED

        return LOSResult.UNKNOWN if has_unknown else LOSResult.CLEAR

    def _costmap_value(self, x, y) -> Optional[int]:
        """H5: map frame (x, y) 의 costmap 값.

        Returns: 0~100 (inflation), -1 (unknown 무시되어 None), None (경계 밖).
        """
        if self.costmap is None:
            return None
        info = self.costmap.info
        res = info.resolution
        ox = info.origin.position.x
        oy = info.origin.position.y
        gx = int((x - ox) / res)
        gy = int((y - oy) / res)
        if gx < 0 or gx >= info.width or gy < 0 or gy >= info.height:
            return None
        idx = gy * info.width + gx
        cost = self.costmap.data[idx]
        if cost == -1:
            return None
        return cost

    # ----- H2: CHECK_VIEW + VIEW_POSE -----

    def check_view(self, target_map) -> bool:
        """현재 와플 위치에서 target_map 을 OMX 가 조준 가능한가?
        
        판정 기준:
            1. LOS clear 또는 unknown (blocked 는 불가)
            2. arm_base 좌표 기준 OMX yaw 한계 안
            3. 거리 적정 범위
        """
        # 1. LOS
        los = self.check_line_of_sight(target_map)
        if los == LOSResult.BLOCKED:
            self.get_logger().info(f"CHECK_VIEW NG: LOS BLOCKED")
            return False

        # 2, 3. arm_base 변환 후 yaw/거리
        arm = self.transform_map_to_arm_base(target_map)
        if arm is None:
            self.get_logger().info(f"CHECK_VIEW NG: TF 변환 실패")
            return False

        ax, ay, az = arm
        yaw_deg = math.degrees(math.atan2(ay, ax))
        distance = math.sqrt(ax*ax + ay*ay + az*az)

        vp = self.cfg.view_pose
        if abs(yaw_deg) > vp.omx_yaw_limit_deg:
            self.get_logger().info(
                f"CHECK_VIEW NG: yaw={yaw_deg:+.1f}° > {vp.omx_yaw_limit_deg}°")
            return False
        if distance < vp.min_distance_m or distance > vp.max_distance_m:
            self.get_logger().info(
                f"CHECK_VIEW NG: dist={distance:.2f}m "
                f"out of [{vp.min_distance_m}, {vp.max_distance_m}]")
            return False

        self.get_logger().info(
            f"CHECK_VIEW OK: yaw={yaw_deg:+.1f}° dist={distance:.2f}m")
        return True

    def compute_view_pose(self, target_map, next_target_map=None):
        """VIEW_POSE v2 (H5): 후보 샘플링 + cost 평가.

        target 주변 candidate_count 방향에서 stand_off_distance 떨어진 후보 생성.
        각 후보의 필수 조건 (costmap free + LOS + OMX aim) 검사,
        통과한 후보 중 최소 cost 선택.

        cost 가중치는 코드 상수. 필요 시 config 로 옮길 수 있음.

        yaw 정책 (H5):
            target 방향과 next_target_map 방향의 짧은 경로 보간.
            yaw_next_weight=0.5 → 중간, =1.0 → next 100% (이전 v1.1), =0.0 → target 100%.

        Args:
            target_map: VIEW_POSE 기준 좌표 (와플 도착 위치 계산용).
            next_target_map: 도착 후 와플이 향할 다음 target. None 이면 target 방향.

        Returns: (x, y, yaw) in map frame, 또는 None (적합 후보 없음).
        """
        waffle = self.get_waffle_xy()
        if waffle is None:
            self.get_logger().warn("compute_view_pose: 와플 위치 모름")
            return None

        tx, ty, _ = target_map
        wx, wy = waffle
        vp_cfg = self.cfg.view_pose
        stand_off = vp_cfg.stand_off_distance

        # cost 가중치 (낮을수록 좋음)
        W_INFL = 1.0    # costmap inflation (벽 가까움 penalty)
        W_DIST = 2.0    # ideal stand_off 거리와의 편차
        W_WAFL = 0.5    # 와플과의 거리 (이동 시간)

        # 후보 N 방향 생성
        n = vp_cfg.candidate_count
        candidates = []
        for i in range(n):
            angle = 2.0 * math.pi * i / n
            cx = tx + stand_off * math.cos(angle)
            cy = ty + stand_off * math.sin(angle)
            candidates.append((cx, cy))

        # 필수 조건 + cost 평가
        valid = []
        rejection_reasons = {'costmap': 0, 'los': 0, 'omx': 0}

        for cx, cy in candidates:
            # 조건 1: costmap free
            cost_val = self._costmap_value(cx, cy)
            if cost_val is None or cost_val >= 80:
                rejection_reasons['costmap'] += 1
                continue

            # 조건 2: LOS from candidate to target
            los = self._los_between((cx, cy), (tx, ty))
            if los == LOSResult.BLOCKED:
                rejection_reasons['los'] += 1
                continue

            # 후보의 target 방향 yaw
            cand_yaw_target = math.atan2(ty - cy, tx - cx)

            # next_target 가중 보간으로 최종 yaw 계산
            if next_target_map is not None:
                nx, ny, _ = next_target_map
                ndx, ndy = nx - cx, ny - cy
                if math.hypot(ndx, ndy) > 1e-3:
                    yaw_next = math.atan2(ndy, ndx)
                    # 짧은 경로 보간: diff ∈ [-π, π]
                    diff = ((yaw_next - cand_yaw_target + math.pi)
                            % (2.0 * math.pi)) - math.pi
                    final_yaw = (cand_yaw_target
                                 + vp_cfg.yaw_next_weight * diff)
                else:
                    final_yaw = cand_yaw_target
            else:
                final_yaw = cand_yaw_target

            # 조건 3: OMX aim feasibility
            # 와플이 final_yaw 향한 상태에서 OMX 가 target 향할 yaw
            omx_req = cand_yaw_target - final_yaw
            omx_req = ((omx_req + math.pi) % (2.0 * math.pi)) - math.pi
            if abs(math.degrees(omx_req)) > vp_cfg.omx_yaw_limit_deg:
                rejection_reasons['omx'] += 1
                continue

            # Cost 계산
            dist_to_target = math.hypot(tx - cx, ty - cy)
            dist_from_waffle = math.hypot(wx - cx, wy - cy)
            cost = (W_INFL * (cost_val / 100.0)
                    + W_DIST * abs(dist_to_target - stand_off)
                    + W_WAFL * dist_from_waffle)

            valid.append((cost, cx, cy, final_yaw, cost_val, dist_from_waffle))

        if not valid:
            self.get_logger().warn(
                f"VIEW_POSE v2: 적합 후보 없음 ({n}개 중 "
                f"costmap={rejection_reasons['costmap']}, "
                f"LOS={rejection_reasons['los']}, "
                f"OMX={rejection_reasons['omx']})")
            return None

        valid.sort()
        cost, bx, by, byaw, infl, dw = valid[0]
        self.get_logger().info(
            f"VIEW_POSE v2: {n}개 중 {len(valid)} 적합, "
            f"선택 cost={cost:.2f} (infl={infl}, "
            f"dist_waffle={dw:.2f}m) "
            f"-> ({bx:+.2f}, {by:+.2f}) yaw={math.degrees(byaw):+.1f}°")
        return (bx, by, byaw)

    def transform_map_to_arm_base(self, coord_map):
        ps = PointStamped()
        ps.header.frame_id = 'map'
        ps.header.stamp = self.get_clock().now().to_msg()
        ps.point.x, ps.point.y, ps.point.z = coord_map

        try:
            transform = self.tf_buffer.lookup_transform(
                target_frame='base_link',
                source_frame='map',
                time=rclpy.time.Time(),
                timeout=Duration(seconds=0.1),
            )
        except TransformException as e:
            self.get_logger().warn(f"TF lookup 실패: {e}")
            return None

        try:
            ps_base = do_transform_point(ps, transform)
        except Exception as e:
            self.get_logger().warn(f"do_transform_point 실패: {e}")
            return None

        return (
            ps_base.point.x - self.arm_offset[0],
            ps_base.point.y - self.arm_offset[1],
            ps_base.point.z - self.arm_offset[2],
        )

    # ----- Subscribers -----

    def on_control_mode(self, msg):
        if msg.data == "idle":
            self.sm.on_abort()
            self.ctrl.go_home()

    def on_target_in_map(self, msg: PointStamped):
        coord = (msg.point.x, msg.point.y, msg.point.z)
        self.sm.on_target(coord)

    def on_boundary_in_map(self, msg: PointStamped):
        # H2: 외부 토픽 입력 (디버그/수동). H4 에서 내부 자동 생성과 공존.
        coord = (msg.point.x, msg.point.y, msg.point.z)
        self.sm.on_boundary(coord)

    def on_patrol_in_map(self, msg: PointStamped):
        coord = (msg.point.x, msg.point.y, msg.point.z)
        self.sm.on_patrol(coord)

    def on_arm_enable(self, msg):
        self.sm.on_arm_enable(msg.data)

    def on_abort(self, msg):
        self.sm.on_abort()
        self.ctrl.go_home()

    def on_nav_result(self, msg: String):
        """H2: waffle_node 가 발행한 Nav2 액션 결과."""
        self.sm.on_nav_result(msg.data)

    def on_boundary_enable(self, msg: String):
        """H4: BOUNDARY 자동 생성 런타임 토글.
        
        메시지 형식 (소문자): 
            'target on' / 'target off'
            'patrol on' / 'patrol off'
            'all on' / 'all off'
        """
        try:
            which, action = msg.data.lower().strip().split()
        except ValueError:
            self.get_logger().warn(
                f"잘못된 형식: '{msg.data}' (예: 'target on', 'all off')")
            return

        on = (action == 'on')
        if action not in ('on', 'off'):
            self.get_logger().warn(f"action 은 on/off: '{action}'")
            return

        if which == 'target':
            self.boundary_gen.set_enabled(target=on)
        elif which == 'patrol':
            self.boundary_gen.set_enabled(patrol=on)
        elif which == 'all':
            self.boundary_gen.set_enabled(target=on, patrol=on)
        else:
            self.get_logger().warn(
                f"unknown target: '{which}' (target/patrol/all 만)")
            return

        self.get_logger().info(
            f"Boundary toggle: T={self.boundary_gen.enabled_target} "
            f"P={self.boundary_gen.enabled_patrol}")

    # ----- Publishers -----

    def publish_periodic(self):
        msg = String()
        prefix = "dry_run_" if self.dry_run else ""
        if self.paused:
            prefix = "paused_"
        msg.data = f"{prefix}{self.sm.state.value}"
        self.pub_status.publish(msg)

        qmsg = Int32()
        qmsg.data = self.sm.queue_size()
        self.pub_queue_size.publish(qmsg)

        self.publish_queue_markers()

    def publish_state_change(self):
        if self.sm.state != self._last_state:
            msg = String()
            msg.data = self.sm.state.value
            self.pub_state.publish(msg)
            self._last_state = self.sm.state

    def publish_detected(self, detected):
        msg = Bool()
        msg.data = detected
        self.pub_detected.publish(msg)

    def publish_error(self, ex, ey):
        msg = Point()
        msg.x = float(ex)
        msg.y = float(ey)
        msg.z = 0.0
        self.pub_error.publish(msg)

    def publish_joint_state(self):
        try:
            positions = self.ctrl.read_joint_positions_rad()
        except Exception:
            return
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = list(positions.keys())
        msg.position = list(positions.values())
        self.pub_joint.publish(msg)

    def publish_progress(self, p):
        msg = Float32()
        msg.data = float(p)
        self.pub_progress.publish(msg)

    def publish_fire(self):
        self.pub_fire.publish(Empty())

    def _make_point_stamped(self, coord_map):
        msg = PointStamped()
        msg.header.frame_id = 'map'
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.point.x, msg.point.y, msg.point.z = coord_map
        return msg

    def publish_processed(self, coord_map):
        if coord_map is None:
            return
        self.pub_processed.publish(self._make_point_stamped(coord_map))

    def publish_target_lost(self, coord_map):
        if coord_map is None:
            return
        self.pub_target_lost.publish(self._make_point_stamped(coord_map))
        self.get_logger().info(f"[target_lost] 발행: {coord_map}")

    def publish_target_blocked(self, coord_map, type_name=""):
        if coord_map is None:
            return
        self.pub_target_blocked.publish(self._make_point_stamped(coord_map))
        self.get_logger().info(
            f"[target_blocked] 발행 ({type_name}): {coord_map}")

    def publish_patrol_complete(self):
        self.pub_patrol_complete.publish(Empty())
        self.get_logger().info("[patrol_complete] 발행")

    def publish_nav_goal(self, view_pose):
        """H2: VIEW_POSE 를 PoseStamped 로 발행."""
        x, y, yaw = view_pose
        msg = PoseStamped()
        msg.header.frame_id = 'map'
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.pose.position.x = x
        msg.pose.position.y = y
        msg.pose.position.z = 0.0
        # yaw -> quaternion
        msg.pose.orientation.z = math.sin(yaw / 2.0)
        msg.pose.orientation.w = math.cos(yaw / 2.0)
        self.pub_nav_goal.publish(msg)
        self.get_logger().info(
            f"[nav_goal] 발행: ({x:+.2f}, {y:+.2f}) "
            f"yaw={math.degrees(yaw):+.1f}°")
        # H3.2: 새 nav 시작 → 옛 nav_result 폐기 (race 방지)
        if self.sm.nav_pending_result is not None:
            self.get_logger().warn(
                f"이전 nav_result ({self.sm.nav_pending_result}) 폐기 "
                f"- 새 nav 시작")
            self.sm.nav_pending_result = None

    def publish_nav_cancel(self):
        """H3: TARGET preempt 시 진행 중 Nav2 cancel 요청."""
        self.pub_nav_cancel.publish(Empty())
        self.get_logger().info("[nav_cancel] 발행 (preempt)")

    def publish_target_not_found(self, coord_map):
        """H3: TARGET 좌표에서 scan_timeout 안에 표적 못 찾음."""
        if coord_map is None:
            return
        self.pub_target_not_found.publish(self._make_point_stamped(coord_map))
        self.get_logger().info(f"[target_not_found] 발행: {coord_map}")

    def publish_queue_markers(self):
        if not self.cfg.patrol.publish_queue_markers:
            return

        marker_array = MarkerArray()
        now_stamp = self.get_clock().now().to_msg()

        type_colors = {
            TargetType.TARGET:   (1.0, 0.2, 0.2),
            TargetType.BOUNDARY: (1.0, 0.6, 0.0),
            TargetType.PATROL:   (1.0, 1.0, 0.2),
        }
        type_sizes = {
            TargetType.TARGET:   0.25,
            TargetType.BOUNDARY: 0.18,
            TargetType.PATROL:   0.12,
        }

        delete_marker = Marker()
        delete_marker.header.frame_id = 'map'
        delete_marker.action = Marker.DELETEALL
        marker_array.markers.append(delete_marker)

        # 현재 focus (parent 또는 boundary)
        if self.sm.current_focus is not None:
            m = Marker()
            m.header.frame_id = 'map'
            m.header.stamp = now_stamp
            m.ns = 'queue_current'
            m.id = 0
            m.type = Marker.SPHERE
            m.action = Marker.ADD
            m.pose.position.x = self.sm.current_focus.coord_map[0]
            m.pose.position.y = self.sm.current_focus.coord_map[1]
            m.pose.position.z = self.sm.current_focus.coord_map[2]
            m.pose.orientation.w = 1.0
            m.scale.x = m.scale.y = m.scale.z = 0.35
            m.color.r = 0.2
            m.color.g = 1.0
            m.color.b = 0.2
            m.color.a = 0.9
            marker_array.markers.append(m)

        # 모든 큐 entry
        all_entries = list(self.sm.main_queue) + list(self.sm.boundary_queue)
        for i, entry in enumerate(all_entries):
            m = Marker()
            m.header.frame_id = 'map'
            m.header.stamp = now_stamp
            m.ns = 'queue'
            m.id = i
            m.type = Marker.SPHERE
            m.action = Marker.ADD
            m.pose.position.x = entry.coord_map[0]
            m.pose.position.y = entry.coord_map[1]
            m.pose.position.z = entry.coord_map[2]
            m.pose.orientation.w = 1.0
            size = type_sizes.get(entry.target_type, 0.15)
            m.scale.x = m.scale.y = m.scale.z = size
            r, g, b = type_colors.get(entry.target_type, (0.5, 0.5, 0.5))
            m.color.r = r
            m.color.g = g
            m.color.b = b
            m.color.a = 0.8
            marker_array.markers.append(m)

            t = Marker()
            t.header.frame_id = 'map'
            t.header.stamp = now_stamp
            t.ns = 'queue_label'
            t.id = i
            t.type = Marker.TEXT_VIEW_FACING
            t.action = Marker.ADD
            t.pose.position.x = entry.coord_map[0]
            t.pose.position.y = entry.coord_map[1]
            t.pose.position.z = entry.coord_map[2] + 0.25
            t.pose.orientation.w = 1.0
            t.scale.z = 0.15
            t.color.r = t.color.g = t.color.b = 1.0
            t.color.a = 0.9
            t.text = f"{entry.type_name} {entry.distance:.1f}m"
            marker_array.markers.append(t)

        self.pub_queue_markers.publish(marker_array)

    # ----- Main loop -----

    def loop(self):
        frame = self.detector.read_frame()
        if frame is None:
            self.get_logger().warn("프레임 읽기 실패")
            return

        # 격발 후 COOLDOWN 동안엔 YOLO detect 스킵.
        # 화면/스트림(read_frame·visualize·debug_stream)은 계속 흐르되,
        # 재검출로 인한 조기 재격발을 막는다.
        # COOLDOWN 은 detect 결과(detected·error_norm)를 쓰지 않으므로 안전.
        if self.sm.state == State.COOLDOWN:
            detected, error_norm, bbox, conf = False, None, None, None
        else:
            detected, error_norm, bbox, conf = self.detector.detect(frame)

        now = time.time()
        action = self.sm.update(detected, error_norm, now)

        # blocked entries 알림
        for entry in action.get('blocked_entries', []):
            self.publish_target_blocked(entry.coord_map, entry.type_name)

        if not self.paused:
            if action['action'] == 'aim':
                coord_map = action['coord_map']
                coord_arm = self.transform_map_to_arm_base(coord_map)
                if coord_arm is None:
                    self.get_logger().warn(
                        f"TF 변환 실패, focus 종료: {coord_map}")
                    self.sm._on_focus_done()
                else:
                    self.ctrl.aim_at_coord(*coord_arm)
                    self.get_logger().info(
                        f"AIM: map{coord_map} -> arm{coord_arm}")

            elif action['action'] == 'track':
                self.ctrl.step_ibvs(*action['error'])

            elif action['action'] == 'fire':
                processed_map = (self.sm.current_focus.coord_map
                                 if self.sm.current_focus else None)
                self.publish_fire()
                self.ctrl.fire()
                self.publish_processed(processed_map)

            elif action['action'] == 'target_lost':
                self.publish_target_lost(action.get('lost_coord_map'))

            elif action['action'] == 'home':
                self.ctrl.go_home()

            elif action['action'] == 'nav_goal':
                # H2 신규
                vp = action['nav_goal_xyyaw']
                if vp is not None:
                    self.publish_nav_goal(vp)

        if action.get('patrol_complete', False):
            self.publish_patrol_complete()

        # H3: TARGET miss 알림
        if action.get('target_not_found_coord') is not None:
            self.publish_target_not_found(action['target_not_found_coord'])

        # H4: BOUNDARY 자동 생성 (WAITING_NAV + PATROL parent 일 때만)
        if (self.sm.state == State.WAITING_NAV
                and self.sm.current_parent is not None):
            coord = self.boundary_gen.maybe_generate(
                now, self.sm.current_parent.target_type)
            if coord is not None:
                self.sm.on_boundary(coord)

        self.publish_detected(detected)
        if error_norm is not None:
            self.publish_error(error_norm[0], error_norm[1])
        self.publish_joint_state()
        self.publish_progress(action.get('confirm_progress', 0.0))
        self.publish_state_change()

        # 그리기: display 또는 stream 둘 중 하나라도 필요하면 호출
        need_viz = (not self.no_display) or (self.debug_stream is not None)
        if need_viz:
            self.visualize(frame, detected, error_norm, bbox, conf, action)
            if self.debug_stream is not None:
                self.debug_stream.update(frame)
                self.debug_stream.update_state(self._make_snapshot(action, error_norm))

        # 표시: GUI 가 있을 때만
        if not self.no_display:
            cv2.imshow("OMX YOLO node", frame)
            key = cv2.waitKey(1) & 0xFF
            self._handle_key(key)

        self.fps_n += 1
        if now - self.fps_t >= 1.0:
            self.fps_disp = self.fps_n / (now - self.fps_t)
            self.fps_t = now
            self.fps_n = 0
    
    def _make_snapshot(self, action, error_norm) -> dict:
        """SSE 로 보낼 상태 스냅샷.

        가벼운 dict 만 만들기 - 직렬화는 Flask 스레드가 함.
        """
        def entry_dict(e):
            if e is None:
                return None
            return {
                'priority': e.priority,
                'type': e.target_type.name,
                'coord': [round(c, 3) for c in e.coord_map],
                'distance': round(e.distance, 3),
            }

        return {
            'ts': time.time(),
            'state': self.sm.state.value,
            'armed': self.sm.armed,
            'paused': self.paused,
            'fps': round(self.fps_disp, 1),
            'confirm_progress': round(action.get('confirm_progress', 0.0), 3),
            'ibvs_error': ([round(error_norm[0], 3), round(error_norm[1], 3)]
                        if error_norm else None),
            'current_parent': entry_dict(self.sm.current_parent),
            'current_focus': entry_dict(self.sm.current_focus),
            'main_queue': [entry_dict(e) for e in self.sm.main_queue],
            'boundary_queue': [entry_dict(e) for e in self.sm.boundary_queue],
            'main_queue_size': len(self.sm.main_queue),
            'boundary_queue_size': len(self.sm.boundary_queue),
        }

    def visualize(self, frame, detected, error_norm, bbox, conf, action):
        h, w = frame.shape[:2]
        cx, cy = w / 2.0, h / 2.0
        deadband_x = self.cfg.ibvs.deadband_x
        deadband_y = self.cfg.ibvs.deadband_y

        cv2.drawMarker(frame, (int(cx), int(cy)),
                       (0, 255, 255), cv2.MARKER_CROSS, 20, 1)
        dz_x = int(deadband_x * cx)
        dz_y = int(deadband_y * cy)
        cv2.rectangle(frame,
                      (int(cx) - dz_x, int(cy) - dz_y),
                      (int(cx) + dz_x, int(cy) + dz_y),
                      (80, 80, 80), 1)

        if detected and bbox:
            x1, y1, x2, y2 = bbox
            state_color = {
                State.IDLE: (180, 180, 180),
                State.AIMING: (255, 200, 0),
                State.SCANNING: (200, 255, 200),
                State.TRACKING: (0, 255, 0),
                State.CONFIRMING: (0, 165, 255),
                State.FIRING: (0, 0, 255),
                State.COOLDOWN: (200, 100, 200),
                State.WAITING_NAV: (100, 200, 255),   # H2: 하늘색
            }
            color = state_color.get(self.sm.state, (255, 255, 255))
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            obj_x = (x1 + x2) / 2.0
            obj_y = (y1 + y2) / 2.0
            cv2.circle(frame, (int(obj_x), int(obj_y)), 4, color, -1)
            cv2.line(frame, (int(cx), int(cy)),
                     (int(obj_x), int(obj_y)), color, 1)
            cv2.putText(frame, f"{self.detector.class_name} {conf:.2f}",
                        (x1, max(y1 - 8, 16)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        state_txt = f"[{self.sm.state.value.upper()}]"
        if self.paused:
            state_txt = f"[PAUSED|{self.sm.state.value}]"
        if self.dry_run:
            state_txt = f"[DRY|{self.sm.state.value}]"

        armed_txt = "ARMED" if self.sm.armed else "DISARMED"
        queue_txt = (f"Q:m{len(self.sm.main_queue)}"
                     f"/b{len(self.sm.boundary_queue)}")
        costmap_txt = "MAP:OK" if self.costmap else "MAP:--"

        focus_txt = ""
        if self.sm.current_focus is not None:
            is_b = action.get('focus_is_boundary', False)
            tag = "B" if is_b else self.sm.current_focus.type_name[0]
            focus_txt = (f" [{tag}:{self.sm.current_focus.distance:.1f}m]")

        cv2.putText(frame,
                    f"{state_txt}{focus_txt} {armed_txt} {queue_txt} {costmap_txt}",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 255, 255), 1)
        cv2.putText(frame,
                    f"yaw={math.degrees(self.ctrl.yaw):+.1f} "
                    f"pitch={math.degrees(self.ctrl.pitch):+.1f} "
                    f"fps={self.fps_disp:.1f}",
                    (10, 45), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 255, 255), 1)

        # TRACKING lost progress
        if (self.sm.state == State.TRACKING
                and self.sm.lost_start_t > 0.0):
            elapsed = time.time() - self.sm.lost_start_t
            timeout = self.cfg.fire.lost_timeout_sec
            lost_progress = min(1.0, elapsed / timeout)
            bar_x, bar_y, bar_w, bar_h = 10, h - 100, 200, 12
            cv2.rectangle(frame, (bar_x, bar_y),
                         (bar_x + bar_w, bar_y + bar_h),
                         (100, 100, 100), 1)
            cv2.rectangle(frame, (bar_x, bar_y),
                         (bar_x + int(bar_w * lost_progress), bar_y + bar_h),
                         (0, 100, 255), -1)
            cv2.putText(frame, f"LOST {elapsed:.1f}/{timeout:.1f}s",
                        (bar_x + bar_w + 10, bar_y + 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 100, 100), 1)

        # SCANNING progress
        if self.sm.state == State.SCANNING:
            scan_timeout = (self.cfg.patrol.scan_timeout_sec
                            if self.cfg.patrol else 2.0)
            elapsed = time.time() - self.sm.scan_start_t
            scan_progress = min(1.0, elapsed / scan_timeout)
            bar_x, bar_y, bar_w, bar_h = 10, h - 80, 200, 12
            cv2.rectangle(frame, (bar_x, bar_y),
                         (bar_x + bar_w, bar_y + bar_h),
                         (100, 100, 100), 1)
            cv2.rectangle(frame, (bar_x, bar_y),
                         (bar_x + int(bar_w * scan_progress), bar_y + bar_h),
                         (100, 255, 100), -1)
            cv2.putText(frame, f"SCAN {elapsed:.1f}/{scan_timeout:.1f}s",
                        (bar_x + bar_w + 10, bar_y + 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        # CONFIRMING progress
        progress = action.get('confirm_progress', 0.0)
        if progress > 0 or self.sm.state == State.CONFIRMING:
            bar_x, bar_y, bar_w, bar_h = 10, h - 60, 200, 15
            cv2.rectangle(frame, (bar_x, bar_y),
                         (bar_x + bar_w, bar_y + bar_h),
                         (100, 100, 100), 1)
            cv2.rectangle(frame, (bar_x, bar_y),
                         (bar_x + int(bar_w * progress), bar_y + bar_h),
                         (0, 165, 255), -1)
            cv2.putText(frame, f"AIM {progress*100:.0f}%",
                        (bar_x + bar_w + 10, bar_y + 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        if error_norm is not None:
            cv2.putText(frame,
                        f"err=({error_norm[0]:+.2f}, {error_norm[1]:+.2f})",
                        (10, h - 40), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (255, 255, 255), 1)

        cv2.putText(frame, "p:pause a:arm h:home/clear ESC:quit",
                    (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (180, 180, 180), 1)

        # cv2.imshow("OMX YOLO node", frame)

    def _handle_key(self, key):
        if key == 27:
            self.get_logger().info("ESC. 종료.")
            rclpy.shutdown()
        elif key == ord('p'):
            self.paused = not self.paused
            self.get_logger().info("일시정지" if self.paused else "재개")
        elif key == ord('a'):
            self.sm.armed = not self.sm.armed
            self.get_logger().info(f"Armed: {self.sm.armed}")
        elif key == ord('h'):
            self.get_logger().info("Home + 모든 큐 비움 (수동)")
            self.sm.on_abort()
            self.ctrl.go_home()

    def destroy_node(self):
        if hasattr(self, 'detector'):
            self.detector.release()
        cv2.destroyAllWindows()
        if hasattr(self, 'ctrl'):
            self.ctrl.disconnect()
        super().destroy_node()


def main(args=None):
    import argparse
    parser = argparse.ArgumentParser(
        description="OMX YOLO ROS 2 node - Stage H4 (modular)")
    parser.add_argument("--dry-run", action="store_true",
                        help="OMX 없이 카메라 + 검출만")
    parser.add_argument("--no-display", action="store_true",
                        help="OpenCV 화면 표시 끔 (헤드리스 SSH 환경 등)")
    parser.add_argument("--debug-stream", action="store_true",
                        help="Flask MJPEG 디버그 스트림 (http://host:port/)")
    parser.add_argument("--debug-port", type=int, default=8080,
                        help="--debug-stream 포트 (기본 8080)")
    parser.add_argument("--debug-fps", type=int, default=15,
                        help="--debug-stream FPS 제한 (기본 15)")
    parser.add_argument("--debug-quality", type=int, default=70,
                        help="--debug-stream JPEG quality 10~95 (기본 70)")
    cli_args, ros_args = parser.parse_known_args()

    rclpy.init(args=ros_args)

    try:
        node = OmxYoloNode(dry_run=cli_args.dry_run,
                   no_display=cli_args.no_display,
                   debug_stream=cli_args.debug_stream,
                   debug_port=cli_args.debug_port,
                   debug_fps=cli_args.debug_fps,
                   debug_quality=cli_args.debug_quality)
        try:
            rclpy.spin(node)
        finally:
            node.destroy_node()
    except KeyboardInterrupt:
        print("\n중단됨.")
    except Exception as e:
        print(f"노드 에러: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()