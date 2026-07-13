"""StateMachine — OMX Auto-Aim 의 상태 머신 + 큐 정책.

이 모듈은 ROS 의존성이 없다 (콜백 패턴으로 ROS 와 분리).
StateMachine 사용자 (OmxYoloNode) 가 다음 콜백을 주입한다:

    los_check_fn(coord_map)         -> LOSResult
    waffle_pos_fn()                 -> (x, y) or None
    check_view_fn(coord_map)        -> bool        (H2: 현 위치에서 조준 가능?)
    compute_view_pose_fn(target,    -> (x,y,yaw) or None  (H2: 와플 이동 위치)
                        next=None)
    nav_cancel_fn()                 -> None        (H3: /omx/nav_cancel publish)
    plan_waypoints_fn(start_xy,     -> [(x,y), ...] or None  (H5.1: A*
                       goal_xy)        waypoint 리스트)

상세 정책은 INTERFACE_v3.md 의 3-4 절 참조.
"""

from __future__ import annotations

import heapq
import math
import time
from typing import Optional

from omx.config import Config
from omx.types import State, TargetType, LOSResult, TargetEntry


class StateMachine:
    def __init__(self, cfg: Config, logger=None):
        self.cfg = cfg
        self.logger = logger
        self.state = State.IDLE

        # H2: 큐 분리
        self.main_queue: list[TargetEntry] = []      # TARGET + PATROL
        self.boundary_queue: list[TargetEntry] = []  # BOUNDARY only

        # H2: 부모 + focus 분리
        self.current_parent: Optional[TargetEntry] = None  # 처리 중 TARGET/PATROL
        self.current_focus: Optional[TargetEntry] = None   # OMX 가 조준 중 (parent or boundary)

        # H2: nav_result 비동기 처리
        self.nav_pending_result: Optional[str] = None
        # H3: preempt cancel 결과 무시용 flag
        self.pending_cancel_for_preempt: bool = False

        # H5.1: A* waypoint crawl -- WAITING_NAV 동안 유지되는 현재 이동 경로.
        # nav_waypoints[0] 이 현재 추적 중인 waypoint, 도착하면 pop.
        self.nav_waypoints: list[tuple] = []
        self.nav_final_view_pose: Optional[tuple] = None
        self.nav_last_eval_t: float = 0.0

        # 타이머/플래그
        self.aim_start_t: float = 0.0      # H2.1: AIMING 진입 시각
        self.scan_start_t: float = 0.0
        self.confirm_start_t: float = 0.0
        self.confirm_progress: float = 0.0
        self.cooldown_until: float = 0.0
        self.cooldown_home_sent: bool = False
        self.fire_start_t: float = 0.0     # 격발 펄스 시작 시각 (home 지연용)
        self.lost_start_t: float = 0.0

        self.armed = cfg.autotrack.default_armed if cfg.autotrack else False
        self.last_processed: Optional[tuple] = None
        self.patrol_complete_sent = True

        # 콜백 (OmxYoloNode 가 주입)
        self.los_check_fn = None              # (coord_map) -> LOSResult
        self.waffle_pos_fn = None             # () -> (x, y) or None
        self.check_view_fn = None             # H2: (coord_map) -> bool
        self.compute_view_pose_fn = None      # H2: (coord_map) -> (x,y,yaw) or None
        self.nav_cancel_fn = None             # H3: () -> publish /omx/nav_cancel
        self.plan_waypoints_fn = None         # H5.1: (start_xy, goal_xy) -> [(x,y)] or None

    def _log(self, msg):
        if self.logger:
            self.logger.info(msg)
        else:
            print(msg)

    def transition(self, new_state: State):
        if self.state != new_state:
            self._log(f"State: {self.state.value} -> {new_state.value}")
            self.state = new_state

    # ----- Queue 조작 -----

    def add_target(self, coord, target_type: TargetType,
                   parent_id: Optional[int] = None) -> bool:
        # 큐 선택
        if target_type == TargetType.BOUNDARY:
            queue = self.boundary_queue
            max_size = (self.cfg.boundary.max_queue_size
                        if self.cfg.boundary else 10)
        else:
            queue = self.main_queue
            max_size = (self.cfg.patrol.max_queue_size
                        if self.cfg.patrol else 20)

        # 격발 중 main 큐 추가 금지 (BOUNDARY 는 OK - 큐만 쌓임)
        if (target_type != TargetType.BOUNDARY
                and self.state in (State.CONFIRMING, State.FIRING)):
            self._log(f"좌표 무시 (state={self.state.value}, 격발 우선): "
                      f"type={target_type.name}")
            return False

        if self._is_duplicate(coord, target_type):
            self._log(f"좌표 중복 무시: {coord} ({target_type.name})")
            return False

        if len(queue) >= max_size:
            removed = self._remove_oldest(queue)
            if not removed:
                self._log(f"큐 가득 ({max_size}), 추가 거부")
                return False

        entry = TargetEntry(
            priority=int(target_type),
            coord_map=coord,
            target_type=target_type,
            arrival_time=time.time(),
            parent_id=parent_id,
        )
        if self.waffle_pos_fn:
            entry.update_distance(self.waffle_pos_fn())

        heapq.heappush(queue, entry)
        if target_type != TargetType.BOUNDARY:
            self.patrol_complete_sent = False

        self._log(f"큐 추가: type={target_type.name} "
                  f"coord={coord} dist={entry.distance:.2f}m, "
                  f"main={len(self.main_queue)} bnd={len(self.boundary_queue)}")
        return True

    def _is_duplicate(self, coord, target_type) -> bool:
        """같은 type 끼리만 중복 비교 (H3.1: PATROL → TARGET 업그레이드 허용)."""
        if not self.cfg.patrol:
            return False
        threshold = self.cfg.patrol.duplicate_threshold_m

        # BOUNDARY 는 BOUNDARY 끼리만 비교
        if target_type == TargetType.BOUNDARY:
            for entry in self.boundary_queue:
                if self._distance(coord, entry.coord_map) < threshold:
                    return True
            return False

        # PATROL/TARGET 처리
        # last_processed 와 비교: PATROL 만 (TARGET 은 외부 신뢰 신호로 재처리 허용)
        if (target_type == TargetType.PATROL
                and self.last_processed
                and self._distance(coord, self.last_processed) < threshold):
            return True

        # main_queue 와 same-type 비교
        for entry in self.main_queue:
            if (entry.target_type == target_type
                    and self._distance(coord, entry.coord_map) < threshold):
                return True

        # current_parent 와 same-type 비교
        if (self.current_parent is not None
                and self.current_parent.target_type == target_type
                and self._distance(coord, self.current_parent.coord_map) < threshold):
            return True

        return False

    def _distance(self, a, b):
        return math.sqrt(sum((ai - bi)**2 for ai, bi in zip(a, b)))

    def _remove_oldest(self, queue) -> bool:
        if not queue:
            return False
        oldest_idx = 0
        for i, entry in enumerate(queue):
            if entry.count < queue[oldest_idx].count:
                oldest_idx = i
        removed = queue.pop(oldest_idx)
        heapq.heapify(queue)
        self._log(f"큐 가득, 오래된 {removed.type_name} 제거")
        return True

    def _pop_with_los(self, queue, waffle_xy=None):
        """LOS 검사 통과한 entry pop. 거리 기반 재정렬 후.
        
        H4: BOUNDARY 는 TTL 도 검사 (오래된 좌표는 와플 위치 달라져 의미 없음).
        """
        blocked_entries = []

        if waffle_xy is not None:
            for entry in queue:
                entry.update_distance(waffle_xy)
            heapq.heapify(queue)

        # H4: BOUNDARY TTL 설정
        now_t = time.time()
        ttl = (self.cfg.boundary.ttl_sec
               if self.cfg.boundary else 10.0)

        while queue:
            entry = heapq.heappop(queue)

            # H4: BOUNDARY TTL 검사
            if entry.target_type == TargetType.BOUNDARY:
                age = now_t - entry.arrival_time
                if age > ttl:
                    self._log(f"BOUNDARY TTL 초과 ({age:.1f}s > {ttl:.1f}s) "
                              f"폐기: {entry.coord_map}")
                    continue  # 다음 entry

            if (not self.cfg.patrol
                    or not self.cfg.patrol.los_check_enabled
                    or self.los_check_fn is None):
                return entry, blocked_entries

            result = self.los_check_fn(entry.coord_map)

            if result == LOSResult.CLEAR:
                return entry, blocked_entries
            elif result == LOSResult.UNKNOWN:
                if entry.target_type == TargetType.BOUNDARY:
                    self._log(f"LOS UNKNOWN, BOUNDARY 탐색 허용: "
                              f"{entry.coord_map}")
                    return entry, blocked_entries
                else:
                    return entry, blocked_entries
            elif result == LOSResult.BLOCKED:
                if entry.target_type == TargetType.BOUNDARY:
                    self._log(f"LOS BLOCKED, BOUNDARY 폐기: {entry.coord_map}")
                    blocked_entries.append(entry)
                    continue
                else:
                    return entry, blocked_entries

        return None, blocked_entries

    def clear_boundary_queue(self):
        cleared = len(self.boundary_queue)
        self.boundary_queue.clear()
        if cleared > 0:
            self._log(f"BOUNDARY 큐 {cleared}개 일괄 폐기")

    def queue_size(self) -> int:
        return len(self.main_queue) + len(self.boundary_queue)

    @property
    def queue(self):
        """RViz 마커용. main + boundary 통합 view."""
        return self.main_queue + self.boundary_queue

    # ----- 입력 핸들러 -----

    def on_target(self, coord) -> bool:
        # H3.1: 1) main_queue 에 같은 위치 PATROL 이 있으면 제거 (업그레이드)
        self._upgrade_patrol_in_queue_to_target(coord)
        # 2) TARGET 큐 추가
        accepted = self.add_target(coord, TargetType.TARGET)
        if accepted:
            # 3) current_parent 가 PATROL 이면 같은/다른 위치 분기 처리
            self._maybe_preempt_for_target(coord)
        return accepted

    def _upgrade_patrol_in_queue_to_target(self, target_coord):
        """main_queue 의 같은 위치 PATROL 항목 제거 (TARGET 으로 대체될 예정).
        
        current_parent 의 PATROL 은 _maybe_preempt_for_target 에서 처리.
        """
        if not self.cfg.patrol:
            return
        threshold = self.cfg.patrol.duplicate_threshold_m

        new_queue = []
        removed = 0
        for entry in self.main_queue:
            if (entry.target_type == TargetType.PATROL
                    and self._distance(entry.coord_map, target_coord) < threshold):
                removed += 1
                self._log(f"PATROL → TARGET 업그레이드 (큐): "
                          f"{entry.coord_map} 제거")
            else:
                new_queue.append(entry)
        if removed > 0:
            self.main_queue = new_queue
            heapq.heapify(self.main_queue)

    def on_boundary(self, coord, parent_id=None) -> bool:
        return self.add_target(coord, TargetType.BOUNDARY, parent_id=parent_id)

    def on_patrol(self, coord) -> bool:
        return self.add_target(coord, TargetType.PATROL)

    def on_nav_result(self, result: str):
        """waffle_node 의 nav_result 비동기 수신. 다음 tick 에서 처리."""
        self.nav_pending_result = result
        self._log(f"nav_result 받음: {result}")

    # ----- H3: TARGET preempt -----

    def _preempt_ok(self) -> bool:
        """현재 상황에서 preempt 가능?
        
        조건:
            - current_parent 가 PATROL
            - state 가 WAITING_NAV / AIMING / SCANNING 중 하나
              (TRACKING/CONFIRMING/FIRING/COOLDOWN 은 끝까지 처리)
        """
        if self.current_parent is None:
            return False
        if self.current_parent.target_type != TargetType.PATROL:
            return False
        if self.state not in (State.WAITING_NAV,
                              State.AIMING,
                              State.SCANNING):
            return False
        return True

    def _is_waffle_navigating(self) -> bool:
        """waffle 이 현재 Nav2 로 이동 중인가?"""
        if self.state == State.WAITING_NAV:
            return True
        # boundary 처리 중 (WAITING_NAV 의 임시 transient)
        if self.state in (State.AIMING, State.SCANNING):
            if (self.current_focus is not None
                    and self.current_focus.target_type == TargetType.BOUNDARY):
                return True
        return False

    def _maybe_preempt_for_target(self, target_coord):
        """방금 TARGET 이 추가되었을 때 PATROL preempt 시도 (H3.1).
        
        같은 위치: PATROL 폐기 (업그레이드)
        다른 위치: PATROL 큐 복귀 (priority 로 자동 재정렬)
        """
        if not self._preempt_ok():
            return

        threshold = (self.cfg.patrol.duplicate_threshold_m
                     if self.cfg.patrol else 0.3)
        parent_coord = self.current_parent.coord_map
        same_location = (self._distance(parent_coord, target_coord) < threshold)

        loc_tag = "same" if same_location else "different"
        self._log(f"=== TARGET preempt 발동 "
                  f"(state={self.state.value}, "
                  f"parent_loc={loc_tag}) ===")

        # 와플 이동 중이면 cancel 요청
        if self._is_waffle_navigating():
            if self.nav_cancel_fn:
                self.nav_cancel_fn()
                self.pending_cancel_for_preempt = True
                self._log("nav_cancel 발송")

        # PATROL 처리 분기
        if same_location:
            # 같은 위치 → 업그레이드 (PATROL 폐기)
            self._log(f"PATROL → TARGET 업그레이드: {parent_coord} 폐기")
            self.current_parent = None
        else:
            # 다른 위치 → PATROL 큐 복귀 (priority 로 다시 정렬됨)
            patrol_entry = self.current_parent
            self.current_parent = None
            # heapq push (priority, distance, count 로 자동 정렬)
            heapq.heappush(self.main_queue, patrol_entry)
            self._log(f"PATROL 큐 복귀: {patrol_entry.coord_map} "
                      f"(TARGET 처리 후 자동 재처리)")

        # 공통 정리
        self.current_focus = None
        self.boundary_queue.clear()
        self.confirm_progress = 0.0
        self.lost_start_t = 0.0
        self.nav_waypoints = []                    # H5.1
        self.nav_final_view_pose = None
        self.transition(State.IDLE)

    def on_abort(self):
        self._log("ABORT - IDLE + 모든 큐 비움")
        # 안전: 상태 무관 무조건 nav_cancel 발송.
        # waffle_node 가 IDLE 이면 어차피 무시 (이미 그렇게 구현돼 있음).
        if self.nav_cancel_fn is not None:
            self.nav_cancel_fn()
            self._log("ABORT: nav_cancel 발송")

        self.transition(State.IDLE)
        self.main_queue.clear()
        self.boundary_queue.clear()
        self.current_parent = None
        self.current_focus = None
        self.confirm_progress = 0.0
        self.cooldown_home_sent = False
        self.patrol_complete_sent = True
        self.lost_start_t = 0.0
        self.nav_pending_result = None
        self.pending_cancel_for_preempt = False    # H3.2
        self.nav_waypoints = []                    # H5.1
        self.nav_final_view_pose = None

    def on_arm_enable(self, armed: bool):
        self.armed = armed
        self._log(f"Armed: {armed}")

    def force_fire_now(self, now: float, *, cancel_nav: bool = True,
                       reason: str = "immediate_detection") -> bool:
        """현재 작업을 접고 즉시 격발 쿨다운 상태로 진입한다."""
        if self.state in (State.FIRING, State.COOLDOWN):
            return False

        if cancel_nav and self._is_waffle_navigating():
            if self.nav_cancel_fn is not None:
                self.nav_cancel_fn()
                self.pending_cancel_for_preempt = True
                self._log(f"{reason}: nav_cancel 발송")

        self.boundary_queue.clear()
        self.current_parent = None
        self.current_focus = None
        self.nav_pending_result = None
        self.nav_waypoints = []                    # H5.1
        self.nav_final_view_pose = None
        self.confirm_progress = 1.0
        self.lost_start_t = 0.0
        self.fire_start_t = now
        self.cooldown_until = now + self.cfg.fire.cooldown_sec
        self.cooldown_home_sent = False
        self.transition(State.COOLDOWN)
        self._log(f"{reason}: 즉시 격발 -> COOLDOWN")
        return True

    # ----- update() 메인 -----

    def update(self, detected: bool, error_norm, now: float, *, vision_valid: bool = True) -> dict:
        action = {
            'action': 'wait',
            'state': self.state,
            'coord_map': None,
            'error': None,
            'confirm_progress': 0.0,
            'patrol_complete': False,
            'lost_coord_map': None,
            'blocked_entries': [],
            'nav_goal_xyyaw': None,         # H2: (x, y, yaw) for /omx/nav_goal
            'focus_is_boundary': False,     # H2: 시각화용
            'target_not_found_coord': None, # H3
            'scan_sweep': False,
            'cancel_navigation': False,
            'cancel_reason': '',
        }

        # 1. nav_result 처리
        #    - H3: preempt cancel 결과는 state 무관하게 무시 (이미 TARGET 처리 중)
        #    - H2.1: 그 외엔 WAITING_NAV state 일 때만 적용
        if self.nav_pending_result is not None:
            if self.pending_cancel_for_preempt:
                self._log(f"preempt cancel 결과 ({self.nav_pending_result}) "
                          f"무시 - TARGET 처리 계속")
                self.nav_pending_result = None
                self.pending_cancel_for_preempt = False
            elif (
                self.state == State.WAITING_NAV
                or (
                    self.state in (State.AIMING, State.SCANNING)
                    and self.current_focus is not None
                    and self.current_focus.target_type == TargetType.BOUNDARY
                )
            ):
                result = self.nav_pending_result
                self.nav_pending_result = None
                self._handle_nav_result(result, action, now)

        detection_preempted_nav = (
            vision_valid
            and
            detected
            and error_norm is not None
            and self.state
            not in (State.TRACKING, State.CONFIRMING, State.FIRING, State.COOLDOWN)
        )
        if detection_preempted_nav:
            self.boundary_queue.clear()
            self.current_focus = None
            self.nav_waypoints = []
            self.nav_final_view_pose = None
            self.pending_cancel_for_preempt = self._is_waffle_navigating()
            action['cancel_navigation'] = True
            action['cancel_reason'] = 'target_detected_track'
            self.confirm_progress = 0.0
            self.lost_start_t = 0.0
            action['action'] = 'track'
            action['error'] = error_norm
            self.transition(State.TRACKING)
            self._log("카메라 탐지: 기존 작업 중단 -> 즉시 PD TRACKING")

        # 2. State 분기
        elif self.state == State.IDLE:
            self._on_idle(detected, action, now)

        elif self.state == State.WAITING_NAV:
            self._on_waiting_nav(action, now)

        elif self.state == State.AIMING:
            # H2.1: aim_settle_sec 동안 OMX 모터가 목표 각도로 이동.
            # 그동안 action='wait', 외부에 별다른 영향 없음.
            is_boundary = (
                self.current_focus is not None
                and self.current_focus.target_type == TargetType.BOUNDARY
            )
            aim_settle = (
                self.cfg.patrol.boundary_aim_settle_sec
                if is_boundary and self.cfg.patrol
                else self.cfg.fire.aim_settle_sec
            )
            if now - self.aim_start_t >= aim_settle:
                self.scan_start_t = now
                self.transition(State.SCANNING)
                action['action'] = 'scan_sweep'
                action['scan_sweep'] = True

        elif self.state == State.SCANNING:
            if vision_valid:
                self._on_scanning(detected, error_norm, now, action)
            else:
                action['action'] = 'scan_sweep'
                action['scan_sweep'] = True

        elif self.state == State.TRACKING:
            if vision_valid:
                self._on_tracking(detected, error_norm, now, action)

        elif self.state == State.CONFIRMING:
            if vision_valid:
                self._on_confirming(detected, error_norm, now, action)

        elif self.state == State.FIRING:
            action['action'] = 'fire'
            self.fire_start_t = now    # 격발 펄스 시작 (home 지연 기준)
            self.transition(State.COOLDOWN)
            self.cooldown_until = now + self.cfg.fire.cooldown_sec
            self.cooldown_home_sent = False

        elif self.state == State.COOLDOWN:
            self._on_cooldown(now, action)

        action['state'] = self.state
        action['confirm_progress'] = self.confirm_progress
        action['focus_is_boundary'] = (
            self.current_focus is not None
            and self.current_focus.target_type == TargetType.BOUNDARY)
        return action

    # ----- 핸들러: nav_result -----

    def _handle_nav_result(self, result: str, action: dict, now: float):
        """nav_result 적용.

        WAITING_NAV 이거나 BOUNDARY AIMING/SCANNING 중일 때 처리된다. 그래서
        OMX 가 주행 중 좌우를 훑는 동안에도 Nav2 다음 hop/도착 처리가 밀리지
        않는다.
        """
        # 도착 정책: boundary 큐 일괄 폐기 (적용 시점에만)
        self.clear_boundary_queue()

        if result == "succeeded":
            # H5.1: 방금 도착한 goal 은 현재 waypoint 하나에 대한 것.
            # 남은 구간이 있으면 다음 hop 으로, 없으면(원래 동작) parent
            # AIMING 진입. 중간 구간이 poll(_on_waypoint_eval) 보다 먼저
            # Nav2 자체 완료 신호로 끝난 경우도 여기서 자연스럽게 처리된다.
            if self.nav_waypoints:
                self.nav_waypoints.pop(0)
            if self.nav_waypoints:
                next_pose = self._next_hop_pose()
                if next_pose is not None:
                    self._log(
                        f"WAYPOINT_CRAWL: nav_result succeeded, 다음 hop -> "
                        f"{next_pose[:2]} (남은 {len(self.nav_waypoints)}개)")
                    action['action'] = 'nav_goal'
                    action['nav_goal_xyyaw'] = next_pose
                    return

            self.nav_final_view_pose = None
            if self.current_parent is not None:
                if self.current_parent.target_type == TargetType.PATROL:
                    self._log(f"PATROL Nav2 도착, parent 조준 생략: "
                              f"{self.current_parent.coord_map}")
                    self.nav_waypoints = []
                    self.current_parent = None
                    self.current_focus = None
                    self.transition(State.IDLE)
                    return

                # parent 의 AIMING 진입
                self.current_focus = self.current_parent
                self.transition(State.AIMING)
                self.aim_start_t = now    # H2.1
                action['action'] = 'aim'
                action['coord_map'] = self.current_parent.coord_map
                self._log(f"와플 도착, parent AIMING: "
                          f"{self.current_parent.coord_map}")
            else:
                self._log("nav_result succeeded 인데 parent 없음. IDLE 로.")
                self.transition(State.IDLE)
        else:
            # aborted / canceled / rejected
            self._log(f"Nav 실패 ({result}), parent 폐기")
            self.current_parent = None
            self.current_focus = None
            self.nav_waypoints = []
            self.nav_final_view_pose = None
            self.transition(State.IDLE)

    # ----- 핸들러: IDLE -----

    def _on_idle(self, detected: bool, action: dict, now: float):
        # 처리 끝났으니 focus/parent 정리
        if self.current_focus is not None:
            self.current_focus = None

        # main_queue 에서 pop 시도
        if self.main_queue:
            waffle_xy = self.waffle_pos_fn() if self.waffle_pos_fn else None
            entry, blocked = self._pop_with_los(self.main_queue, waffle_xy)
            action['blocked_entries'] = blocked

            if entry is None:
                return

            self.current_parent = entry
            self._log(f"main_queue pop: {entry.type_name} "
                      f"{entry.coord_map} dist={entry.distance:.2f}m")

            # TARGET 은 현재 위치에서 바로 보이면 즉시 조준한다.
            # PATROL 은 정찰 주행 자체가 목적이므로 바로 Nav2 를 출발시키고,
            # 이동 중 BOUNDARY sweep 으로 OMX 가 좌우를 훑는다.
            can_view = (
                entry.target_type != TargetType.PATROL
                and (
                    self.check_view_fn is None
                    or self.check_view_fn(entry.coord_map)
                )
            )

            if can_view:
                # 현 위치에서 조준 가능 → AIMING
                self.current_focus = entry
                self.last_processed = entry.coord_map
                self.transition(State.AIMING)
                self.aim_start_t = now    # H2.1
                action['action'] = 'aim'
                action['coord_map'] = entry.coord_map
                self._log("CHECK_VIEW: 현 위치 조준 가능 -> AIMING")
            else:
                # VIEW_POSE 계산 → 와플 이동.
                # H2.1: 도착 시 yaw 는 main_queue 의 다음 entry 방향
                # (다음 작업으로 빨리 출발하기 위해. OMX 가 ±180° 회전해서 조준).
                next_target_map = (self.main_queue[0].coord_map
                                   if self.main_queue else None)
                view_pose = (self.compute_view_pose_fn(
                                 entry.coord_map, next_target_map)
                             if self.compute_view_pose_fn else None)
                if view_pose is None:
                    if entry.target_type == TargetType.PATROL:
                        waffle_xy = (
                            self.waffle_pos_fn() if self.waffle_pos_fn else None
                        )
                        if waffle_xy is not None:
                            yaw = math.atan2(
                                entry.coord_map[1] - waffle_xy[1],
                                entry.coord_map[0] - waffle_xy[0],
                            )
                        else:
                            yaw = 0.0
                        view_pose = (entry.coord_map[0], entry.coord_map[1], yaw)
                        self._log(
                            "VIEW_POSE 계산 실패, PATROL 좌표로 바로 Nav2 출발")
                    else:
                        self._log(f"VIEW_POSE 계산 실패, parent 폐기")
                        self.last_processed = entry.coord_map
                        self.current_parent = None
                        return
                self.last_processed = entry.coord_map

                # H5.1: 옵션이 켜진 경우에만 A* waypoint crawl 을 쓴다.
                # 기본 운용은 최종 VIEW_POSE 로 Nav2 goal 을 바로 보낸다.
                waffle_xy = self.waffle_pos_fn() if self.waffle_pos_fn else None
                nav_cfg = self.cfg.nav_crawl
                if (
                    nav_cfg is not None
                    and nav_cfg.enabled
                    and self.plan_waypoints_fn
                    and waffle_xy is not None
                ):
                    waypoints = self.plan_waypoints_fn(waffle_xy, view_pose[:2])
                else:
                    waypoints = None
                if not waypoints:
                    waypoints = [view_pose[:2]]

                self.nav_waypoints = waypoints
                self.nav_final_view_pose = view_pose
                self.nav_last_eval_t = now
                self.transition(State.WAITING_NAV)
                action['action'] = 'nav_goal'
                action['nav_goal_xyyaw'] = self._next_hop_pose()
                self._log(f"CHECK_VIEW: 불가, VIEW_POSE={view_pose} "
                          f"waypoints={len(self.nav_waypoints)}개 -> WAITING_NAV")

        elif self.armed and detected:
            self._log("Autonomous detection -> TRACKING")
            self.current_focus = None  # autotrack 은 focus 없음
            action['cancel_navigation'] = True
            action['cancel_reason'] = 'target_detected_idle'
            if error_norm is not None:
                action['action'] = 'track'
                action['error'] = error_norm
            self.transition(State.TRACKING)

        else:
            if not self.patrol_complete_sent:
                action['patrol_complete'] = True
                self.patrol_complete_sent = True
                self._log("정찰 완료 - main_queue 비었음")
            if self.armed:
                action['action'] = 'scan_sweep'
                action['scan_sweep'] = True

    # ----- 핸들러: WAITING_NAV -----

    def _on_waiting_nav(self, action: dict, now: float):
        """와플 이동 대기. boundary_queue 에 있으면 처리."""
        # 이미 boundary 처리 중이면 (current_focus 있으면) 그대로 진행
        # 실제로는 AIMING/SCANNING 등 다른 state 에 있어야 하는데
        # WAITING_NAV state 라는 건 boundary 처리 안 하고 그냥 대기 중.

        if self.current_focus is not None:
            return  # 이미 처리 중

        # H5.1: A* waypoint crawl 진행 상황 재평가 (refresh_period_sec 마다).
        # boundary pop 보다 먼저 검사 -- 시야가 이미 확보됐다면 boundary
        # 처리보다 원래 목표(parent) 조준이 우선이다.
        if self.current_parent is not None and self.nav_waypoints:
            nav_cfg = self.cfg.nav_crawl
            period = nav_cfg.refresh_period_sec if nav_cfg else 0.5
            if now - self.nav_last_eval_t >= period:
                self.nav_last_eval_t = now
                if self._on_waypoint_eval(action, now):
                    return

        # boundary_queue 에서 pop 시도
        if self.boundary_queue:
            waffle_xy = self.waffle_pos_fn() if self.waffle_pos_fn else None
            entry, blocked = self._pop_with_los(self.boundary_queue, waffle_xy)
            action['blocked_entries'] = blocked

            if entry is not None:
                self.current_focus = entry
                self.transition(State.AIMING)
                self.aim_start_t = now    # H2.1
                action['action'] = 'aim'
                action['coord_map'] = entry.coord_map
                self._log(f"WAITING_NAV 중 boundary AIMING: "
                          f"{entry.coord_map}")
        # boundary 큐가 잠깐 비어 있어도, Nav2 주행 중에는 OMX 가 멈추지
        # 않고 배경 sweep 을 계속한다.
        if action['action'] == 'wait':
            action['action'] = 'scan_sweep'
            action['scan_sweep'] = True

    # ----- H5.1: A* waypoint crawl 헬퍼 -----

    def _hop_yaw(self, hop_xy, is_final: bool) -> float:
        """구간(hop) 도착 시 향할 yaw. 마지막 구간은 VIEW_POSE 가 계산한
        (OMX 조준 가능성까지 검증된) yaw 를 그대로 쓰고, 중간 구간은 다음
        지점을 바라보게 한다."""
        if is_final and self.nav_final_view_pose is not None:
            return self.nav_final_view_pose[2]
        waffle_xy = self.waffle_pos_fn() if self.waffle_pos_fn else None
        if waffle_xy is None:
            return 0.0
        return math.atan2(hop_xy[1] - waffle_xy[1], hop_xy[0] - waffle_xy[0])

    def _next_hop_pose(self) -> Optional[tuple]:
        if not self.nav_waypoints:
            return None
        hop = self.nav_waypoints[0]
        is_final = len(self.nav_waypoints) == 1
        return (hop[0], hop[1], self._hop_yaw(hop, is_final))

    def _on_waypoint_eval(self, action: dict, now: float) -> bool:
        """WAITING_NAV 동안 refresh_period_sec 마다 호출.

        1) 조기 종료: 이동 도중 이미 목표가 보이면(CHECK_VIEW 통과) 남은
           구간을 취소하고 곧장 AIMING 으로 넘어간다 -- 미리 계산해 둔 먼
           최종 지점까지 굳이 다 가지 않는다.
        2) 구간 도착: 현재 waypoint "분포"(waypoint_tolerance_m) 안에 들어
           오면 다음 구간(이미 계산해 캐싱해 둔 nav_waypoints 의 다음
           원소)으로 즉시 넘어간다. 단, 마지막 구간은 여기서 먼저 AIMING
           으로 넘기지 않고 Nav2 자신의 nav_result(_handle_nav_result) 를
           기다린다 -- 그래야 늦게 도착하는 succeeded 결과가 다음 이동
           시도에 잘못 적용되는 경합을 만들지 않는다.

        아직 도착 전이면 아무 것도 하지 않는다 -- waffle_pos_fn() 은
        AMCL/odom 잡음으로 tick 마다 몇 cm씩 흔들리는데, 여기서 매번
        재계획해서 goal 을 다시 찍으면(구간 지점을 "보정") on_nav_goal
        이 그때마다 진행 중이던 goal 을 취소하고 새로 보내서 Nav2 가 실제
        로 움직일 시간을 못 얻는다(경로는 계속 새로 생기는데 로봇은 안
        가는 증상). 그래서 재계획 없이 이미 캐싱된 다음 지점을 그대로
        쓴다.

        Returns: 이번 tick 에 action 을 채웠으면 True (호출부는 boundary
                 처리 등을 건너뛰고 곧장 return 해야 함).
        """
        nav_cfg = self.cfg.nav_crawl

        # 1) 조기 종료
        early_stop = nav_cfg.early_stop_on_los if nav_cfg else True
        if (
            early_stop
            and self.check_view_fn is not None
            and self.current_parent is not None
            and self.current_parent.target_type != TargetType.PATROL
            and self.check_view_fn(self.current_parent.coord_map)
        ):
            self._log("WAYPOINT_CRAWL: 이동 중 시야 확보 -> 조기 종료")
            if self.nav_cancel_fn is not None:
                self.nav_cancel_fn()
                self.pending_cancel_for_preempt = True
            self.nav_waypoints = []
            self.nav_final_view_pose = None
            self.current_focus = self.current_parent
            self.transition(State.AIMING)
            self.aim_start_t = now
            action['action'] = 'aim'
            action['coord_map'] = self.current_parent.coord_map
            return True

        waffle_xy = self.waffle_pos_fn() if self.waffle_pos_fn else None
        if waffle_xy is None or not self.nav_waypoints:
            return False

        tolerance = nav_cfg.waypoint_tolerance_m if nav_cfg else 0.35
        hop = self.nav_waypoints[0]
        dist = math.hypot(waffle_xy[0] - hop[0], waffle_xy[1] - hop[1])

        # 2) 구간 도착 (마지막 구간 제외 -- 위 docstring 참고)
        if dist <= tolerance:
            if len(self.nav_waypoints) > 1:
                self.nav_waypoints.pop(0)
                next_pose = self._next_hop_pose()
                if next_pose is None:
                    return False
                self._log(
                    f"WAYPOINT_CRAWL: 구간 도착, 다음 hop -> {next_pose[:2]} "
                    f"(남은 {len(self.nav_waypoints)}개)")
                action['action'] = 'nav_goal'
                action['nav_goal_xyyaw'] = next_pose
                return True
            return False

        return False

    # ----- 핸들러: SCANNING / TRACKING / CONFIRMING / COOLDOWN -----

    def _scan_timeout(self) -> float:
        """현재 focus 의 type 별 scan timeout. H3 + H4."""
        if self.cfg.patrol is None or self.current_focus is None:
            return self.cfg.patrol.scan_timeout_sec if self.cfg.patrol else 2.0

        t = self.current_focus.target_type
        if t == TargetType.TARGET:
            return self.cfg.patrol.target_scan_timeout_sec
        elif t == TargetType.BOUNDARY:
            return self.cfg.patrol.boundary_scan_timeout_sec    # H4
        # PATROL
        return self.cfg.patrol.scan_timeout_sec

    def _on_scanning(self, detected, error_norm, now, action):
        if detected:
            self.lost_start_t = 0.0
            if error_norm is not None:
                action['action'] = 'track'
                action['error'] = error_norm
            self.transition(State.TRACKING)
        else:
            action['action'] = 'scan_sweep'
            action['scan_sweep'] = True
            scan_timeout = self._scan_timeout()
            if now - self.scan_start_t >= scan_timeout:
                # H3: TARGET miss 알림
                if (self.current_focus is not None
                        and self.current_focus.target_type == TargetType.TARGET):
                    action['target_not_found_coord'] = (
                        self.current_focus.coord_map)
                    self._log(f"TARGET miss ({scan_timeout:.1f}s scan) "
                              f"-> target_not_found 발행")
                else:
                    self._log(f"SCANNING {scan_timeout:.1f}s 끝, 표적 없음")
                self._on_focus_done()

    def force_track_now(self, reason: str = "detection") -> bool:
        """탐지 즉시 현재 작업을 접고 TRACKING 으로 들어간다."""
        if self.state in (State.FIRING, State.COOLDOWN):
            return False

        self.boundary_queue.clear()
        self.current_parent = None
        self.current_focus = None
        self.nav_pending_result = None
        self.nav_waypoints = []
        self.nav_final_view_pose = None
        self.confirm_progress = 0.0
        self.lost_start_t = 0.0
        self.transition(State.TRACKING)
        self._log(f"{reason}: 즉시 TRACKING 진입")
        return True

    def _on_tracking(self, detected, error_norm, now, action):
        if detected:
            self.lost_start_t = 0.0
            ex, ey = error_norm
            db_x = self.cfg.ibvs.deadband_x
            db_y = self.cfg.ibvs.deadband_y
            if abs(ex) < db_x and abs(ey) < db_y:
                self._log("표적 deadband 진입 -> CONFIRMING")
                self.transition(State.CONFIRMING)
                self.confirm_start_t = now
                self.confirm_progress = 0.0
            else:
                action['action'] = 'track'
                action['error'] = error_norm
        else:
            if self.lost_start_t == 0.0:
                self.lost_start_t = now
                self._log("TRACKING 중 표적 사라짐 (타임아웃 대기)")
            elapsed = now - self.lost_start_t
            timeout = self.cfg.fire.lost_timeout_sec
            if elapsed >= timeout:
                self._log(f"TRACKING 표적 {timeout:.1f}s 잃음")
                if self.current_focus is not None:
                    self.last_processed = self.current_focus.coord_map
                    action['lost_coord_map'] = self.current_focus.coord_map
                action['action'] = 'target_lost'
                self.lost_start_t = 0.0
                self._on_focus_done()

    def _on_confirming(self, detected, error_norm, now, action):
        if not detected:
            self._log("CONFIRMING 중 표적 사라짐 -> TRACKING")
            self.transition(State.TRACKING)
            self.confirm_progress = 0.0
        else:
            ex, ey = error_norm
            scale = self.cfg.fire.confirm_deadband_scale
            confirm_db_x = self.cfg.ibvs.deadband_x * scale
            confirm_db_y = self.cfg.ibvs.deadband_y * scale
            if abs(ex) > confirm_db_x or abs(ey) > confirm_db_y:
                self._log("CONFIRMING 중 이탈 -> TRACKING")
                self.transition(State.TRACKING)
                self.confirm_progress = 0.0
            else:
                elapsed = now - self.confirm_start_t
                self.confirm_progress = min(
                    1.0, elapsed / self.cfg.fire.hold_time_sec)
                if elapsed >= self.cfg.fire.hold_time_sec:
                    self._log(f"조준 {self.cfg.fire.hold_time_sec}s 유지 "
                              f"-> FIRING")
                    self.transition(State.FIRING)
                    self.confirm_progress = 1.0

    def _on_cooldown(self, now, action):
        if now >= self.cooldown_until:
            self._log("Cooldown 끝")
            self.confirm_progress = 0.0
            self.cooldown_home_sent = False
            self._on_focus_done()
        else:
            # 격발 펄스가 끝날 때까지 조준 유지 → 그 후 home.
            # fire_node 의 GPIO HIGH 펄스(fire_duration_sec)와 맞춰서
            # 발사 중에 팔이 빠져나가지 않도록 함.
            fire_pulse = self.cfg.fire.fire_pulse_sec
            pulse_elapsed = now - self.fire_start_t
            if not self.cooldown_home_sent and pulse_elapsed >= fire_pulse:
                action['action'] = 'home'
                self.cooldown_home_sent = True
                self._log(f"격발 펄스 종료 ({pulse_elapsed:.2f}s) -> home")

    # ----- focus 완료 처리 -----

    def _on_focus_done(self):
        """현재 focus 종료. focus 가 parent 면 IDLE, boundary 면 WAITING_NAV 복귀."""
        if self.current_focus is None:
            if self.current_parent is not None:
                heapq.heappush(self.main_queue, self.current_parent)
                self._log(f"중단된 parent 큐 복귀: "
                          f"{self.current_parent.coord_map}")
                self.current_parent = None
                self.nav_waypoints = []
                self.nav_final_view_pose = None
            self.transition(State.IDLE)
            return

        is_boundary = (self.current_focus.target_type == TargetType.BOUNDARY)
        self.current_focus = None
        self.confirm_progress = 0.0

        if is_boundary:
            # boundary 처리 끝. parent 아직 이동 중이면 WAITING_NAV 복귀.
            if self.current_parent is not None:
                self.transition(State.WAITING_NAV)
                self._log("Boundary 처리 끝 -> WAITING_NAV 복귀")
            else:
                # parent 없으면 IDLE (autotrack 케이스 등)
                self.transition(State.IDLE)
        else:
            # main parent 처리 끝
            self.current_parent = None
            self.transition(State.IDLE)
