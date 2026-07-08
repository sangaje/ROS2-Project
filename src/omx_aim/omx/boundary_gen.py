"""BoundaryGenerator — 와플 이동 중 사주 경계 좌표 자동 생성.

Stage H4. ROS 의존성 없음 (waffle_pose_fn 콜백 + BoundaryConfig 만).

Sweep 동작:
    fan_half_angle=45, angle_step=22.5 → [-45, -22.5, 0, 22.5, 45]
    매 period_sec 마다 sweep 한 칸씩 진행하며 BOUNDARY 1개 생성.

좌표 계산 (map frame):
    absolute_angle = reference_yaw + sweep_offset
    reference_yaw 는 기본적으로 waffle.yaw 이며, 호출자가 patrol 방향처럼
    별도 기준을 넘기면 그 방향을 중심으로 fan sweep 한다.
    x = waffle.x + distance_m * cos(absolute)
    y = waffle.y + distance_m * sin(absolute)
    z = boundary.z

parent_type 별 enable 분리:
    - PATROL 처리 중에만 (기본): 탐색 의미
    - TARGET 처리 중에는 (기본 off): TARGET 으로 빠르게
    - /omx/boundary_enable 토픽으로 런타임 토글 가능
"""

from __future__ import annotations

import math
from typing import Callable, Optional

from omx.types import TargetType


class BoundaryGenerator:
    """sweep 방식 BOUNDARY 좌표 생성기."""

    def __init__(self, cfg, waffle_pose_fn: Callable, logger=None):
        """
        Args:
            cfg: BoundaryConfig
            waffle_pose_fn: () -> (x, y, yaw) or None (map frame 와플 pose)
            logger: ROS logger (optional)
        """
        self.cfg = cfg
        self.waffle_pose_fn = waffle_pose_fn
        self.logger = logger

        # parent type 별 토글 (런타임 변경 가능)
        self.enabled_target = cfg.enable_during_target
        self.enabled_patrol = cfg.enable_during_patrol

        # sweep 각도 리스트 (한 번 계산 후 고정)
        half = cfg.fan_half_angle_deg
        step = cfg.angle_step_deg
        self.sweep_angles_deg = []
        a = -half
        while a <= half + 0.01:
            self.sweep_angles_deg.append(a)
            a += step
        self.sweep_idx = 0
        self.last_gen_t = 0.0

    def _log(self, msg):
        if self.logger:
            self.logger.info(msg)

    def set_enabled(self, target: Optional[bool] = None,
                    patrol: Optional[bool] = None):
        """런타임 토글."""
        if target is not None:
            self.enabled_target = target
        if patrol is not None:
            self.enabled_patrol = patrol

    def is_enabled_for(self, parent_type) -> bool:
        """parent (TARGET/PATROL) 에 대해 enabled 여부."""
        if parent_type == TargetType.TARGET:
            return self.enabled_target
        elif parent_type == TargetType.PATROL:
            return self.enabled_patrol
        return False

    def maybe_generate(self, now: float, parent_type,
                       reference_yaw: Optional[float] = None) -> Optional[tuple]:
        """주기 시간 됐고 enabled 면 다음 BOUNDARY 좌표 반환.

        Returns:
            (x, y, z) in map frame, 또는 None.
        """
        if not self.is_enabled_for(parent_type):
            return None
        if now - self.last_gen_t < self.cfg.period_sec:
            return None

        pose = self.waffle_pose_fn()
        if pose is None:
            return None
        wx, wy, wyaw = pose
        base_yaw = wyaw if reference_yaw is None else reference_yaw

        # sweep 한 칸 진행
        offset_deg = self.sweep_angles_deg[self.sweep_idx]
        self.sweep_idx = (self.sweep_idx + 1) % len(self.sweep_angles_deg)

        absolute_angle = base_yaw + math.radians(offset_deg)
        d = self.cfg.distance_m
        bx = wx + d * math.cos(absolute_angle)
        by = wy + d * math.sin(absolute_angle)
        bz = self.cfg.z

        self.last_gen_t = now
        return (bx, by, bz)
