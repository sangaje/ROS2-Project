"""OMX Auto-Aim 시스템의 공통 타입 정의.

State machine, queue 정책, 시각화 등 여러 모듈에서 공유.
ROS 의존성 없음 (순수 Python).
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field
from enum import Enum, IntEnum
from typing import Optional


# ===========================================================
# State enum
# ===========================================================

class State(Enum):
    """State machine 의 상태.
    
    전이 다이어그램은 INTERFACE_v3.md 의 3.2 참조.
    """
    IDLE = "idle"
    AIMING = "aiming"
    SCANNING = "scanning"
    TRACKING = "tracking"
    CONFIRMING = "confirming"
    FIRING = "firing"
    COOLDOWN = "cooldown"
    WAITING_NAV = "waiting_nav"   # H2: 와플 Nav2 이동 중


# ===========================================================
# Target type
# ===========================================================

class TargetType(IntEnum):
    """좌표 종류.
    
    IntEnum 값 = priority. 낮을수록 우선.
        TARGET   (0):  외부 신뢰 좌표. 즉각 처리.
        BOUNDARY (5):  이동 중 사주 경계. 단명.
        PATROL   (10): 탐색 대상. 미확인.
    """
    TARGET = 0
    BOUNDARY = 5
    PATROL = 10


# ===========================================================
# Line-of-Sight 결과
# ===========================================================

class LOSResult(Enum):
    """LOS (Line of Sight) 검사 결과.
    
    costmap 상 와플→좌표 직선이 장애물에 막혔는지 검사.
    """
    CLEAR = "clear"      # 통과
    BLOCKED = "blocked"  # 장애물
    UNKNOWN = "unknown"  # 미관측 (costmap=-1 또는 경계 밖)


# ===========================================================
# TargetEntry (큐 항목)
# ===========================================================

# 큐 내 entry 의 도착 순서 (FIFO 보조 정렬 키)
_entry_counter = itertools.count()


@dataclass(order=True)
class TargetEntry:
    """큐에 들어가는 좌표 항목.

    heapq 정렬은 sort_key 만 사용.
    sort_key = (priority, distance, count):
        priority 낮은 게 우선 (TARGET=0 최우선)
        같은 priority 면 distance 가까운 게 우선
        같으면 count 작은 게 우선 (먼저 들어온 순서, FIFO)
    """
    sort_key: tuple = field(init=False, default=(0, 0.0, 0))

    priority: int = field(compare=False, default=10)
    count: int = field(compare=False,
                       default_factory=lambda: next(_entry_counter))
    coord_map: tuple = field(compare=False, default=(0.0, 0.0, 0.0))
    target_type: TargetType = field(compare=False,
                                     default=TargetType.PATROL)
    arrival_time: float = field(compare=False, default=0.0)
    distance: float = field(compare=False, default=0.0)
    # H2 신규 (H4 에서 사용). BOUNDARY 가 어느 parent 의 자식인지.
    parent_id: Optional[int] = field(compare=False, default=None)

    def __post_init__(self):
        self._update_sort_key()

    def update_distance(self, waffle_xy):
        """와플 위치 기준 거리 갱신 + sort_key 재계산."""
        if waffle_xy is None:
            self.distance = 0.0
        else:
            dx = self.coord_map[0] - waffle_xy[0]
            dy = self.coord_map[1] - waffle_xy[1]
            self.distance = math.sqrt(dx*dx + dy*dy)
        self._update_sort_key()

    def _update_sort_key(self):
        self.sort_key = (self.priority, self.distance, self.count)

    @property
    def type_name(self):
        return self.target_type.name