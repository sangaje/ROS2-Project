"""StateBus — 메인 loop ↔ Flask 스레드 사이 통신 채널.

설계:
    - 프레임: lock-free ref swap (CPython 에서 atomic)
    - 상태 dict: 락으로 보호 (작은 dict 복사 비용 무시)
    - seq 번호로 클라이언트가 stale skip 가능

메인 loop 은 update_frame / update_state 만 호출.
Flask 스레드는 get_frame / get_state 로 가져감.

성능:
    update_frame()  ~ μs 단위 (단순 ref 대입)
    update_state()  ~ μs 단위 (락 + dict 대입)
    → 메인 loop tick 시간에 영향 없음
"""

from __future__ import annotations

import threading
from typing import Any, Dict, Optional, Tuple

import numpy as np


class StateBus:
    def __init__(self):
        # 프레임: lock-free ref swap
        self._frame: Optional[np.ndarray] = None
        self._frame_seq: int = 0

        # 상태: 락으로 보호
        self._state: Dict[str, Any] = {}
        self._state_seq: int = 0
        self._lock = threading.Lock()

    # ----- writers (main loop) -----

    def update_frame(self, frame: np.ndarray) -> None:
        """프레임 ref 교체. 락 없음."""
        self._frame = frame
        self._frame_seq += 1   # CPython int 증가도 atomic

    def update_state(self, snapshot: Dict[str, Any]) -> None:
        """상태 스냅샷 교체. dict 참조 대입만."""
        with self._lock:
            self._state = snapshot
            self._state_seq += 1

    # ----- readers (Flask threads) -----

    def get_frame(self) -> Tuple[Optional[np.ndarray], int]:
        return self._frame, self._frame_seq

    def get_state(self) -> Tuple[Dict[str, Any], int]:
        with self._lock:
            # shallow copy 로 충분: writer 가 매번 새 dict 통째 대입하기 때문
            return self._state, self._state_seq