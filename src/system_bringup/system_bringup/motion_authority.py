"""Single source of truth for non-safety field-robot command ownership."""

from __future__ import annotations

from enum import Enum


class MotionAuthority(str, Enum):
    NONE = 'NONE'
    LOCALIZATION_SPIN = 'LOCALIZATION_SPIN'
    FAILOVER_RECOVERY_NAV = 'FAILOVER_RECOVERY_NAV'
    ACTIVE_SCOUT_RL = 'ACTIVE_SCOUT_RL'
    NORMAL_FOLLOW = 'NORMAL_FOLLOW'


NAV_AUTHORITIES = frozenset({
    MotionAuthority.NORMAL_FOLLOW,
    MotionAuthority.FAILOVER_RECOVERY_NAV,
})


def nav_motion_is_quiescent(active_goal_count: int, cancel_requests: int) -> bool:
    return active_goal_count == 0 and cancel_requests == 0


def authority_allows_nonzero(current: MotionAuthority, source: MotionAuthority) -> bool:
    return current == source
