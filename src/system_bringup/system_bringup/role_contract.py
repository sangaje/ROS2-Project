"""Shared role and epoch parsing for field-robot orchestration.

The wire format intentionally stays JSON/String compatible with the existing
fleet topics.  This module has no ROS dependencies so role handling is easy
to unit test in isolation.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping, Optional


class Role(str, Enum):
    IDLE = 'IDLE'
    FOLLOWER = 'FOLLOWER'
    ACTIVE_SCOUT = 'ACTIVE_SCOUT'
    RECOVERY_NAVIGATING = 'RECOVERY_NAVIGATING'
    ARRIVED_AT_FAILURE_POSE = 'ARRIVED_AT_FAILURE_POSE'
    LOCALIZATION_CHECK = 'LOCALIZATION_CHECK'
    LOCALIZATION_SPIN = 'LOCALIZATION_SPIN'
    LOCALIZATION_SETTLE = 'LOCALIZATION_SETTLE'
    FAILED = 'FAILED'


ROLE_ALIASES = {
    'FOLLOWING': Role.FOLLOWER.value,
    'SCOUT': Role.ACTIVE_SCOUT.value,
    'ACTIVE_SCOUT_EXPLORING': Role.ACTIVE_SCOUT.value,
    'RECOVERY': Role.RECOVERY_NAVIGATING.value,
}


@dataclass(frozen=True)
class RoleMessage:
    role: Role
    robot: str
    epoch: Optional[int]
    active_scout_id: Optional[str]
    localization_ready: Optional[bool]
    recovery_complete: Optional[bool]
    payload: Mapping[str, Any]


def parse_epoch(value: Any) -> Optional[int]:
    """Parse a non-negative integer ownership epoch without coercion."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if value >= 0 else None
    if isinstance(value, str) and value.strip().isdigit():
        return int(value.strip())
    return None


def optional_bool(value: Any) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in ('true', '1', 'yes', 'y'):
            return True
        if normalized in ('false', '0', 'no', 'n'):
            return False
    return None


def normalize_role(raw: Any) -> Role:
    key = str(raw or '').strip().upper()
    key = ROLE_ALIASES.get(key, key)
    return Role.__members__.get(key, Role.IDLE)


def parse_role_message(raw: str, default_robot: str) -> Optional[RoleMessage]:
    """Parse the existing JSON role topic plus legacy simple-role strings."""
    text = str(raw or '').strip()
    if not text:
        return None
    if not text.startswith('{'):
        return RoleMessage(
            role=normalize_role(text), robot=default_robot, epoch=None,
            active_scout_id=None, localization_ready=None,
            recovery_complete=None, payload={},
        )
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    robot = str(payload.get('robot', default_robot)).strip() or default_robot
    return RoleMessage(
        role=normalize_role(payload.get('role', payload.get('command', payload.get('status', 'IDLE')))),
        robot=robot,
        epoch=parse_epoch(payload.get('epoch')),
        active_scout_id=(
            str(payload['active_scout_id']).strip()
            if payload.get('active_scout_id') is not None else None
        ),
        localization_ready=optional_bool(payload.get('localization_ready')),
        recovery_complete=optional_bool(payload.get('recovery_complete')),
        payload=payload,
    )
