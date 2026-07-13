"""Fleet registry parsing shared by launch-time and runtime orchestration."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import yaml


@dataclass(frozen=True)
class FieldRobotSpec:
    robot_name: str
    domain_id: int
    initial_role: str = 'STANDBY'
    map_capable: bool = True
    map_authority: bool = False
    camera_capable: bool = True


def _bool(value: Any, default: bool = True) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    text = str(value).strip().lower()
    if text in ('1', 'true', 'yes', 'on'):
        return True
    if text in ('0', 'false', 'no', 'off'):
        return False
    return default


def _robot_from_mapping(data: dict[str, Any]) -> FieldRobotSpec | None:
    name = str(data.get('robot_name', '')).strip()
    if not name:
        return None
    try:
        domain = int(data.get('domain_id'))
    except (TypeError, ValueError):
        return None
    initial_role = str(data.get('initial_role', 'STANDBY')).strip().upper() or 'STANDBY'
    default_authority = initial_role in ('ACTIVE_SCOUT', 'SCOUT', 'RECOVERING')
    return FieldRobotSpec(
        robot_name=name,
        domain_id=domain,
        initial_role=initial_role,
        map_capable=_bool(data.get('map_capable'), True),
        map_authority=_bool(data.get('map_authority'), default_authority),
        camera_capable=_bool(data.get('camera_capable'), True),
    )


def normalize_registry(data: Any) -> list[FieldRobotSpec]:
    if isinstance(data, str):
        text = data.strip()
        if not text:
            return []
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            data = yaml.safe_load(text)
    if not isinstance(data, dict):
        return []
    robots = data.get('field_robots', [])
    if not isinstance(robots, list):
        return []
    out: list[FieldRobotSpec] = []
    seen: set[str] = set()
    for item in robots:
        if not isinstance(item, dict):
            continue
        spec = _robot_from_mapping(item)
        if spec is None or spec.robot_name in seen:
            continue
        seen.add(spec.robot_name)
        out.append(spec)
    return out


def load_registry_file(path: str | Path) -> list[FieldRobotSpec]:
    registry_path = Path(path).expanduser()
    if not registry_path.exists():
        raise FileNotFoundError(f'fleet_registry_file does not exist: {registry_path}')
    return normalize_registry(registry_path.read_text(encoding='utf-8'))


def build_legacy_registry(
    *,
    active_scout_robot_name: str,
    risk_domain_id: str,
    follower_robot_name: str,
    follower_domain_id: str,
) -> list[FieldRobotSpec]:
    robots: list[FieldRobotSpec] = []
    if str(risk_domain_id).strip():
        robots.append(FieldRobotSpec(
            robot_name=str(active_scout_robot_name).strip() or 'scout22',
            domain_id=int(str(risk_domain_id).strip()),
            initial_role='ACTIVE_SCOUT',
            map_authority=True,
        ))
    if str(follower_domain_id).strip():
        name = str(follower_robot_name).strip() or 'follower21'
        domain = int(str(follower_domain_id).strip())
        if all(robot.robot_name != name for robot in robots):
            robots.append(FieldRobotSpec(
                robot_name=name,
                domain_id=domain,
                initial_role='FOLLOWER',
                map_authority=False,
            ))
    return robots


def registry_to_json(registry: Iterable[FieldRobotSpec]) -> str:
    return json.dumps(
        {
            'field_robots': [
                {
                    'robot_name': robot.robot_name,
                    'domain_id': robot.domain_id,
                    'initial_role': robot.initial_role,
                    'map_capable': robot.map_capable,
                    'map_authority': robot.map_authority,
                    'camera_capable': robot.camera_capable,
                }
                for robot in registry
            ]
        },
        sort_keys=True,
    )
