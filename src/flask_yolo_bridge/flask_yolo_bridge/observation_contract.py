"""Field-robot camera observation metadata helpers.

The authoritative observation key is ``(robot_id, boot_id, sequence)``.
This module is intentionally ROS-free so the race-condition rules can be
unit tested without launching nodes.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
from typing import Iterable, Mapping, Optional


@dataclass(frozen=True)
class PoseSample:
    stamp_sec: float
    x: float
    y: float
    yaw: float


def make_boot_id() -> str:
    return uuid.uuid4().hex


def closest_pose_sample(
    samples: Iterable[PoseSample],
    stamp_sec: float,
    max_error_sec: float,
) -> tuple[Optional[PoseSample], float]:
    """Return the nearest pose sample and absolute time error in seconds."""
    if stamp_sec <= 0.0:
        return None, float('inf')
    best = None
    best_error = float('inf')
    for sample in samples:
        error = abs(float(sample.stamp_sec) - float(stamp_sec))
        if error < best_error:
            best = sample
            best_error = error
    if best is None or best_error > max(0.0, float(max_error_sec)):
        return None, best_error
    return best, best_error


def parse_role_payload(raw: str, fallback_role: str) -> tuple[str, int]:
    text = str(raw or '').strip()
    if not text:
        return fallback_role, 0
    if not text.startswith('{'):
        return text.upper(), 0
    try:
        import json
        payload = json.loads(text)
    except Exception:
        return fallback_role, 0
    if not isinstance(payload, dict):
        return fallback_role, 0
    role = str(
        payload.get('role', payload.get('status', fallback_role))
    ).strip().upper() or fallback_role
    try:
        epoch = int(payload.get('epoch', payload.get('role_epoch', 0)))
    except (TypeError, ValueError):
        epoch = 0
    return role, max(0, epoch)


def build_observation_metadata(
    *,
    robot_id: str,
    boot_id: str,
    sequence: int,
    role: str,
    role_epoch: int,
    frame_id: str,
    camera_hfov_deg: float,
    capture_ros_sec: float,
    capture_wall_sec: float,
    capture_mono_sec: float,
    pose: PoseSample,
    pose_time_error_sec: float,
    send_start_mono_sec: Optional[float] = None,
    image_width: int = 0,
    image_height: int = 0,
    calibration_id: str = '',
) -> dict[str, str]:
    now_mono = time.monotonic() if send_start_mono_sec is None else send_start_mono_sec
    capture_to_send_ms = max(0.0, (now_mono - float(capture_mono_sec)) * 1000.0)
    return {
        'robot_id': str(robot_id),
        'boot_id': str(boot_id),
        'observation_id': str(int(sequence)),
        'sequence': str(int(sequence)),
        'role': str(role).upper(),
        'role_epoch': str(int(role_epoch)),
        'camera_frame_id': str(frame_id),
        'camera_hfov_deg': f'{float(camera_hfov_deg):.6f}',
        'capture_ros_sec': f'{float(capture_ros_sec):.9f}',
        'capture_wall_sec': f'{float(capture_wall_sec):.9f}',
        'capture_pose_x': f'{float(pose.x):.6f}',
        'capture_pose_y': f'{float(pose.y):.6f}',
        'capture_pose_yaw': f'{float(pose.yaw):.9f}',
        'capture_pose_stamp_sec': f'{float(pose.stamp_sec):.9f}',
        'pose_time_error_ms': f'{float(pose_time_error_sec) * 1000.0:.3f}',
        'capture_to_send_delay_ms': f'{capture_to_send_ms:.3f}',
        'image_width': str(int(image_width)),
        'image_height': str(int(image_height)),
        'camera_calibration_id': str(calibration_id),
    }


def echo_observation_metadata(form: Mapping[str, object]) -> dict:
    """Copy recognized observation metadata from HTTP form into JSON payload."""
    keys = (
        'robot_id',
        'boot_id',
        'observation_id',
        'sequence',
        'role',
        'role_epoch',
        'camera_frame_id',
        'camera_hfov_deg',
        'capture_ros_sec',
        'capture_wall_sec',
        'capture_pose_x',
        'capture_pose_y',
        'capture_pose_yaw',
        'capture_pose_stamp_sec',
        'pose_time_error_ms',
        'capture_to_send_delay_ms',
        'camera_calibration_id',
    )
    out = {}
    for key in keys:
        value = form.get(key)
        if value is not None:
            out[key] = str(value)
    return out
