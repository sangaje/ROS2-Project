
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RiskStats:
    risk: float
    front_min_dist: float
    front_density: float
    valid_front_count: int


def _finite_ranges(ranges: np.ndarray) -> np.ndarray:
    arr = np.asarray(ranges, dtype=np.float32)
    return arr[np.isfinite(arr)]


def _sector_ranges(
    ranges: np.ndarray,
    angle_min: float,
    angle_increment: float,
    half_angle_deg: float,
) -> np.ndarray:
    ranges = np.asarray(ranges, dtype=np.float32)
    n = ranges.shape[0]
    angles = angle_min + np.arange(n, dtype=np.float32) * angle_increment
    half_angle = np.deg2rad(half_angle_deg)
    mask = np.abs(angles) <= half_angle
    return _finite_ranges(ranges[mask])


def compute_front_distance_risk(
    ranges: np.ndarray,
    angle_min: float,
    angle_increment: float,
    *,
    front_angle_deg: float = 35.0,
    d_safe: float = 1.20,
    d_min: float = 0.15,
) -> tuple[float, float, int]:
    """
    전방 sector 안의 최소 거리 기반 위험도.

    반환:
        (risk, front_min_dist, valid_front_count)

    risk:
        0.0 = 충분히 안전한 거리
        1.0 = 충돌 직전/측정 불능
    """
    front = _sector_ranges(
        ranges=ranges,
        angle_min=angle_min,
        angle_increment=angle_increment,
        half_angle_deg=front_angle_deg,
    )

    if front.size == 0:
        return 1.0, float("inf"), 0

    d_front = float(np.min(front))

    if d_safe <= d_min:
        raise ValueError("d_safe must be larger than d_min")

    risk = (d_safe - d_front) / (d_safe - d_min)
    risk = float(np.clip(risk, 0.0, 1.0))
    return risk, d_front, int(front.size)


def compute_obstacle_density_risk(
    ranges: np.ndarray,
    angle_min: float,
    angle_increment: float,
    *,
    front_angle_deg: float = 60.0,
    d_safe: float = 1.20,
) -> float:
    """
    전방 넓은 sector 안에서 d_safe보다 가까운 ray 비율.
    얇은 장애물 하나에 과도하게 반응하는 것을 줄이기 위한 보조 위험도.
    """
    front = _sector_ranges(
        ranges=ranges,
        angle_min=angle_min,
        angle_increment=angle_increment,
        half_angle_deg=front_angle_deg,
    )

    if front.size == 0:
        return 1.0

    density = float(np.mean(front < d_safe))
    return float(np.clip(density, 0.0, 1.0))


def compute_cqb_scalar_risk(
    ranges: np.ndarray,
    angle_min: float,
    angle_increment: float,
    *,
    collision: bool = False,
    front_angle_deg: float = 35.0,
    density_angle_deg: float = 60.0,
    d_safe: float = 1.20,
    d_min: float = 0.15,
    w_distance: float = 0.70,
    w_density: float = 0.25,
    w_collision: float = 0.05,
) -> RiskStats:
    """
    CQB scene risk scalar pseudo-label.

    입력은 LiDAR scan이고, 출력 risk는 카메라 이미지 한 장의 라벨로 사용한다.
    카메라 이미지는 모델 입력이고, LiDAR는 학습 라벨 생성용 teacher signal이다.

    현재 정의:
        risk = 0.70 * 전방 최소거리 위험도
             + 0.25 * 전방 장애물 밀도
             + 0.05 * 충돌 플래그

    추후 unknown map ratio, doorway/corner score를 추가할 수 있다.
    """
    r_dist, front_min_dist, valid_front_count = compute_front_distance_risk(
        ranges=ranges,
        angle_min=angle_min,
        angle_increment=angle_increment,
        front_angle_deg=front_angle_deg,
        d_safe=d_safe,
        d_min=d_min,
    )

    r_density = compute_obstacle_density_risk(
        ranges=ranges,
        angle_min=angle_min,
        angle_increment=angle_increment,
        front_angle_deg=density_angle_deg,
        d_safe=d_safe,
    )

    r_collision = 1.0 if collision else 0.0

    risk = (
        w_distance * r_dist
        + w_density * r_density
        + w_collision * r_collision
    )
    risk = float(np.clip(risk, 0.0, 1.0))

    return RiskStats(
        risk=risk,
        front_min_dist=front_min_dist,
        front_density=r_density,
        valid_front_count=valid_front_count,
    )
