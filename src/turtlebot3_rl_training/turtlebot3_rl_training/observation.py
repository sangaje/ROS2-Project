import math
from typing import Sequence

import numpy as np


def normalize_angle(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def downsample_lidar(
    scan_ranges: Sequence[float],
    num_bins: int = 360,
    min_range: float = 0.12,
    max_range: float = 3.5,
) -> np.ndarray:
    ranges = np.asarray(scan_ranges, dtype=np.float32)

    if ranges.size == 0:
        return np.ones(num_bins, dtype=np.float32)

    ranges = np.nan_to_num(
        ranges,
        nan=max_range,
        posinf=max_range,
        neginf=min_range,
    )

    ranges = np.clip(ranges, min_range, max_range)

    if ranges.size < num_bins:
        x_old = np.linspace(0.0, 1.0, ranges.size)
        x_new = np.linspace(0.0, 1.0, num_bins)
        ranges = np.interp(x_new, x_old, ranges).astype(np.float32)

    split_ranges = np.array_split(ranges, num_bins)
    pooled = np.array([np.min(chunk) for chunk in split_ranges], dtype=np.float32)

    normalized = (pooled - min_range) / (max_range - min_range)
    normalized = np.clip(normalized, 0.0, 1.0)

    return normalized.astype(np.float32)


def build_observation(
    scan_ranges: Sequence[float],
    robot_xy: np.ndarray,
    robot_yaw: float,
    goal_xy: np.ndarray,
    prev_action: np.ndarray,
    num_lidar_bins: int = 360,
    min_lidar_range: float = 0.12,
    max_lidar_range: float = 3.5,
    max_goal_distance: float = 5.0,
    max_linear_speed: float = 0.22,
    max_angular_speed: float = 1.5,
) -> np.ndarray:
    """
    기존 navigation observation.

    [
      lidar_0 ... lidar_359,
      goal_distance_norm,
      goal_angle_norm,
      prev_linear_norm,
      prev_angular_norm,
    ]

    총 차원: 360 + 4 = 364
    """
    lidar = downsample_lidar(
        scan_ranges=scan_ranges,
        num_bins=num_lidar_bins,
        min_range=min_lidar_range,
        max_range=max_lidar_range,
    )

    dx = float(goal_xy[0] - robot_xy[0])
    dy = float(goal_xy[1] - robot_xy[1])

    goal_distance = math.sqrt(dx * dx + dy * dy)
    goal_angle_world = math.atan2(dy, dx)
    goal_angle_robot = normalize_angle(goal_angle_world - robot_yaw)

    goal_distance_norm = np.clip(goal_distance / max_goal_distance, 0.0, 1.0)
    goal_angle_norm = np.clip(goal_angle_robot / math.pi, -1.0, 1.0)

    prev_linear_norm = np.clip(
        float(prev_action[0]) / max(max_linear_speed, 1e-6),
        0.0,
        1.0,
    )
    prev_angular_norm = np.clip(
        float(prev_action[1]) / max(max_angular_speed, 1e-6),
        -1.0,
        1.0,
    )

    obs = np.concatenate(
        [
            lidar,
            np.array(
                [
                    goal_distance_norm,
                    goal_angle_norm,
                    prev_linear_norm,
                    prev_angular_norm,
                ],
                dtype=np.float32,
            ),
        ]
    )

    return obs.astype(np.float32)


def build_exploration_observation(
    scan_ranges: Sequence[float],
    coverage_ratio: float,
    coverage_delta: float,
    frontier_distance: float,
    frontier_angle: float,
    target_priority: float,
    mean_confidence: float,
    stale_ratio: float,
    low_confidence_ratio: float,
    prev_action: np.ndarray,
    num_lidar_bins: int = 360,
    min_lidar_range: float = 0.12,
    max_lidar_range: float = 3.5,
    max_frontier_distance: float = 5.0,
    max_linear_speed: float = 0.22,
    max_angular_speed: float = 1.5,
) -> np.ndarray:
    """
    SLAM 보조형 exploration observation.

    [
      lidar_0 ... lidar_359,
      coverage_ratio,
      coverage_delta_norm,
      frontier_distance_norm,
      frontier_angle_norm,
      target_priority,
      mean_confidence_norm,
      stale_ratio,
      low_confidence_ratio,
      prev_linear_norm,
      prev_angular_norm,
    ]

    총 차원: 360 + 10 = 370

    의미:
      - SLAM /map은 기하학적 기준을 담당한다.
      - 이 observation은 RL memory map의 탐색 가치만 요약한다.
      - stale_ratio와 low_confidence_ratio가 커지면 재확인 행동을 학습할 수 있다.
    """
    lidar = downsample_lidar(
        scan_ranges=scan_ranges,
        num_bins=num_lidar_bins,
        min_range=min_lidar_range,
        max_range=max_lidar_range,
    )

    coverage_ratio_norm = np.clip(float(coverage_ratio), 0.0, 1.0)

    # 한 step에서 coverage_delta는 매우 작으므로 확대해서 관측에 넣는다.
    coverage_delta_norm = np.clip(float(coverage_delta) * 100.0, -1.0, 1.0)

    frontier_distance_norm = np.clip(
        float(frontier_distance) / max(max_frontier_distance, 1e-6),
        0.0,
        1.0,
    )

    frontier_angle_norm = np.clip(
        float(frontier_angle) / math.pi,
        -1.0,
        1.0,
    )

    target_priority_norm = np.clip(float(target_priority), 0.0, 1.0)
    mean_confidence_norm = np.clip(float(mean_confidence) / 100.0, 0.0, 1.0)
    stale_ratio_norm = np.clip(float(stale_ratio), 0.0, 1.0)
    low_confidence_ratio_norm = np.clip(float(low_confidence_ratio), 0.0, 1.0)

    prev_linear_norm = np.clip(
        float(prev_action[0]) / max(max_linear_speed, 1e-6),
        0.0,
        1.0,
    )

    prev_angular_norm = np.clip(
        float(prev_action[1]) / max(max_angular_speed, 1e-6),
        -1.0,
        1.0,
    )

    obs = np.concatenate(
        [
            lidar,
            np.array(
                [
                    coverage_ratio_norm,
                    coverage_delta_norm,
                    frontier_distance_norm,
                    frontier_angle_norm,
                    target_priority_norm,
                    mean_confidence_norm,
                    stale_ratio_norm,
                    low_confidence_ratio_norm,
                    prev_linear_norm,
                    prev_angular_norm,
                ],
                dtype=np.float32,
            ),
        ]
    )

    return obs.astype(np.float32)
