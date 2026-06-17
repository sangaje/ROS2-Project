import math
import os
from typing import Optional, Sequence

import numpy as np


def normalize_angle(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def _env_int(name: str, default: int, min_value: int = 1) -> int:
    try:
        value = int(os.environ.get(name, str(default)))
    except Exception:
        value = default
    return max(min_value, value)


def _env_float(name: str, default: float, min_value: float = 0.0) -> float:
    try:
        value = float(os.environ.get(name, str(default)))
    except Exception:
        value = default
    return max(min_value, value)


def _env_bool(name: str, default: bool = True) -> bool:
    raw = os.environ.get(name, "1" if default else "0")
    return str(raw).strip().lower() not in {"0", "false", "no", "off", "disable", "disabled"}


def _make_odd_kernel(value: int, *, min_value: int = 1, max_value: int = 31) -> int:
    value = int(max(min_value, min(max_value, value)))
    if value % 2 == 0:
        value += 1
    return int(max(min_value, min(max_value, value)))


def _circular_median_filter(values: np.ndarray, kernel_size: int) -> np.ndarray:
    """Small circular median filter for LaserScan angular speckle noise."""
    kernel_size = _make_odd_kernel(kernel_size, min_value=1)
    if kernel_size <= 1 or values.size <= 2:
        return values.astype(np.float32, copy=False)

    half = kernel_size // 2
    stacked = np.stack(
        [np.roll(values, shift) for shift in range(-half, half + 1)],
        axis=0,
    )
    return np.median(stacked, axis=0).astype(np.float32)


def _circular_lowpass_filter(values: np.ndarray, kernel_size: int) -> np.ndarray:
    """Gaussian-like circular low-pass filter over angular scan order."""
    kernel_size = _make_odd_kernel(kernel_size, min_value=1)
    if kernel_size <= 1 or values.size <= 2:
        return values.astype(np.float32, copy=False)

    half = kernel_size // 2
    x = np.linspace(-2.0, 2.0, kernel_size, dtype=np.float32)
    weights = np.exp(-0.5 * x * x).astype(np.float32)
    weights /= float(np.sum(weights))

    filtered = np.zeros_like(values, dtype=np.float32)
    for weight, shift in zip(weights, range(-half, half + 1)):
        filtered += float(weight) * np.roll(values, shift)
    return filtered.astype(np.float32)


def _valid_scan_geometry(
    n: int,
    angle_min: Optional[float],
    angle_increment: Optional[float],
) -> bool:
    if n <= 0:
        return False
    if angle_min is None or angle_increment is None:
        return False
    if not np.isfinite(float(angle_min)):
        return False
    if not np.isfinite(float(angle_increment)) or abs(float(angle_increment)) < 1e-12:
        return False
    return True


def _legacy_index_pool(
    robust_ranges: np.ndarray,
    smooth_ranges: np.ndarray,
    num_bins: int,
    min_range: float,
    max_range: float,
    obstacle_margin: float,
) -> np.ndarray:
    """Old behavior: index-space split/min-pooling fallback."""
    if robust_ranges.size < num_bins:
        # Interpolation fallback. Use endpoint=False to avoid duplicating the circular seam.
        x_old = np.linspace(0.0, 1.0, robust_ranges.size, endpoint=False)
        x_new = np.linspace(0.0, 1.0, num_bins, endpoint=False)
        robust_ranges = np.interp(x_new, x_old, robust_ranges, period=1.0).astype(np.float32)
        smooth_ranges = np.interp(x_new, x_old, smooth_ranges, period=1.0).astype(np.float32)

    if robust_ranges.size == num_bins:
        pooled = np.minimum(smooth_ranges, robust_ranges + obstacle_margin)
    else:
        raw_chunks = np.array_split(robust_ranges, num_bins)
        smooth_chunks = np.array_split(smooth_ranges, num_bins)
        min_pool = np.array([np.min(chunk) for chunk in raw_chunks], dtype=np.float32)
        smooth_pool = np.array([np.mean(chunk) for chunk in smooth_chunks], dtype=np.float32)
        pooled = np.minimum(smooth_pool, min_pool + obstacle_margin)

    return np.clip(pooled, min_range, max_range).astype(np.float32)


def _uniform_angle_resample(
    robust_ranges: np.ndarray,
    smooth_ranges: np.ndarray,
    num_bins: int,
    min_range: float,
    max_range: float,
    obstacle_margin: float,
    angle_min: float,
    angle_increment: float,
    angle_max: Optional[float] = None,
) -> np.ndarray:
    """Resample LaserScan to uniformly spaced angular bins.

    Output order intentionally follows LaserScan angle_min. For TurtleBot3 LDS,
    angle_min is typically the first ray in the raw /scan ordering, so this keeps
    the policy's LiDAR index convention stable while making decimation angularly
    uniform.

    Per target bin:
      - use all source rays whose angle falls inside that bin,
      - min-pool robust ranges for obstacle preservation,
      - mean-pool low-pass ranges for anti-alias smoothing,
      - clamp smoothed result so close obstacles cannot be hidden by smoothing.
    """
    n = int(robust_ranges.size)
    if n <= 0 or num_bins <= 0:
        return np.ones(num_bins, dtype=np.float32) * max_range

    angle_min = float(angle_min)
    angle_increment = float(angle_increment)

    # Prefer the actual scan angular extent. For standard full 360-degree scans,
    # n * angle_increment is approximately 2*pi.
    span = abs(float(angle_increment)) * float(n)
    if angle_max is not None and np.isfinite(float(angle_max)):
        meta_span = abs(float(angle_max) - float(angle_min)) + abs(float(angle_increment))
        if np.isfinite(meta_span) and meta_span > 1e-6:
            span = meta_span

    if not np.isfinite(span) or span <= 1e-6:
        return _legacy_index_pool(
            robust_ranges,
            smooth_ranges,
            num_bins,
            min_range,
            max_range,
            obstacle_margin,
        )

    # Cap full-circle scans to exactly 2*pi so the seam does not drift.
    if abs(span - 2.0 * math.pi) < math.radians(5.0):
        span = 2.0 * math.pi

    source_angles = angle_min + np.arange(n, dtype=np.float32) * angle_increment
    target_centers = angle_min + (np.arange(num_bins, dtype=np.float32) + 0.5) * (span / float(num_bins))
    half_width = 0.5 * (span / float(num_bins))

    pooled = np.empty(num_bins, dtype=np.float32)
    full_circle = abs(span - 2.0 * math.pi) < math.radians(5.0)

    for j, center in enumerate(target_centers):
        if full_circle:
            diff = np.arctan2(np.sin(source_angles - center), np.cos(source_angles - center))
            mask = np.abs(diff) <= half_width + 1e-7
        else:
            lo = center - half_width
            hi = center + half_width
            mask = (source_angles >= lo - 1e-7) & (source_angles < hi + 1e-7)

        if not np.any(mask):
            # Rare geometry mismatch fallback: nearest angular ray.
            if full_circle:
                diff = np.arctan2(np.sin(source_angles - center), np.cos(source_angles - center))
            else:
                diff = source_angles - center
            nearest = int(np.argmin(np.abs(diff)))
            raw_min = float(robust_ranges[nearest])
            smooth_mean = float(smooth_ranges[nearest])
        else:
            raw_min = float(np.min(robust_ranges[mask]))
            smooth_mean = float(np.mean(smooth_ranges[mask]))

        pooled[j] = min(smooth_mean, raw_min + obstacle_margin)

    return np.clip(pooled, min_range, max_range).astype(np.float32)


def downsample_lidar(
    scan_ranges: Sequence[float],
    num_bins: int = 360,
    min_range: float = 0.12,
    max_range: float = 3.5,
    scan_angle_min: Optional[float] = None,
    scan_angle_increment: Optional[float] = None,
    scan_angle_max: Optional[float] = None,
) -> np.ndarray:
    """Return a 0~1 normalized LiDAR vector with anti-alias filtering.

    Old behavior was index-space array_split/min-pooling, which assumes the raw
    LaserScan already has exactly the same angular sampling as the NN bins.
    This version uses LaserScan angle_min/angle_increment when available and
    resamples to uniform angular bins. It falls back to the old conservative
    index-space pool if geometry metadata is missing.

    Runtime knobs:
      TB3_RL_LIDAR_UNIFORM_ANGLE_RESAMPLE default 1, set 0 to fallback
      TB3_RL_LIDAR_MEDIAN_KERNEL          default 3, set 1 to disable
      TB3_RL_LIDAR_LOWPASS_KERNEL         default 5, set 1 to disable
      TB3_RL_LIDAR_OBSTACLE_MARGIN_M      default 0.08
    """
    num_bins = max(int(num_bins), 1)
    ranges = np.asarray(scan_ranges, dtype=np.float32)

    if ranges.size == 0:
        return np.ones(num_bins, dtype=np.float32)

    ranges = np.nan_to_num(
        ranges,
        nan=max_range,
        posinf=max_range,
        neginf=min_range,
    )
    ranges = np.clip(ranges, min_range, max_range).astype(np.float32)

    median_kernel = _make_odd_kernel(
        _env_int("TB3_RL_LIDAR_MEDIAN_KERNEL", 3),
        min_value=1,
    )
    lowpass_kernel = _make_odd_kernel(
        _env_int("TB3_RL_LIDAR_LOWPASS_KERNEL", 5),
        min_value=1,
    )
    obstacle_margin = _env_float("TB3_RL_LIDAR_OBSTACLE_MARGIN_M", 0.08)

    robust_ranges = _circular_median_filter(ranges, median_kernel)
    smooth_ranges = _circular_lowpass_filter(robust_ranges, lowpass_kernel)

    if _env_bool("TB3_RL_LIDAR_UNIFORM_ANGLE_RESAMPLE", True) and _valid_scan_geometry(
        robust_ranges.size,
        scan_angle_min,
        scan_angle_increment,
    ):
        pooled = _uniform_angle_resample(
            robust_ranges=robust_ranges,
            smooth_ranges=smooth_ranges,
            num_bins=num_bins,
            min_range=min_range,
            max_range=max_range,
            obstacle_margin=obstacle_margin,
            angle_min=float(scan_angle_min),
            angle_increment=float(scan_angle_increment),
            angle_max=scan_angle_max,
        )
    else:
        pooled = _legacy_index_pool(
            robust_ranges=robust_ranges,
            smooth_ranges=smooth_ranges,
            num_bins=num_bins,
            min_range=min_range,
            max_range=max_range,
            obstacle_margin=obstacle_margin,
        )

    normalized = (pooled - min_range) / max((max_range - min_range), 1e-6)
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
    scan_angle_min: Optional[float] = None,
    scan_angle_increment: Optional[float] = None,
    scan_angle_max: Optional[float] = None,
) -> np.ndarray:
    """
    Navigation observation.

    [lidar_0 ... lidar_359, goal_distance_norm, goal_angle_norm,
     prev_linear_norm, prev_angular_norm]
    """
    lidar = downsample_lidar(
        scan_ranges=scan_ranges,
        num_bins=num_lidar_bins,
        min_range=min_lidar_range,
        max_range=max_lidar_range,
        scan_angle_min=scan_angle_min,
        scan_angle_increment=scan_angle_increment,
        scan_angle_max=scan_angle_max,
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
    scan_angle_min: Optional[float] = None,
    scan_angle_increment: Optional[float] = None,
    scan_angle_max: Optional[float] = None,
) -> np.ndarray:
    """
    SLAM-assisted exploration observation.

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
    """
    lidar = downsample_lidar(
        scan_ranges=scan_ranges,
        num_bins=num_lidar_bins,
        min_range=min_lidar_range,
        max_range=max_lidar_range,
        scan_angle_min=scan_angle_min,
        scan_angle_increment=scan_angle_increment,
        scan_angle_max=scan_angle_max,
    )

    coverage_ratio_norm = np.clip(float(coverage_ratio), 0.0, 1.0)
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
