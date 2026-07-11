import math
import os
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np


@dataclass(frozen=True)
class LidarPreprocessorConfig:
    """Frozen LiDAR semantics for a policy invocation.

    Training keeps its environment-variable compatibility path. Deployment
    passes this object from the policy contract, so sensor callbacks never read
    mutable process environment state.
    """

    canonical_front_zero: bool = True
    front_index: int = 0
    angle_offset_deg: float = 0.0
    flip_lr: bool = False
    uniform_angle_resample: bool = True
    median_kernel: int = 3
    lowpass_kernel: int = 5
    obstacle_margin_m: float = 0.08
    sector_bins: int = 0
    sector_lowpass_kernel: int = 3
    sector_expand_mode: str = 'linear'


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




def _env_optional_int(name: str, default: int = 0, min_value: int = 0) -> int:
    try:
        value = int(os.environ.get(name, str(default)))
    except Exception:
        value = int(default)
    return int(max(int(min_value), value))


def _expand_sector_bins_to_policy_bins(
    sector_ranges: np.ndarray,
    output_bins: int,
    sector_front_index: int,
    output_front_index: int,
    min_range: float,
    max_range: float,
    mode: str = "linear",
) -> np.ndarray:
    """Expand a low-dimensional canonical sector LiDAR vector to policy bins.

    This is a compatibility bridge for checkpoints trained with 360 LiDAR bins.
    It lets the real robot first aggregate noisy/sparse raw scans into stable
    sectors, e.g. 60 sectors, and then present a 360-bin tensor with the same
    shape expected by the existing SAC policy.

    The angular semantics are preserved:
      sector_ranges[sector_front_index] -> robot front
      output[output_front_index]        -> robot front

    `nearest` repeats sectors. `linear` circularly interpolates between sectors
    to avoid a stair-step pattern in the existing 360-bin feature extractor.
    """
    values = np.asarray(sector_ranges, dtype=np.float32)
    s = int(values.size)
    o = max(int(output_bins), 1)
    if s <= 0:
        return np.ones(o, dtype=np.float32) * float(max_range)
    if s == o and (int(sector_front_index) % s) == (int(output_front_index) % o):
        return np.clip(values, min_range, max_range).astype(np.float32)

    sector_front_index = int(sector_front_index) % s
    output_front_index = int(output_front_index) % o
    mode = str(mode or "linear").strip().lower()

    # Output bin j has the same physical angle as sector coordinate x.
    # j=output_front_index maps to x=sector_front_index.
    j = np.arange(o, dtype=np.float32)
    x = (j - float(output_front_index)) * (float(s) / float(o)) + float(sector_front_index)

    if mode in {"nearest", "repeat", "hold"}:
        idx = np.rint(x).astype(np.int64) % s
        out = values[idx]
    else:
        x0 = np.floor(x).astype(np.int64)
        frac = (x - x0.astype(np.float32)).astype(np.float32)
        i0 = x0 % s
        i1 = (x0 + 1) % s
        out = (1.0 - frac) * values[i0] + frac * values[i1]

    return np.clip(out, min_range, max_range).astype(np.float32)


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
    from numpy.lib.stride_tricks import sliding_window_view
    padded = np.concatenate([values[-half:], values, values[:half]])
    windows = sliding_window_view(padded, kernel_size)
    return np.median(windows, axis=1).astype(np.float32)


def _circular_lowpass_filter(values: np.ndarray, kernel_size: int) -> np.ndarray:
    """Gaussian-like circular low-pass filter over angular scan order."""
    kernel_size = _make_odd_kernel(kernel_size, min_value=1)
    if kernel_size <= 1 or values.size <= 2:
        return values.astype(np.float32, copy=False)

    half = kernel_size // 2
    x = np.linspace(-2.0, 2.0, kernel_size, dtype=np.float32)
    weights = np.exp(-0.5 * x * x).astype(np.float32)
    weights /= float(np.sum(weights))

    padded = np.concatenate([values[-half:], values, values[:half]])
    return np.convolve(padded, weights[::-1], mode='valid').astype(values.dtype)


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



def _canonical_angle_resample(
    robust_ranges: np.ndarray,
    smooth_ranges: np.ndarray,
    num_bins: int,
    min_range: float,
    max_range: float,
    obstacle_margin: float,
    angle_min: float,
    angle_increment: float,
    angle_max: Optional[float] = None,
    front_index: int = 0,
    config: Optional[LidarPreprocessorConfig] = None,
) -> np.ndarray:
    """Conservative angle-aware LiDAR resampling for policy input.

    This is deliberately *not* a learned regression/super-resolution step.
    A LaserScan already contains metric samples `(theta_i, r_i)`.  The policy,
    however, expects a fixed 360-bin vector whose index semantics are stable.

    Output convention:
      output[front_index]                 = robot front, angle 0
      output[front_index + num_bins / 4]  = robot left,  +pi/2
      output[front_index + num_bins / 2]  = robot rear,  +/-pi
      output[front_index + 3*num_bins/4] = robot right, -pi/2

    Main difference from the older nearest/bin-center method:
      - Each raw beam is treated as an angular cell, not a point sample.
      - Each target policy bin is also an angular cell.
      - A raw beam contributes to a target bin if the two angular cells overlap.
      - Close obstacles are preserved by min-pooling robust ranges.
      - Smooth ranges are overlap-weight averaged only for anti-aliasing.
      - Final value is `min(weighted_smooth, raw_min + obstacle_margin)`.

    This handles real LDS scans with 253/254 beams without leaving sparse holes
    in the 360 policy vector and without hallucinating obstacles/free-space via
    regression.
    """
    n = int(robust_ranges.size)
    if n <= 0 or num_bins <= 0:
        return np.ones(num_bins, dtype=np.float32) * max_range

    angle_min = float(angle_min)
    angle_increment = float(angle_increment)
    if not np.isfinite(angle_min) or not np.isfinite(angle_increment) or abs(angle_increment) < 1e-12:
        return _legacy_index_pool(
            robust_ranges,
            smooth_ranges,
            num_bins,
            min_range,
            max_range,
            obstacle_margin,
        )

    try:
        front_index = int(front_index) % int(num_bins)
    except Exception:
        front_index = 0

    # Full-circle LaserScan beams in the robot base frame.  ROS convention:
    # angle=0 is forward, positive angle is counter-clockwise/left.
    source_angles = angle_min + np.arange(n, dtype=np.float32) * angle_increment

    # v9: explicit scan-convention correction.
    # Some real LDS/driver paths use a different zero-angle or clockwise ordering
    # than the Gazebo sensor used for training.  Do NOT bake this into TF; keep it
    # local to the policy LiDAR preprocessing and RViz policy-scan debug.
    #
    #   TB3_RL_LIDAR_ANGLE_OFFSET_DEG : added to raw scan angles before binning
    #   TB3_RL_LIDAR_FLIP_LR          : mirror left/right before offset
    #
    # Use offset/flip only after checking /rl_raw_scan_points vs /map.
    config = config or _environment_lidar_config()
    if config.flip_lr:
        source_angles = -source_angles
    angle_offset_deg = float(config.angle_offset_deg)
    if abs(angle_offset_deg) > 1.0e-9:
        source_angles = source_angles + math.radians(float(angle_offset_deg))

    source_angles = np.arctan2(np.sin(source_angles), np.cos(source_angles)).astype(np.float32)

    # Estimate angular support of each raw beam.  For an LDS with ~253 beams this
    # is much wider than a 1-degree target bin; using this support avoids gaps.
    source_half_width = 0.5 * abs(float(angle_increment))
    if not np.isfinite(source_half_width) or source_half_width <= 1e-9:
        source_half_width = math.pi / max(float(n), 1.0)

    # Protect against malformed metadata.  If angle_max gives a clearly different
    # increment, keep the larger support for safety/conservatism.
    if angle_max is not None:
        try:
            meta_span = abs(float(angle_max) - float(angle_min))
            if np.isfinite(meta_span) and n > 1:
                meta_inc = meta_span / float(max(n - 1, 1))
                if np.isfinite(meta_inc) and meta_inc > 1e-9:
                    source_half_width = max(source_half_width, 0.5 * meta_inc)
        except Exception:
            pass

    target_half_width = math.pi / float(num_bins)
    support_width = source_half_width + target_half_width
    eps = 1e-7

    # For output bin j, choose the physical robot-frame angle assigned to that
    # bin.  With front_index=0: j=0 front, j=90 left, j=180 rear, j=270 right.
    bin_offsets = (np.arange(num_bins, dtype=np.float32) - float(front_index))
    target_centers = 2.0 * math.pi * bin_offsets / float(num_bins)
    target_centers = np.arctan2(np.sin(target_centers), np.cos(target_centers)).astype(np.float32)

    # Vectorized: build (n_source, num_bins) broadcast instead of a Python loop.
    source_col = source_angles[:, None]       # (n, 1)
    target_row = target_centers[None, :]     # (1, num_bins)
    diff_matrix = source_col - target_row    # (n, num_bins)
    # Normalize to [-pi, pi] using arctan2 on the whole matrix.
    diff_matrix = np.arctan2(np.sin(diff_matrix), np.cos(diff_matrix))
    abs_diff = np.abs(diff_matrix)           # (n, num_bins)

    support = support_width + eps
    in_support = abs_diff <= support         # bool (n, num_bins)

    # Min pool (obstacle detection)
    r_expanded = np.where(in_support, robust_ranges[:, None], np.inf)  # (n, num_bins)
    pooled_min = np.min(r_expanded, axis=0)  # (num_bins,)
    # For bins with no source in support, fall back to nearest source ray.
    no_support_mask = np.isinf(pooled_min)
    if np.any(no_support_mask):
        nearest_idx = np.argmin(abs_diff, axis=0)  # (num_bins,)
        fallback_min = robust_ranges[nearest_idx]
        pooled_min = np.where(no_support_mask, fallback_min, pooled_min)

    # Weighted smooth mean (triangular overlap weights for anti-aliasing)
    weights_matrix = np.maximum(support - abs_diff, eps) * in_support  # (n, num_bins)
    weight_sum = weights_matrix.sum(axis=0)  # (num_bins,)
    weight_sum_safe = np.maximum(weight_sum, eps)
    pooled_smooth = (weights_matrix * smooth_ranges[:, None]).sum(axis=0) / weight_sum_safe  # (num_bins,)
    # For bins with no support, fall back to nearest source ray smooth value.
    if np.any(no_support_mask):
        fallback_smooth = smooth_ranges[nearest_idx]
        pooled_smooth = np.where(no_support_mask, fallback_smooth, pooled_smooth)

    pooled = np.minimum(pooled_smooth, pooled_min + obstacle_margin)

    return np.clip(pooled, min_range, max_range).astype(np.float32)

_ENVIRONMENT_LIDAR_CONFIG: LidarPreprocessorConfig | None = None


def _environment_lidar_config() -> LidarPreprocessorConfig:
    """Legacy training configuration, resolved only once per process."""
    global _ENVIRONMENT_LIDAR_CONFIG
    if _ENVIRONMENT_LIDAR_CONFIG is None:
        _ENVIRONMENT_LIDAR_CONFIG = LidarPreprocessorConfig(
            canonical_front_zero=_env_bool('TB3_RL_LIDAR_CANONICAL_FRONT_ZERO', True),
            front_index=_env_int('TB3_RL_LIDAR_FRONT_INDEX', 0, min_value=0),
            angle_offset_deg=_env_float('TB3_RL_LIDAR_ANGLE_OFFSET_DEG', 0.0),
            flip_lr=_env_bool('TB3_RL_LIDAR_FLIP_LR', False),
            uniform_angle_resample=_env_bool('TB3_RL_LIDAR_UNIFORM_ANGLE_RESAMPLE', True),
            median_kernel=_env_int('TB3_RL_LIDAR_MEDIAN_KERNEL', 3),
            lowpass_kernel=_env_int('TB3_RL_LIDAR_LOWPASS_KERNEL', 5),
            obstacle_margin_m=_env_float('TB3_RL_LIDAR_OBSTACLE_MARGIN_M', 0.08),
            sector_bins=_env_optional_int('TB3_RL_LIDAR_SECTOR_BINS', 0, min_value=0),
            sector_lowpass_kernel=_env_int('TB3_RL_LIDAR_SECTOR_LOWPASS_KERNEL', 3),
            sector_expand_mode=os.environ.get('TB3_RL_LIDAR_SECTOR_EXPAND_MODE', 'linear'),
        )
    return _ENVIRONMENT_LIDAR_CONFIG


def downsample_lidar(
    scan_ranges: Sequence[float],
    num_bins: int = 360,
    min_range: float = 0.12,
    max_range: float = 3.5,
    scan_angle_min: Optional[float] = None,
    scan_angle_increment: Optional[float] = None,
    scan_angle_max: Optional[float] = None,
    config: Optional[LidarPreprocessorConfig] = None,
) -> np.ndarray:
    """Return a 0~1 normalized LiDAR vector with anti-alias filtering.

    Old behavior was index-space array_split/min-pooling, which assumes the raw
    LaserScan already has exactly the same angular sampling as the NN bins.
    This version uses LaserScan angle_min/angle_increment when available and
    resamples to uniform angular bins. It falls back to the old conservative
    index-space pool if geometry metadata is missing.

    Runtime knobs:
      TB3_RL_LIDAR_CANONICAL_FRONT_ZERO   default 1, set 0 to keep raw angle_min order
      TB3_RL_LIDAR_FRONT_INDEX            default 0, 30 for 60-bin middle-front test
      TB3_RL_LIDAR_ANGLE_OFFSET_DEG       default 0, raw scan angular correction before policy binning
      TB3_RL_LIDAR_FLIP_LR                default 0, mirror raw scan left/right before policy binning
      TB3_RL_LIDAR_SECTOR_BINS            optional intermediate sector count; leave 0 for direct output bins
      TB3_RL_LIDAR_SECTOR_LOWPASS_KERNEL  default 3, sector-level conservative low-pass
      TB3_RL_LIDAR_SECTOR_EXPAND_MODE     default linear, or nearest/repeat when sector bridge is used
      TB3_RL_LIDAR_UNIFORM_ANGLE_RESAMPLE default 1, set 0 to fallback
      TB3_RL_LIDAR_MEDIAN_KERNEL          default 3, set 1 to disable
      TB3_RL_LIDAR_LOWPASS_KERNEL         default 5, set 1 to disable
      TB3_RL_LIDAR_OBSTACLE_MARGIN_M      default 0.08
    """
    config = config or _environment_lidar_config()

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

    median_kernel = _make_odd_kernel(config.median_kernel, min_value=1)
    lowpass_kernel = _make_odd_kernel(config.lowpass_kernel, min_value=1)
    obstacle_margin = max(float(config.obstacle_margin_m), 0.0)

    robust_ranges = _circular_median_filter(ranges, median_kernel)
    smooth_ranges = _circular_lowpass_filter(robust_ranges, lowpass_kernel)

    geometry_valid = _valid_scan_geometry(
        robust_ranges.size,
        scan_angle_min,
        scan_angle_increment,
    )

    if geometry_valid and config.canonical_front_zero:
        front_index = int(config.front_index) % int(num_bins)

        # v4: optional stable-sector bridge.
        # Existing 145000 checkpoint still requires 360 LiDAR inputs, but the
        # real robot scan is only ~252 beams.  A direct 252->360 expansion is
        # over-resolved and can amplify angular aliasing.  Instead:
        #     raw scan -> filtered canonical sectors, e.g. 60 -> expand to 360
        # This keeps the neural network input shape unchanged while reducing
        # beam-count jitter and preserving obstacles conservatively.
        sector_bins = max(int(config.sector_bins), 0)
        if sector_bins >= 2 and sector_bins != num_bins:
            sector_front_index = int(round(float(front_index) * float(sector_bins) / float(num_bins))) % int(sector_bins)
            sector_pooled = _canonical_angle_resample(
                robust_ranges=robust_ranges,
                smooth_ranges=smooth_ranges,
                num_bins=sector_bins,
                min_range=min_range,
                max_range=max_range,
                obstacle_margin=obstacle_margin,
                angle_min=float(scan_angle_min),
                angle_increment=float(scan_angle_increment),
                angle_max=scan_angle_max,
                front_index=sector_front_index,
                config=config,
            )

            sector_lowpass_kernel = _make_odd_kernel(config.sector_lowpass_kernel, min_value=1)
            if sector_lowpass_kernel > 1 and sector_pooled.size > 2:
                # Conservative sector-level low-pass: smooth free-space, but do
                # not allow smoothing to erase a close obstacle by more than the
                # existing obstacle margin.
                sector_smooth = _circular_lowpass_filter(sector_pooled, sector_lowpass_kernel)
                sector_pooled = np.minimum(sector_smooth, sector_pooled + obstacle_margin)
                sector_pooled = np.clip(sector_pooled, min_range, max_range).astype(np.float32)

            pooled = _expand_sector_bins_to_policy_bins(
                sector_ranges=sector_pooled,
                output_bins=num_bins,
                sector_front_index=sector_front_index,
                output_front_index=front_index,
                min_range=min_range,
                max_range=max_range,
                mode=config.sector_expand_mode,
            )
        else:
            pooled = _canonical_angle_resample(
                robust_ranges=robust_ranges,
                smooth_ranges=smooth_ranges,
                num_bins=num_bins,
                min_range=min_range,
                max_range=max_range,
                obstacle_margin=obstacle_margin,
                angle_min=float(scan_angle_min),
                angle_increment=float(scan_angle_increment),
                angle_max=scan_angle_max,
                front_index=front_index,
                config=config,
            )
    elif config.uniform_angle_resample and geometry_valid:
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
    include_target_priority: bool = True,
    trim_extra_stats: bool = True,
    lidar_config: Optional[LidarPreprocessorConfig] = None,
) -> np.ndarray:
    """
    SLAM-assisted exploration observation.

    Full layout (trim_extra_stats=False -- the pre-trim, currently deployed
    contract in scout_rl_policy_contract.json):
      [
        lidar_0 ... lidar_359,
        coverage_ratio,
        coverage_delta_norm,
        frontier_distance_norm,
        frontier_angle_norm,
        target_priority,    # omitted when include_target_priority=False
        mean_confidence_norm,
        stale_ratio,
        low_confidence_ratio,
        prev_linear_norm,
        prev_angular_norm,
      ]                      # lidar+10, or lidar+9 without target_priority

    Trimmed layout (trim_extra_stats=True -- default, new training):
      [
        lidar_0 ... lidar_359,
        coverage_ratio,
        frontier_distance_norm,  # omitted when include_target_priority=False
        frontier_angle_norm,     # omitted when include_target_priority=False
        target_priority,         # omitted when include_target_priority=False
        prev_linear_norm,
        prev_angular_norm,
      ]                      # lidar+6, or lidar+3 without target_priority

    frontier_distance/frontier_angle are only real signal when the priority
    map is enabled (compute_frontier_info() runs).  In no-priority/fast-stats
    mode (include_target_priority=False), exploration_map.py hardcodes them to
    constants (frontier_distance=size_m, frontier_angle=0.0) every step -- a
    dead, zero-information input to the policy -- so they are dropped from the
    trimmed vector together with target_priority in that mode.

    coverage_delta_norm / mean_confidence_norm / stale_ratio /
    low_confidence_ratio are dropped from the trimmed vector: coverage_delta
    and confidence_gain (the CSV-logged proxies) do have real variance, but
    none of these four had a directly-verifiable CSV column and mean_confidence
    in particular is largely redundant with what the map CNN's
    global-average-pool branch already sees in the confidence channel.

    A checkpoint already trained (and frozen behind a policy contract, e.g.
    system_bringup/config/scout_rl_policy_contract.json) on the full-width
    vector needs trim_extra_stats=False to match; gazebo_nav_env.py derives
    this from TB3_RL_OBS_EXTRA_DIM_OVERRIDE so a frozen inference contract
    gets the full vector while new training gets the trimmed default without
    changing this shared function's behavior for both at once.
    """
    lidar = downsample_lidar(
        scan_ranges=scan_ranges,
        num_bins=num_lidar_bins,
        min_range=min_lidar_range,
        max_range=max_lidar_range,
        scan_angle_min=scan_angle_min,
        scan_angle_increment=scan_angle_increment,
        scan_angle_max=scan_angle_max,
        config=lidar_config,
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

    if trim_extra_stats:
        if include_target_priority:
            extra = np.array(
                [
                    coverage_ratio_norm,
                    frontier_distance_norm,
                    frontier_angle_norm,
                    target_priority_norm,
                    prev_linear_norm,
                    prev_angular_norm,
                ],
                dtype=np.float32,
            )
        else:
            # No-priority policy input: remove target_priority AND frontier_* from
            # the actor/critic vector.  target_priority is dropped because there is
            # no priority map to score in this mode; frontier_distance/frontier_angle
            # are dropped because exploration_map.py's fast-stats path hardcodes them
            # to constants here (dead input, see docstring).  This is the lidar+3
            # trimmed layout.
            extra = np.array(
                [
                    coverage_ratio_norm,
                    prev_linear_norm,
                    prev_angular_norm,
                ],
                dtype=np.float32,
            )
    elif include_target_priority:
        extra = np.array(
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
        )
    else:
        # No-priority policy input: remove target_priority from the actor/critic
        # vector.  This changes vector dim from lidar+10 to lidar+9 and must be
        # trained as a new model/checkpoint family.
        extra = np.array(
            [
                coverage_ratio_norm,
                coverage_delta_norm,
                frontier_distance_norm,
                frontier_angle_norm,
                mean_confidence_norm,
                stale_ratio_norm,
                low_confidence_ratio_norm,
                prev_linear_norm,
                prev_angular_norm,
            ],
            dtype=np.float32,
        )

    obs = np.concatenate([lidar, extra])

    return obs.astype(np.float32)
