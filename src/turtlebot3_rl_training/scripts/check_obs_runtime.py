#!/usr/bin/env python3
"""
check_obs_runtime.py
====================
Diagnostic: inspect the latest environment info fields from metrics_compact.csv.

Prints a summary of:
  - scan freshness (scan_age_sec, scan_stale)
  - confidence update stats
  - reward breakdown
  - safety terminal events
  - policy vs. executed v/w

Usage:
  python3 scripts/check_obs_runtime.py [path/to/metrics_compact.csv]

If no path is given, looks for the most recently modified CSV in rl_logs/.
"""

import os
import sys
import glob
import math


def find_latest_csv():
    candidates = glob.glob("rl_logs/**/*.csv", recursive=True)
    if not candidates:
        return None
    return max(candidates, key=os.path.getmtime)


def print_summary(path: str):
    import csv
    rows = []
    try:
        with open(path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(row)
    except Exception as e:
        print(f"Failed to read {path}: {e}", file=sys.stderr)
        return

    if not rows:
        print(f"No data in {path}")
        return

    def fget(row, col, default=float("nan")):
        try:
            v = row.get(col, "")
            return float(v) if v.strip() != "" else default
        except Exception:
            return default

    def iget(row, col, default=0):
        try:
            v = row.get(col, "")
            return int(float(v)) if v.strip() != "" else default
        except Exception:
            return default

    print(f"\n=== metrics_compact: {path} ({len(rows)} rows) ===\n")

    # Last row
    last = rows[-1]
    print("[Last step]")
    print(f"  step        : {iget(last, 'timesteps')}")
    print(f"  terminal    : {last.get('terminal_reason', '?')}")
    print(f"  reward_total: {fget(last, 'reward_total'):+.3f}")
    print()
    print("[Sensor freshness (last step)]")
    print(f"  scan_age_sec        : {fget(last, 'scan_age_sec'):.3f}  (warn if > 0.35)")
    print(f"  odom_age_sec        : {fget(last, 'odom_age_sec'):.3f}")
    print(f"  map_age_sec         : {fget(last, 'map_age_sec'):.3f}")
    print(f"  scan_stale          : {iget(last, 'scan_stale')}")
    print(f"  obs_stale           : {iget(last, 'obs_stale')}")
    print(f"  tf_pose_ok          : {iget(last, 'tf_pose_ok')}")
    print(f"  confidence_pose_ok  : {iget(last, 'confidence_pose_ok')}")
    print()
    print("[LiDAR sectors (last step)]")
    print(f"  raw:    F={fget(last,'raw_scan_front'):.3f}  L={fget(last,'raw_scan_left'):.3f}  "
          f"R={fget(last,'raw_scan_rear'):.3f}  Rt={fget(last,'raw_scan_right'):.3f}")
    print(f"  policy: F={fget(last,'policy_scan_front'):.3f}  L={fget(last,'policy_scan_left'):.3f}  "
          f"R={fget(last,'policy_scan_rear'):.3f}  Rt={fget(last,'policy_scan_right'):.3f}")
    print(f"  stamp_delta: {fget(last, 'lidar60_stamp_delta_sec'):.3f}s")
    print()
    print("[Reward breakdown (last step)]")
    for col in ("r_confidence", "r_slam", "r_priority", "r_wall",
                "r_safety_slow", "r_safety_terminal", "r_collision", "r_step"):
        print(f"  {col:25s}: {fget(last, col):+.4f}")
    print(f"  {'reward_total':25s}: {fget(last, 'reward_total'):+.4f}")
    print()
    print("[Safety (last step)]")
    print(f"  velocity_safety_terminal : {iget(last, 'velocity_safety_terminal')}")
    print(f"  velocity_safety_distance : {fget(last, 'velocity_safety_distance'):.3f}")
    print(f"  policy_v / executed_v    : {fget(last, 'policy_v'):.3f} / {fget(last, 'executed_v'):.3f}")
    print(f"  policy_w / executed_w    : {fget(last, 'policy_w'):.3f} / {fget(last, 'executed_w'):.3f}")
    print()

    # Aggregate over all rows
    n = len(rows)
    stale_count = sum(iget(r, "scan_stale") for r in rows)
    safety_term_count = sum(iget(r, "velocity_safety_terminal") for r in rows)
    collision_count = sum(iget(r, "collision") for r in rows)
    scan_ages = [fget(r, "scan_age_sec") for r in rows if math.isfinite(fget(r, "scan_age_sec")) and fget(r, "scan_age_sec") >= 0]
    conf_updated = [fget(r, "confidence_updated_cells") for r in rows if math.isfinite(fget(r, "confidence_updated_cells"))]

    print(f"[Aggregate over {n} rows]")
    print(f"  scan_stale steps        : {stale_count} / {n}  ({100.0*stale_count/max(n,1):.1f}%)")
    print(f"  safety_terminal steps   : {safety_term_count} / {n}")
    print(f"  collision steps         : {collision_count} / {n}")
    if scan_ages:
        import statistics
        print(f"  scan_age_sec: mean={statistics.mean(scan_ages):.3f}  max={max(scan_ages):.3f}  "
              f"p95={sorted(scan_ages)[int(len(scan_ages)*0.95)]:.3f}")
    if conf_updated:
        mean_conf = sum(conf_updated) / len(conf_updated)
        nonzero_conf = sum(1 for v in conf_updated if v > 0)
        print(f"  confidence_updated_cells: mean={mean_conf:.1f}  nonzero_steps={nonzero_conf}/{len(conf_updated)}")
    print()


def main():
    if len(sys.argv) > 1:
        path = sys.argv[1]
    else:
        path = find_latest_csv()
        if path is None:
            print("No CSV found in rl_logs/. Provide path as argument.", file=sys.stderr)
            sys.exit(1)
        print(f"Using: {path}")

    print_summary(path)


if __name__ == "__main__":
    main()
