#!/usr/bin/env python3
"""
plot_metrics_compact.py
=======================
Plot key training metrics from metrics_compact.csv.

Plots:
  1. Critic loss & Actor loss over time
  2. Episode reward (mean window) over time
  3. Safety terminal count per episode
  4. Confidence updated cells per step
  5. Scan age per step
  6. Policy v/w vs executed v/w

Usage:
  python3 scripts/plot_metrics_compact.py [path/to/metrics_compact.csv]

Requires: matplotlib (pip install matplotlib)
"""

import os
import sys
import glob
import math
import csv
from pathlib import Path


def find_latest_csv():
    candidates = glob.glob("rl_logs/**/*.csv", recursive=True)
    if not candidates:
        return None
    return max(candidates, key=os.path.getmtime)


def load_csv(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def fget(row, col, default=float("nan")):
    try:
        v = row.get(col, "")
        if v.strip() == "":
            return default
        return float(v)
    except Exception:
        return default


def col_array(rows, col, default=float("nan")):
    return [fget(r, col, default) for r in rows]


def finite(arr):
    return [x for x in arr if math.isfinite(x)]


def main():
    try:
        import matplotlib
        matplotlib.use("Agg")  # non-interactive backend works everywhere
        import matplotlib.pyplot as plt
    except ImportError:
        sys.exit("matplotlib not found. Install with: pip install matplotlib")

    if len(sys.argv) > 1:
        path = sys.argv[1]
    else:
        path = find_latest_csv()
        if path is None:
            sys.exit("No CSV found in rl_logs/. Provide path as argument.")
        print(f"Using: {path}")

    rows = load_csv(path)
    if not rows:
        sys.exit("No data in CSV.")

    steps = col_array(rows, "timesteps")
    critic_loss = col_array(rows, "critic_loss")
    actor_loss = col_array(rows, "actor_loss")
    ep_reward = col_array(rows, "mean_episode_reward")
    safety_term_ep = col_array(rows, "safety_terminal_count_ep", 0.0)
    conf_updated = col_array(rows, "confidence_updated_cells", 0.0)
    scan_age = col_array(rows, "scan_age_sec", float("nan"))
    policy_v = col_array(rows, "policy_v", 0.0)
    executed_v = col_array(rows, "executed_v", 0.0)
    policy_w = col_array(rows, "policy_w", 0.0)
    executed_w = col_array(rows, "executed_w", 0.0)

    fig, axes = plt.subplots(3, 2, figsize=(16, 12))
    fig.suptitle(f"SAC Training Metrics\n{Path(path).name}", fontsize=11)

    def _plot(ax, xs, ys, label, color, ylabel, title, hline=None, ymin=None, ymax=None):
        pairs = [(x, y) for x, y in zip(xs, ys) if math.isfinite(x) and math.isfinite(y)]
        if not pairs:
            ax.set_title(title + " (no data)")
            return
        px, py = zip(*pairs)
        ax.plot(px, py, color=color, linewidth=0.8, alpha=0.7, label=label)
        if hline is not None:
            ax.axhline(hline, color="red", linewidth=0.8, linestyle="--", alpha=0.6, label=f"threshold={hline}")
        if ymin is not None:
            ax.set_ylim(bottom=ymin)
        if ymax is not None:
            ax.set_ylim(top=ymax)
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("timesteps", fontsize=8)
        ax.set_ylabel(ylabel, fontsize=8)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    _plot(axes[0, 0], steps, critic_loss, "critic_loss", "steelblue", "loss", "Critic Loss (Q)")
    axes[0, 0].plot(
        [x for x, y in zip(steps, actor_loss) if math.isfinite(x) and math.isfinite(y)],
        [y for x, y in zip(steps, actor_loss) if math.isfinite(x) and math.isfinite(y)],
        color="orange", linewidth=0.8, alpha=0.7, label="actor_loss",
    )
    axes[0, 0].legend(fontsize=7)

    _plot(axes[0, 1], steps, ep_reward, "mean_ep_reward", "green", "reward", "Episode Reward (mean window)")

    _plot(axes[1, 0], steps, safety_term_ep, "safety_terminal_count/ep", "red",
          "count", "Safety Terminal Count (per episode)")

    _plot(axes[1, 1], steps, conf_updated, "conf_updated_cells", "purple",
          "cells", "Confidence Updated Cells (per step)")

    _plot(axes[2, 0], steps, scan_age, "scan_age_sec", "darkorange",
          "seconds", "Scan Age (per step)", hline=0.35, ymin=0.0, ymax=2.0)

    # v/w overlay
    pv_pairs = [(x, y) for x, y in zip(steps, policy_v) if math.isfinite(x) and math.isfinite(y)]
    ev_pairs = [(x, y) for x, y in zip(steps, executed_v) if math.isfinite(x) and math.isfinite(y)]
    pw_pairs = [(x, y) for x, y in zip(steps, policy_w) if math.isfinite(x) and math.isfinite(y)]
    ew_pairs = [(x, y) for x, y in zip(steps, executed_w) if math.isfinite(x) and math.isfinite(y)]
    ax = axes[2, 1]
    if pv_pairs:
        ax.plot(*zip(*pv_pairs), color="blue", linewidth=0.6, alpha=0.6, label="policy_v")
    if ev_pairs:
        ax.plot(*zip(*ev_pairs), color="cyan", linewidth=0.6, alpha=0.6, label="executed_v")
    if pw_pairs:
        ax.plot(*zip(*pw_pairs), color="red", linewidth=0.6, alpha=0.4, label="policy_w")
    if ew_pairs:
        ax.plot(*zip(*ew_pairs), color="salmon", linewidth=0.6, alpha=0.4, label="executed_w")
    ax.set_title("Policy v/w vs Executed v/w", fontsize=9)
    ax.set_xlabel("timesteps", fontsize=8)
    ax.set_ylabel("m/s or rad/s", fontsize=8)
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = str(Path(path).with_suffix(".png"))
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    print(f"Saved: {out_path}")

    # Also try interactive display
    try:
        matplotlib.use("TkAgg")
        plt.show()
    except Exception:
        pass


if __name__ == "__main__":
    main()
