#!/usr/bin/env python3
"""Direct OMX servo scan test.

This bypasses ROS, YOLO, Nav2, and the dashboard. Use it to answer one
question quickly: can the Dynamixel bus lock torque and move pan/lift?
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_OMX = REPO_ROOT / "src" / "omx_aim"
if SRC_OMX.is_dir():
    sys.path.insert(0, str(SRC_OMX))

from omx.config import CalibrationConfig, MotorConfig, SafetyConfig  # noqa: E402
from omx.controller import OmxController  # noqa: E402


def _tuple_pairs(raw: dict) -> dict[str, tuple[float, float]]:
    out = {}
    for key, value in raw.items():
        if len(value) != 2:
            raise ValueError(f"{key}: expected [lo, hi], got {value!r}")
        out[key] = (float(value[0]), float(value[1]))
    return out


def load_motor_only_config(path: Path, port_override: str = ""):
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    cfg = SimpleNamespace(
        motor=MotorConfig(**raw["motor"]),
        calibration=CalibrationConfig(**raw["calibration"]),
        safety=SafetyConfig(
            angle_limits_deg=_tuple_pairs(raw["safety"]["angle_limits_deg"]),
            max_step_deg=raw["safety"]["max_step_deg"],
            large_delta_threshold_tick=raw["safety"]["large_delta_threshold_tick"],
        ),
    )
    if port_override:
        cfg.motor.port = port_override
    return cfg


def try_read_position(ctrl: OmxController) -> dict[str, object]:
    bus = getattr(ctrl, "bus", None)
    if bus is None or not hasattr(bus, "read"):
        return {}
    values = {}
    for motor in ("shoulder_pan", "shoulder_lift"):
        try:
            values[motor] = bus.read("Present_Position", motor, normalize=False)
        except TypeError:
            try:
                values[motor] = bus.read("Present_Position", [motor], normalize=False)
            except Exception as exc:  # noqa: BLE001
                values[motor] = f"{type(exc).__name__}: {exc}"
        except Exception as exc:  # noqa: BLE001
            values[motor] = f"{type(exc).__name__}: {exc}"
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a direct OMX pan scan without ROS."
    )
    parser.add_argument(
        "--config",
        default=str(REPO_ROOT / "src" / "omx_aim" / "config" / "config.yaml"),
        help="Path to omx_aim config.yaml.",
    )
    parser.add_argument("--port", default="", help="Override Dynamixel port.")
    parser.add_argument("--dry-run", action="store_true", help="Do not touch hardware.")
    parser.add_argument("--home-only", action="store_true", help="Connect and home only.")
    parser.add_argument("--duration-sec", type=float, default=12.0)
    parser.add_argument("--period-sec", type=float, default=6.0)
    parser.add_argument("--half-angle-deg", type=float, default=35.0)
    parser.add_argument("--center-yaw-deg", type=float, default=0.0)
    parser.add_argument("--rate-hz", type=float, default=10.0)
    parser.add_argument(
        "--readback",
        action="store_true",
        help="Try to read Present_Position after each command.",
    )
    parser.add_argument(
        "--leave-torque-on",
        action="store_true",
        help="Disconnect serial without disabling torque at the end.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg = load_motor_only_config(Path(args.config), args.port)
    ctrl = OmxController(cfg, dry_run=args.dry_run)
    center_yaw_rad = math.radians(args.center_yaw_deg)
    dt = 1.0 / max(1.0, args.rate_hz)

    print(
        "OMX_SCAN_SWEEP_TEST | "
        f"port={cfg.motor.port} dry_run={args.dry_run} "
        f"half_angle={args.half_angle_deg:.1f}deg period={args.period_sec:.2f}s"
    )

    try:
        ctrl.connect()
        ctrl.go_home()
        if args.home_only:
            print("OMX_SCAN_SWEEP_TEST | home command sent")
            return 0

        deadline = time.monotonic() + max(0.1, args.duration_sec)
        while time.monotonic() < deadline:
            now = time.monotonic()
            ctrl.scan_sweep(
                now,
                args.half_angle_deg,
                args.period_sec,
                center_yaw_rad,
            )
            line = (
                "OMX_SCAN_SWEEP_TEST | "
                f"yaw={math.degrees(ctrl.yaw):+.1f}deg "
                f"pitch={math.degrees(ctrl.pitch):+.1f}deg"
            )
            if args.readback:
                line += f" readback={try_read_position(ctrl)}"
            print(line, flush=True)
            time.sleep(dt)
        return 0
    except KeyboardInterrupt:
        print("OMX_SCAN_SWEEP_TEST | interrupted")
        return 130
    except Exception as exc:  # noqa: BLE001
        print(
            "OMX_SCAN_SWEEP_TEST_ERROR | "
            f"{type(exc).__name__}: {exc}\n"
            "If this fails but video still works in the launch, the ROS node will "
            "fall back to video_only mode and the arm will not move.",
            file=sys.stderr,
        )
        return 1
    finally:
        try:
            bus = getattr(ctrl, "bus", None)
            if args.leave_torque_on and bus is not None and getattr(
                bus, "is_connected", False
            ):
                bus.disconnect(disable_torque=False)
            else:
                ctrl.disconnect()
        except Exception as exc:  # noqa: BLE001
            print(
                "OMX_SCAN_SWEEP_TEST_DISCONNECT_ERROR | "
                f"{type(exc).__name__}: {exc}",
                file=sys.stderr,
            )


if __name__ == "__main__":
    raise SystemExit(main())
