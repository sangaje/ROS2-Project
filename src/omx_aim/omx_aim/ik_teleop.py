#!/usr/bin/env python3
"""OMX-Follower point-at IK teleop (config.yaml 적용).

변경점:
- 모든 캘리브레이션 / 안전 / 모터 설정값을 config.yaml 에서 로드
- 코드 안에 하드코딩된 상수 없음
"""

from __future__ import annotations

import argparse
import math
import sys
import time

from omx.hardware import (
    build_bus,
    get_dxl_symbols,
    ARM_MOTORS,
    # GRIPPER_MOTOR,
    MOTOR_ORDER,
)
from omx.config import load_config, Config


TICKS_PER_REV = 4096
RAD2TICK = TICKS_PER_REV / (2.0 * math.pi)


# ===========================================================
# IK (캘리브 값은 인자로 받음)
# ===========================================================

def aim_angles(x: float, y: float, z: float) -> dict[str, float]:
    if x == 0.0 and y == 0.0 and z == 0.0:
        raise ValueError("원점은 가리킬 수 없음")
    yaw = math.atan2(y, x)
    pitch = math.atan2(z, math.hypot(x, y))
    return {
        "shoulder_pan":  yaw,
        "shoulder_lift": pitch,
        "elbow_flex":    0.0,
        "wrist_flex":    0.0,
        # "wrist_roll":    0.0,
    }


def clamp_angles(
    angles: dict[str, float],
    safe_limits_rad: dict[str, tuple[float, float]],
) -> tuple[dict[str, float], list[str]]:
    clamped = {}
    warnings = []
    for m, a in angles.items():
        lo, hi = safe_limits_rad[m]
        if a < lo or a > hi:
            warnings.append(
                f"{m}: {math.degrees(a):+.1f} deg -> "
                f"clamp to [{math.degrees(lo):+.0f}, {math.degrees(hi):+.0f}]"
            )
        clamped[m] = max(lo, min(hi, a))
    return clamped, warnings


def angles_to_ticks(
    angles: dict[str, float],
    home: dict[str, int],
    sign: dict[str, int],
) -> dict[str, int]:
    return {
        m: int(round(home[m] + sign[m] * angle * RAD2TICK))
        for m, angle in angles.items()
    }


def print_solution(x: float, y: float, z: float, cfg: Config) -> None:
    raw_angles = aim_angles(x, y, z)
    angles, warnings = clamp_angles(raw_angles, cfg.safety.angle_limits_rad)
    ticks = angles_to_ticks(angles, cfg.calibration.home, cfg.calibration.sign)

    print(f"\nTarget = ({x:+.3f}, {y:+.3f}, {z:+.3f})")
    print(f"  {'Joint':<14}  {'Angle':>9}   {'Tick':>6}   {'HOME':>6}")
    for m in ARM_MOTORS:
        a_deg = math.degrees(angles[m])
        print(f"  {m:<14}  {a_deg:+7.2f} d   {ticks[m]:>6}   {cfg.calibration.home[m]:>6}")
    for w in warnings:
        print(f"  ! {w}")


# ===========================================================
# Controller
# ===========================================================

class AimController:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.bus = build_bus(cfg.motor.port)

    def connect_and_configure(self) -> None:
        s = get_dxl_symbols()
        DriveMode = s["DriveMode"]
        OperatingMode = s["OperatingMode"]

        self.bus.connect()
        with self.bus.torque_disabled():
            self.bus.configure_motors(return_delay_time=0)

            for motor in ARM_MOTORS:
                self.bus.write(
                    "Operating_Mode", motor,
                    OperatingMode.EXTENDED_POSITION.value, normalize=False,
                )
            # self.bus.write(
            #     "Operating_Mode", GRIPPER_MOTOR,
            #     OperatingMode.CURRENT_POSITION.value, normalize=False,
            # )
            # self.bus.write(
            #     "Drive_Mode", GRIPPER_MOTOR,
            #     DriveMode.NON_INVERTED.value, normalize=False,
            # )

            self.bus.write("Position_P_Gain", "elbow_flex",
                           self.cfg.motor.elbow_p_gain, normalize=False)
            self.bus.write("Position_I_Gain", "elbow_flex",
                           self.cfg.motor.elbow_i_gain, normalize=False)
            self.bus.write("Position_D_Gain", "elbow_flex",
                           self.cfg.motor.elbow_d_gain, normalize=False)

            for motor in ARM_MOTORS:
                self.bus.write("Profile_Velocity", motor,
                               self.cfg.motor.profile_velocity, normalize=False)
                self.bus.write("Profile_Acceleration", motor,
                               self.cfg.motor.profile_acceleration, normalize=False)

        self.bus.enable_torque(num_retry=3)
        print(f"속도 설정: velocity={self.cfg.motor.profile_velocity}, "
              f"accel={self.cfg.motor.profile_acceleration}")

    def disconnect(self, disable_torque: bool = True) -> None:
        if self.bus.is_connected:
            self.bus.disconnect(disable_torque=disable_torque)

    def read_present(self) -> dict[str, int]:
        return self.bus.sync_read("Present_Position", normalize=False)

    def go_home(self, wait_seconds: float = 2.5) -> None:
        present = self.read_present()
        home = self.cfg.calibration.home
        threshold = self.cfg.safety.large_delta_threshold_tick

        print("\nHome 으로 이동:")
        print(f"  {'Joint':<14}  {'현재':>6} -> {'HOME':>6}   delta")
        max_delta = 0
        for m in ARM_MOTORS:
            d = home[m] - present[m]
            max_delta = max(max_delta, abs(d))
            print(f"  {m:<14}  {present[m]:>6} -> {home[m]:>6}   {d:+6}")

        if max_delta > threshold:
            print(f"!! 큰 변화 (max delta={max_delta} tick)")
        if not _confirm("Home 이동: Enter 계속, Ctrl+C 취소"):
            print("Home 이동 취소.")
            return

        for m in ARM_MOTORS:
            self.bus.write("Goal_Position", m, home[m], normalize=False)
        time.sleep(wait_seconds)
        print("Home 도달 (추정).")

    def aim_at(self, x: float, y: float, z: float) -> None:
        raw_angles = aim_angles(x, y, z)
        angles, warnings = clamp_angles(raw_angles, self.cfg.safety.angle_limits_rad)
        ticks = angles_to_ticks(angles, self.cfg.calibration.home,
                                 self.cfg.calibration.sign)
        present = self.read_present()
        threshold = self.cfg.safety.large_delta_threshold_tick

        print(f"\nTarget = ({x:+.3f}, {y:+.3f}, {z:+.3f})")
        print(f"  {'Joint':<14}  {'Angle':>9}   {'Tick':>6}   {'Δtick':>6}")
        max_delta = 0
        for m in ARM_MOTORS:
            delta = ticks[m] - present[m]
            max_delta = max(max_delta, abs(delta))
            a_deg = math.degrees(angles[m])
            print(f"  {m:<14}  {a_deg:+7.2f} d   {ticks[m]:>6}   {delta:+6}")
        for w in warnings:
            print(f"  ! {w}")

        if max_delta > threshold:
            approx_deg = max_delta / RAD2TICK * 180.0 / math.pi
            print(f"!! 최대 변화 {max_delta} tick (약 {approx_deg:.1f} deg)")
            if not _confirm("실행: Enter 계속, Ctrl+C 취소"):
                print("취소됨.")
                return

        for m in ARM_MOTORS:
            self.bus.write("Goal_Position", m, ticks[m], normalize=False)
        print("명령 전송 완료.")


# ===========================================================
# 모드
# ===========================================================

def _confirm(prompt: str) -> bool:
    try:
        input(prompt + ": ")
        return True
    except (KeyboardInterrupt, EOFError):
        print()
        return False


def _parse_xyz(line: str) -> tuple[float, float, float]:
    parts = line.split()
    if len(parts) != 3:
        raise ValueError("x y z 세 개 필요")
    return float(parts[0]), float(parts[1]), float(parts[2])


def mode_dry_run(cfg: Config) -> int:
    print("=== Dry-run mode (모터 연결 없음) ===")
    print("좌표 입력 예: 0.2 0.0 0.1\n")
    while True:
        try:
            line = input("target x y z > ").strip()
        except (KeyboardInterrupt, EOFError):
            print()
            break
        if line in ("q", "quit", "exit", ""):
            break
        try:
            x, y, z = _parse_xyz(line)
            print_solution(x, y, z, cfg)
        except ValueError as e:
            print(f"입력 오류: {e}")
    return 0


def mode_measure_home(cfg: Config) -> int:
    print("=== Home 측정 mode ===")
    print("팔을 '곧게 펴서 정면 수평' 자세로 손으로 옮긴 뒤 Enter.\n")

    ctrl = AimController(cfg)
    try:
        ctrl.bus.connect()
        if not _confirm("준비됐으면 Enter"):
            return 0
        present = ctrl.read_present()
        print("\n--- 측정 결과 ---")
        print("아래를 config.yaml 의 calibration.home 에 복사하세요:\n")
        for m in MOTOR_ORDER:
            print(f'    {m}: {present[m]}')
        return 0
    finally:
        if ctrl.bus.is_connected:
            ctrl.bus.disconnect(disable_torque=True)


def mode_interactive(cfg: Config, skip_home: bool) -> int:
    print("=== Interactive aim mode ===")
    ctrl = AimController(cfg)
    try:
        ctrl.connect_and_configure()

        if not skip_home:
            print("\n초기 home 복귀를 권장합니다.")
            try:
                ctrl.go_home()
            except KeyboardInterrupt:
                print("\nHome 이동 중단.")

        print("\n명령:")
        print("  x y z       - 좌표로 조준")
        print("  home        - Home 복귀")
        print("  present     - 현재 관절 위치")
        print("  q / Ctrl+C  - 종료\n")

        while True:
            try:
                line = input("aim > ").strip()
            except (KeyboardInterrupt, EOFError):
                print()
                break
            if line in ("q", "quit", "exit"):
                break
            if line == "":
                continue
            if line == "home":
                try:
                    ctrl.go_home()
                except KeyboardInterrupt:
                    print("\nHome 이동 중단.")
                continue
            if line == "present":
                p = ctrl.read_present()
                for m in ARM_MOTORS:
                    print(f"  {m:<14} {p[m]}")
                continue
            try:
                x, y, z = _parse_xyz(line)
            except ValueError as e:
                print(f"입력 오류: {e}")
                continue
            try:
                ctrl.aim_at(x, y, z)
            except KeyboardInterrupt:
                print("\n명령 중단.")
            except Exception as e:
                print(f"오류: {e}")
        return 0
    finally:
        ctrl.disconnect(disable_torque=True)


# ===========================================================
# Entry
# ===========================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="OMX point-at IK teleop.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    p.add_argument("--config", default=None,
                   help="config.yaml 경로 (default: ./config.yaml)")
    p.add_argument("--dry-run", action="store_true",
                   help="모터 연결 없이 IK 계산만")
    p.add_argument("--measure-home", action="store_true",
                   help="현재 자세의 raw tick 측정")
    p.add_argument("--skip-home", action="store_true",
                   help="시작 시 home 복귀 건너뛰기")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    try:
        cfg = load_config(args.config)
    except (FileNotFoundError, ValueError) as e:
        print(f"Config 로드 실패: {e}", file=sys.stderr)
        return 1

    if args.dry_run:
        return mode_dry_run(cfg)
    if args.measure_home:
        return mode_measure_home(cfg)
    return mode_interactive(cfg, args.skip_home)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\n중단됨.")
        sys.exit(0)