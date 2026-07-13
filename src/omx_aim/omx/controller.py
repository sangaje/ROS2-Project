"""OMX 상위 제어 — Dynamixel bus 위에 좌표 → 각도 변환 로직.

omx/hardware.py: 저수준 (DXL bus, MOTOR_ORDER 등)
omx/controller.py (이 파일): 상위 (aim_at_coord, IBVS step)
    
ROS 의존성 없음.

wrist_roll, gripper 는 H2 부터 물리적으로 제거됨.
격발은 별도 MCU (Jetson GPIO + 트랜지스터) 가 /omx/fire 토픽 받아 처리.
"""

from __future__ import annotations

import math
import time

from omx.hardware import build_bus, get_dxl_symbols, ARM_MOTORS, MOTOR_ORDER
from omx.config import Config


TICKS_PER_REV = 4096
RAD2TICK = TICKS_PER_REV / (2.0 * math.pi)


class OmxController:
    """OMX 4 모터 (shoulder_pan/lift, elbow_flex, wrist_flex) 제어.

    좌표 (arm_base 기준 x, y, z) 를 받아 shoulder_pan/lift 각도로 변환.
    elbow_flex, wrist_flex 는 home 위치 고정.

    IBVS 모드에선 영상 오차 (error_x, error_y) 기반 미세 보정.
    """

    def __init__(self, cfg: Config, dry_run: bool = False, logger=None):
        self.cfg = cfg
        self.dry_run = dry_run
        self.bus = None if dry_run else build_bus(cfg.motor.port)
        self.yaw = 0.0
        self.pitch = 0.0
        self.logger = logger
        # IBVS 미분항 상태 (lead compensation, Phase B)
        self._prev_error_x = 0.0
        self._prev_error_y = 0.0
        self._de_x_ema = 0.0
        self._de_y_ema = 0.0
        self._prev_ibvs_t = None
        self._scan_sweep_start_t = None
        self._scan_sweep_center_yaw = 0.0
        self._scan_sweep_center_pitch = 0.0

    def _log(self, msg, level="info"):
        if self.logger:
            getattr(self.logger, level)(msg)
        else:
            print(msg)

    # ----- 연결 관리 -----

    def connect(self):
        if self.dry_run:
            self._log("[dry-run] OMX 연결 생략")
            return
        s = get_dxl_symbols()
        OperatingMode = s["OperatingMode"]

        self.bus.connect()
        with self.bus.torque_disabled():
            self.bus.configure_motors(return_delay_time=0)
            for m in ARM_MOTORS:
                self.bus.write("Operating_Mode", m,
                               OperatingMode.EXTENDED_POSITION.value,
                               normalize=False)
            # gripper 떼어냄 - 격발은 별도 MCU (Jetson GPIO)
            for m in MOTOR_ORDER:
                self.bus.write("Profile_Velocity", m,
                               self.cfg.motor.profile_velocity, normalize=False)
                self.bus.write("Profile_Acceleration", m,
                               self.cfg.motor.profile_acceleration, normalize=False)
        self.bus.enable_torque(num_retry=3)
        self._log("OMX 연결 완료")

    def disconnect(self):
        if self.dry_run or self.bus is None or not self.bus.is_connected:
            return
        self.bus.disconnect(disable_torque=True)
        self._log("OMX 연결 해제")

    def go_home(self):
        self.reset_scan_sweep()
        if self.dry_run:
            self.yaw = 0.0
            self.pitch = 0.0
            self._log("[dry-run] Home 이동 시뮬레이션")
            return
        home = self.cfg.calibration.home
        for m in MOTOR_ORDER:
            self.bus.write("Goal_Position", m, home[m], normalize=False)
        self.yaw = 0.0
        self.pitch = 0.0
        self._log("Home 명령 전송 (non-blocking)")

    def reset_scan_sweep(self):
        self._scan_sweep_start_t = None
        self._scan_sweep_center_yaw = 0.0
        self._scan_sweep_center_pitch = self.pitch

    def _write_angles(self, yaw: float, pitch: float):
        limits = self.cfg.safety.angle_limits_rad
        lo, hi = limits["shoulder_pan"]
        yaw = max(lo, min(hi, yaw))
        lo, hi = limits["shoulder_lift"]
        pitch = max(lo, min(hi, pitch))

        self.yaw = yaw
        self.pitch = pitch

        if self.dry_run:
            return

        home = self.cfg.calibration.home
        sign = self.cfg.calibration.sign
        yaw_tick = int(round(home["shoulder_pan"]
                             + sign["shoulder_pan"] * yaw * RAD2TICK))
        pitch_tick = int(round(home["shoulder_lift"]
                               + sign["shoulder_lift"] * pitch * RAD2TICK))
        self.bus.write("Goal_Position", "shoulder_pan",
                       yaw_tick, normalize=False)
        self.bus.write("Goal_Position", "shoulder_lift",
                       pitch_tick, normalize=False)

    def scan_sweep(self, now: float, half_angle_deg: float,
                   period_sec: float, center_yaw_rad: float | None = None):
        """SCANNING 중에도 pan을 좌우로 연속적으로(호를 그리며) 훑는다.

        예전 구현은 half_period 마다 목표를 반대쪽 끝으로 순간 이동시켰다
        -- 서보가 실제로 그 끝까지 도달하기 전에 목표가 또 뒤집히면
        중간에서 방향이 꺾여서 period_sec 를 아무리 늘려도(서보 속도가
        그에 못 미치면) 절대 끝까지 못 가는 경우가 있었다. 대신 매 tick
        마다 목표 각도를 삼각파로 조금씩(연속적으로) 옮겨서 서보가 항상
        "바로 앞"의 목표를 매끄럽게 뒤쫓게 한다 -- 서보가 못 따라가도
        방향은 위상에 따라서만 바뀌므로 순간이동/중간에 꺾이는 문제
        자체가 없고, 늦게라도 결국 양 끝까지 도달한다.
        """
        period_sec = max(0.5, float(period_sec))
        half_angle = math.radians(max(0.0, float(half_angle_deg)))

        if self._scan_sweep_start_t is None:
            self._scan_sweep_start_t = now
            self._scan_sweep_center_yaw = (
                0.0 if center_yaw_rad is None else float(center_yaw_rad)
            )
            self._scan_sweep_center_pitch = self.pitch
            self._log(
                "OMX_SCAN_SWEEP_START | "
                f"center_yaw={math.degrees(self._scan_sweep_center_yaw):+.1f}deg "
                f"half_angle={math.degrees(half_angle):.1f}deg "
                f"period={period_sec:.2f}s"
            )
        elif center_yaw_rad is not None:
            center_yaw_rad = float(center_yaw_rad)
            if abs(center_yaw_rad - self._scan_sweep_center_yaw) > math.radians(5.0):
                self._scan_sweep_center_yaw = center_yaw_rad
                self._scan_sweep_start_t = now
                self._log(
                    "OMX_SCAN_SWEEP_RECENTER | "
                    f"center_yaw={math.degrees(center_yaw_rad):+.1f}deg"
                )

        elapsed = max(0.0, now - self._scan_sweep_start_t)
        phase = (elapsed % period_sec) / period_sec  # 0..1, 한 바퀴(왕복) 주기
        triangle = 2.0 * abs(2.0 * phase - 1.0) - 1.0  # +1 -> -1 -> +1 삼각파
        yaw = self._scan_sweep_center_yaw + triangle * half_angle
        self._write_angles(yaw, self._scan_sweep_center_pitch)

    # ----- 조준 -----

    def aim_at_coord(self, x, y, z):
        """arm_base 기준 (x, y, z) 좌표를 향해 coarse 조준.
        
        새 yaw = atan2(y, x), pitch = atan2(z, sqrt(x²+y²)).
        safety angle limit 으로 clamp.
        elbow_flex / wrist_flex 는 home 으로 복귀.
        """
        if x == 0.0 and y == 0.0 and z == 0.0:
            self._log("원점 좌표는 가리킬 수 없음", "warn")
            return
        self.reset_scan_sweep()

        new_yaw = math.atan2(y, x)
        new_pitch = math.atan2(z, math.hypot(x, y))

        limits = self.cfg.safety.angle_limits_rad
        lo, hi = limits["shoulder_pan"]
        new_yaw = max(lo, min(hi, new_yaw))
        lo, hi = limits["shoulder_lift"]
        new_pitch = max(lo, min(hi, new_pitch))

        self.yaw = new_yaw
        self.pitch = new_pitch

        if not self.dry_run:
            home = self.cfg.calibration.home
            sign = self.cfg.calibration.sign
            yaw_tick = int(round(home["shoulder_pan"]
                                 + sign["shoulder_pan"] * new_yaw * RAD2TICK))
            pitch_tick = int(round(home["shoulder_lift"]
                                   + sign["shoulder_lift"] * new_pitch * RAD2TICK))
            # wrist_roll 떼어냄
            for m in ("elbow_flex", "wrist_flex"):
                self.bus.write("Goal_Position", m, home[m], normalize=False)
            self.bus.write("Goal_Position", "shoulder_pan",
                           yaw_tick, normalize=False)
            self.bus.write("Goal_Position", "shoulder_lift",
                           pitch_tick, normalize=False)

        self._log(f"Coarse aim: yaw={math.degrees(new_yaw):.1f}, "
                  f"pitch={math.degrees(new_pitch):.1f}")

    def step_ibvs(self, error_x, error_y):
        """IBVS (Image-Based Visual Servoing) 한 스텝.

        영상 오차 (ex, ey, [-1, 1] 정규화) 기반 yaw/pitch 미세 보정.
        deadband 안이면 움직임 없음.
        max_step 으로 한 tick 당 이동 제한.

        Phase B (움직이는 표적):
            kd_yaw / kd_pitch 가 0 이면 순수 P controller (기존 동작 동일).
            kd > 0 이면 EMA 필터링된 de/dt 를 더해 미래 위치 예측
            (lead compensation).

        Returns: True if 움직임 발생, False otherwise.
        """
        max_step = self.cfg.safety.max_step_rad
        self.reset_scan_sweep()
        deadband_x = self.cfg.ibvs.deadband_x
        deadband_y = self.cfg.ibvs.deadband_y

        # ----- 미분항 업데이트 (deadband 와 무관하게 항상 갱신) -----
        now = time.time()
        if self._prev_ibvs_t is None:
            dt = 0.0
        else:
            dt = now - self._prev_ibvs_t
        self._prev_ibvs_t = now

        reset_gap = self.cfg.ibvs.derivative_reset_gap_sec
        if dt <= 0.0 or dt > reset_gap:
            # 첫 호출 또는 오랜만에 재진입 -> 미분 reset
            self._de_x_ema = 0.0
            self._de_y_ema = 0.0
        else:
            de_x_raw = (error_x - self._prev_error_x) / dt
            de_y_raw = (error_y - self._prev_error_y) / dt
            alpha = self.cfg.ibvs.derivative_ema_alpha
            self._de_x_ema = alpha * de_x_raw + (1.0 - alpha) * self._de_x_ema
            self._de_y_ema = alpha * de_y_raw + (1.0 - alpha) * self._de_y_ema
        self._prev_error_x = error_x
        self._prev_error_y = error_y

        # ----- Deadband gate (기존 그대로) -----
        ex = 0.0 if abs(error_x) < deadband_x else error_x
        ey = 0.0 if abs(error_y) < deadband_y else error_y

        if ex == 0.0 and ey == 0.0:
            return False

        # ----- P + D (lead) -----
        kp_y = self.cfg.ibvs.kp_yaw
        kp_p = self.cfg.ibvs.kp_pitch
        kd_y = self.cfg.ibvs.kd_yaw
        kd_p = self.cfg.ibvs.kd_pitch

        delta_yaw   = self.cfg.ibvs.sign_vs_x * (kp_y * ex + kd_y * self._de_x_ema)
        delta_pitch = self.cfg.ibvs.sign_vs_y * (kp_p * ey + kd_p * self._de_y_ema)

        delta_yaw = max(-max_step, min(max_step, delta_yaw))
        delta_pitch = max(-max_step, min(max_step, delta_pitch))

        new_yaw = self.yaw + delta_yaw
        new_pitch = self.pitch + delta_pitch

        limits = self.cfg.safety.angle_limits_rad
        lo, hi = limits["shoulder_pan"]
        new_yaw = max(lo, min(hi, new_yaw))
        lo, hi = limits["shoulder_lift"]
        new_pitch = max(lo, min(hi, new_pitch))

        self.yaw = new_yaw
        self.pitch = new_pitch

        if not self.dry_run:
            home = self.cfg.calibration.home
            sign = self.cfg.calibration.sign
            yaw_tick = int(round(home["shoulder_pan"]
                                + sign["shoulder_pan"] * new_yaw * RAD2TICK))
            pitch_tick = int(round(home["shoulder_lift"]
                                + sign["shoulder_lift"] * new_pitch * RAD2TICK))
            self.bus.write("Goal_Position", "shoulder_pan",
                        yaw_tick, normalize=False)
            self.bus.write("Goal_Position", "shoulder_lift",
                        pitch_tick, normalize=False)
        return True
    def reset_ibvs_filter(self):
        """IBVS 미분 상태 초기화.

        TRACKING 새로 진입할 때 호출하면 깔끔하지만,
        호출 안 해도 derivative_reset_gap_sec 기반 자동 reset 됨.
        """
        self._prev_error_x = 0.0
        self._prev_error_y = 0.0
        self._de_x_ema = 0.0
        self._de_y_ema = 0.0
        self._prev_ibvs_t = None

    # ----- 격발 -----

    def fire(self):
        """격발 신호. 실제 격발은 별도 MCU 가 /omx/fire 토픽 받아 처리.

        OmxYoloNode 가 /omx/fire 토픽을 별도 publish 하므로,
        여기서는 cooldown UX 위한 짧은 대기만.
        """
        if self.dry_run:
            self._log("[dry-run] 격발 신호 시뮬레이션")
        else:
            self._log("격발 신호 발사 (외부 MCU 처리)")
        # time.sleep(0.5)

    # ----- 상태 조회 -----

    def read_joint_positions_rad(self):
        """현재 4 모터의 관절 각도 (rad) 읽기."""
        if self.dry_run or self.bus is None or not self.bus.is_connected:
            return {
                "shoulder_pan": self.yaw,
                "shoulder_lift": self.pitch,
                "elbow_flex": 0.0,
                "wrist_flex": 0.0,
            }
        ticks = self.bus.sync_read("Present_Position", normalize=False)
        home = self.cfg.calibration.home
        sign = self.cfg.calibration.sign
        result = {}
        for name in MOTOR_ORDER:
            result[name] = (ticks[name] - home[name]) / RAD2TICK / sign.get(name, 1)
        return result
