"""OMX hardware abstraction layer.

LeRobot 이 pip 로 설치되어 있다고 가정.
별도 경로 추가 트릭 없음.
"""

from __future__ import annotations

from lerobot.motors import Motor, MotorNormMode
from lerobot.motors.dynamixel import (
    DriveMode,
    DynamixelMotorsBus,
    OperatingMode,
)


# ===== 상수 =====
DEFAULT_PORT = "/dev/omx_follower"

MOTOR_ORDER = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    # "wrist_roll",
    # "gripper",
)
ARM_MOTORS = MOTOR_ORDER[:-1]
# GRIPPER_MOTOR = MOTOR_ORDER[-1]

MOTOR_SPEC = {
    "shoulder_pan":  (11, "xl430-w250"),
    "shoulder_lift": (12, "xl430-w250"),
    "elbow_flex":    (13, "xl430-w250"),
    "wrist_flex":    (14, "xl330-m288"),
    # "wrist_roll":    (15, "xl330-m288"),
    # "gripper":       (16, "xl330-m288"),
}


# ===== LeRobot 심볼 접근 =====
def get_dxl_symbols():
    """기존 코드 호환용. dict 로 반환."""
    return {
        "Motor": Motor,
        "MotorNormMode": MotorNormMode,
        "DriveMode": DriveMode,
        "DynamixelMotorsBus": DynamixelMotorsBus,
        "OperatingMode": OperatingMode,
    }


# ===== Bus 빌드 =====
def build_bus(port: str = DEFAULT_PORT) -> DynamixelMotorsBus:
    """모든 모터 포함한 DynamixelMotorsBus."""
    motors = {}
    for name in MOTOR_ORDER:
        motor_id, model = MOTOR_SPEC[name]
        # gripper 떼어냄 → 모든 모터 RANGE_M100_100 통일
        motors[name] = Motor(motor_id, model, MotorNormMode.RANGE_M100_100)
    return DynamixelMotorsBus(port=port, motors=motors)
