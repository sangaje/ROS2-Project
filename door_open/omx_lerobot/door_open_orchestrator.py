#!/usr/bin/env python3
"""
door_open_orchestrator.py
실행 위치: 10.10.14.95 (ububtu)

동작 순서:
  1. OMX 그리퍼 모터값 모니터링
  2. 그리퍼가 손잡이를 잡으면 (모터값 < 50) Vic Pinky에 후진 명령 발행
  3. 2초 후진 후 정지
  4. 그리퍼 오픈 → 종료

의존성:
  pip install dynamixel-sdk
  ROS2 jazzy + geometry_msgs
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import Bool

import time
from dynamixel_sdk import PortHandler, PacketHandler

# ── DYNAMIXEL 설정 ──────────────────────────────────────────
DEVICE_PORT      = '/dev/ttyACM0'   # OpenRB-150 보드 포트
BAUDRATE         = 1000000
PROTOCOL_VERSION = 2.0

# OMX-AI 모터 ID 구성 (OpenRB-150 경유)
# ID 11: 관절 1 (베이스)
# ID 12: 관절 2 (숄더)
# ID 13: 관절 3 (엘보)
# ID 14: 관절 4 (리스트 피치)
# ID 15: 관절 5 (리스트 롤)
# ID 16: 그리퍼
GRIPPER_ID       = 16

# Control Table 주소 (DYNAMIXEL-X 시리즈)
ADDR_PRESENT_LOAD  = 126   # Present Load  (2 byte)
ADDR_GOAL_POSITION = 116   # Goal Position (4 byte)
ADDR_TORQUE_ENABLE = 64    # Torque Enable (1 byte)

# 그리퍼 포지션 값
GRIPPER_OPEN_POS = 1900    # 열린 상태

# 파지 판단 임계값
# open: 57~59 / 파지 성공: 40 후반대 → 50 기준
GRIP_THRESHOLD = 50
# ─────────────────────────────────────────────────────────────


class DoorOpenOrchestrator(Node):

    def __init__(self):
        super().__init__('door_open_orchestrator')

        # Vic Pinky cmd_vel 퍼블리셔 (ROS_DOMAIN_ID=30 으로 .118에 전달)
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.status_pub  = self.create_publisher(Bool, '/door_open_status', 10)

        # DYNAMIXEL 초기화
        self.port_handler   = PortHandler(DEVICE_PORT)
        self.packet_handler = PacketHandler(PROTOCOL_VERSION)

        if not self.port_handler.openPort():
            self.get_logger().error(f'포트 열기 실패: {DEVICE_PORT}')
            raise RuntimeError('포트 열기 실패')

        if not self.port_handler.setBaudRate(BAUDRATE):
            self.get_logger().error('Baudrate 설정 실패')
            raise RuntimeError('Baudrate 설정 실패')

        self.get_logger().info('DYNAMIXEL 연결 완료')
        self._set_torque(True)

        # 상태 머신
        self.state = 'WAIT_GRIP'
        self.reverse_start_time = None

        # 100ms 주기 타이머
        self.timer = self.create_timer(0.1, self.run_state_machine)

    # ── DYNAMIXEL 유틸 ───────────────────────────────────────

    def _set_torque(self, enable: bool):
        val = 1 if enable else 0
        self.packet_handler.write1ByteTxRx(
            self.port_handler, GRIPPER_ID, ADDR_TORQUE_ENABLE, val)

    def _read_load(self) -> float:
        """그리퍼 모터 load 값 읽기 (0~100 스케일)"""
        val, result, error = self.packet_handler.read2ByteTxRx(
            self.port_handler, GRIPPER_ID, ADDR_PRESENT_LOAD)
        if result != 0:
            self.get_logger().warn(f'load 읽기 실패: {error}')
            return 999.0
        return (val & 0x3FF) / 10.0

    def _set_gripper(self, position: int):
        self.packet_handler.write4ByteTxRx(
            self.port_handler, GRIPPER_ID, ADDR_GOAL_POSITION, position)

    # ── cmd_vel ──────────────────────────────────────────────

    def _publish_reverse(self):
        msg = Twist()
        msg.linear.x  = -0.15
        msg.angular.z = 0.0
        self.cmd_vel_pub.publish(msg)

    def _publish_stop(self):
        self.cmd_vel_pub.publish(Twist())

    # ── 상태 머신 ────────────────────────────────────────────

    def run_state_machine(self):

        if self.state == 'WAIT_GRIP':
            load = self._read_load()
            self.get_logger().info(f'[WAIT_GRIP] load={load:.1f}', throttle_duration_sec=0.5)
            if load < GRIP_THRESHOLD:
                self.get_logger().info(f'파지 감지! load={load:.1f} → 후진 시작')
                self.reverse_start_time = time.time()
                self.state = 'REVERSING'

        elif self.state == 'REVERSING':
            elapsed = time.time() - self.reverse_start_time
            self._publish_reverse()
            self.get_logger().info(f'[REVERSING] {elapsed:.2f}s', throttle_duration_sec=0.3)
            if elapsed >= 2.0:
                self.get_logger().info('2초 후진 완료 → 정지')
                self._publish_stop()
                self.state = 'RELEASE_GRIP'

        elif self.state == 'RELEASE_GRIP':
            self.get_logger().info('그리퍼 오픈')
            self._set_gripper(GRIPPER_OPEN_POS)
            time.sleep(1.0)
            self.state = 'DONE'

        elif self.state == 'DONE':
            msg = Bool()
            msg.data = True
            self.status_pub.publish(msg)
            self.get_logger().info('문 열기 동작 완료')
            self.timer.cancel()

    def destroy_node(self):
        self._publish_stop()
        self._set_torque(False)
        self.port_handler.closePort()
        super().destroy_node()


def main():
    rclpy.init()
    node = DoorOpenOrchestrator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
