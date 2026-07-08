#!/usr/bin/env python3
"""Fire Node — /omx/fire 토픽 수신 시 GPIO HIGH 펄스 발사.

설계:
    yolo_node 가 FIRING state 에서 /omx/fire (Empty) 발행 →
    이 노드가 GPIO pin HIGH (지속시간 동안) → LOW

안전 기능:
    - 부팅 시 LOW (안전 기본값)
    - 종료 시 LOW 보장 (cleanup)
    - cooldown: 직전 격발 후 일정 시간 새 fire 무시 (연발 방지)
    - 명시적 disable 토픽 (/omx/fire_disable Bool)

토픽:
    Sub: /omx/fire          std_msgs/Empty   격발 신호
    Sub: /omx/fire_disable  std_msgs/Bool    true 면 격발 무시 (안전 잠금)
    Pub: /omx/fire_status   std_msgs/String  "armed", "firing", "cooldown", "disabled"

파라미터:
    pin:                31      Jetson GPIO BOARD 핀 번호
    fire_duration_sec:  0.7     HIGH 유지 시간
    cooldown_sec:       1.5     격발 후 다음 격발까지 최소 시간
    active_state:       "HIGH"  발사 시 핀 상태 (회로 따라 "LOW" 도 가능)
    start_disabled:     False   부팅 시 disabled 상태로 시작 (true 면 외부 enable 필요)
"""

from __future__ import annotations

import threading
import time

import rclpy
from rclpy.node import Node

from std_msgs.msg import Empty, Bool, String

try:
    import Jetson.GPIO as GPIO
    GPIO_AVAILABLE = True
except ImportError:
    GPIO_AVAILABLE = False


class FireNode(Node):
    def __init__(self):
        super().__init__('fire_node')

        # ---------- Parameters ----------
        self.declare_parameter('pin', 31)
        self.declare_parameter('fire_duration_sec', 1.5)
        self.declare_parameter('cooldown_sec', 3.0)
        self.declare_parameter('active_state', 'HIGH')
        self.declare_parameter('start_disabled', False)
        self.declare_parameter('dry_run', False)

        self.pin = int(self.get_parameter('pin').value)
        self.fire_duration = float(self.get_parameter('fire_duration_sec').value)
        self.cooldown_sec = float(self.get_parameter('cooldown_sec').value)
        active = str(self.get_parameter('active_state').value).upper()
        self.start_disabled = bool(self.get_parameter('start_disabled').value)
        self.dry_run = bool(self.get_parameter('dry_run').value)

        if active == 'HIGH':
            self.active_level = 1
            self.idle_level = 0
        elif active == 'LOW':
            self.active_level = 0
            self.idle_level = 1
        else:
            raise ValueError(f"active_state must be HIGH or LOW, got: {active}")

        # ---------- State ----------
        self.disabled = self.start_disabled
        self.is_firing = False
        self.last_fire_t = 0.0
        self.fire_count = 0
        self.lock = threading.Lock()

        # ---------- GPIO setup ----------
        if self.dry_run:
            self.get_logger().warn("[dry-run] GPIO 미사용, 시뮬레이션 모드")
        elif not GPIO_AVAILABLE:
            self.get_logger().error(
                "Jetson.GPIO 미설치. pip install Jetson.GPIO 또는 "
                "--ros-args -p dry_run:=true 로 시뮬레이션")
            raise RuntimeError("GPIO unavailable")
        else:
            GPIO.setmode(GPIO.BOARD)
            GPIO.setwarnings(False)
            # 부팅 시 idle 상태 (안전)
            initial = GPIO.HIGH if self.idle_level == 1 else GPIO.LOW
            GPIO.setup(self.pin, GPIO.OUT, initial=initial)

        # ---------- Pub/Sub ----------
        self.create_subscription(Empty, '/omx/fire', self.on_fire, 10)
        self.create_subscription(
            Bool, '/omx/fire_disable', self.on_disable, 10)
        self.create_subscription(
            Bool, '/omx/fire_diable', self.on_disable_alias, 10)
        self.pub_status = self.create_publisher(String, '/omx/fire_status', 10)

        # ---------- Status timer ----------
        self.create_timer(1.0, self.publish_status)

        self.get_logger().info("=" * 50)
        self.get_logger().info("Fire Node")
        self.get_logger().info("=" * 50)
        mode = "[dry-run]" if self.dry_run else ""
        self.get_logger().info(f"{mode} GPIO pin: {self.pin} (BOARD)")
        self.get_logger().info(
            f"Active: {active} ({self.fire_duration:.2f}s pulse)")
        self.get_logger().info(
            f"Cooldown: {self.cooldown_sec:.2f}s (연발 방지)")
        if self.disabled:
            self.get_logger().warn("Started in DISABLED state — "
                                   "send /omx/fire_disable {data: false} to arm")
        self.get_logger().info("=== Ready ===")

    # ----- Subscribers -----

    def on_fire(self, msg):
        """격발 명령 수신."""
        with self.lock:
            now = time.monotonic()

            if self.disabled:
                self.get_logger().warn(
                    "fire 신호 받았지만 DISABLED 상태 — 무시")
                return

            if self.is_firing:
                self.get_logger().warn(
                    "fire 신호 받았지만 이미 firing 중 — 무시")
                return

            since_last = now - self.last_fire_t
            if since_last < self.cooldown_sec:
                self.get_logger().warn(
                    f"fire 신호 받았지만 cooldown 중 "
                    f"({since_last:.2f}s/{self.cooldown_sec:.2f}s) — 무시")
                return

            self.is_firing = True

        # lock 밖에서 실제 격발 (블록킹)
        self._do_fire()

    def on_disable(self, msg: Bool):
        with self.lock:
            self.disabled = msg.data
        state = "DISABLED" if msg.data else "ARMED"
        self.get_logger().info(f"Fire {state}")

    def on_disable_alias(self, msg: Bool):
        self.get_logger().warn(
            "/omx/fire_diable is deprecated typo; use /omx/fire_disable")
        self.on_disable(msg)

    # ----- 실제 GPIO 동작 -----

    def _do_fire(self):
        self.fire_count += 1
        self.get_logger().info(
            f"🔥 FIRE #{self.fire_count} (pulse {self.fire_duration:.2f}s)")

        # 별도 thread 로 timing (메인 콜백 블록 안 함)
        threading.Thread(target=self._fire_pulse, daemon=True).start()

    def _fire_pulse(self):
        try:
            if not self.dry_run and GPIO_AVAILABLE:
                # HIGH (또는 active level)
                GPIO.output(self.pin,
                            GPIO.HIGH if self.active_level == 1 else GPIO.LOW)

            time.sleep(self.fire_duration)

            if not self.dry_run and GPIO_AVAILABLE:
                # LOW (idle)
                GPIO.output(self.pin,
                            GPIO.HIGH if self.idle_level == 1 else GPIO.LOW)

            with self.lock:
                self.is_firing = False
                self.last_fire_t = time.monotonic()

            self.get_logger().info(f"✓ FIRE #{self.fire_count} 완료")

        except Exception as e:
            self.get_logger().error(f"FIRE 중 에러: {e}")
            # 에러 발생해도 idle 로 복귀 시도
            try:
                if not self.dry_run and GPIO_AVAILABLE:
                    GPIO.output(self.pin,
                                GPIO.HIGH if self.idle_level == 1 else GPIO.LOW)
            except Exception:
                pass
            with self.lock:
                self.is_firing = False

    # ----- Status -----

    def publish_status(self):
        msg = String()
        if self.disabled:
            msg.data = "disabled"
        elif self.is_firing:
            msg.data = "firing"
        else:
            now = time.monotonic()
            if now - self.last_fire_t < self.cooldown_sec:
                msg.data = "cooldown"
            else:
                msg.data = "armed"
        self.pub_status.publish(msg)

    # ----- 종료 -----

    def destroy_node(self):
        """안전 정리: GPIO 를 idle 로 복귀시키고 cleanup."""
        try:
            if not self.dry_run and GPIO_AVAILABLE:
                GPIO.output(self.pin,
                            GPIO.HIGH if self.idle_level == 1 else GPIO.LOW)
                self.get_logger().info(f"GPIO {self.pin} → idle, cleanup")
                GPIO.cleanup()
        except Exception as e:
            print(f"GPIO cleanup 에러: {e}")
        super().destroy_node()


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Fire Node - GPIO 격발 제어")
    parser.add_argument("--dry-run", action="store_true",
                        help="GPIO 미사용 시뮬레이션")
    cli_args, ros_args = parser.parse_known_args()

    rclpy.init(args=ros_args)
    node = None
    try:
        # CLI flag 를 parameter 로 매핑
        if cli_args.dry_run:
            ros_args = (ros_args or []) + [
                '--ros-args', '-p', 'dry_run:=true']
            rclpy.shutdown()
            rclpy.init(args=ros_args)

        node = FireNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("\n중단됨.")
    except Exception as e:
        print(f"노드 에러: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
