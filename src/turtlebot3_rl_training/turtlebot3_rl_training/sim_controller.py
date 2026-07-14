import subprocess
import time
from typing import Optional

import rclpy
from rclpy.node import Node
from ros_gz_interfaces.srv import ControlWorld


class GazeboSimController:
    """
    Gazebo Sim world control helper.

    역할:
      - /world/<world>/control bridge 자동 실행
      - simulation pause 상태 유지
      - multi_step으로 Gazebo physics를 원하는 step 수만큼 전진

    핵심:
      RL step마다 wall-clock sleep을 기다리는 것이 아니라,
      Gazebo physics iteration을 명시적으로 진행한다.
    """

    def __init__(
        self,
        node: Node,
        control_service: str = "/world/default/control",
        auto_start_bridge: bool = True,
        service_wait_timeout_sec: float = 8.0,
    ):
        self.node = node
        self.control_service = control_service.strip() or "/world/default/control"
        self.auto_start_bridge = bool(auto_start_bridge)
        self.service_wait_timeout_sec = float(service_wait_timeout_sec)

        self.bridge_proc: Optional[subprocess.Popen] = None

        self.client = self.node.create_client(
            ControlWorld,
            self.control_service,
        )

        self.node.get_logger().info(f"World control service: {self.control_service}")

        if self.auto_start_bridge:
            self._ensure_bridge_process()

    def _ensure_bridge_process(self):
        if self.client.service_is_ready():
            self.node.get_logger().info(
                f"ControlWorld service already available: {self.control_service}"
            )
            return

        bridge_arg = f"{self.control_service}@ros_gz_interfaces/srv/ControlWorld"

        cmd = [
            "ros2",
            "run",
            "ros_gz_bridge",
            "parameter_bridge",
            bridge_arg,
        ]

        self.node.get_logger().warn(
            "ControlWorld service is not available. "
            "Starting ros_gz_bridge internally:\n" + " ".join(cmd)
        )

        self.bridge_proc = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            text=True,
        )

        start = time.time()

        while rclpy.ok() and time.time() - start < self.service_wait_timeout_sec:
            rclpy.spin_once(self.node, timeout_sec=0.05)

            if self.client.service_is_ready():
                self.node.get_logger().info(
                    f"ControlWorld service is now available: {self.control_service}"
                )
                return

            time.sleep(0.05)

        self.node.get_logger().error(
            "Failed to start ControlWorld bridge internally. "
            f"service={self.control_service}. "
            "Check world name, e.g. /world/default/control"
        )

    def wait_until_ready(self, timeout_sec: Optional[float] = None) -> bool:
        timeout = self.service_wait_timeout_sec if timeout_sec is None else timeout_sec

        if self.client.service_is_ready():
            return True

        ok = self.client.wait_for_service(timeout_sec=timeout)

        if not ok:
            self.node.get_logger().error(
                f"ControlWorld service not available: {self.control_service}"
            )
            return False

        return True

    def pause(self, pause: bool = True, timeout_sec: float = 2.0) -> bool:
        if not self.wait_until_ready():
            return False

        req = ControlWorld.Request()
        req.world_control.pause = bool(pause)

        return self._call(req, timeout_sec=timeout_sec)

    def step(self, num_steps: int, timeout_sec: float = 2.0) -> bool:
        """
        paused world를 num_steps만큼 전진시킨다.
        """
        if num_steps <= 0:
            return True

        if not self.wait_until_ready():
            return False

        req = ControlWorld.Request()
        # Keep the world paused after exactly this multi_step batch. If Gazebo
        # was accidentally left running, the first accepted control request must
        # clamp it back to lockstep mode before the next policy action.
        req.world_control.pause = True
        req.world_control.multi_step = int(num_steps)

        return self._call(req, timeout_sec=timeout_sec)

    def reset_world(self, mode: str = "all", timeout_sec: float = 3.0) -> bool:
        """Request Gazebo world reset through ControlWorld.

        mode:
          - "all": reset time + models (equivalent to GUI reset world)
          - "time_only": reset simulation time only
          - "model_only": reset model poses/velocities only
        """
        if not self.wait_until_ready():
            return False

        req = ControlWorld.Request()
        # Critical for RL lockstep: Gazebo's WorldControl reset request leaves
        # pause false unless explicitly set. That lets the world free-run during
        # reset/SLAM/readiness waits, so step_count no longer means fixed sim
        # time. Keep reset results paused.
        req.world_control.pause = True
        world_reset = getattr(req.world_control, "reset", None)
        if world_reset is None:
            self.node.get_logger().error(
                "ControlWorld request has no world_control.reset field; cannot reset world"
            )
            return False

        mode_norm = str(mode or "all").strip().lower()
        try:
            if mode_norm == "time_only":
                world_reset.time_only = True
            elif mode_norm == "model_only":
                world_reset.model_only = True
            else:
                world_reset.all = True
        except Exception as exc:
            self.node.get_logger().error(f"Failed to configure world reset request: {exc}")
            return False

        ok = self._call(req, timeout_sec=timeout_sec)
        if not ok:
            return False
        return self.pause(True, timeout_sec=min(max(float(timeout_sec), 0.25), 0.75))

    def _call(self, req: ControlWorld.Request, timeout_sec: float = 2.0) -> bool:
        future = self.client.call_async(req)

        start = time.time()

        while rclpy.ok() and time.time() - start < timeout_sec:
            rclpy.spin_once(self.node, timeout_sec=0.001)

            if future.done():
                try:
                    response = future.result()
                except Exception as exc:
                    self.node.get_logger().error(f"ControlWorld call failed: {exc}")
                    return False

                if not response.success:
                    self.node.get_logger().error("ControlWorld returned success=False")
                    return False

                return True

        self.node.get_logger().error(f"ControlWorld timeout: {self.control_service}")
        return False

    def close(self):
        if self.bridge_proc is None:
            return

        if self.bridge_proc.poll() is not None:
            return

        self.node.get_logger().info("Stopping internal ControlWorld bridge...")

        self.bridge_proc.terminate()

        try:
            self.bridge_proc.wait(timeout=2.0)
        except subprocess.TimeoutExpired:
            self.bridge_proc.kill()
            self.bridge_proc.wait(timeout=1.0)
