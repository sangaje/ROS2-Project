import math
import random
import re
import subprocess
import time
from dataclasses import dataclass
from typing import Optional

import rclpy
from geometry_msgs.msg import Pose
from ros_gz_interfaces.msg import Entity
from ros_gz_interfaces.srv import SetEntityPose


@dataclass(frozen=True)
class ResetPose:
    x: float
    y: float
    z: float
    yaw: float


class ResetManager:
    """
    TurtleBot3 pose reset manager.

    핵심 수정점:
      - 잘못된 기본 이름(turtlebot3_burger)을 매 reset마다 무작정 호출하지 않는다.
      - /model/<name>/... ROS topic, Gazebo pose/info, gz model --list에서 실제 model name을 먼저 찾는다.
      - 실패한 entity name은 blacklist하여 같은 실행 중 반복 호출하지 않는다.
      - ROS SetEntityPose가 실패하면 gz service /world/<world>/set_pose 직접 호출도 fallback으로 시도한다.

    결과:
      - 실제 entity 이름이 맞으면 Burger를 reset_x/reset_y 중앙으로 보낸다.
      - 못 찾으면 Gazebo error spam을 최소화하고 명확한 진단 로그만 남긴다.
    """

    def __init__(
        self,
        node,
        entity_name: str,
        set_pose_service: str = "/world/default/set_pose",
        reset_z: float = 0.05,
        auto_start_bridge: bool = True,
        service_wait_timeout_sec: float = 8.0,
        auto_discover_entity: bool = True,
    ):
        self.node = node
        self.entity_name = entity_name.strip() or "turtlebot3_burger"
        self.set_pose_service = set_pose_service.strip() or "/world/default/set_pose"
        self.reset_z = float(reset_z)
        self.auto_start_bridge = bool(auto_start_bridge)
        self.service_wait_timeout_sec = float(service_wait_timeout_sec)
        self.auto_discover_entity = bool(auto_discover_entity)

        self.bridge_proc: Optional[subprocess.Popen] = None
        self.failed_entity_names: set[str] = set()
        self.validated_entity_name: Optional[str] = None
        self.discovery_done = False

        self.last_requested_pose: Optional[ResetPose] = None
        self.last_actual_pose: Optional[ResetPose] = None
        self.last_reset_entity_name: Optional[str] = None

        self.node.get_logger().info(f"Reset service     : {self.set_pose_service}")
        self.node.get_logger().info(f"Reset entity hint : {self.entity_name}")
        self.node.get_logger().info(f"Reset z           : {self.reset_z}")

        self.client = self.node.create_client(
            SetEntityPose,
            self.set_pose_service,
        )

        if self.auto_start_bridge:
            self._ensure_bridge_process()

    def _ensure_bridge_process(self):
        if self.client.service_is_ready():
            self.node.get_logger().info(
                f"SetEntityPose service already available: {self.set_pose_service}"
            )
            return

        bridge_arg = f"{self.set_pose_service}@ros_gz_interfaces/srv/SetEntityPose"

        cmd = [
            "ros2",
            "run",
            "ros_gz_bridge",
            "parameter_bridge",
            bridge_arg,
        ]

        self.node.get_logger().warn(
            "SetEntityPose service is not available yet. "
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
                    f"SetEntityPose service is now available: {self.set_pose_service}"
                )
                return

            time.sleep(0.05)

        self.node.get_logger().error(
            "Failed to start SetEntityPose bridge internally. "
            f"service={self.set_pose_service}. "
            "Check world name. Example service: /world/default/set_pose"
        )

    @staticmethod
    def yaw_to_quaternion(yaw: float) -> tuple[float, float, float, float]:
        half = yaw * 0.5
        return 0.0, 0.0, math.sin(half), math.cos(half)

    def build_pose(self, reset_pose: ResetPose) -> Pose:
        pose = Pose()

        pose.position.x = float(reset_pose.x)
        pose.position.y = float(reset_pose.y)
        pose.position.z = float(reset_pose.z)

        qx, qy, qz, qw = self.yaw_to_quaternion(reset_pose.yaw)
        pose.orientation.x = qx
        pose.orientation.y = qy
        pose.orientation.z = qz
        pose.orientation.w = qw
        return pose

    def wait_until_ready(self, timeout_sec: Optional[float] = None) -> bool:
        timeout = self.service_wait_timeout_sec if timeout_sec is None else timeout_sec

        if self.client.service_is_ready():
            return True

        available = self.client.wait_for_service(timeout_sec=timeout)
        if not available:
            self.node.get_logger().error(
                f"SetEntityPose service is not available: {self.set_pose_service}"
            )
            return False
        return True

    def reset_center_pose(
        self,
        x: float = 0.0,
        y: float = 0.0,
        random_yaw: bool = False,
        fixed_yaw: float = 0.0,
    ) -> Optional[ResetPose]:
        """
        Burger를 episode 시작 좌표로 보낸다.
        기본값은 Gazebo world 중앙인 (0, 0)이다.
        """
        yaw = random.uniform(-math.pi, math.pi) if random_yaw else float(fixed_yaw)
        reset_pose = ResetPose(
            x=float(x),
            y=float(y),
            z=float(self.reset_z),
            yaw=float(yaw),
        )

        ok = self.reset_to_pose(reset_pose)
        if not ok:
            return None
        return reset_pose

    def reset_random_pose(
        self,
        candidates: list[tuple[float, float]],
        random_yaw: bool = True,
        fixed_yaw: float = 0.0,
    ) -> Optional[ResetPose]:
        if not candidates:
            raise ValueError("reset pose candidates must not be empty")

        x, y = random.choice(candidates)
        return self.reset_center_pose(
            x=float(x),
            y=float(y),
            random_yaw=random_yaw,
            fixed_yaw=fixed_yaw,
        )

    def reset_to_pose(
        self,
        reset_pose: ResetPose,
        timeout_sec: float = 3.0,
    ) -> bool:
        if not self.wait_until_ready(timeout_sec=self.service_wait_timeout_sec):
            return False

        self.last_requested_pose = reset_pose
        self.last_actual_pose = None
        self.last_reset_entity_name = None

        candidates = self._candidate_entity_names()
        candidates = [name for name in candidates if name not in self.failed_entity_names]

        if not candidates:
            self.node.get_logger().error(
                "No valid Gazebo robot model candidate was found for pose reset. "
                "Run these commands and pass the model name with --entity-name:\n"
                "  ros2 topic list | grep '^/model/'\n"
                "  gz model --list\n"
                "  timeout 2 gz topic -e -t /world/default/pose/info | grep 'name:'"
            )
            return False

        self.node.get_logger().info(
            "Pose reset candidate order: " + ", ".join(candidates)
        )

        for candidate in candidates:
            if self._reset_to_pose_with_entity_name(
                entity_name=candidate,
                reset_pose=reset_pose,
                timeout_sec=timeout_sec,
            ):
                if candidate != self.entity_name:
                    self.node.get_logger().warn(
                        "Gazebo entity auto-detected: "
                        f"'{self.entity_name}' -> '{candidate}'"
                    )
                self.entity_name = candidate
                self.validated_entity_name = candidate
                self.last_reset_entity_name = candidate
                return True

            # 같은 실행에서 같은 잘못된 이름을 계속 때리지 않는다.
            self.failed_entity_names.add(candidate)

        self.node.get_logger().error(
            "Failed to reset pose for all discovered Gazebo robot candidates: "
            + ", ".join(candidates)
            + ". Use --disable-pose-reset temporarily or pass the exact model name with --entity-name."
        )
        return False

    def _reset_to_pose_with_entity_name(
        self,
        entity_name: str,
        reset_pose: ResetPose,
        timeout_sec: float,
    ) -> bool:
        # 1) ROS bridge SetEntityPose 우선 시도.
        # wrong name일 때 Gazebo가 error를 찍으므로 entity type은 MODEL 하나만 시도한다.
        if self._reset_by_ros_service(entity_name, reset_pose, timeout_sec):
            return True

        # 2) bridge 변환 문제가 있을 때 Gazebo transport service 직접 시도.
        if self._reset_by_gz_service(entity_name, reset_pose):
            return True

        return False

    def _reset_by_ros_service(
        self,
        entity_name: str,
        reset_pose: ResetPose,
        timeout_sec: float,
    ) -> bool:
        req = SetEntityPose.Request()
        req.entity.name = entity_name
        req.entity.type = Entity.MODEL
        req.pose = self.build_pose(reset_pose)

        future = self.client.call_async(req)
        start = time.time()

        while rclpy.ok() and time.time() - start < timeout_sec:
            rclpy.spin_once(self.node, timeout_sec=0.01)

            if not future.done():
                continue

            try:
                response = future.result()
            except Exception as exc:
                self.node.get_logger().warn(
                    f"SetEntityPose call failed for entity='{entity_name}': {exc}"
                )
                return False

            if not response.success:
                self.node.get_logger().warn(
                    f"SetEntityPose success=False for entity='{entity_name}'"
                )
                return False

            if not self._verify_actual_gazebo_pose(entity_name, reset_pose):
                return False

            self.node.get_logger().info(
                f"Reset by ROS SetEntityPose: entity='{entity_name}' -> "
                f"requested=(x={reset_pose.x:.3f}, y={reset_pose.y:.3f}, "
                f"z={reset_pose.z:.3f}, yaw={reset_pose.yaw:.3f}), "
                f"actual={self._format_pose(self.last_actual_pose)}"
            )
            return True

        self.node.get_logger().warn(
            f"SetEntityPose timeout for entity='{entity_name}', service={self.set_pose_service}"
        )
        return False

    def _reset_by_gz_service(self, entity_name: str, reset_pose: ResetPose) -> bool:
        world = self._world_name_from_service()
        service = f"/world/{world}/set_pose"
        qx, qy, qz, qw = self.yaw_to_quaternion(reset_pose.yaw)

        req = (
            f'name: "{entity_name}" '
            f'position {{ x: {reset_pose.x:.9f} y: {reset_pose.y:.9f} z: {reset_pose.z:.9f} }} '
            f'orientation {{ x: {qx:.9f} y: {qy:.9f} z: {qz:.9f} w: {qw:.9f} }}'
        )

        commands = [
            [
                "gz",
                "service",
                "-s",
                service,
                "--reqtype",
                "gz.msgs.Pose",
                "--reptype",
                "gz.msgs.Boolean",
                "--timeout",
                "2000",
                "--req",
                req,
            ],
            [
                "ign",
                "service",
                "-s",
                service,
                "--reqtype",
                "ignition.msgs.Pose",
                "--reptype",
                "ignition.msgs.Boolean",
                "--timeout",
                "2000",
                "--req",
                req,
            ],
        ]

        for cmd in commands:
            try:
                completed = subprocess.run(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    timeout=3.0,
                    check=False,
                )
            except Exception:
                continue

            out = (completed.stdout or "") + "\n" + (completed.stderr or "")
            if completed.returncode == 0 and (
                "data: true" in out
                or "data:true" in out
                or "true" in out.lower()
                or "Boolean" in out
            ):
                if not self._verify_actual_gazebo_pose(entity_name, reset_pose):
                    return False

                self.node.get_logger().info(
                    f"Reset by Gazebo service: entity='{entity_name}' -> "
                    f"requested=(x={reset_pose.x:.3f}, y={reset_pose.y:.3f}, "
                    f"z={reset_pose.z:.3f}, yaw={reset_pose.yaw:.3f}), "
                    f"actual={self._format_pose(self.last_actual_pose)}"
                )
                return True

        return False

    def _candidate_entity_names(self) -> list[str]:
        """
        Gazebo pose/info에는 model뿐 아니라 link/base_scan/wheel/환경 모델까지 섞인다.
        이전 버전은 turtlebot3_world 같은 환경 모델을 turtle 키워드 때문에 먼저 잡을 수 있었다.
        여기서는 실제 로봇 모델 후보만 남기고, burger를 최우선으로 둔다.
        """
        raw_candidates: list[str] = []

        if self.validated_entity_name:
            raw_candidates.append(self.validated_entity_name)

        # 사용자가 burger처럼 명시한 이름은 최우선이다.
        # 단, 기본값 turtlebot3_burger는 실제 모델명이 아닐 수 있으므로 fallback으로 둔다.
        if self.entity_name and self.entity_name != "turtlebot3_burger":
            raw_candidates.append(self.entity_name)

        if self.auto_discover_entity:
            raw_candidates.extend(self._ros_model_topic_names())
            raw_candidates.extend(self._gz_model_list())
            raw_candidates.extend(self._gz_pose_info_model_names())

        if self.entity_name and self.entity_name == "turtlebot3_burger":
            raw_candidates.append(self.entity_name)

        filtered = []
        for name in self._unique_preserve_order([n for n in raw_candidates if n]):
            if self._is_bad_entity_candidate(name):
                continue
            if not self._looks_like_robot_model(name):
                continue
            filtered.append(name)

        filtered = self._unique_preserve_order(filtered)
        filtered.sort(key=self._entity_priority_score)
        return filtered

    @staticmethod
    def _looks_like_robot_model(name: str) -> bool:
        lower = name.lower()
        return any(token in lower for token in ("burger", "waffle", "turtlebot3", "tb3"))

    @staticmethod
    def _is_bad_entity_candidate(name: str) -> bool:
        lower = name.strip().lower()
        if not lower:
            return True
        if "::" in name:
            return True

        exact_bad = {
            "world",
            "default",
            "ground_plane",
            "sun",
            "link",
            "base_link",
            "base_footprint",
            "base_scan",
            "imu_link",
            "wheel_left_link",
            "wheel_right_link",
            "ros_symbol",
            "symbol",
        }
        if lower in exact_bad:
            return True

        # 환경/월드 모델 또는 링크/센서 이름을 로봇 모델로 오인하지 않는다.
        bad_substrings = (
            "world",
            "ground",
            "symbol",
            "base_link",
            "base_footprint",
            "base_scan",
            "imu",
            "wheel",
            "link",
            "sensor",
            "collision",
            "visual",
            "camera",
            "lidar",
            "scan",
        )
        if any(token in lower for token in bad_substrings):
            # turtlebot3_burger처럼 실제 모델명에 들어갈 수 있는 경우는 허용.
            if "burger" not in lower and "waffle" not in lower:
                return True
            if "world" in lower:
                return True

        return False

    def _entity_priority_score(self, name: str) -> int:
        lower = name.lower()
        if self.validated_entity_name and name == self.validated_entity_name:
            return -1000
        if self.entity_name and self.entity_name != "turtlebot3_burger" and name == self.entity_name:
            return -900
        if lower == "burger":
            return -800
        if lower.endswith("/burger"):
            return -790
        if "burger" in lower:
            return -700
        if "waffle" in lower:
            return -300
        if "turtlebot3" in lower or "tb3" in lower:
            return -200
        return 0

    def _verify_actual_gazebo_pose(
        self,
        entity_name: str,
        reset_pose: ResetPose,
        tolerance_xy: float = 0.08,
        timeout_sec: float = 0.8,
    ) -> bool:
        actual = self._wait_for_gazebo_pose(entity_name, timeout_sec=timeout_sec)

        # pose/info를 못 읽는 환경이면 service success를 신뢰하되, 로그에는 미검증으로 남긴다.
        if actual is None:
            self.last_actual_pose = None
            self.node.get_logger().warn(
                f"Could not verify actual Gazebo pose for entity='{entity_name}'. "
                "Accepting service success, but check /world/<world>/pose/info if the GUI looks wrong."
            )
            return True

        self.last_actual_pose = actual
        dx = abs(float(actual.x) - float(reset_pose.x))
        dy = abs(float(actual.y) - float(reset_pose.y))

        if dx <= tolerance_xy and dy <= tolerance_xy:
            return True

        self.node.get_logger().warn(
            f"Pose reset verification failed for entity='{entity_name}': "
            f"requested=(x={reset_pose.x:.3f}, y={reset_pose.y:.3f}, z={reset_pose.z:.3f}), "
            f"actual={self._format_pose(actual)}, "
            f"|dx|={dx:.3f}, |dy|={dy:.3f}. Trying next candidate."
        )
        return False

    def _wait_for_gazebo_pose(
        self,
        entity_name: str,
        timeout_sec: float = 0.8,
    ) -> Optional[ResetPose]:
        start = time.time()
        while time.time() - start < timeout_sec:
            pose = self._read_gazebo_pose(entity_name)
            if pose is not None:
                return pose
            time.sleep(0.05)
        return None

    def _read_gazebo_pose(self, entity_name: str) -> Optional[ResetPose]:
        world = self._world_name_from_service()
        topic = f"/world/{world}/pose/info"
        escaped = re.escape(entity_name)

        commands = [
            ["timeout", "1", "gz", "topic", "-e", "-t", topic],
            ["timeout", "1", "ign", "topic", "-e", "-t", topic],
        ]

        for cmd in commands:
            try:
                completed = subprocess.run(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL,
                    text=True,
                    timeout=2.0,
                    check=False,
                )
            except Exception:
                continue

            text = completed.stdout or ""
            if not text:
                continue

            # gz pose/info는 pose { name: "..." position { x: ... y: ... z: ... } } 반복 구조다.
            pattern = (
                r'name:\s*"' + escaped + r'"'
                r'.{0,2500}?position\s*\{\s*'
                r'x:\s*([-+0-9.eE]+)\s*'
                r'y:\s*([-+0-9.eE]+)\s*'
                r'z:\s*([-+0-9.eE]+)'
            )
            m = re.search(pattern, text, flags=re.DOTALL)
            if not m:
                continue

            try:
                x = float(m.group(1))
                y = float(m.group(2))
                z = float(m.group(3))
            except Exception:
                continue

            return ResetPose(x=x, y=y, z=z, yaw=0.0)

        return None

    @staticmethod
    def _format_pose(pose: Optional[ResetPose]) -> str:
        if pose is None:
            return "unverified"
        return f"(x={pose.x:.3f}, y={pose.y:.3f}, z={pose.z:.3f})"

    def _world_name_from_service(self) -> str:
        # /world/default/set_pose -> default
        parts = [p for p in self.set_pose_service.split("/") if p]
        if len(parts) >= 2 and parts[0] == "world":
            return parts[1]
        return "default"

    def _ros_model_topic_names(self) -> list[str]:
        try:
            completed = subprocess.run(
                ["ros2", "topic", "list"],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                timeout=2.0,
                check=False,
            )
        except Exception:
            return []

        if completed.returncode != 0:
            return []

        names = []
        for line in completed.stdout.splitlines():
            m = re.match(r"^/model/([^/]+)/", line.strip())
            if m:
                names.append(m.group(1))

        names = self._unique_preserve_order(names)
        if names:
            self.node.get_logger().info(
                f"Gazebo model candidates from ROS /model topics: {names}"
            )
        return names

    def _gz_model_list(self) -> list[str]:
        commands = [
            ["gz", "model", "--list"],
            ["ign", "model", "--list"],
        ]

        for cmd in commands:
            try:
                completed = subprocess.run(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL,
                    text=True,
                    timeout=2.0,
                    check=False,
                )
            except Exception:
                continue

            if completed.returncode != 0:
                continue

            names = []
            for line in completed.stdout.splitlines():
                stripped = line.strip().strip("- ")
                if not stripped:
                    continue
                if stripped.startswith("[") or stripped.lower().startswith("available"):
                    continue
                candidate = stripped.split()[0].strip()
                if candidate:
                    names.append(candidate)

            names = self._unique_preserve_order(names)
            if names:
                self.node.get_logger().info(
                    "Gazebo model candidates from command "
                    f"'{ ' '.join(cmd) }': {names}"
                )
                return names

        return []

    def _gz_pose_info_model_names(self) -> list[str]:
        world = self._world_name_from_service()
        topic = f"/world/{world}/pose/info"
        commands = [
            ["timeout", "2", "gz", "topic", "-e", "-t", topic],
            ["timeout", "2", "ign", "topic", "-e", "-t", topic],
        ]

        for cmd in commands:
            try:
                completed = subprocess.run(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL,
                    text=True,
                    timeout=3.0,
                    check=False,
                )
            except Exception:
                continue

            if not completed.stdout:
                continue

            raw_names = re.findall(r'name:\s*"([^"]+)"', completed.stdout)
            names = []
            for name in raw_names:
                n = name.strip()
                if not n:
                    continue
                lower = n.lower()
                if "::" in n:
                    continue
                if lower in {"world", "default", "ground_plane", "sun"}:
                    continue
                names.append(n)

            names = self._unique_preserve_order(names)
            if names:
                self.node.get_logger().info(
                    f"Gazebo pose/info model candidates from {topic}: {names}"
                )
                return names

        return []

    @staticmethod
    def _unique_preserve_order(values: list[str]) -> list[str]:
        seen = set()
        out = []
        for value in values:
            if value in seen:
                continue
            seen.add(value)
            out.append(value)
        return out

    def close(self):
        if self.bridge_proc is None:
            return

        if self.bridge_proc.poll() is not None:
            return

        self.node.get_logger().info("Stopping internal SetEntityPose bridge...")
        self.bridge_proc.terminate()

        try:
            self.bridge_proc.wait(timeout=2.0)
        except subprocess.TimeoutExpired:
            self.bridge_proc.kill()
            self.bridge_proc.wait(timeout=1.0)
