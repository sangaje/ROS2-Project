import math
import os
import signal
import subprocess
import time
from typing import Optional

import numpy as np
import rclpy
from geometry_msgs.msg import TwistStamped
from nav_msgs.msg import OccupancyGrid, Odometry
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import LaserScan
from std_srvs.srv import Empty

try:
    import tf2_ros
except Exception:  # pragma: no cover - ROS 환경 의존
    tf2_ros = None


class TurtleBot3RosInterface(Node):
    """
    ROS2 topic I/O 전담 클래스.

    구독:
      /scan
      /odom
      /clock
      /map                 # SLAM이 발행하는 OccupancyGrid. 없으면 None 유지.

    발행:
      /cmd_vel : geometry_msgs/msg/TwistStamped only

    핵심:
      - Twist는 사용하지 않는다.
      - Gazebo Sim world stepping 후 실제 sim time 또는 odom stamp가
        전진했는지 확인할 수 있도록 /clock, odom stamp barrier를 제공한다.
      - SLAM 사용 시 map->odom TF를 이용해 odom pose를 map frame pose로 변환한다.
    """

    def __init__(
        self,
        namespace: str = "",
        scan_topic: str = "scan",
        odom_topic: str = "odom",
        cmd_vel_topic: str = "cmd_vel",
        clock_topic: str = "/clock",
        map_topic: str = "/map",
        enable_tf: bool = True,
        auto_start_slam: bool = False,
        slam_launch_package: str = "slam_toolbox",
        slam_launch_file: str = "online_async_launch.py",
        slam_use_sim_time: bool = True,
        slam_reset_service: str = "/slam_toolbox/reset",
    ):
        super().__init__("turtlebot3_rl_interface")

        self.namespace = namespace.strip("/")

        self.scan_topic = self._topic(scan_topic)
        self.odom_topic = self._topic(odom_topic)
        self.cmd_vel_topic = self._topic(cmd_vel_topic)
        self.clock_topic = clock_topic
        self.map_topic = map_topic.strip() if map_topic is not None else ""
        self.auto_start_slam = bool(auto_start_slam)
        self.slam_launch_package = str(slam_launch_package).strip() or "slam_toolbox"
        self.slam_launch_file = str(slam_launch_file).strip() or "online_async_launch.py"
        self.slam_use_sim_time = bool(slam_use_sim_time)
        self.slam_reset_service = str(slam_reset_service).strip() or "/slam_toolbox/reset"
        self.slam_proc: Optional[subprocess.Popen] = None

        self.scan: Optional[LaserScan] = None
        self.odom: Optional[Odometry] = None
        self.clock: Optional[Clock] = None
        self.slam_map: Optional[OccupancyGrid] = None

        self.last_scan_time: Optional[float] = None
        self.last_odom_time: Optional[float] = None
        self.last_clock_time_sec: Optional[float] = None
        self.last_slam_map_time: Optional[float] = None

        self.tf_buffer = None
        self.tf_listener = None

        if enable_tf and tf2_ros is not None:
            self.tf_buffer = tf2_ros.Buffer()
            self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        elif enable_tf:
            self.get_logger().warn(
                "tf2_ros import failed. map-frame pose transform will be disabled."
            )

        self.cmd_pub = self.create_publisher(
            TwistStamped,
            self.cmd_vel_topic,
            10,
        )

        self.scan_sub = self.create_subscription(
            LaserScan,
            self.scan_topic,
            self._scan_callback,
            10,
        )

        self.odom_sub = self.create_subscription(
            Odometry,
            self.odom_topic,
            self._odom_callback,
            10,
        )

        self.clock_sub = self.create_subscription(
            Clock,
            self.clock_topic,
            self._clock_callback,
            10,
        )

        map_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )

        self.map_sub = None
        if self.map_topic:
            self.map_sub = self.create_subscription(
                OccupancyGrid,
                self.map_topic,
                self._slam_map_callback,
                map_qos,
            )

        self.get_logger().info(f"scan topic    : {self.scan_topic}")
        self.get_logger().info(f"odom topic    : {self.odom_topic}")
        self.get_logger().info(f"cmd_vel topic : {self.cmd_vel_topic}")
        self.get_logger().info(f"clock topic   : {self.clock_topic}")
        self.get_logger().info(f"slam map topic: {self.map_topic or '(disabled)'}")
        self.get_logger().info(f"auto SLAM    : {self.auto_start_slam}")
        self.get_logger().info("cmd msg type  : geometry_msgs/msg/TwistStamped")

    def _topic(self, name: str) -> str:
        name = name.strip("/")

        if self.namespace:
            return f"/{self.namespace}/{name}"

        return f"/{name}"

    @staticmethod
    def _stamp_to_sec(sec: int, nanosec: int) -> float:
        return float(sec) + float(nanosec) * 1e-9

    @staticmethod
    def _normalize_angle(angle: float) -> float:
        return math.atan2(math.sin(angle), math.cos(angle))

    @staticmethod
    def _yaw_from_quaternion_xyzw(
        x: float,
        y: float,
        z: float,
        w: float,
    ) -> float:
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        return math.atan2(siny_cosp, cosy_cosp)

    def _scan_callback(self, msg: LaserScan):
        self.scan = msg
        self.last_scan_time = time.time()

    def _odom_callback(self, msg: Odometry):
        self.odom = msg
        self.last_odom_time = time.time()

    def _clock_callback(self, msg: Clock):
        self.clock = msg
        self.last_clock_time_sec = self._stamp_to_sec(
            msg.clock.sec,
            msg.clock.nanosec,
        )

    def _slam_map_callback(self, msg: OccupancyGrid):
        self.slam_map = msg
        self.last_slam_map_time = time.time()

    def publish_cmd_vel(self, linear_x: float, angular_z: float):
        msg = TwistStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "base_link"
        msg.twist.linear.x = float(linear_x)
        msg.twist.angular.z = float(angular_z)
        self.cmd_pub.publish(msg)

    def stop_robot(self):
        self.publish_cmd_vel(0.0, 0.0)

    def spin_once(self, timeout_sec: float = 0.01):
        rclpy.spin_once(self, timeout_sec=timeout_sec)

    def spin_steps(self, num_spins: int = 20, timeout_sec: float = 0.001):
        for _ in range(num_spins):
            rclpy.spin_once(self, timeout_sec=timeout_sec)

    def spin_for(self, duration_sec: float):
        start = time.time()

        while time.time() - start < duration_sec:
            rclpy.spin_once(self, timeout_sec=0.01)

    def wait_for_sensor_ready(self, timeout_sec: float = 10.0) -> bool:
        start = time.time()

        while time.time() - start < timeout_sec:
            rclpy.spin_once(self, timeout_sec=0.05)

            if self.scan is not None and self.odom is not None:
                self.get_logger().info("Received /scan and /odom.")
                return True

        self.get_logger().error("Timeout while waiting for /scan and /odom.")
        return False

    def wait_for_slam_map_ready(self, timeout_sec: float = 5.0) -> bool:
        if not self.map_topic:
            return False

        start = time.time()

        while time.time() - start < timeout_sec:
            rclpy.spin_once(self, timeout_sec=0.05)

            if self.slam_map is not None:
                self.get_logger().info(f"Received SLAM map from {self.map_topic}.")
                return True

        self.get_logger().warn(
            f"Timeout while waiting for SLAM map topic={self.map_topic}. "
            "RL memory map will still work with LiDAR-only updates."
        )
        return False


    def ensure_slam_toolbox(self, timeout_sec: float = 8.0) -> bool:
        """
        SLAM /map을 별도 터미널에서 실행하지 않도록 slam_toolbox를 내부에서 실행한다.

        이미 /map을 받았거나 slam_toolbox 노드가 떠 있으면 중복 실행하지 않는다.
        이 객체가 직접 실행한 프로세스만 close()/restart에서 종료한다.
        """
        if not self.map_topic:
            return False

        if self.slam_map is not None:
            return True

        if self._slam_node_exists():
            self.get_logger().info("slam_toolbox node already exists. Not starting another one.")
            return True

        return self.start_slam_toolbox(timeout_sec=timeout_sec)

    def start_slam_toolbox(self, timeout_sec: float = 8.0) -> bool:
        if not self.map_topic:
            return False

        if self.slam_proc is not None and self.slam_proc.poll() is None:
            return True

        cmd = [
            "ros2",
            "launch",
            self.slam_launch_package,
            self.slam_launch_file,
            f"use_sim_time:={'true' if self.slam_use_sim_time else 'false'}",
        ]

        self.get_logger().warn("Starting SLAM internally: " + " ".join(cmd))

        try:
            self.slam_proc = subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                text=True,
                start_new_session=True,
            )
        except Exception as exc:
            self.get_logger().error(f"Failed to start slam_toolbox: {exc}")
            self.slam_proc = None
            return False

        start = time.time()
        while rclpy.ok() and time.time() - start < timeout_sec:
            rclpy.spin_once(self, timeout_sec=0.05)
            if self.slam_proc.poll() is not None:
                self.get_logger().error(
                    "slam_toolbox process exited early. "
                    "Try running the launch command manually once to see the error."
                )
                return False
            if self._slam_node_exists() or self.slam_map is not None:
                return True

        self.get_logger().warn(
            "slam_toolbox process was started, but no node/map was detected yet. "
            "It may still be waiting for /scan, /odom, or /tf."
        )
        return True

    def reset_slam_state(self):
        self.slam_map = None
        self.last_slam_map_time = None

    def reset_slam_mapping(
        self,
        timeout_sec: float = 8.0,
        allow_process_restart: bool = False,
        reset_service: Optional[str] = None,
    ) -> bool:
        """
        episode reset 때 SLAM /map 상태를 초기화한다.

        중요한 점:
          - Burger pose만 reset하면 slam_toolbox의 pose-graph/map은 남아 있다.
          - 따라서 /slam_toolbox/reset 같은 SLAM reset service를 호출해야 /map도 비워진다.
          - 기본적으로 프로세스 재시작은 하지 않는다. 서비스 reset이 실패한 경우에만
            allow_process_restart=True일 때 내부에서 시작한 slam_toolbox를 재시작한다.
        """
        if not self.map_topic:
            return False

        self.reset_slam_state()

        preferred = (reset_service or self.slam_reset_service or "").strip()
        service_candidates = self._unique_strings([
            preferred,
            "/slam_toolbox/reset",
            "/slam_toolbox/clear",
            "/slam_toolbox/clear_map",
        ])

        for service_name in service_candidates:
            if self._call_ros_service_auto(service_name, timeout_sec=min(timeout_sec, 4.0)):
                self.get_logger().info(f"Requested SLAM map reset via {service_name}")
                # old transient-local /map이 다시 들어오는 것을 줄이기 위해 내부 캐시를 한 번 더 비운다.
                self.reset_slam_state()
                return True

        if allow_process_restart and self.slam_proc is not None and self.slam_proc.poll() is None:
            self.get_logger().warn(
                "No usable SLAM reset service was available. Restarting internally started "
                "slam_toolbox because allow_process_restart=True."
            )
            self.stop_slam_toolbox()
            time.sleep(0.35)
            self.reset_slam_state()
            return self.start_slam_toolbox(timeout_sec=timeout_sec)

        self.get_logger().warn(
            "No usable SLAM reset service was available. SLAM /map may keep its old cells. "
            "RL memory/confidence maps will still be reset. Check: ros2 service list | grep slam"
        )
        return False

    def _call_ros_service_auto(self, service_name: str, timeout_sec: float = 4.0) -> bool:
        """
        service type을 런타임에 확인한 뒤 ros2 service call로 reset을 호출한다.
        slam_toolbox의 reset service는 환경에 따라 std_srvs/Empty가 아니라
        slam_toolbox/srv/Reset인 경우가 있어 rclpy Empty client만 쓰면 실패한다.
        """
        service_name = str(service_name).strip()
        if not service_name:
            return False

        service_type = self._get_service_type(service_name, timeout_sec=0.8)
        if not service_type:
            return False

        if service_type.endswith("slam_toolbox/srv/Reset"):
            request_payload = "{}"
        elif service_type.endswith("std_srvs/srv/Empty"):
            request_payload = "{}"
        else:
            # reset/clear 계열 이름이 아니면 의도치 않은 service call을 피한다.
            if not any(token in service_name for token in ("reset", "clear")):
                return False
            request_payload = "{}"

        cmd = [
            "ros2",
            "service",
            "call",
            service_name,
            service_type,
            request_payload,
        ]

        try:
            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=max(float(timeout_sec), 1.0),
            )
        except subprocess.TimeoutExpired:
            self.get_logger().warn(f"SLAM reset service timeout: {service_name} [{service_type}]")
            return False
        except Exception as exc:
            self.get_logger().warn(f"SLAM reset service call failed: {service_name}: {exc}")
            return False

        if result.returncode != 0:
            self.get_logger().warn(
                f"SLAM reset service returned non-zero: {service_name} [{service_type}] "
                f"stderr={result.stderr.strip()}"
            )
            return False

        return True

    def _get_service_type(self, service_name: str, timeout_sec: float = 0.8) -> Optional[str]:
        service_name = str(service_name).strip()
        if not service_name:
            return None

        try:
            result = subprocess.run(
                ["ros2", "service", "type", service_name],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=max(float(timeout_sec), 0.5),
            )
        except Exception:
            return None

        if result.returncode != 0:
            return None

        service_type = result.stdout.strip().splitlines()[0].strip() if result.stdout.strip() else ""
        return service_type or None

    # 이전 코드와의 호환용. std_srvs/Empty service만 직접 rclpy로 호출한다.
    def _call_empty_service(self, service_name: str, timeout_sec: float = 2.0) -> bool:
        service_name = str(service_name).strip()
        if not service_name:
            return False

        client = self.create_client(Empty, service_name)
        if not client.wait_for_service(timeout_sec=0.25):
            self.destroy_client(client)
            return False

        future = client.call_async(Empty.Request())
        start = time.time()

        while rclpy.ok() and time.time() - start < timeout_sec:
            rclpy.spin_once(self, timeout_sec=0.01)
            if future.done():
                try:
                    future.result()
                except Exception as exc:
                    self.get_logger().warn(f"Empty service failed: {service_name}: {exc}")
                    self.destroy_client(client)
                    return False
                self.destroy_client(client)
                return True

        self.get_logger().warn(f"Empty service timeout: {service_name}")
        self.destroy_client(client)
        return False

    @staticmethod
    def _unique_strings(values: list[str]) -> list[str]:
        seen = set()
        out = []
        for value in values:
            v = str(value).strip()
            if not v or v in seen:
                continue
            seen.add(v)
            out.append(v)
        return out

    def stop_slam_toolbox(self):
        if self.slam_proc is None:
            return

        if self.slam_proc.poll() is not None:
            self.slam_proc = None
            return

        self.get_logger().info("Stopping internally started slam_toolbox...")

        try:
            os.killpg(os.getpgid(self.slam_proc.pid), signal.SIGTERM)
        except Exception:
            self.slam_proc.terminate()

        try:
            self.slam_proc.wait(timeout=4.0)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(os.getpgid(self.slam_proc.pid), signal.SIGKILL)
            except Exception:
                self.slam_proc.kill()
            self.slam_proc.wait(timeout=2.0)

        self.slam_proc = None

    def _slam_node_exists(self) -> bool:
        try:
            completed = subprocess.run(
                ["ros2", "node", "list"],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                timeout=2.0,
                check=False,
            )
        except Exception:
            return False

        if completed.returncode != 0:
            return False

        for line in completed.stdout.splitlines():
            name = line.strip()
            if "slam_toolbox" in name or "async_slam_toolbox" in name:
                return True

        return False

    def get_sim_time_sec(self) -> Optional[float]:
        return self.last_clock_time_sec

    def get_odom_stamp_sec(self) -> Optional[float]:
        if self.odom is None:
            return None

        stamp = self.odom.header.stamp
        return self._stamp_to_sec(stamp.sec, stamp.nanosec)

    def wait_for_time_advance(
        self,
        start_sim_time_sec: Optional[float],
        start_odom_stamp_sec: Optional[float],
        target_delta_sec: float,
        timeout_wall_sec: float = 1.0,
    ) -> bool:
        """
        /world/control multi_step 이후 실제 시간이 전진했는지 확인한다.

        우선순위:
          1. /clock 기준 sim time 전진
          2. /odom header.stamp 기준 odom time 전진

        둘 다 없으면 stale observation 가능성이 있으므로 False 반환.
        """
        target_delta_sec = float(target_delta_sec)
        wall_start = time.time()

        has_sim_target = start_sim_time_sec is not None
        has_odom_target = start_odom_stamp_sec is not None

        while time.time() - wall_start < timeout_wall_sec:
            rclpy.spin_once(self, timeout_sec=0.001)

            now_sim = self.get_sim_time_sec()
            now_odom = self.get_odom_stamp_sec()

            if has_sim_target and now_sim is not None:
                if now_sim >= start_sim_time_sec + target_delta_sec:
                    return True

            if has_odom_target and now_odom is not None:
                if now_odom >= start_odom_stamp_sec + target_delta_sec:
                    return True

        return False

    def wait_for_new_sensor_frame(
        self,
        prev_scan_wall_time: Optional[float],
        prev_odom_wall_time: Optional[float],
        timeout_wall_sec: float = 0.5,
    ) -> bool:
        """
        scan 또는 odom callback이 새로 들어왔는지 wall-clock 기준으로 확인한다.
        sim time barrier 보조용이다.
        """
        wall_start = time.time()

        while time.time() - wall_start < timeout_wall_sec:
            rclpy.spin_once(self, timeout_sec=0.001)

            scan_updated = (
                prev_scan_wall_time is not None
                and self.last_scan_time is not None
                and self.last_scan_time > prev_scan_wall_time
            )

            odom_updated = (
                prev_odom_wall_time is not None
                and self.last_odom_time is not None
                and self.last_odom_time > prev_odom_wall_time
            )

            if scan_updated or odom_updated:
                return True

        return False

    def get_position(self, frame_id: Optional[str] = None) -> Optional[np.ndarray]:
        pose = self.get_pose2d(frame_id=frame_id)

        if pose is None:
            return None

        xy, _ = pose
        return xy

    def get_quaternion_xyzw(self) -> Optional[tuple[float, float, float, float]]:
        if self.odom is None:
            return None

        q = self.odom.pose.pose.orientation
        return float(q.x), float(q.y), float(q.z), float(q.w)

    def get_roll_pitch_yaw(self) -> Optional[tuple[float, float, float]]:
        quat = self.get_quaternion_xyzw()

        if quat is None:
            return None

        x, y, z, w = quat

        sinr_cosp = 2.0 * (w * x + y * z)
        cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
        roll = math.atan2(sinr_cosp, cosr_cosp)

        sinp = 2.0 * (w * y - z * x)
        sinp = max(-1.0, min(1.0, sinp))
        pitch = math.asin(sinp)

        yaw = self._yaw_from_quaternion_xyzw(x, y, z, w)

        return roll, pitch, yaw

    def get_yaw(self, frame_id: Optional[str] = None) -> Optional[float]:
        pose = self.get_pose2d(frame_id=frame_id)

        if pose is not None:
            _, yaw = pose
            return yaw

        rpy = self.get_roll_pitch_yaw()

        if rpy is None:
            return None

        return rpy[2]

    def get_pose2d(
        self,
        frame_id: Optional[str] = None,
    ) -> Optional[tuple[np.ndarray, float]]:
        """
        로봇의 2D pose를 반환한다.

        frame_id가 None이거나 odom frame과 같으면 odom pose를 그대로 반환한다.
        frame_id가 map이면 TF의 map->odom 변환을 적용한다.
        SLAM을 사용할 때는 frame_id="map"으로 호출하는 것이 맞다.
        """
        if self.odom is None:
            return None

        p = self.odom.pose.pose.position
        q = self.odom.pose.pose.orientation

        odom_xy = np.array([p.x, p.y], dtype=np.float32)
        odom_yaw = self._yaw_from_quaternion_xyzw(
            float(q.x),
            float(q.y),
            float(q.z),
            float(q.w),
        )

        source_frame = self.odom.header.frame_id.strip() or "odom"
        target_frame = frame_id.strip() if frame_id is not None else ""

        if not target_frame or target_frame == source_frame:
            return odom_xy, float(odom_yaw)

        transformed = self._transform_pose2d(
            xy=odom_xy,
            yaw=odom_yaw,
            source_frame=source_frame,
            target_frame=target_frame,
        )

        if transformed is None:
            return odom_xy, float(odom_yaw)

        return transformed

    def _transform_pose2d(
        self,
        xy: np.ndarray,
        yaw: float,
        source_frame: str,
        target_frame: str,
    ) -> Optional[tuple[np.ndarray, float]]:
        if self.tf_buffer is None:
            return None

        try:
            transform = self.tf_buffer.lookup_transform(
                target_frame,
                source_frame,
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.02),
            )
        except Exception:
            return None

        t = transform.transform.translation
        q = transform.transform.rotation
        tf_yaw = self._yaw_from_quaternion_xyzw(
            float(q.x),
            float(q.y),
            float(q.z),
            float(q.w),
        )

        c = math.cos(tf_yaw)
        s = math.sin(tf_yaw)

        x = float(xy[0])
        y = float(xy[1])

        out_x = float(t.x) + c * x - s * y
        out_y = float(t.y) + s * x + c * y
        out_yaw = self._normalize_angle(float(yaw) + tf_yaw)

        return np.array([out_x, out_y], dtype=np.float32), float(out_yaw)

    def is_fallen(
        self,
        max_abs_roll: float = 0.7,
        max_abs_pitch: float = 0.7,
    ) -> bool:
        rpy = self.get_roll_pitch_yaw()

        if rpy is None:
            return False

        roll, pitch, _ = rpy

        return abs(roll) > max_abs_roll or abs(pitch) > max_abs_pitch

    def close(self):
        self.stop_slam_toolbox()

    def destroy_node(self):
        self.close()
        return super().destroy_node()

