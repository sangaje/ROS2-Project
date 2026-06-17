import math
import os
import signal
import subprocess
import threading
import time
import tempfile
import re
import zlib
from pathlib import Path
from typing import Optional

import numpy as np
import rclpy
from geometry_msgs.msg import TransformStamped, TwistStamped
from nav_msgs.msg import OccupancyGrid, Odometry
from nav_msgs.srv import GetMap
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.parameter import Parameter
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


def _occupancy_grid_data_signature(msg: OccupancyGrid) -> tuple[int, int, int, int, int, int]:
    """Return a compact content signature for an OccupancyGrid.

    slam_toolbox can republish OccupancyGrid messages with unchanged metadata
    (same stamp, size, origin) while the occupancy data changes.  RViz renders
    those data changes, but a metadata-only de-duplication filter in the RL node
    drops them.  Include counts and a CRC32 of the grid payload so real map
    content updates are accepted.
    """
    try:
        data = np.asarray(msg.data, dtype=np.int16)
        total = int(data.size)
        if total <= 0:
            return (0, 0, 0, 0, 0, 0)
        unknown = int(np.count_nonzero(data < 0))
        free = int(np.count_nonzero((data >= 0) & (data < 50)))
        occupied = int(np.count_nonzero(data >= 50))
        known = int(total - unknown)
        # Convert signed occupancy values to uint8 for a stable CRC.
        payload = np.asarray(data & 0xFF, dtype=np.uint8).tobytes()
        crc = int(zlib.crc32(payload) & 0xFFFFFFFF)
        return (total, known, free, occupied, unknown, crc)
    except Exception:
        try:
            total = len(msg.data)
        except Exception:
            total = 0
        return (int(total), -1, -1, -1, -1, -1)


class _BackgroundMapSubscriber(Node):
    """
    Dedicated /map subscriber spun by its own executor thread.

    Why this exists:
      - The main TurtleBot3RosInterface is spun manually by the RL loop.
      - slam_toolbox publishes /map slowly and with environment-dependent QoS.
      - On the real robot, `ros2 topic hz /map` can see /map while the RL loop
        misses the callback long enough for the map-locked RL layers to fall back
        forever.

    This node receives /map independently and writes the latest OccupancyGrid into
    the parent interface. It also republishes a latched copy on an internal topic
    so RViz/diagnostics can verify that the RL process itself is seeing the map.
    """

    def __init__(self, parent: "TurtleBot3RosInterface", map_topic: str, relay_topic: str):
        super().__init__("turtlebot3_rl_map_mirror")
        self.parent = parent
        self.map_topic = str(map_topic or "/map")
        self.relay_topic = str(relay_topic or "/map_rl_internal")
        self._last_sig = None
        self._recv_count = 0
        self._last_parent_force_log = 0.0

        relay_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.relay_pub = self.create_publisher(OccupancyGrid, self.relay_topic, relay_qos)

        qoses = [
            ("reliable_transient", QoSProfile(
                history=HistoryPolicy.KEEP_LAST,
                depth=1,
                reliability=ReliabilityPolicy.RELIABLE,
                durability=DurabilityPolicy.TRANSIENT_LOCAL,
            )),
            ("reliable_volatile", QoSProfile(
                history=HistoryPolicy.KEEP_LAST,
                depth=10,
                reliability=ReliabilityPolicy.RELIABLE,
                durability=DurabilityPolicy.VOLATILE,
            )),
            ("best_effort_volatile", QoSProfile(
                history=HistoryPolicy.KEEP_LAST,
                depth=10,
                reliability=ReliabilityPolicy.BEST_EFFORT,
                durability=DurabilityPolicy.VOLATILE,
            )),
        ]
        self.subs = []
        for name, qos in qoses:
            self.subs.append(
                self.create_subscription(
                    OccupancyGrid,
                    self.map_topic,
                    lambda msg, n=name: self._cb(msg, n),
                    qos,
                )
            )
        self.get_logger().info(
            f"MAP_MIRROR_START | {self.map_topic} -> parent + {self.relay_topic} | multi-qos background executor"
        )

    @staticmethod
    def _sig(msg: OccupancyGrid):
        info = msg.info
        origin = info.origin
        q = origin.orientation
        data_sig = _occupancy_grid_data_signature(msg)
        return (
            int(msg.header.stamp.sec),
            int(msg.header.stamp.nanosec),
            str(msg.header.frame_id),
            int(info.width),
            int(info.height),
            float(info.resolution),
            float(origin.position.x),
            float(origin.position.y),
            float(origin.position.z),
            float(getattr(q, "x", 0.0)),
            float(getattr(q, "y", 0.0)),
            float(getattr(q, "z", 0.0)),
            float(getattr(q, "w", 1.0)),
            *data_sig,
        )

    def reset_dedupe(self):
        # Parent reset_slam_state() intentionally clears parent.slam_map on every
        # episode/reset.  slam_toolbox can then republish a map with the same
        # stamp/metadata signature, especially on slow /map update intervals.
        # If the mirror keeps its previous signature, it suppresses the only map
        # sample that could repopulate the parent.  Clear mirror de-duplication
        # whenever the parent clears its SLAM state.
        self._last_sig = None
        self._last_parent_force_log = 0.0

    def _cb(self, msg: OccupancyGrid, qos_name: str):
        sig = self._sig(msg)
        parent_missing = False
        try:
            parent_missing = getattr(self.parent, "slam_map", None) is None
        except Exception:
            parent_missing = False

        duplicate = sig == self._last_sig
        if duplicate and not parent_missing:
            return

        if not duplicate:
            self._last_sig = sig
            self._recv_count += 1

        # Always keep the diagnostic mirror topic alive.  More importantly, if the
        # parent has just reset its cached map, force one delivery even when the
        # OccupancyGrid metadata is identical to the previous map message.
        self.relay_pub.publish(msg)
        try:
            self.parent._slam_map_callback(
                msg,
                source=f"map_mirror:{qos_name}{':force_parent_empty' if parent_missing else ''}",
                force=bool(parent_missing),
            )
        except TypeError:
            self.parent._slam_map_callback(msg)
        except Exception as exc:
            self.get_logger().warn(f"MAP_MIRROR_PARENT_UPDATE_FAILED | {type(exc).__name__}: {exc}")

        now = time.time()
        if parent_missing and now - self._last_parent_force_log > 2.0:
            self._last_parent_force_log = now
            self.get_logger().warn(
                "MAP_MIRROR_FORCE_PARENT_UPDATE | parent slam_map was empty; "
                "delivered latest /map even if metadata signature was unchanged"
            )

        if self._recv_count <= 3 or self._recv_count % 10 == 0 or parent_missing:
            info = msg.info
            try:
                _total, _known, _free, _occ, _unk, _crc = _occupancy_grid_data_signature(msg)
            except Exception:
                _known, _free, _occ, _unk, _crc = -1, -1, -1, -1, -1
            self.get_logger().info(
                f"MAP_MIRROR_RECEIVED | count={self._recv_count} qos={qos_name} "
                f"parent_missing={parent_missing} duplicate={duplicate} "
                f"frame={msg.header.frame_id or '(empty)'} size={info.width}x{info.height} "
                f"res={info.resolution:.3f} known={_known} free={_free} occ={_occ} unknown={_unk} crc={_crc} "
                f"relay={self.relay_topic}"
            )


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
        enable_cmd_vel_pub: bool = True,
        auto_start_slam: bool = False,
        slam_backend: str = "cartographer",
        slam_launch_package: str = "slam_toolbox",
        slam_launch_file: str = "online_async_launch.py",
        slam_use_sim_time: bool = True,
        slam_reset_service: str = "/slam_toolbox/reset",
        use_sim_time: bool = True,
    ):
        super().__init__("turtlebot3_rl_interface")

        self.use_sim_time_requested = bool(use_sim_time)
        self._configure_use_sim_time(self.use_sim_time_requested)

        self.namespace = namespace.strip("/")

        self.scan_topic = self._topic(scan_topic)
        self.odom_topic = self._topic(odom_topic)
        self.cmd_vel_topic = self._topic(cmd_vel_topic)
        self.clock_topic = clock_topic
        self.map_topic = map_topic.strip() if map_topic is not None else ""
        self.enable_cmd_vel_pub = bool(enable_cmd_vel_pub)
        self.auto_start_slam = bool(auto_start_slam)
        self.slam_backend = str(slam_backend or "cartographer").strip().lower()
        if self.slam_backend not in ("cartographer", "slam_toolbox"):
            self.slam_backend = "cartographer"
        self.slam_launch_package = str(slam_launch_package).strip() or "slam_toolbox"
        self.slam_launch_file = str(slam_launch_file).strip() or "online_async_launch.py"
        self.slam_use_sim_time = bool(slam_use_sim_time)
        default_reset = "" if self.slam_backend == "cartographer" else "/slam_toolbox/reset"
        self.slam_reset_service = str(slam_reset_service).strip() or default_reset
        self.slam_proc: Optional[subprocess.Popen] = None
        self.robot_state_publisher_proc: Optional[subprocess.Popen] = None

        # RViz RobotModel/Nav2 need a complete TF chain.  Some Gazebo launches
        # publish /odom but not odom->base_footprint on /tf, and some custom
        # runs omit robot_state_publisher.  Keep these guards enabled by default.
        self.odom_tf_fallback_enabled = True
        self.odom_tf_broadcaster = None
        self._odom_tf_fallback_logged = False

        self._slam_map_lock = threading.RLock()
        self._map_mirror_node = None
        self._map_mirror_executor = None
        self._map_mirror_thread = None
        self._map_mirror_stop_event = threading.Event()
        self.map_mirror_topic = "/map_rl_internal"
        self._slam_map_service_candidates = []
        self._last_slam_service_fetch_log = 0.0
        self._last_slam_service_fetch_attempt = 0.0
        # v23.5: GetMap service calls must never spin this node while it is
        # already being driven by another executor/timer.  Use a short-lived
        # private node/executor for service fetches and avoid live-periodic
        # service polling during policy steps.
        self._slam_service_private_counter = 0
        self.slam_service_private_executor_enabled = True
        self.slam_live_service_poll_enabled = False
        # In real-robot auto-start mode we must not silently reuse a manually
        # launched slam_toolbox with the default raw /scan config.  That path
        # causes variable LaserRangeScan beam counts and stale maps.
        self.force_internal_slam_safe_config = bool(self.auto_start_slam and (not bool(self.slam_use_sim_time)))

        # Internal real-robot SLAM helpers.  The TurtleBot LDS scan sometimes
        # reports variable beam counts and the default slam_toolbox config is
        # tuned for a 20 m lidar.  For real robot tests we launch slam_toolbox
        # with a generated safe config and feed it a fixed-length scan topic.
        self.slam_fixed_scan_topic = "/scan_fixed"
        # 0 means: lock to the first real /scan beam count, then keep it fixed.
        # This avoids Karto errors when the raw scan count jitters by +/- beams.
        self.slam_fixed_scan_count = 0
        self.slam_real_safe_params_enabled = not bool(self.slam_use_sim_time)
        self._scan_fixed_pub = None
        self._scan_fixed_sub = None
        self._scan_fixed_angle_min = None
        self._scan_fixed_angle_max = None
        self._scan_fixed_angle_increment = None
        self._scan_fixed_warned = False
        self._scan_fixed_resample_logged = False
        self._generated_slam_params_file = None
        self._generated_cartographer_lua_file = None

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

        self.enable_tf = bool(enable_tf and tf2_ros is not None)
        if self.enable_tf:
            self._recreate_tf_listener()
        elif enable_tf:
            self.get_logger().warn(
                "tf2_ros import failed. map-frame pose transform will be disabled."
            )

        self.cmd_pub = None
        if self.enable_cmd_vel_pub:
            self.cmd_pub = self.create_publisher(
                TwistStamped,
                self.cmd_vel_topic,
                10,
            )

        # Real TurtleBot3 /scan is usually published with BEST_EFFORT reliability.
        # A RELIABLE subscription is incompatible with a BEST_EFFORT publisher,
        # which makes the policy hang while waiting for scan.  BEST_EFFORT
        # subscription is compatible with both BEST_EFFORT and RELIABLE publishers,
        # so use it for high-rate robot sensor streams in both Gazebo and real robot.
        sensor_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )

        self.scan_sub = self.create_subscription(
            LaserScan,
            self.scan_topic,
            self._scan_callback,
            sensor_qos,
        )

        self.odom_sub = self.create_subscription(
            Odometry,
            self.odom_topic,
            self._odom_callback,
            sensor_qos,
        )

        self.clock_sub = self.create_subscription(
            Clock,
            self.clock_topic,
            self._clock_callback,
            10,
        )

        # /map QoS differs across slam_toolbox builds and launches.
        # Some publish RELIABLE+TRANSIENT_LOCAL, others publish RELIABLE+VOLATILE.
        # A TRANSIENT_LOCAL-only subscriber is incompatible with a VOLATILE publisher
        # in ROS 2, which makes ros2 topic echo/hz show /map while this node never
        # receives it.  Subscribe with multiple compatible QoS profiles and let the
        # common callback de-duplicate by stamp/metadata.
        self.map_sub = None
        self.map_sub_transient = None
        self.map_sub_volatile = None
        self.map_sub_best_effort = None
        self._last_slam_map_sig = None
        self._slam_map_recv_count = 0
        if self.map_topic:
            map_qos_transient = QoSProfile(
                history=HistoryPolicy.KEEP_LAST,
                depth=1,
                reliability=ReliabilityPolicy.RELIABLE,
                durability=DurabilityPolicy.TRANSIENT_LOCAL,
            )
            map_qos_volatile = QoSProfile(
                history=HistoryPolicy.KEEP_LAST,
                depth=5,
                reliability=ReliabilityPolicy.RELIABLE,
                durability=DurabilityPolicy.VOLATILE,
            )
            map_qos_best_effort = QoSProfile(
                history=HistoryPolicy.KEEP_LAST,
                depth=5,
                reliability=ReliabilityPolicy.BEST_EFFORT,
                durability=DurabilityPolicy.VOLATILE,
            )
            self.map_sub_transient = self.create_subscription(
                OccupancyGrid,
                self.map_topic,
                self._slam_map_callback,
                map_qos_transient,
            )
            self.map_sub_volatile = self.create_subscription(
                OccupancyGrid,
                self.map_topic,
                self._slam_map_callback,
                map_qos_volatile,
            )
            self.map_sub_best_effort = self.create_subscription(
                OccupancyGrid,
                self.map_topic,
                self._slam_map_callback,
                map_qos_best_effort,
            )
            # Backward-compatible alias used only as a truthy handle by old code.
            self.map_sub = self.map_sub_volatile
            self._start_background_map_mirror()

        self._ensure_robot_state_publisher_guard()

        self.get_logger().info(f"scan topic    : {self.scan_topic}")
        self.get_logger().info(f"odom topic    : {self.odom_topic}")
        self.get_logger().info(f"cmd_vel topic : {self.cmd_vel_topic}")
        self.get_logger().info(f"clock topic   : {self.clock_topic}")
        self.get_logger().info(f"slam map topic: {self.map_topic or '(disabled)'}")
        self.get_logger().info(f"auto SLAM    : {self.auto_start_slam}")
        self.get_logger().warn(f"SLAM_BACKEND : {self.slam_backend} (/map source)")
        self.get_logger().info("sensor QoS   : BEST_EFFORT/VOLATILE for /scan and /odom")
        self._slam_map_service_candidates = self._make_slam_map_service_candidates()
        self.get_logger().info("map QoS      : multi-sub RELIABLE/TRANSIENT_LOCAL + RELIABLE/VOLATILE + BEST_EFFORT/VOLATILE")
        self.get_logger().info(f"map service  : fallback GetMap candidates={self._slam_map_service_candidates}")
        if self.enable_cmd_vel_pub:
            self.get_logger().info("cmd msg type  : geometry_msgs/msg/TwistStamped")
        else:
            self.get_logger().warn(
                "cmd_vel publisher disabled: Nav2 owns /cmd_vel in action_mode=nav2"
            )



    def _ros_node_list(self, timeout_sec: float = 1.0) -> list[str]:
        try:
            result = subprocess.run(
                ["ros2", "node", "list"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=max(float(timeout_sec), 0.5),
            )
        except Exception:
            return []
        if result.returncode != 0:
            return []
        return [line.strip() for line in result.stdout.splitlines() if line.strip()]

    def _ensure_robot_state_publisher_guard(self) -> bool:
        """Ensure base_footprint->base_link/... static TFs exist for RViz RobotModel.

        turtlebot3_gazebo usually starts this, but custom launches can expose
        /robot_description without publishing the fixed link transforms.  That
        produces RViz errors such as "No transform from base_link".  Starting a
        second RSP is avoided by checking current nodes first.
        """
        nodes = self._ros_node_list(timeout_sec=1.0)
        if any("robot_state_publisher" in n for n in nodes):
            return True

        cmd = [
            "ros2",
            "launch",
            "turtlebot3_bringup",
            "robot_state_publisher.launch.py",
            f"use_sim_time:={'true' if self.use_sim_time_requested else 'false'}",
        ]
        env = os.environ.copy()
        env.setdefault("TURTLEBOT3_MODEL", "burger")
        try:
            self.robot_state_publisher_proc = subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                text=True,
                env=env,
                start_new_session=True,
            )
            self.get_logger().warn(
                "TF_GUARD | robot_state_publisher was missing; started: " + " ".join(cmd)
            )
            return True
        except Exception as exc:
            self.get_logger().warn(f"TF_GUARD | failed to start robot_state_publisher: {exc}")
            self.robot_state_publisher_proc = None
            return False

    def _publish_odom_tf_fallback(self, msg: Odometry) -> None:
        """Publish odom->base_footprint from /odom when Gazebo does not provide TF.

        This is deliberately in-process so RViz RobotModel and Nav2 have the
        odom->base_footprint edge even with minimal Gazebo launches.  The
        transform is exactly the pose already carried by /odom.
        """
        if not self.odom_tf_fallback_enabled or tf2_ros is None:
            return
        if self.odom_tf_broadcaster is None:
            try:
                self.odom_tf_broadcaster = tf2_ros.TransformBroadcaster(self)
            except Exception as exc:
                self.get_logger().warn(f"TF_GUARD | failed to create odom TF broadcaster: {exc}")
                self.odom_tf_fallback_enabled = False
                return

        parent = (msg.header.frame_id or "odom").strip() or "odom"
        child = (msg.child_frame_id or "base_footprint").strip() or "base_footprint"
        if not child:
            child = "base_footprint"

        t = TransformStamped()
        t.header = msg.header
        t.header.frame_id = parent
        t.child_frame_id = child
        t.transform.translation.x = float(msg.pose.pose.position.x)
        t.transform.translation.y = float(msg.pose.pose.position.y)
        t.transform.translation.z = float(msg.pose.pose.position.z)
        t.transform.rotation = msg.pose.pose.orientation
        try:
            self.odom_tf_broadcaster.sendTransform(t)
            if not self._odom_tf_fallback_logged:
                self.get_logger().warn(f"TF_GUARD | odom TF fallback active: {parent} -> {child}")
                self._odom_tf_fallback_logged = True
        except Exception as exc:
            if not self._odom_tf_fallback_logged:
                self.get_logger().warn(f"TF_GUARD | odom TF fallback publish failed: {exc}")
                self._odom_tf_fallback_logged = True

    def _configure_use_sim_time(self, enabled: bool) -> None:
        """Force this node to use Gazebo /clock when the simulator uses sim time.

        Marker, Path, TF lookup, and Nav2 goal stamps must live in the same
        time domain as /tf.  Without this, RViz can report "No transform to
        fixed frame" even when the frame tree is correct, because the marker
        stamp is wall-clock time while Gazebo/SLAM TF is sim time.
        """
        enabled = bool(enabled)
        try:
            self.set_parameters([
                Parameter("use_sim_time", Parameter.Type.BOOL, enabled),
            ])
        except Exception:
            try:
                self.declare_parameter("use_sim_time", enabled)
            except Exception:
                pass
            try:
                self.set_parameters([
                    Parameter("use_sim_time", Parameter.Type.BOOL, enabled),
                ])
            except Exception as exc:
                self.get_logger().warn(f"Failed to set use_sim_time={enabled}: {exc}")
                return

        self.get_logger().info(f"use_sim_time: {enabled}")

    def _recreate_tf_listener(self) -> bool:
        if tf2_ros is None:
            self.tf_buffer = None
            self.tf_listener = None
            return False
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        return True

    def reset_tf_buffer(self) -> bool:
        """Drop cached TF transforms after Gazebo teleport/reset.

        This does not change RViz or slam_toolbox state. It only prevents this
        Python node from reading a stale map->odom transform immediately after
        a model teleport. If RViz fixed frame is `map`, slam_toolbox still owns
        map->odom and must be reset/restarted separately for visual map-frame
        origin reset.
        """
        if not self.enable_tf:
            return False
        ok = self._recreate_tf_listener()
        if ok:
            self.get_logger().info(
                "TF buffer was recreated after pose reset. "
                "This flushes local cached map->odom transforms only."
            )
        return ok

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
        self._publish_odom_tf_fallback(msg)

    def _clock_callback(self, msg: Clock):
        self.clock = msg
        self.last_clock_time_sec = self._stamp_to_sec(
            msg.clock.sec,
            msg.clock.nanosec,
        )

    def _slam_map_callback(self, msg: OccupancyGrid, source: str = "main", force: bool = False):
        # De-duplicate duplicate deliveries from the multi-QoS /map subscriptions.
        # force=True is used by the background map mirror after reset_slam_state()
        # when the parent cache is empty but slam_toolbox republishes an OccupancyGrid
        # with the same metadata/stamp signature as before.
        info = msg.info
        data_sig = _occupancy_grid_data_signature(msg)
        sig = (
            int(msg.header.stamp.sec),
            int(msg.header.stamp.nanosec),
            str(msg.header.frame_id),
            int(info.width),
            int(info.height),
            float(info.resolution),
            float(info.origin.position.x),
            float(info.origin.position.y),
            *data_sig,
        )
        with self._slam_map_lock:
            if sig == getattr(self, "_last_slam_map_sig", None) and not bool(force):
                return
            self._last_slam_map_sig = sig
            self.slam_map = msg
            self.last_slam_map_time = time.time()
            self._slam_map_recv_count = int(getattr(self, "_slam_map_recv_count", 0)) + 1
        if self._slam_map_recv_count <= 3 or self._slam_map_recv_count % 10 == 0:
            try:
                total, known, free, occupied, unknown, crc = _occupancy_grid_data_signature(msg)
                total = max(int(total), 1)
                known_ratio = float(known) / float(total) if known >= 0 else -1.0
            except Exception:
                total, known, free, occupied, unknown, crc = 1, -1, -1, -1, -1, -1
                known_ratio = -1.0
            self.get_logger().info(
                "SLAM_MAP_RECEIVED | "
                f"count={self._slam_map_recv_count} source={source} force={bool(force)} topic={self.map_topic} "
                f"frame={msg.header.frame_id or '(empty)'} "
                f"size={info.width}x{info.height} res={info.resolution:.3f} "
                f"origin=({info.origin.position.x:.2f},{info.origin.position.y:.2f}) "
                f"known={known} free={free} occ={occupied} unknown={unknown} ratio={known_ratio:.3f} crc={crc}"
            )

    def _make_slam_map_service_candidates(self) -> list[str]:
        """Return likely GetMap service names for the current SLAM backend."""
        if str(getattr(self, "slam_backend", "")).strip().lower() == "cartographer":
            # Cartographer ROS publishes /map through occupancy_grid_node; it does
            # not provide slam_toolbox's /dynamic_map service.  Polling those
            # services during Cartographer mode only adds misleading timeout logs.
            return []

        candidates = []
        reset = str(getattr(self, "slam_reset_service", "") or "").strip()
        if reset and "/" in reset:
            ns = reset.rsplit("/", 1)[0].strip()
            if ns:
                candidates.append(ns.rstrip("/") + "/dynamic_map")
        candidates.extend([
            "/slam_toolbox/dynamic_map",
            "/dynamic_map",
        ])
        # Preserve order, remove duplicates/empties.
        return self._unique_strings(candidates)

    def _try_fetch_slam_map_service(self, timeout_sec: float = 0.45, reason: str = "") -> bool:
        """Fetch SLAM map via nav_msgs/srv/GetMap without spinning this node.

        v23.5 fix:
          - Previous implementation called ``rclpy.spin_once(self)`` inside
            service polling.  When this method was invoked from a timer or from
            code already using an executor, rclpy raised
            ``RuntimeError: Executor is already spinning``.
          - It also hammered ``/slam_toolbox/dynamic_map`` during live policy
            updates, causing slam_toolbox response timeouts and making /map
            appear frozen.

        This implementation creates a short-lived private node and a private
        SingleThreadedExecutor for the GetMap request.  It is used only for
        startup/reset strict gates, not for live map refresh during policy steps.
        """
        if not self.map_topic:
            return False

        candidates = list(getattr(self, "_slam_map_service_candidates", None) or [])
        if not candidates:
            candidates = self._make_slam_map_service_candidates()
            self._slam_map_service_candidates = candidates
        if not candidates:
            return False

        total_timeout = max(float(timeout_sec), 0.05)
        per_service_timeout = max(min(total_timeout / max(len(candidates), 1), 0.45), 0.10)

        # Do not fire service calls at a very high rate.  The strict gate already
        # retries periodically; this guard prevents accidental nested callers from
        # flooding slam_toolbox.
        now = time.time()
        min_gap = 0.12
        if now - float(getattr(self, "_last_slam_service_fetch_attempt", 0.0)) < min_gap:
            return False
        self._last_slam_service_fetch_attempt = now

        for service_name in candidates:
            fetch_node = None
            executor = None
            try:
                self._slam_service_private_counter += 1
                fetch_node = rclpy.create_node(
                    f"tb3_rl_getmap_fetch_{os.getpid()}_{self._slam_service_private_counter}",
                    context=self.context,
                )
                executor = SingleThreadedExecutor(context=self.context)
                executor.add_node(fetch_node)

                client = fetch_node.create_client(GetMap, service_name)
                if not client.wait_for_service(timeout_sec=min(per_service_timeout, 0.20)):
                    continue

                future = client.call_async(GetMap.Request())
                start = time.time()
                while rclpy.ok(context=self.context) and time.time() - start < per_service_timeout:
                    executor.spin_once(timeout_sec=0.02)
                    if future.done():
                        try:
                            response = future.result()
                        except Exception as exc:
                            now = time.time()
                            if now - self._last_slam_service_fetch_log > 3.0:
                                self._last_slam_service_fetch_log = now
                                self.get_logger().warn(
                                    f"SLAM_MAP_SERVICE_FETCH_FAILED | service={service_name} "
                                    f"reason={type(exc).__name__}: {exc}"
                                )
                            break

                        msg = getattr(response, "map", None)
                        if msg is None or int(msg.info.width) <= 0 or int(msg.info.height) <= 0:
                            now = time.time()
                            if now - self._last_slam_service_fetch_log > 3.0:
                                self._last_slam_service_fetch_log = now
                                self.get_logger().warn(
                                    f"SLAM_MAP_SERVICE_FETCH_EMPTY | service={service_name} reason={reason}"
                                )
                            break

                        self._slam_map_callback(
                            msg,
                            source=f"service:{service_name}{':' + reason if reason else ''}",
                            force=True,
                        )
                        info = msg.info
                        try:
                            total, known, free, occupied, unknown, crc = _occupancy_grid_data_signature(msg)
                            ratio = float(known) / float(max(total, 1))
                        except Exception:
                            known, ratio, free, occupied, unknown, crc = -1, -1.0, -1, -1, -1, -1
                        self.get_logger().warn(
                            f"SLAM_MAP_SERVICE_FETCH_OK | service={service_name} reason={reason} "
                            f"frame={msg.header.frame_id or '(empty)'} size={info.width}x{info.height} "
                            f"res={info.resolution:.3f} known={known} ratio={ratio:.3f} crc={crc}"
                        )
                        return True

                now = time.time()
                if now - self._last_slam_service_fetch_log > 5.0:
                    self._last_slam_service_fetch_log = now
                    self.get_logger().warn(
                        f"SLAM_MAP_SERVICE_FETCH_TIMEOUT | service={service_name} reason={reason} "
                        f"timeout={per_service_timeout:.2f}s"
                    )
            except Exception as exc:
                now = time.time()
                if now - self._last_slam_service_fetch_log > 3.0:
                    self._last_slam_service_fetch_log = now
                    self.get_logger().warn(
                        f"SLAM_MAP_SERVICE_FETCH_ERROR | service={service_name} "
                        f"reason={type(exc).__name__}: {exc}"
                    )
            finally:
                if executor is not None and fetch_node is not None:
                    try:
                        executor.remove_node(fetch_node)
                    except Exception:
                        pass
                if executor is not None:
                    try:
                        executor.shutdown()
                    except Exception:
                        pass
                if fetch_node is not None:
                    try:
                        fetch_node.destroy_node()
                    except Exception:
                        pass

        return False

    def publish_cmd_vel(self, linear_x: float, angular_z: float):
        if self.cmd_pub is None:
            return
        msg = TwistStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "base_link"
        msg.twist.linear.x = float(linear_x)
        msg.twist.angular.z = float(angular_z)
        self.cmd_pub.publish(msg)

    def stop_robot(self):
        self.publish_cmd_vel(0.0, 0.0)

    def _start_background_map_mirror(self):
        """Start a dedicated /map subscriber in a separate executor thread."""
        if not self.map_topic:
            return
        if self._map_mirror_node is not None:
            return
        try:
            relay_topic = "/map_rl_internal"
            self.map_mirror_topic = relay_topic
            self._map_mirror_node = _BackgroundMapSubscriber(self, self.map_topic, relay_topic)
            self._map_mirror_executor = SingleThreadedExecutor()
            self._map_mirror_executor.add_node(self._map_mirror_node)
            self._map_mirror_stop_event.clear()

            def _spin_loop():
                while not self._map_mirror_stop_event.is_set() and rclpy.ok():
                    try:
                        self._map_mirror_executor.spin_once(timeout_sec=0.10)
                    except Exception:
                        time.sleep(0.05)

            self._map_mirror_thread = threading.Thread(
                target=_spin_loop,
                name="turtlebot3_rl_map_mirror_spin",
                daemon=True,
            )
            self._map_mirror_thread.start()
            self.get_logger().info(
                f"map mirror   : background multi-QoS subscriber active, relay={relay_topic}"
            )
        except Exception as exc:
            self.get_logger().warn(
                f"MAP_MIRROR_START_FAILED | {type(exc).__name__}: {exc}. Falling back to main-node /map subscriber only."
            )
            self._map_mirror_node = None
            self._map_mirror_executor = None
            self._map_mirror_thread = None

    def _stop_background_map_mirror(self):
        self._map_mirror_stop_event.set()
        try:
            if self._map_mirror_executor is not None:
                self._map_mirror_executor.wake()
        except Exception:
            pass
        try:
            if self._map_mirror_thread is not None and self._map_mirror_thread.is_alive():
                self._map_mirror_thread.join(timeout=1.0)
        except Exception:
            pass
        try:
            if self._map_mirror_executor is not None and self._map_mirror_node is not None:
                self._map_mirror_executor.remove_node(self._map_mirror_node)
        except Exception:
            pass
        try:
            if self._map_mirror_node is not None:
                self._map_mirror_node.destroy_node()
        except Exception:
            pass
        try:
            if self._map_mirror_executor is not None:
                self._map_mirror_executor.shutdown(timeout_sec=0.2)
        except Exception:
            pass
        self._map_mirror_node = None
        self._map_mirror_executor = None
        self._map_mirror_thread = None

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

        # First, try the GetMap service immediately.  This avoids waiting for a
        # slow /map topic publication when slam_toolbox already has a valid map.
        if self.slam_map is None:
            self._try_fetch_slam_map_service(timeout_sec=0.55, reason="wait_start")

        start = time.time()
        last_service_try = 0.0

        while time.time() - start < timeout_sec:
            rclpy.spin_once(self, timeout_sec=0.05)

            if self.slam_map is not None:
                self.get_logger().info(f"Received SLAM map from {self.map_topic}.")
                return True

            now = time.time()
            if now - last_service_try >= 0.75:
                last_service_try = now
                if self._try_fetch_slam_map_service(timeout_sec=0.45, reason="wait_loop"):
                    self.get_logger().info(f"Received SLAM map via GetMap service for {self.map_topic}.")
                    return True

        # One last hard service pull before declaring fallback mode.
        if self.slam_map is None and self._try_fetch_slam_map_service(timeout_sec=0.8, reason="wait_timeout_final"):
            self.get_logger().info(f"Received SLAM map via GetMap service after topic wait timeout for {self.map_topic}.")
            return True

        self.get_logger().warn(
            f"Timeout while waiting for SLAM map topic={self.map_topic} and GetMap service candidates={self._slam_map_service_candidates}. "
            "RL memory map will still work with LiDAR-only updates."
        )
        return False



    def _ensure_scan_fixed_relay(self) -> str:
        """Start an in-process /scan -> /scan_fixed relay for slam_toolbox.

        slam_toolbox/Karto is sensitive to LaserScan beam-count changes.  The
        relay pads/truncates the real scan to a stable beam count while keeping
        the original header/frame/timestamps.  It is only for SLAM; the policy
        can still consume the raw /scan.
        """
        if self._scan_fixed_pub is None:
            pub_qos = QoSProfile(
                history=HistoryPolicy.KEEP_LAST,
                depth=10,
                reliability=ReliabilityPolicy.RELIABLE,
                durability=DurabilityPolicy.VOLATILE,
            )
            self._scan_fixed_pub = self.create_publisher(
                LaserScan, self.slam_fixed_scan_topic, pub_qos
            )

        if self._scan_fixed_sub is None:
            sub_qos = QoSProfile(
                history=HistoryPolicy.KEEP_LAST,
                depth=10,
                reliability=ReliabilityPolicy.BEST_EFFORT,
                durability=DurabilityPolicy.VOLATILE,
            )
            self._scan_fixed_sub = self.create_subscription(
                LaserScan, self.scan_topic, self._scan_fixed_callback, sub_qos
            )
            self.get_logger().warn(
                "INTERNAL_SCAN_FIXED_RELAY_ACTIVE | "
                f"{self.scan_topic} -> {self.slam_fixed_scan_topic} | "
                f"fixed_count=auto_first_scan(current={self.slam_fixed_scan_count}) | used only by internally launched SLAM"
            )
        return self.slam_fixed_scan_topic

    def _scan_fixed_callback(self, msg: LaserScan):
        if self._scan_fixed_pub is None:
            return

        in_count = len(msg.ranges)
        if in_count <= 0 or not math.isfinite(float(msg.angle_increment)) or abs(float(msg.angle_increment)) < 1.0e-9:
            return

        if int(getattr(self, "slam_fixed_scan_count", 0)) <= 0:
            # Lock to the first scan geometry seen by the relay.  slam_toolbox/Karto
            # registers the laser once and then expects all future scans for that
            # sensor to have the same ray count and angular geometry.
            self.slam_fixed_scan_count = int(max(1, in_count))
            self._scan_fixed_angle_min = float(msg.angle_min)
            # Important: use the actual range vector length, not msg.angle_max.
            # Some LDS/driver paths publish angle_max as angle_min + N*inc, while
            # others use angle_min + (N-1)*inc.  Karto interprets the metadata
            # inclusively; inconsistent metadata causes errors such as
            # "LaserRangeScan contains 245 range readings, expected 246".
            self._scan_fixed_angle_increment = float(msg.angle_increment)
            self._scan_fixed_angle_max = (
                float(self._scan_fixed_angle_min)
                + float(self._scan_fixed_angle_increment) * float(self.slam_fixed_scan_count - 1)
            )
            self.get_logger().warn(
                "INTERNAL_SCAN_FIXED_RELAY_RESAMPLE_LOCK | "
                f"fixed_count={self.slam_fixed_scan_count} "
                f"angle_min={float(self._scan_fixed_angle_min):.6f} "
                f"angle_max={float(self._scan_fixed_angle_max):.6f} "
                f"angle_increment={float(self._scan_fixed_angle_increment):.9f}"
            )

        fixed_count = int(max(1, self.slam_fixed_scan_count))
        out = LaserScan()
        out.header = msg.header
        out.angle_min = float(self._scan_fixed_angle_min)
        out.angle_increment = float(self._scan_fixed_angle_increment)
        out.angle_max = float(self._scan_fixed_angle_max)
        out.time_increment = msg.time_increment
        out.scan_time = msg.scan_time
        out.range_min = msg.range_min
        out.range_max = msg.range_max

        in_angle_min = float(msg.angle_min)
        in_inc = float(msg.angle_increment)
        in_ranges = list(msg.ranges)
        in_intensities = list(msg.intensities)

        # Resample into the locked angular grid.  Padding/truncation alone is not
        # enough when angle_min/angle_max/increment also jitter or when angle_max
        # has inclusive/exclusive convention drift.  Nearest-neighbor resampling
        # keeps every output scan geometrically identical for slam_toolbox.
        out_ranges = []
        out_intensities = [] if in_intensities else None
        for j in range(fixed_count):
            a = out.angle_min + out.angle_increment * float(j)
            idx = int(round((a - in_angle_min) / in_inc))
            if 0 <= idx < in_count:
                r = float(in_ranges[idx])
                if math.isfinite(r) and msg.range_min <= r <= msg.range_max:
                    out_ranges.append(r)
                else:
                    out_ranges.append(math.inf)
                if out_intensities is not None:
                    out_intensities.append(float(in_intensities[idx]) if idx < len(in_intensities) else 0.0)
            else:
                out_ranges.append(math.inf)
                if out_intensities is not None:
                    out_intensities.append(0.0)

        out.ranges = out_ranges
        out.intensities = out_intensities if out_intensities is not None else []

        expected_by_meta = int(round((float(msg.angle_max) - float(msg.angle_min)) / in_inc)) + 1
        geometry_changed = (
            in_count != fixed_count
            or expected_by_meta != in_count
            or abs(float(msg.angle_min) - out.angle_min) > 1.0e-6
            or abs(float(msg.angle_increment) - out.angle_increment) > 1.0e-9
        )
        if geometry_changed and not self._scan_fixed_warned:
            self._scan_fixed_warned = True
            self.get_logger().warn(
                "INTERNAL_SCAN_FIXED_RELAY_RESAMPLING_UNSTABLE_SCAN | "
                f"input_count={in_count} input_expected_by_meta={expected_by_meta} fixed={fixed_count} "
                f"input_angle_min={float(msg.angle_min):.6f} input_angle_max={float(msg.angle_max):.6f} "
                f"input_inc={in_inc:.9f}"
            )

        self._scan_fixed_pub.publish(out)

    @staticmethod
    def _set_yaml_param_text(text: str, key: str, value: str) -> str:
        pattern = rf"^(\s*{re.escape(key)}:\s*).*$"
        if re.search(pattern, text, flags=re.M):
            return re.sub(pattern, rf"\1{value}", text, flags=re.M)
        if "ros__parameters:\n" in text:
            return text.replace(
                "ros__parameters:\n",
                f"ros__parameters:\n    {key}: {value}\n",
                1,
            )
        return text + f"\n{key}: {value}\n"


    def _resolve_cartographer_configuration_files_dir(self) -> Optional[Path]:
        """Find the directory that contains Cartographer's common Lua includes.

        Cartographer resolves `include "map_builder.lua"` relative to
        `-configuration_directory`.  A generated config in /tmp without those
        include files makes cartographer_node exit immediately, while the
        occupancy_grid_node can keep the shell process alive.  That presents as
        "Cartographer started" but /map never appears.  Keep the generated
        TB3 config and the upstream common Lua files in the same temp dir.
        """
        candidate_dirs = []
        for pkg in ("cartographer_ros", "cartographer"):
            try:
                prefix = subprocess.check_output(
                    ["ros2", "pkg", "prefix", pkg],
                    text=True,
                    stderr=subprocess.DEVNULL,
                ).strip()
            except Exception:
                prefix = ""
            if prefix:
                base = Path(prefix) / "share" / pkg
                candidate_dirs.extend([
                    base / "configuration_files",
                    base / "config",
                ])

        # Common distro paths as a fallback when `ros2 pkg prefix` is slow or
        # unavailable inside an embedded training process.
        candidate_dirs.extend([
            Path("/opt/ros/jazzy/share/cartographer_ros/configuration_files"),
            Path("/opt/ros/jazzy/share/cartographer/configuration_files"),
            Path("/opt/ros/humble/share/cartographer_ros/configuration_files"),
            Path("/opt/ros/humble/share/cartographer/configuration_files"),
        ])

        for d in candidate_dirs:
            try:
                if (d / "map_builder.lua").exists() and (d / "trajectory_builder.lua").exists():
                    return d
            except Exception:
                pass
        return None

    def _make_internal_cartographer_lua_file(self, scan_topic: str) -> Optional[str]:
        """Generate a TurtleBot3 2D Cartographer config for RL mapping.

        Cartographer has no cheap per-episode map-reset service like slam_toolbox.
        For this project SLAM reset is mandatory at every reset, so Cartographer
        is reset by killing and restarting cartographer_node + occupancy_grid_node.
        """
        scan_topic = str(scan_topic or self.scan_topic).strip() or "/scan"
        lua = f"""
include "map_builder.lua"
include "trajectory_builder.lua"

options = {{
  map_builder = MAP_BUILDER,
  trajectory_builder = TRAJECTORY_BUILDER,
  map_frame = "map",
  tracking_frame = "base_scan",
  published_frame = "odom",
  odom_frame = "odom",
  provide_odom_frame = false,
  publish_frame_projected_to_2d = true,
  use_odometry = true,
  use_nav_sat = false,
  use_landmarks = false,
  num_laser_scans = 1,
  num_multi_echo_laser_scans = 0,
  num_subdivisions_per_laser_scan = 1,
  num_point_clouds = 0,
  lookup_transform_timeout_sec = 0.20,
  submap_publish_period_sec = 0.10,
  pose_publish_period_sec = 5e-3,
  trajectory_publish_period_sec = 30e-3,
  rangefinder_sampling_ratio = 1.0,
  odometry_sampling_ratio = 1.0,
  fixed_frame_pose_sampling_ratio = 1.0,
  imu_sampling_ratio = 0.0,
  landmarks_sampling_ratio = 1.0,
}}

MAP_BUILDER.use_trajectory_builder_2d = true
MAP_BUILDER.num_background_threads = 4

TRAJECTORY_BUILDER_2D.min_range = 0.12
TRAJECTORY_BUILDER_2D.max_range = 3.5
TRAJECTORY_BUILDER_2D.missing_data_ray_length = 3.0
TRAJECTORY_BUILDER_2D.use_imu_data = false
TRAJECTORY_BUILDER_2D.use_online_correlative_scan_matching = true
TRAJECTORY_BUILDER_2D.real_time_correlative_scan_matcher.linear_search_window = 0.10
TRAJECTORY_BUILDER_2D.real_time_correlative_scan_matcher.angular_search_window = math.rad(20.)
TRAJECTORY_BUILDER_2D.motion_filter.max_time_seconds = 0.15
TRAJECTORY_BUILDER_2D.motion_filter.max_distance_meters = 0.015
TRAJECTORY_BUILDER_2D.motion_filter.max_angle_radians = 0.006
TRAJECTORY_BUILDER_2D.ceres_scan_matcher.translation_weight = 10.
TRAJECTORY_BUILDER_2D.ceres_scan_matcher.rotation_weight = 40.

POSE_GRAPH.optimize_every_n_nodes = 20
POSE_GRAPH.constraint_builder.min_score = 0.55
POSE_GRAPH.constraint_builder.global_localization_min_score = 0.60
POSE_GRAPH.constraint_builder.sampling_ratio = 0.20
POSE_GRAPH.optimization_problem.huber_scale = 1e1

return options
"""
        include_dir = self._resolve_cartographer_configuration_files_dir()
        tmp_dir = Path(tempfile.mkdtemp(prefix="tb3_rl_cartographer_cfg_"))
        if include_dir is None:
            self.get_logger().error(
                "CARTOGRAPHER_CONFIG_INCLUDE_DIR_NOT_FOUND | install ros-jazzy-cartographer-ros "
                "and ros-jazzy-cartographer. Cannot find map_builder.lua/trajectory_builder.lua."
            )
            return None

        for name in ("map_builder.lua", "trajectory_builder.lua"):
            try:
                (tmp_dir / name).write_text((include_dir / name).read_text())
            except Exception as exc:
                self.get_logger().error(
                    "CARTOGRAPHER_CONFIG_INCLUDE_COPY_FAILED | "
                    f"include_dir={include_dir} name={name} error={exc}"
                )
                return None

        path = tmp_dir / "tb3_rl_cartographer.lua"
        path.write_text(lua)
        self._generated_cartographer_lua_file = str(path)
        self.get_logger().warn(
            "CARTOGRAPHER_CONFIG_WRITTEN | "
            f"file={path} config_dir={tmp_dir} include_src={include_dir} "
            f"scan_topic={scan_topic} use_sim_time={self.slam_use_sim_time} "
            "map_frame=map odom_frame=odom tracking_frame=base_scan"
        )
        return str(path)

    def _make_internal_slam_params_file(self, scan_topic: str) -> Optional[str]:
        """Generate a slam_toolbox params file encoded with real-robot-safe defaults."""
        try:
            prefix = subprocess.check_output(
                ["ros2", "pkg", "prefix", "slam_toolbox"],
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
            default_path = Path(prefix) / "share" / "slam_toolbox" / "config" / "mapper_params_online_async.yaml"
            text = default_path.read_text()
        except Exception as exc:
            self.get_logger().warn(
                "INTERNAL_SLAM_SAFE_CONFIG_DEFAULT_LOAD_FAILED | "
                f"using minimal params | error={exc}"
            )
            text = "slam_toolbox:\n  ros__parameters:\n"

        params = {
            "use_sim_time": "true" if self.slam_use_sim_time else "false",
            "scan_topic": str(scan_topic),
            "map_frame": "map",
            "odom_frame": "odom",
            "base_frame": "base_footprint",
            "mode": "mapping",
        }

        # For real robot runs, keep SLAM stable rather than aggressively fast.
        # Too-fast map publishing or processing every scan can fill the TF
        # message filter queue and make /map *slower* or unavailable.
        if self.slam_real_safe_params_enabled:
            params.update({
                "max_laser_range": "3.5",
                "map_update_interval": "0.75",
                "throttle_scans": "3",
                "minimum_time_interval": "0.15",
                "transform_timeout": "0.50",
                "tf_buffer_duration": "30.0",
                "transform_publish_period": "0.05",
            })
        else:
            # Gazebo can usually tolerate a faster map update.
            params.update({
                "map_update_interval": "0.5",
                "throttle_scans": "1",
                "transform_publish_period": "0.05",
            })

        for key, value in params.items():
            text = self._set_yaml_param_text(text, key, value)

        fd, path = tempfile.mkstemp(prefix="tb3_rl_slam_toolbox_", suffix=".yaml")
        with os.fdopen(fd, "w") as f:
            f.write(text)
        self._generated_slam_params_file = path
        self.get_logger().warn(
            "INTERNAL_SLAM_SAFE_CONFIG_WRITTEN | "
            f"file={path} scan_topic={scan_topic} real_safe={self.slam_real_safe_params_enabled} "
            "map_update_interval=0.75 throttle_scans=3 max_laser_range=3.5"
        )
        return path

    def ensure_slam_toolbox(self, timeout_sec: float = 8.0) -> bool:
        """Ensure the configured SLAM backend is running.

        Kept method name for backward compatibility with train_sac/eval_policy.
        In v25 the default backend is Cartographer for both training and real robot.
        """
        if not self.map_topic:
            return False

        if self.slam_backend == "cartographer":
            return self.ensure_cartographer(timeout_sec=timeout_sec)

        force_internal = bool(getattr(self, "force_internal_slam_safe_config", False))

        if self.slam_map is not None and not force_internal:
            return True

        if self._slam_node_exists():
            if force_internal and (self.slam_proc is None or self.slam_proc.poll() is not None):
                self.get_logger().warn(
                    "INTERNAL_SLAM_RESTART_EXTERNAL | existing slam_toolbox detected, "
                    "but real-robot RL requires the generated /scan_fixed safe config. "
                    "Stopping external slam_toolbox before internal start."
                )
                self._kill_external_slam_toolbox()
                self.reset_slam_state()
                time.sleep(1.0)
            else:
                self.get_logger().info("slam_toolbox node already exists. Not starting another one.")
                return True

        return self.start_slam_toolbox(timeout_sec=timeout_sec)

    def ensure_cartographer(self, timeout_sec: float = 8.0) -> bool:
        if not self.map_topic:
            return False

        if self.slam_map is not None and self._cartographer_node_exists():
            return True

        if self._cartographer_node_exists() and (self.slam_proc is None or self.slam_proc.poll() is not None):
            self.get_logger().warn(
                "CARTOGRAPHER_RESTART_EXTERNAL | existing Cartographer detected. "
                "Stopping it so RL owns the only SLAM backend and reset is deterministic."
            )
            self._kill_external_cartographer()
            self.reset_slam_state()
            time.sleep(0.5)

        return self.start_cartographer(timeout_sec=timeout_sec)

    def start_cartographer(self, timeout_sec: float = 8.0) -> bool:
        if not self.map_topic:
            return False

        if self.slam_proc is not None and self.slam_proc.poll() is None:
            return True

        # v25.2: prefer the official TurtleBot3 Cartographer launch.  The
        # generated bare cartographer_node path was too fragile: a process could
        # start but occupancy_grid_node never produced /map.  The official launch
        # uses the TurtleBot3-tested lua, node composition, and occupancy grid
        # wiring.  This is now the default for training and real eval.
        use_sim = "true" if self.slam_use_sim_time else "false"
        stamp_ms = int(time.time() * 1000)
        carto_log = f"/tmp/tb3_rl_turtlebot3_cartographer_launch_{stamp_ms}.log"
        grid_log = f"/tmp/tb3_rl_cartographer_fallback_{stamp_ms}.log"
        self._cartographer_last_log_files = (carto_log, grid_log)

        def _pkg_exists(pkg: str) -> bool:
            try:
                subprocess.check_output(["ros2", "pkg", "prefix", pkg], text=True, stderr=subprocess.DEVNULL)
                return True
            except Exception:
                return False

        prefer_generated = os.environ.get("TB3_RL_CARTOGRAPHER_GENERATED", "0").strip().lower() in ("1", "true", "yes", "on")
        use_official = (not prefer_generated) and _pkg_exists("turtlebot3_cartographer")

        if use_official:
            # Keep the scan_fixed relay alive for debugging, but the official TB3
            # launch consumes /scan.  Cartographer handles normal TB3 LaserScan
            # streams better than slam_toolbox, and this avoids launch-file
            # remapping incompatibilities across ROS2 distributions.
            try:
                self._ensure_scan_fixed_relay()
            except Exception:
                pass
            shell_cmd = (
                "set -e; "
                f"echo '[tb3_rl] turtlebot3_cartographer launch start' > {carto_log!r}; "
                "export TURTLEBOT3_MODEL=${TURTLEBOT3_MODEL:-burger}; "
                "ros2 launch turtlebot3_cartographer cartographer.launch.py "
                f"use_sim_time:={use_sim} "
                f">> {carto_log!r} 2>&1"
            )
            cmd = ["bash", "-lc", shell_cmd]
            self.get_logger().warn(
                "Starting TurtleBot3 Cartographer launch internally: "
                f"package=turtlebot3_cartographer use_sim_time={use_sim} log={carto_log}"
            )
        else:
            # Fallback for systems without turtlebot3_cartographer installed.
            # Use generated config, but match TB3 more closely by tracking the
            # laser frame when possible.
            slam_scan_topic = self._ensure_scan_fixed_relay()
            lua_path = self._make_internal_cartographer_lua_file(scan_topic=slam_scan_topic)
            if not lua_path:
                return False
            lua_dir = str(Path(lua_path).parent)
            lua_base = str(Path(lua_path).name)
            resolution = "0.05"
            publish_period = "0.15" if self.slam_use_sim_time else "0.30"
            shell_cmd = (
                "set -e; "
                f"echo '[tb3_rl] cartographer generated fallback start' > {grid_log!r}; "
                "ros2 run cartographer_ros cartographer_node "
                f"-configuration_directory {lua_dir!r} "
                f"-configuration_basename {lua_base!r} "
                f"--ros-args -p use_sim_time:={use_sim} -r scan:={slam_scan_topic} "
                f">> {grid_log!r} 2>&1 & "
                "ros2 run cartographer_ros occupancy_grid_node "
                f"-resolution {resolution} -publish_period_sec {publish_period} "
                f"--ros-args -p use_sim_time:={use_sim} "
                f">> {grid_log!r} 2>&1 & "
                "wait"
            )
            cmd = ["bash", "-lc", shell_cmd]
            self.get_logger().warn(
                "Starting generated Cartographer fallback internally: "
                f"scan={slam_scan_topic} config={lua_path} use_sim_time={use_sim} "
                f"occupancy_period={publish_period}s log={grid_log}"
            )

        try:
            self.slam_proc = subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                text=True,
                start_new_session=True,
            )
        except Exception as exc:
            self.get_logger().error(f"Failed to start Cartographer: {exc}")
            self.slam_proc = None
            return False

        start = time.time()
        node_seen = False
        while rclpy.ok() and time.time() - start < timeout_sec:
            rclpy.spin_once(self, timeout_sec=0.05)
            if self.slam_proc.poll() is not None:
                tail_msg = self._cartographer_logs_tail()
                self.get_logger().error(
                    "Cartographer process exited early. "
                    "Install ros-jazzy-turtlebot3-cartographer or inspect the log. " + tail_msg
                )
                return False
            node_seen = node_seen or self._cartographer_node_exists()
            # Do not return just because the node exists.  The old behavior did
            # that and allowed training to continue with no /map.  Wait until
            # at least one OccupancyGrid is actually received.
            if self.slam_map is not None:
                self.get_logger().warn(
                    "CARTOGRAPHER_MAP_READY | /map received from Cartographer "
                    f"after {time.time() - start:.2f}s node_seen={node_seen}"
                )
                return True

        tail_msg = self._cartographer_logs_tail()
        self.get_logger().warn(
            "CARTOGRAPHER_WAIT_NO_MAP | Cartographer process was started, but /map was not received yet. "
            "Check /scan, /scan_fixed, /odom, /tf, and the log files. " + tail_msg
        )
        return True

    def _cartographer_logs_tail(self) -> str:
        try:
            log_files = getattr(self, "_cartographer_last_log_files", None) or ()
            tails = []
            for lf in log_files:
                try:
                    lines = Path(lf).read_text(errors="replace").splitlines()[-16:]
                    if lines:
                        tails.append(f"{lf}:" + " | ".join(lines))
                except Exception:
                    pass
            if tails:
                return "logs_tail=[" + " || ".join(tails) + "]"
        except Exception:
            pass
        return "logs_tail=(none)"

    def start_slam_toolbox(self, timeout_sec: float = 8.0) -> bool:
        if not self.map_topic:
            return False

        if self.slam_proc is not None and self.slam_proc.poll() is None:
            return True

        slam_scan_topic = self.scan_topic
        slam_params_file = None
        if self.slam_real_safe_params_enabled:
            slam_scan_topic = self._ensure_scan_fixed_relay()
            slam_params_file = self._make_internal_slam_params_file(scan_topic=slam_scan_topic)

        cmd = [
            "ros2",
            "launch",
            self.slam_launch_package,
            self.slam_launch_file,
            f"use_sim_time:={'true' if self.slam_use_sim_time else 'false'}",
        ]
        if slam_params_file:
            cmd.append(f"slam_params_file:={slam_params_file}")

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
        with self._slam_map_lock:
            self.slam_map = None
            self.last_slam_map_time = None
            self._last_slam_map_sig = None
        try:
            mirror = getattr(self, "_map_mirror_node", None)
            if mirror is not None and hasattr(mirror, "reset_dedupe"):
                mirror.reset_dedupe()
        except Exception:
            pass

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

        if self.slam_backend == "cartographer":
            self.get_logger().warn(
                "MANDATORY_CARTOGRAPHER_RESET | Cartographer has no per-episode map reset service; "
                "restarting cartographer_node + occupancy_grid_node."
            )
            self.reset_slam_state()
            self.stop_slam_toolbox()
            self._kill_external_cartographer()
            time.sleep(0.35)
            ok = self.start_cartographer(timeout_sec=timeout_sec)
            if ok:
                self.wait_for_slam_map_ready(timeout_sec=max(float(timeout_sec), 1.0))
            return bool(ok)

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
                # Do not leave the episode in map-blind mode when slam_toolbox
                # already has a fresh map available through /dynamic_map.
                self._try_fetch_slam_map_service(timeout_sec=0.8, reason="after_reset_service")
                return True

        if allow_process_restart:
            self.get_logger().warn(
                "MANDATORY_SLAM_RESET | reset service unavailable; restarting slam_toolbox "
                "because stale /map is not allowed."
            )
            if self.slam_proc is not None and self.slam_proc.poll() is None:
                self.stop_slam_toolbox()
                time.sleep(0.35)
            self.reset_slam_state()
            ok = self.start_slam_toolbox(timeout_sec=timeout_sec)
            if ok:
                self.wait_for_slam_map_ready(timeout_sec=max(float(timeout_sec), 1.0))
            return bool(ok and self.slam_map is not None)

        self.get_logger().warn(
            "MANDATORY_SLAM_RESET_FAILED | no usable reset service and process restart disabled. "
            "Enable --restart-slam-on-reset or use the provided training command."
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

    def _kill_external_slam_toolbox(self):
        """Best-effort cleanup for manually launched slam_toolbox instances."""
        patterns = [
            "slam_toolbox",
            "async_slam_toolbox_node",
            "sync_slam_toolbox_node",
        ]
        for pat in patterns:
            try:
                subprocess.run(["pkill", "-f", pat], check=False, timeout=2.0)
            except Exception:
                pass
        try:
            self.get_logger().warn("INTERNAL_SLAM_EXTERNAL_KILL_SENT | pkill -f slam_toolbox")
        except Exception:
            pass

    def _kill_external_cartographer(self):
        """Best-effort cleanup for manually launched Cartographer instances."""
        patterns = [
            "cartographer_node",
            "occupancy_grid_node",
            "cartographer_ros",
            "turtlebot3_cartographer",
        ]
        for pat in patterns:
            try:
                subprocess.run(["pkill", "-f", pat], check=False, timeout=2.0)
            except Exception:
                pass
        try:
            self.get_logger().warn("CARTOGRAPHER_EXTERNAL_KILL_SENT | pkill -f cartographer_node/occupancy_grid_node/turtlebot3_cartographer")
        except Exception:
            pass

    def stop_slam_toolbox(self):
        if self.slam_proc is None:
            return

        if self.slam_proc.poll() is not None:
            self.slam_proc = None
            return

        self.get_logger().info(f"Stopping internally started SLAM backend: {self.slam_backend}...")

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
            if self.slam_backend == "cartographer":
                if "cartographer" in name or "occupancy_grid" in name:
                    return True
            else:
                if "slam_toolbox" in name or "async_slam_toolbox" in name:
                    return True

        return False

    def _cartographer_node_exists(self) -> bool:
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
            if "cartographer" in name or "occupancy_grid" in name:
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
            # Do NOT silently return odom coordinates when the caller asked for map.
            # Publishing map-frame markers or computing map-frame waypoints from
            # odom fallback coordinates makes RViz layers appear to slide apart.
            return None

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
        self._stop_background_map_mirror()
        self.stop_slam_toolbox()

    def destroy_node(self):
        self.close()
        return super().destroy_node()

