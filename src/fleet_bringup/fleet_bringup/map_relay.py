#!/usr/bin/env python3
"""
Receives /map_bridge from domain_bridge with a transient-local
subscription and republishes it on /map (transient_local). By default it
stands by until no other /map publisher is alive. In external-map mode it
relays each bridged update directly and avoids periodic full-map
republishing.

/map is meant to have exactly one live source at a time -- normally a local
SLAM node (e.g. Cartographer) when this robot owns its own mapping. This
relay stays silent while that primary is alive, detected via count_publishers() on the output topic (not
message content, so there is no self-feedback risk once this node starts
publishing too). If the primary disappears for longer than
takeover_grace_sec, the relay takes over: it continues from the primary's
own last published map if one was ever seen, or falls back to the latest
bridged map otherwise, so downstream AMCL/costmaps keep seeing a map instead
of losing it outright.

Stepping back down requires the primary to be seen present continuously for
standby_confirm_sec, not just one detection tick. A real robot's SLAM
process can drop out of DDS discovery for a couple of seconds under CPU
load or a brief restart; without this debounce, count_publishers() flips
low->high->low and the relay repeatedly takes over and stands back down,
so /map alternates between the relay's cached map and the primary's live
map on every blip. While a reappearance is still unconfirmed the relay
keeps serving its last output instead of republishing, so the two sources
never publish to /map at the same time.
"""
import rclpy
import hashlib
import time
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from nav_msgs.msg import OccupancyGrid
from std_msgs.msg import String


class MapRelay(Node):
    def __init__(self):
        super().__init__('map_relay')
        self.declare_parameter('input_topic', '/map_bridge')
        self.declare_parameter('output_topic', '/map')
        self.declare_parameter('check_period_sec', 1.0)
        self.declare_parameter('takeover_grace_sec', 2.0)
        self.declare_parameter('standby_confirm_sec', 2.0)
        self.declare_parameter('relay_without_primary', False)
        self.declare_parameter('max_publish_rate_hz', 1.0)
        self.declare_parameter('cached_republish_period_sec', 0.0)
        self.declare_parameter('active_scout_id_topic', '')
        self.declare_parameter('primary_scout_id', 'scout22')
        self.declare_parameter('follower_scout_id', 'follower21')
        self.declare_parameter('follower_input_topic', '')
        self.declare_parameter('robot_role', '')
        self.declare_parameter('local_map_outbound', False)

        self.input_topic = str(self.get_parameter('input_topic').value)
        self.output_topic = str(self.get_parameter('output_topic').value)
        self.check_period = max(
            0.1, float(self.get_parameter('check_period_sec').value)
        )
        self.takeover_grace = max(
            0.0, float(self.get_parameter('takeover_grace_sec').value)
        )
        self.standby_confirm_sec = max(
            0.0, float(self.get_parameter('standby_confirm_sec').value)
        )
        self.relay_without_primary = bool(
            self.get_parameter('relay_without_primary').value
        )
        self.max_publish_rate_hz = max(
            0.0, float(self.get_parameter('max_publish_rate_hz').value)
        )
        self.cached_republish_period_sec = max(
            0.0, float(self.get_parameter('cached_republish_period_sec').value)
        )
        self.min_publish_period_sec = (
            1.0 / self.max_publish_rate_hz
            if self.max_publish_rate_hz > 0.0 else 0.0
        )
        self.active_scout_id_topic = str(
            self.get_parameter('active_scout_id_topic').value
        ).strip()
        self.primary_scout_id = str(self.get_parameter('primary_scout_id').value).strip()
        self.follower_scout_id = str(self.get_parameter('follower_scout_id').value).strip()
        self.follower_input_topic = str(
            self.get_parameter('follower_input_topic').value
        ).strip()
        self.robot_role = str(self.get_parameter('robot_role').value).strip()
        self.local_map_outbound = bool(
            self.get_parameter('local_map_outbound').value
        )
        self.active_scout_id = self.primary_scout_id

        pub_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
        )
        bridge_sub_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
        )
        output_sub_qos = QoSProfile(
            depth=5,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
        )
        self._latest_bridged = None
        self._latest_follower = None
        self._last_seen_output = None
        self._relaying = False
        self._primary_missing_since = None
        self._primary_present_since = None
        self._first_bridged_logged = False
        self._first_output_logged = False
        self._last_published_signature = None
        self._last_publish_mono_sec = 0.0
        self._pending_rate_limited = False
        self._last_status_log_sec = 0.0

        self._pub = self.create_publisher(
            OccupancyGrid, self.output_topic, pub_qos
        )
        self._own_output_publishers = 1
        self._sub = self.create_subscription(
            OccupancyGrid, self.input_topic, self._on_bridged_map, bridge_sub_qos
        )
        self._follower_sub = None
        if self.follower_input_topic:
            self._follower_sub = self.create_subscription(
                OccupancyGrid,
                self.follower_input_topic,
                self._on_follower_map,
                bridge_sub_qos,
            )
        self._active_scout_sub = None
        if self.active_scout_id_topic:
            self._active_scout_sub = self.create_subscription(
                String,
                self.active_scout_id_topic,
                self._on_active_scout_id,
                pub_qos,
            )
        # Also watch the output topic itself so a takeover can continue from
        # whatever the primary (e.g. Cartographer) last published there,
        # instead of always jumping straight to the bridged leader map.
        self._output_sub = self.create_subscription(
            OccupancyGrid, self.output_topic, self._on_output_seen, output_sub_qos
        )
        self._timer = self.create_timer(
            self.check_period, self._check_primary
        )
        self.get_logger().info(
            f'map relay standing by: {self.input_topic} (transient_local) -> '
            f'{self.output_topic} (transient_local), only if no other '
            f'publisher is active on {self.output_topic}; '
            f'relay_without_primary={self.relay_without_primary} '
            f'max_publish_rate_hz={self.max_publish_rate_hz:.2f} '
            f'cached_republish_period_sec={self.cached_republish_period_sec:.2f} '
            f'active_scout_topic={self.active_scout_id_topic or "(disabled)"} '
            f'follower_input={self.follower_input_topic or "(disabled)"}'
        )

    def _on_bridged_map(self, msg: OccupancyGrid):
        if not self._is_valid_map(msg):
            self.get_logger().warning(
                'MAP_RELAY_INVALID_BRIDGE_MAP | ignoring invalid '
                f'{self.input_topic} sample',
                throttle_duration_sec=5.0,
            )
            return
        if not self._first_bridged_logged:
            self._first_bridged_logged = True
            self.get_logger().info(
                'MAP_BRIDGE_FIRST_RX | '
                f'topic={self.input_topic} width={msg.info.width} '
                f'height={msg.info.height} resolution={msg.info.resolution:.3f}'
            )
        self._latest_bridged = msg
        if self._relaying and self._selected_map_source() is msg:
            self._publish(msg)

    def _on_follower_map(self, msg: OccupancyGrid):
        if not self._is_valid_map(msg):
            self.get_logger().warning(
                'MAP_RELAY_INVALID_FOLLOWER_MAP | ignoring invalid '
                f'{self.follower_input_topic} sample',
                throttle_duration_sec=5.0,
            )
            return
        self._latest_follower = msg
        if self._relaying and self._selected_map_source() is msg:
            self._publish(msg)

    def _on_active_scout_id(self, msg: String):
        scout_id = str(msg.data).strip()
        if not scout_id or scout_id == self.active_scout_id:
            return
        self.active_scout_id = scout_id
        self.get_logger().warning(
            'MAP_RELAY_ACTIVE_SCOUT_CHANGED | '
            f'active_scout={self.active_scout_id} '
            f'selected_input={self._selected_input_topic()}'
        )
        if self._relaying:
            self._publish_latest()

    def _on_output_seen(self, msg: OccupancyGrid):
        if self._relaying:
            return
        if not self._is_valid_map(msg):
            return
        if not self._first_output_logged:
            self._first_output_logged = True
            self.get_logger().info(
                'MAP_OUTPUT_FIRST_RX | '
                f'topic={self.output_topic} width={msg.info.width} '
                f'height={msg.info.height} resolution={msg.info.resolution:.3f}'
            )
        self._last_seen_output = msg

    @staticmethod
    def _is_valid_map(msg: OccupancyGrid) -> bool:
        width = int(msg.info.width)
        height = int(msg.info.height)
        return (
            width > 0
            and height > 0
            and float(msg.info.resolution) > 0.0
            and len(msg.data) == width * height
        )

    @staticmethod
    def _map_signature(msg: OccupancyGrid):
        info = msg.info
        origin = info.origin
        data_hash = hashlib.blake2b(
            bytes((int(value) + 1) & 0xFF for value in msg.data),
            digest_size=16,
        ).hexdigest()
        return (
            msg.header.frame_id,
            int(msg.header.stamp.sec),
            int(msg.header.stamp.nanosec),
            int(info.width),
            int(info.height),
            round(float(info.resolution), 9),
            round(float(origin.position.x), 6),
            round(float(origin.position.y), 6),
            round(float(origin.position.z), 6),
            round(float(origin.orientation.x), 6),
            round(float(origin.orientation.y), 6),
            round(float(origin.orientation.z), 6),
            round(float(origin.orientation.w), 6),
            data_hash,
        )

    def _now_sec(self) -> float:
        return self.get_clock().now().nanoseconds * 1.0e-9

    def _now_mono_sec(self) -> float:
        return time.monotonic()

    def _external_publisher_count(self) -> int:
        # count_publishers() includes our own publishers on this topic, so
        # subtract them to learn whether anyone else is active.
        return max(
            0,
            self.count_publishers(self.output_topic)
            - self._own_output_publishers,
        )

    def _check_primary(self):
        if self.relay_without_primary:
            if not self._relaying:
                self._relaying = True
                self.get_logger().info(
                    'MAP_RELAY_EXTERNAL_MODE | relaying bridged map updates '
                    f'{self.input_topic} -> {self.output_topic}'
                )
                self._publish_latest()
            elif self._pending_rate_limited:
                self._publish_latest()
            elif (
                self.cached_republish_period_sec > 0.0
                and self._selected_map_source() is not None
                and (
                    self._last_publish_mono_sec <= 0.0
                    or self._now_mono_sec() - self._last_publish_mono_sec
                    >= self.cached_republish_period_sec
                )
            ):
                self._publish_latest(force=True)
            return

        now = self._now_sec()
        self._log_map_status(now)
        external = self._external_publisher_count()

        if external > 0:
            self._primary_missing_since = None
            if not self._relaying:
                return
            if self._primary_present_since is None:
                self._primary_present_since = now
            present_sec = now - self._primary_present_since
            if present_sec >= self.standby_confirm_sec:
                self._relaying = False
                self._primary_present_since = None
                self.get_logger().warning(
                    'MAP_RELAY_STANDBY | a primary map publisher has been '
                    f'active for {present_sec:.1f}s; relay standing down'
                )
                return
            # Reappearance not yet confirmed stable -- do not republish
            # (avoid two live sources on the output topic at once), but
            # keep _relaying True so a flicker back to "missing" resumes
            # serving the cached map instead of leaving a gap.
            self.get_logger().info(
                'MAP_RELAY_STANDBY_PENDING | primary map publisher seen, '
                f'confirming for {self.standby_confirm_sec - present_sec:.1f}s more',
                throttle_duration_sec=5.0,
            )
            return

        self._primary_present_since = None
        if self._primary_missing_since is None:
            self._primary_missing_since = now

        missing_sec = now - self._primary_missing_since
        if not self._relaying and missing_sec >= self.takeover_grace:
            self._relaying = True
            self.get_logger().error(
                'MAP_RELAY_TAKEOVER | no primary map publisher for '
                f'{missing_sec:.1f}s; relay taking over {self.output_topic}'
            )
            self._publish_latest()
        elif self._relaying:
            self._publish_latest()
            self.get_logger().info(
                f'Map relay active (no primary for {missing_sec:.1f}s)',
                throttle_duration_sec=10.0,
            )

    def _publish_latest(self, *, force: bool = False):
        # Prefer continuing from whatever the primary itself last
        # published (e.g. Cartographer's own last map before it died);
        # only fall back to the bridged leader map if we never saw one.
        selected = self._selected_map_source()
        source = selected if self.relay_without_primary else (
            self._last_seen_output or selected
        )
        if source is None:
            self.get_logger().warning(
                'MAP_RELAY_NO_CACHED_MAP | taking over but no map has '
                'been seen yet from either source',
                throttle_duration_sec=5.0,
            )
            return
        self._publish(source, force=force)

    def _selected_input_topic(self) -> str:
        if (
            self.follower_input_topic
            and self.follower_scout_id
            and self.active_scout_id == self.follower_scout_id
        ):
            return self.follower_input_topic
        return self.input_topic

    def _selected_map_source(self):
        if self._selected_input_topic() == self.follower_input_topic:
            return self._latest_follower or self._latest_bridged
        return self._latest_bridged

    def _publish(self, source: OccupancyGrid, *, force: bool = False):
        signature = self._map_signature(source)
        if not force and signature == self._last_published_signature:
            self.get_logger().debug(
                'MAP_RELAY_DUPLICATE_SKIPPED | unchanged map sample'
            )
            return False
        now = self._now_mono_sec()
        elapsed = now - self._last_publish_mono_sec
        if (
            self.min_publish_period_sec > 0.0
            and self._last_publish_mono_sec > 0.0
            and elapsed < self.min_publish_period_sec
        ):
            self._pending_rate_limited = True
            self.get_logger().debug(
                'MAP_RELAY_RATE_LIMITED | latest map retained for next slot'
            )
            return False
        self._pub.publish(source)
        self._last_published_signature = signature
        self._last_publish_mono_sec = now
        self._pending_rate_limited = False
        self.get_logger().info(
            f'Map relayed: {source.info.width}x{source.info.height} @ '
            f'{source.info.resolution:.3f}m/cell',
            throttle_duration_sec=10.0,
        )
        return True

    def _log_map_status(self, now_sec: float) -> None:
        if now_sec - self._last_status_log_sec < 10.0:
            return
        self._last_status_log_sec = now_sec
        shared_input = self.input_topic in ('/shared_map_in', '/map_bridge')
        shared_rx = bool(self._latest_bridged is not None and shared_input)
        self.get_logger().info(
            'FOLLOWER_MAP_STATUS | '
            f'role={self.robot_role or "unknown"} '
            f'shared_map_rx={str(shared_rx).lower()} '
            'shared_map_outbound=false '
            f'local_slam_map_rx={str(self._latest_follower is not None).lower()} '
            f'local_map_outbound={str(self.local_map_outbound).lower()} '
            f'input_topic={self.input_topic} output_topic={self.output_topic}'
        )


def main():
    rclpy.init()
    node = MapRelay()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.destroy_node()
        except KeyboardInterrupt:
            pass
        if rclpy.ok():
            try:
                rclpy.shutdown()
            except KeyboardInterrupt:
                pass


if __name__ == '__main__':
    main()
