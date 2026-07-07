#!/usr/bin/env python3
"""
Receives /map_bridge from domain_bridge with a transient-local
subscription and stands by to republish it on /map (transient_local) only
when nothing else is currently publishing there. While relaying, it
periodically republishes the cached map so volatile/default subscribers
such as plain `ros2 topic echo /map --once` can receive a fresh sample too.

/map is meant to have exactly one live source at a time -- normally a local
SLAM node (e.g. Cartographer) when this robot owns its own mapping. This
relay stays silent while that primary is alive, detected via count_publishers() on the output topic (not
message content, so there is no self-feedback risk once this node starts
publishing too). If the primary disappears for longer than
takeover_grace_sec, the relay takes over: it continues from the primary's
own last published map if one was ever seen, or falls back to the latest
bridged map otherwise, so downstream AMCL/costmaps keep seeing a map instead
of losing it outright. It steps back down the moment the primary reappears.
"""
import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from nav_msgs.msg import OccupancyGrid


class MapRelay(Node):
    def __init__(self):
        super().__init__('map_relay')
        self.declare_parameter('input_topic', '/map_bridge')
        self.declare_parameter('output_topic', '/map')
        self.declare_parameter('check_period_sec', 1.0)
        self.declare_parameter('takeover_grace_sec', 2.0)

        self.input_topic = str(self.get_parameter('input_topic').value)
        self.output_topic = str(self.get_parameter('output_topic').value)
        self.check_period = max(
            0.1, float(self.get_parameter('check_period_sec').value)
        )
        self.takeover_grace = max(
            0.0, float(self.get_parameter('takeover_grace_sec').value)
        )

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
        self._last_seen_output = None
        self._relaying = False
        self._primary_missing_since = None
        self._first_bridged_logged = False
        self._first_output_logged = False

        self._pub = self.create_publisher(
            OccupancyGrid, self.output_topic, pub_qos
        )
        self._sub = self.create_subscription(
            OccupancyGrid, self.input_topic, self._on_bridged_map, bridge_sub_qos
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
            f'publisher is active on {self.output_topic}'
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
        if self._relaying:
            self._publish(msg)

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

    def _now_sec(self) -> float:
        return self.get_clock().now().nanoseconds * 1.0e-9

    def _external_publisher_count(self) -> int:
        # count_publishers() always includes our own publisher on this
        # topic, so subtract it to learn whether anyone else is active.
        return max(0, self.count_publishers(self.output_topic) - 1)

    def _check_primary(self):
        now = self._now_sec()
        external = self._external_publisher_count()

        if external > 0:
            self._primary_missing_since = None
            if self._relaying:
                self._relaying = False
                self.get_logger().warning(
                    'MAP_RELAY_STANDBY | a primary map publisher is '
                    'active again; relay standing down'
                )
            return

        if self._primary_missing_since is None:
            self._primary_missing_since = now
            return

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

    def _publish_latest(self):
        # Prefer continuing from whatever the primary itself last
        # published (e.g. Cartographer's own last map before it died);
        # only fall back to the bridged leader map if we never saw one.
        source = self._last_seen_output or self._latest_bridged
        if source is None:
            self.get_logger().warning(
                'MAP_RELAY_NO_CACHED_MAP | taking over but no map has '
                'been seen yet from either source',
                throttle_duration_sec=5.0,
            )
            return
        self._publish(source)

    def _publish(self, source: OccupancyGrid):
        self._pub.publish(source)
        self.get_logger().info(
            f'Map relayed: {source.info.width}x{source.info.height} @ '
            f'{source.info.resolution:.3f}m/cell',
            throttle_duration_sec=10.0,
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
