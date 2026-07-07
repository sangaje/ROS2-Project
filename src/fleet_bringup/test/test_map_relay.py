import rclpy
from nav_msgs.msg import OccupancyGrid

from fleet_bringup.map_relay import MapRelay


def make_node() -> MapRelay:
    if not rclpy.ok():
        rclpy.init()
    return MapRelay()


def destroy_node(node: MapRelay) -> None:
    node.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()


def grid(width: int) -> OccupancyGrid:
    msg = OccupancyGrid()
    msg.info.width = width
    msg.info.height = width
    msg.info.resolution = 0.05
    return msg


def test_relay_stays_silent_while_a_primary_publisher_is_active():
    node = make_node()
    try:
        published = []
        node._pub.publish = lambda msg: published.append(msg)
        node.count_publishers = lambda topic: 2  # us + one real primary
        node._on_bridged_map(grid(10))

        node._check_primary()
        node._check_primary()
        assert published == []
        assert node._relaying is False
    finally:
        destroy_node(node)


def test_relay_takes_over_after_grace_period_once_primary_disappears():
    node = make_node()
    try:
        published = []
        node._pub.publish = lambda msg: published.append(msg)
        node.count_publishers = lambda topic: 1  # only ourselves
        now = [0.0]
        node._now_sec = lambda: now[0]
        node._on_bridged_map(grid(20))

        node._check_primary()
        assert node._relaying is False
        assert published == []

        now[0] += node.takeover_grace + 0.1
        node._check_primary()
        assert node._relaying is True
        assert len(published) == 1
        assert published[0].info.width == 20
    finally:
        destroy_node(node)


def test_takeover_prefers_the_primarys_own_last_output_over_the_bridged_map():
    node = make_node()
    try:
        published = []
        node._pub.publish = lambda msg: published.append(msg)
        node.count_publishers = lambda topic: 1
        now = [0.0]
        node._now_sec = lambda: now[0]

        node._on_bridged_map(grid(20))
        node._on_output_seen(grid(99))  # e.g. Cartographer's last map

        node._check_primary()  # primes _primary_missing_since
        now[0] += node.takeover_grace + 0.1
        node._check_primary()
        assert published[0].info.width == 99
    finally:
        destroy_node(node)


def test_relay_stands_down_the_moment_a_primary_reappears():
    node = make_node()
    try:
        published = []
        node._pub.publish = lambda msg: published.append(msg)
        now = [0.0]
        node._now_sec = lambda: now[0]
        node.count_publishers = lambda topic: 1
        node._on_bridged_map(grid(10))

        node._check_primary()  # primes _primary_missing_since
        now[0] += node.takeover_grace + 0.1
        node._check_primary()
        assert node._relaying is True

        node.count_publishers = lambda topic: 2
        node._check_primary()
        assert node._relaying is False
    finally:
        destroy_node(node)


def test_new_bridged_maps_are_republished_immediately_while_relaying():
    node = make_node()
    try:
        published = []
        node._pub.publish = lambda msg: published.append(msg)
        now = [0.0]
        node._now_sec = lambda: now[0]
        node.count_publishers = lambda topic: 1
        node._on_bridged_map(grid(10))

        node._check_primary()  # primes _primary_missing_since
        now[0] += node.takeover_grace + 0.1
        node._check_primary()
        assert len(published) == 1

        node._on_bridged_map(grid(42))
        assert len(published) == 2
        assert published[-1].info.width == 42
    finally:
        destroy_node(node)


def test_relay_does_not_republish_its_own_stale_output_over_fresh_bridge():
    node = make_node()
    try:
        published = []

        def publish(msg):
            published.append(msg)
            node._on_output_seen(msg)

        node._pub.publish = publish
        now = [0.0]
        node._now_sec = lambda: now[0]
        node.count_publishers = lambda topic: 1
        node._on_bridged_map(grid(10))

        node._check_primary()
        now[0] += node.takeover_grace + 0.1
        node._check_primary()
        assert published[-1].info.width == 10

        node._on_bridged_map(grid(42))
        assert published[-1].info.width == 42
    finally:
        destroy_node(node)
