import rclpy
from nav_msgs.msg import OccupancyGrid
from std_msgs.msg import String

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
    msg.data = [0] * (width * width)
    return msg


def invalid_grid() -> OccupancyGrid:
    msg = OccupancyGrid()
    msg.info.width = 10
    msg.info.height = 10
    msg.info.resolution = 0.05
    msg.data = []
    return msg


def test_relay_stays_silent_while_a_primary_publisher_is_active():
    node = make_node()
    try:
        assert node._own_output_publishers == 1
        published = []
        node._pub.publish = lambda msg: published.append(msg)
        node.count_publishers = (
            lambda topic: node._own_output_publishers + 1
        )
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
        node.count_publishers = lambda topic: node._own_output_publishers
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


def test_zero_grace_takes_over_on_first_missing_primary_check():
    node = make_node()
    try:
        published = []
        node.takeover_grace = 0.0
        node._pub.publish = lambda msg: published.append(msg)
        node.count_publishers = lambda topic: node._own_output_publishers
        node._now_sec = lambda: 0.0
        node._on_bridged_map(grid(20))

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
        node.count_publishers = lambda topic: node._own_output_publishers
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


def test_relay_stands_down_only_after_primary_confirmed_stable():
    node = make_node()
    try:
        published = []
        node._pub.publish = lambda msg: published.append(msg)
        now = [0.0]
        node._now_sec = lambda: now[0]
        node.count_publishers = lambda topic: node._own_output_publishers
        node._on_bridged_map(grid(10))

        node._check_primary()  # primes _primary_missing_since
        now[0] += node.takeover_grace + 0.1
        node._check_primary()
        assert node._relaying is True

        node.count_publishers = (
            lambda topic: node._own_output_publishers + 1
        )
        node._check_primary()
        # Reappearance seen but not yet confirmed stable -- still relaying.
        assert node._relaying is True

        now[0] += node.standby_confirm_sec - 0.1
        node._check_primary()
        assert node._relaying is True

        now[0] += 0.2
        node._check_primary()
        assert node._relaying is False
    finally:
        destroy_node(node)


def test_flickering_primary_presence_does_not_flap_the_relay_down():
    node = make_node()
    try:
        published = []
        node._pub.publish = lambda msg: published.append(msg)
        now = [0.0]
        node._now_sec = lambda: now[0]
        node.count_publishers = lambda topic: node._own_output_publishers
        node._on_bridged_map(grid(10))

        node._check_primary()
        now[0] += node.takeover_grace + 0.1
        node._check_primary()
        assert node._relaying is True

        # Primary flickers present for less than standby_confirm_sec, then
        # disappears again -- the relay must never have stopped serving.
        node.count_publishers = (
            lambda topic: node._own_output_publishers + 1
        )
        now[0] += node.standby_confirm_sec * 0.5
        node._check_primary()
        assert node._relaying is True

        node.count_publishers = lambda topic: node._own_output_publishers
        now[0] += 0.1
        node._check_primary()
        assert node._relaying is True
        assert len(published) >= 1
    finally:
        destroy_node(node)


def test_zero_standby_confirm_preserves_instant_standdown():
    node = make_node()
    try:
        published = []
        node.standby_confirm_sec = 0.0
        node._pub.publish = lambda msg: published.append(msg)
        now = [0.0]
        node._now_sec = lambda: now[0]
        node.count_publishers = lambda topic: node._own_output_publishers
        node._on_bridged_map(grid(10))

        node._check_primary()
        now[0] += node.takeover_grace + 0.1
        node._check_primary()
        assert node._relaying is True

        node.count_publishers = (
            lambda topic: node._own_output_publishers + 1
        )
        node._check_primary()
        assert node._relaying is False
    finally:
        destroy_node(node)


def test_new_bridged_maps_are_republished_immediately_while_relaying():
    node = make_node()
    try:
        published = []
        node.min_publish_period_sec = 0.0
        node._pub.publish = lambda msg: published.append(msg)
        now = [0.0]
        node._now_sec = lambda: now[0]
        node.count_publishers = lambda topic: node._own_output_publishers
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


def test_relay_skips_unchanged_cached_map_while_active():
    node = make_node()
    try:
        published = []
        node._pub.publish = lambda msg: published.append(msg)
        now = [0.0]
        node._now_sec = lambda: now[0]
        node.count_publishers = lambda topic: node._own_output_publishers
        node._on_bridged_map(grid(10))

        node._check_primary()  # primes _primary_missing_since
        now[0] += node.takeover_grace + 0.1
        node._check_primary()
        assert len(published) == 1

        now[0] += node.check_period
        node._check_primary()
        assert len(published) == 1
    finally:
        destroy_node(node)


def test_relay_publishes_when_map_payload_changes():
    node = make_node()
    try:
        published = []
        node.min_publish_period_sec = 0.0
        node._pub.publish = lambda msg: published.append(msg)
        now = [0.0]
        node._now_sec = lambda: now[0]
        node.count_publishers = lambda topic: node._own_output_publishers
        first = grid(10)
        second = grid(10)
        second.data[0] = 42

        node._on_bridged_map(first)
        node._check_primary()
        now[0] += node.takeover_grace + 0.1
        node._check_primary()
        assert len(published) == 1

        node._on_bridged_map(second)
        assert len(published) == 2
        assert published[-1].data[0] == 42
    finally:
        destroy_node(node)


def test_relay_selects_follower_map_after_active_scout_takeover():
    node = make_node()
    try:
        node.follower_input_topic = '/follower21/map_bridge'
        node.follower_scout_id = 'follower21'
        published = []
        node.min_publish_period_sec = 0.0
        node._pub.publish = lambda msg: published.append(msg)
        node.relay_without_primary = True
        node._relaying = True

        node._on_bridged_map(grid(22))
        assert published[-1].info.width == 22

        follower = String()
        follower.data = 'follower21'
        node._on_follower_map(grid(21))
        node._on_active_scout_id(follower)

        assert node._selected_input_topic() == '/follower21/map_bridge'
        assert published[-1].info.width == 21

        # Late scout22 maps must not overwrite the selected follower map.
        node._on_bridged_map(grid(23))
        assert published[-1].info.width == 21
    finally:
        destroy_node(node)


def test_invalid_bridged_map_is_not_treated_as_available():
    node = make_node()
    try:
        node._on_bridged_map(invalid_grid())
        assert node._latest_bridged is None
    finally:
        destroy_node(node)


def test_relay_does_not_republish_its_own_stale_output_over_fresh_bridge():
    node = make_node()
    try:
        published = []
        node.min_publish_period_sec = 0.0

        def publish(msg):
            published.append(msg)
            node._on_output_seen(msg)

        node._pub.publish = publish
        now = [0.0]
        node._now_sec = lambda: now[0]
        node.count_publishers = lambda topic: node._own_output_publishers
        node._on_bridged_map(grid(10))

        node._check_primary()
        now[0] += node.takeover_grace + 0.1
        node._check_primary()
        assert published[-1].info.width == 10

        node._on_bridged_map(grid(42))
        assert published[-1].info.width == 42
    finally:
        destroy_node(node)
