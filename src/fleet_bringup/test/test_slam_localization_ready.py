import rclpy
from nav_msgs.msg import OccupancyGrid
from sensor_msgs.msg import LaserScan

from fleet_bringup.slam_localization_ready import SlamLocalizationReady
from pathlib import Path


SOURCE = (
    Path(__file__).parents[1]
    / 'fleet_bringup'
    / 'slam_localization_ready.py'
).read_text(encoding='utf-8')


def make_node() -> SlamLocalizationReady:
    if not rclpy.ok():
        rclpy.init()
    return SlamLocalizationReady()


def destroy_node(node: SlamLocalizationReady) -> None:
    node.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()


def _map(known_cells: int) -> OccupancyGrid:
    msg = OccupancyGrid()
    msg.info.width = known_cells
    msg.info.height = 1
    msg.info.resolution = 0.05
    msg.data = [50] * known_cells
    return msg


def test_stays_not_ready_until_map_scan_and_tf_all_valid():
    node = make_node()
    try:
        published = []
        node.ready_pub.publish = lambda msg: published.append(msg.data)
        node._now = lambda: 100.0
        node._tf_ok = lambda: False
        node._tf_status = lambda: (False, -1.0)
        node.min_known_map_cells = 10

        node._on_map(_map(50))
        node._on_scan(LaserScan())
        node._tick()

        assert node.done is False
        assert published == []
    finally:
        destroy_node(node)


def test_slam_ready_uses_absolute_latched_topic_and_debug_log():
    assert "self.declare_parameter('ready_topic', '/localization_ready')" in SOURCE
    assert "LEADER_LOCALIZATION_DEBUG |" in SOURCE
    assert "mode=cartographer" in SOURCE
    assert "blocking_reason=" in SOURCE


def test_requires_conditions_to_hold_for_stable_duration_before_latching_ready():
    node = make_node()
    try:
        published = []
        node.ready_pub.publish = lambda msg: published.append(msg.data)
        node.min_known_map_cells = 10
        node.stable_duration_sec = 2.0
        node._tf_ok = lambda: True
        node._tf_status = lambda: (True, 0.0)
        clock = [100.0]
        node._now = lambda: clock[0]

        node._on_map(_map(50))
        node._on_scan(LaserScan())
        node.last_scan_wall = clock[0]

        node._tick()
        assert node.done is False

        clock[0] += 1.0
        node._on_scan(LaserScan())
        node._tick()
        assert node.done is False

        clock[0] += 1.1
        node._on_scan(LaserScan())
        node._tick()
        assert node.done is True
        assert published == [True]
    finally:
        destroy_node(node)


def test_below_min_known_cells_never_latches():
    node = make_node()
    try:
        published = []
        node.ready_pub.publish = lambda msg: published.append(msg.data)
        node.min_known_map_cells = 100
        node.stable_duration_sec = 0.0
        node._tf_ok = lambda: True
        node._tf_status = lambda: (True, 0.0)
        node._now = lambda: 100.0

        node._on_map(_map(5))
        node._on_scan(LaserScan())
        node.last_scan_wall = 100.0
        node._tick()

        assert node.done is False
    finally:
        destroy_node(node)


def test_stale_scan_resets_the_stability_window():
    node = make_node()
    try:
        published = []
        node.ready_pub.publish = lambda msg: published.append(msg.data)
        node.min_known_map_cells = 10
        node.stable_duration_sec = 1.0
        node.max_scan_age_sec = 1.0
        node._tf_ok = lambda: True
        node._tf_status = lambda: (True, 0.0)
        clock = [100.0]
        node._now = lambda: clock[0]

        node._on_map(_map(50))
        node.last_scan_wall = clock[0]
        node._tick()
        assert node.good_since_wall == 100.0

        clock[0] += 5.0  # scan goes stale, no new _on_scan call
        node._tick()
        assert node.good_since_wall is None
        assert node.done is False
    finally:
        destroy_node(node)


def test_latches_exactly_once():
    node = make_node()
    try:
        published = []
        node.ready_pub.publish = lambda msg: published.append(msg.data)
        node.min_known_map_cells = 10
        node.stable_duration_sec = 0.0
        node._tf_ok = lambda: True
        node._tf_status = lambda: (True, 0.0)
        node._now = lambda: 100.0

        node._on_map(_map(50))
        node._on_scan(LaserScan())
        node.last_scan_wall = 100.0

        node._tick()
        node._tick()
        node._tick()

        assert published == [True]
    finally:
        destroy_node(node)
