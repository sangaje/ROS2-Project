from nav_msgs.msg import OccupancyGrid

from system_bringup.takeover_map_exporter import TakeoverMapExporter


def _grid(width, height, origin_x=0.0, origin_y=0.0, fill=-1):
    msg = OccupancyGrid()
    msg.header.frame_id = 'map'
    msg.info.width = width
    msg.info.height = height
    msg.info.resolution = 1.0
    msg.info.origin.position.x = origin_x
    msg.info.origin.position.y = origin_y
    msg.info.origin.orientation.w = 1.0
    msg.data = [fill] * (width * height)
    return msg


def test_takeover_map_exporter_merges_cached_baseline_with_new_local_map():
    node = TakeoverMapExporter.__new__(TakeoverMapExporter)
    baseline = _grid(3, 3, 0.0, 0.0, -1)
    baseline.data[0] = 0
    baseline.data[4] = 100

    current = _grid(2, 2, 2.0, 1.0, -1)
    current.data[0] = 0
    current.data[3] = 100

    merged = node._merge(baseline, current)

    assert merged.info.width == 4
    assert merged.info.height == 3
    assert merged.info.origin.position.x == 0.0
    assert merged.info.origin.position.y == 0.0
    assert merged.data[0] == 0
    assert merged.data[1 * merged.info.width + 1] == 100
    assert merged.data[1 * merged.info.width + 2] == 0
    assert merged.data[2 * merged.info.width + 3] == 100
