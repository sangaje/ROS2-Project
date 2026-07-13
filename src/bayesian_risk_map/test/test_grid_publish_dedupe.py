import numpy as np
from builtin_interfaces.msg import Time
from nav_msgs.msg import OccupancyGrid

from bayesian_risk_map.bayesian_risk_map_node import RoomAwareRiskMapNode


class FakePublisher:
    def __init__(self):
        self.messages = []

    def publish(self, msg):
        self.messages.append(msg)


def make_node():
    node = RoomAwareRiskMapNode.__new__(RoomAwareRiskMapNode)
    node.map_frame = 'map'
    node.occ_grid = np.zeros((4, 4), dtype=np.int16)
    node.latest_map_msg = OccupancyGrid()
    node.latest_map_msg.info.width = 4
    node.latest_map_msg.info.height = 4
    node.latest_map_msg.info.resolution = 0.05
    node._published_layer_signatures = {}
    return node


def test_array_layer_publish_skips_unchanged_payload():
    node = make_node()
    pub = FakePublisher()
    stamp = Time(sec=1)
    arr = np.zeros((4, 4), dtype=np.float32)

    assert node._publish_array_layer('risk', pub, arr, stamp)
    assert len(pub.messages) == 1

    assert not node._publish_array_layer('risk', pub, arr.copy(), Time(sec=2))
    assert len(pub.messages) == 1

    arr[0, 0] = 0.7
    assert node._publish_array_layer('risk', pub, arr, Time(sec=3))
    assert len(pub.messages) == 2
    assert pub.messages[-1].data[0] == 70


def test_region_id_publish_skips_unchanged_payload():
    node = make_node()
    pub = FakePublisher()
    stamp = Time(sec=1)
    node.region_id_map = np.zeros((4, 4), dtype=np.int32)

    assert node._publish_region_id_layer('region_id', pub, stamp)
    assert len(pub.messages) == 1

    assert not node._publish_region_id_layer('region_id', pub, Time(sec=2))
    assert len(pub.messages) == 1

    node.region_id_map[1, 1] = 3
    assert node._publish_region_id_layer('region_id', pub, Time(sec=3))
    assert len(pub.messages) == 2
