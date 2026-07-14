import math

from sensor_msgs.msg import LaserScan

from omx_aim.scan_processor import ScanProcessor


class _Publisher:
    def __init__(self):
        self.messages = []

    def publish(self, msg):
        self.messages.append(msg)


def _bare_processor() -> ScanProcessor:
    node = ScanProcessor.__new__(ScanProcessor)
    node.flip = False
    node.min_valid = 0.20
    node.max_valid = 3.0
    node.mask = [(-0.05, 0.05)]
    node.n_scans = 0
    node.n_points_total = 0
    node.n_self_masked_nan = 0
    node.n_real_inf = 0
    node.n_finite_marking = 0
    node.n_range_rejected = 0
    node.pub = _Publisher()
    return node


def _scan(ranges):
    msg = LaserScan()
    msg.angle_min = -0.2
    msg.angle_increment = 0.1
    msg.angle_max = msg.angle_min + msg.angle_increment * (len(ranges) - 1)
    msg.range_min = 0.12
    msg.range_max = 3.5
    msg.ranges = list(ranges)
    return msg


def test_self_masked_and_rejected_ranges_become_nan_not_inf():
    node = _bare_processor()

    node.on_scan(_scan([1.0, 0.10, 1.2, 4.0, 2.0]))

    out = node.pub.messages[-1]
    assert math.isnan(out.ranges[1])
    assert math.isnan(out.ranges[2])
    assert math.isnan(out.ranges[3])
    assert node.n_self_masked_nan == 1
    assert node.n_range_rejected == 2


def test_real_sensor_inf_is_preserved_for_costmap_clearing():
    node = _bare_processor()

    node.on_scan(_scan([float('inf'), 1.0, 1.2]))

    out = node.pub.messages[-1]
    assert math.isinf(out.ranges[0])
    assert node.n_real_inf == 1
