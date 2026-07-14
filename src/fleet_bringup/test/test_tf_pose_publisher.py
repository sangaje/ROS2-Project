import math

from geometry_msgs.msg import PoseStamped

from fleet_bringup.tf_pose_publisher import TfPosePublisher


class _Logger:
    def info(self, *args, **kwargs):
        pass

    def warn(self, *args, **kwargs):
        pass


def _pose(x: float, y: float, yaw: float = 0.0) -> PoseStamped:
    msg = PoseStamped()
    msg.pose.position.x = x
    msg.pose.position.y = y
    msg.pose.orientation.z = math.sin(0.5 * yaw)
    msg.pose.orientation.w = math.cos(0.5 * yaw)
    return msg


def _bare_filter() -> TfPosePublisher:
    node = TfPosePublisher.__new__(TfPosePublisher)
    node.freeze_when_stationary = True
    node.stationary_linear_threshold_m = 0.02
    node.stationary_angular_threshold_rad = 0.035
    node.stationary_freeze_warmup_sec = 0.0
    node.map_jump_filter_enabled = False
    node.map_jump_min_allowed_m = 0.20
    node.map_jump_odom_scale = 4.0
    node.map_jump_slop_m = 0.12
    node._start_wall = 0.0
    node._last_motion_pose = None
    node._last_accepted_pose = None
    node._freeze_count = 0
    node._last_freeze_log_wall = 0.0
    node.output_topic = '/leader_pose'
    node.get_logger = lambda: _Logger()
    return node


def test_stationary_filter_holds_last_pose_when_odom_does_not_move():
    node = _bare_filter()

    first, frozen = node._select_pose_for_publish(_pose(1.0, 2.0), (0.0, 0.0, 0.0))
    assert not frozen
    assert first.pose.position.x == 1.0

    second, frozen = node._select_pose_for_publish(
        _pose(1.35, 2.25),
        (0.005, 0.004, 0.01),
    )

    assert frozen
    assert second.pose.position.x == 1.0
    assert second.pose.position.y == 2.0


def test_stationary_filter_accepts_new_pose_after_odom_motion():
    node = _bare_filter()
    node._select_pose_for_publish(_pose(1.0, 2.0), (0.0, 0.0, 0.0))

    second, frozen = node._select_pose_for_publish(
        _pose(1.35, 2.25),
        (0.08, 0.0, 0.0),
    )

    assert not frozen
    assert second.pose.position.x == 1.35
    assert second.pose.position.y == 2.25


def test_map_jump_filter_rejects_pose_teleport_larger_than_odom_motion():
    node = _bare_filter()
    node.map_jump_filter_enabled = True
    node._select_pose_for_publish(_pose(1.0, 2.0), (0.0, 0.0, 0.0))

    second, frozen = node._select_pose_for_publish(
        _pose(1.80, 2.00),
        (0.04, 0.0, 0.0),
    )

    assert frozen
    assert second.pose.position.x == 1.0
    assert second.pose.position.y == 2.0


def test_map_jump_filter_accepts_pose_motion_consistent_with_odom():
    node = _bare_filter()
    node.map_jump_filter_enabled = True
    node._select_pose_for_publish(_pose(1.0, 2.0), (0.0, 0.0, 0.0))

    second, frozen = node._select_pose_for_publish(
        _pose(1.15, 2.00),
        (0.04, 0.0, 0.0),
    )

    assert not frozen
    assert second.pose.position.x == 1.15
