import math

import pytest
from geometry_msgs.msg import PoseStamped

from fleet_bringup.global_localize_kickstart import GlobalLocalizeKickstart


def _pose(x: float, y: float, yaw: float) -> PoseStamped:
    msg = PoseStamped()
    msg.pose.position.x = x
    msg.pose.position.y = y
    msg.pose.orientation.z = math.sin(0.5 * yaw)
    msg.pose.orientation.w = math.cos(0.5 * yaw)
    return msg


def _bare_node() -> GlobalLocalizeKickstart:
    node = GlobalLocalizeKickstart.__new__(GlobalLocalizeKickstart)
    node.active_scout_robot_name = 'scout22'
    node.follower_robot_name = 'follower21'
    node.scout_pose_max_age_sec = 8.0
    node._active_scout_id = None
    node._member_pose = None
    node._member_pose_wall = None
    node._burger_pose = None
    node._burger_pose_wall = None
    node._last_scout_pose = None
    node._last_scout_pose_wall = None
    node._pending_seed_source = None
    node._pending_seed_age = None
    node._last_seed_wait_detail = ''
    node._now = lambda: 100.0
    return node


def test_scout_seed_uses_fresh_member_pose_without_active_id():
    node = _bare_node()
    node._member_pose = _pose(1.2, -0.4, 0.7)
    node._member_pose_wall = 97.5

    seed = node._scout_seed_pose()

    assert seed is not None
    x, y, yaw = seed
    assert x == 1.2
    assert y == -0.4
    assert yaw == pytest.approx(0.7)
    assert node._pending_seed_source == 'scout22'
    assert node._pending_seed_age == 2.5


def test_scout_seed_prefers_active_follower_when_available():
    node = _bare_node()
    node._active_scout_id = 'follower21'
    node._member_pose = _pose(1.2, -0.4, 0.7)
    node._member_pose_wall = 99.0
    node._burger_pose = _pose(-0.2, 0.8, -1.1)
    node._burger_pose_wall = 99.5

    seed = node._scout_seed_pose()

    assert seed is not None
    assert seed[0] == -0.2
    assert seed[1] == 0.8
    assert seed[2] == pytest.approx(-1.1)
    assert node._pending_seed_source == 'follower21'


def test_scout_seed_can_fallback_to_latched_last_scout_pose():
    node = _bare_node()
    node._last_scout_pose = _pose(2.0, 0.5, 1.2)
    node._last_scout_pose_wall = 98.0

    seed = node._scout_seed_pose()

    assert seed is not None
    assert seed[0] == 2.0
    assert seed[1] == 0.5
    assert seed[2] == pytest.approx(1.2)
    assert node._pending_seed_source == 'last_scout_pose'
