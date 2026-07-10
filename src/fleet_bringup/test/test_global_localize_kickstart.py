import math

import pytest
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry

from fleet_bringup.global_localize_kickstart import GlobalLocalizeKickstart, State


class _Logger:
    def warning(self, *args, **kwargs):
        pass

    def error(self, *args, **kwargs):
        pass

    def info(self, *args, **kwargs):
        pass


def _pose(x: float, y: float, yaw: float) -> PoseStamped:
    msg = PoseStamped()
    msg.pose.position.x = x
    msg.pose.position.y = y
    msg.pose.orientation.z = math.sin(0.5 * yaw)
    msg.pose.orientation.w = math.cos(0.5 * yaw)
    return msg


def _odom(yaw: float) -> Odometry:
    msg = Odometry()
    msg.pose.pose.orientation.z = math.sin(0.5 * yaw)
    msg.pose.pose.orientation.w = math.cos(0.5 * yaw)
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
    node.get_logger = lambda: _Logger()
    node.last_pose_cov = None
    node.last_pose_cov_wall = None
    node.cov_xy_threshold = 0.35
    node.cov_yaw_threshold = 0.25
    node.max_amcl_pose_age_sec = 1.5
    node.stable_duration_sec = 1.0
    node.good_since_wall = None
    node.reinit_in_flight = False
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


def test_retry_spin_never_reenters_scout_pose_seed_path():
    node = _bare_node()
    transitions = []
    starts = []
    node.retry_count = 0
    node.max_spin_retries = 3
    node._seeded_this_attempt = True
    node._pending_seed_source = 'scout22'
    node._pending_seed_age = 0.2
    node.get_logger = lambda: _Logger()
    node._transition = transitions.append
    node._start_spin = lambda: starts.append('spin')

    node._tick_retry_spin()

    assert starts == ['spin']
    assert State.WAIT_SCOUT_POSE not in transitions
    assert node._seeded_this_attempt is False
    assert node._pending_seed_source is None


def test_completed_bootstrap_latch_blocks_future_ticks():
    node = _bare_node()
    node.done = False
    node.localization_bootstrap_completed = True
    node.bootstrap_failed = False
    called = []
    node._PRE_SPIN_STATES = (State.WAIT_MAP,)
    node.state = State.WAIT_MAP
    node._forced_spin = False
    node.node_start_wall = 0.0
    node.force_spin_after_sec = 1.0
    node._transition = lambda state: called.append(state)
    node._tick_wait_map = lambda: called.append('tick_wait_map')

    node._tick()

    assert called == []


def test_prerequisite_deadline_fails_safe_instead_of_forcing_spin():
    node = _bare_node()
    node.done = False
    node.localization_bootstrap_completed = False
    node.bootstrap_failed = False
    node.state = State.WAIT_MAP
    node.node_start_wall = 0.0
    node.force_spin_after_sec = 10.0
    node.require_valid_map = True
    node.map_received = False
    node.require_amcl_before_spin = True
    node.amcl_active = False
    node.require_scan_before_spin = True
    node.last_scan_wall = None
    node.max_scan_age_sec = 1.0
    node.require_odom_before_spin = True
    node.last_odom_wall = None
    node.last_odom_yaw = None
    node.max_odom_age_sec = 1.0
    node.spin_enabled = True
    node.spin_target_angle = 6.45
    node.spin_margin = 0.0
    transitions = []
    node._transition = transitions.append
    node._tick_wait_map = lambda: pytest.fail('wait handler ran after deadline')

    node._tick()

    assert transitions == [State.FAIL_SAFE]
    assert State.CHECK_LOCALIZATION_QUALITY not in transitions


def test_bad_covariance_after_verified_spin_enters_bounded_retry():
    node = _bare_node()
    node.initial_spin_completed = True
    node.last_pose_cov = {'xy_cov': 2.0, 'yaw_cov': 1.5}
    node.last_pose_cov_wall = 100.0
    node.check_start_wall = 90.0
    node.check_timeout_sec = 2.0
    node.total_attempts = 1
    node.retry_count = 0
    node.max_spin_retries = 1
    twists = []
    transitions = []
    node._publish_twist = twists.append
    node._transition = transitions.append

    node._tick_check_localization()

    assert node.retry_count == 1
    assert transitions == [State.RETRY_SPIN]
    assert State.READY_FOR_NAV not in transitions
    assert twists[-1] == 0.0

    node._tick_check_localization()
    assert transitions[-1] == State.FAIL_SAFE
    assert State.READY_FOR_NAV not in transitions


def test_ready_entry_requires_completed_spin_and_fresh_stable_covariance():
    node = _bare_node()
    node.initial_spin_completed = True
    node.last_pose_cov = {'xy_cov': 0.10, 'yaw_cov': 0.08}
    node.last_pose_cov_wall = 100.0
    node.good_since_wall = 98.0
    node.done = False
    node.localization_bootstrap_completed = False
    published_ready = []
    twists = []
    transitions = []
    node._publish_ready = published_ready.append
    node._publish_twist = twists.append
    node._transition = transitions.append

    node._on_enter_ready_for_nav()

    assert transitions == []
    assert published_ready == [True]
    assert twists == [0.0]
    assert node.done is True
    assert node.localization_bootstrap_completed is True


@pytest.mark.parametrize(
    ('spin_completed', 'covariance_wall'),
    [(False, 100.0), (True, 90.0)],
)
def test_ready_entry_rejects_missing_spin_or_stale_covariance(
    spin_completed, covariance_wall,
):
    node = _bare_node()
    node.initial_spin_completed = spin_completed
    node.last_pose_cov = {'xy_cov': 0.10, 'yaw_cov': 0.08}
    node.last_pose_cov_wall = covariance_wall
    node.good_since_wall = 98.0
    transitions = []
    node._transition = transitions.append

    node._on_enter_ready_for_nav()

    assert transitions == [State.FAIL_SAFE]


def test_retry_budget_counts_each_retry_and_then_fails_safe():
    node = _bare_node()
    node.retry_count = 0
    node.max_spin_retries = 2
    twists = []
    transitions = []
    node._publish_twist = twists.append
    node._transition = transitions.append

    node._retry_or_fail('spin_timeout')
    node._retry_or_fail('sensor_dropout')
    node._retry_or_fail('covariance_not_stable')

    assert node.retry_count == 2
    assert transitions == [
        State.RETRY_SPIN,
        State.RETRY_SPIN,
        State.FAIL_SAFE,
    ]
    assert twists == [0.0, 0.0, 0.0]


def test_amcl_evidence_window_covers_settle_and_stability_contract():
    node = _bare_node()
    node.max_amcl_pose_age_sec = 4.0
    node.last_pose_cov = {'xy_cov': 0.10, 'yaw_cov': 0.08}
    node.last_pose_cov_wall = 96.1

    assert node._covariance_ok() is True

    node.last_pose_cov_wall = 95.9
    assert node._covariance_ok() is False


def test_spin_progress_counts_net_rotation_in_commanded_direction():
    node = _bare_node()
    node.state = State.SPIN
    node.last_odom_yaw = 0.0
    node.last_odom_xy = (0.0, 0.0)
    node.last_odom_wall = 0.0
    node.spin_direction = 1.0
    node.accumulated_yaw = 0.0

    node._on_odom(_odom(-0.20))
    node._on_odom(_odom(0.30))

    assert node.accumulated_yaw == pytest.approx(0.30)
