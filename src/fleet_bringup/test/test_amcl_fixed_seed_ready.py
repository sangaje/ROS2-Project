from pathlib import Path

from fleet_bringup.amcl_fixed_seed_ready import AmclFixedSeedReady


SOURCE = (
    Path(__file__).parents[1]
    / 'fleet_bringup'
    / 'amcl_fixed_seed_ready.py'
).read_text(encoding='utf-8')


def _bare_node(now: float, last_amcl_wall, tf_fresh: bool):
    node = AmclFixedSeedReady.__new__(AmclFixedSeedReady)
    node.last_scan_wall = now
    node.max_scan_age_sec = 1.5
    node.last_scan_ranges = 100
    node.last_scan_finite_ranges = 100
    node.base_frame = 'base_footprint'
    node.last_scan_frame = 'base_footprint'
    node.last_scan_stamp_sec = now
    node.last_scan_source_age_ms = 0.0
    node.last_odom_wall = now
    node.max_odom_age_sec = 1.5
    node.last_amcl_wall = last_amcl_wall
    node.max_amcl_age_sec = 2.0
    node.amcl_pose_finite = True
    node.xy_cov = 0.0
    node.yaw_cov = 0.0
    node.max_xy_cov = 0.22
    node.max_yaw_cov = 0.16
    node.global_frame = 'map'
    node.odom_frame = 'odom'
    node.map_valid = True
    node.map_known_cells = 1000
    node.min_known_map_cells = 100
    node.initial_pose_applied = True
    node._now = lambda: now
    node._tf_status = lambda target, source: (tf_fresh, 10.0)
    node._amcl_active = lambda: True
    return node


def test_fixed_seed_ready_watches_amcl_inputs_without_motion_side_effects():
    assert "class AmclFixedSeedReady" in SOURCE
    assert "self.declare_parameter('map_topic', '/map')" in SOURCE
    assert "self.declare_parameter('scan_topic', '/scan')" in SOURCE
    assert "self.declare_parameter('odom_topic', '/odom')" in SOURCE
    assert "self.declare_parameter('amcl_pose_topic', '/amcl_pose')" in SOURCE
    assert "self.declare_parameter('amcl_get_state_service', '/amcl/get_state')" in SOURCE
    assert "self.declare_parameter('ready_topic', '/localization_ready')" in SOURCE
    assert "self.declare_parameter('fixed_seed_initial_pose_applied', True)" in SOURCE
    assert 'self._publish_ready(True)' in SOURCE

    assert 'Twist' not in SOURCE
    assert "create_publisher(Bool, self.ready_topic" in SOURCE
    assert "create_publisher(Twist" not in SOURCE
    assert 'NavigateToPose' not in SOURCE


def test_fixed_seed_ready_requires_tf_and_lifecycle_active():
    assert 'lookup_transform' in SOURCE
    assert "self.global_frame" in SOURCE
    assert "self.base_frame" in SOURCE
    assert "GetState" in SOURCE
    assert "lifecycle=active" in SOURCE
    assert "LEADER_LOCALIZATION_DEBUG |" in SOURCE
    assert "blocking_reason=" in SOURCE
    assert "readiness_publisher_count=" in SOURCE


def test_fixed_seed_ready_reports_real_scan_blocking_reasons():
    assert "LEADER_SCAN_DEBUG |" in SOURCE
    assert "qos_compatible=best_effort_sensor_data" in SOURCE
    assert "publisher_count=" in SOURCE
    assert "subscription_count=" in SOURCE
    assert "scan_missing" in SOURCE
    assert "scan_stale" in SOURCE
    assert "scan_frame_missing" in SOURCE
    assert "scan_tf_unavailable" in SOURCE
    assert "scan_timestamp_out_of_range" in SOURCE
    assert "scan_empty" in SOURCE
    assert "return 'scan_fresh'" not in SOURCE


def test_stationary_robot_with_stale_amcl_topic_is_not_blocked_if_tf_is_fresh():
    # Regression test for the real-hardware catch-22: nav2's AMCL only
    # republishes /amcl_pose after motion past update_min_d/update_min_a, so
    # a stationary robot right after fixed-seed init can show
    # amcl_pose_age_ms in the hundreds of seconds forever, even with good
    # covariance -- and localization_ready never fires to release the very
    # motion that would let AMCL republish. A fresh map->odom TF (which AMCL
    # keeps rebroadcasting every filter cycle regardless of motion) must be
    # accepted as proof the last pose/covariance are still authoritative.
    now = 1_000_000.0
    node = _bare_node(now, last_amcl_wall=now - 242.0, tf_fresh=True)

    ok, reason, checks = node._readiness_state()

    assert checks['amcl_pose_fresh'] is True
    assert reason == 'none'
    assert ok is True


def test_amcl_pose_never_received_still_blocks_even_with_fresh_tf():
    now = 1_000_000.0
    node = _bare_node(now, last_amcl_wall=None, tf_fresh=True)

    ok, reason, checks = node._readiness_state()

    assert checks['amcl_pose_fresh'] is False
    assert reason == 'amcl_pose_stale'
    assert ok is False


def test_stale_amcl_pose_still_blocks_when_tf_is_also_stale():
    now = 1_000_000.0
    node = _bare_node(now, last_amcl_wall=now - 242.0, tf_fresh=False)

    ok, reason, checks = node._readiness_state()

    assert checks['amcl_pose_fresh'] is False
    assert reason == 'amcl_pose_stale'
    assert ok is False


def test_launch_files_pass_absolute_localization_ready_topic():
    root = Path(__file__).parents[2]
    for relative in (
        'fleet_bringup/launch/leader.launch.py',
        'fleet_bringup/launch/member.launch.py',
        'fleet_bringup/launch/follower.launch.py',
    ):
        text = (root / relative).read_text(encoding='utf-8')
        assert "'ready_topic': '/localization_ready'" in text
