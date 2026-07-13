from pathlib import Path


SOURCE = (
    Path(__file__).parents[1]
    / 'fleet_bringup'
    / 'amcl_fixed_seed_ready.py'
).read_text(encoding='utf-8')


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


def test_launch_files_pass_absolute_localization_ready_topic():
    root = Path(__file__).parents[2]
    for relative in (
        'fleet_bringup/launch/leader.launch.py',
        'fleet_bringup/launch/member.launch.py',
        'fleet_bringup/launch/follower.launch.py',
    ):
        text = (root / relative).read_text(encoding='utf-8')
        assert "'ready_topic': '/localization_ready'" in text
