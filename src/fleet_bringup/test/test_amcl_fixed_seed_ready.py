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
