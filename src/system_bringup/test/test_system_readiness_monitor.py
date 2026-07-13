from pathlib import Path


MONITOR = (
    Path(__file__).parents[1]
    / 'system_bringup'
    / 'system_readiness_monitor.py'
)


def test_system_readiness_monitor_publishes_required_latched_contract():
    source = MONITOR.read_text(encoding='utf-8')

    assert "class ReadinessStage" in source
    for stage in (
        'BOOTING',
        'SENSOR_READY',
        'MAP_TF_READY',
        'LOCALIZATION_READY',
        'NAV2_READY',
        'DASHBOARD_VIDEO_READY',
        'SYSTEM_READY',
        'RUNNING',
    ):
        assert stage in source
    for topic in (
        '/system/ready',
        '/system/readiness',
        '/system/readiness_detail',
    ):
        assert topic in source
    assert 'DurabilityPolicy.TRANSIENT_LOCAL' in source


def test_system_readiness_detail_contains_all_startup_dependencies():
    source = MONITOR.read_text(encoding='utf-8')

    for key in (
        'scout_sensor',
        'scout_localization',
        'leader_localization',
        'follower_localization',
        'leader_nav2',
        'follower_nav2',
        'map_tf',
        'domain_bridges',
        'dashboard',
        'scout_video',
        'yolo_video',
        'system_ready',
        'blocking_reasons',
    ):
        assert key in source


def test_system_readiness_monitor_waits_for_nav2_dashboard_and_bridge_inputs():
    source = MONITOR.read_text(encoding='utf-8')

    assert "ActionClient(self, NavigateToPose" in source
    assert "create_client(GetState" in source
    assert "self.create_subscription(Bool, self.video_ready_topic" in source
    assert "fleet_registry_json" in source
    assert "follower_candidates" in source
    assert "any(self._field_robot_nav_ready(name)" in source
    assert "self.tf_buffer.lookup_transform" in source
