from pathlib import Path


ROOT = Path(__file__).parents[3]


def _read(relpath: str) -> str:
    return (ROOT / relpath).read_text(encoding='utf-8')


def test_start_motion_is_the_only_cross_domain_motion_readiness_topic():
    bridge = _read('src/fleet_bringup/fleet_bringup/domain_bridge_config.py')

    body = bridge.split('def system_readiness_topics', 1)[1].split(
        'def _runtime_output_directory', 1
    )[0]
    assert "'/fleet/start_motion'" in body
    assert "'/fleet/readiness_detail'" in body
    assert "'/system/ready'" not in body
    assert 'dashboard_backend_ready' not in body
    assert 'dashboard_ui_ready' not in body
    assert 'dashboard_readiness_detail' not in body


def test_leader_dashboard_is_the_single_start_motion_publisher():
    sources = [
        path
        for base in ('src/system_bringup', 'src/fleet_bringup', 'src/omx_aim')
        for path in (ROOT / base).rglob('*.py')
        if '/test/' not in str(path)
    ]
    publishers = [
        str(path.relative_to(ROOT))
        for path in sources
        if 'start_motion_pub = self.create_publisher' in path.read_text(encoding='utf-8')
    ]
    assert publishers == [
        'src/system_bringup/system_bringup/leader_unified_dashboard.py'
    ]


def test_default_system_launch_routes_rl_through_unified_motion_authority():
    launch = _read('src/system_bringup/launch/system.launch.py')

    assert "'external_rl_cmd_topic': '/fleet/active_scout_rl_cmd'" in launch
    assert "'cmd_vel_topic': '/fleet/active_scout_rl_cmd'" in launch
    assert "'cmd_vel_topic': DEFAULT_CMD_VEL_TOPIC" in launch


def test_default_leader_shadow_does_not_create_hardware_cmd_publisher():
    source = _read('src/system_bringup/system_bringup/leader_shadow_follow.py')

    assert "self.cmd_pub = None" in source
    assert "if self.direct_shadow_cmd_vel:" in source
