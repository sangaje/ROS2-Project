from pathlib import Path


SYSTEM_LAUNCH = Path(__file__).parents[1] / 'launch' / 'system.launch.py'


def test_three_robot_initial_formation_defaults_are_scout_centered():
    text = SYSTEM_LAUNCH.read_text(encoding='utf-8')

    assert "'leader_initial_y'," in text
    assert "default_value='0.10'" in text
    assert "'scout_initial_y'," in text
    assert "default_value='0.0'" in text
    assert "'follower_initial_y'," in text
    assert "default_value='-0.10'" in text


def test_system_launch_passes_initial_pose_to_fleet_launches():
    text = SYSTEM_LAUNCH.read_text(encoding='utf-8')

    assert "fleet_launch_args['leader_initial_y']" in text
    assert "fleet_launch_args['member_initial_y']" in text
    assert "fleet_launch_args['follower_initial_y']" in text


def test_scout_member_pose_is_stabilized_when_cartographer_corrects_while_stopped():
    text = (
        Path(__file__).parents[2]
        / 'fleet_bringup'
        / 'launch'
        / 'member.launch.py'
    ).read_text(encoding='utf-8')

    assert "'freeze_when_stationary': True" in text
    assert "'stationary_linear_threshold_m': 0.003" in text


def test_system_launch_uses_external_worker_without_in_process_runtime():
    text = SYSTEM_LAUNCH.read_text(encoding='utf-8')

    assert "default_value='external_worker'" in text
    assert "'start_rl_worker'" in text
    assert "'scout_rl_inference.launch.py'" in text
    assert "'initial_role_active': 'false'" in text
    assert "'rl_backend': rl_backend_value" in text


def test_leader_can_own_risk_map_from_scout_sources():
    text = SYSTEM_LAUNCH.read_text(encoding='utf-8')

    assert "'start_leader_risk_map'" in text
    assert 'include_risk_outputs=not launch_bool' in text
    assert "package='bayesian_risk_map'" in text
    assert "'pose_topic': scout_pose_topic.perform(context)" in text
    assert "'detection_source': 'flask_topic'" in text


def test_scout_can_run_cartographer_without_local_risk_map():
    text = SYSTEM_LAUNCH.read_text(encoding='utf-8')

    assert 'or launch_bool(start_cartographer.perform(context))' in text
    assert "'start_risk_map': start_risk_map.perform(context)" in text
