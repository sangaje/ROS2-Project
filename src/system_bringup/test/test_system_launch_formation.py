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
