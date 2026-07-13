from pathlib import Path


ROOT = Path(__file__).parents[3]


def test_field_robot_launch_is_single_entrypoint_for_scout_and_follower():
    source = (
        ROOT / 'src' / 'system_bringup' / 'launch' / 'field_robot.launch.py'
    ).read_text(encoding='utf-8')

    assert "initial_role" in source
    assert "ACTIVE_SCOUT" in source
    assert "FOLLOWER" in source
    assert "'fleet_role': fleet_role" in source
    assert "'start_risk_map': 'false'" in source
    assert "'enable_yolo': 'false'" in source
    assert "'start_camera_sender': _bool_text" in source
    assert "field_enable_exploration" in source
    assert "field_enable_cartographer" in source
    assert "field_enable_amcl" in source


def test_system_launch_routes_field_observations_to_robot_topics():
    source = (
        ROOT / 'src' / 'system_bringup' / 'launch' / 'system.launch.py'
    ).read_text(encoding='utf-8')

    assert "field_observation_topic = f'/field/{scout_robot_name}/risk_observation'" in source
    assert "'output_topic': field_observation_topic" in source
    assert "'pose_topic': (" in source
    assert "'active_roles': 'ACTIVE_SCOUT,SCOUT,FOLLOWER,RECOVERING'" in source
    assert "'observation_topics': [" in source
    assert "f'/field/{active_scout_robot_name.perform(context)}/risk_observation'" in source
    assert "f'/field/{follower_robot_name.perform(context)}/risk_observation'" in source


def test_leader_map_input_uses_field_robot_namespaces():
    leader = (
        ROOT / 'src' / 'fleet_bringup' / 'launch' / 'leader.launch.py'
    ).read_text(encoding='utf-8')
    system = (
        ROOT / 'src' / 'system_bringup' / 'launch' / 'system.launch.py'
    ).read_text(encoding='utf-8')

    assert "f'/field/{active_scout_robot_name.perform(context)}/map'" in leader
    assert "default_value='/field/follower21/map'" in leader
    assert "f'/field/{follower_robot_name.perform(context)}/map'" in system
