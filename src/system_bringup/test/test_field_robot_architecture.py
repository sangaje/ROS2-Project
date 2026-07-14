from pathlib import Path


ROOT = Path(__file__).parents[3]


def test_field_robot_launch_is_single_entrypoint_for_leader_scout_and_follower():
    source = (
        ROOT / 'src' / 'system_bringup' / 'launch' / 'field_robot.launch.py'
    ).read_text(encoding='utf-8')

    assert "initial_role" in source
    assert "LEADER" in source
    assert "ACTIVE_SCOUT" in source
    assert "FOLLOWER" in source
    assert "'role': 'leader'" in source
    assert "'fleet_role': fleet_role" in source
    assert "scout_capable=true" in source
    assert "normal_duty=" in source
    assert "takeover_duty=active_scout" in source
    assert "leader_follow" in source
    assert "'start_risk_map': 'false'" in source
    assert "'enable_yolo': 'false'" in source
    assert "'start_camera_sender': _bool_text" in source
    assert "'forward_field_map_to_main': requested_map_forward" in source
    assert "field_enable_exploration" in source
    assert "field_enable_cartographer" in source
    assert "field_enable_amcl" in source
    assert "initial_role=FOLLOWER cannot start Cartographer" in source
    assert "initial_role=FOLLOWER cannot claim map authority at startup" in source
    assert "requested_map_forward = _bool_text" in source
    assert "is_follower," in source
    assert "default_exploration = True" in source
    assert "DeclareLaunchArgument('enable_rl', default_value='')" in source
    assert "DeclareLaunchArgument('enable_follow'" not in source


def test_system_launch_routes_field_observations_to_robot_topics():
    source = (
        ROOT / 'src' / 'system_bringup' / 'launch' / 'system.launch.py'
    ).read_text(encoding='utf-8')

    assert "field_observation_topic = f'/field/{scout_robot_name}/risk_observation'" in source
    assert "'output_topic': field_observation_topic" in source
    assert "'active_max_upload_mbps': '2.5'" in source
    assert "'standby_max_upload_mbps': '0.8'" in source
    assert "'width': '640'" in source
    assert "'height': '480'" in source
    assert "'pose_topic': (" in source
    assert "'active_roles': 'ACTIVE_SCOUT,SCOUT,RECOVERING'" in source
    assert "'standby_roles': 'FOLLOWER,IDLE,TAKEOVER_PENDING'" in source
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
    assert "map_source_topic=(" in system
    assert "/map_out" in system
    assert "default_value='/field/follower21/map'" in leader
    assert "f'/field/{follower_robot_name.perform(context)}/map'" in system
