import yaml

from fleet_bringup.domain_bridge_config import (
    validate_no_bridge_feedback_cycles,
    validate_no_duplicate_bridge_routes,
    write_leader_to_pc_bridge_config,
    write_field_robot_candidate_bridge_configs,
    write_fleet_bridge_configs,
    write_member_bridge_configs,
    write_risk_to_leader_bridge_config,
)


def test_real_bridge_directions_and_control_qos(tmp_path):
    main_path, follower_path = write_fleet_bridge_configs(
        24,
        25,
        output_directory=tmp_path,
    )
    main = yaml.safe_load(main_path.read_text())
    follower = yaml.safe_load(follower_path.read_text())

    assert (main['from_domain'], main['to_domain']) == (24, 25)
    assert (follower['from_domain'], follower['to_domain']) == (25, 24)
    assert main['topics']['/fleet/follow_command']['qos']['durability'] == (
        'transient_local'
    )
    assert main['topics']['/fleet/coordination_status']['qos']['depth'] == 1
    assert main['topics']['/fleet/collision_warning']['qos']['durability'] == (
        'transient_local'
    )
    assert main['topics']['/fleet/start_motion']['type'] == 'std_msgs/msg/Bool'
    assert main['topics']['/fleet/start_motion']['qos']['durability'] == (
        'transient_local'
    )
    assert main['topics']['/fleet/readiness_detail']['type'] == 'std_msgs/msg/String'
    assert main['topics']['/fleet/start_motion_detail']['type'] == 'std_msgs/msg/String'
    assert main['topics']['/fleet/start_motion_detail']['qos']['durability'] == (
        'transient_local'
    )
    assert main['topics']['/system/ready']['type'] == 'std_msgs/msg/Bool'
    assert main['topics']['/system/ready']['qos']['durability'] == (
        'transient_local'
    )
    assert '/fleet/dashboard_ui_ready' not in main['topics']
    assert main['topics']['/fleet/robot_poses']['type'] == (
        'geometry_msgs/msg/PoseArray'
    )
    assert main['topics']['/leader_pose']['type'] == (
        'geometry_msgs/msg/PoseStamped'
    )
    assert main['topics']['/member_pose']['type'] == (
        'geometry_msgs/msg/PoseStamped'
    )
    assert main['topics']['/omx/observation_status']['type'] == 'std_msgs/msg/String'
    assert main['topics']['/omx/observation_status']['qos'] == {
        'reliability': 'best_effort',
        'durability': 'volatile',
        'history': 'keep_last',
        'depth': 5,
    }
    assert main['topics']['/omx/camera_ready']['type'] == 'std_msgs/msg/Bool'
    assert main['topics']['/failover/active_scout_id']['qos']['durability'] == (
        'transient_local'
    )
    assert '/fleet/hazard_pose' in main['topics']
    assert follower['topics']['/fleet/follow_enabled']['qos']['durability'] == (
        'transient_local'
    )
    assert '/burger_scan_relay' not in follower['topics']
    assert '/risk/risk_map' not in follower['topics']
    assert '/risk/person_probability_map' not in follower['topics']
    assert '/risk/evidence_markers' not in follower['topics']
    assert '/field/follower25/risk_observation' in follower['topics']
    assert '/clock' not in main['topics']
    assert '/cmd_vel' not in follower['topics']
    assert main['topics']['/map']['remap'] == '/map_bridge'
    assert '/scout22/rl_confidence_map' not in main['topics']
    assert '/rl_confidence_seed' not in main['topics']


def test_simulation_bridge_adds_only_simulation_transport_topics(tmp_path):
    main_path, follower_path = write_fleet_bridge_configs(
        24,
        25,
        simulation=True,
        output_directory=tmp_path,
    )
    main = yaml.safe_load(main_path.read_text())
    follower = yaml.safe_load(follower_path.read_text())

    assert '/clock' in main['topics']
    assert '/burger/scan' in main['topics']
    assert '/burger/odom' in main['topics']
    assert follower['topics']['/cmd_vel']['remap'] == '/burger/cmd_vel'


def test_follower_scan_topic_uses_its_domain_id(tmp_path):
    _, follower_path = write_fleet_bridge_configs(
        24,
        31,
        include_follower_scan=True,
        output_directory=tmp_path,
    )
    follower = yaml.safe_load(follower_path.read_text())

    assert follower['topics']['/burger_scan_relay']['remap'] == (
        '/follower31/scan'
    )


def test_member_bridge_keeps_risk_topics_off_the_default_pose_status_path(tmp_path):
    main_path, member_path = write_member_bridge_configs(
        24,
        26,
        output_directory=tmp_path,
    )
    main = yaml.safe_load(main_path.read_text())
    member = yaml.safe_load(member_path.read_text())

    assert (main['from_domain'], main['to_domain']) == (24, 26)
    assert main['topics']['/map']['remap'] == '/map_bridge'
    assert main['topics']['/leader_pose']['type'] == (
        'geometry_msgs/msg/PoseStamped'
    )
    assert main['topics']['/omx/observation_status']['type'] == 'std_msgs/msg/String'
    assert main['topics']['/omx/camera_ready']['type'] == 'std_msgs/msg/Bool'
    assert main['topics']['/fleet/coordination_status']['qos']['durability'] == (
        'transient_local'
    )
    assert main['topics']['/fleet/start_motion']['qos']['durability'] == (
        'transient_local'
    )
    assert main['topics']['/system/ready']['type'] == 'std_msgs/msg/Bool'
    assert '/system/readiness' not in main['topics']
    assert main['topics']['/fleet/robot_poses']['type'] == (
        'geometry_msgs/msg/PoseArray'
    )
    assert (member['from_domain'], member['to_domain']) == (26, 24)
    assert '/risk/risk_map' not in member['topics']
    assert '/risk/person_probability_map' not in member['topics']
    assert '/risk/evidence_markers' not in member['topics']
    assert '/map' not in member['topics']


def test_member_bridge_can_explicitly_forward_risk_topics_to_main(tmp_path):
    _, member_path = write_member_bridge_configs(
        24,
        26,
        forward_risk_to_main=True,
        output_directory=tmp_path,
    )
    member = yaml.safe_load(member_path.read_text())

    assert member['topics']['/risk/risk_map']['type'] == (
        'nav_msgs/msg/OccupancyGrid'
    )
    assert member['topics']['/risk/risk_map']['qos']['durability'] == (
        'transient_local'
    )


def test_member_bridge_can_forward_owned_map_to_main(tmp_path):
    _, member_path = write_member_bridge_configs(
        20,
        22,
        forward_map_to_main=True,
        output_directory=tmp_path,
    )
    member = yaml.safe_load(member_path.read_text())

    assert (member['from_domain'], member['to_domain']) == (22, 20)
    assert member['topics']['/map']['type'] == 'nav_msgs/msg/OccupancyGrid'
    assert member['topics']['/map']['remap'] == '/field/scout22/map'
    assert member['topics']['/map']['qos']['reliability'] == 'reliable'
    assert member['topics']['/map']['qos']['durability'] == 'transient_local'


def test_follower_bridge_can_forward_owned_map_to_main(tmp_path):
    _, follower_path = write_fleet_bridge_configs(
        20,
        21,
        forward_map_to_main=True,
        output_directory=tmp_path,
    )
    follower = yaml.safe_load(follower_path.read_text())

    assert (follower['from_domain'], follower['to_domain']) == (21, 20)
    assert follower['topics']['/local_slam_map']['type'] == 'nav_msgs/msg/OccupancyGrid'
    assert follower['topics']['/local_slam_map']['remap'] == '/field/follower21/map'
    assert follower['topics']['/local_slam_map']['qos']['reliability'] == 'reliable'
    assert follower['topics']['/local_slam_map']['qos']['durability'] == 'transient_local'
    assert follower['topics']['/rl_confidence_map']['remap'] == (
        '/field/follower21/rl_confidence_map'
    )
    assert follower['topics']['/risk/risk_map']['remap'] == '/field/follower21/risk_map'
    assert '/map' not in follower['topics']


def test_bridge_cycle_validator_rejects_map_echo_loop():
    main_config = {
        'name': 'main_to_follower',
        'from_domain': 20,
        'to_domain': 21,
        'topics': {
            '/map': {
                'type': 'nav_msgs/msg/OccupancyGrid',
                'remap': '/map_bridge',
            },
        },
    }
    follower_config = {
        'name': 'follower_to_main',
        'from_domain': 21,
        'to_domain': 20,
        'topics': {
            '/map': {
                'type': 'nav_msgs/msg/OccupancyGrid',
                'remap': '/field/follower21/map',
            },
            '/field/follower21/map': {
                'type': 'nav_msgs/msg/OccupancyGrid',
                'remap': '/map',
            },
        },
    }

    try:
        validate_no_bridge_feedback_cycles(
            [main_config, follower_config],
            relay_edges=[
                ((21, '/map_bridge'), (21, '/map'), 'map_relay'),
                ((20, '/field/follower21/map'), (20, '/map'), 'active_map_mux'),
            ],
        )
    except ValueError as exc:
        assert 'feedback cycle' in str(exc)
    else:
        raise AssertionError('expected map feedback cycle rejection')


def test_risk_to_leader_bridge_is_one_way_map_source(tmp_path):
    path = write_risk_to_leader_bridge_config(
        22,
        24,
        output_directory=tmp_path,
    )
    config = yaml.safe_load(path.read_text())

    assert (config['from_domain'], config['to_domain']) == (22, 24)
    assert config['topics']['/map']['remap'] == '/field/scout22/map'
    assert config['topics']['/map']['qos']['reliability'] == 'reliable'
    assert config['topics']['/map']['qos']['durability'] == 'transient_local'
    assert config['topics']['/map']['qos']['history'] == 'keep_last'
    assert '/member_pose' not in config['topics']
    assert '/scout/signal' not in config['topics']
    assert '/rl_confidence_map' not in config['topics']
    assert config['topics']['/risk/yolo_detections']['type'] == 'std_msgs/msg/String'
    assert '/tf' not in config['topics']
    assert '/tf_static' not in config['topics']


def test_risk_to_leader_bridge_can_exclude_scout_risk_outputs(tmp_path):
    path = write_risk_to_leader_bridge_config(
        22,
        24,
        include_risk_outputs=False,
        output_directory=tmp_path,
    )
    config = yaml.safe_load(path.read_text())

    assert '/map' in config['topics']
    assert '/member_pose' not in config['topics']
    assert '/risk/yolo_detections' in config['topics']
    assert '/risk/risk_map' not in config['topics']
    assert '/risk/person_probability_map' not in config['topics']
    assert '/risk/evidence_markers' not in config['topics']


def test_leader_to_pc_bridge_is_visualization_only(tmp_path):
    path = write_leader_to_pc_bridge_config(
        24,
        30,
        output_directory=tmp_path,
    )
    config = yaml.safe_load(path.read_text())

    assert (config['from_domain'], config['to_domain']) == (24, 30)
    assert config['topics']['/map']['qos']['reliability'] == 'reliable'
    assert config['topics']['/map']['qos']['durability'] == 'transient_local'
    assert config['topics']['/fleet/start_motion']['qos']['durability'] == (
        'transient_local'
    )
    assert '/fleet/readiness_detail' in config['topics']
    assert '/fleet/start_motion_detail' in config['topics']
    assert config['topics']['/system/ready']['type'] == 'std_msgs/msg/Bool'
    assert '/fleet_debug_markers' in config['topics']
    assert '/risk/risk_map' in config['topics']
    assert '/tf' not in config['topics']
    assert '/tf_static' not in config['topics']
    assert '/cmd_vel' not in config['topics']


def test_candidate_field_bridge_uses_identity_namespaces(tmp_path):
    paths = write_field_robot_candidate_bridge_configs(
        20,
        [
            {'robot_name': 'scout22', 'domain_id': 22, 'initial_role': 'ACTIVE_SCOUT'},
            {'robot_name': 'follower21', 'domain_id': 21, 'initial_role': 'FOLLOWER'},
        ],
        output_directory=tmp_path,
    )
    configs = [yaml.safe_load(path.read_text()) for path in paths]
    from_configs = [c for c in configs if c['to_domain'] == 20]

    assert any(
        c['topics']['/member_pose']['remap'] == '/field/scout22/pose'
        for c in from_configs
        if c['from_domain'] == 22
    )
    assert any(
        c['topics']['/burger_pose']['remap'] == '/field/follower21/pose'
        for c in from_configs
        if c['from_domain'] == 21
    )
    assert any(
        c['topics']['/map']['remap'] == '/field/scout22/map'
        for c in from_configs
        if c['from_domain'] == 22
    )
    assert all(
        '/map' not in c['topics']
        for c in from_configs
        if c['from_domain'] == 21
    )


def test_duplicate_bridge_routes_are_rejected():
    duplicate = {
        'name': 'dup',
        'from_domain': 22,
        'to_domain': 20,
        'topics': {'/map': {'type': 'nav_msgs/msg/OccupancyGrid'}},
    }
    try:
        validate_no_duplicate_bridge_routes([duplicate, duplicate])
    except ValueError as exc:
        assert 'duplicate domain_bridge route' in str(exc)
    else:
        raise AssertionError('duplicate route was not rejected')
