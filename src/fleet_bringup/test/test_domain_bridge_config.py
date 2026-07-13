import yaml

from fleet_bringup.domain_bridge_config import (
    write_leader_to_pc_bridge_config,
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
    assert main['topics']['/fleet/video_ready']['type'] == 'std_msgs/msg/Bool'
    assert main['topics']['/fleet/video_ready']['qos']['durability'] == (
        'transient_local'
    )
    assert main['topics']['/system/ready']['type'] == 'std_msgs/msg/Bool'
    assert main['topics']['/system/ready']['qos']['durability'] == (
        'transient_local'
    )
    assert main['topics']['/system/readiness_detail']['type'] == (
        'std_msgs/msg/String'
    )
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
    assert follower['topics']['/burger_scan_relay']['remap'] == (
        '/follower25/scan'
    )
    assert '/risk/risk_map' not in follower['topics']
    assert '/risk/person_probability_map' not in follower['topics']
    assert '/risk/evidence_markers' not in follower['topics']
    assert '/field/follower25/risk_observation' in follower['topics']
    assert '/clock' not in main['topics']
    assert '/cmd_vel' not in follower['topics']
    assert main['topics']['/scout22/rl_confidence_map']['remap'] == (
        '/rl_confidence_seed'
    )


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
    assert main['topics']['/fleet/video_ready']['qos']['durability'] == (
        'transient_local'
    )
    assert main['topics']['/system/ready']['qos']['durability'] == (
        'transient_local'
    )
    assert main['topics']['/system/readiness']['type'] == 'std_msgs/msg/String'
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
    assert follower['topics']['/map']['type'] == 'nav_msgs/msg/OccupancyGrid'
    assert follower['topics']['/map']['remap'] == '/field/follower21/map'
    assert follower['topics']['/map']['qos']['reliability'] == 'reliable'
    assert follower['topics']['/map']['qos']['durability'] == 'transient_local'
    assert follower['topics']['/rl_confidence_map']['remap'] == (
        '/follower21/rl_confidence_map'
    )


def test_risk_to_leader_bridge_is_one_way_map_source(tmp_path):
    path = write_risk_to_leader_bridge_config(
        22,
        24,
        output_directory=tmp_path,
    )
    config = yaml.safe_load(path.read_text())

    assert (config['from_domain'], config['to_domain']) == (22, 24)
    assert config['topics']['/map']['remap'] == '/field/scout22/map'
    assert config['topics']['/rl_confidence_map']['remap'] == (
        '/scout22/rl_confidence_map'
    )
    assert config['topics']['/map']['qos']['reliability'] == 'reliable'
    assert config['topics']['/map']['qos']['durability'] == 'transient_local'
    assert config['topics']['/map']['qos']['history'] == 'keep_last'
    assert config['topics']['/member_pose']['type'] == 'geometry_msgs/msg/PoseStamped'
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
    assert '/member_pose' in config['topics']
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
    assert config['topics']['/system/ready']['qos']['durability'] == (
        'transient_local'
    )
    assert '/fleet_debug_markers' in config['topics']
    assert '/risk/risk_map' in config['topics']
    assert '/tf' not in config['topics']
    assert '/tf_static' not in config['topics']
    assert '/cmd_vel' not in config['topics']
