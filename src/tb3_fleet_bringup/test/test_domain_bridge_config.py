import yaml

from tb3_fleet_bringup.domain_bridge_config import (
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
    assert main['topics']['/fleet/robot_poses']['type'] == (
        'geometry_msgs/msg/PoseArray'
    )
    assert '/fleet/hazard_pose' in main['topics']
    assert follower['topics']['/fleet/follow_enabled']['qos']['durability'] == (
        'transient_local'
    )
    assert follower['topics']['/burger_scan_relay']['remap'] == (
        '/follower25/scan'
    )
    assert follower['topics']['/risk/risk_map']['type'] == (
        'nav_msgs/msg/OccupancyGrid'
    )
    assert follower['topics']['/risk/risk_map']['qos']['durability'] == (
        'transient_local'
    )
    assert follower['topics']['/risk/risk_map']['qos']['depth'] == 1
    assert follower['topics']['/risk/person_probability_map']['type'] == (
        'nav_msgs/msg/OccupancyGrid'
    )
    assert follower['topics']['/risk/evidence_markers']['type'] == (
        'visualization_msgs/msg/MarkerArray'
    )
    assert '/clock' not in main['topics']
    assert '/cmd_vel' not in follower['topics']


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


def test_member_bridge_forwards_core_risk_topics_to_main(tmp_path):
    _, member_path = write_member_bridge_configs(
        24,
        26,
        output_directory=tmp_path,
    )
    member = yaml.safe_load(member_path.read_text())

    assert (member['from_domain'], member['to_domain']) == (26, 24)
    assert member['topics']['/risk/risk_map']['type'] == (
        'nav_msgs/msg/OccupancyGrid'
    )
    assert member['topics']['/risk/risk_map']['qos']['durability'] == (
        'transient_local'
    )
    assert member['topics']['/risk/person_probability_map']['qos']['depth'] == 1
    assert member['topics']['/risk/evidence_markers']['type'] == (
        'visualization_msgs/msg/MarkerArray'
    )
    assert '/map' not in member['topics']


def test_risk_to_leader_bridge_is_one_way_map_source(tmp_path):
    path = write_risk_to_leader_bridge_config(
        22,
        24,
        output_directory=tmp_path,
    )
    config = yaml.safe_load(path.read_text())

    assert (config['from_domain'], config['to_domain']) == (22, 24)
    assert config['topics']['/map']['remap'] == '/map_bridge'
    assert config['topics']['/map']['qos']['durability'] == 'transient_local'
    assert '/tf' not in config['topics']
    assert '/tf_static' not in config['topics']


def test_leader_to_pc_bridge_is_visualization_only(tmp_path):
    path = write_leader_to_pc_bridge_config(
        24,
        30,
        output_directory=tmp_path,
    )
    config = yaml.safe_load(path.read_text())

    assert (config['from_domain'], config['to_domain']) == (24, 30)
    assert config['topics']['/map']['qos']['durability'] == 'transient_local'
    assert '/fleet_debug_markers' in config['topics']
    assert '/risk/risk_map' in config['topics']
    assert '/tf' not in config['topics']
    assert '/tf_static' not in config['topics']
    assert '/cmd_vel' not in config['topics']
