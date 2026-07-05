import yaml

from tb3_fleet_bringup.domain_bridge_config import write_fleet_bridge_configs


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
