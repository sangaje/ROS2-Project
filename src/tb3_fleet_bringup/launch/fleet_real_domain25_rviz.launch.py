#!/usr/bin/env python3

import tempfile
from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    LogInfo,
    OpaqueFunction,
    SetEnvironmentVariable,
    UnsetEnvironmentVariable,
)
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    bringup_share = get_package_share_directory('tb3_fleet_bringup')

    domain_id   = LaunchConfiguration('domain_id')
    rviz_config = LaunchConfiguration('rviz_config')
    robot1_ip   = LaunchConfiguration('robot1_ip')
    robot2_ip   = LaunchConfiguration('robot2_ip')

    default_rviz_config = str(Path(bringup_share) / 'rviz' / 'fleet_debug.rviz')
    marker_script       = str(Path(bringup_share) / 'scripts' / 'fleet_debug_marker.py')
    rviz_clean_script   = str(Path(bringup_share) / 'scripts' / 'run_rviz2_clean.bash')

    def make_fastdds_env(context, *args, **kwargs):
        peers = [p.strip() for p in (robot1_ip.perform(context), robot2_ip.perform(context)) if p.strip()]
        d  = domain_id.perform(context)
        if not peers:
            return [
                UnsetEnvironmentVariable('ROS_STATIC_PEERS'),
                LogInfo(msg=['REAL_FLEET_RVIZ_D', d, ' | DDS discovery=subnet multicast']),
            ]
        locators = '\n'.join(
            f'          <locator><udpv4><address>{peer}</address></udpv4></locator>'
            for peer in peers
        )
        xml = f"""<?xml version="1.0" encoding="UTF-8"?>
<profiles xmlns="http://www.eprosima.com/XMLSchemas/fastRTPS_Profiles">
  <participant profile_name="default_profile" is_default_profile="true">
    <rtps>
      <builtin>
        <initialPeersList>
{locators}
        </initialPeersList>
      </builtin>
    </rtps>
  </participant>
</profiles>
"""
        xml_path = Path(tempfile.gettempdir()) / f'fastdds_fleet_rviz_d{d}.xml'
        xml_path.write_text(xml, encoding='utf-8')
        return [
            SetEnvironmentVariable('FASTRTPS_DEFAULT_PROFILES_FILE', str(xml_path)),
            SetEnvironmentVariable('FASTDDS_DEFAULT_PROFILES_FILE', str(xml_path)),
            SetEnvironmentVariable('ROS_STATIC_PEERS', ';'.join(peers)),
        ]

    marker = ExecuteProcess(
        cmd=[
            'python3', marker_script, '--ros-args',
            '-r', '__node:=fleet_real_debug_marker',
            '-p', 'use_sim_time:=false',
            '-p', 'waffle_pose_topic:=/leader_pose',
            '-p', 'burger_pose_topic:=/burger_pose',
            '-p', 'marker_topic:=/fleet_debug_markers',
            '-p', 'frame_id:=map',
        ],
        output='screen',
        name='fleet_real_debug_marker',
    )
    rviz = ExecuteProcess(
        cmd=[
            rviz_clean_script,
            '-d', rviz_config,
            '--ros-args',
            '-r', '__node:=rviz2_real_domain25_fleet',
            '-p', 'use_sim_time:=false',
        ],
        output='screen',
        name='rviz2_real_domain25_fleet',
    )

    return LaunchDescription([
        DeclareLaunchArgument('domain_id',    default_value='25'),
        DeclareLaunchArgument('rviz_config',  default_value=default_rviz_config),
        DeclareLaunchArgument('robot1_ip',    default_value=''),
        DeclareLaunchArgument('robot2_ip',    default_value=''),
        UnsetEnvironmentVariable('ROS_DISCOVERY_SERVER'),
        UnsetEnvironmentVariable('ROS_LOCALHOST_ONLY'),
        UnsetEnvironmentVariable('FASTRTPS_DEFAULT_PROFILES_FILE'),
        UnsetEnvironmentVariable('FASTDDS_DEFAULT_PROFILES_FILE'),
        SetEnvironmentVariable('ROS_DOMAIN_ID',               domain_id),
        SetEnvironmentVariable('RMW_IMPLEMENTATION',          'rmw_fastrtps_cpp'),
        SetEnvironmentVariable('FASTDDS_BUILTIN_TRANSPORTS',  'UDPv4'),
        SetEnvironmentVariable('ROS_AUTOMATIC_DISCOVERY_RANGE', 'SUBNET'),
        OpaqueFunction(function=make_fastdds_env),
        LogInfo(msg='REAL_FLEET_RVIZ_D25 | /goal_pose → Waffle | /fleet/follow_command → Burger'),
        marker,
        rviz,
    ])
