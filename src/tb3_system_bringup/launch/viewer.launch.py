#!/usr/bin/env python3
"""PC-side unified view: fleet debug markers and the scout's risk map in a
single RViz window, on the leader/PC DDS domain."""

from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, LogInfo
from launch.substitutions import EnvironmentVariable, LaunchConfiguration
from launch_ros.actions import Node

from tb3_fleet_bringup.launch_utils import dds_launch_environment


def generate_launch_description():
    fleet_share = Path(get_package_share_directory('tb3_fleet_bringup'))
    system_share = Path(get_package_share_directory('tb3_system_bringup'))
    domain_id = LaunchConfiguration('domain_id')

    return LaunchDescription([
        DeclareLaunchArgument(
            'domain_id',
            default_value=EnvironmentVariable('ROS_DOMAIN_ID'),
            description='Leader/PC DDS domain that fleet and risk topics are bridged to.',
        ),
        DeclareLaunchArgument(
            'ros_static_peers',
            default_value=EnvironmentVariable('ROS_STATIC_PEERS', default_value=''),
            description=(
                'Optional ROS_STATIC_PEERS value (semicolon-separated '
                'addresses) forcing unicast DDS discovery to specific '
                'peers in addition to SUBNET multicast discovery -- needed '
                'when the robots are only reachable over a link that does '
                'not carry multicast, such as a Tailscale/VPN hop between '
                'machines on different physical LANs.'
            ),
        ),
        *dds_launch_environment(domain_id, LaunchConfiguration('ros_static_peers')),
        LogInfo(msg=['SYSTEM_VIEWER | domain=', domain_id]),
        Node(
            package='tb3_fleet_bringup',
            executable='fleet_debug_marker',
            name='fleet_debug_marker',
            output='screen',
            parameters=[{
                'use_sim_time': False,
                'leader_pose_topic': '/leader_pose',
                'burger_pose_topic': '/burger_pose',
                'marker_topic': '/fleet_debug_markers',
                'frame_id': 'map',
            }],
        ),
        ExecuteProcess(
            cmd=[
                str(fleet_share / 'scripts' / 'run_rviz2_clean.bash'),
                '-d',
                str(system_share / 'rviz' / 'system_view.rviz'),
                '--ros-args',
                '-r',
                '__node:=rviz2_system',
            ],
            output='screen',
            name='rviz2_system',
        ),
    ])
