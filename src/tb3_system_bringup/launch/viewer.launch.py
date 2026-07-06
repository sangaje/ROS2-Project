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
    member_domain_id = LaunchConfiguration('member_domain_id')
    burger_domain_id = LaunchConfiguration('burger_domain_id')

    return LaunchDescription([
        DeclareLaunchArgument(
            'domain_id',
            default_value=EnvironmentVariable('ROS_DOMAIN_ID'),
            description='Leader/PC DDS domain that fleet and risk topics are bridged to.',
        ),
        DeclareLaunchArgument(
            'member_domain_id',
            default_value='22',
            description='Scout/member DDS domain shown in RViz marker labels.',
        ),
        DeclareLaunchArgument(
            'burger_domain_id',
            default_value='22',
            description='Follower/Burger DDS domain shown in RViz marker labels.',
        ),
        *dds_launch_environment(domain_id),
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
                'member_pose_topic': '/member_pose',
                'marker_topic': '/fleet_debug_markers',
                'frame_id': 'map',
                'leader_domain_id': domain_id,
                'member_domain_id': member_domain_id,
                'burger_domain_id': burger_domain_id,
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
