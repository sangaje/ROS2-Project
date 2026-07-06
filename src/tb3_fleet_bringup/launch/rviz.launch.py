#!/usr/bin/env python3
"""Fleet visualization on the leader DDS domain."""

from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, LogInfo
from launch.substitutions import EnvironmentVariable, LaunchConfiguration
from launch_ros.actions import Node

from tb3_fleet_bringup.launch_utils import dds_launch_environment


def generate_launch_description():
    package_share = Path(get_package_share_directory('tb3_fleet_bringup'))
    domain_id = LaunchConfiguration('domain_id')
    member_domain_id = LaunchConfiguration('member_domain_id')
    burger_domain_id = LaunchConfiguration('burger_domain_id')

    return LaunchDescription([
        DeclareLaunchArgument(
            'domain_id',
            default_value=EnvironmentVariable('ROS_DOMAIN_ID'),
            description='Leader/PC DDS domain.',
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
        LogInfo(msg=['FLEET_RVIZ | domain=', domain_id]),
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
                str(package_share / 'scripts' / 'run_rviz2_clean.bash'),
                '-d',
                str(package_share / 'rviz' / 'fleet_debug.rviz'),
                '--ros-args',
                '-r',
                '__node:=rviz2_fleet',
            ],
            output='screen',
            name='rviz2_fleet',
        ),
    ])
