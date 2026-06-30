#!/usr/bin/env python3

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, SetEnvironmentVariable, LogInfo
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    bringup_share = get_package_share_directory('tb3_fleet_bringup')
    rviz_config = LaunchConfiguration('rviz_config')
    marker_script = os.path.join(bringup_share, 'scripts', 'fleet_debug_marker_direct_v41.py')

    marker_node = ExecuteProcess(
        cmd=[
            'python3', marker_script, '--ros-args',
            '-r', '__node:=fleet_debug_marker_domain26',
            '-p', 'use_sim_time:=true',
            '-p', 'waffle_pose_topic:=/leader_pose',
            '-p', 'burger_pose_topic:=/burger_pose',
            '-p', 'marker_topic:=/fleet_debug_markers',
            '-p', 'frame_id:=map',
        ],
        output='screen',
        name='fleet_debug_marker_domain26_v41',
    )

    rviz = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2_domain26_burger_debug',
        output='screen',
        arguments=['-d', rviz_config],
        parameters=[{'use_sim_time': True}],
    )

    return LaunchDescription([
        DeclareLaunchArgument('rviz_config', default_value=os.path.join(bringup_share, 'rviz', 'fleet_domain25_debug_v41.rviz')),
        SetEnvironmentVariable('ROS_DOMAIN_ID', '26'),
        SetEnvironmentVariable('RMW_IMPLEMENTATION', 'rmw_fastrtps_cpp'),
        SetEnvironmentVariable('FASTDDS_BUILTIN_TRANSPORTS', 'UDPv4'),
        LogInfo(msg='V41_DOMAIN26_BURGER_RVIZ | RViz attached directly to Burger domain. Default /goal_pose controls Burger.'),
        marker_node,
        rviz,
    ])
