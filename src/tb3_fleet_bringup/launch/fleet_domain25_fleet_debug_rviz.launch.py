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
    domain_id = LaunchConfiguration('domain_id')
    marker_script = os.path.join(bringup_share, 'scripts', 'fleet_debug_marker.py')
    rviz_clean_script = os.path.join(bringup_share, 'scripts', 'run_rviz2_clean.bash')
    default_rviz_config = os.path.join(bringup_share, 'rviz', 'fleet_debug.rviz')
    if not os.path.exists(marker_script):
        raise RuntimeError(f'Missing fleet debug marker script: {marker_script}')
    if not os.path.exists(rviz_clean_script):
        raise RuntimeError(f'Missing clean RViz launcher script: {rviz_clean_script}')
    if not os.path.exists(default_rviz_config):
        raise RuntimeError(f'Missing fleet RViz config: {default_rviz_config}')

    marker_node = ExecuteProcess(
        cmd=[
            'python3', marker_script, '--ros-args',
            '-r', '__node:=fleet_debug_marker',
            '-p', 'use_sim_time:=true',
            '-p', 'waffle_pose_topic:=/leader_pose',
            '-p', 'burger_pose_topic:=/burger_pose',
            '-p', 'marker_topic:=/fleet_debug_markers',
            '-p', 'frame_id:=map',
        ],
        output='screen',
        name='fleet_debug_marker_v41',
    )

    rviz = ExecuteProcess(
        cmd=[
            rviz_clean_script,
            '-d', rviz_config,
            '--ros-args',
            '-r', '__node:=rviz2_domain25_fleet_debug',
            '-p', 'use_sim_time:=true',
        ],
        output='screen',
        name='rviz2_domain25_fleet_debug',
    )

    return LaunchDescription([
        DeclareLaunchArgument('domain_id', default_value='25'),
        DeclareLaunchArgument('rviz_config', default_value=default_rviz_config),
        SetEnvironmentVariable('ROS_DOMAIN_ID', domain_id),
        SetEnvironmentVariable('RMW_IMPLEMENTATION', 'rmw_fastrtps_cpp'),
        SetEnvironmentVariable('FASTDDS_BUILTIN_TRANSPORTS', 'UDPv4'),
        SetEnvironmentVariable('ROS_AUTOMATIC_DISCOVERY_RANGE', 'LOCALHOST'),
        LogInfo(msg='V55_FLEET_RVIZ | Aggregated RViz on Domain25. Shows /map, /leader_pose, bridged /burger_pose markers. Do not bridge raw /tf.'),
        marker_node,
        rviz,
    ])
