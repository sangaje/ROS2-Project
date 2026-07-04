#!/usr/bin/env python3

from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    LogInfo,
    SetEnvironmentVariable,
    UnsetEnvironmentVariable,
)
from launch.substitutions import EnvironmentVariable, LaunchConfiguration


def generate_launch_description():
    bringup_share = get_package_share_directory('tb3_fleet_bringup')

    domain_id   = LaunchConfiguration('domain_id')
    rviz_config = LaunchConfiguration('rviz_config')

    default_rviz_config = str(Path(bringup_share) / 'rviz' / 'fleet_debug.rviz')
    marker_script       = str(Path(bringup_share) / 'scripts' / 'fleet_debug_marker.py')
    rviz_clean_script   = str(Path(bringup_share) / 'scripts' / 'run_rviz2_clean.bash')

    marker = ExecuteProcess(
        cmd=[
            'python3', marker_script, '--ros-args',
            '-r', '__node:=fleet_debug_marker',
            '-p', 'use_sim_time:=false',
            '-p', 'waffle_pose_topic:=/leader_pose',
            '-p', 'burger_pose_topic:=/burger_pose',
            '-p', 'marker_topic:=/fleet_debug_markers',
            '-p', 'frame_id:=map',
        ],
        output='screen',
        name='fleet_debug_marker',
    )
    rviz = ExecuteProcess(
        cmd=[
            rviz_clean_script,
            '-d', rviz_config,
            '--ros-args',
            '-r', '__node:=rviz2_fleet',
            '-p', 'use_sim_time:=false',
        ],
        output='screen',
        name='rviz2_fleet',
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            'domain_id',
            default_value=EnvironmentVariable('ROS_DOMAIN_ID', default_value='25'),
        ),
        DeclareLaunchArgument('rviz_config',  default_value=default_rviz_config),
        UnsetEnvironmentVariable('ROS_DISCOVERY_SERVER'),
        UnsetEnvironmentVariable('ROS_LOCALHOST_ONLY'),
        UnsetEnvironmentVariable('FASTRTPS_DEFAULT_PROFILES_FILE'),
        UnsetEnvironmentVariable('FASTDDS_DEFAULT_PROFILES_FILE'),
        SetEnvironmentVariable('ROS_DOMAIN_ID',               domain_id),
        SetEnvironmentVariable('ROS_AUTOMATIC_DISCOVERY_RANGE', 'SUBNET'),
        SetEnvironmentVariable('ROS_LOCALHOST_ONLY',           '0'),
        SetEnvironmentVariable('RMW_IMPLEMENTATION',          'rmw_fastrtps_cpp'),
        LogInfo(msg=['FLEET_RVIZ | domain=', domain_id,
                     ' | /goal_pose -> leader | /fleet/follow_command -> follower']),
        marker,
        rviz,
    ])
