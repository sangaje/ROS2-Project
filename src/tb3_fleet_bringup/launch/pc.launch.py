#!/usr/bin/env python3
"""
Debugging PC — runs RViz on the leader domain (D25 by default).
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument, SetEnvironmentVariable, TimerAction,
    UnsetEnvironmentVariable,
)
from launch.substitutions import EnvironmentVariable, LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg = get_package_share_directory('tb3_fleet_bringup')
    domain_id = LaunchConfiguration('domain_id')

    rviz = Node(
        package='rviz2', executable='rviz2',
        name='rviz2', output='screen',
        arguments=['-d', os.path.join(pkg, 'rviz', 'fleet_debug.rviz')],
        additional_env={'ROS_DOMAIN_ID': domain_id},
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            'domain_id',
            default_value=EnvironmentVariable('ROS_DOMAIN_ID', default_value='25'),
            description='ROS domain; defaults to the shell ROS_DOMAIN_ID.',
        ),
        UnsetEnvironmentVariable('ROS_DISCOVERY_SERVER'),
        UnsetEnvironmentVariable('ROS_LOCALHOST_ONLY'),
        UnsetEnvironmentVariable('FASTRTPS_DEFAULT_PROFILES_FILE'),
        UnsetEnvironmentVariable('FASTDDS_DEFAULT_PROFILES_FILE'),
        SetEnvironmentVariable('ROS_DOMAIN_ID',                domain_id),
        SetEnvironmentVariable('ROS_AUTOMATIC_DISCOVERY_RANGE', 'SUBNET'),
        SetEnvironmentVariable('ROS_LOCALHOST_ONLY',           '0'),
        SetEnvironmentVariable('RMW_IMPLEMENTATION',           'rmw_fastrtps_cpp'),
        TimerAction(period=1.0, actions=[rviz]),
    ])
