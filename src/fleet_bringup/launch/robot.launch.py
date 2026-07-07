#!/usr/bin/env python3
"""Standalone TurtleBot3 hardware bringup using the fleet DDS policy."""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import EnvironmentVariable, LaunchConfiguration

from fleet_bringup.launch_utils import dds_launch_environment


def generate_launch_description():
    domain_id = LaunchConfiguration('domain_id')
    upstream_launch = os.path.join(
        get_package_share_directory('turtlebot3_bringup'),
        'launch',
        'robot.launch.py',
    )
    return LaunchDescription([
        DeclareLaunchArgument(
            'domain_id',
            default_value=EnvironmentVariable('ROS_DOMAIN_ID'),
            description='DDS domain for this robot.',
        ),
        *dds_launch_environment(domain_id),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(upstream_launch),
            launch_arguments={
                'use_sim_time': 'false',
                'namespace': '',
            }.items(),
        ),
    ])
