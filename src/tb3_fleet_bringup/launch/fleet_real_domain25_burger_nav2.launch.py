#!/usr/bin/env python3

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    bringup_share = get_package_share_directory('tb3_fleet_bringup')
    leader_launch = os.path.join(
        bringup_share, 'launch', 'fleet_real_domain25_waffle_nav2.launch.py'
    )

    domain_id = LaunchConfiguration('domain_id')

    return LaunchDescription([
        DeclareLaunchArgument('domain_id', default_value='25'),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(leader_launch),
            launch_arguments={
                'domain_id': domain_id,
                'robot_model': 'burger',
            }.items(),
        ),
    ])
