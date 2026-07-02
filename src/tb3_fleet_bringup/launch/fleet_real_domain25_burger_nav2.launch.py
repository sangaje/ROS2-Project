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

    domain_id   = LaunchConfiguration('domain_id')
    use_slam    = LaunchConfiguration('use_slam')
    initial_x   = LaunchConfiguration('initial_x')
    initial_y   = LaunchConfiguration('initial_y')
    initial_yaw = LaunchConfiguration('initial_yaw')

    return LaunchDescription([
        DeclareLaunchArgument('domain_id',   default_value='25'),
        DeclareLaunchArgument(
            'use_slam', default_value='true',
            description='true=Cartographer SLAM; false=AMCL with map from other robot.',
        ),
        DeclareLaunchArgument(
            'initial_x', default_value='1.05',
            description='[AMCL only] Leader x in follower SLAM map frame.',
        ),
        DeclareLaunchArgument('initial_y',   default_value='0.0'),
        DeclareLaunchArgument('initial_yaw', default_value='0.0'),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(leader_launch),
            launch_arguments={
                'domain_id':   domain_id,
                'robot_model': 'burger',
                'use_slam':    use_slam,
                'initial_x':   initial_x,
                'initial_y':   initial_y,
                'initial_yaw': initial_yaw,
            }.items(),
        ),
    ])
