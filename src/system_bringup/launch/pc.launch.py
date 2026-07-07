#!/usr/bin/env python3
"""PC-side tools for the fleet system.

The PC only starts visualization/client tools on its own DDS domain. Leader
system bringup owns debug aggregation and bridges selected topics here.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import (
    EnvironmentVariable,
    LaunchConfiguration,
    PathJoinSubstitution,
)
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    yolo_server_launch = PathJoinSubstitution([
        FindPackageShare('flask_yolo_bridge'),
        'launch',
        'flask_yolo_server.launch.py',
    ])
    viewer_launch = PathJoinSubstitution([
        FindPackageShare('system_bringup'),
        'launch',
        'viewer.launch.py',
    ])

    return LaunchDescription([
        DeclareLaunchArgument('start_yolo_server', default_value='false'),
        DeclareLaunchArgument('start_viewer', default_value='true'),
        DeclareLaunchArgument(
            'domain_id',
            default_value=EnvironmentVariable('ROS_DOMAIN_ID'),
            description='PC DDS domain. Set this shell to pc_domain_id.',
        ),

        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(yolo_server_launch),
            condition=IfCondition(LaunchConfiguration('start_yolo_server')),
        ),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(viewer_launch),
            condition=IfCondition(LaunchConfiguration('start_viewer')),
            launch_arguments={
                'domain_id': LaunchConfiguration('domain_id'),
            }.items(),
        ),
    ])
