#!/usr/bin/env python3
"""PC-side tools for the two-robot system test.

The scout owns the Bayesian risk map. This launch only starts the PC-side YOLO
HTTP server and the unified RViz/debug viewer on the current ROS domain.
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
        FindPackageShare('tb3_flask_yolo_bridge'),
        'launch',
        'flask_yolo_server.launch.py',
    ])
    viewer_launch = PathJoinSubstitution([
        FindPackageShare('tb3_system_bringup'),
        'launch',
        'viewer.launch.py',
    ])

    return LaunchDescription([
        DeclareLaunchArgument('start_yolo_server', default_value='true'),
        DeclareLaunchArgument('start_viewer', default_value='true'),
        DeclareLaunchArgument(
            'domain_id',
            default_value=EnvironmentVariable('ROS_DOMAIN_ID'),
            description='PC DDS domain. Usually the same shell ROS_DOMAIN_ID as the leader domain.',
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
