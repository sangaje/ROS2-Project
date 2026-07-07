#!/usr/bin/env python3
"""PC-side fleet visualization for leader-published debug topics."""

from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, LogInfo
from launch.substitutions import EnvironmentVariable, LaunchConfiguration

from fleet_bringup.launch_utils import dds_launch_environment


def generate_launch_description():
    package_share = Path(get_package_share_directory('fleet_bringup'))
    domain_id = LaunchConfiguration('domain_id')

    return LaunchDescription([
        DeclareLaunchArgument(
            'domain_id',
            default_value=EnvironmentVariable('ROS_DOMAIN_ID'),
            description='PC DDS domain receiving leader debug topics.',
        ),
        *dds_launch_environment(domain_id),
        LogInfo(msg=['FLEET_RVIZ | domain=', domain_id]),
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
