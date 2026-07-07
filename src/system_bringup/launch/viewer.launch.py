#!/usr/bin/env python3
"""PC-side unified view for leader-published fleet/risk debug topics."""

from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, LogInfo
from launch.substitutions import EnvironmentVariable, LaunchConfiguration

from fleet_bringup.launch_utils import dds_launch_environment


def generate_launch_description():
    fleet_share = Path(get_package_share_directory('fleet_bringup'))
    system_share = Path(get_package_share_directory('system_bringup'))
    domain_id = LaunchConfiguration('domain_id')

    return LaunchDescription([
        DeclareLaunchArgument(
            'domain_id',
            default_value=EnvironmentVariable('ROS_DOMAIN_ID'),
            description='PC DDS domain that receives leader visualization/debug topics.',
        ),
        *dds_launch_environment(domain_id),
        LogInfo(msg=['SYSTEM_VIEWER | domain=', domain_id]),
        ExecuteProcess(
            cmd=[
                str(fleet_share / 'scripts' / 'run_rviz2_clean.bash'),
                '-d',
                str(system_share / 'rviz' / 'system_view.rviz'),
                '--ros-args',
                '-r',
                '__node:=rviz2_system',
            ],
            output='screen',
            name='rviz2_system',
        ),
    ])
