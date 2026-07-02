#!/usr/bin/env python3

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    LogInfo,
    SetEnvironmentVariable,
    UnsetEnvironmentVariable,
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PythonExpression


def generate_launch_description():
    role = LaunchConfiguration('role')
    lds_model = LaunchConfiguration('lds_model')
    usb_port = LaunchConfiguration('usb_port')
    domain_id = PythonExpression([
        "'25' if '", role, "' == 'leader' else '24'",
    ])

    robot_launch = os.path.join(
        get_package_share_directory('turtlebot3_bringup'),
        'launch',
        'robot.launch.py',
    )

    pc_ip = LaunchConfiguration('pc_ip')

    return LaunchDescription([
        DeclareLaunchArgument(
            'role',
            default_value='leader',
            description='leader uses Domain25; follower uses Domain24.',
        ),
        DeclareLaunchArgument(
            'lds_model',
            default_value='LDS-01',
            description='Set LDS-01, LDS-02, or LDS-03 to match the robot.',
        ),
        DeclareLaunchArgument('usb_port', default_value='/dev/ttyACM0'),
        DeclareLaunchArgument(
            'pc_ip', default_value='',
            description='PC IP address for unicast DDS discovery (e.g. 10.10.14.5). '
                        'Leave empty to rely on multicast only.',
        ),
        UnsetEnvironmentVariable('ROS_DISCOVERY_SERVER'),
        UnsetEnvironmentVariable('ROS_LOCALHOST_ONLY'),
        UnsetEnvironmentVariable('FASTRTPS_DEFAULT_PROFILES_FILE'),
        UnsetEnvironmentVariable('FASTDDS_DEFAULT_PROFILES_FILE'),
        SetEnvironmentVariable('ROS_DOMAIN_ID', domain_id),
        SetEnvironmentVariable('RMW_IMPLEMENTATION', 'rmw_fastrtps_cpp'),
        SetEnvironmentVariable('FASTDDS_BUILTIN_TRANSPORTS', 'UDPv4'),
        SetEnvironmentVariable('ROS_AUTOMATIC_DISCOVERY_RANGE', 'SUBNET'),
        SetEnvironmentVariable('ROS_STATIC_PEERS', pc_ip),
        SetEnvironmentVariable('TURTLEBOT3_MODEL', 'burger'),
        SetEnvironmentVariable('LDS_MODEL', lds_model),
        LogInfo(
            msg=[
                'REAL_BURGER_BASE | role=', role,
                ' domain=', domain_id,
                ' lds=', lds_model,
            ]
        ),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(robot_launch),
            launch_arguments={
                'use_sim_time': 'false',
                'usb_port': usb_port,
            }.items(),
        ),
    ])
