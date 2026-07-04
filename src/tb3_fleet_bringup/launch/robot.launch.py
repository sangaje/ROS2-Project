#!/usr/bin/env python3

import os
from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    LogInfo,
    OpaqueFunction,
    SetEnvironmentVariable,
    UnsetEnvironmentVariable,
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch.substitutions import EnvironmentVariable
from launch_ros.actions import Node


def generate_launch_description():
    role      = LaunchConfiguration('role')
    domain_id_arg = LaunchConfiguration('domain_id')
    lds_model = LaunchConfiguration('lds_model')
    usb_port  = LaunchConfiguration('usb_port')
    lidar_port = LaunchConfiguration('lidar_port')
    start_state_publisher = LaunchConfiguration('start_state_publisher')
    start_lidar = LaunchConfiguration('start_lidar')
    start_base = LaunchConfiguration('start_base')
    bringup_impl = LaunchConfiguration('bringup_impl')

    domain_id = PythonExpression([
        "'", domain_id_arg, "' if '", domain_id_arg,
        "' else ('25' if '", role, "' == 'leader' else '24')",
    ])

    tb3_bringup_share = Path(get_package_share_directory('turtlebot3_bringup'))
    official_robot_launch = str(tb3_bringup_share / 'launch' / 'robot.launch.py')
    state_publisher_launch = str(tb3_bringup_share / 'launch' / 'turtlebot3_state_publisher.launch.py')
    tb3_param_file = str(tb3_bringup_share / 'param' / 'burger.yaml')

    def enabled(value):
        return value.strip().lower() in ('true', '1', 'yes', 'on')

    def resolve_lidar_port(value):
        port = value.strip()
        if port and port.lower() != 'auto':
            return port

        by_id = Path('/dev/serial/by-id')
        if by_id.exists():
            preferred = []
            fallback = []
            for candidate in sorted(by_id.iterdir()):
                name = candidate.name.lower()
                resolved_name = candidate.resolve().name
                target = str(candidate)
                if not resolved_name.startswith('ttyUSB'):
                    continue
                if any(key in name for key in ('ld', 'lidar', 'cp210', 'silicon', 'usb-serial', 'uart')):
                    preferred.append(target)
                else:
                    fallback.append(target)
            if preferred:
                return preferred[0]
            if fallback:
                return fallback[0]

        for candidate in ('/dev/ttyUSB0', '/dev/ttyUSB1', '/dev/ttyACM1'):
            if Path(candidate).exists():
                return candidate
        return '/dev/ttyUSB0'

    def make_lidar(context, *args, **kwargs):
        model = lds_model.perform(context)
        port = resolve_lidar_port(lidar_port.perform(context))

        if model == 'LDS-02':
            lidar_launch = Path(get_package_share_directory('ld08_driver')) / 'launch' / 'ld08.launch.py'
        elif model == 'LDS-03':
            lidar_launch = Path(get_package_share_directory('coin_d4_driver')) / 'launch' / 'single_lidar_node.launch.py'
        else:
            lidar_launch = Path(get_package_share_directory('hls_lfcd_lds_driver')) / 'launch' / 'hlds_laser.launch.py'

        return [
            LogInfo(msg=['REAL_BURGER_LIDAR | model=', model, ' port=', port, ' frame=base_scan']),
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(str(lidar_launch)),
                launch_arguments={
                    'port': port,
                    'frame_id': 'base_scan',
                    'namespace': '',
                }.items(),
            ),
        ]

    def make_robot_bringup(context, *args, **kwargs):
        impl = bringup_impl.perform(context).strip().lower()
        state_on = enabled(start_state_publisher.perform(context))
        lidar_on = enabled(start_lidar.perform(context))
        base_on = enabled(start_base.perform(context))
        model = lds_model.perform(context).strip()
        lidar_port_value = resolve_lidar_port(lidar_port.perform(context))

        os.environ['TURTLEBOT3_MODEL'] = 'burger'
        os.environ['LDS_MODEL'] = model

        if impl == 'official' and state_on and lidar_on and base_on and lidar_port_value == '/dev/ttyUSB0':
            return [
                LogInfo(msg='REAL_BURGER_BASE | using turtlebot3_bringup robot.launch.py'),
                IncludeLaunchDescription(
                    PythonLaunchDescriptionSource(official_robot_launch),
                    launch_arguments={
                        'use_sim_time': 'false',
                        'usb_port': usb_port,
                        'namespace': '',
                    }.items(),
                ),
            ]

        actions = [LogInfo(msg='REAL_BURGER_BASE | using split bringup path')]

        if state_on:
            actions.append(IncludeLaunchDescription(
                PythonLaunchDescriptionSource(state_publisher_launch),
                launch_arguments={'use_sim_time': 'false', 'namespace': ''}.items(),
            ))
        else:
            actions.append(LogInfo(msg='REAL_BURGER_STATE_PUBLISHER | disabled'))

        if lidar_on:
            actions.extend(make_lidar(context))
        else:
            actions.append(LogInfo(msg='REAL_BURGER_LIDAR | disabled'))

        if base_on:
            actions.append(Node(
                package='turtlebot3_node',
                executable='turtlebot3_ros',
                parameters=[tb3_param_file, {'namespace': ''}],
                arguments=['-i', usb_port],
                output='screen',
            ))
        else:
            actions.append(LogInfo(msg='REAL_BURGER_BASE_DRIVER | disabled'))

        return actions

    return LaunchDescription([
        DeclareLaunchArgument(
            'role', default_value='leader',
            description='Default domain selector when domain_id is empty.',
        ),
        DeclareLaunchArgument('domain_id', default_value='',
                              description='Explicit ROS_DOMAIN_ID. Empty uses role default.'),
        DeclareLaunchArgument('lds_model',
                              default_value=EnvironmentVariable('LDS_MODEL', default_value='LDS-02'),
                              description='LDS-01, LDS-02, or LDS-03. Defaults to the robot LDS_MODEL environment variable.'),
        DeclareLaunchArgument('usb_port', default_value='/dev/ttyACM0'),
        DeclareLaunchArgument('lidar_port',
                              default_value=EnvironmentVariable('LIDAR_PORT', default_value='auto'),
                              description='LiDAR serial port, or auto to scan /dev/serial/by-id and ttyUSB*.'),
        DeclareLaunchArgument('start_state_publisher', default_value='true'),
        DeclareLaunchArgument('start_lidar', default_value='true'),
        DeclareLaunchArgument('start_base', default_value='true',
                              description='Start turtlebot3_node/OpenCR base driver.'),
        DeclareLaunchArgument('bringup_impl', default_value='official',
                              description='official uses turtlebot3_bringup robot.launch.py when all robot parts are enabled; split allows per-part toggles.'),
        UnsetEnvironmentVariable('ROS_DISCOVERY_SERVER'),
        UnsetEnvironmentVariable('ROS_LOCALHOST_ONLY'),
        UnsetEnvironmentVariable('FASTRTPS_DEFAULT_PROFILES_FILE'),
        UnsetEnvironmentVariable('FASTDDS_DEFAULT_PROFILES_FILE'),
        SetEnvironmentVariable('ROS_DOMAIN_ID',               domain_id),
        SetEnvironmentVariable('ROS_AUTOMATIC_DISCOVERY_RANGE', 'SUBNET'),
        SetEnvironmentVariable('ROS_LOCALHOST_ONLY',           '0'),
        SetEnvironmentVariable('RMW_IMPLEMENTATION',          'rmw_fastrtps_cpp'),
        SetEnvironmentVariable('TURTLEBOT3_MODEL', 'burger'),
        SetEnvironmentVariable('LDS_MODEL', lds_model),
        LogInfo(msg=['REAL_BURGER_BASE | role=', role,
                     ' domain=', domain_id, ' lds=', lds_model,
                     ' opencr=', usb_port,
                     ' | state_pub=', start_state_publisher,
                     ' lidar=', start_lidar,
                     ' base=', start_base]),
        OpaqueFunction(function=make_robot_bringup),
    ])
