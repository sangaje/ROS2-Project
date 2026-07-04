#!/usr/bin/env python3

from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch.conditions import IfCondition
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

    domain_id = PythonExpression([
        "'", domain_id_arg, "' if '", domain_id_arg,
        "' else ('25' if '", role, "' == 'leader' else '24')",
    ])

    tb3_bringup_share = Path(get_package_share_directory('turtlebot3_bringup'))
    state_publisher_launch = str(tb3_bringup_share / 'launch' / 'turtlebot3_state_publisher.launch.py')
    tb3_param_file = str(tb3_bringup_share / 'param' / 'burger.yaml')

    def make_lidar(context, *args, **kwargs):
        if start_lidar.perform(context).lower() not in ('true', '1', 'yes', 'on'):
            return [LogInfo(msg='REAL_BURGER_LIDAR | disabled')]

        model = lds_model.perform(context)
        port = lidar_port.perform(context)

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
                    'port': lidar_port,
                    'frame_id': 'base_scan',
                    'namespace': '',
                }.items(),
            ),
        ]

    state_publisher = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(state_publisher_launch),
        launch_arguments={'use_sim_time': 'false', 'namespace': ''}.items(),
        condition=IfCondition(start_state_publisher),
    )
    turtlebot3_node = Node(
        package='turtlebot3_node',
        executable='turtlebot3_ros',
        parameters=[tb3_param_file, {'namespace': ''}],
        arguments=['-i', usb_port],
        output='screen',
        condition=IfCondition(start_base),
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            'role', default_value='leader',
            description='Default domain selector when domain_id is empty.',
        ),
        DeclareLaunchArgument('domain_id', default_value='',
                              description='Explicit ROS_DOMAIN_ID. Empty uses role default.'),
        DeclareLaunchArgument('lds_model',
                              default_value=EnvironmentVariable('LDS_MODEL', default_value='LDS-01'),
                              description='LDS-01, LDS-02, or LDS-03. Defaults to the robot LDS_MODEL environment variable.'),
        DeclareLaunchArgument('usb_port', default_value='/dev/ttyACM0'),
        DeclareLaunchArgument('lidar_port', default_value='/dev/ttyUSB0',
                              description='LiDAR serial port. Check with: ls -l /dev/serial/by-id/'),
        DeclareLaunchArgument('start_state_publisher', default_value='true'),
        DeclareLaunchArgument('start_lidar', default_value='true'),
        DeclareLaunchArgument('start_base', default_value='true',
                              description='Start turtlebot3_node/OpenCR base driver.'),
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
        state_publisher,
        OpaqueFunction(function=make_lidar),
        turtlebot3_node,
    ])
