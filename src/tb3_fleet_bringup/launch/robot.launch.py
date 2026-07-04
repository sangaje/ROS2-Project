#!/usr/bin/env python3
from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument, IncludeLaunchDescription,
    OpaqueFunction, SetEnvironmentVariable, UnsetEnvironmentVariable,
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import EnvironmentVariable, LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    lds_model  = LaunchConfiguration('lds_model')
    usb_port   = LaunchConfiguration('usb_port')
    lidar_port = LaunchConfiguration('lidar_port')

    tb3_share  = Path(get_package_share_directory('turtlebot3_bringup'))
    state_pub  = str(tb3_share / 'launch' / 'turtlebot3_state_publisher.launch.py')
    tb3_params = str(tb3_share / 'param' / 'burger.yaml')

    def make_robot(context, *args, **kwargs):
        model = lds_model.perform(context).strip().upper()
        port  = lidar_port.perform(context).strip()

        if model == 'LDS-02':
            lidar_launch = str(Path(get_package_share_directory('ld08_driver'))
                               / 'launch' / 'ld08.launch.py')
        elif model == 'LDS-03':
            lidar_launch = str(Path(get_package_share_directory('coin_d4_driver'))
                               / 'launch' / 'single_lidar_node.launch.py')
        else:  # LDS-01
            lidar_launch = str(Path(get_package_share_directory('hls_lfcd_lds_driver'))
                               / 'launch' / 'hlds_laser.launch.py')

        return [
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(state_pub),
                launch_arguments={'use_sim_time': 'false', 'namespace': ''}.items(),
            ),
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(lidar_launch),
                launch_arguments={'port': port, 'frame_id': 'base_scan'}.items(),
            ),
            Node(
                package='turtlebot3_node', executable='turtlebot3_ros',
                parameters=[tb3_params, {'namespace': ''}],
                arguments=['-i', usb_port],
                output='screen',
            ),
        ]

    return LaunchDescription([
        DeclareLaunchArgument('role',      default_value='leader'),
        DeclareLaunchArgument('domain_id', default_value=''),
        DeclareLaunchArgument('lds_model',
                              default_value=EnvironmentVariable('LDS_MODEL', default_value='LDS-01')),
        DeclareLaunchArgument('usb_port',  default_value='/dev/ttyACM0'),
        DeclareLaunchArgument('lidar_port',
                              default_value=EnvironmentVariable('LIDAR_PORT', default_value='/dev/ttyUSB0')),
        UnsetEnvironmentVariable('ROS_DISCOVERY_SERVER'),
        UnsetEnvironmentVariable('ROS_LOCALHOST_ONLY'),
        UnsetEnvironmentVariable('FASTRTPS_DEFAULT_PROFILES_FILE'),
        UnsetEnvironmentVariable('FASTDDS_DEFAULT_PROFILES_FILE'),
        SetEnvironmentVariable('ROS_AUTOMATIC_DISCOVERY_RANGE', 'SUBNET'),
        SetEnvironmentVariable('ROS_LOCALHOST_ONLY',            '0'),
        SetEnvironmentVariable('RMW_IMPLEMENTATION',            'rmw_fastrtps_cpp'),
        SetEnvironmentVariable('LDS_MODEL', lds_model),
        OpaqueFunction(function=make_robot),
    ])
