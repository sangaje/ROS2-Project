#!/usr/bin/env python3

import tempfile
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
from launch_ros.actions import Node


def generate_launch_description():
    role      = LaunchConfiguration('role')
    lds_model = LaunchConfiguration('lds_model')
    usb_port  = LaunchConfiguration('usb_port')
    lidar_port = LaunchConfiguration('lidar_port')
    pc_ip     = LaunchConfiguration('pc_ip')

    domain_id = PythonExpression(["'25' if '", role, "' == 'leader' else '24'"])

    tb3_bringup_share = Path(get_package_share_directory('turtlebot3_bringup'))
    state_publisher_launch = str(tb3_bringup_share / 'launch' / 'turtlebot3_state_publisher.launch.py')
    tb3_param_file = str(tb3_bringup_share / 'param' / 'burger.yaml')

    def make_fastdds_env(context, *args, **kwargs):
        p = pc_ip.perform(context)
        d = context.perform_substitution(domain_id)
        xml = f"""<?xml version="1.0" encoding="UTF-8"?>
<profiles xmlns="http://www.eprosima.com/XMLSchemas/fastRTPS_Profiles">
  <participant profile_name="default_profile" is_default_profile="true">
    <rtps>
      <builtin>
        <initialPeersList>
          <locator><udpv4><address>{p}</address></udpv4></locator>
        </initialPeersList>
      </builtin>
    </rtps>
  </participant>
</profiles>
"""
        xml_path = Path(tempfile.gettempdir()) / f'fastdds_robot_d{d}.xml'
        xml_path.write_text(xml, encoding='utf-8')
        return [
            SetEnvironmentVariable('FASTRTPS_DEFAULT_PROFILES_FILE', str(xml_path)),
            SetEnvironmentVariable('FASTDDS_DEFAULT_PROFILES_FILE', str(xml_path)),
            SetEnvironmentVariable('ROS_STATIC_PEERS', p),
        ]

    def make_lidar(context, *args, **kwargs):
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
    )
    turtlebot3_node = Node(
        package='turtlebot3_node',
        executable='turtlebot3_ros',
        parameters=[tb3_param_file, {'namespace': ''}],
        arguments=['-i', usb_port],
        output='screen',
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            'role', default_value='leader',
            description='leader uses Domain 25; follower uses Domain 24.',
        ),
        DeclareLaunchArgument('lds_model', default_value='LDS-01',
                              description='LDS-01, LDS-02, or LDS-03.'),
        DeclareLaunchArgument('usb_port', default_value='/dev/ttyACM0'),
        DeclareLaunchArgument('lidar_port', default_value='/dev/ttyUSB0',
                              description='LiDAR serial port. Check with: ls -l /dev/serial/by-id/'),
        DeclareLaunchArgument('pc_ip',    default_value='10.10.14.58',
                              description='PC IP for unicast DDS discovery.'),
        UnsetEnvironmentVariable('ROS_DISCOVERY_SERVER'),
        UnsetEnvironmentVariable('ROS_LOCALHOST_ONLY'),
        UnsetEnvironmentVariable('FASTRTPS_DEFAULT_PROFILES_FILE'),
        UnsetEnvironmentVariable('FASTDDS_DEFAULT_PROFILES_FILE'),
        SetEnvironmentVariable('ROS_DOMAIN_ID',               domain_id),
        SetEnvironmentVariable('RMW_IMPLEMENTATION',          'rmw_fastrtps_cpp'),
        SetEnvironmentVariable('FASTDDS_BUILTIN_TRANSPORTS',  'UDPv4'),
        SetEnvironmentVariable('ROS_AUTOMATIC_DISCOVERY_RANGE', 'SUBNET'),
        SetEnvironmentVariable('TURTLEBOT3_MODEL', 'burger'),
        SetEnvironmentVariable('LDS_MODEL', lds_model),
        OpaqueFunction(function=make_fastdds_env),
        LogInfo(msg=['REAL_BURGER_BASE | role=', role,
                     ' domain=', domain_id, ' lds=', lds_model,
                     ' opencr=', usb_port, ' pc=', pc_ip]),
        state_publisher,
        OpaqueFunction(function=make_lidar),
        turtlebot3_node,
    ])
