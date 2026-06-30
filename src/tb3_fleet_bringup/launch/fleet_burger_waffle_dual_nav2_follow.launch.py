#!/usr/bin/env python3

import os

from ament_index_python.packages import get_package_share_directory, PackageNotFoundError
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, TimerAction, LogInfo
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _default_map() -> str:
    try:
        tb3_nav_share = get_package_share_directory('turtlebot3_navigation2')
        p = os.path.join(tb3_nav_share, 'map', 'map.yaml')
        if os.path.exists(p):
            return p
    except PackageNotFoundError:
        pass

    try:
        nav2_share = get_package_share_directory('nav2_bringup')
        for rel in ['maps/tb3_sandbox.yaml', 'maps/depot.yaml', 'maps/warehouse.yaml']:
            p = os.path.join(nav2_share, rel)
            if os.path.exists(p):
                return p
    except PackageNotFoundError:
        pass
    return ''


def _nav2_bringup_group(robot: str, params_file: str, map_yaml, delay: float):
    nav2_share = get_package_share_directory('nav2_bringup')
    bringup_launch = os.path.join(nav2_share, 'launch', 'bringup_launch.py')

    # Use the official Nav2 bringup launch instead of manually spawning lifecycle nodes.
    # This is more stable on Jazzy and reliably creates /<robot>/navigate_to_pose.
    return TimerAction(
        period=delay,
        actions=[
            LogInfo(msg=f'V21_START_NAV2_BRINGUP | robot={robot} | expected_action=/{robot}/navigate_to_pose'),
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(bringup_launch),
                launch_arguments={
                    'namespace': robot,
                    'use_namespace': 'True',
                    'slam': 'False',
                    'map': map_yaml,
                    'use_sim_time': 'True',
                    'params_file': params_file,
                    'autostart': 'True',
                    'use_composition': 'False',
                    'use_respawn': 'False',
                    'log_level': 'info',
                }.items(),
            ),
        ],
    )


def generate_launch_description():
    bringup_share = get_package_share_directory('tb3_fleet_bringup')
    spawn_bridge_launch = os.path.join(bringup_share, 'launch', 'fleet_two_burger_spawn_with_ros_bridge.launch.py')

    burger_params = os.path.join(bringup_share, 'config', 'burger_nav2.yaml')
    waffle_params = os.path.join(bringup_share, 'config', 'waffle_nav2.yaml')

    map_yaml = LaunchConfiguration('map')
    burger_x = LaunchConfiguration('burger_x')
    burger_y = LaunchConfiguration('burger_y')
    burger_yaw = LaunchConfiguration('burger_yaw')
    waffle_x = LaunchConfiguration('waffle_x')
    waffle_y = LaunchConfiguration('waffle_y')
    waffle_yaw = LaunchConfiguration('waffle_yaw')

    spawn_and_bridge = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(spawn_bridge_launch),
        launch_arguments={
            'burger_x': burger_x,
            'burger_y': burger_y,
            'burger_yaw': burger_yaw,
            'waffle_x': waffle_x,
            'waffle_y': waffle_y,
            'waffle_yaw': waffle_yaw,
        }.items(),
    )

    frame_tools = Node(
        package='tb3_fleet_bringup',
        executable='nav2_frame_tools',
        name='nav2_frame_tools',
        output='screen',
        parameters=[{
            'robot_names': 'burger,waffle',
            'initial_xs': '-2.9,-1.8',
            'initial_ys': '0.5,0.5',
            'initial_yaws': '0.0,0.0',
            'initial_pose_repeat_count': 80,
            'initial_pose_period_sec': 0.25,
            'log_every_n_odom': 50,
            'log_every_n_scan': 100,
        }],
    )

    nav2_follower = Node(
        package='tb3_fleet_bringup',
        executable='waffle_burger_nav2_follower',
        name='waffle_burger_nav2_follower',
        output='screen',
        parameters=[{
            'leader_name': 'waffle',
            'follower_name': 'burger',
            'follow_distance': 1.05,
            'min_distance': 0.55,
            'goal_period_sec': 1.5,
            'goal_update_distance': 0.20,
        }],
    )

    return LaunchDescription([
        DeclareLaunchArgument('map', default_value=_default_map()),
        DeclareLaunchArgument('burger_x', default_value='-2.9'),
        DeclareLaunchArgument('burger_y', default_value='0.5'),
        DeclareLaunchArgument('burger_yaw', default_value='0.0'),
        DeclareLaunchArgument('waffle_x', default_value='-1.8'),
        DeclareLaunchArgument('waffle_y', default_value='0.5'),
        DeclareLaunchArgument('waffle_yaw', default_value='0.0'),

        LogInfo(msg='V21_DUAL_NAV2_BRINGUP_FIXED | official nav2_bringup per namespace'),
        LogInfo(msg='WAIT 30~45 seconds, then check: ros2 action list | grep navigate_to_pose'),
        LogInfo(msg='Expected actions: /waffle/navigate_to_pose and /burger/navigate_to_pose'),
        LogInfo(msg=['Map yaml: ', map_yaml]),
        spawn_and_bridge,
        TimerAction(period=3.5, actions=[frame_tools]),
        _nav2_bringup_group('waffle', waffle_params, map_yaml, delay=7.0),
        _nav2_bringup_group('burger', burger_params, map_yaml, delay=9.0),
        TimerAction(period=25.0, actions=[nav2_follower]),
    ])
