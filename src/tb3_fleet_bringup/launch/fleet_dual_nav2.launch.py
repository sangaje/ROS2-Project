#!/usr/bin/env python3

import os

from ament_index_python.packages import PackageNotFoundError, get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, LogInfo, TimerAction
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _default_map() -> str:
    try:
        tb3_nav_share = get_package_share_directory('turtlebot3_navigation2')
        path = os.path.join(tb3_nav_share, 'map', 'map.yaml')
        if os.path.exists(path):
            return path
    except PackageNotFoundError:
        pass

    try:
        nav2_share = get_package_share_directory('nav2_bringup')
        for rel_path in ('maps/tb3_sandbox.yaml', 'maps/depot.yaml', 'maps/warehouse.yaml'):
            path = os.path.join(nav2_share, rel_path)
            if os.path.exists(path):
                return path
    except PackageNotFoundError:
        pass

    return ''


def _nav2_bringup(robot_name: str, params_file: str, map_yaml, delay: float) -> TimerAction:
    nav2_share = get_package_share_directory('nav2_bringup')
    bringup_launch = os.path.join(nav2_share, 'launch', 'bringup_launch.py')

    return TimerAction(
        period=delay,
        actions=[
            LogInfo(msg=f'START_NAV2 | robot={robot_name} action=/{robot_name}/navigate_to_pose'),
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(bringup_launch),
                launch_arguments={
                    'namespace': robot_name,
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


def _goal_proxy(robot_name: str, delay: float) -> TimerAction:
    return TimerAction(
        period=delay,
        actions=[
            Node(
                package='tb3_fleet_robot',
                executable='robot_goal_proxy',
                name=f'{robot_name}_goal_proxy',
                output='screen',
                parameters=[{
                    'use_sim_time': True,
                    'robot_name': robot_name,
                    'navigate_action': f'/{robot_name}/navigate_to_pose',
                    'cancel_previous_goal': True,
                    'ignore_duplicate_goals': True,
                    'same_goal_xy_tolerance_m': 0.05,
                    'same_goal_yaw_tolerance_rad': 0.08,
                    'min_resend_period_sec': 1.0,
                    'wait_for_server_sec': 60.0,
                }],
            ),
            Node(
                package='tb3_fleet_robot',
                executable='robot_pose_reporter',
                name=f'{robot_name}_pose_reporter',
                output='screen',
                parameters=[{
                    'use_sim_time': True,
                    'robot_name': robot_name,
                    'map_frame': 'map',
                    'base_frame': f'{robot_name}/base_footprint',
                    'publish_period_sec': 0.20,
                }],
            ),
        ],
    )


def generate_launch_description():
    bringup_share = get_package_share_directory('tb3_fleet_bringup')
    spawn_launch = os.path.join(bringup_share, 'launch', 'fleet_two_burger_spawn_with_ros_bridge.launch.py')
    rviz_config = os.path.join(bringup_share, 'rviz', 'fleet_dual_nav2.rviz')
    burger_params = os.path.join(bringup_share, 'config', 'burger_nav2.yaml')
    waffle_params = os.path.join(bringup_share, 'config', 'waffle_nav2.yaml')

    map_yaml = LaunchConfiguration('map')
    use_rviz = LaunchConfiguration('rviz')
    spacing = LaunchConfiguration('spacing')
    formation_type = LaunchConfiguration('formation_type')

    burger_x = LaunchConfiguration('burger_x')
    burger_y = LaunchConfiguration('burger_y')
    burger_yaw = LaunchConfiguration('burger_yaw')
    waffle_x = LaunchConfiguration('waffle_x')
    waffle_y = LaunchConfiguration('waffle_y')
    waffle_yaw = LaunchConfiguration('waffle_yaw')

    spawn_and_bridge = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(spawn_launch),
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
        name='dual_nav2_frame_tools',
        output='screen',
        parameters=[{
            'use_sim_time': True,
            'robot_names': 'burger,waffle',
            'initial_xs': [burger_x, ',', waffle_x],
            'initial_ys': [burger_y, ',', waffle_y],
            'initial_yaws': [burger_yaw, ',', waffle_yaw],
            'initial_pose_repeat_count': 80,
            'initial_pose_period_sec': 0.25,
            'log_every_n_odom': 100,
            'log_every_n_scan': 200,
        }],
    )

    commander = Node(
        package='tb3_fleet_master',
        executable='fleet_commander_node',
        name='fleet_commander_node',
        output='screen',
        parameters=[{
            'use_sim_time': True,
            'robot_names': 'waffle,burger',
            'spacing': spacing,
            'formation_type': formation_type,
            'frame_id': 'map',
            'republish_count': 3,
            'republish_period_sec': 0.10,
        }],
    )

    state_echo = Node(
        package='tb3_fleet_master',
        executable='fleet_state_echo',
        name='fleet_state_echo',
        output='screen',
        parameters=[{
            'use_sim_time': True,
            'robot_names': 'waffle,burger',
            'print_period_sec': 2.0,
        }],
    )

    rviz = Node(
        package='rviz2',
        executable='rviz2',
        name='fleet_dual_nav2_rviz',
        output='screen',
        arguments=['-d', rviz_config],
        parameters=[{'use_sim_time': True}],
        condition=IfCondition(use_rviz),
    )

    return LaunchDescription([
        DeclareLaunchArgument('map', default_value=_default_map()),
        DeclareLaunchArgument('rviz', default_value='true'),
        DeclareLaunchArgument('spacing', default_value='0.85'),
        DeclareLaunchArgument('formation_type', default_value='column'),
        DeclareLaunchArgument('burger_x', default_value='-2.9'),
        DeclareLaunchArgument('burger_y', default_value='0.5'),
        DeclareLaunchArgument('burger_yaw', default_value='0.0'),
        DeclareLaunchArgument('waffle_x', default_value='-1.8'),
        DeclareLaunchArgument('waffle_y', default_value='0.5'),
        DeclareLaunchArgument('waffle_yaw', default_value='0.0'),
        LogInfo(msg='DUAL_NAV2_FLEET | Gazebo + /waffle Nav2 + /burger Nav2 + fleet group goal + RViz'),
        LogInfo(msg='RViz tool "Fleet Group Goal" publishes /fleet/group_goal. Waffle takes the clicked pose; Burger keeps the formation slot.'),
        LogInfo(msg=['Map yaml: ', map_yaml]),
        spawn_and_bridge,
        TimerAction(period=3.5, actions=[frame_tools]),
        _nav2_bringup('waffle', waffle_params, map_yaml, delay=7.0),
        _nav2_bringup('burger', burger_params, map_yaml, delay=9.0),
        TimerAction(period=12.0, actions=[commander, state_echo]),
        _goal_proxy('waffle', delay=15.0),
        _goal_proxy('burger', delay=15.5),
        TimerAction(period=16.0, actions=[rviz]),
    ])
