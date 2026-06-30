#!/usr/bin/env python3

import os
from typing import Iterable, Tuple

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, TimerAction, SetEnvironmentVariable, LogInfo
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _first_existing_model_sdf(tb3_gazebo_share: str, candidates: Iterable[str]) -> Tuple[str, str]:
    for model_dir_name in candidates:
        sdf = os.path.join(tb3_gazebo_share, 'models', model_dir_name, 'model.sdf')
        if os.path.exists(sdf):
            return model_dir_name, sdf
    tried = ', '.join(candidates)
    raise RuntimeError(f'No TurtleBot3 model.sdf found. Tried model directories: {tried}')


def generate_launch_description():
    tb3_gazebo_share = get_package_share_directory('turtlebot3_gazebo')
    default_world = os.path.join(tb3_gazebo_share, 'worlds', 'turtlebot3_world.world')
    burger_model_dir, burger_sdf = _first_existing_model_sdf(tb3_gazebo_share, ['turtlebot3_burger'])
    waffle_model_dir, waffle_sdf = _first_existing_model_sdf(tb3_gazebo_share, ['turtlebot3_waffle', 'turtlebot3_waffle_pi'])

    tb3_models_dir = os.path.join(tb3_gazebo_share, 'models')
    old_gz_resource_path = os.environ.get('GZ_SIM_RESOURCE_PATH', '')
    gz_resource_path_parts = [tb3_models_dir, tb3_gazebo_share]
    if old_gz_resource_path:
        gz_resource_path_parts.append(old_gz_resource_path)
    gz_resource_path = ':'.join(gz_resource_path_parts)

    world = LaunchConfiguration('world')
    burger_name = LaunchConfiguration('burger_name')
    waffle_name = LaunchConfiguration('waffle_name')

    burger_x = LaunchConfiguration('burger_x')
    burger_y = LaunchConfiguration('burger_y')
    burger_yaw = LaunchConfiguration('burger_yaw')

    waffle_x = LaunchConfiguration('waffle_x')
    waffle_y = LaunchConfiguration('waffle_y')
    waffle_yaw = LaunchConfiguration('waffle_yaw')

    gz_verbosity = LaunchConfiguration('gz_verbosity')

    gz_sim = ExecuteProcess(
        cmd=['gz', 'sim', '-r', '-v', gz_verbosity, world],
        output='screen',
        name='gz_sim_burger_waffle_world',
    )

    spawn_burger = Node(
        package='ros_gz_sim',
        executable='create',
        name='spawn_burger',
        output='screen',
        arguments=['-file', burger_sdf, '-name', burger_name, '-x', burger_x, '-y', burger_y, '-z', '0.05', '-Y', burger_yaw],
    )

    spawn_waffle = Node(
        package='ros_gz_sim',
        executable='create',
        name='spawn_waffle',
        output='screen',
        arguments=['-file', waffle_sdf, '-name', waffle_name, '-x', waffle_x, '-y', waffle_y, '-z', '0.05', '-Y', waffle_yaw],
    )

    return LaunchDescription([
        DeclareLaunchArgument('world', default_value=default_world),
        DeclareLaunchArgument('gz_verbosity', default_value='2'),
        DeclareLaunchArgument('burger_name', default_value='burger'),
        DeclareLaunchArgument('waffle_name', default_value='waffle'),
        DeclareLaunchArgument('burger_x', default_value='-2.0'),
        DeclareLaunchArgument('burger_y', default_value='-0.5'),
        DeclareLaunchArgument('burger_yaw', default_value='0.0'),
        DeclareLaunchArgument('waffle_x', default_value='-2.0'),
        DeclareLaunchArgument('waffle_y', default_value='0.7'),
        DeclareLaunchArgument('waffle_yaw', default_value='0.0'),
        LogInfo(msg='V17_BURGER_WAFFLE_SPAWN_ONLY | names: burger, waffle'),
        LogInfo(msg='SAFE_SPAWN_DEFAULTS | burger=(-2.0,-0.5,0.0) waffle=(-2.0,0.7,0.0) z=0.05'),
        SetEnvironmentVariable('TURTLEBOT3_MODEL', 'burger'),
        SetEnvironmentVariable('GZ_SIM_RESOURCE_PATH', gz_resource_path),
        SetEnvironmentVariable('IGN_GAZEBO_RESOURCE_PATH', gz_resource_path),
        LogInfo(msg=['Using world: ', world]),
        LogInfo(msg=['Using Burger SDF: ', burger_sdf]),
        LogInfo(msg=['Using Waffle SDF: ', waffle_sdf]),
        LogInfo(msg=['GZ_SIM_RESOURCE_PATH: ', gz_resource_path]),
        gz_sim,
        TimerAction(period=3.0, actions=[spawn_burger]),
        TimerAction(period=5.0, actions=[spawn_waffle]),
    ])
