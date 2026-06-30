#!/usr/bin/env python3

import os
import re
import tempfile
from pathlib import Path
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


def _patch_diffdrive_topic(src_sdf: str, robot_name: str, model_label: str) -> str:
    """Create a temporary SDF whose DiffDrive command topic is robot-scoped.

    v17 defaults:
      burger DiffDrive topic -> /burger/cmd_vel
      waffle DiffDrive topic -> /waffle/cmd_vel
    """
    text = Path(src_sdf).read_text(encoding='utf-8')
    target_topic = f'/{robot_name}/cmd_vel'

    plugin_re = re.compile(
        r'(<plugin[^>]*(?:DiffDrive|diff_drive|diff-drive)[^>]*>)(.*?)(</plugin>)',
        re.IGNORECASE | re.DOTALL,
    )

    def repl(match: re.Match) -> str:
        start, body, end = match.group(1), match.group(2), match.group(3)
        if re.search(r'<topic>.*?</topic>', body, flags=re.DOTALL):
            body = re.sub(r'<topic>.*?</topic>', f'<topic>{target_topic}</topic>', body, count=1, flags=re.DOTALL)
        else:
            body = body + f'\n      <topic>{target_topic}</topic>\n'
        return start + body + end

    patched, n = plugin_re.subn(repl, text, count=1)

    if n == 0:
        patched = re.sub(
            r'<topic>[^<]*cmd_vel[^<]*</topic>',
            f'<topic>{target_topic}</topic>',
            text,
            count=1,
            flags=re.IGNORECASE,
        )

    if patched == text:
        raise RuntimeError(
            f'Failed to patch DiffDrive cmd_vel topic in {src_sdf}. '
            'Open the SDF and check the DiffDrive plugin tag.'
        )

    out_dir = Path(tempfile.gettempdir()) / 'tb3_fleet_patched_sdf_v17_burger_waffle'
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f'{robot_name}_{model_label}_cmd_{robot_name}.sdf'
    out_path.write_text(patched, encoding='utf-8')
    return str(out_path)


def generate_launch_description():
    tb3_gazebo_share = get_package_share_directory('turtlebot3_gazebo')
    bringup_share = get_package_share_directory('tb3_fleet_bringup')

    default_world = os.path.join(tb3_gazebo_share, 'worlds', 'turtlebot3_world.world')
    burger_model_dir, burger_sdf = _first_existing_model_sdf(tb3_gazebo_share, ['turtlebot3_burger'])
    waffle_model_dir, waffle_sdf = _first_existing_model_sdf(tb3_gazebo_share, ['turtlebot3_waffle', 'turtlebot3_waffle_pi'])
    default_bridge_config = os.path.join(bringup_share, 'config', 'two_burger_ros_gz_bridge.yaml')

    burger_patched_sdf = _patch_diffdrive_topic(burger_sdf, 'burger', burger_model_dir)
    waffle_patched_sdf = _patch_diffdrive_topic(waffle_sdf, 'waffle', waffle_model_dir)

    tb3_models_dir = os.path.join(tb3_gazebo_share, 'models')
    old_gz_resource_path = os.environ.get('GZ_SIM_RESOURCE_PATH', '')
    gz_resource_path_parts = [tb3_models_dir, tb3_gazebo_share]
    if old_gz_resource_path:
        gz_resource_path_parts.append(old_gz_resource_path)
    gz_resource_path = ':'.join(gz_resource_path_parts)

    world = LaunchConfiguration('world')
    bridge_config = LaunchConfiguration('bridge_config')
    gz_verbosity = LaunchConfiguration('gz_verbosity')

    burger_name = LaunchConfiguration('burger_name')
    waffle_name = LaunchConfiguration('waffle_name')

    burger_x = LaunchConfiguration('burger_x')
    burger_y = LaunchConfiguration('burger_y')
    burger_yaw = LaunchConfiguration('burger_yaw')

    waffle_x = LaunchConfiguration('waffle_x')
    waffle_y = LaunchConfiguration('waffle_y')
    waffle_yaw = LaunchConfiguration('waffle_yaw')

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
        arguments=[
            '-file', burger_patched_sdf,
            '-name', burger_name,
            '-x', burger_x,
            '-y', burger_y,
            '-z', '0.05',
            '-Y', burger_yaw,
        ],
    )

    spawn_waffle = Node(
        package='ros_gz_sim',
        executable='create',
        name='spawn_waffle',
        output='screen',
        arguments=[
            '-file', waffle_patched_sdf,
            '-name', waffle_name,
            '-x', waffle_x,
            '-y', waffle_y,
            '-z', '0.05',
            '-Y', waffle_yaw,
        ],
    )

    twist_stamped_cmdvel_bridge = Node(
        package='tb3_fleet_bringup',
        executable='twist_stamped_to_twist',
        name='twist_stamped_to_twist_bridge',
        output='screen',
        parameters=[{
            'robot_names': 'burger,waffle',
            'cmd_vel_topic': 'cmd_vel',
            'internal_cmd_vel_topics': 'gz_cmd_vel_unstamped,gz_cmd_vel_model_unstamped',
            'cmd_republish_rate_hz': 20.0,
            'watchdog_timeout_sec': 0.8,
            'log_every_n_republish': 100,
        }],
    )

    bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        name='burger_waffle_ros_gz_bridge',
        output='screen',
        parameters=[{'config_file': bridge_config}],
    )

    return LaunchDescription([
        DeclareLaunchArgument('world', default_value=default_world),
        DeclareLaunchArgument('bridge_config', default_value=default_bridge_config),
        DeclareLaunchArgument('gz_verbosity', default_value='2'),
        DeclareLaunchArgument('burger_name', default_value='burger'),
        DeclareLaunchArgument('waffle_name', default_value='waffle'),
        DeclareLaunchArgument('burger_x', default_value='-2.9'),
        DeclareLaunchArgument('burger_y', default_value='0.5'),
        DeclareLaunchArgument('burger_yaw', default_value='0.0'),
        DeclareLaunchArgument('waffle_x', default_value='-1.8'),
        DeclareLaunchArgument('waffle_y', default_value='0.5'),
        DeclareLaunchArgument('waffle_yaw', default_value='0.0'),

        LogInfo(msg='V19_BURGER_WAFFLE_TWIST_STAMPED | Gazebo mixed Burger+Waffle + TwistStamped ROS command API'),
        LogInfo(msg='Names changed: /burger/* and /waffle/* instead of /robot1/* and /robot2/*'),
        LogInfo(msg='ROS public cmd topics: /burger/cmd_vel and /waffle/cmd_vel are geometry_msgs/msg/TwistStamped'),
        LogInfo(msg='Patch: per-model SDF forces DiffDrive topics to /burger/cmd_vel and /waffle/cmd_vel; fallback /model/<name>/cmd_vel is also bridged'),
        LogInfo(msg=['Patched burger SDF: ', burger_patched_sdf]),
        LogInfo(msg=['Patched waffle SDF: ', waffle_patched_sdf]),
        SetEnvironmentVariable('TURTLEBOT3_MODEL', 'burger'),
        SetEnvironmentVariable('GZ_SIM_RESOURCE_PATH', gz_resource_path),
        SetEnvironmentVariable('IGN_GAZEBO_RESOURCE_PATH', gz_resource_path),
        LogInfo(msg=['Using world: ', world]),
        LogInfo(msg=['Using Burger source SDF: ', burger_sdf]),
        LogInfo(msg=['Using Waffle source SDF: ', waffle_sdf]),
        LogInfo(msg=['Using ROS/GZ bridge config: ', bridge_config]),
        gz_sim,
        TimerAction(period=2.0, actions=[twist_stamped_cmdvel_bridge, bridge]),
        TimerAction(period=3.0, actions=[spawn_burger]),
        TimerAction(period=5.0, actions=[spawn_waffle]),
    ])
