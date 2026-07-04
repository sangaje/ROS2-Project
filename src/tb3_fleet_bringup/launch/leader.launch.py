#!/usr/bin/env python3
"""
Leader — D25 Waffle Pi.
  mode:=real  Hardware bringup + Cartographer SLAM + Nav2
  mode:=sim   Gazebo simulation — Cartographer SLAM + Nav2, no hardware
"""
import os
import tempfile
from pathlib import Path

import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument, ExecuteProcess, IncludeLaunchDescription,
    OpaqueFunction, SetEnvironmentVariable, TimerAction,
    UnsetEnvironmentVariable,
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from nav2_common.launch import RewrittenYaml


def generate_launch_description():
    pkg = get_package_share_directory('tb3_fleet_bringup')
    tb3_carto_launch = os.path.join(
        get_package_share_directory('turtlebot3_cartographer'),
        'launch', 'cartographer.launch.py',
    )

    mode          = LaunchConfiguration('mode')
    domain_id     = LaunchConfiguration('domain_id')
    use_slam      = LaunchConfiguration('use_slam')
    initial_x     = LaunchConfiguration('initial_x')
    initial_y     = LaunchConfiguration('initial_y')
    initial_yaw   = LaunchConfiguration('initial_yaw')
    def make_all(context, *args, **kwargs):
        sim = mode.perform(context).lower() == 'sim'
        ust = 'true' if sim else 'false'
        d = domain_id.perform(context)
        extra = {
            'ROS_DOMAIN_ID': d,
            'ROS_AUTOMATIC_DISCOVERY_RANGE': 'SUBNET',
            'ROS_LOCALHOST_ONLY': '0',
            'RMW_IMPLEMENTATION': 'rmw_fastrtps_cpp',
        }

        nav2_params = RewrittenYaml(
            source_file=os.path.join(pkg, 'config', 'domain25_waffle_nav2.yaml'),
            param_rewrites={
                'use_sim_time': ust,
                'odom_topic': '/odom',
                'scan_topic': '/scan',
                'topic': '/scan',
                'enable_stamped_cmd_vel': 'true',
            },
            convert_types=True,
        )

        # Localization: Cartographer SLAM or AMCL
        if use_slam.perform(context).lower() in ('true', '1', 'yes'):
            localization = [IncludeLaunchDescription(
                PythonLaunchDescriptionSource(tb3_carto_launch),
                launch_arguments={
                    'cartographer_config_dir': os.path.join(pkg, 'config'),
                    'configuration_basename': 'cartographer_2d_lidar_odom_v44.lua',
                    'use_sim_time': ust,
                    'use_rviz': 'false',
                }.items(),
            )]
        else:
            ix, iy, iyaw = (float(x.perform(context)) for x in [initial_x, initial_y, initial_yaw])
            pose_yaml = Path(tempfile.gettempdir()) / 'leader_amcl_pose.yaml'
            pose_yaml.write_text(yaml.dump({'amcl': {'ros__parameters': {
                'set_initial_pose': True,
                'initial_pose': {'x': ix, 'y': iy, 'z': 0.0, 'yaw': iyaw},
            }}}), encoding='utf-8')
            amcl = Node(package='nav2_amcl', executable='amcl', name='amcl',
                        output='screen', parameters=[nav2_params, str(pose_yaml)],
                        additional_env=extra)
            lc_loc = Node(package='nav2_lifecycle_manager', executable='lifecycle_manager',
                          name='lifecycle_manager_localization', output='screen',
                          parameters=[nav2_params], additional_env=extra)
            localization = [TimerAction(period=0.5, actions=[amcl]),
                            TimerAction(period=1.0, actions=[lc_loc])]

        leader_pose = ExecuteProcess(
            cmd=['python3', os.path.join(pkg, 'scripts', 'tf_pose_publisher_direct_v44.py'),
                 '--ros-args', '-r', '__node:=leader_pose_pub',
                 '-p', f'use_sim_time:={ust}',
                 '-p', 'target_frame:=map', '-p', 'source_frame:=base_footprint',
                 '-p', 'output_topic:=/leader_pose',
                 '-p', 'publish_rate_hz:=10.0', '-p', 'log_every_n:=100'],
            output='screen', name='leader_pose_pub', additional_env=extra,
        )
        burger_amcl_tf = ExecuteProcess(
            cmd=['python3', os.path.join(pkg, 'scripts', 'pose_to_tf_broadcaster.py'),
                 '--ros-args', '-r', '__node:=burger_tf_on_leader',
                 '-p', f'use_sim_time:={ust}',
                 '-p', 'input_topic:=/burger_pose',
                 '-p', 'parent_frame:=map', '-p', 'child_frame:=burger/base_footprint',
                 '-p', 'republish_hz:=10.0'],
            output='screen', name='burger_tf_on_leader', additional_env=extra,
        )

        controller  = Node(package='nav2_controller', executable='controller_server',
                           name='controller_server', output='screen',
                           parameters=[nav2_params], additional_env=extra)
        planner     = Node(package='nav2_planner', executable='planner_server',
                           name='planner_server', output='screen',
                           parameters=[nav2_params], additional_env=extra)
        behaviors   = Node(package='nav2_behaviors', executable='behavior_server',
                           name='behavior_server', output='screen',
                           parameters=[nav2_params], additional_env=extra)
        bt_nav      = Node(package='nav2_bt_navigator', executable='bt_navigator',
                           name='bt_navigator', output='screen',
                           parameters=[nav2_params], additional_env=extra)
        lifecycle_nav = Node(package='nav2_lifecycle_manager', executable='lifecycle_manager',
                             name='lifecycle_manager_navigation', output='screen',
                             parameters=[nav2_params], additional_env=extra)
        default_goal = ExecuteProcess(
            cmd=['python3', os.path.join(pkg, 'scripts', 'pose_to_nav2_action_direct_v41.py'),
                 '--ros-args', '-r', '__node:=waffle_default_goal',
                 '-p', f'use_sim_time:={ust}',
                 '-p', 'goal_pose_topic:=/goal_pose',
                 '-p', 'navigate_action:=/navigate_to_pose',
                 '-p', 'default_frame_id:=map', '-p', 'cancel_previous_goal:=true'],
            output='screen', name='waffle_default_goal', additional_env=extra,
        )
        named_goal = ExecuteProcess(
            cmd=['python3', os.path.join(pkg, 'scripts', 'pose_to_nav2_action_direct_v41.py'),
                 '--ros-args', '-r', '__node:=waffle_named_goal',
                 '-p', f'use_sim_time:={ust}',
                 '-p', 'goal_pose_topic:=/waffle_goal_pose',
                 '-p', 'navigate_action:=/navigate_to_pose',
                 '-p', 'default_frame_id:=map', '-p', 'cancel_previous_goal:=true'],
            output='screen', name='waffle_named_goal', additional_env=extra,
        )

        if sim:
            return localization + [
                TimerAction(period=1.0, actions=[leader_pose, burger_amcl_tf]),
                TimerAction(period=2.0, actions=[controller, planner, behaviors, bt_nav]),
                TimerAction(period=5.0, actions=[lifecycle_nav]),
                TimerAction(period=7.0, actions=[default_goal, named_goal]),
            ]
        else:
            burger_scan_static_tf = ExecuteProcess(
                cmd=['ros2', 'run', 'tf2_ros', 'static_transform_publisher',
                     '--x', '-0.032', '--y', '0.0', '--z', '0.182',
                     '--roll', '0', '--pitch', '0', '--yaw', '0',
                     '--frame-id', 'burger/base_footprint',
                     '--child-frame-id', 'burger/base_scan'],
                output='screen', name='burger_scan_static_tf', additional_env=extra,
            )
            return localization + [
                TimerAction(period=1.0, actions=[burger_scan_static_tf]),
                TimerAction(period=2.0, actions=[leader_pose, burger_amcl_tf]),
                TimerAction(period=6.0, actions=[controller, planner, behaviors, bt_nav]),
                TimerAction(period=10.0, actions=[lifecycle_nav]),
                TimerAction(period=12.0, actions=[default_goal, named_goal]),
            ]

    return LaunchDescription([
        DeclareLaunchArgument('mode',               default_value='real',
                              description='real = physical robot | sim = Gazebo'),
        DeclareLaunchArgument('domain_id',          default_value='25'),
        DeclareLaunchArgument('use_slam',           default_value='true',
                              description='true = Cartographer SLAM | false = AMCL with saved map'),
        DeclareLaunchArgument('initial_x',          default_value='0.0'),
        DeclareLaunchArgument('initial_y',          default_value='0.0'),
        DeclareLaunchArgument('initial_yaw',        default_value='0.0'),
        UnsetEnvironmentVariable('ROS_DISCOVERY_SERVER'),
        UnsetEnvironmentVariable('ROS_LOCALHOST_ONLY'),
        UnsetEnvironmentVariable('FASTRTPS_DEFAULT_PROFILES_FILE'),
        UnsetEnvironmentVariable('FASTDDS_DEFAULT_PROFILES_FILE'),
        SetEnvironmentVariable('ROS_DOMAIN_ID',                domain_id),
        SetEnvironmentVariable('ROS_AUTOMATIC_DISCOVERY_RANGE', 'SUBNET'),
        SetEnvironmentVariable('ROS_LOCALHOST_ONLY',           '0'),
        SetEnvironmentVariable('RMW_IMPLEMENTATION',           'rmw_fastrtps_cpp'),
        OpaqueFunction(function=make_all),
    ])
