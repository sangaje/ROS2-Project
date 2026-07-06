#!/usr/bin/env python3
"""Shared foundation every real-robot fleet role builds on: TurtleBot3
hardware bringup, the Nav2 navigation core (controller/planner/behavior/
bt_navigator + lifecycle manager), and a goal-pose-to-NavigateToPose proxy.

Deliberately excludes localization (AMCL vs Cartographer), domain bridging,
and any role-specific behaviour -- those differ per fleet role and are
layered on top by whatever includes this file (see member.launch.py, which
adds AMCL + domain bridging, and leader.launch.py, which adds Cartographer +
fleet coordination instead).
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    OpaqueFunction,
    TimerAction,
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

from tb3_fleet_bringup.launch_utils import clean_process_environment, launch_bool


def generate_launch_description():
    robot_launch = os.path.join(
        get_package_share_directory('turtlebot3_bringup'),
        'launch',
        'robot.launch.py',
    )

    domain_id = LaunchConfiguration('domain_id')
    start_robot_bringup = LaunchConfiguration('start_robot_bringup')
    hardware_param_file = LaunchConfiguration('hardware_param_file')
    nav2_params_file = LaunchConfiguration('nav2_params_file')
    goal_pose_topic = LaunchConfiguration('goal_pose_topic')
    goal_proxy_name = LaunchConfiguration('goal_proxy_name')
    nav_delay_sec = LaunchConfiguration('nav_delay_sec')
    lifecycle_delay_sec = LaunchConfiguration('lifecycle_delay_sec')
    goal_delay_sec = LaunchConfiguration('goal_delay_sec')

    def make_stack(context):
        process_env = clean_process_environment(domain_id.perform(context))
        params_file = nav2_params_file.perform(context)
        if not params_file:
            raise ValueError(
                'base.launch.py requires nav2_params_file (a path to a '
                'resolved Nav2 parameters YAML) -- the caller decides '
                'which localization params (AMCL vs Cartographer) apply.'
            )

        navigation = [
            Node(
                package='nav2_controller',
                executable='controller_server',
                name='controller_server',
                output='screen',
                parameters=[params_file],
                env=process_env,
            ),
            Node(
                package='nav2_planner',
                executable='planner_server',
                name='planner_server',
                output='screen',
                parameters=[params_file],
                env=process_env,
            ),
            Node(
                package='nav2_behaviors',
                executable='behavior_server',
                name='behavior_server',
                output='screen',
                parameters=[params_file],
                env=process_env,
            ),
            Node(
                package='nav2_bt_navigator',
                executable='bt_navigator',
                name='bt_navigator',
                output='screen',
                parameters=[params_file],
                env=process_env,
            ),
        ]
        navigation_lifecycle = Node(
            package='nav2_lifecycle_manager',
            executable='lifecycle_manager',
            name='lifecycle_manager_navigation',
            output='screen',
            parameters=[params_file],
            env=process_env,
        )
        goal_proxy = Node(
            package='tb3_fleet_bringup',
            executable='pose_to_nav2',
            name=goal_proxy_name.perform(context),
            output='screen',
            parameters=[{
                'use_sim_time': False,
                'goal_pose_topic': goal_pose_topic.perform(context),
            }],
            env=process_env,
        )

        actions = []
        if launch_bool(start_robot_bringup.perform(context)):
            robot_launch_args = {
                'use_sim_time': 'false',
                'namespace': '',
            }
            param_file = hardware_param_file.perform(context)
            if param_file:
                # Overrides turtlebot3_bringup's own default hardware
                # params, e.g. to disable the wheel odometry's own
                # odom->base_footprint TF broadcast when a Cartographer
                # elsewhere is going to own that transform instead (see
                # leader.launch.py enable_cartographer:=false / member.
                # launch.py's own-SLAM mode).
                robot_launch_args['tb3_param_dir'] = param_file
            actions.append(IncludeLaunchDescription(
                PythonLaunchDescriptionSource(robot_launch),
                launch_arguments=robot_launch_args.items(),
            ))

        actions.extend([
            TimerAction(
                period=float(nav_delay_sec.perform(context)),
                actions=navigation,
            ),
            TimerAction(
                period=float(lifecycle_delay_sec.perform(context)),
                actions=[navigation_lifecycle],
            ),
            TimerAction(
                period=float(goal_delay_sec.perform(context)),
                actions=[goal_proxy],
            ),
        ])
        return actions

    return LaunchDescription([
        DeclareLaunchArgument(
            'domain_id', default_value='0',
            description='DDS domain; the caller always sets this explicitly.',
        ),
        DeclareLaunchArgument(
            'start_robot_bringup', default_value='true',
            choices=['true', 'false'],
            description='Start TurtleBot3 hardware drivers.',
        ),
        DeclareLaunchArgument(
            'hardware_param_file', default_value='',
            description=(
                'Optional path to override turtlebot3_bringup\'s own '
                'hardware parameter YAML (its tb3_param_dir). Empty '
                'means use turtlebot3_bringup\'s own default.'
            ),
        ),
        DeclareLaunchArgument(
            'nav2_params_file', default_value='',
            description=(
                'Required: path to a resolved Nav2 parameters YAML. The '
                'caller owns localization, so this file should not '
                'declare its own SLAM/AMCL choice beyond what Nav2 itself '
                'needs (global_frame, costmaps, controller tuning).'
            ),
        ),
        DeclareLaunchArgument('goal_pose_topic', default_value='/goal_pose'),
        DeclareLaunchArgument('goal_proxy_name', default_value='goal_arbiter'),
        DeclareLaunchArgument('nav_delay_sec', default_value='8.0'),
        DeclareLaunchArgument('lifecycle_delay_sec', default_value='12.0'),
        DeclareLaunchArgument(
            'goal_delay_sec',
            default_value=LaunchConfiguration('lifecycle_delay_sec'),
            description=(
                'When the goal proxy starts. Defaults to the same moment '
                'as lifecycle_delay_sec; override separately if the '
                'caller wants the two staggered.'
            ),
        ),
        OpaqueFunction(function=make_stack),
    ])
