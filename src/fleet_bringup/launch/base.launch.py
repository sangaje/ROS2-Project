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

import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    LogInfo,
    OpaqueFunction,
    TimerAction,
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

from fleet_bringup.launch_utils import clean_process_environment, launch_bool


def _tracked_cmd_vel_adapter_enabled(param_file: str) -> bool:
    if not param_file:
        return False
    try:
        with open(param_file, 'r', encoding='utf-8') as handle:
            data = yaml.safe_load(handle) or {}
    except OSError:
        return False
    params = data.get('tracked_cmd_vel_adapter', {}).get('ros__parameters', {})
    return bool(params.get('enabled', False))


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
    start_nav2 = LaunchConfiguration('start_nav2')
    goal_pose_topic = LaunchConfiguration('goal_pose_topic')
    cancel_topic = LaunchConfiguration('cancel_topic')
    goal_proxy_name = LaunchConfiguration('goal_proxy_name')
    nav_delay_sec = LaunchConfiguration('nav_delay_sec')
    lifecycle_delay_sec = LaunchConfiguration('lifecycle_delay_sec')
    goal_delay_sec = LaunchConfiguration('goal_delay_sec')
    require_localization_ready = LaunchConfiguration('require_localization_ready')
    localization_ready_topic = LaunchConfiguration('localization_ready_topic')

    def make_stack(context):
        process_env = clean_process_environment(domain_id.perform(context))
        params_file = nav2_params_file.perform(context)
        if not params_file:
            raise ValueError(
                'base.launch.py requires nav2_params_file (a path to a '
                'resolved Nav2 parameters YAML) -- the caller decides '
                'which localization params (AMCL vs Cartographer) apply.'
            )
        hardware_params_file = hardware_param_file.perform(context).strip()
        adapter_enabled = _tracked_cmd_vel_adapter_enabled(hardware_params_file)
        command_remappings = (
            [('cmd_vel', '/cmd_vel_nav'), ('/cmd_vel', '/cmd_vel_nav')]
            if adapter_enabled
            else []
        )

        navigation = [
            Node(
                package='nav2_controller',
                executable='controller_server',
                name='controller_server',
                output='screen',
                parameters=[params_file],
                remappings=command_remappings,
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
                remappings=command_remappings,
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
            package='fleet_bringup',
            executable='pose_to_nav2',
            name=goal_proxy_name.perform(context),
            output='screen',
            parameters=[{
                'use_sim_time': False,
                'goal_pose_topic': goal_pose_topic.perform(context),
                'cancel_topic': cancel_topic.perform(context),
                'require_localization_ready': launch_bool(
                    require_localization_ready.perform(context)
                ),
                'localization_ready_topic': localization_ready_topic.perform(context),
            }],
            env=process_env,
        )

        actions = []
        if adapter_enabled:
            actions.append(Node(
                package='fleet_bringup',
                executable='tracked_cmd_vel_adapter',
                name='tracked_cmd_vel_adapter',
                output='screen',
                parameters=[hardware_params_file],
                env=process_env,
                respawn=True,
                respawn_delay=1.0,
            ))
        if launch_bool(start_robot_bringup.perform(context)):
            robot_launch_args = {
                'use_sim_time': 'false',
                'namespace': '',
            }
            param_file = hardware_params_file
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

        if launch_bool(start_nav2.perform(context)):
            actions.extend([
                TimerAction(
                    period=float(nav_delay_sec.perform(context)),
                    actions=[
                        LogInfo(msg=[
                            'BASE_STAGE | starting Nav2 core nodes ',
                            '(controller/planner/behavior/bt_navigator)',
                        ]),
                        *navigation,
                    ],
                ),
                TimerAction(
                    period=float(lifecycle_delay_sec.perform(context)),
                    actions=[
                        LogInfo(msg=[
                            'BASE_STAGE | starting Nav2 navigation lifecycle',
                        ]),
                        navigation_lifecycle,
                    ],
                ),
                TimerAction(
                    period=float(goal_delay_sec.perform(context)),
                    actions=[
                        LogInfo(msg=[
                            'BASE_STAGE | starting Nav2 goal proxy ',
                            goal_proxy_name.perform(context),
                        ]),
                        goal_proxy,
                    ],
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
        DeclareLaunchArgument(
            'start_nav2',
            default_value='true',
            choices=['true', 'false'],
            description=(
                'Start the Nav2 controller/planner/behavior/bt_navigator '
                'and goal proxy. Hardware bringup is controlled separately '
                'by start_robot_bringup.'
            ),
        ),
        DeclareLaunchArgument('goal_pose_topic', default_value='/goal_pose'),
        DeclareLaunchArgument(
            'cancel_topic',
            default_value='',
            description='Optional std_msgs/Bool topic that cancels the current Nav2 action goal.',
        ),
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
        DeclareLaunchArgument(
            'require_localization_ready',
            default_value='false',
            choices=['true', 'false'],
            description=(
                'Hold the Nav2 goal proxy until /localization_ready is true. '
                'Used by leader/member AMCL kickstart so initial spin owns cmd_vel.'
            ),
        ),
        DeclareLaunchArgument(
            'localization_ready_topic',
            default_value='/localization_ready',
            description='Latched Bool topic published by global_localize_kickstart.',
        ),
        OpaqueFunction(function=make_stack),
    ])
