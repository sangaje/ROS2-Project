#!/usr/bin/env python3

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    LogInfo,
    OpaqueFunction,
    SetEnvironmentVariable,
    TimerAction,
    UnsetEnvironmentVariable,
)
from launch.substitutions import LaunchConfiguration


def _enabled(value):
    return value.strip().lower() in ('true', '1', 'yes', 'on')


def generate_launch_description():
    mode = LaunchConfiguration('mode')
    main_domain_id = LaunchConfiguration('main_domain_id')
    follower_domain_id = LaunchConfiguration('follower_domain_id')
    robot_domains = LaunchConfiguration('robot_domains')
    start_rviz = LaunchConfiguration('start_rviz')
    start_gazebo = LaunchConfiguration('start_gazebo')
    start_bridge = LaunchConfiguration('start_bridge')
    start_gz_client = LaunchConfiguration('start_gz_client')
    rviz_delay = LaunchConfiguration('rviz_delay')

    def make_actions(context, *args, **kwargs):
        selected_mode = mode.perform(context).strip().lower()
        main_domain = main_domain_id.perform(context)
        follower_domain = follower_domain_id.perform(context)

        if selected_mode not in ('real', 'sim'):
            raise RuntimeError("mode must be 'real' or 'sim'")

        actions = [
            LogInfo(msg=[
                'PC_DEBUG | mode=', selected_mode,
                ' main_domain=', main_domain,
                ' follower_domain=', follower_domain,
            ])
        ]

        if selected_mode == 'sim' and _enabled(start_gazebo.perform(context)):
            actions.append(ExecuteProcess(
                cmd=[
                    'ros2', 'launch', 'tb3_fleet_bringup', 'sim_world.launch.py',
                    f'domain_id:={main_domain}',
                    f'start_gz_client:={start_gz_client.perform(context)}',
                ],
                output='screen',
                name='sim_world',
            ))

        if _enabled(start_bridge.perform(context)):
            actions.append(ExecuteProcess(
                cmd=[
                    'ros2', 'launch', 'tb3_fleet_bridge', 'bridges.launch.py',
                    f'main_domain_id:={main_domain}',
                    f'robot_domains:={robot_domains.perform(context)}',
                ],
                output='screen',
                name='pc_bridge',
            ))

        if _enabled(start_rviz.perform(context)):
            actions.append(TimerAction(
                period=float(rviz_delay.perform(context)),
                actions=[ExecuteProcess(
                    cmd=[
                        'ros2', 'launch', 'tb3_fleet_bringup', 'rviz.launch.py',
                        f'domain_id:={main_domain}',
                    ],
                    output='screen',
                    name='rviz',
                )],
            ))

        return actions

    return LaunchDescription([
        DeclareLaunchArgument('mode', default_value='real',
                              description='real or sim.'),
        DeclareLaunchArgument('main_domain_id', default_value='25',
                              description='Leader/main domain used by RViz.'),
        DeclareLaunchArgument('follower_domain_id', default_value='24'),
        DeclareLaunchArgument('robot_domains', default_value='burger:24',
                              description='Only used when start_bridge=true.'),
        DeclareLaunchArgument('start_rviz', default_value='true'),
        DeclareLaunchArgument('start_gazebo', default_value='true',
                              description='Only used in mode=sim.'),
        DeclareLaunchArgument('start_bridge', default_value='false',
                              description='Normally false: follower.launch.py owns its bridge.'),
        DeclareLaunchArgument('start_gz_client', default_value='true'),
        DeclareLaunchArgument('rviz_delay', default_value='3.0'),
        UnsetEnvironmentVariable('ROS_DISCOVERY_SERVER'),
        UnsetEnvironmentVariable('FASTRTPS_DEFAULT_PROFILES_FILE'),
        UnsetEnvironmentVariable('FASTDDS_DEFAULT_PROFILES_FILE'),
        SetEnvironmentVariable('ROS_AUTOMATIC_DISCOVERY_RANGE', 'SUBNET'),
        SetEnvironmentVariable('ROS_LOCALHOST_ONLY', '0'),
        SetEnvironmentVariable('RMW_IMPLEMENTATION', 'rmw_fastrtps_cpp'),
        OpaqueFunction(function=make_actions),
    ])
