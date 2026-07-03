#!/usr/bin/env python3

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    LogInfo,
    OpaqueFunction,
    TimerAction,
)
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    follower_initial_x   = LaunchConfiguration('follower_initial_x')
    follower_initial_y   = LaunchConfiguration('follower_initial_y')
    follower_initial_yaw = LaunchConfiguration('follower_initial_yaw')
    follow_distance      = LaunchConfiguration('follow_distance')
    start_following      = LaunchConfiguration('start_following')
    start_rviz           = LaunchConfiguration('start_rviz')
    main_domain_id       = LaunchConfiguration('main_domain_id')
    leader_domain_id     = LaunchConfiguration('leader_domain_id')
    follower_domain_id   = LaunchConfiguration('follower_domain_id')

    def selected_main_domain(context):
        legacy = leader_domain_id.perform(context).strip()
        return legacy if legacy else main_domain_id.perform(context)

    def make_processes(context, *args, **kwargs):
        main_domain = selected_main_domain(context)
        leader_cmd = [
            'ros2', 'launch', 'tb3_fleet_bringup',
            'fleet_real_leader_nav2.launch.py',
            f'domain_id:={main_domain}',
            'robot_model:=burger',
            'use_slam:=true',
        ]
        follower_cmd = [
            'ros2', 'launch', 'tb3_fleet_bringup',
            'fleet_real_follower_nav2.launch.py',
            f'domain_id:={follower_domain_id.perform(context)}',
            f'main_domain_id:={main_domain}',
            'use_slam:=false',
            f'slam_domain:={main_domain}',
            f'start_following:={start_following.perform(context)}',
            f'follow_distance:={follow_distance.perform(context)}',
            'enable_path_yield:=true',
            'path_block_distance:=0.55',
            'yield_lateral_distance:=0.75',
            f'follower_initial_x:={follower_initial_x.perform(context)}',
            f'follower_initial_y:={follower_initial_y.perform(context)}',
            f'follower_initial_yaw:={follower_initial_yaw.perform(context)}',
        ]
        rviz_cmd = [
            'ros2', 'launch', 'tb3_fleet_bringup',
            'fleet_rviz.launch.py',
            f'domain_id:={main_domain}',
        ]

        actions = [
            TimerAction(period=0.0, actions=[
                ExecuteProcess(
                    cmd=leader_cmd,
                    output='screen',
                    name='real_nav2_leader_slam',
                ),
            ]),
            TimerAction(period=10.0, actions=[
                ExecuteProcess(
                    cmd=follower_cmd,
                    output='screen',
                    name='real_nav2_follower_amcl',
                ),
            ]),
        ]
        if start_rviz.perform(context).lower() in ('true', '1', 'yes', 'on'):
            actions.append(TimerAction(period=18.0, actions=[
                ExecuteProcess(
                    cmd=rviz_cmd,
                    output='screen',
                    name='two_burgers_rviz',
                ),
            ]))
        return actions

    return LaunchDescription([
        DeclareLaunchArgument('follower_initial_x', default_value='-1.05',
                              description='Follower initial x in leader SLAM map.'),
        DeclareLaunchArgument('follower_initial_y',   default_value='0.0'),
        DeclareLaunchArgument('follower_initial_yaw', default_value='0.0'),
        DeclareLaunchArgument('follow_distance',  default_value='1.05'),
        DeclareLaunchArgument('start_following',  default_value='false',
                              description='Safe default: wait for fleet_follow_signal.'),
        DeclareLaunchArgument('start_rviz',  default_value='true'),
        DeclareLaunchArgument('main_domain_id', default_value='25',
                              description='Main/leader ROS domain. RViz, SLAM, leader Nav2 run here.'),
        DeclareLaunchArgument('leader_domain_id', default_value='',
                              description='Deprecated alias for main_domain_id. Empty uses main_domain_id.'),
        DeclareLaunchArgument('follower_domain_id', default_value='24'),
        LogInfo(msg=['REAL_NAV2_CLEAN_PC | main domain=', main_domain_id,
                     ' SLAM, follower domain=', follower_domain_id, ' AMCL/Nav2',
                     ' | start_following=', start_following]),
        OpaqueFunction(function=make_processes),
    ])
