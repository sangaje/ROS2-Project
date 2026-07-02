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
    robot1_ip            = LaunchConfiguration('robot1_ip')
    robot2_ip            = LaunchConfiguration('robot2_ip')

    def make_processes(context, *args, **kwargs):
        r1 = robot1_ip.perform(context).strip()
        r2 = robot2_ip.perform(context).strip()
        peer_args = []
        if r1:
            peer_args.append(f'robot1_ip:={r1}')
        if r2:
            peer_args.append(f'robot2_ip:={r2}')

        leader_cmd = [
            'ros2', 'launch', 'tb3_fleet_bringup',
            'fleet_real_domain25_burger_nav2.launch.py',
            'domain_id:=25',
            'use_slam:=true',
        ] + peer_args
        follower_cmd = [
            'ros2', 'launch', 'tb3_fleet_bringup',
            'fleet_real_domain24_burger_nav2_follower.launch.py',
            'domain_id:=24',
            'leader_domain_id:=25',
            'use_slam:=false',
            'slam_domain:=25',
            f'start_following:={start_following.perform(context)}',
            f'follow_distance:={follow_distance.perform(context)}',
            'enable_path_yield:=true',
            'path_block_distance:=0.55',
            'yield_lateral_distance:=0.75',
            f'follower_initial_x:={follower_initial_x.perform(context)}',
            f'follower_initial_y:={follower_initial_y.perform(context)}',
            f'follower_initial_yaw:={follower_initial_yaw.perform(context)}',
        ] + peer_args
        rviz_cmd = [
            'ros2', 'launch', 'tb3_fleet_bringup',
            'fleet_real_domain25_rviz.launch.py',
            'domain_id:=25',
        ] + peer_args

        actions = [
            TimerAction(period=0.0, actions=[
                ExecuteProcess(
                    cmd=leader_cmd,
                    output='screen',
                    name='real_nav2_leader_domain25_slam',
                ),
            ]),
            TimerAction(period=10.0, actions=[
                ExecuteProcess(
                    cmd=follower_cmd,
                    output='screen',
                    name='real_nav2_follower_domain24_amcl',
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
        DeclareLaunchArgument('robot1_ip',   default_value='',
                              description='Optional leader robot IP for static DDS peers. Empty uses subnet multicast.'),
        DeclareLaunchArgument('robot2_ip',   default_value='',
                              description='Optional follower robot IP for static DDS peers. Empty uses subnet multicast.'),
        LogInfo(msg=['REAL_NAV2_CLEAN_PC | leader Domain25 SLAM, follower Domain24 AMCL/Nav2',
                     ' | start_following=', start_following,
                     ' | static_peers=', robot1_ip, ';', robot2_ip]),
        OpaqueFunction(function=make_processes),
    ])
