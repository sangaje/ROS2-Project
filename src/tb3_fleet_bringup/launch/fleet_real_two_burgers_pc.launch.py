#!/usr/bin/env python3

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    LogInfo,
    TimerAction,
)
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    leader_initial_x = LaunchConfiguration('leader_initial_x')
    leader_initial_y = LaunchConfiguration('leader_initial_y')
    leader_initial_yaw = LaunchConfiguration('leader_initial_yaw')
    follow_distance = LaunchConfiguration('follow_distance')
    start_following = LaunchConfiguration('start_following')
    start_rviz = LaunchConfiguration('start_rviz')

    follower = ExecuteProcess(
        cmd=[
            'ros2', 'launch', 'tb3_fleet_bringup',
            'fleet_real_domain24_burger_nav2_follower.launch.py',
            'domain_id:=24',
            'leader_domain_id:=25',
            ['start_following:=', start_following],
            ['follow_distance:=', follow_distance],
            'enable_path_yield:=true',
            'path_block_distance:=0.55',
            'yield_lateral_distance:=0.75',
        ],
        output='screen',
        name='two_burgers_follower_stack',
    )
    leader = ExecuteProcess(
        cmd=[
            'ros2', 'launch', 'tb3_fleet_bringup',
            'fleet_real_domain25_burger_nav2.launch.py',
            'domain_id:=25',
            ['initial_x:=', leader_initial_x],
            ['initial_y:=', leader_initial_y],
            ['initial_yaw:=', leader_initial_yaw],
        ],
        output='screen',
        name='two_burgers_leader_stack',
    )
    rviz = ExecuteProcess(
        cmd=[
            'ros2', 'launch', 'tb3_fleet_bringup',
            'fleet_real_domain25_rviz.launch.py',
            'domain_id:=25',
        ],
        output='screen',
        name='two_burgers_rviz',
        condition=IfCondition(start_rviz),
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            'leader_initial_x',
            default_value='0.95',
            description='Leader Burger x relative to follower Burger at startup.',
        ),
        DeclareLaunchArgument('leader_initial_y', default_value='0.0'),
        DeclareLaunchArgument('leader_initial_yaw', default_value='0.0'),
        DeclareLaunchArgument('follow_distance', default_value='1.05'),
        DeclareLaunchArgument(
            'start_following',
            default_value='false',
            description='Safe default: wait for the follow command.',
        ),
        DeclareLaunchArgument('start_rviz', default_value='true'),
        LogInfo(
            msg=[
                'REAL_TWO_BURGERS_PC | leader Domain25 at x=',
                leader_initial_x,
                ' | follower Domain24 | start_following=',
                start_following,
            ]
        ),
        follower,
        TimerAction(period=2.0, actions=[leader]),
        TimerAction(period=5.0, actions=[rviz]),
    ])
