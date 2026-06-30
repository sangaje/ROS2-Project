#!/usr/bin/env python3

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, LogInfo, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    base_launch = PathJoinSubstitution([
        FindPackageShare('tb3_fleet_bringup'),
        'launch',
        'fleet_two_burger_spawn_with_ros_bridge.launch.py',
    ])

    follow_distance = LaunchConfiguration('follow_distance')
    min_leader_distance = LaunchConfiguration('min_leader_distance')
    hard_min_leader_distance = LaunchConfiguration('hard_min_leader_distance')
    front_stop_distance = LaunchConfiguration('front_stop_distance')
    front_slow_distance = LaunchConfiguration('front_slow_distance')
    max_linear = LaunchConfiguration('max_linear')
    max_angular = LaunchConfiguration('max_angular')

    follower = Node(
        package='tb3_fleet_bringup',
        executable='waffle_burger_follower',
        name='waffle_burger_follower',
        output='screen',
        parameters=[{
            'leader_name': 'waffle',
            'follower_name': 'burger',
            'leader_cmd_topic': '/waffle/cmd_vel',
            'leader_odom_topic': '/waffle/odom',
            'follower_odom_topic': '/burger/odom',
            'follower_scan_topic': '/burger/scan',
            'follower_cmd_topic': '/burger/cmd_vel',
            'cmd_frame_id': 'base_link',
            'control_rate_hz': 20.0,
            'follow_distance': follow_distance,
            'goal_tolerance': 0.12,
            'min_leader_distance': min_leader_distance,
            'hard_min_leader_distance': hard_min_leader_distance,
            'front_stop_distance': front_stop_distance,
            'front_slow_distance': front_slow_distance,
            'max_linear': max_linear,
            'max_angular': max_angular,
            'kp_dist': 0.85,
            'kp_yaw': 2.20,
            'front_sector_deg': 42.0,
            'side_sector_deg': 75.0,
            'avoid_turn_speed': 0.75,
            'stale_timeout_sec': 1.5,
            'use_cmd_relay_fallback': True,
            'allow_follow_without_scan': True,
            'cmd_relay_scale_linear': 0.90,
            'cmd_relay_scale_angular': 0.90,
            'cmd_relay_timeout_sec': 2.0,
            'log_every_n': 20,
        }],
    )

    return LaunchDescription([
        DeclareLaunchArgument('follow_distance', default_value='0.95'),
        DeclareLaunchArgument('min_leader_distance', default_value='0.55'),
        DeclareLaunchArgument('hard_min_leader_distance', default_value='0.38'),
        DeclareLaunchArgument('front_stop_distance', default_value='0.42'),
        DeclareLaunchArgument('front_slow_distance', default_value='0.75'),
        DeclareLaunchArgument('max_linear', default_value='0.22'),
        DeclareLaunchArgument('max_angular', default_value='1.20'),
        LogInfo(msg='V19_HYBRID_WAFFLE_LEADER_BURGER_FOLLOWER | Odom follow + /waffle/cmd_vel relay fallback + front-stop guard.'),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(base_launch),
            launch_arguments={
                'burger_x': '-2.9',
                'burger_y': '0.5',
                'burger_yaw': '0.0',
                'waffle_x': '-1.8',
                'waffle_y': '0.5',
                'waffle_yaw': '0.0',
            }.items(),
        ),
        TimerAction(period=8.0, actions=[follower]),
    ])
