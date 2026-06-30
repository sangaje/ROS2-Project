#!/usr/bin/env python3
import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, LogInfo, SetEnvironmentVariable
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    bringup_share = get_package_share_directory('tb3_fleet_bringup')
    astar_script = os.path.join(bringup_share, 'scripts', 'astar_cmd_vel_follower_direct_v55.py')
    target_mode = LaunchConfiguration('astar_target_mode')
    domain_id = LaunchConfiguration('domain_id')

    burger_astar = ExecuteProcess(
        cmd=[
            'python3', astar_script,
            '--ros-args',
            '-r', '__node:=burger_astar_cmd_vel_follower',
            '-p', 'use_sim_time:=true',
            '-p', 'robot_name:=burger',
            '-p', 'map_topic:=/map',
            '-p', 'robot_pose_topic:=/burger_pose',
            '-p', 'leader_pose_topic:=/leader_pose',
            '-p', 'manual_goal_topic:=/burger_goal_pose',
            '-p', 'scan_topic:=/scan_nav',
            '-p', 'cmd_vel_topic:=/cmd_vel',
            '-p', 'path_topic:=/astar_path',
            '-p', ['target_mode:=', target_mode],
            '-p', 'leader_goal_mode:=line_between_robots',
            '-p', 'follow_distance:=0.85',
            '-p', 'goal_tolerance:=0.25',
            '-p', 'replan_period_sec:=0.7',
            '-p', 'control_rate_hz:=10.0',
            '-p', 'lookahead_distance:=0.28',
            '-p', 'occupied_threshold:=45',
            '-p', 'treat_unknown_as_obstacle:=false',
            '-p', 'inflation_radius_m:=0.22',
            '-p', 'soft_inflation_radius_m:=0.42',
            '-p', 'clearance_cost_weight:=7.0',
            '-p', 'unknown_cost_weight:=3.0',
            '-p', 'diagonal_motion:=false',
            '-p', 'max_linear:=0.075',
            '-p', 'min_linear:=0.02',
            '-p', 'max_angular:=0.50',
            '-p', 'front_stop_distance:=0.35',
            '-p', 'front_slow_distance:=0.60',
            '-p', 'stale_goal_sec:=0.0',
            '-p', 'direct_fallback_if_no_path:=false',
            '-p', 'log_period_sec:=1.0',
        ],
        output='screen',
        name='burger_astar_cmd_vel_follower_only_v55'
    )

    return LaunchDescription([
        DeclareLaunchArgument('domain_id', default_value='24'),
        DeclareLaunchArgument('astar_target_mode', default_value='leader'),
        LogInfo(msg=['V55_BURGER_ASTAR_ONLY | starts forced Burger A* follower mode=', target_mode]),
        SetEnvironmentVariable('ROS_DOMAIN_ID', domain_id),
        SetEnvironmentVariable('RMW_IMPLEMENTATION', 'rmw_fastrtps_cpp'),
        SetEnvironmentVariable('FASTDDS_BUILTIN_TRANSPORTS', 'UDPv4'),
        burger_astar,
    ])
