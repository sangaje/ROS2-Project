#!/usr/bin/env python3
import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, LogInfo, SetEnvironmentVariable
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    bringup_share = get_package_share_directory('tb3_fleet_bringup')
    astar_script = os.path.join(bringup_share, 'scripts', 'astar_cmd_vel_follower_direct_v55.py')
    astar_goal_topic = LaunchConfiguration('astar_goal_topic')
    domain_id = LaunchConfiguration('domain_id')

    waffle_astar = ExecuteProcess(
        cmd=[
            'python3', astar_script,
            '--ros-args',
            '-r', '__node:=waffle_astar_cmd_vel_controller',
            '-p', 'use_sim_time:=true',
            '-p', 'robot_name:=waffle',
            '-p', 'map_topic:=/map',
            '-p', 'robot_pose_topic:=/leader_pose',
            '-p', 'leader_pose_topic:=/leader_pose',
            '-p', ['manual_goal_topic:=', astar_goal_topic],
            '-p', 'scan_topic:=/scan_nav',
            '-p', 'cmd_vel_topic:=/cmd_vel',
            '-p', 'path_topic:=/waffle_astar_path',
            '-p', 'target_mode:=manual',
            '-p', 'leader_goal_mode:=leader_pose',
            '-p', 'follow_distance:=0.0',
            '-p', 'goal_tolerance:=0.22',
            '-p', 'replan_period_sec:=0.5',
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
        name='waffle_astar_cmd_vel_controller_only_v55'
    )

    return LaunchDescription([
        DeclareLaunchArgument('domain_id', default_value='25'),
        DeclareLaunchArgument('astar_goal_topic', default_value='/goal_pose'),
        LogInfo(msg=['V55_WAFFLE_ASTAR_ONLY | starts forced Waffle A* subscriber on ', astar_goal_topic]),
        SetEnvironmentVariable('ROS_DOMAIN_ID', domain_id),
        SetEnvironmentVariable('RMW_IMPLEMENTATION', 'rmw_fastrtps_cpp'),
        SetEnvironmentVariable('FASTDDS_BUILTIN_TRANSPORTS', 'UDPv4'),
        waffle_astar,
    ])
