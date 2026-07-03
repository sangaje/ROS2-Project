#!/usr/bin/env python3
"""
Leader stack for simulation.
Cartographer SLAM + Nav2.
Expects leader burger robot already running in Gazebo (fleet_sim_gazebo_world).
"""

import os
import sys

from ament_index_python.packages import get_package_share_directory
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
from launch_ros.actions import Node
from nav2_common.launch import RewrittenYaml


def _arg(name: str, default: str) -> str:
    for a in sys.argv:
        if a.startswith(f'{name}:='):
            return a.split(':=', 1)[1]
    return default


def generate_launch_description():
    _did = _arg('domain_id', '25')

    os.environ['ROS_DOMAIN_ID'] = _did
    os.environ['RMW_IMPLEMENTATION'] = 'rmw_fastrtps_cpp'
    os.environ.pop('ROS_DISCOVERY_SERVER', None)
    os.environ.pop('ROS_LOCALHOST_ONLY', None)
    os.environ.pop('FASTRTPS_DEFAULT_PROFILES_FILE', None)
    os.environ.pop('FASTDDS_DEFAULT_PROFILES_FILE', None)

    bringup_share = get_package_share_directory('tb3_fleet_bringup')

    domain_id          = LaunchConfiguration('domain_id')
    initial_x          = LaunchConfiguration('initial_x')
    initial_y          = LaunchConfiguration('initial_y')
    initial_yaw        = LaunchConfiguration('initial_yaw')
    follower_initial_x = LaunchConfiguration('follower_initial_x')
    follower_initial_y = LaunchConfiguration('follower_initial_y')

    # Use the burger slam yaml, rewrite for sim
    nav2_params = RewrittenYaml(
        source_file=os.path.join(bringup_share, 'config', 'domain25_waffle_nav2.yaml'),
        param_rewrites={
            'use_sim_time': 'true',
            'odom_topic': '/odom',
            'topic': '/scan',
        },
        convert_types=True,
    )

    cartographer_config_dir  = os.path.join(bringup_share, 'config')
    tf_pose_script           = os.path.join(bringup_share, 'scripts', 'tf_pose_publisher_direct_v44.py')
    goal_proxy_script        = os.path.join(bringup_share, 'scripts', 'pose_to_nav2_action_direct_v41.py')
    pose_tf_script = os.path.join(bringup_share, 'scripts', 'pose_to_tf_broadcaster.py')

    cartographer = Node(
        package='cartographer_ros', executable='cartographer_node',
        name='cartographer_node', output='screen',
        parameters=[{'use_sim_time': True}],
        arguments=[
            '-configuration_directory', cartographer_config_dir,
            '-configuration_basename', 'cartographer_2d_lidar_odom_v44.lua',
        ],
    )
    occ_grid = Node(
        package='cartographer_ros', executable='cartographer_occupancy_grid_node',
        name='cartographer_occupancy_grid_node', output='screen',
        parameters=[{'use_sim_time': True}],
        arguments=['-resolution', '0.05', '-publish_period_sec', '1.0'],
    )
    leader_env = {'ROS_DOMAIN_ID': _did, 'RMW_IMPLEMENTATION': 'rmw_fastrtps_cpp'}
    burger_amcl_tf = ExecuteProcess(
        cmd=['python3', pose_tf_script, '--ros-args',
             '-r', '__node:=sim_burger_amcl_tf_on_leader_domain',
             '-p', 'use_sim_time:=true',
             '-p', 'input_topic:=/burger_pose',
             '-p', 'parent_frame:=map',
             '-p', 'child_frame:=burger/base_footprint',
             '-p', 'republish_hz:=10.0'],
        output='screen', name='sim_burger_amcl_tf_on_leader_domain',
        additional_env=leader_env,
    )
    leader_pose = ExecuteProcess(
        cmd=['python3', tf_pose_script, '--ros-args',
             '-r', '__node:=sim_leader_pose_publisher',
             '-p', 'use_sim_time:=true',
             '-p', 'target_frame:=map', '-p', 'source_frame:=base_footprint',
             '-p', 'output_topic:=/leader_pose',
             '-p', 'publish_rate_hz:=10.0', '-p', 'log_every_n:=100'],
        output='screen', name='sim_leader_pose_publisher',
        additional_env=leader_env,
    )
    controller_server = Node(package='nav2_controller', executable='controller_server',
                             output='screen', parameters=[nav2_params])
    planner_server    = Node(package='nav2_planner', executable='planner_server',
                             output='screen', parameters=[nav2_params])
    behavior_server   = Node(package='nav2_behaviors', executable='behavior_server',
                             output='screen', parameters=[nav2_params])
    bt_navigator      = Node(package='nav2_bt_navigator', executable='bt_navigator',
                             output='screen', parameters=[nav2_params])
    lifecycle_nav     = Node(package='nav2_lifecycle_manager', executable='lifecycle_manager',
                             name='lifecycle_manager_navigation', output='screen',
                             parameters=[nav2_params])
    named_goal = ExecuteProcess(
        cmd=['python3', goal_proxy_script, '--ros-args',
             '-r', '__node:=sim_leader_goal_to_nav2',
             '-p', 'use_sim_time:=true',
             '-p', 'goal_pose_topic:=/goal_pose',
             '-p', 'navigate_action:=/navigate_to_pose',
             '-p', 'default_frame_id:=map', '-p', 'cancel_previous_goal:=true'],
        output='screen', name='sim_leader_goal_to_nav2',
        additional_env=leader_env,
    )

    return LaunchDescription([
        DeclareLaunchArgument('domain_id',          default_value=_did,
                              description='ROS domain ID for leader (default 25)'),
        DeclareLaunchArgument('initial_x',          default_value='-1.5'),
        DeclareLaunchArgument('initial_y',          default_value='-0.5'),
        DeclareLaunchArgument('initial_yaw',        default_value='0.0'),
        DeclareLaunchArgument('follower_initial_x', default_value='-1.0',
                              description='Follower start X in map frame (leader=origin)'),
        DeclareLaunchArgument('follower_initial_y', default_value='0.0'),
        UnsetEnvironmentVariable('ROS_DISCOVERY_SERVER'),
        UnsetEnvironmentVariable('ROS_LOCALHOST_ONLY'),
        UnsetEnvironmentVariable('FASTRTPS_DEFAULT_PROFILES_FILE'),
        UnsetEnvironmentVariable('FASTDDS_DEFAULT_PROFILES_FILE'),
        SetEnvironmentVariable('ROS_DOMAIN_ID',               domain_id),
        SetEnvironmentVariable('RMW_IMPLEMENTATION',          'rmw_fastrtps_cpp'),
        SetEnvironmentVariable('TURTLEBOT3_MODEL',            'burger'),
        LogInfo(msg=['SIM_LEADER | domain=', domain_id, ' | Cartographer SLAM + Nav2']),
        TimerAction(period=1.0, actions=[cartographer, occ_grid, burger_amcl_tf]),
        TimerAction(period=2.0, actions=[leader_pose]),
        TimerAction(period=3.0, actions=[controller_server, planner_server,
                                          behavior_server, bt_navigator]),
        TimerAction(period=6.0, actions=[lifecycle_nav]),
        TimerAction(period=8.0, actions=[named_goal]),
    ])
