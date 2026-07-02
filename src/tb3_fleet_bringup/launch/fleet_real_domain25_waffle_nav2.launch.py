#!/usr/bin/env python3

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    LogInfo,
    SetEnvironmentVariable,
    TimerAction,
    UnsetEnvironmentVariable,
)
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node
from nav2_common.launch import RewrittenYaml


def generate_launch_description():
    bringup_share = get_package_share_directory('tb3_fleet_bringup')

    domain_id = LaunchConfiguration('domain_id')
    robot_model = LaunchConfiguration('robot_model')

    waffle_nav2_source = os.path.join(
        bringup_share, 'config', 'domain25_waffle_nav2.yaml'
    )
    burger_nav2_source = os.path.join(
        bringup_share, 'config', 'domain26_burger_nav2_slam.yaml'
    )
    nav2_source = PythonExpression([
        "'", burger_nav2_source, "' if '", robot_model,
        "' == 'burger' else '", waffle_nav2_source, "'",
    ])
    nav2_params = RewrittenYaml(
        source_file=nav2_source,
        param_rewrites={
            'use_sim_time': 'false',
            'odom_topic': '/odom',
            'topic': '/scan',
        },
        convert_types=True,
    )

    cartographer_config_dir = os.path.join(bringup_share, 'config')
    tf_pose_script = os.path.join(
        bringup_share, 'scripts', 'tf_pose_publisher_direct_v44.py'
    )
    goal_proxy_script = os.path.join(
        bringup_share, 'scripts', 'pose_to_nav2_action_direct_v41.py'
    )

    cartographer = Node(
        package='cartographer_ros',
        executable='cartographer_node',
        name='cartographer_node',
        output='screen',
        parameters=[{'use_sim_time': False}],
        arguments=[
            '-configuration_directory', cartographer_config_dir,
            '-configuration_basename', 'cartographer_2d_lidar_odom_v44.lua',
        ],
    )
    occupancy_grid = Node(
        package='cartographer_ros',
        executable='cartographer_occupancy_grid_node',
        name='cartographer_occupancy_grid_node',
        output='screen',
        parameters=[{'use_sim_time': False}],
        arguments=['-resolution', '0.05', '-publish_period_sec', '1.0'],
    )

    leader_pose = ExecuteProcess(
        cmd=[
            'python3', tf_pose_script, '--ros-args',
            '-r', '__node:=waffle_real_leader_pose_publisher',
            '-p', 'use_sim_time:=false',
            '-p', 'target_frame:=map',
            '-p', 'source_frame:=base_footprint',
            '-p', 'output_topic:=/leader_pose',
            '-p', 'publish_rate_hz:=10.0',
            '-p', 'log_every_n:=100',
        ],
        output='screen',
        name='waffle_real_leader_pose_publisher',
    )

    controller_server = Node(
        package='nav2_controller',
        executable='controller_server',
        name='controller_server',
        output='screen',
        parameters=[nav2_params],
    )
    planner_server = Node(
        package='nav2_planner',
        executable='planner_server',
        name='planner_server',
        output='screen',
        parameters=[nav2_params],
    )
    behavior_server = Node(
        package='nav2_behaviors',
        executable='behavior_server',
        name='behavior_server',
        output='screen',
        parameters=[nav2_params],
    )
    bt_navigator = Node(
        package='nav2_bt_navigator',
        executable='bt_navigator',
        name='bt_navigator',
        output='screen',
        parameters=[nav2_params],
    )
    lifecycle = Node(
        package='nav2_lifecycle_manager',
        executable='lifecycle_manager',
        name='lifecycle_manager_navigation',
        output='screen',
        parameters=[nav2_params],
    )

    default_goal = ExecuteProcess(
        cmd=[
            'python3', goal_proxy_script, '--ros-args',
            '-r', '__node:=waffle_real_default_goal_to_nav2',
            '-p', 'use_sim_time:=false',
            '-p', 'goal_pose_topic:=/goal_pose',
            '-p', 'navigate_action:=/navigate_to_pose',
            '-p', 'default_frame_id:=map',
            '-p', 'cancel_previous_goal:=true',
        ],
        output='screen',
        name='waffle_real_default_goal_to_nav2',
    )
    named_goal = ExecuteProcess(
        cmd=[
            'python3', goal_proxy_script, '--ros-args',
            '-r', '__node:=waffle_real_named_goal_to_nav2',
            '-p', 'use_sim_time:=false',
            '-p', 'goal_pose_topic:=/waffle_goal_pose',
            '-p', 'navigate_action:=/navigate_to_pose',
            '-p', 'default_frame_id:=map',
            '-p', 'cancel_previous_goal:=true',
        ],
        output='screen',
        name='waffle_real_named_goal_to_nav2',
    )

    return LaunchDescription([
        DeclareLaunchArgument('domain_id', default_value='25'),
        DeclareLaunchArgument(
            'robot_model',
            default_value='waffle_pi',
            description='Physical TurtleBot3 model: waffle, waffle_pi, or burger.',
        ),
        UnsetEnvironmentVariable('ROS_DISCOVERY_SERVER'),
        UnsetEnvironmentVariable('ROS_LOCALHOST_ONLY'),
        UnsetEnvironmentVariable('FASTRTPS_DEFAULT_PROFILES_FILE'),
        UnsetEnvironmentVariable('FASTDDS_DEFAULT_PROFILES_FILE'),
        SetEnvironmentVariable('ROS_DOMAIN_ID', domain_id),
        SetEnvironmentVariable('RMW_IMPLEMENTATION', 'rmw_fastrtps_cpp'),
        SetEnvironmentVariable('FASTDDS_BUILTIN_TRANSPORTS', 'UDPv4'),
        SetEnvironmentVariable('ROS_AUTOMATIC_DISCOVERY_RANGE', 'SUBNET'),
        SetEnvironmentVariable('TURTLEBOT3_MODEL', robot_model),
        LogInfo(
            msg=[
                'REAL_LEADER_DOMAIN25 | model=', robot_model,
                ' | Cartographer SLAM → map→odom TF | /leader_pose → bridge → Domain24',
            ]
        ),
        TimerAction(period=0.5, actions=[cartographer, occupancy_grid]),
        TimerAction(period=1.0, actions=[leader_pose]),
        TimerAction(
            period=2.0,
            actions=[controller_server, planner_server, behavior_server, bt_navigator],
        ),
        TimerAction(period=5.0, actions=[lifecycle]),
        TimerAction(period=7.0, actions=[default_goal, named_goal]),
    ])
