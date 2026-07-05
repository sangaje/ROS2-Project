#!/usr/bin/env python3
"""Leader stack: TurtleBot3 bringup, Cartographer, Nav2 and fleet coordination."""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    OpaqueFunction,
    TimerAction,
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import EnvironmentVariable, LaunchConfiguration
from launch_ros.actions import Node
from nav2_common.launch import RewrittenYaml

from tb3_fleet_bringup.launch_utils import (
    clean_process_environment,
    dds_launch_environment,
    launch_bool,
)


def generate_launch_description():
    package_share = get_package_share_directory('tb3_fleet_bringup')
    robot_launch = os.path.join(
        get_package_share_directory('turtlebot3_bringup'),
        'launch',
        'robot.launch.py',
    )
    cartographer_launch = os.path.join(
        get_package_share_directory('turtlebot3_cartographer'),
        'launch',
        'cartographer.launch.py',
    )

    use_sim_time = LaunchConfiguration('use_sim_time')
    domain_id = LaunchConfiguration('domain_id')
    start_robot_bringup = LaunchConfiguration('start_robot_bringup')

    def make_stack(context):
        simulation = launch_bool(use_sim_time.perform(context))
        domain = domain_id.perform(context)
        process_env = clean_process_environment(domain)

        nav2_params = RewrittenYaml(
            source_file=os.path.join(
                package_share, 'config', 'leader_nav2.yaml'
            ),
            param_rewrites={
                'use_sim_time': str(simulation).lower(),
                'odom_topic': '/odom',
                'scan_topic': '/scan',
                'topic': '/scan',
                'enable_stamped_cmd_vel': 'true',
            },
            convert_types=True,
        )

        cartographer = IncludeLaunchDescription(
            PythonLaunchDescriptionSource(cartographer_launch),
            launch_arguments={
                'cartographer_config_dir': os.path.join(package_share, 'config'),
                'configuration_basename': 'leader_cartographer.lua',
                'use_sim_time': str(simulation).lower(),
                'use_rviz': 'false',
            }.items(),
        )

        leader_pose = Node(
            package='tb3_fleet_bringup',
            executable='tf_pose_publisher',
            name='leader_pose_pub',
            output='screen',
            parameters=[{
                'use_sim_time': simulation,
                'output_topic': '/leader_pose',
                'publish_rate_hz': 10.0,
                'log_every_n': 100,
            }],
            env=process_env,
        )
        follower_tf = Node(
            package='tb3_fleet_bringup',
            executable='pose_to_tf',
            name='burger_tf_on_leader',
            output='screen',
            parameters=[{'use_sim_time': simulation}],
            env=process_env,
        )
        leader_scan = Node(
            package='tb3_fleet_bringup',
            executable='scan_frame_relay',
            name='leader_fleet_scan_relay',
            output='screen',
            parameters=[{
                'input_topic': '/scan',
                'output_topic': '/leader/scan',
                'output_frame': 'base_scan',
                'input_reliability': 'best_effort',
                'output_reliability': 'reliable',
            }],
            env=process_env,
        )

        navigation = [
            Node(
                package='nav2_controller',
                executable='controller_server',
                name='controller_server',
                output='screen',
                parameters=[nav2_params],
                env=process_env,
            ),
            Node(
                package='nav2_planner',
                executable='planner_server',
                name='planner_server',
                output='screen',
                parameters=[nav2_params],
                env=process_env,
            ),
            Node(
                package='nav2_behaviors',
                executable='behavior_server',
                name='behavior_server',
                output='screen',
                parameters=[nav2_params],
                env=process_env,
            ),
            Node(
                package='nav2_bt_navigator',
                executable='bt_navigator',
                name='bt_navigator',
                output='screen',
                parameters=[nav2_params],
                env=process_env,
            ),
        ]
        lifecycle = Node(
            package='nav2_lifecycle_manager',
            executable='lifecycle_manager',
            name='lifecycle_manager_navigation',
            output='screen',
            parameters=[nav2_params],
            env=process_env,
        )

        goal_nodes = [
            Node(
                package='tb3_fleet_bringup',
                executable='pose_to_nav2',
                name='leader_goal_arbiter_output',
                output='screen',
                parameters=[{
                    'use_sim_time': simulation,
                    'goal_pose_topic': '/fleet/leader_coord_goal',
                }],
                env=process_env,
            ),
        ]
        coordinator = Node(
            package='tb3_fleet_bringup',
            executable='fleet_path_coordinator',
            name='fleet_path_coordinator',
            output='screen',
            parameters=[{'use_sim_time': simulation}],
            env=process_env,
            respawn=True,
            respawn_delay=3.0,
        )

        actions = []
        if (
            not simulation
            and launch_bool(start_robot_bringup.perform(context))
        ):
            actions.append(IncludeLaunchDescription(
                PythonLaunchDescriptionSource(robot_launch),
                launch_arguments={
                    'use_sim_time': 'false',
                    'namespace': '',
                }.items(),
            ))

        if not simulation:
            actions.append(TimerAction(
                period=1.0,
                actions=[Node(
                    package='tf2_ros',
                    executable='static_transform_publisher',
                    name='burger_scan_static_tf',
                    output='screen',
                    arguments=[
                        '--x', '-0.032', '--y', '0.0', '--z', '0.182',
                        '--roll', '0', '--pitch', '0', '--yaw', '0',
                        '--frame-id', 'burger/base_footprint',
                        '--child-frame-id', 'burger/base_scan',
                    ],
                    env=process_env,
                )],
            ))

        timing = (
            (5.0, 8.0, 12.0, 16.0, 18.0, 20.0)
            if not simulation
            else (0.5, 1.0, 2.0, 5.0, 7.0, 9.0)
        )
        cartographer_t, pose_t, nav_t, lifecycle_t, goals_t, coordinator_t = timing
        actions.extend([
            TimerAction(period=cartographer_t, actions=[cartographer]),
            TimerAction(
                period=pose_t,
                actions=[leader_pose, follower_tf, leader_scan],
            ),
            TimerAction(period=nav_t, actions=navigation),
            TimerAction(period=lifecycle_t, actions=[lifecycle]),
            TimerAction(period=goals_t, actions=goal_nodes),
            TimerAction(period=coordinator_t, actions=[coordinator]),
        ])
        return actions

    return LaunchDescription([
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='false',
            choices=['true', 'false'],
            description='Use the Gazebo clock and simulated sensor path.',
        ),
        DeclareLaunchArgument(
            'domain_id',
            default_value=EnvironmentVariable('ROS_DOMAIN_ID', default_value='24'),
            description='Leader DDS domain.',
        ),
        DeclareLaunchArgument(
            'start_robot_bringup',
            default_value='true',
            choices=['true', 'false'],
            description='Start TurtleBot3 hardware drivers in real mode.',
        ),
        *dds_launch_environment(domain_id),
        OpaqueFunction(function=make_stack),
    ])
