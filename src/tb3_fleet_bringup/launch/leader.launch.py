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
    base_launch = os.path.join(package_share, 'launch', 'base.launch.py')
    cartographer_launch = os.path.join(
        get_package_share_directory('turtlebot3_cartographer'),
        'launch',
        'cartographer.launch.py',
    )

    use_sim_time = LaunchConfiguration('use_sim_time')
    domain_id = LaunchConfiguration('domain_id')
    start_robot_bringup = LaunchConfiguration('start_robot_bringup')
    require_follower_pose = LaunchConfiguration('require_follower_pose')

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
            respawn=True,
            respawn_delay=3.0,
        )
        follower_tf = Node(
            package='tb3_fleet_bringup',
            executable='pose_to_tf',
            name='burger_tf_on_leader',
            output='screen',
            parameters=[{'use_sim_time': simulation}],
            env=process_env,
            respawn=True,
            respawn_delay=3.0,
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

        # base.launch.py owns the Nav2 core + goal proxy in real mode, so
        # leader/follower/member all get the exact same node definitions
        # instead of three copies. Simulation keeps its own inline copy
        # below since base.launch.py is real-hardware only (no Gazebo
        # clock/sim sensor path).
        navigation = None
        lifecycle = None
        goal_nodes = None
        base_include = None
        if simulation:
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
        else:
            base_include = IncludeLaunchDescription(
                PythonLaunchDescriptionSource(base_launch),
                launch_arguments={
                    'domain_id': domain,
                    'start_robot_bringup': start_robot_bringup.perform(context),
                    'nav2_params_file': nav2_params,
                    'goal_pose_topic': '/fleet/leader_coord_goal',
                    'goal_proxy_name': 'leader_goal_arbiter_output',
                    'nav_delay_sec': '12.0',
                    'lifecycle_delay_sec': '16.0',
                    'goal_delay_sec': '18.0',
                }.items(),
            )

        coordinator = Node(
            package='tb3_fleet_bringup',
            executable='fleet_path_coordinator',
            name='fleet_path_coordinator',
            output='screen',
            parameters=[{
                'use_sim_time': simulation,
                'require_follower_pose': launch_bool(
                    require_follower_pose.perform(context)
                ),
            }],
            env=process_env,
            respawn=True,
            respawn_delay=3.0,
        )

        actions = []
        if base_include is not None:
            # Real mode: base.launch.py brings up the hardware drivers
            # itself (via start_robot_bringup), so it is not also included
            # directly here.
            actions.append(base_include)

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

        if simulation:
            timing = (0.5, 1.0, 2.0, 5.0, 7.0, 9.0)
            (
                cartographer_t, pose_t, nav_t, lifecycle_t, goals_t,
                coordinator_t,
            ) = timing
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
        else:
            # nav/lifecycle/goal timing for real mode lives inside
            # base.launch.py (nav_delay_sec=12.0, lifecycle_delay_sec=16.0,
            # goal_delay_sec=18.0 above), measured from this same t=0.
            cartographer_t, pose_t, coordinator_t = 5.0, 8.0, 20.0
            actions.extend([
                TimerAction(period=cartographer_t, actions=[cartographer]),
                TimerAction(
                    period=pose_t,
                    actions=[leader_pose, follower_tf, leader_scan],
                ),
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
        DeclareLaunchArgument(
            'require_follower_pose',
            default_value='true',
            choices=['true', 'false'],
            description=(
                'Whether a follower.launch.py robot is expected in this '
                'fleet. When true (default) the coordinator holds the '
                "leader in place until BOTH /leader_pose and /burger_pose "
                'are fresh -- correct for a leader+follower fleet, but it '
                'freezes the leader forever if no follower ever '
                'publishes. Set false for a leader-only or '
                'leader+member fleet with no follower robot.'
            ),
        ),
        *dds_launch_environment(domain_id),
        OpaqueFunction(function=make_stack),
    ])
