#!/usr/bin/env python3
"""Follower stack: domain bridge, AMCL, Nav2 and leader following."""

import os
import tempfile
from pathlib import Path

import yaml
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

from tb3_fleet_bringup.domain_bridge_config import write_fleet_bridge_configs
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

    use_sim_time = LaunchConfiguration('use_sim_time')
    domain_id = LaunchConfiguration('domain_id')
    main_domain_id = LaunchConfiguration('main_domain_id')
    start_robot_bringup = LaunchConfiguration('start_robot_bringup')
    follow_distance = LaunchConfiguration('follow_distance')
    initial_x = LaunchConfiguration('follower_initial_x')
    initial_y = LaunchConfiguration('follower_initial_y')
    initial_yaw = LaunchConfiguration('follower_initial_yaw')
    auto_localize = LaunchConfiguration('auto_localize')

    def make_stack(context):
        simulation = launch_bool(use_sim_time.perform(context))
        follower_domain = int(domain_id.perform(context))
        main_domain = int(main_domain_id.perform(context))
        process_env = clean_process_environment(str(follower_domain))

        nav2_source = os.path.join(
            package_share,
            'config',
            (
                'follower_nav2_amcl_sim.yaml'
                if simulation
                else 'follower_nav2_amcl.yaml'
            ),
        )
        nav2_params = RewrittenYaml(
            source_file=nav2_source,
            param_rewrites={
                'use_sim_time': str(simulation).lower(),
                'odom_topic': '/odom',
                'scan_topic': '/scan',
                'topic': '/scan',
                'enable_stamped_cmd_vel': 'true',
            },
            convert_types=True,
        )

        main_to_follower, follower_to_main = write_fleet_bridge_configs(
            main_domain,
            follower_domain,
            simulation=simulation,
        )
        bridges = [
            Node(
                package='domain_bridge',
                executable='domain_bridge',
                name='bridge_main_to_follower',
                output='screen',
                arguments=[
                    str(main_to_follower),
                    '--wait-for-publisher',
                    'false',
                ],
                env=process_env,
                respawn=True,
                respawn_delay=3.0,
            ),
            Node(
                package='domain_bridge',
                executable='domain_bridge',
                name='bridge_follower_to_main',
                output='screen',
                arguments=[
                    str(follower_to_main),
                    '--wait-for-publisher',
                    'false',
                ],
                env=process_env,
                respawn=True,
                respawn_delay=3.0,
            ),
        ]

        map_relay = Node(
            package='tb3_fleet_bringup',
            executable='map_relay',
            name='follower_map_relay',
            output='screen',
            parameters=[{'use_sim_time': simulation}],
            env=process_env,
            respawn=True,
            respawn_delay=3.0,
        )
        follower_pose = Node(
            package='tb3_fleet_bringup',
            executable='tf_pose_publisher',
            name='burger_pose_pub',
            output='screen',
            parameters=[{
                'use_sim_time': simulation,
                'output_topic': '/burger_pose',
                'publish_rate_hz': 10.0,
                'log_every_n': 100,
            }],
            env=process_env,
            respawn=True,
            respawn_delay=3.0,
        )

        relay_nodes = [map_relay]
        robot_state_publisher = None
        if simulation:
            gazebo_share = get_package_share_directory('turtlebot3_gazebo')
            robot_description = Path(
                gazebo_share, 'urdf', 'turtlebot3_burger.urdf'
            ).read_text(encoding='utf-8')
            robot_state_publisher = Node(
                package='robot_state_publisher',
                executable='robot_state_publisher',
                name='robot_state_publisher',
                output='screen',
                parameters=[{
                    'use_sim_time': True,
                    'robot_description': robot_description,
                }],
                env=process_env,
            )
            relay_nodes.extend([
                Node(
                    package='tb3_fleet_bringup',
                    executable='sim_burger_tf_relay',
                    name='follower_tf_relay',
                    output='screen',
                    parameters=[{'use_sim_time': True}],
                    env=process_env,
                ),
                Node(
                    package='tb3_fleet_bringup',
                    executable='sim_burger_scan_relay',
                    name='follower_scan_relay',
                    output='screen',
                    parameters=[{'use_sim_time': True}],
                    env=process_env,
                ),
            ])
        else:
            relay_nodes.append(Node(
                package='tb3_fleet_bringup',
                executable='scan_frame_relay',
                name='burger_scan_relay',
                output='screen',
                parameters=[{
                    'input_topic': '/scan',
                    'output_topic': '/burger_scan_relay',
                    'output_frame': 'burger/base_scan',
                    'input_reliability': 'best_effort',
                    'output_reliability': 'reliable',
                }],
                env=process_env,
            ))

        auto = launch_bool(auto_localize.perform(context))
        pose_override = Path(tempfile.gettempdir()) / (
            f'tb3_follower_{follower_domain}_initial_pose.yaml'
        )
        if auto:
            # Let AMCL search the whole map instead of trusting a fixed
            # seed; global_localize_kickstart triggers the actual search
            # once the localization stack is active.
            amcl_overrides = {'set_initial_pose': False}
        else:
            amcl_overrides = {
                'set_initial_pose': True,
                'initial_pose': {
                    'x': float(initial_x.perform(context)),
                    'y': float(initial_y.perform(context)),
                    'z': 0.0,
                    'yaw': float(initial_yaw.perform(context)),
                },
            }
        pose_override.write_text(yaml.safe_dump({
            'amcl': {'ros__parameters': amcl_overrides},
        }), encoding='utf-8')

        amcl = Node(
            package='nav2_amcl',
            executable='amcl',
            name='amcl',
            output='screen',
            parameters=[nav2_params, str(pose_override)],
            env=process_env,
            respawn=True,
            respawn_delay=3.0,
        )
        localization_lifecycle = Node(
            package='nav2_lifecycle_manager',
            executable='lifecycle_manager',
            name='lifecycle_manager_localization',
            output='screen',
            parameters=[nav2_params],
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
        navigation_lifecycle = Node(
            package='nav2_lifecycle_manager',
            executable='lifecycle_manager',
            name='lifecycle_manager_navigation',
            output='screen',
            parameters=[nav2_params],
            env=process_env,
        )

        kickstart_node = None
        if auto:
            kickstart_node = Node(
                package='tb3_fleet_bringup',
                executable='global_localize_kickstart',
                name='burger_global_localize',
                output='screen',
                parameters=[{
                    'use_sim_time': simulation,
                    'spin_enabled': not simulation,
                    'spin_duration_sec': 8.0,
                    'spin_speed_rad_s': 0.6,
                    'cmd_vel_topic': '/cmd_vel',
                    'use_stamped_cmd_vel': True,
                }],
                env=process_env,
            )

        goal_proxy = Node(
            package='tb3_fleet_bringup',
            executable='pose_to_nav2',
            name='burger_named_goal',
            output='screen',
            parameters=[{
                'use_sim_time': simulation,
                'goal_pose_topic': '/burger_goal_pose',
            }],
            env=process_env,
        )
        follower = Node(
            package='tb3_fleet_bringup',
            executable='fleet_follower',
            name='fleet_follower',
            output='screen',
            parameters=[{
                'use_sim_time': simulation,
                'follow_distance': float(follow_distance.perform(context)),
                'start_following': True,
            }],
            env=process_env,
            respawn=True,
            respawn_delay=3.0,
        )

        actions = []
        if robot_state_publisher is not None:
            actions.append(robot_state_publisher)
        elif launch_bool(start_robot_bringup.perform(context)):
            actions.append(IncludeLaunchDescription(
                PythonLaunchDescriptionSource(robot_launch),
                launch_arguments={
                    'use_sim_time': 'false',
                    'namespace': '',
                }.items(),
            ))

        timing = (
            (0.5, 1.0, 5.0, 5.5, 6.5, 8.0, 12.0, 14.0)
            if not simulation
            else (0.5, 1.0, 5.0, 5.5, 6.5, 7.0, 10.0, 13.0)
        )
        (
            bridge_t, relay_t, amcl_t, localization_t, kickstart_t,
            nav_t, lifecycle_t, behavior_t,
        ) = timing
        actions.extend([
            TimerAction(period=bridge_t, actions=bridges),
            TimerAction(period=relay_t, actions=relay_nodes + [follower_pose]),
            TimerAction(period=amcl_t, actions=[amcl]),
            TimerAction(period=localization_t, actions=[localization_lifecycle]),
            TimerAction(period=nav_t, actions=navigation),
            TimerAction(period=lifecycle_t, actions=[navigation_lifecycle]),
            TimerAction(period=behavior_t, actions=[goal_proxy, follower]),
        ])
        if kickstart_node is not None:
            actions.append(
                TimerAction(period=kickstart_t, actions=[kickstart_node])
            )
        return actions

    return LaunchDescription([
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='false',
            choices=['true', 'false'],
            description='Use the Gazebo clock and simulated sensor relays.',
        ),
        DeclareLaunchArgument(
            'domain_id',
            default_value=EnvironmentVariable('ROS_DOMAIN_ID', default_value='25'),
            description='Follower DDS domain.',
        ),
        DeclareLaunchArgument(
            'main_domain_id',
            default_value='24',
            description='Leader/PC DDS domain used by domain_bridge.',
        ),
        DeclareLaunchArgument(
            'start_robot_bringup',
            default_value='true',
            choices=['true', 'false'],
            description='Start TurtleBot3 hardware drivers in real mode.',
        ),
        DeclareLaunchArgument(
            'follow_distance',
            default_value='0.70',
            description='Desired distance behind the leader in metres.',
        ),
        DeclareLaunchArgument('follower_initial_x', default_value='-0.70'),
        DeclareLaunchArgument('follower_initial_y', default_value='0.0'),
        DeclareLaunchArgument('follower_initial_yaw', default_value='0.0'),
        DeclareLaunchArgument(
            'auto_localize',
            default_value='true',
            choices=['true', 'false'],
            description=(
                'Let AMCL search the whole map via '
                'reinitialize_global_localization instead of trusting '
                'follower_initial_x/y/yaw. Set false to fall back to the '
                'fixed seed.'
            ),
        ),
        *dds_launch_environment(domain_id),
        OpaqueFunction(function=make_stack),
    ])
