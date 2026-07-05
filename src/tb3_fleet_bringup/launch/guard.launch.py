#!/usr/bin/env python3
"""Guard stack: domain bridge, AMCL and Nav2 for a passive fleet member.

The guard never leads and never follows. It only reports its pose to the
coordinator and executes whatever short yield/return goal the coordinator
sends on /guard_goal_pose when the leader or follower needs to pass.
"""

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

from tb3_fleet_bringup.domain_bridge_config import write_guard_bridge_configs
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

    domain_id = LaunchConfiguration('domain_id')
    main_domain_id = LaunchConfiguration('main_domain_id')
    start_robot_bringup = LaunchConfiguration('start_robot_bringup')
    initial_x = LaunchConfiguration('guard_initial_x')
    initial_y = LaunchConfiguration('guard_initial_y')
    initial_yaw = LaunchConfiguration('guard_initial_yaw')
    auto_localize = LaunchConfiguration('auto_localize')
    enable_amcl = LaunchConfiguration('enable_amcl')

    def make_stack(context):
        guard_domain = int(domain_id.perform(context))
        main_domain = int(main_domain_id.perform(context))
        process_env = clean_process_environment(str(guard_domain))

        # Reuses the follower's Burger Nav2/AMCL tuning; the guard is the
        # same robot class localizing against the same shared map.
        nav2_params = RewrittenYaml(
            source_file=os.path.join(
                package_share, 'config', 'follower_nav2_amcl.yaml'
            ),
            param_rewrites={
                'use_sim_time': 'false',
                'odom_topic': '/odom',
                'scan_topic': '/scan',
                'topic': '/scan',
                'enable_stamped_cmd_vel': 'true',
            },
            convert_types=True,
        )

        main_to_guard, guard_to_main = write_guard_bridge_configs(
            main_domain, guard_domain,
        )
        bridges = [
            Node(
                package='domain_bridge',
                executable='domain_bridge',
                name='bridge_main_to_guard',
                output='screen',
                arguments=[
                    str(main_to_guard), '--wait-for-publisher', 'false',
                ],
                env=process_env,
                respawn=True,
                respawn_delay=3.0,
            ),
            Node(
                package='domain_bridge',
                executable='domain_bridge',
                name='bridge_guard_to_main',
                output='screen',
                arguments=[
                    str(guard_to_main), '--wait-for-publisher', 'false',
                ],
                env=process_env,
                respawn=True,
                respawn_delay=3.0,
            ),
        ]

        map_relay = Node(
            package='tb3_fleet_bringup',
            executable='map_relay',
            name='guard_map_relay',
            output='screen',
            parameters=[{'use_sim_time': False}],
            env=process_env,
            respawn=True,
            respawn_delay=3.0,
        )
        guard_pose = Node(
            package='tb3_fleet_bringup',
            executable='tf_pose_publisher',
            name='guard_pose_pub',
            output='screen',
            parameters=[{
                'use_sim_time': False,
                'output_topic': '/guard_pose',
                'publish_rate_hz': 10.0,
                'log_every_n': 100,
            }],
            env=process_env,
            respawn=True,
            respawn_delay=3.0,
        )

        amcl_enabled = launch_bool(enable_amcl.perform(context))
        auto = launch_bool(auto_localize.perform(context))

        amcl = None
        localization_lifecycle = None
        if amcl_enabled:
            # AMCL is the one TF source this stack owns by default (map->odom
            # over a shared, bridged map). It is deliberately the only thing
            # that publishes that transform here; anything else that wants
            # to own SLAM/TF for this robot (e.g. a risk-map Cartographer)
            # must come with enable_amcl:=false so the two never collide.
            pose_override = Path(tempfile.gettempdir()) / (
                f'tb3_guard_{guard_domain}_initial_pose.yaml'
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
        if amcl_enabled and auto:
            kickstart_node = Node(
                package='tb3_fleet_bringup',
                executable='global_localize_kickstart',
                name='guard_global_localize',
                output='screen',
                parameters=[{
                    'spin_enabled': True,
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
            name='guard_coord_goal',
            output='screen',
            parameters=[{
                'use_sim_time': False,
                'goal_pose_topic': '/guard_goal_pose',
            }],
            env=process_env,
        )

        actions = []
        if launch_bool(start_robot_bringup.perform(context)):
            actions.append(IncludeLaunchDescription(
                PythonLaunchDescriptionSource(robot_launch),
                launch_arguments={
                    'use_sim_time': 'false',
                    'namespace': '',
                }.items(),
            ))

        timing = (0.5, 1.0, 5.0, 5.5, 6.5, 8.0, 12.0)
        (
            bridge_t, relay_t, amcl_t, localization_t, kickstart_t,
            nav_t, lifecycle_t,
        ) = timing
        actions.extend([
            TimerAction(period=bridge_t, actions=bridges),
            TimerAction(period=relay_t, actions=[map_relay, guard_pose]),
            TimerAction(period=nav_t, actions=navigation),
            TimerAction(
                period=lifecycle_t,
                actions=[navigation_lifecycle, goal_proxy],
            ),
        ])
        if amcl is not None:
            actions.append(TimerAction(period=amcl_t, actions=[amcl]))
        if localization_lifecycle is not None:
            actions.append(TimerAction(
                period=localization_t, actions=[localization_lifecycle],
            ))
        if kickstart_node is not None:
            actions.append(
                TimerAction(period=kickstart_t, actions=[kickstart_node])
            )
        return actions

    return LaunchDescription([
        DeclareLaunchArgument(
            'domain_id',
            default_value=EnvironmentVariable('ROS_DOMAIN_ID', default_value='26'),
            description='Guard DDS domain.',
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
            description='Start TurtleBot3 hardware drivers.',
        ),
        DeclareLaunchArgument('guard_initial_x', default_value='0.0'),
        DeclareLaunchArgument('guard_initial_y', default_value='0.0'),
        DeclareLaunchArgument('guard_initial_yaw', default_value='0.0'),
        DeclareLaunchArgument(
            'enable_amcl',
            default_value='true',
            choices=['true', 'false'],
            description=(
                'Run AMCL as this stack\'s map->odom TF source. Set false '
                'only when something else (e.g. a risk-map Cartographer) '
                'will own SLAM/TF for this robot instead -- running both '
                'at once fights over the same transform.'
            ),
        ),
        DeclareLaunchArgument(
            'auto_localize',
            default_value='true',
            choices=['true', 'false'],
            description=(
                'Let AMCL search the whole map via '
                'reinitialize_global_localization instead of trusting '
                'guard_initial_x/y/yaw. Set false to fall back to the '
                'fixed seed.'
            ),
        ),
        *dds_launch_environment(domain_id),
        OpaqueFunction(function=make_stack),
    ])
