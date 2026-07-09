#!/usr/bin/env python3
"""Member stack: domain bridge, AMCL and (via base.launch.py) Nav2 for a
generic fleet member.

A member reports its pose and executes direct Nav2 goals on /member_goal_pose.
follower.launch.py builds on top of this file by adding trailing behaviour;
leader.launch.py is a sibling that builds on base.launch.py directly with
Cartographer instead of AMCL.
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
    LogInfo,
    OpaqueFunction,
    TimerAction,
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import EnvironmentVariable, LaunchConfiguration
from launch_ros.actions import Node
from nav2_common.launch import RewrittenYaml

from fleet_bringup.domain_bridge_config import write_member_bridge_configs
from fleet_bringup.launch_utils import (
    clean_process_environment,
    dds_launch_environment,
    launch_bool,
)


def generate_launch_description():
    package_share = get_package_share_directory('fleet_bringup')
    base_launch = os.path.join(package_share, 'launch', 'base.launch.py')

    domain_id = LaunchConfiguration('domain_id')
    main_domain_id = LaunchConfiguration('main_domain_id')
    start_robot_bringup = LaunchConfiguration('start_robot_bringup')
    start_nav2 = LaunchConfiguration('start_nav2')
    hardware_param_file = LaunchConfiguration('hardware_param_file')
    forward_map_to_main = LaunchConfiguration('forward_map_to_main')
    initial_x = LaunchConfiguration('member_initial_x')
    initial_y = LaunchConfiguration('member_initial_y')
    initial_yaw = LaunchConfiguration('member_initial_yaw')
    auto_localize = LaunchConfiguration('auto_localize')
    enable_amcl = LaunchConfiguration('enable_amcl')

    def make_stack(context):
        member_domain = int(domain_id.perform(context))
        main_domain_value = main_domain_id.perform(context).strip()
        if not main_domain_value:
            raise ValueError(
                'main_domain_id is required for member.launch.py domain_bridge. '
                'Pass the launch option main_domain_id:=<leader_domain>.'
            )
        main_domain = int(main_domain_value)
        process_env = clean_process_environment(str(member_domain))

        # Reuses the follower's Burger Nav2/AMCL tuning; a plain member is
        # the same robot class localizing against the same shared map.
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

        main_to_member, member_to_main = write_member_bridge_configs(
            main_domain,
            member_domain,
            forward_map_to_main=launch_bool(forward_map_to_main.perform(context)),
        )
        bridges = [
            Node(
                package='domain_bridge',
                executable='domain_bridge',
                name='bridge_main_to_member',
                output='screen',
                arguments=[
                    str(main_to_member), '--wait-for-publisher', 'false',
                ],
                env=process_env,
                respawn=True,
                respawn_delay=3.0,
            ),
            Node(
                package='domain_bridge',
                executable='domain_bridge',
                name='bridge_member_to_main',
                output='screen',
                arguments=[
                    str(member_to_main), '--wait-for-publisher', 'false',
                ],
                env=process_env,
                respawn=True,
                respawn_delay=3.0,
            ),
        ]

        map_relay = Node(
            package='fleet_bringup',
            executable='map_relay',
            name='member_map_relay',
            output='screen',
            parameters=[{'use_sim_time': False}],
            env=process_env,
            respawn=True,
            respawn_delay=3.0,
        )
        member_pose = Node(
            package='fleet_bringup',
            executable='tf_pose_publisher',
            name='member_pose_pub',
            output='screen',
            parameters=[{
                'use_sim_time': False,
                'output_topic': '/member_pose',
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
                f'member_{member_domain}_initial_pose.yaml'
            )
            initial_pose = {
                'x': float(initial_x.perform(context)),
                'y': float(initial_y.perform(context)),
                'z': 0.0,
                'yaw': float(initial_yaw.perform(context)),
            }
            amcl_overrides = {
                'set_initial_pose': True,
                'initial_pose': initial_pose,
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
            localization_config_log = LogInfo(msg=[
                'AMCL_INITIAL_POSE_CONFIG | robot=member',
                ' | role=member',
                ' | global_frame=map',
                ' | odom_frame=odom',
                ' | base_frame=base_footprint',
                ' | scan_topic=/scan',
                ' | tf_broadcast=true',
                ' | mode=',
                'seeded_global_localize' if auto else 'fixed_seed',
                ' | x=', str(initial_pose['x']),
                ' | y=', str(initial_pose['y']),
                ' | yaw=', str(initial_pose['yaw']),
            ])

        kickstart_node = None
        if amcl_enabled and auto:
            kickstart_node = Node(
                package='fleet_bringup',
                executable='global_localize_kickstart',
                name='member_global_localize',
                output='screen',
                parameters=[{
                    'spin_enabled': True,
                    'spin_speed_rad_s': 0.40,
                    'spin_target_angle_rad': 7.10,
                    'spin_timeout_sec': 42.0,
                    'spin_sensor_dropout_grace_sec': 1.5,
                    'settle_duration_sec': 3.0,
                    'require_valid_map': True,
                    'min_known_map_cells': 100,
                    'require_scan_before_spin': True,
                    'require_odom_before_spin': True,
                    'require_amcl_before_spin': True,
                    'max_scan_age_sec': 1.2,
                    'max_odom_age_sec': 1.2,
                    'cmd_vel_topic': '/cmd_vel',
                    'use_stamped_cmd_vel': True,
                    'amcl_pose_topic': '/amcl_pose',
                    'localization_cov_xy_threshold': 0.22,
                    'localization_cov_yaw_threshold': 0.16,
                    'localization_stable_duration_sec': 2.5,
                    'localization_check_timeout_sec': 9.0,
                    'max_spin_retries': 3,
                    'force_spin_after_sec': 14.0,
                }],
                env=process_env,
                respawn=True,
                respawn_delay=3.0,
            )

        base = IncludeLaunchDescription(
            PythonLaunchDescriptionSource(base_launch),
            launch_arguments={
                'domain_id': str(member_domain),
                'start_robot_bringup': start_robot_bringup.perform(context),
                'hardware_param_file': (
                    hardware_param_file.perform(context)
                    or os.path.join(
                        package_share,
                        'config',
                        'turtlebot3_burger_stamped_cmd_vel.yaml',
                    )
                ),
                'nav2_params_file': nav2_params,
                'start_nav2': start_nav2.perform(context),
                'goal_pose_topic': '/member_goal_pose',
                'goal_proxy_name': 'member_coord_goal',
                'nav_delay_sec': '8.0',
                'lifecycle_delay_sec': '12.0',
                'require_localization_ready': (
                    'true' if amcl_enabled and auto else 'false'
                ),
                'localization_ready_topic': '/localization_ready',
            }.items(),
        )

        timing = (0.5, 1.0, 5.0, 5.5, 6.5)
        bridge_t, relay_t, amcl_t, localization_t, kickstart_t = timing
        actions = [
            base,
            LogInfo(msg=[
                'LEADER_EGRESS_BRIDGE | source_domain=', str(main_domain),
                ' | destination_domain=', str(member_domain),
                ' | topics=/map,/leader_pose',
                ' | bridge_topic=/map_bridge',
                ' | map_type=nav_msgs/msg/OccupancyGrid',
                ' | pose_type=geometry_msgs/msg/PoseStamped',
            ]),
            TimerAction(period=bridge_t, actions=bridges),
            TimerAction(period=relay_t, actions=[map_relay, member_pose]),
        ]
        if amcl is not None:
            actions.append(TimerAction(
                period=amcl_t, actions=[localization_config_log, amcl],
            ))
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
            default_value=EnvironmentVariable('ROS_DOMAIN_ID'),
            description='Member DDS domain.',
        ),
        DeclareLaunchArgument(
            'main_domain_id',
            default_value='',
            description=(
                'Leader DDS domain used by domain_bridge. Required; pass '
                'main_domain_id:=<leader_domain>.'
            ),
        ),
        DeclareLaunchArgument(
            'start_robot_bringup',
            default_value='true',
            choices=['true', 'false'],
            description='Start TurtleBot3 hardware drivers.',
        ),
        DeclareLaunchArgument(
            'start_nav2',
            default_value='true',
            choices=['true', 'false'],
            description=(
                'Start this member robot Nav2 stack. Set false for a '
                'mapping-only scout where RL/manual control owns cmd_vel.'
            ),
        ),
        DeclareLaunchArgument(
            'hardware_param_file', default_value='',
            description=(
                'Optional override for turtlebot3_bringup\'s own hardware '
                'parameter YAML, passed through to base.launch.py. Needed '
                'when this member owns its own SLAM (enable_amcl:=false + '
                'a Cartographer elsewhere) so the wheel odometry\'s own '
                'odom->base_footprint TF broadcast can be disabled -- '
                'otherwise it conflicts with Cartographer\'s own TF.'
            ),
        ),
        DeclareLaunchArgument(
            'forward_map_to_main',
            default_value='false',
            choices=['true', 'false'],
            description=(
                'Bridge this robot-owned /map back to the leader domain as '
                '/map_bridge. Enable only when this member/scout owns SLAM.'
            ),
        ),
        DeclareLaunchArgument('member_initial_x', default_value='0.0'),
        DeclareLaunchArgument('member_initial_y', default_value='0.0'),
        DeclareLaunchArgument('member_initial_yaw', default_value='0.0'),
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
                'reinitialize_global_localization after seeding from '
                'member_initial_x/y/yaw. Set false to use only the fixed '
                'seed.'
            ),
        ),
        *dds_launch_environment(domain_id),
        OpaqueFunction(function=make_stack),
    ])
