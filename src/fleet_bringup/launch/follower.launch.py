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
    LogInfo,
    OpaqueFunction,
    TimerAction,
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import EnvironmentVariable, LaunchConfiguration
from launch_ros.actions import Node
from nav2_common.launch import RewrittenYaml

from fleet_bringup.domain_bridge_config import write_fleet_bridge_configs
from fleet_bringup.launch_utils import (
    clean_process_environment,
    dds_launch_environment,
    launch_bool,
)


def generate_launch_description():
    package_share = get_package_share_directory('fleet_bringup')
    robot_launch = os.path.join(
        get_package_share_directory('turtlebot3_bringup'),
        'launch',
        'robot.launch.py',
    )

    use_sim_time = LaunchConfiguration('use_sim_time')
    domain_id = LaunchConfiguration('domain_id')
    main_domain_id = LaunchConfiguration('main_domain_id')
    start_robot_bringup = LaunchConfiguration('start_robot_bringup')
    hardware_param_file = LaunchConfiguration('hardware_param_file')
    forward_map_to_main = LaunchConfiguration('forward_map_to_main')
    include_follower_scan = LaunchConfiguration('include_follower_scan')
    follow_distance = LaunchConfiguration('follow_distance')
    start_legacy_follower = LaunchConfiguration('start_legacy_follower')
    initial_x = LaunchConfiguration('follower_initial_x')
    initial_y = LaunchConfiguration('follower_initial_y')
    initial_yaw = LaunchConfiguration('follower_initial_yaw')
    auto_localize = LaunchConfiguration('auto_localize')
    enable_amcl = LaunchConfiguration('enable_amcl')

    def make_stack(context):
        simulation = launch_bool(use_sim_time.perform(context))
        follower_domain = int(domain_id.perform(context))
        main_domain_value = main_domain_id.perform(context).strip()
        if not main_domain_value:
            raise ValueError(
                'main_domain_id is required for follower.launch.py domain_bridge. '
                'Pass the launch option main_domain_id:=<leader_domain>.'
            )
        main_domain = int(main_domain_value)
        process_env = clean_process_environment(str(follower_domain))
        amcl_enabled = launch_bool(enable_amcl.perform(context))

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
            forward_map_to_main=launch_bool(forward_map_to_main.perform(context)),
            include_follower_scan=launch_bool(include_follower_scan.perform(context)),
            include_leader_map=amcl_enabled,
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
            package='fleet_bringup',
            executable='map_relay',
            name='follower_map_relay',
            output='screen',
            parameters=[{'use_sim_time': simulation}],
            env=process_env,
            respawn=True,
            respawn_delay=3.0,
        )
        follower_pose = Node(
            package='fleet_bringup',
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

        relay_nodes = [map_relay] if amcl_enabled else []
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
                    package='fleet_bringup',
                    executable='sim_burger_tf_relay',
                    name='follower_tf_relay',
                    output='screen',
                    parameters=[{'use_sim_time': True}],
                    env=process_env,
                ),
                Node(
                    package='fleet_bringup',
                    executable='sim_burger_scan_relay',
                    name='follower_scan_relay',
                    output='screen',
                    parameters=[{'use_sim_time': True}],
                    env=process_env,
                ),
            ])
        else:
            relay_nodes.append(Node(
                package='fleet_bringup',
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
        require_ready_for_follow = bool(amcl_enabled and not simulation)

        amcl = None
        localization_lifecycle = None
        fixed_seed_ready = None
        if amcl_enabled:
            # AMCL is the one TF source this stack owns by default (map->
            # odom over a shared, bridged map). Set enable_amcl:=false only
            # when something else (e.g. a risk-map Cartographer) will own
            # SLAM/TF for this robot instead -- running both fights over
            # the same transform.
            pose_override = Path(tempfile.gettempdir()) / (
                f'follower_{follower_domain}_initial_pose.yaml'
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
                'AMCL_INITIAL_POSE_CONFIG | robot=burger',
                ' | role=follower',
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
                package='fleet_bringup',
                executable='global_localize_kickstart',
                name='burger_global_localize',
                output='screen',
                parameters=[{
                    'use_sim_time': simulation,
                    'enable_scout_pose_seed': False,
                    'active_scout_id_topic': '/failover/active_scout_id',
                    'active_scout_robot_name': 'scout22',
                    'follower_robot_name': 'follower21',
                    'member_pose_topic': '/member_pose',
                    'burger_pose_topic': '/burger_pose',
                    'last_scout_pose_topic': '/failover/last_scout_pose',
                    'scout_pose_max_age_sec': 8.0,
                    'scout_pose_wait_timeout_sec': 2.0,
                    'initial_pose_topic': '/initialpose',
                    'initial_pose_xy_std_m': 1.0,
                    'initial_pose_yaw_std_deg': 45.0,
                    'initial_pose_settle_sec': 0.5,
                    'allow_blind_global_reinit': False,
                    'spin_enabled': not simulation,
                    'spin_speed_rad_s': 0.40,
                    'spin_target_angle_rad': 7.10,
                    'spin_timeout_sec': 35.0,
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
                    'localization_cov_xy_threshold': 1.0,
                    'localization_cov_yaw_threshold': 0.8,
                    'localization_stable_duration_sec': 2.5,
                    'localization_check_timeout_sec': 9.0,
                    'max_spin_retries': 0,
                    'force_spin_after_sec': 14.0,
                }],
                env=process_env,
                respawn=True,
                respawn_delay=3.0,
            )
        elif amcl_enabled and not simulation:
            fixed_seed_ready = Node(
                package='fleet_bringup',
                executable='amcl_fixed_seed_ready',
                name='follower_amcl_fixed_seed_ready',
                output='screen',
                parameters=[{
                    'map_topic': '/map',
                    'scan_topic': '/scan',
                    'odom_topic': '/odom',
                    'amcl_pose_topic': '/amcl_pose',
                    'amcl_get_state_service': '/amcl/get_state',
                    'global_frame': 'map',
                    'base_frame': 'base_footprint',
                    'ready_topic': '/localization_ready',
                    'fixed_seed_initial_pose_applied': True,
                }],
                env=process_env,
                respawn=True,
                respawn_delay=3.0,
            )

        goal_proxy = Node(
            package='fleet_bringup',
            executable='pose_to_nav2',
            name='burger_named_goal',
            output='screen',
            parameters=[{
                'use_sim_time': simulation,
                'goal_pose_topic': '/burger_goal_pose',
                'require_system_ready': False,
                'system_ready_topic': '/system/ready',
                'require_start_motion': True,
                'start_motion_topic': '/fleet/start_motion',
            }],
            env=process_env,
        )
        follower = Node(
            package='fleet_bringup',
            executable='fleet_follower',
            name='fleet_follower',
            output='screen',
            parameters=[{
                'use_sim_time': simulation,
                'follow_distance': float(follow_distance.perform(context)),
                'start_following': True,
                'require_localization_ready': require_ready_for_follow,
            }],
            env=process_env,
            respawn=True,
            respawn_delay=3.0,
        )

        actions = []
        if robot_state_publisher is not None:
            actions.append(robot_state_publisher)
        elif launch_bool(start_robot_bringup.perform(context)):
            robot_launch_args = {'use_sim_time': 'false', 'namespace': ''}
            param_file = hardware_param_file.perform(context)
            if not param_file:
                param_file = os.path.join(
                    package_share, 'config', 'turtlebot3_burger_stamped_cmd_vel.yaml'
                )
            if param_file:
                robot_launch_args['tb3_param_dir'] = param_file
            actions.append(IncludeLaunchDescription(
                PythonLaunchDescriptionSource(robot_launch),
                launch_arguments=robot_launch_args.items(),
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
        behavior_nodes = [goal_proxy]
        if launch_bool(start_legacy_follower.perform(context)):
            behavior_nodes.append(follower)

        actions.extend([
            LogInfo(msg=[
                'LEADER_EGRESS_BRIDGE | source_domain=', str(main_domain),
                ' | destination_domain=', str(follower_domain),
                ' | topics=/map,/leader_pose',
                ' | bridge_topic=/map_bridge',
                ' | map_type=nav_msgs/msg/OccupancyGrid',
                ' | pose_type=geometry_msgs/msg/PoseStamped',
            ]),
            TimerAction(period=bridge_t, actions=bridges),
            TimerAction(period=relay_t, actions=relay_nodes + [follower_pose]),
            TimerAction(period=nav_t, actions=navigation),
            TimerAction(period=lifecycle_t, actions=[navigation_lifecycle]),
            TimerAction(period=behavior_t, actions=behavior_nodes),
        ])
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
        if fixed_seed_ready is not None:
            actions.append(TimerAction(
                period=kickstart_t,
                actions=[
                    LogInfo(msg=[
                        'FOLLOWER_STAGE | starting fixed-seed AMCL ready watcher',
                    ]),
                    fixed_seed_ready,
                ],
            ))
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
            default_value=EnvironmentVariable('ROS_DOMAIN_ID'),
            description='Follower DDS domain.',
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
            description='Start TurtleBot3 hardware drivers in real mode.',
        ),
        DeclareLaunchArgument(
            'hardware_param_file', default_value='',
            description=(
                'Optional override for turtlebot3_bringup\'s own hardware '
                'parameter YAML (real mode only). Needed when this '
                'follower owns its own SLAM (enable_amcl:=false + a '
                'Cartographer elsewhere) so the wheel odometry\'s own '
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
                '/map_bridge. Enable only when this follower/scout owns SLAM.'
            ),
        ),
        DeclareLaunchArgument(
            'include_follower_scan',
            default_value='false',
            choices=['true', 'false'],
            description=(
                'Bridge /burger_scan_relay to the leader domain. Default '
                'false because leader control uses compact pose/status/ack '
                'topics, not follower raw LaserScan.'
            ),
        ),
        DeclareLaunchArgument(
            'follow_distance',
            default_value='0.70',
            description='Desired distance behind the leader in metres.',
        ),
        DeclareLaunchArgument(
            'start_legacy_follower',
            default_value='false',
            choices=['true', 'false'],
            description=(
                'Start fleet_follower. Set false when system_bringup '
                'unified_field_robot owns FOLLOWER mode.'
            ),
        ),
        DeclareLaunchArgument('follower_initial_x', default_value='0.0'),
        DeclareLaunchArgument('follower_initial_y', default_value='-0.10'),
        DeclareLaunchArgument('follower_initial_yaw', default_value='0.0'),
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
            default_value='false',
            choices=['true', 'false'],
            description=(
                'Let AMCL search the whole map via '
                'reinitialize_global_localization after seeding from '
                'follower_initial_x/y/yaw. Default false uses only the fixed '
                'seed and amcl_fixed_seed_ready.'
            ),
        ),
        *dds_launch_environment(domain_id),
        OpaqueFunction(function=make_stack),
    ])
