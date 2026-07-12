#!/usr/bin/env python3
"""Leader stack: TurtleBot3 bringup, AMCL/Nav2 and fleet coordination.

In real mode the leader defaults to receiving the scout/risk domain's SLAM
map through domain_bridge on /map_bridge, republishing it as this domain's
/map, and running AMCL against that shared map. enable_cartographer:=true is
kept as an explicit compatibility escape hatch for single-leader SLAM mode.
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

from fleet_bringup.launch_utils import (
    clean_process_environment,
    dds_launch_environment,
    launch_bool,
)


def _tracked_cmd_vel_adapter_enabled(param_file: str) -> bool:
    if not param_file:
        return False
    try:
        with open(param_file, 'r', encoding='utf-8') as handle:
            data = yaml.safe_load(handle) or {}
    except OSError:
        return False
    params = data.get('tracked_cmd_vel_adapter', {}).get('ros__parameters', {})
    return bool(params.get('enabled', False))


def generate_launch_description():
    package_share = get_package_share_directory('fleet_bringup')
    base_launch = os.path.join(package_share, 'launch', 'base.launch.py')

    use_sim_time = LaunchConfiguration('use_sim_time')
    domain_id = LaunchConfiguration('domain_id')
    start_robot_bringup = LaunchConfiguration('start_robot_bringup')
    start_nav2 = LaunchConfiguration('start_nav2')
    require_follower_pose = LaunchConfiguration('require_follower_pose')
    enable_cartographer = LaunchConfiguration('enable_cartographer')
    auto_localize = LaunchConfiguration('auto_localize')
    localization_scan_topic = LaunchConfiguration('localization_scan_topic')
    hardware_param_file = LaunchConfiguration('hardware_param_file')
    initial_x = LaunchConfiguration('leader_initial_x')
    initial_y = LaunchConfiguration('leader_initial_y')
    initial_yaw = LaunchConfiguration('leader_initial_yaw')

    def make_stack(context):
        simulation = launch_bool(use_sim_time.perform(context))
        domain = domain_id.perform(context)
        process_env = clean_process_environment(domain)
        # Simulation has no bridged-map infrastructure set up, so it always
        # owns Cartographer regardless of enable_cartographer.
        cartographer_owned = simulation or launch_bool(
            enable_cartographer.perform(context)
        )
        scan_topic_value = (
            '/scan'
            if simulation
            else localization_scan_topic.perform(context).strip() or '/scan'
        )
        auto_localize_enabled = (
            not cartographer_owned
            and launch_bool(auto_localize.perform(context))
        )
        leader_hardware_param_file = hardware_param_file.perform(context).strip()
        if not leader_hardware_param_file:
            leader_hardware_param_file = os.path.join(
                package_share,
                'config',
                'tracked_waffle_kinematics.yaml',
            )
        tracked_adapter_enabled = _tracked_cmd_vel_adapter_enabled(
            leader_hardware_param_file
        )
        leader_cmd_vel_topic = (
            '/cmd_vel_nav' if tracked_adapter_enabled else '/cmd_vel'
        )

        nav2_params = RewrittenYaml(
            source_file=os.path.join(
                package_share,
                'config',
                (
                    'leader_nav2.yaml'
                    if cartographer_owned
                    else 'leader_waffle_pi_nav2.yaml'
                ),
            ),
            param_rewrites={
                'use_sim_time': str(simulation).lower(),
                'odom_topic': '/odom',
                'scan_topic': scan_topic_value,
                'map_topic': '/map',
                'topic': scan_topic_value,
                'enable_stamped_cmd_vel': 'true',
            },
            convert_types=True,
        )

        cartographer = None
        amcl = None
        localization_lifecycle = None
        map_relay = None
        kickstart_node = None
        if cartographer_owned:
            cartographer_launch = os.path.join(
                get_package_share_directory('turtlebot3_cartographer'),
                'launch',
                'cartographer.launch.py',
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
        else:
            # Receive the map from the risk/scout SLAM domain as
            # /map_bridge, then republish it as this domain's /map for
            # AMCL/Nav2 and downstream fan-out bridges.
            map_relay = Node(
                package='fleet_bringup',
                executable='map_relay',
                name='leader_map_relay',
                output='screen',
                parameters=[{
                    'use_sim_time': False,
                    'input_topic': '/map_bridge',
                    'output_topic': '/map',
                    'check_period_sec': 0.2,
                    'takeover_grace_sec': 0.0,
                    'relay_without_primary': True,
                }],
                env=process_env,
                respawn=True,
                respawn_delay=3.0,
            )

            auto = auto_localize_enabled
            initial_pose = {
                'x': float(initial_x.perform(context)),
                'y': float(initial_y.perform(context)),
                'z': 0.0,
                'yaw': float(initial_yaw.perform(context)),
            }
            pose_override = Path(tempfile.gettempdir()) / (
                f'leader_{domain}_initial_pose.yaml'
            )
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
            localization_config_log = LogInfo(msg=[
                'LEADER_LOCALIZATION_CONFIG | global_frame=map',
                ' | odom_frame=odom',
                ' | base_frame=base_footprint',
                ' | map_topic=/map',
                ' | scan_topic=', scan_topic_value,
                ' | tf_broadcast=true',
                ' | initial_pose_mode=',
                'seeded_global_localize' if auto else 'fixed_seed',
                ' | x=', str(initial_pose['x']),
                ' | y=', str(initial_pose['y']),
                ' | yaw=', str(initial_pose['yaw']),
            ])
            localization_lifecycle = Node(
                package='nav2_lifecycle_manager',
                executable='lifecycle_manager',
                name='lifecycle_manager_localization',
                output='screen',
                parameters=[nav2_params],
                env=process_env,
                respawn=True,
                respawn_delay=3.0,
            )
            if auto:
                kickstart_node = Node(
                    package='fleet_bringup',
                    executable='global_localize_kickstart',
                    name='leader_global_localize',
                    output='screen',
                    parameters=[{
                        'scan_topic': scan_topic_value,
                        # Seed AMCL's initial pose from the active scout's
                        # known map-frame pose before spinning, instead of
                        # blindly scattering particles across a still-sparse
                        # shared map -- falls back to blind reinit if no
                        # scout pose is available in time.
                        'enable_scout_pose_seed': True,
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
                        'spin_enabled': True,
                        # 좌우 바퀴 비대칭이 심할수록(고속 회전일수록) 더
                        # 벌어져서 "제자리" spin 이 실제로는 호를 그리며
                        # 이동한다 -- 낮은 회전 속도로 그 영향을 줄임.
                        'spin_speed_rad_s': 0.25,
                        'spin_target_angle_rad': 7.10,
                        'spin_timeout_sec': 35.0,
                        'spin_sensor_dropout_grace_sec': 1.5,
                        'settle_duration_sec': 3.0,
                        'spin_max_drift_m': 0.35,
                        'require_valid_map': True,
                        'min_known_map_cells': 100,
                        'require_scan_before_spin': True,
                        'require_odom_before_spin': True,
                        'require_amcl_before_spin': True,
                        'max_scan_age_sec': 1.2,
                        'max_odom_age_sec': 1.2,
                        'cmd_vel_topic': leader_cmd_vel_topic,
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

        leader_pose = Node(
            package='fleet_bringup',
            executable='tf_pose_publisher',
            name='leader_pose_pub',
            output='screen',
            parameters=[{
                'use_sim_time': simulation,
                'output_topic': '/leader_pose',
                'target_frame': 'map',
                'source_frame': 'base_footprint',
                'source_frame_candidates': ['base_footprint', 'base_link'],
                'publish_rate_hz': 10.0,
                'freeze_when_stationary': True,
                'stationary_target_frame': 'odom',
                'stationary_linear_threshold_m': 0.02,
                'stationary_angular_threshold_rad': 0.035,
                'stationary_freeze_warmup_sec': 12.0,
                'log_every_n': 100,
            }],
            env=process_env,
            respawn=True,
            respawn_delay=3.0,
        )
        follower_tf = Node(
            package='fleet_bringup',
            executable='pose_to_tf',
            name='burger_tf_on_leader',
            output='screen',
            parameters=[{'use_sim_time': simulation}],
            env=process_env,
            respawn=True,
            respawn_delay=3.0,
        )
        leader_scan = Node(
            package='fleet_bringup',
            executable='scan_frame_relay',
            name='leader_fleet_scan_relay',
            output='screen',
            parameters=[{
                'input_topic': scan_topic_value,
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
                    package='fleet_bringup',
                    executable='pose_to_nav2',
                    name='leader_goal_arbiter_output',
                    output='screen',
                    parameters=[{
                        'use_sim_time': simulation,
                        'goal_pose_topic': '/fleet/leader_coord_goal',
                        'cancel_topic': '/fleet/leader_nav_cancel',
                    }],
                    env=process_env,
                ),
            ]
        else:
            if cartographer_owned:
                nav_delay_sec = '12.0'
                lifecycle_delay_sec = '16.0'
                goal_delay_sec = '18.0'
            else:
                # External-map mode waits for /map to cross the risk->leader
                # bridge, then gives AMCL a clear head start before Nav2
                # starts asking for map->odom transforms.
                nav_delay_sec = '6.0'
                lifecycle_delay_sec = '7.0'
                goal_delay_sec = '8.0'

            base_include = IncludeLaunchDescription(
                PythonLaunchDescriptionSource(base_launch),
                launch_arguments={
                    'domain_id': domain,
                    'start_robot_bringup': start_robot_bringup.perform(context),
                    'hardware_param_file': leader_hardware_param_file,
                    'start_nav2': start_nav2.perform(context),
                    'nav2_params_file': nav2_params,
                    'goal_pose_topic': '/fleet/leader_coord_goal',
                    'cancel_topic': '/fleet/leader_nav_cancel',
                    'goal_proxy_name': 'leader_goal_arbiter_output',
                    'nav_delay_sec': nav_delay_sec,
                    'lifecycle_delay_sec': lifecycle_delay_sec,
                    'goal_delay_sec': goal_delay_sec,
                    'require_localization_ready': (
                        'true' if auto_localize_enabled else 'false'
                    ),
                    'localization_ready_topic': '/localization_ready',
                }.items(),
            )

        coordinator = Node(
            package='fleet_bringup',
            executable='fleet_path_coordinator',
            name='fleet_path_coordinator',
            output='screen',
            parameters=[{
                'use_sim_time': simulation,
                'require_follower_pose': launch_bool(
                    require_follower_pose.perform(context)
                ),
                # External-map AMCL must physically spin first. Holding the
                # coordinator prevents it from publishing leader/member goals
                # while global_localize_kickstart owns /cmd_vel.
                'require_localization_ready': auto_localize_enabled,
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
                period=0.2,
                actions=[
                    LogInfo(msg=[
                        'LEADER_STAGE | starting follower scan static TF',
                    ]),
                    Node(
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
                    ),
                ],
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
            # base.launch.py, measured from this same t=0. External-map
            # leader mode deliberately starts Nav2 later than normal SLAM
            # leader mode so AMCL is active before navigation needs TF.
            pose_t, coordinator_t = 1.0, 6.0
            actions.extend([
                TimerAction(
                    period=pose_t,
                    actions=[
                        LogInfo(msg=[
                            'LEADER_STAGE | starting leader/follower pose ',
                            'and scan relay nodes',
                        ]),
                        leader_pose,
                        follower_tf,
                        leader_scan,
                    ],
                ),
                TimerAction(
                    period=coordinator_t,
                    actions=[
                        LogInfo(msg=[
                            'LEADER_STAGE | starting fleet coordinator',
                        ]),
                        coordinator,
                    ],
                ),
            ])
            if cartographer_owned:
                actions.append(
                    TimerAction(
                        period=5.0,
                        actions=[
                            LogInfo(msg=[
                                'LEADER_STAGE | starting Cartographer',
                            ]),
                            cartographer,
                        ],
                    )
                )
                # global_localize_kickstart never runs in this branch (its
                # whole state machine is AMCL-specific), so nothing else
                # would ever publish ready_topic here -- a downstream
                # bootstrap gate (e.g. scout_failover_coordinator) would
                # wait forever. Watch Cartographer's own map/TF/scan
                # instead.
                actions.append(
                    TimerAction(
                        period=6.0,
                        actions=[
                            LogInfo(msg=[
                                'LEADER_STAGE | starting SLAM localization '
                                'ready watcher',
                            ]),
                            Node(
                                package='fleet_bringup',
                                executable='slam_localization_ready',
                                name='leader_slam_localization_ready',
                                output='screen',
                                parameters=[{
                                    'map_topic': '/map',
                                    'scan_topic': scan_topic_value,
                                    'global_frame': 'map',
                                    'base_frame': 'base_footprint',
                                    'ready_topic': 'localization_ready',
                                }],
                                env=process_env,
                                respawn=True,
                                respawn_delay=3.0,
                            ),
                        ],
                    )
                )
            else:
                actions.append(
                    TimerAction(
                        period=0.2,
                        actions=[
                            LogInfo(msg=[
                                'LEADER_STAGE | starting bridged-map relay',
                            ]),
                            map_relay,
                        ],
                    )
                )
                actions.append(
                    TimerAction(
                        period=2.0,
                        actions=[
                            localization_config_log,
                            LogInfo(msg=[
                                'LEADER_STAGE | starting AMCL',
                            ]),
                            amcl,
                        ],
                    )
                )
                actions.append(TimerAction(
                    period=3.0,
                    actions=[
                        LogInfo(msg=[
                            'LEADER_STAGE | starting AMCL lifecycle manager',
                        ]),
                        localization_lifecycle,
                    ],
                ))
                if kickstart_node is not None:
                    actions.append(
                        TimerAction(
                            period=4.0,
                            actions=[
                                LogInfo(msg=[
                                    'LEADER_STAGE | starting AMCL global ',
                                    'localize kickstart',
                                ]),
                                kickstart_node,
                            ],
                        )
                    )
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
            default_value=EnvironmentVariable('ROS_DOMAIN_ID'),
            description='Leader DDS domain.',
        ),
        DeclareLaunchArgument(
            'start_robot_bringup',
            default_value='true',
            choices=['true', 'false'],
            description='Start TurtleBot3 hardware drivers in real mode.',
        ),
        DeclareLaunchArgument(
            'start_nav2',
            default_value='true',
            choices=['true', 'false'],
            description=(
                'Start the leader Nav2 navigation core and goal proxy in '
                'real mode. Hardware bringup is controlled separately.'
            ),
        ),
        DeclareLaunchArgument(
            'require_follower_pose',
            default_value='true',
            choices=['true', 'false'],
            description=(
                'Whether a follower.launch.py robot is expected in this '
                'fleet. When true (default) the coordinator monitors both '
                '/leader_pose and /burger_pose for safety telemetry. Set '
                'false for a leader-only or leader+member fleet with no '
                'follower robot to suppress follower-pose wait warnings. '
                'Direct Nav2 goal passthrough never publishes a hold goal.'
            ),
        ),
        DeclareLaunchArgument(
            'enable_cartographer',
            default_value='false',
            choices=['true', 'false'],
            description=(
                'Real mode only (ignored in simulation): default false '
                'receives the scout/risk SLAM map on /map_bridge and runs '
                'AMCL against the shared /map. Set true only for '
                'single-leader SLAM compatibility.'
            ),
        ),
        DeclareLaunchArgument(
            'auto_localize',
            default_value='false',
            choices=['true', 'false'],
            description=(
                'Only used when enable_cartographer:=false. Let AMCL '
                'start from a one-shot scout-pose /initialpose seed when '
                'available, then refine via verified in-place spin. Default '
                'false keeps a stationary leader on its fixed initial pose.'
            ),
        ),
        DeclareLaunchArgument(
            'localization_scan_topic',
            default_value='/scan',
            description=(
                'Real external-map leader mode: LaserScan topic consumed by '
                'AMCL/Nav2. Defaults to raw /scan so localization does not '
                'deadlock when optional OMX scan_processor (/scan_filtered) '
                'is not running. Pass /scan_filtered only after verifying it '
                'is publishing.'
            ),
        ),
        DeclareLaunchArgument(
            'hardware_param_file',
            default_value='',
            description=(
                'Optional TurtleBot3 hardware params. Empty uses the '
                'tracked Waffle kinematics profile.'
            ),
        ),
        DeclareLaunchArgument('leader_initial_x', default_value='0.0'),
        DeclareLaunchArgument('leader_initial_y', default_value='0.0'),
        DeclareLaunchArgument('leader_initial_yaw', default_value='0.0'),
        *dds_launch_environment(domain_id),
        OpaqueFunction(function=make_stack),
    ])
