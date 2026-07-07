#!/usr/bin/env python3
"""Leader stack: TurtleBot3 bringup, AMCL/Nav2 and fleet coordination.

In real mode the leader defaults to receiving the risk/scout domain's SLAM
map through domain_bridge on /map_bridge, republishing it as this domain's
/map, and running AMCL against that shared map. enable_cartographer:=true is
kept as an explicit compatibility escape hatch.
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
    start_nav2 = LaunchConfiguration('start_nav2')
    require_follower_pose = LaunchConfiguration('require_follower_pose')
    enable_cartographer = LaunchConfiguration('enable_cartographer')
    auto_localize = LaunchConfiguration('auto_localize')
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

        nav2_params = RewrittenYaml(
            source_file=os.path.join(
                package_share,
                'config',
                (
                    'leader_nav2.yaml'
                    if cartographer_owned
                    else 'follower_nav2_amcl.yaml'
                ),
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

        cartographer = None
        amcl = None
        localization_lifecycle = None
        map_relay = None
        kickstart_node = None
        if cartographer_owned:
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
                package='tb3_fleet_bringup',
                executable='map_relay',
                name='leader_map_relay',
                output='screen',
                parameters=[{
                    'use_sim_time': False,
                    'input_topic': '/map_bridge',
                    'output_topic': '/map',
                }],
                env=process_env,
                respawn=True,
                respawn_delay=3.0,
            )

            auto = launch_bool(auto_localize.perform(context))
            pose_override = Path(tempfile.gettempdir()) / (
                f'tb3_leader_{domain}_initial_pose.yaml'
            )
            if auto:
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
                respawn=True,
                respawn_delay=3.0,
            )
            if auto:
                kickstart_node = Node(
                    package='tb3_fleet_bringup',
                    executable='global_localize_kickstart',
                    name='leader_global_localize',
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
            if cartographer_owned:
                nav_delay_sec = '12.0'
                lifecycle_delay_sec = '16.0'
                goal_delay_sec = '18.0'
            else:
                # External-map mode waits for /map to cross the risk->leader
                # bridge, then gives AMCL a clear head start before Nav2
                # starts asking for map->odom transforms.
                nav_delay_sec = '18.0'
                lifecycle_delay_sec = '22.0'
                goal_delay_sec = '24.0'

            base_include = IncludeLaunchDescription(
                PythonLaunchDescriptionSource(base_launch),
                launch_arguments={
                    'domain_id': domain,
                    'start_robot_bringup': start_robot_bringup.perform(context),
                    'start_nav2': start_nav2.perform(context),
                    'nav2_params_file': nav2_params,
                    'goal_pose_topic': '/fleet/leader_coord_goal',
                    'goal_proxy_name': 'leader_goal_arbiter_output',
                    'nav_delay_sec': nav_delay_sec,
                    'lifecycle_delay_sec': lifecycle_delay_sec,
                    'goal_delay_sec': goal_delay_sec,
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
            # base.launch.py, measured from this same t=0. External-map
            # leader mode deliberately starts Nav2 later than normal SLAM
            # leader mode so AMCL is active before navigation needs TF.
            pose_t, coordinator_t = 8.0, 20.0
            actions.extend([
                TimerAction(
                    period=pose_t,
                    actions=[leader_pose, follower_tf, leader_scan],
                ),
                TimerAction(period=coordinator_t, actions=[coordinator]),
            ])
            if cartographer_owned:
                actions.append(
                    TimerAction(period=5.0, actions=[cartographer])
                )
            else:
                actions.append(
                    TimerAction(period=1.0, actions=[map_relay])
                )
                actions.append(TimerAction(period=8.0, actions=[amcl]))
                actions.append(TimerAction(
                    period=12.0, actions=[localization_lifecycle],
                ))
                if kickstart_node is not None:
                    actions.append(
                        TimerAction(period=15.0, actions=[kickstart_node])
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
                'fleet. When true (default) the coordinator holds the '
                "leader in place until BOTH /leader_pose and /burger_pose "
                'are fresh -- correct for a leader+follower fleet, but it '
                'freezes the leader forever if no follower ever '
                'publishes. Set false for a leader-only or '
                'leader+member fleet with no follower robot.'
            ),
        ),
        DeclareLaunchArgument(
            'enable_cartographer',
            default_value='false',
            choices=['true', 'false'],
            description=(
                'Real mode only (ignored in simulation): default false '
                'receives the risk/scout SLAM map on /map_bridge and runs '
                'AMCL against the shared /map. Set true only for legacy '
                'single-leader SLAM operation.'
            ),
        ),
        DeclareLaunchArgument(
            'auto_localize',
            default_value='true',
            choices=['true', 'false'],
            description=(
                'Only used when enable_cartographer:=false. Let AMCL '
                'search the whole received map via '
                'reinitialize_global_localization instead of trusting '
                'leader_initial_x/y/yaw.'
            ),
        ),
        DeclareLaunchArgument('leader_initial_x', default_value='0.0'),
        DeclareLaunchArgument('leader_initial_y', default_value='0.0'),
        DeclareLaunchArgument('leader_initial_yaw', default_value='0.0'),
        *dds_launch_environment(domain_id),
        OpaqueFunction(function=make_stack),
    ])
