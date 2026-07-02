#!/usr/bin/env python3

import os
import tempfile
from pathlib import Path

import yaml
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


def generate_launch_description():
    bringup_share = get_package_share_directory('tb3_fleet_bringup')

    domain_id           = LaunchConfiguration('domain_id')
    leader_domain_id    = LaunchConfiguration('leader_domain_id')
    robot_model         = LaunchConfiguration('robot_model')
    follow_distance     = LaunchConfiguration('follow_distance')
    start_following     = LaunchConfiguration('start_following')
    enable_path_yield   = LaunchConfiguration('enable_path_yield')
    path_block_distance = LaunchConfiguration('path_block_distance')
    yield_lateral_distance = LaunchConfiguration('yield_lateral_distance')
    follower_initial_x  = LaunchConfiguration('follower_initial_x')
    follower_initial_y  = LaunchConfiguration('follower_initial_y')
    follower_initial_yaw = LaunchConfiguration('follower_initial_yaw')
    use_slam            = LaunchConfiguration('use_slam')
    slam_domain         = LaunchConfiguration('slam_domain')

    # ── yaml selection: SLAM mode uses slam yaml; AMCL mode uses amcl yaml ──
    nav2_source = os.path.join(bringup_share, 'config', 'domain24_burger_nav2_amcl.yaml')
    nav2_slam_source = os.path.join(bringup_share, 'config', 'domain26_burger_nav2_slam.yaml')

    nav2_params = RewrittenYaml(
        source_file=nav2_source,
        param_rewrites={
            'use_sim_time': 'false',
            'odom_topic':   '/odom',
            'topic':        '/scan',
        },
        convert_types=True,
    )
    nav2_slam_params = RewrittenYaml(
        source_file=nav2_slam_source,
        param_rewrites={
            'use_sim_time': 'false',
            'odom_topic':   '/odom',
            'topic':        '/scan',
        },
        convert_types=True,
    )

    follower_script   = os.path.join(bringup_share, 'scripts', 'domain_bridge_nav2_follower_direct_v40.py')
    tf_pose_script    = os.path.join(bringup_share, 'scripts', 'tf_pose_publisher_direct_v44.py')
    goal_proxy_script = os.path.join(bringup_share, 'scripts', 'pose_to_nav2_action_direct_v41.py')
    cartographer_config_dir = os.path.join(bringup_share, 'config')

    # ── Domain bridge (map direction depends on which domain does SLAM) ────────
    def make_domain_bridges(context, *args, **kwargs):
        leader_domain = leader_domain_id.perform(context)
        burger_domain = domain_id.perform(context)
        sd = slam_domain.perform(context)   # '25' or '24'
        out_dir = Path(tempfile.gettempdir()) / 'tb3_fleet_real_domain_bridge'
        out_dir.mkdir(parents=True, exist_ok=True)

        l2b = out_dir / f'real_leader_{leader_domain}_to_burger_{burger_domain}.yaml'
        b2l = out_dir / f'real_burger_{burger_domain}_to_leader_{leader_domain}.yaml'

        # /map always flows from the SLAM domain to the non-SLAM domain.
        # When slam_domain=25 (leader SLAM): map goes 25→24.
        # When slam_domain=24 (follower SLAM): map goes 24→25.
        map_25_to_24 = (sd == '25')

        l2b_map_block = (
            """  /map:
    type: nav_msgs/msg/OccupancyGrid
    qos:
      reliability: reliable
      durability: transient_local
      history: keep_last
      depth: 1
  /map_metadata:
    type: nav_msgs/msg/MapMetaData
    qos:
      reliability: reliable
      durability: transient_local
      history: keep_last
      depth: 1
"""
            if map_25_to_24 else ""
        )
        b2l_map_block = (
            """  /map:
    type: nav_msgs/msg/OccupancyGrid
    qos:
      reliability: reliable
      durability: transient_local
      history: keep_last
      depth: 1
  /map_metadata:
    type: nav_msgs/msg/MapMetaData
    qos:
      reliability: reliable
      durability: transient_local
      history: keep_last
      depth: 1
"""
            if not map_25_to_24 else ""
        )

        l2b.write_text(
            f"""name: real_leader_{leader_domain}_to_burger_{burger_domain}
from_domain: {leader_domain}
to_domain: {burger_domain}

topics:
{l2b_map_block}  /leader_pose:
    type: geometry_msgs/msg/PoseStamped
    qos:
      reliability: reliable
      durability: volatile
      history: keep_last
      depth: 10
  /plan:
    type: nav_msgs/msg/Path
    remap: /waffle_plan
    qos:
      reliability: reliable
      durability: volatile
      history: keep_last
      depth: 10
  /burger_goal_pose:
    type: geometry_msgs/msg/PoseStamped
    qos:
      reliability: reliable
      durability: volatile
      history: keep_last
      depth: 10
  /fleet/follow_command:
    type: std_msgs/msg/String
    qos:
      reliability: reliable
      durability: volatile
      history: keep_last
      depth: 10
""",
            encoding='utf-8',
        )

        b2l.write_text(
            f"""name: real_burger_{burger_domain}_to_leader_{leader_domain}
from_domain: {burger_domain}
to_domain: {leader_domain}

topics:
{b2l_map_block}  /burger_pose:
    type: geometry_msgs/msg/PoseStamped
    qos:
      reliability: reliable
      durability: volatile
      history: keep_last
      depth: 10
  /scan:
    type: sensor_msgs/msg/LaserScan
    remap: /burger_scan
    qos:
      reliability: best_effort
      durability: volatile
      history: keep_last
      depth: 10
  /plan:
    type: nav_msgs/msg/Path
    remap: /burger_plan
    qos:
      reliability: reliable
      durability: volatile
      history: keep_last
      depth: 10
  /fleet/follow_enabled:
    type: std_msgs/msg/Bool
    qos:
      reliability: reliable
      durability: transient_local
      history: keep_last
      depth: 1
""",
            encoding='utf-8',
        )

        return [
            ExecuteProcess(
                cmd=['ros2', 'run', 'domain_bridge', 'domain_bridge', str(l2b)],
                output='screen',
                name='real_leader_to_burger_domain_bridge',
            ),
            ExecuteProcess(
                cmd=['ros2', 'run', 'domain_bridge', 'domain_bridge', str(b2l)],
                output='screen',
                name='real_burger_to_leader_domain_bridge',
            ),
        ]

    # ── Localization: AMCL (default) or Cartographer SLAM ─────────────────────
    def make_localization(context, *args, **kwargs):
        slam_mode = use_slam.perform(context).lower() in ('true', '1', 'yes')

        if slam_mode:
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
            return [TimerAction(period=0.5, actions=[cartographer, occupancy_grid])]

        # ── AMCL (receives map from SLAM domain via bridge) ───────────────────
        ix   = float(follower_initial_x.perform(context))
        iy   = float(follower_initial_y.perform(context))
        iyaw = float(follower_initial_yaw.perform(context))

        amcl_pose_overrides = {
            'amcl': {
                'ros__parameters': {
                    'set_initial_pose': True,
                    'initial_pose': {'x': ix, 'y': iy, 'z': 0.0, 'yaw': iyaw},
                }
            }
        }
        pose_yaml = Path(tempfile.gettempdir()) / 'burger_amcl_initial_pose.yaml'
        pose_yaml.write_text(yaml.dump(amcl_pose_overrides), encoding='utf-8')

        amcl = Node(
            package='nav2_amcl',
            executable='amcl',
            name='amcl',
            output='screen',
            parameters=[nav2_params, str(pose_yaml)],
        )
        lifecycle_loc = Node(
            package='nav2_lifecycle_manager',
            executable='lifecycle_manager',
            name='lifecycle_manager_localization',
            output='screen',
            parameters=[nav2_params],
        )
        return [
            TimerAction(period=2.0, actions=[amcl]),
            TimerAction(period=2.5, actions=[lifecycle_loc]),
        ]

    # ── Burger pose publisher (TF: map → base_footprint, works for both modes) ─
    burger_pose = ExecuteProcess(
        cmd=[
            'python3', tf_pose_script, '--ros-args',
            '-r', '__node:=burger_real_pose_publisher',
            '-p', 'use_sim_time:=false',
            '-p', 'target_frame:=map',
            '-p', 'source_frame:=base_footprint',
            '-p', 'output_topic:=/burger_pose',
            '-p', 'publish_rate_hz:=10.0',
            '-p', 'log_every_n:=100',
        ],
        output='screen',
        name='burger_real_pose_publisher',
    )

    # ── Nav2 nodes ─────────────────────────────────────────────────────────────
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
    lifecycle_navigation = Node(
        package='nav2_lifecycle_manager',
        executable='lifecycle_manager',
        name='lifecycle_manager_navigation',
        output='screen',
        parameters=[nav2_params],
    )
    burger_named_goal = ExecuteProcess(
        cmd=[
            'python3', goal_proxy_script, '--ros-args',
            '-r', '__node:=burger_real_named_goal_to_nav2',
            '-p', 'use_sim_time:=false',
            '-p', 'goal_pose_topic:=/burger_goal_pose',
            '-p', 'navigate_action:=/navigate_to_pose',
            '-p', 'default_frame_id:=map',
            '-p', 'cancel_previous_goal:=true',
        ],
        output='screen',
        name='burger_real_named_goal_to_nav2',
    )
    follower = ExecuteProcess(
        cmd=[
            'python3', follower_script, '--ros-args',
            '-r', '__node:=domain_bridge_nav2_follower',
            '-p', 'use_sim_time:=false',
            '-p', 'leader_pose_topic:=/leader_pose',
            '-p', 'leader_path_topic:=/waffle_plan',
            '-p', 'follower_pose_topic:=/burger_pose',
            '-p', 'map_topic:=/map',
            '-p', 'navigate_action:=/navigate_to_pose',
            '-p', ['follow_distance:=', follow_distance],
            '-p', 'goal_period_sec:=1.0',
            '-p', 'goal_update_distance:=0.20',
            '-p', 'cancel_previous_goal:=false',
            '-p', 'follow_command_topic:=/fleet/follow_command',
            '-p', 'follow_status_topic:=/fleet/follow_enabled',
            '-p', ['start_following:=', start_following],
            '-p', ['enable_path_yield:=', enable_path_yield],
            '-p', ['path_block_distance:=', path_block_distance],
            '-p', 'path_lookahead_min:=0.30',
            '-p', 'path_lookahead_max:=2.50',
            '-p', ['yield_lateral_distance:=', yield_lateral_distance],
            '-p', 'yield_release_distance:=0.80',
            '-p', 'yield_map_clearance:=0.18',
            '-p', 'yield_min_hold_sec:=4.0',
            '-p', 'yield_max_hold_sec:=12.0',
        ],
        output='screen',
        name='burger_real_nav2_follower',
    )

    return LaunchDescription([
        DeclareLaunchArgument('domain_id',        default_value='24'),
        DeclareLaunchArgument('leader_domain_id', default_value='25'),
        DeclareLaunchArgument('robot_model',      default_value='burger'),
        DeclareLaunchArgument('follow_distance',  default_value='1.05'),
        DeclareLaunchArgument(
            'start_following', default_value='false',
            description='true=start in FOLLOWING; false=wait for FOLLOW command.',
        ),
        DeclareLaunchArgument('enable_path_yield',      default_value='true'),
        DeclareLaunchArgument('path_block_distance',    default_value='0.55'),
        DeclareLaunchArgument('yield_lateral_distance', default_value='0.75'),
        DeclareLaunchArgument(
            'follower_initial_x', default_value='-1.05',
            description='[AMCL] Follower start x in SLAM map frame.',
        ),
        DeclareLaunchArgument('follower_initial_y',   default_value='0.0'),
        DeclareLaunchArgument('follower_initial_yaw', default_value='0.0'),
        DeclareLaunchArgument(
            'use_slam', default_value='false',
            description='false=AMCL with received map; true=Cartographer SLAM.',
        ),
        DeclareLaunchArgument(
            'slam_domain', default_value='25',
            description='Which domain runs SLAM (25 or 24). Controls /map bridge direction.',
        ),
        UnsetEnvironmentVariable('ROS_DISCOVERY_SERVER'),
        UnsetEnvironmentVariable('ROS_LOCALHOST_ONLY'),
        UnsetEnvironmentVariable('FASTRTPS_DEFAULT_PROFILES_FILE'),
        UnsetEnvironmentVariable('FASTDDS_DEFAULT_PROFILES_FILE'),
        SetEnvironmentVariable('ROS_DOMAIN_ID',               domain_id),
        SetEnvironmentVariable('RMW_IMPLEMENTATION',          'rmw_fastrtps_cpp'),
        SetEnvironmentVariable('FASTDDS_BUILTIN_TRANSPORTS',  'UDPv4'),
        SetEnvironmentVariable('ROS_AUTOMATIC_DISCOVERY_RANGE', 'SUBNET'),
        SetEnvironmentVariable('ROS_STATIC_PEERS',            '10.10.14.10;10.10.14.14'),
        SetEnvironmentVariable('TURTLEBOT3_MODEL',            robot_model),
        LogInfo(
            msg=[
                'REAL_BURGER_D24 | use_slam=', use_slam,
                ' | slam_domain=', slam_domain,
                ' | follower_initial=(', follower_initial_x, ',', follower_initial_y, ')',
                ' | start_following=', start_following,
            ]
        ),
        # Bridge: /map direction depends on slam_domain
        TimerAction(period=0.5, actions=[OpaqueFunction(function=make_domain_bridges)]),
        # Localization: AMCL or Cartographer
        OpaqueFunction(function=make_localization),
        # Burger pose: reads map→base_footprint TF
        TimerAction(period=3.5, actions=[burger_pose]),
        # Nav2 planning stack
        TimerAction(
            period=4.5,
            actions=[controller_server, planner_server, behavior_server, bt_navigator],
        ),
        TimerAction(period=8.0, actions=[lifecycle_navigation]),
        TimerAction(period=10.0, actions=[burger_named_goal]),
        TimerAction(period=11.0, actions=[follower]),
    ])
