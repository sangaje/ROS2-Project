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
    IncludeLaunchDescription,
    LogInfo,
    OpaqueFunction,
    SetEnvironmentVariable,
    TimerAction,
    UnsetEnvironmentVariable,
)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node
from nav2_common.launch import RewrittenYaml


def generate_launch_description():
    bringup_share = get_package_share_directory('tb3_fleet_bringup')

    mode                 = LaunchConfiguration('mode')
    domain_id            = LaunchConfiguration('domain_id')
    main_domain_id       = LaunchConfiguration('main_domain_id')
    leader_domain_id     = LaunchConfiguration('leader_domain_id')
    robot_name           = LaunchConfiguration('robot_name')
    robot_model          = LaunchConfiguration('robot_model')
    follow_distance      = LaunchConfiguration('follow_distance')
    start_following      = LaunchConfiguration('start_following')
    enable_path_yield    = LaunchConfiguration('enable_path_yield')
    path_block_distance  = LaunchConfiguration('path_block_distance')
    yield_lateral_distance = LaunchConfiguration('yield_lateral_distance')
    follower_initial_x   = LaunchConfiguration('follower_initial_x')
    follower_initial_y   = LaunchConfiguration('follower_initial_y')
    follower_initial_yaw = LaunchConfiguration('follower_initial_yaw')
    use_slam             = LaunchConfiguration('use_slam')
    slam_domain          = LaunchConfiguration('slam_domain')
    start_domain_bridge  = LaunchConfiguration('start_domain_bridge')
    bridge_start_delay   = LaunchConfiguration('bridge_start_delay')
    start_robot_bringup  = LaunchConfiguration('start_robot_bringup')
    start_state_publisher = LaunchConfiguration('start_state_publisher')
    start_lidar          = LaunchConfiguration('start_lidar')
    start_base           = LaunchConfiguration('start_base')
    lds_model            = LaunchConfiguration('lds_model')
    usb_port             = LaunchConfiguration('usb_port')
    lidar_port           = LaunchConfiguration('lidar_port')

    real_condition = IfCondition(PythonExpression(["'", mode, "' == 'real'"]))
    sim_condition = IfCondition(PythonExpression(["'", mode, "' == 'sim'"]))
    robot_condition = IfCondition(PythonExpression([
        "'", mode, "' == 'real' and '", start_robot_bringup,
        "'.lower() in ['true', '1', 'yes', 'on']",
    ]))

    nav2_params = RewrittenYaml(
        source_file=os.path.join(bringup_share, 'config', 'domain24_burger_nav2_amcl.yaml'),
        param_rewrites={
            'use_sim_time': 'false',
            'odom_topic': '/odom',
            'scan_topic': '/scan_nav',
            'topic': '/scan_nav',
        },
        convert_types=True,
    )

    follower_script   = os.path.join(bringup_share, 'scripts', 'domain_bridge_nav2_follower_direct_v40.py')
    tf_pose_script    = os.path.join(bringup_share, 'scripts', 'tf_pose_publisher_direct_v44.py')
    goal_proxy_script = os.path.join(bringup_share, 'scripts', 'pose_to_nav2_action_direct_v41.py')
    map_relay_script  = os.path.join(bringup_share, 'scripts', 'sim_map_relay.py')
    scan_relay_script = os.path.join(bringup_share, 'scripts', 'scan_frame_relay.py')
    cartographer_config_dir = os.path.join(bringup_share, 'config')
    follower_robot_launch = os.path.join(bringup_share, 'launch', 'robot.launch.py')
    sim_follower_launch = os.path.join(bringup_share, 'launch', 'sim_follower.launch.py')

    def main_topic(name, suffix):
        return f'/burger_{suffix}' if name == 'burger' else f'/{name}_{suffix}'

    def selected_main_domain(context):
        legacy = leader_domain_id.perform(context).strip()
        return legacy if legacy else main_domain_id.perform(context)

    def enabled(value):
        return value.lower() in ('true', '1', 'yes', 'on')

    # ── Domain bridge (/map direction depends on slam_domain) ─────────────────
    def make_domain_bridges(context, *args, **kwargs):
        if not enabled(start_domain_bridge.perform(context)):
            return [LogInfo(msg='FOLLOWER_BRIDGE | disabled')]

        ld = selected_main_domain(context)
        bd = domain_id.perform(context)
        sd = slam_domain.perform(context).strip() or ld
        rn = robot_name.perform(context).strip() or 'burger'
        out = Path(tempfile.gettempdir()) / 'tb3_fleet_domain_bridge'
        out.mkdir(parents=True, exist_ok=True)

        l2b = out / f'{rn}_{ld}_to_{bd}.yaml'
        b2l = out / f'{rn}_{bd}_to_{ld}.yaml'
        goal_main_topic = main_topic(rn, 'goal_pose')
        pose_main_topic = main_topic(rn, 'pose')
        scan_main_topic = main_topic(rn, 'scan')
        plan_main_topic = main_topic(rn, 'plan')
        follow_status_topic = '/fleet/follow_enabled' if rn == 'burger' else f'/fleet/{rn}/follow_enabled'

        l2b_map_block = """  /map:
    type: nav_msgs/msg/OccupancyGrid
    remap: /map_bridge
    qos:
      reliability: reliable
      durability: volatile
      history: keep_last
      depth: 5
  /map_metadata:
    type: nav_msgs/msg/MapMetaData
    qos:
      reliability: reliable
      durability: volatile
      history: keep_last
      depth: 1
"""
        b2l_map_block = """  /map:
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
        l2b_map = l2b_map_block if sd == ld else ''
        b2l_map = b2l_map_block if sd == bd else ''

        l2b.write_text(
            f"""name: {rn}_{ld}_to_{bd}
from_domain: {ld}
to_domain: {bd}

topics:
{l2b_map}  /leader_pose:
    type: geometry_msgs/msg/PoseStamped
    qos:
      reliability: reliable
      durability: volatile
      history: keep_last
      depth: 10
  /initialpose:
    type: geometry_msgs/msg/PoseWithCovarianceStamped
    qos:
      reliability: reliable
      durability: volatile
      history: keep_last
      depth: 1
  /plan:
    type: nav_msgs/msg/Path
    remap: /waffle_plan
    qos:
      reliability: reliable
      durability: volatile
      history: keep_last
      depth: 10
  {goal_main_topic}:
    type: geometry_msgs/msg/PoseStamped
    remap: /burger_goal_pose
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
""", encoding='utf-8')

        b2l.write_text(
            f"""name: {rn}_{bd}_to_{ld}
from_domain: {bd}
to_domain: {ld}

topics:
{b2l_map}  /burger_pose:
    type: geometry_msgs/msg/PoseStamped
    remap: {pose_main_topic}
    qos:
      reliability: reliable
      durability: volatile
      history: keep_last
      depth: 10
  /burger_scan_relay:
    type: sensor_msgs/msg/LaserScan
    remap: {scan_main_topic}
    qos:
      reliability: best_effort
      durability: volatile
      history: keep_last
      depth: 10
  /plan:
    type: nav_msgs/msg/Path
    remap: {plan_main_topic}
    qos:
      reliability: reliable
      durability: volatile
      history: keep_last
      depth: 10
  /fleet/follow_enabled:
    type: std_msgs/msg/Bool
    remap: {follow_status_topic}
    qos:
      reliability: reliable
      durability: transient_local
      history: keep_last
      depth: 1
""", encoding='utf-8')

        return [
            ExecuteProcess(cmd=['ros2', 'run', 'domain_bridge', 'domain_bridge', str(l2b)],
                           output='screen',
                           name=f'{rn}_main_to_robot_bridge'),
            ExecuteProcess(cmd=['ros2', 'run', 'domain_bridge', 'domain_bridge', str(b2l)],
                           output='screen',
                           name=f'{rn}_robot_to_main_bridge'),
        ]

    # ── Localization: AMCL (default) or Cartographer ──────────────────────────
    def make_localization(context, *args, **kwargs):
        slam_mode = use_slam.perform(context).lower() in ('true', '1', 'yes')
        d = domain_id.perform(context)

        extra_env = {
            'ROS_DOMAIN_ID': d,
            'ROS_AUTOMATIC_DISCOVERY_RANGE': 'SUBNET',
            'ROS_LOCALHOST_ONLY': '0',
            'RMW_IMPLEMENTATION': 'rmw_fastrtps_cpp',
        }

        if slam_mode:
            cartographer = Node(
                package='cartographer_ros', executable='cartographer_node',
                name='cartographer_node', output='screen',
                parameters=[{'use_sim_time': False}],
                arguments=['-configuration_directory', cartographer_config_dir,
                           '-configuration_basename', 'cartographer_2d_lidar_odom_v44.lua'],
                remappings=[('scan', '/scan_nav')],
                additional_env=extra_env,
            )
            occ_grid = Node(
                package='cartographer_ros', executable='cartographer_occupancy_grid_node',
                name='cartographer_occupancy_grid_node', output='screen',
                parameters=[{'use_sim_time': False}],
                arguments=['-resolution', '0.05', '-publish_period_sec', '1.0'],
                additional_env=extra_env,
            )
            return [TimerAction(period=0.5, actions=[cartographer, occ_grid])]

        ix   = float(follower_initial_x.perform(context))
        iy   = float(follower_initial_y.perform(context))
        iyaw = float(follower_initial_yaw.perform(context))
        pose_overrides = {'amcl': {'ros__parameters': {
            'set_initial_pose': True,
            'initial_pose': {'x': ix, 'y': iy, 'z': 0.0, 'yaw': iyaw},
        }}}
        pose_yaml = Path(tempfile.gettempdir()) / 'burger_amcl_initial_pose.yaml'
        pose_yaml.write_text(yaml.dump(pose_overrides), encoding='utf-8')

        amcl = Node(
            package='nav2_amcl', executable='amcl', name='amcl', output='screen',
            parameters=[nav2_params, str(pose_yaml)],
            additional_env=extra_env,
        )
        lifecycle_loc = Node(
            package='nav2_lifecycle_manager', executable='lifecycle_manager',
            name='lifecycle_manager_localization', output='screen',
            parameters=[nav2_params], additional_env=extra_env,
        )
        return [
            TimerAction(period=2.0, actions=[amcl]),
            TimerAction(period=2.5, actions=[lifecycle_loc]),
        ]

    burger_pose = ExecuteProcess(
        cmd=['python3', tf_pose_script, '--ros-args',
             '-r', '__node:=burger_real_pose_publisher',
             '-p', 'use_sim_time:=false', '-p', 'target_frame:=map',
             '-p', 'source_frame:=base_footprint', '-p', 'output_topic:=/burger_pose',
             '-p', 'publish_rate_hz:=10.0', '-p', 'log_every_n:=100'],
        output='screen', name='burger_real_pose_publisher',
    )
    map_relay = ExecuteProcess(
        cmd=['python3', map_relay_script, '--ros-args',
             '-r', '__node:=real_follower_map_relay',
             '-p', 'use_sim_time:=false',
             '-p', 'input_topic:=/map_bridge',
             '-p', 'output_topic:=/map'],
        output='screen', name='real_follower_map_relay',
    )
    scan_relay = ExecuteProcess(
        cmd=['python3', scan_relay_script, '--ros-args',
             '-r', '__node:=real_burger_scan_frame_relay',
             '-p', 'use_sim_time:=false',
             '-p', 'input_topic:=/scan',
             '-p', 'output_topic:=/burger_scan_relay',
             '-p', 'output_frame:=burger/base_scan',
             '-p', 'input_reliability:=best_effort',
             '-p', 'output_reliability:=reliable'],
        output='screen', name='real_burger_scan_frame_relay',
    )
    scan_nav_relay = ExecuteProcess(
        cmd=['python3', scan_relay_script, '--ros-args',
             '-r', '__node:=burger_scan_nav_relay',
             '-p', 'use_sim_time:=false',
             '-p', 'input_topic:=/scan',
             '-p', 'output_topic:=/scan_nav',
             '-p', 'output_frame:=base_scan',
             '-p', 'input_reliability:=best_effort',
             '-p', 'output_reliability:=reliable'],
        output='screen', name='burger_scan_nav_relay',
    )
    domain_bridge_check = ExecuteProcess(
        cmd=['ros2', 'pkg', 'prefix', 'domain_bridge'],
        output='screen',
        name='check_domain_bridge_package',
    )
    robot_bringup = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(follower_robot_launch),
        launch_arguments={
            'role': 'follower',
            'domain_id': domain_id,
            'lds_model': lds_model,
            'usb_port': usb_port,
            'lidar_port': lidar_port,
            'start_state_publisher': start_state_publisher,
            'start_lidar': start_lidar,
            'start_base': start_base,
        }.items(),
        condition=robot_condition,
    )
    sim_follower = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(sim_follower_launch),
        launch_arguments={
            'domain_id': domain_id,
            'leader_domain_id': main_domain_id,
            'follow_distance': follow_distance,
            'start_following': start_following,
            'follower_initial_x': follower_initial_x,
            'follower_initial_y': follower_initial_y,
            'follower_initial_yaw': follower_initial_yaw,
        }.items(),
        condition=sim_condition,
    )

    controller_server = Node(package='nav2_controller', executable='controller_server',
                             name='controller_server', output='screen', parameters=[nav2_params])
    planner_server    = Node(package='nav2_planner', executable='planner_server',
                             name='planner_server', output='screen', parameters=[nav2_params])
    behavior_server   = Node(package='nav2_behaviors', executable='behavior_server',
                             name='behavior_server', output='screen', parameters=[nav2_params])
    bt_navigator      = Node(package='nav2_bt_navigator', executable='bt_navigator',
                             name='bt_navigator', output='screen', parameters=[nav2_params])
    lifecycle_nav     = Node(package='nav2_lifecycle_manager', executable='lifecycle_manager',
                             name='lifecycle_manager_navigation', output='screen',
                             parameters=[nav2_params])
    burger_named_goal = ExecuteProcess(
        cmd=['python3', goal_proxy_script, '--ros-args',
             '-r', '__node:=burger_named_goal',
             '-p', 'use_sim_time:=false', '-p', 'goal_pose_topic:=/burger_goal_pose',
             '-p', 'navigate_action:=/navigate_to_pose', '-p', 'default_frame_id:=map',
             '-p', 'cancel_previous_goal:=true'],
        output='screen', name='burger_named_goal',
    )
    follower = ExecuteProcess(
        cmd=['python3', follower_script, '--ros-args',
             '-r', '__node:=domain_bridge_follower',
             '-p', 'use_sim_time:=false',
             '-p', 'leader_pose_topic:=/leader_pose', '-p', 'leader_path_topic:=/waffle_plan',
             '-p', 'follower_pose_topic:=/burger_pose', '-p', 'map_topic:=/map',
             '-p', 'navigate_action:=/navigate_to_pose',
             '-p', ['follow_distance:=', follow_distance],
             '-p', 'goal_period_sec:=1.0', '-p', 'goal_update_distance:=0.20',
             '-p', 'cancel_previous_goal:=false',
             '-p', 'follow_command_topic:=/fleet/follow_command',
             '-p', 'follow_status_topic:=/fleet/follow_enabled',
             '-p', ['start_following:=', start_following],
             '-p', ['enable_path_yield:=', enable_path_yield],
             '-p', ['path_block_distance:=', path_block_distance],
             '-p', 'path_lookahead_min:=0.30', '-p', 'path_lookahead_max:=2.50',
             '-p', ['yield_lateral_distance:=', yield_lateral_distance],
             '-p', 'yield_release_distance:=0.80', '-p', 'yield_map_clearance:=0.18',
             '-p', 'yield_min_hold_sec:=4.0', '-p', 'yield_max_hold_sec:=12.0'],
        output='screen', name='burger_follower',
    )

    return LaunchDescription([
        DeclareLaunchArgument('mode',               default_value='real',
                              description='real or sim.'),
        DeclareLaunchArgument('domain_id',          default_value='24'),
        DeclareLaunchArgument('main_domain_id',     default_value='25',
                              description='Main/leader ROS domain used by domain_bridge.'),
        DeclareLaunchArgument('leader_domain_id',   default_value='',
                              description='Deprecated alias for main_domain_id. Empty uses main_domain_id.'),
        DeclareLaunchArgument('robot_name',          default_value='burger',
                              description='Main-domain name for this follower. Use burger for the first robot.'),
        DeclareLaunchArgument('robot_model',        default_value='burger'),
        DeclareLaunchArgument('follow_distance',    default_value='1.05'),
        DeclareLaunchArgument('start_following',    default_value='false'),
        DeclareLaunchArgument('enable_path_yield',      default_value='true'),
        DeclareLaunchArgument('path_block_distance',    default_value='0.55'),
        DeclareLaunchArgument('yield_lateral_distance', default_value='0.75'),
        DeclareLaunchArgument('follower_initial_x',  default_value='-1.05'),
        DeclareLaunchArgument('follower_initial_y',  default_value='0.0'),
        DeclareLaunchArgument('follower_initial_yaw', default_value='0.0'),
        DeclareLaunchArgument('use_slam',    default_value='false'),
        DeclareLaunchArgument('slam_domain', default_value='',
                              description='Domain that publishes /map. Empty uses main_domain_id.'),
        DeclareLaunchArgument('start_domain_bridge', default_value='true',
                              description='Start robot<->main domain_bridge from the follower side.'),
        DeclareLaunchArgument('bridge_start_delay', default_value='0.5'),
        DeclareLaunchArgument('start_robot_bringup', default_value='true',
                              description='Start follower base, lidar, and state publisher in this launch.'),
        DeclareLaunchArgument('start_state_publisher', default_value='true'),
        DeclareLaunchArgument('start_lidar', default_value='true'),
        DeclareLaunchArgument('start_base', default_value='true'),
        DeclareLaunchArgument('lds_model', default_value='LDS-02'),
        DeclareLaunchArgument('usb_port', default_value='/dev/ttyACM0'),
        DeclareLaunchArgument('lidar_port', default_value='/dev/ttyUSB0'),
        UnsetEnvironmentVariable('ROS_DISCOVERY_SERVER'),
        UnsetEnvironmentVariable('ROS_LOCALHOST_ONLY'),
        UnsetEnvironmentVariable('FASTRTPS_DEFAULT_PROFILES_FILE'),
        UnsetEnvironmentVariable('FASTDDS_DEFAULT_PROFILES_FILE'),
        SetEnvironmentVariable('ROS_DOMAIN_ID',               domain_id),
        SetEnvironmentVariable('ROS_AUTOMATIC_DISCOVERY_RANGE', 'SUBNET'),
        SetEnvironmentVariable('ROS_LOCALHOST_ONLY',           '0'),
        SetEnvironmentVariable('RMW_IMPLEMENTATION',          'rmw_fastrtps_cpp'),
        SetEnvironmentVariable('TURTLEBOT3_MODEL',            robot_model),
        LogInfo(msg=['FOLLOWER | mode=', mode, ' domain=', domain_id,
                     ' | robot=', robot_name,
                     ' | main_domain=', main_domain_id,
                     ' | use_slam=', use_slam,
                     ' | bridge=', start_domain_bridge,
                     ' | robot_bringup=', start_robot_bringup,
                     ' | slam_domain=', slam_domain,
                     ' | init=(', follower_initial_x, ',', follower_initial_y, ')',
                     ' | start_following=', start_following]),
        sim_follower,
        robot_bringup,
        TimerAction(period=0.2, actions=[domain_bridge_check], condition=real_condition),
        TimerAction(period=bridge_start_delay, actions=[OpaqueFunction(function=make_domain_bridges)],
                    condition=real_condition),
        TimerAction(period=1.0, actions=[map_relay, scan_relay, scan_nav_relay],
                    condition=real_condition),
        OpaqueFunction(function=make_localization, condition=real_condition),
        TimerAction(period=3.5, actions=[burger_pose], condition=real_condition),
        TimerAction(period=4.5, actions=[controller_server, planner_server,
                                          behavior_server, bt_navigator],
                    condition=real_condition),
        TimerAction(period=8.0, actions=[lifecycle_nav], condition=real_condition),
        TimerAction(period=10.0, actions=[burger_named_goal], condition=real_condition),
        TimerAction(period=11.0, actions=[follower], condition=real_condition),
    ])
