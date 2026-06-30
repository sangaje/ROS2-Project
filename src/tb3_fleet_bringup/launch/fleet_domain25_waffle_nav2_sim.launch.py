#!/usr/bin/env python3

import os
import re
import tempfile
from pathlib import Path
from typing import Iterable, Tuple

from ament_index_python.packages import get_package_share_directory, PackageNotFoundError
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, TimerAction, SetEnvironmentVariable, LogInfo
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch.conditions import IfCondition
from launch_ros.actions import Node


def _first_existing(paths):
    for p in paths:
        if p and os.path.exists(p):
            return p
    return ''


def _default_map() -> str:
    candidates = []
    try:
        tb3_nav_share = get_package_share_directory('turtlebot3_navigation2')
        candidates.extend([
            os.path.join(tb3_nav_share, 'map', 'map.yaml'),
            os.path.join(tb3_nav_share, 'maps', 'map.yaml'),
            os.path.join(tb3_nav_share, 'map', 'turtlebot3_world.yaml'),
            os.path.join(tb3_nav_share, 'maps', 'turtlebot3_world.yaml'),
        ])
    except PackageNotFoundError:
        pass
    try:
        nav2_share = get_package_share_directory('nav2_bringup')
        candidates.extend([
            os.path.join(nav2_share, 'maps', 'tb3_sandbox.yaml'),
            os.path.join(nav2_share, 'maps', 'depot.yaml'),
            os.path.join(nav2_share, 'maps', 'warehouse.yaml'),
        ])
    except PackageNotFoundError:
        pass
    return _first_existing(candidates)


def _default_house_map() -> str:
    candidates = []
    try:
        tb3_nav_share = get_package_share_directory('turtlebot3_navigation2')
        candidates.extend([
            os.path.join(tb3_nav_share, 'map', 'turtlebot3_house.yaml'),
            os.path.join(tb3_nav_share, 'maps', 'turtlebot3_house.yaml'),
            os.path.join(tb3_nav_share, 'map', 'house.yaml'),
            os.path.join(tb3_nav_share, 'maps', 'house.yaml'),
        ])
    except PackageNotFoundError:
        pass
    try:
        tb3_gazebo_share = get_package_share_directory('turtlebot3_gazebo')
        candidates.extend([
            os.path.join(tb3_gazebo_share, 'map', 'turtlebot3_house.yaml'),
            os.path.join(tb3_gazebo_share, 'maps', 'turtlebot3_house.yaml'),
            os.path.join(tb3_gazebo_share, 'map', 'house.yaml'),
            os.path.join(tb3_gazebo_share, 'maps', 'house.yaml'),
        ])
    except PackageNotFoundError:
        pass
    # Do NOT fall back to map.yaml here. map.yaml is usually the TurtleBot3 world map,
    # and silently using it with turtlebot3_house.world makes RViz show the wrong map.
    return _first_existing(candidates)


def _first_existing_model_sdf(tb3_gazebo_share: str, candidates: Iterable[str]) -> Tuple[str, str]:
    for model_dir_name in candidates:
        sdf = os.path.join(tb3_gazebo_share, 'models', model_dir_name, 'model.sdf')
        if os.path.exists(sdf):
            return model_dir_name, sdf
    raise RuntimeError(f'No TurtleBot3 model.sdf found. Tried: {", ".join(candidates)}')


def _patch_model_topics(src_sdf: str, robot_name: str, model_label: str) -> str:
    text = Path(src_sdf).read_text(encoding='utf-8')
    cmd_topic = f'/{robot_name}/cmd_vel'
    odom_topic = f'/{robot_name}/odometry'
    scan_topic = f'/{robot_name}/scan'

    plugin_re = re.compile(r'(<plugin[^>]*(?:DiffDrive|diff_drive|diff-drive)[^>]*>)(.*?)(</plugin>)', re.IGNORECASE | re.DOTALL)

    def patch_tag(body: str, tag: str, value: str) -> str:
        if re.search(rf'<{tag}>.*?</{tag}>', body, flags=re.DOTALL):
            return re.sub(rf'<{tag}>.*?</{tag}>', f'<{tag}>{value}</{tag}>', body, count=1, flags=re.DOTALL)
        return body + f'\n      <{tag}>{value}</{tag}>\n'

    def repl_plugin(match: re.Match) -> str:
        start, body, end = match.group(1), match.group(2), match.group(3)
        body = patch_tag(body, 'topic', cmd_topic)
        body = patch_tag(body, 'odom_topic', odom_topic)
        body = patch_tag(body, 'tf_topic', f'/{robot_name}/tf')
        body = patch_tag(body, 'frame_id', 'odom')
        body = patch_tag(body, 'child_frame_id', 'base_footprint')
        return start + body + end

    patched, n = plugin_re.subn(repl_plugin, text, count=1)
    if n == 0:
        patched = re.sub(r'<topic>[^<]*cmd_vel[^<]*</topic>', f'<topic>{cmd_topic}</topic>', text, count=1, flags=re.IGNORECASE)

    sensor_re = re.compile(r'(<sensor[^>]*(?:hls_lfcd_lds|gpu_lidar|lidar|ray)[^>]*>)(.*?)(</sensor>)', re.IGNORECASE | re.DOTALL)

    def repl_sensor(match: re.Match) -> str:
        start, body, end = match.group(1), match.group(2), match.group(3)
        body = patch_tag(body, 'topic', scan_topic)
        return start + body + end

    patched, _ = sensor_re.subn(repl_sensor, patched, count=1)
    for required in [cmd_topic, odom_topic, scan_topic]:
        if required not in patched:
            raise RuntimeError(f'Failed to patch required topic {required} in {src_sdf}')

    out_dir = Path(tempfile.gettempdir()) / 'tb3_fleet_patched_sdf_v41_domain_split'
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f'{robot_name}_{model_label}_v41.sdf'
    out_path.write_text(patched, encoding='utf-8')
    return str(out_path)


def generate_launch_description():
    tb3_gazebo_share = get_package_share_directory('turtlebot3_gazebo')
    bringup_share = get_package_share_directory('tb3_fleet_bringup')

    default_world = os.path.join(tb3_gazebo_share, 'worlds', 'turtlebot3_world.world')
    default_house = os.path.join(tb3_gazebo_share, 'worlds', 'turtlebot3_house.world')
    if not os.path.exists(default_house):
        default_house = default_world

    default_map = _default_map()
    default_house_map = _default_house_map()

    burger_model_dir, burger_sdf = _first_existing_model_sdf(tb3_gazebo_share, ['turtlebot3_burger'])
    waffle_model_dir, waffle_sdf = _first_existing_model_sdf(tb3_gazebo_share, ['turtlebot3_waffle', 'turtlebot3_waffle_pi'])
    burger_patched_sdf = _patch_model_topics(burger_sdf, 'burger', burger_model_dir)
    waffle_patched_sdf = _patch_model_topics(waffle_sdf, 'waffle', waffle_model_dir)

    tb3_models_dir = os.path.join(tb3_gazebo_share, 'models')
    old_gz_resource_path = os.environ.get('GZ_SIM_RESOURCE_PATH', '')
    gz_resource_path = ':'.join([tb3_models_dir, tb3_gazebo_share, old_gz_resource_path] if old_gz_resource_path else [tb3_models_dir, tb3_gazebo_share])

    world_preset = LaunchConfiguration('world_preset')
    world_override = LaunchConfiguration('world')
    localization_mode = LaunchConfiguration('localization_mode')
    world = PythonExpression([
        "'", world_override, "' if '", world_override, "' != '' else (",
        "'", default_house, "' if '", world_preset, "' == 'house' else ",
        "'", default_world, "' if '", world_preset, "' == 'world' else '", world_preset, "')"
    ])

    map_override = LaunchConfiguration('map')
    map_preset = LaunchConfiguration('map_preset')
    map_yaml = PythonExpression([
        "'", map_override, "' if '", map_override, "' != '' else (",
        "'", default_map, "' if '", map_preset, "' == 'world' else (",
        "'", default_house_map, "' if '", map_preset, "' == 'house' else (",
        "'", default_house_map, "' if '", world_preset, "' == 'house' else '", default_map, "')))"
    ])

    gz_verbosity = LaunchConfiguration('gz_verbosity')
    burger_x = LaunchConfiguration('burger_x')
    burger_y = LaunchConfiguration('burger_y')
    burger_yaw = LaunchConfiguration('burger_yaw')
    waffle_x = LaunchConfiguration('waffle_x')
    waffle_y = LaunchConfiguration('waffle_y')
    waffle_yaw = LaunchConfiguration('waffle_yaw')
    slam_map_publish_period_sec = LaunchConfiguration('slam_map_publish_period_sec')
    slam_resolution = LaunchConfiguration('slam_resolution')
    control_mode = LaunchConfiguration('control_mode')
    domain_id = LaunchConfiguration('domain_id')
    astar_target_mode = LaunchConfiguration('astar_target_mode')
    astar_goal_topic = LaunchConfiguration('astar_goal_topic')

    bridge_config = os.path.join(bringup_share, 'config', 'domain25_waffle_ros_gz_bridge.yaml')
    nav2_static_params = os.path.join(bringup_share, 'config', 'domain25_waffle_nav2.yaml')
    nav2_slam_params = os.path.join(bringup_share, 'config', 'domain25_waffle_nav2_slam.yaml')
    cartographer_config_dir = os.path.join(bringup_share, 'config')
    cartographer_config_basename = LaunchConfiguration('cartographer_config_basename')

    single_twist_script = os.path.join(bringup_share, 'scripts', 'single_twist_stamped_to_twist_direct_v36.py')
    frame_tools_script = os.path.join(bringup_share, 'scripts', 'single_domain_nav2_frame_tools_direct_v40.py')
    tf_pose_script = os.path.join(bringup_share, 'scripts', 'tf_pose_publisher_direct_v44.py')
    map_odom_localization_script = os.path.join(bringup_share, 'scripts', 'map_odom_localization_direct_v44.py')
    goal_proxy_script = os.path.join(bringup_share, 'scripts', 'pose_to_nav2_action_direct_v41.py')
    astar_cmd_script = os.path.join(bringup_share, 'scripts', 'astar_cmd_vel_follower_direct_v55.py')

    is_slam = IfCondition(PythonExpression(["'", localization_mode, "' == 'slam'"]))
    is_static = IfCondition(PythonExpression(["'", localization_mode, "' != 'slam'"]))
    is_nav2_control = IfCondition(PythonExpression(["'", control_mode, "' == 'nav2'"]))
    is_astar_control = IfCondition(PythonExpression(["'", control_mode, "' == 'astar_cmd'"]))

    gz_sim = ExecuteProcess(cmd=['gz', 'sim', '-r', '-v', gz_verbosity, world], output='screen', name='gz_sim_burger_waffle_v55')

    spawn_burger = Node(package='ros_gz_sim', executable='create', name='spawn_burger', output='screen', arguments=['-file', burger_patched_sdf, '-name', 'burger', '-x', burger_x, '-y', burger_y, '-z', '0.05', '-Y', burger_yaw])
    spawn_waffle = Node(package='ros_gz_sim', executable='create', name='spawn_waffle', output='screen', arguments=['-file', waffle_patched_sdf, '-name', 'waffle', '-x', waffle_x, '-y', waffle_y, '-z', '0.05', '-Y', waffle_yaw])

    converter = ExecuteProcess(cmd=['python3', single_twist_script, '--ros-args', '-r', '__node:=single_twist_stamped_to_twist_bridge', '-p', 'use_sim_time:=true', '-p', 'robot_name:=waffle', '-p', 'cmd_vel_topic:=/cmd_vel', '-p', 'internal_cmd_vel_topics:=/gz_cmd_vel_unstamped,/gz_cmd_vel_model_unstamped', '-p', 'cmd_republish_rate_hz:=0.0', '-p', 'watchdog_timeout_sec:=0.5', '-p', 'log_every_n_republish:=100'], output='screen', name='waffle_twist_stamped_to_twist_v55')

    bridge = Node(package='ros_gz_bridge', executable='parameter_bridge', name='waffle_ros_gz_bridge_domain25', output='screen', parameters=[{'config_file': bridge_config}])

    frame_tools = ExecuteProcess(cmd=['python3', frame_tools_script, '--ros-args', '-r', '__node:=single_domain_nav2_frame_tools', '-p', 'use_sim_time:=true', '-p', 'robot_name:=waffle', '-p', ['initial_x:=', waffle_x], '-p', ['initial_y:=', waffle_y], '-p', ['initial_yaw:=', waffle_yaw], '-p', 'reset_odom_origin_on_first_msg:=true', '-p', 'initial_pose_repeat_count:=40', '-p', 'initial_pose_period_sec:=0.25'], output='screen', name='waffle_frame_tools_v55')

    # Static-map mode only. In SLAM mode Cartographer owns map->odom.
    map_odom_localization = ExecuteProcess(cmd=['python3', map_odom_localization_script, '--ros-args', '-r', '__node:=map_odom_localization', '-p', 'use_sim_time:=true', '-p', 'robot_name:=waffle', '-p', 'odom_topic:=/odom_nav', '-p', 'map_frame:=map', '-p', 'odom_frame:=odom', '-p', 'base_frame:=base_footprint', '-p', ['initial_x:=', waffle_x], '-p', ['initial_y:=', waffle_y], '-p', ['initial_yaw:=', waffle_yaw], '-p', 'publish_rate_hz:=30.0', '-p', 'publish_amcl_pose:=true'], output='screen', name='waffle_map_odom_localization_v55', condition=is_static)

    cartographer = Node(
        package='cartographer_ros',
        executable='cartographer_node',
        name='cartographer_node',
        output='screen',
        parameters=[{'use_sim_time': True}],
        arguments=['-configuration_directory', cartographer_config_dir, '-configuration_basename', cartographer_config_basename],
        remappings=[('scan', '/scan_nav'), ('odom', '/odom_nav')],
        condition=is_slam,
    )
    occupancy_grid = Node(
        package='cartographer_ros',
        executable='cartographer_occupancy_grid_node',
        name='cartographer_occupancy_grid_node',
        output='screen',
        parameters=[{'use_sim_time': True}],
        arguments=['-resolution', slam_resolution, '-publish_period_sec', slam_map_publish_period_sec],
        condition=is_slam,
    )

    leader_pose = ExecuteProcess(cmd=['python3', tf_pose_script, '--ros-args', '-r', '__node:=leader_pose_tf_publisher', '-p', 'use_sim_time:=true', '-p', 'target_frame:=map', '-p', 'source_frame:=base_footprint', '-p', 'output_topic:=/leader_pose', '-p', 'publish_rate_hz:=10.0', '-p', 'log_every_n:=100'], output='screen', name='waffle_leader_pose_tf_publisher_v55')

    map_server = Node(package='nav2_map_server', executable='map_server', name='map_server', output='screen', parameters=[nav2_static_params, {'use_sim_time': True, 'yaml_filename': map_yaml}], condition=is_static)
    map_lifecycle = Node(package='nav2_lifecycle_manager', executable='lifecycle_manager', name='lifecycle_manager_map', output='screen', parameters=[nav2_static_params], condition=is_static)

    controller_server = Node(package='nav2_controller', executable='controller_server', name='controller_server', output='screen', parameters=[nav2_slam_params])
    planner_server = Node(package='nav2_planner', executable='planner_server', name='planner_server', output='screen', parameters=[nav2_slam_params])
    behavior_server = Node(package='nav2_behaviors', executable='behavior_server', name='behavior_server', output='screen', parameters=[nav2_slam_params])
    bt_navigator = Node(package='nav2_bt_navigator', executable='bt_navigator', name='bt_navigator', output='screen', parameters=[nav2_slam_params])
    nav_lifecycle = Node(package='nav2_lifecycle_manager', executable='lifecycle_manager', name='lifecycle_manager_navigation', output='screen', parameters=[nav2_slam_params])

    goal_proxy_default = ExecuteProcess(cmd=['python3', goal_proxy_script, '--ros-args', '-r', '__node:=waffle_default_goal_pose_to_nav2', '-p', 'use_sim_time:=true', '-p', 'goal_pose_topic:=/goal_pose', '-p', 'navigate_action:=/navigate_to_pose', '-p', 'default_frame_id:=map', '-p', 'cancel_previous_goal:=true'], output='screen', name='waffle_default_goal_pose_to_nav2_v55')
    goal_proxy_named = ExecuteProcess(cmd=['python3', goal_proxy_script, '--ros-args', '-r', '__node:=waffle_named_goal_pose_to_nav2', '-p', 'use_sim_time:=true', '-p', 'goal_pose_topic:=/waffle_goal_pose', '-p', 'navigate_action:=/navigate_to_pose', '-p', 'default_frame_id:=map', '-p', 'cancel_previous_goal:=true'], output='screen', name='waffle_named_goal_pose_to_nav2_v55')

    waffle_astar_cmd = ExecuteProcess(cmd=[
        'python3', astar_cmd_script,
        '--ros-args',
        '-r', '__node:=waffle_astar_cmd_vel_controller',
        '-p', 'use_sim_time:=true',
        '-p', 'robot_name:=waffle',
        '-p', 'map_topic:=/map',
        '-p', 'robot_pose_topic:=/leader_pose',
        '-p', 'leader_pose_topic:=/leader_pose',
        '-p', ['manual_goal_topic:=', astar_goal_topic],
        '-p', 'scan_topic:=/scan_nav',
        '-p', 'cmd_vel_topic:=/cmd_vel',
        '-p', 'path_topic:=/waffle_astar_path',
        '-p', ['target_mode:=', astar_target_mode],
        '-p', 'leader_goal_mode:=leader_pose',
        '-p', 'follow_distance:=0.0',
        '-p', 'goal_tolerance:=0.22',
        '-p', 'replan_period_sec:=0.5',
        '-p', 'control_rate_hz:=10.0',
        '-p', 'lookahead_distance:=0.28',
        '-p', 'occupied_threshold:=45',
        '-p', 'treat_unknown_as_obstacle:=false',
        '-p', 'inflation_radius_m:=0.22',
            '-p', 'soft_inflation_radius_m:=0.42',
            '-p', 'clearance_cost_weight:=7.0',
            '-p', 'unknown_cost_weight:=3.0',
            '-p', 'diagonal_motion:=false',
        '-p', 'max_linear:=0.075',
        '-p', 'min_linear:=0.02',
        '-p', 'max_angular:=0.50',
        '-p', 'front_stop_distance:=0.35',
        '-p', 'front_slow_distance:=0.60',
        '-p', 'stale_goal_sec:=0.0',
        '-p', 'direct_fallback_if_no_path:=false',
        '-p', 'log_period_sec:=1.0',
    ], output='screen', name='waffle_astar_cmd_vel_controller_v55')

    return LaunchDescription([
        DeclareLaunchArgument('domain_id', default_value='25', description='ROS_DOMAIN_ID used by Waffle/Gazebo owner domain.'),
        DeclareLaunchArgument('localization_mode', default_value='slam', description='slam: Cartographer live mapping owns /map and map->odom. static: map_server owns /map and deterministic map->odom is used.'),
        DeclareLaunchArgument('world_preset', default_value='house', description='world | house | absolute world file path. Use world:=... to force override.'),
        DeclareLaunchArgument('world', default_value='', description='Absolute Gazebo world path override. Empty means use world_preset.'),
        DeclareLaunchArgument('map', default_value='', description='Static-map mode only: absolute Nav2 map yaml override.'),
        DeclareLaunchArgument('map_preset', default_value='auto', description='Static-map mode only: auto | world | house.'),
        DeclareLaunchArgument('cartographer_config_basename', default_value='cartographer_2d_lidar_odom_v44.lua'),
        DeclareLaunchArgument('control_mode', default_value='astar_cmd', description='nav2: Waffle uses Nav2 NavigateToPose. astar_cmd: Waffle directly publishes /cmd_vel TwistStamped from A* path tracking.'),
        DeclareLaunchArgument('astar_target_mode', default_value='manual', description='Waffle A* mode: manual uses astar_goal_topic. leader is not useful for Waffle.'),
        DeclareLaunchArgument('astar_goal_topic', default_value='/goal_pose', description='Waffle A* mode goal topic. RViz 2D Nav Goal publishes /goal_pose by default.'),
        DeclareLaunchArgument('slam_resolution', default_value='0.05'),
        DeclareLaunchArgument('slam_map_publish_period_sec', default_value='1.0'),
        DeclareLaunchArgument('gz_verbosity', default_value='2'),
        DeclareLaunchArgument('burger_x', default_value='-3.20'),
        DeclareLaunchArgument('burger_y', default_value='-1.75'),
        DeclareLaunchArgument('burger_yaw', default_value='0.0'),
        DeclareLaunchArgument('waffle_x', default_value='-2.25'),
        DeclareLaunchArgument('waffle_y', default_value='-1.75'),
        DeclareLaunchArgument('waffle_yaw', default_value='0.0'),
        SetEnvironmentVariable('ROS_DOMAIN_ID', domain_id),
        SetEnvironmentVariable('RMW_IMPLEMENTATION', 'rmw_fastrtps_cpp'),
        SetEnvironmentVariable('FASTDDS_BUILTIN_TRANSPORTS', 'UDPv4'),
        SetEnvironmentVariable('TURTLEBOT3_MODEL', 'waffle'),
        SetEnvironmentVariable('GZ_SIM_RESOURCE_PATH', gz_resource_path),
        SetEnvironmentVariable('IGN_GAZEBO_RESOURCE_PATH', gz_resource_path),
        LogInfo(msg='V55_DOMAIN25_WAFFLE_OWNER | Gazebo + Waffle Nav2 + Cartographer/static-map selectable + shared /map owner.'),
        LogInfo(msg=['V55_LOCALIZATION_MODE | ', localization_mode]),
        LogInfo(msg=['V55_WORLD_SELECTED | ', world]),
        LogInfo(msg=['V55_STATIC_MAP_SELECTED | ', map_yaml]),
        LogInfo(msg=['V55_CARTOGRAPHER_CONFIG | ', cartographer_config_dir, '/', cartographer_config_basename]),
        LogInfo(msg=['V55_WAFFLE_CONTROL_MODE | ', control_mode, ' | nav2 or astar_cmd']),
        LogInfo(msg='V55_SAFE_HOUSE_SPAWN | default Waffle=(-2.25,-1.75,0.0), Burger=(-3.20,-1.75,0.0). Override burger_x/y and waffle_x/y if needed.'),
        gz_sim,
        TimerAction(period=2.0, actions=[converter, bridge]),
        TimerAction(period=3.0, actions=[spawn_burger]),
        TimerAction(period=5.0, actions=[spawn_waffle]),
        TimerAction(period=7.0, actions=[frame_tools]),
        TimerAction(period=8.0, actions=[map_odom_localization, cartographer, occupancy_grid, leader_pose]),
        TimerAction(period=10.0, actions=[map_server, map_lifecycle]),
        TimerAction(period=20.0, actions=[controller_server, planner_server, behavior_server, bt_navigator], condition=is_nav2_control),
        TimerAction(period=28.0, actions=[nav_lifecycle], condition=is_nav2_control),
        TimerAction(period=30.0, actions=[goal_proxy_default, goal_proxy_named], condition=is_nav2_control),
        TimerAction(period=9.0, actions=[waffle_astar_cmd]),
    ])
