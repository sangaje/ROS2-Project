#!/usr/bin/env python3
"""Unified Waffle launch — simulation and real robot.

sim=true  : Gazebo + ros_gz_bridge + spawn + Cartographer + Nav2
sim=false : Cartographer + Nav2 (no simulation nodes)
"""

import os
import re
import tempfile
from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    TimerAction,
    SetEnvironmentVariable,
    LogInfo,
    OpaqueFunction,
)
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _require_files(paths):
    missing = [p for p in paths if not os.path.exists(p)]
    if missing:
        raise RuntimeError('Missing required fleet bringup files: ' + ', '.join(missing))


def _patch_model_topics(src_sdf, robot_name):
    text = Path(src_sdf).read_text(encoding='utf-8')
    cmd_topic = f'/{robot_name}/cmd_vel'
    odom_topic = f'/{robot_name}/odometry'
    scan_topic = f'/{robot_name}/scan'

    plugin_re = re.compile(
        r'(<plugin[^>]*(?:DiffDrive|diff_drive|diff-drive)[^>]*>)(.*?)(</plugin>)',
        re.IGNORECASE | re.DOTALL,
    )

    def patch_tag(body, tag, value):
        if re.search(rf'<{tag}>.*?</{tag}>', body, flags=re.DOTALL):
            return re.sub(rf'<{tag}>.*?</{tag}>', f'<{tag}>{value}</{tag}>', body,
                          count=1, flags=re.DOTALL)
        return body + f'\n      <{tag}>{value}</{tag}>\n'

    def repl_plugin(match):
        start, body, end = match.group(1), match.group(2), match.group(3)
        body = patch_tag(body, 'topic', cmd_topic)
        body = patch_tag(body, 'odom_topic', odom_topic)
        body = patch_tag(body, 'tf_topic', f'/{robot_name}/tf')
        body = patch_tag(body, 'frame_id', 'odom')
        body = patch_tag(body, 'child_frame_id', 'base_footprint')
        return start + body + end

    patched, n = plugin_re.subn(repl_plugin, text, count=1)
    if n == 0:
        patched = re.sub(
            r'<topic>[^<]*cmd_vel[^<]*</topic>', f'<topic>{cmd_topic}</topic>',
            text, count=1, flags=re.IGNORECASE,
        )

    sensor_re = re.compile(
        r'(<sensor[^>]*(?:hls_lfcd_lds|gpu_lidar|lidar|ray)[^>]*>)(.*?)(</sensor>)',
        re.IGNORECASE | re.DOTALL,
    )

    def repl_sensor(match):
        start, body, end = match.group(1), match.group(2), match.group(3)
        body = patch_tag(body, 'topic', scan_topic)
        return start + body + end

    patched, _ = sensor_re.subn(repl_sensor, patched, count=1)
    out_dir = Path(tempfile.gettempdir()) / 'tb3_fleet_patched_sdf'
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f'{robot_name}_patched.sdf'
    out_path.write_text(patched, encoding='utf-8')
    return str(out_path)


def _first_existing_sdf(tb3_gz_share, candidates):
    for name in candidates:
        sdf = os.path.join(tb3_gz_share, 'models', name, 'model.sdf')
        if os.path.exists(sdf):
            return sdf
    raise RuntimeError(f'No TurtleBot3 model.sdf found in: {", ".join(candidates)}')


# ---------------------------------------------------------------------------
# generate_launch_description
# ---------------------------------------------------------------------------

def generate_launch_description():
    bringup_share = get_package_share_directory('tb3_fleet_bringup')

    # ---- Script / config paths -----------------------------------------
    bridge_config = os.path.join(bringup_share, 'config', 'domain25_waffle_ros_gz_bridge.yaml')
    nav2_params = os.path.join(bringup_share, 'config', 'domain25_waffle_nav2_slam.yaml')
    carto_config_dir = os.path.join(bringup_share, 'config')
    carto_config_basename = 'cartographer_2d_lidar_odom.lua'

    single_twist_script = os.path.join(bringup_share, 'scripts', 'single_twist_stamped_to_twist.py')
    frame_tools_script = os.path.join(bringup_share, 'scripts', 'single_domain_nav2_frame_tools_direct_v40.py')
    tf_pose_script = os.path.join(bringup_share, 'scripts', 'tf_pose_publisher.py')
    map_odom_script = os.path.join(bringup_share, 'scripts', 'map_odom_localization.py')
    map_cleaner_script = os.path.join(bringup_share, 'scripts', 'robot_footprint_map_free_space_filter.py')
    through_poses_script = os.path.join(bringup_share, 'scripts', 'path_to_nav2_through_poses.py')
    gz_sim_clean = os.path.join(bringup_share, 'scripts', 'run_gz_sim_clean.bash')

    _require_files([
        bridge_config,
        nav2_params,
        single_twist_script,
        frame_tools_script,
        tf_pose_script,
        map_odom_script,
        map_cleaner_script,
        through_poses_script,
    ])

    # ---- LaunchConfiguration handles -----------------------------------
    sim = LaunchConfiguration('sim')
    localization_mode = LaunchConfiguration('localization_mode')
    world = LaunchConfiguration('world')
    waffle_x = LaunchConfiguration('waffle_x')
    waffle_y = LaunchConfiguration('waffle_y')
    waffle_yaw = LaunchConfiguration('waffle_yaw')
    burger_x = LaunchConfiguration('burger_x')
    burger_y = LaunchConfiguration('burger_y')
    domain_id = LaunchConfiguration('domain_id')

    # ---- Conditions -------------------------------------------------------
    is_sim = IfCondition(PythonExpression(["'", sim, "' == 'true'"]))
    is_real = IfCondition(PythonExpression(["'", sim, "' != 'true'"]))
    is_slam = IfCondition(PythonExpression(["'", localization_mode, "' == 'slam'"]))
    is_not_slam = IfCondition(PythonExpression(["'", localization_mode, "' != 'slam'"]))

    # use_sim_time string for -p flag
    use_sim_time_str = PythonExpression(["'true' if '", sim, "' == 'true' else 'false'"])

    # ---- Gazebo (sim only) ------------------------------------------------
    def _make_sim_nodes(context, *args, **kwargs):
        sim_val = sim.perform(context)
        if sim_val != 'true':
            return []

        world_val = world.perform(context)
        waffle_x_val = waffle_x.perform(context)
        waffle_y_val = waffle_y.perform(context)
        burger_x_val = burger_x.perform(context)
        burger_y_val = burger_y.perform(context)

        try:
            tb3_gz_share = get_package_share_directory('turtlebot3_gazebo')
        except Exception:
            raise RuntimeError('turtlebot3_gazebo package not found')

        tb3_models_dir = os.path.join(tb3_gz_share, 'models')
        old_gz_path = os.environ.get('GZ_SIM_RESOURCE_PATH', '')
        gz_path = ':'.join(
            [tb3_models_dir, tb3_gz_share, old_gz_path]
            if old_gz_path else [tb3_models_dir, tb3_gz_share]
        )

        waffle_sdf_src = _first_existing_sdf(tb3_gz_share,
                                              ['turtlebot3_waffle', 'turtlebot3_waffle_pi'])
        burger_sdf_src = _first_existing_sdf(tb3_gz_share, ['turtlebot3_burger'])
        waffle_sdf = _patch_model_topics(waffle_sdf_src, 'waffle')
        burger_sdf = _patch_model_topics(burger_sdf_src, 'burger')

        if not world_val:
            worlds = [
                os.path.join(tb3_gz_share, 'worlds', 'turtlebot3_house.world'),
                os.path.join(tb3_gz_share, 'worlds', 'turtlebot3_world.world'),
            ]
            world_val = next((w for w in worlds if os.path.exists(w)), worlds[-1])

        os.environ['GZ_SIM_RESOURCE_PATH'] = gz_path
        os.environ['IGN_GAZEBO_RESOURCE_PATH'] = gz_path

        gz_cmd = [gz_sim_clean, '-r', '-v', '2', world_val]
        if not os.path.exists(gz_sim_clean):
            gz_cmd = ['gz', 'sim', '-r', '-v', '2', world_val]

        gz = ExecuteProcess(cmd=gz_cmd, output='screen', name='gz_sim_waffle')
        bridge = Node(
            package='ros_gz_bridge', executable='parameter_bridge',
            name='waffle_ros_gz_bridge', output='screen',
            parameters=[{'config_file': bridge_config}],
        )
        spawn_waffle = Node(
            package='ros_gz_sim', executable='create', name='spawn_waffle',
            output='screen',
            arguments=['-file', waffle_sdf, '-name', 'waffle',
                       '-x', waffle_x_val, '-y', waffle_y_val, '-z', '0.05',
                       '-Y', waffle_yaw.perform(context)],
        )
        spawn_burger = Node(
            package='ros_gz_sim', executable='create', name='spawn_burger',
            output='screen',
            arguments=['-file', burger_sdf, '-name', 'burger',
                       '-x', burger_x_val, '-y', burger_y_val, '-z', '0.05', '-Y', '0.0'],
        )
        converter = ExecuteProcess(
            cmd=[
                'python3', single_twist_script, '--ros-args',
                '-r', '__node:=waffle_twist_stamped_to_twist',
                '-p', 'use_sim_time:=true',
                '-p', 'robot_name:=waffle',
                '-p', 'cmd_vel_topic:=/cmd_vel_stamped',
                '-p', 'internal_cmd_vel_topics:=/cmd_vel,/gz_cmd_vel_unstamped',
                '-p', 'watchdog_timeout_sec:=0.5',
            ],
            output='screen', name='waffle_twist_stamped_to_twist',
        )

        return [
            gz,
            TimerAction(period=2.0, actions=[bridge, converter]),
            TimerAction(period=3.0, actions=[spawn_burger]),
            TimerAction(period=5.0, actions=[spawn_waffle]),
        ]

    def _make_real_twist(context, *args, **kwargs):
        sim_val = sim.perform(context)
        if sim_val == 'true':
            return []
        converter = ExecuteProcess(
            cmd=[
                'python3', single_twist_script, '--ros-args',
                '-r', '__node:=waffle_twist_stamped_to_twist',
                '-p', 'use_sim_time:=false',
                '-p', 'robot_name:=waffle',
                '-p', 'cmd_vel_topic:=/cmd_vel_stamped',
                '-p', 'internal_cmd_vel_topics:=/cmd_vel',
                '-p', 'watchdog_timeout_sec:=0.5',
            ],
            output='screen', name='waffle_twist_stamped_to_twist',
        )
        return [TimerAction(period=1.0, actions=[converter])]

    # ---- Always-on nodes --------------------------------------------------
    # Delay differs sim vs real but we unify by using the same fixed delays.
    # Sim-only nodes won't run on real (handled by OpaqueFunction above).

    frame_tools = ExecuteProcess(
        cmd=[
            'python3', frame_tools_script, '--ros-args',
            '-r', '__node:=waffle_frame_tools',
            '-p', ['use_sim_time:=', use_sim_time_str],
            '-p', 'robot_name:=waffle',
            '-p', ['initial_x:=', waffle_x],
            '-p', ['initial_y:=', waffle_y],
            '-p', ['initial_yaw:=', waffle_yaw],
            '-p', 'reset_odom_origin_on_first_msg:=true',
            '-p', 'initial_pose_repeat_count:=40',
            '-p', 'initial_pose_period_sec:=0.25',
            '-p', 'scan_out:=/scan_nav',
        ],
        output='screen', name='waffle_frame_tools',
    )

    map_odom = ExecuteProcess(
        cmd=[
            'python3', map_odom_script, '--ros-args',
            '-r', '__node:=waffle_map_odom_localization',
            '-p', ['use_sim_time:=', use_sim_time_str],
            '-p', 'robot_name:=waffle',
            '-p', 'odom_topic:=/odom_nav',
            '-p', 'map_frame:=map',
            '-p', 'odom_frame:=odom',
            '-p', 'base_frame:=base_footprint',
            '-p', ['initial_x:=', waffle_x],
            '-p', ['initial_y:=', waffle_y],
            '-p', ['initial_yaw:=', waffle_yaw],
            '-p', 'publish_rate_hz:=30.0',
            '-p', 'publish_amcl_pose:=true',
        ],
        output='screen', name='waffle_map_odom_localization',
        condition=is_not_slam,
    )

    cartographer = OpaqueFunction(function=lambda context, *a, **k: [Node(
        package='cartographer_ros',
        executable='cartographer_node',
        name='cartographer_node',
        output='screen',
        parameters=[{'use_sim_time': sim.perform(context) == 'true'}],
        arguments=[
            '-configuration_directory', carto_config_dir,
            '-configuration_basename', carto_config_basename,
        ],
        remappings=[('scan', '/scan_nav'), ('odom', '/odom_nav')],
        condition=is_slam,
    )])

    occupancy_grid = OpaqueFunction(function=lambda context, *a, **k: [Node(
        package='cartographer_ros',
        executable='cartographer_occupancy_grid_node',
        name='cartographer_occupancy_grid_node',
        output='screen',
        parameters=[{'use_sim_time': sim.perform(context) == 'true'}],
        arguments=['-resolution', '0.05', '-publish_period_sec', '0.5'],
        remappings=[('map', '/map_raw'), ('map_updates', '/map_raw_updates')],
        condition=is_slam,
    )])

    map_cleaner = ExecuteProcess(
        cmd=[
            'python3', map_cleaner_script, '--ros-args',
            '-r', '__node:=robot_footprint_map_free_space_filter',
            '-p', ['use_sim_time:=', use_sim_time_str],
            '-p', 'map_in_topic:=/map_raw',
            '-p', 'map_out_topic:=/map',
            '-p', 'map_metadata_out_topic:=/map_metadata',
            '-p', 'pose_topics:=/leader_pose,/burger_pose',
            '-p', 'clear_radius_m:=0.26',
            '-p', 'stale_pose_sec:=2.0',
        ],
        output='screen', name='robot_footprint_map_free_space_filter',
        condition=is_slam,
    )

    leader_pose = ExecuteProcess(
        cmd=[
            'python3', tf_pose_script, '--ros-args',
            '-r', '__node:=waffle_leader_pose_tf_publisher',
            '-p', ['use_sim_time:=', use_sim_time_str],
            '-p', 'target_frame:=map',
            '-p', 'source_frame:=base_footprint',
            '-p', 'output_topic:=/leader_pose',
            '-p', 'publish_rate_hz:=10.0',
            '-p', 'log_every_n:=100',
        ],
        output='screen', name='waffle_leader_pose_tf_publisher',
    )

    controller_server = Node(
        package='nav2_controller', executable='controller_server',
        name='controller_server', output='screen',
        parameters=[nav2_params],
        remappings=[('cmd_vel', '/cmd_vel_stamped')],
    )
    planner_server = Node(
        package='nav2_planner', executable='planner_server',
        name='planner_server', output='screen',
        parameters=[nav2_params],
    )
    behavior_server = Node(
        package='nav2_behaviors', executable='behavior_server',
        name='behavior_server', output='screen',
        parameters=[nav2_params],
    )
    bt_navigator = Node(
        package='nav2_bt_navigator', executable='bt_navigator',
        name='bt_navigator', output='screen',
        parameters=[nav2_params],
    )
    nav_lifecycle = Node(
        package='nav2_lifecycle_manager', executable='lifecycle_manager',
        name='lifecycle_manager_navigation', output='screen',
        parameters=[nav2_params],
    )

    through_poses = ExecuteProcess(
        cmd=[
            'python3', through_poses_script, '--ros-args',
            '-r', '__node:=waffle_through_poses',
            '-p', ['use_sim_time:=', use_sim_time_str],
            '-p', 'path_topic:=/waffle_waypoints',
            '-p', 'action_name:=/navigate_through_poses',
            '-p', 'default_frame_id:=map',
            '-p', 'change_threshold_m:=0.25',
            '-p', 'min_resend_sec:=1.5',
        ],
        output='screen', name='waffle_through_poses',
    )

    return LaunchDescription([
        # ---- Arguments ----------------------------------------------------
        DeclareLaunchArgument('sim', default_value='true',
                              description='true=Gazebo simulation, false=real robot'),
        DeclareLaunchArgument('localization_mode', default_value='slam',
                              description='slam=Cartographer, amcl=map_odom_localization'),
        DeclareLaunchArgument('world', default_value='',
                              description='Gazebo world file (sim only). Empty=auto house.'),
        DeclareLaunchArgument('waffle_x', default_value='1.38'),
        DeclareLaunchArgument('waffle_y', default_value='3.49'),
        DeclareLaunchArgument('waffle_yaw', default_value='0.0'),
        DeclareLaunchArgument('burger_x', default_value='0.58'),
        DeclareLaunchArgument('burger_y', default_value='3.49'),
        DeclareLaunchArgument('domain_id', default_value='25'),
        # ---- Environment --------------------------------------------------
        SetEnvironmentVariable('ROS_DOMAIN_ID', domain_id),
        SetEnvironmentVariable('RMW_IMPLEMENTATION', 'rmw_fastrtps_cpp'),
        SetEnvironmentVariable('FASTDDS_BUILTIN_TRANSPORTS', 'UDPv4'),
        SetEnvironmentVariable('TURTLEBOT3_MODEL', 'waffle'),
        LogInfo(msg='FLEET_WAFFLE_LAUNCH | unified sim/real waffle launch'),
        LogInfo(msg=['SIM | ', sim, ' | LOCALIZATION | ', localization_mode]),
        # ---- Sim-only nodes -----------------------------------------------
        OpaqueFunction(function=_make_sim_nodes),
        # ---- Real-only twist bridge ----------------------------------------
        OpaqueFunction(function=_make_real_twist),
        # ---- Always-on (frame tools at 6 s) --------------------------------
        TimerAction(period=6.0, actions=[frame_tools]),
        TimerAction(period=7.0, actions=[leader_pose]),
        # map_odom (static mode only)
        TimerAction(period=10.0, actions=[map_odom]),
        # Cartographer (slam mode only)
        TimerAction(period=10.0, actions=[cartographer, occupancy_grid]),
        TimerAction(period=11.0, actions=[map_cleaner]),
        # Nav2 stack
        TimerAction(period=20.0, actions=[controller_server, planner_server,
                                          behavior_server, bt_navigator]),
        TimerAction(period=28.0, actions=[nav_lifecycle]),
        TimerAction(period=32.0, actions=[through_poses]),
    ])
