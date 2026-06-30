#!/usr/bin/env python3
"""Unified Burger launch — simulation and real robot.

sim=true  : ros_gz_bridge + twist bridge + map_odom_localization + Nav2
sim=false : map_odom_localization + Nav2 (no simulation nodes)

Domain 26. Receives /map, /leader_pose, /burger_waypoints from Domain 25 via bridge.
"""

import os
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


def _require_files(paths):
    missing = [p for p in paths if not os.path.exists(p)]
    if missing:
        raise RuntimeError('Missing required fleet bringup files: ' + ', '.join(missing))


def generate_launch_description():
    bringup_share = get_package_share_directory('tb3_fleet_bringup')

    bridge_config = os.path.join(bringup_share, 'config', 'domain26_burger_ros_gz_bridge.yaml')
    nav2_params = os.path.join(bringup_share, 'config', 'domain26_burger_nav2_slam.yaml')

    single_twist_script = os.path.join(bringup_share, 'scripts', 'single_twist_stamped_to_twist.py')
    frame_tools_script = os.path.join(bringup_share, 'scripts', 'single_domain_nav2_frame_tools_direct_v40.py')
    map_odom_script = os.path.join(bringup_share, 'scripts', 'map_odom_localization.py')
    tf_pose_script = os.path.join(bringup_share, 'scripts', 'tf_pose_publisher.py')
    through_poses_script = os.path.join(bringup_share, 'scripts', 'path_to_nav2_through_poses.py')

    _require_files([
        bridge_config,
        nav2_params,
        single_twist_script,
        frame_tools_script,
        map_odom_script,
        tf_pose_script,
        through_poses_script,
    ])

    sim = LaunchConfiguration('sim')
    burger_x = LaunchConfiguration('burger_x')
    burger_y = LaunchConfiguration('burger_y')
    burger_yaw = LaunchConfiguration('burger_yaw')
    map_origin_x = LaunchConfiguration('map_origin_x')
    map_origin_y = LaunchConfiguration('map_origin_y')
    map_origin_yaw = LaunchConfiguration('map_origin_yaw')
    domain_id = LaunchConfiguration('domain_id')
    leader_domain_id = LaunchConfiguration('leader_domain_id')

    use_sim_time_str = PythonExpression(["'true' if '", sim, "' == 'true' else 'false'"])

    # ---- Domain bridge (OpaqueFunction) -----------------------------------
    def _write_bridge_configs(context, *args, **kwargs):
        leader_domain = leader_domain_id.perform(context)
        burger_domain = domain_id.perform(context)
        out_dir = Path(tempfile.gettempdir()) / 'tb3_fleet_domain_bridge'
        out_dir.mkdir(parents=True, exist_ok=True)

        shared_path = out_dir / f'fleet_25_to_{burger_domain}.yaml'
        debug_path = out_dir / f'fleet_{burger_domain}_to_25.yaml'

        shared_yaml = f"""name: fleet_shared_{leader_domain}_to_{burger_domain}
from_domain: {leader_domain}
to_domain: {burger_domain}

topics:
  /map:
    type: nav_msgs/msg/OccupancyGrid
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
      depth: 5
  /leader_pose:
    type: geometry_msgs/msg/PoseStamped
    qos:
      reliability: reliable
      durability: volatile
      history: keep_last
      depth: 10
  /waffle_waypoints:
    type: nav_msgs/msg/Path
    qos:
      reliability: reliable
      durability: volatile
      history: keep_last
      depth: 10
  /burger_waypoints:
    type: nav_msgs/msg/Path
    qos:
      reliability: reliable
      durability: volatile
      history: keep_last
      depth: 10
"""

        debug_yaml = f"""name: fleet_debug_{burger_domain}_to_{leader_domain}
from_domain: {burger_domain}
to_domain: {leader_domain}

topics:
  /burger_pose:
    type: geometry_msgs/msg/PoseStamped
    qos:
      reliability: reliable
      durability: volatile
      history: keep_last
      depth: 10
"""

        shared_path.write_text(shared_yaml)
        debug_path.write_text(debug_yaml)

        return [
            ExecuteProcess(
                cmd=['ros2', 'run', 'domain_bridge', 'domain_bridge', str(shared_path)],
                output='screen', name='fleet_shared_domain_bridge',
            ),
            ExecuteProcess(
                cmd=['ros2', 'run', 'domain_bridge', 'domain_bridge', str(debug_path)],
                output='screen', name='fleet_debug_domain_bridge',
            ),
        ]

    domain_bridges = OpaqueFunction(function=_write_bridge_configs)

    # ---- Sim-only nodes ---------------------------------------------------
    def _make_sim_nodes(context, *args, **kwargs):
        if sim.perform(context) != 'true':
            return []
        bridge = Node(
            package='ros_gz_bridge', executable='parameter_bridge',
            name='burger_ros_gz_bridge', output='screen',
            parameters=[{'config_file': bridge_config}],
        )
        converter = ExecuteProcess(
            cmd=[
                'python3', single_twist_script, '--ros-args',
                '-r', '__node:=burger_twist_stamped_to_twist',
                '-p', 'use_sim_time:=true',
                '-p', 'robot_name:=burger',
                '-p', 'cmd_vel_topic:=/cmd_vel_stamped',
                '-p', 'internal_cmd_vel_topics:=/cmd_vel,/gz_cmd_vel_unstamped',
                '-p', 'watchdog_timeout_sec:=0.5',
            ],
            output='screen', name='burger_twist_stamped_to_twist',
        )
        return [bridge, converter]

    def _make_real_twist(context, *args, **kwargs):
        if sim.perform(context) == 'true':
            return []
        converter = ExecuteProcess(
            cmd=[
                'python3', single_twist_script, '--ros-args',
                '-r', '__node:=burger_twist_stamped_to_twist',
                '-p', 'use_sim_time:=false',
                '-p', 'robot_name:=burger',
                '-p', 'cmd_vel_topic:=/cmd_vel_stamped',
                '-p', 'internal_cmd_vel_topics:=/cmd_vel',
                '-p', 'watchdog_timeout_sec:=0.5',
            ],
            output='screen', name='burger_twist_stamped_to_twist',
        )
        return [converter]

    # ---- Always-on nodes --------------------------------------------------
    frame_tools = ExecuteProcess(
        cmd=[
            'python3', frame_tools_script, '--ros-args',
            '-r', '__node:=burger_frame_tools',
            '-p', ['use_sim_time:=', use_sim_time_str],
            '-p', 'robot_name:=burger',
            '-p', ['initial_x:=', burger_x],
            '-p', ['initial_y:=', burger_y],
            '-p', ['initial_yaw:=', burger_yaw],
            '-p', 'reset_odom_origin_on_first_msg:=true',
            '-p', 'initial_pose_repeat_count:=40',
            '-p', 'initial_pose_period_sec:=0.25',
            '-p', 'scan_out:=/scan_nav',
        ],
        output='screen', name='burger_frame_tools',
    )

    map_odom = ExecuteProcess(
        cmd=[
            'python3', map_odom_script, '--ros-args',
            '-r', '__node:=burger_map_odom_localization',
            '-p', ['use_sim_time:=', use_sim_time_str],
            '-p', 'robot_name:=burger',
            '-p', 'odom_topic:=/odom_nav',
            '-p', 'map_frame:=map',
            '-p', 'odom_frame:=odom',
            '-p', 'base_frame:=base_footprint',
            '-p', ['initial_x:=', burger_x],
            '-p', ['initial_y:=', burger_y],
            '-p', ['initial_yaw:=', burger_yaw],
            '-p', 'relative_to_world_origin:=true',
            '-p', ['world_origin_x:=', map_origin_x],
            '-p', ['world_origin_y:=', map_origin_y],
            '-p', ['world_origin_yaw:=', map_origin_yaw],
            '-p', 'publish_rate_hz:=30.0',
            '-p', 'publish_amcl_pose:=true',
        ],
        output='screen', name='burger_map_odom_localization',
    )

    burger_pose = ExecuteProcess(
        cmd=[
            'python3', tf_pose_script, '--ros-args',
            '-r', '__node:=burger_pose_tf_publisher',
            '-p', ['use_sim_time:=', use_sim_time_str],
            '-p', 'target_frame:=map',
            '-p', 'source_frame:=base_footprint',
            '-p', 'output_topic:=/burger_pose',
            '-p', 'publish_rate_hz:=10.0',
            '-p', 'log_every_n:=100',
        ],
        output='screen', name='burger_pose_tf_publisher',
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
            '-r', '__node:=burger_through_poses',
            '-p', ['use_sim_time:=', use_sim_time_str],
            '-p', 'path_topic:=/burger_waypoints',
            '-p', 'action_name:=/navigate_through_poses',
            '-p', 'default_frame_id:=map',
            '-p', 'change_threshold_m:=0.25',
            '-p', 'min_resend_sec:=1.5',
        ],
        output='screen', name='burger_through_poses',
    )

    return LaunchDescription([
        DeclareLaunchArgument('sim', default_value='true',
                              description='true=Gazebo sim, false=real robot'),
        DeclareLaunchArgument('burger_x', default_value='0.58'),
        DeclareLaunchArgument('burger_y', default_value='3.49'),
        DeclareLaunchArgument('burger_yaw', default_value='0.0'),
        DeclareLaunchArgument('map_origin_x', default_value='1.38',
                              description='Waffle initial x (SLAM map origin)'),
        DeclareLaunchArgument('map_origin_y', default_value='3.49',
                              description='Waffle initial y (SLAM map origin)'),
        DeclareLaunchArgument('map_origin_yaw', default_value='0.0',
                              description='Waffle initial yaw (SLAM map origin)'),
        DeclareLaunchArgument('domain_id', default_value='26'),
        DeclareLaunchArgument('leader_domain_id', default_value='25'),
        SetEnvironmentVariable('ROS_DOMAIN_ID', domain_id),
        SetEnvironmentVariable('RMW_IMPLEMENTATION', 'rmw_fastrtps_cpp'),
        SetEnvironmentVariable('FASTDDS_BUILTIN_TRANSPORTS', 'UDPv4'),
        SetEnvironmentVariable('TURTLEBOT3_MODEL', 'burger'),
        LogInfo(msg='FLEET_BURGER_LAUNCH | unified sim/real burger launch'),
        LogInfo(msg=['SIM | ', sim, ' | domain_id=', domain_id,
                     ' leader_domain_id=', leader_domain_id]),
        TimerAction(period=0.5, actions=[domain_bridges]),
        TimerAction(period=1.5, actions=[OpaqueFunction(function=_make_sim_nodes)]),
        TimerAction(period=1.5, actions=[OpaqueFunction(function=_make_real_twist)]),
        TimerAction(period=4.0, actions=[frame_tools]),
        TimerAction(period=5.0, actions=[map_odom, burger_pose]),
        TimerAction(period=10.0, actions=[controller_server, planner_server,
                                          behavior_server, bt_navigator]),
        TimerAction(period=18.0, actions=[nav_lifecycle]),
        TimerAction(period=22.0, actions=[through_poses]),
    ])
