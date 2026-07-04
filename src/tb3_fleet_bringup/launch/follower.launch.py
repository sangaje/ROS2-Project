#!/usr/bin/env python3
"""
Follower — D24 Burger.
  mode:=real  Hardware bringup + domain_bridge + AMCL + Nav2 + follower script
  mode:=sim   Gazebo simulation — domain_bridge + relays + AMCL + Nav2 + follower script
"""
import os
import tempfile
from pathlib import Path

import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument, ExecuteProcess, IncludeLaunchDescription,
    OpaqueFunction, SetEnvironmentVariable, TimerAction,
    UnsetEnvironmentVariable,
)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node
from nav2_common.launch import RewrittenYaml


def generate_launch_description():
    pkg = get_package_share_directory('tb3_fleet_bringup')

    mode               = LaunchConfiguration('mode')
    domain_id          = LaunchConfiguration('domain_id')
    main_domain_id     = LaunchConfiguration('main_domain_id')
    follow_distance    = LaunchConfiguration('follow_distance')
    start_following    = LaunchConfiguration('start_following')
    enable_path_yield  = LaunchConfiguration('enable_path_yield')
    path_block_dist    = LaunchConfiguration('path_block_distance')
    yield_lateral_dist = LaunchConfiguration('yield_lateral_distance')
    initial_x          = LaunchConfiguration('follower_initial_x')
    initial_y          = LaunchConfiguration('follower_initial_y')
    initial_yaw        = LaunchConfiguration('follower_initial_yaw')
    def make_all(context, *args, **kwargs):
        sim = mode.perform(context).lower() == 'sim'
        ust = 'true' if sim else 'false'
        d  = domain_id.perform(context)
        ld = main_domain_id.perform(context)
        extra = {
            'ROS_DOMAIN_ID': d,
            'ROS_AUTOMATIC_DISCOVERY_RANGE': 'SUBNET',
            'ROS_LOCALHOST_ONLY': '0',
            'RMW_IMPLEMENTATION': 'rmw_fastrtps_cpp',
        }

        # Nav2 + AMCL params
        if sim:
            nav2_params = os.path.join(pkg, 'config', 'domain24_burger_nav2_amcl_sim.yaml')
        else:
            nav2_params = RewrittenYaml(
                source_file=os.path.join(pkg, 'config', 'domain24_burger_nav2_amcl.yaml'),
                param_rewrites={
                    'use_sim_time': 'false',
                    'odom_topic': '/odom',
                    'scan_topic': '/scan',
                    'topic': '/scan',
                    'enable_stamped_cmd_vel': 'true',
                },
                convert_types=True,
            )

        # Domain bridge YAMLs
        out = Path(tempfile.gettempdir()) / 'tb3_fleet_domain_bridge'
        out.mkdir(parents=True, exist_ok=True)
        l2b = out / f'l2b_{ld}_to_{d}.yaml'
        b2l = out / f'b2l_{d}_to_{ld}.yaml'

        if sim:
            l2b.write_text(f"""\
name: sim_{ld}_to_{d}
from_domain: {ld}
to_domain: {d}
topics:
  /clock:
    type: rosgraph_msgs/msg/Clock
    qos: {{reliability: best_effort, durability: volatile, history: keep_last, depth: 10}}
  /map:
    type: nav_msgs/msg/OccupancyGrid
    remap: /map_bridge
    qos: {{reliability: reliable, durability: volatile, history: keep_last, depth: 5}}
  /leader_pose:
    type: geometry_msgs/msg/PoseStamped
    qos: {{reliability: reliable, durability: volatile, history: keep_last, depth: 10}}
  /plan:
    type: nav_msgs/msg/Path
    remap: /waffle_plan
    qos: {{reliability: reliable, durability: volatile, history: keep_last, depth: 10}}
  /burger_goal_pose:
    type: geometry_msgs/msg/PoseStamped
    qos: {{reliability: reliable, durability: volatile, history: keep_last, depth: 10}}
  /fleet/follow_command:
    type: std_msgs/msg/String
    qos: {{reliability: reliable, durability: volatile, history: keep_last, depth: 10}}
  /burger/scan:
    type: sensor_msgs/msg/LaserScan
    remap: /scan_bridge
    qos: {{reliability: best_effort, durability: volatile, history: keep_last, depth: 10}}
  /burger/odom:
    type: nav_msgs/msg/Odometry
    remap: /odom_bridge
    qos: {{reliability: reliable, durability: volatile, history: keep_last, depth: 10}}
  /burger/joint_states:
    type: sensor_msgs/msg/JointState
    remap: /joint_states
    qos: {{reliability: reliable, durability: volatile, history: keep_last, depth: 10}}
  /burger/tf:
    type: tf2_msgs/msg/TFMessage
    qos: {{reliability: reliable, durability: volatile, history: keep_last, depth: 100}}
""", encoding='utf-8')
            b2l.write_text(f"""\
name: sim_{d}_to_{ld}
from_domain: {d}
to_domain: {ld}
topics:
  /burger_pose:
    type: geometry_msgs/msg/PoseStamped
    qos: {{reliability: reliable, durability: volatile, history: keep_last, depth: 10}}
  /plan:
    type: nav_msgs/msg/Path
    remap: /burger_plan
    qos: {{reliability: reliable, durability: volatile, history: keep_last, depth: 10}}
  /fleet/follow_enabled:
    type: std_msgs/msg/Bool
    qos: {{reliability: reliable, durability: transient_local, history: keep_last, depth: 1}}
  /cmd_vel:
    type: geometry_msgs/msg/TwistStamped
    remap: /burger/cmd_vel
    qos: {{reliability: reliable, durability: volatile, history: keep_last, depth: 10}}
  /burger_scan_relay:
    type: sensor_msgs/msg/LaserScan
    remap: /burger_scan
    qos: {{reliability: best_effort, durability: volatile, history: keep_last, depth: 10}}
""", encoding='utf-8')
        else:
            l2b.write_text(f"""\
name: l2b_{ld}_to_{d}
from_domain: {ld}
to_domain: {d}
topics:
  /map:
    type: nav_msgs/msg/OccupancyGrid
    remap: /map_bridge
    qos: {{reliability: reliable, durability: volatile, history: keep_last, depth: 5}}
  /leader_pose:
    type: geometry_msgs/msg/PoseStamped
    qos: {{reliability: reliable, durability: volatile, history: keep_last, depth: 10}}
  /plan:
    type: nav_msgs/msg/Path
    remap: /waffle_plan
    qos: {{reliability: reliable, durability: volatile, history: keep_last, depth: 10}}
  /burger_goal_pose:
    type: geometry_msgs/msg/PoseStamped
    qos: {{reliability: reliable, durability: volatile, history: keep_last, depth: 10}}
  /fleet/follow_command:
    type: std_msgs/msg/String
    qos: {{reliability: reliable, durability: volatile, history: keep_last, depth: 10}}
""", encoding='utf-8')
            b2l.write_text(f"""\
name: b2l_{d}_to_{ld}
from_domain: {d}
to_domain: {ld}
topics:
  /burger_pose:
    type: geometry_msgs/msg/PoseStamped
    qos: {{reliability: reliable, durability: volatile, history: keep_last, depth: 10}}
  /burger_scan_relay:
    type: sensor_msgs/msg/LaserScan
    remap: /burger_scan
    qos: {{reliability: best_effort, durability: volatile, history: keep_last, depth: 10}}
  /plan:
    type: nav_msgs/msg/Path
    remap: /burger_plan
    qos: {{reliability: reliable, durability: volatile, history: keep_last, depth: 10}}
  /fleet/follow_enabled:
    type: std_msgs/msg/Bool
    qos: {{reliability: reliable, durability: transient_local, history: keep_last, depth: 1}}
""", encoding='utf-8')

        bridges = [
            ExecuteProcess(cmd=['ros2', 'run', 'domain_bridge', 'domain_bridge', str(l2b)],
                           output='screen', name='bridge_l2b'),
            ExecuteProcess(cmd=['ros2', 'run', 'domain_bridge', 'domain_bridge', str(b2l)],
                           output='screen', name='bridge_b2l'),
        ]

        map_relay = ExecuteProcess(
            cmd=['python3', os.path.join(pkg, 'scripts', 'sim_map_relay.py'),
                 '--ros-args', '-r', '__node:=follower_map_relay',
                 '-p', f'use_sim_time:={ust}',
                 '-p', 'input_topic:=/map_bridge', '-p', 'output_topic:=/map'],
            output='screen', name='follower_map_relay', additional_env=extra,
        )
        burger_pose = ExecuteProcess(
            cmd=['python3', os.path.join(pkg, 'scripts', 'tf_pose_publisher_direct_v44.py'),
                 '--ros-args', '-r', '__node:=burger_pose_pub',
                 '-p', f'use_sim_time:={ust}',
                 '-p', 'target_frame:=map', '-p', 'source_frame:=base_footprint',
                 '-p', 'output_topic:=/burger_pose',
                 '-p', 'publish_rate_hz:=10.0', '-p', 'log_every_n:=100'],
            output='screen', name='burger_pose_pub', additional_env=extra,
        )

        if sim:
            tb3_gz_share = get_package_share_directory('turtlebot3_gazebo')
            with open(os.path.join(tb3_gz_share, 'urdf', 'turtlebot3_burger.urdf')) as f:
                robot_desc = f.read()
            rsp = Node(
                package='robot_state_publisher', executable='robot_state_publisher',
                name='robot_state_publisher', output='screen',
                parameters=[{'use_sim_time': True, 'robot_description': robot_desc}],
                additional_env=extra,
            )
            tf_relay = ExecuteProcess(
                cmd=['python3', os.path.join(pkg, 'scripts', 'sim_burger_tf_relay.py'),
                     '--ros-args', '-r', '__node:=follower_tf_relay',
                     '-p', 'use_sim_time:=true'],
                output='screen', name='follower_tf_relay', additional_env=extra,
            )
            scan_relay = ExecuteProcess(
                cmd=['python3', os.path.join(pkg, 'scripts', 'sim_burger_scan_relay.py'),
                     '--ros-args', '-r', '__node:=follower_scan_relay',
                     '-p', 'use_sim_time:=true',
                     '-p', 'scan_input_topic:=/scan_bridge',
                     '-p', 'scan_output_topic:=/scan',
                     '-p', 'burger_scan_output_topic:=/burger_scan_relay',
                     '-p', 'odom_input_topic:=/odom_bridge',
                     '-p', 'odom_output_topic:=/odom'],
                output='screen', name='follower_scan_relay', additional_env=extra,
            )
            relay_nodes = [tf_relay, scan_relay, map_relay]
        else:
            rsp = None
            scan_relay = ExecuteProcess(
                cmd=['python3', os.path.join(pkg, 'scripts', 'scan_frame_relay.py'),
                     '--ros-args', '-r', '__node:=burger_scan_relay',
                     '-p', 'input_topic:=/scan',
                     '-p', 'output_topic:=/burger_scan_relay',
                     '-p', 'output_frame:=burger/base_scan',
                     '-p', 'input_reliability:=best_effort',
                     '-p', 'output_reliability:=reliable'],
                output='screen', name='burger_scan_relay', additional_env=extra,
            )
            relay_nodes = [scan_relay, map_relay]

        # AMCL
        ix, iy, iyaw = (float(x.perform(context)) for x in [initial_x, initial_y, initial_yaw])
        pose_yaml = Path(tempfile.gettempdir()) / 'burger_amcl_pose.yaml'
        pose_yaml.write_text(yaml.dump({'amcl': {'ros__parameters': {
            'set_initial_pose': True,
            'initial_pose': {'x': ix, 'y': iy, 'z': 0.0, 'yaw': iyaw},
        }}}), encoding='utf-8')
        amcl_kwargs = dict(respawn=True, respawn_delay=3.0) if sim else {}
        amcl = Node(package='nav2_amcl', executable='amcl', name='amcl',
                    output='screen', parameters=[nav2_params, str(pose_yaml)],
                    additional_env=extra, **amcl_kwargs)
        lc_loc = Node(package='nav2_lifecycle_manager', executable='lifecycle_manager',
                      name='lifecycle_manager_localization', output='screen',
                      parameters=[nav2_params], additional_env=extra)

        controller  = Node(package='nav2_controller', executable='controller_server',
                           name='controller_server', output='screen',
                           parameters=[nav2_params], additional_env=extra)
        planner     = Node(package='nav2_planner', executable='planner_server',
                           name='planner_server', output='screen',
                           parameters=[nav2_params], additional_env=extra)
        behaviors   = Node(package='nav2_behaviors', executable='behavior_server',
                           name='behavior_server', output='screen',
                           parameters=[nav2_params], additional_env=extra)
        bt_nav      = Node(package='nav2_bt_navigator', executable='bt_navigator',
                           name='bt_navigator', output='screen',
                           parameters=[nav2_params], additional_env=extra)
        lifecycle_nav = Node(package='nav2_lifecycle_manager', executable='lifecycle_manager',
                             name='lifecycle_manager_navigation', output='screen',
                             parameters=[nav2_params], additional_env=extra)
        burger_goal = ExecuteProcess(
            cmd=['python3', os.path.join(pkg, 'scripts', 'pose_to_nav2_action_direct_v41.py'),
                 '--ros-args', '-r', '__node:=burger_named_goal',
                 '-p', f'use_sim_time:={ust}',
                 '-p', 'goal_pose_topic:=/burger_goal_pose',
                 '-p', 'navigate_action:=/navigate_to_pose',
                 '-p', 'default_frame_id:=map', '-p', 'cancel_previous_goal:=true'],
            output='screen', name='burger_named_goal', additional_env=extra,
        )

        # Follower behavior script
        follower_cmd = [
            'python3', os.path.join(pkg, 'scripts', 'domain_bridge_nav2_follower_direct_v40.py'),
            '--ros-args', '-r', '__node:=fleet_follower',
            '-p', f'use_sim_time:={ust}',
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
        ]
        if sim:
            follower_cmd += ['-p', 'enable_path_yield:=false', '-p', 'path_block_distance:=0.55']
        else:
            follower_cmd += [
                '-p', ['enable_path_yield:=', enable_path_yield],
                '-p', ['path_block_distance:=', path_block_dist],
                '-p', 'path_lookahead_min:=0.30', '-p', 'path_lookahead_max:=2.50',
                '-p', ['yield_lateral_distance:=', yield_lateral_dist],
                '-p', 'yield_release_distance:=0.80',
                '-p', 'yield_map_clearance:=0.18',
                '-p', 'yield_min_hold_sec:=4.0',
                '-p', 'yield_max_hold_sec:=12.0',
            ]
        follower_proc = ExecuteProcess(
            cmd=follower_cmd, output='screen', name='fleet_follower', additional_env=extra,
        )

        if sim:
            return [rsp,
                    TimerAction(period=0.5, actions=bridges),
                    TimerAction(period=1.0, actions=relay_nodes),
                    TimerAction(period=5.0, actions=[amcl]),
                    TimerAction(period=5.5, actions=[lc_loc]),
                    TimerAction(period=7.0, actions=[controller, planner, behaviors, bt_nav]),
                    TimerAction(period=10.0, actions=[lifecycle_nav]),
                    TimerAction(period=12.0, actions=[burger_pose]),
                    TimerAction(period=13.0, actions=[burger_goal, follower_proc])]
        else:
            return [TimerAction(period=0.5, actions=bridges + [scan_relay]),
                    TimerAction(period=1.0, actions=[map_relay, burger_pose]),
                    TimerAction(period=2.0, actions=[amcl]),
                    TimerAction(period=2.5, actions=[lc_loc]),
                    TimerAction(period=5.0, actions=[controller, planner, behaviors, bt_nav]),
                    TimerAction(period=9.0, actions=[lifecycle_nav]),
                    TimerAction(period=11.0, actions=[burger_goal, follower_proc])]

    return LaunchDescription([
        DeclareLaunchArgument('mode',               default_value='real',
                              description='real = physical robot | sim = Gazebo'),
        DeclareLaunchArgument('domain_id',          default_value='24'),
        DeclareLaunchArgument('main_domain_id',     default_value='25'),
        DeclareLaunchArgument('follow_distance',    default_value='1.05'),
        DeclareLaunchArgument('start_following',    default_value='false'),
        DeclareLaunchArgument('enable_path_yield',  default_value='true'),
        DeclareLaunchArgument('path_block_distance', default_value='0.55'),
        DeclareLaunchArgument('yield_lateral_distance', default_value='0.75'),
        DeclareLaunchArgument('follower_initial_x', default_value='-1.0'),
        DeclareLaunchArgument('follower_initial_y', default_value='0.0'),
        DeclareLaunchArgument('follower_initial_yaw', default_value='0.0'),
        UnsetEnvironmentVariable('ROS_DISCOVERY_SERVER'),
        UnsetEnvironmentVariable('ROS_LOCALHOST_ONLY'),
        UnsetEnvironmentVariable('FASTRTPS_DEFAULT_PROFILES_FILE'),
        UnsetEnvironmentVariable('FASTDDS_DEFAULT_PROFILES_FILE'),
        SetEnvironmentVariable('ROS_DOMAIN_ID',                domain_id),
        SetEnvironmentVariable('ROS_AUTOMATIC_DISCOVERY_RANGE', 'SUBNET'),
        SetEnvironmentVariable('ROS_LOCALHOST_ONLY',           '0'),
        SetEnvironmentVariable('RMW_IMPLEMENTATION',           'rmw_fastrtps_cpp'),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(os.path.join(
                get_package_share_directory('turtlebot3_bringup'), 'launch', 'robot.launch.py')),
            condition=IfCondition(PythonExpression(["'", mode, "'.lower() != 'sim'"])),
        ),

        OpaqueFunction(function=make_all),
    ])
