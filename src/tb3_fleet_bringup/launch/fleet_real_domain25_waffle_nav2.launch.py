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
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node
from nav2_common.launch import RewrittenYaml


def generate_launch_description():
    bringup_share = get_package_share_directory('tb3_fleet_bringup')

    domain_id   = LaunchConfiguration('domain_id')
    robot_model = LaunchConfiguration('robot_model')
    use_slam    = LaunchConfiguration('use_slam')
    initial_x   = LaunchConfiguration('initial_x')
    initial_y   = LaunchConfiguration('initial_y')
    initial_yaw = LaunchConfiguration('initial_yaw')
    robot1_ip   = LaunchConfiguration('robot1_ip')
    robot2_ip   = LaunchConfiguration('robot2_ip')

    burger_slam_yaml = os.path.join(bringup_share, 'config', 'domain26_burger_nav2_slam.yaml')
    burger_amcl_yaml = os.path.join(bringup_share, 'config', 'domain24_burger_nav2_amcl.yaml')
    waffle_yaml      = os.path.join(bringup_share, 'config', 'domain25_waffle_nav2.yaml')

    nav2_source = PythonExpression([
        "'", burger_amcl_yaml, "' if '", robot_model, "' == 'burger' and '",
        use_slam, "' == 'false' else ('",
        burger_slam_yaml, "' if '", robot_model, "' == 'burger' else '",
        waffle_yaml, "')",
    ])
    nav2_params = RewrittenYaml(
        source_file=nav2_source,
        param_rewrites={'use_sim_time': 'false', 'odom_topic': '/odom', 'topic': '/scan'},
        convert_types=True,
    )

    cartographer_config_dir = os.path.join(bringup_share, 'config')
    tf_pose_script    = os.path.join(bringup_share, 'scripts', 'tf_pose_publisher_direct_v44.py')
    goal_proxy_script = os.path.join(bringup_share, 'scripts', 'pose_to_nav2_action_direct_v41.py')

    # ── FastDDS XML with configurable IPs (written at launch-time) ─────────────
    def make_fastdds_env(context, *args, **kwargs):
        r1 = robot1_ip.perform(context)
        r2 = robot2_ip.perform(context)
        d  = domain_id.perform(context)
        xml = f"""<?xml version="1.0" encoding="UTF-8"?>
<profiles xmlns="http://www.eprosima.com/XMLSchemas/fastRTPS_Profiles">
  <participant profile_name="default_profile" is_default_profile="true">
    <rtps>
      <builtin>
        <initialPeersList>
          <locator><udpv4><address>{r1}</address></udpv4></locator>
          <locator><udpv4><address>{r2}</address></udpv4></locator>
        </initialPeersList>
      </builtin>
    </rtps>
  </participant>
</profiles>
"""
        xml_path = Path(tempfile.gettempdir()) / f'fastdds_fleet_d{d}.xml'
        xml_path.write_text(xml, encoding='utf-8')
        return [
            SetEnvironmentVariable('FASTDDS_DEFAULT_PROFILES_FILE', str(xml_path)),
            SetEnvironmentVariable('ROS_STATIC_PEERS', f'{r1};{r2}'),
        ]

    # ── Localization: Cartographer SLAM or AMCL ────────────────────────────────
    def make_localization(context, *args, **kwargs):
        slam_mode = use_slam.perform(context).lower() in ('true', '1', 'yes')
        d = domain_id.perform(context)

        extra_env = {
            'ROS_DOMAIN_ID': d,
            'RMW_IMPLEMENTATION': 'rmw_fastrtps_cpp',
            'FASTDDS_BUILTIN_TRANSPORTS': 'UDPv4',
            'ROS_AUTOMATIC_DISCOVERY_RANGE': 'SUBNET',
        }

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
                additional_env=extra_env,
            )
            occupancy_grid = Node(
                package='cartographer_ros',
                executable='cartographer_occupancy_grid_node',
                name='cartographer_occupancy_grid_node',
                output='screen',
                parameters=[{'use_sim_time': False}],
                arguments=['-resolution', '0.05', '-publish_period_sec', '1.0'],
                additional_env=extra_env,
            )
            return [TimerAction(period=0.5, actions=[cartographer, occupancy_grid])]

        ix   = float(initial_x.perform(context))
        iy   = float(initial_y.perform(context))
        iyaw = float(initial_yaw.perform(context))
        amcl_pose_overrides = {
            'amcl': {'ros__parameters': {
                'set_initial_pose': True,
                'initial_pose': {'x': ix, 'y': iy, 'z': 0.0, 'yaw': iyaw},
            }}
        }
        pose_yaml = Path(tempfile.gettempdir()) / 'leader_amcl_initial_pose.yaml'
        pose_yaml.write_text(yaml.dump(amcl_pose_overrides), encoding='utf-8')
        amcl = Node(
            package='nav2_amcl', executable='amcl', name='amcl',
            output='screen',
            parameters=[nav2_params, str(pose_yaml)],
            additional_env=extra_env,
        )
        lifecycle_loc = Node(
            package='nav2_lifecycle_manager', executable='lifecycle_manager',
            name='lifecycle_manager_localization', output='screen',
            parameters=[nav2_params],
            additional_env=extra_env,
        )
        return [
            TimerAction(period=0.5, actions=[amcl]),
            TimerAction(period=1.0, actions=[lifecycle_loc]),
        ]

    leader_pose = ExecuteProcess(
        cmd=[
            'python3', tf_pose_script, '--ros-args',
            '-r', '__node:=waffle_real_leader_pose_publisher',
            '-p', 'use_sim_time:=false',
            '-p', 'target_frame:=map',
            '-p', 'source_frame:=base_footprint',
            '-p', 'output_topic:=/leader_pose',
            '-p', 'publish_rate_hz:=10.0',
            '-p', 'log_every_n:=100',
        ],
        output='screen', name='waffle_real_leader_pose_publisher',
    )

    controller_server = Node(
        package='nav2_controller', executable='controller_server',
        name='controller_server', output='screen', parameters=[nav2_params],
    )
    planner_server = Node(
        package='nav2_planner', executable='planner_server',
        name='planner_server', output='screen', parameters=[nav2_params],
    )
    behavior_server = Node(
        package='nav2_behaviors', executable='behavior_server',
        name='behavior_server', output='screen', parameters=[nav2_params],
    )
    bt_navigator = Node(
        package='nav2_bt_navigator', executable='bt_navigator',
        name='bt_navigator', output='screen', parameters=[nav2_params],
    )
    lifecycle_nav = Node(
        package='nav2_lifecycle_manager', executable='lifecycle_manager',
        name='lifecycle_manager_navigation', output='screen', parameters=[nav2_params],
    )
    default_goal = ExecuteProcess(
        cmd=[
            'python3', goal_proxy_script, '--ros-args',
            '-r', '__node:=waffle_real_default_goal_to_nav2',
            '-p', 'use_sim_time:=false',
            '-p', 'goal_pose_topic:=/goal_pose',
            '-p', 'navigate_action:=/navigate_to_pose',
            '-p', 'default_frame_id:=map',
            '-p', 'cancel_previous_goal:=true',
        ],
        output='screen', name='waffle_real_default_goal_to_nav2',
    )
    named_goal = ExecuteProcess(
        cmd=[
            'python3', goal_proxy_script, '--ros-args',
            '-r', '__node:=waffle_real_named_goal_to_nav2',
            '-p', 'use_sim_time:=false',
            '-p', 'goal_pose_topic:=/waffle_goal_pose',
            '-p', 'navigate_action:=/navigate_to_pose',
            '-p', 'default_frame_id:=map',
            '-p', 'cancel_previous_goal:=true',
        ],
        output='screen', name='waffle_real_named_goal_to_nav2',
    )

    return LaunchDescription([
        DeclareLaunchArgument('domain_id',    default_value='25'),
        DeclareLaunchArgument('robot_model',  default_value='waffle_pi'),
        DeclareLaunchArgument('use_slam',     default_value='true'),
        DeclareLaunchArgument('initial_x',    default_value='1.05'),
        DeclareLaunchArgument('initial_y',    default_value='0.0'),
        DeclareLaunchArgument('initial_yaw',  default_value='0.0'),
        DeclareLaunchArgument('robot1_ip',    default_value='10.10.14.10'),
        DeclareLaunchArgument('robot2_ip',    default_value='10.10.14.14'),
        UnsetEnvironmentVariable('ROS_DISCOVERY_SERVER'),
        UnsetEnvironmentVariable('ROS_LOCALHOST_ONLY'),
        UnsetEnvironmentVariable('FASTRTPS_DEFAULT_PROFILES_FILE'),
        SetEnvironmentVariable('ROS_DOMAIN_ID',               domain_id),
        SetEnvironmentVariable('RMW_IMPLEMENTATION',          'rmw_fastrtps_cpp'),
        SetEnvironmentVariable('FASTDDS_BUILTIN_TRANSPORTS',  'UDPv4'),
        SetEnvironmentVariable('ROS_AUTOMATIC_DISCOVERY_RANGE', 'SUBNET'),
        SetEnvironmentVariable('TURTLEBOT3_MODEL',            robot_model),
        OpaqueFunction(function=make_fastdds_env),   # writes XML + sets env vars
        LogInfo(msg=['LEADER_D', domain_id, ' | model=', robot_model,
                     ' | use_slam=', use_slam,
                     ' | peers=', robot1_ip, ';', robot2_ip]),
        OpaqueFunction(function=make_localization),
        TimerAction(period=1.5, actions=[leader_pose]),
        TimerAction(period=2.0, actions=[controller_server, planner_server,
                                         behavior_server, bt_navigator]),
        TimerAction(period=5.0, actions=[lifecycle_nav]),
        TimerAction(period=7.0, actions=[default_goal, named_goal]),
    ])
