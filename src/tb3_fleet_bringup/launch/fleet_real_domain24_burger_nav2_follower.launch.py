#!/usr/bin/env python3

import os
import tempfile
from pathlib import Path

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

    domain_id = LaunchConfiguration('domain_id')
    leader_domain_id = LaunchConfiguration('leader_domain_id')
    robot_model = LaunchConfiguration('robot_model')
    follow_distance = LaunchConfiguration('follow_distance')
    start_following = LaunchConfiguration('start_following')
    enable_path_yield = LaunchConfiguration('enable_path_yield')
    path_block_distance = LaunchConfiguration('path_block_distance')
    yield_lateral_distance = LaunchConfiguration('yield_lateral_distance')

    nav2_source = os.path.join(
        bringup_share, 'config', 'domain26_burger_nav2_slam.yaml'
    )
    nav2_params = RewrittenYaml(
        source_file=nav2_source,
        param_rewrites={
            'use_sim_time': 'false',
            'odom_topic': '/odom',
            'topic': '/scan',
        },
        convert_types=True,
    )

    follower_script = os.path.join(
        bringup_share, 'scripts', 'domain_bridge_nav2_follower_direct_v40.py'
    )
    tf_pose_script = os.path.join(
        bringup_share, 'scripts', 'tf_pose_publisher_direct_v44.py'
    )
    goal_proxy_script = os.path.join(
        bringup_share, 'scripts', 'pose_to_nav2_action_direct_v41.py'
    )
    cartographer_config_dir = os.path.join(bringup_share, 'config')

    def make_domain_bridges(context, *args, **kwargs):
        leader_domain = leader_domain_id.perform(context)
        burger_domain = domain_id.perform(context)
        out_dir = Path(tempfile.gettempdir()) / 'tb3_fleet_real_domain_bridge'
        out_dir.mkdir(parents=True, exist_ok=True)

        leader_to_burger = out_dir / (
            f'real_leader_{leader_domain}_to_burger_{burger_domain}.yaml'
        )
        burger_to_leader = out_dir / (
            f'real_burger_{burger_domain}_to_leader_{leader_domain}.yaml'
        )

        leader_to_burger.write_text(
            f"""name: real_leader_{leader_domain}_to_burger_{burger_domain}
from_domain: {leader_domain}
to_domain: {burger_domain}

topics:
  /leader_pose:
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

        burger_to_leader.write_text(
            f"""name: real_burger_{burger_domain}_to_leader_{leader_domain}
from_domain: {burger_domain}
to_domain: {leader_domain}

topics:
  /map:
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
  /burger_pose:
    type: geometry_msgs/msg/PoseStamped
    qos:
      reliability: reliable
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
                cmd=[
                    'ros2', 'run', 'domain_bridge', 'domain_bridge',
                    str(leader_to_burger),
                ],
                output='screen',
                name='real_leader_to_burger_domain_bridge',
            ),
            ExecuteProcess(
                cmd=[
                    'ros2', 'run', 'domain_bridge', 'domain_bridge',
                    str(burger_to_leader),
                ],
                output='screen',
                name='real_burger_to_leader_domain_bridge',
            ),
        ]

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
    lifecycle = Node(
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
        DeclareLaunchArgument('domain_id', default_value='24'),
        DeclareLaunchArgument('leader_domain_id', default_value='25'),
        DeclareLaunchArgument('robot_model', default_value='burger'),
        DeclareLaunchArgument('follow_distance', default_value='1.05'),
        DeclareLaunchArgument(
            'start_following',
            default_value='true',
            description='true starts in FOLLOWING; false waits for FOLLOW/RESUME.',
        ),
        DeclareLaunchArgument('enable_path_yield', default_value='true'),
        DeclareLaunchArgument('path_block_distance', default_value='0.55'),
        DeclareLaunchArgument('yield_lateral_distance', default_value='0.75'),
        UnsetEnvironmentVariable('ROS_DISCOVERY_SERVER'),
        UnsetEnvironmentVariable('ROS_LOCALHOST_ONLY'),
        UnsetEnvironmentVariable('FASTRTPS_DEFAULT_PROFILES_FILE'),
        UnsetEnvironmentVariable('FASTDDS_DEFAULT_PROFILES_FILE'),
        SetEnvironmentVariable('ROS_DOMAIN_ID', domain_id),
        SetEnvironmentVariable('RMW_IMPLEMENTATION', 'rmw_fastrtps_cpp'),
        SetEnvironmentVariable('FASTDDS_BUILTIN_TRANSPORTS', 'UDPv4'),
        SetEnvironmentVariable('ROS_AUTOMATIC_DISCOVERY_RANGE', 'SUBNET'),
        SetEnvironmentVariable('TURTLEBOT3_MODEL', robot_model),
        LogInfo(
            msg=[
                'REAL_BURGER_DOMAIN24 | hardware /odom,/scan,/cmd_vel | '
                'Cartographer map owner | start_following=', start_following,
            ]
        ),
        TimerAction(
            period=0.5,
            actions=[OpaqueFunction(function=make_domain_bridges)],
        ),
        TimerAction(period=2.0, actions=[cartographer, occupancy_grid]),
        TimerAction(period=3.0, actions=[burger_pose]),
        TimerAction(
            period=4.0,
            actions=[controller_server, planner_server, behavior_server, bt_navigator],
        ),
        TimerAction(period=7.0, actions=[lifecycle]),
        TimerAction(period=9.0, actions=[burger_named_goal]),
        TimerAction(period=10.0, actions=[follower]),
    ])
