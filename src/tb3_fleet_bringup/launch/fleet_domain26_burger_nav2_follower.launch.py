#!/usr/bin/env python3

import os
import tempfile
from pathlib import Path
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, TimerAction, LogInfo, SetEnvironmentVariable, OpaqueFunction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution, PythonExpression
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    bringup_share = get_package_share_directory('tb3_fleet_bringup')

    burger_x = LaunchConfiguration('burger_x')
    burger_y = LaunchConfiguration('burger_y')
    burger_yaw = LaunchConfiguration('burger_yaw')
    map_origin_x = LaunchConfiguration('map_origin_x')
    map_origin_y = LaunchConfiguration('map_origin_y')
    map_origin_yaw = LaunchConfiguration('map_origin_yaw')
    auto_follow = LaunchConfiguration('auto_follow')
    control_mode = LaunchConfiguration('control_mode')
    astar_target_mode = LaunchConfiguration('astar_target_mode')
    domain_id = LaunchConfiguration('domain_id')
    leader_domain_id = LaunchConfiguration('leader_domain_id')

    bridge_config = os.path.join(bringup_share, 'config', 'domain26_burger_ros_gz_bridge.yaml')
    nav2_params = os.path.join(bringup_share, 'config', 'domain26_burger_nav2_slam.yaml')
    single_twist_script = os.path.join(bringup_share, 'scripts', 'single_twist_stamped_to_twist_direct_v36.py')
    frame_tools_script = os.path.join(bringup_share, 'scripts', 'single_domain_nav2_frame_tools_direct_v40.py')
    follower_script = os.path.join(bringup_share, 'scripts', 'domain_bridge_nav2_follower_direct_v40.py')
    map_odom_localization_script = os.path.join(bringup_share, 'scripts', 'map_odom_localization_direct_v44.py')
    tf_pose_script = os.path.join(bringup_share, 'scripts', 'tf_pose_publisher_direct_v44.py')
    goal_proxy_script = os.path.join(bringup_share, 'scripts', 'pose_to_nav2_action_direct_v41.py')
    astar_cmd_script = os.path.join(bringup_share, 'scripts', 'astar_cmd_vel_follower_direct_v55.py')

    def _write_domain_bridge_configs(context, *args, **kwargs):
        leader_domain = leader_domain_id.perform(context)
        burger_domain = domain_id.perform(context)
        out_dir = Path(tempfile.gettempdir()) / 'tb3_fleet_v55_domain_bridge'
        out_dir.mkdir(parents=True, exist_ok=True)

        shared_path = out_dir / f'shared_slam_map_leader_goal_{leader_domain}_to_{burger_domain}.yaml'
        debug_path = out_dir / f'burger_debug_{burger_domain}_to_{leader_domain}.yaml'

        shared_yaml = f"""name: shared_slam_map_leader_goal_{leader_domain}_to_{burger_domain}_v55
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
  /burger_goal_pose:
    type: geometry_msgs/msg/PoseStamped
    qos:
      reliability: reliable
      durability: volatile
      history: keep_last
      depth: 10
"""

        debug_yaml = f"""name: burger_debug_{burger_domain}_to_{leader_domain}_v55
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
  /astar_path:
    type: nav_msgs/msg/Path
    qos:
      reliability: reliable
      durability: volatile
      history: keep_last
      depth: 10
"""

        shared_path.write_text(shared_yaml)
        debug_path.write_text(debug_yaml)

        return [
            ExecuteProcess(cmd=['ros2', 'run', 'domain_bridge', 'domain_bridge', str(shared_path)], output='screen', name='shared_slam_map_leader_goal_domain_bridge_v55'),
            ExecuteProcess(cmd=['ros2', 'run', 'domain_bridge', 'domain_bridge', str(debug_path)], output='screen', name='burger_debug_domain_bridge_v55'),
        ]

    domain_bridges = OpaqueFunction(function=_write_domain_bridge_configs)

    converter = ExecuteProcess(cmd=['python3', single_twist_script, '--ros-args', '-r', '__node:=single_twist_stamped_to_twist_bridge', '-p', 'use_sim_time:=true', '-p', 'robot_name:=burger', '-p', 'cmd_vel_topic:=/cmd_vel', '-p', 'internal_cmd_vel_topics:=/gz_cmd_vel_unstamped,/gz_cmd_vel_model_unstamped', '-p', 'cmd_republish_rate_hz:=0.0', '-p', 'watchdog_timeout_sec:=0.5', '-p', 'log_every_n_republish:=100'], output='screen', name='burger_twist_stamped_to_twist_v55')

    bridge = Node(package='ros_gz_bridge', executable='parameter_bridge', name='burger_ros_gz_bridge_domain_burger', output='screen', parameters=[{'config_file': bridge_config}])

    frame_tools = ExecuteProcess(cmd=['python3', frame_tools_script, '--ros-args', '-r', '__node:=single_domain_nav2_frame_tools', '-p', 'use_sim_time:=true', '-p', 'robot_name:=burger', '-p', ['initial_x:=', burger_x], '-p', ['initial_y:=', burger_y], '-p', ['initial_yaw:=', burger_yaw], '-p', 'reset_odom_origin_on_first_msg:=true', '-p', 'initial_pose_repeat_count:=40', '-p', 'initial_pose_period_sec:=0.25'], output='screen', name='burger_frame_tools_v55')

    map_odom_localization = ExecuteProcess(cmd=['python3', map_odom_localization_script, '--ros-args', '-r', '__node:=map_odom_localization', '-p', 'use_sim_time:=true', '-p', 'robot_name:=burger', '-p', 'odom_topic:=/odom_nav', '-p', 'map_frame:=map', '-p', 'odom_frame:=odom', '-p', 'base_frame:=base_footprint', '-p', ['initial_x:=', burger_x], '-p', ['initial_y:=', burger_y], '-p', ['initial_yaw:=', burger_yaw], '-p', 'relative_to_world_origin:=true', '-p', ['world_origin_x:=', map_origin_x], '-p', ['world_origin_y:=', map_origin_y], '-p', ['world_origin_yaw:=', map_origin_yaw], '-p', 'publish_rate_hz:=30.0', '-p', 'publish_amcl_pose:=true'], output='screen', name='burger_map_odom_localization_v55')

    burger_pose = ExecuteProcess(cmd=['python3', tf_pose_script, '--ros-args', '-r', '__node:=burger_pose_tf_publisher', '-p', 'use_sim_time:=true', '-p', 'target_frame:=map', '-p', 'source_frame:=base_footprint', '-p', 'output_topic:=/burger_pose', '-p', 'publish_rate_hz:=10.0', '-p', 'log_every_n:=100'], output='screen', name='burger_pose_tf_publisher_v55')

    controller_server = Node(package='nav2_controller', executable='controller_server', name='controller_server', output='screen', parameters=[nav2_params])
    planner_server = Node(package='nav2_planner', executable='planner_server', name='planner_server', output='screen', parameters=[nav2_params])
    behavior_server = Node(package='nav2_behaviors', executable='behavior_server', name='behavior_server', output='screen', parameters=[nav2_params])
    bt_navigator = Node(package='nav2_bt_navigator', executable='bt_navigator', name='bt_navigator', output='screen', parameters=[nav2_params])
    nav_lifecycle = Node(package='nav2_lifecycle_manager', executable='lifecycle_manager', name='lifecycle_manager_navigation', output='screen', parameters=[nav2_params])

    goal_proxy_default = ExecuteProcess(cmd=['python3', goal_proxy_script, '--ros-args', '-r', '__node:=burger_default_goal_pose_to_nav2', '-p', 'use_sim_time:=true', '-p', 'goal_pose_topic:=/goal_pose', '-p', 'navigate_action:=/navigate_to_pose', '-p', 'default_frame_id:=map', '-p', 'cancel_previous_goal:=true'], output='screen', name='burger_default_goal_pose_to_nav2_v55')
    goal_proxy_named = ExecuteProcess(cmd=['python3', goal_proxy_script, '--ros-args', '-r', '__node:=burger_named_goal_pose_to_nav2', '-p', 'use_sim_time:=true', '-p', 'goal_pose_topic:=/burger_goal_pose', '-p', 'navigate_action:=/navigate_to_pose', '-p', 'default_frame_id:=map', '-p', 'cancel_previous_goal:=true'], output='screen', name='burger_named_goal_pose_to_nav2_v55')

    follower = ExecuteProcess(cmd=['python3', follower_script, '--ros-args', '-r', '__node:=domain_bridge_nav2_follower', '-p', 'use_sim_time:=true', '-p', 'leader_pose_topic:=/leader_pose', '-p', 'navigate_action:=/navigate_to_pose', '-p', 'follow_distance:=1.05', '-p', 'goal_period_sec:=1.5', '-p', 'goal_update_distance:=0.25', '-p', 'cancel_previous_goal:=false'], output='screen', name='domain_bridge_nav2_follower_v55')

    astar_cmd = ExecuteProcess(cmd=[
        'python3', astar_cmd_script,
        '--ros-args',
        '-r', '__node:=burger_astar_cmd_vel_follower',
        '-p', 'use_sim_time:=true',
        '-p', 'robot_name:=burger',
        '-p', 'map_topic:=/map',
        '-p', 'robot_pose_topic:=/burger_pose',
        '-p', 'leader_pose_topic:=/leader_pose',
        '-p', 'manual_goal_topic:=/burger_goal_pose',
        '-p', 'scan_topic:=/scan_nav',
        '-p', 'cmd_vel_topic:=/cmd_vel',
        '-p', 'path_topic:=/astar_path',
        '-p', ['target_mode:=', astar_target_mode],
        '-p', 'leader_goal_mode:=line_between_robots',
        '-p', 'follow_distance:=0.85',
        '-p', 'goal_tolerance:=0.16',
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
        '-p', 'stale_goal_sec:=5.0',
        '-p', 'direct_fallback_if_no_path:=false',
        '-p', 'log_period_sec:=1.0',
    ], output='screen', name='burger_astar_cmd_vel_follower_v55')

    is_nav2_control = IfCondition(PythonExpression(["'", control_mode, "' == 'nav2'"]))
    is_astar_control = IfCondition(PythonExpression(["'", control_mode, "' == 'astar_cmd'"]))

    return LaunchDescription([
        DeclareLaunchArgument('domain_id', default_value='24', description='ROS_DOMAIN_ID used by Burger follower domain.'),
        DeclareLaunchArgument('leader_domain_id', default_value='25', description='ROS_DOMAIN_ID of Waffle/SLAM owner domain.'),
        DeclareLaunchArgument('burger_x', default_value='-3.20'),
        DeclareLaunchArgument('burger_y', default_value='-1.75'),
        DeclareLaunchArgument('burger_yaw', default_value='0.0'),
        DeclareLaunchArgument('map_origin_x', default_value='-2.25', description='Map frame origin x. For Cartographer SLAM: set to waffle initial x so burger coords are relative to the SLAM map origin. For static pre-built map: use 0.0 (Gazebo absolute coords).'),
        DeclareLaunchArgument('map_origin_y', default_value='-1.75', description='Map frame origin y. For Cartographer SLAM: set to waffle initial y. For static pre-built map: use 0.0.'),
        DeclareLaunchArgument('map_origin_yaw', default_value='0.0', description='Map frame origin yaw.'),
        DeclareLaunchArgument('auto_follow', default_value='true', description='Nav2 mode only: true means Burger follows /leader_pose with NavigateToPose goals.'),
        DeclareLaunchArgument('control_mode', default_value='astar_cmd', description='nav2: use Nav2 NavigateToPose. astar_cmd: bypass Nav2 and publish /cmd_vel TwistStamped from A* path tracking.'),
        DeclareLaunchArgument('astar_target_mode', default_value='leader', description='astar_cmd mode: leader follows /leader_pose; manual follows /burger_goal_pose.'),
        SetEnvironmentVariable('ROS_DOMAIN_ID', domain_id),
        SetEnvironmentVariable('RMW_IMPLEMENTATION', 'rmw_fastrtps_cpp'),
        SetEnvironmentVariable('FASTDDS_BUILTIN_TRANSPORTS', 'UDPv4'),
        SetEnvironmentVariable('ROS_AUTOMATIC_DISCOVERY_RANGE', 'LOCALHOST'),
        SetEnvironmentVariable('TURTLEBOT3_MODEL', 'burger'),
        LogInfo(msg='V55_DOMAIN24_BURGER | Burger Nav2 receives live SLAM /map and /leader_pose from Domain25. Burger map frame is relative to Waffle SLAM origin.'),
        LogInfo(msg=['V55_BRIDGE_DOMAINS | leader_domain_id=', leader_domain_id, ' -> burger_domain_id=', domain_id, ' and debug back.']),
        LogInfo(msg='V55_SAFE_HOUSE_SPAWN | default Burger=(-3.20,-1.75,0.0), map_origin=(-2.25,-1.75,0.0). Keep map_origin equal to Waffle initial pose.'),
        LogInfo(msg=['V55_BURGER_CONTROL_MODE | ', control_mode, ' | nav2 or astar_cmd']),
        TimerAction(period=0.5, actions=[domain_bridges]),
        TimerAction(period=1.5, actions=[converter, bridge]),
        TimerAction(period=4.0, actions=[frame_tools]),
        TimerAction(period=5.0, actions=[map_odom_localization, burger_pose]),
        TimerAction(period=10.0, actions=[controller_server, planner_server, behavior_server, bt_navigator], condition=is_nav2_control),
        TimerAction(period=18.0, actions=[nav_lifecycle], condition=is_nav2_control),
        TimerAction(period=20.0, actions=[goal_proxy_default, goal_proxy_named], condition=is_nav2_control),
        TimerAction(period=30.0, actions=[follower], condition=IfCondition(PythonExpression(["'", control_mode, "' == 'nav2' and '", auto_follow, "' == 'true'"]))),
        TimerAction(period=9.0, actions=[astar_cmd], condition=is_astar_control),
    ])
