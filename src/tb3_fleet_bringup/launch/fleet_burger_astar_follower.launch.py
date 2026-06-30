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
    astar_cmd_script = os.path.join(bringup_share, 'scripts', 'astar_cmd_vel_follower_direct_v61.py')
    odom_pose_fallback_script = os.path.join(bringup_share, 'scripts', 'odom_pose_publisher_direct_v59.py')

    def _write_domain_bridge_configs(context, *args, **kwargs):
        leader_domain = leader_domain_id.perform(context)
        burger_domain = domain_id.perform(context)
        out_dir = Path(tempfile.gettempdir()) / 'tb3_fleet_v63_domain_bridge'
        out_dir.mkdir(parents=True, exist_ok=True)

        shared_path = out_dir / f'shared_slam_map_leader_goal_{leader_domain}_to_{burger_domain}.yaml'
        debug_path = out_dir / f'burger_debug_{burger_domain}_to_{leader_domain}.yaml'

        shared_yaml = f"""name: shared_slam_map_leader_goal_{leader_domain}_to_{burger_domain}_v63
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
  /goal_pose:
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

        debug_yaml = f"""name: burger_debug_{burger_domain}_to_{leader_domain}_v63
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
            ExecuteProcess(cmd=['ros2', 'run', 'domain_bridge', 'domain_bridge', str(shared_path)], output='screen', name='shared_slam_map_leader_goal_domain_bridge_v63'),
            ExecuteProcess(cmd=['ros2', 'run', 'domain_bridge', 'domain_bridge', str(debug_path)], output='screen', name='burger_debug_domain_bridge_v63'),
        ]

    domain_bridges = OpaqueFunction(function=_write_domain_bridge_configs)

    converter = ExecuteProcess(cmd=['python3', single_twist_script, '--ros-args', '-r', '__node:=single_twist_stamped_to_twist_bridge', '-p', 'use_sim_time:=true', '-p', 'robot_name:=burger', '-p', 'cmd_vel_topic:=/cmd_vel', '-p', 'internal_cmd_vel_topics:=/gz_cmd_vel_unstamped,/gz_cmd_vel_model_unstamped', '-p', 'cmd_republish_rate_hz:=0.0', '-p', 'watchdog_timeout_sec:=0.5', '-p', 'log_every_n_republish:=100'], output='screen', name='burger_twist_stamped_to_twist_v63')

    bridge = Node(package='ros_gz_bridge', executable='parameter_bridge', name='burger_ros_gz_bridge_v63', output='screen', parameters=[{'config_file': bridge_config}])

    frame_tools = ExecuteProcess(cmd=['python3', frame_tools_script, '--ros-args', '-r', '__node:=single_domain_nav2_frame_tools', '-p', 'use_sim_time:=true', '-p', 'robot_name:=burger', '-p', ['initial_x:=', burger_x], '-p', ['initial_y:=', burger_y], '-p', ['initial_yaw:=', burger_yaw], '-p', 'reset_odom_origin_on_first_msg:=true', '-p', 'initial_pose_repeat_count:=40', '-p', 'initial_pose_period_sec:=0.25'], output='screen', name='burger_frame_tools_v63')

    map_odom_localization = ExecuteProcess(cmd=['python3', map_odom_localization_script, '--ros-args', '-r', '__node:=map_odom_localization', '-p', 'use_sim_time:=true', '-p', 'robot_name:=burger', '-p', 'odom_topic:=/odom_nav', '-p', 'map_frame:=map', '-p', 'odom_frame:=odom', '-p', 'base_frame:=base_footprint', '-p', ['initial_x:=', burger_x], '-p', ['initial_y:=', burger_y], '-p', ['initial_yaw:=', burger_yaw], '-p', 'relative_to_world_origin:=true', '-p', ['world_origin_x:=', map_origin_x], '-p', ['world_origin_y:=', map_origin_y], '-p', ['world_origin_yaw:=', map_origin_yaw], '-p', 'publish_rate_hz:=30.0', '-p', 'publish_amcl_pose:=true'], output='screen', name='burger_map_odom_localization_v63')

    burger_pose = ExecuteProcess(cmd=['python3', tf_pose_script, '--ros-args', '-r', '__node:=burger_pose_tf_publisher', '-p', 'use_sim_time:=true', '-p', 'target_frame:=map', '-p', 'source_frame:=base_footprint', '-p', 'output_topic:=/burger_pose', '-p', 'publish_rate_hz:=10.0', '-p', 'log_every_n:=100'], output='screen', name='burger_pose_tf_publisher_v63')

    burger_pose_fallback = ExecuteProcess(cmd=['python3', odom_pose_fallback_script, '--ros-args', '-r', '__node:=burger_pose_odom_fallback_publisher', '-p', 'use_sim_time:=true', '-p', 'odom_topic:=/odom_nav', '-p', 'output_topic:=/burger_pose_odom_fallback', '-p', 'frame_id:=map', '-p', ['initial_x:=', burger_x], '-p', ['initial_y:=', burger_y], '-p', ['initial_yaw:=', burger_yaw], '-p', 'log_every_n:=100'], output='screen', name='burger_pose_odom_fallback_v63')

    controller_server = Node(package='nav2_controller', executable='controller_server', name='controller_server', output='screen', parameters=[nav2_params])
    planner_server = Node(package='nav2_planner', executable='planner_server', name='planner_server', output='screen', parameters=[nav2_params])
    behavior_server = Node(package='nav2_behaviors', executable='behavior_server', name='behavior_server', output='screen', parameters=[nav2_params])
    bt_navigator = Node(package='nav2_bt_navigator', executable='bt_navigator', name='bt_navigator', output='screen', parameters=[nav2_params])
    nav_lifecycle = Node(package='nav2_lifecycle_manager', executable='lifecycle_manager', name='lifecycle_manager_navigation', output='screen', parameters=[nav2_params])

    goal_proxy_default = ExecuteProcess(cmd=['python3', goal_proxy_script, '--ros-args', '-r', '__node:=burger_default_goal_pose_to_nav2', '-p', 'use_sim_time:=true', '-p', 'goal_pose_topic:=/goal_pose', '-p', 'navigate_action:=/navigate_to_pose', '-p', 'default_frame_id:=map', '-p', 'cancel_previous_goal:=true'], output='screen', name='burger_default_goal_pose_to_nav2_v63')
    goal_proxy_named = ExecuteProcess(cmd=['python3', goal_proxy_script, '--ros-args', '-r', '__node:=burger_named_goal_pose_to_nav2', '-p', 'use_sim_time:=true', '-p', 'goal_pose_topic:=/burger_goal_pose', '-p', 'navigate_action:=/navigate_to_pose', '-p', 'default_frame_id:=map', '-p', 'cancel_previous_goal:=true'], output='screen', name='burger_named_goal_pose_to_nav2_v63')

    follower = ExecuteProcess(cmd=['python3', follower_script, '--ros-args', '-r', '__node:=domain_bridge_nav2_follower', '-p', 'use_sim_time:=true', '-p', 'leader_pose_topic:=/leader_pose', '-p', 'navigate_action:=/navigate_to_pose', '-p', 'follow_distance:=1.05', '-p', 'goal_period_sec:=1.5', '-p', 'goal_update_distance:=0.25', '-p', 'cancel_previous_goal:=false'], output='screen', name='domain_bridge_nav2_follower_v63')

    astar_cmd = ExecuteProcess(cmd=[
        'python3', astar_cmd_script,
        '--ros-args',
        '-r', '__node:=burger_astar_cmd_vel_follower',
        '-p', 'use_sim_time:=true',
        '-p', 'robot_name:=burger',
        '-p', 'map_topic:=/map',
        '-p', 'robot_pose_topic:=/burger_pose',
        '-p', 'robot_pose_backup_topic:=/burger_pose_odom_fallback',
        '-p', 'leader_pose_topic:=/leader_pose',
        '-p', 'manual_goal_topic:=/goal_pose',
        '-p', 'dynamic_obstacle_pose_topic:=/leader_pose',
        '-p', 'manual_goal_slot_dx_m:=0.0',
        '-p', 'manual_goal_slot_dy_m:=-0.65',
        '-p', 'manual_goal_slot_use_goal_yaw:=true',
        '-p', 'scan_topic:=/scan_nav',
        '-p', 'cmd_vel_topic:=/cmd_vel',
        '-p', 'path_topic:=/astar_path',
        '-p', ['target_mode:=', astar_target_mode],
        '-p', 'leader_goal_mode:=leader_pose',
        '-p', 'follow_distance:=0.0',
        '-p', 'goal_tolerance:=0.24',
        '-p', 'replan_period_sec:=0.5',
        '-p', 'control_rate_hz:=10.0',
        '-p', 'lookahead_distance:=0.30',
        '-p', 'occupied_threshold:=45',
        '-p', 'treat_unknown_as_obstacle:=false',
        '-p', 'inflation_radius_m:=0.22',
            '-p', 'soft_inflation_radius_m:=0.42',
            '-p', 'clearance_cost_weight:=7.0',
            '-p', 'unknown_cost_weight:=1.5',
            '-p', 'diagonal_motion:=false',
        '-p', 'dynamic_obstacle_hard_radius_m:=0.48',
        '-p', 'dynamic_obstacle_soft_radius_m:=0.95',
        '-p', 'dynamic_obstacle_cost_weight:=10.0',
        '-p', 'peer_collision_stop_distance:=0.50',
        '-p', 'peer_collision_slow_distance:=0.90',
        '-p', 'max_linear:=0.070',
        '-p', 'min_linear:=0.02',
        '-p', 'max_angular:=0.50',
        '-p', 'front_stop_distance:=0.37',
        '-p', 'front_slow_distance:=0.65',
        '-p', 'stale_goal_sec:=0.0',
        '-p', 'direct_fallback_if_no_path:=true',
        '-p', 'allow_direct_goal_without_map:=true',
        '-p', 'publish_direct_path_on_fallback:=true',
        '-p', 'log_period_sec:=1.0',
    ], output='screen', name='burger_astar_cmd_vel_follower_v63')

    is_nav2_control = IfCondition(PythonExpression(["'", control_mode, "' == 'nav2'"]))
    is_astar_control = IfCondition(PythonExpression(["'", control_mode, "' == 'astar_cmd'"]))

    return LaunchDescription([
        DeclareLaunchArgument('domain_id', default_value='26', description='ROS_DOMAIN_ID used by Burger follower domain.'),
        DeclareLaunchArgument('leader_domain_id', default_value='25', description='ROS_DOMAIN_ID of Waffle/SLAM owner domain.'),
        DeclareLaunchArgument('burger_x', default_value='0.45'),
        DeclareLaunchArgument('burger_y', default_value='-1.60'),
        DeclareLaunchArgument('burger_yaw', default_value='0.0'),
        DeclareLaunchArgument('map_origin_x', default_value='1.20', description='Waffle initial x. Cartographer map origin is Waffle start.'),
        DeclareLaunchArgument('map_origin_y', default_value='-1.60', description='Waffle initial y. Cartographer map origin is Waffle start.'),
        DeclareLaunchArgument('map_origin_yaw', default_value='0.0', description='Waffle initial yaw. Used to align Burger into Waffle SLAM map.'),
        DeclareLaunchArgument('auto_follow', default_value='false', description='Nav2 mode only: true means Burger follows /leader_pose with NavigateToPose goals.'),
        DeclareLaunchArgument('control_mode', default_value='nav2', description='nav2: use Nav2 NavigateToPose. astar_cmd: bypass Nav2 and publish /cmd_vel TwistStamped from A* path tracking.'),
        DeclareLaunchArgument('astar_target_mode', default_value='manual', description='astar_cmd mode: manual follows the same bridged /goal_pose as Waffle; leader keeps old follow-the-leader mode.'),
        SetEnvironmentVariable('ROS_DOMAIN_ID', domain_id),
        SetEnvironmentVariable('RMW_IMPLEMENTATION', 'rmw_fastrtps_cpp'),
        SetEnvironmentVariable('FASTDDS_BUILTIN_TRANSPORTS', 'UDPv4'),
        SetEnvironmentVariable('TURTLEBOT3_MODEL', 'burger'),
        LogInfo(msg='V63_BURGER_NAV2_GROUP | Burger Nav2 receives live SLAM /map and /leader_pose from Domain25. Burger map frame is relative to Waffle SLAM origin.'),
        LogInfo(msg=['V63_BRIDGE_DOMAINS | leader_domain_id=', leader_domain_id, ' -> burger_domain_id=', domain_id, ' and debug back.']),
        LogInfo(msg='V63_SAFE_HOUSE_SPAWN | default Burger=(0.45,-1.60,0.0), map_origin=(1.20,-1.60,0.0). Keep map_origin equal to Waffle initial pose.'),
        LogInfo(msg=['V63_BURGER_CONTROL_MODE | ', control_mode, ' | nav2 or astar_cmd']),
        TimerAction(period=0.5, actions=[domain_bridges]),
        TimerAction(period=1.5, actions=[converter, bridge]),
        TimerAction(period=4.0, actions=[frame_tools]),
        TimerAction(period=5.0, actions=[map_odom_localization, burger_pose, burger_pose_fallback]),
        TimerAction(period=10.0, actions=[controller_server, planner_server, behavior_server, bt_navigator], condition=is_nav2_control),
        TimerAction(period=18.0, actions=[nav_lifecycle], condition=is_nav2_control),
        TimerAction(period=20.0, actions=[goal_proxy_named], condition=is_nav2_control),
        # v63 group mode: no follow-the-leader Nav2 follower; Burger only consumes /burger_goal_pose from dispatcher.
        # TimerAction(period=30.0, actions=[follower], condition=is_nav2_control),
        TimerAction(period=9.0, actions=[astar_cmd], condition=is_astar_control),
    ])
