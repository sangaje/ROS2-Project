#!/usr/bin/env python3

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, TimerAction, LogInfo, SetEnvironmentVariable
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
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

    bridge_config = os.path.join(bringup_share, 'config', 'domain26_burger_ros_gz_bridge.yaml')
    nav2_params = os.path.join(bringup_share, 'config', 'domain26_burger_nav2_slam.yaml')
    single_twist_script = os.path.join(bringup_share, 'scripts', 'single_twist_stamped_to_twist_direct_v36.py')
    frame_tools_script = os.path.join(bringup_share, 'scripts', 'single_domain_nav2_frame_tools_direct_v40.py')
    follower_script = os.path.join(bringup_share, 'scripts', 'domain_bridge_nav2_follower_direct_v40.py')
    map_odom_localization_script = os.path.join(bringup_share, 'scripts', 'map_odom_localization_direct_v44.py')
    tf_pose_script = os.path.join(bringup_share, 'scripts', 'tf_pose_publisher_direct_v44.py')
    goal_proxy_script = os.path.join(bringup_share, 'scripts', 'pose_to_nav2_action_direct_v41.py')

    shared_map_leader_goal_config = PathJoinSubstitution([FindPackageShare('tb3_fleet_bridge'), 'config', 'shared_slam_map_leader_goal_25_to_26_v44.yaml'])
    burger_debug_config = PathJoinSubstitution([FindPackageShare('tb3_fleet_bridge'), 'config', 'burger_debug_26_to_25_v41.yaml'])

    domain_bridge_shared = ExecuteProcess(cmd=['ros2', 'run', 'domain_bridge', 'domain_bridge', shared_map_leader_goal_config], output='screen', name='shared_slam_map_leader_goal_domain_bridge_25_to_26_v44')
    domain_bridge_burger_debug = ExecuteProcess(cmd=['ros2', 'run', 'domain_bridge', 'domain_bridge', burger_debug_config], output='screen', name='burger_debug_domain_bridge_26_to_25_v44')

    converter = ExecuteProcess(cmd=['python3', single_twist_script, '--ros-args', '-r', '__node:=single_twist_stamped_to_twist_bridge', '-p', 'use_sim_time:=true', '-p', 'robot_name:=burger', '-p', 'cmd_vel_topic:=/cmd_vel', '-p', 'internal_cmd_vel_topics:=/gz_cmd_vel_unstamped,/gz_cmd_vel_model_unstamped', '-p', 'cmd_republish_rate_hz:=0.0', '-p', 'watchdog_timeout_sec:=0.5', '-p', 'log_every_n_republish:=100'], output='screen', name='burger_twist_stamped_to_twist_v44')

    bridge = Node(package='ros_gz_bridge', executable='parameter_bridge', name='burger_ros_gz_bridge_domain26', output='screen', parameters=[{'config_file': bridge_config}])

    frame_tools = ExecuteProcess(cmd=['python3', frame_tools_script, '--ros-args', '-r', '__node:=single_domain_nav2_frame_tools', '-p', 'use_sim_time:=true', '-p', 'robot_name:=burger', '-p', ['initial_x:=', burger_x], '-p', ['initial_y:=', burger_y], '-p', ['initial_yaw:=', burger_yaw], '-p', 'reset_odom_origin_on_first_msg:=true', '-p', 'initial_pose_repeat_count:=40', '-p', 'initial_pose_period_sec:=0.25'], output='screen', name='burger_frame_tools_v44')

    map_odom_localization = ExecuteProcess(cmd=['python3', map_odom_localization_script, '--ros-args', '-r', '__node:=map_odom_localization', '-p', 'use_sim_time:=true', '-p', 'robot_name:=burger', '-p', 'odom_topic:=/odom_nav', '-p', 'map_frame:=map', '-p', 'odom_frame:=odom', '-p', 'base_frame:=base_footprint', '-p', ['initial_x:=', burger_x], '-p', ['initial_y:=', burger_y], '-p', ['initial_yaw:=', burger_yaw], '-p', 'relative_to_world_origin:=true', '-p', ['world_origin_x:=', map_origin_x], '-p', ['world_origin_y:=', map_origin_y], '-p', ['world_origin_yaw:=', map_origin_yaw], '-p', 'publish_rate_hz:=30.0', '-p', 'publish_amcl_pose:=true'], output='screen', name='burger_map_odom_localization_v44')

    burger_pose = ExecuteProcess(cmd=['python3', tf_pose_script, '--ros-args', '-r', '__node:=burger_pose_tf_publisher', '-p', 'use_sim_time:=true', '-p', 'target_frame:=map', '-p', 'source_frame:=base_footprint', '-p', 'output_topic:=/burger_pose', '-p', 'publish_rate_hz:=10.0', '-p', 'log_every_n:=100'], output='screen', name='burger_pose_tf_publisher_v44')

    controller_server = Node(package='nav2_controller', executable='controller_server', name='controller_server', output='screen', parameters=[nav2_params])
    planner_server = Node(package='nav2_planner', executable='planner_server', name='planner_server', output='screen', parameters=[nav2_params])
    behavior_server = Node(package='nav2_behaviors', executable='behavior_server', name='behavior_server', output='screen', parameters=[nav2_params])
    bt_navigator = Node(package='nav2_bt_navigator', executable='bt_navigator', name='bt_navigator', output='screen', parameters=[nav2_params])
    nav_lifecycle = Node(package='nav2_lifecycle_manager', executable='lifecycle_manager', name='lifecycle_manager_navigation', output='screen', parameters=[nav2_params])

    goal_proxy_default = ExecuteProcess(cmd=['python3', goal_proxy_script, '--ros-args', '-r', '__node:=burger_default_goal_pose_to_nav2', '-p', 'use_sim_time:=true', '-p', 'goal_pose_topic:=/goal_pose', '-p', 'navigate_action:=/navigate_to_pose', '-p', 'default_frame_id:=map', '-p', 'cancel_previous_goal:=true'], output='screen', name='burger_default_goal_pose_to_nav2_v44')
    goal_proxy_named = ExecuteProcess(cmd=['python3', goal_proxy_script, '--ros-args', '-r', '__node:=burger_named_goal_pose_to_nav2', '-p', 'use_sim_time:=true', '-p', 'goal_pose_topic:=/burger_goal_pose', '-p', 'navigate_action:=/navigate_to_pose', '-p', 'default_frame_id:=map', '-p', 'cancel_previous_goal:=true'], output='screen', name='burger_named_goal_pose_to_nav2_v44')

    follower = ExecuteProcess(cmd=['python3', follower_script, '--ros-args', '-r', '__node:=domain_bridge_nav2_follower', '-p', 'use_sim_time:=true', '-p', 'leader_pose_topic:=/leader_pose', '-p', 'navigate_action:=/navigate_to_pose', '-p', 'follow_distance:=1.05', '-p', 'goal_period_sec:=1.5', '-p', 'goal_update_distance:=0.25', '-p', 'cancel_previous_goal:=false'], output='screen', name='domain_bridge_nav2_follower_v44')

    return LaunchDescription([
        DeclareLaunchArgument('burger_x', default_value='-3.20'),
        DeclareLaunchArgument('burger_y', default_value='-1.75'),
        DeclareLaunchArgument('burger_yaw', default_value='0.0'),
        DeclareLaunchArgument('map_origin_x', default_value='-2.25', description='Waffle initial x. Cartographer map origin is Waffle start.'),
        DeclareLaunchArgument('map_origin_y', default_value='-1.75', description='Waffle initial y. Cartographer map origin is Waffle start.'),
        DeclareLaunchArgument('map_origin_yaw', default_value='0.0', description='Waffle initial yaw. Used to align Burger into Waffle SLAM map.'),
        DeclareLaunchArgument('auto_follow', default_value='true', description='true: Burger follows /leader_pose. false: manual RViz/CLI Burger goals only.'),
        SetEnvironmentVariable('ROS_DOMAIN_ID', '26'),
        SetEnvironmentVariable('RMW_IMPLEMENTATION', 'rmw_fastrtps_cpp'),
        SetEnvironmentVariable('FASTDDS_BUILTIN_TRANSPORTS', 'UDPv4'),
        SetEnvironmentVariable('TURTLEBOT3_MODEL', 'burger'),
        LogInfo(msg='V45_DOMAIN26_BURGER | Burger Nav2 receives live SLAM /map and /leader_pose from Domain25. Burger map frame is relative to Waffle SLAM origin.'),
        LogInfo(msg=['V45_SHARED_SLAM_BRIDGE_CONFIG | ', shared_map_leader_goal_config]),
        LogInfo(msg=['V45_BURGER_DEBUG_BRIDGE_CONFIG | ', burger_debug_config]),
        LogInfo(msg='V45_SAFE_HOUSE_SPAWN | default Burger=(-3.20,-1.75,0.0), map_origin=(-2.25,-1.75,0.0). Keep map_origin equal to Waffle initial pose.'),
        TimerAction(period=0.5, actions=[domain_bridge_shared, domain_bridge_burger_debug]),
        TimerAction(period=1.5, actions=[converter, bridge]),
        TimerAction(period=4.0, actions=[frame_tools]),
        TimerAction(period=5.0, actions=[map_odom_localization, burger_pose]),
        TimerAction(period=10.0, actions=[controller_server, planner_server, behavior_server, bt_navigator]),
        TimerAction(period=18.0, actions=[nav_lifecycle]),
        TimerAction(period=20.0, actions=[goal_proxy_default, goal_proxy_named]),
        TimerAction(period=30.0, actions=[follower], condition=IfCondition(auto_follow)),
    ])
