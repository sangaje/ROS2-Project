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
    RegisterEventHandler,
    SetEnvironmentVariable,
    TimerAction,
    UnsetEnvironmentVariable,
)
from launch.conditions import IfCondition
from launch.event_handlers import OnProcessExit
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import EnvironmentVariable, LaunchConfiguration, PythonExpression
from launch_ros.actions import Node
from nav2_common.launch import RewrittenYaml


def generate_launch_description():
    bringup_share = get_package_share_directory('tb3_fleet_bringup')

    mode        = LaunchConfiguration('mode')
    domain_id   = LaunchConfiguration('domain_id')
    robot_model = LaunchConfiguration('robot_model')
    use_slam    = LaunchConfiguration('use_slam')
    slam_impl   = LaunchConfiguration('slam_impl')
    initial_x   = LaunchConfiguration('initial_x')
    initial_y   = LaunchConfiguration('initial_y')
    initial_yaw = LaunchConfiguration('initial_yaw')
    start_robot_bringup = LaunchConfiguration('start_robot_bringup')
    start_state_publisher = LaunchConfiguration('start_state_publisher')
    start_lidar = LaunchConfiguration('start_lidar')
    start_base = LaunchConfiguration('start_base')
    lds_model = LaunchConfiguration('lds_model')
    usb_port = LaunchConfiguration('usb_port')
    lidar_port = LaunchConfiguration('lidar_port')

    real_condition = IfCondition(PythonExpression(["'", mode, "' == 'real'"]))
    sim_condition = IfCondition(PythonExpression(["'", mode, "' == 'sim'"]))
    robot_condition = IfCondition(PythonExpression([
        "'", mode, "' == 'real' and '", start_robot_bringup,
        "'.lower() in ['true', '1', 'yes', 'on']",
    ]))

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
        param_rewrites={
            'use_sim_time': 'false',
            'odom_topic': '/odom',
            'scan_topic': '/scan_nav',
            'topic': '/scan_nav',
            'enable_stamped_cmd_vel': 'true',
        },
        convert_types=True,
    )

    cartographer_config_dir = os.path.join(bringup_share, 'config')
    tb3_cartographer_share = get_package_share_directory('turtlebot3_cartographer')
    tb3_cartographer_launch = os.path.join(
        tb3_cartographer_share, 'launch', 'cartographer.launch.py'
    )
    robot_launch = os.path.join(bringup_share, 'launch', 'robot.launch.py')
    tf_pose_script    = os.path.join(bringup_share, 'scripts', 'tf_pose_publisher_direct_v44.py')
    goal_proxy_script = os.path.join(bringup_share, 'scripts', 'pose_to_nav2_action_direct_v41.py')
    pose_tf_script    = os.path.join(bringup_share, 'scripts', 'pose_to_tf_broadcaster.py')
    scan_relay_script = os.path.join(bringup_share, 'scripts', 'scan_frame_relay.py')
    wait_ready_script = os.path.join(bringup_share, 'scripts', 'wait_for_fleet_ready.bash')

    # ── Localization: Cartographer SLAM or AMCL ────────────────────────────────
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
            if slam_impl.perform(context).strip().lower() == 'official':
                return [TimerAction(period=0.5, actions=[
                    LogInfo(msg='LEADER_SLAM | using turtlebot3_cartographer cartographer.launch.py'),
                    IncludeLaunchDescription(
                        PythonLaunchDescriptionSource(tb3_cartographer_launch),
                        launch_arguments={
                            'use_sim_time': 'false',
                            'use_rviz': 'false',
                            'resolution': '0.05',
                            'publish_period_sec': '1.0',
                        }.items(),
                    ),
                ])]

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
                remappings=[('scan', '/scan_cartographer')],
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
    scan_cartographer_relay = ExecuteProcess(
        cmd=[
            'python3', scan_relay_script, '--ros-args',
            '-r', '__node:=leader_scan_cartographer_relay',
            '-p', 'use_sim_time:=false',
            '-p', 'input_topic:=/scan',
            '-p', 'output_topic:=/scan_cartographer',
            '-p', 'output_frame:=base_scan',
            '-p', 'input_reliability:=best_effort',
            '-p', 'output_reliability:=reliable',
        ],
        output='screen', name='leader_scan_cartographer_relay',
    )
    scan_nav_relay = ExecuteProcess(
        cmd=[
            'python3', scan_relay_script, '--ros-args',
            '-r', '__node:=leader_scan_nav_relay',
            '-p', 'use_sim_time:=false',
            '-p', 'input_topic:=/scan',
            '-p', 'output_topic:=/scan_nav',
            '-p', 'output_frame:=base_scan',
            '-p', 'input_reliability:=best_effort',
            '-p', 'output_reliability:=reliable',
        ],
        output='screen', name='leader_scan_nav_relay',
    )

    def make_scan_relays(context, *args, **kwargs):
        actions = [scan_nav_relay]
        slam_on = use_slam.perform(context).strip().lower() in ('true', '1', 'yes', 'on')
        official_slam = slam_impl.perform(context).strip().lower() == 'official'
        if not (slam_on and official_slam):
            actions.insert(0, scan_cartographer_relay)
        return [TimerAction(period=0.2, actions=actions)]

    burger_amcl_tf = ExecuteProcess(
        cmd=[
            'python3', pose_tf_script, '--ros-args',
            '-r', '__node:=real_burger_amcl_tf_on_leader_domain',
            '-p', 'use_sim_time:=false',
            '-p', 'input_topic:=/burger_pose',
            '-p', 'parent_frame:=map',
            '-p', 'child_frame:=burger/base_footprint',
            '-p', 'republish_hz:=10.0',
        ],
        output='screen', name='real_burger_amcl_tf_on_leader_domain',
    )
    burger_scan_static_tf = ExecuteProcess(
        cmd=[
            'ros2', 'run', 'tf2_ros', 'static_transform_publisher',
            '--x', '-0.032',
            '--y', '0.0',
            '--z', '0.182',
            '--roll', '0',
            '--pitch', '0',
            '--yaw', '0',
            '--frame-id', 'burger/base_footprint',
            '--child-frame-id', 'burger/base_scan',
        ],
        output='screen', name='real_burger_scan_static_tf',
    )
    robot_bringup = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(robot_launch),
        launch_arguments={
            'role': 'leader',
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
    sim_leader = ExecuteProcess(
        cmd=[
            'ros2', 'launch', 'tb3_fleet_bringup', 'sim_leader.launch.py',
            ['domain_id:=', domain_id],
            ['follower_initial_x:=', initial_x],
            ['follower_initial_y:=', initial_y],
        ],
        output='screen',
        name='sim_leader',
        condition=sim_condition,
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
    wait_scan = ExecuteProcess(
        cmd=['bash', wait_ready_script, 'scan', '/scan', 'unused', '0'],
        output='screen',
        name='wait_leader_scan',
    )
    wait_map_tf = ExecuteProcess(
        cmd=['bash', wait_ready_script, 'tf', 'map', 'base_footprint', '0'],
        output='screen',
        name='wait_leader_map_tf',
    )
    default_goal = ExecuteProcess(
        cmd=[
            'python3', goal_proxy_script, '--ros-args',
            '-r', '__node:=waffle_default_goal',
            '-p', 'use_sim_time:=false',
            '-p', 'goal_pose_topic:=/goal_pose',
            '-p', 'navigate_action:=/navigate_to_pose',
            '-p', 'default_frame_id:=map',
            '-p', 'cancel_previous_goal:=true',
        ],
        output='screen', name='waffle_default_goal',
    )
    named_goal = ExecuteProcess(
        cmd=[
            'python3', goal_proxy_script, '--ros-args',
            '-r', '__node:=waffle_named_goal',
            '-p', 'use_sim_time:=false',
            '-p', 'goal_pose_topic:=/waffle_goal_pose',
            '-p', 'navigate_action:=/navigate_to_pose',
            '-p', 'default_frame_id:=map',
            '-p', 'cancel_previous_goal:=true',
        ],
        output='screen', name='waffle_named_goal',
    )

    return LaunchDescription([
        DeclareLaunchArgument('mode',         default_value='real',
                              description='real or sim.'),
        DeclareLaunchArgument('domain_id',    default_value='25'),
        DeclareLaunchArgument('robot_model',  default_value='burger'),
        DeclareLaunchArgument('use_slam',     default_value='true'),
        DeclareLaunchArgument('slam_impl',    default_value='official',
                              description='official uses installed turtlebot3_cartographer; custom uses tb3_fleet_bringup config.'),
        DeclareLaunchArgument('initial_x',    default_value='1.05'),
        DeclareLaunchArgument('initial_y',    default_value='0.0'),
        DeclareLaunchArgument('initial_yaw',  default_value='0.0'),
        DeclareLaunchArgument('start_robot_bringup', default_value='true',
                              description='Real mode only: start base, lidar, and state publisher.'),
        DeclareLaunchArgument('start_state_publisher', default_value='true'),
        DeclareLaunchArgument('start_lidar', default_value='true'),
        DeclareLaunchArgument('start_base', default_value='true'),
        DeclareLaunchArgument('lds_model',
                              default_value=EnvironmentVariable('LDS_MODEL', default_value='LDS-02'),
                              description='LDS-01, LDS-02, or LDS-03. Defaults to LDS_MODEL env.'),
        DeclareLaunchArgument('usb_port', default_value='/dev/ttyACM0'),
        DeclareLaunchArgument('lidar_port',
                              default_value=EnvironmentVariable('LIDAR_PORT', default_value='auto')),
        UnsetEnvironmentVariable('ROS_DISCOVERY_SERVER'),
        UnsetEnvironmentVariable('ROS_LOCALHOST_ONLY'),
        UnsetEnvironmentVariable('FASTRTPS_DEFAULT_PROFILES_FILE'),
        UnsetEnvironmentVariable('FASTDDS_DEFAULT_PROFILES_FILE'),
        SetEnvironmentVariable('ROS_DOMAIN_ID',               domain_id),
        SetEnvironmentVariable('ROS_AUTOMATIC_DISCOVERY_RANGE', 'SUBNET'),
        SetEnvironmentVariable('ROS_LOCALHOST_ONLY',           '0'),
        SetEnvironmentVariable('RMW_IMPLEMENTATION',          'rmw_fastrtps_cpp'),
        SetEnvironmentVariable('TURTLEBOT3_MODEL',            robot_model),
        LogInfo(msg=['LEADER | mode=', mode, ' domain=', domain_id,
                     ' | model=', robot_model, ' | use_slam=', use_slam,
                     ' | robot_bringup=', start_robot_bringup]),
        sim_leader,
        robot_bringup,
        TimerAction(period=1.0, actions=[wait_scan], condition=real_condition),
        OpaqueFunction(function=make_scan_relays, condition=real_condition),
        OpaqueFunction(function=make_localization, condition=real_condition),
        TimerAction(period=4.0, actions=[leader_pose, burger_scan_static_tf, burger_amcl_tf],
                    condition=real_condition),
        TimerAction(period=6.0, actions=[controller_server, planner_server,
                                         behavior_server, bt_navigator],
                    condition=real_condition),
        TimerAction(period=7.0, actions=[wait_map_tf], condition=real_condition),
        RegisterEventHandler(
            OnProcessExit(
                target_action=wait_map_tf,
                on_exit=[
                    lifecycle_nav,
                    TimerAction(period=2.0, actions=[default_goal, named_goal]),
                ],
            ),
            condition=real_condition,
        ),
    ])
