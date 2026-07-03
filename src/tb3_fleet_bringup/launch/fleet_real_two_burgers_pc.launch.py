#!/usr/bin/env python3

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    LogInfo,
    OpaqueFunction,
    TimerAction,
)
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    follower_initial_x   = LaunchConfiguration('follower_initial_x')
    follower_initial_y   = LaunchConfiguration('follower_initial_y')
    follower_initial_yaw = LaunchConfiguration('follower_initial_yaw')
    follow_distance      = LaunchConfiguration('follow_distance')
    start_following      = LaunchConfiguration('start_following')
    start_rviz           = LaunchConfiguration('start_rviz')
    start_leader_stack   = LaunchConfiguration('start_leader_stack')
    start_follower_stack = LaunchConfiguration('start_follower_stack')
    leader_use_slam      = LaunchConfiguration('leader_use_slam')
    start_leader_robot_bringup = LaunchConfiguration('start_leader_robot_bringup')
    start_follower_robot_bringup = LaunchConfiguration('start_follower_robot_bringup')
    main_domain_id       = LaunchConfiguration('main_domain_id')
    leader_domain_id     = LaunchConfiguration('leader_domain_id')
    follower_domain_id   = LaunchConfiguration('follower_domain_id')
    leader_lds_model     = LaunchConfiguration('leader_lds_model')
    follower_lds_model   = LaunchConfiguration('follower_lds_model')
    leader_usb_port      = LaunchConfiguration('leader_usb_port')
    follower_usb_port    = LaunchConfiguration('follower_usb_port')
    leader_lidar_port    = LaunchConfiguration('leader_lidar_port')
    follower_lidar_port  = LaunchConfiguration('follower_lidar_port')
    leader_stack_delay   = LaunchConfiguration('leader_stack_delay')
    follower_stack_delay = LaunchConfiguration('follower_stack_delay')
    rviz_delay           = LaunchConfiguration('rviz_delay')

    def selected_main_domain(context):
        legacy = leader_domain_id.perform(context).strip()
        return legacy if legacy else main_domain_id.perform(context)

    def enabled(value):
        return value.lower() in ('true', '1', 'yes', 'on')

    def make_processes(context, *args, **kwargs):
        main_domain = selected_main_domain(context)
        follower_domain = follower_domain_id.perform(context)
        leader_slam = leader_use_slam.perform(context)
        leader_cmd = [
            'ros2', 'launch', 'tb3_fleet_bringup',
            'fleet_real_leader_nav2.launch.py',
            f'domain_id:={main_domain}',
            'robot_model:=burger',
            f'use_slam:={leader_slam}',
        ]
        follower_cmd = [
            'ros2', 'launch', 'tb3_fleet_bringup',
            'fleet_real_follower_nav2.launch.py',
            f'domain_id:={follower_domain}',
            f'main_domain_id:={main_domain}',
            'use_slam:=false',
            f'slam_domain:={main_domain}',
            f'start_following:={start_following.perform(context)}',
            f'follow_distance:={follow_distance.perform(context)}',
            'enable_path_yield:=true',
            'path_block_distance:=0.55',
            'yield_lateral_distance:=0.75',
            f'follower_initial_x:={follower_initial_x.perform(context)}',
            f'follower_initial_y:={follower_initial_y.perform(context)}',
            f'follower_initial_yaw:={follower_initial_yaw.perform(context)}',
        ]
        rviz_cmd = [
            'ros2', 'launch', 'tb3_fleet_bringup',
            'fleet_rviz.launch.py',
            f'domain_id:={main_domain}',
        ]
        leader_robot_cmd = [
            'ros2', 'launch', 'tb3_fleet_bringup',
            'fleet_real_burger_robot.launch.py',
            'role:=leader',
            f'domain_id:={main_domain}',
            f'lds_model:={leader_lds_model.perform(context)}',
            f'usb_port:={leader_usb_port.perform(context)}',
            f'lidar_port:={leader_lidar_port.perform(context)}',
        ]
        follower_robot_cmd = [
            'ros2', 'launch', 'tb3_fleet_bringup',
            'fleet_real_burger_robot.launch.py',
            'role:=follower',
            f'domain_id:={follower_domain}',
            f'lds_model:={follower_lds_model.perform(context)}',
            f'usb_port:={follower_usb_port.perform(context)}',
            f'lidar_port:={follower_lidar_port.perform(context)}',
        ]

        actions = []
        if enabled(start_leader_robot_bringup.perform(context)):
            actions.append(TimerAction(period=0.0, actions=[
                ExecuteProcess(
                    cmd=leader_robot_cmd,
                    output='screen',
                    name='real_leader_robot_bringup',
                ),
            ]))
        if enabled(start_follower_robot_bringup.perform(context)):
            actions.append(TimerAction(period=0.0, actions=[
                ExecuteProcess(
                    cmd=follower_robot_cmd,
                    output='screen',
                    name='real_follower_robot_bringup',
                ),
            ]))
        if enabled(start_leader_stack.perform(context)):
            actions.append(TimerAction(period=float(leader_stack_delay.perform(context)), actions=[
                ExecuteProcess(
                    cmd=leader_cmd,
                    output='screen',
                    name='real_nav2_leader_slam',
                ),
            ]))
        if enabled(start_follower_stack.perform(context)):
            actions.append(TimerAction(period=float(follower_stack_delay.perform(context)), actions=[
                ExecuteProcess(
                    cmd=follower_cmd,
                    output='screen',
                    name='real_nav2_follower_amcl',
                ),
            ]))
        if enabled(start_rviz.perform(context)):
            actions.append(TimerAction(period=float(rviz_delay.perform(context)), actions=[
                ExecuteProcess(
                    cmd=rviz_cmd,
                    output='screen',
                    name='two_burgers_rviz',
                ),
            ]))
        return actions

    return LaunchDescription([
        DeclareLaunchArgument('follower_initial_x', default_value='-1.05',
                              description='Follower initial x in leader SLAM map.'),
        DeclareLaunchArgument('follower_initial_y',   default_value='0.0'),
        DeclareLaunchArgument('follower_initial_yaw', default_value='0.0'),
        DeclareLaunchArgument('follow_distance',  default_value='1.05'),
        DeclareLaunchArgument('start_following',  default_value='false',
                              description='Safe default: wait for fleet_follow_signal.'),
        DeclareLaunchArgument('start_rviz',  default_value='true'),
        DeclareLaunchArgument('start_leader_stack', default_value='true',
                              description='Start leader SLAM/Nav2 stack on this machine.'),
        DeclareLaunchArgument('start_follower_stack', default_value='false',
                              description='Start follower AMCL/Nav2/bridge stack on this machine. Default false: run it on the follower robot.'),
        DeclareLaunchArgument('leader_use_slam', default_value='true',
                              description='Use Cartographer SLAM for the main/leader stack. false starts AMCL mode.'),
        DeclareLaunchArgument('start_leader_robot_bringup', default_value='false',
                              description='Start leader hardware bringup locally. Use only on the leader robot PC.'),
        DeclareLaunchArgument('start_follower_robot_bringup', default_value='false',
                              description='Start follower hardware bringup locally. Use only on the follower robot PC.'),
        DeclareLaunchArgument('main_domain_id', default_value='25',
                              description='Main/leader ROS domain. RViz, SLAM, leader Nav2 run here.'),
        DeclareLaunchArgument('leader_domain_id', default_value='',
                              description='Deprecated alias for main_domain_id. Empty uses main_domain_id.'),
        DeclareLaunchArgument('follower_domain_id', default_value='24'),
        DeclareLaunchArgument('leader_lds_model', default_value='LDS-02'),
        DeclareLaunchArgument('follower_lds_model', default_value='LDS-02'),
        DeclareLaunchArgument('leader_usb_port', default_value='/dev/ttyACM0'),
        DeclareLaunchArgument('follower_usb_port', default_value='/dev/ttyACM0'),
        DeclareLaunchArgument('leader_lidar_port', default_value='/dev/ttyUSB0'),
        DeclareLaunchArgument('follower_lidar_port', default_value='/dev/ttyUSB0'),
        DeclareLaunchArgument('leader_stack_delay', default_value='0.0'),
        DeclareLaunchArgument('follower_stack_delay', default_value='10.0'),
        DeclareLaunchArgument('rviz_delay', default_value='18.0'),
        LogInfo(msg=['REAL_NAV2_CLEAN_PC | main domain=', main_domain_id,
                     ' leader_use_slam=', leader_use_slam,
                     ' follower domain=', follower_domain_id, ' AMCL/Nav2',
                     ' | local_follower_stack=', start_follower_stack,
                     ' | start_following=', start_following,
                     ' | local_robot_bringup leader=', start_leader_robot_bringup,
                     ' follower=', start_follower_robot_bringup]),
        OpaqueFunction(function=make_processes),
    ])
