#!/usr/bin/env python3

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, LogInfo, OpaqueFunction
from launch.substitutions import LaunchConfiguration


def _enabled(value):
    return value.strip().lower() in ('true', '1', 'yes', 'on')


def generate_launch_description():
    host = LaunchConfiguration('host')
    main_domain_id = LaunchConfiguration('main_domain_id')
    follower_domain_id = LaunchConfiguration('follower_domain_id')
    robot_name = LaunchConfiguration('robot_name')
    robot_domains = LaunchConfiguration('robot_domains')
    leader_use_slam = LaunchConfiguration('leader_use_slam')
    start_rviz = LaunchConfiguration('start_rviz')
    start_following = LaunchConfiguration('start_following')
    start_pc_bridges = LaunchConfiguration('start_pc_bridges')
    start_domain_bridge = LaunchConfiguration('start_domain_bridge')
    follow_distance = LaunchConfiguration('follow_distance')
    follower_initial_x = LaunchConfiguration('follower_initial_x')
    follower_initial_y = LaunchConfiguration('follower_initial_y')
    follower_initial_yaw = LaunchConfiguration('follower_initial_yaw')
    lds_model = LaunchConfiguration('lds_model')
    usb_port = LaunchConfiguration('usb_port')
    lidar_port = LaunchConfiguration('lidar_port')
    start_robot_bringup = LaunchConfiguration('start_robot_bringup')
    start_state_publisher = LaunchConfiguration('start_state_publisher')
    start_lidar = LaunchConfiguration('start_lidar')
    start_base = LaunchConfiguration('start_base')

    def make_actions(context, *args, **kwargs):
        selected = host.perform(context).strip().lower()
        if selected not in ('pc', 'leader', 'follower'):
            raise RuntimeError("host must be one of: pc, leader, follower")

        main_domain = main_domain_id.perform(context)
        follower_domain = follower_domain_id.perform(context)
        lds = lds_model.perform(context)
        usb = usb_port.perform(context)
        lidar = lidar_port.perform(context)

        actions = [LogInfo(msg=['REAL | host=', selected,
                                ' main_domain=', main_domain,
                                ' follower_domain=', follower_domain])]

        if selected == 'pc':
            actions.append(ExecuteProcess(
                cmd=[
                    'ros2', 'launch', 'tb3_fleet_bringup', 'pc.launch.py',
                    f'main_domain_id:={main_domain}',
                    'start_leader_stack:=true',
                    'start_follower_stack:=false',
                    f'leader_use_slam:={leader_use_slam.perform(context)}',
                    f'start_rviz:={start_rviz.perform(context)}',
                    f'start_following:={start_following.perform(context)}',
                    f'follower_domain_id:={follower_domain}',
                    f'follower_initial_x:={follower_initial_x.perform(context)}',
                    f'follower_initial_y:={follower_initial_y.perform(context)}',
                    f'follower_initial_yaw:={follower_initial_yaw.perform(context)}',
                    f'follow_distance:={follow_distance.perform(context)}',
                ],
                output='screen',
                name='pc',
            ))
            if _enabled(start_pc_bridges.perform(context)):
                actions.append(ExecuteProcess(
                    cmd=[
                        'ros2', 'launch', 'tb3_fleet_bridge', 'bridges.launch.py',
                        f'main_domain_id:={main_domain}',
                        f'robot_domains:={robot_domains.perform(context)}',
                    ],
                    output='screen',
                    name='pc_bridges',
                ))

        elif selected == 'leader':
            actions.append(ExecuteProcess(
                cmd=[
                    'ros2', 'launch', 'tb3_fleet_bringup', 'robot.launch.py',
                    'role:=leader',
                    f'domain_id:={main_domain}',
                    f'lds_model:={lds}',
                    f'usb_port:={usb}',
                    f'lidar_port:={lidar}',
                    f'start_state_publisher:={start_state_publisher.perform(context)}',
                    f'start_lidar:={start_lidar.perform(context)}',
                    f'start_base:={start_base.perform(context)}',
                ],
                output='screen',
                name='leader_robot',
            ))

        else:
            actions.append(ExecuteProcess(
                cmd=[
                    'ros2', 'launch', 'tb3_fleet_bringup', 'follower.launch.py',
                    f'domain_id:={follower_domain}',
                    f'main_domain_id:={main_domain}',
                    f'robot_name:={robot_name.perform(context)}',
                    'use_slam:=false',
                    f'slam_domain:={main_domain}',
                    f'start_domain_bridge:={start_domain_bridge.perform(context)}',
                    f'start_robot_bringup:={start_robot_bringup.perform(context)}',
                    f'start_state_publisher:={start_state_publisher.perform(context)}',
                    f'start_lidar:={start_lidar.perform(context)}',
                    f'start_base:={start_base.perform(context)}',
                    f'lds_model:={lds}',
                    f'usb_port:={usb}',
                    f'lidar_port:={lidar}',
                    f'start_following:={start_following.perform(context)}',
                    f'follow_distance:={follow_distance.perform(context)}',
                    f'follower_initial_x:={follower_initial_x.perform(context)}',
                    f'follower_initial_y:={follower_initial_y.perform(context)}',
                    f'follower_initial_yaw:={follower_initial_yaw.perform(context)}',
                ],
                output='screen',
                name='follower',
            ))

        return actions

    return LaunchDescription([
        DeclareLaunchArgument('host', default_value='pc',
                              description='pc, leader, or follower.'),
        DeclareLaunchArgument('main_domain_id', default_value='25'),
        DeclareLaunchArgument('follower_domain_id', default_value='24'),
        DeclareLaunchArgument('robot_name', default_value='burger'),
        DeclareLaunchArgument('robot_domains', default_value='burger:24'),
        DeclareLaunchArgument('leader_use_slam', default_value='true'),
        DeclareLaunchArgument('start_rviz', default_value='true'),
        DeclareLaunchArgument('start_following', default_value='false'),
        DeclareLaunchArgument('start_pc_bridges', default_value='false',
                              description='Use false when follower.launch.py starts its own bridge.'),
        DeclareLaunchArgument('start_domain_bridge', default_value='true',
                              description='Follower-side bridge.'),
        DeclareLaunchArgument('follow_distance', default_value='1.05'),
        DeclareLaunchArgument('follower_initial_x', default_value='-1.05'),
        DeclareLaunchArgument('follower_initial_y', default_value='0.0'),
        DeclareLaunchArgument('follower_initial_yaw', default_value='0.0'),
        DeclareLaunchArgument('lds_model', default_value='LDS-02'),
        DeclareLaunchArgument('usb_port', default_value='/dev/ttyACM0'),
        DeclareLaunchArgument('lidar_port', default_value='/dev/ttyUSB0'),
        DeclareLaunchArgument('start_robot_bringup', default_value='true'),
        DeclareLaunchArgument('start_state_publisher', default_value='true'),
        DeclareLaunchArgument('start_lidar', default_value='true'),
        DeclareLaunchArgument('start_base', default_value='true'),
        OpaqueFunction(function=make_actions),
    ])
