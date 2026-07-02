#!/usr/bin/env python3

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    LogInfo,
    TimerAction,
)
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PythonExpression


def generate_launch_description():
    slam_domain          = LaunchConfiguration('slam_domain')
    follower_initial_x   = LaunchConfiguration('follower_initial_x')
    follower_initial_y   = LaunchConfiguration('follower_initial_y')
    follower_initial_yaw = LaunchConfiguration('follower_initial_yaw')
    leader_initial_x     = LaunchConfiguration('leader_initial_x')
    leader_initial_y     = LaunchConfiguration('leader_initial_y')
    leader_initial_yaw   = LaunchConfiguration('leader_initial_yaw')
    follow_distance      = LaunchConfiguration('follow_distance')
    start_following      = LaunchConfiguration('start_following')
    start_rviz           = LaunchConfiguration('start_rviz')
    robot1_ip            = LaunchConfiguration('robot1_ip')
    robot2_ip            = LaunchConfiguration('robot2_ip')

    leader_use_slam   = PythonExpression(["'true' if '", slam_domain, "' == '25' else 'false'"])
    follower_use_slam = PythonExpression(["'true' if '", slam_domain, "' == '24' else 'false'"])
    slam_on_leader    = PythonExpression(["'", slam_domain, "' == '25'"])
    slam_on_follower  = PythonExpression(["'", slam_domain, "' == '24'"])

    def make_follower(name):
        return ExecuteProcess(
            cmd=[
                'ros2', 'launch', 'tb3_fleet_bringup',
                'fleet_real_domain24_burger_nav2_follower.launch.py',
                'domain_id:=24',
                'leader_domain_id:=25',
                ['use_slam:=',    follower_use_slam],
                ['slam_domain:=', slam_domain],
                ['start_following:=',    start_following],
                ['follow_distance:=',    follow_distance],
                'enable_path_yield:=true',
                'path_block_distance:=0.55',
                'yield_lateral_distance:=0.75',
                ['follower_initial_x:=',   follower_initial_x],
                ['follower_initial_y:=',   follower_initial_y],
                ['follower_initial_yaw:=', follower_initial_yaw],
                ['robot1_ip:=', robot1_ip],
                ['robot2_ip:=', robot2_ip],
            ],
            output='screen',
            name=name,
        )

    def make_leader(name):
        return ExecuteProcess(
            cmd=[
                'ros2', 'launch', 'tb3_fleet_bringup',
                'fleet_real_domain25_burger_nav2.launch.py',
                'domain_id:=25',
                ['use_slam:=',    leader_use_slam],
                ['initial_x:=',   leader_initial_x],
                ['initial_y:=',   leader_initial_y],
                ['initial_yaw:=', leader_initial_yaw],
                ['robot1_ip:=', robot1_ip],
                ['robot2_ip:=', robot2_ip],
            ],
            output='screen',
            name=name,
        )

    leader_slam_first = make_leader('two_burgers_leader_stack_slam_first')
    follower_after_leader_slam = make_follower('two_burgers_follower_stack_after_leader_slam')
    follower_slam_first = make_follower('two_burgers_follower_stack_slam_first')
    leader_after_follower_slam = make_leader('two_burgers_leader_stack_after_follower_slam')
    rviz = ExecuteProcess(
        cmd=[
            'ros2', 'launch', 'tb3_fleet_bringup',
            'fleet_real_domain25_rviz.launch.py',
            'domain_id:=25',
            ['robot1_ip:=', robot1_ip],
            ['robot2_ip:=', robot2_ip],
        ],
        output='screen',
        name='two_burgers_rviz',
        condition=IfCondition(start_rviz),
    )

    return LaunchDescription([
        DeclareLaunchArgument('slam_domain', default_value='25',
                              description='Which burger does SLAM: 25=leader, 24=follower.'),
        DeclareLaunchArgument('follower_initial_x', default_value='-1.05',
                              description='[slam_domain=25] Follower x in leader SLAM map.'),
        DeclareLaunchArgument('follower_initial_y',   default_value='0.0'),
        DeclareLaunchArgument('follower_initial_yaw', default_value='0.0'),
        DeclareLaunchArgument('leader_initial_x', default_value='1.05',
                              description='[slam_domain=24] Leader x in follower SLAM map.'),
        DeclareLaunchArgument('leader_initial_y',   default_value='0.0'),
        DeclareLaunchArgument('leader_initial_yaw', default_value='0.0'),
        DeclareLaunchArgument('follow_distance',  default_value='1.05'),
        DeclareLaunchArgument('start_following',  default_value='false',
                              description='Safe default: wait for fleet_follow_signal.'),
        DeclareLaunchArgument('start_rviz',  default_value='true'),
        DeclareLaunchArgument('robot1_ip',   default_value='10.10.14.10',
                              description='Leader robot IP (Domain 25).'),
        DeclareLaunchArgument('robot2_ip',   default_value='10.10.14.14',
                              description='Follower robot IP (Domain 24).'),
        LogInfo(msg=['REAL_TWO_BURGERS_PC | slam_domain=', slam_domain,
                     ' | start_following=', start_following,
                     ' | robot1=', robot1_ip, ' robot2=', robot2_ip]),
        LogInfo(msg='REAL_TWO_BURGERS_PC | start order: SLAM domain first, follower/AMCL second, RViz after map warmup.'),
        TimerAction(period=0.0, actions=[leader_slam_first], condition=IfCondition(slam_on_leader)),
        TimerAction(period=8.0, actions=[follower_after_leader_slam], condition=IfCondition(slam_on_leader)),
        TimerAction(period=0.0, actions=[follower_slam_first], condition=IfCondition(slam_on_follower)),
        TimerAction(period=8.0, actions=[leader_after_follower_slam], condition=IfCondition(slam_on_follower)),
        TimerAction(period=18.0, actions=[rviz]),
    ])
