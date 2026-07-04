#!/usr/bin/env python3
"""
Master launcher for the full domain-bridge fleet simulation test.

Topology:
  Gazebo on leader domain ─ two burger robots ─┐
  Leader: Cartographer SLAM + Nav2
  Follower: domain_bridge + TF relay + AMCL + follower script
  RViz on leader domain: fleet_debug.rviz

Start order:
  T=0   Gazebo world (robot spawn at T+3, T+4 inside)
  T=5   leader stack
  T=10  follower stack
  T=15  RViz
"""

import os

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    LogInfo,
    TimerAction,
)
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    follow_distance = LaunchConfiguration('follow_distance')
    start_following = LaunchConfiguration('start_following')
    start_rviz      = LaunchConfiguration('start_rviz')
    start_gz_client = LaunchConfiguration('start_gz_client')
    leader_domain_id = LaunchConfiguration('leader_domain_id')
    follower_domain_id = LaunchConfiguration('follower_domain_id')
    follower_initial_x = LaunchConfiguration('follower_initial_x')
    follower_initial_y = LaunchConfiguration('follower_initial_y')
    follower_initial_yaw = LaunchConfiguration('follower_initial_yaw')

    gazebo = ExecuteProcess(
        cmd=['ros2', 'launch', 'tb3_fleet_bringup',
             'sim_world.launch.py',
             ['domain_id:=', leader_domain_id],
             ['start_gz_client:=', start_gz_client]],
        output='screen', name='sim_gazebo_world',
    )
    leader = ExecuteProcess(
        cmd=['ros2', 'launch', 'tb3_fleet_bringup',
             'sim_leader.launch.py',
             ['domain_id:=', leader_domain_id],
             ['follower_initial_x:=', follower_initial_x],
             ['follower_initial_y:=', follower_initial_y]],
        output='screen', name='sim_leader',
    )
    follower = ExecuteProcess(
        cmd=['ros2', 'launch', 'tb3_fleet_bringup',
             'sim_follower.launch.py',
             ['domain_id:=', follower_domain_id],
             ['leader_domain_id:=', leader_domain_id],
             ['follow_distance:=', follow_distance],
             ['start_following:=', start_following],
             ['follower_initial_x:=', follower_initial_x],
             ['follower_initial_y:=', follower_initial_y],
             ['follower_initial_yaw:=', follower_initial_yaw]],
        output='screen', name='sim_follower',
    )
    rviz = ExecuteProcess(
        cmd=['ros2', 'launch', 'tb3_fleet_bringup',
             'rviz.launch.py',
             ['domain_id:=', leader_domain_id]],
        output='screen', name='sim_rviz',
        condition=IfCondition(start_rviz),
    )

    return LaunchDescription([
        DeclareLaunchArgument('follow_distance', default_value='1.05'),
        DeclareLaunchArgument('start_following', default_value='false',
                              description='true=start following immediately'),
        DeclareLaunchArgument('start_rviz',      default_value='true'),
        DeclareLaunchArgument('start_gz_client', default_value='false',
                              description='Start Gazebo GUI client in addition to the headless server.'),
        DeclareLaunchArgument('leader_domain_id', default_value='25'),
        DeclareLaunchArgument('follower_domain_id', default_value='24'),
        DeclareLaunchArgument('follower_initial_x', default_value='-1.0',
                              description='Follower initial x in the leader SLAM map.'),
        DeclareLaunchArgument('follower_initial_y', default_value='0.0'),
        DeclareLaunchArgument('follower_initial_yaw', default_value='0.0'),
        LogInfo(msg=['SIM_FLEET_MASTER | leader_domain=', leader_domain_id,
                     ' follower_domain=', follower_domain_id,
                     ' | follow_distance=', follow_distance,
                     ' | start_following=', start_following]),
        gazebo,
        TimerAction(period=5.0,  actions=[leader]),
        TimerAction(period=10.0, actions=[follower]),
        TimerAction(period=15.0, actions=[rviz]),
    ])
