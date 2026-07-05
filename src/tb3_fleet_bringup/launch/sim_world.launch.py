#!/usr/bin/env python3
"""
Single Gazebo world with two burger robots on the configured ROS domain.

Leader  : standard topics (/scan, /odom, /tf, /cmd_vel …)
Follower: burger-prefixed topics (/burger/scan, /burger/odom, /burger/tf …)
          Frame IDs also prefixed (burger/odom → burger/base_footprint)

Run follower.launch.py with use_sim_time:=true and
start_robot_bringup:=false to bridge the follower into its DDS domain.
"""

import os
import tempfile
from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    AppendEnvironmentVariable,
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    OpaqueFunction,
    TimerAction,
)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

from tb3_fleet_bringup.launch_utils import dds_launch_environment


def generate_launch_description():
    tb3_gz_share = get_package_share_directory('turtlebot3_gazebo')
    ros_gz_share  = get_package_share_directory('ros_gz_sim')

    world       = LaunchConfiguration('world')
    domain_id = LaunchConfiguration('domain_id')
    start_gz_client = LaunchConfiguration('start_gz_client')

    default_world = os.path.join(tb3_gz_share, 'worlds', 'turtlebot3_world.world')
    leader_sdf    = os.path.join(tb3_gz_share, 'models', 'turtlebot3_burger', 'model.sdf')
    leader_bridge_yaml = os.path.join(tb3_gz_share, 'params', 'turtlebot3_burger_bridge.yaml')
    leader_urdf   = os.path.join(tb3_gz_share, 'urdf', 'turtlebot3_burger.urdf')

    with open(leader_urdf, 'r') as f:
        robot_desc = f.read()

    gz_server = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(ros_gz_share, 'launch', 'gz_sim.launch.py')
        ),
        launch_arguments={
            'gz_args': ['-r -s -v2 ', world],
            'on_exit_shutdown': 'true',
        }.items(),
    )
    gz_client = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(ros_gz_share, 'launch', 'gz_sim.launch.py')
        ),
        launch_arguments={'gz_args': '-g -v2 ', 'on_exit_shutdown': 'true'}.items(),
        condition=IfCondition(start_gz_client),
    )

    # Leader RSP on the leader domain, with standard frame names.
    leader_rsp = Node(
        package='robot_state_publisher', executable='robot_state_publisher',
        name='robot_state_publisher', output='screen',
        parameters=[{'use_sim_time': True, 'robot_description': robot_desc}],
    )

    # Follower RSP on the leader domain, with burger/ frame prefixes.
    follower_rsp = Node(
        package='robot_state_publisher', executable='robot_state_publisher',
        name='burger_state_publisher', output='screen',
        parameters=[{
            'use_sim_time': True,
            'robot_description': robot_desc,
            'frame_prefix': 'burger/',
        }],
        remappings=[('joint_states', '/burger/joint_states')],
    )

    # Spawn leader
    spawn_leader = Node(
        package='ros_gz_sim', executable='create', name='spawn_leader', output='screen',
        arguments=['-name', 'turtlebot3_burger', '-file', leader_sdf,
                   '-x', '-1.5', '-y', '-0.5', '-z', '0.01'],
    )
    # Leader Gazebo-to-ROS bridge with standard topic names.
    leader_bridge = Node(
        package='ros_gz_bridge', executable='parameter_bridge',
        name='leader_gz_bridge', output='screen',
        arguments=['--ros-args', '-p', f'config_file:={leader_bridge_yaml}'],
    )

    def make_follower(context, *args, **kwargs):
        # Modify burger SDF: prefix all topics and frame IDs with 'burger/'
        sdf_text = Path(leader_sdf).read_text()
        subs = [
            ('<topic>scan</topic>',                   '<topic>burger/scan</topic>'),
            ('<gz_frame_id>base_scan</gz_frame_id>',  '<gz_frame_id>burger/base_scan</gz_frame_id>'),
            ('<odom_topic>odom</odom_topic>',          '<odom_topic>burger/odom</odom_topic>'),
            ('<frame_id>odom</frame_id>',              '<frame_id>burger/odom</frame_id>'),
            ('<child_frame_id>base_footprint</child_frame_id>',
             '<child_frame_id>burger/base_footprint</child_frame_id>'),
            ('<tf_topic>/tf</tf_topic>',               '<tf_topic>/burger/tf</tf_topic>'),
            ('<topic>cmd_vel</topic>',                 '<topic>burger/cmd_vel</topic>'),
            ('<topic>joint_states</topic>',            '<topic>burger/joint_states</topic>'),
            ('<topic>imu</topic>',                     '<topic>burger/imu</topic>'),
        ]
        for old, new in subs:
            sdf_text = sdf_text.replace(old, new)
        follower_sdf = Path(tempfile.gettempdir()) / 'turtlebot3_burger_follower.sdf'
        follower_sdf.write_text(sdf_text)

        # Follower Gazebo-to-ROS bridge yaml with burger/ prefixed topics.
        bridge_yaml_text = """\
- ros_topic_name: "/burger/joint_states"
  gz_topic_name: "burger/joint_states"
  ros_type_name: "sensor_msgs/msg/JointState"
  gz_type_name: "gz.msgs.Model"
  direction: GZ_TO_ROS
- ros_topic_name: "/burger/odom"
  gz_topic_name: "burger/odom"
  ros_type_name: "nav_msgs/msg/Odometry"
  gz_type_name: "gz.msgs.Odometry"
  direction: GZ_TO_ROS
- ros_topic_name: "/burger/tf"
  gz_topic_name: "burger/tf"
  ros_type_name: "tf2_msgs/msg/TFMessage"
  gz_type_name: "gz.msgs.Pose_V"
  direction: GZ_TO_ROS
- ros_topic_name: "/burger/cmd_vel"
  gz_topic_name: "burger/cmd_vel"
  ros_type_name: "geometry_msgs/msg/TwistStamped"
  gz_type_name: "gz.msgs.Twist"
  direction: ROS_TO_GZ
- ros_topic_name: "/burger/scan"
  gz_topic_name: "burger/scan"
  ros_type_name: "sensor_msgs/msg/LaserScan"
  gz_type_name: "gz.msgs.LaserScan"
  direction: GZ_TO_ROS
"""
        bridge_yaml = Path(tempfile.gettempdir()) / 'turtlebot3_burger_follower_bridge.yaml'
        bridge_yaml.write_text(bridge_yaml_text)

        return [
            Node(package='ros_gz_sim', executable='create',
                 name='spawn_follower', output='screen',
                 arguments=['-name', 'turtlebot3_burger_follower',
                             '-file', str(follower_sdf),
                             '-x', '-2.2', '-y', '-0.5', '-z', '0.01']),
            Node(package='ros_gz_bridge', executable='parameter_bridge',
                 name='follower_gz_bridge', output='screen',
                 arguments=['--ros-args', '-p', f'config_file:={bridge_yaml}']),
        ]

    return LaunchDescription([
        DeclareLaunchArgument('world',        default_value=default_world),
        DeclareLaunchArgument('domain_id',    default_value='24'),
        DeclareLaunchArgument('start_gz_client', default_value='false',
                              description='Start the Gazebo GUI client. False keeps the headless server running.'),
        *dds_launch_environment(domain_id),
        AppendEnvironmentVariable('GZ_SIM_RESOURCE_PATH',
                                  os.path.join(tb3_gz_share, 'models')),
        gz_server,
        gz_client,
        leader_rsp,
        follower_rsp,
        TimerAction(period=3.0, actions=[spawn_leader, leader_bridge]),
        TimerAction(period=4.0, actions=[OpaqueFunction(function=make_follower)]),
    ])
