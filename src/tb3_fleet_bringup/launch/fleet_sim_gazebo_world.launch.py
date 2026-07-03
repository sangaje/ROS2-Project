#!/usr/bin/env python3
"""
Single Gazebo world with two burger robots on the configured ROS domain.

Leader  : standard topics (/scan, /odom, /tf, /cmd_vel …)
Follower: burger-prefixed topics (/burger/scan, /burger/odom, /burger/tf …)
          Frame IDs also prefixed (burger/odom → burger/base_footprint)

A separate follower stack (fleet_sim_follower_nav2.launch.py) bridges the
follower's sensor data into the configured follower domain for AMCL/Nav2.
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
    SetEnvironmentVariable,
    TimerAction,
    UnsetEnvironmentVariable,
)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    os.environ.pop('ROS_DISCOVERY_SERVER', None)
    os.environ.pop('ROS_LOCALHOST_ONLY', None)
    os.environ.pop('FASTRTPS_DEFAULT_PROFILES_FILE', None)
    os.environ.pop('FASTDDS_DEFAULT_PROFILES_FILE', None)

    tb3_gz_share = get_package_share_directory('turtlebot3_gazebo')
    ros_gz_share  = get_package_share_directory('ros_gz_sim')

    world       = LaunchConfiguration('world')
    use_sim_time = LaunchConfiguration('use_sim_time')
    domain_id = LaunchConfiguration('domain_id')
    start_gz_client = LaunchConfiguration('start_gz_client')
    leader_x    = LaunchConfiguration('leader_x')
    leader_y    = LaunchConfiguration('leader_y')
    follower_x  = LaunchConfiguration('follower_x')
    follower_y  = LaunchConfiguration('follower_y')

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
                   '-x', leader_x, '-y', leader_y, '-z', '0.01'],
    )
    # Leader Gazebo-to-ROS bridge with standard topic names.
    leader_bridge = Node(
        package='ros_gz_bridge', executable='parameter_bridge',
        name='leader_gz_bridge', output='screen',
        arguments=['--ros-args', '-p', f'config_file:={leader_bridge_yaml}'],
    )

    def make_follower(context, *args, **kwargs):
        fx = follower_x.perform(context)
        fy = follower_y.perform(context)

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
                             '-x', fx, '-y', fy, '-z', '0.01']),
            Node(package='ros_gz_bridge', executable='parameter_bridge',
                 name='follower_gz_bridge', output='screen',
                 arguments=['--ros-args', '-p', f'config_file:={bridge_yaml}']),
        ]

    return LaunchDescription([
        DeclareLaunchArgument('world',        default_value=default_world),
        DeclareLaunchArgument('use_sim_time', default_value='true'),
        DeclareLaunchArgument('domain_id',    default_value='25'),
        DeclareLaunchArgument('start_gz_client', default_value='false',
                              description='Start the Gazebo GUI client. False keeps the headless server running.'),
        DeclareLaunchArgument('leader_x',     default_value='-1.5'),
        DeclareLaunchArgument('leader_y',     default_value='-0.5'),
        DeclareLaunchArgument('follower_x',   default_value='-2.5'),
        DeclareLaunchArgument('follower_y',   default_value='-0.5'),
        UnsetEnvironmentVariable('ROS_DISCOVERY_SERVER'),
        UnsetEnvironmentVariable('ROS_LOCALHOST_ONLY'),
        UnsetEnvironmentVariable('FASTRTPS_DEFAULT_PROFILES_FILE'),
        UnsetEnvironmentVariable('FASTDDS_DEFAULT_PROFILES_FILE'),
        SetEnvironmentVariable('TURTLEBOT3_MODEL', 'burger'),
        SetEnvironmentVariable('ROS_DOMAIN_ID', domain_id),
        AppendEnvironmentVariable('GZ_SIM_RESOURCE_PATH',
                                  os.path.join(tb3_gz_share, 'models')),
        gz_server,
        gz_client,
        leader_rsp,
        follower_rsp,
        TimerAction(period=3.0, actions=[spawn_leader, leader_bridge]),
        TimerAction(period=4.0, actions=[OpaqueFunction(function=make_follower)]),
    ])
