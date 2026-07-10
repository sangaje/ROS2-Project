#!/usr/bin/env python3
"""Same-domain RViz + cmd_vel preview for debugging a scout's RL exploration.

Run this on the same ROS_DOMAIN_ID as the scout (no bridging -- the scout
runs unnamespaced on its own domain). Shows /map, TF, /scan, RobotModel, a
short unicycle-preview of /cmd_vel (green/moving vs. gray/STALE), and the
RL policy's own /rl_debug_overlay markers.
"""

from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, LogInfo
from launch.substitutions import EnvironmentVariable, LaunchConfiguration
from launch_ros.actions import Node

from fleet_bringup.launch_utils import dds_launch_environment


def generate_launch_description():
    fleet_share = Path(get_package_share_directory('fleet_bringup'))
    system_share = Path(get_package_share_directory('system_bringup'))
    domain_id = LaunchConfiguration('domain_id')
    cmd_vel_topic = LaunchConfiguration('cmd_vel_topic')
    use_stamped_cmd_vel = LaunchConfiguration('use_stamped_cmd_vel')
    base_frame_id = LaunchConfiguration('base_frame_id')

    return LaunchDescription([
        DeclareLaunchArgument(
            'domain_id',
            default_value=EnvironmentVariable('ROS_DOMAIN_ID'),
            description=(
                'DDS domain to view -- must match the scout being debugged '
                '(e.g. 22), not a leader/PC bridge domain.'
            ),
        ),
        DeclareLaunchArgument(
            'cmd_vel_topic', default_value='/cmd_vel',
            description='Scout cmd_vel topic to preview.',
        ),
        DeclareLaunchArgument(
            'use_stamped_cmd_vel', default_value='true',
            choices=['true', 'false'],
            description='Whether cmd_vel_topic carries TwistStamped (true) or Twist (false).',
        ),
        DeclareLaunchArgument(
            'base_frame_id', default_value='base_footprint',
            description='Frame the cmd_vel preview is drawn relative to.',
        ),
        *dds_launch_environment(domain_id),
        LogInfo(msg=['SCOUT_RL_DEBUG_VIEW | domain=', domain_id]),
        Node(
            package='fleet_bringup',
            executable='cmd_vel_marker',
            name='cmd_vel_marker',
            output='screen',
            parameters=[{
                'cmd_vel_topic': cmd_vel_topic,
                'use_stamped_cmd_vel': use_stamped_cmd_vel,
                'base_frame_id': base_frame_id,
                'marker_topic': '/cmd_vel_debug_markers',
            }],
        ),
        ExecuteProcess(
            cmd=[
                str(fleet_share / 'scripts' / 'run_rviz2_clean.bash'),
                '-d',
                str(system_share / 'rviz' / 'scout_rl_debug.rviz'),
                '--ros-args',
                '-r',
                '__node:=rviz2_scout_rl_debug',
            ],
            output='screen',
            name='rviz2_scout_rl_debug',
        ),
    ])
