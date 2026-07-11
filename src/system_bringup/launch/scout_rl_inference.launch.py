#!/usr/bin/env python3
"""Run only the ACTIVE_SCOUT RL inference process.

Use this with the main system launch set to ``enable_exploration:=false`` when
you want RL inference isolated from the rest of ``unified_field_robot``.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo
from launch.substitutions import EnvironmentVariable, LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue

from fleet_bringup.launch_utils import dds_launch_environment


def generate_launch_description():
    domain_id = LaunchConfiguration('domain_id')
    robot_name = LaunchConfiguration('robot_name')
    role_topic = LaunchConfiguration('role_topic')
    initial_role_active = LaunchConfiguration('initial_role_active')
    cmd_vel_topic = LaunchConfiguration('cmd_vel_topic')
    use_stamped_cmd_vel = LaunchConfiguration('use_stamped_cmd_vel')
    enable_velocity_safety_filter = LaunchConfiguration(
        'enable_velocity_safety_filter'
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            'domain_id',
            default_value=EnvironmentVariable('ROS_DOMAIN_ID'),
            description='DDS domain where /scan, /map, TF and /cmd_vel live.',
        ),
        DeclareLaunchArgument(
            'robot_name',
            default_value='scout22',
            description='Robot name used to derive the default role topic.',
        ),
        DeclareLaunchArgument(
            'role_topic',
            default_value='',
            description='Latched role topic. Empty means /<robot_name>/role.',
        ),
        DeclareLaunchArgument(
            'initial_role_active',
            default_value='true',
            choices=['true', 'false'],
            description='Start inference immediately instead of waiting for ACTIVE_SCOUT.',
        ),
        DeclareLaunchArgument(
            'cmd_vel_topic',
            default_value='/cmd_vel',
            description='Velocity topic owned by this inference process.',
        ),
        DeclareLaunchArgument(
            'use_stamped_cmd_vel',
            default_value='true',
            choices=['true', 'false'],
            description='Publish TwistStamped when true, Twist when false.',
        ),
        DeclareLaunchArgument(
            'enable_velocity_safety_filter',
            default_value='true',
            choices=['true', 'false'],
            description='Apply the runtime backup/slowdown safety projection.',
        ),
        *dds_launch_environment(domain_id),
        LogInfo(msg=[
            'SCOUT_RL_INFERENCE | model=',
            'sac_turtlebot3_burger_emergency.zip vector_dim=63 domain=',
            domain_id,
        ]),
        Node(
            package='system_bringup',
            executable='scout_rl_policy_worker',
            name='scout_rl_policy_worker',
            output='screen',
            parameters=[{
                'robot_name': robot_name,
                'role_topic': role_topic,
                'initial_role_active': ParameterValue(
                    initial_role_active,
                    value_type=bool,
                ),
                'cmd_vel_topic': cmd_vel_topic,
                'use_stamped_cmd_vel': ParameterValue(
                    use_stamped_cmd_vel,
                    value_type=bool,
                ),
                'enable_velocity_safety_filter': ParameterValue(
                    enable_velocity_safety_filter,
                    value_type=bool,
                ),
            }],
            respawn=True,
            respawn_delay=3.0,
        ),
    ])
