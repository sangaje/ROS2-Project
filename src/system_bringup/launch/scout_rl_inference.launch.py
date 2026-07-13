#!/usr/bin/env python3
"""Run only the ACTIVE_SCOUT RL inference process.

``system.launch.py`` includes this launch for the canonical external-worker
backend.  It remains directly runnable for diagnostics.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo
from launch.substitutions import EnvironmentVariable, LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue

from fleet_bringup.launch_utils import dds_launch_environment
from system_bringup.launch_defaults import (
    DEFAULT_ACTIVE_SCOUT,
    DEFAULT_CMD_VEL_TOPIC,
)


def generate_launch_description():
    domain_id = LaunchConfiguration('domain_id')
    robot_name = LaunchConfiguration('robot_name')
    role_topic = LaunchConfiguration('role_topic')
    initial_role_active = LaunchConfiguration('initial_role_active')
    failover_state_topic = LaunchConfiguration('failover_state_topic')
    active_scout_id_topic = LaunchConfiguration('active_scout_id_topic')
    scout_epoch_topic = LaunchConfiguration('scout_epoch_topic')
    localization_ready_topic = LaunchConfiguration('localization_ready_topic')
    field_robot_status_topic = LaunchConfiguration('field_robot_status_topic')
    require_failover_activation = LaunchConfiguration('require_failover_activation')
    require_localization_ready = LaunchConfiguration('require_localization_ready')
    require_system_ready = LaunchConfiguration('require_system_ready')
    system_ready_topic = LaunchConfiguration('system_ready_topic')
    require_video_ready = LaunchConfiguration('require_video_ready')
    video_ready_topic = LaunchConfiguration('video_ready_topic')
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
            'require_localization_ready',
            default_value='true',
            choices=['true', 'false'],
            description=(
                'Require /localization_ready in addition to scan/map/TF. '
                'Set false for a Cartographer-owned active scout.'
            ),
        ),
        DeclareLaunchArgument(
            'robot_name',
            default_value=DEFAULT_ACTIVE_SCOUT,
            description='Robot name used to derive the default role topic.',
        ),
        DeclareLaunchArgument(
            'role_topic',
            default_value='',
            description='Latched role topic. Empty means /<robot_name>/role.',
        ),
        DeclareLaunchArgument(
            'initial_role_active',
            default_value='false',
            choices=['true', 'false'],
            description='Debug only: start inference without the failover activation gate.',
        ),
        DeclareLaunchArgument(
            'failover_state_topic',
            default_value='/failover/state',
            description='Latched failover state topic used by the activation gate.',
        ),
        DeclareLaunchArgument(
            'active_scout_id_topic',
            default_value='/failover/active_scout_id',
            description='Latched active scout owner topic used by the activation gate.',
        ),
        DeclareLaunchArgument(
            'scout_epoch_topic',
            default_value='/failover/scout_epoch',
            description='Latched scout ownership epoch topic.',
        ),
        DeclareLaunchArgument(
            'localization_ready_topic',
            default_value='/localization_ready',
            description='Localization readiness topic for this robot domain.',
        ),
        DeclareLaunchArgument(
            'field_robot_status_topic',
            default_value='/fleet/field_robot_status',
            description='Field robot status topic carrying motion authority state.',
        ),
        DeclareLaunchArgument(
            'require_failover_activation',
            default_value='true',
            choices=['true', 'false'],
            description='Require active scout id, epoch, localization and motion-release gates.',
        ),
        DeclareLaunchArgument(
            'require_video_ready',
            default_value='true',
            choices=['true', 'false'],
            description='Hold RL movement until the leader-owned start_motion latch is true.',
        ),
        DeclareLaunchArgument(
            'require_system_ready',
            default_value='false',
            choices=['true', 'false'],
            description='Legacy internal gate; default false because /fleet/start_motion owns motion release.',
        ),
        DeclareLaunchArgument(
            'system_ready_topic',
            default_value='/system/ready',
            description='Latched fleet-wide startup readiness topic.',
        ),
        DeclareLaunchArgument(
            'video_ready_topic',
            default_value='/fleet/start_motion',
            description='Latched leader-owned final motion permission topic.',
        ),
        DeclareLaunchArgument(
            'cmd_vel_topic',
            default_value=DEFAULT_CMD_VEL_TOPIC,
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
                'failover_state_topic': failover_state_topic,
                'active_scout_id_topic': active_scout_id_topic,
                'scout_epoch_topic': scout_epoch_topic,
                'localization_ready_topic': localization_ready_topic,
                'field_robot_status_topic': field_robot_status_topic,
                'require_failover_activation': ParameterValue(
                    require_failover_activation,
                    value_type=bool,
                ),
                'require_localization_ready': ParameterValue(
                    require_localization_ready,
                    value_type=bool,
                ),
                'require_system_ready': ParameterValue(
                    require_system_ready,
                    value_type=bool,
                ),
                'system_ready_topic': system_ready_topic,
                'require_video_ready': ParameterValue(
                    require_video_ready,
                    value_type=bool,
                ),
                'video_ready_topic': video_ready_topic,
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
