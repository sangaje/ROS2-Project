#!/usr/bin/env python3
"""Unified Field Robot entry point for scout22 and follower21.

Scout and Follower are the same robot stack.  The initial role and runtime
enable flags choose whether the robot starts as ACTIVE_SCOUT or FOLLOWER.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, LogInfo, OpaqueFunction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import EnvironmentVariable, LaunchConfiguration


def _bool_text(value: str, default: bool) -> str:
    text = str(value or '').strip().lower()
    if not text:
        return 'true' if default else 'false'
    return 'true' if text in ('1', 'true', 'yes', 'on') else 'false'


def generate_launch_description():
    robot_name = LaunchConfiguration('robot_name')
    domain_id = LaunchConfiguration('domain_id')
    main_domain_id = LaunchConfiguration('main_domain_id')
    initial_role = LaunchConfiguration('initial_role')
    field_enable_exploration = LaunchConfiguration('field_enable_exploration')
    enable_follow = LaunchConfiguration('enable_follow')
    enable_rl = LaunchConfiguration('enable_rl')
    enable_observation_tx = LaunchConfiguration('enable_observation_tx')
    field_enable_cartographer = LaunchConfiguration('field_enable_cartographer')
    field_enable_amcl = LaunchConfiguration('field_enable_amcl')
    map_authority_eligible = LaunchConfiguration('map_authority_eligible')
    forward_field_map_to_main = LaunchConfiguration('forward_field_map_to_main')
    camera_sender_device = LaunchConfiguration('camera_sender_device')
    flask_server_url = LaunchConfiguration('flask_server_url')

    def make_stack(context):
        name = robot_name.perform(context).strip()
        role = initial_role.perform(context).strip().upper()
        if role not in ('ACTIVE_SCOUT', 'FOLLOWER'):
            raise ValueError(
                'initial_role must be ACTIVE_SCOUT or FOLLOWER, '
                f'got {role!r}'
            )
        is_follower = role == 'FOLLOWER'
        fleet_role = 'follower' if is_follower else 'member'
        default_exploration = True
        default_follow = is_follower
        default_cartographer = not is_follower
        default_amcl = is_follower
        default_map_authority = not is_follower

        system_launch = os.path.join(
            get_package_share_directory('system_bringup'),
            'launch',
            'system.launch.py',
        )
        launch_args = {
            'role': 'scout',
            'fleet_role': fleet_role,
            'domain_id': domain_id.perform(context),
            'main_domain_id': main_domain_id.perform(context),
            'active_scout_robot_name': name if not is_follower else 'scout22',
            'follower_robot_name': name if is_follower else 'follower21',
            'enable_exploration': _bool_text(
                field_enable_exploration.perform(context),
                default_exploration,
            ),
            'start_rl_worker': _bool_text(enable_rl.perform(context), True),
            'start_camera_sender': _bool_text(
                enable_observation_tx.perform(context),
                True,
            ),
            'start_cartographer': _bool_text(
                field_enable_cartographer.perform(context),
                default_cartographer,
            ),
            'enable_cartographer': 'false',
            'enable_amcl': _bool_text(field_enable_amcl.perform(context), default_amcl),
            # Bayesian risk is centralized on the leader; this only controls
            # Cartographer/map production on a field robot.
            'start_risk_map': 'false',
            'enable_yolo': 'false',
            'detection_source': 'flask_topic',
            'camera_sender_device': camera_sender_device.perform(context),
            'flask_server_url': flask_server_url.perform(context),
            'enable_scout_failover': 'true',
            'forward_field_map_to_main': _bool_text(
                forward_field_map_to_main.perform(context),
                False,
            ),
        }
        if not _bool_text(map_authority_eligible.perform(context), default_map_authority) == 'true':
            launch_args['start_cartographer'] = 'false'
        return [
            LogInfo(msg=[
                'FIELD_ROBOT_LAUNCH | robot=', name,
                ' role=', role,
                ' fleet_role=', fleet_role,
                ' domain=', domain_id.perform(context),
                ' main_domain=', main_domain_id.perform(context),
            ]),
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(system_launch),
                launch_arguments=launch_args.items(),
            ),
        ]

    return LaunchDescription([
        DeclareLaunchArgument('robot_name', default_value='scout22'),
        DeclareLaunchArgument(
            'domain_id',
            default_value=EnvironmentVariable('ROS_DOMAIN_ID'),
        ),
        DeclareLaunchArgument('main_domain_id', default_value='20'),
        DeclareLaunchArgument(
            'initial_role',
            default_value='ACTIVE_SCOUT',
            choices=['ACTIVE_SCOUT', 'FOLLOWER'],
        ),
        DeclareLaunchArgument(
            'field_enable_exploration',
            default_value='',
            description=(
                'Field robot wrapper override. Empty chooses from initial_role.'
            ),
        ),
        DeclareLaunchArgument('enable_follow', default_value=''),
        DeclareLaunchArgument('enable_rl', default_value='true'),
        DeclareLaunchArgument('enable_observation_tx', default_value='true'),
        DeclareLaunchArgument(
            'field_enable_cartographer',
            default_value='',
            description=(
                'Field robot wrapper override. Empty chooses from initial_role.'
            ),
        ),
        DeclareLaunchArgument(
            'field_enable_amcl',
            default_value='',
            description=(
                'Field robot wrapper override. Empty chooses from initial_role.'
            ),
        ),
        DeclareLaunchArgument('map_authority_eligible', default_value=''),
        DeclareLaunchArgument(
            'forward_field_map_to_main',
            default_value='false',
            choices=['true', 'false'],
            description=(
                'Forward this field robot local /map through its member/'
                'follower bridge. Keep false for normal ACTIVE_SCOUT and '
                'FOLLOWER operation; use only for explicit takeover/commit '
                'tests.'
            ),
        ),
        DeclareLaunchArgument('camera_sender_device', default_value='/dev/video1'),
        DeclareLaunchArgument('flask_server_url', default_value='http://orin-jetson:5005/detect'),
        OpaqueFunction(function=make_stack),
    ])
