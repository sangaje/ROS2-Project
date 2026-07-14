#!/usr/bin/env python3
"""Unified robot entry point for leader, scout22 and follower21.

ACTIVE_SCOUT and FOLLOWER are both scout-capable field robots.  The follower
starts as a leader-following robot, then takes over the scout mission after the
active scout is confirmed dead.
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
    active_scout_robot_name = LaunchConfiguration('active_scout_robot_name')
    follower_robot_name = LaunchConfiguration('follower_robot_name')
    risk_domain_id = LaunchConfiguration('risk_domain_id')
    member_domain_id = LaunchConfiguration('member_domain_id')
    follower_domain_id = LaunchConfiguration('follower_domain_id')
    require_follower_pose = LaunchConfiguration('require_follower_pose')
    leader_enable_cartographer = LaunchConfiguration('leader_enable_cartographer')
    field_enable_exploration = LaunchConfiguration('field_enable_exploration')
    enable_rl = LaunchConfiguration('enable_rl')
    enable_observation_tx = LaunchConfiguration('enable_observation_tx')
    field_enable_cartographer = LaunchConfiguration('field_enable_cartographer')
    field_enable_amcl = LaunchConfiguration('field_enable_amcl')
    map_authority_eligible = LaunchConfiguration('map_authority_eligible')
    field_forward_map_to_main = LaunchConfiguration('field_forward_map_to_main')
    camera_sender_device = LaunchConfiguration('camera_sender_device')
    flask_server_url = LaunchConfiguration('flask_server_url')

    def make_stack(context):
        name = robot_name.perform(context).strip()
        role = initial_role.perform(context).strip().upper()
        if role not in ('LEADER', 'ACTIVE_SCOUT', 'FOLLOWER'):
            raise ValueError(
                'initial_role must be LEADER, ACTIVE_SCOUT or FOLLOWER, '
                f'got {role!r}'
            )
        is_leader = role == 'LEADER'
        is_follower = role == 'FOLLOWER'

        system_launch = os.path.join(
            get_package_share_directory('system_bringup'),
            'launch',
            'system.launch.py',
        )
        if is_leader:
            launch_args = {
                'role': 'leader',
                'fleet_role': 'leader',
                'domain_id': domain_id.perform(context),
                'main_domain_id': main_domain_id.perform(context),
                'risk_domain_id': risk_domain_id.perform(context),
                'member_domain_id': member_domain_id.perform(context),
                'follower_domain_id': follower_domain_id.perform(context),
                'active_scout_robot_name': active_scout_robot_name.perform(context),
                'follower_robot_name': follower_robot_name.perform(context),
                'enable_cartographer': leader_enable_cartographer.perform(context),
                'require_follower_pose': require_follower_pose.perform(context),
                'enable_scout_failover': 'true',
            }
            return [
                LogInfo(msg=[
                    'FIELD_ROBOT_LAUNCH | robot=leader role=LEADER',
                    ' domain=', domain_id.perform(context),
                    ' main_domain=', main_domain_id.perform(context),
                    ' active_scout=', active_scout_robot_name.perform(context),
                    ' follower_scout=', follower_robot_name.perform(context),
                ]),
                IncludeLaunchDescription(
                    PythonLaunchDescriptionSource(system_launch),
                    launch_arguments=launch_args.items(),
                ),
            ]

        fleet_role = 'follower' if is_follower else 'member'
        default_exploration = True
        default_follow = is_follower
        default_cartographer = not is_follower
        default_amcl = is_follower
        default_map_authority = not is_follower
        requested_cartographer = _bool_text(
            field_enable_cartographer.perform(context),
            default_cartographer,
        )
        requested_rl = _bool_text(enable_rl.perform(context), default_exploration)
        requested_map_authority = _bool_text(
            map_authority_eligible.perform(context),
            default_map_authority,
        )
        # FOLLOWER does not own /map at startup, but it needs this outbound
        # route pre-created so a takeover Cartographer/Risk stack can be seen
        # by the Leader immediately after ACTIVE_SCOUT handoff.
        requested_map_forward = _bool_text(
            field_forward_map_to_main.perform(context),
            is_follower,
        )
        if is_follower and requested_cartographer == 'true':
            raise ValueError(
                'initial_role=FOLLOWER cannot start Cartographer. '
                'Follower uses shared map + AMCL until takeover is confirmed.'
            )
        if is_follower and requested_map_authority == 'true':
            raise ValueError(
                'initial_role=FOLLOWER cannot claim map authority at startup. '
                'Takeover SLAM starts only after ACTIVE_SCOUT authority.'
            )
        launch_args = {
            'role': 'scout',
            'fleet_role': fleet_role,
            'domain_id': domain_id.perform(context),
            'main_domain_id': main_domain_id.perform(context),
            'active_scout_robot_name': (
                name if not is_follower else active_scout_robot_name.perform(context)
            ),
            'follower_robot_name': (
                name if is_follower else follower_robot_name.perform(context)
            ),
            'enable_exploration': _bool_text(
                field_enable_exploration.perform(context),
                default_exploration,
            ),
            'start_rl_worker': requested_rl,
            'start_camera_sender': _bool_text(
                enable_observation_tx.perform(context),
                True,
            ),
            'start_cartographer': requested_cartographer,
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
            'forward_field_map_to_main': requested_map_forward,
        }
        if requested_map_authority != 'true':
            launch_args['start_cartographer'] = 'false'
        return [
            LogInfo(msg=[
                'FIELD_ROBOT_LAUNCH | robot=', name,
                ' role=', role,
                ' fleet_role=', fleet_role,
                ' scout_capable=true',
                ' normal_duty=',
                'leader_follow' if is_follower else 'active_scout',
                ' takeover_duty=active_scout',
                ' domain=', domain_id.perform(context),
                ' main_domain=', main_domain_id.perform(context),
                ' cartographer=', launch_args['start_cartographer'],
                ' rl=', launch_args['start_rl_worker'],
                ' map_authority=', requested_map_authority,
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
            choices=['LEADER', 'ACTIVE_SCOUT', 'FOLLOWER'],
        ),
        DeclareLaunchArgument('active_scout_robot_name', default_value='scout22'),
        DeclareLaunchArgument('follower_robot_name', default_value='follower21'),
        DeclareLaunchArgument('risk_domain_id', default_value='22'),
        DeclareLaunchArgument('member_domain_id', default_value='22'),
        DeclareLaunchArgument('follower_domain_id', default_value='21'),
        DeclareLaunchArgument(
            'require_follower_pose',
            default_value='false',
            choices=['true', 'false'],
        ),
        DeclareLaunchArgument(
            'leader_enable_cartographer',
            default_value='false',
            choices=['true', 'false'],
        ),
        DeclareLaunchArgument(
            'field_enable_exploration',
            default_value='',
            description=(
                'Field robot wrapper override. Empty chooses from initial_role.'
            ),
        ),
        DeclareLaunchArgument('enable_rl', default_value=''),
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
            'field_forward_map_to_main',
            default_value='',
            description=(
                'Forward this field robot local /map through its member/'
                'follower bridge. Empty keeps ACTIVE_SCOUT on its dedicated '
                'map gateway and pre-opens FOLLOWER takeover map egress.'
            ),
        ),
        DeclareLaunchArgument('camera_sender_device', default_value='/dev/video1'),
        DeclareLaunchArgument('flask_server_url', default_value='http://orin-jetson:5005/detect'),
        OpaqueFunction(function=make_stack),
    ])
