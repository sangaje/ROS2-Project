#!/usr/bin/env python3
"""Single entry point that turns on everything a robot's fleet role needs.

role:=scout  -> fleet bringup (default fleet_role=guard) + Bayesian risk map
                + the RL-trained exploration policy driving cmd_vel directly.
role:=waffle -> fleet bringup only (default fleet_role=leader). This role is
                a placeholder: the waffle-specific behaviour stack is still
                beta and is meant to be merged in later.
"""

import os
import shlex

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    IncludeLaunchDescription,
    LogInfo,
    OpaqueFunction,
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import EnvironmentVariable, LaunchConfiguration

from tb3_fleet_bringup.launch_utils import (
    clean_process_environment,
    dds_launch_environment,
    launch_bool,
)


FLEET_LAUNCH_FILES = {
    'leader': 'leader.launch.py',
    'follower': 'follower.launch.py',
    'guard': 'guard.launch.py',
}
DEFAULT_FLEET_ROLE = {
    'scout': 'guard',
    'waffle': 'leader',
}


def generate_launch_description():
    role = LaunchConfiguration('role')
    domain_id = LaunchConfiguration('domain_id')
    main_domain_id = LaunchConfiguration('main_domain_id')
    fleet_role = LaunchConfiguration('fleet_role')
    start_robot_bringup = LaunchConfiguration('start_robot_bringup')
    auto_localize = LaunchConfiguration('auto_localize')
    enable_amcl = LaunchConfiguration('enable_amcl')
    start_risk_map = LaunchConfiguration('start_risk_map')
    start_cartographer = LaunchConfiguration('start_cartographer')
    cartographer_configuration_basename = LaunchConfiguration(
        'cartographer_configuration_basename'
    )
    start_camera = LaunchConfiguration('start_camera')
    risk_model_path = LaunchConfiguration('risk_model_path')
    start_rl_policy = LaunchConfiguration('start_rl_policy')
    rl_model_path = LaunchConfiguration('rl_model_path')
    rl_disable_slam_map = LaunchConfiguration('rl_disable_slam_map')
    rl_extra_args = LaunchConfiguration('rl_extra_args')
    start_rviz = LaunchConfiguration('start_rviz')

    def make_stack(context):
        role_value = role.perform(context).strip().lower()
        if role_value not in DEFAULT_FLEET_ROLE:
            raise ValueError(
                f"role must be 'scout' or 'waffle', got {role_value!r}"
            )

        domain = int(domain_id.perform(context))
        main_domain = int(main_domain_id.perform(context))

        fleet_role_value = fleet_role.perform(context).strip().lower()
        if not fleet_role_value:
            fleet_role_value = DEFAULT_FLEET_ROLE[role_value]
        if fleet_role_value not in FLEET_LAUNCH_FILES:
            raise ValueError(
                f"fleet_role must be one of {sorted(FLEET_LAUNCH_FILES)}, "
                f'got {fleet_role_value!r}'
            )

        fleet_share = get_package_share_directory('tb3_fleet_bringup')
        fleet_launch_path = os.path.join(
            fleet_share, 'launch', FLEET_LAUNCH_FILES[fleet_role_value]
        )
        fleet_launch_args = {
            'domain_id': str(domain),
            'start_robot_bringup': start_robot_bringup.perform(context),
        }
        if fleet_role_value in ('follower', 'guard'):
            fleet_launch_args['main_domain_id'] = str(main_domain)
            fleet_launch_args['auto_localize'] = (
                auto_localize.perform(context)
            )
        # Whether AMCL (fleet bringup's own map->odom TF source) will be
        # running underneath, so start_cartographer below can refuse to
        # also claim that transform. guard.launch.py is the only fleet
        # role with an enable_amcl switch; follower.launch.py always runs
        # AMCL, and leader.launch.py never does (it runs Cartographer).
        if fleet_role_value == 'guard':
            fleet_launch_args['enable_amcl'] = enable_amcl.perform(context)
            amcl_will_run = launch_bool(enable_amcl.perform(context))
        elif fleet_role_value == 'follower':
            amcl_will_run = True
        else:
            amcl_will_run = False

        actions = [
            LogInfo(msg=[
                'SYSTEM_BRINGUP | role=', role_value,
                ' fleet_role=', fleet_role_value,
                ' domain=', str(domain),
            ]),
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(fleet_launch_path),
                launch_arguments=fleet_launch_args.items(),
            ),
        ]

        if role_value != 'scout':
            if role_value == 'waffle':
                actions.append(LogInfo(msg=[
                    'SYSTEM_BRINGUP | role=waffle is a beta placeholder: '
                    'only fleet bringup runs. Risk map and RL policy are '
                    'scout-only for now.'
                ]))
            return actions

        if launch_bool(start_risk_map.perform(context)):
            risk_share = get_package_share_directory('tb3_bayesian_risk_map')
            risk_launch_path = os.path.join(
                risk_share, 'launch', 'real_robot_risk_slam.launch.py'
            )
            cartographer_on = launch_bool(start_cartographer.perform(context))
            if cartographer_on and amcl_will_run:
                raise ValueError(
                    'start_cartographer:=true and enable_amcl:=true would '
                    'both try to own map->odom for this robot. Set '
                    'enable_amcl:=false when turning start_cartographer on.'
                )
            actions.append(IncludeLaunchDescription(
                PythonLaunchDescriptionSource(risk_launch_path),
                launch_arguments={
                    'use_sim_time': 'false',
                    # The fleet stack above already brings up hardware.
                    # start_cartographer defaults to false because AMCL
                    # (fleet bringup's own map->odom source) is on by
                    # default; flip enable_amcl:=false first if you want
                    # this Cartographer to own SLAM/TF instead.
                    'start_robot_bringup': 'false',
                    'start_cartographer': (
                        'true' if cartographer_on else 'false'
                    ),
                    'cartographer_configuration_basename': (
                        cartographer_configuration_basename.perform(context)
                    ),
                    'start_camera': start_camera.perform(context),
                    'model_path': risk_model_path.perform(context),
                    'start_rviz': 'false',
                }.items(),
            ))

        if launch_bool(start_rl_policy.perform(context)):
            process_env = clean_process_environment(str(domain))
            rl_command = [
                'ros2', 'run', 'turtlebot3_rl_training', 'eval_policy',
                '--model', rl_model_path.perform(context),
                '--real-robot',
            ]
            if launch_bool(rl_disable_slam_map.perform(context)):
                # eval_policy unconditionally tries to own SLAM (its own
                # Cartographer/slam_toolbox + map->odom TF) whenever
                # --real-robot is passed, unless map use is disabled here.
                # The fleet stack above already owns localization for this
                # robot, so leave this on unless you deliberately run the
                # scout WITHOUT tb3_fleet_bringup's own localization.
                rl_command.append('--disable-slam-map')
            extra = rl_extra_args.perform(context).strip()
            if extra:
                rl_command.extend(shlex.split(extra))
            actions.append(ExecuteProcess(
                cmd=rl_command,
                output='screen',
                name='scout_rl_policy',
                env=process_env,
            ))

        if launch_bool(start_rviz.perform(context)):
            system_share = get_package_share_directory('tb3_system_bringup')
            viewer_launch_path = os.path.join(
                system_share, 'launch', 'viewer.launch.py'
            )
            actions.append(IncludeLaunchDescription(
                PythonLaunchDescriptionSource(viewer_launch_path),
                launch_arguments={'domain_id': str(main_domain)}.items(),
            ))

        return actions

    return LaunchDescription([
        DeclareLaunchArgument(
            'role',
            default_value='scout',
            choices=['scout', 'waffle'],
            description=(
                "This robot's fleet role: 'scout' (fleet bringup + risk "
                "map + RL policy) or 'waffle' (fleet bringup only, beta "
                'placeholder).'
            ),
        ),
        DeclareLaunchArgument(
            'domain_id',
            default_value=EnvironmentVariable('ROS_DOMAIN_ID', default_value='26'),
            description='This robot\'s DDS domain.',
        ),
        DeclareLaunchArgument(
            'main_domain_id',
            default_value='24',
            description='Leader/PC DDS domain used by domain_bridge.',
        ),
        DeclareLaunchArgument(
            'fleet_role',
            default_value='',
            choices=['', 'leader', 'follower', 'guard'],
            description=(
                'Which tb3_fleet_bringup stack to run underneath. Empty '
                "picks a default from role: scout->guard, waffle->leader."
            ),
        ),
        DeclareLaunchArgument(
            'start_robot_bringup', default_value='true',
            choices=['true', 'false'],
        ),
        DeclareLaunchArgument(
            'auto_localize', default_value='true',
            choices=['true', 'false'],
            description='Passed through to follower/guard AMCL global localization.',
        ),
        DeclareLaunchArgument(
            'enable_amcl', default_value='true',
            choices=['true', 'false'],
            description=(
                'guard fleet_role only: run AMCL as the map->odom TF '
                'source (fleet bringup\'s own default). Set false only '
                'when start_cartographer below will own SLAM/TF instead.'
            ),
        ),
        DeclareLaunchArgument(
            'start_risk_map', default_value='true',
            choices=['true', 'false'],
            description='Scout only: turn on the Bayesian risk map stack.',
        ),
        DeclareLaunchArgument(
            'start_cartographer', default_value='false',
            choices=['true', 'false'],
            description=(
                'Scout only: let the risk map\'s own Cartographer own '
                'SLAM/TF instead of AMCL. Off by default -- requires '
                'enable_amcl:=false, and the TF chain still needs the '
                'robot bringup\'s own odom broadcast reconciled by hand '
                '(see tb3_system_bringup README).'
            ),
        ),
        DeclareLaunchArgument(
            'cartographer_configuration_basename',
            default_value='turtlebot3_lds_2d_risk_safe.lua',
            description=(
                'Passed through to real_robot_risk_slam.launch.py when '
                'start_cartographer:=true.'
            ),
        ),
        DeclareLaunchArgument(
            'start_camera', default_value='true',
            choices=['true', 'false'],
            description='Scout only: start the USB camera feeding the risk map YOLO detector.',
        ),
        DeclareLaunchArgument('risk_model_path', default_value='yolo11n.pt'),
        DeclareLaunchArgument(
            'start_rl_policy', default_value='true',
            choices=['true', 'false'],
            description='Scout only: run the trained RL policy against cmd_vel.',
        ),
        DeclareLaunchArgument(
            'rl_model_path',
            default_value='rl_models/sac_turtlebot3_burger.zip',
        ),
        DeclareLaunchArgument(
            'rl_disable_slam_map', default_value='true',
            choices=['true', 'false'],
            description=(
                'Stop eval_policy from starting its own SLAM/map->odom TF, '
                "since the fleet stack above already owns it. Turn off "
                'only if this scout runs without tb3_fleet_bringup.'
            ),
        ),
        DeclareLaunchArgument(
            'rl_extra_args', default_value='',
            description='Extra raw CLI flags appended to `ros2 run turtlebot3_rl_training eval_policy`.',
        ),
        DeclareLaunchArgument(
            'start_rviz', default_value='false',
            choices=['true', 'false'],
            description='Also bring up the unified fleet+risk RViz view (see viewer.launch.py).',
        ),
        *dds_launch_environment(domain_id),
        OpaqueFunction(function=make_stack),
    ])
