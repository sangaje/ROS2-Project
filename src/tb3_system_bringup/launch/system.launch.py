#!/usr/bin/env python3
"""Single entry point that turns on everything a robot's fleet role needs.

role:=scout  -> fleet bringup (default fleet_role=member) + Bayesian risk map
                + the RL-trained exploration policy driving cmd_vel directly.
role:=leader -> fleet bringup only (default fleet_role=leader).
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
    'member': 'member.launch.py',
}
DEFAULT_FLEET_ROLE = {
    'scout': 'member',
    'leader': 'leader',
}


def generate_launch_description():
    role = LaunchConfiguration('role')
    domain_id = LaunchConfiguration('domain_id')
    main_domain_id = LaunchConfiguration('main_domain_id')
    fleet_role = LaunchConfiguration('fleet_role')
    start_robot_bringup = LaunchConfiguration('start_robot_bringup')
    ros_static_peers = LaunchConfiguration('ros_static_peers')
    require_follower_pose = LaunchConfiguration('require_follower_pose')
    enable_cartographer = LaunchConfiguration('enable_cartographer')
    auto_localize = LaunchConfiguration('auto_localize')
    enable_amcl = LaunchConfiguration('enable_amcl')
    start_risk_map = LaunchConfiguration('start_risk_map')
    start_cartographer = LaunchConfiguration('start_cartographer')
    cartographer_configuration_basename = LaunchConfiguration(
        'cartographer_configuration_basename'
    )
    start_camera = LaunchConfiguration('start_camera')
    risk_model_path = LaunchConfiguration('risk_model_path')
    detection_source = LaunchConfiguration('detection_source')
    enable_yolo = LaunchConfiguration('enable_yolo')
    external_detection_topic = LaunchConfiguration('external_detection_topic')
    start_camera_sender = LaunchConfiguration('start_camera_sender')
    camera_sender_device = LaunchConfiguration('camera_sender_device')
    flask_server_url = LaunchConfiguration('flask_server_url')
    start_rl_policy = LaunchConfiguration('start_rl_policy')
    rl_model_path = LaunchConfiguration('rl_model_path')
    rl_disable_slam_map = LaunchConfiguration('rl_disable_slam_map')
    rl_extra_args = LaunchConfiguration('rl_extra_args')
    start_rviz = LaunchConfiguration('start_rviz')

    def make_stack(context):
        role_value = role.perform(context).strip().lower()
        if role_value not in DEFAULT_FLEET_ROLE:
            raise ValueError(
                f"role must be 'scout' or 'leader', got {role_value!r}"
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

        # Will this scout end up owning its own SLAM (risk map's
        # Cartographer, via start_cartographer:=true + enable_amcl:=false)?
        # Applies whether the scout is currently in member or follower
        # fleet_role -- a follower is just a scout temporarily tailing the
        # leader instead of exploring (RL suspended above), and its risk
        # map/camera/Cartographer keep running regardless. If so it needs
        # the wheel odometry's own odom->base_footprint TF broadcast
        # disabled, or Cartographer's map->odom(->base_footprint) fights
        # it and TF splits into two disconnected trees.
        scout_owns_slam = (
            role_value == 'scout'
            and fleet_role_value in ('member', 'follower')
            and launch_bool(start_risk_map.perform(context))
            and launch_bool(start_cartographer.perform(context))
            and not launch_bool(enable_amcl.perform(context))
        )

        peers = ros_static_peers.perform(context)
        fleet_launch_args = {
            'domain_id': str(domain),
            'start_robot_bringup': start_robot_bringup.perform(context),
            'ros_static_peers': peers,
        }
        if fleet_role_value == 'leader':
            fleet_launch_args['require_follower_pose'] = (
                require_follower_pose.perform(context)
            )
            fleet_launch_args['enable_cartographer'] = (
                enable_cartographer.perform(context)
            )
        if fleet_role_value in ('follower', 'member', 'leader'):
            fleet_launch_args['auto_localize'] = (
                auto_localize.perform(context)
            )
        if fleet_role_value in ('follower', 'member'):
            fleet_launch_args['main_domain_id'] = str(main_domain)
        if scout_owns_slam:
            fleet_launch_args['hardware_param_file'] = os.path.join(
                get_package_share_directory('tb3_bayesian_risk_map'),
                'config', 'turtlebot3_burger_no_odom_tf.yaml',
            )
        # Whether the fleet stack underneath already owns a map->odom TF
        # source, so start_cartographer below can refuse to also claim
        # that transform. member.launch.py, follower.launch.py and
        # leader.launch.py can each give SLAM up (enable_amcl:=false /
        # enable_cartographer:=false).
        if fleet_role_value in ('member', 'follower'):
            fleet_launch_args['enable_amcl'] = enable_amcl.perform(context)
            fleet_stack_owns_slam = launch_bool(enable_amcl.perform(context))
        else:  # leader
            fleet_stack_owns_slam = launch_bool(
                enable_cartographer.perform(context)
            )

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
            return actions

        camera_sender_on = launch_bool(start_camera_sender.perform(context))

        if launch_bool(start_risk_map.perform(context)):
            risk_share = get_package_share_directory('tb3_bayesian_risk_map')
            risk_launch_path = os.path.join(
                risk_share, 'launch', 'real_robot_risk_slam.launch.py'
            )
            cartographer_on = launch_bool(start_cartographer.perform(context))
            if cartographer_on and fleet_stack_owns_slam:
                if fleet_role_value in ('member', 'follower'):
                    hint = 'Set enable_amcl:=false when turning start_cartographer on.'
                else:
                    hint = (
                        'Set enable_cartographer:=false (leader.launch.py '
                        'will then run AMCL against a map bridged in from '
                        'a member instead) when turning start_cartographer '
                        'on for fleet_role:=leader.'
                    )
                raise ValueError(
                    'start_cartographer:=true would fight with the '
                    f"{fleet_role_value} fleet stack's own map->odom "
                    f'source for this robot. {hint}'
                )
            configured_lua = cartographer_configuration_basename.perform(context)
            if scout_owns_slam and configured_lua == 'turtlebot3_lds_2d_risk_safe.lua':
                # Auto-upgrade to the variant that properly owns the full
                # map->odom->base_footprint chain -- paired with the
                # no-odom-TF hardware param file set above, since the
                # plain default only publishes map->base_footprint
                # directly and leaves odom disconnected.
                configured_lua = 'turtlebot3_lds_2d_risk_safe_no_odom.lua'

            # start_camera_sender:=true means opencv_camera_to_flask_yolo
            # (started below) owns the camera and forwards frames to a
            # PC-side flask_yolo_server instead -- force the matching
            # detection_source wiring so the two never drift apart.
            risk_start_camera = start_camera.perform(context)
            risk_enable_yolo = enable_yolo.perform(context)
            risk_detection_source = detection_source.perform(context)
            if camera_sender_on:
                risk_start_camera = 'false'
                risk_enable_yolo = 'false'
                risk_detection_source = 'flask_topic'

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
                    'cartographer_configuration_basename': configured_lua,
                    'start_camera': risk_start_camera,
                    'model_path': risk_model_path.perform(context),
                    'detection_source': risk_detection_source,
                    'enable_yolo': risk_enable_yolo,
                    'external_detection_topic': (
                        external_detection_topic.perform(context)
                    ),
                    'start_rviz': 'false',
                }.items(),
            ))

        if camera_sender_on:
            sender_share = get_package_share_directory('tb3_flask_yolo_bridge')
            sender_launch_path = os.path.join(
                sender_share, 'launch', 'opencv_camera_to_flask_yolo.launch.py'
            )
            actions.append(IncludeLaunchDescription(
                PythonLaunchDescriptionSource(sender_launch_path),
                launch_arguments={
                    'device': camera_sender_device.perform(context),
                    'server_url': flask_server_url.perform(context),
                    'output_topic': external_detection_topic.perform(context),
                }.items(),
            ))

        if fleet_role_value == 'follower':
            # A follower is a scout that is temporarily suspending its own
            # recon duty to tail the leader instead -- follower.launch.py's
            # Nav2 stack already drives cmd_vel to chase the leader, so the
            # RL policy (which also drives cmd_vel directly) must not run
            # at the same time or the two fight over the motors. The risk
            # map/camera/YOLO stay on above: they only accumulate risk data
            # passively and never command movement themselves.
            if launch_bool(start_rl_policy.perform(context)):
                actions.append(LogInfo(msg=[
                    'SYSTEM_BRINGUP | fleet_role=follower: suspending the '
                    'RL policy (follower.launch.py\'s own Nav2 pursuit '
                    'drives cmd_vel instead). Risk map/camera stay on.'
                ]))
        elif launch_bool(start_rl_policy.perform(context)):
            process_env = clean_process_environment(str(domain), peers)
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
                launch_arguments={
                    'domain_id': str(main_domain),
                    'ros_static_peers': peers,
                }.items(),
            ))

        return actions

    return LaunchDescription([
        DeclareLaunchArgument(
            'role',
            default_value='scout',
            choices=['scout', 'leader'],
            description=(
                "This robot's fleet role: 'scout' (fleet bringup + risk "
                "map + RL policy) or 'leader' (fleet bringup only)."
            ),
        ),
        DeclareLaunchArgument(
            'domain_id',
            default_value=EnvironmentVariable('ROS_DOMAIN_ID'),
            description='This robot\'s DDS domain.',
        ),
        DeclareLaunchArgument(
            'main_domain_id',
            default_value=EnvironmentVariable('MAIN_DOMAIN_ID'),
            description='Leader/PC DDS domain used by domain_bridge.',
        ),
        DeclareLaunchArgument(
            'fleet_role',
            default_value='',
            choices=['', 'leader', 'follower', 'member'],
            description=(
                'Which tb3_fleet_bringup stack to run underneath. Empty '
                "picks a default from role: scout->member, leader->leader."
            ),
        ),
        DeclareLaunchArgument(
            'start_robot_bringup', default_value='true',
            choices=['true', 'false'],
        ),
        DeclareLaunchArgument(
            'ros_static_peers',
            default_value=EnvironmentVariable('ROS_STATIC_PEERS', default_value=''),
            description=(
                'Optional ROS_STATIC_PEERS value (semicolon-separated '
                'addresses) forcing unicast DDS discovery to specific '
                'peers in addition to SUBNET multicast discovery. Needed '
                'when this fleet spans machines that are not on the same '
                'physical LAN and only reachable over a link that does '
                'not carry multicast, e.g. a Tailscale/VPN hop -- without '
                'this, cross-machine topics (like a member-owned /map '
                'reaching the leader, or the PC viewer) can silently show '
                'up in `ros2 topic list` (discovery) but never actually '
                'deliver any messages (`ros2 topic echo`/`hz` hang).'
            ),
        ),
        DeclareLaunchArgument(
            'require_follower_pose', default_value='true',
            choices=['true', 'false'],
            description=(
                "leader fleet_role only (role:=leader): set false if no "
                "follower.launch.py robot is in this fleet, otherwise the "
                'coordinator holds the leader in place forever waiting '
                'for a follower pose that will never arrive.'
            ),
        ),
        DeclareLaunchArgument(
            'enable_cartographer', default_value='true',
            choices=['true', 'false'],
            description=(
                'leader fleet_role only (role:=leader), real mode only: '
                'run Cartographer here (default). Set false to instead '
                'run AMCL against a map received from a member robot '
                'that owns its own SLAM (that member needs '
                'enable_amcl:=false start_cartographer:=true).'
            ),
        ),
        DeclareLaunchArgument(
            'auto_localize', default_value='true',
            choices=['true', 'false'],
            description=(
                'Passed through to follower/member AMCL global '
                'localization, and to leader AMCL when '
                'enable_cartographer:=false.'
            ),
        ),
        DeclareLaunchArgument(
            'enable_amcl', default_value='true',
            choices=['true', 'false'],
            description=(
                'member/follower fleet_role only: run AMCL as the '
                'map->odom TF source (fleet bringup\'s own default). Set '
                'false only when start_cartographer below will own '
                'SLAM/TF instead.'
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
            'detection_source', default_value='local_yolo',
            description=(
                'Scout only, passed through to real_robot_risk_slam.'
                'launch.py. Default local_yolo runs YOLO on this robot. '
                "Set flask_topic to consume detections published by "
                'tb3_flask_yolo_bridge/opencv_camera_to_flask_yolo.'
                'launch.py instead (offload YOLO to a PC running '
                'flask_yolo_server.launch.py) -- also set '
                'enable_yolo:=false and start_camera:=false in that case, '
                'since the sender node owns the camera instead.'
            ),
        ),
        DeclareLaunchArgument(
            'enable_yolo', default_value='true',
            choices=['true', 'false'],
            description=(
                'Scout only: run YOLO inference inside the risk map node '
                'itself. Set false when detection_source:=flask_topic '
                '(a PC-side flask_yolo_server does the inference instead).'
            ),
        ),
        DeclareLaunchArgument(
            'external_detection_topic', default_value='/risk/yolo_detections',
            description=(
                'Scout only: topic the risk map node reads external '
                'detections from when detection_source is not local_yolo '
                '-- must match opencv_camera_to_flask_yolo\'s output_topic.'
            ),
        ),
        DeclareLaunchArgument(
            'start_camera_sender', default_value='false',
            choices=['true', 'false'],
            description=(
                'Scout only: run tb3_flask_yolo_bridge/'
                'opencv_camera_to_flask_yolo.launch.py on this robot to '
                'offload YOLO inference to a PC running '
                'flask_yolo_server.launch.py, instead of running YOLO '
                'locally. When true, this auto-forces start_camera:=false, '
                "enable_yolo:=false and detection_source:=flask_topic so "
                "the risk map reads detections from the sender's "
                'output_topic (kept in sync with external_detection_topic) '
                'instead of running its own camera/YOLO.'
            ),
        ),
        DeclareLaunchArgument(
            'camera_sender_device', default_value='/dev/video1',
            description='Scout only, start_camera_sender:=true: camera device for opencv_camera_to_flask_yolo.',
        ),
        DeclareLaunchArgument(
            'flask_server_url', default_value='http://seil:5005/detect',
            description=(
                'Scout only, start_camera_sender:=true: PC-side '
                'flask_yolo_server URL. Defaults to the PC\'s Tailscale '
                'hostname.'
            ),
        ),
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
        *dds_launch_environment(domain_id, LaunchConfiguration('ros_static_peers')),
        OpaqueFunction(function=make_stack),
    ])
