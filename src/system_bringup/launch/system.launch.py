#!/usr/bin/env python3
"""Single entry point that turns on everything a robot's fleet role needs.

role:=scout  -> fleet bringup (default fleet_role=member) + Bayesian risk map
                + the RL-trained exploration policy driving cmd_vel directly.
role:=leader -> fleet bringup (default fleet_role=leader) + leader-side
                Jetson/OMX AIM integration.
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
    TimerAction,
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import EnvironmentVariable, LaunchConfiguration
from launch.substitutions import PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackagePrefix

from fleet_bringup.domain_bridge_config import (
    write_leader_to_pc_bridge_config,
    write_risk_to_leader_bridge_config,
)
from fleet_bringup.launch_utils import (
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
    start_nav2 = LaunchConfiguration('start_nav2')
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
    risk_domain_id = LaunchConfiguration('risk_domain_id')
    pc_domain_id = LaunchConfiguration('pc_domain_id')
    enable_risk_to_leader_bridge = LaunchConfiguration(
        'enable_risk_to_leader_bridge'
    )
    enable_pc_visualization_bridge = LaunchConfiguration(
        'enable_pc_visualization_bridge'
    )
    member_domain_id = LaunchConfiguration('member_domain_id')
    follower_domain_id = LaunchConfiguration('follower_domain_id')
    start_omx_aim = LaunchConfiguration('start_omx_aim')
    start_yolo_server = LaunchConfiguration('start_yolo_server')
    yolo_server_delay_sec = LaunchConfiguration('yolo_server_delay_sec')
    yolo_server_host = LaunchConfiguration('yolo_server_host')
    yolo_server_port = LaunchConfiguration('yolo_server_port')
    yolo_server_model_path = LaunchConfiguration('yolo_server_model_path')
    yolo_server_device = LaunchConfiguration('yolo_server_device')
    yolo_server_half = LaunchConfiguration('yolo_server_half')
    omx_yolo_node_delay_sec = LaunchConfiguration('omx_yolo_node_delay_sec')
    start_patrol_planner = LaunchConfiguration('start_patrol_planner')
    patrol_planner_delay_sec = LaunchConfiguration('patrol_planner_delay_sec')
    patrol_min_risk = LaunchConfiguration('patrol_min_risk')
    patrol_relative_threshold_ratio = LaunchConfiguration(
        'patrol_relative_threshold_ratio'
    )
    patrol_min_fallback_risk = LaunchConfiguration('patrol_min_fallback_risk')
    patrol_max_candidate_cells = LaunchConfiguration('patrol_max_candidate_cells')
    debug_stream = LaunchConfiguration('debug_stream')
    debug_port = LaunchConfiguration('debug_port')
    unified_dashboard = LaunchConfiguration('unified_dashboard')
    dashboard_host = LaunchConfiguration('dashboard_host')
    dashboard_port = LaunchConfiguration('dashboard_port')

    def make_stack(context):
        role_value = role.perform(context).strip().lower()
        if role_value not in DEFAULT_FLEET_ROLE:
            raise ValueError(
                f"role must be 'scout' or 'leader', got {role_value!r}"
            )

        domain = int(domain_id.perform(context))

        fleet_role_value = fleet_role.perform(context).strip().lower()
        if not fleet_role_value:
            fleet_role_value = DEFAULT_FLEET_ROLE[role_value]
        if fleet_role_value not in FLEET_LAUNCH_FILES:
            raise ValueError(
                f"fleet_role must be one of {sorted(FLEET_LAUNCH_FILES)}, "
                f'got {fleet_role_value!r}'
            )

        main_domain_value = main_domain_id.perform(context).strip()
        main_domain = domain
        if fleet_role_value in ('follower', 'member'):
            if not main_domain_value:
                raise ValueError(
                    'main_domain_id is required for scout/member/follower '
                    'bridging. Pass the launch option '
                    'main_domain_id:=<leader_domain>.'
                )
            main_domain = int(main_domain_value)

        fleet_share = get_package_share_directory('fleet_bringup')
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
        scout_rl_owns_cmd_vel = (
            role_value == 'scout'
            and fleet_role_value == 'member'
            and launch_bool(start_rl_policy.perform(context))
        )

        fleet_launch_args = {
            'domain_id': str(domain),
            'start_robot_bringup': start_robot_bringup.perform(context),
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
            fleet_launch_args['forward_map_to_main'] = (
                'true' if scout_owns_slam else 'false'
            )
        if fleet_role_value in ('leader', 'member'):
            fleet_launch_args['start_nav2'] = (
                'false' if scout_rl_owns_cmd_vel
                else start_nav2.perform(context)
            )
        if scout_owns_slam:
            fleet_launch_args['hardware_param_file'] = os.path.join(
                get_package_share_directory('bayesian_risk_map'),
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
        if scout_rl_owns_cmd_vel and launch_bool(start_nav2.perform(context)):
            actions.append(LogInfo(msg=[
                'SYSTEM_BRINGUP | scout member RL owns /cmd_vel; forcing ',
                'fleet start_nav2:=false to avoid Nav2/RL command conflict.',
            ]))

        if role_value == 'leader':
            process_env = clean_process_environment(str(domain))
            risk_domain_value = risk_domain_id.perform(context).strip()
            if (
                launch_bool(enable_risk_to_leader_bridge.perform(context))
                and risk_domain_value
            ):
                risk_domain = int(risk_domain_value)
                if risk_domain != domain:
                    risk_bridge_config = write_risk_to_leader_bridge_config(
                        risk_domain, domain,
                    )
                    actions.append(LogInfo(msg=[
                        'MAP_BRIDGE_STAGE_A | source_domain=',
                        str(risk_domain),
                        ' | destination_domain=', str(domain),
                        ' | source_topic=/map',
                        ' | bridge_topic=/map_bridge',
                        ' | final_topic=/map',
                        ' | type=nav_msgs/msg/OccupancyGrid',
                    ]))
                    actions.append(TimerAction(
                        period=0.5,
                        actions=[Node(
                            package='domain_bridge',
                            executable='domain_bridge',
                            name='bridge_risk_to_leader',
                            output='screen',
                            arguments=[
                                str(risk_bridge_config),
                                '--wait-for-publisher',
                                'false',
                            ],
                            env=process_env,
                            respawn=True,
                            respawn_delay=3.0,
                        )],
                    ))
                else:
                    actions.append(LogInfo(msg=[
                        'SYSTEM_BRINGUP | risk_domain_id equals leader '
                        'domain; risk->leader bridge skipped.'
                    ]))

            actions.append(TimerAction(
                period=3.0,
                actions=[Node(
                    package='fleet_bringup',
                    executable='fleet_debug_marker',
                    name='fleet_debug_marker',
                    output='screen',
                    parameters=[{
                        'use_sim_time': False,
                        'leader_pose_topic': '/leader_pose',
                        'burger_pose_topic': '/burger_pose',
                        'member_pose_topic': '/member_pose',
                        'marker_topic': '/fleet_debug_markers',
                        'frame_id': 'map',
                        'leader_domain_id': str(domain),
                        'member_domain_id': member_domain_id.perform(context),
                        'burger_domain_id': follower_domain_id.perform(context),
                    }],
                    env=process_env,
                    respawn=True,
                    respawn_delay=3.0,
                )],
            ))

            pc_domain_value = pc_domain_id.perform(context).strip()
            if (
                launch_bool(enable_pc_visualization_bridge.perform(context))
                and pc_domain_value
            ):
                pc_domain = int(pc_domain_value)
                if pc_domain != domain:
                    pc_bridge_config = write_leader_to_pc_bridge_config(
                        domain, pc_domain,
                    )
                    actions.append(TimerAction(
                        period=2.0,
                        actions=[Node(
                            package='domain_bridge',
                            executable='domain_bridge',
                            name='bridge_leader_to_pc_debug',
                            output='screen',
                            arguments=[
                                str(pc_bridge_config),
                                '--wait-for-publisher',
                                'false',
                            ],
                            env=process_env,
                            respawn=True,
                            respawn_delay=3.0,
                        )],
                    ))
                else:
                    actions.append(LogInfo(msg=[
                        'SYSTEM_BRINGUP | pc_domain_id equals leader '
                        'domain; leader->PC bridge skipped.'
                    ]))

            if launch_bool(start_yolo_server.perform(context)):
                flask_yolo_exe = PathJoinSubstitution([
                    FindPackagePrefix('flask_yolo_bridge'),
                    'lib',
                    'flask_yolo_bridge',
                    'flask_yolo_server',
                ])
                flask_yolo_server = ExecuteProcess(
                    cmd=[
                        flask_yolo_exe,
                        '--host', yolo_server_host.perform(context),
                        '--port', yolo_server_port.perform(context),
                        '--model-path', yolo_server_model_path.perform(context),
                        '--device', yolo_server_device.perform(context),
                        '--half', yolo_server_half.perform(context),
                        '--fast-forward', 'true',
                        '--conf', '0.20',
                        '--iou', '0.45',
                        '--max-det', '64',
                        '--imgsz', '960',
                        '--debug-jpeg-quality', '75',
                        '--max-capture-age-sec', '1.5',
                        '--max-queue-wait-sec', '0.05',
                        '--person-only',
                    ],
                    output='screen',
                    name='flask_yolo_server',
                    env=process_env,
                )
                yolo_delay = float(yolo_server_delay_sec.perform(context))
                if yolo_delay > 0.0:
                    actions.append(TimerAction(
                        period=yolo_delay,
                        actions=[
                            LogInfo(msg=[
                                'SYSTEM_BRINGUP | starting flask_yolo_server ',
                                'after ', str(yolo_delay),
                                's stagger to avoid Jetson startup contention',
                            ]),
                            flask_yolo_server,
                        ],
                    ))
                else:
                    actions.append(flask_yolo_server)

            if launch_bool(start_omx_aim.perform(context)):
                omx_launch_path = os.path.join(
                    get_package_share_directory('omx_aim'),
                    'launch',
                    'jetson.launch.py',
                )
                actions.append(IncludeLaunchDescription(
                    PythonLaunchDescriptionSource(omx_launch_path),
                    launch_arguments={
                        # system_bringup owns the leader Flask YOLO server.
                        # Keep the OMX component from starting a duplicate
                        # process on the same port/camera/GPU.
                        'start_yolo_server': 'false',
                        'yolo_server_host': yolo_server_host.perform(context),
                        'yolo_server_port': yolo_server_port.perform(context),
                        'yolo_server_model_path': (
                            yolo_server_model_path.perform(context)
                        ),
                        'yolo_server_device': yolo_server_device.perform(context),
                        'yolo_server_half': yolo_server_half.perform(context),
                        'yolo_node_delay_sec': (
                            omx_yolo_node_delay_sec.perform(context)
                        ),
                        'start_patrol_planner': (
                            start_patrol_planner.perform(context)
                        ),
                        'patrol_planner_delay_sec': (
                            patrol_planner_delay_sec.perform(context)
                        ),
                        'patrol_min_risk': patrol_min_risk.perform(context),
                        'patrol_relative_threshold_ratio': (
                            patrol_relative_threshold_ratio.perform(context)
                        ),
                        'patrol_min_fallback_risk': (
                            patrol_min_fallback_risk.perform(context)
                        ),
                        'patrol_max_candidate_cells': (
                            patrol_max_candidate_cells.perform(context)
                        ),
                        'debug_stream': debug_stream.perform(context),
                        'debug_port': debug_port.perform(context),
                    }.items(),
                ))
            if launch_bool(unified_dashboard.perform(context)):
                actions.append(Node(
                    package='system_bringup',
                    executable='leader_unified_dashboard',
                    name='leader_unified_dashboard',
                    output='screen',
                    parameters=[{
                        'host': dashboard_host.perform(context),
                        'port': int(dashboard_port.perform(context)),
                        'omx_debug_port': int(debug_port.perform(context)),
                        'omx_stream_path': '/stream.mjpg',
                        'omx_state_path': '/state.json',
                        'yolo_server_port': int(yolo_server_port.perform(context)),
                        'yolo_raw_stream_path': '/stream/raw.mjpg',
                        'yolo_overlay_stream_path': '/stream/yolo.mjpg',
                        'yolo_status_path': '/api/status',
                        'map_topic': '/map',
                        'risk_topic': '/risk/risk_map',
                        'leader_pose_topic': '/leader_pose',
                        'follower_pose_topic': '/burger_pose',
                        'follower_name': 'follower21',
                        'member_pose_topic': '/member_pose',
                        'second_follower_pose_topic': '/member_pose',
                        'second_follower_name': 'scout22',
                        'second_follower_role': 'scout',
                        'fleet_poses_topic': '/fleet/robot_poses',
                        'fleet_status_topic': '/fleet/coordination_status',
                        'collision_warning_topic': '/fleet/collision_warning',
                    }],
                    env=process_env,
                    respawn=True,
                    respawn_delay=3.0,
                ))

        if role_value != 'scout':
            if launch_bool(start_rviz.perform(context)):
                actions.append(LogInfo(msg=[
                    'SYSTEM_BRINGUP | start_rviz is ignored on robot-side '
                    'system.launch.py. Run system_bringup pc.launch.py '
                    'on the PC domain instead.'
                ]))
            return actions

        camera_sender_on = launch_bool(start_camera_sender.perform(context))

        if launch_bool(start_risk_map.perform(context)):
            risk_share = get_package_share_directory('bayesian_risk_map')
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
            # remote flask_yolo_server -- force the matching
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
                    # In the target architecture this risk-map layer owns
                    # SLAM by default; keep enable_amcl:=false underneath so
                    # there is only one map->odom source in this domain.
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
            sender_share = get_package_share_directory('flask_yolo_bridge')
            sender_launch_path = os.path.join(
                sender_share, 'launch', 'opencv_camera_to_flask_yolo.launch.py'
            )
            actions.append(IncludeLaunchDescription(
                PythonLaunchDescriptionSource(sender_launch_path),
                launch_arguments={
                    'device': camera_sender_device.perform(context),
                    'server_url': flask_server_url.perform(context),
                    'output_topic': external_detection_topic.perform(context),
                    'width': '1920',
                    'height': '1080',
                    'send_width': '960',
                    'send_height': '540',
                    'camera_fps': '15.0',
                    'max_rate_hz': '5.0',
                    'http_worker_count': '1',
                    'jpeg_quality': '65',
                    'timeout_sec': '1.0',
                    'connect_timeout_sec': '0.3',
                    'read_timeout_sec': '1.2',
                    'max_http_roundtrip_sec': '1.5',
                    'max_frame_age_sec': '1.0',
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
                # scout WITHOUT fleet_bringup's own localization.
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
            actions.append(LogInfo(msg=[
                'SYSTEM_BRINGUP | start_rviz is ignored on robot-side '
                'system.launch.py. Run system_bringup pc.launch.py on '
                'the PC domain instead.'
            ]))

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
            default_value='',
            description=(
                'Leader DDS domain used by domain_bridge. Required for '
                'scout/member/follower bridging; optional '
                'for role:=leader. Pass as main_domain_id:=<leader_domain>.'
            ),
        ),
        DeclareLaunchArgument(
            'fleet_role',
            default_value='',
            choices=['', 'leader', 'follower', 'member'],
            description=(
                'Which fleet_bringup stack to run underneath. Empty '
                "picks a default from role: scout->member, leader->leader."
            ),
        ),
        DeclareLaunchArgument(
            'start_robot_bringup', default_value='true',
            choices=['true', 'false'],
        ),
        DeclareLaunchArgument(
            'start_nav2', default_value='true',
            choices=['true', 'false'],
            description=(
                'member fleet_role only: start Nav2. Set false for a '
                'mapping-only scout where Cartographer/risk map run but '
                'Nav2 should not create planner/controller nodes.'
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
            'enable_cartographer', default_value='false',
            choices=['true', 'false'],
            description=(
                'leader fleet_role only (role:=leader), real mode only: '
                'run Cartographer here. Default false in leader.launch.py '
                'receives the risk/scout SLAM map and runs AMCL against it.'
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
            'enable_amcl', default_value='false',
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
            'start_cartographer', default_value='true',
            choices=['true', 'false'],
            description=(
                'Scout only: let the risk map\'s own Cartographer own '
                'SLAM/TF instead of AMCL. Off by default -- requires '
                'enable_amcl:=false, and the TF chain still needs the '
                'robot bringup\'s own odom broadcast reconciled by hand '
                '(see system_bringup README).'
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
                'flask_yolo_bridge/opencv_camera_to_flask_yolo.'
                'launch.py instead (offload YOLO to a remote '
                'flask_yolo_server, normally the leader Jetson) -- also set '
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
                '(a remote flask_yolo_server, normally the leader Jetson, '
                'does the inference instead).'
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
                'Scout only: run flask_yolo_bridge/'
                'opencv_camera_to_flask_yolo.launch.py on this robot to '
                'offload YOLO inference to a remote '
                'flask_yolo_server, normally the leader Jetson, instead of running YOLO '
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
            'flask_server_url', default_value='http://orin-jetson:5005/detect',
            description=(
                'Scout only, start_camera_sender:=true: remote '
                'flask_yolo_server URL. Defaults to the leader Jetson '
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
                'only if this scout runs without fleet_bringup.'
            ),
        ),
        DeclareLaunchArgument(
            'rl_extra_args', default_value='',
            description='Extra raw CLI flags appended to `ros2 run turtlebot3_rl_training eval_policy`.',
        ),
        DeclareLaunchArgument(
            'start_rviz', default_value='false',
            choices=['true', 'false'],
            description=(
                'Deprecated on robot-side system.launch.py. RViz now runs '
                'only on the PC via pc.launch.py/viewer.launch.py.'
            ),
        ),
        DeclareLaunchArgument(
            'risk_domain_id',
            default_value='',
            description=(
                'Leader role only: risk/scout DDS domain that owns SLAM and '
                'publishes /map. When set, system bringup starts a one-way '
                'risk->leader bridge.'
            ),
        ),
        DeclareLaunchArgument(
            'pc_domain_id',
            default_value='',
            description=(
                'Leader role only: PC visualization DDS domain. When set, '
                'system bringup starts a one-way leader->PC debug bridge.'
            ),
        ),
        DeclareLaunchArgument(
            'enable_risk_to_leader_bridge',
            default_value='false',
            choices=['true', 'false'],
            description=(
                'Leader role only: bridge /map and risk debug topics from '
                'risk_domain_id to leader. Default false because a '
                'Cartographer-owning scout now forwards its own /map upstream.'
            ),
        ),
        DeclareLaunchArgument(
            'enable_pc_visualization_bridge',
            default_value='true',
            choices=['true', 'false'],
            description='Leader role only: bridge selected visualization/debug topics to pc_domain_id.',
        ),
        DeclareLaunchArgument(
            'member_domain_id',
            default_value='',
            description='Leader debug marker label for a member/scout domain.',
        ),
        DeclareLaunchArgument(
            'follower_domain_id',
            default_value='',
            description='Leader debug marker label for a follower domain.',
        ),
        DeclareLaunchArgument(
            'start_omx_aim',
            default_value='true',
            choices=['true', 'false'],
            description='Leader role only: include Jetson/OMX AIM component launch.',
        ),
        DeclareLaunchArgument(
            'start_yolo_server',
            default_value='true',
            choices=['true', 'false'],
            description='Leader role only: run flask_yolo_server on the Jetson.',
        ),
        DeclareLaunchArgument(
            'yolo_server_delay_sec',
            default_value='6.0',
            description='Leader role only: delay heavy Flask YOLO model startup.',
        ),
        DeclareLaunchArgument(
            'yolo_server_host',
            default_value='0.0.0.0',
            description='Leader role only: flask_yolo_server bind address.',
        ),
        DeclareLaunchArgument(
            'yolo_server_port',
            default_value='5005',
            description='Leader role only: flask_yolo_server HTTP port.',
        ),
        DeclareLaunchArgument(
            'yolo_server_model_path',
            default_value='best.pt',
            description='Leader role only: YOLO model path for flask_yolo_server.',
        ),
        DeclareLaunchArgument(
            'yolo_server_device',
            default_value='0',
            description='Leader role only: YOLO device for flask_yolo_server.',
        ),
        DeclareLaunchArgument(
            'yolo_server_half',
            default_value='true',
            choices=['true', 'false'],
            description='Leader role only: use half precision when supported.',
        ),
        DeclareLaunchArgument(
            'omx_yolo_node_delay_sec',
            default_value='14.0',
            description='Leader role only: delay heavy OMX YOLO/camera/model startup.',
        ),
        DeclareLaunchArgument(
            'start_patrol_planner',
            default_value='true',
            choices=['true', 'false'],
            description='Leader role only: start OMX patrol planner from bridged risk map.',
        ),
        DeclareLaunchArgument(
            'patrol_planner_delay_sec',
            default_value='6.0',
            description='Leader role only: small grace before starting patrol planner.',
        ),
        DeclareLaunchArgument(
            'patrol_min_risk',
            default_value='40',
            description='Leader role only: absolute 0-100 risk cutoff for patrol planner.',
        ),
        DeclareLaunchArgument(
            'patrol_relative_threshold_ratio',
            default_value='0.55',
            description='Leader role only: fallback cutoff ratio of current risk peak.',
        ),
        DeclareLaunchArgument(
            'patrol_min_fallback_risk',
            default_value='5',
            description='Leader role only: noise floor for relative patrol candidates.',
        ),
        DeclareLaunchArgument(
            'patrol_max_candidate_cells',
            default_value='2000',
            description='Leader role only: maximum top-risk cells evaluated by patrol NMS.',
        ),
        DeclareLaunchArgument(
            'debug_stream',
            default_value='false',
            choices=['true', 'false'],
            description='Leader role only: enable the OMX MJPEG debug stream.',
        ),
        DeclareLaunchArgument(
            'debug_port',
            default_value='8080',
            description='Leader role only: OMX MJPEG debug stream port.',
        ),
        DeclareLaunchArgument(
            'unified_dashboard',
            default_value='true',
            choices=['true', 'false'],
            description='Leader role only: run the integrated leader dashboard when debug_stream is true.',
        ),
        DeclareLaunchArgument(
            'dashboard_host',
            default_value='0.0.0.0',
            description='Leader role only: integrated dashboard bind address.',
        ),
        DeclareLaunchArgument(
            'dashboard_port',
            default_value='8091',
            description='Leader role only: integrated dashboard HTTP port.',
        ),
        *dds_launch_environment(domain_id),
        OpaqueFunction(function=make_stack),
    ])
