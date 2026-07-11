#!/usr/bin/env python3
"""Single entry point that turns on everything a robot's fleet role needs.

role:=scout  -> fleet bringup (default fleet_role=member) + Scout-owned
                Cartographer/risk map + Jetson-offloaded camera sender.
role:=leader -> fleet bringup (default fleet_role=leader) + shared-map
                AMCL/Nav2 and Jetson/OMX AIM integration.
"""

import os

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
    with_virtualenv_site_packages,
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
    leader_auto_localize = LaunchConfiguration('leader_auto_localize')
    enable_amcl = LaunchConfiguration('enable_amcl')
    start_risk_map = LaunchConfiguration('start_risk_map')
    start_cartographer = LaunchConfiguration('start_cartographer')
    cartographer_configuration_basename = LaunchConfiguration(
        'cartographer_configuration_basename'
    )
    start_camera = LaunchConfiguration('start_camera')
    start_teleop = LaunchConfiguration('start_teleop')
    risk_model_path = LaunchConfiguration('risk_model_path')
    risk_target_class = LaunchConfiguration('risk_target_class')
    detection_source = LaunchConfiguration('detection_source')
    enable_yolo = LaunchConfiguration('enable_yolo')
    external_detection_topic = LaunchConfiguration('external_detection_topic')
    start_camera_sender = LaunchConfiguration('start_camera_sender')
    camera_sender_device = LaunchConfiguration('camera_sender_device')
    flask_server_url = LaunchConfiguration('flask_server_url')
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
    enable_scout_failover = LaunchConfiguration('enable_scout_failover')
    leader_robot_name = LaunchConfiguration('leader_robot_name')
    active_scout_robot_name = LaunchConfiguration('active_scout_robot_name')
    follower_robot_name = LaunchConfiguration('follower_robot_name')
    scout_liveness_topic = LaunchConfiguration('scout_liveness_topic')
    scout_liveness_timeout_sec = LaunchConfiguration('scout_liveness_timeout_sec')
    scout_failure_confirm_sec = LaunchConfiguration('scout_failure_confirm_sec')
    scout_pose_topic = LaunchConfiguration('scout_pose_topic')
    scout_pose_timeout_sec = LaunchConfiguration('scout_pose_timeout_sec')
    enable_leader_shadow_follow = LaunchConfiguration('enable_leader_shadow_follow')
    leader_shadow_direct_cmd_vel = LaunchConfiguration('leader_shadow_direct_cmd_vel')
    leader_shadow_follow_distance_m = LaunchConfiguration('leader_shadow_follow_distance_m')
    leader_shadow_stop_distance_m = LaunchConfiguration('leader_shadow_stop_distance_m')
    leader_shadow_resume_distance_m = LaunchConfiguration('leader_shadow_resume_distance_m')
    leader_shadow_far_distance_m = LaunchConfiguration('leader_shadow_far_distance_m')
    leader_shadow_max_linear_vel = LaunchConfiguration('leader_shadow_max_linear_vel')
    leader_shadow_catchup_max_linear_vel = LaunchConfiguration(
        'leader_shadow_catchup_max_linear_vel'
    )
    leader_shadow_max_angular_vel = LaunchConfiguration('leader_shadow_max_angular_vel')
    leader_shadow_goal_update_period_sec = LaunchConfiguration(
        'leader_shadow_goal_update_period_sec'
    )
    leader_shadow_goal_min_change_m = LaunchConfiguration(
        'leader_shadow_goal_min_change_m'
    )
    leader_shadow_cmd_linear_scale = LaunchConfiguration(
        'leader_shadow_cmd_linear_scale'
    )
    leader_shadow_cmd_angular_scale = LaunchConfiguration(
        'leader_shadow_cmd_angular_scale'
    )
    leader_shadow_cmd_max_linear_vel = LaunchConfiguration(
        'leader_shadow_cmd_max_linear_vel'
    )
    leader_shadow_cmd_max_angular_vel = LaunchConfiguration(
        'leader_shadow_cmd_max_angular_vel'
    )
    leader_shadow_heading_min_motion_m = LaunchConfiguration(
        'leader_shadow_heading_min_motion_m'
    )
    enable_leader_continuous_scan = LaunchConfiguration(
        'enable_leader_continuous_scan'
    )
    leader_scan_fov_deg = LaunchConfiguration('leader_scan_fov_deg')
    leader_scan_update_rate_hz = LaunchConfiguration('leader_scan_update_rate_hz')
    leader_scan_timeout_sec = LaunchConfiguration('leader_scan_timeout_sec')
    leader_recovery_standoff_m = LaunchConfiguration('leader_recovery_standoff_m')
    leader_failure_arrival_tolerance_m = LaunchConfiguration(
        'leader_failure_arrival_tolerance_m'
    )
    follower_recovery_standoff_m = LaunchConfiguration('follower_recovery_standoff_m')
    scout_takeover_arrival_tolerance_m = LaunchConfiguration(
        'scout_takeover_arrival_tolerance_m'
    )
    enable_localization_spin_on_takeover = LaunchConfiguration(
        'enable_localization_spin_on_takeover'
    )
    enable_exploration = LaunchConfiguration('enable_exploration')
    start_omx_aim = LaunchConfiguration('start_omx_aim')
    start_yolo_server = LaunchConfiguration('start_yolo_server')
    yolo_server_delay_sec = LaunchConfiguration('yolo_server_delay_sec')
    yolo_server_host = LaunchConfiguration('yolo_server_host')
    yolo_server_port = LaunchConfiguration('yolo_server_port')
    yolo_server_model_path = LaunchConfiguration('yolo_server_model_path')
    yolo_server_target_class = LaunchConfiguration('yolo_server_target_class')
    yolo_server_conf = LaunchConfiguration('yolo_server_conf')
    yolo_server_device = LaunchConfiguration('yolo_server_device')
    yolo_server_half = LaunchConfiguration('yolo_server_half')
    omx_yolo_node_delay_sec = LaunchConfiguration('omx_yolo_node_delay_sec')
    omx_camera_index = LaunchConfiguration('omx_camera_index')
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
        process_env = with_virtualenv_site_packages(
            clean_process_environment(str(domain))
        )

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
            if main_domain_value:
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
            fleet_launch_args['auto_localize'] = (
                leader_auto_localize.perform(context)
            )
        if fleet_role_value in ('follower', 'member'):
            fleet_launch_args['auto_localize'] = (
                auto_localize.perform(context)
            )
        if (
            role_value == 'scout'
            and fleet_role_value == 'follower'
            and launch_bool(enable_scout_failover.perform(context))
        ):
            fleet_launch_args['start_legacy_follower'] = 'false'
        if fleet_role_value in ('follower', 'member'):
            fleet_launch_args['main_domain_id'] = str(main_domain)
            fleet_launch_args['forward_map_to_main'] = (
                'true' if scout_owns_slam else 'false'
            )
        if fleet_role_value in ('leader', 'member'):
            nav2_value = start_nav2.perform(context)
            if scout_owns_slam:
                nav2_value = 'false'
            elif scout_rl_owns_cmd_vel:
                nav2_value = 'false'
            fleet_launch_args['start_nav2'] = (
                nav2_value
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
        if fleet_role_value in ('member', 'follower') and not main_domain_value:
            actions.append(LogInfo(msg=[
                'SYSTEM_BRINGUP | main_domain_id not set; running ',
                fleet_role_value,
                ' standalone on domain ', str(domain),
                ' without cross-domain fleet bridge.',
            ]))
        if scout_rl_owns_cmd_vel and launch_bool(start_nav2.perform(context)):
            actions.append(LogInfo(msg=[
                'SYSTEM_BRINGUP | scout member RL owns /cmd_vel; forcing ',
                'fleet start_nav2:=false to avoid Nav2/RL command conflict.',
            ]))
        if scout_owns_slam and launch_bool(start_nav2.perform(context)):
            actions.append(LogInfo(msg=[
                'SYSTEM_BRINGUP | scout owns Cartographer /map; forcing ',
                'fleet start_nav2:=false for lightweight Scout operation. ',
                'Leader/member AMCL/Nav2 remains available in their own roles.',
            ]))

        scout_robot_name = (
            follower_robot_name.perform(context)
            if role_value == 'scout' and fleet_role_value == 'follower'
            else active_scout_robot_name.perform(context)
        )

        if (
            role_value == 'scout'
            and (
                fleet_role_value == 'member'
                or launch_bool(enable_scout_failover.perform(context))
            )
        ):
            local_exploration = launch_bool(enable_exploration.perform(context))
            if local_exploration:
                actions.append(LogInfo(msg=[
                    'SYSTEM_BRINGUP | ACTIVE_SCOUT RL is in-process in '
                    'unified_field_robot; no eval_policy subprocess is started.',
                ]))
            else:
                actions.append(LogInfo(msg=[
                    'SYSTEM_BRINGUP | enable_exploration:=false -- this robot '
                    'will not publish ACTIVE_SCOUT RL commands.'
                ]))
            local_robot_name = (
                follower_robot_name.perform(context)
                if fleet_role_value == 'follower'
                else active_scout_robot_name.perform(context)
            )
            initial_field_role = (
                'FOLLOWER' if fleet_role_value == 'follower' else 'ACTIVE_SCOUT'
            )
            self_pose_topic = (
                '/burger_pose' if fleet_role_value == 'follower' else '/member_pose'
            )
            actions.append(TimerAction(
                period=2.5,
                actions=[Node(
                    package='system_bringup',
                    executable='unified_field_robot',
                    name='unified_field_robot',
                    output='screen',
                    parameters=[{
                        'robot_name': local_robot_name,
                        'fleet_role': fleet_role_value,
                        'active_scout_robot_name': active_scout_robot_name.perform(context),
                        'initial_role': initial_field_role,
                        'enable_follow_mode': True,
                        'enable_scout_mode': True,
                        'enable_recovery_mode': True,
                        'enable_localization_spin': launch_bool(
                            enable_localization_spin_on_takeover.perform(context)
                        ),
                        'enable_exploration': local_exploration,
                        'leader_pose_topic': '/leader_pose',
                        'self_pose_topic': self_pose_topic,
                        'require_localization_ready': not scout_owns_slam,
                        'localization_ready_topic': '/localization_ready',
                        'follow_distance_m': 0.70,
                        'recovery_arrival_tolerance_m': float(
                            scout_takeover_arrival_tolerance_m.perform(context)
                        ),
                        'max_xy_covariance': 0.22,
                        'max_yaw_covariance': 0.16,
                        'spin_speed_rad_s': 0.40,
                        'spin_target_angle_rad': 7.10,
                        'spin_timeout_sec': 42.0,
                        'settle_duration_sec': 3.0,
                        'max_spin_retries': 3,
                        'cmd_vel_topic': '/cmd_vel',
                        'use_stamped_cmd_vel': True,
                    }],
                    env=process_env,
                    respawn=True,
                    respawn_delay=3.0,
                )],
            ))

        if role_value == 'leader':
            leader_localization_ready_gate = (
                not launch_bool(enable_cartographer.perform(context))
                and launch_bool(leader_auto_localize.perform(context))
            )
            risk_domain_value = risk_domain_id.perform(context).strip()
            if (
                launch_bool(enable_risk_to_leader_bridge.perform(context))
                and risk_domain_value
            ):
                risk_domain = int(risk_domain_value)
                if risk_domain != domain:
                    leader_owns_map = launch_bool(
                        enable_cartographer.perform(context)
                    )
                    risk_bridge_config = write_risk_to_leader_bridge_config(
                        risk_domain,
                        domain,
                        include_map=not leader_owns_map,
                    )
                    if leader_owns_map:
                        actions.append(LogInfo(msg=[
                            'SYSTEM_BRINGUP | leader Cartographer owns /map; ',
                            'risk->leader bridge excludes /map and only carries ',
                            'risk/debug topics.',
                        ]))
                    else:
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
                virtual_env = os.environ.get('VIRTUAL_ENV', '').strip()
                python_exe = (
                    os.path.join(virtual_env, 'bin', 'python3')
                    if virtual_env else 'python3'
                )
                yolo_model_path = yolo_server_model_path.perform(context)
                if yolo_model_path and not os.path.isabs(yolo_model_path):
                    yolo_model_path = os.path.abspath(yolo_model_path)
                target_class_value = yolo_server_target_class.perform(context).strip().lower()
                target_class_args = (
                    ['--all-classes']
                    if target_class_value in ('', 'all', 'none', '-1')
                    else ['--target-class', target_class_value]
                )
                flask_yolo_server = ExecuteProcess(
                    cmd=[
                        python_exe,
                        '-m', 'flask_yolo_bridge.flask_yolo_server',
                        '--host', yolo_server_host.perform(context),
                        '--port', yolo_server_port.perform(context),
                        '--model-path', yolo_model_path,
                        *target_class_args,
                        '--device', yolo_server_device.perform(context),
                        '--half', yolo_server_half.perform(context),
                        '--fast-forward', 'true',
                        '--conf', yolo_server_conf.perform(context),
                        '--iou', '0.45',
                        '--max-det', '64',
                        '--imgsz', '960',
                        '--debug-jpeg-quality', '75',
                        '--max-capture-age-sec', '1.5',
                        '--max-queue-wait-sec', '0.05',
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
                        'yolo_node_model_path': (
                            yolo_server_model_path.perform(context)
                        ),
                        'omx_camera_index': omx_camera_index.perform(context),
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
            if launch_bool(enable_leader_shadow_follow.perform(context)):
                actions.append(TimerAction(
                    period=9.0,
                    actions=[Node(
                        package='system_bringup',
                        executable='leader_shadow_follow',
                        name='leader_shadow_follow',
                        output='screen',
                        parameters=[{
                            'enable_leader_shadow_follow': True,
                            'leader_pose_topic': '/leader_pose',
                            'active_scout_pose_topic': scout_pose_topic.perform(context),
                            'follower_scout_pose_topic': '/burger_pose',
                            'leader_goal_topic': '/fleet/leader_coord_goal',
                            'leader_cancel_topic': '/fleet/leader_nav_cancel',
                            'cmd_vel_topic': '/cmd_vel',
                            'use_stamped_cmd_vel': True,
                            'direct_shadow_cmd_vel': launch_bool(
                                leader_shadow_direct_cmd_vel.perform(context)
                            ),
                            'map_topic': '/map',
                            'failover_state_topic': '/failover/state',
                            'active_scout_id_topic': '/failover/active_scout_id',
                            'active_scout_robot_name': active_scout_robot_name.perform(context),
                            'follower_robot_name': follower_robot_name.perform(context),
                            'require_localization_ready': leader_localization_ready_gate,
                            'localization_ready_topic': '/localization_ready',
                            'scout_pose_timeout_sec': float(
                                scout_pose_timeout_sec.perform(context)
                            ),
                            'startup_grace_sec': 8.0,
                            'leader_shadow_follow_distance_m': float(
                                leader_shadow_follow_distance_m.perform(context)
                            ),
                            'leader_shadow_stop_distance_m': float(
                                leader_shadow_stop_distance_m.perform(context)
                            ),
                            'leader_shadow_resume_distance_m': float(
                                leader_shadow_resume_distance_m.perform(context)
                            ),
                            'leader_shadow_far_distance_m': float(
                                leader_shadow_far_distance_m.perform(context)
                            ),
                            'leader_shadow_max_linear_vel': float(
                                leader_shadow_max_linear_vel.perform(context)
                            ),
                            'leader_shadow_catchup_max_linear_vel': float(
                                leader_shadow_catchup_max_linear_vel.perform(context)
                            ),
                            'leader_shadow_max_angular_vel': float(
                                leader_shadow_max_angular_vel.perform(context)
                            ),
                            'leader_shadow_goal_update_period_sec': float(
                                leader_shadow_goal_update_period_sec.perform(context)
                            ),
                            'leader_shadow_goal_min_change_m': float(
                                leader_shadow_goal_min_change_m.perform(context)
                            ),
                            'leader_shadow_cmd_linear_scale': float(
                                leader_shadow_cmd_linear_scale.perform(context)
                            ),
                            'leader_shadow_cmd_angular_scale': float(
                                leader_shadow_cmd_angular_scale.perform(context)
                            ),
                            'leader_shadow_cmd_max_linear_vel': float(
                                leader_shadow_cmd_max_linear_vel.perform(context)
                            ),
                            'leader_shadow_cmd_max_angular_vel': float(
                                leader_shadow_cmd_max_angular_vel.perform(context)
                            ),
                            'leader_shadow_heading_min_motion_m': float(
                                leader_shadow_heading_min_motion_m.perform(context)
                            ),
                            'enable_leader_continuous_scan': launch_bool(
                                enable_leader_continuous_scan.perform(context)
                            ),
                            'leader_scan_topic': '/scan',
                            'leader_scan_fov_deg': float(
                                leader_scan_fov_deg.perform(context)
                            ),
                            'leader_scan_update_rate_hz': float(
                                leader_scan_update_rate_hz.perform(context)
                            ),
                            'leader_scan_timeout_sec': float(
                                leader_scan_timeout_sec.perform(context)
                            ),
                        }],
                        env=process_env,
                        respawn=True,
                        respawn_delay=3.0,
                    )],
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
                        'leader_nav_path_topic': '/plan',
                        'leader_bridged_nav_path_topic': '/leader_plan',
                        'follower_nav_path_topic': '/burger_plan',
                        'member_nav_path_topic': '/member_plan',
                        'omx_waypoint_route_topic': '/omx/waypoint_route',
                    }],
                    env=process_env,
                    respawn=True,
                    respawn_delay=3.0,
                ))
            if launch_bool(enable_scout_failover.perform(context)):
                actions.append(Node(
                    package='system_bringup',
                    executable='scout_failover_coordinator',
                    name='scout_failover_coordinator',
                    output='screen',
                    parameters=[{
                        'enable_scout_failover': True,
                        'leader_robot_name': leader_robot_name.perform(context),
                        'active_scout_robot_name': active_scout_robot_name.perform(context),
                        'follower_robot_name': follower_robot_name.perform(context),
                        'scout_liveness_topic': scout_liveness_topic.perform(context),
                        'scout_liveness_timeout_sec': float(
                            scout_liveness_timeout_sec.perform(context)
                        ),
                        'scout_failure_confirm_sec': float(
                            scout_failure_confirm_sec.perform(context)
                        ),
                        'scout_pose_topic': scout_pose_topic.perform(context),
                        'scout_pose_timeout_sec': float(
                            scout_pose_timeout_sec.perform(context)
                        ),
                        'require_bootstrap_complete': True,
                        'bootstrap_ready_topic': '/localization_ready',
                        'leader_recovery_standoff_m': float(
                            leader_recovery_standoff_m.perform(context)
                        ),
                        'leader_failure_arrival_tolerance_m': float(
                            leader_failure_arrival_tolerance_m.perform(context)
                        ),
                        'follower_recovery_standoff_m': float(
                            follower_recovery_standoff_m.perform(context)
                        ),
                        'scout_takeover_arrival_tolerance_m': float(
                            scout_takeover_arrival_tolerance_m.perform(context)
                        ),
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
                        'a scout/member instead) when turning '
                        'start_cartographer on for fleet_role:=leader.'
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
                    # In the target architecture this Scout risk-map layer
                    # owns SLAM by default; keep enable_amcl:=false
                    # underneath so there is only one map->odom source in
                    # this domain.
                    'start_robot_bringup': 'false',
                    'start_cartographer': (
                        'true' if cartographer_on else 'false'
                    ),
                    'cartographer_configuration_basename': configured_lua,
                    'start_camera': risk_start_camera,
                    'start_teleop': start_teleop.perform(context),
                    'model_path': risk_model_path.perform(context),
                    'target_class': risk_target_class.perform(context),
                    'detection_source': risk_detection_source,
                    'enable_yolo': risk_enable_yolo,
                    'external_detection_topic': (
                        external_detection_topic.perform(context)
                    ),
                    'camera_hfov_deg': leader_scan_fov_deg.perform(context),
                    'start_rviz': 'false',
                }.items(),
            ))

        if camera_sender_on:
            sender_share = get_package_share_directory('flask_yolo_bridge')
            sender_launch_path = os.path.join(
                sender_share, 'launch', 'opencv_camera_to_flask_yolo.launch.py'
            )
            role_gating_on = launch_bool(enable_scout_failover.perform(context))
            actions.append(IncludeLaunchDescription(
                PythonLaunchDescriptionSource(sender_launch_path),
                launch_arguments={
                    'device': camera_sender_device.perform(context),
                    'server_url': flask_server_url.perform(context),
                    'output_topic': external_detection_topic.perform(context),
                    'width': '1920',
                    'height': '1080',
                    'send_width': '1280',
                    'send_height': '720',
                    'camera_fps': '15.0',
                    'max_rate_hz': '5.0',
                    'http_worker_count': '1',
                    'jpeg_quality': '80',
                    'timeout_sec': '1.0',
                    'connect_timeout_sec': '0.3',
                    'read_timeout_sec': '1.8',
                    'max_http_roundtrip_sec': '2.0',
                    'max_frame_age_sec': '1.5',
                    'enable_role_gating': 'true' if role_gating_on else 'false',
                    'robot_name': scout_robot_name,
                    'role_topic': f'/{scout_robot_name}/role',
                    'initial_role_active': (
                        'true'
                        if (not role_gating_on or fleet_role_value == 'member')
                        else 'false'
                    ),
                    'active_roles': 'ACTIVE_SCOUT,SCOUT',
                }.items(),
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
                "map + camera sender) or 'leader' (fleet bringup + "
                "Jetson/OMX AIM)."
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
                'Leader DDS domain used by domain_bridge. Leave empty for '
                'single-robot scout/member testing; optional for '
                'role:=leader. Pass as main_domain_id:=<leader_domain> '
                'when bridging to a leader.'
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
                'run Cartographer here. Default false receives the '
                'Scout-owned /map via risk_domain_id and runs AMCL against '
                'it. Set true only for single-leader SLAM compatibility.'
            ),
        ),
        DeclareLaunchArgument(
            'auto_localize', default_value='true',
            choices=['true', 'false'],
            description=(
                'Passed through to follower/member AMCL global '
                'localization. Leader uses leader_auto_localize so a '
                'stationary leader does not drift by default.'
            ),
        ),
        DeclareLaunchArgument(
            'leader_auto_localize', default_value='false',
            choices=['true', 'false'],
            description=(
                'Leader fleet_role only: enable AMCL scout-pose seed plus '
                'verified in-place spin. Default false keeps the leader '
                'map pose anchored to leader_initial_x/y/yaw while the '
                'robot is standing still.'
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
            description=(
                'Scout only: turn on the Bayesian risk map stack. Default '
                'true restores Scout-owned Cartographer/risk processing; '
                'local YOLO/camera capture remain disabled when '
                'start_camera_sender:=true.'
            ),
        ),
        DeclareLaunchArgument(
            'start_cartographer', default_value='true',
            choices=['true', 'false'],
            description=(
                'Scout only: let the risk map launch own Cartographer '
                'SLAM/TF. Default true for the Domain 22 authoritative map; '
                'requires '
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
            'start_camera', default_value='false',
            choices=['true', 'false'],
            description=(
                'Scout only: start the USB camera feeding the local risk map '
                'YOLO detector. Default false because start_camera_sender '
                'owns the camera in the lightweight Scout path.'
            ),
        ),
        DeclareLaunchArgument(
            'start_teleop', default_value='false',
            choices=['true', 'false'],
            description=(
                'Scout only: pass through to the risk SLAM launch to start '
                'turtlebot3_teleop. Default false because teleop_keyboard '
                'needs an interactive terminal.'
            ),
        ),
        DeclareLaunchArgument('risk_model_path', default_value='model/best.engine'),
        DeclareLaunchArgument(
            'risk_target_class',
            default_value='-1',
            description=(
                'Scout only: target class accepted by the risk map. -1 '
                'accepts all YOLO classes while debugging best engine class ids.'
            ),
        ),
        DeclareLaunchArgument(
            'detection_source', default_value='flask_topic',
            description=(
                'Scout only, passed through to real_robot_risk_slam.'
                'launch.py. Default flask_topic consumes detections published by '
                'flask_yolo_bridge/opencv_camera_to_flask_yolo.'
                'launch.py (offloaded to the leader Jetson). Set local_yolo '
                'only for explicit Scout-local inference debugging.'
            ),
        ),
        DeclareLaunchArgument(
            'enable_yolo', default_value='false',
            choices=['true', 'false'],
            description=(
                'Scout only: run YOLO inference inside the risk map node '
                'itself. Default false; the leader Jetson Flask server does '
                'inference in the lightweight Scout path.'
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
            'start_camera_sender', default_value='true',
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
            default_value='true',
            choices=['true', 'false'],
            description=(
                'Leader role only: bridge Scout/risk /map and risk debug '
                'topics from risk_domain_id to leader. With the default '
                'leader enable_cartographer:=false this is the critical '
                '22->20 shared-map path.'
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
            'enable_scout_failover',
            default_value='true',
            choices=['true', 'false'],
            description='Enable active-scout liveness watchdog and follower takeover orchestration.',
        ),
        DeclareLaunchArgument('leader_robot_name', default_value='leader'),
        DeclareLaunchArgument('active_scout_robot_name', default_value='scout22'),
        DeclareLaunchArgument('follower_robot_name', default_value='follower21'),
        DeclareLaunchArgument(
            'scout_liveness_topic',
            default_value='/scout/signal',
            description='Leader-domain heartbeat topic bridged from the active scout.',
        ),
        DeclareLaunchArgument(
            'scout_liveness_timeout_sec',
            default_value='2.0',
            description='Seconds without scout heartbeat before suspected-dead.',
        ),
        DeclareLaunchArgument(
            'scout_failure_confirm_sec',
            default_value='0.5',
            description='Additional confirmation time before declaring scout dead.',
        ),
        DeclareLaunchArgument(
            'scout_pose_topic',
            default_value='/member_pose',
            description='Leader-domain active-scout map-frame pose topic.',
        ),
        DeclareLaunchArgument(
            'scout_pose_timeout_sec',
            default_value='5.0',
            description='Maximum age of scout pose allowed for failure target freeze.',
        ),
        DeclareLaunchArgument(
            'enable_leader_shadow_follow',
            default_value='true',
            choices=['true', 'false'],
            description='Leader role only: move behind the active scout during normal operation.',
        ),
        DeclareLaunchArgument(
            'leader_shadow_direct_cmd_vel',
            default_value='true',
            choices=['true', 'false'],
            description=(
                'Leader role only: use continuous /cmd_vel for normal '
                'shadow follow instead of repeatedly preempting Nav2 goals.'
            ),
        ),
        DeclareLaunchArgument(
            'leader_shadow_follow_distance_m',
            default_value='2.8',
            description='Leader shadow target distance behind active scout movement direction.',
        ),
        DeclareLaunchArgument(
            'leader_shadow_stop_distance_m',
            default_value='2.2',
            description='Leader stops shadow goal updates when closer than this to the active scout.',
        ),
        DeclareLaunchArgument(
            'leader_shadow_resume_distance_m',
            default_value='3.0',
            description='Leader resumes shadow follow when scout distance reaches this hysteresis threshold.',
        ),
        DeclareLaunchArgument(
            'leader_shadow_far_distance_m',
            default_value='4.5',
            description='Distance where shadow follow permits catch-up speed limits.',
        ),
        DeclareLaunchArgument(
            'leader_shadow_max_linear_vel',
            default_value='0.38',
            description='Best-effort DWB linear velocity cap while shadow following.',
        ),
        DeclareLaunchArgument(
            'leader_shadow_catchup_max_linear_vel',
            default_value='0.46',
            description='Best-effort DWB linear velocity cap when the leader is far behind the active scout.',
        ),
        DeclareLaunchArgument(
            'leader_shadow_max_angular_vel',
            default_value='0.85',
            description='Best-effort DWB angular velocity cap while shadow following.',
        ),
        DeclareLaunchArgument(
            'leader_shadow_goal_update_period_sec',
            default_value='2.0',
            description='Minimum time between leader shadow Nav2 goal updates.',
        ),
        DeclareLaunchArgument(
            'leader_shadow_goal_min_change_m',
            default_value='0.60',
            description='Minimum shadow target displacement before sending another leader goal.',
        ),
        DeclareLaunchArgument(
            'leader_shadow_cmd_linear_scale',
            default_value='3.0',
            description=(
                'Direct /cmd_vel compensation for a loaded leader that only '
                'achieves a fraction of commanded linear speed.'
            ),
        ),
        DeclareLaunchArgument(
            'leader_shadow_cmd_angular_scale',
            default_value='1.0',
            description='Direct /cmd_vel angular compensation scale.',
        ),
        DeclareLaunchArgument(
            'leader_shadow_cmd_max_linear_vel',
            default_value='0.75',
            description='Hard cap for compensated direct /cmd_vel linear.x.',
        ),
        DeclareLaunchArgument(
            'leader_shadow_cmd_max_angular_vel',
            default_value='1.20',
            description='Hard cap for compensated direct /cmd_vel angular.z.',
        ),
        DeclareLaunchArgument(
            'leader_shadow_heading_min_motion_m',
            default_value='0.15',
            description='Scout displacement required before updating movement-heading estimate.',
        ),
        DeclareLaunchArgument(
            'enable_leader_continuous_scan',
            default_value='true',
            choices=['true', 'false'],
            description='Publish leader scan freshness/FOV state independently from navigation.',
        ),
        DeclareLaunchArgument(
            'leader_scan_fov_deg',
            default_value='60.0',
            description='Risk/visibility scan FOV in degrees; Nav2 obstacle LaserScan is not clipped.',
        ),
        DeclareLaunchArgument(
            'leader_scan_update_rate_hz',
            default_value='10.0',
            description='Leader scan freshness/status update rate.',
        ),
        DeclareLaunchArgument(
            'leader_scan_timeout_sec',
            default_value='1.0',
            description='Maximum leader scan age before scan state becomes stale.',
        ),
        DeclareLaunchArgument(
            'leader_recovery_standoff_m',
            default_value='0.70',
            description='Leader goal offset behind failed scout pose.',
        ),
        DeclareLaunchArgument(
            'leader_failure_arrival_tolerance_m',
            default_value='0.80',
            description='If leader is already this close to failure pose, skip recovery goal and only cancel shadow goal.',
        ),
        DeclareLaunchArgument(
            'follower_recovery_standoff_m',
            default_value='0.15',
            description='Follower goal offset behind failed scout pose.',
        ),
        DeclareLaunchArgument(
            'scout_takeover_arrival_tolerance_m',
            default_value='0.40',
            description='Follower distance to failed scout pose required for takeover.',
        ),
        DeclareLaunchArgument(
            'enable_localization_spin_on_takeover',
            default_value='true',
            choices=['true', 'false'],
            description='Follower-side takeover agent performs 360deg AMCL spin when covariance is poor.',
        ),
        DeclareLaunchArgument(
            'enable_exploration',
            default_value='true',
            choices=['true', 'false'],
            description=(
                'Enable the deterministic in-process ACTIVE_SCOUT SAC runtime. '
                'It is role-gated and never starts a remote or eval_policy '
                'subprocess; false leaves ACTIVE_SCOUT motion stopped.'
            ),
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
            default_value='model/best.engine',
            description=(
                'Leader role only: YOLO model path for flask_yolo_server. '
                'Use model/best.engine on Jetson to avoid PyTorch CUDA sm_87 issues.'
            ),
        ),
        DeclareLaunchArgument(
            'yolo_server_target_class',
            default_value='all',
            description=(
                'Leader role only: class id for flask_yolo_server. Use '
                '"all" to disable class filtering while debugging best engine.'
            ),
        ),
        DeclareLaunchArgument(
            'yolo_server_conf',
            default_value='0.20',
            description=(
                'Leader role only: YOLO confidence threshold for '
                'flask_yolo_server. Lower this only while debugging misses.'
            ),
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
            'omx_camera_index', default_value='0',
            description='Leader role only: OpenCV camera index for OMX debug/YOLO video.',
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
            default_value='true',
            choices=['true', 'false'],
            description=(
                'Leader role only: enable the OMX MJPEG debug stream shown '
                'inside the integrated dashboard.'
            ),
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
            description='Leader role only: run the integrated leader dashboard.',
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
