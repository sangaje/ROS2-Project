from launch_ros.parameter_descriptions import ParameterValue


def explorer_params(use_sim_time, robot_frame, cmd_vel_stamped, prefer_internal_grid_planner):
    return {
        'version_name': 'v31_cartographer_external_slam',
        'use_sim_time': ParameterValue(use_sim_time, value_type=bool),

        # Frames / topics
        'map_topic': '/map',
        'scan_topic': '/scan',
        'cmd_vel_topic': '/cmd_vel',
        'global_frame': 'map',
        'robot_frame': robot_frame,
        'cmd_vel_stamped': ParameterValue(cmd_vel_stamped, value_type=bool),
        'path_topic': '/visibility_explorer/path',
        'plan_alias_topic': '/plan',
        'scan_viz_topic': '/scan_reliable',
        'publish_scan_reliable': True,
        'scan_sub_best_effort': True,

        # Main loop
        'planning_period': 0.28,
        'map_stable_time': 0.15,
        'max_plan_trials': 24,

        # Map interpretation
        'free_threshold': 20,
        'occupied_threshold': 65,
        'unknown_is_obstacle_for_clearance': False,

        # Frontier / candidate generation
        'min_frontier_cluster_size': 4,
        'frontier_goal_offset': 0.55,
        'candidate_stride_cells': 3,
        'min_goal_distance': 0.22,
        'max_goal_distance': 5.5,
        'max_goal_bearing_deg': 170.0,

        # Safety / clearance
        'robot_radius': 0.23,
        'min_goal_clearance': 0.28,
        'min_path_clearance': 0.16,
        'front_stop_distance': 0.24,
        'front_slow_distance': 0.42,
        'side_stop_distance': 0.16,

        # Visual coverage / NBV
        'view_fov_deg': 60.0,
        'view_ray_count': 31,
        'view_max_range': 3.5,
        'coverage_robot_radius': 0.28,
        'coverage_publish_period': 0.6,
        'visual_unchecked_threshold': 95,
        'coverage_candidate_max_checked': 95,

        # Scoring: v31 keeps v30 unmapped priority but still values coverage.
        'unknown_gain_weight': 12.0,
        'visual_gain_weight': 3.0,
        'frontier_gain_weight': 1.8,
        'clearance_weight': 0.55,
        'distance_weight': 0.20,
        'blacklist_weight': 5.0,
        'heading_weight': 0.10,
        'coverage_source_score_bonus': 20.0,

        # Planner/execution. v31 can run with no Nav2: internal grid A* is available.
        'planner_id': '',
        'enable_internal_grid_planner': True,
        'prefer_internal_grid_planner': ParameterValue(prefer_internal_grid_planner, value_type=bool),
        'internal_planner_max_expansions': 60000,
        'nav2_result_timeout': 75.0,
        'stuck_timeout': 10.0,
        'stuck_min_progress': 0.10,
        'goal_success_radius': 0.35,
        'blacklist_radius': 0.55,
        'blacklist_duration': 45.0,
        'execute_with_nav2_navigator': False,
        'direct_follow_lookahead_distance': 0.35,
        'direct_follow_max_linear_speed': 0.20,
        'direct_follow_min_linear_speed': 0.035,
        'direct_follow_max_angular_speed': 1.20,
        'direct_follow_angular_gain': 1.8,
        'direct_follow_stuck_timeout': 3.8,
        'direct_follow_progress_epsilon': 0.04,
        'direct_goal_tolerance': 0.18,

        # View check after arrival
        'enable_sector_scan': False,
        'sector_scan_angle_deg': 45.0,
        'sector_scan_angular_speed': 0.75,
        'enable_view_yaw_align': True,
        'view_align_tolerance_deg': 12.0,
        'view_align_timeout': 0.8,
        'view_align_angular_speed': 0.75,

        # SLAM warmup: useful for real robot even with Cartographer.
        'enable_slam_warmup': True,
        'slam_warmup_min_duration': 4.0,
        'slam_warmup_max_duration': 8.0,
        'slam_warmup_min_known_cells': 350,
        'slam_warmup_min_free_cells': 150,
        'slam_warmup_angular_speed': 0.16,
        'map_health_log_period': 2.0,

        # LiDAR probe for unmapped/open areas.
        'enable_short_lidar_probe_fallback': True,
        'enable_unknown_first_probe': True,
        'enable_scan_open_space_probe': True,
        'allow_probe_during_map_stabilizing': True,
        'prefer_scan_probe_before_nav2': True,
        'coverage_priority_over_probe': True,
        'unmapped_priority_over_coverage': True,
        'unmapped_priority_probe_gain': 7,
        'unknown_first_min_gain': 2,
        'nav2_min_goal_distance': 0.0,
        'probe_success_radius': 0.12,
        'probe_min_distance': 0.40,
        'probe_fov_deg': 130.0,
        'probe_distance': 0.75,
        'probe_min_front_clearance': 0.46,
        'probe_side_clearance': 0.20,
        'probe_linear_speed': 0.16,
        'probe_angular_gain': 1.8,
        'probe_timeout': 4.0,
        'probe_fail_cooldown': 2.0,
        'probe_safety_margin': 0.05,
        'probe_fail_blacklist_radius': 0.60,
        'planner_first_when_probe_blocked': True,
        'micro_rotate_on_total_failure': True,
        'micro_rotate_duration': 0.45,
        'micro_rotate_speed': 0.35,
    }

