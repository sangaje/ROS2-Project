from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    use_sim_time = LaunchConfiguration('use_sim_time')
    tb3_model = LaunchConfiguration('tb3_model')
    slam_backend = LaunchConfiguration('slam_backend')
    robot_frame = LaunchConfiguration('robot_frame')
    global_frame = LaunchConfiguration('global_frame')
    auto_start = LaunchConfiguration('auto_start')
    auto_mapper_delay_sec = LaunchConfiguration('auto_mapper_delay_sec')
    max_linear_x = LaunchConfiguration('max_linear_x')
    max_angular_z = LaunchConfiguration('max_angular_z')
    front_stop_distance_m = LaunchConfiguration('front_stop_distance_m')
    front_slow_distance_m = LaunchConfiguration('front_slow_distance_m')
    planning_period_sec = LaunchConfiguration('planning_period_sec')
    replan_if_goal_older_sec = LaunchConfiguration('replan_if_goal_older_sec')
    goal_lock_min_sec = LaunchConfiguration('goal_lock_min_sec')
    lidar_arc_avoidance = LaunchConfiguration('lidar_arc_avoidance')
    path_lookahead_m = LaunchConfiguration('path_lookahead_m')
    min_linear_x = LaunchConfiguration('min_linear_x')
    cruise_linear_x = LaunchConfiguration('cruise_linear_x')
    goal_progress_stall_sec = LaunchConfiguration('goal_progress_stall_sec')
    preserve_goal_on_map_resize = LaunchConfiguration('preserve_goal_on_map_resize')
    lock_goal_until_reached = LaunchConfiguration('lock_goal_until_reached')
    conservative_astar = LaunchConfiguration('conservative_astar')
    path_min_clearance_m = LaunchConfiguration('path_min_clearance_m')
    path_prefer_clearance_m = LaunchConfiguration('path_prefer_clearance_m')
    path_wall_cost_weight = LaunchConfiguration('path_wall_cost_weight')
    path_unknown_cost = LaunchConfiguration('path_unknown_cost')
    path_region_boundary_cost = LaunchConfiguration('path_region_boundary_cost')
    path_diagonal_cost_multiplier = LaunchConfiguration('path_diagonal_cost_multiplier')
    candidate_min_clearance_m = LaunchConfiguration('candidate_min_clearance_m')
    dense_coverage_marking = LaunchConfiguration('dense_coverage_marking')
    coverage_brush_radius_m = LaunchConfiguration('coverage_brush_radius_m')
    coverage_robot_radius_m = LaunchConfiguration('coverage_robot_radius_m')
    coverage_downsample_angle_step_deg = LaunchConfiguration('coverage_downsample_angle_step_deg')
    coverage_front_only = LaunchConfiguration('coverage_front_only')
    coverage_fov_deg = LaunchConfiguration('coverage_fov_deg')
    coverage_yaw_offset_deg = LaunchConfiguration('coverage_yaw_offset_deg')
    coverage_mark_robot_footprint = LaunchConfiguration('coverage_mark_robot_footprint')
    allow_cross_region_view_gain = LaunchConfiguration('allow_cross_region_view_gain')
    w_cross_region_unknown = LaunchConfiguration('w_cross_region_unknown')
    w_cross_region_frontier = LaunchConfiguration('w_cross_region_frontier')
    frontier_candidate_sampling = LaunchConfiguration('frontier_candidate_sampling')
    frontier_candidate_max_count = LaunchConfiguration('frontier_candidate_max_count')
    frontier_candidate_min_unknown_neighbors = LaunchConfiguration('frontier_candidate_min_unknown_neighbors')
    view_fov_deg = LaunchConfiguration('view_fov_deg')
    path_sparsify_max_step_m = LaunchConfiguration('path_sparsify_max_step_m')
    region_completion_min_active_sec = LaunchConfiguration('region_completion_min_active_sec')
    region_completion_min_cells = LaunchConfiguration('region_completion_min_cells')
    mission_lock_skip_region_replan = LaunchConfiguration('mission_lock_skip_region_replan')
    no_stop_on_missing_region_stats = LaunchConfiguration('no_stop_on_missing_region_stats')
    emergency_stop_distance_m = LaunchConfiguration('emergency_stop_distance_m')
    fast_cruise_when_locked = LaunchConfiguration('fast_cruise_when_locked')
    house_spawn_x = LaunchConfiguration('house_spawn_x')
    house_spawn_y = LaunchConfiguration('house_spawn_y')
    house_spawn_z = LaunchConfiguration('house_spawn_z')
    house_spawn_yaw = LaunchConfiguration('house_spawn_yaw')
    creep_linear_x = LaunchConfiguration('creep_linear_x')
    linear_k = LaunchConfiguration('linear_k')
    angular_k = LaunchConfiguration('angular_k')
    linear_accel_limit = LaunchConfiguration('linear_accel_limit')
    angular_accel_limit = LaunchConfiguration('angular_accel_limit')
    waypoint_abandon_enabled = LaunchConfiguration('waypoint_abandon_enabled')
    waypoint_abandon_front_distance_m = LaunchConfiguration('waypoint_abandon_front_distance_m')
    waypoint_abandon_time_sec = LaunchConfiguration('waypoint_abandon_time_sec')
    waypoint_abandon_cooldown_sec = LaunchConfiguration('waypoint_abandon_cooldown_sec')
    abandoned_goal_radius_m = LaunchConfiguration('abandoned_goal_radius_m')
    abandoned_goal_memory_sec = LaunchConfiguration('abandoned_goal_memory_sec')
    abandon_turn_speed = LaunchConfiguration('abandon_turn_speed')
    idle_spin_enabled = LaunchConfiguration('idle_spin_enabled')
    reopen_completed_region_on_frontier = LaunchConfiguration('reopen_completed_region_on_frontier')
    reopen_frontier_margin = LaunchConfiguration('reopen_frontier_margin')
    keep_moving_when_no_goal = LaunchConfiguration('keep_moving_when_no_goal')
    clear_completed_regions_when_no_goal = LaunchConfiguration('clear_completed_regions_when_no_goal')
    search_motion_linear_x = LaunchConfiguration('search_motion_linear_x')
    search_motion_angular_z = LaunchConfiguration('search_motion_angular_z')
    search_motion_front_clearance_m = LaunchConfiguration('search_motion_front_clearance_m')
    search_motion_side_balance_gain = LaunchConfiguration('search_motion_side_balance_gain')
    search_motion_min_turn_z = LaunchConfiguration('search_motion_min_turn_z')

    core_launch = PathJoinSubstitution([
        FindPackageShare('tb3_region_mapper'), 'launch', 'sim_house_region_graph.launch.py'
    ])
    mapper_launch = PathJoinSubstitution([
        FindPackageShare('tb3_region_mapper'), 'launch', 'region_auto_mapper.launch.py'
    ])

    return LaunchDescription([
        DeclareLaunchArgument('use_sim_time', default_value='true'),
        DeclareLaunchArgument('tb3_model', default_value='burger'),
        DeclareLaunchArgument('slam_backend', default_value='cartographer'),
        DeclareLaunchArgument('robot_frame', default_value='base_footprint'),
        DeclareLaunchArgument('global_frame', default_value='map'),
        DeclareLaunchArgument('auto_start', default_value='true'),
        DeclareLaunchArgument('auto_mapper_delay_sec', default_value='14.0'),
        DeclareLaunchArgument('max_linear_x', default_value='0.20'),
        DeclareLaunchArgument('max_angular_z', default_value='0.85'),
        DeclareLaunchArgument('front_stop_distance_m', default_value='0.30'),
        DeclareLaunchArgument('front_slow_distance_m', default_value='0.68'),
        DeclareLaunchArgument('planning_period_sec', default_value='2.50'),
        DeclareLaunchArgument('replan_if_goal_older_sec', default_value='180.0'),
        DeclareLaunchArgument('goal_lock_min_sec', default_value='90.0'),
        DeclareLaunchArgument('lidar_arc_avoidance', default_value='true'),
        DeclareLaunchArgument('path_lookahead_m', default_value='0.80'),
        DeclareLaunchArgument('min_linear_x', default_value='0.05'),
        DeclareLaunchArgument('cruise_linear_x', default_value='0.16'),
        DeclareLaunchArgument('goal_progress_stall_sec', default_value='25.0'),
        DeclareLaunchArgument('preserve_goal_on_map_resize', default_value='true'),
        DeclareLaunchArgument('lock_goal_until_reached', default_value='true'),
        DeclareLaunchArgument('conservative_astar', default_value='true'),
        DeclareLaunchArgument('path_min_clearance_m', default_value='0.40'),
        DeclareLaunchArgument('path_prefer_clearance_m', default_value='0.95'),
        DeclareLaunchArgument('path_wall_cost_weight', default_value='28.0'),
        DeclareLaunchArgument('path_unknown_cost', default_value='100.0'),
        DeclareLaunchArgument('path_region_boundary_cost', default_value='8.0'),
        DeclareLaunchArgument('path_diagonal_cost_multiplier', default_value='1.08'),
        DeclareLaunchArgument('candidate_min_clearance_m', default_value='0.40'),
        DeclareLaunchArgument('dense_coverage_marking', default_value='true'),
        DeclareLaunchArgument('coverage_brush_radius_m', default_value='0.18'),
        DeclareLaunchArgument('coverage_robot_radius_m', default_value='0.26'),
        DeclareLaunchArgument('coverage_downsample_angle_step_deg', default_value='1.0'),
        DeclareLaunchArgument('coverage_front_only', default_value='true'),
        DeclareLaunchArgument('coverage_fov_deg', default_value='90.0'),
        DeclareLaunchArgument('coverage_yaw_offset_deg', default_value='0.0'),
        DeclareLaunchArgument('coverage_mark_robot_footprint', default_value='false'),
        DeclareLaunchArgument('allow_cross_region_view_gain', default_value='true'),
        DeclareLaunchArgument('w_cross_region_unknown', default_value='4.5'),
        DeclareLaunchArgument('w_cross_region_frontier', default_value='3.5'),
        DeclareLaunchArgument('frontier_candidate_sampling', default_value='true'),
        DeclareLaunchArgument('frontier_candidate_max_count', default_value='260'),
        DeclareLaunchArgument('frontier_candidate_min_unknown_neighbors', default_value='1'),
        DeclareLaunchArgument('view_fov_deg', default_value='100.0'),
        DeclareLaunchArgument('path_sparsify_max_step_m', default_value='0.35'),
        DeclareLaunchArgument('region_completion_min_active_sec', default_value='18.0'),
        DeclareLaunchArgument('region_completion_min_cells', default_value='160'),
        DeclareLaunchArgument('waypoint_abandon_enabled', default_value='true'),
        DeclareLaunchArgument('waypoint_abandon_front_distance_m', default_value='0.24'),
        DeclareLaunchArgument('waypoint_abandon_time_sec', default_value='0.70'),
        DeclareLaunchArgument('waypoint_abandon_cooldown_sec', default_value='2.0'),
        DeclareLaunchArgument('abandoned_goal_radius_m', default_value='0.70'),
        DeclareLaunchArgument('abandoned_goal_memory_sec', default_value='55.0'),
        DeclareLaunchArgument('abandon_turn_speed', default_value='0.38'),
        DeclareLaunchArgument('idle_spin_enabled', default_value='false'),
        DeclareLaunchArgument('reopen_completed_region_on_frontier', default_value='true'),
        DeclareLaunchArgument('reopen_frontier_margin', default_value='1.15'),
        DeclareLaunchArgument('keep_moving_when_no_goal', default_value='true'),
        DeclareLaunchArgument('clear_completed_regions_when_no_goal', default_value='true'),
        DeclareLaunchArgument('search_motion_linear_x', default_value='0.055'),
        DeclareLaunchArgument('search_motion_angular_z', default_value='0.32'),
        DeclareLaunchArgument('search_motion_front_clearance_m', default_value='0.48'),
        DeclareLaunchArgument('search_motion_side_balance_gain', default_value='0.45'),
        DeclareLaunchArgument('search_motion_min_turn_z', default_value='0.18'),
        DeclareLaunchArgument('mission_lock_skip_region_replan', default_value='true'),
        DeclareLaunchArgument('no_stop_on_missing_region_stats', default_value='true'),
        DeclareLaunchArgument('emergency_stop_distance_m', default_value='0.18'),
        DeclareLaunchArgument('fast_cruise_when_locked', default_value='false'),
        DeclareLaunchArgument('creep_linear_x', default_value='0.04'),
        DeclareLaunchArgument('linear_k', default_value='1.20'),
        DeclareLaunchArgument('angular_k', default_value='1.45'),
        DeclareLaunchArgument('linear_accel_limit', default_value='0.60'),
        DeclareLaunchArgument('angular_accel_limit', default_value='2.20'),
        DeclareLaunchArgument('house_spawn_x', default_value='-2.0'),
        DeclareLaunchArgument('house_spawn_y', default_value='-0.5'),
        DeclareLaunchArgument('house_spawn_z', default_value='0.01'),
        DeclareLaunchArgument('house_spawn_yaw', default_value='0.0'),

        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(core_launch),
            launch_arguments={
                'use_sim_time': use_sim_time,
                'tb3_model': tb3_model,
                'slam_backend': slam_backend,
                'robot_frame': robot_frame,
                'global_frame': global_frame,
                'use_rviz': 'false',
                'house_spawn_x': house_spawn_x,
                'house_spawn_y': house_spawn_y,
                'house_spawn_z': house_spawn_z,
                'house_spawn_yaw': house_spawn_yaw,
            }.items(),
        ),

        TimerAction(
            period=auto_mapper_delay_sec,
            actions=[
                IncludeLaunchDescription(
                    PythonLaunchDescriptionSource(mapper_launch),
                    launch_arguments={
                        'use_sim_time': use_sim_time,
                        'auto_start': auto_start,
                        'cmd_vel_topic': '/cmd_vel',
                        'global_frame': global_frame,
                        'robot_frame': robot_frame,
                        'max_linear_x': max_linear_x,
                        'max_angular_z': max_angular_z,
                        'front_stop_distance_m': front_stop_distance_m,
                        'front_slow_distance_m': front_slow_distance_m,
                        'planning_period_sec': planning_period_sec,
                        'replan_if_goal_older_sec': replan_if_goal_older_sec,
                        'goal_lock_min_sec': goal_lock_min_sec,
                        'lidar_arc_avoidance': lidar_arc_avoidance,
                        'path_lookahead_m': path_lookahead_m,
                        'min_linear_x': min_linear_x,
                        'creep_linear_x': creep_linear_x,
                        'cruise_linear_x': cruise_linear_x,
                        'linear_k': linear_k,
                        'angular_k': angular_k,
                        'linear_accel_limit': linear_accel_limit,
                        'angular_accel_limit': angular_accel_limit,
                        'goal_progress_stall_sec': goal_progress_stall_sec,
                        'preserve_goal_on_map_resize': preserve_goal_on_map_resize,
                        'lock_goal_until_reached': lock_goal_until_reached,
                        'conservative_astar': conservative_astar,
                        'path_min_clearance_m': path_min_clearance_m,
                        'path_prefer_clearance_m': path_prefer_clearance_m,
                        'path_wall_cost_weight': path_wall_cost_weight,
                        'path_unknown_cost': path_unknown_cost,
                        'path_region_boundary_cost': path_region_boundary_cost,
                        'path_diagonal_cost_multiplier': path_diagonal_cost_multiplier,
                        'candidate_min_clearance_m': candidate_min_clearance_m,
                        'dense_coverage_marking': dense_coverage_marking,
                        'coverage_brush_radius_m': coverage_brush_radius_m,
                        'coverage_robot_radius_m': coverage_robot_radius_m,
                        'coverage_downsample_angle_step_deg': coverage_downsample_angle_step_deg,
                        'coverage_front_only': coverage_front_only,
                        'coverage_fov_deg': coverage_fov_deg,
                        'coverage_yaw_offset_deg': coverage_yaw_offset_deg,
                        'coverage_mark_robot_footprint': coverage_mark_robot_footprint,
                        'allow_cross_region_view_gain': allow_cross_region_view_gain,
                        'w_cross_region_unknown': w_cross_region_unknown,
                        'w_cross_region_frontier': w_cross_region_frontier,
                        'frontier_candidate_sampling': frontier_candidate_sampling,
                        'frontier_candidate_max_count': frontier_candidate_max_count,
                        'frontier_candidate_min_unknown_neighbors': frontier_candidate_min_unknown_neighbors,
                        'view_fov_deg': view_fov_deg,
                        'path_sparsify_max_step_m': path_sparsify_max_step_m,
                        'region_completion_min_active_sec': region_completion_min_active_sec,
                        'region_completion_min_cells': region_completion_min_cells,
                        'mission_lock_skip_region_replan': mission_lock_skip_region_replan,
                        'no_stop_on_missing_region_stats': no_stop_on_missing_region_stats,
                        'emergency_stop_distance_m': emergency_stop_distance_m,
                        'fast_cruise_when_locked': fast_cruise_when_locked,
                        'waypoint_abandon_enabled': waypoint_abandon_enabled,
                        'waypoint_abandon_front_distance_m': waypoint_abandon_front_distance_m,
                        'waypoint_abandon_time_sec': waypoint_abandon_time_sec,
                        'waypoint_abandon_cooldown_sec': waypoint_abandon_cooldown_sec,
                        'abandoned_goal_radius_m': abandoned_goal_radius_m,
                        'abandoned_goal_memory_sec': abandoned_goal_memory_sec,
                        'abandon_turn_speed': abandon_turn_speed,
                        'idle_spin_enabled': idle_spin_enabled,
                        'reopen_completed_region_on_frontier': reopen_completed_region_on_frontier,
                        'reopen_frontier_margin': reopen_frontier_margin,
                        'keep_moving_when_no_goal': keep_moving_when_no_goal,
                        'clear_completed_regions_when_no_goal': clear_completed_regions_when_no_goal,
                        'search_motion_linear_x': search_motion_linear_x,
                        'search_motion_angular_z': search_motion_angular_z,
                        'search_motion_front_clearance_m': search_motion_front_clearance_m,
                        'search_motion_side_balance_gain': search_motion_side_balance_gain,
                        'search_motion_min_turn_z': search_motion_min_turn_z,
                    }.items(),
                )
            ],
        ),
    ])
