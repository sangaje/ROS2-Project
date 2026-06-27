from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    use_sim_time = LaunchConfiguration('use_sim_time')
    map_topic = LaunchConfiguration('map_topic')
    robot_frame = LaunchConfiguration('robot_frame')
    global_frame = LaunchConfiguration('global_frame')

    return LaunchDescription([
        DeclareLaunchArgument('use_sim_time', default_value='true'),
        DeclareLaunchArgument('map_topic', default_value='/map'),
        DeclareLaunchArgument('global_frame', default_value='map'),
        DeclareLaunchArgument('robot_frame', default_value='base_footprint'),

        Node(
            package='tb3_region_mapper',
            executable='slam_region_graph_node',
            name='slam_region_graph',
            output='screen',
            parameters=[{
                'use_sim_time': use_sim_time,
                'map_topic': map_topic,
                'global_frame': global_frame,
                'robot_frame': robot_frame,

                # Update cadence. Region graph is intentionally slower than control/planning.
                'timer_period': 0.25,
                'region_update_period': 1.0,
                'map_stable_time': 0.60,
                'force_update_without_map_delta': True,

                # Occupancy interpretation.
                'free_threshold': 62,
                'occupied_threshold': 65,
                'enable_adaptive_occupancy_filter': False,
                'adaptive_occupied_min': 45,
                'adaptive_occupied_max': 78,
                'adaptive_occupied_margin': 2,
                'enable_local_dark_obstacle_filter': False,
                'dark_obstacle_min_value': 55,
                'dark_obstacle_local_contrast': 18,
                'dark_obstacle_neighbor_radius_m': 0.12,
                'wall_inflation_radius_m': 0.02,
                'obstacle_min_cluster_cells': 3,
                'enable_iterative_highpass_wall_filter': True,
                'highpass_wall_iterations': 4,
                'highpass_smooth_radius_m': 0.18,
                'highpass_radius_growth_m': 0.04,
                'highpass_wall_min_value': 52,
                'highpass_wall_contrast': 12,
                'highpass_wall_votes_min': 3,
                'highpass_hysteresis_min_value': 45,
                'known_non_obstacle_is_free': True,
                'free_mask_denoise_iterations': 6,
                'free_fill_neighbor_min': 4,
                'free_keep_neighbor_min': 1,
                'fill_unknown_holes_as_free': True,
                'unknown_hole_fill_max_area_m2': 1.20,
                'unknown_hole_fill_min_free_boundary_ratio': 0.58,
                'unknown_hole_fill_max_occ_boundary_ratio': 0.22,
                'region_dense_fill_iterations': 12,
                'region_dense_fill_neighbor_min': 2,
                'use_reachable_only': True,
                'fallback_to_all_free_without_tf': True,

                # Region filtering.
                'min_region_area_m2': 0.10,
                'min_region_cells': 12,
                'region_connectivity_8': False,
                'max_regions': 80,
                'min_frontier_cluster_size': 4,

                # Approximate GVD + bottleneck cuts.
                'enable_gvd': True,
                'enable_bottleneck_cuts': True,
                'gvd_min_clearance': 0.18,
                'door_clearance_min': 0.18,
                'door_clearance_max': 0.55,
                'max_gateway_cuts': 20,
                'cut_line_half_length': 0.62,
                'cut_line_width': 0.10,
                'cut_test_min_component_area_m2': 0.12,
                'cut_test_max_candidates': 32,
                'enable_low_clearance_doorway_cuts': True,
                'doorway_cut_clearance_min': 0.16,
                'doorway_cut_clearance_max': 0.42,
                'doorway_cut_local_min_margin': 0.035,
                'doorway_cut_min_cluster_cells': 2,
                'doorway_cut_max_candidates': 40,
                'doorway_cut_duplicate_distance_m': 0.32,
                'doorway_cut_force_half_length_m': 0.68,
                'doorway_cut_force_width_m': 0.12,

                # Main room/zone split. Increase room_seed_erosion_radius_m
                # when rooms stay merged through doors; decrease it when
                # too many small fragments appear.
                'enable_morphological_room_split': True,
                'room_seed_erosion_radius_m': 0.34,
                'room_seed_min_area_m2': 0.10,
                'room_seed_min_cells': 18,
                'room_split_connectivity_8': False,
                'region_separator_width_cells': 1,
                'region_separator_max_clearance_m': 0.55,
                'merge_tiny_split_regions': True,
                'tiny_split_region_area_m2': 0.08,
                'min_morphological_regions': 2,
                'use_clearance_priority_watershed': True,
                'watershed_conflict_clearance_max_m': 0.55,
                'watershed_separator_width_cells': 1,

                # Online tracking/state classification.
                'region_match_max_distance': 0.85,
                'region_match_area_ratio_min': 0.25,
                'stable_confirm_updates': 3,
                'open_unknown_boundary_ratio': 0.28,
                'stable_unknown_boundary_ratio': 0.16,
                'stable_closure_min': 0.50,
                'corridor_elongation_min': 4.5,

                # Visualization.
                'region_map_id_only': True,
                'publish_gvd_markers': False,
                'publish_region_text': True,
                'region_text_id_only': True,
                'publish_latent_frontiers': False,
                'publish_gateway_markers': False,
                'region_marker_alpha': 0.84,
                'max_region_marker_cells': 250000,
                'marker_z': 0.035,
                'text_z': 0.35,
            }],
        )
    ])
