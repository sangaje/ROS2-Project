import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, SetEnvironmentVariable, TimerAction
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import Command, LaunchConfiguration, PathJoinSubstitution, PythonExpression
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    use_sim_time = LaunchConfiguration('use_sim_time')
    tb3_model = LaunchConfiguration('tb3_model')
    slam_backend = LaunchConfiguration('slam_backend')
    robot_frame = LaunchConfiguration('robot_frame')
    global_frame = LaunchConfiguration('global_frame')
    map_topic = LaunchConfiguration('map_topic')
    slam_delay_sec = LaunchConfiguration('slam_delay_sec')
    region_delay_sec = LaunchConfiguration('region_delay_sec')
    cartographer_config_basename = LaunchConfiguration('cartographer_config_basename')
    map_resolution = LaunchConfiguration('map_resolution')
    map_publish_period_sec = LaunchConfiguration('map_publish_period_sec')
    house_spawn_x = LaunchConfiguration('house_spawn_x')
    house_spawn_y = LaunchConfiguration('house_spawn_y')
    house_spawn_z = LaunchConfiguration('house_spawn_z')
    house_spawn_yaw = LaunchConfiguration('house_spawn_yaw')

    gazebo_launch = os.path.join(
        get_package_share_directory('turtlebot3_gazebo'),
        'launch',
        'turtlebot3_house.launch.py',
    )

    turtlebot3_cartographer_share = get_package_share_directory('turtlebot3_cartographer')
    cartographer_config_dir = os.path.join(turtlebot3_cartographer_share, 'config')

    robot_urdf = PathJoinSubstitution([
        FindPackageShare('turtlebot3_description'),
        'urdf',
        PythonExpression(["'turtlebot3_' + '", tb3_model, "' + '.urdf'"]),
    ])

    return LaunchDescription([
        DeclareLaunchArgument('use_sim_time', default_value='true'),
        DeclareLaunchArgument('tb3_model', default_value='burger'),
        DeclareLaunchArgument(
            'slam_backend',
            default_value='cartographer',
            description='Compatibility argument. Ignored: this launch always starts Cartographer and never starts slam_toolbox.',
        ),
        # Compatibility-only argument. This core launch intentionally never starts RViz.
        # Use rviz_region_graph.launch.py or rviz_region_graph_keepalive.launch.py separately.
        DeclareLaunchArgument('use_rviz', default_value='false', description='Ignored. RViz is intentionally decoupled from the core stack.'),
        DeclareLaunchArgument('map_topic', default_value='/map'),
        DeclareLaunchArgument('global_frame', default_value='map'),
        DeclareLaunchArgument('robot_frame', default_value='base_footprint'),
        DeclareLaunchArgument('slam_delay_sec', default_value='3.0'),
        DeclareLaunchArgument('region_delay_sec', default_value='8.0'),
        DeclareLaunchArgument('cartographer_config_basename', default_value='turtlebot3_lds_2d.lua'),
        DeclareLaunchArgument('map_resolution', default_value='0.05'),
        DeclareLaunchArgument('map_publish_period_sec', default_value='1.0'),
        DeclareLaunchArgument(
            'house_spawn_x',
            default_value='1.20',
            description='TurtleBot3 spawn x inside turtlebot3_house. Default is a known indoor house pose.',
        ),
        DeclareLaunchArgument(
            'house_spawn_y',
            default_value='-1.60',
            description='TurtleBot3 spawn y inside turtlebot3_house. Default is a known indoor house pose.',
        ),
        DeclareLaunchArgument('house_spawn_z', default_value='0.01'),
        DeclareLaunchArgument('house_spawn_yaw', default_value='0.0'),

        SetEnvironmentVariable('TURTLEBOT3_MODEL', tb3_model),

        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(gazebo_launch),
            # turtlebot3_gazebo/turtlebot3_house.launch.py exposes x_pose/y_pose
            # on most ROS2 TurtleBot3 releases.  Passing them here prevents
            # accidental spawn at the world origin/outside the house.
            launch_arguments={
                # Different turtlebot3_gazebo releases use slightly different
                # spawn argument names. Provide the common aliases so the pose is
                # not silently ignored and the robot does not appear outside the house.
                'x_pose': house_spawn_x,
                'y_pose': house_spawn_y,
                'z_pose': house_spawn_z,
                'yaw': house_spawn_yaw,
                'x': house_spawn_x,
                'y': house_spawn_y,
                'z': house_spawn_z,
                'Y': house_spawn_yaw,
                'spawn_x': house_spawn_x,
                'spawn_y': house_spawn_y,
                'spawn_z': house_spawn_z,
                'spawn_yaw': house_spawn_yaw,
            }.items(),
        ),

        # Ensure RViz RobotModel has /robot_description even when the Gazebo
        # launch variant does not publish it as a transient-local topic.
        # This node also publishes the standard TurtleBot3 TF tree from URDF.
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            name='robot_state_publisher_region_graph',
            output='screen',
            parameters=[{
                'use_sim_time': use_sim_time,
                'robot_description': Command(['xacro ', robot_urdf]),
            }],
        ),

        # Cartographer is launched directly instead of including
        # turtlebot3_cartographer/cartographer.launch.py, because that launch file
        # starts its own RViz.  This package should own the only RViz window.
        TimerAction(
            period=slam_delay_sec,
            actions=[
                Node(
                    package='cartographer_ros',
                    executable='cartographer_node',
                    name='cartographer_node',
                    output='screen',
                    parameters=[{'use_sim_time': use_sim_time}],
                    arguments=[
                        '-configuration_directory', cartographer_config_dir,
                        '-configuration_basename', cartographer_config_basename,
                    ],
                ),
                Node(
                    package='cartographer_ros',
                    executable='cartographer_occupancy_grid_node',
                    name='cartographer_occupancy_grid_node',
                    output='screen',
                    parameters=[{'use_sim_time': use_sim_time}],
                    arguments=[
                        '-resolution', map_resolution,
                        '-publish_period_sec', map_publish_period_sec,
                    ],
                ),
            ],
        ),

        TimerAction(
            period=region_delay_sec,
            actions=[
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
                        'region_update_period': 2.5,
                        'map_stable_time': 0.70,
                        'force_update_without_map_delta': True,
                        'hold_region_graph_while_moving': False,
                        'region_hold_linear_speed_mps': 0.055,
                        'region_hold_angular_speed_rps': 0.22,
                        'region_hold_republish_sec': 0.75,

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
                'region_dense_fill_iterations': 30,
                'region_dense_fill_neighbor_min': 1,
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
                        'publish_latent_frontiers': True,
                'publish_gateway_markers': True,
                'region_marker_alpha': 0.84,
                'max_region_marker_cells': 250000,
                        'marker_z': 0.035,
                        'text_z': 0.35,
                    }],
                )
            ],
        ),

        # RViz intentionally removed from the core launch.
    ])
