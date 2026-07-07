from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    args = [
        DeclareLaunchArgument('use_sim_time', default_value='true'),
        DeclareLaunchArgument('map_topic', default_value='/map'),
        DeclareLaunchArgument('region_map_topic', default_value='/slam_region_graph/region_map'),
        DeclareLaunchArgument('scan_topic', default_value='/scan'),
        DeclareLaunchArgument('global_frame', default_value='map'),
        DeclareLaunchArgument('robot_frame', default_value='base_footprint'),
        DeclareLaunchArgument('enable_goal_publishing', default_value='false'),
        DeclareLaunchArgument('publish_to_nav2_goal_pose', default_value='false'),
        DeclareLaunchArgument('view_fov_deg', default_value='360.0'),
        DeclareLaunchArgument('view_max_range_m', default_value='3.2'),
        DeclareLaunchArgument('candidate_grid_step_m', default_value='0.25'),
        DeclareLaunchArgument('region_coverage_threshold', default_value='0.85'),
        DeclareLaunchArgument('region_frontier_threshold', default_value='8'),
    ]

    explorer = Node(
        package='tb3_region_mapper',
        executable='region_explorer_node',
        name='region_explorer',
        output='screen',
        parameters=[{
            'use_sim_time': LaunchConfiguration('use_sim_time'),
            'map_topic': LaunchConfiguration('map_topic'),
            'region_map_topic': LaunchConfiguration('region_map_topic'),
            'scan_topic': LaunchConfiguration('scan_topic'),
            'global_frame': LaunchConfiguration('global_frame'),
            'robot_frame': LaunchConfiguration('robot_frame'),
            'enable_goal_publishing': LaunchConfiguration('enable_goal_publishing'),
            'publish_to_nav2_goal_pose': LaunchConfiguration('publish_to_nav2_goal_pose'),
            'view_fov_deg': LaunchConfiguration('view_fov_deg'),
            'view_max_range_m': LaunchConfiguration('view_max_range_m'),
            'candidate_grid_step_m': LaunchConfiguration('candidate_grid_step_m'),
            'region_coverage_threshold': LaunchConfiguration('region_coverage_threshold'),
            'region_frontier_threshold': LaunchConfiguration('region_frontier_threshold'),
        }],
    )

    return LaunchDescription(args + [explorer])
