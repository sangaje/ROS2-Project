import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    use_sim_time = LaunchConfiguration('use_sim_time')
    autostart = LaunchConfiguration('autostart')
    explorer_delay_sec = LaunchConfiguration('explorer_delay_sec')

    slam_launch = os.path.join(
        get_package_share_directory('slam_toolbox'),
        'launch',
        'online_async_launch.py',
    )

    nav2_launch = os.path.join(
        get_package_share_directory('nav2_bringup'),
        'launch',
        'navigation_launch.py',
    )

    return LaunchDescription([
        DeclareLaunchArgument('use_sim_time', default_value='true'),
        DeclareLaunchArgument('autostart', default_value='true'),
        DeclareLaunchArgument('explorer_delay_sec', default_value='15.0'),

        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(slam_launch),
            launch_arguments={'use_sim_time': use_sim_time}.items(),
        ),

        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(nav2_launch),
            launch_arguments={
                'use_sim_time': use_sim_time,
                'autostart': autostart,
            }.items(),
        ),

        TimerAction(
            period=explorer_delay_sec,
            actions=[
                Node(
                    package='tb3_visibility_explorer',
                    executable='visibility_explorer_node',
                    name='visibility_explorer',
                    output='screen',
                    parameters=[{
                        'version_name': 'v30_unmapped_priority_real_robot',
                        'use_sim_time': use_sim_time,

                        'map_topic': '/map',
                        'scan_topic': '/scan',
                        'cmd_vel_topic': '/cmd_vel',
                        'global_frame': 'map',
                        'robot_frame': 'base_link',
                        'cmd_vel_stamped': True,
                        'path_topic': '/visibility_explorer/path',
                        'plan_alias_topic': '/plan',
                        'scan_viz_topic': '/scan_reliable',
                        'publish_scan_reliable': True,
                        'scan_sub_best_effort': True,

                        'planning_period': 0.25,
                        'map_stable_time': 0.10,
                        'max_plan_trials': 20,

                        'free_threshold': 20,
                        'occupied_threshold': 65,
                        'unknown_is_obstacle_for_clearance': False,

                        'min_frontier_cluster_size': 5,
                        'frontier_goal_offset': 0.55,
                        'candidate_stride_cells': 3,
                        'min_goal_distance': 0.25,
                        'max_goal_distance': 5.0,
                        'max_goal_bearing_deg': 160.0,

                        'robot_radius': 0.23,
                        'min_goal_clearance': 0.30,
                        'min_path_clearance': 0.18,
                        'front_stop_distance': 0.24,
                        'front_slow_distance': 0.42,
                        'side_stop_distance': 0.16,

                        'view_fov_deg': 60.0,
                        'view_ray_count': 31,
                        'view_max_range': 3.5,
                        'coverage_robot_radius': 0.28,
                        'coverage_publish_period': 0.4,
                        'visual_unchecked_threshold': 95,
                        'coverage_candidate_max_checked': 95,

                        'unknown_gain_weight': 12.0,
                        'visual_gain_weight': 3.0,
                        'frontier_gain_weight': 1.6,
                        'clearance_weight': 0.55,
                        'distance_weight': 0.22,
                        'blacklist_weight': 5.0,
                        'heading_weight': 0.12,
                        'coverage_source_score_bonus': 22.0,

                        'planner_id': '',
                        'nav2_result_timeout': 75.0,
                        'stuck_timeout': 10.0,
                        'stuck_min_progress': 0.10,
                        'goal_success_radius': 0.35,
                        'blacklist_radius': 0.55,
                        'blacklist_duration': 45.0,

                        'enable_sector_scan': False,
                        'sector_scan_angle_deg': 45.0,
                        'sector_scan_angular_speed': 0.75,
                        'enable_view_yaw_align': True,
                        'view_align_tolerance_deg': 12.0,
                        'view_align_timeout': 1.0,
                        'view_align_angular_speed': 0.75,

                        'enable_short_lidar_probe_fallback': True,
                        'enable_unknown_first_probe': True,
                        'enable_scan_open_space_probe': True,
                        'allow_probe_during_map_stabilizing': True,
                        'prefer_scan_probe_before_nav2': True,
                        'coverage_priority_over_probe': True,
                        'unmapped_priority_over_coverage': True,
                        'unmapped_priority_probe_gain': 8,
                        'unknown_first_min_gain': 2,
                        'nav2_min_goal_distance': 0.0,
                        'execute_with_nav2_navigator': False,
                        'direct_follow_lookahead_distance': 0.35,
                        'direct_follow_max_linear_speed': 0.24,
                        'direct_follow_min_linear_speed': 0.045,
                        'direct_follow_max_angular_speed': 1.35,
                        'direct_follow_angular_gain': 1.8,
                        'direct_follow_stuck_timeout': 3.5,
                        'direct_follow_progress_epsilon': 0.04,
                        'direct_goal_tolerance': 0.18,
                        'probe_success_radius': 0.12,
                        'probe_min_distance': 0.45,
                        'probe_fov_deg': 120.0,
                        'probe_distance': 0.75,
                        'probe_min_front_clearance': 0.48,
                        'probe_side_clearance': 0.22,
                        'probe_linear_speed': 0.20,
                        'probe_angular_gain': 1.8,
                        'probe_timeout': 4.0,
                        'probe_fail_cooldown': 2.0,
                        'probe_safety_margin': 0.05,
                        'probe_fail_blacklist_radius': 0.60,
                        'planner_first_when_probe_blocked': True,
                        'micro_rotate_on_total_failure': True,
                        'micro_rotate_duration': 0.45,
                        'micro_rotate_speed': 0.45,
                    }],
                )
            ],
        ),
    ])
