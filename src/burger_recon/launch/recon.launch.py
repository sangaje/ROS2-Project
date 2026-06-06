from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    SetEnvironmentVariable,
    TimerAction,
)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    use_sim_time = LaunchConfiguration("use_sim_time")
    rviz = LaunchConfiguration("rviz")

    # Frontier explorer options
    goal_mode = LaunchConfiguration("goal_mode")
    base_frame = LaunchConfiguration("base_frame")

    goal_backoff_distance = LaunchConfiguration("goal_backoff_distance")
    goal_clearance_cells = LaunchConfiguration("goal_clearance_cells")
    unknown_policy = LaunchConfiguration("unknown_policy")
    max_unknown_ratio = LaunchConfiguration("max_unknown_ratio")

    use_distance_filter = LaunchConfiguration("use_distance_filter")
    use_rejected_filter = LaunchConfiguration("use_rejected_filter")

    min_goal_distance = LaunchConfiguration("min_goal_distance")
    max_goal_distance = LaunchConfiguration("max_goal_distance")
    min_frontier_size = LaunchConfiguration("min_frontier_size")
    information_gain_weight = LaunchConfiguration("information_gain_weight")

    log_frontier_stats = LaunchConfiguration("log_frontier_stats")
    log_goal_debug = LaunchConfiguration("log_goal_debug")

    slam_params_file = LaunchConfiguration("slam_params_file")

    # ---------------------------------------------------------------------
    # Gazebo TurtleBot3
    # ---------------------------------------------------------------------
    gazebo_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [
                    FindPackageShare("turtlebot3_gazebo"),
                    "launch",
                    "turtlebot3_world.launch.py",
                ]
            )
        )
    )

    # ---------------------------------------------------------------------
    # SLAM Toolbox
    # ---------------------------------------------------------------------
    slam_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [
                    FindPackageShare("slam_toolbox"),
                    "launch",
                    "online_async_launch.py",
                ]
            )
        ),
        launch_arguments={
            "use_sim_time": use_sim_time,
            "slam_params_file": slam_params_file,
        }.items(),
    )

    # ---------------------------------------------------------------------
    # Nav2
    # ---------------------------------------------------------------------
    nav2_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [
                    FindPackageShare("nav2_bringup"),
                    "launch",
                    "navigation_launch.py",
                ]
            )
        ),
        launch_arguments={
            "use_sim_time": use_sim_time,
            "autostart": "true",
        }.items(),
    )

    # ---------------------------------------------------------------------
    # RViz
    # ---------------------------------------------------------------------
    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        output="screen",
        condition=IfCondition(rviz),
        parameters=[
            {
                "use_sim_time": use_sim_time,
            }
        ],
    )

    # ---------------------------------------------------------------------
    # Frontier Explorer
    # ---------------------------------------------------------------------
    frontier_explorer_node = Node(
        package="burger_recon",
        executable="frontier_explorer",
        name="frontier_explorer",
        output="screen",
        parameters=[
            {
                "use_sim_time": use_sim_time,
                "base_frame": base_frame,
                # goal mode: raw / relaxed / safe
                "goal_mode": goal_mode,
                # goal generation
                "goal_backoff_distance": goal_backoff_distance,
                "goal_clearance_cells": goal_clearance_cells,
                # unknown handling
                "unknown_policy": unknown_policy,
                "max_unknown_ratio": max_unknown_ratio,
                # optional filters
                "use_distance_filter": use_distance_filter,
                "use_rejected_filter": use_rejected_filter,
                # distance constraints
                "min_goal_distance": min_goal_distance,
                "max_goal_distance": max_goal_distance,
                # frontier extraction / scoring
                "min_frontier_size": min_frontier_size,
                "information_gain_weight": information_gain_weight,
                # debug
                "log_frontier_stats": log_frontier_stats,
                "log_goal_debug": log_goal_debug,
            }
        ],
    )

    return LaunchDescription(
        [
            # -----------------------------------------------------------------
            # Common args
            # -----------------------------------------------------------------
            DeclareLaunchArgument(
                "use_sim_time",
                default_value="true",
                description="Use Gazebo simulation time",
            ),
            DeclareLaunchArgument(
                "rviz",
                default_value="true",
                description="Launch RViz2",
            ),
            # -----------------------------------------------------------------
            # Frontier args
            # -----------------------------------------------------------------
            DeclareLaunchArgument(
                "goal_mode",
                default_value="relaxed",
                description="Frontier goal mode: raw, relaxed, safe",
            ),
            DeclareLaunchArgument(
                "base_frame",
                default_value="base_footprint",
                description="Robot base frame",
            ),
            DeclareLaunchArgument(
                "goal_backoff_distance",
                default_value="0.25",
                description="Backoff distance from frontier toward robot",
            ),
            DeclareLaunchArgument(
                "goal_clearance_cells",
                default_value="1",
                description="Safety clearance radius in grid cells",
            ),
            DeclareLaunchArgument(
                "unknown_policy",
                default_value="ratio",
                description="Unknown handling: strict, ratio, ignore",
            ),
            DeclareLaunchArgument(
                "max_unknown_ratio",
                default_value="0.60",
                description="Maximum unknown cell ratio in goal clearance window",
            ),
            DeclareLaunchArgument(
                "use_distance_filter",
                default_value="true",
                description="Enable min/max goal distance filter",
            ),
            DeclareLaunchArgument(
                "use_rejected_filter",
                default_value="true",
                description="Avoid recently failed goals",
            ),
            DeclareLaunchArgument(
                "min_goal_distance",
                default_value="0.20",
                description="Minimum goal distance from robot",
            ),
            DeclareLaunchArgument(
                "max_goal_distance",
                default_value="8.0",
                description="Maximum goal distance from robot",
            ),
            DeclareLaunchArgument(
                "min_frontier_size",
                default_value="8",
                description="Minimum frontier cluster size",
            ),
            DeclareLaunchArgument(
                "information_gain_weight",
                default_value="0.20",
                description="Information gain weight in goal scoring",
            ),
            DeclareLaunchArgument(
                "log_frontier_stats",
                default_value="true",
                description="Log frontier cluster statistics",
            ),
            DeclareLaunchArgument(
                "log_goal_debug",
                default_value="true",
                description="Log goal selection debug counts",
            ),
            DeclareLaunchArgument(
                "slam_params_file",
                default_value=PathJoinSubstitution(
                    [
                        FindPackageShare("burger_recon"),
                        "config",
                        "slam_fast.yaml",
                    ]
                ),
                description="SLAM Toolbox params file",
            ),
            # -----------------------------------------------------------------
            # Environment
            # -----------------------------------------------------------------
            SetEnvironmentVariable(
                name="TURTLEBOT3_MODEL",
                value="burger",
            ),
            # -----------------------------------------------------------------
            # Launch order
            # -----------------------------------------------------------------
            gazebo_launch,
            TimerAction(
                period=4.0,
                actions=[
                    slam_launch,
                ],
            ),
            TimerAction(
                period=9.0,
                actions=[
                    nav2_launch,
                ],
            ),
            TimerAction(
                period=10.0,
                actions=[
                    rviz_node,
                ],
            ),
            TimerAction(
                period=18.0,
                actions=[
                    frontier_explorer_node,
                ],
            ),
        ]
    )
