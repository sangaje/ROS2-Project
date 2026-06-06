#!/usr/bin/env python3

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

    base_frame = LaunchConfiguration("base_frame")

    linear_speed = LaunchConfiguration("linear_speed")
    angular_kp = LaunchConfiguration("angular_kp")
    max_angular_speed = LaunchConfiguration("max_angular_speed")
    angle_tolerance = LaunchConfiguration("angle_tolerance")
    goal_tolerance = LaunchConfiguration("goal_tolerance")

    safe_front_distance = LaunchConfiguration("safe_front_distance")
    front_angle_deg = LaunchConfiguration("front_angle_deg")
    avoid_turn_speed = LaunchConfiguration("avoid_turn_speed")

    min_goal_distance = LaunchConfiguration("min_goal_distance")
    max_goal_distance = LaunchConfiguration("max_goal_distance")
    min_frontier_size = LaunchConfiguration("min_frontier_size")
    information_gain_weight = LaunchConfiguration("information_gain_weight")
    goal_recompute_period = LaunchConfiguration("goal_recompute_period")

    log_frontier_stats = LaunchConfiguration("log_frontier_stats")

    gazebo_launch_file = LaunchConfiguration("gazebo_launch_file")

    gazebo_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [
                    FindPackageShare("turtlebot3_gazebo"),
                    "launch",
                    gazebo_launch_file,
                ]
            )
        )
    )
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
        }.items(),
    )

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

    frontier_cmd_explorer_node = Node(
        package="burger_recon",
        executable="frontier_cmd_explorer",
        name="frontier_cmd_explorer",
        output="screen",
        parameters=[
            {
                "use_sim_time": use_sim_time,
                "map_topic": "/map",
                "scan_topic": "/scan",
                "cmd_vel_topic": "/cmd_vel",
                "map_frame": "map",
                "base_frame": base_frame,
                "linear_speed": linear_speed,
                "angular_kp": angular_kp,
                "max_angular_speed": max_angular_speed,
                "angle_tolerance": angle_tolerance,
                "goal_tolerance": goal_tolerance,
                "safe_front_distance": safe_front_distance,
                "front_angle_deg": front_angle_deg,
                "avoid_turn_speed": avoid_turn_speed,
                "min_goal_distance": min_goal_distance,
                "max_goal_distance": max_goal_distance,
                "min_frontier_size": min_frontier_size,
                "information_gain_weight": information_gain_weight,
                "goal_recompute_period": goal_recompute_period,
                "log_frontier_stats": log_frontier_stats,
            }
        ],
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "gazebo_launch_file",
                default_value="turtlebot3_house.launch.py",
                description="TurtleBot3 Gazebo launch file",
            ),
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
            DeclareLaunchArgument(
                "base_frame",
                default_value="base_footprint",
                description="Robot base frame",
            ),
            DeclareLaunchArgument(
                "linear_speed",
                default_value="0.08",
                description="Forward linear speed",
            ),
            DeclareLaunchArgument(
                "angular_kp",
                default_value="1.5",
                description="Angular P gain",
            ),
            DeclareLaunchArgument(
                "max_angular_speed",
                default_value="0.7",
                description="Maximum angular speed",
            ),
            DeclareLaunchArgument(
                "angle_tolerance",
                default_value="0.35",
                description="Allow forward motion if yaw error is below this",
            ),
            DeclareLaunchArgument(
                "goal_tolerance",
                default_value="0.25",
                description="Distance threshold for goal reached",
            ),
            DeclareLaunchArgument(
                "safe_front_distance",
                default_value="0.35",
                description="Minimum allowed front obstacle distance",
            ),
            DeclareLaunchArgument(
                "front_angle_deg",
                default_value="35.0",
                description="Front scan sector angle",
            ),
            DeclareLaunchArgument(
                "avoid_turn_speed",
                default_value="0.6",
                description="Turn speed during obstacle avoidance",
            ),
            DeclareLaunchArgument(
                "min_goal_distance",
                default_value="0.35",
                description="Minimum frontier goal distance",
            ),
            DeclareLaunchArgument(
                "max_goal_distance",
                default_value="8.0",
                description="Maximum frontier goal distance",
            ),
            DeclareLaunchArgument(
                "min_frontier_size",
                default_value="8",
                description="Minimum frontier cluster size",
            ),
            DeclareLaunchArgument(
                "information_gain_weight",
                default_value="0.20",
                description="Information gain weight in frontier scoring",
            ),
            DeclareLaunchArgument(
                "goal_recompute_period",
                default_value="2.0",
                description="Goal recomputation period",
            ),
            DeclareLaunchArgument(
                "log_frontier_stats",
                default_value="true",
                description="Log frontier cluster statistics",
            ),
            SetEnvironmentVariable(
                name="TURTLEBOT3_MODEL",
                value="burger",
            ),
            # 1. Gazebo
            gazebo_launch,
            # 2. SLAM Toolbox
            TimerAction(
                period=4.0,
                actions=[
                    slam_launch,
                ],
            ),
            # 3. RViz
            TimerAction(
                period=6.0,
                actions=[
                    rviz_node,
                ],
            ),
            # 4. Direct cmd_vel frontier explorer
            TimerAction(
                period=8.0,
                actions=[
                    frontier_cmd_explorer_node,
                ],
            ),
        ]
    )
