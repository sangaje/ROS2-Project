from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    robot_name = LaunchConfiguration('robot_name')
    map_frame = LaunchConfiguration('map_frame')
    base_frame = LaunchConfiguration('base_frame')
    navigate_action = LaunchConfiguration('navigate_action')

    # goal_topic/hold_topic/cancel_topic/status_topic are left empty.
    # robot_goal_proxy derives them from robot_name as /fleet/<robot>/...
    goal_proxy = Node(
        package='tb3_fleet_robot',
        executable='robot_goal_proxy',
        name='robot_goal_proxy',
        output='screen',
        parameters=[{
            'robot_name': robot_name,
            'navigate_action': navigate_action,
        }],
    )

    pose_reporter = Node(
        package='tb3_fleet_robot',
        executable='robot_pose_reporter',
        name='robot_pose_reporter',
        output='screen',
        parameters=[{
            'robot_name': robot_name,
            'map_frame': map_frame,
            'base_frame': base_frame,
        }],
    )

    return LaunchDescription([
        DeclareLaunchArgument('robot_name', default_value='burger'),
        DeclareLaunchArgument('map_frame', default_value='map'),
        DeclareLaunchArgument('base_frame', default_value='burger/base_footprint'),
        DeclareLaunchArgument('navigate_action', default_value='/burger/navigate_to_pose'),
        goal_proxy,
        pose_reporter,
    ])
