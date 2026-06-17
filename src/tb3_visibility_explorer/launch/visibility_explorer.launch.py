from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

from tb3_visibility_explorer.explorer_launch_params import explorer_params


def generate_launch_description():
    use_sim_time = LaunchConfiguration('use_sim_time')
    robot_frame = LaunchConfiguration('robot_frame')
    cmd_vel_stamped = LaunchConfiguration('cmd_vel_stamped')
    prefer_internal_grid_planner = LaunchConfiguration('prefer_internal_grid_planner')

    return LaunchDescription([
        DeclareLaunchArgument('use_sim_time', default_value='false'),
        DeclareLaunchArgument('robot_frame', default_value='base_footprint'),
        DeclareLaunchArgument('cmd_vel_stamped', default_value='true'),
        DeclareLaunchArgument('prefer_internal_grid_planner', default_value='true'),
        Node(
            package='tb3_visibility_explorer',
            executable='visibility_explorer_node',
            name='visibility_explorer',
            output='screen',
            parameters=[explorer_params(use_sim_time, robot_frame, cmd_vel_stamped, prefer_internal_grid_planner)],
        ),
    ])
