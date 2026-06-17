import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, SetEnvironmentVariable, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue

from tb3_visibility_explorer.explorer_launch_params import explorer_params


def generate_launch_description():
    use_sim_time = LaunchConfiguration('use_sim_time')
    explorer_delay_sec = LaunchConfiguration('explorer_delay_sec')
    turtlebot3_model = LaunchConfiguration('turtlebot3_model')
    robot_frame = LaunchConfiguration('robot_frame')
    cmd_vel_stamped = LaunchConfiguration('cmd_vel_stamped')
    prefer_internal_grid_planner = LaunchConfiguration('prefer_internal_grid_planner')

    cartographer_launch = os.path.join(
        get_package_share_directory('turtlebot3_cartographer'),
        'launch',
        'cartographer.launch.py',
    )

    return LaunchDescription([
        DeclareLaunchArgument('use_sim_time', default_value='false'),
        DeclareLaunchArgument('explorer_delay_sec', default_value='10.0'),
        DeclareLaunchArgument('turtlebot3_model', default_value='burger'),
        DeclareLaunchArgument('robot_frame', default_value='base_footprint'),
        DeclareLaunchArgument('cmd_vel_stamped', default_value='true'),
        DeclareLaunchArgument('prefer_internal_grid_planner', default_value='true'),
        SetEnvironmentVariable('TURTLEBOT3_MODEL', turtlebot3_model),

        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(cartographer_launch),
            launch_arguments={
                'use_sim_time': use_sim_time,
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
                    parameters=[explorer_params(use_sim_time, robot_frame, cmd_vel_stamped, prefer_internal_grid_planner)],
                )
            ],
        ),
    ])
