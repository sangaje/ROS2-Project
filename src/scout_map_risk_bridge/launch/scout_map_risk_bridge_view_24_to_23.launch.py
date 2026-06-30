from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    bridge_launch = PathJoinSubstitution([
        FindPackageShare('scout_map_risk_bridge'),
        'launch',
        'scout_map_risk_bridge_24_to_23.launch.py',
    ])
    rviz_launch = PathJoinSubstitution([
        FindPackageShare('scout_map_risk_bridge'),
        'launch',
        'scout_map_risk_receiver_rviz.launch.py',
    ])
    return LaunchDescription([
        DeclareLaunchArgument('start_rviz', default_value='true'),
        IncludeLaunchDescription(PythonLaunchDescriptionSource(bridge_launch)),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(rviz_launch),
            condition=IfCondition(LaunchConfiguration('start_rviz')),
            launch_arguments={
                'domain_id': '23',
            }.items(),
        ),
    ])
