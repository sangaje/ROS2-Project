import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    use_sim_time = LaunchConfiguration('use_sim_time')
    pkg_share = get_package_share_directory('tb3_region_mapper')
    rviz_config = os.path.join(pkg_share, 'rviz', 'region_graph.rviz')

    return LaunchDescription([
        DeclareLaunchArgument('use_sim_time', default_value='true'),
        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2_region_graph',
            output='screen',
            arguments=['-d', rviz_config],
            parameters=[{'use_sim_time': use_sim_time}],
        )
    ])
