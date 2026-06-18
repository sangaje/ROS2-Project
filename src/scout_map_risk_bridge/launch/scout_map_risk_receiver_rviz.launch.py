from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    default_rviz = os.path.join(get_package_share_directory('scout_map_risk_bridge'), 'rviz', 'scout_map_risk_receiver.rviz')
    return LaunchDescription([
        DeclareLaunchArgument('rviz_config', default_value=default_rviz),
        Node(package='rviz2', executable='rviz2', name='scout_map_risk_receiver_rviz', output='screen',
             arguments=['-d', LaunchConfiguration('rviz_config')]),
    ])
