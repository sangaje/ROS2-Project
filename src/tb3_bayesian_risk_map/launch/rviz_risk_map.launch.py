from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    pkg_share = get_package_share_directory('tb3_bayesian_risk_map')
    rviz_config = os.path.join(pkg_share, 'rviz', 'bayesian_risk_map.rviz')
    return LaunchDescription([
        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2_risk_map',
            output='screen',
            arguments=['-d', rviz_config],
        )
    ])
