from launch import LaunchDescription
from launch.actions import ExecuteProcess, SetEnvironmentVariable
from launch.substitutions import PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    pkg_share = get_package_share_directory('tb3_bayesian_risk_map')
    rviz_config = os.path.join(pkg_share, 'rviz', 'bayesian_risk_map.rviz')
    rviz_clean = PathJoinSubstitution([
        FindPackageShare('tb3_bayesian_risk_map'),
        'scripts',
        'rviz2_clean_env.bash',
    ])
    return LaunchDescription([
        SetEnvironmentVariable('RMW_IMPLEMENTATION', 'rmw_fastrtps_cpp'),
        ExecuteProcess(
            cmd=[rviz_clean, '-d', rviz_config],
            name='rviz2_risk_map',
            output='screen',
        )
    ])
