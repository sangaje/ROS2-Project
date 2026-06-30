from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, SetEnvironmentVariable
from launch.substitutions import LaunchConfiguration
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    pkg_share = get_package_share_directory('scout_map_risk_bridge')
    default_rviz = os.path.join(pkg_share, 'rviz', 'scout_map_risk_receiver.rviz')
    rviz_clean = os.path.join(pkg_share, 'scripts', 'run_rviz2_clean.bash')
    return LaunchDescription([
        DeclareLaunchArgument('domain_id', default_value='20'),
        DeclareLaunchArgument('rviz_config', default_value=default_rviz),
        SetEnvironmentVariable('ROS_DOMAIN_ID', LaunchConfiguration('domain_id')),
        SetEnvironmentVariable('RMW_IMPLEMENTATION', 'rmw_fastrtps_cpp'),
        SetEnvironmentVariable('FASTDDS_BUILTIN_TRANSPORTS', 'UDPv4'),
        ExecuteProcess(
            cmd=[rviz_clean, '-d', LaunchConfiguration('rviz_config')],
            name='scout_map_risk_receiver_rviz',
            output='screen',
        ),
    ])
