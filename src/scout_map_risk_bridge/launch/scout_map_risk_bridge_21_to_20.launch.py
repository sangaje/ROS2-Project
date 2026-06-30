from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    bridge_launch = PathJoinSubstitution([
        FindPackageShare('scout_map_risk_bridge'),
        'launch',
        'scout_map_risk_bridge.launch.py',
    ])
    return LaunchDescription([
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(bridge_launch),
            launch_arguments={
                'from_domain': '21',
                'to_domain': '20',
                'map_in': '/map',
                'risk_in': '/risk/risk_map',
                'map_out': '/scout/map',
                'risk_out': '/scout/risk_map',
                'status_out': '/scout_bridge/status',
                'target_frame': 'scout_map',
                'republish_period_sec': '1.0',
            }.items(),
        ),
    ])
