from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('from_domain', default_value='21'),
        DeclareLaunchArgument('to_domain', default_value='20'),
        DeclareLaunchArgument('map_in', default_value='/map'),
        DeclareLaunchArgument('risk_in', default_value='/risk/risk_map'),
        DeclareLaunchArgument('map_out', default_value='/scout/map'),
        DeclareLaunchArgument('risk_out', default_value='/scout/risk_map'),
        DeclareLaunchArgument('status_out', default_value='/scout_bridge/status'),
        DeclareLaunchArgument('initialpose_in', default_value='/initialpose'),
        DeclareLaunchArgument('initialpose_out', default_value='/initialpose'),
        DeclareLaunchArgument('source_initialpose_frame', default_value='map'),
        DeclareLaunchArgument('target_frame', default_value='scout_map'),
        DeclareLaunchArgument('republish_period_sec', default_value='1.0'),
        Node(
            package='scout_map_risk_bridge',
            executable='scout_map_risk_bridge_node',
            output='screen',
            arguments=[
                '--from-domain', LaunchConfiguration('from_domain'),
                '--to-domain', LaunchConfiguration('to_domain'),
                '--map-in', LaunchConfiguration('map_in'),
                '--risk-in', LaunchConfiguration('risk_in'),
                '--map-out', LaunchConfiguration('map_out'),
                '--risk-out', LaunchConfiguration('risk_out'),
                '--status-out', LaunchConfiguration('status_out'),
                '--initialpose-in', LaunchConfiguration('initialpose_in'),
                '--initialpose-out', LaunchConfiguration('initialpose_out'),
                '--source-initialpose-frame', LaunchConfiguration('source_initialpose_frame'),
                '--target-frame', LaunchConfiguration('target_frame'),
                '--republish-period-sec', LaunchConfiguration('republish_period_sec'),
            ],
        ),
    ])
