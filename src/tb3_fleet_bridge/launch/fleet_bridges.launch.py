from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.substitutions import PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


def _bridge_process(config_file_name):
    config_path = PathJoinSubstitution([
        FindPackageShare('tb3_fleet_bridge'),
        'config',
        config_file_name,
    ])
    return ExecuteProcess(
        cmd=['ros2', 'run', 'domain_bridge', 'domain_bridge', config_path],
        output='screen',
    )


def generate_launch_description():
    # Domain layout:
    #   Domain 25: waffle + fleet master + RViz
    #   Domain 24: burger Nav2
    # Waffle and the fleet master share domain 25 so no waffle bridge is needed.
    return LaunchDescription([
        DeclareLaunchArgument('note', default_value='domain25_waffle_master_domain24_burger'),
        _bridge_process('burger_bridge.yaml'),
    ])
