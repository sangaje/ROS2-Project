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
    #   Domain 25: fleet master and RViz
    #   Domain 26: burger Nav2
    #   Domain 27: waffle Nav2
    # The normal simulator launch uses one ROS domain and does not need this.
    # Keep these bridges for hardware or split-domain debugging.
    return LaunchDescription([
        DeclareLaunchArgument('note', default_value='domain25_master_domain26_burger_domain27_waffle'),
        _bridge_process('burger_bridge.yaml'),
        _bridge_process('waffle_bridge.yaml'),
    ])
