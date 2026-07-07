from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration

from multi.launch_utils import (
    make_dynamic_domain_bridge,
    make_physical_robot_controller,
    make_robot_signal_node,
)


def generate_launch_description():
    namespace = LaunchConfiguration('namespace')
    domain_id = LaunchConfiguration('domain_id')
    base_domain_id = LaunchConfiguration('base_domain_id')
    use_sim_time = LaunchConfiguration('use_sim_time')
    enable_test_controller = LaunchConfiguration('enable_test_controller')
    bridge_map = LaunchConfiguration('bridge_map')
    enable_signal_node = LaunchConfiguration('enable_signal_node')
    publish_signal = LaunchConfiguration('publish_signal')

    return LaunchDescription([
        DeclareLaunchArgument('namespace', default_value='waffle1'),
        DeclareLaunchArgument('domain_id', default_value='23'),
        DeclareLaunchArgument('base_domain_id', default_value='0'),
        DeclareLaunchArgument('use_sim_time', default_value='false'),
        DeclareLaunchArgument('enable_test_controller', default_value='false'),
        DeclareLaunchArgument('bridge_map', default_value='false'),
        DeclareLaunchArgument('enable_signal_node', default_value='true'),
        DeclareLaunchArgument('publish_signal', default_value='false'),
        DeclareLaunchArgument('initial_x', default_value='-1.80'),
        DeclareLaunchArgument('initial_y', default_value='0.55'),
        DeclareLaunchArgument('initial_yaw', default_value='0.0'),
        make_physical_robot_controller(
            namespace,
            LaunchConfiguration('initial_x'),
            LaunchConfiguration('initial_y'),
            LaunchConfiguration('initial_yaw'),
            use_sim_time,
            enable_test_controller,
        ),
        make_robot_signal_node(
            namespace,
            publish_signal,
            enable_signal_node,
        ),
        make_dynamic_domain_bridge(namespace, domain_id, base_domain_id, bridge_map),
    ])
