from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration

from multi.launch_utils import (
    make_dynamic_domain_bridge,
    make_physical_robot_controller,
    make_region_nav2_node,
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
    enable_region_nav2 = LaunchConfiguration('enable_region_nav2')
    publish_region_nav2_goal = LaunchConfiguration('publish_region_nav2_goal')
    map_topic = LaunchConfiguration('map_topic')
    nav2_goal_topic = LaunchConfiguration('nav2_goal_topic')

    return LaunchDescription([
        DeclareLaunchArgument('namespace', default_value='burger1'),
        DeclareLaunchArgument('domain_id', default_value='22'),
        DeclareLaunchArgument('base_domain_id', default_value='0'),
        DeclareLaunchArgument('use_sim_time', default_value='false'),
        DeclareLaunchArgument('enable_test_controller', default_value='false'),
        DeclareLaunchArgument('bridge_map', default_value='false'),
        DeclareLaunchArgument('enable_signal_node', default_value='true'),
        DeclareLaunchArgument('publish_signal', default_value='true'),
        DeclareLaunchArgument('enable_region_nav2', default_value='false'),
        DeclareLaunchArgument('publish_region_nav2_goal', default_value='false'),
        DeclareLaunchArgument('map_topic', default_value='/map'),
        DeclareLaunchArgument('nav2_goal_topic', default_value='/goal_pose'),
        DeclareLaunchArgument('initial_x', default_value='-1.90'),
        DeclareLaunchArgument('initial_y', default_value='0.00'),
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
        make_region_nav2_node(
            use_sim_time,
            map_topic,
            nav2_goal_topic,
            publish_region_nav2_goal,
            condition=IfCondition(enable_region_nav2),
        ),
        make_dynamic_domain_bridge(namespace, domain_id, base_domain_id, bridge_map),
    ])
