import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    AppendEnvironmentVariable,
    DeclareLaunchArgument,
    IncludeLaunchDescription,
)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

from tb3_multi.launch_utils import (
    make_auto_patrol_node,
    make_region_nav2_node,
    make_ros_gz_bridge_args,
    make_sim_controller_nodes,
    ROBOT_NAMES,
)


def generate_launch_description():
    pkg_tb3_multi = get_package_share_directory('tb3_multi')
    pkg_tb3_gazebo = get_package_share_directory('turtlebot3_gazebo')
    pkg_ros_gz_sim = get_package_share_directory('ros_gz_sim')

    world = os.path.join(pkg_tb3_multi, 'worlds', 'multi_tb3_world.world')
    map_yaml = os.path.join(pkg_tb3_multi, 'maps', 'turtlemap.yaml')
    rviz_config = os.path.join(pkg_tb3_multi, 'rviz', 'multi_tb3_gz.rviz')

    use_gazebo = LaunchConfiguration('use_gazebo')
    use_rviz = LaunchConfiguration('use_rviz')
    use_sim_time = LaunchConfiguration('use_sim_time')
    publish_static_map = LaunchConfiguration('publish_static_map')
    map_topic = LaunchConfiguration('map_topic')
    auto_patrol = LaunchConfiguration('auto_patrol')
    rescue_offset_m = LaunchConfiguration('rescue_offset_m')
    enable_sim_controllers = LaunchConfiguration('enable_sim_controllers')
    enable_region_nav2 = LaunchConfiguration('enable_region_nav2')
    publish_region_nav2_goal = LaunchConfiguration('publish_region_nav2_goal')
    nav2_goal_topic = LaunchConfiguration('nav2_goal_topic')

    return LaunchDescription([
        DeclareLaunchArgument('use_gazebo', default_value='true'),
        DeclareLaunchArgument('use_rviz', default_value='true'),
        DeclareLaunchArgument('use_sim_time', default_value='true'),
        DeclareLaunchArgument('publish_static_map', default_value='false'),
        DeclareLaunchArgument('map_topic', default_value='/map'),
        DeclareLaunchArgument('auto_patrol', default_value='true'),
        DeclareLaunchArgument('rescue_offset_m', default_value='0.0'),
        DeclareLaunchArgument('enable_sim_controllers', default_value='false'),
        DeclareLaunchArgument('enable_region_nav2', default_value='false'),
        DeclareLaunchArgument('publish_region_nav2_goal', default_value='false'),
        DeclareLaunchArgument('nav2_goal_topic', default_value='/burger1/goal_pose'),
        AppendEnvironmentVariable(
            'GZ_SIM_RESOURCE_PATH',
            os.path.join(pkg_tb3_gazebo, 'models'),
            condition=IfCondition(use_gazebo),
        ),
        AppendEnvironmentVariable(
            'GZ_SIM_RESOURCE_PATH',
            os.path.join(pkg_tb3_multi, 'models'),
            condition=IfCondition(use_gazebo),
        ),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(pkg_ros_gz_sim, 'launch', 'gz_sim.launch.py')
            ),
            launch_arguments={
                'gz_args': f'-r -v 4 {world}',
                'on_exit_shutdown': 'true',
            }.items(),
            condition=IfCondition(use_gazebo),
        ),
        Node(
            package='ros_gz_bridge',
            executable='parameter_bridge',
            arguments=make_ros_gz_bridge_args(),
            condition=IfCondition(use_gazebo),
            output='screen',
        ),
        Node(
            package='tb3_multi',
            executable='static_map_publisher',
            name='static_map_publisher',
            parameters=[{
                'use_sim_time': use_sim_time,
                'map_yaml': map_yaml,
                'frame_id': 'map',
                'map_topic': map_topic,
            }],
            condition=IfCondition(publish_static_map),
            output='screen',
        ),
        Node(
            package='tb3_multi',
            executable='goal_dispatcher',
            name='goal_dispatcher',
            parameters=[{
                'use_sim_time': use_sim_time,
                'robots': ROBOT_NAMES,
                'default_robot': 'burger1',
            }],
            output='screen',
        ),
        make_auto_patrol_node(
            use_sim_time,
            auto_patrol,
            rescue_offset_m,
        ),
        make_region_nav2_node(
            use_sim_time,
            map_topic,
            nav2_goal_topic,
            publish_region_nav2_goal,
            condition=IfCondition(enable_region_nav2),
        ),
        *make_sim_controller_nodes(
            use_sim_time=use_sim_time,
            condition=IfCondition(enable_sim_controllers),
        ),
        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            arguments=['-d', rviz_config],
            parameters=[{'use_sim_time': use_sim_time}],
            condition=IfCondition(use_rviz),
            output='screen',
        ),
    ])
