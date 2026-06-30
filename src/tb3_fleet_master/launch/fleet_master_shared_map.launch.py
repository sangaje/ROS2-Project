from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    robot_names = LaunchConfiguration('robot_names')
    spacing = LaunchConfiguration('spacing')
    formation_type = LaunchConfiguration('formation_type')
    frame_id = LaunchConfiguration('frame_id')
    map_yaml = LaunchConfiguration('map')
    use_sim_time = LaunchConfiguration('use_sim_time')

    # Domain 25 master-side saved map publisher.
    # Followers receive /map through domain_bridge and run their own AMCL/Nav2 locally.
    map_server = Node(
        package='nav2_map_server',
        executable='map_server',
        name='map_server',
        output='screen',
        parameters=[{
            'yaml_filename': map_yaml,
            'use_sim_time': use_sim_time,
        }],
    )

    lifecycle_manager_map = Node(
        package='nav2_lifecycle_manager',
        executable='lifecycle_manager',
        name='lifecycle_manager_map_server',
        output='screen',
        parameters=[{
            'use_sim_time': use_sim_time,
            'autostart': True,
            'node_names': ['map_server'],
        }],
    )

    commander = Node(
        package='tb3_fleet_master',
        executable='fleet_commander_node',
        name='fleet_commander_node',
        output='screen',
        parameters=[{
            'robot_names': robot_names,
            'spacing': spacing,
            'formation_type': formation_type,
            'frame_id': frame_id,
        }],
    )

    echo = Node(
        package='tb3_fleet_master',
        executable='fleet_state_echo',
        name='fleet_state_echo',
        output='screen',
        parameters=[{
            'robot_names': robot_names,
        }],
    )

    return LaunchDescription([
        DeclareLaunchArgument('robot_names', default_value='robot1,robot2'),
        DeclareLaunchArgument('spacing', default_value='0.60'),
        DeclareLaunchArgument('formation_type', default_value='wedge'),
        DeclareLaunchArgument('frame_id', default_value='map'),
        DeclareLaunchArgument('map', default_value=''),
        DeclareLaunchArgument('use_sim_time', default_value='true'),
        map_server,
        lifecycle_manager_map,
        commander,
        echo,
    ])
