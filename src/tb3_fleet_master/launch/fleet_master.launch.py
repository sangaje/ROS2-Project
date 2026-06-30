from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    robot_names = LaunchConfiguration('robot_names')
    spacing = LaunchConfiguration('spacing')
    formation_type = LaunchConfiguration('formation_type')
    frame_id = LaunchConfiguration('frame_id')
    # robot_names is intentionally a comma-separated string here to avoid
    # launch-time string-array quoting issues. The node accepts both list and CSV.
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

    # Launch cannot directly condition on an arbitrary string without extra imports;
    # keep echo enabled by default because it is useful during v1 debugging.
    return LaunchDescription([
        DeclareLaunchArgument('robot_names', default_value='waffle,burger'),
        DeclareLaunchArgument('spacing', default_value='0.85'),
        DeclareLaunchArgument('formation_type', default_value='column'),
        DeclareLaunchArgument('frame_id', default_value='map'),
        commander,
        echo,
    ])
