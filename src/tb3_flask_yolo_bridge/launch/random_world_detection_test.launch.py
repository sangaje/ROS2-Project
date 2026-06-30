from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('output_topic', default_value='/risk/yolo_detections'),
        DeclareLaunchArgument('use_sim_time', default_value='false'),
        DeclareLaunchArgument('map_topic', default_value='/map'),
        DeclareLaunchArgument('map_frame', default_value='map'),
        DeclareLaunchArgument('base_frame', default_value='base_link'),
        DeclareLaunchArgument('x_min', default_value='-2.0'),
        DeclareLaunchArgument('x_max', default_value='2.0'),
        DeclareLaunchArgument('y_min', default_value='-2.0'),
        DeclareLaunchArgument('y_max', default_value='2.0'),
        DeclareLaunchArgument('camera_hfov_deg', default_value='62.0'),
        DeclareLaunchArgument('min_range_m', default_value='0.5'),
        DeclareLaunchArgument('max_range_m', default_value='5.0'),
        DeclareLaunchArgument('use_map_free_cells', default_value='true'),
        Node(
            package='tb3_flask_yolo_bridge',
            executable='random_world_detection_test',
            name='random_world_detection_test',
            output='screen',
            parameters=[{
                'output_topic': LaunchConfiguration('output_topic'),
                'use_sim_time': LaunchConfiguration('use_sim_time'),
                'map_topic': LaunchConfiguration('map_topic'),
                'map_frame': LaunchConfiguration('map_frame'),
                'base_frame': LaunchConfiguration('base_frame'),
                'x_min': LaunchConfiguration('x_min'),
                'x_max': LaunchConfiguration('x_max'),
                'y_min': LaunchConfiguration('y_min'),
                'y_max': LaunchConfiguration('y_max'),
                'camera_hfov_deg': LaunchConfiguration('camera_hfov_deg'),
                'min_range_m': LaunchConfiguration('min_range_m'),
                'max_range_m': LaunchConfiguration('max_range_m'),
                'use_map_free_cells': LaunchConfiguration('use_map_free_cells'),
            }],
        ),
    ])
