from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('image_topic', default_value='/risk/debug_yolo_image'),
        DeclareLaunchArgument('resize_width', default_value='960'),
        Node(
            package='tb3_bayesian_risk_map',
            executable='opencv_yolo_viewer_node',
            name='opencv_yolo_viewer_node',
            output='screen',
            parameters=[{
                'image_topic': LaunchConfiguration('image_topic'),
                'resize_width': LaunchConfiguration('resize_width'),
            }],
        ),
    ])
