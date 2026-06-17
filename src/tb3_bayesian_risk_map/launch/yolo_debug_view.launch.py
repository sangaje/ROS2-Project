from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('image_topic', default_value='/risk/debug_yolo_image'),
        Node(
            package='rqt_image_view',
            executable='rqt_image_view',
            name='rqt_yolo_debug_view',
            output='screen',
            arguments=[LaunchConfiguration('image_topic')],
        ),
    ])
