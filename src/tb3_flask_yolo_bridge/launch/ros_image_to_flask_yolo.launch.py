from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('image_topic', default_value='/camera/image_raw'),
        DeclareLaunchArgument('input_type', default_value='raw'),
        DeclareLaunchArgument('server_url', default_value='http://127.0.0.1:5005/detect'),
        DeclareLaunchArgument('output_topic', default_value='/risk/yolo_detections'),
        DeclareLaunchArgument('max_rate_hz', default_value='10.0'),
        DeclareLaunchArgument('jpeg_quality', default_value='75'),
        DeclareLaunchArgument('timeout_sec', default_value='5.0'),
        DeclareLaunchArgument('publish_debug_image', default_value='true'),
        DeclareLaunchArgument('debug_image_topic', default_value='/risk/debug_yolo_image'),
        Node(
            package='tb3_flask_yolo_bridge',
            executable='ros_image_to_flask_yolo',
            name='ros_image_to_flask_yolo',
            output='screen',
            parameters=[{
                'image_topic': LaunchConfiguration('image_topic'),
                'input_type': LaunchConfiguration('input_type'),
                'server_url': LaunchConfiguration('server_url'),
                'output_topic': LaunchConfiguration('output_topic'),
                'max_rate_hz': LaunchConfiguration('max_rate_hz'),
                'jpeg_quality': LaunchConfiguration('jpeg_quality'),
                'timeout_sec': LaunchConfiguration('timeout_sec'),
                'publish_debug_image': LaunchConfiguration('publish_debug_image'),
                'debug_image_topic': LaunchConfiguration('debug_image_topic'),
            }],
        ),
    ])
