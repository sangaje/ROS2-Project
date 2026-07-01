from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('device', default_value='/dev/video0'),
        DeclareLaunchArgument('frame_id', default_value='camera_link'),
        DeclareLaunchArgument('width', default_value='320'),
        DeclareLaunchArgument('height', default_value='240'),
        DeclareLaunchArgument('send_width', default_value='320'),
        DeclareLaunchArgument('send_height', default_value='240'),
        DeclareLaunchArgument('camera_fps', default_value='15.0'),
        DeclareLaunchArgument('buffer_size', default_value='1'),
        DeclareLaunchArgument('fourcc', default_value='MJPG'),
        DeclareLaunchArgument('server_url', default_value='http://127.0.0.1:5005/detect'),
        DeclareLaunchArgument('output_topic', default_value='/risk/yolo_detections'),
        DeclareLaunchArgument('max_rate_hz', default_value='5.0'),
        DeclareLaunchArgument('jpeg_quality', default_value='60'),
        DeclareLaunchArgument('timeout_sec', default_value='1.0'),
        DeclareLaunchArgument('publish_empty_detections', default_value='true'),
        Node(
            package='tb3_flask_yolo_bridge',
            executable='opencv_camera_to_flask_yolo',
            name='opencv_camera_to_flask_yolo',
            output='screen',
            parameters=[{
                'device': LaunchConfiguration('device'),
                'frame_id': LaunchConfiguration('frame_id'),
                'width': LaunchConfiguration('width'),
                'height': LaunchConfiguration('height'),
                'send_width': LaunchConfiguration('send_width'),
                'send_height': LaunchConfiguration('send_height'),
                'camera_fps': LaunchConfiguration('camera_fps'),
                'buffer_size': LaunchConfiguration('buffer_size'),
                'fourcc': LaunchConfiguration('fourcc'),
                'server_url': LaunchConfiguration('server_url'),
                'output_topic': LaunchConfiguration('output_topic'),
                'max_rate_hz': LaunchConfiguration('max_rate_hz'),
                'jpeg_quality': LaunchConfiguration('jpeg_quality'),
                'timeout_sec': LaunchConfiguration('timeout_sec'),
                'publish_empty_detections': LaunchConfiguration('publish_empty_detections'),
            }],
        ),
    ])
