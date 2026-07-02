from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('device', default_value='/dev/video1'),
        DeclareLaunchArgument('fallback_devices', default_value='/dev/video1,/dev/video0,/dev/video2,/dev/video3'),
        DeclareLaunchArgument('frame_id', default_value='camera_link'),
        DeclareLaunchArgument('width', default_value='320'),
        DeclareLaunchArgument('height', default_value='240'),
        DeclareLaunchArgument('send_width', default_value='320'),
        DeclareLaunchArgument('send_height', default_value='240'),
        DeclareLaunchArgument('camera_fps', default_value='10.0'),
        DeclareLaunchArgument('buffer_size', default_value='1'),
        DeclareLaunchArgument('fourcc', default_value='MJPG'),
        DeclareLaunchArgument('server_url', default_value='http://100.96.193.2:5005/detect'),
        DeclareLaunchArgument('output_topic', default_value='/risk/yolo_detections'),
        DeclareLaunchArgument('max_rate_hz', default_value='2.0'),
        DeclareLaunchArgument('jpeg_quality', default_value='45'),
        DeclareLaunchArgument('timeout_sec', default_value='0.8'),
        DeclareLaunchArgument('max_http_roundtrip_sec', default_value='1.0'),
        DeclareLaunchArgument('max_frame_age_sec', default_value='1.2'),
        DeclareLaunchArgument('retry_open_period_sec', default_value='1.0'),
        DeclareLaunchArgument('publish_empty_detections', default_value='true'),
        Node(
            package='tb3_flask_yolo_bridge',
            executable='opencv_camera_to_flask_yolo',
            name='opencv_camera_to_flask_yolo',
            output='screen',
            parameters=[{
                'device': LaunchConfiguration('device'),
                'fallback_devices': LaunchConfiguration('fallback_devices'),
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
                'max_http_roundtrip_sec': LaunchConfiguration('max_http_roundtrip_sec'),
                'max_frame_age_sec': LaunchConfiguration('max_frame_age_sec'),
                'retry_open_period_sec': LaunchConfiguration('retry_open_period_sec'),
                'publish_empty_detections': LaunchConfiguration('publish_empty_detections'),
            }],
        ),
    ])
