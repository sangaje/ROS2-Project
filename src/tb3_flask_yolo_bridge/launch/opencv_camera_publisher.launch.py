from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('device', default_value='/dev/video0'),
        DeclareLaunchArgument('image_topic', default_value='/camera/image_raw'),
        DeclareLaunchArgument('frame_id', default_value='camera_link'),
        DeclareLaunchArgument('width', default_value='640'),
        DeclareLaunchArgument('height', default_value='480'),
        DeclareLaunchArgument('fps', default_value='15.0'),
        DeclareLaunchArgument('show_preview', default_value='false'),
        Node(
            package='tb3_flask_yolo_bridge',
            executable='opencv_camera_publisher',
            name='opencv_camera_publisher',
            output='screen',
            parameters=[{
                'device': LaunchConfiguration('device'),
                'image_topic': LaunchConfiguration('image_topic'),
                'frame_id': LaunchConfiguration('frame_id'),
                'width': LaunchConfiguration('width'),
                'height': LaunchConfiguration('height'),
                'fps': LaunchConfiguration('fps'),
                'show_preview': LaunchConfiguration('show_preview'),
            }],
        ),
    ])
