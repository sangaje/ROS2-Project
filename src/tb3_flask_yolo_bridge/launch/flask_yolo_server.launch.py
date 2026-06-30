from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.substitutions import LaunchConfiguration
from launch_ros.substitutions import FindPackagePrefix
from launch.substitutions import PathJoinSubstitution


def generate_launch_description():
    exe = PathJoinSubstitution([FindPackagePrefix('tb3_flask_yolo_bridge'), 'lib', 'tb3_flask_yolo_bridge', 'flask_yolo_server'])
    return LaunchDescription([
        DeclareLaunchArgument('host', default_value='0.0.0.0'),
        DeclareLaunchArgument('port', default_value='5005'),
        DeclareLaunchArgument('model_path', default_value='yolo11n.pt'),
        DeclareLaunchArgument('device', default_value='cpu'),
        DeclareLaunchArgument('conf', default_value='0.20'),
        DeclareLaunchArgument('imgsz', default_value='640'),
        DeclareLaunchArgument('debug_jpeg_quality', default_value='80'),
        ExecuteProcess(
            cmd=[
                exe,
                '--host', LaunchConfiguration('host'),
                '--port', LaunchConfiguration('port'),
                '--model-path', LaunchConfiguration('model_path'),
                '--device', LaunchConfiguration('device'),
                '--conf', LaunchConfiguration('conf'),
                '--imgsz', LaunchConfiguration('imgsz'),
                '--debug-jpeg-quality', LaunchConfiguration('debug_jpeg_quality'),
                '--person-only',
            ],
            output='screen',
            name='flask_yolo_server',
        ),
    ])
