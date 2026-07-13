import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    virtual_env = os.environ.get('VIRTUAL_ENV', '').strip()
    python_exe = (
        os.path.join(virtual_env, 'bin', 'python3')
        if virtual_env else 'python3'
    )
    return LaunchDescription([
        DeclareLaunchArgument('host', default_value='0.0.0.0'),
        DeclareLaunchArgument('port', default_value='5005'),
        DeclareLaunchArgument('model_path', default_value='model/target_v3.engine'),
        DeclareLaunchArgument('target_class', default_value='0'),
        DeclareLaunchArgument('device', default_value='0'),
        DeclareLaunchArgument('half', default_value='true'),
        DeclareLaunchArgument('conf', default_value='0.20'),
        DeclareLaunchArgument('iou', default_value='0.45'),
        DeclareLaunchArgument('max_det', default_value='64'),
        DeclareLaunchArgument('imgsz', default_value='640'),
        DeclareLaunchArgument('debug_jpeg_quality', default_value='52'),
        DeclareLaunchArgument('enable_raw_debug_stream', default_value='false'),
        DeclareLaunchArgument('max_capture_age_sec', default_value='1.5'),
        DeclareLaunchArgument('max_queue_wait_sec', default_value='0.05'),
        ExecuteProcess(
            cmd=[
                python_exe,
                '-m', 'flask_yolo_bridge.flask_yolo_server',
                '--host', LaunchConfiguration('host'),
                '--port', LaunchConfiguration('port'),
                '--model-path', LaunchConfiguration('model_path'),
                '--target-class', LaunchConfiguration('target_class'),
                '--device', LaunchConfiguration('device'),
                '--half', LaunchConfiguration('half'),
                '--conf', LaunchConfiguration('conf'),
                '--iou', LaunchConfiguration('iou'),
                '--max-det', LaunchConfiguration('max_det'),
                '--imgsz', LaunchConfiguration('imgsz'),
                '--debug-jpeg-quality', LaunchConfiguration('debug_jpeg_quality'),
                '--enable-raw-debug-stream', LaunchConfiguration('enable_raw_debug_stream'),
                '--max-capture-age-sec', LaunchConfiguration('max_capture_age_sec'),
                '--max-queue-wait-sec', LaunchConfiguration('max_queue_wait_sec'),
            ],
            output='screen',
            name='flask_yolo_server',
        ),
    ])
