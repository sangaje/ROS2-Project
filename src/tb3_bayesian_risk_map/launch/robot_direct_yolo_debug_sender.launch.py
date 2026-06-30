from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, SetEnvironmentVariable
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    real_robot_launch = PathJoinSubstitution([
        FindPackageShare('tb3_bayesian_risk_map'),
        'launch',
        'real_robot_risk_slam.launch.py',
    ])

    return LaunchDescription([
        DeclareLaunchArgument('domain_id', default_value='25'),
        DeclareLaunchArgument('use_sim_time', default_value='false'),
        DeclareLaunchArgument('start_robot_bringup', default_value='false'),
        DeclareLaunchArgument('start_cartographer', default_value='true'),
        DeclareLaunchArgument('start_risk_map', default_value='true'),
        DeclareLaunchArgument('start_teleop', default_value='false'),
        DeclareLaunchArgument('opencv_camera_device', default_value='/dev/video1'),
        DeclareLaunchArgument('opencv_camera_width', default_value='640'),
        DeclareLaunchArgument('opencv_camera_height', default_value='480'),
        DeclareLaunchArgument('opencv_camera_fps', default_value='15.0'),
        DeclareLaunchArgument('opencv_camera_fourcc', default_value='MJPG'),
        DeclareLaunchArgument('model_path', default_value='yolo11n.pt'),
        DeclareLaunchArgument('device', default_value='cpu'),
        DeclareLaunchArgument('yolo_imgsz', default_value='320'),
        DeclareLaunchArgument('yolo_max_rate_hz', default_value='2.0'),
        DeclareLaunchArgument('conf_threshold', default_value='0.25'),
        DeclareLaunchArgument('debug_compressed_image_topic', default_value='/risk/debug_yolo_image/compressed'),
        DeclareLaunchArgument('debug_compressed_jpeg_quality', default_value='70'),

        SetEnvironmentVariable('ROS_DOMAIN_ID', LaunchConfiguration('domain_id')),
        SetEnvironmentVariable('RMW_IMPLEMENTATION', 'rmw_fastrtps_cpp'),
        SetEnvironmentVariable('FASTDDS_BUILTIN_TRANSPORTS', 'UDPv4'),

        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(real_robot_launch),
            launch_arguments={
                'use_sim_time': LaunchConfiguration('use_sim_time'),
                'start_robot_bringup': LaunchConfiguration('start_robot_bringup'),
                'start_camera': 'false',
                'start_cartographer': LaunchConfiguration('start_cartographer'),
                'start_risk_map': LaunchConfiguration('start_risk_map'),
                'start_rviz': 'false',
                'start_teleop': LaunchConfiguration('start_teleop'),
                'start_opencv_yolo_view': 'false',
                'start_rqt_yolo_view': 'false',
                'detection_source': 'opencv_camera',
                'enable_yolo': 'true',
                'model_path': LaunchConfiguration('model_path'),
                'device': LaunchConfiguration('device'),
                'yolo_imgsz': LaunchConfiguration('yolo_imgsz'),
                'yolo_max_rate_hz': LaunchConfiguration('yolo_max_rate_hz'),
                'conf_threshold': LaunchConfiguration('conf_threshold'),
                'opencv_camera_device': LaunchConfiguration('opencv_camera_device'),
                'opencv_camera_width': LaunchConfiguration('opencv_camera_width'),
                'opencv_camera_height': LaunchConfiguration('opencv_camera_height'),
                'opencv_camera_fps': LaunchConfiguration('opencv_camera_fps'),
                'opencv_camera_fourcc': LaunchConfiguration('opencv_camera_fourcc'),
                'publish_overlay': 'false',
                'publish_debug_image': 'false',
                'publish_debug_compressed_image': 'true',
                'debug_compressed_image_topic': LaunchConfiguration('debug_compressed_image_topic'),
                'debug_compressed_jpeg_quality': LaunchConfiguration('debug_compressed_jpeg_quality'),
                'debug_show_opencv': 'false',
                'debug_save_images': 'false',
                'debug_log_image_status': 'true',
                'teleop_mode': 'true',
            }.items(),
        ),
    ])
