from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    package_config_dir = PathJoinSubstitution([
        FindPackageShare('tb3_bayesian_risk_map'),
        'config',
    ])

    real_robot_launch = PathJoinSubstitution([
        FindPackageShare('tb3_bayesian_risk_map'),
        'launch',
        'real_robot_risk_slam.launch.py',
    ])

    return LaunchDescription([
        DeclareLaunchArgument('use_sim_time', default_value='false'),
        DeclareLaunchArgument('model_path', default_value='yolo11n.pt'),
        DeclareLaunchArgument('camera_device', default_value='/dev/video0'),
        DeclareLaunchArgument('camera_width', default_value='640'),
        DeclareLaunchArgument('camera_height', default_value='480'),
        DeclareLaunchArgument('camera_fps', default_value='15'),
        DeclareLaunchArgument('yolo_imgsz', default_value='320'),
        DeclareLaunchArgument('yolo_max_rate_hz', default_value='1.0'),
        DeclareLaunchArgument('conf_threshold', default_value='0.25'),

        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(real_robot_launch),
            launch_arguments={
                'use_sim_time': LaunchConfiguration('use_sim_time'),
                'start_robot_bringup': 'false',
                'start_camera': 'false',
                'start_cartographer': 'true',
                'start_risk_map': 'true',
                'start_rviz': 'false',
                'start_teleop': 'false',
                'start_opencv_yolo_view': 'false',
                'start_rqt_yolo_view': 'false',
                'cartographer_configuration_directory': package_config_dir,
                'cartographer_configuration_basename': 'turtlebot3_lds_2d_risk_safe.lua',
                'teleop_mode': 'true',
                'risk_publish_rate_hz': '5.0',
                'region_update_period_sec': '1.5',
                'visibility_num_rays': '48',
                'enable_room_probability': 'false',
                'enable_region_segmentation': 'true',
                'enable_visibility_tracking': 'true',
                'detection_source': 'opencv_camera',
                'enable_yolo': 'true',
                'model_path': LaunchConfiguration('model_path'),
                'device': 'cpu',
                'yolo_imgsz': LaunchConfiguration('yolo_imgsz'),
                'yolo_max_rate_hz': LaunchConfiguration('yolo_max_rate_hz'),
                'conf_threshold': LaunchConfiguration('conf_threshold'),
                'detection_timeout_sec': '2.0',
                'detection_reuse_max_distance_m': '0.50',
                'external_detection_max_count': '64',
                'camera_device': LaunchConfiguration('camera_device'),
                'camera_width': LaunchConfiguration('camera_width'),
                'camera_height': LaunchConfiguration('camera_height'),
                'camera_fps': LaunchConfiguration('camera_fps'),
                'camera_pixel_format': 'MJPG',
                'camera_output_encoding': 'rgb8',
                'publish_overlay': 'false',
                'publish_debug_image': 'true',
                'debug_show_opencv': 'false',
            }.items(),
        ),
    ])
