from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, SetEnvironmentVariable
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    pkg_share = get_package_share_directory('tb3_bayesian_risk_map')
    safe_lua = os.path.join(pkg_share, 'config', 'turtlebot3_lds_2d_risk_safe_no_odom.lua')
    carto_basename = (
        'turtlebot3_lds_2d_risk_safe_no_odom.lua'
        if os.path.exists(safe_lua)
        else 'turtlebot3_lds_2d.lua'
    )

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
        DeclareLaunchArgument('domain_id', default_value='25'),
        DeclareLaunchArgument('model_path', default_value='yolo11n.pt'),
        DeclareLaunchArgument('camera_device', default_value='/dev/video1'),
        DeclareLaunchArgument('camera_width', default_value='640'),
        DeclareLaunchArgument('camera_height', default_value='480'),
        DeclareLaunchArgument('camera_fps', default_value='15'),
        DeclareLaunchArgument('yolo_imgsz', default_value='256'),
        DeclareLaunchArgument('yolo_max_rate_hz', default_value='2.0'),
        DeclareLaunchArgument('yolo_async', default_value='true'),
        DeclareLaunchArgument('conf_threshold', default_value='0.25'),
        DeclareLaunchArgument('debug_compressed_image_topic', default_value='/risk/debug_yolo_image/compressed'),
        DeclareLaunchArgument('debug_compressed_jpeg_quality', default_value='55'),
        DeclareLaunchArgument('debug_compressed_resize_width', default_value='480'),
        DeclareLaunchArgument('debug_compressed_publish_rate_hz', default_value='3.0'),
        DeclareLaunchArgument('opencv_async_capture', default_value='true'),
        DeclareLaunchArgument('risk_persist_in_unknown', default_value='true'),

        SetEnvironmentVariable('ROS_DOMAIN_ID', LaunchConfiguration('domain_id')),
        SetEnvironmentVariable('RMW_IMPLEMENTATION', 'rmw_fastrtps_cpp'),

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
                'cartographer_configuration_basename': carto_basename,
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
                'yolo_async': LaunchConfiguration('yolo_async'),
                'conf_threshold': LaunchConfiguration('conf_threshold'),
                'detection_timeout_sec': '2.0',
                'detection_reuse_max_distance_m': '0.50',
                'external_detection_max_count': '64',
                'camera_device': LaunchConfiguration('camera_device'),
                'camera_width': LaunchConfiguration('camera_width'),
                'camera_height': LaunchConfiguration('camera_height'),
                'camera_fps': LaunchConfiguration('camera_fps'),
                'opencv_camera_device': LaunchConfiguration('camera_device'),
                'opencv_camera_width': LaunchConfiguration('camera_width'),
                'opencv_camera_height': LaunchConfiguration('camera_height'),
                'opencv_camera_fps': LaunchConfiguration('camera_fps'),
                'opencv_camera_buffer_size': '1',
                'opencv_async_capture': LaunchConfiguration('opencv_async_capture'),
                'camera_pixel_format': 'MJPG',
                'camera_output_encoding': 'rgb8',
                'risk_persist_in_unknown': LaunchConfiguration('risk_persist_in_unknown'),
                'publish_overlay': 'false',
                'publish_debug_image': 'false',
                'publish_debug_compressed_image': 'true',
                'debug_compressed_image_topic': LaunchConfiguration('debug_compressed_image_topic'),
                'debug_compressed_jpeg_quality': LaunchConfiguration('debug_compressed_jpeg_quality'),
                'debug_compressed_resize_width': LaunchConfiguration('debug_compressed_resize_width'),
                'debug_compressed_publish_rate_hz': LaunchConfiguration('debug_compressed_publish_rate_hz'),
                'debug_show_opencv': 'false',
            }.items(),
        ),
    ])
