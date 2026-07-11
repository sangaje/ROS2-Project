from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction, ExecuteProcess
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from ament_index_python.packages import get_package_share_directory
from fleet_bringup.launch_utils import dds_launch_environment
import os


def _as_int(context, name, default):
    value = LaunchConfiguration(name).perform(context)
    try:
        return int(value)
    except Exception:
        return int(default)


def _make_camera_node(context, *args, **kwargs):
    width = _as_int(context, 'camera_width', 640)
    height = _as_int(context, 'camera_height', 480)
    return [
        Node(
            condition=IfCondition(LaunchConfiguration('start_camera')),
            package='v4l2_camera',
            executable='v4l2_camera_node',
            name='usb_camera',
            output='screen',
            parameters=[{
                'use_sim_time': LaunchConfiguration('use_sim_time'),
                'video_device': LaunchConfiguration('camera_device'),
                'image_size': [width, height],
                'pixel_format': LaunchConfiguration('camera_pixel_format'),
                'output_encoding': LaunchConfiguration('camera_output_encoding'),
                'time_per_frame': [1, max(1, _as_int(context, 'camera_fps', 15))],
            }],
            remappings=[
                ('image_raw', LaunchConfiguration('image_topic')),
                ('camera_info', LaunchConfiguration('camera_info_topic')),
            ],
        )
    ]


def generate_launch_description():
    pkg_share = get_package_share_directory('bayesian_risk_map')
    default_config = os.path.join(pkg_share, 'config', 'bayesian_risk_map.yaml')
    default_rviz = os.path.join(pkg_share, 'rviz', 'bayesian_risk_map.rviz')
    rviz_clean = PathJoinSubstitution([
        FindPackageShare('bayesian_risk_map'),
        'scripts',
        'rviz2_clean_env.bash',
    ])

    cartographer_config_dir = PathJoinSubstitution([
        FindPackageShare('bayesian_risk_map'),
        'config',
    ])

    robot_launch = PathJoinSubstitution([
        FindPackageShare('turtlebot3_bringup'),
        'launch',
        'robot.launch.py',
    ])

    return LaunchDescription([
        DeclareLaunchArgument('use_sim_time', default_value='false'),

        DeclareLaunchArgument('start_robot_bringup', default_value='false'),
        DeclareLaunchArgument('start_camera', default_value='true'),
        DeclareLaunchArgument('start_cartographer', default_value='true'),
        DeclareLaunchArgument('start_risk_map', default_value='true'),
        DeclareLaunchArgument('start_rviz', default_value='false'),
        DeclareLaunchArgument('start_teleop', default_value='false'),
        DeclareLaunchArgument('start_opencv_yolo_view', default_value='false'),
        DeclareLaunchArgument('start_rqt_yolo_view', default_value='false'),

        DeclareLaunchArgument('camera_device', default_value='/dev/video0'),
        DeclareLaunchArgument('camera_width', default_value='640'),
        DeclareLaunchArgument('camera_height', default_value='480'),
        DeclareLaunchArgument('camera_fps', default_value='15'),
        DeclareLaunchArgument('camera_pixel_format', default_value='MJPG'),
        DeclareLaunchArgument('camera_output_encoding', default_value='rgb8'),
        DeclareLaunchArgument('camera_info_topic', default_value='/camera/camera_info'),

        DeclareLaunchArgument('cartographer_configuration_basename', default_value='turtlebot3_lds_2d_risk_safe.lua'),
        DeclareLaunchArgument('cartographer_configuration_directory', default_value=cartographer_config_dir),
        DeclareLaunchArgument('cartographer_resolution', default_value='0.05'),
        DeclareLaunchArgument('cartographer_publish_period_sec', default_value='1.0'),

        DeclareLaunchArgument('config_file', default_value=default_config),
        DeclareLaunchArgument('map_topic', default_value='/map'),
        DeclareLaunchArgument('image_topic', default_value='/camera/image_raw'),
        DeclareLaunchArgument('map_frame', default_value='map'),
        DeclareLaunchArgument('base_frame', default_value='base_footprint'),
        *dds_launch_environment(None),

        DeclareLaunchArgument('detection_source', default_value='local_yolo'),
        DeclareLaunchArgument('external_detection_topic', default_value='/risk/yolo_detections'),
        DeclareLaunchArgument('debug_image_topic', default_value='/risk/debug_yolo_image'),
        DeclareLaunchArgument('enable_yolo', default_value='true'),
        DeclareLaunchArgument('enable_fake_detection', default_value='false'),
        DeclareLaunchArgument('model_path', default_value='yolo11n.pt'),
        DeclareLaunchArgument('device', default_value='cpu'),
        DeclareLaunchArgument('conf_threshold', default_value='0.25'),
        DeclareLaunchArgument('camera_hfov_deg', default_value='60.0'),
        DeclareLaunchArgument('yolo_imgsz', default_value='320'),
        DeclareLaunchArgument('yolo_max_rate_hz', default_value='1.0'),
        DeclareLaunchArgument('yolo_async', default_value='true'),
        DeclareLaunchArgument('detection_timeout_sec', default_value='2.0'),
        DeclareLaunchArgument('detection_reuse_max_distance_m', default_value='0.50'),
        DeclareLaunchArgument('external_detection_max_count', default_value='64'),
        DeclareLaunchArgument('update_rate_hz', default_value='2.0'),
        DeclareLaunchArgument('risk_publish_rate_hz', default_value='5.0'),
        DeclareLaunchArgument('diagnostic_publish_rate_hz', default_value='1.0'),

        DeclareLaunchArgument('opencv_camera_device', default_value='/dev/video0'),
        DeclareLaunchArgument('opencv_camera_width', default_value='640'),
        DeclareLaunchArgument('opencv_camera_height', default_value='480'),
        DeclareLaunchArgument('opencv_camera_fps', default_value='15.0'),
        DeclareLaunchArgument('opencv_camera_buffer_size', default_value='1'),
        DeclareLaunchArgument('opencv_async_capture', default_value='true'),
        DeclareLaunchArgument('opencv_camera_fourcc', default_value='MJPG'),
        DeclareLaunchArgument('risk_persist_in_unknown', default_value='true'),

        DeclareLaunchArgument('publish_overlay', default_value='true'),
        DeclareLaunchArgument('publish_debug_image', default_value='true'),
        DeclareLaunchArgument('publish_debug_compressed_image', default_value='false'),
        DeclareLaunchArgument('debug_compressed_image_topic', default_value='/risk/debug_yolo_image/compressed'),
        DeclareLaunchArgument('debug_compressed_jpeg_quality', default_value='70'),
        DeclareLaunchArgument('debug_compressed_resize_width', default_value='480'),
        DeclareLaunchArgument('debug_compressed_publish_rate_hz', default_value='3.0'),
        DeclareLaunchArgument('debug_show_opencv', default_value='false'),
        DeclareLaunchArgument('debug_save_images', default_value='false'),
        DeclareLaunchArgument('debug_log_image_status', default_value='true'),
        DeclareLaunchArgument('preserve_risk_on_map_resize', default_value='true'),
        DeclareLaunchArgument('teleop_mode', default_value='true'),
        DeclareLaunchArgument('region_update_period_sec', default_value='1.5'),
        DeclareLaunchArgument('visibility_num_rays', default_value='48'),
        DeclareLaunchArgument('enable_room_probability', default_value='false'),
        DeclareLaunchArgument('enable_region_segmentation', default_value='true'),
        DeclareLaunchArgument('enable_visibility_tracking', default_value='true'),
        DeclareLaunchArgument('source_halo_radius_m', default_value='0.75'),
        DeclareLaunchArgument('source_halo_sigma_m', default_value='0.35'),

        DeclareLaunchArgument('rviz_config', default_value=default_rviz),
        DeclareLaunchArgument('opencv_view_resize_width', default_value='960'),

        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(robot_launch),
            condition=IfCondition(LaunchConfiguration('start_robot_bringup')),
            launch_arguments={'use_sim_time': LaunchConfiguration('use_sim_time')}.items(),
        ),

        OpaqueFunction(function=_make_camera_node),

        Node(
            condition=IfCondition(LaunchConfiguration('start_cartographer')),
            package='cartographer_ros',
            executable='cartographer_node',
            name='cartographer_node',
            output='screen',
            parameters=[{'use_sim_time': LaunchConfiguration('use_sim_time')}],
            arguments=[
                '-configuration_directory', LaunchConfiguration('cartographer_configuration_directory'),
                '-configuration_basename', LaunchConfiguration('cartographer_configuration_basename'),
            ],
        ),

        Node(
            condition=IfCondition(LaunchConfiguration('start_cartographer')),
            package='cartographer_ros',
            executable='cartographer_occupancy_grid_node',
            name='cartographer_occupancy_grid_node',
            output='screen',
            parameters=[{'use_sim_time': LaunchConfiguration('use_sim_time')}],
            arguments=[
                '-resolution', LaunchConfiguration('cartographer_resolution'),
                '-publish_period_sec', LaunchConfiguration('cartographer_publish_period_sec'),
            ],
        ),

        Node(
            condition=IfCondition(LaunchConfiguration('start_risk_map')),
            package='bayesian_risk_map',
            executable='bayesian_risk_map_node',
            name='bayesian_risk_map_node',
            output='screen',
            parameters=[
                LaunchConfiguration('config_file'),
                {
                    'use_sim_time': LaunchConfiguration('use_sim_time'),
                    'map_topic': LaunchConfiguration('map_topic'),
                    'image_topic': LaunchConfiguration('image_topic'),
                    'map_frame': LaunchConfiguration('map_frame'),
                    'base_frame': LaunchConfiguration('base_frame'),
                    'detection_source': LaunchConfiguration('detection_source'),
                    'external_detection_topic': LaunchConfiguration('external_detection_topic'),
                    'debug_image_topic': LaunchConfiguration('debug_image_topic'),
                    'enable_yolo': LaunchConfiguration('enable_yolo'),
                    'enable_fake_detection': LaunchConfiguration('enable_fake_detection'),
                    'model_path': LaunchConfiguration('model_path'),
                    'device': LaunchConfiguration('device'),
                    'conf_threshold': LaunchConfiguration('conf_threshold'),
                    'camera_hfov_deg': LaunchConfiguration('camera_hfov_deg'),
                    'yolo_imgsz': LaunchConfiguration('yolo_imgsz'),
                    'yolo_max_rate_hz': LaunchConfiguration('yolo_max_rate_hz'),
                    'yolo_async': LaunchConfiguration('yolo_async'),
                    'detection_timeout_sec': LaunchConfiguration('detection_timeout_sec'),
                    'detection_reuse_max_distance_m': LaunchConfiguration('detection_reuse_max_distance_m'),
                    'external_detection_max_count': LaunchConfiguration('external_detection_max_count'),
                    'update_rate_hz': LaunchConfiguration('update_rate_hz'),
                    'risk_publish_rate_hz': LaunchConfiguration('risk_publish_rate_hz'),
                    'diagnostic_publish_rate_hz': LaunchConfiguration('diagnostic_publish_rate_hz'),
                    'opencv_camera_device': LaunchConfiguration('opencv_camera_device'),
                    'opencv_camera_width': LaunchConfiguration('opencv_camera_width'),
                    'opencv_camera_height': LaunchConfiguration('opencv_camera_height'),
                    'opencv_camera_fps': LaunchConfiguration('opencv_camera_fps'),
                    'opencv_camera_buffer_size': LaunchConfiguration('opencv_camera_buffer_size'),
                    'opencv_async_capture': LaunchConfiguration('opencv_async_capture'),
                    'opencv_camera_fourcc': LaunchConfiguration('opencv_camera_fourcc'),
                    'risk_persist_in_unknown': LaunchConfiguration('risk_persist_in_unknown'),
                    'publish_overlay': LaunchConfiguration('publish_overlay'),
                    'publish_debug_image': LaunchConfiguration('publish_debug_image'),
                    'publish_debug_compressed_image': LaunchConfiguration('publish_debug_compressed_image'),
                    'debug_compressed_image_topic': LaunchConfiguration('debug_compressed_image_topic'),
                    'debug_compressed_jpeg_quality': LaunchConfiguration('debug_compressed_jpeg_quality'),
                    'debug_compressed_resize_width': LaunchConfiguration('debug_compressed_resize_width'),
                    'debug_compressed_publish_rate_hz': LaunchConfiguration('debug_compressed_publish_rate_hz'),
                    'debug_show_opencv': LaunchConfiguration('debug_show_opencv'),
                    'debug_save_images': LaunchConfiguration('debug_save_images'),
                    'debug_log_image_status': LaunchConfiguration('debug_log_image_status'),
                    'preserve_risk_on_map_resize': LaunchConfiguration('preserve_risk_on_map_resize'),
                    'teleop_mode': LaunchConfiguration('teleop_mode'),
                    'region_update_period_sec': LaunchConfiguration('region_update_period_sec'),
                    'visibility_num_rays': LaunchConfiguration('visibility_num_rays'),
                    'enable_room_probability': LaunchConfiguration('enable_room_probability'),
                    'enable_region_segmentation': LaunchConfiguration('enable_region_segmentation'),
                    'enable_visibility_tracking': LaunchConfiguration('enable_visibility_tracking'),
                    'source_halo_radius_m': LaunchConfiguration('source_halo_radius_m'),
                    'source_halo_sigma_m': LaunchConfiguration('source_halo_sigma_m'),
                },
            ],
        ),

        ExecuteProcess(
            condition=IfCondition(LaunchConfiguration('start_teleop')),
            cmd=['ros2', 'run', 'turtlebot3_teleop', 'teleop_keyboard'],
            output='screen',
            emulate_tty=True,
        ),

        ExecuteProcess(
            condition=IfCondition(LaunchConfiguration('start_rviz')),
            cmd=[
                rviz_clean,
                '-d', LaunchConfiguration('rviz_config'),
                '--ros-args',
                '-r', '__node:=rviz2_real_robot_risk_map',
                '-p', ['use_sim_time:=', LaunchConfiguration('use_sim_time')],
            ],
            name='rviz2_real_robot_risk_map',
            output='screen',
        ),

        Node(
            condition=IfCondition(LaunchConfiguration('start_opencv_yolo_view')),
            package='bayesian_risk_map',
            executable='opencv_yolo_viewer_node',
            name='opencv_yolo_viewer_node',
            output='screen',
            parameters=[{
                'image_topic': LaunchConfiguration('debug_image_topic'),
                'resize_width': LaunchConfiguration('opencv_view_resize_width'),
            }],
        ),

        Node(
            condition=IfCondition(LaunchConfiguration('start_rqt_yolo_view')),
            package='rqt_image_view',
            executable='rqt_image_view',
            name='rqt_yolo_debug_view',
            output='screen',
            arguments=[LaunchConfiguration('debug_image_topic')],
        ),
    ])
