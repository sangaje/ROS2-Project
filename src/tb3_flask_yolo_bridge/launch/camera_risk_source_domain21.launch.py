from launch import LaunchDescription
import os

from ament_index_python.packages import get_package_share_directory
from launch.actions import DeclareLaunchArgument, ExecuteProcess, IncludeLaunchDescription, SetEnvironmentVariable, TimerAction
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    bringup_share = get_package_share_directory('tb3_fleet_bringup')
    tf_pose_script = os.path.join(bringup_share, 'scripts', 'tf_pose_publisher_direct_v44.py')

    flask_server_launch = PathJoinSubstitution([
        FindPackageShare('tb3_flask_yolo_bridge'),
        'launch',
        'flask_yolo_server.launch.py',
    ])
    image_sender_launch = PathJoinSubstitution([
        FindPackageShare('tb3_flask_yolo_bridge'),
        'launch',
        'ros_image_to_flask_yolo.launch.py',
    ])
    risk_launch = PathJoinSubstitution([
        FindPackageShare('tb3_bayesian_risk_map'),
        'launch',
        'cartographer_risk_rviz.launch.py',
    ])

    return LaunchDescription([
        DeclareLaunchArgument('domain_id', default_value='21'),
        DeclareLaunchArgument('use_sim_time', default_value='true'),
        DeclareLaunchArgument('start_flask_server', default_value='true'),
        DeclareLaunchArgument('start_image_sender', default_value='true'),
        DeclareLaunchArgument('start_cartographer', default_value='true'),
        DeclareLaunchArgument('start_risk_map', default_value='false'),
        DeclareLaunchArgument('start_pose_publisher', default_value='true'),
        DeclareLaunchArgument('start_domain21_rviz', default_value='false'),
        DeclareLaunchArgument('start_opencv_yolo_view', default_value='false'),
        DeclareLaunchArgument('image_topic', default_value='/camera/image_raw'),
        DeclareLaunchArgument('input_type', default_value='raw'),
        DeclareLaunchArgument('server_url', default_value='http://127.0.0.1:5005/detect'),
        DeclareLaunchArgument('output_topic', default_value='/risk/yolo_detections'),
        DeclareLaunchArgument('debug_image_topic', default_value='/risk/debug_yolo_image'),
        DeclareLaunchArgument('max_rate_hz', default_value='3.0'),
        DeclareLaunchArgument('jpeg_quality', default_value='70'),
        DeclareLaunchArgument('timeout_sec', default_value='1.0'),
        DeclareLaunchArgument('flask_host', default_value='0.0.0.0'),
        DeclareLaunchArgument('flask_port', default_value='5005'),
        DeclareLaunchArgument('model_path', default_value='yolo11n.pt'),
        DeclareLaunchArgument('device', default_value='cpu'),
        DeclareLaunchArgument('conf', default_value='0.20'),
        DeclareLaunchArgument('imgsz', default_value='640'),
        DeclareLaunchArgument('map_topic', default_value='/map'),
        DeclareLaunchArgument('map_frame', default_value='map'),
        DeclareLaunchArgument('base_frame', default_value='base_footprint'),
        DeclareLaunchArgument('pose_topic', default_value='/leader_pose'),

        SetEnvironmentVariable('ROS_DOMAIN_ID', LaunchConfiguration('domain_id')),
        SetEnvironmentVariable('RMW_IMPLEMENTATION', 'rmw_fastrtps_cpp'),
        SetEnvironmentVariable('FASTDDS_BUILTIN_TRANSPORTS', 'UDPv4'),

        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(flask_server_launch),
            condition=IfCondition(LaunchConfiguration('start_flask_server')),
            launch_arguments={
                'host': LaunchConfiguration('flask_host'),
                'port': LaunchConfiguration('flask_port'),
                'model_path': LaunchConfiguration('model_path'),
                'device': LaunchConfiguration('device'),
                'conf': LaunchConfiguration('conf'),
                'imgsz': LaunchConfiguration('imgsz'),
            }.items(),
        ),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(image_sender_launch),
            condition=IfCondition(LaunchConfiguration('start_image_sender')),
            launch_arguments={
                'image_topic': LaunchConfiguration('image_topic'),
                'input_type': LaunchConfiguration('input_type'),
                'server_url': LaunchConfiguration('server_url'),
                'output_topic': LaunchConfiguration('output_topic'),
                'publish_debug_image': 'true',
                'debug_image_topic': LaunchConfiguration('debug_image_topic'),
                'max_rate_hz': LaunchConfiguration('max_rate_hz'),
                'jpeg_quality': LaunchConfiguration('jpeg_quality'),
                'timeout_sec': LaunchConfiguration('timeout_sec'),
            }.items(),
        ),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(risk_launch),
            launch_arguments={
                'use_sim_time': LaunchConfiguration('use_sim_time'),
                'start_cartographer': LaunchConfiguration('start_cartographer'),
                'start_risk_map': LaunchConfiguration('start_risk_map'),
                'start_rviz': LaunchConfiguration('start_domain21_rviz'),
                'start_opencv_yolo_view': LaunchConfiguration('start_opencv_yolo_view'),
                'start_yolo_view': 'false',
                'yolo_view_topic': LaunchConfiguration('debug_image_topic'),
                'map_topic': LaunchConfiguration('map_topic'),
                'map_frame': LaunchConfiguration('map_frame'),
                'base_frame': LaunchConfiguration('base_frame'),
                'detection_source': 'flask_topic',
                'enable_yolo': 'false',
                'external_detection_topic': LaunchConfiguration('output_topic'),
                'conf_threshold': LaunchConfiguration('conf'),
                'model_path': LaunchConfiguration('model_path'),
                'device': LaunchConfiguration('device'),
            }.items(),
        ),
        TimerAction(
            period=3.0,
            actions=[
                ExecuteProcess(
                    cmd=[
                        'python3', tf_pose_script, '--ros-args',
                        '-r', '__node:=domain21_tf_pose_publisher',
                        '-p', ['use_sim_time:=', LaunchConfiguration('use_sim_time')],
                        '-p', ['target_frame:=', LaunchConfiguration('map_frame')],
                        '-p', ['source_frame:=', LaunchConfiguration('base_frame')],
                        '-p', ['output_topic:=', LaunchConfiguration('pose_topic')],
                        '-p', 'publish_rate_hz:=10.0',
                        '-p', 'log_every_n:=100',
                    ],
                    output='screen',
                    name='domain21_tf_pose_publisher',
                    condition=IfCondition(LaunchConfiguration('start_pose_publisher')),
                )
            ],
        ),
    ])
