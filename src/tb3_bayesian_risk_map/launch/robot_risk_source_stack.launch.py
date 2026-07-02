from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, IncludeLaunchDescription, SetEnvironmentVariable, TimerAction
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    bringup_share = get_package_share_directory('tb3_fleet_bringup')
    tf_pose_script = os.path.join(bringup_share, 'scripts', 'tf_pose_publisher_direct_v44.py')

    real_robot_risk_slam_launch = PathJoinSubstitution([
        FindPackageShare('tb3_bayesian_risk_map'),
        'launch',
        'real_robot_risk_slam.launch.py',
    ])
    opencv_sender_launch = PathJoinSubstitution([
        FindPackageShare('tb3_flask_yolo_bridge'),
        'launch',
        'opencv_camera_to_flask_yolo.launch.py',
    ])

    return LaunchDescription([
        DeclareLaunchArgument('domain_id', default_value='21'),
        DeclareLaunchArgument('use_sim_time', default_value='false'),
        DeclareLaunchArgument('turtlebot3_model', default_value='burger'),

        DeclareLaunchArgument('start_robot_bringup', default_value='true'),
        DeclareLaunchArgument('start_cartographer', default_value='true'),
        DeclareLaunchArgument('start_pose_publisher', default_value='true'),
        DeclareLaunchArgument('start_camera_sender', default_value='true'),
        DeclareLaunchArgument('start_teleop', default_value='true'),
        DeclareLaunchArgument('start_robot_rviz', default_value='false'),

        DeclareLaunchArgument('server_url', default_value='http://100.96.193.2:5005/detect'),

        DeclareLaunchArgument('camera_device', default_value='/dev/video1'),
        DeclareLaunchArgument('camera_width', default_value='320'),
        DeclareLaunchArgument('camera_height', default_value='240'),
        DeclareLaunchArgument('send_width', default_value='320'),
        DeclareLaunchArgument('send_height', default_value='240'),
        DeclareLaunchArgument('camera_fps', default_value='10.0'),
        DeclareLaunchArgument('camera_fourcc', default_value='MJPG'),
        DeclareLaunchArgument('jpeg_quality', default_value='45'),
        DeclareLaunchArgument('yolo_max_rate_hz', default_value='2.0'),
        DeclareLaunchArgument('http_timeout_sec', default_value='0.8'),
        DeclareLaunchArgument('max_http_roundtrip_sec', default_value='1.0'),
        DeclareLaunchArgument('max_frame_age_sec', default_value='1.2'),

        DeclareLaunchArgument('map_topic', default_value='/map'),
        DeclareLaunchArgument('map_frame', default_value='map'),
        DeclareLaunchArgument('base_frame', default_value='base_footprint'),
        DeclareLaunchArgument('pose_topic', default_value='/leader_pose'),
        DeclareLaunchArgument('detection_topic', default_value='/risk/yolo_detections'),
        DeclareLaunchArgument('cartographer_configuration_basename', default_value='turtlebot3_lds_2d_risk_safe.lua'),
        DeclareLaunchArgument('cartographer_publish_period_sec', default_value='1.0'),

        SetEnvironmentVariable('ROS_DOMAIN_ID', LaunchConfiguration('domain_id')),
        SetEnvironmentVariable('RMW_IMPLEMENTATION', 'rmw_fastrtps_cpp'),
        SetEnvironmentVariable('FASTDDS_BUILTIN_TRANSPORTS', 'UDPv4'),
        SetEnvironmentVariable('ROS_LOCALHOST_ONLY', '0'),
        SetEnvironmentVariable('ROS_AUTOMATIC_DISCOVERY_RANGE', 'SUBNET'),
        SetEnvironmentVariable('TURTLEBOT3_MODEL', LaunchConfiguration('turtlebot3_model')),

        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(real_robot_risk_slam_launch),
            launch_arguments={
                'use_sim_time': LaunchConfiguration('use_sim_time'),
                'start_robot_bringup': LaunchConfiguration('start_robot_bringup'),
                'start_camera': 'false',
                'start_cartographer': LaunchConfiguration('start_cartographer'),
                'start_risk_map': 'false',
                'start_rviz': LaunchConfiguration('start_robot_rviz'),
                'start_teleop': LaunchConfiguration('start_teleop'),
                'start_opencv_yolo_view': 'false',
                'start_rqt_yolo_view': 'false',
                'cartographer_configuration_basename': LaunchConfiguration('cartographer_configuration_basename'),
                'cartographer_publish_period_sec': LaunchConfiguration('cartographer_publish_period_sec'),
                'map_topic': LaunchConfiguration('map_topic'),
                'map_frame': LaunchConfiguration('map_frame'),
                'base_frame': LaunchConfiguration('base_frame'),
                'detection_source': 'flask_topic',
                'enable_yolo': 'false',
                'external_detection_topic': LaunchConfiguration('detection_topic'),
            }.items(),
        ),
        TimerAction(
            period=4.0,
            actions=[
                IncludeLaunchDescription(
                    PythonLaunchDescriptionSource(opencv_sender_launch),
                    condition=IfCondition(LaunchConfiguration('start_camera_sender')),
                    launch_arguments={
                        'device': LaunchConfiguration('camera_device'),
                        'frame_id': 'camera_link',
                        'width': LaunchConfiguration('camera_width'),
                        'height': LaunchConfiguration('camera_height'),
                        'send_width': LaunchConfiguration('send_width'),
                        'send_height': LaunchConfiguration('send_height'),
                        'camera_fps': LaunchConfiguration('camera_fps'),
                        'fourcc': LaunchConfiguration('camera_fourcc'),
                        'server_url': LaunchConfiguration('server_url'),
                        'output_topic': LaunchConfiguration('detection_topic'),
                        'max_rate_hz': LaunchConfiguration('yolo_max_rate_hz'),
                        'jpeg_quality': LaunchConfiguration('jpeg_quality'),
                        'timeout_sec': LaunchConfiguration('http_timeout_sec'),
                        'max_http_roundtrip_sec': LaunchConfiguration('max_http_roundtrip_sec'),
                        'max_frame_age_sec': LaunchConfiguration('max_frame_age_sec'),
                    }.items(),
                ),
            ],
        ),
        TimerAction(
            period=6.0,
            actions=[
                ExecuteProcess(
                    cmd=[
                        'python3', tf_pose_script, '--ros-args',
                        '-r', '__node:=robot_tf_pose_publisher',
                        '-p', ['use_sim_time:=', LaunchConfiguration('use_sim_time')],
                        '-p', ['target_frame:=', LaunchConfiguration('map_frame')],
                        '-p', ['source_frame:=', LaunchConfiguration('base_frame')],
                        '-p', ['output_topic:=', LaunchConfiguration('pose_topic')],
                        '-p', 'publish_rate_hz:=10.0',
                        '-p', 'timeout_sec:=0.05',
                        '-p', 'log_every_n:=100',
                    ],
                    output='screen',
                    name='robot_tf_pose_publisher',
                    condition=IfCondition(LaunchConfiguration('start_pose_publisher')),
                ),
            ],
        ),
    ])
