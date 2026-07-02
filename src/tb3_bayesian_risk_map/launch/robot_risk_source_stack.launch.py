from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, IncludeLaunchDescription, SetEnvironmentVariable, TimerAction
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    bringup_share = get_package_share_directory('tb3_fleet_bringup')
    tf_pose_script = os.path.join(bringup_share, 'scripts', 'tf_pose_publisher_direct_v44.py')

    robot_launch = PathJoinSubstitution([
        FindPackageShare('turtlebot3_bringup'),
        'launch',
        'robot.launch.py',
    ])
    opencv_sender_launch = PathJoinSubstitution([
        FindPackageShare('tb3_flask_yolo_bridge'),
        'launch',
        'opencv_camera_to_flask_yolo.launch.py',
    ])
    cartographer_config_dir = PathJoinSubstitution([
        FindPackageShare('tb3_bayesian_risk_map'),
        'config',
    ])
    default_tb3_param_dir = PathJoinSubstitution([
        FindPackageShare('tb3_bayesian_risk_map'),
        'config',
        'turtlebot3_burger_no_odom_tf.yaml',
    ])
    rviz_config = PathJoinSubstitution([
        FindPackageShare('tb3_bayesian_risk_map'),
        'rviz',
        'slam_risk_live.rviz',
    ])

    return LaunchDescription([
        DeclareLaunchArgument('domain_id', default_value='21'),
        DeclareLaunchArgument('use_sim_time', default_value='false'),
        DeclareLaunchArgument('turtlebot3_model', default_value='burger'),
        DeclareLaunchArgument('lds_model', default_value='LDS-02'),
        DeclareLaunchArgument('tb3_param_dir', default_value=default_tb3_param_dir),

        DeclareLaunchArgument('start_robot_bringup', default_value='true'),
        DeclareLaunchArgument('start_cartographer', default_value='true'),
        DeclareLaunchArgument('start_pose_publisher', default_value='true'),
        DeclareLaunchArgument('start_camera_sender', default_value='true'),
        DeclareLaunchArgument('start_teleop', default_value='false'),
        DeclareLaunchArgument('start_robot_rviz', default_value='false'),
        DeclareLaunchArgument('camera_sender_start_delay_sec', default_value='4.0'),
        DeclareLaunchArgument('cartographer_start_delay_sec', default_value='12.0'),
        DeclareLaunchArgument('pose_publisher_start_delay_sec', default_value='18.0'),
        DeclareLaunchArgument('rviz_config', default_value=rviz_config),

        DeclareLaunchArgument('server_url', default_value='http://10.10.14.58:5005/detect'),

        DeclareLaunchArgument('camera_device', default_value='/dev/video1'),
        DeclareLaunchArgument('camera_fallback_devices', default_value='/dev/video1,/dev/video0,/dev/video2,/dev/video3'),
        DeclareLaunchArgument('camera_width', default_value='320'),
        DeclareLaunchArgument('camera_height', default_value='240'),
        DeclareLaunchArgument('send_width', default_value='320'),
        DeclareLaunchArgument('send_height', default_value='240'),
        DeclareLaunchArgument('camera_fps', default_value='15.0'),
        DeclareLaunchArgument('camera_fourcc', default_value='MJPG'),
        DeclareLaunchArgument('jpeg_quality', default_value='40'),
        DeclareLaunchArgument('yolo_max_rate_hz', default_value='4.0'),
        DeclareLaunchArgument('http_timeout_sec', default_value='0.8'),
        DeclareLaunchArgument('http_connect_timeout_sec', default_value='0.25'),
        DeclareLaunchArgument('http_read_timeout_sec', default_value='0.8'),
        DeclareLaunchArgument('max_http_roundtrip_sec', default_value='0.9'),
        DeclareLaunchArgument('max_frame_age_sec', default_value='1.0'),
        DeclareLaunchArgument('camera_retry_open_period_sec', default_value='1.0'),

        DeclareLaunchArgument('map_topic', default_value='/map'),
        DeclareLaunchArgument('map_frame', default_value='map'),
        DeclareLaunchArgument('base_frame', default_value='base_footprint'),
        DeclareLaunchArgument('pose_topic', default_value='/leader_pose'),
        DeclareLaunchArgument('detection_topic', default_value='/risk/yolo_detections'),
        DeclareLaunchArgument('cartographer_configuration_directory', default_value=cartographer_config_dir),
        DeclareLaunchArgument('cartographer_configuration_basename', default_value='turtlebot3_lds_2d_risk_safe_no_odom.lua'),
        DeclareLaunchArgument('cartographer_resolution', default_value='0.05'),
        DeclareLaunchArgument('cartographer_publish_period_sec', default_value='1.0'),

        SetEnvironmentVariable('ROS_DOMAIN_ID', LaunchConfiguration('domain_id')),
        SetEnvironmentVariable('RMW_IMPLEMENTATION', 'rmw_fastrtps_cpp'),
        SetEnvironmentVariable('FASTDDS_BUILTIN_TRANSPORTS', 'UDPv4'),
        SetEnvironmentVariable('ROS_LOCALHOST_ONLY', '0'),
        SetEnvironmentVariable('ROS_AUTOMATIC_DISCOVERY_RANGE', 'SUBNET'),
        SetEnvironmentVariable('TURTLEBOT3_MODEL', LaunchConfiguration('turtlebot3_model')),
        SetEnvironmentVariable('LDS_MODEL', LaunchConfiguration('lds_model')),

        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(robot_launch),
            condition=IfCondition(LaunchConfiguration('start_robot_bringup')),
            launch_arguments={
                'use_sim_time': LaunchConfiguration('use_sim_time'),
                'tb3_param_dir': LaunchConfiguration('tb3_param_dir'),
            }.items(),
        ),
        TimerAction(
            period=LaunchConfiguration('cartographer_start_delay_sec'),
            actions=[
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
            ],
        ),
        TimerAction(
            period=LaunchConfiguration('camera_sender_start_delay_sec'),
            actions=[
                IncludeLaunchDescription(
                    PythonLaunchDescriptionSource(opencv_sender_launch),
                    condition=IfCondition(LaunchConfiguration('start_camera_sender')),
                    launch_arguments={
                        'device': LaunchConfiguration('camera_device'),
                        'frame_id': 'camera_link',
                        'fallback_devices': LaunchConfiguration('camera_fallback_devices'),
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
                        'connect_timeout_sec': LaunchConfiguration('http_connect_timeout_sec'),
                        'read_timeout_sec': LaunchConfiguration('http_read_timeout_sec'),
                        'max_http_roundtrip_sec': LaunchConfiguration('max_http_roundtrip_sec'),
                        'max_frame_age_sec': LaunchConfiguration('max_frame_age_sec'),
                        'retry_open_period_sec': LaunchConfiguration('camera_retry_open_period_sec'),
                    }.items(),
                ),
            ],
        ),
        TimerAction(
            period=LaunchConfiguration('pose_publisher_start_delay_sec'),
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
        Node(
            condition=IfCondition(LaunchConfiguration('start_robot_rviz')),
            package='rviz2',
            executable='rviz2',
            name='rviz2_robot_risk_source_stack',
            output='screen',
            arguments=['-d', LaunchConfiguration('rviz_config')],
            parameters=[{'use_sim_time': LaunchConfiguration('use_sim_time')}],
        ),
    ])
