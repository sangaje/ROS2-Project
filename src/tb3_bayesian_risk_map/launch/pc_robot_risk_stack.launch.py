from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    IncludeLaunchDescription,
    SetEnvironmentVariable,
    TimerAction,
    UnsetEnvironmentVariable,
)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    flask_server_launch = PathJoinSubstitution([
        FindPackageShare('tb3_flask_yolo_bridge'),
        'launch',
        'flask_yolo_server.launch.py',
    ])
    central_risk_launch = PathJoinSubstitution([
        FindPackageShare('tb3_bayesian_risk_map'),
        'launch',
        'central_risk_map_domain20.launch.py',
    ])
    monitor_launch = PathJoinSubstitution([
        FindPackageShare('tb3_bayesian_risk_map'),
        'launch',
        'pc_risk_debug_monitor.launch.py',
    ])

    clean_flask_port = ExecuteProcess(
        condition=IfCondition(LaunchConfiguration('start_flask_server')),
        cmd=[
            'bash',
            '-lc',
            'if [ "$1" = "true" ]; then fuser -k "$2"/tcp || true; fi',
            'clean_flask_port',
            LaunchConfiguration('restart_flask_port'),
            LaunchConfiguration('flask_port'),
        ],
        output='screen',
        name='clean_flask_yolo_port',
    )

    return LaunchDescription([
        DeclareLaunchArgument('central_domain_id', default_value='25'),
        DeclareLaunchArgument('robot_domain_id', default_value='24'),
        DeclareLaunchArgument('use_sim_time', default_value='false'),

        DeclareLaunchArgument('start_flask_server', default_value='true'),
        DeclareLaunchArgument('restart_flask_port', default_value='false'),
        DeclareLaunchArgument('start_domain_bridges', default_value='true'),
        DeclareLaunchArgument('start_risk_map', default_value='true'),
        DeclareLaunchArgument('start_rviz', default_value='true'),
        DeclareLaunchArgument('bridge_rviz_topics', default_value='true'),

        DeclareLaunchArgument('flask_host', default_value='0.0.0.0'),
        DeclareLaunchArgument('flask_port', default_value='5005'),
        DeclareLaunchArgument('model_path', default_value='yolo11n.pt'),
        DeclareLaunchArgument('device', default_value='0'),
        DeclareLaunchArgument('half', default_value='true'),
        DeclareLaunchArgument('conf', default_value='0.20'),
        DeclareLaunchArgument('imgsz', default_value='640'),
        DeclareLaunchArgument('debug_jpeg_quality', default_value='65'),
        DeclareLaunchArgument('max_capture_age_sec', default_value='0.8'),
        DeclareLaunchArgument('max_queue_wait_sec', default_value='0.0'),

        DeclareLaunchArgument('map_topic', default_value='/map'),
        DeclareLaunchArgument('map_frame', default_value='map'),
        DeclareLaunchArgument('base_frame', default_value='base_footprint'),
        DeclareLaunchArgument('pose_topic', default_value='/leader_pose'),
        DeclareLaunchArgument('external_detection_topic', default_value='/risk/yolo_detections'),
        DeclareLaunchArgument('risk_publish_rate_hz', default_value='5.0'),

        UnsetEnvironmentVariable('FASTRTPS_DEFAULT_PROFILES_FILE'),
        UnsetEnvironmentVariable('RMW_FASTRTPS_DEFAULT_PROFILES_FILE'),
        UnsetEnvironmentVariable('FASTDDS_DEFAULT_PROFILES_FILE'),
        UnsetEnvironmentVariable('ROS_DISCOVERY_SERVER'),
        UnsetEnvironmentVariable('ROS_STATIC_PEERS'),
        SetEnvironmentVariable('ROS_DOMAIN_ID', LaunchConfiguration('central_domain_id')),
        SetEnvironmentVariable('RMW_IMPLEMENTATION', 'rmw_fastrtps_cpp'),
        SetEnvironmentVariable('FASTDDS_BUILTIN_TRANSPORTS', 'UDPv4'),
        SetEnvironmentVariable('ROS_LOCALHOST_ONLY', '0'),
        SetEnvironmentVariable('ROS_AUTOMATIC_DISCOVERY_RANGE', 'SUBNET'),

        clean_flask_port,
        TimerAction(
            period=1.0,
            actions=[
                IncludeLaunchDescription(
                    PythonLaunchDescriptionSource(flask_server_launch),
                    condition=IfCondition(LaunchConfiguration('start_flask_server')),
                    launch_arguments={
                        'host': LaunchConfiguration('flask_host'),
                        'port': LaunchConfiguration('flask_port'),
                        'model_path': LaunchConfiguration('model_path'),
                        'device': LaunchConfiguration('device'),
                        'half': LaunchConfiguration('half'),
                        'conf': LaunchConfiguration('conf'),
                        'imgsz': LaunchConfiguration('imgsz'),
                        'debug_jpeg_quality': LaunchConfiguration('debug_jpeg_quality'),
                        'max_capture_age_sec': LaunchConfiguration('max_capture_age_sec'),
                        'max_queue_wait_sec': LaunchConfiguration('max_queue_wait_sec'),
                    }.items(),
                ),
            ],
        ),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(central_risk_launch),
            launch_arguments={
                'central_domain_id': LaunchConfiguration('central_domain_id'),
                'source_domain_id': LaunchConfiguration('robot_domain_id'),
                'risk_sink_domain_ids': LaunchConfiguration('robot_domain_id'),
                'use_sim_time': LaunchConfiguration('use_sim_time'),
                'start_domain_bridges': LaunchConfiguration('start_domain_bridges'),
                'start_risk_map': LaunchConfiguration('start_risk_map'),
                'bridge_rviz_topics': LaunchConfiguration('bridge_rviz_topics'),
                'map_topic': LaunchConfiguration('map_topic'),
                'map_frame': LaunchConfiguration('map_frame'),
                'base_frame': LaunchConfiguration('base_frame'),
                'pose_topic': LaunchConfiguration('pose_topic'),
                'external_detection_topic': LaunchConfiguration('external_detection_topic'),
                'risk_publish_rate_hz': LaunchConfiguration('risk_publish_rate_hz'),
            }.items(),
        ),
        TimerAction(
            period=3.0,
            actions=[
                IncludeLaunchDescription(
                    PythonLaunchDescriptionSource(monitor_launch),
                    condition=IfCondition(LaunchConfiguration('start_rviz')),
                    launch_arguments={
                        'domain_id': LaunchConfiguration('central_domain_id'),
                        'use_sim_time': LaunchConfiguration('use_sim_time'),
                        'start_rviz': LaunchConfiguration('start_rviz'),
                        'start_opencv_debug_view': 'false',
                    }.items(),
                ),
            ],
        ),
    ])
