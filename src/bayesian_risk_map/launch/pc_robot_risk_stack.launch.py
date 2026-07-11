from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    IncludeLaunchDescription,
    TimerAction,
)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import EnvironmentVariable, LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare
from fleet_bringup.launch_utils import dds_launch_environment


def generate_launch_description():
    flask_server_launch = PathJoinSubstitution([
        FindPackageShare('flask_yolo_bridge'),
        'launch',
        'flask_yolo_server.launch.py',
    ])
    central_risk_launch = PathJoinSubstitution([
        FindPackageShare('bayesian_risk_map'),
        'launch',
        'central_risk_map_bridge.launch.py',
    ])
    monitor_launch = PathJoinSubstitution([
        FindPackageShare('bayesian_risk_map'),
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
        DeclareLaunchArgument('central_domain_id', default_value=EnvironmentVariable('ROS_DOMAIN_ID')),
        DeclareLaunchArgument(
            'robot_domain_id',
            default_value='',
            description=(
                'Robot DDS domain for PC-side bridges. Required when '
                'start_domain_bridges:=true; pass robot_domain_id:=<robot_domain>.'
            ),
        ),
        DeclareLaunchArgument('use_sim_time', default_value='false'),

        DeclareLaunchArgument('start_flask_server', default_value='true'),
        DeclareLaunchArgument('restart_flask_port', default_value='false'),
        DeclareLaunchArgument('start_domain_bridges', default_value='true'),
        DeclareLaunchArgument('start_risk_map', default_value='true'),
        DeclareLaunchArgument('start_rviz', default_value='true'),
        DeclareLaunchArgument('bridge_rviz_topics', default_value='true'),

        DeclareLaunchArgument('flask_host', default_value='0.0.0.0'),
        DeclareLaunchArgument('flask_port', default_value='5005'),
        DeclareLaunchArgument('model_path', default_value='model/best.pt'),
        DeclareLaunchArgument('device', default_value='0'),
        DeclareLaunchArgument('half', default_value='true'),
        DeclareLaunchArgument('conf', default_value='0.20'),
        DeclareLaunchArgument('imgsz', default_value='960'),
        DeclareLaunchArgument('debug_jpeg_quality', default_value='75'),
        DeclareLaunchArgument('max_capture_age_sec', default_value='1.5'),
        DeclareLaunchArgument('max_queue_wait_sec', default_value='0.05'),

        DeclareLaunchArgument('map_topic', default_value='/map'),
        DeclareLaunchArgument('map_frame', default_value='map'),
        DeclareLaunchArgument('base_frame', default_value='base_footprint'),
        DeclareLaunchArgument('pose_topic', default_value='/leader_pose'),
        DeclareLaunchArgument('external_detection_topic', default_value='/risk/yolo_detections'),
        DeclareLaunchArgument('risk_publish_rate_hz', default_value='5.0'),

        *dds_launch_environment(LaunchConfiguration('central_domain_id')),

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
