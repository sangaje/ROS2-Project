import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, IncludeLaunchDescription, SetEnvironmentVariable
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def default_tb3_world_map():
    candidates = []
    for pkg in ('turtlebot3_navigation2', 'nav2_bringup'):
        try:
            share = get_package_share_directory(pkg)
        except Exception:
            continue
        candidates.extend([
            os.path.join(share, 'map', 'map.yaml'),
            os.path.join(share, 'maps', 'map.yaml'),
            os.path.join(share, 'maps', 'tb3_sandbox.yaml'),
        ])
    for path in candidates:
        if os.path.exists(path):
            return path
    return ''


def generate_launch_description():
    webcam_launch = PathJoinSubstitution([
        FindPackageShare('tb3_flask_yolo_bridge'),
        'launch',
        'opencv_camera_publisher.launch.py',
    ])
    risk_launch = PathJoinSubstitution([
        FindPackageShare('tb3_bayesian_risk_map'),
        'launch',
        'bayesian_risk_map.launch.py',
    ])
    map_odom_script = os.path.join(
        get_package_share_directory('tb3_fleet_bringup'),
        'scripts',
        'map_odom_localization.py',
    )

    return LaunchDescription([
        DeclareLaunchArgument('domain_id', default_value='21'),
        DeclareLaunchArgument('use_sim_time', default_value='true'),
        DeclareLaunchArgument('map', default_value=default_tb3_world_map()),
        DeclareLaunchArgument('start_map_server', default_value='true'),
        DeclareLaunchArgument('start_map_odom_tf', default_value='true'),
        DeclareLaunchArgument('start_webcam_source', default_value='true'),
        DeclareLaunchArgument('start_risk_map', default_value='true'),
        DeclareLaunchArgument('start_opencv_yolo_view', default_value='true'),
        DeclareLaunchArgument('image_topic', default_value='/camera/image_raw'),
        DeclareLaunchArgument('debug_image_topic', default_value='/risk/debug_yolo_image'),
        DeclareLaunchArgument('webcam_device', default_value='/dev/video0'),
        DeclareLaunchArgument('webcam_width', default_value='640'),
        DeclareLaunchArgument('webcam_height', default_value='480'),
        DeclareLaunchArgument('webcam_fps', default_value='15.0'),
        DeclareLaunchArgument('show_webcam_preview', default_value='false'),
        DeclareLaunchArgument('opencv_view_resize_width', default_value='960'),
        DeclareLaunchArgument('model_path', default_value='yolo11n.pt'),
        DeclareLaunchArgument('device', default_value='cpu'),
        DeclareLaunchArgument('conf', default_value='0.20'),
        DeclareLaunchArgument('map_topic', default_value='/map'),
        DeclareLaunchArgument('map_frame', default_value='map'),
        DeclareLaunchArgument('odom_frame', default_value='odom'),
        DeclareLaunchArgument('base_frame', default_value='base_footprint'),
        DeclareLaunchArgument('odom_topic', default_value='/odom'),
        DeclareLaunchArgument('initial_x', default_value='0.0'),
        DeclareLaunchArgument('initial_y', default_value='0.0'),
        DeclareLaunchArgument('initial_yaw', default_value='0.0'),

        SetEnvironmentVariable('ROS_DOMAIN_ID', LaunchConfiguration('domain_id')),
        SetEnvironmentVariable('RMW_IMPLEMENTATION', 'rmw_fastrtps_cpp'),
        SetEnvironmentVariable('FASTDDS_BUILTIN_TRANSPORTS', 'UDPv4'),

        Node(
            condition=IfCondition(LaunchConfiguration('start_map_server')),
            package='nav2_map_server',
            executable='map_server',
            name='risk_static_map_server',
            output='screen',
            parameters=[{
                'use_sim_time': LaunchConfiguration('use_sim_time'),
                'yaml_filename': LaunchConfiguration('map'),
                'topic_name': LaunchConfiguration('map_topic'),
                'frame_id': LaunchConfiguration('map_frame'),
            }],
        ),
        Node(
            condition=IfCondition(LaunchConfiguration('start_map_server')),
            package='nav2_lifecycle_manager',
            executable='lifecycle_manager',
            name='risk_static_map_lifecycle',
            output='screen',
            parameters=[{
                'use_sim_time': LaunchConfiguration('use_sim_time'),
                'autostart': True,
                'node_names': ['risk_static_map_server'],
            }],
        ),
        ExecuteProcess(
            condition=IfCondition(LaunchConfiguration('start_map_odom_tf')),
            cmd=[
                'python3', map_odom_script,
                '--ros-args',
                '-r', '__node:=risk_static_map_odom_localization',
                '-p', ['use_sim_time:=', LaunchConfiguration('use_sim_time')],
                '-p', 'robot_name:=tb3_static_world_test',
                '-p', ['map_frame:=', LaunchConfiguration('map_frame')],
                '-p', ['odom_frame:=', LaunchConfiguration('odom_frame')],
                '-p', ['base_frame:=', LaunchConfiguration('base_frame')],
                '-p', ['odom_topic:=', LaunchConfiguration('odom_topic')],
                '-p', ['initial_x:=', LaunchConfiguration('initial_x')],
                '-p', ['initial_y:=', LaunchConfiguration('initial_y')],
                '-p', ['initial_yaw:=', LaunchConfiguration('initial_yaw')],
                '-p', 'publish_rate_hz:=30.0',
                '-p', 'publish_amcl_pose:=true',
            ],
            output='screen',
            name='risk_static_map_odom_localization',
        ),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(webcam_launch),
            condition=IfCondition(LaunchConfiguration('start_webcam_source')),
            launch_arguments={
                'device': LaunchConfiguration('webcam_device'),
                'image_topic': LaunchConfiguration('image_topic'),
                'frame_id': 'camera_link',
                'width': LaunchConfiguration('webcam_width'),
                'height': LaunchConfiguration('webcam_height'),
                'fps': LaunchConfiguration('webcam_fps'),
                'show_preview': LaunchConfiguration('show_webcam_preview'),
            }.items(),
        ),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(risk_launch),
            condition=IfCondition(LaunchConfiguration('start_risk_map')),
            launch_arguments={
                'use_sim_time': LaunchConfiguration('use_sim_time'),
                'map_topic': LaunchConfiguration('map_topic'),
                'image_topic': LaunchConfiguration('image_topic'),
                'map_frame': LaunchConfiguration('map_frame'),
                'base_frame': LaunchConfiguration('base_frame'),
                'detection_source': 'local_yolo',
                'enable_yolo': 'true',
                'debug_image_topic': LaunchConfiguration('debug_image_topic'),
                'conf_threshold': LaunchConfiguration('conf'),
                'model_path': LaunchConfiguration('model_path'),
                'device': LaunchConfiguration('device'),
                'debug_show_opencv': 'false',
            }.items(),
        ),
        Node(
            condition=IfCondition(LaunchConfiguration('start_opencv_yolo_view')),
            package='tb3_bayesian_risk_map',
            executable='opencv_yolo_viewer_node',
            name='opencv_yolo_viewer_node',
            output='screen',
            parameters=[{
                'image_topic': LaunchConfiguration('debug_image_topic'),
                'resize_width': LaunchConfiguration('opencv_view_resize_width'),
                'enable_image_view': True,
                'grid_topics': '/risk/detection_candidate_map,/risk/positive_memory_map,/risk/risk_map,/risk/combined_priority_map',
            }],
        ),
    ])
