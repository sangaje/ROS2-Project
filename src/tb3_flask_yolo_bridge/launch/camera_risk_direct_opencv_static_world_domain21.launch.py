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
        DeclareLaunchArgument('start_risk_map', default_value='true'),
        DeclareLaunchArgument('debug_show_opencv', default_value='true'),
        DeclareLaunchArgument('opencv_camera_device', default_value='/dev/video0'),
        DeclareLaunchArgument('opencv_camera_width', default_value='640'),
        DeclareLaunchArgument('opencv_camera_height', default_value='480'),
        DeclareLaunchArgument('opencv_camera_fps', default_value='15.0'),
        DeclareLaunchArgument('yolo_max_rate_hz', default_value='3.0'),
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
            PythonLaunchDescriptionSource(risk_launch),
            condition=IfCondition(LaunchConfiguration('start_risk_map')),
            launch_arguments={
                'use_sim_time': LaunchConfiguration('use_sim_time'),
                'map_topic': LaunchConfiguration('map_topic'),
                'map_frame': LaunchConfiguration('map_frame'),
                'base_frame': LaunchConfiguration('base_frame'),
                'detection_source': 'opencv_camera',
                'enable_yolo': 'true',
                'conf_threshold': LaunchConfiguration('conf'),
                'model_path': LaunchConfiguration('model_path'),
                'device': LaunchConfiguration('device'),
                'yolo_max_rate_hz': LaunchConfiguration('yolo_max_rate_hz'),
                'opencv_camera_device': LaunchConfiguration('opencv_camera_device'),
                'opencv_camera_width': LaunchConfiguration('opencv_camera_width'),
                'opencv_camera_height': LaunchConfiguration('opencv_camera_height'),
                'opencv_camera_fps': LaunchConfiguration('opencv_camera_fps'),
                'publish_overlay': 'false',
                'publish_debug_image': 'false',
                'debug_show_opencv': LaunchConfiguration('debug_show_opencv'),
            }.items(),
        ),
    ])
