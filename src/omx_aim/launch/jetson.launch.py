"""Jetson/OMX AIM component launch for the system_bringup leader role.

사용:
    ros2 launch system_bringup system.launch.py role:=leader debug_stream:=true

Standalone component debug only:
    ros2 launch omx_aim jetson.launch.py debug_stream:=true

start_yolo_server:=true 이면 정찰/팔로워 로봇이 HTTP로 프레임을 보낼
flask_yolo_server 도 이 Jetson에서 같이 실행한다.
debug_stream:=true 이면 yolo_node 가 기존 in-process Flask MJPEG 스트림을 띄운다.
    브라우저에서 http://<jetson-ip>:<debug_port>/ 로 확인
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _is_true(value):
    return str(value).strip().lower() in ('1', 'true', 'yes', 'on')


def launch_setup(context, *args, **kwargs):
    debug_stream = _is_true(LaunchConfiguration('debug_stream').perform(context))
    start_yolo_server = _is_true(
        LaunchConfiguration('start_yolo_server').perform(context)
    )
    debug_port = LaunchConfiguration('debug_port').perform(context)

    # yolo_node CLI 인자 조립
    yolo_args = ['--no-display']
    if debug_stream:
        yolo_args += ['--debug-stream', '--debug-port', debug_port]

    actions = []

    if start_yolo_server:
        yolo_server_launch = os.path.join(
            get_package_share_directory('flask_yolo_bridge'),
            'launch',
            'flask_yolo_server.launch.py',
        )
        actions.append(IncludeLaunchDescription(
            PythonLaunchDescriptionSource(yolo_server_launch),
            launch_arguments={
                'host': LaunchConfiguration('yolo_server_host').perform(context),
                'port': LaunchConfiguration('yolo_server_port').perform(context),
                'model_path': LaunchConfiguration('yolo_server_model_path').perform(context),
                'device': LaunchConfiguration('yolo_server_device').perform(context),
                'half': LaunchConfiguration('yolo_server_half').perform(context),
            }.items(),
        ))

    actions.extend([
        Node(
            package='omx_aim', executable='waffle_node', name='waffle_node',
            output='screen',
        ),
        Node(
            package='omx_aim', executable='yolo_node', name='omx_yolo_node',
            output='screen',
            arguments=yolo_args,
            parameters=[{
                'waffle_frame_candidates': ['base_link', 'base_footprint'],
            }],
        ),
        Node(
            package='omx_aim', executable='fire_node', name='fire_node',
            output='screen',
        ),
        Node(
            package='omx_aim', executable='target_bridge', name='target_bridge',
            output='screen',
        ),
        Node(
            package='omx_aim', executable='scan_processor', name='scan_processor',
            output='screen',
        ),
        Node(
            package='omx_aim', executable='patrol_planner', name='patrol_planner',
            output='screen',
        ),
    ])

    return actions


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            'start_yolo_server', default_value='true',
            choices=['true', 'false'],
            description='Run flask_yolo_server on this Jetson for scout/follower camera offload.'),
        DeclareLaunchArgument(
            'yolo_server_host', default_value='0.0.0.0',
            description='flask_yolo_server bind address.'),
        DeclareLaunchArgument(
            'yolo_server_port', default_value='5005',
            description='flask_yolo_server HTTP port.'),
        DeclareLaunchArgument(
            'yolo_server_model_path', default_value='yolo11n.pt',
            description='YOLO model path for flask_yolo_server.'),
        DeclareLaunchArgument(
            'yolo_server_device', default_value='0',
            description='YOLO device for flask_yolo_server.'),
        DeclareLaunchArgument(
            'yolo_server_half', default_value='true',
            choices=['true', 'false'],
            description='Use half precision in flask_yolo_server when supported.'),
        DeclareLaunchArgument(
            'debug_stream', default_value='false',
            description='yolo_node 의 Flask MJPEG 디버그 스트림 켜기'),
        DeclareLaunchArgument(
            'debug_port', default_value='8080',
            description='디버그 스트림 포트'),
        OpaqueFunction(function=launch_setup),
    ])
