"""Jetson(Waffle) 측 노드 일괄 실행: waffle_node + yolo_node + fire_node + target_bridge + scan_processor.

turtlebot3_bringup 은 별도 실행 (README.md 참고).

사용:
    ros2 launch omx_aim jetson.launch.py
    ros2 launch omx_aim jetson.launch.py debug_stream:=true
    ros2 launch omx_aim jetson.launch.py debug_stream:=true debug_port:=8090

debug_stream:=true 이면 yolo_node 가 in-process Flask MJPEG 스트림을 띄움.
    브라우저에서 http://<jetson-ip>:<debug_port>/ 로 확인
    (Tailscale 쓰면 http://100.79.57.117:8080/ 처럼 orin-jetson IP 로 접속)
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def launch_setup(context, *args, **kwargs):
    debug_stream = LaunchConfiguration('debug_stream').perform(context) == 'true'
    debug_port = LaunchConfiguration('debug_port').perform(context)

    # yolo_node CLI 인자 조립
    yolo_args = ['--no-display']
    if debug_stream:
        yolo_args += ['--debug-stream', '--debug-port', debug_port]

    return [
        Node(
            package='omx_aim', executable='waffle_node', name='waffle_node',
            output='screen',
        ),
        Node(
            package='omx_aim', executable='yolo_node', name='omx_yolo_node',
            output='screen',
            arguments=yolo_args,
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
    ]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            'debug_stream', default_value='false',
            description='yolo_node 의 Flask MJPEG 디버그 스트림 켜기'),
        DeclareLaunchArgument(
            'debug_port', default_value='8080',
            description='디버그 스트림 포트'),
        OpaqueFunction(function=launch_setup),
    ])