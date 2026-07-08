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
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    LogInfo,
    OpaqueFunction,
    TimerAction,
)
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
    start_patrol_planner = _is_true(
        LaunchConfiguration('start_patrol_planner').perform(context)
    )
    debug_port = LaunchConfiguration('debug_port').perform(context)
    yolo_node_delay = float(
        LaunchConfiguration('yolo_node_delay_sec').perform(context)
    )
    patrol_delay = float(
        LaunchConfiguration('patrol_planner_delay_sec').perform(context)
    )

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

    yolo_node = Node(
        package='omx_aim', executable='yolo_node', name='omx_yolo_node',
        output='screen',
        arguments=yolo_args,
        parameters=[{
            'waffle_frame_candidates': ['base_link', 'base_footprint'],
        }],
        respawn=True,
        respawn_delay=3.0,
    )

    actions.extend([
        Node(
            package='omx_aim', executable='waffle_node', name='waffle_node',
            output='screen',
            parameters=[{
                'require_amcl_ready': False,
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
    ])
    if yolo_node_delay > 0.0:
        actions.append(TimerAction(
            period=yolo_node_delay,
            actions=[
                LogInfo(msg=[
                    'OMX_AIM | starting omx_yolo_node after ',
                    str(yolo_node_delay),
                    's stagger to avoid Jetson startup contention',
                ]),
                yolo_node,
            ],
        ))
    else:
        actions.append(yolo_node)

    patrol_planner = Node(
        package='omx_aim', executable='patrol_planner', name='patrol_planner',
        output='screen',
        parameters=[{
            'min_risk': int(LaunchConfiguration('patrol_min_risk').perform(context)),
            'relative_threshold_ratio': float(
                LaunchConfiguration('patrol_relative_threshold_ratio').perform(context)
            ),
            'min_fallback_risk': int(
                LaunchConfiguration('patrol_min_fallback_risk').perform(context)
            ),
            'max_candidate_cells': int(
                LaunchConfiguration('patrol_max_candidate_cells').perform(context)
            ),
        }],
    )
    if start_patrol_planner:
        if patrol_delay > 0.0:
            actions.append(TimerAction(
                period=patrol_delay,
                actions=[
                    LogInfo(msg=[
                        'OMX_AIM | starting patrol_planner after ',
                        str(patrol_delay),
                        's risk-map bridge grace period',
                    ]),
                    patrol_planner,
                ],
            ))
        else:
            actions.append(patrol_planner)

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
            'yolo_server_model_path', default_value='best.pt',
            description='YOLO model path for flask_yolo_server.'),
        DeclareLaunchArgument(
            'yolo_server_device', default_value='0',
            description='YOLO device for flask_yolo_server.'),
        DeclareLaunchArgument(
            'yolo_server_half', default_value='true',
            choices=['true', 'false'],
            description='Use half precision in flask_yolo_server when supported.'),
        DeclareLaunchArgument(
            'start_patrol_planner', default_value='true',
            choices=['true', 'false'],
            description='Start patrol_planner on the leader after the risk bridge grace period.'),
        DeclareLaunchArgument(
            'yolo_node_delay_sec', default_value='14.0',
            description='Delay heavy OMX YOLO/camera/model startup on constrained Jetson hardware.'),
        DeclareLaunchArgument(
            'patrol_planner_delay_sec', default_value='6.0',
            description='Small grace before starting patrol_planner.'),
        DeclareLaunchArgument(
            'patrol_min_risk', default_value='40',
            description='Absolute 0-100 risk cutoff for patrol candidate extraction.'),
        DeclareLaunchArgument(
            'patrol_relative_threshold_ratio', default_value='0.55',
            description='Fallback cutoff ratio of current risk peak when peak is below patrol_min_risk.'),
        DeclareLaunchArgument(
            'patrol_min_fallback_risk', default_value='5',
            description='Noise floor for relative patrol candidate extraction.'),
        DeclareLaunchArgument(
            'patrol_max_candidate_cells', default_value='2000',
            description='Maximum top-risk cells evaluated by patrol_planner NMS per cycle.'),
        DeclareLaunchArgument(
            'debug_stream', default_value='false',
            description='yolo_node 의 Flask MJPEG 디버그 스트림 켜기'),
        DeclareLaunchArgument(
            'debug_port', default_value='8080',
            description='디버그 스트림 포트'),
        OpaqueFunction(function=launch_setup),
    ])
