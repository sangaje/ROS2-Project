"""Leader Jetson bringup: fleet leader stack + OMX AIM + debug dashboards.

사용:
    ros2 launch omx_aim jetson.launch.py
    ros2 launch omx_aim jetson.launch.py debug_stream:=true
    ros2 launch omx_aim jetson.launch.py debug_stream:=true debug_port:=8090 dashboard_port:=8091

debug_stream:=true 이면 yolo_node 가 기존 in-process Flask MJPEG 스트림을 띄우고,
leader_unified_dashboard 도 별도 Flask 서버로 같이 실행한다.
    브라우저에서 http://<jetson-ip>:<debug_port>/ 로 확인
    통합 상황판은 http://<jetson-ip>:<dashboard_port>/ 로 확인
    (Tailscale 쓰면 http://100.79.57.117:8080/ 처럼 orin-jetson IP 로 접속)
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import EnvironmentVariable, LaunchConfiguration
from launch_ros.actions import Node


def _is_true(value):
    return str(value).strip().lower() in ('1', 'true', 'yes', 'on')


def launch_setup(context, *args, **kwargs):
    debug_stream = _is_true(LaunchConfiguration('debug_stream').perform(context))
    unified_dashboard = _is_true(
        LaunchConfiguration('unified_dashboard').perform(context)
    )
    start_fleet_leader = _is_true(
        LaunchConfiguration('start_fleet_leader').perform(context)
    )
    debug_port = LaunchConfiguration('debug_port').perform(context)
    dashboard_port = LaunchConfiguration('dashboard_port').perform(context)
    dashboard_host = LaunchConfiguration('dashboard_host').perform(context)

    # yolo_node CLI 인자 조립
    yolo_args = ['--no-display']
    if debug_stream:
        yolo_args += ['--debug-stream', '--debug-port', debug_port]

    actions = []

    if start_fleet_leader:
        system_launch = os.path.join(
            get_package_share_directory('system_bringup'),
            'launch',
            'system.launch.py',
        )
        actions.append(IncludeLaunchDescription(
            PythonLaunchDescriptionSource(system_launch),
            launch_arguments={
                'role': 'leader',
                'domain_id': LaunchConfiguration('domain_id').perform(context),
                'risk_domain_id': LaunchConfiguration('risk_domain_id').perform(context),
                'pc_domain_id': LaunchConfiguration('pc_domain_id').perform(context),
                'member_domain_id': LaunchConfiguration('member_domain_id').perform(context),
                'follower_domain_id': LaunchConfiguration('follower_domain_id').perform(context),
                'require_follower_pose': LaunchConfiguration('require_follower_pose').perform(context),
                'enable_cartographer': LaunchConfiguration('enable_cartographer').perform(context),
                'auto_localize': LaunchConfiguration('auto_localize').perform(context),
                'start_robot_bringup': LaunchConfiguration('start_robot_bringup').perform(context),
                'start_nav2': LaunchConfiguration('start_nav2').perform(context),
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

    if debug_stream and unified_dashboard:
        actions.append(Node(
            package='omx_aim',
            executable='unified_dashboard',
            name='leader_unified_dashboard',
            output='screen',
            parameters=[{
                'host': dashboard_host,
                'port': int(dashboard_port),
                'omx_debug_port': int(debug_port),
                'omx_stream_path': '/stream.mjpg',
                'omx_state_path': '/state.json',
                'map_topic': '/map',
                'risk_topic': '/risk/risk_map',
                'leader_pose_topic': '/leader_pose',
                'follower_pose_topic': '/burger_pose',
                'member_pose_topic': '/member_pose',
                'fleet_poses_topic': '/fleet/robot_poses',
                'fleet_status_topic': '/fleet/coordination_status',
                'collision_warning_topic': '/fleet/collision_warning',
            }],
        ))

    return actions


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            'domain_id', default_value=EnvironmentVariable('ROS_DOMAIN_ID'),
            description='Leader DDS domain passed through to system_bringup.'),
        DeclareLaunchArgument(
            'start_fleet_leader', default_value='true',
            choices=['true', 'false'],
            description='system_bringup role:=leader 를 함께 실행'),
        DeclareLaunchArgument(
            'risk_domain_id', default_value='',
            description='Risk/scout DDS domain to bridge /map and /risk topics into leader.'),
        DeclareLaunchArgument(
            'pc_domain_id', default_value='',
            description='PC DDS domain for optional leader->PC visualization bridge.'),
        DeclareLaunchArgument(
            'member_domain_id', default_value='',
            description='Member/scout domain label for leader debug markers.'),
        DeclareLaunchArgument(
            'follower_domain_id', default_value='',
            description='Follower domain label for leader debug markers.'),
        DeclareLaunchArgument(
            'require_follower_pose', default_value='true',
            choices=['true', 'false'],
            description='Set false for leader-only or leader+member fleets without /burger_pose.'),
        DeclareLaunchArgument(
            'enable_cartographer', default_value='false',
            choices=['true', 'false'],
            description='Leader fleet stack: run local Cartographer instead of AMCL on bridged /map.'),
        DeclareLaunchArgument(
            'auto_localize', default_value='true',
            choices=['true', 'false'],
            description='Leader AMCL global localization when enable_cartographer:=false.'),
        DeclareLaunchArgument(
            'start_robot_bringup', default_value='true',
            choices=['true', 'false'],
            description='Start TurtleBot3 hardware bringup through the fleet leader stack.'),
        DeclareLaunchArgument(
            'start_nav2', default_value='true',
            choices=['true', 'false'],
            description='Start Nav2 through the fleet leader stack.'),
        DeclareLaunchArgument(
            'debug_stream', default_value='false',
            description='yolo_node 의 Flask MJPEG 디버그 스트림 켜기'),
        DeclareLaunchArgument(
            'debug_port', default_value='8080',
            description='디버그 스트림 포트'),
        DeclareLaunchArgument(
            'unified_dashboard', default_value='true',
            choices=['true', 'false'],
            description='debug_stream:=true 일 때 leader 통합 상황판 실행'),
        DeclareLaunchArgument(
            'dashboard_host', default_value='0.0.0.0',
            description='통합 상황판 HTTP bind address'),
        DeclareLaunchArgument(
            'dashboard_port', default_value='8091',
            description='통합 상황판 HTTP 포트'),
        OpaqueFunction(function=launch_setup),
    ])
