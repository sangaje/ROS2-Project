"""Desktop 측 노드 일괄 실행: map_relay + patrol_planner + auto_initialpose + scout_watchdog.

domain_bridge, Nav2, RViz 는 별도 실행 (README.md 참고).

사용:
    ros2 launch omx_aim desktop.launch.py
    ros2 launch omx_aim desktop.launch.py scout_watchdog:=false
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    scout_watchdog_arg = DeclareLaunchArgument(
        'scout_watchdog', default_value='true',
        description='Burger heartbeat 감시 + 수색 TARGET 발행 노드 실행 여부')

    return LaunchDescription([
        scout_watchdog_arg,
        # Node(
        #     package='omx_aim', executable='map_relay', name='map_relay',
        #     output='screen',
        # ),
        Node(
            package='omx_aim', executable='patrol_planner', name='patrol_planner',
            output='screen',
        ),
        Node(
            package='omx_aim', executable='auto_initialpose', name='auto_initialpose',
            output='screen',
        ),
        # Node(
        #     package='omx_aim', executable='scout_watchdog', name='scout_watchdog',
        #     output='screen',
        #     condition=IfCondition(LaunchConfiguration('scout_watchdog')),
        # ),
    ])
