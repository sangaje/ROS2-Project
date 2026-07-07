"""Burger 없이 정찰 시스템 시뮬: fake_static_map + fake_risk_map.

/scout/map, /scout/risk_map 을 가짜로 발행해 map_relay/patrol_planner 를 테스트.

사용:
    ros2 launch omx_aim sim.launch.py
"""
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package='omx_aim', executable='fake_static_map', name='fake_static_map',
            output='screen',
        ),
        Node(
            package='omx_aim', executable='fake_risk_map', name='fake_risk_map',
            output='screen',
        ),
    ])
