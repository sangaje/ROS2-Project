#!/usr/bin/env python3
"""
통합 시뮬레이션 런치: Gazebo 헤드리스(GUI 없음) + RViz2
==========================================================
fast_turtlebot3_world.launch.py 를 include해서 Gazebo 서버/RSP/스폰을
그대로 재사용하고, RViz2 + 선택적 Cartographer만 추가한다.

사용법:
  ros2 launch turtlebot3_rl_training sim_headless_rviz.launch.py

주요 인수:
  gui:=false               (기본) Gazebo GUI 없이 서버만 실행
  gui:=true                Gazebo GUI도 함께 실행
  start_cartographer:=false  (기본) train_sac이 Cartographer를 관리
  start_cartographer:=true   이 런치에서 직접 Cartographer 시작
  start_rviz:=true         (기본) RViz2 실행
  x_pose:=-2.80            로봇 초기 X 위치
  y_pose:=0.96             로봇 초기 Y 위치
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    TimerAction,
)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _pkg_share(package_name: str, fallback: str = "") -> str:
    try:
        return get_package_share_directory(package_name)
    except Exception:
        return fallback


def generate_launch_description() -> LaunchDescription:
    # ── 패키지 경로 ────────────────────────────────────────────────────
    tb3_cartographer_share = _pkg_share("turtlebot3_cartographer")
    this_pkg_share = _pkg_share("turtlebot3_rl_training")

    # 이 패키지의 fast_turtlebot3_world.launch.py (Gazebo 서버+RSP+스폰 담당)
    source_fast_launch = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "fast_turtlebot3_world.launch.py")
    )
    installed_fast_launch = os.path.join(
        this_pkg_share, "launch", "fast_turtlebot3_world.launch.py"
    )
    prefer_source_launch = os.environ.get("TB3_RL_PREFER_SOURCE_LAUNCH", "1").strip().lower()
    fast_launch_path = (
        source_fast_launch
        if prefer_source_launch not in {"0", "false", "no", "off"} and os.path.exists(source_fast_launch)
        else installed_fast_launch
        if os.path.exists(installed_fast_launch)
        else source_fast_launch
    )

    # RViz 설정 파일: 설치된 경로 우선, 없으면 소스 트리
    source_rviz = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "rviz", "rl_sim_headless.rviz")
    )
    installed_rviz = os.path.join(this_pkg_share, "rviz", "rl_sim_headless.rviz")
    rviz_default = (
        source_rviz
        if prefer_source_launch not in {"0", "false", "no", "off"} and os.path.exists(source_rviz)
        else installed_rviz
        if os.path.exists(installed_rviz)
        else source_rviz
    )

    source_training_world = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "world", "training_house.sdf")
    )
    installed_training_world = os.path.join(this_pkg_share, "world", "training_house.sdf")
    tb3_gazebo_share = _pkg_share("turtlebot3_gazebo")
    tb3_house_world = os.path.join(tb3_gazebo_share, "worlds", "turtlebot3_house.world") if tb3_gazebo_share else ""
    world_default = os.environ.get("SIM_WORLD", "").strip()
    if not world_default:
        if os.path.exists(source_training_world):
            world_default = source_training_world
        elif os.path.exists(installed_training_world):
            world_default = installed_training_world
        else:
            world_default = tb3_house_world

    cartographer_config_default = (
        os.path.join(tb3_cartographer_share, "config") if tb3_cartographer_share else ""
    )

    # ── Launch 인수 ────────────────────────────────────────────────────
    use_sim_time = LaunchConfiguration("use_sim_time")
    gui = LaunchConfiguration("gui")
    verbose = LaunchConfiguration("verbose")
    update_rate = LaunchConfiguration("update_rate")
    x_pose = LaunchConfiguration("x_pose")
    y_pose = LaunchConfiguration("y_pose")
    start_rviz = LaunchConfiguration("start_rviz")
    start_cartographer = LaunchConfiguration("start_cartographer")
    rviz_config = LaunchConfiguration("rviz_config")
    world = LaunchConfiguration("world")

    args = [
        DeclareLaunchArgument("use_sim_time", default_value="true"),
        DeclareLaunchArgument(
            "gui",
            default_value="false",
            description="Gazebo GUI 클라이언트 실행 여부. false=헤드리스.",
        ),
        DeclareLaunchArgument("verbose", default_value="1"),
        DeclareLaunchArgument(
            "update_rate",
            default_value=os.environ.get("SIM_UPDATE_RATE", "200"),
            description="Gazebo server update rate cap in Hz. Use 0 for uncapped.",
        ),
        DeclareLaunchArgument("x_pose", default_value="-2.80"),
        DeclareLaunchArgument("y_pose", default_value="0.96"),
        DeclareLaunchArgument("start_rviz", default_value="true"),
        DeclareLaunchArgument(
            "start_cartographer",
            default_value="false",
            description="train_sac 사용 시 false 유지.",
        ),
        DeclareLaunchArgument("rviz_config", default_value=rviz_default),
        DeclareLaunchArgument(
            "world",
            default_value=world_default,
            description="Optional SDF world override. Default uses training_house.sdf.",
        ),
    ]

    # ── Gazebo 서버 + RSP + 로봇 스폰 (fast_turtlebot3_world 재사용) ──
    gazebo_cmd = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(fast_launch_path),
        launch_arguments={
            "use_sim_time": use_sim_time,
            "gui": gui,
            "verbose": verbose,
            "update_rate": update_rate,
            "x_pose": x_pose,
            "y_pose": y_pose,
            "world": world,
        }.items(),
    )

    # ── Cartographer SLAM (선택 사항) ──────────────────────────────────
    cartographer_nodes: list = []
    if tb3_cartographer_share:
        cartographer_nodes.append(
            Node(
                package="cartographer_ros",
                executable="cartographer_node",
                name="cartographer_node",
                output="screen",
                parameters=[{"use_sim_time": use_sim_time}],
                arguments=[
                    "-configuration_directory",
                    cartographer_config_default,
                    "-configuration_basename",
                    "turtlebot3_lds_2d.lua",
                ],
                condition=IfCondition(start_cartographer),
            )
        )
        cartographer_nodes.append(
            Node(
                package="cartographer_ros",
                executable="occupancy_grid_node",
                name="occupancy_grid_node",
                output="screen",
                parameters=[
                    {"use_sim_time": use_sim_time},
                    {"resolution": 0.05},
                    {"publish_period_sec": 0.5},
                ],
                condition=IfCondition(start_cartographer),
            )
        )

    # ── RViz2 (2초 딜레이: Gazebo/RSP가 먼저 뜬 후 실행) ──────────────
    rviz_node = TimerAction(
        period=2.0,
        actions=[
            Node(
                package="rviz2",
                executable="rviz2",
                name="rviz2_rl",
                output="screen",
                arguments=["-d", rviz_config],
                parameters=[{"use_sim_time": use_sim_time}],
                condition=IfCondition(start_rviz),
            )
        ],
    )

    # ── LaunchDescription 조립 ─────────────────────────────────────────
    ld = LaunchDescription(args)
    ld.add_action(gazebo_cmd)
    for node in cartographer_nodes:
        ld.add_action(node)
    ld.add_action(rviz_node)
    return ld
