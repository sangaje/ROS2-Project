#!/usr/bin/env python3

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import AppendEnvironmentVariable
from launch.actions import DeclareLaunchArgument
from launch.actions import IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    tb3_gazebo_share = get_package_share_directory("turtlebot3_gazebo")
    rl_training_share = get_package_share_directory("turtlebot3_rl_training")
    ros_gz_sim_share = get_package_share_directory("ros_gz_sim")

    launch_file_dir = os.path.join(tb3_gazebo_share, "launch")

    use_sim_time = LaunchConfiguration("use_sim_time")
    x_pose = LaunchConfiguration("x_pose")
    y_pose = LaunchConfiguration("y_pose")
    gui = LaunchConfiguration("gui")
    verbose = LaunchConfiguration("verbose")
    update_rate = LaunchConfiguration("update_rate")

    source_training_world = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "world", "training_house.sdf")
    )
    installed_training_world = os.path.join(
        rl_training_share, "world", "training_house.sdf"
    )
    tb3_house_world = os.path.join(tb3_gazebo_share, "worlds", "turtlebot3_house.world")
    default_world = os.environ.get("SIM_WORLD", "").strip()
    if not default_world:
        if os.path.exists(source_training_world):
            default_world = source_training_world
        elif os.path.exists(installed_training_world):
            default_world = installed_training_world
        else:
            default_world = tb3_house_world
    world = LaunchConfiguration("world")

    declare_use_sim_time = DeclareLaunchArgument(
        "use_sim_time",
        default_value="true",
    )

    declare_x_pose = DeclareLaunchArgument(
        "x_pose",
        default_value="-2.80",
    )

    declare_y_pose = DeclareLaunchArgument(
        "y_pose",
        default_value="0.96",
    )

    declare_gui = DeclareLaunchArgument(
        "gui",
        default_value="false",
        description="Whether to launch Gazebo GUI client.",
    )

    declare_verbose = DeclareLaunchArgument(
        "verbose",
        default_value="1",
        description="Gazebo verbosity level.",
    )

    declare_update_rate = DeclareLaunchArgument(
        "update_rate",
        default_value=os.environ.get("SIM_UPDATE_RATE", "200"),
        description="Gazebo server update rate cap in Hz. 200Hz matches max_step_size=0.005s; use 0 for uncapped.",
    )

    declare_world = DeclareLaunchArgument(
        "world",
        default_value=default_world,
        description="SDF world used for RL training. Must contain obstacle geometry.",
    )

    set_env_vars_resources = AppendEnvironmentVariable(
        "GZ_SIM_RESOURCE_PATH",
        os.path.join(tb3_gazebo_share, "models"),
    )

    gzserver_cmd = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(ros_gz_sim_share, "launch", "gz_sim.launch.py")
        ),
        launch_arguments={
            "gz_args": ["-r -s -z ", update_rate, " -v", verbose, " ", world],
            "on_exit_shutdown": "true",
        }.items(),
    )

    gzclient_cmd = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(ros_gz_sim_share, "launch", "gz_sim.launch.py")
        ),
        launch_arguments={
            "gz_args": ["-g -v", verbose],
            "on_exit_shutdown": "true",
        }.items(),
        condition=IfCondition(gui),
    )

    robot_state_publisher_cmd = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(launch_file_dir, "robot_state_publisher.launch.py")
        ),
        launch_arguments={
            "use_sim_time": use_sim_time,
        }.items(),
    )

    spawn_turtlebot_cmd = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(launch_file_dir, "spawn_turtlebot3.launch.py")
        ),
        launch_arguments={
            "x_pose": x_pose,
            "y_pose": y_pose,
        }.items(),
    )

    ld = LaunchDescription()

    ld.add_action(declare_use_sim_time)
    ld.add_action(declare_x_pose)
    ld.add_action(declare_y_pose)
    ld.add_action(declare_gui)
    ld.add_action(declare_verbose)
    ld.add_action(declare_update_rate)
    ld.add_action(declare_world)

    # 중요: Gazebo 실행 전에 resource path를 먼저 잡는다.
    ld.add_action(set_env_vars_resources)

    # Gazebo server
    ld.add_action(gzserver_cmd)

    # GUI는 기본 false. 필요할 때만 gui:=true
    ld.add_action(gzclient_cmd)

    # Robot
    ld.add_action(robot_state_publisher_cmd)
    ld.add_action(spawn_turtlebot_cmd)

    return ld
