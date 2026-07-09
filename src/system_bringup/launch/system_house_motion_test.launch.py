#!/usr/bin/env python3
"""Motion-only three-robot Gazebo House integration test.

This is intentionally perception-free: no YOLO, no Bayesian risk map, no OMX,
and no camera decision nodes.  system_bringup remains the root while this file
uses fleet_bringup and multi/region providers underneath.
"""

from __future__ import annotations

import copy
import os
import tempfile
from pathlib import Path

import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    AppendEnvironmentVariable,
    DeclareLaunchArgument,
    ExecuteProcess,
    LogInfo,
    OpaqueFunction,
    TimerAction,
)
from launch.conditions import IfCondition, UnlessCondition
from launch.substitutions import EnvironmentVariable, LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.actions import PushRosNamespace
from launch.actions import GroupAction

from fleet_bringup.launch_utils import dds_launch_environment


RUNTIME_DIR = Path(tempfile.gettempdir()) / 'system_house_motion_test'


def _read_yaml(path: Path) -> dict:
    with path.open('r', encoding='utf-8') as handle:
        return yaml.safe_load(handle) or {}


def _write_yaml(path: Path, data: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding='utf-8')
    return path


def _tb3_share() -> Path:
    return Path(get_package_share_directory('turtlebot3_gazebo'))


def _find_tb3_file(*parts: str) -> Path:
    candidate = _tb3_share().joinpath(*parts)
    if candidate.exists():
        return candidate
    source_candidate = Path('/home/seil/turtlebot3_ws/src/turtlebot3_simulations/turtlebot3_gazebo').joinpath(*parts)
    if source_candidate.exists():
        return source_candidate
    raise FileNotFoundError(f'turtlebot3_gazebo file not found: {"/".join(parts)}')


def _nav_map_yaml() -> Path:
    try:
        nav_share = Path(get_package_share_directory('turtlebot3_navigation2'))
        candidate = nav_share / 'map' / 'map.yaml'
        if candidate.exists():
            return candidate
    except Exception:
        pass
    multi_share = Path(get_package_share_directory('multi'))
    return multi_share / 'maps' / 'turtlemap.yaml'


def _replace_robot_sdf(template: Path, namespace: str) -> str:
    text = template.read_text(encoding='utf-8')
    replacements = {
        '<model name="turtlebot3_burger">': f'<model name="{namespace}">',
        '<model name="turtlebot3_waffle_pi">': f'<model name="{namespace}">',
        '<model name="turtlebot3_waffle">': f'<model name="{namespace}">',
        '<topic>imu</topic>': f'<topic>/{namespace}/imu</topic>',
        '<topic>scan</topic>': f'<topic>/{namespace}/scan</topic>',
        '<gz_frame_id>base_scan</gz_frame_id>': f'<gz_frame_id>{namespace}/base_scan</gz_frame_id>',
        '<topic>camera/image_raw</topic>': f'<topic>/{namespace}/camera/image_raw</topic>',
        '<gz_frame_id>camera_rgb_frame</gz_frame_id>': f'<gz_frame_id>{namespace}/camera_rgb_frame</gz_frame_id>',
        '<topic>cmd_vel</topic>': f'<topic>/{namespace}/cmd_vel</topic>',
        '<odom_topic>odom</odom_topic>': f'<odom_topic>/{namespace}/odom</odom_topic>',
        '<frame_id>odom</frame_id>': f'<frame_id>{namespace}/odom</frame_id>',
        '<child_frame_id>base_footprint</child_frame_id>': (
            f'<child_frame_id>{namespace}/base_footprint</child_frame_id>'
        ),
        '<topic>joint_states</topic>': f'<topic>/{namespace}/joint_states</topic>',
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def _model_template_for(model: str) -> Path:
    model_dir = 'turtlebot3_waffle_pi' if model == 'waffle_pi' else 'turtlebot3_burger'
    return _find_tb3_file('models', model_dir, 'model.sdf')


def _urdf_for(model: str) -> str:
    name = 'turtlebot3_waffle_pi.urdf' if model == 'waffle_pi' else 'turtlebot3_burger.urdf'
    return _find_tb3_file('urdf', name).read_text(encoding='utf-8')


def _write_model_dir(namespace: str, model: str) -> None:
    model_dir = RUNTIME_DIR / 'models' / namespace
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / 'model.sdf').write_text(
        _replace_robot_sdf(_model_template_for(model), namespace),
        encoding='utf-8',
    )
    (model_dir / 'model.config').write_text(
        f"""<?xml version="1.0"?>
<model>
  <name>{namespace}</name>
  <version>1.0</version>
  <sdf version="1.8">model.sdf</sdf>
  <author><name>system_bringup</name></author>
  <description>Runtime namespaced TurtleBot3 for motion-only House tests.</description>
</model>
""",
        encoding='utf-8',
    )


def _write_world(config: dict) -> Path:
    house_model = _find_tb3_file('models', 'turtlebot3_house')
    world_path = RUNTIME_DIR / 'worlds' / 'system_house_motion_test.world'
    world_path.parent.mkdir(parents=True, exist_ok=True)
    robots = config['robots']

    include_blocks = []
    for key in ('leader', 'field_a', 'field_b'):
        robot = robots[key]
        ns = robot['namespace']
        spawn = robot['spawn']
        include_blocks.append(f"""
    <include>
      <uri>model://{ns}</uri>
      <name>{ns}</name>
      <pose>{spawn['x']} {spawn['y']} 0.01 0 0 {spawn['yaw']}</pose>
    </include>
""")

    world_path.write_text(
        f"""<?xml version="1.0"?>
<sdf version="1.8">
  <world name="default">
    <scene><shadows>0</shadows></scene>
    <physics type="ode">
      <real_time_update_rate>1000.0</real_time_update_rate>
      <max_step_size>0.001</max_step_size>
      <real_time_factor>1</real_time_factor>
    </physics>
    <plugin filename="gz-sim-physics-system" name="gz::sim::systems::Physics"/>
    <plugin filename="gz-sim-user-commands-system" name="gz::sim::systems::UserCommands"/>
    <plugin filename="gz-sim-scene-broadcaster-system" name="gz::sim::systems::SceneBroadcaster"/>
    <plugin filename="gz-sim-sensors-system" name="gz::sim::systems::Sensors">
      <render_engine>ogre2</render_engine>
    </plugin>
    <plugin filename="gz-sim-imu-system" name="gz::sim::systems::Imu"/>
    <include>
      <uri>https://fuel.gazebosim.org/1.0/OpenRobotics/models/Ground Plane</uri>
    </include>
    <include>
      <uri>https://fuel.gazebosim.org/1.0/OpenRobotics/models/Sun</uri>
    </include>
    <model name="turtlebot3_house">
      <static>1</static>
      <include><uri>model://turtlebot3_house</uri></include>
    </model>
{''.join(include_blocks)}
  </world>
</sdf>
""",
        encoding='utf-8',
    )
    return world_path


def _nav2_base_config(config: dict) -> dict:
    nav = config['nav2']
    return {
        'amcl': {'ros__parameters': {
            'use_sim_time': True,
            'alpha1': 0.05,
            'alpha2': 0.05,
            'alpha3': 0.05,
            'alpha4': 0.05,
            'alpha5': 0.03,
            'base_frame_id': 'base_footprint',
            'odom_frame_id': 'odom',
            'global_frame_id': 'map',
            'scan_topic': 'scan',
            'map_topic': '/map',
            'robot_model_type': 'nav2_amcl::DifferentialMotionModel',
            'laser_model_type': 'likelihood_field',
            'laser_min_range': 0.12,
            'laser_max_range': 3.5,
            'laser_likelihood_max_dist': 2.0,
            'max_beams': 90,
            'min_particles': 1000,
            'max_particles': 4000,
            'pf_err': 0.02,
            'pf_z': 0.99,
            'update_min_a': 0.05,
            'update_min_d': 0.05,
            'transform_tolerance': 1.5,
            'set_initial_pose': True,
            'initial_pose': {'x': 0.0, 'y': 0.0, 'z': 0.0, 'yaw': 0.0},
        }},
        'lifecycle_manager_localization': {'ros__parameters': {
            'use_sim_time': True,
            'autostart': True,
            'bond_timeout': 8.0,
            'node_names': ['amcl'],
        }},
        'controller_server': {'ros__parameters': {
            'use_sim_time': True,
            'controller_frequency': float(nav['controller_frequency']),
            'min_x_velocity_threshold': 0.001,
            'min_y_velocity_threshold': 0.5,
            'min_theta_velocity_threshold': 0.001,
            'failure_tolerance': 2.0,
            'odom_topic': 'odom',
            'enable_stamped_cmd_vel': True,
            'progress_checker_plugins': ['progress_checker'],
            'goal_checker_plugins': ['goal_checker'],
            'controller_plugins': ['FollowPath'],
            'progress_checker': {
                'plugin': 'nav2_controller::SimpleProgressChecker',
                'required_movement_radius': 0.02,
                'movement_time_allowance': 45.0,
            },
            'goal_checker': {
                'plugin': 'nav2_controller::SimpleGoalChecker',
                'stateful': False,
                'xy_goal_tolerance': 0.16,
                'yaw_goal_tolerance': 0.65,
            },
            'FollowPath': {
                'plugin': 'dwb_core::DWBLocalPlanner',
                'debug_trajectory_details': False,
                'min_vel_x': -0.04,
                'min_vel_y': 0.0,
                'max_vel_x': float(nav['max_vel_x']),
                'max_vel_y': 0.0,
                'max_vel_theta': float(nav['max_vel_theta']),
                'min_speed_xy': 0.0,
                'max_speed_xy': float(nav['max_vel_x']),
                'min_speed_theta': 0.0,
                'acc_lim_x': float(nav['acceleration_x']),
                'acc_lim_y': 0.0,
                'acc_lim_theta': float(nav['acceleration_theta']),
                'decel_lim_x': -float(nav['acceleration_x']),
                'decel_lim_y': 0.0,
                'decel_lim_theta': -float(nav['acceleration_theta']),
                'vx_samples': 24,
                'vy_samples': 1,
                'vtheta_samples': 36,
                'sim_time': 1.0,
                'linear_granularity': 0.025,
                'angular_granularity': 0.02,
                'transform_tolerance': 1.5,
                'xy_goal_tolerance': 0.16,
                'trans_stopped_velocity': 0.03,
                'short_circuit_trajectory_evaluation': True,
                'stateful': True,
                'critics': ['RotateToGoal', 'Oscillation', 'BaseObstacle', 'GoalAlign', 'PathAlign', 'PathDist', 'GoalDist'],
                'BaseObstacle.scale': 0.03,
                'PathAlign.scale': 20.0,
                'PathAlign.forward_point_distance': 0.08,
                'GoalAlign.scale': 12.0,
                'GoalAlign.forward_point_distance': 0.08,
                'PathDist.scale': 24.0,
                'GoalDist.scale': 16.0,
                'RotateToGoal.scale': 16.0,
                'RotateToGoal.slowing_factor': 3.0,
                'RotateToGoal.lookahead_time': -1.0,
            },
        }},
        'planner_server': {'ros__parameters': {
            'use_sim_time': True,
            'expected_planner_frequency': float(nav['planner_frequency']),
            'planner_plugins': ['GridBased'],
            'GridBased': {
                'plugin': 'nav2_navfn_planner::NavfnPlanner',
                'tolerance': 0.20,
                'use_astar': True,
                'allow_unknown': False,
            },
        }},
        'behavior_server': {'ros__parameters': {
            'use_sim_time': True,
            'local_costmap_topic': 'local_costmap/costmap_raw',
            'global_costmap_topic': 'global_costmap/costmap_raw',
            'local_footprint_topic': 'local_costmap/published_footprint',
            'global_footprint_topic': 'global_costmap/published_footprint',
            'cycle_frequency': 10.0,
            'behavior_plugins': ['spin', 'backup', 'drive_on_heading', 'wait'],
            'spin': {'plugin': 'nav2_behaviors::Spin'},
            'backup': {'plugin': 'nav2_behaviors::BackUp'},
            'drive_on_heading': {'plugin': 'nav2_behaviors::DriveOnHeading'},
            'wait': {'plugin': 'nav2_behaviors::Wait'},
            'global_frame': 'map',
            'robot_base_frame': 'base_footprint',
            'transform_tolerance': 1.5,
            'simulate_ahead_time': 1.2,
            'max_rotational_vel': float(nav['max_vel_theta']),
            'min_rotational_vel': 0.10,
            'rotational_acc_lim': float(nav['acceleration_theta']),
            'enable_stamped_cmd_vel': True,
        }},
        'bt_navigator': {'ros__parameters': {
            'use_sim_time': True,
            'global_frame': 'map',
            'robot_base_frame': 'base_footprint',
            'odom_topic': 'odom',
            'bt_loop_duration': 10,
            'default_server_timeout': 20,
            'wait_for_service_timeout': 1000,
            'navigators': ['navigate_to_pose', 'navigate_through_poses'],
            'navigate_to_pose': {'plugin': 'nav2_bt_navigator::NavigateToPoseNavigator'},
            'navigate_through_poses': {'plugin': 'nav2_bt_navigator::NavigateThroughPosesNavigator'},
        }},
        'local_costmap': {'local_costmap': {'ros__parameters': {
            'use_sim_time': True,
            'update_frequency': 10.0,
            'publish_frequency': 5.0,
            'global_frame': 'odom',
            'robot_base_frame': 'base_footprint',
            'rolling_window': True,
            'width': int(round(float(nav['local_costmap_size_m']))),
            'height': int(round(float(nav['local_costmap_size_m']))),
            'resolution': 0.05,
            'robot_radius': float(nav['robot_radius_m']),
            'plugins': ['voxel_layer', 'inflation_layer'],
            'inflation_layer': {
                'plugin': 'nav2_costmap_2d::InflationLayer',
                'cost_scaling_factor': 8.0,
                'inflation_radius': float(nav['inflation_radius_m']),
            },
            'voxel_layer': {
                'plugin': 'nav2_costmap_2d::VoxelLayer',
                'enabled': True,
                'publish_voxel_map': False,
                'origin_z': 0.0,
                'z_resolution': 0.05,
                'z_voxels': 16,
                'max_obstacle_height': 2.0,
                'mark_threshold': 0,
                'observation_sources': 'scan',
                'scan': {
                    'topic': 'scan',
                    'max_obstacle_height': 2.0,
                    'clearing': True,
                    'marking': True,
                    'data_type': 'LaserScan',
                    'raytrace_max_range': 3.5,
                    'raytrace_min_range': 0.0,
                    'obstacle_max_range': 3.0,
                    'obstacle_min_range': 0.0,
                },
            },
            'always_send_full_costmap': True,
        }}},
        'global_costmap': {'global_costmap': {'ros__parameters': {
            'use_sim_time': True,
            'update_frequency': 2.0,
            'publish_frequency': 1.0,
            'global_frame': 'map',
            'robot_base_frame': 'base_footprint',
            'robot_radius': float(nav['robot_radius_m']),
            'resolution': 0.05,
            'track_unknown_space': True,
            'plugins': ['static_layer', 'inflation_layer'],
            'static_layer': {
                'plugin': 'nav2_costmap_2d::StaticLayer',
                'map_topic': '/map',
                'map_subscribe_transient_local': True,
            },
            'inflation_layer': {
                'plugin': 'nav2_costmap_2d::InflationLayer',
                'cost_scaling_factor': 8.0,
                'inflation_radius': float(nav['inflation_radius_m']),
            },
            'always_send_full_costmap': True,
        }}},
        'lifecycle_manager_navigation': {'ros__parameters': {
            'use_sim_time': True,
            'autostart': True,
            'bond_timeout': 8.0,
            'node_names': ['controller_server', 'planner_server', 'behavior_server', 'bt_navigator'],
        }},
    }


def _robot_nav2_params(config: dict, robot: dict) -> Path:
    ns = robot['namespace']
    spawn = robot['spawn']
    params = copy.deepcopy(_nav2_base_config(config))
    params['amcl']['ros__parameters'].update({
        'base_frame_id': f'{ns}/base_footprint',
        'odom_frame_id': f'{ns}/odom',
        'initial_pose': {
            'x': float(spawn['x']),
            'y': float(spawn['y']),
            'z': 0.0,
            'yaw': float(spawn['yaw']),
        },
    })
    for key in ('behavior_server', 'bt_navigator'):
        params[key]['ros__parameters']['robot_base_frame'] = f'{ns}/base_footprint'
        if 'odom_topic' in params[key]['ros__parameters']:
            params[key]['ros__parameters']['odom_topic'] = 'odom'
    params['local_costmap']['local_costmap']['ros__parameters']['robot_base_frame'] = f'{ns}/base_footprint'
    params['local_costmap']['local_costmap']['ros__parameters']['global_frame'] = f'{ns}/odom'
    params['global_costmap']['global_costmap']['ros__parameters']['robot_base_frame'] = f'{ns}/base_footprint'
    fq_params = {}
    for node_name, node_params in params.items():
        if node_name in ('local_costmap', 'global_costmap'):
            fq_params[f'/{ns}/{node_name}/{node_name}'] = node_params[node_name]
        else:
            fq_params[f'/{ns}/{node_name}'] = node_params
    return _write_yaml(RUNTIME_DIR / 'params' / f'{ns}_nav2.yaml', fq_params)


def _prepare_runtime_assets(config: dict) -> tuple[Path, Path, dict[str, Path]]:
    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    for robot in config['robots'].values():
        _write_model_dir(robot['namespace'], robot['model'])
    world = _write_world(config)
    nav2_params = {
        key: _robot_nav2_params(config, robot)
        for key, robot in config['robots'].items()
    }
    return world, _nav_map_yaml(), nav2_params


def _bridge_args(namespaces: list[str]) -> list[str]:
    args = [
        '/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock',
        '/tf@tf2_msgs/msg/TFMessage[gz.msgs.Pose_V',
    ]
    for ns in namespaces:
        args.extend([
            f'/{ns}/cmd_vel@geometry_msgs/msg/TwistStamped]gz.msgs.Twist',
            f'/{ns}/odom@nav_msgs/msg/Odometry[gz.msgs.Odometry',
            f'/{ns}/joint_states@sensor_msgs/msg/JointState[gz.msgs.Model',
            f'/{ns}/scan@sensor_msgs/msg/LaserScan[gz.msgs.LaserScan',
            f'/{ns}/imu@sensor_msgs/msg/Imu[gz.msgs.IMU',
        ])
    return args


def _nav2_group(ns: str, params: Path, goal_topic: str, cancel_topic: str = '') -> GroupAction:
    extra_env = {'RCUTILS_LOGGING_BUFFERED_STREAM': '1'}
    return GroupAction([
        PushRosNamespace(ns),
        Node(
            package='nav2_amcl',
            executable='amcl',
            name='amcl',
            output='screen',
            parameters=[str(params)],
            additional_env=extra_env,
        ),
        Node(
            package='nav2_lifecycle_manager',
            executable='lifecycle_manager',
            name='lifecycle_manager_localization',
            output='screen',
            parameters=[str(params)],
            additional_env=extra_env,
        ),
        TimerAction(period=2.0, actions=[
            Node(
                package='nav2_controller',
                executable='controller_server',
                name='controller_server',
                output='screen',
                parameters=[str(params)],
                additional_env=extra_env,
            ),
            Node(
                package='nav2_planner',
                executable='planner_server',
                name='planner_server',
                output='screen',
                parameters=[str(params)],
                additional_env=extra_env,
            ),
            Node(
                package='nav2_behaviors',
                executable='behavior_server',
                name='behavior_server',
                output='screen',
                parameters=[str(params)],
                additional_env=extra_env,
            ),
            Node(
                package='nav2_bt_navigator',
                executable='bt_navigator',
                name='bt_navigator',
                output='screen',
                parameters=[str(params)],
                additional_env=extra_env,
            ),
        ]),
        TimerAction(period=3.0, actions=[
            Node(
                package='nav2_lifecycle_manager',
                executable='lifecycle_manager',
                name='lifecycle_manager_navigation',
                output='screen',
                parameters=[str(params)],
                additional_env=extra_env,
            ),
        ]),
        TimerAction(period=4.0, actions=[
            Node(
                package='fleet_bringup',
                executable='pose_to_nav2',
                name='goal_proxy',
                output='screen',
                parameters=[{
                    'use_sim_time': True,
                    'goal_pose_topic': goal_topic,
                    'cancel_topic': cancel_topic,
                    'navigate_action': f'/{ns}/navigate_to_pose',
                    'require_localization_ready': False,
                }],
                additional_env=extra_env,
            ),
        ]),
    ])


def _robot_state_publisher(ns: str, model: str) -> Node:
    return Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name=f'{ns}_robot_state_publisher',
        output='screen',
        parameters=[{
            'use_sim_time': True,
            'robot_description': _urdf_for(model),
            'frame_prefix': f'{ns}/',
        }],
        remappings=[('joint_states', f'/{ns}/joint_states')],
    )


def _exploration_command(ns: str) -> str:
    return (
        'ros2 run region_mapper region_nav2_explorer_node '
        '--ros-args '
        f'-r __ns:=/{ns} '
        '-p use_sim_time:=true '
        '-p map_topic:=/map '
        '-p region_map_topic:=/slam_region_graph/region_map '
        '-p scan_topic:=scan '
        '-p global_frame:=map '
        f'-p robot_frame:={ns}/base_footprint '
        f'-p navigate_action_name:=/{ns}/navigate_to_pose '
        '-p auto_start:=true '
        '-p coverage_front_only:=true '
        '-p coverage_fov_deg:=60.0 '
        '-p view_fov_deg:=80.0'
    )


def generate_launch_description():
    pkg_share = Path(get_package_share_directory('system_bringup'))
    config_path = pkg_share / 'config' / 'house_motion_test.yaml'
    config = _read_yaml(config_path)
    world_path, map_yaml, nav2_params = _prepare_runtime_assets(config)

    scenario = LaunchConfiguration('scenario')
    ros_domain_id = LaunchConfiguration('ros_domain_id')
    rviz = LaunchConfiguration('rviz')
    headless = LaunchConfiguration('headless')

    rviz_config = pkg_share / 'rviz' / 'system_view.rviz'
    robots = config['robots']
    namespaces = [robots[key]['namespace'] for key in ('leader', 'field_a', 'field_b')]

    def validate_scenario(context, *args, **kwargs):
        del args, kwargs
        selected = scenario.perform(context).strip().lower()
        if selected != 'house':
            raise ValueError('system_house_motion_test.launch.py currently supports only scenario:=house')
        return []

    gz_server = ExecuteProcess(
        cmd=['gz', 'sim', '-r', '-s', '-v', '2', str(world_path)],
        name='gz_sim_server',
        output='screen',
    )
    gz_client = ExecuteProcess(
        cmd=['gz', 'sim', '-g', '-v', '2'],
        name='gz_sim_client',
        output='screen',
        condition=UnlessCondition(headless),
    )

    map_server = Node(
        package='nav2_map_server',
        executable='map_server',
        name='map_server',
        output='screen',
        parameters=[{'use_sim_time': True, 'yaml_filename': str(map_yaml)}],
    )
    map_lifecycle = Node(
        package='nav2_lifecycle_manager',
        executable='lifecycle_manager',
        name='lifecycle_manager_map',
        output='screen',
        parameters=[{
            'use_sim_time': True,
            'autostart': True,
            'node_names': ['map_server'],
        }],
    )

    leader = robots['leader']
    field_a = robots['field_a']
    field_b = robots['field_b']
    field_runtime = config['field_runtime']
    localization = config['localization']
    failover = config['failover']
    leader_cfg = config['leader']
    yield_cfg = config['yield']

    actions = [
        DeclareLaunchArgument('scenario', default_value='house'),
        DeclareLaunchArgument('ros_domain_id', default_value=EnvironmentVariable('ROS_DOMAIN_ID', default_value='30')),
        DeclareLaunchArgument('rviz', default_value='true', choices=['true', 'false']),
        DeclareLaunchArgument('headless', default_value='false', choices=['true', 'false']),
        *dds_launch_environment(ros_domain_id),
        OpaqueFunction(function=validate_scenario),
        AppendEnvironmentVariable('GZ_SIM_RESOURCE_PATH', str(_tb3_share() / 'models')),
        AppendEnvironmentVariable('GZ_SIM_RESOURCE_PATH', str(_find_tb3_file('models', 'turtlebot3_house').parent)),
        AppendEnvironmentVariable('GZ_SIM_RESOURCE_PATH', str(RUNTIME_DIR / 'models')),
        LogInfo(msg=[
            'SYSTEM_HOUSE_MOTION_TEST | scenario=house motion_only=true ',
            'risk=false yolo=false camera_decision=false world=', str(world_path),
            ' map=', str(map_yaml),
        ]),
        gz_server,
        gz_client,
        TimerAction(period=1.0, actions=[
            Node(
                package='ros_gz_bridge',
                executable='parameter_bridge',
                name='house_motion_gz_bridge',
                output='screen',
                arguments=_bridge_args(namespaces),
            ),
        ]),
        TimerAction(period=1.2, actions=[
            _robot_state_publisher(leader['namespace'], leader['model']),
            _robot_state_publisher(field_a['namespace'], field_a['model']),
            _robot_state_publisher(field_b['namespace'], field_b['model']),
        ]),
        TimerAction(period=1.5, actions=[map_server, map_lifecycle]),
        TimerAction(period=3.0, actions=[
            _nav2_group(leader['namespace'], nav2_params['leader'], leader['goal_topic'], '/fleet/leader_nav_cancel'),
            _nav2_group(field_a['namespace'], nav2_params['field_a'], field_a['goal_topic']),
            _nav2_group(field_b['namespace'], nav2_params['field_b'], field_b['goal_topic']),
        ]),
        TimerAction(period=5.0, actions=[
            Node(
                package='fleet_bringup',
                executable='tf_pose_publisher',
                name='leader_pose_pub',
                output='screen',
                parameters=[{
                    'use_sim_time': True,
                    'output_topic': leader['pose_topic'],
                    'target_frame': 'map',
                    'source_frame': f"{leader['namespace']}/base_footprint",
                    'publish_rate_hz': 10.0,
                    'log_every_n': 100,
                }],
            ),
            Node(
                package='fleet_bringup',
                executable='tf_pose_publisher',
                name='field_a_pose_pub',
                output='screen',
                parameters=[{
                    'use_sim_time': True,
                    'output_topic': field_a['pose_topic'],
                    'target_frame': 'map',
                    'source_frame': f"{field_a['namespace']}/base_footprint",
                    'publish_rate_hz': 10.0,
                    'log_every_n': 100,
                }],
            ),
            Node(
                package='fleet_bringup',
                executable='tf_pose_publisher',
                name='field_b_pose_pub',
                output='screen',
                parameters=[{
                    'use_sim_time': True,
                    'output_topic': field_b['pose_topic'],
                    'target_frame': 'map',
                    'source_frame': f"{field_b['namespace']}/base_footprint",
                    'publish_rate_hz': 10.0,
                    'log_every_n': 100,
                }],
            ),
        ]),
        TimerAction(period=6.0, actions=[
            Node(
                package='fleet_bringup',
                executable='fleet_path_coordinator',
                name='fleet_path_coordinator',
                output='screen',
                parameters=[{
                    'use_sim_time': True,
                    'leader_pose_topic': leader['pose_topic'],
                    'follower_pose_topic': field_b['pose_topic'],
                    'member_pose_topic': field_a['pose_topic'],
                    'map_topic': '/map',
                    'leader_user_goal_topic': '/goal_pose',
                    'leader_coord_goal_topic': leader['goal_topic'],
                    'follower_coord_goal_topic': field_b['goal_topic'],
                    'member_coord_goal_topic': field_a['goal_topic'],
                    'require_follower_pose': True,
                    'require_localization_ready': False,
                    'minimum_robot_separation_m': float(yield_cfg['minimum_robot_separation_m']),
                    'evasion_offset_m': float(yield_cfg['evasion_offset_m']),
                    'evasion_offset_max_m': float(yield_cfg['evasion_offset_max_m']),
                    'motion_trigger_distance_m': float(yield_cfg['motion_trigger_distance_m']),
                    'cooldown_sec': float(yield_cfg['cooldown_sec']),
                }],
                respawn=True,
                respawn_delay=3.0,
            ),
            Node(
                package='system_bringup',
                executable='scout_failover_coordinator',
                name='scout_failover_coordinator',
                output='screen',
                parameters=[{
                    'use_sim_time': True,
                    'enable_scout_failover': True,
                    'leader_robot_name': 'leader',
                    'active_scout_robot_name': field_a['robot_name'],
                    'follower_robot_name': field_b['robot_name'],
                    'scout_liveness_topic': '/scout/signal',
                    'scout_pose_topic': field_a['pose_topic'],
                    'leader_pose_topic': leader['pose_topic'],
                    'follower_pose_topic': field_b['pose_topic'],
                    'leader_goal_topic': leader['goal_topic'],
                    'leader_cancel_topic': '/fleet/leader_nav_cancel',
                    'role_command_topic': '/fleet/field_robot_role_cmd',
                    'field_robot_status_topic': '/fleet/field_robot_status',
                    'scout_liveness_timeout_sec': float(failover['scout_liveness_timeout_sec']),
                    'scout_failure_confirm_sec': float(failover['scout_failure_confirm_sec']),
                    'scout_pose_timeout_sec': float(failover['scout_pose_timeout_sec']),
                    'leader_recovery_standoff_m': float(failover['leader_recovery_standoff_m']),
                    'leader_failure_arrival_tolerance_m': float(failover['leader_failure_arrival_tolerance_m']),
                    'follower_recovery_standoff_m': float(failover['follower_recovery_standoff_m']),
                    'scout_takeover_arrival_tolerance_m': float(field_runtime['recovery_arrival_tolerance_m']),
                }],
                respawn=True,
                respawn_delay=3.0,
            ),
            Node(
                package='system_bringup',
                executable='leader_shadow_follow',
                name='leader_shadow_follow',
                output='screen',
                parameters=[{
                    'use_sim_time': True,
                    'enable_leader_shadow_follow': True,
                    'leader_pose_topic': leader['pose_topic'],
                    'active_scout_pose_topic': field_a['pose_topic'],
                    'follower_scout_pose_topic': field_b['pose_topic'],
                    'leader_goal_topic': leader['goal_topic'],
                    'leader_cancel_topic': '/fleet/leader_nav_cancel',
                    'controller_set_parameters_service': f"/{leader['namespace']}/controller_server/set_parameters",
                    'map_topic': '/map',
                    'active_scout_robot_name': field_a['robot_name'],
                    'follower_robot_name': field_b['robot_name'],
                    'scout_pose_timeout_sec': float(failover['scout_pose_timeout_sec']),
                    'startup_grace_sec': 6.0,
                    'leader_shadow_follow_distance_m': float(leader_cfg['shadow_distance_m']),
                    'leader_shadow_stop_distance_m': float(leader_cfg['shadow_stop_distance_m']),
                    'leader_shadow_resume_distance_m': float(leader_cfg['shadow_resume_distance_m']),
                    'leader_shadow_far_distance_m': float(leader_cfg['shadow_far_distance_m']),
                    'leader_shadow_max_linear_vel': float(leader_cfg['max_linear_vel']),
                    'leader_shadow_max_angular_vel': float(leader_cfg['max_angular_vel']),
                    'enable_leader_continuous_scan': bool(leader_cfg['scan_enabled']),
                    'leader_scan_topic': f"/{leader['namespace']}/scan",
                    'leader_scan_fov_deg': float(leader_cfg['scan_fov_deg']),
                    'leader_scan_update_rate_hz': float(leader_cfg['scan_update_rate_hz']),
                    'leader_scan_timeout_sec': float(leader_cfg['scan_timeout_sec']),
                }],
                respawn=True,
                respawn_delay=3.0,
            ),
        ]),
        TimerAction(period=7.0, actions=[
            Node(
                package='system_bringup',
                executable='unified_field_robot',
                name='field_a_unified_runtime',
                output='screen',
                parameters=[{
                    'use_sim_time': True,
                    'robot_name': field_a['robot_name'],
                    'initial_role': field_a['initial_role'],
                    'leader_pose_topic': leader['pose_topic'],
                    'self_pose_topic': field_a['pose_topic'],
                    'navigate_action': f"/{field_a['namespace']}/navigate_to_pose",
                    'cmd_vel_topic': f"/{field_a['namespace']}/cmd_vel",
                    'amcl_pose_topic': f"/{field_a['namespace']}/amcl_pose",
                    'odom_topic': f"/{field_a['namespace']}/odom",
                    'enable_follow_mode': True,
                    'enable_scout_mode': True,
                    'enable_recovery_mode': True,
                    'enable_localization_spin': bool(config['features']['localization_spin']),
                    'enable_exploration': True,
                    'exploration_command': _exploration_command(field_a['namespace']),
                    'follow_distance_m': float(field_runtime['follow_distance_m']),
                    'follow_goal_period_sec': float(field_runtime['follow_goal_period_sec']),
                    'follow_goal_update_distance_m': float(field_runtime['follow_goal_update_distance_m']),
                    'recovery_arrival_tolerance_m': float(field_runtime['recovery_arrival_tolerance_m']),
                    'spin_speed_rad_s': float(localization['spin_speed_rad_s']),
                    'spin_target_angle_rad': float(localization['spin_target_angle_rad']),
                    'spin_timeout_sec': float(localization['spin_timeout_sec']),
                    'max_spin_retries': int(localization['max_spin_retries']),
                    'max_xy_covariance': float(localization['max_xy_covariance']),
                    'max_yaw_covariance': float(localization['max_yaw_covariance']),
                }],
            ),
            Node(
                package='system_bringup',
                executable='unified_field_robot',
                name='field_b_unified_runtime',
                output='screen',
                parameters=[{
                    'use_sim_time': True,
                    'robot_name': field_b['robot_name'],
                    'initial_role': field_b['initial_role'],
                    'leader_pose_topic': leader['pose_topic'],
                    'self_pose_topic': field_b['pose_topic'],
                    'navigate_action': f"/{field_b['namespace']}/navigate_to_pose",
                    'cmd_vel_topic': f"/{field_b['namespace']}/cmd_vel",
                    'amcl_pose_topic': f"/{field_b['namespace']}/amcl_pose",
                    'odom_topic': f"/{field_b['namespace']}/odom",
                    'enable_follow_mode': True,
                    'enable_scout_mode': True,
                    'enable_recovery_mode': True,
                    'enable_localization_spin': bool(config['features']['localization_spin']),
                    'enable_exploration': True,
                    'exploration_command': _exploration_command(field_b['namespace']),
                    'follow_distance_m': float(field_runtime['follow_distance_m']),
                    'follow_goal_period_sec': float(field_runtime['follow_goal_period_sec']),
                    'follow_goal_update_distance_m': float(field_runtime['follow_goal_update_distance_m']),
                    'recovery_arrival_tolerance_m': float(field_runtime['recovery_arrival_tolerance_m']),
                    'spin_speed_rad_s': float(localization['spin_speed_rad_s']),
                    'spin_target_angle_rad': float(localization['spin_target_angle_rad']),
                    'spin_timeout_sec': float(localization['spin_timeout_sec']),
                    'max_spin_retries': int(localization['max_spin_retries']),
                    'max_xy_covariance': float(localization['max_xy_covariance']),
                    'max_yaw_covariance': float(localization['max_yaw_covariance']),
                }],
            ),
        ]),
        TimerAction(period=8.0, actions=[
            Node(
                package='rviz2',
                executable='rviz2',
                name='rviz2',
                output='screen',
                arguments=['-d', str(rviz_config)],
                parameters=[{'use_sim_time': True}],
                condition=IfCondition(rviz),
            ),
        ]),
    ]

    return LaunchDescription(actions)
