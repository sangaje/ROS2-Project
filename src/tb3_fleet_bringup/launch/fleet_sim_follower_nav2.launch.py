#!/usr/bin/env python3
"""
Follower stack for simulation.

Bridges:
  leader → follower : /map, /leader_pose, /burger_goal_pose, /fleet/follow_command
                      /burger/scan  → /scan_bridge   (relay strips frame to /scan)
                      /burger/odom  → /odom_bridge   (relay strips frame to /odom)
                      /burger/joint_states → /joint_states
                      /burger/tf    → /burger/tf
  follower → leader : /burger_pose, /plan→/burger_plan, /fleet/follow_enabled
                      /cmd_vel → /burger/cmd_vel  (Nav2 output → Gz follower robot)

The follower domain also runs robot_state_publisher, TF/sensor relays, AMCL, Nav2,
and the follower controller script.
"""

import os
import sys
import tempfile
from pathlib import Path

import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    LogInfo,
    OpaqueFunction,
    SetEnvironmentVariable,
    TimerAction,
    UnsetEnvironmentVariable,
)
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _arg(name: str, default: str) -> str:
    """Read a name:=value launch argument from sys.argv at parse time."""
    for a in sys.argv:
        if a.startswith(f'{name}:='):
            return a.split(':=', 1)[1]
    return default


def generate_launch_description():
    # Parse domain IDs early so os.environ is set before any nodes are created.
    # This ensures ExecuteProcess children inherit the correct ROS_DOMAIN_ID even
    # when SetEnvironmentVariable doesn't propagate to them reliably.
    _did  = _arg('domain_id',        '24')   # follower domain
    _ldid = _arg('leader_domain_id', '25')   # leader domain

    os.environ['ROS_DOMAIN_ID'] = _did
    os.environ['RMW_IMPLEMENTATION'] = 'rmw_fastrtps_cpp'
    follower_env = {'ROS_DOMAIN_ID': _did, 'RMW_IMPLEMENTATION': 'rmw_fastrtps_cpp'}
    os.environ.pop('ROS_DISCOVERY_SERVER', None)
    os.environ.pop('ROS_LOCALHOST_ONLY', None)
    os.environ.pop('FASTRTPS_DEFAULT_PROFILES_FILE', None)
    os.environ.pop('FASTDDS_DEFAULT_PROFILES_FILE', None)

    bringup_share = get_package_share_directory('tb3_fleet_bringup')
    tb3_gz_share  = get_package_share_directory('turtlebot3_gazebo')

    domain_id        = LaunchConfiguration('domain_id')
    leader_domain_id = LaunchConfiguration('leader_domain_id')
    follow_distance  = LaunchConfiguration('follow_distance')
    start_following  = LaunchConfiguration('start_following')
    follower_initial_x = LaunchConfiguration('follower_initial_x')
    follower_initial_y = LaunchConfiguration('follower_initial_y')
    follower_initial_yaw = LaunchConfiguration('follower_initial_yaw')

    nav2_params = os.path.join(bringup_share, 'config', 'domain24_burger_nav2_amcl_sim.yaml')

    follower_script   = os.path.join(bringup_share, 'scripts',
                                     'domain_bridge_nav2_follower_direct_v40.py')
    tf_pose_script    = os.path.join(bringup_share, 'scripts',
                                     'tf_pose_publisher_direct_v44.py')
    goal_proxy_script = os.path.join(bringup_share, 'scripts',
                                     'pose_to_nav2_action_direct_v41.py')
    tf_relay_script   = os.path.join(bringup_share, 'scripts', 'sim_burger_tf_relay.py')
    scan_relay_script = os.path.join(bringup_share, 'scripts', 'sim_burger_scan_relay.py')
    map_relay_script  = os.path.join(bringup_share, 'scripts', 'sim_map_relay.py')

    # Follower robot_state_publisher uses standard frames without the burger/ prefix.
    follower_urdf = os.path.join(tb3_gz_share, 'urdf', 'turtlebot3_burger.urdf')
    with open(follower_urdf, 'r') as f:
        robot_desc = f.read()

    follower_rsp = Node(
        package='robot_state_publisher', executable='robot_state_publisher',
        name='robot_state_publisher', output='screen',
        parameters=[{'use_sim_time': True, 'robot_description': robot_desc}],
        additional_env=follower_env,
    )

    # TF relay: /burger/tf (burger/odom -> burger/base_footprint) -> /tf (odom -> base_footprint)
    tf_relay = ExecuteProcess(
        cmd=['python3', tf_relay_script, '--ros-args',
             '-r', '__node:=sim_burger_tf_relay',
             '-p', 'use_sim_time:=true'],
        output='screen', name='sim_burger_tf_relay',
        additional_env=follower_env,
    )
    # Sensor relay: bridge input frames burger/* -> follower standard frames for Nav2.
    scan_relay = ExecuteProcess(
        cmd=['python3', scan_relay_script, '--ros-args',
             '-r', '__node:=sim_burger_scan_relay',
             '-p', 'use_sim_time:=true',
             '-p', 'scan_input_topic:=/scan_bridge',
             '-p', 'scan_output_topic:=/scan',
             '-p', 'burger_scan_output_topic:=/burger_scan_relay',
             '-p', 'odom_input_topic:=/odom_bridge',
             '-p', 'odom_output_topic:=/odom'],
        output='screen', name='sim_burger_scan_relay',
        additional_env=follower_env,
    )
    # Map relay: /map_bridge (volatile, from domain_bridge) -> /map (transient_local for AMCL)
    map_relay = ExecuteProcess(
        cmd=['python3', map_relay_script, '--ros-args',
             '-r', '__node:=sim_follower_map_relay',
             '-p', 'use_sim_time:=true'],
        output='screen', name='sim_follower_map_relay',
        additional_env=follower_env,
    )

    # -- Domain bridges ---------------------------------------------------------
    def make_domain_bridges(context, *args, **kwargs):
        ld = leader_domain_id.perform(context)   # e.g. '25'
        bd = domain_id.perform(context)          # e.g. '24'

        out = Path(tempfile.gettempdir()) / 'tb3_sim_domain_bridge'
        out.mkdir(parents=True, exist_ok=True)

        leader_to_follower = out / f'sim_{ld}_to_{bd}.yaml'
        follower_to_leader = out / f'sim_{bd}_to_{ld}.yaml'

        leader_to_follower.write_text(f"""\
name: sim_{ld}_to_{bd}
from_domain: {ld}
to_domain: {bd}

topics:
  /clock:
    type: rosgraph_msgs/msg/Clock
    qos:
      reliability: best_effort
      durability: volatile
      history: keep_last
      depth: 10
  /map:
    type: nav_msgs/msg/OccupancyGrid
    remap: /map_bridge
    qos:
      reliability: reliable
      durability: volatile
      history: keep_last
      depth: 5
  /map_metadata:
    type: nav_msgs/msg/MapMetaData
    qos:
      reliability: reliable
      durability: volatile
      history: keep_last
      depth: 1
  /leader_pose:
    type: geometry_msgs/msg/PoseStamped
    qos:
      reliability: reliable
      durability: volatile
      history: keep_last
      depth: 10
  /initialpose:
    type: geometry_msgs/msg/PoseWithCovarianceStamped
    qos:
      reliability: reliable
      durability: volatile
      history: keep_last
      depth: 1
  /burger_goal_pose:
    type: geometry_msgs/msg/PoseStamped
    qos:
      reliability: reliable
      durability: volatile
      history: keep_last
      depth: 10
  /fleet/follow_command:
    type: std_msgs/msg/String
    qos:
      reliability: reliable
      durability: volatile
      history: keep_last
      depth: 10
  /plan:
    type: nav_msgs/msg/Path
    remap: /waffle_plan
    qos:
      reliability: reliable
      durability: volatile
      history: keep_last
      depth: 10
  /burger/scan:
    type: sensor_msgs/msg/LaserScan
    remap: /scan_bridge
    qos:
      reliability: best_effort
      durability: volatile
      history: keep_last
      depth: 10
  /burger/odom:
    type: nav_msgs/msg/Odometry
    remap: /odom_bridge
    qos:
      reliability: reliable
      durability: volatile
      history: keep_last
      depth: 10
  /burger/joint_states:
    type: sensor_msgs/msg/JointState
    remap: /joint_states
    qos:
      reliability: reliable
      durability: volatile
      history: keep_last
      depth: 10
  /burger/tf:
    type: tf2_msgs/msg/TFMessage
    qos:
      reliability: reliable
      durability: volatile
      history: keep_last
      depth: 100
""", encoding='utf-8')

        follower_to_leader.write_text(f"""\
name: sim_{bd}_to_{ld}
from_domain: {bd}
to_domain: {ld}

topics:
  /plan:
    type: nav_msgs/msg/Path
    remap: /burger_plan
    qos:
      reliability: reliable
      durability: volatile
      history: keep_last
      depth: 10
  /burger_pose:
    type: geometry_msgs/msg/PoseStamped
    qos:
      reliability: reliable
      durability: volatile
      history: keep_last
      depth: 10
  /fleet/follow_enabled:
    type: std_msgs/msg/Bool
    qos:
      reliability: reliable
      durability: transient_local
      history: keep_last
      depth: 1
  /cmd_vel:
    type: geometry_msgs/msg/TwistStamped
    remap: /burger/cmd_vel
    qos:
      reliability: reliable
      durability: volatile
      history: keep_last
      depth: 10
  /burger_scan_relay:
    type: sensor_msgs/msg/LaserScan
    remap: /burger_scan
    qos:
      reliability: best_effort
      durability: volatile
      history: keep_last
      depth: 10
""", encoding='utf-8')

        return [
            ExecuteProcess(
                cmd=['ros2', 'run', 'domain_bridge', 'domain_bridge', str(leader_to_follower)],
                output='screen', name='sim_leader_to_follower_bridge'),
            ExecuteProcess(
                cmd=['ros2', 'run', 'domain_bridge', 'domain_bridge', str(follower_to_leader)],
                output='screen', name='sim_follower_to_leader_bridge'),
        ]

    def make_amcl(context, *args, **kwargs):
        ix = float(follower_initial_x.perform(context))
        iy = float(follower_initial_y.perform(context))
        iyaw = float(follower_initial_yaw.perform(context))
        pose_overrides = {'amcl': {'ros__parameters': {
            'set_initial_pose': True,
            'initial_pose': {'x': ix, 'y': iy, 'z': 0.0, 'yaw': iyaw},
        }}}
        pose_yaml = Path(tempfile.gettempdir()) / 'sim_burger_amcl_initial_pose.yaml'
        pose_yaml.write_text(yaml.dump(pose_overrides), encoding='utf-8')
        return [Node(
            package='nav2_amcl', executable='amcl', name='amcl', output='screen',
            parameters=[nav2_params, str(pose_yaml)],
            additional_env=follower_env,
            respawn=True, respawn_delay=3.0,
        )]

    lifecycle_loc = Node(
        package='nav2_lifecycle_manager', executable='lifecycle_manager',
        name='lifecycle_manager_localization', output='screen',
        parameters=[nav2_params],
        additional_env=follower_env,
    )
    controller_server = Node(package='nav2_controller', executable='controller_server',
                             output='screen', parameters=[nav2_params],
                             additional_env=follower_env)
    planner_server    = Node(package='nav2_planner', executable='planner_server',
                             output='screen', parameters=[nav2_params],
                             additional_env=follower_env)
    behavior_server   = Node(package='nav2_behaviors', executable='behavior_server',
                             output='screen', parameters=[nav2_params],
                             additional_env=follower_env)
    bt_navigator      = Node(package='nav2_bt_navigator', executable='bt_navigator',
                             output='screen', parameters=[nav2_params],
                             additional_env=follower_env)
    lifecycle_nav     = Node(package='nav2_lifecycle_manager', executable='lifecycle_manager',
                             name='lifecycle_manager_navigation', output='screen',
                             parameters=[nav2_params],
                             additional_env=follower_env)
    burger_pose = ExecuteProcess(
        cmd=['python3', tf_pose_script, '--ros-args',
             '-r', '__node:=sim_burger_pose_publisher',
             '-p', 'use_sim_time:=true',
             '-p', 'target_frame:=map', '-p', 'source_frame:=base_footprint',
             '-p', 'output_topic:=/burger_pose',
             '-p', 'publish_rate_hz:=10.0', '-p', 'log_every_n:=100'],
        output='screen', name='sim_burger_pose_publisher',
        additional_env=follower_env,
    )
    burger_named_goal = ExecuteProcess(
        cmd=['python3', goal_proxy_script, '--ros-args',
             '-r', '__node:=sim_burger_goal_to_nav2',
             '-p', 'use_sim_time:=true',
             '-p', 'goal_pose_topic:=/burger_goal_pose',
             '-p', 'navigate_action:=/navigate_to_pose',
             '-p', 'default_frame_id:=map', '-p', 'cancel_previous_goal:=true'],
        output='screen', name='sim_burger_goal_to_nav2',
        additional_env=follower_env,
    )
    follower = ExecuteProcess(
        cmd=['python3', follower_script, '--ros-args',
             '-r', '__node:=sim_domain_bridge_nav2_follower',
             '-p', 'use_sim_time:=true',
             '-p', 'leader_pose_topic:=/leader_pose',
             '-p', 'leader_path_topic:=/waffle_plan',
             '-p', 'follower_pose_topic:=/burger_pose',
             '-p', 'map_topic:=/map',
             '-p', 'navigate_action:=/navigate_to_pose',
             '-p', ['follow_distance:=', follow_distance],
             '-p', 'goal_period_sec:=1.0', '-p', 'goal_update_distance:=0.20',
             '-p', 'cancel_previous_goal:=false',
             '-p', 'follow_command_topic:=/fleet/follow_command',
             '-p', 'follow_status_topic:=/fleet/follow_enabled',
             '-p', ['start_following:=', start_following],
             '-p', 'enable_path_yield:=false',
             '-p', 'path_block_distance:=0.55'],
        output='screen', name='sim_burger_nav2_follower',
        additional_env=follower_env,
    )

    return LaunchDescription([
        DeclareLaunchArgument('domain_id',        default_value=_did,
                              description='ROS domain ID for follower (default 24)'),
        DeclareLaunchArgument('leader_domain_id', default_value=_ldid,
                              description='ROS domain ID for leader  (default 25)'),
        DeclareLaunchArgument('follow_distance',  default_value='1.05'),
        DeclareLaunchArgument('start_following',  default_value='false'),
        DeclareLaunchArgument('follower_initial_x', default_value='-1.0'),
        DeclareLaunchArgument('follower_initial_y', default_value='0.0'),
        DeclareLaunchArgument('follower_initial_yaw', default_value='0.0'),
        UnsetEnvironmentVariable('ROS_DISCOVERY_SERVER'),
        UnsetEnvironmentVariable('ROS_LOCALHOST_ONLY'),
        UnsetEnvironmentVariable('FASTRTPS_DEFAULT_PROFILES_FILE'),
        UnsetEnvironmentVariable('FASTDDS_DEFAULT_PROFILES_FILE'),
        SetEnvironmentVariable('ROS_DOMAIN_ID',       domain_id),
        SetEnvironmentVariable('RMW_IMPLEMENTATION',  'rmw_fastrtps_cpp'),
        SetEnvironmentVariable('TURTLEBOT3_MODEL',    'burger'),
        LogInfo(msg=['SIM_FOLLOWER | domain=', domain_id, ' leader=', leader_domain_id,
                     ' | start_following=', start_following]),
        follower_rsp,
        TimerAction(period=0.5, actions=[OpaqueFunction(function=make_domain_bridges)]),
        TimerAction(period=1.0, actions=[tf_relay, scan_relay, map_relay]),
        TimerAction(period=5.0, actions=[
            LogInfo(msg='SIM_FOLLOWER: launching AMCL...'),
            OpaqueFunction(function=make_amcl),
        ]),
        TimerAction(period=5.5, actions=[lifecycle_loc]),
        TimerAction(period=7.0, actions=[controller_server, planner_server,
                                          behavior_server, bt_navigator]),
        TimerAction(period=10.0, actions=[lifecycle_nav]),
        TimerAction(period=12.0, actions=[burger_pose]),
        TimerAction(period=13.0, actions=[burger_named_goal]),
        TimerAction(period=14.0, actions=[follower]),
    ])
