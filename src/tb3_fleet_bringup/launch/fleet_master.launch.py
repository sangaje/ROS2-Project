#!/usr/bin/env python3
"""Fleet Master launch — Domain 25.

Runs on the same PC as Waffle (Domain 25).
- domain_bridge 25->24: /map, /map_metadata, /leader_pose
- domain_bridge 24->25: /burger_pose
- fleet_goal_dispatcher_direct_v72.py  (publishes /waffle_goal_pose, /burger_goal_pose)
- fleet_debug_marker.py
- RViz (optional, fleet_fleet_debug_v71.rviz)
"""

import os
import tempfile
from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    TimerAction,
    SetEnvironmentVariable,
    LogInfo,
    OpaqueFunction,
)
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PythonExpression


def _require_files(paths):
    missing = [p for p in paths if not os.path.exists(p)]
    if missing:
        raise RuntimeError('Missing required fleet master files: ' + ', '.join(missing))


def generate_launch_description():
    bringup_share = get_package_share_directory('tb3_fleet_bringup')

    dispatcher_script = os.path.join(bringup_share, 'scripts', 'fleet_goal_dispatcher_direct_v72.py')
    marker_script = os.path.join(bringup_share, 'scripts', 'fleet_debug_marker.py')
    rviz_clean = os.path.join(bringup_share, 'scripts', 'run_rviz2_clean.bash')
    default_rviz = os.path.join(bringup_share, 'rviz', 'fleet_fleet_debug_v71.rviz')

    _require_files([dispatcher_script])

    waffle_domain_id = LaunchConfiguration('waffle_domain_id')
    burger_domain_id = LaunchConfiguration('burger_domain_id')
    show_rviz = LaunchConfiguration('show_rviz')
    sim = LaunchConfiguration('sim')
    rviz_config = LaunchConfiguration('rviz_config')

    use_sim_time_str = PythonExpression(["'true' if '", sim, "' == 'true' else 'false'"])
    show_rviz_cond = IfCondition(PythonExpression(["'", show_rviz, "' == 'true'"]))

    # ---- Domain bridges (OpaqueFunction) ----------------------------------
    def _write_bridge_configs(context, *args, **kwargs):
        waffle_domain = waffle_domain_id.perform(context)
        burger_domain = burger_domain_id.perform(context)
        out_dir = Path(tempfile.gettempdir()) / 'tb3_fleet_master_bridge'
        out_dir.mkdir(parents=True, exist_ok=True)

        path_25_to_26 = out_dir / f'master_{waffle_domain}_to_{burger_domain}.yaml'
        path_26_to_25 = out_dir / f'master_{burger_domain}_to_{waffle_domain}.yaml'

        yaml_25_to_26 = f"""name: fleet_master_{waffle_domain}_to_{burger_domain}
from_domain: {waffle_domain}
to_domain: {burger_domain}

topics:
  /map:
    type: nav_msgs/msg/OccupancyGrid
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
      depth: 5
  /leader_pose:
    type: geometry_msgs/msg/PoseStamped
    qos:
      reliability: reliable
      durability: volatile
      history: keep_last
      depth: 10
"""

        yaml_26_to_25 = f"""name: fleet_master_{burger_domain}_to_{waffle_domain}
from_domain: {burger_domain}
to_domain: {waffle_domain}

topics:
  /burger_pose:
    type: geometry_msgs/msg/PoseStamped
    qos:
      reliability: reliable
      durability: volatile
      history: keep_last
      depth: 10
"""

        path_25_to_26.write_text(yaml_25_to_26)
        path_26_to_25.write_text(yaml_26_to_25)

        return [
            ExecuteProcess(
                cmd=['ros2', 'run', 'domain_bridge', 'domain_bridge', str(path_25_to_26)],
                output='screen',
                name=f'fleet_bridge_{waffle_domain}_to_{burger_domain}',
            ),
            ExecuteProcess(
                cmd=['ros2', 'run', 'domain_bridge', 'domain_bridge', str(path_26_to_25)],
                output='screen',
                name=f'fleet_bridge_{burger_domain}_to_{waffle_domain}',
            ),
        ]

    domain_bridges = OpaqueFunction(function=_write_bridge_configs)

    # ---- Fleet goal dispatcher --------------------------------------------
    dispatcher = ExecuteProcess(
        cmd=[
            'python3', dispatcher_script, '--ros-args',
            '-r', '__node:=fleet_goal_dispatcher',
            '-p', ['use_sim_time:=', use_sim_time_str],
            '-p', 'input_goal_topic:=/fleet_goal_pose',
            '-p', 'input_alias_topics:=/goal_pose,/move_base_simple/goal',
            '-p', 'waffle_goal_topic:=/waffle_goal_pose',
            '-p', 'burger_goal_topic:=/burger_goal_pose',
            '-p', 'waffle_pose_topic:=/leader_pose',
            '-p', 'burger_pose_topic:=/burger_pose',
            '-p', 'map_topic:=/map',
            '-p', 'frame_id:=map',
            '-p', 'formation_separation_m:=1.20',
            '-p', 'min_pair_distance_m:=0.85',
            '-p', 'search_rings:=4',
            '-p', 'search_angles:=20',
            '-p', 'clearance_check_radius_m:=0.55',
            '-p', 'occupied_threshold:=45',
            '-p', 'republish_count:=3',
            '-p', 'republish_period_sec:=1.0',
            '-p', 'converge_mode:=true',
        ],
        output='screen', name='fleet_goal_dispatcher',
    )

    # ---- Debug marker node ------------------------------------------------
    def _make_marker_node(context, *args, **kwargs):
        if not os.path.exists(marker_script):
            return []
        use_st = use_sim_time_str.perform(context)
        return [
            ExecuteProcess(
                cmd=[
                    'python3', marker_script, '--ros-args',
                    '-r', '__node:=fleet_debug_marker',
                    '-p', f'use_sim_time:={use_st}',
                    '-p', 'waffle_pose_topic:=/leader_pose',
                    '-p', 'burger_pose_topic:=/burger_pose',
                    '-p', 'marker_topic:=/fleet_debug_markers',
                    '-p', 'frame_id:=map',
                ],
                output='screen', name='fleet_debug_marker',
            )
        ]

    # ---- RViz (optional) -------------------------------------------------
    def _make_rviz(context, *args, **kwargs):
        if show_rviz.perform(context) != 'true':
            return []
        rviz_cfg = rviz_config.perform(context)
        if not rviz_cfg or not os.path.exists(rviz_cfg):
            rviz_cfg = default_rviz
        use_st = use_sim_time_str.perform(context)
        if os.path.exists(rviz_clean):
            cmd = [rviz_clean, '-d', rviz_cfg, '--ros-args',
                   '-r', '__node:=rviz2_fleet_master',
                   '-p', f'use_sim_time:={use_st}']
        else:
            cmd = ['rviz2', '-d', rviz_cfg, '--ros-args',
                   '-r', '__node:=rviz2_fleet_master',
                   '-p', f'use_sim_time:={use_st}']
        return [
            ExecuteProcess(cmd=cmd, output='screen', name='rviz2_fleet_master')
        ]

    return LaunchDescription([
        DeclareLaunchArgument('waffle_domain_id', default_value='25'),
        DeclareLaunchArgument('burger_domain_id', default_value='24'),
        DeclareLaunchArgument('show_rviz', default_value='true'),
        DeclareLaunchArgument('sim', default_value='true'),
        DeclareLaunchArgument('rviz_config', default_value=default_rviz),
        SetEnvironmentVariable('ROS_DOMAIN_ID', waffle_domain_id),
        SetEnvironmentVariable('RMW_IMPLEMENTATION', 'rmw_fastrtps_cpp'),
        SetEnvironmentVariable('FASTDDS_BUILTIN_TRANSPORTS', 'UDPv4'),
        SetEnvironmentVariable('ROS_AUTOMATIC_DISCOVERY_RANGE', 'LOCALHOST'),
        LogInfo(msg='FLEET_MASTER_LAUNCH | domain bridge + dispatcher + RViz on Domain 25'),
        LogInfo(msg=['waffle_domain=', waffle_domain_id, ' burger_domain=', burger_domain_id]),
        TimerAction(period=0.5, actions=[domain_bridges]),
        TimerAction(period=2.0, actions=[dispatcher]),
        TimerAction(period=2.0, actions=[OpaqueFunction(function=_make_marker_node)]),
        TimerAction(period=3.0, actions=[OpaqueFunction(function=_make_rviz)]),
    ])
