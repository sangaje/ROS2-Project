"""Launch central risk-map processing and configurable domain_bridge links."""

import os
import tempfile
from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    OpaqueFunction,
    TimerAction,
)
from launch.conditions import IfCondition
from launch.substitutions import EnvironmentVariable, LaunchConfiguration
from launch_ros.actions import Node
from tb3_fleet_bringup.launch_utils import dds_launch_environment


def _topic_block(topic: str, msg_type: str, reliability='reliable', durability='volatile', depth=10) -> str:
    return f"""  {topic}:
    type: {msg_type}
    qos:
      reliability: {reliability}
      durability: {durability}
      history: keep_last
      depth: {depth}
"""


def _write_runtime_yaml(prefix: str, text: str, output_dir: Path) -> Path:
    with tempfile.NamedTemporaryFile(
        mode='w',
        suffix='.yaml',
        prefix=prefix,
        dir=output_dir,
        delete=False,
        encoding='utf-8',
    ) as handle:
        handle.write(text)
        return Path(handle.name)


def generate_launch_description():
    pkg_share = get_package_share_directory('tb3_bayesian_risk_map')
    default_config = os.path.join(pkg_share, 'config', 'bayesian_risk_map.yaml')

    central_domain_id = LaunchConfiguration('central_domain_id')
    source_domain_id = LaunchConfiguration('source_domain_id')
    risk_sink_domain_ids = LaunchConfiguration('risk_sink_domain_ids')
    bridge_rviz_topics = LaunchConfiguration('bridge_rviz_topics')

    def _write_bridge_configs(context, *args, **kwargs):
        central_domain = central_domain_id.perform(context)
        source_domain = source_domain_id.perform(context).strip()
        if not source_domain:
            raise ValueError(
                'source_domain_id is required when start_domain_bridges:=true. '
                'Pass the launch option source_domain_id:=<robot_domain>.'
            )
        include_rviz_topics = bridge_rviz_topics.perform(context).strip().lower() in (
            '1', 'true', 'yes', 'on'
        )
        sink_domains = [
            item.strip()
            for item in risk_sink_domain_ids.perform(context).split(',')
            if item.strip()
        ]

        out_dir = Path(tempfile.gettempdir()) / 'tb3_central_risk_domain_bridge'
        out_dir.mkdir(parents=True, exist_ok=True)

        source_yaml = f"""name: risk_source_{source_domain}_to_central_{central_domain}
from_domain: {source_domain}
to_domain: {central_domain}

topics:
"""
        source_yaml += _topic_block('/clock', 'rosgraph_msgs/msg/Clock', reliability='best_effort', depth=1)
        source_yaml += _topic_block('/map', 'nav_msgs/msg/OccupancyGrid', durability='transient_local', depth=1)
        if include_rviz_topics:
            source_yaml += _topic_block('/scan', 'sensor_msgs/msg/LaserScan', reliability='best_effort', depth=3)
            source_yaml += _topic_block('/odom', 'nav_msgs/msg/Odometry', depth=3)
        source_yaml += _topic_block('/leader_pose', 'geometry_msgs/msg/PoseStamped', depth=1)
        source_yaml += _topic_block('/risk/yolo_detections', 'std_msgs/msg/String', reliability='best_effort', depth=1)
        source_to_central = _write_runtime_yaml(
            f'risk_source_{source_domain}_to_central_{central_domain}_',
            source_yaml,
            out_dir,
        )

        actions = [
            ExecuteProcess(
                cmd=['ros2', 'run', 'domain_bridge', 'domain_bridge', str(source_to_central)],
                output='screen',
                name=f'risk_source_{source_domain}_to_central_{central_domain}',
            )
        ]

        risk_topics = [
            ('/risk/risk_map', 'nav_msgs/msg/OccupancyGrid'),
            ('/risk/person_probability_map', 'nav_msgs/msg/OccupancyGrid'),
            ('/risk/evidence_markers', 'visualization_msgs/msg/MarkerArray'),
        ]
        for sink_domain in sink_domains:
            if sink_domain == central_domain:
                continue
            sink_yaml = f"""name: risk_maps_{central_domain}_to_{sink_domain}
from_domain: {central_domain}
to_domain: {sink_domain}

topics:
"""
            for topic, msg_type in risk_topics:
                sink_yaml += _topic_block(topic, msg_type, durability='transient_local', depth=1)
            central_to_sink = _write_runtime_yaml(
                f'risk_maps_{central_domain}_to_{sink_domain}_',
                sink_yaml,
                out_dir,
            )
            actions.append(
                ExecuteProcess(
                    cmd=['ros2', 'run', 'domain_bridge', 'domain_bridge', str(central_to_sink)],
                    output='screen',
                    name=f'risk_maps_{central_domain}_to_{sink_domain}',
                )
            )

        return actions

    risk_node = Node(
        condition=IfCondition(LaunchConfiguration('start_risk_map')),
        package='tb3_bayesian_risk_map',
        executable='bayesian_risk_map_node',
        name='bayesian_risk_map_node',
        output='screen',
        parameters=[
            LaunchConfiguration('config_file'),
            {
                'use_sim_time': LaunchConfiguration('use_sim_time'),
                'map_topic': LaunchConfiguration('map_topic'),
                'map_qos_durability': 'volatile',
                'map_frame': LaunchConfiguration('map_frame'),
                'base_frame': LaunchConfiguration('base_frame'),
                'pose_source': 'topic',
                'pose_topic': LaunchConfiguration('pose_topic'),
                'pose_topic_stale_sec': LaunchConfiguration('pose_topic_stale_sec'),
                'detection_source': 'flask_topic',
                'external_detection_topic': LaunchConfiguration('external_detection_topic'),
                'enable_yolo': False,
                'publish_overlay': False,
                'publish_debug_image': False,
                'publish_debug_compressed_image': False,
                'publish_diagnostic_maps': False,
                'debug_show_opencv': False,
                'teleop_mode': True,
                'risk_publish_rate_hz': LaunchConfiguration('risk_publish_rate_hz'),
                'diagnostic_publish_rate_hz': LaunchConfiguration('diagnostic_publish_rate_hz'),
                'region_update_period_sec': LaunchConfiguration('region_update_period_sec'),
                'enable_room_probability': LaunchConfiguration('enable_room_probability'),
                'enable_region_segmentation': LaunchConfiguration('enable_region_segmentation'),
                'enable_visibility_tracking': LaunchConfiguration('enable_visibility_tracking'),
            },
        ],
    )

    map_frame_anchor = ExecuteProcess(
        cmd=[
            'ros2', 'run', 'tf2_ros', 'static_transform_publisher',
            '0', '0', '0', '0', '0', '0',
            LaunchConfiguration('map_frame'),
            'risk_map_anchor',
        ],
        output='screen',
        name='risk_map_frame_anchor',
    )

    return LaunchDescription([
        DeclareLaunchArgument('central_domain_id', default_value=EnvironmentVariable('ROS_DOMAIN_ID')),
        DeclareLaunchArgument(
            'source_domain_id',
            default_value='',
            description=(
                'Robot DDS domain to bridge into the central risk map. Required '
                'when start_domain_bridges:=true; pass source_domain_id:=<robot_domain>.'
            ),
        ),
        DeclareLaunchArgument(
            'risk_sink_domain_ids',
            default_value='',
            description='Comma-separated domains that should receive central /risk maps.',
        ),
        DeclareLaunchArgument('use_sim_time', default_value='false'),
        DeclareLaunchArgument('start_domain_bridges', default_value='true'),
        DeclareLaunchArgument('start_risk_map', default_value='true'),
        DeclareLaunchArgument('bridge_rviz_topics', default_value='false'),
        DeclareLaunchArgument('config_file', default_value=default_config),
        DeclareLaunchArgument('map_topic', default_value='/map'),
        DeclareLaunchArgument('map_frame', default_value='map'),
        DeclareLaunchArgument('base_frame', default_value='base_footprint'),
        DeclareLaunchArgument('pose_topic', default_value='/leader_pose'),
        DeclareLaunchArgument('pose_topic_stale_sec', default_value='2.5'),
        DeclareLaunchArgument('external_detection_topic', default_value='/risk/yolo_detections'),
        DeclareLaunchArgument('risk_publish_rate_hz', default_value='5.0'),
        DeclareLaunchArgument('diagnostic_publish_rate_hz', default_value='1.0'),
        DeclareLaunchArgument('region_update_period_sec', default_value='1.5'),
        DeclareLaunchArgument('enable_room_probability', default_value='false'),
        DeclareLaunchArgument('enable_region_segmentation', default_value='false'),
        DeclareLaunchArgument('enable_visibility_tracking', default_value='true'),
        *dds_launch_environment(central_domain_id),
        TimerAction(
            period=0.5,
            actions=[map_frame_anchor, OpaqueFunction(function=_write_bridge_configs)],
            condition=IfCondition(LaunchConfiguration('start_domain_bridges')),
        ),
        TimerAction(period=1.5, actions=[risk_node]),
    ])
