"""Shared launch helpers for tb3_multi."""

from __future__ import annotations

import os
import tempfile
from typing import Iterable, List, Tuple

from launch.actions import OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


RobotSpec = Tuple[str, str, float, float, float]


ROBOTS: List[RobotSpec] = [
    ('burger1', 'burger1.sdf', -1.90, 0.00, 0.0),
    ('waffle1', 'waffle1.sdf', -1.80, 0.55, 0.0),
]

ROBOT_NAMES = [robot[0] for robot in ROBOTS]
BURGER1_WAYPOINTS = (
    '[[ -1.90, 0.00], [ -1.45, 0.00], [ -1.90, 0.55], '
    '[ -1.90, -0.55], [ -1.90, 0.00]]'
)


def robot_initial_parameters() -> dict:
    params = {}
    for name, _, x, y, yaw in ROBOTS:
        params[f'{name}_initial_x'] = x
        params[f'{name}_initial_y'] = y
        params[f'{name}_initial_yaw'] = yaw
    return params


def make_goal_controller(
    robot: str,
    initial_x: float,
    initial_y: float,
    initial_yaw: float,
    *,
    use_sim_time=False,
    publish_visual_markers: bool = False,
    condition=None,
) -> Node:
    return Node(
        package='tb3_multi',
        executable='simple_goal_controller',
        name=f'{robot}_goal_controller',
        parameters=[{
            'use_sim_time': use_sim_time,
            'robot_name': robot,
            'odom_topic': f'/{robot}/odom',
            'cmd_vel_topic': f'/{robot}/cmd_vel',
            'goal_topic': f'/{robot}/goal_point',
            'rescue_goal_topic': f'/{robot}/rescue_goal',
            'initial_x': initial_x,
            'initial_y': initial_y,
            'initial_yaw': initial_yaw,
            'goal_tolerance': 0.12,
            'max_linear_speed': 0.22,
            'max_angular_speed': 1.4,
            'publish_visual_markers': publish_visual_markers,
        }],
        condition=condition,
        output='screen',
    )


def make_sim_controller_nodes(use_sim_time=True, condition=None) -> List[Node]:
    return [
        make_goal_controller(
            robot,
            x,
            y,
            yaw,
            use_sim_time=use_sim_time,
            publish_visual_markers=False,
            condition=condition,
        )
        for robot, _, x, y, yaw in ROBOTS
    ]


def make_auto_patrol_node(map_use_sim_time, auto_patrol, rescue_offset_m) -> Node:
    params = {
        'use_sim_time': map_use_sim_time,
        'robots': ROBOT_NAMES,
        'burger_robots': ['burger1'],
        'goal_tolerance': 0.25,
        'signal_timeout_sec': 3.0,
        'startup_grace_sec': 8.0,
        'rescue_offset_m': ParameterValue(rescue_offset_m, value_type=float),
        'enable_patrol': auto_patrol,
        'patrol_robots': ['burger1'],
        'enable_timeout_detection': False,
        'burger1_waypoints': BURGER1_WAYPOINTS,
        'waffle1_waypoints': '[[ -1.80, 0.55]]',
    }
    params.update(robot_initial_parameters())
    return Node(
        package='tb3_multi',
        executable='auto_patrol_rescue',
        name='auto_patrol_rescue',
        parameters=[params],
        output='screen',
    )


def make_region_nav2_node(
    use_sim_time,
    map_topic,
    nav2_goal_topic,
    publish_nav2_goal,
    condition=None,
) -> Node:
    return Node(
        package='tb3_multi',
        executable='region_nav2_goal',
        name='region_nav2_goal',
        parameters=[{
            'use_sim_time': use_sim_time,
            'map_topic': map_topic,
            'nav2_goal_topic': nav2_goal_topic,
            'publish_nav2_goal': publish_nav2_goal,
            'marker_topic': '/tb3_multi/region_markers',
            'region_centers_topic': '/tb3_multi/region_centers',
            'selected_region_topic': '/tb3_multi/selected_region',
        }],
        condition=condition,
        output='screen',
    )


def make_ros_gz_bridge_args(robots: Iterable[str] = ROBOT_NAMES) -> List[str]:
    args = ['/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock']
    for robot in robots:
        args.extend([
            f'/{robot}/cmd_vel@geometry_msgs/msg/Twist]gz.msgs.Twist',
            f'/{robot}/odom@nav_msgs/msg/Odometry[gz.msgs.Odometry',
            f'/{robot}/joint_states@sensor_msgs/msg/JointState[gz.msgs.Model',
            f'/{robot}/scan@sensor_msgs/msg/LaserScan[gz.msgs.LaserScan',
            f'/{robot}/imu@sensor_msgs/msg/Imu[gz.msgs.IMU',
        ])
    return args


def _topic_yaml_entry(topic: str, msg_type: str, from_domain: str, to_domain: str) -> str:
    return (
        f'  {topic}:\n'
        f'    type: {msg_type}\n'
        f'    from_domain: {from_domain}\n'
        f'    to_domain: {to_domain}\n'
    )


def _bridge_topics_yaml(
    namespace: str,
    robot_domain: str,
    base_domain: str,
    bridge_map: bool = False,
) -> str:
    ns = namespace.strip().strip('/')
    if not ns:
        raise RuntimeError('namespace launch argument must not be empty')
    robot_to_base = [
        (f'/{ns}/odom', 'nav_msgs/msg/Odometry'),
        (f'/{ns}/scan', 'sensor_msgs/msg/LaserScan'),
        (f'/{ns}/imu', 'sensor_msgs/msg/Imu'),
        (f'/{ns}/joint_states', 'sensor_msgs/msg/JointState'),
        (f'/{ns}/map_pose', 'geometry_msgs/msg/PoseStamped'),
        (f'/{ns}/signal', 'std_msgs/msg/String'),
        ('/tb3_multi/robot_failure_report', 'std_msgs/msg/String'),
        ('/tb3_multi/region_markers', 'visualization_msgs/msg/MarkerArray'),
        ('/tb3_multi/region_centers', 'geometry_msgs/msg/PoseArray'),
        ('/tb3_multi/selected_region', 'geometry_msgs/msg/PointStamped'),
    ]
    if bridge_map:
        robot_to_base.append(('/map', 'nav_msgs/msg/OccupancyGrid'))
    base_to_robot = [
        (f'/{ns}/goal_point', 'geometry_msgs/msg/PointStamped'),
        (f'/{ns}/rescue_goal', 'geometry_msgs/msg/PointStamped'),
        (f'/{ns}/goal_pose', 'geometry_msgs/msg/PoseStamped'),
    ]
    for robot in ROBOT_NAMES:
        if robot != ns:
            base_to_robot.append((f'/{robot}/signal', 'std_msgs/msg/String'))
    bidirectional = [
        ('/tb3_multi/fail_robot', 'std_msgs/msg/String'),
        ('/tb3_multi/recover_robot', 'std_msgs/msg/String'),
    ]

    lines = [
        f'name: tb3_multi_{ns}_domain_bridge',
        f'from_domain: {robot_domain}',
        f'to_domain: {base_domain}',
        'topics:',
    ]
    for topic, msg_type in robot_to_base:
        lines.append(_topic_yaml_entry(topic, msg_type, robot_domain, base_domain).rstrip())
    for topic, msg_type in base_to_robot:
        lines.append(_topic_yaml_entry(topic, msg_type, base_domain, robot_domain).rstrip())
    for topic, msg_type in bidirectional:
        lines.append(
            _topic_yaml_entry(topic, msg_type, base_domain, robot_domain).rstrip()
            + '\n'
            + '    bidirectional: true'
        )
    return '\n'.join(lines) + '\n'


def make_dynamic_domain_bridge(
    namespace: LaunchConfiguration,
    robot_domain_id: LaunchConfiguration,
    base_domain_id: LaunchConfiguration,
    bridge_map: LaunchConfiguration,
):
    def launch_setup(context, *args, **kwargs):
        del args, kwargs
        ns = namespace.perform(context)
        robot_domain = robot_domain_id.perform(context)
        base_domain = base_domain_id.perform(context)
        should_bridge_map = bridge_map.perform(context).strip().lower() in (
            '1',
            'true',
            'yes',
            'on',
        )
        yaml_text = _bridge_topics_yaml(
            ns,
            robot_domain,
            base_domain,
            bridge_map=should_bridge_map,
        )
        fd, path = tempfile.mkstemp(
            prefix=f'tb3_multi_{ns.strip().strip("/") or "robot"}_',
            suffix='_domain_bridge.yaml',
            dir='/tmp',
            text=True,
        )
        with os.fdopen(fd, 'w', encoding='utf-8') as f:
            f.write(yaml_text)
        return [
            Node(
                package='domain_bridge',
                executable='domain_bridge',
                name=f'{ns.strip().strip("/")}_domain_bridge',
                arguments=[path],
                output='screen',
            )
        ]

    return OpaqueFunction(function=launch_setup)


def make_physical_robot_controller(
    namespace: LaunchConfiguration,
    initial_x: LaunchConfiguration,
    initial_y: LaunchConfiguration,
    initial_yaw: LaunchConfiguration,
    use_sim_time: LaunchConfiguration,
    enable_test_controller: LaunchConfiguration,
):
    def launch_setup(context, *args, **kwargs):
        del args, kwargs
        enabled = enable_test_controller.perform(context).strip().lower()
        if enabled not in ('1', 'true', 'yes', 'on'):
            return []
        robot = namespace.perform(context).strip().strip('/')
        if not robot:
            raise RuntimeError('namespace launch argument must not be empty')
        return [
            make_goal_controller(
                robot,
                float(initial_x.perform(context)),
                float(initial_y.perform(context)),
                float(initial_yaw.perform(context)),
                use_sim_time=use_sim_time,
                publish_visual_markers=False,
            )
        ]

    return OpaqueFunction(function=launch_setup)


def make_robot_signal_node(
    namespace: LaunchConfiguration,
    publish_signal: LaunchConfiguration,
    enable_signal_node: LaunchConfiguration,
):
    def launch_setup(context, *args, **kwargs):
        del args, kwargs
        enabled = enable_signal_node.perform(context).strip().lower()
        if enabled not in ('1', 'true', 'yes', 'on'):
            return []
        robot = namespace.perform(context).strip().strip('/')
        if not robot:
            raise RuntimeError('namespace launch argument must not be empty')
        peers = [name for name in ROBOT_NAMES if name != robot]
        return [
            Node(
                package='tb3_multi',
                executable='robot_signal',
                name=f'{robot}_signal',
                parameters=[{
                    'robot_name': robot,
                    'peer_robots': peers,
                    'publish_signal': publish_signal,
                }],
                output='screen',
            )
        ]

    return OpaqueFunction(function=launch_setup)
