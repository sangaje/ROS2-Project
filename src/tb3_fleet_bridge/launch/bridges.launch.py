#!/usr/bin/env python3

import tempfile
from pathlib import Path

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, LogInfo, OpaqueFunction
from launch.substitutions import LaunchConfiguration


def _enabled(value: str) -> bool:
    return value.strip().lower() in ('true', '1', 'yes', 'on')


def _parse_robot_domains(spec: str):
    robots = []
    for item in spec.split(','):
        item = item.strip()
        if not item:
            continue
        if ':' in item:
            name, domain = item.split(':', 1)
        else:
            name, domain = 'burger', item
        name = name.strip()
        domain = domain.strip()
        if not name or not domain:
            raise RuntimeError(f'Invalid robot_domains entry: {item!r}')
        robots.append((name, domain))
    if not robots:
        raise RuntimeError('robot_domains must contain at least one entry, for example burger:24')
    return robots


def _main_topic(robot_name: str, suffix: str) -> str:
    if robot_name == 'burger':
        return f'/burger_{suffix}'
    return f'/{robot_name}_{suffix}'


def generate_launch_description():
    main_domain_id = LaunchConfiguration('main_domain_id')
    robot_domains = LaunchConfiguration('robot_domains')
    bridge_map = LaunchConfiguration('bridge_map')
    bridge_scan = LaunchConfiguration('bridge_scan')
    bridge_nav = LaunchConfiguration('bridge_nav')
    bridge_follow_command = LaunchConfiguration('bridge_follow_command')

    def make_bridges(context, *args, **kwargs):
        main_domain = main_domain_id.perform(context)
        robots = _parse_robot_domains(robot_domains.perform(context))
        do_map = _enabled(bridge_map.perform(context))
        do_scan = _enabled(bridge_scan.perform(context))
        do_nav = _enabled(bridge_nav.perform(context))
        do_follow = _enabled(bridge_follow_command.perform(context))

        out_dir = Path(tempfile.gettempdir()) / 'tb3_fleet_bridge_dynamic'
        out_dir.mkdir(parents=True, exist_ok=True)
        actions = [
            LogInfo(msg=[
                'FLEET_BRIDGES_DYNAMIC | main_domain=', main_domain,
                ' robots=', robot_domains,
                ' out=', str(out_dir),
            ])
        ]

        for robot_name, robot_domain in robots:
            main_to_robot = out_dir / f'{robot_name}_{main_domain}_to_{robot_domain}.yaml'
            robot_to_main = out_dir / f'{robot_name}_{robot_domain}_to_{main_domain}.yaml'

            main_topics = []
            if do_map:
                main_topics.append("""  /map:
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
""")
            main_topics.append("""  /leader_pose:
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
  /plan:
    type: nav_msgs/msg/Path
    remap: /waffle_plan
    qos:
      reliability: reliable
      durability: volatile
      history: keep_last
      depth: 10
""")
            if do_nav:
                main_topics.append(f"""  {_main_topic(robot_name, 'goal_pose')}:
    type: geometry_msgs/msg/PoseStamped
    remap: /burger_goal_pose
    qos:
      reliability: reliable
      durability: volatile
      history: keep_last
      depth: 10
""")
            if do_follow:
                main_topics.append(f"""  /fleet/{robot_name}/follow_command:
    type: std_msgs/msg/String
    remap: /fleet/follow_command
    qos:
      reliability: reliable
      durability: volatile
      history: keep_last
      depth: 10
""")

            robot_topics = [f"""  /burger_pose:
    type: geometry_msgs/msg/PoseStamped
    remap: {_main_topic(robot_name, 'pose')}
    qos:
      reliability: reliable
      durability: volatile
      history: keep_last
      depth: 10
"""]
            if do_scan:
                robot_topics.append(f"""  /burger_scan_relay:
    type: sensor_msgs/msg/LaserScan
    remap: {_main_topic(robot_name, 'scan')}
    qos:
      reliability: best_effort
      durability: volatile
      history: keep_last
      depth: 10
""")
            if do_nav:
                robot_topics.append(f"""  /plan:
    type: nav_msgs/msg/Path
    remap: {_main_topic(robot_name, 'plan')}
    qos:
      reliability: reliable
      durability: volatile
      history: keep_last
      depth: 10
""")
            if do_follow:
                robot_topics.append(f"""  /fleet/follow_enabled:
    type: std_msgs/msg/Bool
    remap: /fleet/{robot_name}/follow_enabled
    qos:
      reliability: reliable
      durability: transient_local
      history: keep_last
      depth: 1
""")

            main_to_robot.write_text(
                f"""name: {robot_name}_{main_domain}_to_{robot_domain}
from_domain: {main_domain}
to_domain: {robot_domain}

topics:
{''.join(main_topics)}""",
                encoding='utf-8',
            )
            robot_to_main.write_text(
                f"""name: {robot_name}_{robot_domain}_to_{main_domain}
from_domain: {robot_domain}
to_domain: {main_domain}

topics:
{''.join(robot_topics)}""",
                encoding='utf-8',
            )

            actions.extend([
                ExecuteProcess(
                    cmd=['ros2', 'run', 'domain_bridge', 'domain_bridge', str(main_to_robot)],
                    output='screen',
                    name=f'{robot_name}_main_to_robot_bridge',
                ),
                ExecuteProcess(
                    cmd=['ros2', 'run', 'domain_bridge', 'domain_bridge', str(robot_to_main)],
                    output='screen',
                    name=f'{robot_name}_robot_to_main_bridge',
                ),
            ])
        return actions

    return LaunchDescription([
        DeclareLaunchArgument('main_domain_id', default_value='25'),
        DeclareLaunchArgument(
            'robot_domains',
            default_value='burger:24',
            description='Comma-separated robot bridge specs, e.g. burger:24,burger2:26',
        ),
        DeclareLaunchArgument('bridge_map', default_value='true'),
        DeclareLaunchArgument('bridge_scan', default_value='true'),
        DeclareLaunchArgument('bridge_nav', default_value='true'),
        DeclareLaunchArgument('bridge_follow_command', default_value='true'),
        OpaqueFunction(function=make_bridges),
    ])
