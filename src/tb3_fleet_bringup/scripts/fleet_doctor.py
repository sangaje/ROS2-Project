#!/usr/bin/env python3

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path


NODE_PATTERNS = [
    'cartographer',
    'occupancy',
    'controller_server',
    'planner_server',
    'bt_navigator',
    'lifecycle',
    'fleet_goal_dispatcher',
    'pose_to_nav2',
    'twist',
    'robot_footprint_map_free_space_filter',
]

TOPICS = [
    '/fleet_goal_pose',
    '/waffle_goal_pose',
    '/burger_goal_pose',
    '/cmd_vel',
    '/cmd_vel_stamped',
    '/gz_cmd_vel_unstamped',
    '/map',
    '/map_raw',
    '/scan_nav',
    '/odom_nav',
]

REQUIRED_RELATIVE_FILES = [
    'launch/fleet_waffle_nav2_group.launch.py',
    'launch/fleet_burger_nav2_group.launch.py',
    'launch/fleet_debug_rviz.launch.py',
    'rviz/fleet_debug.rviz',
    'scripts/fleet_goal_dispatcher.py',
    'scripts/pose_to_nav2_action.py',
    'scripts/robot_footprint_map_free_space_filter.py',
    'scripts/single_twist_stamped_to_twist.py',
]


def run(cmd: list[str], timeout: float = 5.0) -> tuple[int, str]:
    try:
        proc = subprocess.run(cmd, text=True, capture_output=True, timeout=timeout, check=False)
    except Exception as exc:
        return 99, str(exc)
    return proc.returncode, (proc.stdout + proc.stderr).strip()


def print_section(title: str) -> None:
    print(f'\n== {title} ==')


def find_share_dir() -> Path:
    code, out = run(['ros2', 'pkg', 'prefix', 'tb3_fleet_bringup'])
    if code == 0 and out:
        return Path(out.splitlines()[0]) / 'share' / 'tb3_fleet_bringup'
    here = Path(__file__).resolve()
    for parent in here.parents:
        if parent.name == 'tb3_fleet_bringup':
            return parent
    return Path.cwd()


def topic_types(topic_info: str) -> list[str]:
    types: list[str] = []
    for line in topic_info.splitlines():
        if 'Type:' in line:
            types.append(line.split('Type:', 1)[1].strip())
    return types


def main() -> int:
    domain = os.environ.get('ROS_DOMAIN_ID', '<unset>')
    print(f'FLEET_DOCTOR | ROS_DOMAIN_ID={domain}')

    print_section('Nodes')
    code, out = run(['ros2', 'node', 'list', '--no-daemon'])
    if code != 0:
        print(f'WARN: ros2 node list failed: {out}')
    else:
        nodes = out.splitlines()
        for pattern in NODE_PATTERNS:
            matches = [n for n in nodes if pattern in n]
            print(f'{pattern}: {matches if matches else "MISSING"}')

    print_section('Topics')
    cmd_vel_types: list[str] = []
    for topic in TOPICS:
        code, out = run(['ros2', 'topic', 'info', topic, '-v', '--no-daemon'])
        if code != 0:
            print(f'{topic}: MISSING_OR_NO_GRAPH')
            continue
        types = topic_types(out)
        if topic == '/cmd_vel':
            cmd_vel_types = types
        print(f'{topic}:\n{out}\n')

    unique_cmd_vel_types = sorted(set(cmd_vel_types))
    if len(unique_cmd_vel_types) > 1:
        print(f'WARN: /cmd_vel has multiple types: {unique_cmd_vel_types}')
    elif unique_cmd_vel_types and unique_cmd_vel_types[0] != 'geometry_msgs/msg/Twist':
        print(f'WARN: /cmd_vel should be geometry_msgs/msg/Twist after conversion, got {unique_cmd_vel_types[0]}')

    print_section('Files')
    share = find_share_dir()
    print(f'share={share}')
    for rel in REQUIRED_RELATIVE_FILES:
        path = share / rel
        print(f'{rel}: {"OK" if path.exists() else "MISSING"}')

    print_section('Static Warnings')
    text = ''
    launch_text = ''
    for rel in ('launch', 'scripts', 'config'):
        root = share / rel
        if root.exists():
            for path in root.rglob('*'):
                if path.is_file() and path.suffix in ('.py', '.yaml', '.lua'):
                    try:
                        body = path.read_text(errors='ignore')
                        text += f'\n# {path}\n' + body
                        if rel == 'launch':
                            launch_text += f'\n# {path}\n' + body
                    except Exception:
                        pass
    forbidden_slam = '|'.join(['slam_' + 'toolbox', 'online_' + 'async', 'async_' + 'slam', 'sync_' + 'slam'])
    if re.search(forbidden_slam, text, re.IGNORECASE):
        print('WARN: forbidden SLAM-related text found')
    else:
        print('OK: no forbidden SLAM references found in installed fleet files')
    if 'executable=\'occupancy_grid_node\'' in launch_text or 'executable="occupancy_grid_node"' in launch_text:
        print('WARN: old occupancy_grid_node executable reference found')
    else:
        print('OK: no old occupancy_grid_node executable reference found')
    if re.search(r'pose_to_nav2_action_direct_v41|fleet_goal_dispatcher_direct_v41|v41|v44|v66|v70|v71', launch_text):
        print('WARN: stale version reference found')
    else:
        print('OK: no stale version launch refs found')

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
