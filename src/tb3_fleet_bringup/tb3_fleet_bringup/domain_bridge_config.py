from pathlib import Path
import tempfile
from typing import Dict, Optional, Tuple

import yaml


def qos(
    reliability: str = 'reliable',
    durability: str = 'volatile',
    depth: int = 10,
) -> Dict:
    return {
        'reliability': reliability,
        'durability': durability,
        'history': 'keep_last',
        'depth': depth,
    }


def topic(message_type: str, *, remap: Optional[str] = None, profile=None) -> Dict:
    config = {
        'type': message_type,
        'qos': profile or qos(),
    }
    if remap:
        config['remap'] = remap
    return config


def write_fleet_bridge_configs(
    main_domain: int,
    follower_domain: int,
    *,
    simulation: bool = False,
    output_directory: Optional[Path] = None,
) -> Tuple[Path, Path]:
    """Create the two directional domain_bridge configurations.

    Control and acknowledgement topics use transient-local durability so a
    briefly restarting bridge or follower cannot miss the latest fleet state.
    """
    output = output_directory or (
        Path(tempfile.gettempdir()) / 'tb3_fleet_domain_bridge'
    )
    output.mkdir(parents=True, exist_ok=True)

    prefix = 'sim_' if simulation else ''
    main_to_follower = output / (
        f'{prefix}main_{main_domain}_to_follower_{follower_domain}.yaml'
    )
    follower_to_main = output / (
        f'{prefix}follower_{follower_domain}_to_main_{main_domain}.yaml'
    )

    main_topics = {}
    if simulation:
        main_topics['/clock'] = topic(
            'rosgraph_msgs/msg/Clock',
            profile=qos('best_effort', depth=10),
        )
    main_topics.update({
        '/map': topic(
            'nav_msgs/msg/OccupancyGrid',
            remap='/map_bridge',
            profile=qos(depth=5),
        ),
        '/leader_pose': topic('geometry_msgs/msg/PoseStamped'),
        '/plan': topic(
            'nav_msgs/msg/Path',
            remap='/waffle_plan',
            profile=qos(depth=3),
        ),
        '/burger_goal_pose': topic('geometry_msgs/msg/PoseStamped'),
        '/fleet/follow_command': topic(
            'std_msgs/msg/String',
            profile=qos(durability='transient_local', depth=1),
        ),
        '/fleet/coordination_status': topic(
            'std_msgs/msg/String',
            profile=qos(durability='transient_local', depth=1),
        ),
        '/fleet/robot_poses': topic(
            'geometry_msgs/msg/PoseArray',
            profile=qos(depth=5),
        ),
        '/fleet/collision_warning': topic(
            'std_msgs/msg/Bool',
            profile=qos(durability='transient_local', depth=1),
        ),
        '/fleet/hazard_pose': topic(
            'geometry_msgs/msg/PoseStamped',
            profile=qos(depth=5),
        ),
        '/initialpose': topic(
            'geometry_msgs/msg/PoseWithCovarianceStamped',
            profile=qos(depth=1),
        ),
    })
    if simulation:
        main_topics.update({
            '/burger/scan': topic(
                'sensor_msgs/msg/LaserScan',
                remap='/scan_bridge',
                profile=qos('best_effort'),
            ),
            '/burger/odom': topic(
                'nav_msgs/msg/Odometry',
                remap='/odom_bridge',
            ),
            '/burger/joint_states': topic(
                'sensor_msgs/msg/JointState',
                remap='/joint_states',
            ),
            '/burger/tf': topic(
                'tf2_msgs/msg/TFMessage',
                profile=qos(depth=100),
            ),
        })

    follower_topics = {
        '/burger_pose': topic('geometry_msgs/msg/PoseStamped'),
        '/plan': topic(
            'nav_msgs/msg/Path',
            remap='/burger_plan',
            profile=qos(depth=3),
        ),
        '/fleet/follow_enabled': topic(
            'std_msgs/msg/Bool',
            profile=qos(durability='transient_local', depth=1),
        ),
        '/burger_scan_relay': topic(
            'sensor_msgs/msg/LaserScan',
            remap='/burger_scan',
            profile=qos('best_effort'),
        ),
    }
    if simulation:
        follower_topics['/cmd_vel'] = topic(
            'geometry_msgs/msg/TwistStamped',
            remap='/burger/cmd_vel',
        )

    main_to_follower.write_text(yaml.safe_dump({
        'name': f'{prefix}main_{main_domain}_to_follower_{follower_domain}',
        'from_domain': int(main_domain),
        'to_domain': int(follower_domain),
        'topics': main_topics,
    }, sort_keys=False), encoding='utf-8')
    follower_to_main.write_text(yaml.safe_dump({
        'name': f'{prefix}follower_{follower_domain}_to_main_{main_domain}',
        'from_domain': int(follower_domain),
        'to_domain': int(main_domain),
        'topics': follower_topics,
    }, sort_keys=False), encoding='utf-8')

    return main_to_follower, follower_to_main
