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


def risk_topics() -> Dict[str, Dict]:
    """Core Bayesian risk outputs a scout should forward to the main domain."""
    grid_profile = qos(durability='transient_local', depth=1)
    marker_profile = qos(durability='transient_local', depth=1)
    return {
        '/risk/risk_map': topic(
            'nav_msgs/msg/OccupancyGrid',
            profile=grid_profile,
        ),
        '/risk/person_probability_map': topic(
            'nav_msgs/msg/OccupancyGrid',
            profile=grid_profile,
        ),
        '/risk/evidence_markers': topic(
            'visualization_msgs/msg/MarkerArray',
            profile=marker_profile,
        ),
    }


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
            remap=f'/follower{int(follower_domain)}/scan',
            profile=qos('best_effort'),
        ),
        **risk_topics(),
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


def write_member_bridge_configs(
    main_domain: int,
    member_domain: int,
    *,
    output_directory: Optional[Path] = None,
) -> Tuple[Path, Path]:
    """Create the two directional domain_bridge configurations for a plain
    fleet member: it never leads or follows, it only reports its pose and
    receives a short yield goal from the coordinator when another robot
    needs to pass. Map and initial-pose flow mirror the follower's bridge
    so the same PC RViz can localize it.
    """
    output = output_directory or (
        Path(tempfile.gettempdir()) / 'tb3_fleet_domain_bridge'
    )
    output.mkdir(parents=True, exist_ok=True)

    main_to_member = output / (
        f'main_{main_domain}_to_member_{member_domain}.yaml'
    )
    member_to_main = output / (
        f'member_{member_domain}_to_main_{main_domain}.yaml'
    )

    main_topics = {
        '/map': topic(
            'nav_msgs/msg/OccupancyGrid',
            remap='/map_bridge',
            profile=qos(depth=5),
        ),
        '/member_goal_pose': topic('geometry_msgs/msg/PoseStamped'),
        '/initialpose': topic(
            'geometry_msgs/msg/PoseWithCovarianceStamped',
            profile=qos(depth=1),
        ),
    }
    member_topics = {
        '/member_pose': topic('geometry_msgs/msg/PoseStamped'),
        # Only actually publishes if this member owns its own SLAM
        # (enable_amcl:=false + start_cartographer:=true) -- otherwise
        # nothing ever appears here and the bridge just idles. Lets a
        # leader with enable_cartographer:=false receive this member's
        # map instead of building its own.
        '/map': topic(
            'nav_msgs/msg/OccupancyGrid',
            remap='/map_from_member',
            profile=qos(depth=5),
        ),
        **risk_topics(),
    }

    main_to_member.write_text(yaml.safe_dump({
        'name': f'main_{main_domain}_to_member_{member_domain}',
        'from_domain': int(main_domain),
        'to_domain': int(member_domain),
        'topics': main_topics,
    }, sort_keys=False), encoding='utf-8')
    member_to_main.write_text(yaml.safe_dump({
        'name': f'member_{member_domain}_to_main_{main_domain}',
        'from_domain': int(member_domain),
        'to_domain': int(main_domain),
        'topics': member_topics,
    }, sort_keys=False), encoding='utf-8')

    return main_to_member, member_to_main
