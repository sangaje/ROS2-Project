import tempfile
from pathlib import Path
from typing import Dict, Optional, Tuple

import yaml


def qos(
    reliability: Optional[str] = 'reliable',
    durability: Optional[str] = 'volatile',
    depth: int = 10,
) -> Dict:
    profile = {
        'history': 'keep_last',
        'depth': depth,
    }
    if reliability is not None:
        profile['reliability'] = reliability
    if durability is not None:
        profile['durability'] = durability
    return profile


def map_qos(depth: int = 5) -> Dict:
    """Preserve map late-joiner semantics across domain_bridge."""
    return qos(reliability='reliable', durability='transient_local', depth=depth)


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
        '/risk/yolo_detections': topic(
            'std_msgs/msg/String',
            profile=qos(reliability='best_effort', durability='volatile', depth=1),
        ),
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
        '/risk/positive_memory_map': topic(
            'nav_msgs/msg/OccupancyGrid',
            profile=grid_profile,
        ),
        '/risk/visibility_map': topic(
            'nav_msgs/msg/OccupancyGrid',
            profile=grid_profile,
        ),
        '/risk/observed_empty_map': topic(
            'nav_msgs/msg/OccupancyGrid',
            profile=grid_profile,
        ),
        '/risk/bearing_consensus_map': topic(
            'nav_msgs/msg/OccupancyGrid',
            profile=grid_profile,
        ),
    }


def _runtime_output_directory(output_directory: Optional[Path]) -> Path:
    output = output_directory or Path(tempfile.gettempdir())
    output.mkdir(parents=True, exist_ok=True)
    return output


def _write_runtime_config(
    prefix: str,
    config: Dict,
    output_directory: Optional[Path] = None,
) -> Path:
    output = _runtime_output_directory(output_directory)
    with tempfile.NamedTemporaryFile(
        mode='w',
        suffix='.yaml',
        prefix=prefix,
        dir=output,
        delete=False,
        encoding='utf-8',
    ) as handle:
        yaml.safe_dump(config, handle, sort_keys=False)
        return Path(handle.name)


def write_fleet_bridge_configs(
    main_domain: int,
    follower_domain: int,
    *,
    simulation: bool = False,
    forward_map_to_main: bool = False,
    output_directory: Optional[Path] = None,
) -> Tuple[Path, Path]:
    """Create the two directional domain_bridge configurations.

    Control and acknowledgement topics use transient-local durability so a
    briefly restarting bridge or follower cannot miss the latest fleet state.
    """
    prefix = 'sim_' if simulation else ''

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
            profile=map_qos(depth=5),
        ),
        '/leader_pose': topic('geometry_msgs/msg/PoseStamped'),
        '/plan': topic(
            'nav_msgs/msg/Path',
            remap='/leader_plan',
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
        '/fleet/scout_takeover': topic(
            'std_msgs/msg/String',
            profile=qos(durability='transient_local', depth=1),
        ),
        '/fleet/scout_role': topic(
            'std_msgs/msg/String',
            profile=qos(durability='transient_local', depth=1),
        ),
        '/fleet/field_robot_role_cmd': topic(
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
        '/fleet/scout_takeover_status': topic(
            'std_msgs/msg/String',
            profile=qos(durability='transient_local', depth=1),
        ),
        '/fleet/field_robot_status': topic(
            'std_msgs/msg/String',
            profile=qos(durability='transient_local', depth=1),
        ),
        '/burger_scan_relay': topic(
            'sensor_msgs/msg/LaserScan',
            remap=f'/follower{int(follower_domain)}/scan',
            profile=qos('best_effort'),
        ),
        **risk_topics(),
    }
    if forward_map_to_main:
        follower_topics['/map'] = topic(
            'nav_msgs/msg/OccupancyGrid',
            remap='/map_bridge',
            profile=map_qos(depth=5),
        )
    if simulation:
        follower_topics['/cmd_vel'] = topic(
            'geometry_msgs/msg/TwistStamped',
            remap='/burger/cmd_vel',
        )

    main_to_follower = _write_runtime_config(
        f'{prefix}main_{main_domain}_to_follower_{follower_domain}_',
        {
        'name': f'{prefix}main_{main_domain}_to_follower_{follower_domain}',
        'from_domain': int(main_domain),
        'to_domain': int(follower_domain),
        'topics': main_topics,
        },
        output_directory,
    )
    follower_to_main = _write_runtime_config(
        f'{prefix}follower_{follower_domain}_to_main_{main_domain}_',
        {
        'name': f'{prefix}follower_{follower_domain}_to_main_{main_domain}',
        'from_domain': int(follower_domain),
        'to_domain': int(main_domain),
        'topics': follower_topics,
        },
        output_directory,
    )

    return main_to_follower, follower_to_main


def write_member_bridge_configs(
    main_domain: int,
    member_domain: int,
    *,
    forward_map_to_main: bool = False,
    output_directory: Optional[Path] = None,
) -> Tuple[Path, Path]:
    """Create the two directional domain_bridge configurations for a plain
    fleet member: it never leads or follows, it only reports its pose and
    receives a short yield goal from the coordinator when another robot
    needs to pass. Map and initial-pose flow mirror the follower's bridge
    so the same PC RViz can localize it.
    """
    main_topics = {
        '/map': topic(
            'nav_msgs/msg/OccupancyGrid',
            remap='/map_bridge',
            profile=map_qos(depth=5),
        ),
        '/leader_pose': topic('geometry_msgs/msg/PoseStamped'),
        '/member_goal_pose': topic('geometry_msgs/msg/PoseStamped'),
        '/fleet/coordination_status': topic(
            'std_msgs/msg/String',
            profile=qos(durability='transient_local', depth=1),
        ),
        '/fleet/scout_takeover': topic(
            'std_msgs/msg/String',
            profile=qos(durability='transient_local', depth=1),
        ),
        '/fleet/scout_role': topic(
            'std_msgs/msg/String',
            profile=qos(durability='transient_local', depth=1),
        ),
        '/fleet/field_robot_role_cmd': topic(
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
    }
    member_topics = {
        '/member_pose': topic('geometry_msgs/msg/PoseStamped'),
        '/scout/signal': topic(
            'std_msgs/msg/String',
            profile=qos(reliability='best_effort', durability='volatile', depth=5),
        ),
        '/fleet/scout_takeover_status': topic(
            'std_msgs/msg/String',
            profile=qos(durability='transient_local', depth=1),
        ),
        '/fleet/field_robot_status': topic(
            'std_msgs/msg/String',
            profile=qos(durability='transient_local', depth=1),
        ),
        **risk_topics(),
    }
    if forward_map_to_main:
        member_topics['/map'] = topic(
            'nav_msgs/msg/OccupancyGrid',
            remap='/map_bridge',
            profile=map_qos(depth=5),
        )

    main_to_member = _write_runtime_config(
        f'main_{main_domain}_to_member_{member_domain}_',
        {
        'name': f'main_{main_domain}_to_member_{member_domain}',
        'from_domain': int(main_domain),
        'to_domain': int(member_domain),
        'topics': main_topics,
        },
        output_directory,
    )
    member_to_main = _write_runtime_config(
        f'member_{member_domain}_to_main_{main_domain}_',
        {
        'name': f'member_{member_domain}_to_main_{main_domain}',
        'from_domain': int(member_domain),
        'to_domain': int(main_domain),
        'topics': member_topics,
        },
        output_directory,
    )

    return main_to_member, member_to_main


def write_risk_to_leader_bridge_config(
    risk_domain: int,
    leader_domain: int,
    *,
    include_map: bool = True,
    output_directory: Optional[Path] = None,
) -> Path:
    """Create the one-way scout/risk bridge into the leader domain.

    include_map=True is the target fleet mode: the scout/risk domain owns
    Cartographer and publishes the authoritative /map. The explicit
    leader-SLAM compatibility mode sets this false so domain 20's local
    Cartographer remains the only /map source in that mode.
    """
    topics = risk_topics()
    topics['/scout/signal'] = topic(
        'std_msgs/msg/String',
        profile=qos(reliability='best_effort', durability='volatile', depth=5),
    )
    if include_map:
        topics['/map'] = topic(
            'nav_msgs/msg/OccupancyGrid',
            remap='/map_bridge',
            profile=map_qos(depth=5),
        )
    return _write_runtime_config(
        f'risk_{risk_domain}_to_leader_{leader_domain}_',
        {
            'name': f'risk_{risk_domain}_to_leader_{leader_domain}',
            'from_domain': int(risk_domain),
            'to_domain': int(leader_domain),
            'topics': topics,
        },
        output_directory,
    )


def write_leader_to_pc_bridge_config(
    leader_domain: int,
    pc_domain: int,
    *,
    output_directory: Optional[Path] = None,
) -> Path:
    """Create the one-way visualization/debug bridge from leader to PC."""
    topics = {
        '/map': topic(
            'nav_msgs/msg/OccupancyGrid',
            profile=map_qos(depth=5),
        ),
        **risk_topics(),
        '/leader_pose': topic('geometry_msgs/msg/PoseStamped'),
        '/burger_pose': topic('geometry_msgs/msg/PoseStamped'),
        '/member_pose': topic('geometry_msgs/msg/PoseStamped'),
        '/leader_plan': topic(
            'nav_msgs/msg/Path',
            profile=qos(depth=3),
        ),
        '/burger_plan': topic(
            'nav_msgs/msg/Path',
            profile=qos(depth=3),
        ),
        '/goal_pose': topic('geometry_msgs/msg/PoseStamped'),
        '/fleet/leader_coord_goal': topic('geometry_msgs/msg/PoseStamped'),
        '/leader_shadow/goal': topic('geometry_msgs/msg/PoseStamped'),
        '/leader_shadow/state': topic(
            'std_msgs/msg/String',
            profile=qos(durability='transient_local', depth=1),
        ),
        '/leader_scan/state': topic(
            'std_msgs/msg/String',
            profile=qos(durability='transient_local', depth=1),
        ),
        '/burger_goal_pose': topic('geometry_msgs/msg/PoseStamped'),
        '/member_goal_pose': topic('geometry_msgs/msg/PoseStamped'),
        '/fleet/coordination_status': topic(
            'std_msgs/msg/String',
            profile=qos(durability='transient_local', depth=1),
        ),
        '/fleet/scout_takeover': topic(
            'std_msgs/msg/String',
            profile=qos(durability='transient_local', depth=1),
        ),
        '/fleet/scout_role': topic(
            'std_msgs/msg/String',
            profile=qos(durability='transient_local', depth=1),
        ),
        '/fleet/field_robot_role_cmd': topic(
            'std_msgs/msg/String',
            profile=qos(durability='transient_local', depth=1),
        ),
        '/fleet/scout_takeover_status': topic(
            'std_msgs/msg/String',
            profile=qos(durability='transient_local', depth=1),
        ),
        '/fleet/field_robot_status': topic(
            'std_msgs/msg/String',
            profile=qos(durability='transient_local', depth=1),
        ),
        '/failover/state': topic(
            'std_msgs/msg/String',
            profile=qos(durability='transient_local', depth=1),
        ),
        '/failover/active_scout_id': topic(
            'std_msgs/msg/String',
            profile=qos(durability='transient_local', depth=1),
        ),
        '/failover/scout_epoch': topic(
            'std_msgs/msg/String',
            profile=qos(durability='transient_local', depth=1),
        ),
        '/failover/scout_alive': topic(
            'std_msgs/msg/Bool',
            profile=qos(durability='transient_local', depth=1),
        ),
        '/failover/last_scout_pose': topic(
            'geometry_msgs/msg/PoseStamped',
            profile=qos(durability='transient_local', depth=1),
        ),
        '/failover/failure_pose': topic(
            'geometry_msgs/msg/PoseStamped',
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
        '/fleet/coordination_markers': topic(
            'visualization_msgs/msg/MarkerArray',
            profile=qos(durability='transient_local', depth=1),
        ),
        '/fleet_debug_markers': topic(
            'visualization_msgs/msg/MarkerArray',
            profile=qos(durability='transient_local', depth=1),
        ),
    }
    return _write_runtime_config(
        f'leader_{leader_domain}_to_pc_{pc_domain}_',
        {
            'name': f'leader_{leader_domain}_to_pc_{pc_domain}',
            'from_domain': int(leader_domain),
            'to_domain': int(pc_domain),
            'topics': topics,
        },
        output_directory,
    )
