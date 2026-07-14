import tempfile
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

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


def map_qos(depth: int = 1) -> Dict:
    """Preserve map late-joiner semantics across domain_bridge."""
    return qos(reliability='reliable', durability='transient_local', depth=depth)


def live_map_qos(depth: int = 1) -> Dict:
    """Match live Cartographer map publishers that may offer volatile durability."""
    return qos(reliability='reliable', durability='volatile', depth=depth)


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


def system_readiness_topics() -> Dict[str, Dict]:
    """Latched fleet-wide startup outputs that cross robot domains."""
    latched = qos(durability='transient_local', depth=1)
    return {
        '/system/ready': topic('std_msgs/msg/Bool', profile=latched),
        '/fleet/start_motion': topic('std_msgs/msg/Bool', profile=latched),
        '/fleet/readiness_detail': topic('std_msgs/msg/String', profile=latched),
        '/fleet/start_motion_detail': topic('std_msgs/msg/String', profile=latched),
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
    validate_no_duplicate_bridge_routes([config])
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


def validate_no_duplicate_bridge_routes(configs: Iterable[Dict]) -> None:
    """Reject duplicate bridge routes before domain_bridge processes start."""
    seen: dict[tuple[int, int, str, str], str] = {}
    for config in configs:
        try:
            from_domain = int(config['from_domain'])
            to_domain = int(config['to_domain'])
        except (KeyError, TypeError, ValueError):
            continue
        name = str(config.get('name', '<unnamed>'))
        topics = config.get('topics', {})
        if not isinstance(topics, dict):
            continue
        for source_topic, spec in topics.items():
            if not isinstance(spec, dict):
                continue
            destination = str(spec.get('remap', source_topic))
            key = (from_domain, to_domain, str(source_topic), destination)
            if key in seen:
                raise ValueError(
                    'duplicate domain_bridge route: '
                    f'{source_topic}->{destination} {from_domain}->{to_domain} '
                    f'in {seen[key]} and {name}'
                )
            seen[key] = name


def _topic_is_sensitive(topic_name: str) -> bool:
    topic_name = str(topic_name)
    exact = {
        '/map',
        '/map_bridge',
        '/shared_map_in',
        '/local_slam_map',
        '/rl_confidence_map',
        '/rl_confidence_seed',
        '/scan',
        '/tf',
        '/tf_static',
        '/plan',
    }
    return (
        topic_name in exact
        or topic_name.startswith('/risk/')
        or topic_name.endswith('/map')
        or topic_name.endswith('/rl_confidence_map')
    )


def validate_no_bridge_feedback_cycles(
    configs: Iterable[Dict],
    *,
    relay_edges: Iterable[Tuple[Tuple[int, str], Tuple[int, str], str]] = (),
) -> None:
    """Reject sensitive bridge+relay cycles before launching domain_bridge.

    The important failure mode is a large topic entering a robot domain,
    being republished by a relay under an authoritative name, then being
    bridged back to the original domain.  This catches that structure while
    still allowing unrelated control topics to move in both directions.
    """
    graph: dict[tuple[int, str], list[tuple[tuple[int, str], str]]] = {}

    def add_edge(src: tuple[int, str], dst: tuple[int, str], label: str) -> None:
        graph.setdefault(src, []).append((dst, label))

    for config in configs:
        try:
            from_domain = int(config['from_domain'])
            to_domain = int(config['to_domain'])
        except (KeyError, TypeError, ValueError):
            continue
        name = str(config.get('name', '<unnamed>'))
        topics = config.get('topics', {})
        if not isinstance(topics, dict):
            continue
        for source_topic, spec in topics.items():
            if not isinstance(spec, dict):
                continue
            destination = str(spec.get('remap', source_topic))
            add_edge(
                (from_domain, str(source_topic)),
                (to_domain, destination),
                f'{name}:{source_topic}->{destination}',
            )

    for src, dst, label in relay_edges:
        add_edge(src, dst, label)

    def walk(
        start: tuple[int, str],
        node: tuple[int, str],
        path: list[tuple[int, str]],
        labels: list[str],
    ) -> None:
        for nxt, label in graph.get(node, []):
            if nxt == start and len(path) > 1:
                cycle_topics = [item[1] for item in path + [nxt]]
                if any(_topic_is_sensitive(item) for item in cycle_topics):
                    route = ' -> '.join(
                        f'{domain}:{topic}' for domain, topic in path + [nxt]
                    )
                    raise ValueError(
                        'sensitive bridge feedback cycle detected: '
                        f'{route} via {", ".join(labels + [label])}'
                    )
            if nxt in path or len(path) >= 8:
                continue
            walk(start, nxt, path + [nxt], labels + [label])

    for start in list(graph):
        walk(start, start, [start], [])


def validate_bridge_config_files(paths: Iterable[Path]) -> None:
    """Load generated domain_bridge YAML files and reject duplicate routes."""
    configs = []
    for path in paths:
        with Path(path).open('r', encoding='utf-8') as handle:
            configs.append(yaml.safe_load(handle) or {})
    validate_no_duplicate_bridge_routes(configs)
    validate_no_bridge_feedback_cycles(configs)


def field_robot_identity_topics(
    robot_name: str,
    *,
    pose_source: str,
    include_map: bool = True,
) -> Dict[str, Dict]:
    """Robot-domain outputs remapped into stable leader-domain identity topics."""
    robot = str(robot_name).strip()
    if not robot:
        raise ValueError('robot_name is required for identity bridge topics')
    topics = {
        pose_source: topic(
            'geometry_msgs/msg/PoseStamped',
            remap=f'/field/{robot}/pose',
        ),
        '/scout/signal': topic(
            'std_msgs/msg/String',
            remap=f'/field/{robot}/heartbeat',
            profile=qos(reliability='best_effort', durability='volatile', depth=5),
        ),
        '/fleet/field_robot_status': topic(
            'std_msgs/msg/String',
            remap=f'/field/{robot}/status',
            profile=qos(durability='transient_local', depth=1),
        ),
        f'/field/{robot}/risk_observation': topic(
            'std_msgs/msg/String',
            profile=qos(reliability='best_effort', durability='volatile', depth=5),
        ),
    }
    if include_map:
        topics['/map'] = topic(
            'nav_msgs/msg/OccupancyGrid',
            remap=f'/field/{robot}/map',
            profile=map_qos(),
        )
    return topics


def write_fleet_bridge_configs(
    main_domain: int,
    follower_domain: int,
    *,
    simulation: bool = False,
    forward_map_to_main: bool = False,
    include_follower_scan: bool = False,
    include_leader_map: bool = True,
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
        **system_readiness_topics(),
        '/leader_pose': topic('geometry_msgs/msg/PoseStamped'),
        '/member_pose': topic('geometry_msgs/msg/PoseStamped'),
        '/omx/target_detected': topic(
            'std_msgs/msg/Bool',
            profile=qos(reliability='best_effort', durability='volatile', depth=5),
        ),
        '/omx/camera_ready': topic(
            'std_msgs/msg/Bool',
            profile=qos(reliability='best_effort', durability='volatile', depth=5),
        ),
        '/omx/camera_yaw': topic(
            'std_msgs/msg/Float32',
            profile=qos(reliability='best_effort', durability='volatile', depth=5),
        ),
        '/omx/observation_status': topic(
            'std_msgs/msg/String',
            profile=qos(reliability='best_effort', durability='volatile', depth=5),
        ),
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
        '/failover/active_scout_id': topic(
            'std_msgs/msg/String',
            profile=qos(durability='transient_local', depth=1),
        ),
        '/failover/last_scout_pose': topic(
            'geometry_msgs/msg/PoseStamped',
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
        '/fleet/video_ready': topic(
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
    if include_leader_map:
        main_topics['/map'] = topic(
            'nav_msgs/msg/OccupancyGrid',
            remap='/shared_map_in',
            profile=map_qos(),
        )
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
        '/fleet/scout_motion_ready_detail': topic(
            'std_msgs/msg/String',
            profile=qos(durability='transient_local', depth=1),
        ),
        f'/field/follower{int(follower_domain)}/risk_observation': topic(
            'std_msgs/msg/String',
            profile=qos(reliability='best_effort', durability='volatile', depth=5),
        ),
    }
    if include_follower_scan:
        follower_topics['/burger_scan_relay'] = topic(
            'sensor_msgs/msg/LaserScan',
            remap=f'/follower{int(follower_domain)}/scan',
            profile=qos('best_effort'),
        )
    if forward_map_to_main:
        follower_topics['/local_slam_map'] = topic(
            'nav_msgs/msg/OccupancyGrid',
            remap=f'/field/follower{int(follower_domain)}/map',
            profile=map_qos(),
        )
    if simulation:
        follower_topics['/cmd_vel'] = topic(
            'geometry_msgs/msg/TwistStamped',
            remap='/burger/cmd_vel',
        )

    main_config = {
        'name': f'{prefix}main_{main_domain}_to_follower_{follower_domain}',
        'from_domain': int(main_domain),
        'to_domain': int(follower_domain),
        'topics': main_topics,
    }
    follower_config = {
        'name': f'{prefix}follower_{follower_domain}_to_main_{main_domain}',
        'from_domain': int(follower_domain),
        'to_domain': int(main_domain),
        'topics': follower_topics,
    }
    relay_edges = [
        (
            (int(follower_domain), '/shared_map_in'),
            (int(follower_domain), '/map'),
            'follower_map_relay:/shared_map_in->/map',
        )
    ]
    validate_no_duplicate_bridge_routes([main_config, follower_config])
    validate_no_bridge_feedback_cycles(
        [main_config, follower_config],
        relay_edges=relay_edges,
    )

    main_to_follower = _write_runtime_config(
        f'{prefix}main_{main_domain}_to_follower_{follower_domain}_',
        main_config,
        output_directory,
    )
    follower_to_main = _write_runtime_config(
        f'{prefix}follower_{follower_domain}_to_main_{main_domain}_',
        follower_config,
        output_directory,
    )

    return main_to_follower, follower_to_main


def write_member_bridge_configs(
    main_domain: int,
    member_domain: int,
    *,
    forward_map_to_main: bool = False,
    forward_risk_to_main: bool = False,
    include_leader_map: bool = True,
    output_directory: Optional[Path] = None,
) -> Tuple[Path, Path]:
    """Create the two directional domain_bridge configurations for a plain
    fleet member: it never leads or follows, it reports its pose and accepts
    direct Nav2 goals on /member_goal_pose. Map and initial-pose flow mirror
    the follower's bridge so the same PC RViz can localize it.
    """
    main_topics = {
        **system_readiness_topics(),
        '/leader_pose': topic('geometry_msgs/msg/PoseStamped'),
        '/omx/target_detected': topic(
            'std_msgs/msg/Bool',
            profile=qos(reliability='best_effort', durability='volatile', depth=5),
        ),
        '/omx/camera_ready': topic(
            'std_msgs/msg/Bool',
            profile=qos(reliability='best_effort', durability='volatile', depth=5),
        ),
        '/omx/camera_yaw': topic(
            'std_msgs/msg/Float32',
            profile=qos(reliability='best_effort', durability='volatile', depth=5),
        ),
        '/omx/observation_status': topic(
            'std_msgs/msg/String',
            profile=qos(reliability='best_effort', durability='volatile', depth=5),
        ),
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
        '/fleet/video_ready': topic(
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
    if include_leader_map:
        main_topics['/map'] = topic(
            'nav_msgs/msg/OccupancyGrid',
            remap='/map_bridge',
            profile=map_qos(),
        )
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
        '/fleet/scout_motion_ready_detail': topic(
            'std_msgs/msg/String',
            profile=qos(durability='transient_local', depth=1),
        ),
        f'/field/scout{int(member_domain)}/risk_observation': topic(
            'std_msgs/msg/String',
            profile=qos(reliability='best_effort', durability='volatile', depth=5),
        ),
    }
    if forward_risk_to_main:
        member_topics.update(risk_topics())
    if forward_map_to_main:
        member_topics['/map'] = topic(
            'nav_msgs/msg/OccupancyGrid',
            remap=f'/field/scout{int(member_domain)}/map',
            profile=map_qos(),
        )
        member_topics['/rl_confidence_map'] = topic(
            'nav_msgs/msg/OccupancyGrid',
            remap='/scout22/rl_confidence_map',
            profile=qos(durability='transient_local', depth=1),
        )

    main_config = {
        'name': f'main_{main_domain}_to_member_{member_domain}',
        'from_domain': int(main_domain),
        'to_domain': int(member_domain),
        'topics': main_topics,
    }
    member_config = {
        'name': f'member_{member_domain}_to_main_{main_domain}',
        'from_domain': int(member_domain),
        'to_domain': int(main_domain),
        'topics': member_topics,
    }
    validate_no_duplicate_bridge_routes([main_config, member_config])

    main_to_member = _write_runtime_config(
        f'main_{main_domain}_to_member_{member_domain}_',
        main_config,
        output_directory,
    )
    member_to_main = _write_runtime_config(
        f'member_{member_domain}_to_main_{main_domain}_',
        member_config,
        output_directory,
    )

    return main_to_member, member_to_main


def write_risk_to_leader_bridge_config(
    risk_domain: int,
    leader_domain: int,
    *,
    include_map: bool = True,
    map_source_topic: str = '/map',
    include_identity_topics: bool = False,
    include_rl_confidence_map: bool = False,
    include_risk_outputs: bool = True,
    output_directory: Optional[Path] = None,
) -> Path:
    """Create the one-way scout/risk bridge into the leader domain.

    include_map=True is the target fleet mode: the scout/risk domain owns
    Cartographer and publishes the authoritative /map. The explicit
    leader-SLAM compatibility mode sets this false so domain 20's local
    Cartographer remains the only /map source in that mode.
    """
    all_risk_topics = risk_topics()
    topics = {
        '/risk/yolo_detections': all_risk_topics['/risk/yolo_detections'],
    }
    if include_identity_topics:
        topics['/member_pose'] = topic('geometry_msgs/msg/PoseStamped')
        topics['/scout/signal'] = topic(
            'std_msgs/msg/String',
            profile=qos(reliability='best_effort', durability='volatile', depth=5),
        )
    if include_risk_outputs:
        topics.update(
            {
                name: spec
                for name, spec in all_risk_topics.items()
                if name != '/risk/yolo_detections'
            }
        )
    if include_map:
        source = str(map_source_topic).strip() or '/map'
        topics[source] = topic(
            'nav_msgs/msg/OccupancyGrid',
            remap=f'/field/scout{int(risk_domain)}/map',
            profile=live_map_qos(),
        )
    if include_rl_confidence_map:
        topics['/rl_confidence_map'] = topic(
            'nav_msgs/msg/OccupancyGrid',
            remap='/scout22/rl_confidence_map',
            profile=qos(durability='transient_local', depth=1),
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


def write_field_robot_candidate_bridge_configs(
    main_domain: int,
    field_robots: Iterable[Dict],
    *,
    output_directory: Optional[Path] = None,
) -> Tuple[Path, ...]:
    """Create identity-namespaced bridges for every candidate field robot.

    This is the scalable bridge shape used by the leader when a registry is
    provided.  Legacy writers still exist elsewhere for compatibility, but
    these configs keep every candidate under /field/<robot_name>/... so role
    changes do not rename topics.
    """
    paths: list[Path] = []
    configs: list[Dict] = []
    main_topics = {
        **system_readiness_topics(),
        '/fleet/video_ready': topic(
            'std_msgs/msg/Bool',
            profile=qos(durability='transient_local', depth=1),
        ),
        '/fleet/field_robot_role_cmd': topic(
            'std_msgs/msg/String',
            profile=qos(durability='transient_local', depth=1),
        ),
        '/fleet/scout_role': topic(
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
        '/leader_pose': topic('geometry_msgs/msg/PoseStamped'),
        '/fleet/robot_poses': topic('geometry_msgs/msg/PoseArray', profile=qos(depth=5)),
        '/initialpose': topic('geometry_msgs/msg/PoseWithCovarianceStamped', profile=qos(depth=1)),
    }
    for item in field_robots:
        robot = str(item.get('robot_name', '')).strip()
        if not robot:
            continue
        domain = int(item['domain_id'])
        pose_source = str(item.get('pose_source', '')).strip()
        if not pose_source:
            initial_role = str(item.get('initial_role', '')).strip().upper()
            pose_source = '/burger_pose' if initial_role == 'FOLLOWER' else '/member_pose'
        to_robot = {
            'name': f'main_{main_domain}_to_field_{robot}_{domain}',
            'from_domain': int(main_domain),
            'to_domain': domain,
            'topics': main_topics,
        }
        role = str(item.get('initial_role', '')).strip().upper()
        include_map = bool(item.get('map_capable', True)) and bool(
            item.get('map_authority', role in ('ACTIVE_SCOUT', 'SCOUT', 'RECOVERING'))
        )
        from_robot = {
            'name': f'field_{robot}_{domain}_to_main_{main_domain}',
            'from_domain': domain,
            'to_domain': int(main_domain),
            'topics': field_robot_identity_topics(
                robot,
                pose_source=pose_source,
                include_map=include_map,
            ),
        }
        configs.extend([to_robot, from_robot])
    validate_no_duplicate_bridge_routes(configs)
    for config in configs:
        paths.append(_write_runtime_config(
            f'{config["name"]}_',
            config,
            output_directory,
        ))
    return tuple(paths)


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
            profile=map_qos(),
        ),
        **system_readiness_topics(),
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
        '/fleet/video_ready': topic(
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
