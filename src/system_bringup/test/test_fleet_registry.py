from system_bringup.fleet_registry import build_legacy_registry, normalize_registry


def test_follower_can_be_map_capable_without_map_authority():
    registry = normalize_registry({
        'field_robots': [
            {
                'robot_name': 'scout22',
                'domain_id': 22,
                'initial_role': 'ACTIVE_SCOUT',
                'map_capable': True,
            },
            {
                'robot_name': 'follower21',
                'domain_id': 21,
                'initial_role': 'FOLLOWER',
                'map_capable': True,
            },
        ],
    })

    by_name = {robot.robot_name: robot for robot in registry}
    assert by_name['scout22'].map_authority is True
    assert by_name['follower21'].map_capable is True
    assert by_name['follower21'].map_authority is False


def test_legacy_registry_makes_only_active_scout_map_authoritative():
    registry = build_legacy_registry(
        active_scout_robot_name='scout22',
        risk_domain_id='22',
        follower_robot_name='follower21',
        follower_domain_id='21',
    )

    by_name = {robot.robot_name: robot for robot in registry}
    assert by_name['scout22'].map_authority is True
    assert by_name['follower21'].map_authority is False
