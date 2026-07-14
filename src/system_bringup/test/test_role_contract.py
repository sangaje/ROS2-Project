from system_bringup.role_contract import Role, parse_epoch, parse_role_message


def test_role_message_preserves_json_fields_and_legacy_string_compatibility():
    message = parse_role_message(
        '{"role":"ACTIVE_SCOUT","robot":"follower21","epoch":2,'
        '"active_scout_id":"follower21","localization_ready":true,'
        '"recovery_complete":true}',
        'scout22',
    )

    assert message is not None
    assert message.role == Role.ACTIVE_SCOUT
    assert message.robot == 'follower21'
    assert message.epoch == 2
    assert message.active_scout_id == 'follower21'
    assert message.localization_ready is True
    assert message.recovery_complete is True

    legacy = parse_role_message('RECOVERY_NAVIGATING', 'follower21')
    assert legacy is not None
    assert legacy.role == Role.RECOVERY_NAVIGATING
    assert legacy.robot == 'follower21'


def test_role_message_rejects_invalid_json_and_epoch_coercion():
    assert parse_role_message('{bad json', 'scout22') is None
    assert parse_epoch(True) is None
    assert parse_epoch(1.0) is None
    assert parse_epoch('2') == 2
