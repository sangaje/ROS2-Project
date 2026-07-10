import pytest

from launch import LaunchContext, LaunchDescription, LaunchService
from launch.actions import OpaqueFunction, SetEnvironmentVariable

from fleet_bringup import launch_utils


def _cyclone_shell(monkeypatch, domain_id='22', cyclonedds_uri=None):
    monkeypatch.setenv('ROS_DOMAIN_ID', domain_id)
    monkeypatch.setenv('RMW_IMPLEMENTATION', 'rmw_cyclonedds_cpp')
    monkeypatch.setenv('FLEET_VALIDATE_CYCLONEDDS_BUFFERS', '0')
    if cyclonedds_uri is None:
        monkeypatch.delenv('CYCLONEDDS_URI', raising=False)
    else:
        monkeypatch.setenv('CYCLONEDDS_URI', cyclonedds_uri)


def test_prepare_process_environment_uses_packaged_uri_when_absent(monkeypatch):
    _cyclone_shell(monkeypatch)
    packaged_uri = 'file:///tmp/fleet-cyclonedds.xml'
    monkeypatch.setattr(
        launch_utils,
        '_packaged_cyclonedds_uri',
        lambda: packaged_uri,
    )

    env = launch_utils._prepare_process_environment('22')

    assert env['ROS_DOMAIN_ID'] == '22'
    assert env['CYCLONEDDS_URI'] == packaged_uri


def test_prepare_process_environment_preserves_explicit_uri(monkeypatch):
    explicit_uri = 'file:///tmp/operator-cyclonedds.xml'
    _cyclone_shell(monkeypatch, cyclonedds_uri=explicit_uri)
    monkeypatch.setattr(
        launch_utils,
        '_packaged_cyclonedds_uri',
        lambda: pytest.fail('packaged URI must not replace an explicit URI'),
    )

    env = launch_utils._prepare_process_environment('22')

    assert env['ROS_DOMAIN_ID'] == '22'
    assert env['CYCLONEDDS_URI'] == explicit_uri


def test_dds_launch_environment_applies_values_to_subsequent_actions(monkeypatch):
    _cyclone_shell(monkeypatch)
    packaged_uri = 'file:///tmp/fleet-cyclonedds.xml'
    monkeypatch.setattr(
        launch_utils,
        '_packaged_cyclonedds_uri',
        lambda: packaged_uri,
    )
    observed = {}

    def observe_environment(context):
        observed['ROS_DOMAIN_ID'] = context.environment.get('ROS_DOMAIN_ID')
        observed['CYCLONEDDS_URI'] = context.environment.get('CYCLONEDDS_URI')
        return []

    service = LaunchService()
    service.include_launch_description(LaunchDescription([
        *launch_utils.dds_launch_environment('22'),
        OpaqueFunction(function=observe_environment),
    ]))

    assert service.run() == 0
    assert observed == {
        'ROS_DOMAIN_ID': '22',
        'CYCLONEDDS_URI': packaged_uri,
    }


def test_dds_launch_environment_preserves_context_uri(monkeypatch):
    _cyclone_shell(monkeypatch)
    explicit_uri = 'file:///tmp/context-cyclonedds.xml'
    action = launch_utils.dds_launch_environment('22')[0]
    context = LaunchContext()
    context.environment['CYCLONEDDS_URI'] = explicit_uri

    returned_actions = action.execute(context)

    assert all(isinstance(item, SetEnvironmentVariable) for item in returned_actions)
    for returned_action in returned_actions:
        returned_action.execute(context)
    assert context.environment['ROS_DOMAIN_ID'] == '22'
    assert context.environment['CYCLONEDDS_URI'] == explicit_uri


def test_dds_launch_environment_rejects_shell_domain_mismatch(monkeypatch):
    _cyclone_shell(monkeypatch, domain_id='21')

    action = launch_utils.dds_launch_environment('22')[0]

    with pytest.raises(RuntimeError, match='does not match shell ROS_DOMAIN_ID'):
        action.execute(LaunchContext())


def test_dds_launch_environment_rejects_missing_shell_domain(monkeypatch):
    monkeypatch.delenv('ROS_DOMAIN_ID', raising=False)
    action = launch_utils.dds_launch_environment(None)[0]

    with pytest.raises(RuntimeError, match='Missing required shell environment'):
        action.execute(LaunchContext())


def test_dds_launch_environment_none_uses_current_domain(monkeypatch):
    _cyclone_shell(monkeypatch, domain_id='22')
    action = launch_utils.dds_launch_environment(None)[0]
    context = LaunchContext()

    returned_actions = action.execute(context)
    for returned_action in returned_actions:
        returned_action.execute(context)

    assert context.environment['ROS_DOMAIN_ID'] == '22'
