import pytest

from launch import LaunchContext

from fleet_bringup import launch_utils


def _shell(monkeypatch, domain_id='22', cyclonedds_uri=None):
    monkeypatch.setenv('ROS_DOMAIN_ID', domain_id)
    monkeypatch.setenv('RMW_IMPLEMENTATION', 'rmw_cyclonedds_cpp')
    monkeypatch.setenv('FLEET_VALIDATE_CYCLONEDDS_BUFFERS', '0')
    if cyclonedds_uri is None:
        monkeypatch.delenv('CYCLONEDDS_URI', raising=False)
    else:
        monkeypatch.setenv('CYCLONEDDS_URI', cyclonedds_uri)


def test_clean_process_environment_preserves_shell_dds_values(monkeypatch):
    uri = 'file:///operator/cyclonedds.xml'
    _shell(monkeypatch, cyclonedds_uri=uri)

    env = launch_utils.clean_process_environment('22')

    assert env['ROS_DOMAIN_ID'] == '22'
    assert env['RMW_IMPLEMENTATION'] == 'rmw_cyclonedds_cpp'
    assert env['CYCLONEDDS_URI'] == uri


def test_dds_launch_environment_validates_without_set_environment_actions(monkeypatch):
    _shell(monkeypatch, cyclonedds_uri='file:///operator/cyclonedds.xml')
    action = launch_utils.dds_launch_environment('22')[0]
    context = LaunchContext()
    before = {
        'ROS_DOMAIN_ID': context.environment.get('ROS_DOMAIN_ID'),
        'CYCLONEDDS_URI': context.environment.get('CYCLONEDDS_URI'),
    }

    assert action.execute(context) == []
    assert context.environment.get('ROS_DOMAIN_ID') == before['ROS_DOMAIN_ID']
    assert context.environment.get('CYCLONEDDS_URI') == before['CYCLONEDDS_URI']


def test_dds_launch_environment_rejects_shell_domain_mismatch(monkeypatch):
    _shell(monkeypatch, domain_id='21')

    with pytest.raises(RuntimeError, match='does not match shell ROS_DOMAIN_ID'):
        launch_utils.dds_launch_environment('22')[0].execute(LaunchContext())


def test_dds_launch_environment_rejects_missing_shell_domain(monkeypatch):
    monkeypatch.delenv('ROS_DOMAIN_ID', raising=False)

    with pytest.raises(RuntimeError, match='Missing required shell environment'):
        launch_utils.dds_launch_environment(None)[0].execute(LaunchContext())
