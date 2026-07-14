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


def test_clean_process_environment_preserves_unreadable_shell_dds_values(monkeypatch):
    uri = 'file:///operator/cyclonedds.xml'
    _shell(monkeypatch, cyclonedds_uri=uri)

    env = launch_utils.clean_process_environment('22')

    assert env['ROS_DOMAIN_ID'] == '22'
    assert env['RMW_IMPLEMENTATION'] == 'rmw_cyclonedds_cpp'
    assert env['CYCLONEDDS_URI'] == uri


def test_clean_process_environment_preserves_small_cyclone_participant_range(monkeypatch, tmp_path):
    source = tmp_path / 'cyclonedds.xml'
    source.write_text(
        '''<?xml version="1.0"?>
<CycloneDDS xmlns="https://cdds.io/config">
  <Domain id="any"><Discovery>
    <ParticipantIndex>auto</ParticipantIndex>
    <MaxAutoParticipantIndex>30</MaxAutoParticipantIndex>
    <Peers><Peer address="10.0.0.2"/></Peers>
  </Discovery></Domain>
</CycloneDDS>''',
        encoding='utf-8',
    )
    _shell(monkeypatch, cyclonedds_uri=f'file://{source}')

    env = launch_utils.clean_process_environment('22')

    assert env['CYCLONEDDS_URI'] == f'file://{source}'
    assert source.read_text(encoding='utf-8').count('>30<') == 1


def test_dds_launch_environment_does_not_rewrite_cyclonedds_uri(monkeypatch, tmp_path):
    source = tmp_path / 'cyclonedds.xml'
    source.write_text(
        '''<CycloneDDS xmlns="https://cdds.io/config"><Domain id="any">
<Discovery><ParticipantIndex>auto</ParticipantIndex>
<MaxAutoParticipantIndex>30</MaxAutoParticipantIndex></Discovery>
</Domain></CycloneDDS>''',
        encoding='utf-8',
    )
    _shell(monkeypatch, cyclonedds_uri=f'file://{source}')
    action = launch_utils.dds_launch_environment('22')[0]
    context = LaunchContext()

    actions = action.execute(context)

    assert actions == []


def test_dds_launch_environment_rejects_shell_domain_mismatch(monkeypatch):
    _shell(monkeypatch, domain_id='21')

    with pytest.raises(RuntimeError, match='does not match shell ROS_DOMAIN_ID'):
        launch_utils.dds_launch_environment('22')[0].execute(LaunchContext())


def test_dds_launch_environment_rejects_missing_shell_domain(monkeypatch):
    monkeypatch.delenv('ROS_DOMAIN_ID', raising=False)

    with pytest.raises(RuntimeError, match='Missing required shell environment'):
        launch_utils.dds_launch_environment(None)[0].execute(LaunchContext())
