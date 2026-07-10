import importlib.util
from pathlib import Path


def _load_module():
    launch_path = (
        Path(__file__).resolve().parents[1] / 'launch' / 'pc_remote_rl.launch.py'
    )
    spec = importlib.util.spec_from_file_location('pc_remote_rl_launch', launch_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cyclonedds_peer_config_disables_multicast_and_targets_scout_host():
    module = _load_module()

    path = module._write_cyclonedds_peer_config('pi2.taile3321c.ts.net')

    text = path.read_text()
    assert '<AllowMulticast>false</AllowMulticast>' in text
    assert '<Peer address="pi2.taile3321c.ts.net"/>' in text


def test_cyclonedds_peer_config_is_rewritten_for_a_different_host():
    module = _load_module()
    module._write_cyclonedds_peer_config('scout22.example.ts.net')

    path = module._write_cyclonedds_peer_config('follower21.example.ts.net')

    text = path.read_text()
    assert 'scout22.example.ts.net' not in text
    assert 'follower21.example.ts.net' in text
