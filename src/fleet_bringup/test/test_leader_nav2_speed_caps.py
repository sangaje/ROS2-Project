from pathlib import Path

import pytest
import yaml


CONFIG_DIR = Path(__file__).parents[1] / 'config'


def _follow_path_params(name: str) -> dict:
    data = yaml.safe_load((CONFIG_DIR / name).read_text(encoding='utf-8'))
    return data['controller_server']['ros__parameters']['FollowPath']


@pytest.mark.parametrize(
    'filename',
    ['leader_nav2.yaml', 'leader_waffle_pi_nav2.yaml'],
)
def test_leader_nav2_caps_allow_faster_safe_shadow_motion(filename):
    follow = _follow_path_params(filename)

    assert follow['min_vel_x'] == pytest.approx(-0.08)
    assert follow['max_vel_x'] == pytest.approx(0.26)
    assert follow['max_speed_xy'] == pytest.approx(0.26)
    assert follow['max_vel_theta'] == pytest.approx(1.00)
    assert follow['acc_lim_x'] == pytest.approx(1.20)
    assert follow['decel_lim_x'] == pytest.approx(-1.20)
    assert follow['acc_lim_theta'] == pytest.approx(2.20)
    assert follow['decel_lim_theta'] == pytest.approx(-2.20)
