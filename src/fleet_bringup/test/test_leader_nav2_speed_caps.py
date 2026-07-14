from pathlib import Path

import pytest
import yaml


CONFIG_DIR = Path(__file__).parents[1] / 'config'


def _follow_path_params(name: str) -> dict:
    data = yaml.safe_load((CONFIG_DIR / name).read_text(encoding='utf-8'))
    return data['controller_server']['ros__parameters']['FollowPath']


@pytest.mark.parametrize(
    ('filename', 'max_linear', 'max_angular', 'acc_linear', 'decel_linear', 'acc_angular', 'decel_angular'),
    [
        ('leader_nav2.yaml', 0.20, 0.80, 0.80, -0.90, 1.60, -1.80),
        ('leader_waffle_pi_nav2.yaml', 0.20, 0.75, 0.70, -0.90, 1.40, -1.60),
    ],
)
def test_leader_nav2_caps_stay_conservative_for_stable_shadow_motion(
    filename,
    max_linear,
    max_angular,
    acc_linear,
    decel_linear,
    acc_angular,
    decel_angular,
):
    follow = _follow_path_params(filename)

    assert follow['min_vel_x'] == pytest.approx(-0.08)
    assert follow['max_vel_x'] == pytest.approx(max_linear)
    assert follow['max_speed_xy'] == pytest.approx(max_linear)
    assert follow['max_vel_theta'] == pytest.approx(max_angular)
    assert follow['acc_lim_x'] == pytest.approx(acc_linear)
    assert follow['decel_lim_x'] == pytest.approx(decel_linear)
    assert follow['acc_lim_theta'] == pytest.approx(acc_angular)
    assert follow['decel_lim_theta'] == pytest.approx(decel_angular)
