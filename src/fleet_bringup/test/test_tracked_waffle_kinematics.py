from pathlib import Path

import pytest
import yaml

from fleet_bringup.tracked_waffle_calibration import (
    Sample,
    calculate_radius,
    calculate_separation,
)


CONFIG = Path(__file__).parents[1] / 'config' / 'tracked_waffle_kinematics.yaml'


def test_tracked_waffle_yaml_contains_effective_kinematics():
    data = yaml.safe_load(CONFIG.read_text(encoding='utf-8'))
    meta = data['tracked_waffle_kinematics']['ros__parameters']
    tb3 = data['/**']['turtlebot3_node']['ros__parameters']
    odom = data['/**']['diff_drive_controller']['ros__parameters']['odometry']

    assert meta['effective_wheel_radius'] == pytest.approx(0.040)
    assert meta['effective_track_separation'] == pytest.approx(0.447)
    assert tb3['wheels']['radius'] == pytest.approx(meta['effective_wheel_radius'])
    assert tb3['wheels']['separation'] == pytest.approx(
        meta['effective_track_separation']
    )
    assert odom['use_imu'] is True


def test_tracked_adapter_defaults_to_separate_nav_and_hardware_topics():
    data = yaml.safe_load(CONFIG.read_text(encoding='utf-8'))
    adapter = data['tracked_cmd_vel_adapter']['ros__parameters']

    assert adapter['enabled'] is True
    assert adapter['input_topic'] == '/cmd_vel_nav'
    assert adapter['output_topic'] == '/cmd_vel'
    assert adapter['input_topic'] != adapter['output_topic']
    assert adapter['linear_gain'] == pytest.approx(0.825)
    assert adapter['angular_gain'] == pytest.approx(1.286)


def test_straight_calibration_uses_measured_over_odom_ratio():
    value, candidates = calculate_radius(
        0.040,
        [Sample(odom=1.0, measured=1.08), Sample(odom=2.0, measured=2.10)],
    )

    assert candidates[0] == pytest.approx(0.0432)
    assert value == pytest.approx((0.0432 + 0.0420) / 2.0)


def test_rotation_calibration_uses_odom_over_measured_ratio():
    value, candidates = calculate_separation(
        0.447,
        [Sample(odom=1800.0, measured=1710.0), Sample(odom=1800.0, measured=1720.0)],
    )

    assert candidates[0] == pytest.approx(0.4705263158)
    assert value == pytest.approx((candidates[0] + candidates[1]) / 2.0)
