import pytest
from geometry_msgs.msg import Twist

from fleet_bringup.tracked_cmd_vel_adapter import (
    TrackedCmdVelAdapter,
    _slew,
    _validate_scale,
)


def _adapter() -> TrackedCmdVelAdapter:
    node = TrackedCmdVelAdapter.__new__(TrackedCmdVelAdapter)
    node.enabled = True
    node.linear_gain = 1.0
    node.angular_gain = 1.0
    node.left_wheel_command_scale = 0.5
    node.right_wheel_command_scale = 0.5
    node.linear_command_scale = 0.5
    node.angular_command_scale = 0.5
    node.effective_wheel_radius = 0.040
    node.effective_track_separation = 0.447
    node.max_linear_velocity = 10.0
    node.max_angular_velocity = 10.0
    node._last_reject_reason = ''
    return node


def _twist(v: float, w: float) -> Twist:
    msg = Twist()
    msg.linear.x = v
    msg.angular.z = w
    return msg


def test_common_scale_halves_virtual_wheel_targets_for_straight_command():
    node = _adapter()

    evaluation = node._evaluate_twist(_twist(0.28, 0.0))

    assert evaluation is not None
    assert evaluation.raw_wheel.left == pytest.approx(7.0)
    assert evaluation.raw_wheel.right == pytest.approx(7.0)
    assert evaluation.output_wheel.left == pytest.approx(3.5)
    assert evaluation.output_wheel.right == pytest.approx(3.5)
    assert evaluation.target.linear_x == pytest.approx(0.14)
    assert evaluation.target.angular_z == pytest.approx(0.0)


def test_common_scale_halves_virtual_wheel_targets_for_in_place_rotation():
    node = _adapter()

    evaluation = node._evaluate_twist(_twist(0.0, 0.70))

    assert evaluation is not None
    assert evaluation.output_wheel.left == pytest.approx(0.5 * evaluation.raw_wheel.left)
    assert evaluation.output_wheel.right == pytest.approx(0.5 * evaluation.raw_wheel.right)
    assert evaluation.target.linear_x == pytest.approx(0.0)
    assert evaluation.target.angular_z == pytest.approx(0.35)


def test_zero_command_stays_exactly_zero():
    node = _adapter()

    evaluation = node._evaluate_twist(_twist(0.0, 0.0))

    assert evaluation is not None
    assert evaluation.target.linear_x == 0.0
    assert evaluation.target.angular_z == 0.0
    assert evaluation.output_wheel.left == 0.0
    assert evaluation.output_wheel.right == 0.0


def test_non_finite_command_is_rejected():
    node = _adapter()

    evaluation = node._evaluate_twist(_twist(float('nan'), 0.0))

    assert evaluation is None
    assert node._last_reject_reason == 'non_finite_twist'


def test_scale_validation_rejects_unsafe_values():
    with pytest.raises(RuntimeError):
        _validate_scale('left_wheel_command_scale', 0.0)
    with pytest.raises(RuntimeError):
        _validate_scale('right_wheel_command_scale', 1.2)


def test_deceleration_limit_is_separate_from_acceleration_limit():
    assert _slew(0.20, 0.0, accel_limit=0.10, decel_limit=0.25, dt=0.4) == pytest.approx(0.10)
    assert _slew(0.0, 0.20, accel_limit=0.10, decel_limit=0.25, dt=0.4) == pytest.approx(0.04)


def test_left_right_scale_params_do_not_fake_independent_wheel_control():
    node = _adapter()
    node.left_wheel_command_scale = 0.48
    node.right_wheel_command_scale = 0.52

    evaluation = node._evaluate_twist(_twist(0.28, 0.0))

    assert evaluation is not None
    assert evaluation.output_wheel.left == pytest.approx(0.5 * evaluation.raw_wheel.left)
    assert evaluation.output_wheel.right == pytest.approx(0.5 * evaluation.raw_wheel.right)
