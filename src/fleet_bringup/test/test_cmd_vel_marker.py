import pytest
import rclpy
from geometry_msgs.msg import Twist, TwistStamped

from fleet_bringup.cmd_vel_marker import CmdVelMarker


def make_node() -> CmdVelMarker:
    if not rclpy.ok():
        rclpy.init()
    return CmdVelMarker()


def destroy_node(node: CmdVelMarker) -> None:
    node.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()


def _capture(node: CmdVelMarker) -> list:
    published = []
    node.pub.publish = lambda msg: published.append(msg)
    return published


def _by_ns(markers, ns):
    return next(m for m in markers.markers if m.ns == ns)


def _twist(linear_x=0.0, angular_z=0.0) -> Twist:
    msg = Twist()
    msg.linear.x = float(linear_x)
    msg.angular.z = float(angular_z)
    return msg


def _twist_stamped(linear_x=0.0, angular_z=0.0) -> TwistStamped:
    msg = TwistStamped()
    msg.twist = _twist(linear_x, angular_z)
    return msg


def test_moving_command_draws_curved_preview_and_green_text():
    node = make_node()
    try:
        published = _capture(node)
        node._now = lambda: 100.0
        node._on_velocity(_twist(0.2, 0.5))

        node._tick()

        arr = published[-1]
        text = _by_ns(arr, 'cmd_vel_text')
        assert 'STALE' not in text.text
        assert 'lin=+0.200' in text.text
        assert 'ang=+0.500' in text.text
        assert text.color.g > text.color.r  # green when actively moving

        path = _by_ns(arr, 'cmd_vel_preview')
        assert len(path.points) == node.preview_samples + 1
        # Unicycle rollout must actually curve (not a straight line) when
        # angular.z is nonzero.
        assert path.points[-1].y != 0.0
    finally:
        destroy_node(node)


def test_straight_line_when_angular_is_zero():
    node = make_node()
    try:
        published = _capture(node)
        node._now = lambda: 100.0
        node._on_velocity(_twist(0.3, 0.0))

        node._tick()

        path = _by_ns(published[-1], 'cmd_vel_preview')
        expected_x = 0.3 * node.preview_scale * node.preview_seconds
        assert path.points[-1].x == pytest.approx(expected_x)
        assert path.points[-1].y == 0.0
    finally:
        destroy_node(node)


def test_below_deadband_is_not_treated_as_moving():
    node = make_node()
    try:
        published = _capture(node)
        node._now = lambda: 100.0
        node._on_velocity(_twist(0.001, 0.001))

        node._tick()

        arr = published[-1]
        text = _by_ns(arr, 'cmd_vel_text')
        assert 'STALE' not in text.text
        assert text.color.g <= text.color.r  # not the "moving" green
        path = _by_ns(arr, 'cmd_vel_preview')
        assert len(path.points) == 1
    finally:
        destroy_node(node)


def test_no_message_ever_reports_stale():
    node = make_node()
    try:
        published = _capture(node)
        node._now = lambda: 100.0

        node._tick()

        text = _by_ns(published[-1], 'cmd_vel_text')
        assert text.text.startswith('cmd_vel STALE')
    finally:
        destroy_node(node)


def test_old_message_goes_stale_after_timeout():
    node = make_node()
    try:
        published = _capture(node)
        clock = [100.0]
        node._now = lambda: clock[0]
        node._on_velocity(_twist(0.2, 0.0))
        clock[0] += node.stale_timeout_sec + 0.1

        node._tick()

        text = _by_ns(published[-1], 'cmd_vel_text')
        assert text.text.startswith('cmd_vel STALE')
    finally:
        destroy_node(node)


def test_stamped_and_unstamped_inputs_produce_the_same_state():
    node = make_node()
    try:
        node._now = lambda: 50.0
        node._on_twist_stamped(_twist_stamped(0.4, -0.2))
        assert node.last_linear_x == pytest.approx(0.4)
        assert node.last_angular_z == pytest.approx(-0.2)

        node._on_twist(_twist(0.1, 0.9))
        assert node.last_linear_x == pytest.approx(0.1)
        assert node.last_angular_z == pytest.approx(0.9)
    finally:
        destroy_node(node)
