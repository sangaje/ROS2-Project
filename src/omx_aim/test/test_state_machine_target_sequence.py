from types import SimpleNamespace

from omx.state_machine import StateMachine
from omx.types import State


def _cfg(*, armed=True, lost_timeout_sec=3.0):
    return SimpleNamespace(
        autotrack=SimpleNamespace(default_armed=armed),
        fire=SimpleNamespace(
            lost_timeout_sec=lost_timeout_sec,
            cooldown_sec=3.0,
            confirm_deadband_scale=1.0,
            hold_time_sec=0.5,
        ),
        ibvs=SimpleNamespace(deadband_x=0.03, deadband_y=0.09),
        patrol=None,
    )


def test_armed_idle_scans_when_no_target_or_queue():
    sm = StateMachine(_cfg(armed=True))

    action = sm.update(False, None, 10.0, vision_valid=True)

    assert sm.state == State.IDLE
    assert action['action'] == 'scan_sweep'
    assert action['scan_sweep'] is True


def test_detection_preempts_navigation_before_tracking():
    sm = StateMachine(_cfg(armed=True))
    sm.state = State.WAITING_NAV

    action = sm.update(True, (0.25, -0.10), 10.0, vision_valid=True)

    assert sm.state == State.TRACKING
    assert action['action'] == 'track'
    assert action['error'] == (0.25, -0.10)
    assert action['cancel_navigation'] is True
    assert action['cancel_reason'] == 'target_detected_track'


def test_detection_auto_arms_and_enters_tracking_when_disarmed():
    sm = StateMachine(_cfg(armed=False))

    action = sm.update(True, (0.18, 0.12), 10.0, vision_valid=True)

    assert sm.armed is True
    assert sm.state == State.TRACKING
    assert action['auto_armed'] is True
    assert action['action'] == 'track'
    assert action['error'] == (0.18, 0.12)


def test_tracking_holds_after_target_disappears_until_timeout():
    sm = StateMachine(_cfg(armed=True, lost_timeout_sec=3.0))
    sm.state = State.TRACKING

    action = sm.update(True, (0.20, -0.10), 9.5, vision_valid=True)
    assert action['action'] == 'track'

    action = sm.update(False, None, 10.0, vision_valid=True)
    assert sm.state == State.TRACKING
    assert action['action'] == 'track'
    assert action['error'] == (0.20, -0.10)
    assert action['stale_track'] is True

    action = sm.update(False, None, 12.9, vision_valid=True)
    assert sm.state == State.TRACKING
    assert action['action'] == 'track'
    assert action['error'] == (0.20, -0.10)
    assert action['stale_track'] is True

    action = sm.update(False, None, 13.0, vision_valid=True)
    assert sm.state == State.IDLE
    assert action['action'] == 'target_lost'


def test_confirming_uses_last_error_for_stale_pd_when_target_disappears():
    sm = StateMachine(_cfg(armed=True, lost_timeout_sec=3.0))
    sm.state = State.CONFIRMING
    sm.last_track_error = (0.08, -0.04)

    action = sm.update(False, None, 10.0, vision_valid=True)

    assert sm.state == State.TRACKING
    assert action['action'] == 'track'
    assert action['error'] == (0.08, -0.04)
    assert action['stale_track'] is True


def test_confirming_immediately_resumes_pd_when_target_leaves_deadband():
    sm = StateMachine(_cfg(armed=True))
    sm.state = State.CONFIRMING

    action = sm.update(True, (0.20, 0.02), 10.0, vision_valid=True)

    assert sm.state == State.TRACKING
    assert action['action'] == 'track'
    assert action['error'] == (0.20, 0.02)
