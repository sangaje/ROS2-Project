from types import SimpleNamespace

from omx.state_machine import StateMachine
from omx.types import State


def _cfg(*, armed=True, lost_timeout_sec=3.0):
    return SimpleNamespace(
        autotrack=SimpleNamespace(default_armed=armed),
        fire=SimpleNamespace(lost_timeout_sec=lost_timeout_sec, cooldown_sec=3.0),
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


def test_tracking_holds_after_target_disappears_until_timeout():
    sm = StateMachine(_cfg(armed=True, lost_timeout_sec=3.0))
    sm.state = State.TRACKING

    action = sm.update(False, None, 10.0, vision_valid=True)
    assert sm.state == State.TRACKING
    assert action['action'] == 'wait'

    action = sm.update(False, None, 12.9, vision_valid=True)
    assert sm.state == State.TRACKING
    assert action['action'] == 'wait'

    action = sm.update(False, None, 13.0, vision_valid=True)
    assert sm.state == State.IDLE
    assert action['action'] == 'target_lost'
