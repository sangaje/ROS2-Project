"""Behavioral regression test for the camera/upload/publish role gate.

Every other test in this package checks role-gating by grepping source text
for parameter names. None of them actually drove on_role() with a FOLLOWER
message and checked what happened -- which is exactly how the real bug
shipped: FOLLOWER sat in both standby_roles and publish_roles, so the union
gate (`role in active_roles or standby_roles or publish_roles`) left the
camera open and uploading for a follower robot at the same time as the real
active scout.
"""

import threading

try:
    from std_msgs.msg import String
except ModuleNotFoundError:
    class String:
        def __init__(self, data=''):
            self.data = data

from flask_yolo_bridge.opencv_camera_to_flask_yolo import OpenCVCameraToFlaskYolo


class _Logger:
    def warning(self, message):
        pass

    def warn(self, message, **kwargs):
        pass


def _bare_node(initial_role='ACTIVE_SCOUT'):
    node = OpenCVCameraToFlaskYolo.__new__(OpenCVCameraToFlaskYolo)
    node.current_role = initial_role
    node.current_role_epoch = 0
    node.active_roles = {'ACTIVE_SCOUT', 'SCOUT', 'RECOVERING'}
    node.standby_roles = {'FOLLOWER', 'IDLE', 'TAKEOVER_PENDING'}
    node.publish_roles = {'ACTIVE_SCOUT', 'SCOUT', 'RECOVERING'}
    active = initial_role in node.active_roles
    node.role_topic_camera_enabled = active
    node.role_topic_publish_enabled = initial_role in node.publish_roles
    node.active_scout_id_enabled = None
    node.camera_process_enabled = active
    node.camera_upload_enabled = active
    node.risk_observation_publish_enabled = initial_role in node.publish_roles
    node.camera_active = active
    node.cap = None
    node.active_device = ''
    node.read_fail_streak = 0
    node.next_open_attempt_mono_sec = 0.0
    node.latest_frame = None
    node.latest_seq = 0
    node.sent_seq = 0
    node.frame_condition = threading.Condition()
    node.robot_name = 'scout22'
    node.get_logger = lambda: _Logger()
    return node


def test_follower_role_disables_camera_upload_and_publish_all_at_once():
    node = _bare_node(initial_role='ACTIVE_SCOUT')
    assert node.camera_process_enabled is True
    assert node.camera_upload_enabled is True
    assert node.risk_observation_publish_enabled is True

    node.on_role(String(data='FOLLOWER'))

    assert node.camera_process_enabled is False
    assert node.camera_upload_enabled is False
    assert node.risk_observation_publish_enabled is False


def test_takeover_to_active_scout_re_enables_all_three_gates():
    node = _bare_node(initial_role='FOLLOWER')
    assert node.camera_process_enabled is False
    assert node.camera_upload_enabled is False
    assert node.risk_observation_publish_enabled is False

    node.on_role(String(data='ACTIVE_SCOUT'))

    assert node.camera_process_enabled is True
    assert node.camera_upload_enabled is True
    assert node.risk_observation_publish_enabled is True


def test_active_scout_id_handoff_enables_follower_camera_and_survives_stale_role():
    node = _bare_node(initial_role='FOLLOWER')
    node.robot_name = 'follower21'

    node.on_active_scout_id(String(data='follower21'))

    assert node.current_role == 'ACTIVE_SCOUT'
    assert node.camera_process_enabled is True
    assert node.camera_upload_enabled is True
    assert node.risk_observation_publish_enabled is True

    node.on_role(String(data='FOLLOWER'))

    assert node.camera_process_enabled is True
    assert node.camera_upload_enabled is True
    assert node.risk_observation_publish_enabled is True


def test_idle_and_takeover_pending_also_get_no_camera_capability():
    for role in ('IDLE', 'TAKEOVER_PENDING', 'RECOVERY_NAVIGATING'):
        node = _bare_node(initial_role='ACTIVE_SCOUT')
        node.on_role(String(data=role))
        assert node.camera_process_enabled is False, role
        assert node.camera_upload_enabled is False, role
        assert node.risk_observation_publish_enabled is False, role
