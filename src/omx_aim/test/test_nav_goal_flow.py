from pathlib import Path

from geometry_msgs.msg import PoseStamped

from omx_aim.waffle_node import WaffleNavNode, WaffleState


def _pose(frame='map', finite=True, valid_quaternion=True):
    msg = PoseStamped()
    msg.header.frame_id = frame
    msg.pose.position.x = 1.0 if finite else float('nan')
    msg.pose.position.y = 2.0
    msg.pose.orientation.w = 1.0 if valid_quaternion else 0.0
    return msg


def test_waffle_goal_validation_rejects_bad_frame_and_pose():
    assert WaffleNavNode._validate_goal(_pose())[0] is True
    assert WaffleNavNode._validate_goal(_pose(frame='odom')) == (
        False,
        'unsupported_frame_odom',
    )
    assert WaffleNavNode._validate_goal(_pose(finite=False)) == (
        False,
        'non_finite_pose',
    )
    assert WaffleNavNode._validate_goal(_pose(valid_quaternion=False)) == (
        False,
        'invalid_quaternion',
    )


def test_waffle_diagnostics_exposes_waiting_localization_state():
    node = WaffleNavNode.__new__(WaffleNavNode)
    node.state = WaffleState.WAITING_LOCALIZATION
    node._current_goal_id = 42
    node._goal_epoch = 7
    node.dry_run = False
    node.action_client = type('ActionClient', (), {'server_is_ready': lambda self: True})()
    node._localization_ready = lambda: False
    node._amcl_ready = True
    node._pending_goal = object()
    node._goal_accepted = False
    node._cancel_requested_before_accept = False
    node._last_error = 'localization_not_ready'

    payload = node._diagnostics_payload()

    assert payload['state'] == 'WAITING_LOCALIZATION'
    assert payload['goal_id'] == 42
    assert payload['goal_pending'] is True
    assert payload['localization_ready'] is False


def test_yolo_retry_suppresses_all_busy_waffle_states_in_source():
    source = Path(__file__).parents[1] / 'omx_aim' / 'yolo_node.py'
    text = source.read_text(encoding='utf-8')

    assert 'def _parse_waffle_state' in text
    assert 'json.loads(text)' in text
    for state in (
        'GOAL_RECEIVED',
        'WAITING_SERVER',
        'WAITING_LOCALIZATION',
        'SENDING_GOAL',
        'WAITING_ACCEPT',
        'NAVIGATING',
        'CANCELING',
    ):
        assert repr(state) in text


def test_yolo_node_no_longer_publishes_direct_base_cmd_vel():
    source = Path(__file__).parents[1] / 'omx_aim' / 'yolo_node.py'
    text = source.read_text(encoding='utf-8')

    assert "create_publisher(\\n            TwistStamped, '/cmd_vel'" not in text
    assert "'/cmd_vel_nav'" not in text
    assert 'publish_detection_stop_cmd' not in text
