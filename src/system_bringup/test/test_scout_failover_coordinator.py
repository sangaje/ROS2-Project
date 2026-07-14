import json
import math

from geometry_msgs.msg import PoseStamped
import pytest
from rclpy.qos import DurabilityPolicy, HistoryPolicy, ReliabilityPolicy
from std_msgs.msg import Bool, String

from system_bringup.scout_failover_coordinator import (
    FailoverState,
    heartbeat_qos_profile,
    is_finite_map_pose,
    parse_epoch,
    ScoutFailoverCoordinator,
)


class _Logger:
    def __init__(self):
        self.messages = []

    def _record(self, level, message, *args, **kwargs):
        self.messages.append((level, str(message)))

    def info(self, message, *args, **kwargs):
        self._record('info', message, *args, **kwargs)

    def warning(self, message, *args, **kwargs):
        self._record('warning', message, *args, **kwargs)

    def error(self, message, *args, **kwargs):
        self._record('error', message, *args, **kwargs)


class _Publisher:
    def __init__(self):
        self.messages = []

    def publish(self, message):
        self.messages.append(message)


def _pose(x=0.0, y=0.0, yaw=0.0, frame='map') -> PoseStamped:
    msg = PoseStamped()
    msg.header.frame_id = frame
    msg.pose.position.x = float(x)
    msg.pose.position.y = float(y)
    msg.pose.orientation.z = math.sin(0.5 * float(yaw))
    msg.pose.orientation.w = math.cos(0.5 * float(yaw))
    return msg


def _string(data) -> String:
    msg = String()
    msg.data = data if isinstance(data, str) else json.dumps(data)
    return msg


def _bare_node(now=0.0):
    node = ScoutFailoverCoordinator.__new__(ScoutFailoverCoordinator)
    clock = [float(now)]
    logger = _Logger()
    node._now = lambda: clock[0]
    node.get_logger = lambda: logger
    node.get_clock = lambda: type('Clock', (), {
        'now': lambda self: type('Now', (), {'to_msg': lambda self: None})()
    })()

    node.enabled = True
    node.require_bootstrap_complete = True
    node.bootstrap_ready = True
    node.bootstrap_ready_wall = 0.0
    node.bootstrap_ready_topic = '/localization_ready'
    node.start_wall = 0.0
    node.startup_grace = 5.0
    node.liveness_timeout = 2.0
    node.confirm_sec = 0.5
    node.pose_timeout = 5.0
    node.robot_pose_timeout = 2.0
    node.recovery_timeout = 10.0
    node.goal_republish_sec = 2.0
    node.max_goal_republishes = 5
    node.leader_arrival_tolerance = 0.8
    node.arrival_tolerance = 0.4
    node.leader_standoff = 0.7
    node.follower_standoff = 0.15

    node.state = FailoverState.NORMAL_OPERATION
    node.leader_name = 'leader'
    node.leader_cancel_topic = '/fleet/leader_nav_cancel'
    node.active_scout_id = 'scout22'
    node.original_scout_id = 'scout22'
    node.follower_name = 'follower21'
    node.scout_epoch = 0
    node.last_liveness_wall = None
    node.suspected_since = None
    node.last_scout_pose = None
    node.last_scout_pose_wall = None
    node.failure_pose = None
    node.leader_pose = None
    node.leader_pose_wall = None
    node.follower_pose = None
    node.follower_pose_wall = None
    node.leader_goal = None
    node.follower_goal = None
    node.leader_recovery_position_reached = False
    node.recovery_goal_publish_count = 0
    node.last_recovery_goal_wall = -1.0e9
    node.recovery_started_wall = None

    node.state_pub = _Publisher()
    node.active_scout_pub = _Publisher()
    node.epoch_pub = _Publisher()
    node.scout_alive_pub = _Publisher()
    node.role_pub = _Publisher()
    node.last_pose_pub = _Publisher()
    node.failure_pose_pub = _Publisher()
    node.leader_goal_pub = _Publisher()
    node.leader_cancel_pub = _Publisher()
    node.follow_command_pub = _Publisher()
    node.role_command_pub = _Publisher()
    return node, clock, logger


def test_heartbeat_qos_matches_best_effort_bridge():
    profile = heartbeat_qos_profile()

    assert profile.reliability == ReliabilityPolicy.BEST_EFFORT
    assert profile.durability == DurabilityPolicy.VOLATILE
    assert profile.history == HistoryPolicy.KEEP_LAST
    assert profile.depth == 5


@pytest.mark.parametrize(
    ('value', 'expected'),
    [
        (0, 0),
        (3, 3),
        (' 7 ', 7),
        (-1, None),
        (True, None),
        (1.0, None),
        ('1.5', None),
        ('bad', None),
        (None, None),
    ],
)
def test_parse_epoch_rejects_malformed_values(value, expected):
    assert parse_epoch(value) == expected


def test_missing_initial_heartbeat_is_suspected_after_bootstrap_and_grace():
    node, clock, _ = _bare_node()
    transitions = []
    original_transition = node._transition

    def record_transition(state):
        if state != node.state:
            transitions.append(state)
        original_transition(state)

    node._transition = record_transition

    clock[0] = 6.9
    node._tick()
    assert node.state == FailoverState.NORMAL_OPERATION

    clock[0] = 7.1
    node._tick()
    node._tick()

    assert node.state == FailoverState.SCOUT_SUSPECTED_DEAD
    assert transitions.count(FailoverState.SCOUT_SUSPECTED_DEAD) == 1


def test_bootstrap_ready_edge_rearms_initial_heartbeat_window():
    node, clock, _ = _bare_node()
    node.bootstrap_ready = False
    node.bootstrap_ready_wall = None

    clock[0] = 10.0
    ready = Bool()
    ready.data = True
    node._on_bootstrap_ready(ready)

    node._check_liveness(now=11.9)
    assert node.state == FailoverState.NORMAL_OPERATION
    node._check_liveness(now=12.1)
    assert node.state == FailoverState.SCOUT_SUSPECTED_DEAD


def test_heartbeat_requires_current_owner_and_epoch():
    node, clock, _ = _bare_node(now=8.0)

    node._on_liveness(_string({'robot': 'scout22', 'epoch': 'bad'}))
    node._on_liveness(_string({'robot': 'old-scout', 'epoch': 0}))
    node._on_liveness(_string({'robot': 'scout22', 'epoch': 1}))
    assert node.last_liveness_wall is None

    node.state = FailoverState.SCOUT_SUSPECTED_DEAD
    node.suspected_since = 7.5
    node._on_liveness(_string({'robot': 'scout22', 'epoch': 0}))

    assert node.last_liveness_wall == 8.0
    assert node.state == FailoverState.NORMAL_OPERATION
    assert node.suspected_since is None


def test_original_active_status_does_not_disarm_normal_watchdog():
    node, _, _ = _bare_node()

    node._on_field_status(_string({
        'robot': 'scout22',
        'epoch': 0,
        'status': 'ACTIVE_SCOUT_READY',
        'active_scout_ready': True,
        'recovery_complete': True,
        'localization_ready': True,
        'motion_authority': 'NONE',
    }))

    assert node.state == FailoverState.NORMAL_OPERATION
    assert node.active_scout_id == 'scout22'
    assert node.role_pub.messages == []


def test_only_current_epoch_follower_completes_valid_takeover_once():
    node, _, logger = _bare_node()
    node.state = FailoverState.FOLLOWER_SCOUT_TAKEOVER
    node.scout_epoch = 2

    # Reconnecting original scout, stale follower, and malformed status are ignored.
    node._on_field_status(_string({
        'robot': 'scout22', 'epoch': 2, 'status': 'ACTIVE_SCOUT_READY',
        'active_scout_ready': True, 'recovery_complete': True,
        'localization_ready': True, 'motion_authority': 'NONE',
    }))
    node._on_field_status(_string({
        'robot': 'follower21', 'epoch': 1, 'status': 'ACTIVE_SCOUT_READY',
        'active_scout_ready': True, 'recovery_complete': True,
        'localization_ready': True, 'motion_authority': 'NONE',
    }))
    node._on_field_status(_string({
        'robot': 'follower21', 'epoch': 2.0, 'status': 'ACTIVE_SCOUT_READY',
        'active_scout_ready': True, 'recovery_complete': True,
        'localization_ready': True, 'motion_authority': 'NONE',
    }))
    assert node.state == FailoverState.FOLLOWER_SCOUT_TAKEOVER
    assert node.active_scout_id == 'scout22'

    completion = _string({
        'robot': 'follower21',
        'epoch': 2,
        'status': 'ACTIVE_SCOUT_READY',
        'active_scout_ready': True,
        'recovery_complete': True,
        'localization_ready': True,
        'motion_authority': 'NONE',
        'nav_goal_active': False,
        'pending_goal_count': 0,
        'active_goal_count': 0,
    })
    node._on_field_status(completion)
    node._on_field_status(completion)

    assert node.state == FailoverState.NEW_SCOUT_EXPLORING
    assert node.active_scout_id == 'follower21'
    assert len(node.role_pub.messages) == 2
    handoff = [message for _, message in logger.messages if 'ACTIVE_SCOUT_HANDOFF' in message]
    assert len(handoff) == 1
    resumed = [message for _, message in logger.messages if 'EXPLORATION_RESUMED' in message]
    assert len(resumed) == 1


def test_follower_active_role_hands_off_source_before_rl_ready():
    node, _, logger = _bare_node()
    node.state = FailoverState.FOLLOWER_SCOUT_TAKEOVER
    node.scout_epoch = 3

    node._on_field_status(_string({
        'robot': 'follower21',
        'epoch': 3,
        'role': 'ACTIVE_SCOUT',
        'status': 'LOCALIZING',
        'active_scout_ready': False,
        'recovery_complete': True,
        'localization_ready': True,
        'motion_authority': 'RECOVERY_NAV',
        'nav_goal_active': False,
        'pending_goal_count': 0,
        'active_goal_count': 0,
    }))

    assert node.state == FailoverState.FOLLOWER_SCOUT_TAKEOVER
    assert node.active_scout_id == 'follower21'
    assert len(node.active_scout_pub.messages) >= 1
    handoff = [message for _, message in logger.messages if 'ACTIVE_SCOUT_HANDOFF' in message]
    assert len(handoff) == 1


def test_follower_completion_is_rejected_before_takeover_state():
    node, _, _ = _bare_node()
    node.scout_epoch = 1
    node.state = FailoverState.RECOVERY_NAVIGATING

    node._on_field_status(_string({
        'robot': 'follower21',
        'epoch': 1,
        'status': 'ACTIVE_SCOUT_READY',
        'active_scout_ready': True,
        'recovery_complete': True,
        'localization_ready': True,
        'motion_authority': 'NONE',
    }))

    assert node.state == FailoverState.RECOVERY_NAVIGATING
    assert node.active_scout_id == 'scout22'


def test_pose_callbacks_require_finite_map_pose_and_record_receipt_time():
    node, clock, _ = _bare_node(now=20.0)
    wrong_frame = _pose(frame='odom')
    invalid = _pose()
    invalid.pose.position.x = float('nan')

    node._on_scout_pose(wrong_frame)
    node._on_follower_pose(invalid)
    node._on_leader_pose(invalid)
    assert node.last_scout_pose is None
    assert node.follower_pose is None
    assert node.leader_pose is None

    scout = _pose(1.0, 2.0)
    follower = _pose(0.5, 0.5)
    leader = _pose(-0.5, -0.5)
    node._on_scout_pose(scout)
    node._on_follower_pose(follower)
    node._on_leader_pose(leader)

    assert is_finite_map_pose(scout)
    assert node.last_scout_pose_wall == 20.0
    assert node.follower_pose_wall == 20.0
    assert node.leader_pose_wall == 20.0
    assert node.last_pose_pub.messages == [scout]


def test_stale_follower_pose_cannot_prove_arrival():
    node, _, _ = _bare_node(now=20.0)
    node.failure_pose = _pose(1.0, 1.0)
    node.follower_pose = _pose(1.0, 1.0)
    node.follower_pose_wall = 17.0

    assert node._follower_arrived(now=20.0) is False

    node.follower_pose_wall = 19.0
    assert node._follower_arrived(now=20.0) is True


def test_stale_leader_pose_cannot_suppress_recovery_goal():
    node, _, _ = _bare_node(now=20.0)
    node.failure_pose = _pose(1.0, 1.0)
    node.leader_pose = _pose(1.0, 1.0)
    node.leader_pose_wall = 17.0

    assert node._leader_already_near_failure() is False

    node.leader_pose_wall = 19.0
    assert node._leader_already_near_failure() is True


def test_stale_scout_pose_cannot_be_frozen_as_failure_pose():
    node, _, _ = _bare_node(now=20.0)
    node.state = FailoverState.SCOUT_SUSPECTED_DEAD
    node.last_scout_pose = _pose(2.0, -1.0)
    node.last_scout_pose_wall = 14.0

    node._confirm_dead(now=20.0)

    assert node.state == FailoverState.FAILOVER_FAILED
    assert node.failure_pose is None
    assert node.scout_epoch == 0
    assert node.failure_pose_pub.messages == []


def test_recovery_timeout_transitions_and_terminal_command_exactly_once():
    node, _, logger = _bare_node(now=11.0)
    node.state = FailoverState.RECOVERY_NAVIGATING
    node.scout_epoch = 4
    node.failure_pose = _pose(1.0, 1.0)
    node.leader_goal = _pose(0.3, 1.0)
    node.follower_goal = _pose(0.85, 1.0)
    node.recovery_started_wall = 0.0

    node._recovery_loop(now=11.0)
    node._recovery_loop(now=12.0)

    assert node.state == FailoverState.FAILOVER_FAILED
    assert len(node.leader_cancel_pub.messages) == 1
    assert len(node.role_command_pub.messages) == 1
    terminal = json.loads(node.role_command_pub.messages[0].data)
    assert terminal['role'] == 'FAILED'
    assert terminal['epoch'] == 4
    assert terminal['robot'] == 'follower21'
    failures = [message for level, message in logger.messages if level == 'error']
    assert len(failures) == 1


def test_failure_pose_freeze_and_dead_edge_are_exactly_once():
    node, _, logger = _bare_node(now=10.0)
    node.state = FailoverState.SCOUT_SUSPECTED_DEAD
    node.last_scout_pose = _pose(2.0, -1.0, 0.5)
    node.last_scout_pose_wall = 9.5
    triggered = []
    node._copy_pose = lambda pose: pose
    node._offset_pose = lambda pose, standoff: pose
    node._leader_already_near_failure = lambda: False
    node._trigger_failover = lambda: triggered.append(node.scout_epoch)

    node._confirm_dead(now=10.0)
    node._confirm_dead(now=10.1)

    assert node.scout_epoch == 1
    assert len(node.failure_pose_pub.messages) == 1
    assert triggered == [1]
    confirmed = [
        message
        for _, message in logger.messages
        if 'SCOUT_DEAD_CONFIRMED | epoch=' in message
    ]
    frozen = [message for _, message in logger.messages if 'LAST_POSE_FROZEN' in message]
    assert len(confirmed) == 1
    assert len(frozen) == 1


def test_recovery_role_command_targets_exact_failed_scout_pose():
    node, _, _ = _bare_node(now=10.0)
    node.scout_epoch = 2
    node.failure_pose = _pose(2.0, -1.0, 0.5)
    node.follower_goal = _pose(1.5, -1.0, 0.5)

    node._publish_recovery_role_command()

    assert len(node.role_command_pub.messages) == 1
    payload = json.loads(node.role_command_pub.messages[0].data)
    assert payload['role'] == 'RECOVERY_NAVIGATING'
    assert payload['target_pose']['x'] == 2.0
    assert payload['target_pose']['y'] == -1.0
    assert payload['failure_pose']['x'] == 2.0
    assert payload['failure_pose']['y'] == -1.0


def test_follower_recovery_goal_is_failed_scout_pose_not_standoff():
    node, _, _ = _bare_node(now=10.0)
    node.state = FailoverState.SCOUT_SUSPECTED_DEAD
    node.last_scout_pose = _pose(2.0, -1.0, 0.5)
    node.last_scout_pose_wall = 9.5
    node.follower_standoff = 0.5
    node._trigger_failover = lambda: None

    node._confirm_dead(now=10.0)

    assert node.follower_goal.pose.position.x == pytest.approx(2.0)
    assert node.follower_goal.pose.position.y == pytest.approx(-1.0)
