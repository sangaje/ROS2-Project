import json

from builtin_interfaces.msg import Time
from geometry_msgs.msg import PoseStamped, TwistStamped
from std_msgs.msg import Bool, String

from system_bringup.leader_shadow_follow import LeaderMode, LeaderShadowFollow
from system_bringup.nav_goal_manager import NavGoalManager
from system_bringup.unified_field_robot import (
    MotionAuthority,
    parse_epoch,
    Role,
    UnifiedFieldRobot,
)


class _Logger:
    def __init__(self):
        self.messages = []

    def _add(self, level, message, *args, **kwargs):
        self.messages.append((level, str(message)))

    def info(self, message, *args, **kwargs):
        self._add('info', message, *args, **kwargs)

    def warning(self, message, *args, **kwargs):
        self._add('warning', message, *args, **kwargs)

    def error(self, message, *args, **kwargs):
        self._add('error', message, *args, **kwargs)


class _Publisher:
    def __init__(self):
        self.messages = []

    def publish(self, message):
        self.messages.append(message)


class _Now:
    def to_msg(self):
        return Time()


class _Clock:
    def now(self):
        return _Now()


class _Future:
    def __init__(self, value=None):
        self.value = value
        self.callbacks = []

    def add_done_callback(self, callback):
        self.callbacks.append(callback)

    def result(self):
        return self.value


class _ActionClient:
    def __init__(self):
        self.sent = []

    def server_is_ready(self):
        return True

    def send_goal_async(self, goal):
        future = _Future()
        self.sent.append((goal, future))
        return future


class _ServiceClient:
    def service_is_ready(self):
        return False


class _Runtime:
    active = True


class _Result:
    def __init__(self, status):
        self.status = status


def _pose(x=0.0, y=0.0, frame='map'):
    msg = PoseStamped()
    msg.header.frame_id = frame
    msg.pose.position.x = float(x)
    msg.pose.position.y = float(y)
    msg.pose.orientation.w = 1.0
    return msg


def _bare_field(now=10.0):
    node = UnifiedFieldRobot.__new__(UnifiedFieldRobot)
    logger = _Logger()
    node.get_logger = lambda: logger
    node.get_clock = lambda: _Clock()
    node._now = lambda: float(now)
    node.robot_name = 'follower21'
    node.epoch = 2
    node.role = Role.RECOVERY_NAVIGATING
    node.motion_authority = MotionAuthority.NONE
    node.rl_runtime = None
    node.rl_backend = 'external_worker'
    node.scout_rl_enabled = True
    node.in_process_rl_enabled = False
    node.nav_client = _ActionClient()
    node.last_odom_xy = (0.0, 0.0)
    node.nav_start_odom_xy = None
    node.movement_started = False
    node.movement_sample_count = 0
    node.self_pose = None
    node.self_pose_wall = None
    node.self_pose_timeout = 2.0
    node.arrival_tolerance = 0.4
    node.navigate_action = '/field_b/navigate_to_pose'
    node.require_localization_ready = True
    node.localization_ready = True
    node.recovery_nav_failures = 0
    node.recovery_nav_succeeded = False
    node.recovery_arrived = False
    node.max_recovery_nav_retries = 3
    node.recovery_nav_retry_sec = 1.0
    node.recovery_stop_linear_epsilon = 0.03
    node.recovery_stop_angular_epsilon = 0.08
    node.last_linear_speed = 0.0
    node.last_angular_speed = 0.0
    node.nav_retry_not_before = -1.0e9
    node.nav = NavGoalManager(
        node.nav_client,
        now_stamp=lambda: Time(),
        copy_pose=node._copy_pose,
        log=logger,
        set_authority=node._set_authority,
        current_authority=lambda: node.motion_authority,
        on_goal_sent=node._on_nav_goal_sent,
        on_failure=node._handle_nav_failure,
        on_result=node._on_nav_result,
    )
    return node, logger


def test_epoch_parser_rejects_truncated_and_boolean_values():
    assert parse_epoch(3) == 3
    assert parse_epoch(' 4 ') == 4
    assert parse_epoch(1.0) is None
    assert parse_epoch(True) is None
    assert parse_epoch(-1) is None


def test_recovery_goal_has_one_inflight_send_only():
    node, _ = _bare_field()
    node._queue_nav_goal(_pose(1.0, 2.0), 'RECOVERY')

    node._dispatch_pending_nav_goal()
    node._dispatch_pending_nav_goal()

    assert len(node.nav_client.sent) == 1
    assert node.nav.inflight_goal_ids == {1}
    assert node.nav.pending_goal is None
    assert node.motion_authority == MotionAuthority.FAILOVER_RECOVERY_NAV


def test_stale_recovery_result_cannot_advance_role():
    node, logger = _bare_field()
    node.nav.goal_epoch = 8
    transitions = []
    node._enter_role = lambda role, reason: transitions.append((role, reason))

    node.nav._goal_result_cb(_Future(_Result(4)), goal_id=7, source='RECOVERY')

    assert transitions == []
    assert any('STALE_FIELD_NAV_RESULT_IGNORED' in text for _, text in logger.messages)


def test_success_without_fresh_arrival_does_not_advance_mission():
    node, _ = _bare_field()
    node.nav.goal_epoch = 3
    node.nav.active_goal_handle = object()
    node.nav.active_goal_source = 'RECOVERY'
    node.motion_authority = MotionAuthority.FAILOVER_RECOVERY_NAV
    node.recovery_target = _pose(3.0, 0.0)
    node._at_pose = lambda target: False
    transitions = []
    node._enter_role = lambda role, reason: transitions.append((role, reason))

    node.nav._goal_result_cb(_Future(_Result(4)), goal_id=3, source='RECOVERY')

    assert transitions == []
    assert node.recovery_nav_failures == 1
    assert node.nav_retry_not_before == 11.0
    assert node.movement_started is False


def test_spin_timeout_without_odom_motion_is_not_success():
    node, logger = _bare_field(now=20.0)
    node.role = Role.LOCALIZATION_SPIN
    node.motion_authority = MotionAuthority.LOCALIZATION_SPIN
    node.spin_command_started = True
    node.spin_start_wall = 5.0
    node.spin_timeout = 10.0
    node.spin_target = 6.45
    node.accumulated_yaw = 0.0
    node.spin_motion_detected = False
    node.spin_last_attempt_completed = False
    node.cmd_vel_topic = '/field_b/cmd_vel'
    node._non_rl_motion_quiesced = lambda: True
    twists = []
    transitions = []
    node._publish_twist = twists.append
    node._enter_role = lambda role, reason: transitions.append((role, reason))

    node._tick_spin()

    assert twists == [0.0]
    assert node.spin_last_attempt_completed is False
    assert transitions == [(Role.LOCALIZATION_SETTLE, 'spin_timeout')]
    assert any('SPIN_FAILED_NO_MOTION' in text for _, text in logger.messages)


def test_recovery_arrival_requires_fresh_matching_frame_pose():
    node, _ = _bare_field(now=10.0)
    node.self_pose = _pose(1.0, 1.0, frame='map')
    node.self_pose_wall = 7.0
    node.self_pose_timeout = 2.0
    node.arrival_tolerance = 0.4
    assert node._at_pose(_pose(1.0, 1.0, frame='map')) is False

    node.self_pose_wall = 10.0
    assert node._at_pose(_pose(1.0, 1.0, frame='odom')) is False
    assert node._at_pose(_pose(1.1, 1.1, frame='map')) is True


def test_recovery_does_not_advance_on_pose_distance_before_nav_success():
    node, _ = _bare_field(now=10.0)
    node.role = Role.RECOVERY_NAVIGATING
    node.recovery_target = _pose(1.0, 1.0)
    node.self_pose = _pose(1.0, 1.0)
    node.self_pose_wall = 10.0
    node.movement_started = True
    node.recovery_nav_succeeded = False
    node._queue_nav_goal = lambda pose, source: None
    transitions = []
    node._enter_role = lambda role, reason: transitions.append((role, reason))

    node._tick_recovery()

    assert transitions == []


def test_recovery_nav_success_pose_and_stop_mark_arrival():
    node, _ = _bare_field(now=10.0)
    node.role = Role.RECOVERY_NAVIGATING
    node.nav.goal_epoch = 3
    node.nav.active_goal_handle = object()
    node.nav.active_goal_source = 'RECOVERY'
    node.recovery_target = _pose(1.0, 1.0)
    node.self_pose = _pose(1.0, 1.0)
    node.self_pose_wall = 10.0
    node.movement_started = True
    node.last_linear_speed = 0.0
    node.last_angular_speed = 0.0
    transitions = []
    node._enter_role = lambda role, reason: transitions.append((role, reason))

    node.nav._goal_result_cb(_Future(_Result(4)), goal_id=3, source='RECOVERY')

    assert node.recovery_arrived is True
    assert transitions == [
        (Role.ARRIVED_AT_FAILURE_POSE, 'recovery_arrival_verified'),
        (Role.LOCALIZATION_CHECK, 'recovery_arrival_verified'),
    ]


def test_rl_heartbeat_requires_localization_and_live_runtime():
    node, _ = _bare_field()
    node.role = Role.ACTIVE_SCOUT
    node.robot_name = 'scout22'
    node.scout_rl_enabled = True
    node.require_localization_ready = True
    node.localization_ready = False
    node.motion_authority = MotionAuthority.ACTIVE_SCOUT_RL
    node.rl_runtime = _Runtime()
    node.heartbeat_seq = 0
    node.heartbeat_pub = _Publisher()

    node._publish_heartbeat()
    assert node.heartbeat_pub.messages == []

    node.localization_ready = True
    node._publish_heartbeat()
    assert len(node.heartbeat_pub.messages) == 1
    payload = json.loads(node.heartbeat_pub.messages[0].data)
    assert payload['robot'] == 'scout22'
    assert payload['epoch'] == 2


def test_external_worker_backend_grants_only_motion_authority_not_runtime():
    node, _ = _bare_field()
    node.role = Role.ACTIVE_SCOUT
    node.rl_backend = 'external_worker'
    node.scout_rl_enabled = True
    node.rl_runtime = None
    node.require_localization_ready = False
    node._nav_motion_quiesced = lambda: True

    node._activate_rl()

    assert node.rl_runtime is None
    assert node.motion_authority == MotionAuthority.ACTIVE_SCOUT_RL


def test_shadow_goal_cancel_is_edge_triggered_on_failover():
    node = LeaderShadowFollow.__new__(LeaderShadowFollow)
    node.failover_state = 'NORMAL_OPERATION'
    node.shadow_goal_active = True
    node.last_goal = _pose()
    node.cancel_pub = _Publisher()
    logger = _Logger()
    node.get_logger = lambda: logger

    message = String(data='SCOUT_SUSPECTED_DEAD')
    node._on_failover_state(message)
    node._on_failover_state(message)

    assert node.cancel_pub.messages == [Bool(data=True), Bool(data=False)]
    assert node.shadow_goal_active is False


def test_direct_shadow_cmd_compensates_loaded_leader_speed():
    node = LeaderShadowFollow.__new__(LeaderShadowFollow)
    node.leader_pose = _pose()
    node.cmd_pub = _Publisher()
    node.use_stamped_cmd_vel = True
    node.get_clock = lambda: _Clock()
    node._stop_direct_cmd = lambda reason: None
    node.cmd_goal_tolerance = 0.16
    node.shadow_linear_vel = 0.38
    node.catchup_linear_vel = 0.46
    node.shadow_angular_vel = 0.85
    node.linear_kp = 0.70
    node.angular_kp = 1.40
    node.heading_slowdown_rad = 0.75
    node.cmd_linear_scale = 3.0
    node.cmd_angular_scale = 1.0
    node.cmd_max_linear_vel = 0.75
    node.cmd_max_angular_vel = 1.20
    node.direct_cmd_active = False

    node._publish_direct_shadow_cmd(_pose(2.0, 0.0), catchup=False)

    assert len(node.cmd_pub.messages) == 1
    command = node.cmd_pub.messages[0]
    assert isinstance(command, TwistStamped)
    assert command.twist.linear.x == 0.75
    assert command.twist.angular.z == 0.0
    assert node.direct_cmd_active is True


def test_nav2_shadow_goal_starts_even_when_leader_begins_next_to_scout():
    node = LeaderShadowFollow.__new__(LeaderShadowFollow)
    logger = _Logger()
    node.get_logger = lambda: logger
    node.get_clock = lambda: _Clock()
    node._now = lambda: 10.0
    node.enabled = True
    node.start_wall = 0.0
    node.startup_grace = 0.0
    node.require_localization_ready = False
    node.localization_ready = True
    node.pause_on_omx_aiming = False
    node.pause_on_raw_target_detection = False
    node.failover_state = 'NORMAL_OPERATION'
    node.active_scout_id = 'scout22'
    node.original_scout_id = 'scout22'
    node.follower_robot_name = 'follower21'
    node.leader_pose = _pose(0.0, 0.0)
    node.original_scout_pose = _pose(0.10, 0.0)
    node.original_scout_wall = 10.0
    node.follower_scout_pose = None
    node.follower_scout_wall = -1.0e9
    node.scout_pose_timeout = 5.0
    node.mode = LeaderMode.IDLE
    node.omx_state = ''
    node.direct_shadow_cmd_vel = False
    node.shadow_active = True
    node.shadow_goal_active = False
    node.direct_cmd_active = False
    node.map_msg = None
    node.heading = 0.0
    node.previous_scout_sample = None
    node.last_goal = None
    node.last_goal_wall = -1.0e9
    node.last_nominal_target = None
    node.goal_period = 0.0
    node.goal_min_change = 0.05
    node.follow_distance = 1.2
    node.cmd_goal_tolerance = 0.16
    node.far_distance = 4.5
    node.shadow_linear_vel = 0.14
    node.catchup_linear_vel = 0.14
    node.shadow_angular_vel = 0.35
    node.restore_linear_vel = 0.14
    node.restore_angular_vel = 0.35
    node.map_clearance = 0.22
    node.search_radius = 1.2
    node.search_step = 0.15
    node.occupied_threshold = 50
    node.allow_unknown = False
    node.scan_fov_deg = 60.0
    node.speed_limit_pending = False
    node.speed_profile = None
    node.controller_client = _ServiceClient()
    node.goal_pub = _Publisher()
    node.goal_debug_pub = _Publisher()
    node.cancel_pub = _Publisher()
    node.cmd_pub = _Publisher()
    node.state_pub = _Publisher()

    node._tick()

    assert len(node.goal_pub.messages) == 1
    goal = node.goal_pub.messages[0]
    assert -1.3 < goal.pose.position.x < -0.9
    assert node.shadow_goal_active is True
