import rclpy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import OccupancyGrid, Path
from std_msgs.msg import Bool

from tb3_fleet_bringup.fleet_path_coordinator import (
    FleetPathCoordinator,
    distance,
)


def pose(x: float, y: float) -> PoseStamped:
    msg = PoseStamped()
    msg.header.frame_id = 'map'
    msg.pose.position.x = x
    msg.pose.position.y = y
    msg.pose.orientation.w = 1.0
    return msg


def path(points) -> Path:
    msg = Path()
    msg.header.frame_id = 'map'
    msg.poses = [pose(x, y) for x, y in points]
    return msg


def free_map() -> OccupancyGrid:
    msg = OccupancyGrid()
    msg.header.frame_id = 'map'
    msg.info.resolution = 0.05
    msg.info.width = 240
    msg.info.height = 240
    msg.info.origin.position.x = -6.0
    msg.info.origin.position.y = -6.0
    msg.info.origin.orientation.w = 1.0
    msg.data = [0] * (msg.info.width * msg.info.height)
    return msg


def make_node() -> FleetPathCoordinator:
    if not rclpy.ok():
        rclpy.init()
    node = FleetPathCoordinator()
    node._map_cb(free_map())
    return node


def destroy_node(node: FleetPathCoordinator) -> None:
    node.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()


def test_follow_mode_crossing_preserves_leader_goal_and_moves_only_follower():
    node = make_node()
    try:
        node._follower_status_cb(Bool(data=True))
        node._leader_pose_cb(pose(-0.5, 0.0))
        node._follower_pose_cb(pose(0.0, -0.5))
        node._leader_path_cb(path([(x / 10.0, 0.0) for x in range(-10, 11)]))
        node._follower_path_cb(path([(0.0, y / 10.0) for y in range(-10, 11)]))
        now = node._now()
        node.leader_velocity = (0.20, 0.0)
        node.follower_velocity = (0.0, 0.20)
        node.leader_motion_sample = (now, (-0.5, 0.0))
        node.follower_motion_sample = (now, (0.0, -0.5))

        node._tick()
        node._tick()
        assert node.state == node.CLEARING
        assert node.priority_robot == node.LEADER
        assert node.leader_evasion_goal is None
        assert node.follower_evasion_goal is not None
        assert node._xy(node.follower_evasion_goal) != node._xy(node.follower_pose)

    finally:
        destroy_node(node)


def test_independent_follower_gets_priority_when_it_is_the_moving_robot():
    node = make_node()
    try:
        node._follower_status_cb(Bool(data=False))
        node._leader_pose_cb(pose(-0.5, 0.0))
        node._follower_pose_cb(pose(0.0, -0.5))
        node._leader_path_cb(path([(x / 10.0, 0.0) for x in range(-10, 11)]))
        node._follower_path_cb(path([(0.0, y / 10.0) for y in range(-3, 11)]))
        now = node._now()
        node.leader_velocity = (0.05, 0.0)
        node.follower_velocity = (0.0, 0.20)
        node.leader_motion_sample = (now, (-0.5, 0.0))
        node.follower_motion_sample = (now, (0.0, -0.5))

        node._tick()
        node._tick()
        assert node.state == node.CLEARING
        assert node.priority_robot == node.FOLLOWER
        assert node.follower_was_following is False
        assert node.follower_resume_goal is not None
        assert node.leader_evasion_goal is not None
        assert node.follower_evasion_goal is None
    finally:
        destroy_node(node)


def test_waffle_keeps_priority_when_both_independent_robots_are_moving():
    node = make_node()
    try:
        node._follower_status_cb(Bool(data=False))
        node._leader_pose_cb(pose(-0.5, 0.0))
        node._follower_pose_cb(pose(0.0, -0.5))
        now = node._now()
        node.leader_velocity = (0.10, 0.0)
        node.follower_velocity = (0.0, 0.22)
        node.leader_motion_sample = (now, (-0.5, 0.0))
        node.follower_motion_sample = (now, (0.0, -0.5))

        node._tick()
        node._tick()
        assert node.priority_robot == node.LEADER
        assert node.leader_evasion_goal is None
        assert node.follower_evasion_goal is not None
    finally:
        destroy_node(node)


def test_pose_motion_without_paths_moves_only_the_yielding_robot():
    node = make_node()
    try:
        node._follower_status_cb(Bool(data=False))
        node._leader_pose_cb(pose(-0.4, 0.0))
        node._follower_pose_cb(pose(0.4, 0.0))
        now = node._now()
        node.leader_velocity = (0.20, 0.0)
        node.follower_velocity = (0.0, 0.0)
        node.leader_motion_sample = (now, (-0.4, 0.0))
        node.follower_motion_sample = (now, (0.4, 0.0))

        risk, _, _ = node._motion_risk()
        assert risk is True
        node._tick()
        node._tick()
        assert node.state == node.CLEARING
        assert node.leader_evasion_goal is None
        assert node.follower_evasion_goal is not None
        assert distance(
            node._xy(node.follower_pose),
            node._xy(node.follower_evasion_goal),
        ) <= node.evasion_offset_max + 1.0e-6
    finally:
        destroy_node(node)


def test_normal_same_direction_follow_motion_is_not_a_proximity_hazard():
    node = make_node()
    try:
        node._follower_status_cb(Bool(data=True))
        node._leader_pose_cb(pose(0.0, 0.0))
        node._follower_pose_cb(pose(-0.70, 0.0))
        now = node._now()
        node.leader_velocity = (0.15, 0.0)
        node.follower_velocity = (0.15, 0.0)
        node.leader_motion_sample = (now, (0.0, 0.0))
        node.follower_motion_sample = (now, (-0.70, 0.0))

        risk, _, _ = node._motion_risk()
        assert risk is False
    finally:
        destroy_node(node)


def test_robot_moving_away_does_not_trigger_yield_maneuver():
    node = make_node()
    try:
        node._follower_status_cb(Bool(data=False))
        node._leader_pose_cb(pose(0.0, 0.0))
        node._follower_pose_cb(pose(0.65, 0.0))
        now = node._now()
        node.leader_velocity = (-0.20, 0.0)
        node.follower_velocity = (0.0, 0.0)
        node.leader_motion_sample = (now, (0.0, 0.0))
        node.follower_motion_sample = (now, (0.65, 0.0))

        risk, _, _ = node._motion_risk()
        assert risk is False
        node._tick()
        node._tick()
        assert node.state == node.IDLE
    finally:
        destroy_node(node)


def test_path_intersection_alone_does_not_trigger_evasion():
    node = make_node()
    try:
        node._leader_pose_cb(pose(-1.0, 0.0))
        node._follower_pose_cb(pose(0.0, -1.0))
        node._leader_path_cb(path([(x / 10.0, 0.0) for x in range(-10, 11)]))
        node._follower_path_cb(path([(0.0, y / 10.0) for y in range(-10, 11)]))

        node._tick()
        assert node.state == node.IDLE
    finally:
        destroy_node(node)


def test_user_goals_remain_persistent_across_evasion_updates():
    node = make_node()
    try:
        leader_goal = pose(2.0, 1.0)
        follower_goal = pose(-1.0, 2.0)
        node._leader_goal_cb(leader_goal)
        node._follower_goal_cb(follower_goal)
        assert node._xy(node.leader_user_goal) == (2.0, 1.0)
        assert node._xy(node.follower_user_goal) == (-1.0, 2.0)

        node.state = node.CLEARING
        node.priority_robot = node.LEADER
        replacement = pose(3.0, -1.0)
        node._leader_goal_cb(replacement)
        assert node._xy(node.leader_user_goal) == (3.0, -1.0)
        assert node._xy(node.leader_resume_goal) == (3.0, -1.0)
    finally:
        destroy_node(node)


def test_independent_follower_goal_is_reasserted_after_pause_acknowledgement():
    node = make_node()
    try:
        published = []
        node.follower_user_goal = pose(-1.0, 2.0)
        node.follower_desired_following = False
        node._publish_goal = lambda publisher, message: published.append(message)

        node._follower_status_cb(Bool(data=False))
        assert published == [node.follower_user_goal]
    finally:
        destroy_node(node)


def test_follower_yield_goal_is_reasserted_after_pause_acknowledgement():
    node = make_node()
    try:
        published = []
        node.state = node.CLEARING
        node.priority_robot = node.LEADER
        node.follower_evasion_goal = pose(0.5, 1.0)
        node._publish_goal = lambda publisher, message: published.append(message)

        node._follower_status_cb(Bool(data=False))
        assert published == [node.follower_evasion_goal]
    finally:
        destroy_node(node)


def test_releasing_right_of_way_does_not_resend_priority_goal():
    node = make_node()
    try:
        published = []
        node.priority_robot = node.LEADER
        node.leader_resume_goal = pose(2.0, 0.0)
        node.state = node.CLEARING
        node._publish_goal = lambda publisher, message: published.append(message)

        node._release_priority()
        assert node.state == node.PRIORITY_PASS
        assert published == []
    finally:
        destroy_node(node)


def test_only_yielding_robot_resumes_its_saved_destination():
    node = make_node()
    try:
        published = []
        node.priority_robot = node.LEADER
        node.follower_was_following = False
        node.leader_resume_goal = pose(2.0, 0.0)
        node.follower_resume_goal = pose(-2.0, 0.0)
        node.leader_user_goal = pose(2.0, 0.0)
        node.follower_user_goal = pose(-2.0, 0.0)
        node._publish_goal = (
            lambda publisher, message: published.append((publisher, message))
        )

        node._restore_after_maneuver()
        assert published == [
            (node.follower_goal_pub, node.follower_user_goal),
        ]
        assert node.leader_user_goal is not None
        assert node.follower_user_goal is not None
    finally:
        destroy_node(node)


def test_unsafe_clearing_timeout_enters_blocked_instead_of_restoring_goals():
    node = make_node()
    try:
        node._leader_pose_cb(pose(-0.25, 0.0))
        node._follower_pose_cb(pose(0.25, 0.0))
        node.state = node.CLEARING
        node.state_since = node._now() - node.clearing_timeout - 0.1
        node.leader_evasion_goal = pose(-1.0, 0.0)
        node.follower_evasion_goal = None
        node.priority_robot = node.FOLLOWER

        node._tick()
        assert node.state == node.BLOCKED
    finally:
        destroy_node(node)


def test_collision_risk_can_interrupt_cooldown():
    node = make_node()
    try:
        node._follower_status_cb(Bool(data=False))
        node._leader_pose_cb(pose(-0.4, 0.0))
        node._follower_pose_cb(pose(0.4, 0.0))
        now = node._now()
        node.leader_velocity = (0.20, 0.0)
        node.follower_velocity = (-0.20, 0.0)
        node.leader_motion_sample = (now, (-0.4, 0.0))
        node.follower_motion_sample = (now, (0.4, 0.0))
        node.state = node.COOLDOWN
        node.cooldown_until = now + 10.0

        node._tick()
        assert node.state == node.COOLDOWN
        node._tick()
        assert node.state == node.CLEARING
        assert node.collision_warning is True
    finally:
        destroy_node(node)


def test_guard_yields_when_leader_approaches_and_returns_when_clear():
    node = make_node()
    try:
        node._leader_pose_cb(pose(-0.5, 0.0))
        node._follower_pose_cb(pose(5.0, 5.0))
        node._guard_pose_cb(pose(0.4, 0.0))
        now = node._now()
        node.leader_velocity = (0.20, 0.0)
        node.leader_motion_sample = (now, (-0.5, 0.0))

        node._tick()
        node._tick()
        assert node.guard_state == node.CLEARING
        assert node.guard_evasion_goal is not None
        assert node.guard_resume_pose is not None
        assert node._xy(node.guard_resume_pose) == (0.4, 0.0)

        # Leader moves away again; guard should resume its original spot.
        expected_resume = node.guard_resume_pose
        node._leader_pose_cb(pose(-3.0, 0.0))
        node.leader_velocity = (0.0, 0.0)
        node.leader_motion_sample = (node._now(), (-3.0, 0.0))
        published = []
        node._publish_goal = lambda publisher, message: published.append(
            (publisher, message)
        )
        node._tick()
        assert node.guard_state == node.COOLDOWN
        assert published == [(node.guard_goal_pub, expected_resume)]
    finally:
        destroy_node(node)


def test_guard_without_a_pose_never_triggers_any_guard_state():
    node = make_node()
    try:
        node._leader_pose_cb(pose(-0.5, 0.0))
        node._follower_pose_cb(pose(0.0, -0.5))
        now = node._now()
        node.leader_velocity = (0.20, 0.0)
        node.follower_velocity = (0.0, 0.20)
        node.leader_motion_sample = (now, (-0.5, 0.0))
        node.follower_motion_sample = (now, (0.0, -0.5))

        node._tick()
        node._tick()
        # Existing leader/follower behaviour is unaffected by an absent guard.
        assert node.state == node.CLEARING
        assert node.guard_state == node.IDLE
    finally:
        destroy_node(node)


def test_stale_pose_forces_safety_warning():
    node = make_node()
    try:
        node._leader_pose_cb(pose(0.0, 0.0))
        node._follower_pose_cb(pose(-0.7, 0.0))
        node.leader_pose_time = node._now() - node.pose_stale - 0.1

        node._tick()
        assert node.collision_warning is True
        assert node.state == node.IDLE
        assert node.pose_hold_active is True

        node._leader_pose_cb(pose(0.0, 0.0))
        node._follower_pose_cb(pose(-0.7, 0.0))
        node._tick()
        assert node.pose_hold_active is False
    finally:
        destroy_node(node)
