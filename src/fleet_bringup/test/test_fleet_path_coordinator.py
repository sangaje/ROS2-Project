import rclpy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import OccupancyGrid
from std_msgs.msg import Bool

from fleet_bringup.fleet_path_coordinator import FleetPathCoordinator


def pose(x: float, y: float) -> PoseStamped:
    msg = PoseStamped()
    msg.header.frame_id = 'map'
    msg.pose.position.x = x
    msg.pose.position.y = y
    msg.pose.orientation.w = 1.0
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
    assert node.direct_goal_passthrough is True
    return node


def destroy_node(node: FleetPathCoordinator) -> None:
    node.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()


def test_leader_goal_passes_through_without_intermediate_goal():
    node = make_node()
    try:
        published = []
        node._publish_goal = lambda publisher, message: published.append(
            (publisher, message)
        )

        goal = pose(2.0, 1.0)
        node._leader_goal_cb(goal)

        assert node._xy(node.leader_user_goal) == (2.0, 1.0)
        assert node.leader_evasion_goal is None
        assert node.follower_evasion_goal is None
        assert published == [(node.leader_goal_pub, node.leader_user_goal)]
    finally:
        destroy_node(node)


def test_burger_goal_pauses_follow_and_passes_through():
    node = make_node()
    try:
        published = []
        commands = []
        node._publish_goal = lambda publisher, message: published.append(
            (publisher, message)
        )
        node._publish_follow_command = (
            lambda command, force=False: commands.append((command, force))
        )

        goal = pose(-1.0, 2.0)
        node._follower_goal_cb(goal)

        assert node.follower_desired_following is False
        assert commands == [('PAUSE', True)]
        assert node.follower_evasion_goal is None
        assert published == [(node.follower_goal_pub, node.follower_user_goal)]
    finally:
        destroy_node(node)


def test_motion_risk_warns_but_does_not_split_nav2_goal():
    node = make_node()
    try:
        published = []
        node._publish_goal = lambda publisher, message: published.append(
            (publisher, message)
        )
        node._follower_status_cb(Bool(data=False))
        node._leader_pose_cb(pose(-0.4, 0.0))
        node._follower_pose_cb(pose(0.4, 0.0))
        now = node._now()
        node.leader_velocity = (0.20, 0.0)
        node.follower_velocity = (-0.20, 0.0)
        node.leader_motion_sample = (now, (-0.4, 0.0))
        node.follower_motion_sample = (now, (0.4, 0.0))

        node._tick()
        node._tick()

        assert node.state == node.IDLE
        assert node.collision_warning is True
        assert node.leader_evasion_goal is None
        assert node.follower_evasion_goal is None
        assert published == []
    finally:
        destroy_node(node)


def test_stale_pose_preserves_saved_goal_without_hold_goal():
    node = make_node()
    try:
        published = []
        node._publish_goal = lambda publisher, message: published.append(
            (publisher, message)
        )
        leader_goal = pose(3.0, 0.0)
        node._leader_goal_cb(leader_goal)
        assert published == [(node.leader_goal_pub, node.leader_user_goal)]

        node._leader_pose_cb(pose(0.0, 0.0))
        node._follower_pose_cb(pose(-0.7, 0.0))
        node.leader_pose_time = node._now() - node.pose_stale - 0.1
        node._tick()

        assert node.collision_warning is True
        assert node.pose_hold_active is False
        assert node._xy(node.leader_user_goal) == (3.0, 0.0)
        assert published == [(node.leader_goal_pub, node.leader_user_goal)]
    finally:
        destroy_node(node)


def test_no_follower_required_keeps_direct_leader_nav_unheld():
    node = make_node()
    try:
        published = []
        node.require_follower_pose = False
        node._publish_goal = lambda publisher, message: published.append(
            (publisher, message)
        )

        node._leader_pose_cb(pose(0.0, 0.0))
        node._tick()

        assert node.state == node.IDLE
        assert node.collision_warning is False
        assert published == []
    finally:
        destroy_node(node)
