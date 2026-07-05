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


def test_follow_mode_crossing_moves_both_and_gives_leader_priority():
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
        assert node.state == node.CLEARING
        assert node.priority_robot == node.LEADER
        assert node.leader_evasion_goal is not None
        assert node.follower_evasion_goal is not None
        assert node._xy(node.leader_evasion_goal) != node._xy(node.leader_pose)
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
        assert node.state == node.CLEARING
        assert node.priority_robot == node.FOLLOWER
        assert node.follower_was_following is False
        assert node.follower_resume_goal is not None
    finally:
        destroy_node(node)


def test_pose_motion_without_paths_triggers_opposite_escape_goals():
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
        assert node.state == node.CLEARING
        assert node.leader_evasion_goal is not None
        assert node.follower_evasion_goal is not None
        assert distance(
            node._xy(node.leader_evasion_goal),
            node._xy(node.follower_evasion_goal),
        ) > 0.8
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


def test_path_intersection_without_position_motion_does_not_trigger_evasion():
    node = make_node()
    try:
        node._leader_pose_cb(pose(-1.0, 0.0))
        node._follower_pose_cb(pose(0.0, -1.0))
        node._leader_path_cb(path([(x / 10.0, 0.0) for x in range(-10, 11)]))
        node._follower_path_cb(path([(0.0, y / 10.0) for y in range(-10, 11)]))

        assert node._find_conflict() is not None
        node._tick()
        assert node.state == node.IDLE
    finally:
        destroy_node(node)
