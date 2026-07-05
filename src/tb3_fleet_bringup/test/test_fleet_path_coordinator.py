import rclpy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import OccupancyGrid, Path
from std_msgs.msg import Bool

from tb3_fleet_bringup.fleet_path_coordinator import FleetPathCoordinator


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


def test_crossing_paths_wait_for_pause_acknowledgement():
    node = make_node()
    try:
        node._leader_pose_cb(pose(-1.0, 0.0))
        node._follower_pose_cb(pose(0.0, -1.0))
        node._leader_path_cb(path([(x / 10.0, 0.0) for x in range(-10, 11)]))
        node._follower_path_cb(path([(0.0, y / 10.0) for y in range(-10, 11)]))

        assert node._find_conflict() is not None
        node._tick()
        assert node.state == node.PAUSING

        node._follower_status_cb(Bool(data=False))
        node._tick()
        assert node.state == node.MOVE_ASIDE
        assert node.leader_yield is not None
        assert node.follower_yield is not None
    finally:
        destroy_node(node)


def test_same_direction_following_is_not_a_path_conflict():
    node = make_node()
    try:
        shared_route = [(x / 10.0, 0.0) for x in range(-15, 21)]
        node._leader_pose_cb(pose(-0.5, 0.0))
        node._follower_pose_cb(pose(-1.2, 0.0))
        node._leader_path_cb(path(shared_route))
        node._follower_path_cb(path(shared_route))

        assert node._find_conflict() is None
    finally:
        destroy_node(node)
