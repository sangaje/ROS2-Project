import rclpy
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import String

from tb3_fleet_bringup.fleet_follower import FleetFollower


def make_node() -> FleetFollower:
    if not rclpy.ok():
        rclpy.init()
    return FleetFollower()


def destroy_node(node: FleetFollower) -> None:
    node.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()


def test_default_target_is_seventy_centimetres_behind_leader():
    node = make_node()
    try:
        leader = PoseStamped()
        leader.header.frame_id = 'map'
        leader.pose.position.x = 1.0
        leader.pose.position.y = 2.0
        leader.pose.orientation.w = 1.0
        node._leader_pose_callback(leader)

        target = node._target_behind_leader()
        assert abs(target.pose.position.x - 0.30) < 1.0e-6
        assert abs(target.pose.position.y - 2.0) < 1.0e-6
    finally:
        destroy_node(node)


def test_pause_and_resume_commands_update_follower_state():
    node = make_node()
    try:
        node._command_callback(String(data='PAUSE'))
        assert node.follow_enabled is False
        node._command_callback(String(data='RESUME'))
        assert node.follow_enabled is True
    finally:
        destroy_node(node)
