import rclpy
from geometry_msgs.msg import PoseStamped

from tb3_fleet_bringup.pose_to_tf import PoseToTfBroadcaster


def make_node():
    if not rclpy.ok():
        rclpy.init()
    return PoseToTfBroadcaster()


def destroy_node(node):
    node.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()


def test_stale_pose_is_not_republished_as_live_tf():
    node = make_node()
    try:
        now = [10.0]
        broadcasts = []
        node._now = lambda: now[0]
        node._broadcast = lambda message: broadcasts.append(message)

        message = PoseStamped()
        message.header.frame_id = 'map'
        message.pose.orientation.w = 1.0
        node._pose_cb(message)
        assert broadcasts == [message]

        now[0] += node.stale_timeout + 0.1
        node._republish()
        assert broadcasts == [message]
    finally:
        destroy_node(node)
