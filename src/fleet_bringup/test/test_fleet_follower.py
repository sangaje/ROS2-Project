import rclpy
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import String

from fleet_bringup.fleet_follower import FleetFollower


class FakeFuture:
    def __init__(self, value=None):
        self.value = value
        self.callbacks = []

    def result(self):
        return self.value

    def add_done_callback(self, callback):
        self.callbacks.append(callback)


class FakeGoalHandle:
    accepted = True

    def __init__(self):
        self.cancel_count = 0
        self.result_future = FakeFuture()

    def cancel_goal_async(self):
        self.cancel_count += 1
        return FakeFuture()

    def get_result_async(self):
        return self.result_future


def make_node() -> FleetFollower:
    if not rclpy.ok():
        rclpy.init()
    return FleetFollower()


def destroy_node(node: FleetFollower) -> None:
    node.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()


def test_default_target_is_fifty_centimetres_behind_leader():
    node = make_node()
    try:
        leader = PoseStamped()
        leader.header.frame_id = 'map'
        leader.pose.position.x = 1.0
        leader.pose.position.y = 2.0
        leader.pose.orientation.w = 1.0
        node._leader_pose_callback(leader)

        target = node._target_behind_leader()
        assert abs(target.pose.position.x - 0.50) < 1.0e-6
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


def test_stale_follow_action_response_cannot_replace_latest_handle():
    node = make_node()
    try:
        node.goal_count = 2
        old_handle = FakeGoalHandle()
        latest_handle = FakeGoalHandle()

        node._goal_response_callback(FakeFuture(old_handle), 1)
        assert old_handle.cancel_count == 1
        assert node.active_goal_handle is None

        node._goal_response_callback(FakeFuture(latest_handle), 2)
        assert node.active_goal_handle is latest_handle
        assert node.active_goal_id == 2
    finally:
        destroy_node(node)


def test_pending_follow_goal_response_timeout_allows_retry():
    node = make_node()
    try:
        node.last_goal_outcome = 'pending'
        node.pending_goal_response_since = 1.0
        node.goal_response_timeout = 2.0
        node.active_goal_handle = None
        node.active_goal_id = 0
        node.goal_count = 4
        node.last_goal_xy = (1.0, 2.0)

        node._recover_goal_response_timeout(3.5)

        assert node.goal_count == 5
        assert node.last_goal_outcome == 'failed'
        assert node.last_goal_xy is None
        assert node.pending_goal_response_since < 0.0
    finally:
        destroy_node(node)
