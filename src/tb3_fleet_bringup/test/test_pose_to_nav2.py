import rclpy
from geometry_msgs.msg import PoseStamped

from tb3_fleet_bringup.pose_to_nav2 import PoseToNav2Action


class FakeFuture:
    def __init__(self):
        self.callbacks = []
        self.value = None

    def add_done_callback(self, callback):
        self.callbacks.append(callback)

    def result(self):
        return self.value

    def resolve(self, value):
        self.value = value
        for callback in list(self.callbacks):
            callback(self)


class FakeGoalHandle:
    def __init__(self):
        self.accepted = True
        self.cancel_count = 0
        self.result_future = FakeFuture()

    def cancel_goal_async(self):
        self.cancel_count += 1
        return FakeFuture()

    def get_result_async(self):
        return self.result_future


class FakeActionClient:
    def __init__(self, ready=False):
        self.ready = ready
        self.sent = []

    def server_is_ready(self):
        return self.ready

    def send_goal_async(self, goal, feedback_callback=None):
        future = FakeFuture()
        self.sent.append((goal, feedback_callback, future))
        return future


def goal(x, y):
    message = PoseStamped()
    message.header.frame_id = 'map'
    message.pose.position.x = x
    message.pose.position.y = y
    message.pose.orientation.w = 1.0
    return message


def make_node():
    if not rclpy.ok():
        rclpy.init()
    return PoseToNav2Action()


def destroy_node(node):
    node.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()


def test_latest_goal_is_retained_until_action_server_is_ready():
    node = make_node()
    try:
        client = FakeActionClient(ready=False)
        node.client = client

        node._on_goal_pose(goal(1.0, 2.0))
        assert node.pending_goal is not None
        assert client.sent == []

        client.ready = True
        node._try_send_pending()
        assert node.pending_goal is None
        assert len(client.sent) == 1
    finally:
        destroy_node(node)


def test_out_of_order_action_response_cannot_replace_latest_goal():
    node = make_node()
    try:
        client = FakeActionClient(ready=True)
        node.client = client

        node._on_goal_pose(goal(1.0, 0.0))
        node._on_goal_pose(goal(2.0, 0.0))
        first_future = client.sent[0][2]
        second_future = client.sent[1][2]
        first_handle = FakeGoalHandle()
        second_handle = FakeGoalHandle()

        first_future.resolve(first_handle)
        assert first_handle.cancel_count == 1
        assert node.current_goal_handle is None

        second_future.resolve(second_handle)
        assert node.current_goal_handle is second_handle
        assert node.current_goal_id == 2
    finally:
        destroy_node(node)
