import rclpy
from action_msgs.msg import GoalStatus
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Bool

from fleet_bringup.pose_to_nav2 import PoseToNav2Action


class FakeFuture:
    def __init__(self):
        self.callbacks = []
        self.value = None
        self.error = None

    def add_done_callback(self, callback):
        self.callbacks.append(callback)

    def result(self):
        if self.error is not None:
            raise self.error
        return self.value

    def resolve(self, value):
        self.value = value
        for callback in list(self.callbacks):
            callback(self)

    def reject(self, error):
        self.error = error
        for callback in list(self.callbacks):
            callback(self)


class FakeGoalHandle:
    def __init__(self, accepted=True):
        self.accepted = accepted
        self.cancel_count = 0
        self.cancel_future = FakeFuture()
        self.result_future = FakeFuture()

    def cancel_goal_async(self):
        self.cancel_count += 1
        return self.cancel_future

    def get_result_async(self):
        return self.result_future


class FakeResult:
    def __init__(self, status):
        self.status = status


class FakeCancelResponse:
    return_code = 0
    goals_canceling = [object()]


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


def test_failed_accepted_goal_is_requeued_with_retry_limit():
    node = make_node()
    try:
        client = FakeActionClient(ready=True)
        node.client = client
        node.result_retry_sec = 0.5
        node.max_result_retries = 1

        node._on_goal_pose(goal(1.0, 0.0))
        first_handle = FakeGoalHandle()
        client.sent[0][2].resolve(first_handle)
        first_handle.result_future.resolve(
            FakeResult(GoalStatus.STATUS_ABORTED)
        )

        assert node.pending_goal is not None
        assert node.failed_result_retries == 1

        node.retry_not_before = -1.0e9
        node._try_send_pending()
        assert node.pending_goal is None
        assert len(client.sent) == 2

        second_handle = FakeGoalHandle()
        client.sent[1][2].resolve(second_handle)
        second_handle.result_future.resolve(
            FakeResult(GoalStatus.STATUS_ABORTED)
        )

        assert node.pending_goal is None
        assert node.failed_result_retries == 1
    finally:
        destroy_node(node)


def test_explicit_cancel_invalidates_goal_accepted_after_cancel_race():
    node = make_node()
    try:
        client = FakeActionClient(ready=True)
        node.client = client

        node._on_goal_pose(goal(1.0, 0.0))
        send_future = client.sent[0][2]
        node._on_cancel(Bool(data=True))

        late_handle = FakeGoalHandle()
        send_future.resolve(late_handle)

        assert node.pending_goal is None
        assert node.current_goal_handle is None
        assert late_handle.cancel_count == 1
        assert len(late_handle.cancel_future.callbacks) == 1
        late_handle.cancel_future.resolve(FakeCancelResponse())
    finally:
        destroy_node(node)


def test_localization_falling_false_invalidates_inflight_send():
    node = make_node()
    try:
        client = FakeActionClient(ready=True)
        node.client = client
        node.localization_ready = True

        node._on_goal_pose(goal(1.0, 0.0))
        send_future = client.sent[0][2]
        node._on_localization_ready(Bool(data=False))

        late_handle = FakeGoalHandle()
        send_future.resolve(late_handle)

        assert node.localization_ready is False
        assert node.current_goal_handle is None
        assert late_handle.cancel_count == 1
    finally:
        destroy_node(node)


def test_canceled_result_is_never_requeued():
    node = make_node()
    try:
        client = FakeActionClient(ready=True)
        node.client = client

        node._on_goal_pose(goal(1.0, 0.0))
        handle = FakeGoalHandle()
        client.sent[0][2].resolve(handle)
        handle.result_future.resolve(FakeResult(GoalStatus.STATUS_CANCELED))

        assert node.pending_goal is None
        assert node.retry_attempts['result'] == 0
    finally:
        destroy_node(node)


def test_rejected_goal_retries_are_bounded_with_exponential_backoff():
    node = make_node()
    try:
        client = FakeActionClient(ready=True)
        node.client = client
        node.max_send_retries = 2
        node.rejected_retry_sec = 0.5

        node._on_goal_pose(goal(1.0, 0.0))
        client.sent[0][2].resolve(FakeGoalHandle(accepted=False))
        first_retry_at = node.retry_not_before
        assert node.pending_goal is not None
        assert node.retry_attempts['rejected'] == 1

        node.retry_not_before = -1.0e9
        node._try_send_pending()
        client.sent[1][2].resolve(FakeGoalHandle(accepted=False))
        second_retry_at = node.retry_not_before
        assert node.pending_goal is not None
        assert node.retry_attempts['rejected'] == 2
        assert second_retry_at > first_retry_at

        node.retry_not_before = -1.0e9
        node._try_send_pending()
        client.sent[2][2].resolve(FakeGoalHandle(accepted=False))
        assert node.pending_goal is None
        assert node.retry_attempts['rejected'] == 2
        assert len(client.sent) == 3
    finally:
        destroy_node(node)


def test_send_exception_retry_is_replaced_by_newer_goal():
    node = make_node()
    try:
        client = FakeActionClient(ready=True)
        node.client = client

        node._on_goal_pose(goal(1.0, 0.0))
        client.sent[0][2].reject(RuntimeError('transport failed'))
        assert node.pending_goal.pose.position.x == 1.0
        assert node.retry_attempts['send_exception'] == 1

        node._on_goal_pose(goal(2.0, 0.0))

        assert node.pending_goal is None
        assert len(client.sent) == 2
        assert client.sent[1][0].pose.pose.position.x == 2.0
        assert node.retry_attempts['send_exception'] == 0
    finally:
        destroy_node(node)
