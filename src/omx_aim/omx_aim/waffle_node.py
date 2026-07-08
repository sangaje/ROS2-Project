#!/usr/bin/env python3
"""Waffle Nav2 client - 단계 H1 (골격).

OMX yolo_node 가 발행한 /omx/nav_goal (VIEW_POSE) 을 Nav2 NavigateToPose
액션으로 와플에게 전달한다. /omx/nav_cancel 받으면 현재 이동 취소.

이 노드는 큐도 정책도 없다. 그냥 "yolo_node 가 시키는 대로" 와플을 옮기는
얇은 어댑터.

토픽:
    Subscribe:
        /omx/nav_goal      PoseStamped   yolo_node 가 계산한 VIEW_POSE
        /omx/nav_cancel    Empty         이동 취소

    Publish:
        /waffle/nav_result String   "succeeded"/"aborted"/"canceled"/"rejected"
        /waffle/status     String   1 Hz 상태 (dry_idle, navigating, ...)
        /waffle/state      String   상태 변경 시

상태:
    IDLE         - 명령 대기
    NAVIGATING   - Nav2 액션 실행 중

액션:
    /navigate_to_pose (config.waffle.nav_action_name)
    타입: nav2_msgs/action/NavigateToPose

실행:
    python3 apps/waffle_node.py            # 실 Nav2 와 통신
    python3 apps/waffle_node.py --dry-run  # Nav2 없이 시뮬레이션
                                           # (nav_goal 받으면 1초 뒤 succeeded)

다음 단계 (H2~):
    yolo_node 가 PATROL/TARGET 처리할 때 현재 위치에서 조준 불가능하면
    VIEW_POSE 계산 → /omx/nav_goal 발행 → 여기서 Nav2 호출.
    TARGET 우선처리 시 /omx/nav_cancel 발행.
"""

from __future__ import annotations

import sys
import time
from enum import Enum
from functools import partial
from typing import Optional

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient

from std_msgs.msg import String, Empty
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped

try:
    from nav2_msgs.action import NavigateToPose
except ImportError:
    print()
    print("ERROR: nav2_msgs 패키지가 없습니다.")
    print("  sudo apt install ros-jazzy-nav2-msgs")
    sys.exit(1)

from action_msgs.msg import GoalStatus

from omx.config import load_config


# ===========================================================
# State
# ===========================================================

class WaffleState(Enum):
    IDLE = "idle"
    NAVIGATING = "navigating"


# ===========================================================
# WaffleNavNode
# ===========================================================

class WaffleNavNode(Node):
    def __init__(self, dry_run: bool = False):
        super().__init__('waffle_nav_node')

        self.cfg = load_config()
        self.dry_run = dry_run

        if self.cfg.waffle is None:
            raise RuntimeError("config.yaml 에 waffle 섹션 필요")

        self.state = WaffleState.IDLE
        self._pending_goal: Optional[PoseStamped] = None
        self._pending_goal_received_wall: Optional[float] = None
        self._last_amcl_pose_wall: Optional[float] = None
        self._amcl_ready = False
        self._amcl_cov_text = "no_amcl_pose"

        # Nav2 액션 핸들
        self.current_goal_handle = None
        self.send_goal_future = None
        self.result_future = None
        # 매 _send_nav_goal 마다 증가 -- 취소된 이전 goal 의 응답/결과
        # 콜백이 늦게 도착해서 그새 시작된 새 goal 의 상태를 덮어쓰는 것을
        # 막는다 (H5.1: waypoint crawl 로 preemption 이 빈번해지면서 필요).
        self._goal_epoch = 0
        self.nav_start_t: float = 0.0

        # dry-run 시뮬레이션 타이머 (one-shot)
        self._dry_timer = None

        # Action client
        action_name = self.cfg.waffle.nav_action_name
        self.action_client = ActionClient(self, NavigateToPose, action_name)
        self.action_name = action_name
        self._nav_server_ready_logged = False
        self._nav_server_timer = None

        self.declare_parameter('require_amcl_ready', True)
        self.declare_parameter('amcl_pose_topic', '/amcl_pose')
        self.declare_parameter('max_amcl_pose_age_sec', 3.0)
        self.declare_parameter('max_xy_covariance', 2.00)
        self.declare_parameter('max_yaw_covariance', 1.50)
        self.declare_parameter('pending_goal_retry_period_sec', 0.5)
        self.declare_parameter('max_pending_goal_age_sec', 60.0)
        self.require_amcl_ready = bool(
            self.get_parameter('require_amcl_ready').value)
        self.amcl_pose_topic = str(
            self.get_parameter('amcl_pose_topic').value)
        self.max_amcl_pose_age_sec = float(
            self.get_parameter('max_amcl_pose_age_sec').value)
        self.max_xy_covariance = float(
            self.get_parameter('max_xy_covariance').value)
        self.max_yaw_covariance = float(
            self.get_parameter('max_yaw_covariance').value)
        self.max_pending_goal_age_sec = float(
            self.get_parameter('max_pending_goal_age_sec').value)
        pending_retry_period = float(
            self.get_parameter('pending_goal_retry_period_sec').value)

        if dry_run:
            self.get_logger().info(
                "[dry-run] Nav2 액션 서버 대기 생략 - 시뮬레이션 모드")
        else:
            self.get_logger().info(
                f"Nav2 액션 서버 '{action_name}' 비동기 확인 시작")
            self._nav_server_timer = self.create_timer(
                1.0, self._check_action_server_ready
            )

        # Subscribers
        self.create_subscription(PoseStamped, '/omx/nav_goal',
                                 self.on_nav_goal, 10)
        self.create_subscription(Empty, '/omx/nav_cancel',
                                 self.on_nav_cancel, 10)
        self.create_subscription(
            PoseWithCovarianceStamped,
            self.amcl_pose_topic,
            self.on_amcl_pose,
            10,
        )

        # Publishers
        self.pub_result = self.create_publisher(
            String, '/waffle/nav_result', 10)
        self.pub_status = self.create_publisher(
            String, '/waffle/status', 10)
        self.pub_state = self.create_publisher(
            String, '/waffle/state', 10)

        # Status timer
        self.create_timer(1.0, self.publish_status)
        self.create_timer(max(0.1, pending_retry_period),
                          self._try_send_pending_goal)

        self.get_logger().info("=" * 50)
        self.get_logger().info("Waffle Nav Node (H1 골격)")
        self.get_logger().info("=" * 50)
        self.get_logger().info(f"Nav2 액션: {action_name}")
        self.get_logger().info(f"Map frame: {self.cfg.waffle.frame}")
        self.get_logger().info(
            f"AMCL gate: require={self.require_amcl_ready} "
            f"topic={self.amcl_pose_topic} "
            f"xy_cov<={self.max_xy_covariance:.3f} "
            f"yaw_cov<={self.max_yaw_covariance:.3f}")
        self.get_logger().info("입력:")
        self.get_logger().info("  /omx/nav_goal    PoseStamped")
        self.get_logger().info("  /omx/nav_cancel  Empty")
        self.get_logger().info("출력:")
        self.get_logger().info("  /waffle/nav_result, /waffle/status, /waffle/state")
        if self.dry_run:
            self.get_logger().info("MODE: dry-run (nav_goal 받으면 1초 후 succeeded)")
        self.get_logger().info("=== Node ready ===")

    # ----- Action server 대기 -----

    def _check_action_server_ready(self):
        """Constructor/startup path를 막지 않고 Nav2 action 서버를 확인."""
        if self.action_client.server_is_ready():
            if not self._nav_server_ready_logged:
                self.get_logger().info("Nav2 액션 서버 연결 확인")
                self._nav_server_ready_logged = True
            if self._nav_server_timer is not None:
                self._nav_server_timer.cancel()
                self._nav_server_timer = None
            return

        self.get_logger().warn(
            f"Nav2 액션 서버 '{self.action_name}' 아직 준비 안 됨. "
            f"nav_goal 수신 시 다시 시도.",
            throttle_duration_sec=5.0,
        )

    # ----- State -----

    def transition(self, new_state: WaffleState):
        if self.state != new_state:
            self.get_logger().info(
                f"State: {self.state.value} -> {new_state.value}")
            self.state = new_state
            # 상태 변경 즉시 발행
            msg = String()
            msg.data = new_state.value
            self.pub_state.publish(msg)

    # ----- Subscribers -----

    def on_amcl_pose(self, msg: PoseWithCovarianceStamped):
        cov = msg.pose.covariance
        xy_cov = max(abs(float(cov[0])), abs(float(cov[7])))
        yaw_cov = abs(float(cov[35]))
        ready = (
            xy_cov <= self.max_xy_covariance
            and yaw_cov <= self.max_yaw_covariance
        )
        was_ready = self._amcl_ready
        self._last_amcl_pose_wall = time.time()
        self._amcl_cov_text = f"xy={xy_cov:.3f}, yaw={yaw_cov:.3f}"
        if ready and not was_ready:
            self.get_logger().info(
                f"AMCL pose accepted ({self._amcl_cov_text})")
        elif not ready and was_ready:
            self.get_logger().warn(
                f"AMCL pose confidence dropped ({self._amcl_cov_text})")
        self._amcl_ready = ready

    def _localization_ready(self) -> bool:
        if self.dry_run or not self.require_amcl_ready:
            return True
        if not self._amcl_ready or self._last_amcl_pose_wall is None:
            return False
        age = time.time() - self._last_amcl_pose_wall
        return age <= self.max_amcl_pose_age_sec

    def _queue_nav_goal(self, msg: PoseStamped, reason: str):
        self._pending_goal = msg
        self._pending_goal_received_wall = time.time()
        self.get_logger().warn(
            f"nav_goal 보류: {reason}. AMCL={self._amcl_cov_text}",
            throttle_duration_sec=2.0,
        )

    def _try_send_pending_goal(self):
        if self._pending_goal is None:
            return
        if self.state == WaffleState.NAVIGATING:
            return

        age = time.time() - (self._pending_goal_received_wall or time.time())
        if age > self.max_pending_goal_age_sec:
            self.get_logger().warn(
                f"pending nav_goal 폐기: {age:.1f}s old")
            self._pending_goal = None
            self._pending_goal_received_wall = None
            self._publish_result("rejected")
            return

        if not self.action_client.server_is_ready():
            self.get_logger().warn(
                f"pending nav_goal 대기: Nav2 액션 서버 '{self.action_name}' "
                "아직 준비 안 됨",
                throttle_duration_sec=5.0,
            )
            return
        if not self._localization_ready():
            self.get_logger().warn(
                f"pending nav_goal 대기: AMCL not ready ({self._amcl_cov_text})",
                throttle_duration_sec=5.0,
            )
            return

        goal_msg = self._pending_goal
        self._pending_goal = None
        self._pending_goal_received_wall = None
        self.get_logger().info(
            f"pending nav_goal 전송: AMCL ready ({self._amcl_cov_text})")
        self._send_nav_goal(goal_msg)

    def on_nav_goal(self, msg: PoseStamped):
        """yolo_node 가 발행한 와플 이동 목표 (VIEW_POSE)."""
        x = msg.pose.position.x
        y = msg.pose.position.y
        frame = msg.header.frame_id or "(none)"
        self.get_logger().info(
            f"nav_goal 수신: ({x:+.2f}, {y:+.2f}) frame={frame}")

        # 이미 NAVIGATING 이면 이전 goal 취소 후 새 goal
        if self.state == WaffleState.NAVIGATING:
            self.get_logger().warn(
                "이미 NAVIGATING - 이전 goal 취소 후 새 goal")
            self._cancel_current_goal()

        # frame_id 비어 있으면 보정
        if not msg.header.frame_id:
            msg.header.frame_id = self.cfg.waffle.frame
            self.get_logger().warn(
                f"frame_id 비어있음 - '{self.cfg.waffle.frame}' 으로 보정")

        if self.dry_run:
            self._dry_run_navigate()
            return

        if not self.action_client.server_is_ready():
            self._queue_nav_goal(
                msg, f"Nav2 액션 서버 '{self.action_name}' 준비 전")
            return
        if not self._localization_ready():
            self._queue_nav_goal(msg, "AMCL localization 준비 전")
            return

        self._send_nav_goal(msg)

    def _send_nav_goal(self, msg: PoseStamped):
        # Nav2 goal 생성 및 전송
        goal = NavigateToPose.Goal()
        goal.pose = msg

        self.nav_start_t = time.time()
        self.transition(WaffleState.NAVIGATING)

        self._goal_epoch += 1
        epoch = self._goal_epoch
        self.send_goal_future = self.action_client.send_goal_async(
            goal, feedback_callback=self._feedback_callback)
        self.send_goal_future.add_done_callback(
            partial(self._goal_response_callback, epoch=epoch))

    def on_nav_cancel(self, msg):
        if self.state == WaffleState.NAVIGATING:
            self.get_logger().info("nav_cancel 수신 - 이동 취소")
            self._cancel_current_goal()
        else:
            self.get_logger().info("nav_cancel 수신했지만 IDLE 상태")

    # ----- Nav2 callbacks -----

    def _feedback_callback(self, feedback_msg):
        """Nav2 피드백 (남은 거리). 로그 너무 많아지지 않게 debug 로."""
        fb = feedback_msg.feedback
        if hasattr(fb, 'distance_remaining'):
            d = fb.distance_remaining
            self.get_logger().debug(f"남은 거리: {d:.2f} m")

    def _goal_response_callback(self, future, epoch: int):
        """Nav2 가 goal accept/reject 응답."""
        if epoch != self._goal_epoch:
            # 이미 취소되고 그 뒤로 새 goal 이 전송된, 낡은 goal 의 응답 --
            # 지금 상태(state/current_goal_handle)를 건드리면 안 됨.
            self.get_logger().debug(
                f"stale goal_response 무시 (epoch={epoch}, "
                f"current={self._goal_epoch})")
            return
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error("Nav2 goal 거부됨")
            self._publish_result("rejected")
            self.transition(WaffleState.IDLE)
            return

        self.current_goal_handle = goal_handle
        self.get_logger().info("Nav2 goal accepted, 이동 시작")

        self.result_future = goal_handle.get_result_async()
        self.result_future.add_done_callback(
            partial(self._result_callback, epoch=epoch))

    def _result_callback(self, future, epoch: int):
        """Nav2 액션 종료."""
        if epoch != self._goal_epoch:
            self.get_logger().debug(
                f"stale nav result 무시 (epoch={epoch}, "
                f"current={self._goal_epoch})")
            return
        status = future.result().status
        elapsed = time.time() - self.nav_start_t

        status_str = {
            GoalStatus.STATUS_SUCCEEDED: "succeeded",
            GoalStatus.STATUS_ABORTED: "aborted",
            GoalStatus.STATUS_CANCELED: "canceled",
        }.get(status, f"unknown_{status}")

        self.get_logger().info(
            f"Nav2 결과: {status_str} ({elapsed:.1f}s 소요)")

        self._publish_result(status_str)
        self.current_goal_handle = None
        self.transition(WaffleState.IDLE)

    def _cancel_current_goal(self):
        """현재 goal cancel 요청. 결과는 _result_callback 에서 CANCELED 로."""
        if self.current_goal_handle is None:
            return
        self.get_logger().info("Goal cancel 요청 보냄")
        self.current_goal_handle.cancel_goal_async()

    # ----- Dry-run 시뮬레이션 -----

    def _dry_run_navigate(self):
        """nav_goal 받으면 1초 후 succeeded."""
        self.transition(WaffleState.NAVIGATING)
        self.nav_start_t = time.time()
        # one-shot timer (첫 호출에서 cancel)
        self._dry_timer = self.create_timer(1.0, self._dry_run_complete)

    def _dry_run_complete(self):
        if self._dry_timer is not None:
            self._dry_timer.cancel()
            self._dry_timer = None
        if self.state == WaffleState.NAVIGATING:
            elapsed = time.time() - self.nav_start_t
            self.get_logger().info(
                f"[dry-run] 도착 시뮬레이션 ({elapsed:.1f}s)")
            self._publish_result("succeeded")
            self.transition(WaffleState.IDLE)

    # ----- Publishers -----

    def _publish_result(self, result_str: str):
        msg = String()
        msg.data = result_str
        self.pub_result.publish(msg)

    def publish_status(self):
        msg = String()
        prefix = "dry_" if self.dry_run else ""
        msg.data = f"{prefix}{self.state.value}"
        self.pub_status.publish(msg)


# ===========================================================
# Entry
# ===========================================================

def main(args=None):
    import argparse
    parser = argparse.ArgumentParser(
        description="Waffle Nav2 client - H1 골격")
    parser.add_argument("--dry-run", action="store_true",
                        help="Nav2 없이 시뮬레이션 (nav_goal 1초 후 succeeded)")
    cli_args, ros_args = parser.parse_known_args()

    rclpy.init(args=ros_args)

    node = None
    try:
        node = WaffleNavNode(dry_run=cli_args.dry_run)
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("\n중단됨.")
    except Exception as e:
        print(f"노드 에러: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
