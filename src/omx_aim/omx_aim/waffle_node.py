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
import json
import math
import time
from enum import Enum
from functools import partial
from typing import Optional

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy

from std_msgs.msg import Bool, String, Empty
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
    IDLE = "IDLE"
    GOAL_RECEIVED = "GOAL_RECEIVED"
    WAITING_SERVER = "WAITING_SERVER"
    WAITING_LOCALIZATION = "WAITING_LOCALIZATION"
    SENDING_GOAL = "SENDING_GOAL"
    WAITING_ACCEPT = "WAITING_ACCEPT"
    NAVIGATING = "NAVIGATING"
    CANCELING = "CANCELING"


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
        self._localization_ready_flag = False
        self._last_nav_goal_meta = {}
        self._current_goal_id = 0
        self._current_goal_type = ''
        self._current_goal_created_at = 0.0
        self._cancel_requested_before_accept = False
        self._last_error = ''
        self._last_feedback_wall: Optional[float] = None
        self._goal_accepted = False

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

        self.declare_parameter('require_amcl_ready', False)
        self.declare_parameter('require_localization_ready', False)
        self.declare_parameter('localization_ready_topic', '/localization_ready')
        self.declare_parameter('require_start_motion', True)
        self.declare_parameter('start_motion_topic', '/fleet/start_motion')
        self.declare_parameter('amcl_pose_topic', '/amcl_pose')
        self.declare_parameter('max_amcl_pose_age_sec', 3.0)
        self.declare_parameter('max_xy_covariance', 2.00)
        self.declare_parameter('max_yaw_covariance', 1.50)
        self.declare_parameter('pending_goal_retry_period_sec', 0.5)
        self.declare_parameter('max_pending_goal_age_sec', 300.0)
        self.declare_parameter('goal_ack_timeout_sec', 5.0)
        self.declare_parameter('cancel_timeout_sec', 3.0)
        self.require_amcl_ready = bool(
            self.get_parameter('require_amcl_ready').value)
        self.require_localization_ready = bool(
            self.get_parameter('require_localization_ready').value)
        self.localization_ready_topic = str(
            self.get_parameter('localization_ready_topic').value)
        self.require_start_motion = bool(
            self.get_parameter('require_start_motion').value)
        self.start_motion_topic = str(
            self.get_parameter('start_motion_topic').value)
        self._start_motion_flag = not self.require_start_motion
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
        self.goal_ack_timeout_sec = max(
            0.5, float(self.get_parameter('goal_ack_timeout_sec').value))
        self.cancel_timeout_sec = max(
            0.5, float(self.get_parameter('cancel_timeout_sec').value))
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
        self.create_subscription(String, '/omx/nav_goal_meta',
                                 self.on_nav_goal_meta, 10)
        self.create_subscription(Empty, '/omx/nav_cancel',
                                 self.on_nav_cancel, 10)
        self.create_subscription(
            PoseWithCovarianceStamped,
            self.amcl_pose_topic,
            self.on_amcl_pose,
            10,
        )
        if self.require_localization_ready:
            latched_qos = QoSProfile(
                depth=1,
                reliability=ReliabilityPolicy.RELIABLE,
                durability=DurabilityPolicy.TRANSIENT_LOCAL,
                history=HistoryPolicy.KEEP_LAST,
            )
            self.create_subscription(
                Bool,
                self.localization_ready_topic,
                self.on_localization_ready,
                latched_qos,
            )
        if self.require_start_motion:
            latched_qos = QoSProfile(
                depth=1,
                reliability=ReliabilityPolicy.RELIABLE,
                durability=DurabilityPolicy.TRANSIENT_LOCAL,
                history=HistoryPolicy.KEEP_LAST,
            )
            self.create_subscription(
                Bool,
                self.start_motion_topic,
                self.on_start_motion,
                latched_qos,
            )

        # Publishers
        self.pub_result = self.create_publisher(
            String, '/waffle/nav_result', 10)
        self.pub_status = self.create_publisher(
            String, '/waffle/status', 10)
        self.pub_state = self.create_publisher(
            String, '/waffle/state', 10)
        self.pub_goal_ack = self.create_publisher(
            String, '/waffle/nav_goal_ack', 10)
        self.pub_diagnostics = self.create_publisher(
            String, '/waffle/nav_diagnostics', 10)

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
        self.get_logger().info(
            f"Localization spin gate: require={self.require_localization_ready} "
            f"topic={self.localization_ready_topic}")
        self.get_logger().info(
            f"Start motion gate: require={self.require_start_motion} "
            f"topic={self.start_motion_topic}")
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
            self.publish_status()
            self._publish_ack(new_state.value)

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

    def on_localization_ready(self, msg: Bool):
        previous = self._localization_ready_flag
        self._localization_ready_flag = bool(msg.data)
        if self._localization_ready_flag and not previous:
            self.get_logger().warn(
                f"localization_ready 수신: {self.localization_ready_topic}=true")
            self._try_send_pending_goal()

    def on_start_motion(self, msg: Bool):
        previous = self._start_motion_flag
        self._start_motion_flag = bool(msg.data)
        if previous and not self._start_motion_flag:
            self._pending_goal = None
            self._pending_goal_received_wall = None
            self._last_error = 'start_motion_false'
            self._cancel_current_goal()
        if self._start_motion_flag != previous:
            self.get_logger().warn(
                f"motion gate 수신: {self.start_motion_topic}={self._start_motion_flag}")
        if self._start_motion_flag and not previous:
            self._try_send_pending_goal()

    def on_nav_goal_meta(self, msg: String):
        try:
            data = json.loads(msg.data)
        except (json.JSONDecodeError, TypeError):
            self.get_logger().warn('OMX_NAV_GOAL_META_IGNORED | malformed')
            return
        if isinstance(data, dict):
            self._last_nav_goal_meta = data

    def _localization_ready(self) -> bool:
        if self.dry_run:
            return True
        if self.require_localization_ready and not self._localization_ready_flag:
            return False
        if not self.require_amcl_ready:
            return True
        if not self._amcl_ready or self._last_amcl_pose_wall is None:
            return False
        age = time.time() - self._last_amcl_pose_wall
        return age <= self.max_amcl_pose_age_sec

    def _start_motion_ready(self) -> bool:
        return self.dry_run or (not self.require_start_motion or self._start_motion_flag)

    def _queue_nav_goal(self, msg: PoseStamped, reason: str):
        self._pending_goal = msg
        self._pending_goal_received_wall = time.time()
        self._last_error = reason
        self.transition(
            WaffleState.WAITING_LOCALIZATION
            if 'localization' in reason.lower() or 'amcl' in reason.lower()
            else WaffleState.WAITING_SERVER
        )
        spin_text = (
            "ready"
            if (not self.require_localization_ready or self._localization_ready_flag)
            else "waiting_spin"
        )
        self.get_logger().warn(
            'WAFFLE_NAV_HOLD | '
            f'reason={reason} goal_id={self._current_goal_id} '
            f'start_motion={self._start_motion_flag}/{self.require_start_motion} '
            f'localization_flag={self._localization_ready_flag} '
            f'localization={spin_text} AMCL={self._amcl_cov_text}',
            throttle_duration_sec=2.0,
        )

    def _try_send_pending_goal(self):
        if self._pending_goal is None:
            return
        if self.state in (
            WaffleState.SENDING_GOAL,
            WaffleState.WAITING_ACCEPT,
            WaffleState.NAVIGATING,
            WaffleState.CANCELING,
        ):
            return

        age = time.time() - (self._pending_goal_received_wall or time.time())
        if age > self.max_pending_goal_age_sec:
            self.get_logger().warn(
                f"pending nav_goal 폐기: {age:.1f}s old")
            self._pending_goal = None
            self._pending_goal_received_wall = None
            self._publish_result("rejected")
            return

        if not self._start_motion_ready():
            self.transition(WaffleState.WAITING_SERVER)
            self.get_logger().warn(
                f"pending nav_goal 폐기: start_motion 준비 전 "
                f"({self.start_motion_topic}=false)",
                throttle_duration_sec=5.0,
            )
            self._pending_goal = None
            self._pending_goal_received_wall = None
            return
        if not self.action_client.server_is_ready():
            self.transition(WaffleState.WAITING_SERVER)
            self.get_logger().warn(
                f"pending nav_goal 대기: Nav2 액션 서버 '{self.action_name}' "
                "아직 준비 안 됨",
                throttle_duration_sec=5.0,
            )
            return
        if not self._localization_ready():
            self.transition(WaffleState.WAITING_LOCALIZATION)
            self.get_logger().warn(
                "pending nav_goal 대기: localization/AMCL not ready "
                f"(spin_ready={self._localization_ready_flag}, "
                f"AMCL={self._amcl_cov_text})",
                throttle_duration_sec=5.0,
            )
            return

        goal_msg = self._pending_goal
        self._pending_goal = None
        self._pending_goal_received_wall = None
        self.get_logger().info(
            f"pending nav_goal 전송: AMCL ready ({self._amcl_cov_text})")
        self._send_nav_goal(goal_msg)

    def _capture_goal_meta(self):
        data = self._last_nav_goal_meta if isinstance(self._last_nav_goal_meta, dict) else {}
        goal_id = data.get('goal_id')
        try:
            self._current_goal_id = int(goal_id)
        except (TypeError, ValueError):
            self._goal_epoch += 1
            self._current_goal_id = self._goal_epoch
        self._current_goal_type = str(data.get('goal_type', 'OMX_VIEW_POSE'))
        try:
            self._current_goal_created_at = float(data.get('created_at', time.time()))
        except (TypeError, ValueError):
            self._current_goal_created_at = time.time()

    @staticmethod
    def _validate_goal(msg: PoseStamped) -> tuple[bool, str]:
        frame = str(msg.header.frame_id or '').strip().lstrip('/')
        if frame and frame != 'map':
            return False, f'unsupported_frame_{frame}'
        p = msg.pose.position
        q = msg.pose.orientation
        values = (p.x, p.y, p.z, q.x, q.y, q.z, q.w)
        if not all(math.isfinite(float(value)) for value in values):
            return False, 'non_finite_pose'
        norm_sq = (
            float(q.x) ** 2
            + float(q.y) ** 2
            + float(q.z) ** 2
            + float(q.w) ** 2
        )
        if norm_sq <= 1.0e-12:
            return False, 'invalid_quaternion'
        return True, ''

    def _publish_ack(self, state: str, *, accepted: bool = False, reason: str = ''):
        msg = String()
        msg.data = json.dumps({
            'goal_id': self._current_goal_id,
            'goal_epoch': self._goal_epoch,
            'state': state,
            'accepted': bool(accepted),
            'reason': reason,
        }, sort_keys=True)
        self.pub_goal_ack.publish(msg)

    def _diagnostics_payload(self) -> dict:
        return {
            'state': self.state.value,
            'goal_id': self._current_goal_id,
            'goal_epoch': self._goal_epoch,
            'action_server_ready': bool(self.dry_run or self.action_client.server_is_ready()),
            'localization_ready': bool(self._localization_ready()),
            'start_motion': bool(self._start_motion_ready()),
            'amcl_ready': bool(self._amcl_ready),
            'goal_pending': self._pending_goal is not None,
            'goal_accepted': bool(self._goal_accepted),
            'cancel_pending': bool(self._cancel_requested_before_accept or self.state == WaffleState.CANCELING),
            'last_error': self._last_error,
        }

    def on_nav_goal(self, msg: PoseStamped):
        """yolo_node 가 발행한 와플 이동 목표 (VIEW_POSE)."""
        x = msg.pose.position.x
        y = msg.pose.position.y
        frame = msg.header.frame_id or "(none)"
        replacing_active_goal = self.state in (
            WaffleState.NAVIGATING,
            WaffleState.WAITING_ACCEPT,
            WaffleState.CANCELING,
        )
        self._capture_goal_meta()
        self.get_logger().info(
            'WAFFLE_NAV_GOAL_RECEIVED | '
            f'id={self._current_goal_id} pose=({x:+.2f},{y:+.2f}) frame={frame}')
        self.transition(WaffleState.GOAL_RECEIVED)

        valid, reason = self._validate_goal(msg)
        if not valid:
            self._last_error = reason
            self.get_logger().error(
                f'OMX_NAV_GOAL_REJECTED_LOCAL | reason={reason} '
                f'goal_id={self._current_goal_id}')
            self._publish_result("rejected")
            self.transition(WaffleState.IDLE)
            return

        # 이미 NAVIGATING 이면 이전 goal 취소 후 새 goal
        if replacing_active_goal:
            self.get_logger().warn(
                "active goal exists - queue replacement after cancel")
            self._pending_goal = msg
            self._pending_goal_received_wall = time.time()
            self._cancel_current_goal()
            self._publish_ack(self.state.value, reason='queued_after_cancel')
            return

        # frame_id 비어 있으면 보정
        if not msg.header.frame_id:
            msg.header.frame_id = self.cfg.waffle.frame
            self.get_logger().warn(
                f"frame_id 비어있음 - '{self.cfg.waffle.frame}' 으로 보정")

        if self.dry_run:
            self._dry_run_navigate()
            return

        if not self._start_motion_ready():
            self._last_error = 'start_motion_false'
            self._publish_result("rejected")
            self.transition(WaffleState.IDLE)
            self.get_logger().warn(
                f'WAFFLE_NAV_GOAL_REJECTED_START_MOTION_FALSE | topic={self.start_motion_topic}'
            )
            return
        if not self.action_client.server_is_ready():
            self._queue_nav_goal(
                msg, f"Nav2 액션 서버 '{self.action_name}' 준비 전")
            return
        if not self._localization_ready():
            self._queue_nav_goal(msg, "localization spin/AMCL 준비 전")
            return

        self._send_nav_goal(msg)

    def _send_nav_goal(self, msg: PoseStamped):
        # Nav2 goal 생성 및 전송
        goal = NavigateToPose.Goal()
        goal.pose = msg

        self.nav_start_t = time.time()
        self._last_feedback_wall = None
        self._goal_accepted = False
        self._cancel_requested_before_accept = False
        self.transition(WaffleState.SENDING_GOAL)

        self._goal_epoch += 1
        epoch = self._goal_epoch
        self.get_logger().info(
            f'WAFFLE_NAV_GOAL_SENT | id={self._current_goal_id} '
            f'action={self.action_name}')
        self.send_goal_future = self.action_client.send_goal_async(
            goal, feedback_callback=self._feedback_callback)
        self.transition(WaffleState.WAITING_ACCEPT)
        self.send_goal_future.add_done_callback(
            partial(self._goal_response_callback, epoch=epoch))

    def on_nav_cancel(self, msg):
        if self.state in (
            WaffleState.WAITING_ACCEPT,
            WaffleState.NAVIGATING,
            WaffleState.SENDING_GOAL,
        ):
            self.get_logger().info("nav_cancel 수신 - 이동 취소")
            self._cancel_current_goal()
        else:
            self.get_logger().info("nav_cancel 수신했지만 IDLE 상태")

    # ----- Nav2 callbacks -----

    def _feedback_callback(self, feedback_msg):
        """Nav2 피드백 (남은 거리). 로그 너무 많아지지 않게 debug 로."""
        self._last_feedback_wall = time.time()
        fb = feedback_msg.feedback
        if hasattr(fb, 'distance_remaining'):
            d = fb.distance_remaining
            self.get_logger().debug(
                f"WAFFLE_NAV_FEEDBACK | id={self._current_goal_id} "
                f"distance_remaining={d:.2f}")

    def _goal_response_callback(self, future, epoch: int):
        """Nav2 가 goal accept/reject 응답."""
        try:
            goal_handle = future.result()
        except Exception as exc:  # noqa: BLE001
            if epoch == self._goal_epoch:
                self._last_error = str(exc)
                self._publish_result("rejected")
                self.transition(WaffleState.IDLE)
            self.get_logger().error(f'WAFFLE_NAV_GOAL_RESPONSE_ERROR | {exc}')
            return
        if epoch != self._goal_epoch:
            # 이미 취소되고 그 뒤로 새 goal 이 전송된, 낡은 goal 의 응답 --
            # 지금 상태(state/current_goal_handle)를 건드리면 안 됨.
            self.get_logger().warn(
                f"stale goal_response 무시 (epoch={epoch}, "
                f"current={self._goal_epoch})")
            if goal_handle.accepted:
                goal_handle.cancel_goal_async()
                self.get_logger().warn(
                    f"WAFFLE_NAV_STALE_ACCEPTED_CANCEL | epoch={epoch}")
            return
        if not goal_handle.accepted:
            self.get_logger().error(
                f"WAFFLE_NAV_GOAL_REJECTED | id={self._current_goal_id}")
            self._publish_result("rejected")
            self.transition(WaffleState.IDLE)
            return

        self.current_goal_handle = goal_handle
        self._goal_accepted = True
        self.get_logger().info(
            f"WAFFLE_NAV_GOAL_ACCEPTED | id={self._current_goal_id}")
        if self._cancel_requested_before_accept:
            self.get_logger().warn(
                f"WAFFLE_NAV_CANCEL_AFTER_ACCEPT | id={self._current_goal_id}")
            self._cancel_current_goal()
            return
        self.transition(WaffleState.NAVIGATING)
        self._publish_ack(WaffleState.NAVIGATING.value, accepted=True)

        self.result_future = goal_handle.get_result_async()
        self.result_future.add_done_callback(
            partial(self._result_callback, epoch=epoch))

    def _result_callback(self, future, epoch: int):
        """Nav2 액션 종료."""
        if epoch != self._goal_epoch:
            self.get_logger().warn(
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
            f"WAFFLE_NAV_RESULT | id={self._current_goal_id} "
            f"result={status_str} elapsed={elapsed:.1f}s")

        self._publish_result(status_str)
        self.current_goal_handle = None
        self.transition(WaffleState.IDLE)
        self._try_send_pending_goal()

    def _cancel_current_goal(self):
        """현재 goal cancel 요청. 결과는 _result_callback 에서 CANCELED 로."""
        if self.state == WaffleState.WAITING_ACCEPT and self.current_goal_handle is None:
            self._cancel_requested_before_accept = True
            self.transition(WaffleState.CANCELING)
            self.get_logger().info(
                f"WAFFLE_NAV_CANCEL_DEFERRED | id={self._current_goal_id}")
            return
        if self.current_goal_handle is None:
            return
        self.transition(WaffleState.CANCELING)
        self.get_logger().info(
            f"WAFFLE_NAV_CANCEL_SENT | id={self._current_goal_id}")
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
        msg.data = f"{prefix}{self.state.value.lower()}"
        self.pub_status.publish(msg)
        diag = String()
        diag.data = json.dumps(self._diagnostics_payload(), sort_keys=True)
        self.pub_diagnostics.publish(diag)


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
