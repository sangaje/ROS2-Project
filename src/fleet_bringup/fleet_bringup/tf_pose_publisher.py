#!/usr/bin/env python3

from __future__ import annotations

import math
import time
from typing import Iterable, Optional

import rclpy
from rclpy.node import Node
from rclpy.exceptions import ParameterAlreadyDeclaredException
from geometry_msgs.msg import PoseStamped
from tf2_ros import Buffer, TransformListener, TransformException


def _safe_declare(node: Node, name: str, default):
    try:
        node.declare_parameter(name, default)
    except ParameterAlreadyDeclaredException:
        pass
    return node.get_parameter(name).value


def _frame_candidates(raw: object, primary: str) -> list[str]:
    if isinstance(raw, str):
        values: Iterable[object] = raw.split(',')
    else:
        values = raw if isinstance(raw, Iterable) else []
    frames = []
    for value in values:
        frame = str(value).strip().strip('/')
        if frame and frame not in frames:
            frames.append(frame)
    primary = primary.strip().strip('/')
    if primary and primary not in frames:
        frames.insert(0, primary)
    return frames or ['base_footprint']


def _yaw_from_xyzw(x: float, y: float, z: float, w: float) -> float:
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _angle_delta(a: float, b: float) -> float:
    return math.atan2(math.sin(a - b), math.cos(a - b))


class TfPosePublisher(Node):
    """Publish PoseStamped from a TF transform.

    v44 purpose:
      - Leader pose under Cartographer: publish map->base_footprint, not dead-reckoned odom.
      - Burger debug pose: publish map->base_footprint using its local map->odom publisher.
    """

    def __init__(self) -> None:
        super().__init__('tf_pose_publisher_v44')
        _safe_declare(self, 'use_sim_time', True)
        self.output_topic = self._abs(str(_safe_declare(self, 'output_topic', '/leader_pose')))
        self.target_frame = str(_safe_declare(self, 'target_frame', 'map'))
        self.source_frame = str(_safe_declare(self, 'source_frame', 'base_footprint'))
        self.source_frame_candidates = _frame_candidates(
            _safe_declare(
                self,
                'source_frame_candidates',
                [self.source_frame, 'base_link'],
            ),
            self.source_frame,
        )
        self.publish_rate_hz = max(1.0, float(_safe_declare(self, 'publish_rate_hz', 10.0)))
        self.timeout_sec = max(0.01, float(_safe_declare(self, 'timeout_sec', 0.05)))
        self.log_every_n = max(0, int(_safe_declare(self, 'log_every_n', 200)))
        self.freeze_when_stationary = bool(
            _safe_declare(self, 'freeze_when_stationary', False)
        )
        self.stationary_target_frame = str(
            _safe_declare(self, 'stationary_target_frame', 'odom')
        )
        self.stationary_linear_threshold_m = max(
            0.0, float(_safe_declare(self, 'stationary_linear_threshold_m', 0.02))
        )
        self.stationary_angular_threshold_rad = max(
            0.0,
            float(_safe_declare(self, 'stationary_angular_threshold_rad', 0.035)),
        )
        self.stationary_freeze_warmup_sec = max(
            0.0, float(_safe_declare(self, 'stationary_freeze_warmup_sec', 10.0))
        )
        self.map_jump_filter_enabled = bool(
            _safe_declare(self, 'map_jump_filter_enabled', False)
        )
        self.map_jump_min_allowed_m = max(
            0.0, float(_safe_declare(self, 'map_jump_min_allowed_m', 0.20))
        )
        self.map_jump_odom_scale = max(
            1.0, float(_safe_declare(self, 'map_jump_odom_scale', 4.0))
        )
        self.map_jump_slop_m = max(
            0.0, float(_safe_declare(self, 'map_jump_slop_m', 0.12))
        )

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.pub = self.create_publisher(PoseStamped, self.output_topic, 10)
        self.count = 0
        self.miss_count = 0
        self.first_pose_logged = False
        self._start_wall = time.monotonic()
        self._last_motion_pose: Optional[tuple[float, float, float]] = None
        self._last_accepted_pose: Optional[PoseStamped] = None
        self._freeze_count = 0
        self._last_freeze_log_wall = 0.0
        self.create_timer(1.0 / self.publish_rate_hz, self._tick)
        self.get_logger().info(
            f'V44_TF_POSE_PUBLISHER_READY | target={self.target_frame} '
            f'sources={self.source_frame_candidates} -> {self.output_topic} '
            f'rate={self.publish_rate_hz:.1f}Hz '
            f'freeze_stationary={self.freeze_when_stationary} '
            f'map_jump_filter={self.map_jump_filter_enabled}'
        )

    @staticmethod
    def _abs(topic: str) -> str:
        topic = topic.strip()
        return topic if topic.startswith('/') else '/' + topic

    def _tick(self) -> None:
        last_error = None
        selected_source = None
        tf = None
        for source_frame in self.source_frame_candidates:
            try:
                tf = self.tf_buffer.lookup_transform(
                    self.target_frame,
                    source_frame,
                    rclpy.time.Time(),
                    timeout=rclpy.duration.Duration(seconds=self.timeout_sec),
                )
                selected_source = source_frame
                break
            except TransformException as exc:
                last_error = exc

        if tf is None or selected_source is None:
            self.miss_count += 1
            if self.miss_count <= 5 or (self.log_every_n and self.miss_count % self.log_every_n == 0):
                self.get_logger().warn(
                    f'V44_TF_POSE_WAIT | target={self.target_frame} '
                    f'sources={self.source_frame_candidates} '
                    f'miss={self.miss_count} reason={last_error}'
                )
            return

        motion_pose = None
        if self.freeze_when_stationary:
            motion_pose = self._lookup_motion_pose(selected_source)

        self._publish_pose(tf, selected_source, motion_pose)

    def _lookup_motion_pose(
        self,
        selected_source: str,
    ) -> Optional[tuple[float, float, float]]:
        try:
            tf = self.tf_buffer.lookup_transform(
                self.stationary_target_frame,
                selected_source,
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=self.timeout_sec),
            )
        except TransformException as exc:
            now = time.monotonic()
            if now - self._last_freeze_log_wall > 5.0:
                self._last_freeze_log_wall = now
                self.get_logger().warn(
                    'V44_TF_POSE_STATIONARY_FILTER_WAIT | '
                    f'target={self.stationary_target_frame} '
                    f'source={selected_source} reason={exc}'
                )
            return None

        return self._motion_pose_from_tf(tf)

    @staticmethod
    def _motion_pose_from_tf(tf) -> tuple[float, float, float]:
        q = tf.transform.rotation
        return (
            float(tf.transform.translation.x),
            float(tf.transform.translation.y),
            _yaw_from_xyzw(float(q.x), float(q.y), float(q.z), float(q.w)),
        )

    @staticmethod
    def _copy_pose_with_header(source: PoseStamped, header) -> PoseStamped:
        msg = PoseStamped()
        msg.header = header
        msg.pose.position.x = source.pose.position.x
        msg.pose.position.y = source.pose.position.y
        msg.pose.position.z = source.pose.position.z
        msg.pose.orientation.x = source.pose.orientation.x
        msg.pose.orientation.y = source.pose.orientation.y
        msg.pose.orientation.z = source.pose.orientation.z
        msg.pose.orientation.w = source.pose.orientation.w
        return msg

    def _select_pose_for_publish(
        self,
        candidate: PoseStamped,
        motion_pose: Optional[tuple[float, float, float]],
    ) -> tuple[PoseStamped, bool]:
        if not self.freeze_when_stationary or motion_pose is None:
            self._last_accepted_pose = self._copy_pose_with_header(
                candidate,
                candidate.header,
            )
            if motion_pose is not None:
                self._last_motion_pose = motion_pose
            return candidate, False

        warmup_done = (
            time.monotonic() - self._start_wall
        ) >= self.stationary_freeze_warmup_sec
        if (
            not warmup_done
            or self._last_motion_pose is None
            or self._last_accepted_pose is None
        ):
            self._last_motion_pose = motion_pose
            self._last_accepted_pose = self._copy_pose_with_header(
                candidate,
                candidate.header,
            )
            return candidate, False

        dx = motion_pose[0] - self._last_motion_pose[0]
        dy = motion_pose[1] - self._last_motion_pose[1]
        dyaw = _angle_delta(motion_pose[2], self._last_motion_pose[2])
        odom_delta = math.hypot(dx, dy)
        stationary = (
            odom_delta <= self.stationary_linear_threshold_m
            and abs(dyaw) <= self.stationary_angular_threshold_rad
        )
        if stationary:
            self._freeze_count += 1
            now = time.monotonic()
            if self._freeze_count <= 3 or now - self._last_freeze_log_wall > 5.0:
                self._last_freeze_log_wall = now
                self.get_logger().info(
                    'V44_TF_POSE_STATIONARY_FREEZE | '
                    f'topic={self.output_topic} count={self._freeze_count} '
                    f'odom_delta=({math.hypot(dx, dy):.3f}m,'
                    f'{abs(dyaw):.3f}rad)'
                )
            return self._copy_pose_with_header(
                self._last_accepted_pose,
                candidate.header,
            ), True

        if self.map_jump_filter_enabled:
            map_delta = math.hypot(
                candidate.pose.position.x - self._last_accepted_pose.pose.position.x,
                candidate.pose.position.y - self._last_accepted_pose.pose.position.y,
            )
            allowed_delta = max(
                self.map_jump_min_allowed_m,
                odom_delta * self.map_jump_odom_scale + self.map_jump_slop_m,
            )
            if map_delta > allowed_delta:
                self._freeze_count += 1
                now = time.monotonic()
                if self._freeze_count <= 3 or now - self._last_freeze_log_wall > 5.0:
                    self._last_freeze_log_wall = now
                    self.get_logger().warn(
                        'V44_TF_POSE_MAP_JUMP_REJECT | '
                        f'topic={self.output_topic} count={self._freeze_count} '
                        f'map_delta={map_delta:.3f}m '
                        f'odom_delta={odom_delta:.3f}m '
                        f'allowed_delta={allowed_delta:.3f}m'
                    )
                return self._copy_pose_with_header(
                    self._last_accepted_pose,
                    candidate.header,
                ), True

        self._freeze_count = 0
        self._last_motion_pose = motion_pose
        self._last_accepted_pose = self._copy_pose_with_header(
            candidate,
            candidate.header,
        )
        return candidate, False

    def _publish_pose(
        self,
        tf,
        selected_source: str,
        motion_pose: Optional[tuple[float, float, float]],
    ) -> None:
        if self.miss_count:
            self.get_logger().info(
                f'V44_TF_POSE_RECOVERED | target={self.target_frame} '
                f'source={selected_source} topic={self.output_topic}'
            )
            self.miss_count = 0

        msg = PoseStamped()
        msg.header = tf.header
        msg.header.frame_id = self.target_frame
        msg.pose.position.x = tf.transform.translation.x
        msg.pose.position.y = tf.transform.translation.y
        msg.pose.position.z = tf.transform.translation.z
        msg.pose.orientation = tf.transform.rotation
        msg, frozen = self._select_pose_for_publish(msg, motion_pose)
        self.pub.publish(msg)

        self.count += 1
        if not self.first_pose_logged:
            self.first_pose_logged = True
            tag = (
                'LEADER_LOCAL_POSE_FIRST_RX'
                if self.output_topic == '/leader_pose'
                else 'TF_POSE_FIRST_RX'
            )
            self.get_logger().info(
                f'{tag} | topic={self.output_topic} target={self.target_frame} '
                f'source={selected_source} frame_id={msg.header.frame_id} '
                f'xy=({msg.pose.position.x:.2f},{msg.pose.position.y:.2f})'
            )
        if self.log_every_n and self.count % self.log_every_n == 0:
            q = msg.pose.orientation
            yaw = _yaw_from_xyzw(q.x, q.y, q.z, q.w)
            self.get_logger().info(
                f'V44_TF_POSE | topic={self.output_topic} source={selected_source} '
                f'n={self.count} xy=({msg.pose.position.x:.2f},{msg.pose.position.y:.2f}) '
                f'yaw={yaw:.2f} stationary_freeze={frozen}'
            )

def main() -> None:
    rclpy.init()
    node = TfPosePublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.destroy_node()
        except KeyboardInterrupt:
            pass
        if rclpy.ok():
            try:
                rclpy.shutdown()
            except KeyboardInterrupt:
                pass


if __name__ == '__main__':
    main()
