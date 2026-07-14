#!/usr/bin/env python3
"""Minimal ACTIVE_SCOUT RL probe.

This intentionally bypasses launch, Cartographer, TF, scan, odom, dashboard,
and the confidence-map updater. It only verifies that the deployment SAC
checkpoint loads, accepts the frozen observation contract, predicts actions,
and optionally publishes a TwistStamped command to /cmd_vel.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
for relative in ('src/system_bringup', 'src/turtlebot3_rl_training'):
    source_path = str(ROOT / relative)
    if source_path not in sys.path:
        sys.path.insert(0, source_path)

from system_bringup.rl_policy_contract import (  # noqa: E402
    PolicyContractError,
    load_contract,
    load_deployment_model,
    probe_checkpoint,
)


def _now_ms(start: float) -> float:
    return (time.perf_counter() - start) * 1000.0


def _shape(contract: dict[str, Any], key: str) -> tuple[int, ...]:
    return tuple(int(value) for value in contract['observation_contract'][key]['shape'])


def build_synthetic_observation(contract: dict[str, Any]) -> dict[str, np.ndarray]:
    """Create a simple, valid observation that looks like open space."""
    map_shape = _shape(contract, 'map')
    map_seq_shape = _shape(contract, 'map_seq')
    vector_shape = _shape(contract, 'vector')
    seq_shape = _shape(contract, 'seq')
    lidar_bins = int(contract['observation_contract']['vector']['lidar_bins'])

    map_obs = np.zeros(map_shape, dtype=np.float32)
    if map_shape[0] > 1:
        map_obs[1, :, :] = 1.0
    if map_shape[0] > 3:
        map_obs[3, :, :] = 0.5

    vector = np.zeros(vector_shape, dtype=np.float32)
    vector[:lidar_bins] = 1.0
    if vector.shape[0] >= lidar_bins + 3:
        vector[lidar_bins:lidar_bins + 3] = np.asarray([1.0, 0.0, 0.5], dtype=np.float32)

    map_seq = np.repeat(map_obs[None, :, :, :], map_seq_shape[0], axis=0).astype(np.float32)
    seq = np.repeat(vector[None, :], seq_shape[0], axis=0).astype(np.float32)

    return {
        'map': map_obs,
        'map_seq': map_seq,
        'seq': seq,
        'vector': vector,
    }


def _clip_action(action: np.ndarray, contract: dict[str, Any]) -> tuple[float, float]:
    low = np.asarray(contract['action_contract']['low'], dtype=np.float32)
    high = np.asarray(contract['action_contract']['high'], dtype=np.float32)
    clipped = np.clip(action.astype(np.float32), low, high)
    return float(clipped[0]), float(clipped[1])


def _publish_twist_stamped(
    topic: str,
    linear_x: float,
    angular_z: float,
    publish_seconds: float,
    rate_hz: float,
) -> int:
    import rclpy
    from geometry_msgs.msg import TwistStamped

    rclpy.init(args=None)
    node = rclpy.create_node('scout_rl_minimal_probe')
    publisher = node.create_publisher(TwistStamped, topic, 10)
    period = 1.0 / max(rate_hz, 0.1)
    deadline = time.monotonic() + max(publish_seconds, 0.0)
    count = 0

    try:
        while time.monotonic() < deadline:
            msg = TwistStamped()
            msg.header.stamp = node.get_clock().now().to_msg()
            msg.header.frame_id = 'base_footprint'
            msg.twist.linear.x = linear_x
            msg.twist.angular.z = angular_z
            publisher.publish(msg)
            rclpy.spin_once(node, timeout_sec=0.0)
            count += 1
            time.sleep(period)

        for _ in range(3):
            stop = TwistStamped()
            stop.header.stamp = node.get_clock().now().to_msg()
            stop.header.frame_id = 'base_footprint'
            publisher.publish(stop)
            rclpy.spin_once(node, timeout_sec=0.0)
            time.sleep(0.05)
    finally:
        node.destroy_node()
        rclpy.shutdown()

    return count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--iterations', type=int, default=5)
    parser.add_argument('--publish-seconds', type=float, default=0.0)
    parser.add_argument('--publish-rate-hz', type=float, default=5.0)
    parser.add_argument('--cmd-vel-topic', default='/cmd_vel')
    parser.add_argument('--allow-motion', action='store_true')
    parser.add_argument(
        '--fixed-cmd',
        nargs=2,
        type=float,
        metavar=('LINEAR_X', 'ANGULAR_Z'),
        help='Publish this explicit command instead of the policy action.',
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    start = time.perf_counter()
    print(
        'SCOUT_RL_MINIMAL_PROBE_START | '
        f'root={ROOT} iterations={args.iterations} publish_seconds={args.publish_seconds}',
        flush=True,
    )

    try:
        contract = load_contract()
        model_start = time.perf_counter()
        model = load_deployment_model(contract)
        probe = probe_checkpoint(contract, model=model)
    except PolicyContractError as exc:
        print(f'SCOUT_RL_MINIMAL_PROBE_FAILED | stage=model_load error={exc}', flush=True)
        return 2

    print(
        'SCOUT_RL_MINIMAL_MODEL_READY | '
        f'load_ms={(time.perf_counter() - model_start) * 1000.0:.1f} '
        + json.dumps({'checkpoint': probe['checkpoint']}, sort_keys=True),
        flush=True,
    )

    observation = build_synthetic_observation(contract)
    action = np.zeros(2, dtype=np.float32)
    iterations = max(1, int(args.iterations))
    for index in range(iterations):
        predict_start = time.perf_counter()
        raw_action, _ = model.predict(observation, deterministic=True)
        predict_ms = (time.perf_counter() - predict_start) * 1000.0
        action = np.asarray(raw_action, dtype=np.float32).reshape(-1)[:2]
        linear_x, angular_z = _clip_action(action, contract)
        print(
            'SCOUT_RL_MINIMAL_PREDICT | '
            f'iter={index + 1} predict_ms={predict_ms:.1f} '
            f'action_linear_x={linear_x:.4f} action_angular_z={angular_z:.4f}',
            flush=True,
        )

    if args.fixed_cmd is not None:
        linear_x, angular_z = float(args.fixed_cmd[0]), float(args.fixed_cmd[1])
    else:
        linear_x, angular_z = _clip_action(action, contract)

    if args.publish_seconds > 0.0:
        if not args.allow_motion:
            print(
                'SCOUT_RL_MINIMAL_PUBLISH_SKIPPED | '
                'reason=allow_motion_not_set use="--allow-motion --publish-seconds N"',
                flush=True,
            )
            return 3
        count = _publish_twist_stamped(
            args.cmd_vel_topic,
            linear_x,
            angular_z,
            args.publish_seconds,
            args.publish_rate_hz,
        )
        print(
            'SCOUT_RL_MINIMAL_CMD_PUBLISHED | '
            f'topic={args.cmd_vel_topic} count={count} '
            f'linear_x={linear_x:.4f} angular_z={angular_z:.4f}',
            flush=True,
        )

    print(f'SCOUT_RL_MINIMAL_PROBE_PASS | elapsed_ms={_now_ms(start):.1f}', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
