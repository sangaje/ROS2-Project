#!/usr/bin/env python3
"""Calibration calculator for the tracked Waffle kinematics YAML."""

from __future__ import annotations

import argparse
import statistics
from dataclasses import dataclass


@dataclass(frozen=True)
class Sample:
    odom: float
    measured: float


def _positive(value: str) -> float:
    parsed = float(value)
    if parsed <= 0.0:
        raise argparse.ArgumentTypeError('value must be positive')
    return parsed


def _sample(value: str) -> Sample:
    if ':' not in value:
        raise argparse.ArgumentTypeError('sample must be ODOM:MEASURED')
    left, right = value.split(':', 1)
    odom = float(left)
    measured = float(right)
    if odom <= 0.0 or measured <= 0.0:
        raise argparse.ArgumentTypeError('sample values must be positive')
    return Sample(odom=odom, measured=measured)


def _median(values: list[float]) -> float:
    if not values:
        raise ValueError('at least one sample is required')
    return float(statistics.median(values))


def calculate_radius(current_radius: float, samples: list[Sample]) -> tuple[float, list[float]]:
    candidates = [
        current_radius * sample.measured / sample.odom
        for sample in samples
    ]
    return _median(candidates), candidates


def calculate_separation(
    current_separation: float,
    samples: list[Sample],
) -> tuple[float, list[float]]:
    candidates = [
        current_separation * sample.odom / sample.measured
        for sample in samples
    ]
    return _median(candidates), candidates


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            'Compute tracked Waffle effective wheel radius or separation from '
            'odom/external measurements. Use repeated forward/backward or '
            'CW/CCW samples and put each as ODOM:MEASURED.'
        )
    )
    sub = parser.add_subparsers(dest='mode', required=True)

    straight = sub.add_parser('straight', help='wheel radius from distance samples')
    straight.add_argument('--current-radius', type=_positive, required=True)
    straight.add_argument(
        '--sample',
        action='append',
        type=_sample,
        required=True,
        help='distance sample as odom_distance:measured_distance, meters',
    )

    rotate = sub.add_parser('rotate', help='wheel separation from rotation samples')
    rotate.add_argument('--current-separation', type=_positive, required=True)
    rotate.add_argument(
        '--sample',
        action='append',
        type=_sample,
        required=True,
        help='rotation sample as odom_rotation:measured_rotation, same units',
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.mode == 'straight':
        value, candidates = calculate_radius(args.current_radius, args.sample)
        label = 'effective_wheel_radius'
    else:
        value, candidates = calculate_separation(args.current_separation, args.sample)
        label = 'effective_track_separation'

    print(f'{label}: {value:.6f}')
    print('candidates:')
    for idx, candidate in enumerate(candidates, start=1):
        print(f'  {idx}: {candidate:.6f}')
    print('yaml patch:')
    print(f'  {label}: {value:.6f}')
    if label == 'effective_wheel_radius':
        print('  /**.turtlebot3_node.ros__parameters.wheels.radius: '
              f'{value:.6f}')
    else:
        print('  /**.turtlebot3_node.ros__parameters.wheels.separation: '
              f'{value:.6f}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
