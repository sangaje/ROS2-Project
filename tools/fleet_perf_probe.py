#!/usr/bin/env python3
"""Collect lightweight fleet network/DDS/Nav2 diagnostics for one run stage."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import time
from pathlib import Path
from typing import Any


def run(cmd: list[str], timeout: float = 3.0) -> dict[str, Any]:
    started = time.time()
    try:
        proc = subprocess.run(
            cmd,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout,
            check=False,
        )
        return {
            'cmd': cmd,
            'returncode': proc.returncode,
            'elapsed_sec': time.time() - started,
            'output': proc.stdout[-8000:],
        }
    except Exception as exc:  # noqa: BLE001
        return {
            'cmd': cmd,
            'returncode': -1,
            'elapsed_sec': time.time() - started,
            'error': str(exc),
        }


def ping_summary(host: str, count: int) -> dict[str, Any]:
    result = run(['ping', '-n', '-c', str(count), '-i', '0.2', host], timeout=max(5.0, count * 0.4))
    output = str(result.get('output', ''))
    values = [float(match) for match in re.findall(r'time=([0-9.]+) ms', output)]
    values.sort()
    def pct(p: float) -> float | None:
        if not values:
            return None
        idx = min(len(values) - 1, max(0, int(round((len(values) - 1) * p))))
        return values[idx]
    return {
        'host': host,
        'sent': count,
        'received': len(values),
        'p50_ms': pct(0.50),
        'p95_ms': pct(0.95),
        'p99_ms': pct(0.99),
        'max_ms': max(values) if values else None,
        'raw': result,
    }


def snapshot(args) -> dict[str, Any]:
    topics = [
        '/map',
        '/risk/risk_map',
        '/leader_pose',
        '/member_pose',
        '/burger_pose',
        '/fleet/video_ready',
        '/system/ready',
        '/leader_shadow/state',
        '/fleet/field_robot_status',
    ]
    actions = {
        'stage': args.stage,
        'stamp': time.time(),
        'ping': ping_summary(args.ping_host, args.ping_count) if args.ping_host else None,
        'ss_summary': run(['ss', '-s']),
        'ip_link': run(['ip', '-s', 'link', 'show', args.iface]) if args.iface else None,
        'processes': run(['bash', '-lc', "ps -eo pid,ppid,comm,pcpu,pmem,nlwp,args | grep -E 'ros2|python|domain_bridge|flask|yolo|nav2|cartographer' | grep -v grep"]),
        'ros_nodes': run(['ros2', 'node', 'list'], timeout=5.0),
        'ros_actions': run(['ros2', 'action', 'info', '/navigate_to_pose'], timeout=5.0),
        'topics': {},
    }
    for topic in topics:
        actions['topics'][topic] = run(['ros2', 'topic', 'info', '-v', topic], timeout=5.0)
    return actions


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', required=True, help='A-K stage label or short description')
    parser.add_argument('--duration-sec', type=float, default=60.0)
    parser.add_argument('--interval-sec', type=float, default=5.0)
    parser.add_argument('--ping-host', default='')
    parser.add_argument('--ping-count', type=int, default=20)
    parser.add_argument('--iface', default='')
    parser.add_argument('--out', default='fleet_perf_probe.jsonl')
    args = parser.parse_args()

    deadline = time.time() + max(1.0, args.duration_sec)
    output = Path(args.out).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('a', encoding='utf-8') as handle:
        while time.time() < deadline:
            handle.write(json.dumps(snapshot(args), sort_keys=True) + '\n')
            handle.flush()
            time.sleep(max(1.0, args.interval_sec))
    print(f'FLEET_PERF_PROBE_DONE | stage={args.stage} out={output}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
