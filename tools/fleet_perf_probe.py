#!/usr/bin/env python3
"""Collect lightweight fleet network/DDS/Nav2 diagnostics for one run stage."""

from __future__ import annotations

import argparse
import csv
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


def parse_ip_link_mbps(output: str, interval_sec: float) -> dict[str, float | None]:
    byte_lines = []
    capture_next = False
    for line in output.splitlines():
        stripped = line.strip()
        if stripped.startswith('RX:') or stripped.startswith('TX:'):
            capture_next = True
            continue
        if capture_next:
            parts = stripped.split()
            if parts and parts[0].isdigit():
                byte_lines.append(int(parts[0]))
            capture_next = False
    if len(byte_lines) < 4 or interval_sec <= 0.0:
        return {'rx_mbps': None, 'tx_mbps': None}
    rx_delta = max(0, byte_lines[2] - byte_lines[0])
    tx_delta = max(0, byte_lines[3] - byte_lines[1])
    return {
        'rx_mbps': rx_delta * 8.0 / interval_sec / 1_000_000.0,
        'tx_mbps': tx_delta * 8.0 / interval_sec / 1_000_000.0,
    }


def ip_link_delta(iface: str, interval_sec: float) -> dict[str, Any] | None:
    if not iface:
        return None
    before = run(['ip', '-s', 'link', 'show', iface], timeout=2.0)
    time.sleep(max(0.2, min(5.0, interval_sec)))
    after = run(['ip', '-s', 'link', 'show', iface], timeout=2.0)
    return {
        'before': before,
        'after': after,
        **parse_ip_link_mbps(
            str(before.get('output', '')) + str(after.get('output', '')),
            max(0.2, min(5.0, interval_sec)),
        ),
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


def ros_topic_bw(topic: str, duration_sec: float) -> dict[str, Any]:
    timeout = max(3.0, duration_sec + 1.5)
    result = run(
        ['timeout', f'{duration_sec:.1f}', 'ros2', 'topic', 'bw', topic],
        timeout=timeout,
    )
    output = str(result.get('output', ''))
    match = re.search(r'average:\s*([0-9.]+)\s*([KMG]?B/s)', output)
    bps = None
    if match:
        value = float(match.group(1))
        unit = match.group(2)
        scale = {'B/s': 1.0, 'KB/s': 1000.0, 'MB/s': 1_000_000.0, 'GB/s': 1_000_000_000.0}.get(unit, 1.0)
        bps = value * scale
    return {
        'topic': topic,
        'bytes_per_sec': bps,
        'mbps': (bps * 8.0 / 1_000_000.0) if bps is not None else None,
        'raw': result,
    }


def process_counts(process_output: str) -> dict[str, int]:
    lines = [line for line in process_output.splitlines() if line.strip()]
    return {
        'domain_bridge': sum('domain_bridge' in line for line in lines),
        'cartographer': sum('cartographer' in line for line in lines),
        'nav2': sum('nav2_' in line or 'bt_navigator' in line for line in lines),
        'camera_sender': sum('opencv_camera_to_flask_yolo' in line for line in lines),
        'flask_yolo': sum('flask_yolo_server' in line for line in lines),
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
        'ip_link_delta': ip_link_delta(args.iface, args.link_interval_sec),
        'processes': run(['bash', '-lc', "ps -eo pid,ppid,comm,pcpu,pmem,nlwp,args | grep -E 'ros2|python|domain_bridge|flask|yolo|nav2|cartographer' | grep -v grep"]),
        'ros_nodes': run(['ros2', 'node', 'list'], timeout=5.0),
        'ros_actions': run(['ros2', 'action', 'info', '/navigate_to_pose'], timeout=5.0),
        'topics': {},
    }
    for topic in topics:
        actions['topics'][topic] = run(['ros2', 'topic', 'info', '-v', topic], timeout=5.0)
    actions['process_counts'] = process_counts(str(actions['processes'].get('output', '')))
    if args.topic_bw_sec > 0.0:
        bw_topics = [item.strip() for item in args.topic_bw_topics.split(',') if item.strip()]
        actions['topic_bw'] = [ros_topic_bw(topic, args.topic_bw_sec) for topic in bw_topics]
        actions['topic_bw_top'] = sorted(
            actions['topic_bw'],
            key=lambda item: float(item.get('mbps') or 0.0),
            reverse=True,
        )[:20]
    return actions


def flatten_row(record: dict[str, Any]) -> dict[str, Any]:
    ping = record.get('ping') or {}
    link = record.get('ip_link_delta') or {}
    counts = record.get('process_counts') or {}
    topic_bw = record.get('topic_bw_top') or []
    camera_mbps = sum(
        float(item.get('mbps') or 0.0)
        for item in topic_bw
        if 'yolo' in str(item.get('topic', '')).lower()
        or 'risk_observation' in str(item.get('topic', '')).lower()
    )
    map_mbps = sum(
        float(item.get('mbps') or 0.0)
        for item in topic_bw
        if 'map' in str(item.get('topic', '')).lower()
    )
    return {
        'stamp': record.get('stamp'),
        'stage': record.get('stage'),
        'ping_p50_ms': ping.get('p50_ms'),
        'ping_p95_ms': ping.get('p95_ms'),
        'ping_p99_ms': ping.get('p99_ms'),
        'ping_max_ms': ping.get('max_ms'),
        'tx_mbps': link.get('tx_mbps'),
        'rx_mbps': link.get('rx_mbps'),
        'domain_bridge_processes': counts.get('domain_bridge'),
        'cartographer_processes': counts.get('cartographer'),
        'nav2_processes': counts.get('nav2'),
        'camera_sender_processes': counts.get('camera_sender'),
        'flask_yolo_processes': counts.get('flask_yolo'),
        'camera_mbps': camera_mbps,
        'map_mbps': map_mbps,
        'top_topic_mbps': (
            f"{topic_bw[0].get('topic')}={float(topic_bw[0].get('mbps') or 0.0):.3f}"
            if topic_bw else ''
        ),
    }


def print_health(row: dict[str, Any]) -> None:
    print(
        'NETWORK_CONTROL_HEALTH | '
        f"stage={row.get('stage')} "
        f"ping_p95_ms={row.get('ping_p95_ms')} "
        f"tx_mbps={row.get('tx_mbps')} "
        f"rx_mbps={row.get('rx_mbps')} "
        f"camera_mbps={row.get('camera_mbps'):.3f} "
        f"map_mbps={row.get('map_mbps'):.3f} "
        f"domain_bridge_processes={row.get('domain_bridge_processes')} "
        f"top_topic={row.get('top_topic_mbps')}"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', required=True, help='A-K stage label or short description')
    parser.add_argument('--duration-sec', type=float, default=60.0)
    parser.add_argument('--interval-sec', type=float, default=5.0)
    parser.add_argument('--ping-host', default='')
    parser.add_argument('--ping-count', type=int, default=20)
    parser.add_argument('--iface', default='')
    parser.add_argument('--link-interval-sec', type=float, default=1.0)
    parser.add_argument('--topic-bw-sec', type=float, default=0.0)
    parser.add_argument(
        '--topic-bw-topics',
        default='/map,/risk/risk_map,/leader_pose,/member_pose,/burger_pose,/field/scout22/risk_observation,/field/follower21/risk_observation,/scout22/rl_confidence_map,/follower21/rl_confidence_map',
    )
    parser.add_argument('--out', default='fleet_perf_probe.jsonl')
    parser.add_argument('--csv-out', default='')
    args = parser.parse_args()

    deadline = time.time() + max(1.0, args.duration_sec)
    output = Path(args.out).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    csv_output = Path(args.csv_out).expanduser() if args.csv_out else output.with_suffix('.csv')
    csv_output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(flatten_row({'stage': args.stage}).keys())
    write_header = not csv_output.exists()
    with output.open('a', encoding='utf-8') as handle:
        with csv_output.open('a', encoding='utf-8', newline='') as csv_handle:
            writer = csv.DictWriter(csv_handle, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
                csv_handle.flush()
            while time.time() < deadline:
                record = snapshot(args)
                row = flatten_row(record)
                handle.write(json.dumps(record, sort_keys=True) + '\n')
                handle.flush()
                writer.writerow(row)
                csv_handle.flush()
                print_health(row)
                time.sleep(max(1.0, args.interval_sec))
    print(f'FLEET_PERF_PROBE_DONE | stage={args.stage} out={output} csv={csv_output}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
