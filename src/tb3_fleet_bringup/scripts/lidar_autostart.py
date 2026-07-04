#!/usr/bin/env python3

import argparse
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from threading import Event

import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import LaserScan


class ScanProbe(Node):
    def __init__(self, topic: str):
        super().__init__('tb3_lidar_autostart_probe')
        self.seen = Event()
        qos = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )
        self.create_subscription(LaserScan, topic, self._scan_cb, qos)

    def reset(self):
        self.seen.clear()

    def _scan_cb(self, _msg):
        self.seen.set()


def unique(items):
    out = []
    for item in items:
        if item and item not in out:
            out.append(item)
    return out


def port_candidates(requested: str):
    requested = (requested or '').strip()
    if requested and requested.lower() != 'auto':
        return [requested]

    candidates = []
    by_id = Path('/dev/serial/by-id')
    if by_id.exists():
        preferred = []
        fallback = []
        for path in sorted(by_id.iterdir()):
            try:
                resolved = path.resolve()
            except OSError:
                continue
            if not resolved.name.startswith('ttyUSB'):
                continue
            name = path.name.lower()
            target = str(path)
            if any(key in name for key in ('ld', 'lidar', 'cp210', 'silicon', 'usb-serial', 'uart')):
                preferred.append(target)
            else:
                fallback.append(target)
        candidates.extend(preferred)
        candidates.extend(fallback)

    for idx in range(4):
        candidates.append(f'/dev/ttyUSB{idx}')
    for idx in range(1, 3):
        candidates.append(f'/dev/ttyACM{idx}')
    return unique(candidates)


def launch_cmd(model: str, port: str, frame_id: str, namespace: str):
    model = model.strip().upper()
    if model == 'LDS-02':
        return [
            'ros2', 'launch', 'ld08_driver', 'ld08.launch.py',
            f'port:={port}',
            f'frame_id:={frame_id}',
            f'namespace:={namespace}',
        ]
    if model == 'LDS-03':
        return [
            'ros2', 'launch', 'coin_d4_driver', 'single_lidar_node.launch.py',
            f'port:={port}',
            f'frame_id:={frame_id}',
            f'namespace:={namespace}',
        ]
    return [
        'ros2', 'launch', 'hls_lfcd_lds_driver', 'hlds_laser.launch.py',
        f'port:={port}',
        f'frame_id:={frame_id}',
        f'namespace:={namespace}',
    ]


def stop_process(proc):
    if proc is None or proc.poll() is not None:
        return
    proc.send_signal(signal.SIGINT)
    try:
        proc.wait(timeout=3.0)
        return
    except subprocess.TimeoutExpired:
        pass
    proc.terminate()
    try:
        proc.wait(timeout=2.0)
        return
    except subprocess.TimeoutExpired:
        pass
    proc.kill()
    proc.wait(timeout=2.0)


def wait_for_scan(node: ScanProbe, timeout_sec: float):
    deadline = time.monotonic() + timeout_sec
    while time.monotonic() < deadline:
        rclpy.spin_once(node, timeout_sec=0.2)
        if node.seen.is_set():
            return True
    return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default=os.environ.get('LDS_MODEL', 'LDS-02'))
    parser.add_argument('--port', default=os.environ.get('LIDAR_PORT', 'auto'))
    parser.add_argument('--frame-id', default='base_scan')
    parser.add_argument('--namespace', default='')
    parser.add_argument('--scan-topic', default='/scan')
    parser.add_argument('--probe-timeout', type=float, default=8.0)
    parser.add_argument('--retry-sleep', type=float, default=1.0)
    args = parser.parse_args()

    rclpy.init(args=None)
    node = ScanProbe(args.scan_topic)
    proc = None
    stopping = False

    def _stop(_signum, _frame):
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)

    try:
        candidates = port_candidates(args.port)
        node.get_logger().info(
            f'lidar_autostart: model={args.model} candidates={candidates} frame={args.frame_id}'
        )
        while not stopping:
            for port in candidates:
                if stopping:
                    break
                if not Path(port).exists():
                    node.get_logger().warn(f'lidar_autostart: skip missing port {port}')
                    continue
                cmd = launch_cmd(args.model, port, args.frame_id, args.namespace)
                node.get_logger().info(f'lidar_autostart: trying port {port}: {" ".join(cmd)}')
                node.reset()
                proc = subprocess.Popen(cmd)
                if wait_for_scan(node, args.probe_timeout):
                    node.get_logger().info(f'lidar_autostart: /scan OK on {port}; keeping driver alive')
                    while not stopping and proc.poll() is None:
                        rclpy.spin_once(node, timeout_sec=0.5)
                    if not stopping:
                        node.get_logger().warn(
                            f'lidar_autostart: driver on {port} exited with code {proc.returncode}; retrying'
                        )
                    stop_process(proc)
                    proc = None
                    break
                node.get_logger().error(f'lidar_autostart: no /scan from {port}; stopping driver')
                stop_process(proc)
                proc = None
                time.sleep(args.retry_sleep)
            else:
                node.get_logger().error('lidar_autostart: no candidate produced /scan; retrying all ports')
                time.sleep(max(2.0, args.retry_sleep))
    finally:
        stop_process(proc)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    sys.exit(main())
