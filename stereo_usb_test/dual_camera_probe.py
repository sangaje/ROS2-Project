#!/usr/bin/env python3
"""Probe two USB cameras at the same time without ROS.

This is intentionally small and dependency-light:

    python3 stereo_usb_test/dual_camera_probe.py --left /dev/video0 --right /dev/video2

It answers the first practical question before building a stereo/YOLO pipeline:
"Can this PC read two cameras at the target size/FPS without exploding?"
"""

from __future__ import annotations

import argparse
import glob
import os
import shutil
import subprocess
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, Optional

import cv2
import numpy as np


def parse_device(value: str):
    """OpenCV accepts either an integer camera index or a device path."""
    if value.isdigit():
        return int(value)
    return value


def make_fourcc(name: str) -> int:
    name = (name or "").strip()
    if len(name) != 4:
        raise argparse.ArgumentTypeError("fourcc must be exactly 4 chars, e.g. MJPG or YUYV")
    return cv2.VideoWriter_fourcc(*name)


def list_camera_candidates() -> None:
    devices = sorted(glob.glob("/dev/video*"))
    print("OpenCV version:", cv2.__version__)
    print("Detected /dev/video* devices:")
    if devices:
        for dev in devices:
            print(f"  {dev}")
    else:
        print("  none")

    if shutil.which("v4l2-ctl"):
        print("\nv4l2-ctl --list-devices:")
        subprocess.run(["v4l2-ctl", "--list-devices"], check=False)
    else:
        print("\nv4l2-ctl not found. Install with: sudo apt install v4l-utils")


@dataclass
class CameraSnapshot:
    ok: bool
    frame: Optional[np.ndarray]
    timestamp: float
    seq: int
    fps: float
    failures: int
    shape: str


class CameraWorker:
    def __init__(
        self,
        name: str,
        device,
        width: int,
        height: int,
        fps: float,
        fourcc: str,
        backend: int,
        warmup_sec: float,
    ) -> None:
        self.name = name
        self.device = device
        self.width = width
        self.height = height
        self.target_fps = fps
        self.fourcc = fourcc
        self.backend = backend
        self.warmup_sec = warmup_sec

        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._cap: Optional[cv2.VideoCapture] = None

        self._frame: Optional[np.ndarray] = None
        self._timestamp = 0.0
        self._seq = 0
        self._failures = 0
        self._times: Deque[float] = deque(maxlen=60)

    def start(self) -> None:
        self._cap = cv2.VideoCapture(self.device, self.backend)
        if not self._cap.isOpened():
            raise RuntimeError(f"{self.name}: cannot open camera {self.device!r}")

        # Keep latency low. Some backends ignore this, but it helps when supported.
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self._cap.set(cv2.CAP_PROP_FOURCC, make_fourcc(self.fourcc))
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self._cap.set(cv2.CAP_PROP_FPS, self.target_fps)

        actual_w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = self._cap.get(cv2.CAP_PROP_FPS)
        print(
            f"{self.name}: opened {self.device!r}, requested "
            f"{self.width}x{self.height}@{self.target_fps:g} {self.fourcc}, "
            f"driver reports {actual_w}x{actual_h}@{actual_fps:.2f}"
        )

        self._thread = threading.Thread(target=self._loop, name=f"camera-{self.name}", daemon=True)
        self._thread.start()

        if self.warmup_sec > 0:
            time.sleep(self.warmup_sec)

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        if self._cap is not None:
            self._cap.release()

    def snapshot(self) -> CameraSnapshot:
        with self._lock:
            frame = None if self._frame is None else self._frame.copy()
            shape = "none" if frame is None else f"{frame.shape[1]}x{frame.shape[0]}"
            return CameraSnapshot(
                ok=frame is not None,
                frame=frame,
                timestamp=self._timestamp,
                seq=self._seq,
                fps=self._estimate_fps_locked(),
                failures=self._failures,
                shape=shape,
            )

    def _estimate_fps_locked(self) -> float:
        if len(self._times) < 2:
            return 0.0
        dt = self._times[-1] - self._times[0]
        if dt <= 1e-6:
            return 0.0
        return (len(self._times) - 1) / dt

    def _loop(self) -> None:
        assert self._cap is not None
        while not self._stop.is_set():
            ok, frame = self._cap.read()
            now = time.monotonic()
            with self._lock:
                if ok and frame is not None:
                    self._frame = frame
                    self._timestamp = now
                    self._seq += 1
                    self._times.append(now)
                else:
                    self._failures += 1
            if not ok:
                time.sleep(0.02)


def overlay_status(
    frame: np.ndarray,
    label: str,
    snap: CameraSnapshot,
    now: float,
    skew_ms: Optional[float],
) -> np.ndarray:
    out = frame.copy()
    age_ms = (now - snap.timestamp) * 1000.0 if snap.timestamp > 0 else float("inf")
    lines = [
        f"{label}  {snap.shape}",
        f"fps={snap.fps:4.1f} age={age_ms:5.0f}ms seq={snap.seq}",
        f"fail={snap.failures}",
    ]
    if skew_ms is not None:
        lines.append(f"L/R skew={skew_ms:5.0f}ms")

    bg_h = 22 * len(lines) + 10
    cv2.rectangle(out, (0, 0), (out.shape[1], bg_h), (0, 0, 0), thickness=-1)
    color = (80, 255, 80) if age_ms < 300 else (80, 160, 255)
    if age_ms > 800:
        color = (80, 80, 255)
    for idx, text in enumerate(lines):
        cv2.putText(
            out,
            text,
            (8, 24 + 22 * idx),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            color if idx == 1 else (230, 230, 230),
            1,
            cv2.LINE_AA,
        )
    return out


def blank_frame(width: int, height: int, text: str) -> np.ndarray:
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    cv2.putText(frame, text, (20, height // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (80, 80, 255), 2)
    return frame


def resize_to_same_height(left: np.ndarray, right: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    h = min(left.shape[0], right.shape[0])
    if left.shape[0] != h:
        w = int(left.shape[1] * (h / left.shape[0]))
        left = cv2.resize(left, (w, h), interpolation=cv2.INTER_AREA)
    if right.shape[0] != h:
        w = int(right.shape[1] * (h / right.shape[0]))
        right = cv2.resize(right, (w, h), interpolation=cv2.INTER_AREA)
    return left, right


def save_pair(save_dir: Path, left: CameraSnapshot, right: CameraSnapshot) -> None:
    save_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    ms = int((time.time() % 1.0) * 1000)
    if left.frame is not None:
        path = save_dir / f"{stamp}_{ms:03d}_left_seq{left.seq}.jpg"
        cv2.imwrite(str(path), left.frame)
        print("saved", path)
    if right.frame is not None:
        path = save_dir / f"{stamp}_{ms:03d}_right_seq{right.seq}.jpg"
        cv2.imwrite(str(path), right.frame)
        print("saved", path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Two USB camera FPS/skew probe without ROS.")
    parser.add_argument("--list", action="store_true", help="list /dev/video* candidates and exit")
    parser.add_argument("--left", default="/dev/video0", help="left camera index/path, e.g. 0 or /dev/video0")
    parser.add_argument("--right", default="/dev/video2", help="right camera index/path, e.g. 2 or /dev/video2")
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--height", type=int, default=240)
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--fourcc", default="MJPG", help="camera pixel format: MJPG usually lighter USB bandwidth than YUYV")
    parser.add_argument("--duration", type=float, default=0.0, help="seconds to run; 0 means until q/ESC/Ctrl-C")
    parser.add_argument("--no-display", action="store_true", help="run without cv2.imshow, useful over SSH")
    parser.add_argument("--save-dir", default="stereo_usb_test/captures", help="where 's' snapshots are written")
    parser.add_argument("--warmup-sec", type=float, default=0.5)
    parser.add_argument("--print-every", type=float, default=1.0, help="console status period in seconds")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.list:
        list_camera_candidates()
        return 0

    if args.left == args.right:
        print("ERROR: --left and --right are the same device. Pick two different cameras.", file=sys.stderr)
        return 2

    backend = cv2.CAP_V4L2 if os.name == "posix" else cv2.CAP_ANY
    left = CameraWorker(
        "LEFT",
        parse_device(args.left),
        args.width,
        args.height,
        args.fps,
        args.fourcc,
        backend,
        args.warmup_sec,
    )
    right = CameraWorker(
        "RIGHT",
        parse_device(args.right),
        args.width,
        args.height,
        args.fps,
        args.fourcc,
        backend,
        args.warmup_sec,
    )

    skew_values_ms: Deque[float] = deque(maxlen=500)
    start = time.monotonic()
    last_print = 0.0
    save_dir = Path(args.save_dir)

    try:
        left.start()
        right.start()
        print("Running. Press q/ESC to quit, s to save a pair.")

        while True:
            now = time.monotonic()
            l_snap = left.snapshot()
            r_snap = right.snapshot()

            skew_ms: Optional[float] = None
            if l_snap.timestamp > 0 and r_snap.timestamp > 0:
                skew_ms = abs(l_snap.timestamp - r_snap.timestamp) * 1000.0
                skew_values_ms.append(skew_ms)

            if now - last_print >= args.print_every:
                last_print = now
                l_age = (now - l_snap.timestamp) * 1000.0 if l_snap.timestamp > 0 else float("inf")
                r_age = (now - r_snap.timestamp) * 1000.0 if r_snap.timestamp > 0 else float("inf")
                skew_text = "n/a" if skew_ms is None else f"{skew_ms:.0f}ms"
                print(
                    f"L fps={l_snap.fps:4.1f} age={l_age:5.0f}ms fail={l_snap.failures:3d} | "
                    f"R fps={r_snap.fps:4.1f} age={r_age:5.0f}ms fail={r_snap.failures:3d} | "
                    f"skew={skew_text}"
                )

            if not args.no_display:
                l_frame = l_snap.frame if l_snap.frame is not None else blank_frame(args.width, args.height, "LEFT no frame")
                r_frame = r_snap.frame if r_snap.frame is not None else blank_frame(args.width, args.height, "RIGHT no frame")
                l_frame, r_frame = resize_to_same_height(l_frame, r_frame)
                l_vis = overlay_status(l_frame, "LEFT", l_snap, now, skew_ms)
                r_vis = overlay_status(r_frame, "RIGHT", r_snap, now, skew_ms)
                combined = np.hstack([l_vis, r_vis])
                cv2.imshow("dual USB camera probe", combined)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                if key == ord("s"):
                    save_pair(save_dir, l_snap, r_snap)
            else:
                time.sleep(0.01)

            if args.duration > 0 and (now - start) >= args.duration:
                break

    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        left.stop()
        right.stop()
        if not args.no_display:
            cv2.destroyAllWindows()

    l_snap = left.snapshot()
    r_snap = right.snapshot()
    elapsed = max(1e-6, time.monotonic() - start)
    print("\nSummary")
    print(f"  elapsed: {elapsed:.1f}s")
    print(f"  left:  seq={l_snap.seq} fps_window={l_snap.fps:.1f} failures={l_snap.failures}")
    print(f"  right: seq={r_snap.seq} fps_window={r_snap.fps:.1f} failures={r_snap.failures}")
    if skew_values_ms:
        values = np.array(skew_values_ms, dtype=np.float32)
        print(
            "  skew_ms: "
            f"mean={float(values.mean()):.1f} "
            f"p95={float(np.percentile(values, 95)):.1f} "
            f"max={float(values.max()):.1f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
