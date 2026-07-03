#!/usr/bin/env python3
"""Estimate distance from a parallel two-camera pair without ROS.

This is a deliberately practical probe, not a full stereo calibration tool.

Core equation:

    Z = focal_px * baseline_m / disparity_px

If the camera FOV/focal length is unknown, put an object at a known distance,
click the same point in the left/right images, pass --known-distance-cm, then
press 'k'. The script will estimate focal_px from that one measurement.
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from dual_camera_probe import CameraWorker, make_fourcc, parse_device, save_pair


WINDOW_NAME = "stereo distance probe"


class PointPicker:
    def __init__(self) -> None:
        self.left: Optional[tuple[int, int]] = None
        self.right: Optional[tuple[int, int]] = None
        self.left_width = 0
        self.scale = 1.0

    def reset(self) -> None:
        self.left = None
        self.right = None

    def set_layout(self, left_width: int, scale: float = 1.0) -> None:
        self.left_width = left_width
        self.scale = scale

    def mouse_cb(self, event, x, y, flags, param) -> None:  # noqa: ANN001
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        if self.scale <= 0:
            self.scale = 1.0
        raw_x = int(round(x / self.scale))
        raw_y = int(round(y / self.scale))
        if raw_x < self.left_width:
            self.left = (raw_x, raw_y)
            print(f"left point = {self.left}")
        else:
            self.right = (raw_x - self.left_width, raw_y)
            print(f"right point = {self.right}")


class OptionalYolo:
    def __init__(self, model_path: str, conf: float) -> None:
        self.model_path = model_path
        self.conf = conf
        self.model = None
        if not model_path:
            return
        try:
            from ultralytics import YOLO  # type: ignore

            self.model = YOLO(model_path)
            print(f"YOLO loaded: {model_path}")
        except Exception as exc:  # noqa: BLE001
            print(f"WARNING: failed to load YOLO model {model_path!r}: {exc}")
            print("Manual click mode still works.")
            self.model = None

    def detect_largest_person(self, frame: np.ndarray):
        if self.model is None:
            return None
        result = self.model.predict(frame, conf=self.conf, classes=[0], verbose=False)[0]
        if result.boxes is None or len(result.boxes) == 0:
            return None
        boxes = result.boxes.xyxy.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy()
        best = None
        best_area = -1.0
        for box, conf in zip(boxes, confs):
            x1, y1, x2, y2 = box
            area = max(0.0, float(x2 - x1)) * max(0.0, float(y2 - y1))
            if area > best_area:
                best_area = area
                cx = 0.5 * float(x1 + x2)
                cy = 0.5 * float(y1 + y2)
                # Bottom-center is often more stable for "where is the person on the floor",
                # but center is better for pure stereo correspondence. Report both.
                foot = (cx, float(y2))
                best = {
                    "box": (float(x1), float(y1), float(x2), float(y2)),
                    "center": (cx, cy),
                    "foot": foot,
                    "conf": float(conf),
                    "area": area,
                }
        return best


def focal_px_from_hfov(width_px: int, hfov_deg: float) -> float:
    hfov_rad = math.radians(max(1e-3, min(179.0, hfov_deg)))
    return width_px / (2.0 * math.tan(hfov_rad / 2.0))


def distance_from_disparity(focal_px: float, baseline_m: float, disparity_px: float) -> Optional[float]:
    if abs(disparity_px) < 0.5:
        return None
    return focal_px * baseline_m / abs(disparity_px)


def draw_cross(img: np.ndarray, pt: tuple[int, int], color: tuple[int, int, int], label: str) -> None:
    x, y = pt
    cv2.drawMarker(img, (x, y), color, cv2.MARKER_CROSS, 22, 2, cv2.LINE_AA)
    cv2.putText(img, label, (x + 8, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)


def draw_detection(img: np.ndarray, det, color: tuple[int, int, int], label: str) -> None:
    if det is None:
        return
    x1, y1, x2, y2 = [int(round(v)) for v in det["box"]]
    cx, cy = [int(round(v)) for v in det["center"]]
    fx, fy = [int(round(v)) for v in det["foot"]]
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    cv2.circle(img, (cx, cy), 4, color, -1)
    cv2.circle(img, (fx, fy), 4, (0, 180, 255), -1)
    cv2.putText(
        img,
        f"{label} conf={det['conf']:.2f}",
        (x1, max(18, y1 - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        color,
        2,
        cv2.LINE_AA,
    )


def put_lines(img: np.ndarray, lines: list[str], x: int = 8, y: int = 24) -> None:
    if not lines:
        return
    cv2.rectangle(img, (0, 0), (img.shape[1], 28 + 23 * len(lines)), (0, 0, 0), -1)
    for idx, line in enumerate(lines):
        color = (230, 230, 230)
        if line.startswith("DIST"):
            color = (80, 255, 80)
        if line.startswith("WARN"):
            color = (80, 160, 255)
        cv2.putText(img, line, (x, y + 23 * idx), cv2.FONT_HERSHEY_SIMPLEX, 0.58, color, 1, cv2.LINE_AA)


def load_static_pair(args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray]:
    left = cv2.imread(args.left_image, cv2.IMREAD_COLOR)
    right = cv2.imread(args.right_image, cv2.IMREAD_COLOR)
    if left is None:
        raise RuntimeError(f"cannot read left image: {args.left_image}")
    if right is None:
        raise RuntimeError(f"cannot read right image: {args.right_image}")
    return left, right


def capture_live_pair(left_worker: CameraWorker, right_worker: CameraWorker) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    left = left_worker.snapshot().frame
    right = right_worker.snapshot().frame
    return left, right


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Estimate distance from two parallel USB camera images.")
    parser.add_argument("--left", default="/dev/video0", help="left camera path/index for live mode")
    parser.add_argument("--right", default="/dev/video2", help="right camera path/index for live mode")
    parser.add_argument("--left-image", help="static left image path")
    parser.add_argument("--right-image", help="static right image path")
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--height", type=int, default=240)
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--fourcc", default="MJPG")
    parser.add_argument("--baseline-cm", type=float, default=7.15, help="lens-center baseline in centimeters")
    parser.add_argument("--hfov-deg", type=float, default=60.0, help="initial horizontal FOV guess")
    parser.add_argument("--focal-px", type=float, default=0.0, help="override focal length in pixels")
    parser.add_argument("--known-distance-cm", type=float, default=0.0, help="press k after clicking a known-distance point")
    parser.add_argument("--yolo-model", default="", help="optional YOLO model path, e.g. yolo11n.pt")
    parser.add_argument("--yolo-conf", type=float, default=0.35)
    parser.add_argument("--auto-yolo", action="store_true", help="start in YOLO person auto-distance mode")
    parser.add_argument("--save-dir", default="stereo_usb_test/captures")
    parser.add_argument("--display-scale", type=float, default=1.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    static_mode = bool(args.left_image or args.right_image)
    if static_mode and not (args.left_image and args.right_image):
        raise SystemExit("--left-image and --right-image must be used together")

    baseline_m = args.baseline_cm / 100.0
    focal_px_override = args.focal_px if args.focal_px > 0 else None
    yolo = OptionalYolo(args.yolo_model, args.yolo_conf)
    auto_yolo = bool(args.auto_yolo and yolo.model is not None)
    picker = PointPicker()
    save_dir = Path(args.save_dir)

    left_worker = None
    right_worker = None
    static_left = None
    static_right = None

    if static_mode:
        static_left, static_right = load_static_pair(args)
    else:
        backend = cv2.CAP_V4L2
        left_worker = CameraWorker("LEFT", parse_device(args.left), args.width, args.height, args.fps, args.fourcc, backend, 0.3)
        right_worker = CameraWorker("RIGHT", parse_device(args.right), args.width, args.height, args.fps, args.fourcc, backend, 0.3)
        left_worker.start()
        right_worker.start()

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(WINDOW_NAME, picker.mouse_cb)

    print("Controls:")
    print("  click left/right image: select same physical point")
    print("  r: reset clicked points")
    print("  [: decrease HFOV guess by 1 deg   ]: increase HFOV guess by 1 deg")
    print("  -/=: decrease/increase focal_px by 2%")
    print("  k: calibrate focal_px from clicked disparity and --known-distance-cm")
    print("  y: toggle YOLO auto mode if --yolo-model is available")
    print("  s: save current raw pair")
    print("  q or ESC: quit")

    try:
        while True:
            if static_mode:
                assert static_left is not None and static_right is not None
                left_frame = static_left.copy()
                right_frame = static_right.copy()
            else:
                assert left_worker is not None and right_worker is not None
                left_frame, right_frame = capture_live_pair(left_worker, right_worker)
                if left_frame is None or right_frame is None:
                    time.sleep(0.02)
                    continue

            if left_frame.shape[0] != right_frame.shape[0]:
                h = min(left_frame.shape[0], right_frame.shape[0])
                left_frame = cv2.resize(left_frame, (int(left_frame.shape[1] * h / left_frame.shape[0]), h))
                right_frame = cv2.resize(right_frame, (int(right_frame.shape[1] * h / right_frame.shape[0]), h))

            left_width = left_frame.shape[1]
            picker.set_layout(left_width=left_width, scale=args.display_scale)

            focal_px = focal_px_override or focal_px_from_hfov(left_width, args.hfov_deg)
            selected_left = picker.left
            selected_right = picker.right
            mode = "manual-click"

            left_det = None
            right_det = None
            if auto_yolo:
                mode = "YOLO-center"
                left_det = yolo.detect_largest_person(left_frame)
                right_det = yolo.detect_largest_person(right_frame)
                if left_det is not None and right_det is not None:
                    selected_left = tuple(int(round(v)) for v in left_det["center"])
                    selected_right = tuple(int(round(v)) for v in right_det["center"])

            disparity = None
            distance_m = None
            if selected_left is not None and selected_right is not None:
                disparity = float(selected_left[0] - selected_right[0])
                distance_m = distance_from_disparity(focal_px, baseline_m, disparity)

            draw_detection(left_frame, left_det, (80, 255, 80), "L person")
            draw_detection(right_frame, right_det, (80, 255, 80), "R person")
            if selected_left is not None:
                draw_cross(left_frame, selected_left, (255, 80, 80), "L")
            if selected_right is not None:
                draw_cross(right_frame, selected_right, (255, 80, 80), "R")

            combined = np.hstack([left_frame, right_frame])
            lines = [
                f"mode={mode}  baseline={args.baseline_cm:.2f}cm  focal={focal_px:.1f}px  hfov_guess={args.hfov_deg:.1f}deg",
                "click same point L/R, then tune HFOV or press k with known distance",
            ]
            if disparity is not None:
                lines.append(f"disparity={disparity:.1f}px abs={abs(disparity):.1f}px")
                if distance_m is None:
                    lines.append("WARN disparity too small; distance is unstable")
                else:
                    lines.append(f"DIST {distance_m:.3f}m  ({distance_m * 100.0:.1f}cm)")
                    if abs(disparity) < 5.0:
                        lines.append("WARN disparity < 5px: far/unstable with 7.15cm baseline")
            else:
                lines.append("select corresponding points or enable YOLO auto mode")
            put_lines(combined, lines)

            if args.display_scale != 1.0:
                combined_show = cv2.resize(
                    combined,
                    (int(combined.shape[1] * args.display_scale), int(combined.shape[0] * args.display_scale)),
                    interpolation=cv2.INTER_AREA if args.display_scale < 1.0 else cv2.INTER_LINEAR,
                )
            else:
                combined_show = combined
            cv2.imshow(WINDOW_NAME, combined_show)

            key = cv2.waitKey(1 if not static_mode else 30) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("r"):
                picker.reset()
            elif key == ord("["):
                args.hfov_deg = max(5.0, args.hfov_deg - 1.0)
                focal_px_override = None
            elif key == ord("]"):
                args.hfov_deg = min(170.0, args.hfov_deg + 1.0)
                focal_px_override = None
            elif key in (ord("-"), ord("_")):
                focal_px_override = focal_px * 0.98
            elif key in (ord("="), ord("+")):
                focal_px_override = focal_px * 1.02
            elif key == ord("y"):
                if yolo.model is None:
                    print("YOLO model is not loaded. Start with --yolo-model yolo11n.pt")
                else:
                    auto_yolo = not auto_yolo
                    print("YOLO auto mode:", auto_yolo)
            elif key == ord("k"):
                if args.known_distance_cm <= 0:
                    print("Set --known-distance-cm first, e.g. --known-distance-cm 100")
                elif disparity is None or abs(disparity) < 0.5:
                    print("Click left/right corresponding points first.")
                else:
                    known_m = args.known_distance_cm / 100.0
                    focal_px_override = known_m * abs(disparity) / baseline_m
                    print(
                        f"calibrated focal_px={focal_px_override:.2f} "
                        f"from known distance {args.known_distance_cm:.1f}cm and disparity {abs(disparity):.2f}px"
                    )
            elif key == ord("s"):
                if static_mode:
                    stamp = time.strftime("%Y%m%d_%H%M%S")
                    save_dir.mkdir(parents=True, exist_ok=True)
                    out = save_dir / f"{stamp}_annotated_distance.jpg"
                    cv2.imwrite(str(out), combined)
                    print("saved", out)
                else:
                    assert left_worker is not None and right_worker is not None
                    save_pair(save_dir, left_worker.snapshot(), right_worker.snapshot())

    finally:
        if left_worker is not None:
            left_worker.stop()
        if right_worker is not None:
            right_worker.stop()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
