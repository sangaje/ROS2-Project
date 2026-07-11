#!/usr/bin/env python3
"""Quick OpenCV camera smoke test for model/best.pt.

Example:
  python3 tools/test_best_pt_camera.py --camera /dev/video0 --model model/best.pt
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='model/best.pt')
    parser.add_argument('--camera', default='/dev/video0')
    parser.add_argument('--width', type=int, default=1280)
    parser.add_argument('--height', type=int, default=720)
    parser.add_argument('--conf', type=float, default=0.25)
    parser.add_argument(
        '--class-id',
        type=int,
        default=None,
        help='Optional class id filter. Omit to show all classes.',
    )
    parser.add_argument(
        '--device',
        default=None,
        help='Ultralytics device, for example 0, cpu, or cuda:0.',
    )
    return parser.parse_args()


def camera_source(raw: str):
    if raw.isdigit():
        return int(raw)
    return raw


def main() -> int:
    args = parse_args()
    try:
        import cv2
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError(
            'This test needs opencv-python and ultralytics installed in the '
            f'current Python environment: {exc}'
        ) from exc

    model_path = Path(args.model).expanduser()
    if not model_path.is_absolute():
        model_path = Path.cwd() / model_path
    if not model_path.exists():
        raise FileNotFoundError(f'model not found: {model_path}')

    model = YOLO(str(model_path))
    cap = cv2.VideoCapture(camera_source(args.camera), cv2.CAP_V4L2)
    if not cap.isOpened():
        raise RuntimeError(f'camera open failed: {args.camera}')

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    fps_ema = 0.0
    last = time.time()
    classes = [args.class_id] if args.class_id is not None else None
    print(f'BEST_PT_CAMERA_TEST | model={model_path} camera={args.camera}')
    print('Press q or ESC to quit.')

    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            print('camera frame read failed')
            break

        result = model.predict(
            frame,
            conf=args.conf,
            classes=classes,
            device=args.device,
            verbose=False,
        )[0]
        annotated = result.plot()

        now = time.time()
        dt = max(1e-6, now - last)
        last = now
        fps = 1.0 / dt
        fps_ema = fps if fps_ema <= 0.0 else 0.9 * fps_ema + 0.1 * fps
        count = 0 if result.boxes is None else len(result.boxes)
        cv2.putText(
            annotated,
            f'best.pt detections={count} fps={fps_ema:.1f}',
            (16, 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )

        cv2.imshow('best.pt camera test', annotated)
        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord('q')):
            break

    cap.release()
    cv2.destroyAllWindows()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
