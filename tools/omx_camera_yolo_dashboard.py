#!/usr/bin/env python3
"""Standalone OMX camera + YOLO smoke test, served over Flask (headless-safe).

Same camera+model pipeline as test_best_pt_camera.py, but streamed as MJPEG
instead of cv2.imshow so it works over SSH with no X display. Does not import
anything from the omx_aim ROS package -- avoids any ROS workspace overlay
ambiguity, so this always runs the exact model/args passed on the command
line.

Example:
  python3 tools/omx_camera_yolo_dashboard.py --camera /dev/video0 \
      --model model/target_v3.pt --device 0 --port 8082

Then open http://<jetson-host>:8082/ in a browser.
"""

from __future__ import annotations

import argparse
import threading
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='model/target_v3.pt')
    parser.add_argument('--camera', default='/dev/video0')
    parser.add_argument('--width', type=int, default=1280)
    parser.add_argument('--height', type=int, default=720)
    parser.add_argument('--conf', type=float, default=0.25)
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument(
        '--class-id', type=int, default=None,
        help='Optional class id filter. Omit to show all classes.',
    )
    parser.add_argument(
        '--device', default=None,
        help='Ultralytics device, for example 0, cpu, or cuda:0. Omit to '
             'let ultralytics pick.',
    )
    parser.add_argument('--half', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--host', default='0.0.0.0')
    parser.add_argument('--port', type=int, default=8082)
    parser.add_argument('--fps', type=int, default=15, help='MJPEG output fps cap.')
    parser.add_argument('--jpeg-quality', type=int, default=75)
    return parser.parse_args()


def camera_source(raw: str):
    if raw.isdigit():
        return int(raw)
    return raw


class InferenceLoop:
    """Owns the camera + model and keeps the latest annotated frame/state."""

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self._lock = threading.Lock()
        self._frame = None
        self._state = {
            'ok': False,
            'model': str(args.model),
            'camera': str(args.camera),
            'device_requested': args.device,
            'device_used': None,
            'fps': 0.0,
            'detections': 0,
            'classes': [],
            'frames': 0,
            'last_error': '',
            'started_at': time.time(),
        }
        self._stop = False

    def get_frame(self):
        with self._lock:
            return self._frame

    def get_state(self):
        with self._lock:
            return dict(self._state)

    def _set_error(self, message: str) -> None:
        with self._lock:
            self._state['last_error'] = message
            self._state['ok'] = False

    def stop(self) -> None:
        self._stop = True

    def run(self) -> None:
        import cv2
        from ultralytics import YOLO

        args = self.args
        model_path = Path(args.model).expanduser()
        if not model_path.is_absolute():
            model_path = Path.cwd() / model_path
        if not model_path.exists():
            self._set_error(f'model not found: {model_path}')
            return

        print(f'OMX_CAMERA_YOLO_TEST | loading model {model_path}', flush=True)
        model = YOLO(str(model_path))

        cap = cv2.VideoCapture(camera_source(args.camera), cv2.CAP_V4L2)
        if not cap.isOpened():
            self._set_error(f'camera open failed: {args.camera}')
            return
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

        classes = [args.class_id] if args.class_id is not None else None
        fps_ema = 0.0
        last = time.time()
        print(
            f'OMX_CAMERA_YOLO_TEST | camera={args.camera} model={model_path.name} '
            f'device={args.device} ready',
            flush=True,
        )

        while not self._stop:
            ok, frame = cap.read()
            if not ok or frame is None:
                self._set_error('camera frame read failed')
                time.sleep(0.5)
                continue

            result = model.predict(
                frame,
                conf=args.conf,
                imgsz=args.imgsz,
                classes=classes,
                device=args.device,
                half=args.half,
                verbose=False,
            )[0]
            annotated = result.plot()

            now = time.time()
            dt = max(1e-6, now - last)
            last = now
            fps = 1.0 / dt
            fps_ema = fps if fps_ema <= 0.0 else 0.9 * fps_ema + 0.1 * fps

            boxes = result.boxes
            count = 0 if boxes is None else len(boxes)
            names = result.names or {}
            classes_seen = []
            if boxes is not None and count:
                class_ids = boxes.cls.int().tolist()
                classes_seen = sorted({names.get(c, str(c)) for c in class_ids})

            ok_enc, buf = cv2.imencode(
                '.jpg', annotated, [int(cv2.IMWRITE_JPEG_QUALITY), args.jpeg_quality]
            )
            with self._lock:
                if ok_enc:
                    self._frame = buf.tobytes()
                self._state.update({
                    'ok': True,
                    'device_used': str(getattr(model.predictor, 'device', args.device)),
                    'fps': round(fps_ema, 1),
                    'detections': count,
                    'classes': classes_seen,
                    'frames': self._state['frames'] + 1,
                    'last_error': '',
                })

            time.sleep(max(0.0, (1.0 / args.fps) - (time.time() - now)))

        cap.release()


def build_app(loop: InferenceLoop):
    from flask import Flask, Response, jsonify

    app = Flask(__name__)

    index_html = """<!doctype html>
<html><head><meta charset="utf-8"><title>OMX camera + YOLO test</title>
<style>
body{background:#0b0d10;color:#e6e6e6;font-family:sans-serif;margin:0;padding:16px}
img{max-width:100%;border:1px solid #333}
pre{background:#161a1e;padding:12px;border-radius:6px;overflow-x:auto}
</style></head>
<body>
<h2>OMX camera + YOLO smoke test</h2>
<img src="/stream.mjpg" alt="stream">
<h3>State</h3>
<pre id="state">loading...</pre>
<script>
async function poll(){
  try{
    const r = await fetch('/state.json', {cache:'no-store'});
    document.getElementById('state').textContent = JSON.stringify(await r.json(), null, 2);
  }catch(e){}
  setTimeout(poll, 500);
}
poll();
</script>
</body></html>"""

    @app.get('/')
    def index():
        return index_html

    @app.get('/state.json')
    def state_json():
        return jsonify(loop.get_state())

    @app.get('/stream.mjpg')
    def stream():
        def generate():
            while True:
                frame = loop.get_frame()
                if frame is None:
                    time.sleep(0.1)
                    continue
                yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n'
                       + frame + b'\r\n')
                time.sleep(1.0 / 15)

        return Response(
            generate(), mimetype='multipart/x-mixed-replace; boundary=frame'
        )

    return app


def main() -> int:
    args = parse_args()
    loop = InferenceLoop(args)
    worker = threading.Thread(target=loop.run, daemon=True)
    worker.start()

    app = build_app(loop)
    print(
        f'OMX_CAMERA_YOLO_TEST | dashboard on http://{args.host}:{args.port}/',
        flush=True,
    )
    app.run(host=args.host, port=args.port, debug=False, use_reloader=False, threaded=True)
    loop.stop()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
