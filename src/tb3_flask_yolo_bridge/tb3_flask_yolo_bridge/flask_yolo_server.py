#!/usr/bin/env python3

from __future__ import annotations

import argparse
import logging
import threading
import time
from dataclasses import dataclass, field

import numpy as np


DEBUG_PAGE = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>TB3 Flask YOLO Debug</title>
  <style>
    :root { color-scheme: dark; font-family: system-ui, sans-serif; }
    body { margin: 0; background: #101318; color: #eef2f6; }
    header { padding: 14px 18px; background: #171c23; position: sticky; top: 0; z-index: 2; }
    h1 { margin: 0 0 6px; font-size: 20px; }
    #status { color: #8fd3ff; font-family: monospace; font-size: 13px; }
    main { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 12px; padding: 12px; }
    section { background: #171c23; border: 1px solid #29313c; border-radius: 8px; overflow: hidden; }
    h2 { margin: 0; padding: 9px 12px; font-size: 15px; background: #202731; }
    img { width: 100%; display: block; min-height: 240px; object-fit: contain; background: #050607; }
    @media (max-width: 900px) { main { grid-template-columns: 1fr; } }
  </style>
</head>
<body>
  <header>
    <h1>TB3 Camera / YOLO Debug</h1>
    <div id="status">Waiting for OpenCV camera frames...</div>
  </header>
  <main>
    <section><h2>OpenCV input — latest frame</h2><img id="raw" alt="raw camera frame"></section>
    <section><h2>YOLO person result — latest frame</h2><img id="yolo" alt="YOLO frame"></section>
  </main>
  <script>
    async function refreshStatus() {
      try {
        const s = await (await fetch('/api/status', {cache: 'no-store'})).json();
        document.getElementById('status').textContent =
          `frames=${s.frames} | people=${s.people} | inference=${s.latency_ms.toFixed(1)} ms | ` +
          `capture-to-web=${s.capture_age_ms.toFixed(1)} ms | ` +
          `size=${s.image_width}x${s.image_height} | server-age=${s.frame_age_sec.toFixed(2)} s`;
      } catch (_) {
        document.getElementById('status').textContent = 'Debug server disconnected';
      }
    }
    async function fetchLatest(kind) {
      const response = await fetch(`/frame/${kind}.jpg?t=${Date.now()}`, {cache: 'no-store'});
      if (!response.ok) return;
      const blob = await response.blob();
      const image = document.getElementById(kind);
      const nextUrl = URL.createObjectURL(blob);
      const oldUrl = image.dataset.objectUrl;
      image.src = nextUrl;
      image.dataset.objectUrl = nextUrl;
      if (oldUrl) URL.revokeObjectURL(oldUrl);
    }
    let polling = false;
    async function pollLatest() {
      if (polling) return;
      polling = true;
      try {
        await Promise.all([fetchLatest('raw'), fetchLatest('yolo')]);
      } catch (_) {
        // The next poll retries. Never queue old browser requests.
      } finally {
        polling = false;
        setTimeout(pollLatest, 60);
      }
    }
    setInterval(refreshStatus, 500);
    refreshStatus();
    pollLatest();
  </script>
</body>
</html>
"""


@dataclass
class DebugState:
    condition: threading.Condition = field(default_factory=threading.Condition)
    raw_jpeg: bytes | None = None
    yolo_jpeg: bytes | None = None
    version: int = 0
    frames: int = 0
    people: int = 0
    latency_ms: float = 0.0
    image_width: int = 0
    image_height: int = 0
    capture_age_ms: float = -1.0
    last_frame_wall_sec: float = 0.0

    def update(self, raw_jpeg, yolo_jpeg, people, latency_ms, width, height, capture_age_ms):
        with self.condition:
            self.raw_jpeg = raw_jpeg
            self.yolo_jpeg = yolo_jpeg
            self.version += 1
            self.frames += 1
            self.people = people
            self.latency_ms = latency_ms
            self.image_width = width
            self.image_height = height
            self.capture_age_ms = capture_age_ms
            self.last_frame_wall_sec = time.time()
            self.condition.notify_all()

    def status(self):
        with self.condition:
            age = time.time() - self.last_frame_wall_sec if self.last_frame_wall_sec else -1.0
            return {
                'ok': self.frames > 0,
                'frames': self.frames,
                'people': self.people,
                'latency_ms': self.latency_ms,
                'image_width': self.image_width,
                'image_height': self.image_height,
                'capture_age_ms': self.capture_age_ms,
                'frame_age_sec': age,
            }

    def latest_frame(self, kind):
        with self.condition:
            frame = self.raw_jpeg if kind == 'raw' else self.yolo_jpeg
            return self.version, frame

    def wait_for_frame(self, kind, previous_version):
        with self.condition:
            self.condition.wait_for(
                lambda: self.version != previous_version
                and (self.raw_jpeg if kind == 'raw' else self.yolo_jpeg) is not None,
                timeout=1.0,
            )
            frame = self.raw_jpeg if kind == 'raw' else self.yolo_jpeg
            return self.version, frame


def _decode_image(file_storage):
    import cv2

    encoded = file_storage.read()
    data = np.frombuffer(encoded, dtype=np.uint8)
    if data.size == 0:
        raise ValueError('empty image upload')
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError('cv2.imdecode failed')
    return encoded, img


def _encode_jpeg(frame, quality):
    import cv2

    ok, buf = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        raise ValueError('cv2.imencode failed')
    return bytes(buf)


def _draw_yolo_overlay(frame, detections, latency_ms):
    import cv2

    output = frame.copy()
    for det in detections:
        x1, y1, x2, y2 = [int(round(v)) for v in det['bbox']]
        cv2.rectangle(output, (x1, y1), (x2, y2), (40, 230, 70), 2)
        label = f"{det['label']} {det['conf']:.2f}"
        cv2.putText(
            output, label, (x1, max(20, y1 - 7)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.58, (40, 230, 70), 2,
        )
    cv2.putText(
        output,
        f'people={len(detections)} inference={latency_ms:.1f}ms',
        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 220, 255), 2,
    )
    if not detections:
        cv2.putText(
            output, 'NO PERSON', (10, 55),
            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 180, 255), 2,
        )
    return output


def _mjpeg_stream(state, kind):
    version = -1
    while True:
        version, frame = state.wait_for_frame(kind, version)
        if frame is None:
            continue
        yield (
            b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n'
            b'Cache-Control: no-cache\r\n\r\n' + frame + b'\r\n'
        )


def build_app(args):
    from flask import Flask, Response, jsonify, request
    from ultralytics import YOLO

    app = Flask(__name__)
    model = YOLO(args.model_path)
    model_lock = threading.Lock()
    state = DebugState()

    @app.get('/')
    def index():
        return Response(
            DEBUG_PAGE,
            mimetype='text/html',
            headers={'Cache-Control': 'no-store, no-cache, must-revalidate, max-age=0'},
        )

    @app.get('/health')
    def health():
        return jsonify({
            'ok': True,
            'model_path': args.model_path,
            'device': args.device,
            'debug_url': f'http://127.0.0.1:{args.port}/',
        })

    @app.get('/api/status')
    def status():
        return jsonify(state.status())

    @app.get('/stream/<kind>.mjpg')
    def stream(kind):
        if kind not in ('raw', 'yolo'):
            return jsonify({'ok': False, 'error': 'kind must be raw or yolo'}), 404
        return Response(
            _mjpeg_stream(state, kind),
            mimetype='multipart/x-mixed-replace; boundary=frame',
            headers={'Cache-Control': 'no-store, no-cache, must-revalidate, max-age=0'},
        )

    @app.get('/frame/<kind>.jpg')
    def latest_frame(kind):
        if kind not in ('raw', 'yolo'):
            return jsonify({'ok': False, 'error': 'kind must be raw or yolo'}), 404
        version, frame = state.latest_frame(kind)
        if frame is None:
            return jsonify({'ok': False, 'error': 'waiting for first frame'}), 503
        return Response(
            frame,
            mimetype='image/jpeg',
            headers={
                'Cache-Control': 'no-store, no-cache, must-revalidate, max-age=0',
                'Pragma': 'no-cache',
                'X-Frame-Version': str(version),
            },
        )

    @app.post('/detect')
    def detect():
        t0 = time.perf_counter()
        if 'image' not in request.files:
            return jsonify({'ok': False, 'error': 'missing multipart file field: image'}), 400
        try:
            original_jpeg, frame = _decode_image(request.files['image'])
        except Exception as exc:
            return jsonify({'ok': False, 'error': str(exc)}), 400

        h, w = frame.shape[:2]
        try:
            with model_lock:
                results = model.predict(
                    source=frame,
                    imgsz=args.imgsz,
                    conf=args.conf,
                    classes=[0] if args.person_only else None,
                    device=args.device,
                    verbose=False,
                )
        except Exception as exc:
            return jsonify({'ok': False, 'error': f'YOLO inference failed: {exc}'}), 500

        inference_ms = (time.perf_counter() - t0) * 1000.0
        detections = []
        if results and results[0].boxes is not None:
            boxes = results[0].boxes
            xyxy = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            clss = boxes.cls.cpu().numpy() if boxes.cls is not None else np.zeros(len(confs), dtype=np.float32)
            names = getattr(results[0], 'names', {}) or {}
            for box, conf, cls in zip(xyxy, confs, clss):
                class_id = int(cls)
                detections.append({
                    'bbox': [float(v) for v in box],
                    'conf': float(conf),
                    'class_id': class_id,
                    'label': str(names.get(class_id, class_id)),
                })

        capture_ros_sec = float(
            request.form.get('capture_ros_sec')
            or request.form.get('capture_wall_sec')
            or 0.0
        )
        wall_delta_sec = time.time() - capture_ros_sec if capture_ros_sec > 0.0 else -1.0
        capture_age_ms = (
            max(0.0, wall_delta_sec * 1000.0)
            if 0.0 <= wall_delta_sec <= 60.0 else -1.0
        )
        overlay = _draw_yolo_overlay(frame, detections, inference_ms)
        annotated_jpeg = _encode_jpeg(overlay, args.debug_jpeg_quality)
        state.update(
            original_jpeg, annotated_jpeg, len(detections),
            inference_ms, w, h, capture_age_ms,
        )

        return jsonify({
            'ok': True,
            'stamp_wall_sec': time.time(),
            'latency_ms': inference_ms,
            'capture_age_ms': capture_age_ms,
            'capture_ros_sec': capture_ros_sec,
            'image_width': int(w),
            'image_height': int(h),
            'detections': detections,
            'debug_url': f'http://127.0.0.1:{args.port}/',
        })

    return app


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', default='0.0.0.0')
    parser.add_argument('--port', type=int, default=5005)
    parser.add_argument('--model-path', default='yolo11n.pt')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--conf', type=float, default=0.20)
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--debug-jpeg-quality', type=int, default=80)
    parser.add_argument('--person-only', action='store_true', default=True)
    parser.add_argument('--all-classes', action='store_false', dest='person_only')
    return parser.parse_args()


def main():
    args = parse_args()
    app = build_app(args)
    logging.getLogger('werkzeug').setLevel(logging.WARNING)
    print(f'[flask_yolo_server] debug dashboard: http://127.0.0.1:{args.port}/', flush=True)
    app.run(host=args.host, port=args.port, threaded=True, use_reloader=False)


if __name__ == '__main__':
    main()
