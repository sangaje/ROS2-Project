#!/usr/bin/env python3

from __future__ import annotations

import argparse
import logging
import queue
import threading
import time
from collections import deque
from dataclasses import dataclass, field

import numpy as np


class InferenceBusyError(RuntimeError):
    pass


DEBUG_PAGE = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>TB3 YOLO Debug</title>
  <style>
    :root {
      color-scheme: dark;
      --bg: #0c1016;
      --panel: #151b24;
      --panel-2: #1d2530;
      --line: #2a3544;
      --text: #edf3fb;
      --muted: #93a4b8;
      --good: #42d392;
      --warn: #ffd166;
      --bad: #ff6b6b;
      --accent: #72c7ff;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    * { box-sizing: border-box; }
    body { margin: 0; background: radial-gradient(circle at top, #152033 0, var(--bg) 42%); color: var(--text); }
    header {
      display: grid;
      grid-template-columns: 1fr auto;
      gap: 14px;
      align-items: center;
      padding: 14px 18px;
      background: rgba(12, 16, 22, 0.92);
      border-bottom: 1px solid var(--line);
      position: sticky;
      top: 0;
      z-index: 2;
      backdrop-filter: blur(10px);
    }
    h1 { margin: 0; font-size: 20px; letter-spacing: 0.2px; }
    .subtitle { margin-top: 4px; color: var(--muted); font-size: 12px; }
    .pill {
      display: inline-flex;
      align-items: center;
      gap: 7px;
      padding: 7px 10px;
      border: 1px solid var(--line);
      border-radius: 999px;
      background: var(--panel);
      color: var(--muted);
      font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
      font-size: 12px;
      white-space: nowrap;
    }
    .dot { width: 8px; height: 8px; border-radius: 999px; background: var(--warn); box-shadow: 0 0 14px currentColor; }
    .ok .dot { background: var(--good); }
    .stale .dot { background: var(--warn); }
    .dead .dot { background: var(--bad); }
    main { display: grid; grid-template-columns: minmax(0, 2fr) minmax(320px, 0.9fr); gap: 12px; padding: 12px; }
    .video-grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 12px; }
    section, aside {
      background: rgba(21, 27, 36, 0.94);
      border: 1px solid var(--line);
      border-radius: 12px;
      overflow: hidden;
      box-shadow: 0 10px 32px rgba(0,0,0,0.22);
    }
    h2 { margin: 0; padding: 10px 12px; font-size: 14px; background: var(--panel-2); color: #dbe8f7; }
    img { width: 100%; display: block; min-height: 260px; object-fit: contain; background: #030509; }
    .cards { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 8px; padding: 10px; }
    .card { padding: 10px; border: 1px solid var(--line); border-radius: 10px; background: #101722; }
    .label { color: var(--muted); font-size: 11px; text-transform: uppercase; letter-spacing: 0.06em; }
    .value { margin-top: 5px; font: 700 22px/1.1 ui-monospace, SFMono-Regular, Menlo, monospace; }
    .value.good { color: var(--good); }
    .value.warn { color: var(--warn); }
    .value.bad { color: var(--bad); }
    .wide { grid-column: 1 / -1; }
    #detections { padding: 0 10px 10px; }
    .det { display: grid; grid-template-columns: 1fr auto; gap: 6px; padding: 8px 0; border-top: 1px solid var(--line); font-size: 13px; }
    .det:first-child { border-top: 0; }
    .muted { color: var(--muted); }
    .mono { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }
    @media (max-width: 1100px) {
      main { grid-template-columns: 1fr; }
      .video-grid { grid-template-columns: 1fr; }
    }
    @media (max-width: 640px) {
      header { grid-template-columns: 1fr; }
      .cards { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <header>
    <div>
      <h1>TB3 YOLO Debug</h1>
      <div class="subtitle">Robot camera JPEG → PC Flask YOLO → ROS detection JSON</div>
    </div>
    <div id="connection" class="pill stale"><span class="dot"></span><span id="connectionText">waiting for frames</span></div>
  </header>
  <main>
    <div class="video-grid">
      <section><h2>Camera input</h2><img src="/stream/raw.mjpg" alt="raw camera stream"></section>
      <section><h2>YOLO overlay</h2><img src="/stream/yolo.mjpg" alt="YOLO stream"></section>
    </div>
    <aside>
      <h2>Live status</h2>
      <div class="cards">
        <div class="card"><div class="label">People</div><div id="people" class="value">—</div></div>
        <div class="card"><div class="label">FPS</div><div id="fps" class="value">—</div></div>
        <div class="card"><div class="label">YOLO</div><div id="latency" class="value">—</div></div>
        <div class="card"><div class="label">Capture age</div><div id="captureAge" class="value">—</div></div>
        <div class="card"><div class="label">Predict only</div><div id="predictMs" class="value">—</div></div>
        <div class="card"><div class="label">Decode/Post</div><div id="decodePostMs" class="value">—</div></div>
        <div class="card"><div class="label">Frame age</div><div id="frameAge" class="value">—</div></div>
        <div class="card"><div class="label">Size</div><div id="size" class="value">—</div></div>
        <div class="card wide"><div class="label">Frames</div><div id="frames" class="value">—</div></div>
      </div>
      <h2>Detections</h2>
      <div id="detections"><div class="muted">No detections yet.</div></div>
    </aside>
  </main>
  <script>
    const fmt = (v, digits = 1, suffix = '') => Number.isFinite(v) && v >= 0 ? `${v.toFixed(digits)}${suffix}` : '—';
    function level(value, good, warn) {
      if (!Number.isFinite(value) || value < 0) return '';
      if (value <= good) return 'good';
      if (value <= warn) return 'warn';
      return 'bad';
    }
    function setValue(id, text, cls = '') {
      const el = document.getElementById(id);
      el.textContent = text;
      el.className = `value ${cls}`.trim();
    }
    function setConnection(status, text) {
      const el = document.getElementById('connection');
      el.className = `pill ${status}`;
      document.getElementById('connectionText').textContent = text;
    }
    async function refreshStatus() {
      try {
        const s = await (await fetch('/api/status', {cache: 'no-store'})).json();
        const age = Number(s.frame_age_sec);
        if (!s.ok) setConnection('stale', 'waiting for first frame');
        else if (age > 2.0) setConnection('dead', `stale ${age.toFixed(1)}s`);
        else if (age > 0.7) setConnection('stale', `slow ${age.toFixed(1)}s`);
        else setConnection('ok', 'live');

        setValue('people', String(s.people ?? 0), s.people > 0 ? 'good' : '');
        const fps = Number(s.fps);
        setValue('fps', fmt(fps, 1), Number.isFinite(fps) ? (fps >= 4 ? 'good' : fps >= 2 ? 'warn' : 'bad') : '');
        setValue('latency', fmt(Number(s.latency_ms), 1, 'ms'), level(Number(s.latency_ms), 80, 180));
        setValue('captureAge', fmt(Number(s.capture_age_ms), 1, 'ms'), level(Number(s.capture_age_ms), 200, 600));
        setValue('predictMs', fmt(Number(s.predict_ms), 1, 'ms'), level(Number(s.predict_ms), 50, 120));
        const decodePost = Number(s.decode_ms) + Number(s.post_ms);
        setValue('decodePostMs', fmt(decodePost, 1, 'ms'), level(decodePost, 30, 90));
        setValue('frameAge', fmt(age * 1000.0, 0, 'ms'), level(age * 1000.0, 250, 800));
        setValue('size', s.image_width && s.image_height ? `${s.image_width}×${s.image_height}` : '—');
        setValue('frames', String(s.frames ?? 0));

        const root = document.getElementById('detections');
        const detections = Array.isArray(s.detections) ? s.detections : [];
        if (!detections.length) {
          root.innerHTML = '<div class="muted">No person in latest frame.</div>';
        } else {
          root.innerHTML = detections.map((d, i) => {
            const conf = Number(d.conf ?? d.confidence ?? 0);
            const box = Array.isArray(d.bbox) ? d.bbox.map(v => Number(v).toFixed(0)).join(', ') : 'bbox unavailable';
            return `<div class="det"><div><b>${i + 1}. ${d.label ?? 'person'}</b><div class="muted mono">${box}</div></div><div class="mono">${(conf * 100).toFixed(0)}%</div></div>`;
          }).join('');
        }
      } catch (_) {
        setConnection('dead', 'server disconnected');
      }
    }
    setInterval(refreshStatus, 500);
    refreshStatus();
  </script>
</body>
</html>
"""


@dataclass
class DebugState:
    condition: threading.Condition = field(default_factory=threading.Condition)
    raw_jpeg: bytes | None = None
    yolo_jpeg: bytes | None = None
    raw_version: int = 0
    yolo_version: int = 0
    raw_frames: int = 0
    yolo_frames: int = 0
    people: int = 0
    latency_ms: float = 0.0
    decode_ms: float = 0.0
    predict_ms: float = 0.0
    post_ms: float = 0.0
    image_width: int = 0
    image_height: int = 0
    capture_age_ms: float = -1.0
    last_raw_wall_sec: float = 0.0
    last_yolo_wall_sec: float = 0.0
    detections: list = field(default_factory=list)
    raw_wall_times: deque = field(default_factory=lambda: deque(maxlen=90))
    yolo_wall_times: deque = field(default_factory=lambda: deque(maxlen=90))

    def update_raw(self, raw_jpeg, width, height, capture_age_ms):
        with self.condition:
            self.raw_jpeg = raw_jpeg
            self.raw_version += 1
            self.raw_frames += 1
            self.image_width = width
            self.image_height = height
            self.capture_age_ms = capture_age_ms
            self.last_raw_wall_sec = time.time()
            self.raw_wall_times.append(self.last_raw_wall_sec)
            self.condition.notify_all()

    def update_yolo(
        self, yolo_jpeg, people, latency_ms, width, height, capture_age_ms,
        detections=None, decode_ms=0.0, predict_ms=0.0, post_ms=0.0,
    ):
        with self.condition:
            self.yolo_jpeg = yolo_jpeg
            self.yolo_version += 1
            self.yolo_frames += 1
            self.people = people
            self.latency_ms = latency_ms
            self.decode_ms = decode_ms
            self.predict_ms = predict_ms
            self.post_ms = post_ms
            self.image_width = width
            self.image_height = height
            self.capture_age_ms = capture_age_ms
            self.last_yolo_wall_sec = time.time()
            self.detections = list(detections or [])
            self.yolo_wall_times.append(self.last_yolo_wall_sec)
            self.condition.notify_all()

    def status(self):
        with self.condition:
            now = time.time()
            raw_age = now - self.last_raw_wall_sec if self.last_raw_wall_sec else -1.0
            yolo_age = now - self.last_yolo_wall_sec if self.last_yolo_wall_sec else -1.0
            raw_fps = 0.0
            if len(self.raw_wall_times) >= 2:
                dt = self.raw_wall_times[-1] - self.raw_wall_times[0]
                if dt > 1e-6:
                    raw_fps = (len(self.raw_wall_times) - 1) / dt
            yolo_fps = 0.0
            if len(self.yolo_wall_times) >= 2:
                dt = self.yolo_wall_times[-1] - self.yolo_wall_times[0]
                if dt > 1e-6:
                    yolo_fps = (len(self.yolo_wall_times) - 1) / dt
            return {
                'ok': self.raw_frames > 0,
                'frames': self.raw_frames,
                'raw_frames': self.raw_frames,
                'yolo_frames': self.yolo_frames,
                'people': self.people,
                'fps': raw_fps,
                'raw_fps': raw_fps,
                'yolo_fps': yolo_fps,
                'latency_ms': self.latency_ms,
                'decode_ms': self.decode_ms,
                'predict_ms': self.predict_ms,
                'post_ms': self.post_ms,
                'image_width': self.image_width,
                'image_height': self.image_height,
                'capture_age_ms': self.capture_age_ms,
                'frame_age_sec': raw_age,
                'raw_frame_age_sec': raw_age,
                'yolo_frame_age_sec': yolo_age,
                'detections': self.detections,
            }

    def latest_frame(self, kind):
        with self.condition:
            frame = self.raw_jpeg if kind == 'raw' else self.yolo_jpeg
            version = self.raw_version if kind == 'raw' else self.yolo_version
            return version, frame

    def wait_for_frame(self, kind, previous_version):
        with self.condition:
            self.condition.wait_for(
                lambda: (
                    self.raw_version if kind == 'raw' else self.yolo_version
                ) != previous_version
                and (self.raw_jpeg if kind == 'raw' else self.yolo_jpeg) is not None,
                timeout=1.0,
            )
            frame = self.raw_jpeg if kind == 'raw' else self.yolo_jpeg
            version = self.raw_version if kind == 'raw' else self.yolo_version
            return version, frame


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


def _normalize_device(device):
    text = str(device).strip()
    if text.isdigit():
        return f'cuda:{text}'
    return text


def _cuda_synchronize_if_needed(device):
    if 'cuda' not in str(device).lower():
        return
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:
        pass


def _request_capture_age_sec(req):
    capture_wall_sec = float(req.form.get('capture_wall_sec') or 0.0)
    if capture_wall_sec > 0.0:
        wall_delta_sec = time.time() - capture_wall_sec
        if 0.0 <= wall_delta_sec <= 60.0:
            return wall_delta_sec

    robot_frame_age_ms_at_send = float(req.form.get('robot_frame_age_ms_at_send') or -1.0)
    if robot_frame_age_ms_at_send >= 0.0:
        return robot_frame_age_ms_at_send / 1000.0

    return -1.0


def build_app(args):
    from flask import Flask, Response, jsonify, request
    from ultralytics import YOLO

    app = Flask(__name__)
    args.device = _normalize_device(args.device)
    model = YOLO(args.model_path)
    use_half = bool(args.half) and str(args.device).lower() not in ('cpu', 'none', '')
    use_fast_forward = bool(args.fast_forward)
    fast_net = None
    letterbox = None
    non_max_suppression = None
    scale_boxes = None
    torch = None
    class_names = getattr(model, 'names', {}) or {}
    if use_fast_forward:
        try:
            import torch as _torch
            from ultralytics.data.augment import LetterBox
            from ultralytics.utils.nms import non_max_suppression as _nms
            from ultralytics.utils.ops import scale_boxes as _scale_boxes

            torch = _torch
            fast_net = model.model.to(args.device).eval()
            if use_half:
                fast_net = fast_net.half()
            letterbox = LetterBox(new_shape=(int(args.imgsz), int(args.imgsz)), auto=False, stride=32)
            non_max_suppression = _nms
            scale_boxes = _scale_boxes
            class_names = getattr(model.model, 'names', class_names) or class_names
        except Exception as exc:
            print(f'[flask_yolo_server] fast forward setup failed; falling back to model.predict: {exc}', flush=True)
            use_fast_forward = False
    warmup_ms = -1.0
    try:
        if 'cuda' in str(args.device).lower():
            torch.backends.cudnn.benchmark = True
            try:
                torch.set_float32_matmul_precision('high')
            except Exception:
                pass
        dummy = np.zeros((max(32, int(args.imgsz)), max(32, int(args.imgsz)), 3), dtype=np.uint8)
        t_warmup = time.perf_counter()
        if use_fast_forward:
            im = letterbox(image=dummy)
            im = im[..., ::-1].transpose(2, 0, 1)
            im = np.ascontiguousarray(im)
            tensor = torch.from_numpy(im).to(args.device, non_blocking=True)
            tensor = tensor.half() if use_half else tensor.float()
            tensor = tensor.unsqueeze(0) / 255.0
            with torch.inference_mode():
                pred = fast_net(tensor)
                if isinstance(pred, (tuple, list)):
                    pred = pred[0]
                non_max_suppression(
                    pred,
                    conf_thres=args.conf,
                    iou_thres=args.iou,
                    classes=[0] if args.person_only else None,
                    max_det=args.max_det,
                )
        else:
            model.predict(
                source=dummy,
                imgsz=args.imgsz,
                conf=args.conf,
                classes=[0] if args.person_only else None,
                device=args.device,
                half=use_half,
                verbose=False,
            )
        _cuda_synchronize_if_needed(args.device)
        warmup_ms = (time.perf_counter() - t_warmup) * 1000.0
        print(
            f'[flask_yolo_server] model ready | device={args.device} half={use_half} '
            f'imgsz={args.imgsz} fast_forward={use_fast_forward} warmup_ms={warmup_ms:.1f}',
            flush=True,
        )
    except Exception as exc:
        print(f'[flask_yolo_server] warmup failed: {exc}', flush=True)
    state = DebugState()
    runtime = {'warmup_ms': warmup_ms}

    class InferenceWorker:
        def __init__(self):
            self.jobs = queue.Queue(maxsize=1)
            self.inflight_lock = threading.Lock()
            self.ready = threading.Event()
            self.thread = threading.Thread(
                target=self.loop,
                name='flask_yolo_inference_worker',
                daemon=True,
            )
            self.thread.start()
            self.ready.wait(timeout=10.0)

        def warmup_in_worker_thread(self):
            dummy = np.zeros((max(32, int(args.imgsz)), max(32, int(args.imgsz)), 3), dtype=np.uint8)
            t_warmup = time.perf_counter()
            self.run_predict(dummy)
            runtime['warmup_ms'] = (time.perf_counter() - t_warmup) * 1000.0
            print(
                f'[flask_yolo_server] inference worker ready | device={args.device} half={use_half} '
                f'imgsz={args.imgsz} fast_forward={use_fast_forward} worker_warmup_ms={runtime["warmup_ms"]:.1f}',
                flush=True,
            )

        def loop(self):
            try:
                self.warmup_in_worker_thread()
            except Exception as exc:
                print(f'[flask_yolo_server] inference worker warmup failed: {exc}', flush=True)
            finally:
                self.ready.set()

            while True:
                frame, result_queue = self.jobs.get()
                try:
                    result_queue.put((True, self.run_predict(frame)))
                except Exception as exc:
                    result_queue.put((False, exc))

        def infer(self, frame):
            if args.max_queue_wait_sec > 0.0:
                acquired = self.inflight_lock.acquire(timeout=args.max_queue_wait_sec)
            else:
                acquired = self.inflight_lock.acquire(blocking=False)
            if not acquired:
                raise InferenceBusyError('inference worker busy; dropped stale request instead of queueing')

            result_queue = queue.Queue(maxsize=1)
            try:
                try:
                    self.jobs.put_nowait((frame, result_queue))
                except queue.Full as exc:
                    raise InferenceBusyError('inference job queue full; dropped request') from exc
                ok, payload = result_queue.get()
                if not ok:
                    raise payload
                return payload
            finally:
                self.inflight_lock.release()

        def run_predict(self, frame):
            t_predict0 = time.perf_counter()
            if use_fast_forward:
                im = letterbox(image=frame)
                im = im[..., ::-1].transpose(2, 0, 1)
                im = np.ascontiguousarray(im)
                tensor = torch.from_numpy(im).to(args.device, non_blocking=True)
                tensor = tensor.half() if use_half else tensor.float()
                tensor = tensor.unsqueeze(0) / 255.0
                with torch.inference_mode():
                    pred = fast_net(tensor)
                    if isinstance(pred, (tuple, list)):
                        pred = pred[0]
                    results = non_max_suppression(
                        pred,
                        conf_thres=args.conf,
                        iou_thres=args.iou,
                        classes=[0] if args.person_only else None,
                        max_det=args.max_det,
                    )
                input_shape = tensor.shape[2:]
            else:
                results = model.predict(
                    source=frame,
                    imgsz=args.imgsz,
                    conf=args.conf,
                    classes=[0] if args.person_only else None,
                    device=args.device,
                    half=use_half,
                    verbose=False,
                )
                input_shape = None
            _cuda_synchronize_if_needed(args.device)
            predict_ms = (time.perf_counter() - t_predict0) * 1000.0
            return results, input_shape, predict_ms

    inference_worker = InferenceWorker()

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
            'half': use_half,
            'imgsz': args.imgsz,
            'fast_forward': use_fast_forward,
            'warmup_ms': runtime.get('warmup_ms', -1.0),
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
            t_decode0 = time.perf_counter()
            original_jpeg, frame = _decode_image(request.files['image'])
            decode_ms = (time.perf_counter() - t_decode0) * 1000.0
        except Exception as exc:
            return jsonify({'ok': False, 'error': str(exc)}), 400

        h, w = frame.shape[:2]
        capture_ros_sec = float(request.form.get('capture_ros_sec') or 0.0)
        capture_wall_sec = float(request.form.get('capture_wall_sec') or 0.0)
        robot_frame_age_ms_at_send = float(request.form.get('robot_frame_age_ms_at_send') or -1.0)
        raw_age_sec = _request_capture_age_sec(request)
        capture_age_ms = max(0.0, raw_age_sec * 1000.0) if raw_age_sec >= 0.0 else -1.0

        # Keep the debug camera view live even when this frame is too old or YOLO is busy.
        state.update_raw(original_jpeg, int(w), int(h), capture_age_ms)

        if args.max_capture_age_sec > 0.0 and raw_age_sec > args.max_capture_age_sec:
            return jsonify({
                'ok': False,
                'stale': True,
                'error': 'stale frame rejected before inference',
                'capture_age_ms': capture_age_ms,
                'max_capture_age_ms': args.max_capture_age_sec * 1000.0,
                'detections': [],
            })

        try:
            results, input_shape, predict_ms = inference_worker.infer(frame)
        except InferenceBusyError as exc:
            busy_age_sec = _request_capture_age_sec(request)
            return jsonify({
                'ok': False,
                'stale': True,
                'busy': True,
                'error': str(exc),
                'capture_age_ms': max(0.0, busy_age_sec * 1000.0) if busy_age_sec >= 0.0 else -1.0,
                'detections': [],
            })
        except Exception as exc:
            return jsonify({'ok': False, 'error': f'YOLO inference failed: {exc}'}), 500

        t_post0 = time.perf_counter()
        detections = []
        if use_fast_forward:
            det = results[0] if results else None
            if det is not None and len(det):
                det = det.clone()
                det[:, :4] = scale_boxes(input_shape, det[:, :4], frame.shape).round()
                for row in det.detach().float().cpu().numpy():
                    x1, y1, x2, y2, conf, cls = row[:6]
                    class_id = int(cls)
                    detections.append({
                        'bbox': [float(x1), float(y1), float(x2), float(y2)],
                        'conf': float(conf),
                        'class_id': class_id,
                        'label': str(class_names.get(class_id, class_id)),
                    })
        elif results and results[0].boxes is not None:
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

        post_age_sec = _request_capture_age_sec(request)
        if post_age_sec >= 0.0:
            capture_age_ms = max(0.0, post_age_sec * 1000.0)
        if (
            args.max_capture_age_sec > 0.0
            and capture_age_ms >= 0.0
            and capture_age_ms > args.max_capture_age_sec * 1000.0
        ):
            return jsonify({
                'ok': False,
                'stale': True,
                'error': 'stale frame rejected after inference',
                'capture_age_ms': capture_age_ms,
                'max_capture_age_ms': args.max_capture_age_sec * 1000.0,
                'detections': [],
            })

        overlay = _draw_yolo_overlay(frame, detections, predict_ms)
        annotated_jpeg = _encode_jpeg(overlay, args.debug_jpeg_quality)
        post_ms = (time.perf_counter() - t_post0) * 1000.0
        inference_ms = (time.perf_counter() - t0) * 1000.0
        state.update_yolo(
            annotated_jpeg, len(detections),
            inference_ms, w, h, capture_age_ms, detections,
            decode_ms=decode_ms, predict_ms=predict_ms, post_ms=post_ms,
        )

        return jsonify({
            'ok': True,
            'stamp_wall_sec': time.time(),
            'latency_ms': inference_ms,
            'decode_ms': decode_ms,
            'predict_ms': predict_ms,
            'post_ms': post_ms,
            'capture_age_ms': capture_age_ms,
            'capture_ros_sec': capture_ros_sec,
            'capture_wall_sec': capture_wall_sec,
            'robot_frame_age_ms_at_send': robot_frame_age_ms_at_send,
            'image_width': int(w),
            'image_height': int(h),
            'detections': detections,
            'debug_url': f'http://127.0.0.1:{args.port}/',
        })

    return app


def parse_args():
    def as_bool(value):
        if isinstance(value, bool):
            return value
        return str(value).strip().lower() in ('1', 'true', 'yes', 'on')

    parser = argparse.ArgumentParser()
    parser.add_argument('--host', default='0.0.0.0')
    parser.add_argument('--port', type=int, default=5005)
    parser.add_argument('--model-path', default='yolo11s.pt')
    parser.add_argument('--device', default='0')
    parser.add_argument('--half', type=as_bool, default=True)
    parser.add_argument('--fast-forward', type=as_bool, default=True)
    parser.add_argument('--conf', type=float, default=0.20)
    parser.add_argument('--iou', type=float, default=0.45)
    parser.add_argument('--max-det', type=int, default=64)
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--debug-jpeg-quality', type=int, default=80)
    parser.add_argument('--max-capture-age-sec', type=float, default=0.8)
    parser.add_argument('--max-queue-wait-sec', type=float, default=0.0)
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
