"""DebugStream — yolo_node annotated frame + state 를 웹 대시보드로 송출.

기존 단일 파일 omx/debug_stream.py 를 패키지로 승격.
yolo_node 에서 `from omx.debug_stream import DebugStream` 은 그대로 동작.

엔드포인트:
    GET /              base.html (Live + Ops 탭)
    GET /stream.mjpg   MJPEG 영상 스트림
    GET /events        SSE 상태 스트림 (5Hz)
    GET /state.json    현재 상태 1회 fetch (디버깅용)

사용:
    stream = DebugStream(port=8080, fps=15, quality=70)
    stream.start()
    ...
    stream.update(annotated_frame)         # 매 tick
    stream.update_state(snapshot_dict)     # 매 tick (선택)
"""

from __future__ import annotations

import json
import logging
import threading
import time

import cv2
import numpy as np

from .state_bus import StateBus


class DebugStream:
    def __init__(
        self,
        port: int = 8080,
        fps: int = 10,
        quality: int = 52,
        width: int = 640,
        height: int = 360,
    ):
        self.port = port
        self.fps = max(1, fps)
        self.quality = max(10, min(95, quality))
        self.width = max(0, int(width))
        self.height = max(0, int(height))
        self.bus = StateBus()
        self._started = False
        self._app = None
        self._thread = None
        self._jpeg_condition = threading.Condition()
        self._jpeg_bytes: bytes | None = None
        self._jpeg_seq = 0
        self._jpeg_source_seq = -1
        self._jpeg_encode_ms = 0.0
        self._jpeg_size_kb = 0.0

    # ----- public API -----

    def update(self, frame) -> None:
        """메인 loop 에서 호출. annotated frame 푸시."""
        self.bus.update_frame(frame)
        _, seq = self.bus.get_frame()
        if seq != self._jpeg_source_seq:
            ok, encoded = self._encode_frame(frame)
            if ok:
                payload = encoded.tobytes()
                with self._jpeg_condition:
                    self._jpeg_bytes = payload
                    self._jpeg_seq += 1
                    self._jpeg_source_seq = seq
                    self._jpeg_size_kb = len(payload) / 1024.0
                    self._jpeg_condition.notify_all()

    def update_state(self, snapshot: dict) -> None:
        """메인 loop 에서 호출. 상태 dict 푸시 (SSE 로 전송됨)."""
        self.bus.update_state(snapshot)

    def start(self) -> None:
        """Flask 를 daemon thread 로 시작. Flask 는 lazy import."""
        if self._started:
            return
        try:
            from flask import Flask
        except ImportError:
            raise RuntimeError("Flask 미설치. pip install flask")

        app = Flask(__name__)
        app.add_url_rule('/', 'index', self._index)
        app.add_url_rule('/stream.mjpg', 'stream', self._stream_view)
        app.add_url_rule('/frame.jpg', 'frame', self._frame_view)
        app.add_url_rule('/events', 'events', self._events_view)
        app.add_url_rule('/state.json', 'state_json', self._state_json_view)

        logging.getLogger('werkzeug').setLevel(logging.WARNING)

        self._app = app
        self._thread = threading.Thread(target=self._serve, daemon=True)
        self._thread.start()
        self._started = True

    def _serve(self):
        self._app.run(host='0.0.0.0', port=self.port,
                      debug=False, use_reloader=False, threaded=True)

    # ----- routes -----

    def _index(self):
        from flask import render_template
        return render_template('base.html')

    def _stream_view(self):
        from flask import Response
        return Response(self._frame_gen(),
                        mimetype='multipart/x-mixed-replace; boundary=frame')

    def _frame_view(self):
        """Single latest JPEG for polling dashboards (reload-safe)."""
        from flask import Response

        frame, _ = self.bus.get_frame()
        jpeg, seq = self._latest_jpeg()
        if frame is None or jpeg is None:
            return Response('waiting for OMX frame\n', status=503, mimetype='text/plain')
        return Response(
            jpeg, mimetype='image/jpeg',
            headers={
                'Cache-Control': 'no-store, no-cache, max-age=0',
                'X-Frame-Version': str(seq),
            },
        )

    def _events_view(self):
        from flask import Response
        return Response(
            self._sse_gen(),
            mimetype='text/event-stream',
            headers={
                'Cache-Control': 'no-cache',
                'X-Accel-Buffering': 'no',  # reverse proxy 버퍼 끔
            })

    def _state_json_view(self):
        from flask import jsonify
        state, _ = self.bus.get_state()
        return jsonify(state)

    # ----- generators -----

    def _frame_gen(self):
        """MJPEG generator. seq 비교로 stale frame 스킵.

        A camera open/read failure used to yield no bytes at all, which makes
        browser ``<img>`` elements look permanently black.  Keep the stream
        alive with a clear diagnostic frame until the detector reconnects.
        """
        last_seq = -1
        next_emit = 0.0

        while True:
            now = time.monotonic()
            if now < next_emit:
                time.sleep(next_emit - now)
            jpeg, seq = self._wait_for_jpeg(last_seq)
            if jpeg is None:
                jpeg = self._waiting_jpeg()
            elif seq == last_seq:
                continue
            else:
                last_seq = seq
            next_emit = time.monotonic() + (1.0 / self.fps)

            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n'
                   + jpeg + b'\r\n')

    def _latest_jpeg(self) -> tuple[bytes | None, int]:
        with self._jpeg_condition:
            return self._jpeg_bytes, self._jpeg_seq

    def _wait_for_jpeg(self, previous_seq: int) -> tuple[bytes | None, int]:
        with self._jpeg_condition:
            self._jpeg_condition.wait_for(
                lambda: self._jpeg_seq != previous_seq and self._jpeg_bytes is not None,
                timeout=max(0.1, 1.0 / self.fps),
            )
            return self._jpeg_bytes, self._jpeg_seq

    def _waiting_jpeg(self) -> bytes:
        frame = np.zeros((360, 640, 3), dtype=np.uint8)
        cv2.putText(
            frame, 'WAITING FOR OMX CAMERA', (78, 160),
            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 190, 255), 2,
        )
        cv2.putText(
            frame, 'camera reconnecting / yolo starting', (115, 205),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1,
        )
        ok, buf = self._encode_frame(frame)
        return buf.tobytes() if ok else b''

    def _encode_frame(self, frame, *, params=None):
        started = time.perf_counter()
        if (
            self.width > 0
            and self.height > 0
            and frame is not None
            and len(frame.shape) >= 2
            and (frame.shape[1] != self.width or frame.shape[0] != self.height)
        ):
            frame = self._resize_full_frame(frame)
        params = params or [int(cv2.IMWRITE_JPEG_QUALITY), self.quality]
        ok, encoded = cv2.imencode('.jpg', frame, params)
        self._jpeg_encode_ms = (time.perf_counter() - started) * 1000.0
        return ok, encoded

    def _resize_full_frame(self, frame):
        source_h, source_w = frame.shape[:2]
        if source_w <= 0 or source_h <= 0 or self.width <= 0 or self.height <= 0:
            return frame
        source_ratio = source_w / float(source_h)
        target_ratio = self.width / float(self.height)
        if abs(source_ratio - target_ratio) <= 1.0e-3:
            return cv2.resize(
                frame,
                (self.width, self.height),
                interpolation=(
                    cv2.INTER_AREA
                    if self.width < source_w or self.height < source_h
                    else cv2.INTER_LINEAR
                ),
            )
        scale = min(self.width / float(source_w), self.height / float(source_h))
        content_w = max(1, int(round(source_w * scale)))
        content_h = max(1, int(round(source_h * scale)))
        resized = cv2.resize(
            frame,
            (content_w, content_h),
            interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR,
        )
        pad_x = max(0, (self.width - content_w) // 2)
        pad_y = max(0, (self.height - content_h) // 2)
        right = max(0, self.width - content_w - pad_x)
        bottom = max(0, self.height - content_h - pad_y)
        return cv2.copyMakeBorder(
            resized,
            pad_y,
            bottom,
            pad_x,
            right,
            cv2.BORDER_CONSTANT,
            value=(0, 0, 0),
        )

    def _sse_gen(self):
        """SSE: state 변경 시에만 push. 최대 5Hz."""
        interval = 0.2
        last_seq = -1
        while True:
            time.sleep(interval)
            state, seq = self.bus.get_state()
            if seq == last_seq:
                continue
            last_seq = seq
            try:
                payload = json.dumps(state, default=str)
            except (TypeError, ValueError):
                continue
            yield f"data: {payload}\n\n"
