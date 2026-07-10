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
    def __init__(self, port: int = 8080, fps: int = 15, quality: int = 70):
        self.port = port
        self.fps = max(1, fps)
        self.quality = max(10, min(95, quality))
        self.bus = StateBus()
        self._started = False
        self._app = None
        self._thread = None

    # ----- public API -----

    def update(self, frame) -> None:
        """메인 loop 에서 호출. annotated frame 푸시."""
        self.bus.update_frame(frame)

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
        interval = 1.0 / self.fps
        params = [int(cv2.IMWRITE_JPEG_QUALITY), self.quality]
        last_seq = -1

        while True:
            time.sleep(interval)
            frame, seq = self.bus.get_frame()
            if frame is None:
                frame = np.zeros((360, 640, 3), dtype=np.uint8)
                cv2.putText(
                    frame, 'WAITING FOR OMX CAMERA', (78, 160),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 190, 255), 2,
                )
                cv2.putText(
                    frame, 'camera reconnecting / yolo starting', (115, 205),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1,
                )
                # Emit the diagnostic periodically even though its source
                # frame sequence has not changed.
                last_seq = seq
            elif seq == last_seq:
                continue
            else:
                last_seq = seq

            ok, buf = cv2.imencode('.jpg', frame, params)
            if not ok:
                continue

            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n'
                   + buf.tobytes() + b'\r\n')

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
