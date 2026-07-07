"""Debug Stream — yolo_node annotated frame 을 MJPEG over HTTP 로 송출.

설계:
    - Flask 서버를 daemon thread 로 실행 (메인 loop 비차단)
    - update(frame) 는 ref 만 교체 (lock 없음, CPython atomic)
    - GET /stream.mjpg : 가장 최근 frame 을 JPEG 인코딩해서 스트림
    - 클라이언트 없으면 인코딩 비용 0 (generator 가 안 돔)

사용:
    stream = DebugStream(port=8080, fps=15, quality=70)
    stream.start()
    ...
    stream.update(annotated_frame)   # 메인 loop 에서

브라우저:  http://<host>:<port>/
원시 스트림:  http://<host>:<port>/stream.mjpg
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Optional

import cv2
import numpy as np


class DebugStream:
    def __init__(self, port: int = 8080, fps: int = 15, quality: int = 70):
        self.port = port
        self.fps = max(1, fps)
        self.quality = max(10, min(95, quality))
        self.latest: Optional[np.ndarray] = None
        self._started = False
        self._app = None
        self._thread = None

    def update(self, frame: np.ndarray) -> None:
        """메인 loop 에서 호출. 복사 없이 ref 만 교체."""
        self.latest = frame

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

        # werkzeug request 로그 끄기
        logging.getLogger('werkzeug').setLevel(logging.WARNING)

        self._app = app
        self._thread = threading.Thread(target=self._serve, daemon=True)
        self._thread.start()
        self._started = True

    def _serve(self):
        self._app.run(host='0.0.0.0', port=self.port,
                      debug=False, use_reloader=False, threaded=True)

    # ----- Flask routes -----

    def _index(self):
        return (
            '<html><head><title>OMX Debug Stream</title>'
            '<style>body{margin:0;background:#111;display:flex;'
            'justify-content:center;align-items:center;min-height:100vh}'
            'img{max-width:100%;height:auto}</style></head>'
            '<body><img src="/stream.mjpg"></body></html>')

    def _stream_view(self):
        from flask import Response
        return Response(self._gen(),
                        mimetype='multipart/x-mixed-replace; boundary=frame')

    def _gen(self):
        interval = 1.0 / self.fps
        encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), self.quality]
        last_t = 0.0

        while True:
            now = time.monotonic()
            dt = now - last_t
            if dt < interval:
                time.sleep(interval - dt)
            last_t = time.monotonic()

            frame = self.latest
            if frame is None:
                time.sleep(0.1)
                continue

            ok, buf = cv2.imencode('.jpg', frame, encode_params)
            if not ok:
                continue

            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n'
                   + buf.tobytes() + b'\r\n')