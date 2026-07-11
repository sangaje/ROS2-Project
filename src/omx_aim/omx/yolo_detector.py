"""YOLO 기반 표적 검출.

OpenCV VideoCapture + Ultralytics YOLO. ROS 의존성 없음.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import cv2
from ultralytics import YOLO

from omx.config import Config


def _model_backend(model_path: str) -> str:
    suffix = Path(str(model_path)).suffix.lower()
    if suffix in ('.engine', '.plan'):
        return 'tensorrt'
    return 'pytorch'


def _safe_model_names(model) -> dict:
    try:
        names = getattr(model, "names", {}) or {}
    except Exception:
        names = {}
    return names if isinstance(names, dict) else {}


def _resolve_supported_device(requested: str) -> tuple[str, str | None]:
    """Prevent an unsupported Jetson CUDA binary from killing OMX video."""
    device = str(requested).strip()
    if device.isdigit():
        device = f'cuda:{device}'
    if device.lower() in ('cpu', 'none', ''):
        return 'cpu', None
    if not device.lower().startswith('cuda'):
        return device, None
    try:
        import torch
        if not torch.cuda.is_available():
            return 'cpu', 'CUDA unavailable'
        index = int(device.split(':', 1)[1]) if ':' in device else 0
        major, minor = torch.cuda.get_device_capability(index)
        needed = f'sm_{major}{minor}'
        supported = set(torch.cuda.get_arch_list())
        if not supported or needed not in supported:
            return 'cpu', (
                f'GPU CC {major}.{minor} requires {needed}; '
                f'torch supports {sorted(supported) or ["none"]}'
            )
    except Exception as exc:  # noqa: BLE001
        return 'cpu', f'CUDA capability check failed: {exc}'
    return device, None


class YoloDetector:
    """카메라 프레임 + YOLO 검출 + 영상 중심 기준 정규화 오차 계산.

    detect() 반환:
        (detected, error_norm, bbox, conf)
            detected: bool
            error_norm: (ex, ey) ∈ [-1, 1], 또는 None
            bbox: (x1, y1, x2, y2), 또는 None
            conf: float, 또는 None
    """

    def __init__(self, cfg: Config, logger=None):
        self.cfg = cfg
        self.logger = logger

        self.cam_idx = cfg.ibvs.camera_index
        self.cap = None
        self._last_reopen_t = 0.0
        self._reopen_period_sec = 1.0
        self._consecutive_read_failures = 0
        self._open_camera(initial=True)

        self.backend = _model_backend(cfg.yolo.model_path)
        requested_device = str(os.environ.get(
            "OMX_YOLO_DEVICE",
            getattr(cfg.yolo, "device", "0"),
        )).strip()
        if self.backend == 'pytorch':
            self.device, cpu_fallback_reason = _resolve_supported_device(requested_device)
            if cpu_fallback_reason:
                self._warn(f'OMX_YOLO_CUDA_FALLBACK_CPU | {cpu_fallback_reason}')
        else:
            self.device = requested_device
            self._log(
                f'OMX_YOLO_TENSORRT_BACKEND | model={cfg.yolo.model_path} '
                f'device={self.device}; skipping torch CUDA capability fallback'
            )
        self.use_half = bool(getattr(cfg.yolo, "half", True)) and self.backend == 'pytorch'
        if self.device.lower() in ("cpu", "none", ""):
            self.use_half = False

        self.model = YOLO(cfg.yolo.model_path, task="detect")
        self.target_class = cfg.yolo.target_class
        self.class_name = _safe_model_names(self.model).get(
            self.target_class, f"cls_{self.target_class}")
        self._log(f"YOLO 로드: {cfg.yolo.model_path}, "
                  f"클래스 {self.target_class} ({self.class_name}), "
                  f"backend={self.backend}, device={self.device}, half={self.use_half}")

    def _log(self, msg):
        if self.logger:
            self.logger.info(msg)
        else:
            print(msg)

    def _warn(self, msg):
        if self.logger:
            self.logger.warn(msg)
        else:
            print(msg)

    def _open_camera(self, *, initial: bool = False) -> bool:
        now = time.time()
        if not initial and now - self._last_reopen_t < self._reopen_period_sec:
            return False
        self._last_reopen_t = now

        if self.cap is not None:
            self.cap.release()

        self.cap = cv2.VideoCapture(self.cam_idx)
        if self.cap.isOpened():
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            self._consecutive_read_failures = 0
            self._log(f"카메라 {self.cam_idx} 열림")
            return True

        self._warn(f"카메라 {self.cam_idx} 열기 실패 - 재연결 대기")
        return False

    def read_frame(self):
        """카메라 1 프레임 읽기. 실패 시 None."""
        if self.cap is None or not self.cap.isOpened():
            self._open_camera()
            return None

        ok, frame = self.cap.read()
        if ok and frame is not None:
            self._consecutive_read_failures = 0
            return frame

        self._consecutive_read_failures += 1
        if self._consecutive_read_failures >= 5:
            self._open_camera()
        return None

    def detect(self, frame):
        """프레임에서 target_class 최고 conf 객체 검출.

        Returns:
            (True, (ex, ey), (x1,y1,x2,y2), conf) - 검출됨
            (False, None, None, None)             - 없음

        ex, ey: 영상 중심 (cx, cy) 기준 정규화 오차.
            ex > 0: 객체가 오른쪽
            ey > 0: 객체가 아래쪽
        """
        h, w = frame.shape[:2]
        cx, cy = w / 2.0, h / 2.0

        results = self.model.predict(
            frame, imgsz=self.cfg.yolo.imgsz,
            conf=self.cfg.yolo.conf_threshold,
            classes=[self.target_class],
            device=self.device,
            half=self.use_half,
            verbose=False)
        boxes = results[0].boxes

        if boxes is None or len(boxes) == 0:
            return False, None, None, None

        # 최고 confidence
        confs = boxes.conf.cpu().numpy()
        idx = confs.argmax()
        xyxy = boxes.xyxy[idx].cpu().numpy()
        x1, y1, x2, y2 = [int(v) for v in xyxy]
        conf = float(confs[idx])

        obj_x = (x1 + x2) / 2.0
        obj_y = (y1 + y2) / 2.0
        ex = (obj_x - cx) / cx
        ey = (obj_y - cy) / cy

        return True, (ex, ey), (x1, y1, x2, y2), conf

    def release(self):
        """카메라 자원 해제."""
        if self.cap:
            self.cap.release()
