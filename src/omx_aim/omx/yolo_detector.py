"""YOLO 기반 표적 검출.

OpenCV VideoCapture + Ultralytics YOLO TensorRT engine. ROS 의존성 없음.
"""

from __future__ import annotations

import os
import subprocess
import time

import cv2

from omx.config import Config


class YoloRuntimeDependencyError(RuntimeError):
    """ultralytics/numpy/torch import failed -- not a missing model file."""


def _safe_model_names(model) -> dict:
    try:
        names = getattr(model, "names", {}) or {}
    except Exception:
        names = {}
    return names if isinstance(names, dict) else {}


def _import_yolo():
    """Import ultralytics.YOLO lazily and report *why* it failed.

    A bare `from ultralytics import YOLO` at module level meant one broken
    dependency (e.g. a NumPy 1.x binary extension shadowing NumPy 2.x on
    Jetson) crashed on import of this module, before the ROS node even
    started -- so the failure looked like "the whole node is gone" instead
    of a diagnosable dependency problem.
    """
    try:
        from ultralytics import YOLO
        return YOLO
    except Exception as exc:  # noqa: BLE001
        import sys
        numpy_version = numpy_path = matplotlib_path = 'unavailable'
        try:
            import numpy
            numpy_version = numpy.__version__
            numpy_path = numpy.__file__
        except Exception:  # noqa: BLE001
            pass
        try:
            import matplotlib
            matplotlib_path = matplotlib.__file__
        except Exception:  # noqa: BLE001
            pass
        raise YoloRuntimeDependencyError(
            'YOLO_DEPENDENCY_ERROR | '
            f'python={sys.executable} '
            f'numpy_version={numpy_version} '
            f'numpy_path={numpy_path} '
            f'matplotlib_path={matplotlib_path} '
            f'ultralytics_path=unavailable '
            f'error={type(exc).__name__}: {exc}'
        ) from exc


def _validate_runtime_model_path(model_path: str) -> None:
    suffix = os.path.splitext(str(model_path))[1].lower()
    if suffix == ".pt":
        raise ValueError(
            f"PyTorch YOLO checkpoints are not allowed at runtime: {model_path}. "
            "Export and launch with model/target_v3.engine instead."
        )
    if suffix not in (".engine", ".plan"):
        raise ValueError(
            f"YOLO runtime model must be a TensorRT .engine/.plan file, got: {model_path}"
        )


def _resolve_supported_device(requested: str, model_path: str) -> tuple[str, str | None]:
    """Prevent an unsupported Jetson CUDA binary from killing OMX video."""
    device = str(requested).strip()
    if device.isdigit():
        device = f'cuda:{device}'
    suffix = os.path.splitext(str(model_path))[1].lower()
    if suffix in (".engine", ".plan"):
        if device.lower() in ("cpu", "none", ""):
            raise ValueError("TensorRT YOLO engines require a CUDA device, not cpu.")
        return device, None
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

        configured_device = str(getattr(cfg.ibvs, 'camera_device', '')).strip()
        self.camera_source = configured_device or cfg.ibvs.camera_index
        self.camera_backend = str(
            getattr(cfg.ibvs, 'camera_backend', 'v4l2')
        ).strip().lower()
        self.cap = None
        self._last_reopen_t = 0.0
        self._reconnect_attempt = 0
        self._reopen_period_sec = max(
            0.1, float(getattr(cfg.ibvs, 'camera_reconnect_period_sec', 1.0))
        )
        self._consecutive_read_failures = 0
        self.camera_ready = False
        self.camera_failure_reason = 'startup'
        self.frame_width = 0
        self.frame_height = 0
        self._pending_first_frame = None
        self._active_camera_source = None
        self._active_camera_backend = None
        self._open_camera(initial=True)

        requested_device = str(os.environ.get(
            "OMX_YOLO_DEVICE",
            getattr(cfg.yolo, "device", "0"),
        )).strip()
        _validate_runtime_model_path(cfg.yolo.model_path)
        self.device, cpu_fallback_reason = _resolve_supported_device(
            requested_device,
            cfg.yolo.model_path,
        )
        if cpu_fallback_reason:
            self._warn(f'OMX_YOLO_CUDA_FALLBACK_CPU | {cpu_fallback_reason}')
        self.use_half = bool(getattr(cfg.yolo, "half", True))
        if self.device.lower() in ("cpu", "none", ""):
            self.use_half = False

        YOLO = _import_yolo()
        self.model = YOLO(cfg.yolo.model_path, task="detect")
        self.target_class = cfg.yolo.target_class
        self.class_name = _safe_model_names(self.model).get(
            self.target_class, f"cls_{self.target_class}")
        self._log(f"YOLO 로드: {cfg.yolo.model_path}, "
                  f"클래스 {self.target_class} ({self.class_name}), "
                  f"device={self.device}, half={self.use_half}")

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

    def _camera_label(self) -> str:
        return str(self.camera_source)

    def _camera_source_candidates(self):
        sources = [self.camera_source]
        source_text = str(self.camera_source)
        if source_text.startswith('/dev/video'):
            suffix = source_text[len('/dev/video'):]
            if suffix.isdigit():
                index = int(suffix)
                if index not in sources:
                    sources.append(index)
        elif isinstance(self.camera_source, str) and source_text.isdigit():
            index = int(source_text)
            if index not in sources:
                sources.append(index)
        return sources

    def _camera_backend_candidates(self):
        if self.camera_backend == 'v4l2':
            return [('V4L2', cv2.CAP_V4L2), ('AUTO', cv2.CAP_ANY)]
        return [('AUTO', cv2.CAP_ANY)]

    def _device_preflight(self) -> tuple[bool, str]:
        if not isinstance(self.camera_source, str) or not self.camera_source.startswith('/'):
            return True, 'index'
        if not os.path.exists(self.camera_source):
            return False, 'device_missing'
        if not os.access(self.camera_source, os.R_OK):
            return False, 'device_not_readable'
        return True, 'ready'

    def _busy_owner(self) -> str:
        if not isinstance(self.camera_source, str) or not self.camera_source.startswith('/'):
            return ''
        try:
            result = subprocess.run(
                ['fuser', '-v', self.camera_source],
                capture_output=True, text=True, timeout=1.0, check=False,
            )
        except (OSError, subprocess.SubprocessError):
            return ''
        return ' '.join((result.stdout + ' ' + result.stderr).split())

    def _set_camera_health(self, ready: bool, reason: str) -> bool:
        changed = ready != self.camera_ready or reason != self.camera_failure_reason
        self.camera_ready = ready
        self.camera_failure_reason = reason
        if not changed:
            return False
        if ready:
            self._log(
                'OMX_CAMERA | RESTORED | '
                f'device={self._camera_label()} width={self.frame_width} '
                f'height={self.frame_height}'
            )
        else:
            self._warn(
                f'OMX_CAMERA | LOST | device={self._camera_label()} reason={reason}'
            )
        return True

    def _open_camera(self, *, initial: bool = False) -> bool:
        now = time.time()
        if not initial and now - self._last_reopen_t < self._reopen_period_sec:
            return False
        self._last_reopen_t = now
        if not initial:
            self._reconnect_attempt += 1
            self._log(
                'OMX_CAMERA | RECONNECTING | '
                f'attempt={self._reconnect_attempt} device={self._camera_label()}'
            )

        if self.cap is not None:
            self.cap.release()

        preflight_ok, preflight_reason = self._device_preflight()
        if not preflight_ok:
            changed = self._set_camera_health(False, preflight_reason)
            exists = os.path.exists(self.camera_source)
            if changed:
                self._warn(
                    'OMX_CAMERA_UNAVAILABLE | '
                    f'requested={self._camera_label()} exists={str(exists).lower()} '
                    f'reason={preflight_reason}'
                )
            return False

        for source in self._camera_source_candidates():
            for backend_name, backend in self._camera_backend_candidates():
                cap = cv2.VideoCapture(source, backend)
                if cap.isOpened():
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    ok, frame = cap.read()
                    if ok and frame is not None:
                        self.cap = cap
                        self._active_camera_source = source
                        self._active_camera_backend = backend_name
                        self._consecutive_read_failures = 0
                        self.frame_height, self.frame_width = frame.shape[:2]
                        self._pending_first_frame = frame
                        self._set_camera_health(True, 'ready')
                        self._reconnect_attempt = 0
                        self._log(
                            'OMX_CAMERA_PREFLIGHT | '
                            f'requested={self._camera_label()} active={source} '
                            f'exists=true readable=true backend={backend_name} '
                            f'opened=true first_frame=true '
                            f'width={self.frame_width} height={self.frame_height}'
                        )
                        return True
                cap.release()
        self.cap = None
        self._active_camera_source = None
        self._active_camera_backend = None
        owner = self._busy_owner()
        reason = 'camera_busy' if owner else 'open_failed'
        changed = self._set_camera_health(False, reason)
        if changed:
            if owner:
                self._warn(f'OMX_CAMERA_BUSY | device={self._camera_label()} owner={owner}')
            else:
                self._warn(
                    'OMX_CAMERA_UNAVAILABLE | '
                    f'requested={self._camera_label()} exists=true reason={reason}'
                )
        return False

    def read_frame(self):
        """카메라 1 프레임 읽기. 실패 시 None."""
        if self.cap is None or not self.cap.isOpened():
            self._open_camera()
            return None

        pending = getattr(self, '_pending_first_frame', None)
        if pending is not None:
            self._pending_first_frame = None
            return pending

        ok, frame = self.cap.read()
        if ok and frame is not None:
            self._consecutive_read_failures = 0
            self.frame_height, self.frame_width = frame.shape[:2]
            self._set_camera_health(True, 'ready')
            return frame

        self._consecutive_read_failures += 1
        if self._consecutive_read_failures >= 5:
            self._set_camera_health(False, 'read_failed')
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
