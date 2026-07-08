"""YOLO 기반 표적 검출.

OpenCV VideoCapture + Ultralytics YOLO. ROS 의존성 없음.
"""

from __future__ import annotations

import os

import cv2
from ultralytics import YOLO

from omx.config import Config


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

        self.camera_source = self._camera_source(cfg)
        self.fallback_sources = self._camera_fallback_sources()
        self.cap = None
        self.active_camera = None
        self.open_camera()

        self.device = str(os.environ.get(
            "OMX_YOLO_DEVICE",
            getattr(cfg.yolo, "device", "0"),
        )).strip()
        self.use_half = bool(getattr(cfg.yolo, "half", True))
        if self.device.lower() in ("cpu", "none", ""):
            self.use_half = False

        self.model = YOLO(cfg.yolo.model_path)
        self.target_class = cfg.yolo.target_class
        self.class_name = self.model.names.get(
            self.target_class, f"cls_{self.target_class}")
        self._log(f"YOLO 로드: {cfg.yolo.model_path}, "
                  f"클래스 {self.target_class} ({self.class_name}), "
                  f"device={self.device}, half={self.use_half}")

    def _camera_source(self, cfg: Config):
        return str(os.environ.get(
            "OMX_CAMERA_INDEX",
            getattr(cfg.ibvs, "camera_index", 0),
        )).strip()

    def _camera_fallback_sources(self) -> list[str]:
        raw = os.environ.get(
            "OMX_CAMERA_FALLBACK_DEVICES",
            "auto",
        )
        return [item.strip() for item in str(raw).split(",") if item.strip()]

    def _camera_candidates(self) -> list[str]:
        candidates = []

        def add(value) -> None:
            text = str(value).strip()
            if not text:
                return
            if text.lower() == "auto":
                for index in range(8):
                    add(f"/dev/video{index}")
                    add(str(index))
                return
            if text not in candidates:
                candidates.append(text)

        add(self.camera_source)
        for source in self.fallback_sources:
            add(source)
        return candidates

    @staticmethod
    def _open_arg(source: str):
        return int(source) if str(source).isdigit() else source

    def open_camera(self):
        self.release()
        tried = []
        for source in self._camera_candidates():
            tried.append(source)
            cap = cv2.VideoCapture(self._open_arg(source), cv2.CAP_V4L2)
            if not cap.isOpened():
                cap.release()
                continue
            ok, frame = cap.read()
            if not ok or frame is None:
                cap.release()
                continue
            self.cap = cap
            self.active_camera = source
            self._log(
                f"카메라 열림: active={source} requested={self.camera_source} "
                f"fallback={','.join(self.fallback_sources)}"
            )
            return
        raise RuntimeError(
            f"카메라 열기 실패: requested={self.camera_source} "
            f"tried={tried}"
        )

    def _log(self, msg):
        if self.logger:
            self.logger.info(msg)
        else:
            print(msg)

    def read_frame(self):
        """카메라 1 프레임 읽기. 실패 시 None."""
        ok, frame = self.cap.read()
        return frame if ok else None

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
