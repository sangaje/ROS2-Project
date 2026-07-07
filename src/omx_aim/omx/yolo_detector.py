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

        cam_idx = cfg.ibvs.camera_index
        self.cap = cv2.VideoCapture(cam_idx)
        if not self.cap.isOpened():
            raise RuntimeError(f"카메라 {cam_idx} 열기 실패")
        self._log(f"카메라 {cam_idx} 열림")

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
