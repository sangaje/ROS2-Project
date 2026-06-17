
#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image, LaserScan

from risk_common import compute_cqb_scalar_risk


class CQBRiskDatasetCollector(Node):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__("cqb_risk_dataset_collector")

        self.args = args
        self.bridge = CvBridge()

        self.out_dir = Path(args.out).expanduser().resolve()
        self.image_dir = self.out_dir / "images"
        self.image_dir.mkdir(parents=True, exist_ok=True)
        self.csv_path = self.out_dir / "labels.csv"

        self._ensure_csv_header()

        self.last_scan: Optional[LaserScan] = None
        self.last_scan_time_sec: Optional[float] = None
        self.saved_count = self._count_existing_images()
        self.image_count = 0
        self.last_save_wall_time = 0.0

        qos = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
        )

        self.scan_sub = self.create_subscription(
            LaserScan,
            args.scan_topic,
            self._on_scan,
            qos,
        )
        self.image_sub = self.create_subscription(
            Image,
            args.image_topic,
            self._on_image,
            qos,
        )

        self.get_logger().info(f"Saving dataset to: {self.out_dir}")
        self.get_logger().info(f"Image topic: {args.image_topic}")
        self.get_logger().info(f"Scan topic : {args.scan_topic}")

    def _ensure_csv_header(self) -> None:
        if self.csv_path.exists() and self.csv_path.stat().st_size > 0:
            return

        with self.csv_path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "filename",
                    "risk",
                    "front_min_dist",
                    "front_density",
                    "valid_front_count",
                    "image_stamp",
                    "scan_age_sec",
                    "image_topic",
                    "scan_topic",
                    "width",
                    "height",
                ]
            )

    def _count_existing_images(self) -> int:
        return len(list(self.image_dir.glob("*.png")))

    @staticmethod
    def _stamp_to_sec(msg) -> float:
        return float(msg.header.stamp.sec) + float(msg.header.stamp.nanosec) * 1e-9

    def _on_scan(self, msg: LaserScan) -> None:
        self.last_scan = msg
        self.last_scan_time_sec = self._stamp_to_sec(msg)

    def _should_save(self) -> bool:
        self.image_count += 1

        if self.image_count % self.args.save_every_n != 0:
            return False

        now = time.monotonic()
        if now - self.last_save_wall_time < self.args.min_save_interval:
            return False

        self.last_save_wall_time = now
        return True

    def _on_image(self, msg: Image) -> None:
        if self.last_scan is None or self.last_scan_time_sec is None:
            return

        if not self._should_save():
            return

        image_stamp = self._stamp_to_sec(msg)
        scan_age = abs(image_stamp - self.last_scan_time_sec)

        if scan_age > self.args.max_scan_age:
            self.get_logger().warn(
                f"Skip image: scan too old. scan_age={scan_age:.3f}s"
            )
            return

        try:
            image_bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as exc:
            self.get_logger().error(f"cv_bridge conversion failed: {exc}")
            return

        if self.args.resize_width > 0 and self.args.resize_height > 0:
            image_bgr = cv2.resize(
                image_bgr,
                (self.args.resize_width, self.args.resize_height),
                interpolation=cv2.INTER_AREA,
            )

        scan = self.last_scan
        ranges = np.asarray(scan.ranges, dtype=np.float32)

        stats = compute_cqb_scalar_risk(
            ranges=ranges,
            angle_min=float(scan.angle_min),
            angle_increment=float(scan.angle_increment),
            front_angle_deg=self.args.front_angle_deg,
            density_angle_deg=self.args.density_angle_deg,
            d_safe=self.args.d_safe,
            d_min=self.args.d_min,
        )

        filename = f"{self.saved_count:08d}.png"
        image_path = self.image_dir / filename

        ok = cv2.imwrite(str(image_path), image_bgr)
        if not ok:
            self.get_logger().error(f"Failed to save image: {image_path}")
            return

        h, w = image_bgr.shape[:2]

        with self.csv_path.open("a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    filename,
                    f"{stats.risk:.6f}",
                    f"{stats.front_min_dist:.6f}",
                    f"{stats.front_density:.6f}",
                    stats.valid_front_count,
                    f"{image_stamp:.9f}",
                    f"{scan_age:.6f}",
                    self.args.image_topic,
                    self.args.scan_topic,
                    w,
                    h,
                ]
            )

        self.saved_count += 1

        if self.saved_count % 50 == 0:
            self.get_logger().info(
                "saved=%d risk=%.3f d_front=%.3f density=%.3f"
                % (
                    self.saved_count,
                    stats.risk,
                    stats.front_min_dist,
                    stats.front_density,
                )
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect RGB images and LiDAR-derived scalar CQB risk labels."
    )
    parser.add_argument("--out", type=str, default="~/cqb_risk_dataset")
    parser.add_argument("--image-topic", type=str, default="/camera/image_raw")
    parser.add_argument("--scan-topic", type=str, default="/scan")
    parser.add_argument("--save-every-n", type=int, default=3)
    parser.add_argument("--min-save-interval", type=float, default=0.05)
    parser.add_argument("--max-scan-age", type=float, default=0.25)

    parser.add_argument("--resize-width", type=int, default=320)
    parser.add_argument("--resize-height", type=int, default=240)

    parser.add_argument("--front-angle-deg", type=float, default=35.0)
    parser.add_argument("--density-angle-deg", type=float, default=60.0)
    parser.add_argument("--d-safe", type=float, default=1.20)
    parser.add_argument("--d-min", type=float, default=0.15)

    args, _ = parser.parse_known_args()
    return args


def main() -> None:
    args = parse_args()
    rclpy.init()
    node = CQBRiskDatasetCollector(args)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
