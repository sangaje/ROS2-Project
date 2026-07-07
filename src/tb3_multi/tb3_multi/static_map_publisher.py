#!/usr/bin/env python3
"""Publish a static OccupancyGrid map without using nav2_map_server lifecycle."""

from pathlib import Path
from typing import Dict, Tuple

from nav_msgs.msg import OccupancyGrid
import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy


def _parse_simple_yaml(path: Path) -> Dict[str, object]:
    """Small fallback parser for the simple ROS map yaml format."""
    try:
        import yaml  # type: ignore
        with path.open('r', encoding='utf-8') as f:
            data = yaml.safe_load(f) or {}
        return data
    except Exception:
        data: Dict[str, object] = {}
        for raw_line in path.read_text(encoding='utf-8').splitlines():
            line = raw_line.split('#', 1)[0].strip()
            if not line or ':' not in line:
                continue
            key, value = line.split(':', 1)
            key = key.strip()
            value = value.strip()
            if value.startswith('[') and value.endswith(']'):
                items = [x.strip() for x in value[1:-1].split(',') if x.strip()]
                data[key] = [float(x) for x in items]
            else:
                try:
                    data[key] = float(value)
                except ValueError:
                    data[key] = value.strip('"\'')
        return data


def _read_pgm(path: Path) -> Tuple[int, int, bytes]:
    """Read binary P5 PGM files used by ROS maps."""
    raw = path.read_bytes()
    pos = 0

    def next_token() -> bytes:
        nonlocal pos
        while pos < len(raw):
            c = raw[pos]
            if c == ord('#'):
                while pos < len(raw) and raw[pos] not in b'\r\n':
                    pos += 1
            elif chr(c).isspace():
                pos += 1
            else:
                break
        start = pos
        while pos < len(raw) and not chr(raw[pos]).isspace():
            pos += 1
        return raw[start:pos]

    magic = next_token()
    if magic != b'P5':
        raise ValueError(f'{path} is not a binary P5 PGM file')
    width = int(next_token())
    height = int(next_token())
    maxval = int(next_token())
    if maxval <= 0 or maxval > 255:
        raise ValueError(f'Unsupported PGM max value: {maxval}')
    while pos < len(raw) and chr(raw[pos]).isspace():
        pos += 1
    pixels = raw[pos:pos + width * height]
    if len(pixels) != width * height:
        raise ValueError(f'PGM size mismatch: expected {width * height}, got {len(pixels)}')
    return width, height, pixels


class StaticMapPublisher(Node):
    """Load a ROS map yaml/PGM pair and publish it as a latched map."""

    def __init__(self) -> None:
        super().__init__('static_map_publisher')
        self.declare_parameter('map_yaml', '')
        self.declare_parameter('frame_id', 'map')
        self.declare_parameter('map_topic', '/map')
        map_yaml = str(self.get_parameter('map_yaml').value)
        self.frame_id = str(self.get_parameter('frame_id').value)
        map_topic = str(self.get_parameter('map_topic').value)
        if not map_yaml:
            raise RuntimeError('map_yaml parameter is required')

        self.map_msg = self._load_map(Path(map_yaml).expanduser())
        qos = QoSProfile(depth=1)
        qos.durability = DurabilityPolicy.TRANSIENT_LOCAL
        qos.reliability = ReliabilityPolicy.RELIABLE
        self.pub = self.create_publisher(OccupancyGrid, map_topic, qos)
        # TRANSIENT_LOCAL makes this latched for late RViz subscribers.
        # Re-publishing the whole map every second can make RViz look like it is
        # blinking/reloading, so publish once at startup.
        self.publish_map()
        self.get_logger().info(
            f'Publishing static map {map_yaml} on {map_topic} '
            f'({self.map_msg.info.width}x{self.map_msg.info.height}, frame={self.frame_id})'
        )

    def _load_map(self, yaml_path: Path) -> OccupancyGrid:
        if not yaml_path.exists():
            raise FileNotFoundError(yaml_path)
        cfg = _parse_simple_yaml(yaml_path)
        image_name = str(cfg.get('image', '')).strip()
        if not image_name:
            raise ValueError(f'Map yaml {yaml_path} has no image field')
        image_path = Path(image_name)
        if not image_path.is_absolute():
            image_path = yaml_path.parent / image_path

        width, height, pixels = _read_pgm(image_path)
        resolution = float(cfg.get('resolution', 0.05))
        origin = cfg.get('origin', [0.0, 0.0, 0.0])
        if not isinstance(origin, list) or len(origin) < 3:
            origin = [0.0, 0.0, 0.0]
        negate = int(cfg.get('negate', 0))
        occupied_thresh = float(cfg.get('occupied_thresh', 0.65))
        free_thresh = float(cfg.get('free_thresh', 0.196))

        msg = OccupancyGrid()
        msg.header.frame_id = self.frame_id
        msg.info.resolution = resolution
        msg.info.width = width
        msg.info.height = height
        msg.info.origin.position.x = float(origin[0])
        msg.info.origin.position.y = float(origin[1])
        msg.info.origin.position.z = 0.0
        msg.info.origin.orientation.w = 1.0

        data = []
        for y in range(height):
            # OccupancyGrid starts at bottom-left. PGM stores from top-left.
            src_y = height - 1 - y
            for x in range(width):
                gray = pixels[src_y * width + x]
                occ = (gray / 255.0) if negate else ((255 - gray) / 255.0)
                if occ > occupied_thresh:
                    data.append(100)
                elif occ < free_thresh:
                    data.append(0)
                else:
                    data.append(-1)
        msg.data = data
        return msg

    def publish_map(self) -> None:
        self.map_msg.header.stamp = self.get_clock().now().to_msg()
        self.pub.publish(self.map_msg)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = StaticMapPublisher()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
