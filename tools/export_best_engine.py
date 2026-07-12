#!/usr/bin/env python3
"""Export the project YOLO checkpoint to a TensorRT engine.

Run this on the Jetson, not on a plain PC:

  python3 tools/export_best_engine.py --pt model/target_v3.pt --engine model/target_v3.engine
"""

from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--pt', default='model/target_v3.pt')
    parser.add_argument('--engine', default='model/target_v3.engine')
    parser.add_argument('--imgsz', type=int, default=960)
    parser.add_argument('--device', default='0')
    parser.add_argument('--half', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--workspace', type=float, default=2.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    pt_path = Path(args.pt).expanduser().resolve()
    engine_path = Path(args.engine).expanduser().resolve()
    if not pt_path.is_file():
        raise FileNotFoundError(f'YOLO checkpoint not found: {pt_path}')

    from ultralytics import YOLO

    model = YOLO(str(pt_path))
    exported = model.export(
        format='engine',
        imgsz=int(args.imgsz),
        half=bool(args.half),
        device=str(args.device),
        workspace=float(args.workspace),
    )
    exported_path = Path(str(exported)).expanduser().resolve()
    if exported_path != engine_path:
        engine_path.parent.mkdir(parents=True, exist_ok=True)
        exported_path.replace(engine_path)
    print(f'EXPORTED_ENGINE={engine_path}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
