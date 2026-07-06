import ast
from pathlib import Path


def test_launch_inventory_is_current():
    launch_dir = Path(__file__).parents[1] / 'launch'
    launch_files = sorted(path.name for path in launch_dir.glob('*.launch.py'))

    assert launch_files == [
        'flask_yolo_server.launch.py',
        'opencv_camera_to_flask_yolo.launch.py',
    ]


def test_launch_files_parse():
    launch_dir = Path(__file__).parents[1] / 'launch'
    for path in launch_dir.glob('*.launch.py'):
        ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
