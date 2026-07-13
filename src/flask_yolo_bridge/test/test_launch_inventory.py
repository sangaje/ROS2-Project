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


def test_camera_sender_has_role_based_rates_and_publish_gate():
    node = (
        Path(__file__).parents[1]
        / 'flask_yolo_bridge'
        / 'opencv_camera_to_flask_yolo.py'
    ).read_text(encoding='utf-8')
    launch = (
        Path(__file__).parents[1]
        / 'launch'
        / 'opencv_camera_to_flask_yolo.launch.py'
    ).read_text(encoding='utf-8')

    assert 'active_max_rate_hz' in node
    assert 'standby_max_rate_hz' in node
    assert 'active_max_upload_mbps' in node
    assert 'standby_max_upload_mbps' in node
    assert 'OPENCV_HTTP_YOLO_TX_BUDGET_DROP' in node
    assert 'tx_mbps=' in node
    assert 'publish_roles' in node
    assert 'def _current_upload_rate_hz' in node
    assert 'def _current_role_allows_publish' in node
    assert "'publish_roles': LaunchConfiguration('publish_roles')" in launch
    assert "'active_max_upload_mbps': LaunchConfiguration('active_max_upload_mbps')" in launch


def test_camera_sender_pose_history_is_bounded():
    node = (
        Path(__file__).parents[1]
        / 'flask_yolo_bridge'
        / 'opencv_camera_to_flask_yolo.py'
    ).read_text(encoding='utf-8')

    assert 'pose_history_max_samples' in node
    assert 'self.pose_history = deque(maxlen=self.pose_history_max_samples)' in node
