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
    assert 'def prepare_frame_for_upload' in node
    assert 'letterbox_full_frame' in node
    assert 'cv2.copyMakeBorder' in node
    assert 'camera_resize_ms_p50=' in node
    assert 'CAMERA_NETWORK_STATUS |' in node
    assert 'publish_roles' in node
    assert 'standby_roles' in node
    assert 'active_scout_id_topic' in node
    assert 'def on_active_scout_id' in node
    assert 'def _current_upload_rate_hz' in node
    assert 'def _current_role_allows_publish' in node
    assert 'camera_process_enabled' in node
    assert 'camera_upload_enabled' in node
    assert 'risk_observation_publish_enabled' in node
    assert "DeclareLaunchArgument('width', default_value='640')" in launch
    assert "DeclareLaunchArgument('height', default_value='480')" in launch
    assert "DeclareLaunchArgument('send_width', default_value='640')" in launch
    assert "DeclareLaunchArgument('send_height', default_value='480')" in launch
    assert "DeclareLaunchArgument('active_max_rate_hz', default_value='5.0')" in launch
    assert "DeclareLaunchArgument('standby_max_rate_hz', default_value='1.0')" in launch
    assert "DeclareLaunchArgument('jpeg_quality', default_value='65')" in launch
    assert "DeclareLaunchArgument('letterbox_color', default_value='0')" in launch
    assert "DeclareLaunchArgument('active_scout_id_topic', default_value='/failover/active_scout_id')" in launch
    assert "'standby_roles': LaunchConfiguration('standby_roles')" in launch
    assert "'publish_roles': LaunchConfiguration('publish_roles')" in launch
    assert "'active_scout_id_topic': LaunchConfiguration('active_scout_id_topic')" in launch
    assert "'active_max_upload_mbps': LaunchConfiguration('active_max_upload_mbps')" in launch


def test_camera_sender_pose_history_is_bounded():
    node = (
        Path(__file__).parents[1]
        / 'flask_yolo_bridge'
        / 'opencv_camera_to_flask_yolo.py'
    ).read_text(encoding='utf-8')

    assert 'pose_history_max_samples' in node
    assert 'self.pose_history = deque(maxlen=self.pose_history_max_samples)' in node
