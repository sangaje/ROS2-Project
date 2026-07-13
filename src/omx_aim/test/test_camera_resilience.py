from pathlib import Path


def test_omx_camera_uses_v4l2_preflight_and_reconnect_contract():
    source = (Path(__file__).parents[1] / 'omx' / 'yolo_detector.py').read_text(
        encoding='utf-8'
    )

    assert 'cv2.CAP_V4L2' in source
    assert 'OMX_CAMERA_PREFLIGHT' in source
    assert 'OMX_CAMERA_UNAVAILABLE' in source
    assert 'OMX_CAMERA_BUSY' in source
    assert 'self._reopen_period_sec' in source
    assert 'def _camera_source_candidates' in source
    assert 'active={source}' in source
    assert "('AUTO', cv2.CAP_ANY)" in source
    assert 'def _capture_loop(self)' in source
    assert 'cv2.CAP_PROP_FOURCC' in source
    assert 'self._latest_frame.copy()' in source


def test_scan_sweep_is_centered_on_forward_heading():
    source = (Path(__file__).parents[1] / 'omx' / 'controller.py').read_text(
        encoding='utf-8'
    )

    assert 'self._scan_sweep_center_yaw = 0.0' in source
    assert 'self._scan_sweep_center_yaw = self.yaw' not in source


def test_invalid_camera_frames_do_not_short_circuit_omx_navigation_loop():
    source = (Path(__file__).parents[1] / 'omx_aim' / 'yolo_node.py').read_text(
        encoding='utf-8'
    )

    assert 'vision_valid=vision_valid' in source
    assert 'self.maybe_retry_waiting_nav_goal(now)' in source
    assert 'self.publish_observation_status(' in source
    assert "'detected': detected if inference_ran else None" in source
    assert "payload['bbox_xyxy']" in source
    assert 'self.publish_vision_safe_fire_lock()' in source


def test_fire_disable_is_latched_and_not_republished_every_frame():
    package_root = Path(__file__).parents[1]
    node_source = (package_root / 'omx_aim' / 'yolo_node.py').read_text(
        encoding='utf-8'
    )
    fire_source = (package_root / 'omx_aim' / 'fire_node.py').read_text(
        encoding='utf-8'
    )

    assert 'TRANSIENT_LOCAL' in node_source
    assert 'self._last_fire_disable_pub' in node_source
    assert 'def _publish_fire_disable(' in node_source
    assert node_source.count('self.pub_fire_disable.publish') == 1
    assert 'TRANSIENT_LOCAL' in fire_source
    assert 'if previous == msg.data:' in fire_source


def test_omx_launch_supports_video_only_dry_run_during_motor_faults():
    package_root = Path(__file__).parents[1]
    launch_source = (package_root / 'launch' / 'jetson.launch.py').read_text(
        encoding='utf-8'
    )
    node_source = (package_root / 'omx_aim' / 'yolo_node.py').read_text(
        encoding='utf-8'
    )

    assert "LaunchConfiguration('omx_dry_run')" in launch_source
    assert "yolo_args.append('--dry-run')" in launch_source
    assert "'omx_dry_run', default_value='false'" in launch_source
    assert 'OMX_CONTROL_DISCONNECT_ERROR' in node_source
    assert 'OMX_CONTROL_RUNTIME_FALLBACK_VIDEO_ONLY' in node_source
    assert "self._controller_call('go_home', self.ctrl.go_home)" in node_source
    assert "'scan_sweep'," in node_source
    assert 'def execute_scan_sweep(self, now: float)' in node_source
    assert 'self.scan_sweep_center_yaw(now)' in node_source


def test_omx_scan_sweep_can_center_on_risk_map():
    package_root = Path(__file__).parents[1]
    node_source = (package_root / 'omx_aim' / 'yolo_node.py').read_text(
        encoding='utf-8'
    )
    config_source = (package_root / 'config' / 'config.yaml').read_text(
        encoding='utf-8'
    )
    controller_source = (package_root / 'omx' / 'controller.py').read_text(
        encoding='utf-8'
    )

    assert 'risk_map_topic: "/risk/risk_map"' in config_source
    assert 'self.on_risk_map' in node_source
    assert 'def _risk_map_scan_center_yaw' in node_source
    assert 'OMX_RISK_SCAN_CENTER' in node_source
    assert 'center_yaw_rad: float | None = None' in controller_source


def test_jetson_launch_prioritizes_risk_patrol_without_localization_gate():
    launch_source = (
        Path(__file__).parents[1] / 'launch' / 'jetson.launch.py'
    ).read_text(encoding='utf-8')

    assert "'require_localization_ready': False" in launch_source
    assert "'patrol_min_risk', default_value='8'" in launch_source
    assert "'patrol_publish_period_sec', default_value='3.0'" in launch_source
    assert "'patrol_view_standoff_distance_m', default_value='1.2'" in launch_source


def test_camera_read_exceptions_do_not_kill_omx_loop():
    node_source = (
        Path(__file__).parents[1] / 'omx_aim' / 'yolo_node.py'
    ).read_text(encoding='utf-8')

    assert 'OMX_CAMERA_READ_ERROR' in node_source
    assert "vision_reason = f'camera_read_failed:{type(exc).__name__}'" in node_source
    assert "getattr(self.detector, 'reset_camera', None)" in node_source
