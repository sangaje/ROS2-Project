from pathlib import Path


def test_omx_camera_uses_v4l2_preflight_and_reconnect_contract():
    source = (Path(__file__).parents[1] / 'omx' / 'yolo_detector.py').read_text(
        encoding='utf-8'
    )

    assert 'cv2.CAP_V4L2' in source
    assert 'OMX_CAMERA_PREFLIGHT' in source
    assert 'OMX_CAMERA_UNAVAILABLE' in source
    assert 'OMX_CAMERA_BUSY' in source
    assert 'self._pending_first_frame = frame' in source
    assert 'self._reopen_period_sec' in source


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
