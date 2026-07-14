from pathlib import Path
from types import SimpleNamespace

import pytest


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


def test_aim_reference_can_be_lowered_below_image_center():
    package_root = Path(__file__).parents[1]
    config_source = (package_root / 'config' / 'config.yaml').read_text(
        encoding='utf-8'
    )
    detector_source = (package_root / 'omx' / 'yolo_detector.py').read_text(
        encoding='utf-8'
    )
    node_source = (package_root / 'omx_aim' / 'yolo_node.py').read_text(
        encoding='utf-8'
    )

    assert 'aim_target_offset_y_norm: 0.20' in config_source
    assert 'aim_target_offset_x_norm: 0.08' in config_source
    assert 'def aim_reference_pixel' in detector_source
    assert "getattr(self.cfg.ibvs, 'aim_target_offset_y_norm'" in detector_source
    assert "getattr(self.cfg.ibvs, 'aim_target_offset_x_norm'" in detector_source
    assert 'self.detector.aim_reference_pixel(w, h)' in node_source


def test_aim_reference_offsets_shift_both_axes_and_stay_in_bounds():
    from omx.yolo_detector import YoloDetector

    detector = YoloDetector.__new__(YoloDetector)
    detector.cfg = SimpleNamespace(
        ibvs=SimpleNamespace(
            aim_target_offset_x_norm=0.08,
            aim_target_offset_y_norm=0.20,
        )
    )

    cx, cy = detector.aim_reference_pixel(1280, 720)

    assert cx == pytest.approx(640.0 + 0.08 * 640.0)
    assert cy == pytest.approx(360.0 + 0.20 * 360.0)

    # Extreme offsets must clamp inside the frame instead of going negative
    # or past the far edge.
    detector.cfg = SimpleNamespace(
        ibvs=SimpleNamespace(
            aim_target_offset_x_norm=-5.0,
            aim_target_offset_y_norm=5.0,
        )
    )
    cx, cy = detector.aim_reference_pixel(1280, 720)
    assert cx == 0.0
    assert cy == 719.0


def test_runtime_motor_fallback_is_visible_in_status_and_observation():
    node_source = (
        Path(__file__).parents[1] / 'omx_aim' / 'yolo_node.py'
    ).read_text(encoding='utf-8')

    assert 'self._control_video_only = bool(dry_run)' in node_source
    assert 'self._control_video_only = True' in node_source
    assert 'prefix = "video_only_"' in node_source
    assert "'control_video_only': bool" in node_source


def test_omx_aim_debug_topic_exposes_pd_tracking_contract():
    node_source = (
        Path(__file__).parents[1] / 'omx_aim' / 'yolo_node.py'
    ).read_text(encoding='utf-8')

    assert "'/omx/aim_debug'" in node_source
    assert 'def publish_aim_debug(' in node_source
    assert "'track_requested': action.get('action') == 'track'" in node_source
    assert "'track_moved': None if track_moved is None else bool(track_moved)" in node_source
    assert "'auto_armed': bool(action.get('auto_armed', False))" in node_source
    assert "track_moved = self._controller_call(" in node_source


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
    assert 'math.sin(2.0 * math.pi * phase)' in controller_source
    assert 'phase=continuous' in controller_source
    assert controller_source.count('self._scan_sweep_start_t = now') == 1


def test_direct_omx_scan_sweep_tool_exists_for_motor_diagnostics():
    tool_source = (
        Path(__file__).parents[3] / 'tools' / 'omx_scan_sweep_test.py'
    ).read_text(encoding='utf-8')

    assert 'OMX_SCAN_SWEEP_TEST' in tool_source
    assert '--half-angle-deg' in tool_source
    assert '--leave-torque-on' in tool_source
    assert 'ctrl.scan_sweep(' in tool_source


def test_jetson_launch_prioritizes_risk_patrol_without_localization_gate():
    launch_source = (
        Path(__file__).parents[1] / 'launch' / 'jetson.launch.py'
    ).read_text(encoding='utf-8')

    assert "'require_localization_ready': False" in launch_source
    assert "'require_start_motion': _is_true(require_start_motion)" in launch_source
    assert "'start_motion_topic': start_motion_topic" in launch_source
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


def test_target_tracking_repeatedly_cancels_leader_base_motion():
    node_source = (
        Path(__file__).parents[1] / 'omx_aim' / 'yolo_node.py'
    ).read_text(encoding='utf-8')
    start = node_source.index('    def maybe_stop_nav_on_detection')
    end = node_source.index('    def _make_point_stamped', start)
    function_source = node_source[start:end]

    assert 'State.TRACKING' in function_source
    assert 'State.CONFIRMING' in function_source
    assert '_detection_streak < 3' not in function_source
    assert 'if not self._waffle_nav_busy()' not in function_source
    assert 'if self._waffle_nav_busy():' in function_source
    assert 'self.publish_leader_nav_cancel()' in function_source
    assert 'target tracking -> leader cancel' in function_source
