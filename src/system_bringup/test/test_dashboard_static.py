from pathlib import Path


def test_dashboard_draws_risk_grid_with_its_own_metadata():
    source = (
        Path(__file__).parents[1] / 'static' / 'dashboard.js'
    ).read_text(encoding='utf-8')

    assert 'function cellToWorld(meta, cellX, cellY)' in source
    assert 'function drawGridImage(img, overlayMeta, baseMeta, vp)' in source
    assert 'drawGridImage(riskImg, latest.risk.metadata, meta, vp)' in source
    assert 'ctx.drawImage(riskImg, vp.x, vp.y, vp.w, vp.h)' not in source


def test_dashboard_publishes_latched_video_ready_after_all_streams_arrive():
    source = (
        Path(__file__).parents[1] / 'system_bringup' / 'leader_unified_dashboard.py'
    ).read_text(encoding='utf-8')

    assert "video_ready_topic" in source
    assert "self.video_ready_pub" in source
    assert "start_motion_topic" in source
    assert "self.start_motion_pub" in source
    assert "readiness_detail_topic" in source
    assert "self.readiness_detail_pub" in source
    assert "start_motion_detail_topic" in source
    assert "self.start_motion_detail_pub" in source
    assert "blocking_reasons" in source
    assert "DASHBOARD_READINESS_DETAIL |" in source
    assert "blocking_panels=" in source
    assert "system_readiness_detail_topic" in source
    assert "self._publish_start_motion(False" in source
    assert "DurabilityPolicy.TRANSIENT_LOCAL" in source
    assert "yolo_frames" in source
    assert "inference_frames" in source
    assert "scout_stream_ready" in source
    assert "omx_stream_ready" in source
    assert "video_ready_max_age_sec" in source


def test_dashboard_requires_browser_rendered_panel_manifest():
    source = (
        Path(__file__).parents[1] / 'system_bringup' / 'leader_unified_dashboard.py'
    ).read_text(encoding='utf-8')
    js = (
        Path(__file__).parents[1] / 'static' / 'dashboard.js'
    ).read_text(encoding='utf-8')

    assert "/api/dashboard_readiness" in source
    assert "dashboard_ui_ready_topic" not in source
    assert "dashboard_readiness_detail_topic" not in source
    assert "readiness_detail_topic" in source
    assert "_dashboard_ui_panels_ready" in source
    assert "rendered" in source
    assert "backend_ready and ui_ready" in source
    assert "ready and system_ready" in source
    assert "publishDashboardReadiness" in js
    assert "rendered:" in js
    assert "naturalWidth > 0" in js
    assert "risk_map" in js
    removed_terms = [
        'scout' + '_raw',
        'scout' + 'RawStream',
        'raw' + '_frame_age_sec',
    ]
    for term in removed_terms:
        assert term not in js
        assert term not in source
    assert "yolo_frame_age_sec" in source
    assert "inference_frame_age_sec" in source
    assert "observation_status_received_wall_sec" in source
    assert "'/omx/observation_status'" in source
    assert "'/omx/camera_ready'" in source


def test_dashboard_subscribes_to_omx_target_detected_with_best_effort_qos():
    source = (
        Path(__file__).parents[1] / 'system_bringup' / 'leader_unified_dashboard.py'
    ).read_text(encoding='utf-8')

    assert "self.create_subscription(Bool, '/omx/target_detected'" in source
    assert "'std_msgs/msg/Bool'), latest_best_effort_qos)" in source


def test_dashboard_has_only_two_default_video_streams():
    source = (
        Path(__file__).parents[1] / 'system_bringup' / 'leader_unified_dashboard.py'
    ).read_text(encoding='utf-8')
    js = (
        Path(__file__).parents[1] / 'static' / 'dashboard.js'
    ).read_text(encoding='utf-8')
    html = (
        Path(__file__).parents[1] / 'templates' / 'dashboard.html'
    ).read_text(encoding='utf-8')

    assert 'CachedMjpegStream' in source
    assert "'scout_yolo': CachedMjpegStream" in source
    assert "'omx': CachedMjpegStream" in source
    assert "streamSources.scoutYoloStream = '/api/yolo_stream/yolo.mjpg'" in js
    assert "streamSources.omxStream = '/api/omx_stream.mjpg'" in js
    assert 'setInterval(refresh, 500)' in js
    removed_terms = [
        'scout' + 'RawStream',
        'Scout ' + 'raw',
        'raw ' + 'debug stream disabled',
    ]
    for term in removed_terms:
        assert term not in js
        assert term not in html
        assert term not in source
