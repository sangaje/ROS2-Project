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
    assert "STARTUP_COORDINATOR |" in source
    assert "STARTUP_NOT_RELEASED" in source
    assert "leader_unified_dashboard_startup_coordinator" in source
    assert "publisher_count" in source
    assert "DurabilityPolicy.TRANSIENT_LOCAL" in source
    assert "yolo_frames" in source
    assert "inference_frames" in source
    assert "scout_stream_ready" in source
    assert "omx_stream_ready" in source
    assert "video_ready_max_age_sec" in source


def test_system_launch_exposes_omx_and_scout_video_ready_requirements():
    # Regression test: leader_unified_dashboard.py has always defaulted
    # require_omx_video_ready/require_scout_video_ready to True internally,
    # but system.launch.py never exposed them as launch arguments -- a
    # leader run without OMX/arm hardware physically attached had no way to
    # ever satisfy dashboard_backend_ready, and start_motion (therefore RL
    # motion) would block forever with no way to test around it.
    text = (
        Path(__file__).parents[1] / 'launch' / 'system.launch.py'
    ).read_text(encoding='utf-8')

    assert "DeclareLaunchArgument(\n            'require_scout_video_ready'," in text
    assert "DeclareLaunchArgument(\n            'require_omx_video_ready'," in text
    assert "require_scout_video_ready = LaunchConfiguration('require_scout_video_ready')" in text
    assert "require_omx_video_ready = LaunchConfiguration('require_omx_video_ready')" in text
    assert (
        "'require_scout_video_ready': launch_bool(\n"
        "                            require_scout_video_ready.perform(context)\n"
        "                        ),"
    ) in text
    assert (
        "'require_omx_video_ready': launch_bool(\n"
        "                            require_omx_video_ready.perform(context)\n"
        "                        ),"
    ) in text


def test_start_motion_is_one_shot_scout_runtime_release():
    # Dashboard/browser/video readiness is diagnostic only. Startup motion is
    # released once from the Scout runtime's minimum safety detail and is not
    # pulled low by later dashboard/video misses.
    source = (
        Path(__file__).parents[1] / 'system_bringup' / 'leader_unified_dashboard.py'
    ).read_text(encoding='utf-8')

    assert 'scout_motion_ready_detail_topic' in source
    assert 'def _evaluate_motion_release' in source
    assert 'previous_start_motion or release_ready' in source
    assert 'MOTION_RELEASE_DEBUG |' in source
    assert 'MOTION_RELEASED |' in source
    assert 'reason=minimum_scout_runtime_ready' in source
    assert 'raw_motion_ok = bool(ready and system_ready)' not in source


def test_video_streams_self_heal_without_a_manual_page_refresh():
    # Regression test: map/risk grids already retried on a failed image
    # request, but the two MJPEG <img> streams (omxStream, scoutYoloStream)
    # had no equivalent -- a stream that went silently idle (server stops
    # sending frames without closing the connection) never fires 'error',
    # so it just froze on the last frame until the user manually reloaded
    # the page. A watchdog must now force a fresh connection itself.
    js = (
        Path(__file__).parents[1] / 'static' / 'dashboard.js'
    ).read_text(encoding='utf-8')

    assert 'lastFrameAtMs' in js
    assert 'streamStaleTimeoutMs' in js
    assert 'function checkStreamFreshness()' in js
    assert 'setInterval(checkStreamFreshness, 2000)' in js
    assert 'now - last > streamStaleTimeoutMs' in js


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
    assert "dashboard_ready=ready" in source
    assert "previous_start_motion or release_ready" in source
    assert "publishDashboardReadiness" in js
    assert "rendered:" in js
    assert "naturalWidth > 0" in js
    assert "risk_map" in js
    assert "backend_seq" in js
    assert "png_seq" in js
    assert "grid_received" in js
    assert "png_bytes" in js
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


def test_dashboard_risk_map_rendering_requires_real_grid_and_png_state():
    source = (
        Path(__file__).parents[1] / 'system_bringup' / 'leader_unified_dashboard.py'
    ).read_text(encoding='utf-8')
    js = (
        Path(__file__).parents[1] / 'static' / 'dashboard.js'
    ).read_text(encoding='utf-8')

    assert 'DASHBOARD_RISK_SUBSCRIBER |' in source
    assert 'DASHBOARD_RISK_RENDER_BACKEND' in source
    assert 'NO_TOPIC' in source
    assert 'WAITING_FIRST_GRID' in source
    assert 'EMPTY_RISK_MAP' in source
    assert 'ACTIVE_RISK_MAP' in source
    assert 'STALE_RISK_MAP' in source
    assert 'RISK_MAP_ALIGNMENT |' in source
    assert "status in ('EMPTY_RISK_MAP', 'ACTIVE_RISK_MAP')" in source
    assert 'risk_map_has_positive_evidence=' in source
    assert 'risk_map_status=' in source
    assert 'No active risk evidence' in js
    assert 'Risk max:' in js
    assert 'Positive cells:' in js
    assert 'seq === backendSeq && pngSeq === backendSeq' in js


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
