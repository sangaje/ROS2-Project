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
    assert "DurabilityPolicy.TRANSIENT_LOCAL" in source
    assert "raw_frames" in source
    assert "yolo_frames" in source
    assert "inference_frames" in source
    assert "'/omx/observation_status'" in source
    assert "'/omx/camera_ready'" in source
