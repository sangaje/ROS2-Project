from pathlib import Path


SERVER = (
    Path(__file__).parents[1]
    / 'flask_yolo_bridge'
    / 'flask_yolo_server.py'
)


def test_detect_defaults_to_async_latest_frame_pipeline():
    source = SERVER.read_text(encoding='utf-8')

    assert '--async-latest' in source
    assert 'default=True' in source
    assert 'submit_latest' in source
    assert "'async_ack': True" in source
    assert 'state.record_capture(' in source
    assert 'update_debug_overlay(frame' not in source.split('@app.post')[1]


def test_video_pipeline_reports_latency_and_drop_metrics():
    source = SERVER.read_text(encoding='utf-8')

    for key in (
        'capture_age_ms',
        'server_decode_ms',
        'queue_wait_ms',
        'tensorrt_predict_ms',
        'gpu_sync_ms',
        'postprocess_ms',
        'overlay_draw_ms',
        'server_jpeg_encode_ms',
        'response_ms',
        'end_to_end_frame_age_ms',
        'dropped_inference_frames',
        'capture_frames',
        'yolo_stream_clients',
    ):
        assert key in source


def test_tensorrt_runtime_rejects_cpu_fallback():
    source = SERVER.read_text(encoding='utf-8')

    assert "backend': 'tensorrt'" in source
    assert 'CUDA fallback is not allowed' in source
    assert "model_suffix in ('.engine', '.plan')" in source
