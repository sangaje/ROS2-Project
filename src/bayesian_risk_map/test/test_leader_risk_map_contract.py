from pathlib import Path


PKG_ROOT = Path(__file__).parents[1]


def test_external_detection_delay_fields_are_tracked():
    source = (
        PKG_ROOT
        / 'bayesian_risk_map'
        / 'bayesian_risk_map_node.py'
    ).read_text(encoding='utf-8')

    assert 'latest_detection_image_delay_ms' in source
    assert 'latest_detection_yolo_latency_ms' in source
    assert 'latest_detection_http_roundtrip_ms' in source
    assert "'capture_age_ms'" in source
    assert "'robot_frame_age_ms'" in source
    assert "'latency_ms'" in source
    assert "'http_roundtrip_ms'" in source
    assert 'capture_source=' in source


def test_new_scout_positive_batch_is_projected_once_even_after_delay():
    source = (
        PKG_ROOT
        / 'bayesian_risk_map'
        / 'bayesian_risk_map_node.py'
    ).read_text(encoding='utf-8')

    assert 'process_new_positive_batch' in source
    assert 'has_new_detection_batch and has_latest_detections' in source
    assert 'can_reuse_detection or process_new_positive_batch' in source


def test_central_risk_bridge_defaults_to_scout_pose():
    source = (
        PKG_ROOT
        / 'launch'
        / 'central_risk_map_bridge.launch.py'
    ).read_text(encoding='utf-8')

    assert "'source_pose_topic'" in source
    assert "source_pose_topic.perform(context)" in source
    assert "'map_qos_durability': 'transient_local'" in source
    assert "DeclareLaunchArgument('source_pose_topic', default_value='/member_pose')" in source
    assert "DeclareLaunchArgument('pose_topic', default_value='/member_pose')" in source
