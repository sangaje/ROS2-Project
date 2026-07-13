from pathlib import Path


def test_external_detection_does_not_fallback_to_current_pose():
    source = (
        Path(__file__).parents[1]
        / 'bayesian_risk_map'
        / 'bayesian_risk_map_node.py'
    ).read_text(encoding='utf-8')
    start = source.index('    def on_external_detections')
    end = source.index('    def parse_payload_capture_pose', start)
    handler = source[start:end]

    assert 'capture_pose = self.parse_payload_capture_pose(payload)' in handler
    assert 'OBSERVATION_MISSING_POSE_DROPPED' in handler
    assert 'self.latest_detection_pose = capture_pose' in handler
    assert 'self.latest_detection_pose = self.get_robot_pose()' not in handler


def test_capture_pose_parser_supports_flat_and_nested_payloads():
    source = (
        Path(__file__).parents[1]
        / 'bayesian_risk_map'
        / 'bayesian_risk_map_node.py'
    ).read_text(encoding='utf-8')
    start = source.index('    def parse_payload_capture_pose')
    end = source.index('    def maybe_make_fake_detection', start)
    parser = source[start:end]

    assert "pose = payload.get('capture_pose')" in parser
    assert "float(pose['x'])" in parser
    assert "float(payload['capture_pose_x'])" in parser
    assert "float(payload['capture_pose_y'])" in parser
    assert "float(payload.get('capture_pose_yaw', 0.0))" in parser


def test_external_observations_are_fenced_to_active_source_and_epoch():
    source = (
        Path(__file__).parents[1]
        / 'bayesian_risk_map'
        / 'bayesian_risk_map_node.py'
    ).read_text(encoding='utf-8')
    handler_start = source.index('    def on_external_detections')
    handler_end = source.index('    def parse_payload_capture_pose', handler_start)
    handler = source[handler_start:handler_end]

    assert 'require_active_observation_source' in source
    assert 'active_scout_id_topic' in source
    assert 'scout_epoch_topic' in source
    assert 'def observation_source_allowed' in source
    assert 'OBSERVATION_INACTIVE_ROLE_DROPPED' in source
    assert 'OBSERVATION_INACTIVE_SOURCE_DROPPED' in source
    assert 'OBSERVATION_STALE_EPOCH_DROPPED' in source
    assert 'if not self.observation_source_allowed(payload, robot_id):' in handler


def test_external_detection_pipeline_reports_filter_stage_counts():
    source = (
        Path(__file__).parents[1]
        / 'bayesian_risk_map'
        / 'bayesian_risk_map_node.py'
    ).read_text(encoding='utf-8')
    handler_start = source.index('    def on_external_detections')
    handler_end = source.index('    def parse_payload_capture_pose', handler_start)
    handler = source[handler_start:handler_end]

    assert 'RISK_DETECTION_PIPELINE |' in source
    for key in (
        'raw_detection_count',
        'schema_valid_count',
        'active_source_valid_count',
        'class_match_count',
        'label_match_count',
        'confidence_pass_count',
        'pose_match_count',
        'range_valid_count',
        'evidence_created_count',
        'rejected_schema',
        'rejected_source',
        'rejected_class',
        'rejected_label',
        'rejected_confidence',
        'rejected_pose',
        'rejected_stale',
        'rejected_range',
    ):
        assert key in handler


def test_risk_map_publish_debug_reports_empty_vs_positive_grid_stats():
    source = (
        Path(__file__).parents[1]
        / 'bayesian_risk_map'
        / 'bayesian_risk_map_node.py'
    ).read_text(encoding='utf-8')

    assert 'RISK_MAP_PUBLISH_DEBUG |' in source
    assert "state={grid_state}" in source
    assert "'positive_grid'" in source
    assert "'empty_grid'" in source
    assert 'unknown_count=' in source
    assert 'zero_count=' in source
    assert 'positive_count=' in source
    assert 'mean_positive=' in source
    assert "force=True" in source
