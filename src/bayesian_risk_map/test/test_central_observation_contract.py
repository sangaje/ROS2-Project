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
