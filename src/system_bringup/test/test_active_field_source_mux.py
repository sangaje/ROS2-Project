from pathlib import Path


MUX = (
    Path(__file__).parents[1]
    / 'system_bringup'
    / 'active_field_source_mux.py'
)


def test_active_field_source_mux_uses_registry_identity_topics_and_epoch_filter():
    source = MUX.read_text(encoding='utf-8')

    assert "fleet_registry_json" in source
    assert "f'/field/{robot}/map'" in source
    assert "f'/field/{robot}/pose'" in source
    assert "f'/field/{robot}/heartbeat'" in source
    assert "f'/field/{robot}/risk_observation'" in source
    assert "'/active_scout/map'" in source
    assert "'/active_scout/pose'" in source
    assert "'/active_scout/risk_observation'" in source
    assert "parse_epoch(payload.get('epoch'" in source
    assert "return epoch == self.epoch" in source
