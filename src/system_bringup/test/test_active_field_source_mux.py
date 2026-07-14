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


def test_best_effort_qos_actually_sets_reliability_best_effort():
    # Regression test: QoSProfile(depth=10) with no explicit reliability=
    # defaults to RELIABLE in rclpy, despite the variable being named
    # best_effort. That silently requested RELIABLE against the
    # domain_bridge's actual BEST_EFFORT republish of /scout/signal and
    # /field/<robot>/risk_observation, so this node never received either
    # topic -- confirmed on hardware via a
    # "requesting incompatible QoS... RELIABILITY_QOS_POLICY" bridge warning.
    source = MUX.read_text(encoding='utf-8')

    assert (
        'best_effort = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)'
        in source
    )


def test_risk_observation_subscription_covers_every_registry_robot():
    # Regression test: this subscription used to sit outside the
    # `for robot in self.robot_names:` loop, so it only ever covered
    # whichever robot happened to be last in the list instead of every
    # field robot in the registry.
    source = MUX.read_text(encoding='utf-8')
    loop_start = source.index('for robot in self.robot_names:')
    loop_end = source.index('if self.robot_names:')
    loop_body = source[loop_start:loop_end]

    assert "f'/field/{robot}/risk_observation'" in loop_body
