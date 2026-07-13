from pathlib import Path

from system_bringup.motion_authority import (
    MotionAuthority,
    authority_allows_nonzero,
    nav_motion_is_quiescent,
)
from system_bringup.rl_activation_gate import (
    BackendGateInputs,
    evaluate_backend_activation,
)
from system_bringup.role_contract import Role


def test_motion_authority_allows_only_the_current_command_owner():
    assert authority_allows_nonzero(
        MotionAuthority.ACTIVE_SCOUT_RL, MotionAuthority.ACTIVE_SCOUT_RL
    )
    assert not authority_allows_nonzero(
        MotionAuthority.FAILOVER_RECOVERY_NAV, MotionAuthority.ACTIVE_SCOUT_RL
    )


def test_nav_motion_quiescence_requires_no_goal_or_pending_cancel():
    assert nav_motion_is_quiescent(0, 0)
    assert not nav_motion_is_quiescent(1, 0)
    assert not nav_motion_is_quiescent(0, 1)


def test_backend_activation_requires_active_role_localization_and_nav_release():
    ready = BackendGateInputs(
        role=Role.ACTIVE_SCOUT,
        scout_enabled=True,
        require_localization_ready=True,
        localization_ready=True,
        nav_idle=True,
    )
    assert evaluate_backend_activation(ready) == (True, 'activation_gate_passed')
    assert evaluate_backend_activation(
        BackendGateInputs(**{**ready.__dict__, 'nav_idle': False})
    ) == (False, 'nav_goal_active')


def test_leader_shadow_pauses_for_target_lock_states():
    source = (Path(__file__).parents[1] / 'system_bringup' / 'leader_shadow_follow.py').read_text(
        encoding='utf-8'
    )

    assert "@staticmethod\n    def _is_omx_aiming" in source
    assert "'TRACKING'" in source
    assert "'CONFIRMING'" in source
    assert "'FIRING'" in source
    assert "'COOLDOWN'" in source


def test_leader_shadow_hard_stops_on_best_effort_target_detection():
    source = (Path(__file__).parents[1] / 'system_bringup' / 'leader_shadow_follow.py').read_text(
        encoding='utf-8'
    )

    assert "self.declare_parameter('pause_on_raw_target_detection', True)" in source
    assert "self.declare_parameter('target_detected_stop_hold_sec', 3.0)" in source
    assert "self.declare_parameter('target_detected_cancel_period_sec', 0.25)" in source
    assert 'ReliabilityPolicy.BEST_EFFORT' in source
    assert 'def _force_leader_stop_for_target' in source
    assert "self._pulse_cancel()" in source
    assert "self._publish_twist(0.0, 0.0)" in source
    assert "base_motion_stopped=true omx_pd_allowed=true" in source


def test_leader_shadow_blocks_nav2_goal_publish_during_target_hold():
    source = (Path(__file__).parents[1] / 'system_bringup' / 'leader_shadow_follow.py').read_text(
        encoding='utf-8'
    )
    publish_nav = source.split('def _publish_nav2_goal', 1)[1].split(
        'def _on_nav_goal_response',
        1,
    )[0]

    assert 'target_reason = self._target_hold_reason()' in publish_nav
    assert 'self._hold_for_omx_target(target_reason)' in publish_nav
    assert "self.declare_parameter('target_memory_hold_sec', 3.0)" in source


def test_leader_shadow_backend_is_explicit_and_debug_logs_motion_chain():
    source = (Path(__file__).parents[1] / 'system_bringup' / 'leader_shadow_follow.py').read_text(
        encoding='utf-8'
    )
    launch = (Path(__file__).parents[1] / 'launch' / 'system.launch.py').read_text(
        encoding='utf-8'
    )

    assert "self.declare_parameter('leader_follow_backend', 'nav2')" in source
    assert "self.follow_backend not in ('nav2', 'direct')" in source
    assert "LEADER_FOLLOW_DEBUG |" in source
    assert "LEADER_NAV2_PIPELINE |" in source
    assert "goal_accepted=" in source
    assert "target_behind_scout=" in source
    assert "target_mode=" in source
    assert "hold_no_safe_rear_goal" in source
    assert "stopped_close_to_scout" in source
    assert "hold_resume_hysteresis" in source
    assert "path_age_ms" in source
    assert "controller_cmd_age_ms" in source
    assert "hardware_cmd_age_ms" in source
    assert "nonzero_cmd_age_ms" in source
    assert "odom_motion" in source
    assert "DeclareLaunchArgument(\n            'leader_follow_backend'" in launch
