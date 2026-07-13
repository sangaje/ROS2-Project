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
from system_bringup.leader_shadow_follow import LeaderShadowFollow


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
    assert LeaderShadowFollow._is_omx_aiming('TRACKING')
    assert LeaderShadowFollow._is_omx_aiming('CONFIRMING')
    assert LeaderShadowFollow._is_omx_aiming('firing')
    assert not LeaderShadowFollow._is_omx_aiming('SCANNING')
