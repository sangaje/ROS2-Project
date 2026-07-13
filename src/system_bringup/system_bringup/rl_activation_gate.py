"""Pure role-gated activation decision for the external scout RL worker."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .motion_authority import MotionAuthority
from .role_contract import Role


class RLWorkerState(str, Enum):
    STANDBY = 'STANDBY'
    RECOVERY_NAVIGATING = 'RECOVERY_NAVIGATING'
    WAIT_LOCALIZATION = 'WAIT_LOCALIZATION'
    WAIT_MOTION_RELEASE = 'WAIT_MOTION_RELEASE'
    WAIT_SENSOR_READY = 'WAIT_SENSOR_READY'
    WAIT_OBSERVATION_READY = 'WAIT_OBSERVATION_READY'
    ACTIVE = 'ACTIVE'
    FAILED = 'FAILED'


@dataclass(frozen=True)
class GateInputs:
    role: str
    role_robot_matches: bool
    role_epoch: int
    failover_epoch: int
    active_scout_matches: bool
    failover_state: str
    localization_ready: bool
    recovery_complete: bool
    nav_goal_inactive: bool
    motion_authority: str
    model_ready: bool
    sensor_ready: bool
    tf_ready: bool
    require_failover_activation: bool
    require_localization_ready: bool
    # Raw scan/odom/SLAM-map freshness (``sensor_ready``) is not sufficient
    # for ACTIVE: the internal MapSnapshot the policy actually reads for
    # inference can still be stale (e.g. the confidence-grid tick fell
    # behind) even while every raw topic is fresh. Defaults to True so
    # existing callers that construct GateInputs without this field keep
    # their previous behavior.
    observation_ready: bool = True


@dataclass(frozen=True)
class ActivationDecision:
    state: RLWorkerState
    reason: str

    @property
    def allowed(self) -> bool:
        return self.state == RLWorkerState.ACTIVE


@dataclass(frozen=True)
class BackendGateInputs:
    role: Role
    scout_enabled: bool
    require_localization_ready: bool
    localization_ready: bool
    nav_idle: bool


def evaluate_backend_activation(gate: BackendGateInputs) -> tuple[bool, str]:
    """Check common preconditions before either RL backend gets authority."""
    if gate.role != Role.ACTIVE_SCOUT:
        return False, 'role_not_active_scout'
    if not gate.scout_enabled:
        return False, 'scout_rl_disabled'
    if gate.require_localization_ready and not gate.localization_ready:
        return False, 'localization_not_ready'
    if not gate.nav_idle:
        return False, 'nav_goal_active'
    return True, 'activation_gate_passed'


def evaluate_activation(gate: GateInputs) -> ActivationDecision:
    role = Role.__members__.get(gate.role.strip().upper(), Role.IDLE)
    failover_state = gate.failover_state.strip().upper()
    motion_authority = gate.motion_authority.strip().upper()
    require_failover = gate.require_failover_activation
    if role == Role.FAILED:
        return ActivationDecision(RLWorkerState.FAILED, 'role_failed')
    if (
        role in (Role.RECOVERY_NAVIGATING, Role.FOLLOWER, Role.IDLE)
        or (
            require_failover
            and (
                failover_state in ('RECOVERY_NAVIGATING', 'FAILOVER_TRIGGERED')
                or motion_authority == MotionAuthority.FAILOVER_RECOVERY_NAV.value
            )
        )
    ):
        return ActivationDecision(RLWorkerState.RECOVERY_NAVIGATING, 'recovery_or_non_scout_role')
    if role != Role.ACTIVE_SCOUT or not gate.role_robot_matches:
        return ActivationDecision(RLWorkerState.STANDBY, 'role_not_active_scout')
    if require_failover and gate.role_epoch < gate.failover_epoch:
        return ActivationDecision(RLWorkerState.STANDBY, 'stale_epoch')
    if require_failover and not gate.active_scout_matches:
        return ActivationDecision(RLWorkerState.STANDBY, 'active_scout_id_mismatch')
    if gate.require_localization_ready and not gate.localization_ready:
        return ActivationDecision(RLWorkerState.WAIT_LOCALIZATION, 'localization_not_ready')
    if require_failover and not gate.recovery_complete:
        return ActivationDecision(RLWorkerState.WAIT_LOCALIZATION, 'recovery_not_complete')
    if (
        not gate.nav_goal_inactive
        or (
            require_failover
            and motion_authority not in ('NONE', MotionAuthority.ACTIVE_SCOUT_RL.value, '')
        )
    ):
        return ActivationDecision(RLWorkerState.WAIT_MOTION_RELEASE, 'motion_authority_busy')
    if not gate.model_ready or not gate.sensor_ready or not gate.tf_ready:
        return ActivationDecision(RLWorkerState.WAIT_SENSOR_READY, 'runtime_inputs_not_ready')
    if not gate.observation_ready:
        return ActivationDecision(RLWorkerState.WAIT_OBSERVATION_READY, 'observation_stale')
    return ActivationDecision(RLWorkerState.ACTIVE, 'all_runtime_inputs_ready')


def evaluate_activation_gate(gate: GateInputs) -> tuple[RLWorkerState, str]:
    """Compatibility API retained for existing callers and tests."""
    decision = evaluate_activation(gate)
    return decision.state, decision.reason
