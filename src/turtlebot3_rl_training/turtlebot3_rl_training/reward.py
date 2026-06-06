import math
import numpy as np


TARGET_NONE = "none"
TARGET_UNKNOWN = "unknown"
TARGET_STALE = "stale"
TARGET_LOW_CONFIDENCE = "low_confidence"
TARGET_PRIORITY_GAP = "priority_gap"


def compute_reward(
    prev_distance: float,
    curr_distance: float,
    action: np.ndarray,
    prev_action: np.ndarray,
    collision: bool,
    goal_reached: bool,
    fallen: bool = False,
) -> float:
    """
    TurtleBot3 navigation용 기본 reward.

    기본 navigation task용 보상이다. exploration 학습에서는
    compute_exploration_reward()가 주로 사용된다.
    """
    linear_x = float(action[0])
    angular_z = float(action[1])

    if collision:
        return -200.0

    if fallen:
        return -200.0

    reward = 0.0

    progress = prev_distance - curr_distance
    reward += 5.0 * progress

    reward -= 0.01
    reward -= 0.015 * abs(angular_z)
    reward -= 0.006 * float(np.linalg.norm(action - prev_action))

    if linear_x < 0.03 and abs(angular_z) > 0.5:
        reward -= 0.03

    if goal_reached:
        reward += 100.0

    return float(reward)


def _gaussian_alignment(angle_rad: float, sigma_deg: float) -> float:
    """angle=0일 때 1, sigma_deg를 벗어날수록 빠르게 0으로 줄어드는 정렬도."""
    sigma = max(math.radians(float(sigma_deg)), 1e-6)
    a = float(angle_rad)
    return float(math.exp(-0.5 * (a / sigma) ** 2))


def _normalize_angle(angle_rad: float) -> float:
    return math.atan2(math.sin(float(angle_rad)), math.cos(float(angle_rad)))


def _signed_path_alignment(
    action_path_error: float,
    threshold_deg: float = 60.0,
) -> tuple[float, float, float]:
    """
    Path 방향과 command arc 방향의 차이를 signed score로 변환한다.

    기준:
      - |error| < threshold_deg  => positive reward region
      - |error| = threshold_deg  => neutral
      - |error| > threshold_deg  => negative reward region

    cos(error)를 그대로 쓰면 90도 기준이 된다. 여기서는 사용자가 의도한
    60도 기준을 명시적으로 적용하기 위해 cos(60deg)를 임계값으로 둔다.

    반환:
      signed   : [-1, 1], 0은 threshold_deg 경계
      positive : [0, 1], threshold 안쪽 정렬도
      negative : [0, 1], threshold 바깥 반대/이탈 정도
    """
    threshold_rad = max(math.radians(float(threshold_deg)), 1e-6)
    threshold_cos = math.cos(threshold_rad)
    c = math.cos(float(action_path_error))

    if c >= threshold_cos:
        positive = (c - threshold_cos) / max(1.0 - threshold_cos, 1e-6)
        negative = 0.0
    else:
        positive = 0.0
        # c는 [-1, threshold_cos) 범위다. 180도에서 negative=1이 되게 정규화한다.
        negative = (threshold_cos - c) / max(threshold_cos + 1.0, 1e-6)

    positive = float(np.clip(positive, 0.0, 1.0))
    negative = float(np.clip(negative, 0.0, 1.0))
    signed = positive if positive > 0.0 else -negative
    return float(signed), positive, negative


def _commanded_arc_angle(
    linear_x: float,
    angular_z: float,
    horizon_sec: float = 0.35,
) -> tuple[float, bool]:
    """
    현재 action이 짧은 시간 뒤 만들 이동 arc의 평균 진행 방향을 로봇 기준 각도로 근사한다.

    반환값:
      - arc_angle: 로봇 정면 기준 이동 방향. 양수는 왼쪽, 음수는 오른쪽.
      - has_translation: 선속도가 충분해서 path 추종 각도 비교가 의미 있는지.

    이유:
      path_angle은 경로 next waypoint 방향이다. 그런데 action=(v,w)가 실제로 만들
      짧은 궤적은 단순히 로봇 정면이 아니라 v와 w가 결합된 arc다. 따라서
      reward는 path_angle과 commanded arc angle의 차이도 봐야 한다.
    """
    v = max(float(linear_x), 0.0)
    w = float(angular_z)
    if v < 1e-3:
        return 0.0, False

    t = max(float(horizon_sec), 1e-3)
    wt = w * t

    if abs(w) < 1e-5:
        return 0.0, True

    # Unicycle local-frame displacement after t seconds. x=forward, y=left.
    dx = (v / w) * math.sin(wt)
    dy = (v / w) * (1.0 - math.cos(wt))

    if abs(dx) < 1e-9 and abs(dy) < 1e-9:
        return 0.0, False

    arc_angle = math.atan2(dy, dx)
    # 지나치게 큰 회전 action 하나가 reward를 비정상적으로 지배하지 않게 제한한다.
    arc_angle = float(np.clip(arc_angle, -math.radians(75.0), math.radians(75.0)))
    return arc_angle, True


def _target_type_weight(target_type: str) -> float:
    """
    target_priority/frontier_angle이 어떤 종류의 target에서 온 것인지에 따라
    reward 영향력을 다르게 둔다.

    핵심:
      - priority_gap은 사용자가 의도한 문/통로 후보이므로 강하게 사용한다.
      - unknown / low_confidence는 보조 탐색 target일 뿐 priority로 과해석하지 않는다.
    """
    t = str(target_type or TARGET_NONE)
    if t == TARGET_PRIORITY_GAP:
        return 1.0
    if t == TARGET_UNKNOWN:
        return 0.30
    if t == TARGET_LOW_CONFIDENCE:
        return 0.18
    if t == TARGET_STALE:
        return 0.14
    return 0.0


def _turn_toward_target_reward(
    angular_z: float,
    target_angle: float,
    max_angular_speed: float,
    target_weight: float,
    strict_front_alignment: float,
) -> float:
    """
    target이 정면에서 벗어나 있을 때 target 방향으로 도는 action을 보상한다.
    target이 이미 정면이면 계속 도는 행동은 벌점 처리한다.
    """
    angle_abs = abs(float(target_angle))
    turn_abs = abs(float(angular_z))
    turn_norm = float(np.clip(turn_abs / max(float(max_angular_speed), 1e-6), 0.0, 1.0))

    if target_weight <= 1e-6 or turn_norm <= 0.03:
        return 0.0

    # 10도 이내면 이미 정면 근처다. 더 돌면 overshoot/빙글빙글에 가깝다.
    if angle_abs < math.radians(10.0):
        return float(-0.55 * target_weight * strict_front_alignment * turn_norm)

    same_direction = np.sign(float(target_angle)) == np.sign(float(angular_z))
    angle_need = float(np.clip(angle_abs / math.radians(75.0), 0.0, 1.0))
    off_axis = 1.0 - float(strict_front_alignment)

    if same_direction:
        # target이 옆에 있을수록, 그리고 priority target일수록 회전 보상이 커진다.
        return float(1.55 * target_weight * angle_need * turn_norm * off_axis)

    # 반대로 도는 행동은 target switching / dithering을 강화하므로 강하게 억제한다.
    return float(-1.25 * target_weight * angle_need * turn_norm)


def compute_exploration_reward(
    new_known_cells: int,
    coverage_delta: float,
    coverage_ratio: float,
    frontier_count: int,
    robot_visit_count: int,
    action: np.ndarray,
    prev_action: np.ndarray,
    collision: bool,
    fallen: bool,
    stale_refresh_cells: int = 0,
    confidence_gain: float = 0.0,
    mean_confidence: float = 0.0,
    stale_ratio: float = 0.0,
    low_confidence_ratio: float = 0.0,
    target_priority: float = 0.0,
    frontier_angle: float = 0.0,
    target_type: str = TARGET_NONE,
    target_switched: bool = False,
    target_reachable: bool = False,
    path_distance: float = 0.0,
    path_angle: float = 0.0,
    path_progress: float = 0.0,
    priority_score: float = 0.0,
    priority_gain: float = 0.0,
    priority_cleared_cells: int = 0,
    priority_clear_gain: float = 0.0,
    wall_support_score: float = 0.0,
    open_space_score: float = 0.0,
    open_space_forward_penalty: float = 0.85,
    explored_stall_steps: int = 0,
    explored_stall_start_steps: int = 8,
    explored_stall_growth: float = 0.008,
    explored_stall_power: float = 1.45,
    explored_stall_max_penalty: float = 1.20,
    max_linear_speed: float = 0.22,
    max_angular_speed: float = 1.5,
) -> float:
    """
    SLAM + confidence + priority 기반 exploration reward.

    핵심 변경점:
      - priority 좌표를 직선으로 바라보는 reward를 제거한다.
      - target으로 실제 갈 수 있는 path가 있을 때만 path next-waypoint 방향 보상을 준다.
      - path distance가 줄어드는 progress를 dense reward로 사용한다.
      - path가 없는 벽 뒤 target은 dense reward를 거의 주지 않고, 비비는 행동을 벌점화한다.
      - priority clear는 여전히 가장 강한 positive event다.
    """
    linear_x = float(action[0])
    angular_z = float(action[1])

    if collision:
        return -200.0
    if fallen:
        return -200.0

    reward = 0.0

    # ------------------------------------------------------------------
    # 1) Normalization / semantics
    # ------------------------------------------------------------------
    target_type = str(target_type or TARGET_NONE)
    is_priority_target = target_type == TARGET_PRIORITY_GAP
    semantic_weight = _target_type_weight(target_type)

    target_priority_norm = float(np.clip(float(target_priority), 0.0, 1.0))
    target_reachable = bool(target_reachable)

    # Unknown/low-confidence target은 약하게, priority_gap은 강하게 본다.
    # 단, reachable path가 없으면 dense shaping은 거의 죽인다.
    reach_gate = 1.0 if target_reachable else 0.12
    semantic_target = target_priority_norm * semantic_weight * reach_gate

    priority_score_norm = float(np.clip(float(priority_score), 0.0, 1.0))
    priority_gain_norm = float(np.clip(float(priority_gain) / 0.20, 0.0, 1.0))
    priority_clear_norm = float(np.clip(float(priority_clear_gain) / 6.0, 0.0, 1.0))
    priority_clear_cell_norm = float(np.clip(float(priority_cleared_cells) / 100.0, 0.0, 1.0))
    priority_event = max(priority_clear_norm, priority_clear_cell_norm)

    wall_support_norm = float(np.clip(float(wall_support_score), 0.0, 1.0))
    open_space_norm = float(np.clip(float(open_space_score), 0.0, 1.0))

    # frontier_angle은 이제 ExplorationGridMap에서 reachable path가 있으면
    # path next-waypoint angle로 채워진다. path_angle도 같은 의미로 명시 전달된다.
    actionable_angle = float(path_angle if target_reachable else frontier_angle)
    if not np.isfinite(actionable_angle):
        actionable_angle = 0.0

    strict_path_alignment = _gaussian_alignment(actionable_angle, sigma_deg=14.0) if frontier_count > 0 and target_reachable else 0.0
    soft_path_alignment = _gaussian_alignment(actionable_angle, sigma_deg=26.0) if frontier_count > 0 and target_reachable else 0.0

    forward_norm = float(
        np.clip(max(linear_x, 0.0) / max(float(max_linear_speed), 1e-6), 0.0, 1.0)
    )
    turn_norm = float(
        np.clip(abs(angular_z) / max(float(max_angular_speed), 1e-6), 0.0, 1.0)
    )

    path_progress_m = float(path_progress) if np.isfinite(float(path_progress)) else 0.0
    path_progress_pos = float(np.clip(path_progress_m / 0.08, 0.0, 1.0))
    path_progress_neg = float(np.clip((-path_progress_m) / 0.08, 0.0, 1.0))

    # 현재 action이 만들어낼 짧은 arc가 path next-waypoint 방향과 얼마나 다른지 계산한다.
    # strict_path_alignment는 "로봇 heading이 path를 향하는가"이고,
    # action_path_alignment는 "이번 action 궤적이 path를 따라가는가"이다.
    commanded_arc_angle, has_translation_arc = _commanded_arc_angle(
        linear_x=linear_x,
        angular_z=angular_z,
    )
    if target_reachable and has_translation_arc and forward_norm > 0.05:
        action_path_error = _normalize_angle(actionable_angle - commanded_arc_angle)
        # 범위를 줄인다: Gaussian도 16도 sigma로 좁혀 path와 거의 같은 arc만 강하게 인정한다.
        action_path_alignment = _gaussian_alignment(action_path_error, sigma_deg=16.0)

        # Signed alignment by 60-degree rule.
        #   |error| < 60 deg  => positive reward
        #   |error| = 60 deg  => neutral
        #   |error| > 60 deg  => negative reward
        # 90도 기준보다 훨씬 엄격하게 path 이탈을 벌점화한다.
        action_path_signed, action_path_positive, action_path_negative = _signed_path_alignment(
            action_path_error,
            threshold_deg=60.0,
        )
        action_path_mismatch = float(
            np.clip(abs(action_path_error) / math.radians(60.0), 0.0, 1.0)
        )
    else:
        action_path_error = 0.0
        action_path_alignment = 0.0
        action_path_signed = 0.0
        action_path_positive = 0.0
        action_path_negative = 0.0
        action_path_mismatch = 0.0

    # path가 없으면 view_structure를 target 방향으로 인정하지 않는다.
    view_structure = float(
        np.clip(
            max(
                0.70 * wall_support_norm,
                semantic_target * soft_path_alignment,
                0.90 * priority_event,
            ),
            0.0,
            1.0,
        )
    )
    empty_void = float(np.clip(open_space_norm * (1.0 - view_structure), 0.0, 1.0))

    # ------------------------------------------------------------------
    # 2) Information reward: real exploration, not empty-space harvesting
    # ------------------------------------------------------------------
    info_gain = float(np.clip(min(float(new_known_cells), 80.0) / 80.0, 0.0, 1.0))
    coverage_gain = float(max(float(coverage_delta), 0.0))
    confidence_gain_norm = float(np.clip(float(confidence_gain) / 4.0, 0.0, 1.0))
    stale_refresh_gain = float(np.clip(min(float(stale_refresh_cells), 80.0) / 80.0, 0.0, 1.0))

    info_gate = float(
        np.clip(
            0.04
            + 0.25 * wall_support_norm
            + 0.35 * semantic_target * soft_path_alignment
            + 0.90 * priority_event,
            0.04,
            1.0,
        )
    )

    reward += 2.80 * info_gain * info_gate
    reward += 7.00 * coverage_gain * info_gate
    reward += 0.70 * confidence_gain_norm * info_gate
    reward += 0.22 * stale_refresh_gain * (0.25 + 0.75 * info_gate)

    if empty_void > 0.20:
        reward -= empty_void * (
            3.20 * info_gain
            + 4.80 * coverage_gain
            + 1.00 * confidence_gain_norm
            + 0.18 * stale_refresh_gain
        )

    # ------------------------------------------------------------------
    # 3) Path-based priority shaping
    # ------------------------------------------------------------------
    # priority가 있다는 사실 자체는 거의 보상하지 않는다. 실제 reachable path
    # progress와 clear event가 중심이다.
    reward += 0.015 * priority_score_norm
    reward += 0.06 * priority_gain_norm

    if is_priority_target and target_reachable:
        # 핵심: 목표 좌표를 바라보는 것이 아니라 path distance가 줄어드는 것을 보상.
        # 단, 현재 command arc가 path 방향과 60도 이상 어긋나면(path_signed < 0)
        # path_progress가 우연히 양수로 잡히더라도 positive reward를 주지 않는다.
        # 사용자가 지정한 priority로 가는 path의 반대 방향 action은 질적으로 잘못된 행동이다.
        correct_arc_gate = action_path_positive if has_translation_arc else strict_path_alignment
        wrong_arc_gate = action_path_negative if has_translation_arc else 0.0

        reward += 8.80 * target_priority_norm * path_progress_pos * correct_arc_gate
        reward -= 5.80 * target_priority_norm * path_progress_neg

        # heading이 path next waypoint를 향하는지만 보는 보상은 매우 약하게 둔다.
        # 핵심은 action arc와 path 방향의 60도 threshold signed alignment다.
        # |path_error| < 60deg 이면 +, |path_error| > 60deg 이면 강한 - reward.
        reward += 0.18 * forward_norm * target_priority_norm * strict_path_alignment * correct_arc_gate
        reward += 4.80 * forward_norm * target_priority_norm * action_path_positive
        reward -= 13.50 * forward_norm * target_priority_norm * action_path_negative

        # reachable path가 있고 그 방향을 바라보는 상태 자체는 약하게만 보상하되,
        # action이 반대 방향이면 이 항도 꺼서 priority 직선 attraction이 남지 않게 한다.
        reward += 0.04 * target_priority_norm * soft_path_alignment * correct_arc_gate

        # 60도 이상 반대 방향으로 움직이는데 다른 정보획득/coverage 보상으로
        # 총 reward가 양수가 되는 exploit을 막기 위한 추가 hard penalty.
        if forward_norm > 0.08 and wrong_arc_gate > 0.0:
            reward -= 10.50 * forward_norm * target_priority_norm * (0.45 + wrong_arc_gate)
            reward -= 3.50 * target_priority_norm * path_progress_pos * wrong_arc_gate
    elif (not is_priority_target) and target_reachable:
        # unknown / low-confidence path following은 보조 탐색이다.
        reward += 1.10 * semantic_target * path_progress_pos
        reward -= 0.45 * semantic_target * path_progress_neg
        reward += 0.10 * forward_norm * semantic_target * strict_path_alignment
        reward += 0.85 * forward_norm * semantic_target * action_path_positive
        reward -= 0.85 * forward_norm * semantic_target * action_path_negative

    # 실제 priority를 확인해서 -1 checked로 만든 이벤트는 가장 강한 보상.
    reward += 7.50 * priority_clear_norm
    reward += 2.50 * priority_clear_cell_norm

    # ------------------------------------------------------------------
    # 4) Forward / turning policy using path angle, not target bearing
    # ------------------------------------------------------------------
    # 일반 전진은 구조 support가 있을 때만 약하게 보상.
    path_forward_alignment = max(soft_path_alignment, 0.75 * action_path_positive)
    reward += 0.22 * forward_norm * (0.20 + 0.80 * wall_support_norm) * (0.30 + 0.70 * path_forward_alignment)

    if target_reachable and semantic_target > 1e-6:
        # path next waypoint 방향으로 도는 것만 보상한다.
        reward += _turn_toward_target_reward(
            angular_z=angular_z,
            target_angle=actionable_angle,
            max_angular_speed=max_angular_speed,
            target_weight=semantic_target,
            strict_front_alignment=strict_path_alignment,
        )
    else:
        # 벽 뒤 priority / unreachable target의 직선 방향 회전은 보상하지 않는다.
        if target_priority_norm > 0.20 and frontier_count > 0:
            reward -= 0.18 * target_priority_norm * turn_norm

    # path가 있는데 정렬 전 무작정 전진하면 벌점.
    if is_priority_target and target_reachable and target_priority_norm > 0.20 and forward_norm > 0.12:
        if strict_path_alignment < 0.45:
            reward -= 2.40 * forward_norm * target_priority_norm * (1.0 - strict_path_alignment)
        if action_path_signed <= 0.0:
            # The commanded arc is more than 60 degrees away from the planned path.
            # This is qualitatively wrong-path motion, not merely weak alignment.
            # Make it dominant enough that coverage/confidence side rewards cannot make
            # wrong-way priority motion profitable.
            reward -= 10.50 * forward_norm * target_priority_norm * (1.0 + 1.80 * action_path_negative)
            reward -= 3.00 * target_priority_norm * path_progress_pos * (1.0 + action_path_negative)
        elif action_path_positive < 0.35:
            reward -= 2.40 * forward_norm * target_priority_norm * (1.0 - action_path_positive)

    # path가 없는 target인데 앞으로 밀고 나가면 벽 비비기/허공 전진으로 간주.
    if is_priority_target and (not target_reachable) and target_priority_norm > 0.20:
        reward -= 0.75 * forward_norm * target_priority_norm

    effective_open_penalty = max(float(open_space_forward_penalty), 0.85)
    reward -= effective_open_penalty * forward_norm * empty_void

    # 목표가 없으면 큰 회전은 작게 벌점. 목표가 reachable이면 회전은 path reward가 판단한다.
    if frontier_count <= 0 or semantic_target <= 1e-6:
        reward -= 0.025 * turn_norm
    else:
        reward -= 0.004 * turn_norm

    # ------------------------------------------------------------------
    # 5) Target switching / jitter penalties
    # ------------------------------------------------------------------
    if bool(target_switched) and priority_event <= 0.05:
        reward -= 0.30 if is_priority_target else 0.16

    action_delta = float(np.linalg.norm(action - prev_action))
    reward -= 0.012 * action_delta

    prev_w = float(prev_action[1])
    curr_w = float(action[1])
    sign_flip = (
        abs(prev_w) > 0.25
        and abs(curr_w) > 0.25
        and np.sign(prev_w) != np.sign(curr_w)
    )
    no_information = (
        int(new_known_cells) <= 1
        and int(stale_refresh_cells) <= 1
        and float(confidence_gain) <= 0.02
        and priority_event <= 0.03
        and path_progress_pos <= 0.02
    )

    if sign_flip:
        if priority_event > 0.15:
            reward -= 0.04
        elif no_information:
            reward -= 0.45
        else:
            reward -= 0.18

    # ------------------------------------------------------------------
    # 6) Revisit / stall / spin penalties
    # ------------------------------------------------------------------
    if robot_visit_count > 8:
        reward -= 0.045 * min(int(robot_visit_count) - 8, 25)

    stall_excess = max(int(explored_stall_steps) - int(explored_stall_start_steps), 0)
    if no_information and stall_excess > 0:
        stall_penalty = float(explored_stall_growth) * (
            float(stall_excess) ** float(explored_stall_power)
        )
        stall_penalty = min(float(explored_stall_max_penalty), stall_penalty)
        reward -= stall_penalty

    if max(linear_x, 0.0) < 0.035 and turn_norm > 0.35 and robot_visit_count > 8:
        if no_information:
            spin_repeat = min(max(int(robot_visit_count) - 8, 0), 30) / 30.0
            reward -= 0.36 * spin_repeat * turn_norm
        elif not is_priority_target:
            reward -= 0.08 * turn_norm

    if no_information:
        reward -= 0.06
        if forward_norm > 0.10:
            reward -= 0.35 * forward_norm * empty_void

    if stale_ratio > 0.20 and stale_refresh_cells <= 1:
        reward -= 0.020 * min(float(stale_ratio), 1.0)

    if low_confidence_ratio > 0.30 and confidence_gain <= 0.02:
        reward -= 0.020 * min(float(low_confidence_ratio), 1.0)

    reward += 0.010 * float(coverage_ratio)
    reward += 0.006 * np.clip(float(mean_confidence) / 100.0, 0.0, 1.0)

    return float(reward)
