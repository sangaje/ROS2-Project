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
        return -100.0

    if fallen:
        return -100.0

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
        return float(-0.70 * target_weight * strict_front_alignment * turn_norm)

    same_direction = np.sign(float(target_angle)) == np.sign(float(angular_z))
    angle_need = float(np.clip(angle_abs / math.radians(75.0), 0.0, 1.0))
    off_axis = 1.0 - float(strict_front_alignment)

    if same_direction:
        # target이 옆에 있을수록 회전 보상을 주되, translation path-following보다 작게 둔다.
        # 회전 보상이 너무 크면 SAC가 path를 실제로 따라가기보다 바라보기/제자리 회전을
        # 보상 exploit으로 사용할 수 있다.
        return float(0.95 * target_weight * angle_need * turn_norm * off_axis)

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
    target_lock_age: int = 0,
    target_reachable: bool = False,
    path_distance: float = 0.0,
    path_angle: float = 0.0,
    path_progress: float = 0.0,
    alternative_path_angles: tuple[float, ...] | None = None,
    priority_score: float = 0.0,
    priority_gain: float = 0.0,
    priority_cleared_cells: int = 0,
    priority_clear_gain: float = 0.0,
    priority_rechecked_cells: int = 0,
    priority_rechecked_gain: float = 0.0,
    wall_support_score: float = 0.0,
    open_space_score: float = 0.0,
    open_space_forward_penalty: float = 0.85,
    explored_stall_steps: int = 0,
    explored_stall_start_steps: int = 8,
    explored_stall_growth: float = 0.008,
    explored_stall_power: float = 1.45,
    explored_stall_max_penalty: float = 1.20,
    confidence_stall_steps: int = 0,
    confidence_stall_start_steps: int = 6,
    confidence_stall_growth: float = 0.010,
    confidence_stall_power: float = 1.35,
    confidence_stall_max_penalty: float = 1.60,
    confidence_stall_gain_threshold: float = 0.02,
    confidence_stall_low_ratio_threshold: float = 0.20,
    sustained_rotation_steps: int = 0,
    sustained_rotation_start_steps: int = 8,
    sustained_rotation_growth: float = 0.030,
    sustained_rotation_power: float = 1.45,
    sustained_rotation_max_penalty: float = 2.50,
    orbit_stall_steps: int = 0,
    orbit_stall_start_steps: int = 5,
    orbit_stall_growth: float = 0.026,
    orbit_stall_power: float = 1.45,
    orbit_stall_max_penalty: float = 3.00,
    orbit_path_efficiency: float = 1.0,
    orbit_path_length: float = 0.0,
    orbit_yaw_accum: float = 0.0,
    max_linear_speed: float = 0.22,
    max_angular_speed: float = 1.5,
    nearest_obstacle_distance: float = 999.0,
    obstacle_proximity_score: float = 0.0,
    lidar_action_obstacle_distance: float = 999.0,
    lidar_action_obstacle_score: float = 0.0,
    lidar_front_obstacle_distance: float = 999.0,
    use_path_reward: bool = False,
    use_wall_proximity_penalty: bool = True,
    use_corridor_priority_reward: bool = False,
    corridor_priority_reward_weight: float = 1.65,
    confidence_reward_weight: float = 1.0,
) -> float:
    """
    v25.9 anti-orbit reward.

    설계 원칙:
      1. priority가 "존재/생성"됐다는 사실은 보상하지 않는다.
         -> 생성된 빨간 영역을 바라보기/빙빙돌기로 farm하는 것을 차단.
      2. 양의 reward는 실제 정보 증가와 priority 확인/제거에만 준다.
      3. per-step dense reward는 작게, terminal penalty만 크게 둔다.
      4. 제자리 회전/작은 원형 궤도/무정보 반복 방문은 음수로 만든다.
      5. terminal(-100)은 그대로 두고, non-terminal reward는 clip한다.
      6. 전진+고각속도 원형 주행은 명시적으로 음수화한다.
    """
    linear_x = float(action[0]) if action is not None and len(action) > 0 else 0.0
    angular_z = float(action[1]) if action is not None and len(action) > 1 else 0.0

    if collision:
        return -100.0
    if fallen:
        return -100.0

    max_v = max(float(max_linear_speed), 1e-6)
    max_w = max(float(max_angular_speed), 1e-6)
    forward_norm = float(np.clip(max(linear_x, 0.0) / max_v, 0.0, 1.0))
    turn_norm = float(np.clip(abs(angular_z) / max_w, 0.0, 1.0))

    try:
        prev = np.asarray(prev_action, dtype=np.float32)
        cur = np.asarray(action, dtype=np.float32)
        action_delta = float(np.linalg.norm(cur - prev))
    except Exception:
        action_delta = 0.0

    # ------------------------------------------------------------------
    # Base cost: every non-terminal step is slightly negative.
    # ------------------------------------------------------------------
    reward = -0.025
    # 회전 자체의 기본 비용을 키운다. 이전 값은 너무 작아서
    # 원형 주행이 confidence/priority 보상을 먹고 이기는 문제가 있었다.
    reward -= 0.055 * turn_norm
    reward -= 0.012 * action_delta

    # 제자리 회전은 필요한 heading alignment만 허용하고, 장기적으로 음수.
    stationary_spin = float(
        np.clip((0.12 - forward_norm) / 0.12, 0.0, 1.0)
        * np.clip((turn_norm - 0.35) / 0.65, 0.0, 1.0)
    )
    if stationary_spin > 0.0:
        reward -= 0.45 * stationary_spin * (0.35 + 0.65 * turn_norm)

    # 전진하면서 같은 자리 주변을 도는 작은 반경 orbit을 직접 억제한다.
    # gazebo_nav_env가 orbit_stall_steps를 넘기지 않는 구버전이어도 이 항은 동작한다.
    v_abs = abs(float(linear_x))
    w_abs = abs(float(angular_z))
    if v_abs > 0.015 and w_abs > 0.045:
        turn_radius = v_abs / max(w_abs, 1e-6)
        tight_turn = float(np.clip((0.55 - turn_radius) / 0.55, 0.0, 1.0))
        turn_excess = float(np.clip((turn_norm - 0.24) / 0.76, 0.0, 1.0))
        if tight_turn > 0.0 and turn_excess > 0.0:
            reward -= 0.85 * tight_turn * (turn_excess ** 1.15) * (0.35 + 0.65 * forward_norm)

    # 고각속도 원호 주행은 지도/priority 업데이트를 farm하기 쉬우므로
    # 정보 증가가 있더라도 기본적으로 비용을 부과한다.
    if forward_norm > 0.06 and turn_norm > 0.30:
        reward -= 0.32 * forward_norm * ((turn_norm - 0.30) / 0.70) ** 1.25

    # ------------------------------------------------------------------
    # Real information gain reward.
    # motion_gate를 둔 이유: 같은 자리에서 회전만 해도 LiDAR FoV 변화로
    # confidence/known delta가 조금 생기는 exploit을 막기 위해서다.
    # ------------------------------------------------------------------
    new_known = max(int(new_known_cells), 0)
    stale_cells = max(int(stale_refresh_cells), 0)
    cov_gain = max(float(coverage_delta), 0.0)
    conf_gain = max(float(confidence_gain), 0.0)
    conf_weight = max(float(confidence_reward_weight), 0.0)

    info_cells_norm = float(np.clip(new_known / 120.0, 0.0, 1.0))
    stale_norm = float(np.clip(stale_cells / 120.0, 0.0, 1.0))
    conf_norm = float(np.clip(conf_gain / 8.0, 0.0, 1.0))

    # 정보 보상은 '전진 기반'으로만 크게 준다.
    # 회전이 클수록 LiDAR FoV 변화로 생기는 가짜 gain 가능성이 크므로 보상 게이트를 줄인다.
    forward_info_gate = float(np.clip((forward_norm - 0.06) / 0.34, 0.0, 1.0))
    turn_suppression = float(np.clip(1.0 - 0.85 * turn_norm, 0.08, 1.0))
    motion_gate = forward_info_gate * turn_suppression
    meaningful_info = bool(new_known >= 4 or stale_cells >= 4 or cov_gain >= 1e-4 or conf_gain >= max(float(confidence_stall_gain_threshold), 0.02))

    if meaningful_info:
        reward += 0.95 * info_cells_norm * motion_gate
        reward += 4.00 * cov_gain * motion_gate
        reward += 0.30 * conf_weight * conf_norm * motion_gate
        reward += 0.10 * stale_norm * motion_gate
    else:
        # 움직이거나 돌았는데 정보가 없으면 명확히 손해.
        activity = float(np.clip(max(forward_norm, turn_norm), 0.0, 1.0))
        reward -= 0.055 * (0.35 + 0.65 * activity)

    # ------------------------------------------------------------------
    # Priority reward: 생성/존재/바라보기는 보상하지 않는다.
    # 오직 실제 확인(clear/recheck)만 보상한다.
    # ------------------------------------------------------------------
    clear_sum = max(float(priority_clear_gain), 0.0)
    recheck_sum = max(float(priority_rechecked_gain), 0.0)
    clear_cells = max(int(priority_cleared_cells), 0)
    recheck_cells = max(int(priority_rechecked_cells), 0)

    # priority_gain은 새 영역이 생긴 것일 뿐, 로봇이 뭔가 잘한 증거가 아니다.
    # 따라서 reward에 넣지 않는다.
    priority_check_reward = 0.0
    # priority는 '처음 직접 제거(clear)'했을 때만 보상한다.
    # recheck는 원형 주행 중 같은 영역을 계속 훑으며 보상을 farm하는 경로라 제거한다.
    if clear_sum > 0.0 or clear_cells > 0:
        priority_check_reward += 0.010 * clear_sum
        priority_check_reward += 0.00035 * min(clear_cells, 250)
    reward += min(priority_check_reward, 0.60) * motion_gate

    if recheck_sum > 0.0 or recheck_cells > 0:
        # 재확인은 유용한 정보가 아니라 중복 관측에 가깝다.
        # 약한 비용을 줘서 뺑뺑이 중복 관측을 보상 루프로 쓰지 못하게 한다.
        reward -= min(0.35, 0.003 * recheck_sum + 0.00008 * min(recheck_cells, 400))

    # ------------------------------------------------------------------
    # Obstacle / wall safety shaping.
    # terminal collision 전 dense gradient를 제공하되 너무 지배하지 않게 제한.
    # ------------------------------------------------------------------
    if bool(use_wall_proximity_penalty):
        try:
            lad = float(lidar_action_obstacle_distance)
        except Exception:
            lad = 999.0
        try:
            lfd = float(lidar_front_obstacle_distance)
        except Exception:
            lfd = 999.0
        try:
            nod = float(nearest_obstacle_distance)
        except Exception:
            nod = 999.0

        la_score = float(np.clip(float(lidar_action_obstacle_score), 0.0, 1.0))
        map_score = float(np.clip(float(obstacle_proximity_score), 0.0, 1.0))

        if la_score > 0.0:
            reward -= 0.45 * la_score
            reward -= 1.25 * forward_norm * (la_score ** 1.25)
        if lad < 0.26 and forward_norm > 0.04:
            reward -= 1.20 * forward_norm * float(np.clip((0.26 - lad) / 0.14, 0.0, 1.0))
        if lfd < 0.34 and forward_norm > 0.08:
            reward -= 1.00 * forward_norm * float(np.clip((0.34 - lfd) / 0.20, 0.0, 1.0))
        if map_score > 0.0:
            reward -= 0.35 * map_score
            reward -= 0.65 * map_score * forward_norm
        if nod < 0.22 and forward_norm > 0.05:
            reward -= 0.80 * forward_norm

    # ------------------------------------------------------------------
    # Revisit / confidence stall / rotation / orbit penalties.
    # ------------------------------------------------------------------
    if robot_visit_count > 8:
        reward -= 0.018 * min(int(robot_visit_count) - 8, 30)

    conf_thresh = max(float(confidence_stall_gain_threshold), 0.0)
    if conf_gain <= conf_thresh:
        conf_excess = max(int(confidence_stall_steps) - int(confidence_stall_start_steps), 0)
        low_ratio = float(np.clip(float(low_confidence_ratio), 0.0, 1.0))
        low_thr = float(np.clip(float(confidence_stall_low_ratio_threshold), 0.0, 1.0))
        low_gate = float(np.clip((low_ratio - low_thr) / max(1.0 - low_thr, 1e-6), 0.25, 1.0))
        reward -= 0.035 * conf_weight * low_gate * (0.40 + 0.60 * max(forward_norm, turn_norm))
        if conf_excess > 0:
            p = float(confidence_stall_growth) * (float(conf_excess) ** float(confidence_stall_power))
            p = min(float(confidence_stall_max_penalty), p)
            reward -= 0.55 * p * low_gate

    rot_excess = max(int(sustained_rotation_steps) - int(sustained_rotation_start_steps), 0)
    if rot_excess > 0 and forward_norm < 0.18 and turn_norm > 0.30:
        p = float(sustained_rotation_growth) * (float(rot_excess) ** float(sustained_rotation_power))
        p = min(float(sustained_rotation_max_penalty), p)
        reward -= 0.75 * p * float(np.clip((0.18 - forward_norm) / 0.18, 0.0, 1.0))

    orbit_excess = max(int(orbit_stall_steps) - int(orbit_stall_start_steps), 0)
    if orbit_excess > 0:
        try:
            orbit_eff = float(np.clip(float(orbit_path_efficiency), 0.0, 1.0))
        except Exception:
            orbit_eff = 1.0
        try:
            orbit_len = max(float(orbit_path_length), 0.0)
        except Exception:
            orbit_len = 0.0
        try:
            orbit_yaw = abs(float(orbit_yaw_accum))
        except Exception:
            orbit_yaw = 0.0
        loop_gate = float(
            np.clip((0.45 - orbit_eff) / 0.45, 0.0, 1.0)
            * np.clip((orbit_len - 0.18) / 0.45, 0.0, 1.0)
            * np.clip((orbit_yaw - math.radians(70.0)) / math.radians(150.0), 0.0, 1.0)
        )
        if loop_gate > 0.0:
            p = float(orbit_stall_growth) * (float(orbit_excess) ** float(orbit_stall_power))
            p = min(float(orbit_stall_max_penalty), p)
            reward -= 1.20 * p * loop_gate

    # target switch jitter는 아주 약하게만 유지.
    if bool(target_switched) and clear_sum <= 0.0 and recheck_sum <= 0.0:
        reward -= 0.06

    # Priority stuck penalty는 제거한다. Pstuck은 reset 조건에서도 끄고,
    # reward에서도 강한 항으로 쓰지 않는다. 그래야 priority 생성/유지 정책 변화가
    # critic target을 흔들지 않는다.

    if stale_ratio > 0.25 and stale_cells <= 1:
        reward -= 0.010 * min(float(stale_ratio), 1.0)
    if low_confidence_ratio > 0.35 and conf_gain <= conf_thresh:
        reward -= 0.010 * min(float(low_confidence_ratio), 1.0)

    # Terminal은 위에서 -100으로 따로 반환한다. non-terminal dense reward는 bounded.
    return float(np.clip(reward, -8.0, 3.0))


def compute_waypoint_macro_reward_adjustment(
    *,
    collision: bool,
    fallen: bool,
    target_reachable: bool,
    path_progress: float,
    waypoint_reached: bool,
    waypoint_timed_out: bool,
    waypoint_distance: float,
    waypoint_final_error: float,
    waypoint_reached_tolerance: float,
    controller_steps: int,
    waypoint_max_control_steps: int,
    waypoint_heading_delta: float = 0.0,
    waypoint_lateral_offset: float = 0.0,
    waypoint_lateral_max_offset: float = 0.20,
    waypoint_path_conditioned: bool = True,
    use_path_reward: bool = False,
) -> float:
    """
    Macro-action waypoint reward adjustment.

    기존 compute_exploration_reward()는 저수준 cmd_vel arc가 path와 맞는지,
    정보획득/priority clear/path progress가 있었는지 평가한다. waypoint mode에서는
    policy action 하나가 여러 controller step으로 실행되므로, 추가로 다음을 본다.

      - waypoint에 실제 도착했는가
      - 도착/실행 결과 path_distance가 줄었는가
      - timeout으로 끝났는가
      - 이전 waypoint 방향과 과하게 달라져 zigzag가 생겼는가
      - path-conditioned mode에서 lateral offset을 과하게 쓰는가

    반환값은 base reward에 더하는 작은 보정항이다. collision/fallen은 base reward에서
    이미 큰 penalty를 주므로 여기서는 0을 반환한다.
    """
    if collision or fallen:
        return 0.0

    path_reward_enabled = bool(use_path_reward)
    reachable = bool(target_reachable and path_reward_enabled)
    reached = bool(waypoint_reached)
    timed_out = bool(waypoint_timed_out)

    max_steps = max(int(waypoint_max_control_steps), 1)
    steps = int(max(int(controller_steps), 0))
    step_frac = float(np.clip(steps / max_steps, 0.0, 1.0))

    wp_dist = max(float(waypoint_distance), 1e-3)
    tol = max(float(waypoint_reached_tolerance), 1e-3)
    final_err = max(float(waypoint_final_error), 0.0)
    final_err_norm = float(np.clip(final_err / max(wp_dist, tol), 0.0, 2.0))

    progress = float(path_progress) if path_reward_enabled else 0.0
    progress_pos = float(np.clip(progress / 0.45, 0.0, 1.0))
    progress_neg = float(np.clip((-progress) / 0.35, 0.0, 1.0))

    heading_delta_abs = abs(float(waypoint_heading_delta))
    zigzag_norm = float(np.clip(heading_delta_abs / math.radians(120.0), 0.0, 1.0))

    lateral_max = max(float(waypoint_lateral_max_offset), 1e-6)
    lateral_norm = float(np.clip(abs(float(waypoint_lateral_offset)) / lateral_max, 0.0, 1.5))

    reward = 0.0

    # Path reward가 켜진 상황에서만 macro-action의 1차 목표를 path distance 감소로 둔다.
    if path_reward_enabled:
        if reachable:
            reward += 2.20 * progress_pos
            reward -= 2.70 * progress_neg
        else:
            # path가 없는 target은 waypoint 도달 자체를 크게 보상하지 않는다.
            reward -= 0.12 * step_frac

    if reached:
        # 도달은 보상하되, path progress가 동반될수록 더 크게 준다.
        reward += 0.50 if reachable else 0.12
        reward += 0.40 * progress_pos
        # 빠르게 도달한 waypoint는 controller oscillation이 적다는 의미다.
        reward += 0.18 * (1.0 - step_frac)
    else:
        # 도달하지 못했으면 남은 거리만큼 약한 penalty.
        reward -= 0.24 * min(final_err_norm, 1.0)

    if timed_out:
        # v6: waypoint/controller timeout itself is not penalized.
        # Remaining distance is already handled by the non-reached term above;
        # adding a separate timeout penalty biases the policy against cautious
        # long-horizon behavior and makes max-control-step cutoffs look like failures.
        pass

    # 이전 waypoint와 방향이 크게 바뀌는 지그재그를 억제한다.
    # path-conditioned mode에서는 원칙적으로 path tangent 주변에서 움직여야 하므로 더 강하게 건다.
    zigzag_weight = 0.48 if bool(waypoint_path_conditioned) else 0.32
    reward -= zigzag_weight * (zigzag_norm ** 1.25)

    # path-conditioned mode의 lateral offset penalty는 path reward 사용 시에만 적용한다.
    # polar goal 학습에서 /rl_path의 영향도를 완전히 제거하려면 이 항도 꺼져야 한다.
    if path_reward_enabled and bool(waypoint_path_conditioned):
        reward -= 0.12 * (min(lateral_norm, 1.0) ** 2)

    # reachable path가 있는데 실행했지만 path progress가 거의 없고 오래 움직였으면 낭비 행동이다.
    if reachable and steps >= max(6, int(0.35 * max_steps)) and progress <= 0.01 and not reached:
        reward -= 0.26 * step_frac

    return float(reward)
