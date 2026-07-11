#!/usr/bin/env bash
# ============================================================
# run_train_v132_clean.sh
# TurtleBot3 SAC v132: clean training (per-episode SLAM reset)
# ============================================================
#
# 핵심 변경 (v131 대비):
#   - Cartographer는 학습 프로세스가 시작/관리
#   - 매 에피소드 SLAM reset/restart로 /map, TF, confidence 기준 초기화
#   - pose-frame=map, safety-boundary-frame=odom
#   - strict SLAM gate는 완화하되 reset 자체는 필수
#   - 백그라운드 맵 미러 비활성화
#   - 스폰 위치: 집 내부 안전 후보 중 랜덤 선택
#
# 사전 조건:
#   터미널 1에서 먼저 실행:
#     cd ~/Desktop/ROS2_Project && bash run_gazebo.sh
#   (Gazebo만 시작; Cartographer는 이 스크립트가 관리)

set -e

if [[ -n "${ZSH_VERSION:-}" ]]; then
    _TB3_RL_SCRIPT_PATH="${(%):-%x}"
else
    _TB3_RL_SCRIPT_PATH="${BASH_SOURCE[0]:-$0}"
fi
SCRIPT_DIR="$(cd "$(dirname "${_TB3_RL_SCRIPT_PATH}")" && pwd)"
cd "${SCRIPT_DIR}"
source "${SCRIPT_DIR}/setup_env.sh"

# ===== ROS / DDS =====
# ROS_DOMAIN_ID는 여기서 설정하지 않는다 -- 사용자 ~/.zshrc 값을 그대로 쓴다.
export TURTLEBOT3_MODEL=burger
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export TB3_RL_MODEL_ODOM_TOPIC=/model/burger/odometry

export TB3_RL_DISABLE_SHM_TRANSPORT=1
export TB3_RL_DISABLE_BUFFER_GUARD=0

# ===== v132: Cartographer는 학습 프로세스가 관리 =====
# reset 때 SLAM map/process를 초기화해서 episode 간 stale map/TF를 끊는다.
export TB3_RL_CARTOGRAPHER_RESTART_DELAY_SEC=0.5

# reset이 어디서 오래 걸리는지 단계별(pose_reset/slam_reset/post_reset_stabilize/
# map_reset/ready_gate/obs_info) ms로 stdout에 RESET_PROFILE 로그를 찍는다.
export TB3_RL_RESET_PROFILER=1

# ===== LiDAR 설정 =====
export TB3_RL_MAX_SCAN_AGE_SEC=0.50
export TB3_RL_LIDAR_CANONICAL_FRONT_ZERO=1
export TB3_RL_LIDAR_FRONT_INDEX=0
export TB3_RL_LIDAR_ANGLE_OFFSET_DEG=0
export TB3_RL_LIDAR_FLIP_LR=0
export TB3_RL_LIDAR_UNIFORM_ANGLE_RESAMPLE=1
export TB3_RL_LIDAR_MEDIAN_KERNEL=3
export TB3_RL_LIDAR_LOWPASS_KERNEL=5
export TB3_RL_LIDAR_OBSTACLE_MARGIN_M=0.08

# 라이더 빔 하나만 obstacle을 찍어도 confidence map이 그 뒤로 계속 칠해지지 않도록,
# 그 hit 지점 주변 차단 반경을 기본 2셀(~10cm)에서 3셀(~15cm)로 넓힌다.
# 기존에 이미 칠해진 confidence를 지우는 것이 아니라, 이후 업데이트에서 그 지점을
# 넘어가는 새 painting만 막는다 (occlusion_threshold, FOV, range는 그대로).
export TB3_RL_CONFIDENCE_LIDAR_OCCLUSION_RADIUS_CELLS=3

# v133: control-dt=0.20 keeps SAC decisions/gradient updates cheap, but that
# alone means confidence is only painted every 0.2s of sim time (sparser than
# the original 0.1s). Split every control_dt advance into 2 physics sub-ticks
# of 0.1s each, painting confidence after every sub-tick, while the agent
# still only sees/acts once per 0.2s (step_count, train-freq-steps etc. are
# unaffected). Collision/fallen are also checked after the intermediate
# sub-tick so a hit mid-macro-step is not missed.
export TB3_RL_MAP_SUBSTEPS_PER_ACTION=2
export TB3_RL_DEBUG_MAP_UPDATE_COUNT=0

# RViz debug visibility: publish policy scan / map overlay diagnostics at a
# throttled rate (every Nth step) instead of every step, so RViz has something
# to show without reintroducing a per-step publish bottleneck. Override
# TB3_RL_*_EVERY_N=1000000 (or blank the topic vars) from the calling shell to
# go back to throughput-only mode for long unattended runs.
export TB3_RL_POLICY_SCAN_TOPIC=
export TB3_RL_POLICY_SCAN_60_TOPIC=
export TB3_RL_POLICY_SCAN_PUBLISH_EVERY_N=1000000
export TB3_RL_POLICY_SCAN_MARKER_TOPIC=
export TB3_RL_RAW_SCAN_MARKER_TOPIC=
export TB3_RL_RAW_SCAN_MARKER_UNCORRECTED=0

# ===== priority: disabled completely for this training run =====
export TB3_RL_FORCE_NO_PRIORITY=1
export TB3_RL_NO_PRIORITY_MODEL_INPUT=1
export TB3_RL_PRIORITY_CLUSTER_SPAWN_INTERVAL_STEPS=100
export TB3_RL_PRIORITY_MAX_SEED_POINTS=6
export TB3_RL_PRIORITY_SPAWN_MIN_RANGE_M=0.90
export TB3_RL_PRIORITY_SPAWN_MAX_RANGE_M=2.40
export TB3_RL_PRIORITY_SPAWN_IN_CURRENT_FOV=0
export TB3_RL_PRIORITY_BIRTH_DELTA=6.0

# ===== logging (조용하게) =====
# EPISODE_RESET_REASON, POST_RESET_STABILIZE*, "Unsafe terminal state detected"
# 등 나머지 개별 QUIET_* 플래그로 안 잡히는 INFO/WARN 로그를 전부 죽이고,
# tqdm 스타일 "[SAC] ..." 진행 블록(plain print라 이 설정과 무관)만 남긴다.
export TB3_RL_QUIET_ALL_ROS_LOGS="${TB3_RL_QUIET_ALL_ROS_LOGS:-1}"
export TB3_RL_QUIET_MAP_LOGS=1
export TB3_RL_QUIET_STARTUP_LOGS=1
export TB3_RL_QUIET_VELOCITY_SAFETY_LOGS=1
export TB3_RL_QUIET_PRIORITY_LOGS=1
export TB3_RL_QUIET_RESET_LOGS=1
export TB3_RL_CONFIDENCE_PUBLISH_DEBUG_EVERY_N=0
export TB3_RL_CONFIDENCE_DEBUG_EVERY_N=0
export TB3_RL_CONFIDENCE_ORIGIN_TOPIC=
export TB3_RL_DEBUG_OVERLAY_EVERY_N=1000000
export TB3_RL_G_ONLY_DEBUG=1

# ===== confidence pose = odom 기반 (TF 대기 불필요) =====
export TB3_RL_CONFIDENCE_UNIFY_WITH_TF_CUBE=1
export TB3_RL_CONFIDENCE_UNIFY_STRICT=0
export TB3_RL_CLEAR_CONFIDENCE_ON_SLAM_OCCUPIED=0
export TB3_RL_CLEAR_CONFIDENCE_ON_SLAM_CANVAS_CHANGE=0
export TB3_RL_CONFIDENCE_OCCUPIED_CONFIRM_STEPS=3
export TB3_RL_CONFIDENCE_DECAY_NEAR_OBSTACLE_SCALE=0.00
export TB3_RL_CONFIDENCE_OBSTACLE_RING_RADIUS=5
export TB3_RL_CONFIDENCE_OBSTACLE_FLOOR_RATIO=1.00
export TB3_RL_CONFIDENCE_SINGLE_CUBE_MARKER=1
export TB3_RL_CONFIDENCE_CUBE_FRAME=base_footprint
export TB3_RL_CONFIDENCE_PREFER_TF_BUFFER=1
export TB3_RL_USE_MANUAL_TF_CACHE_FOR_CONFIDENCE=1
export TB3_RL_CONFIDENCE_TF_BUFFER_FALLBACK=1
export TB3_RL_CONFIDENCE_CUBE_TF_BUFFER_FALLBACK=1
export TB3_RL_MANUAL_TF_MAX_AGE_SEC=60.0
export TB3_RL_CONFIDENCE_TF_HOLD_SEC=2.0
export TB3_RL_TF_CUBE_POSE_WARN=0
export TB3_RL_CONFIDENCE_ANCHOR_RESYNC=1
export TB3_RL_CONFIDENCE_ANCHOR_RESYNC_TOL_M=0.05
export TB3_RL_FORCE_MAP_PUBLISH_EVERY_UPDATE=0
# TF pose lookup 실패 시 confidence 업데이트를 멈추지 않고
# 현재 exploration map frame 기준 pose fallback을 허용한다.
export TB3_RL_CONFIDENCE_ALLOW_ODOM_FALLBACK_WHEN_TF_MISSING=1

# ===== odom TF fallback (Cartographer 없어도 동작) =====
export TB3_RL_ODOM_TF_FALLBACK=0
export TB3_RL_MANUAL_TF_CACHE=1

# ===== profiler =====
# 50Hz/속도 최적화를 어디부터 손댈지 정하기 전에, 스텝 하나에서 시간이
# action_execute(cmd_vel 전송)/wait_sync(Gazebo multi_step 응답 대기)/map_update
# (confidence+SLAM 반영)/reward_compute/obs_build 중 어디에 제일 많이 쓰이는지
# 먼저 로그로 확인한다. STEP_PROFILE 로그를 몇 분 보고 나서 최적화 대상을 정한다.
export TB3_RL_STEP_PROFILER="${TB3_RL_STEP_PROFILER:-0}"
# fps dropped once training passed learning-starts and gradient updates began
# firing (train-freq-steps=24). Time each SAC.train() call directly to see if
# gradient updates are really the new bottleneck before touching control_dt.
export TB3_RL_TIME_GRAD_UPDATE="${TB3_RL_TIME_GRAD_UPDATE:-0}"
export TB3_RL_STEP_PROFILER_EVERY_N=100
export TB3_RL_STEP_PROFILER_SLOW_MS=500
# action_execute stayed ~38ms even though gz_step/wait_time_adv/wait_sensor
# (the actual Gazebo multi_step wait) measured ~0ms -- so the cost is inside
# _execute_velocity_action's own Python logic (safety-shield distance/raycast
# checks), not the physics wait. cProfile one call every N steps to find
# exactly which function is hot.
export TB3_RL_STEP_CPROFILE=0
export TB3_RL_STEP_CPROFILE_EVERY_N=150
# action_execute is now cheap (~8ms); map_update (~38ms) is the remaining
# biggest cost. Profile _update_exploration_map() itself in isolation now
# that the redundant background timer is gone.
export TB3_RL_MAP_CPROFILE=0
export TB3_RL_MAP_CPROFILE_EVERY_N=150
export TB3_RL_FAST_NO_PRIORITY_STATS="${TB3_RL_FAST_NO_PRIORITY_STATS:-1}"
export TB3_RL_WORLD_STEP_CONTRACT_WAIT_SEC="${TB3_RL_WORLD_STEP_CONTRACT_WAIT_SEC:-0.25}"

# ===== world reset (episode-level) =====
# 정체/충돌 루프에서 물리 상태까지 확실히 초기화하려면 world reset을 함께 건다.
export TB3_RL_WORLD_RESET_ON_EPISODE=1
export TB3_RL_WORLD_RESET_MODE=all
export TB3_RL_WORLD_RESET_TIMEOUT_SEC=3.0
# world reset(all)이 15~20개 random obstacle까지 다시 초기화하는 동안 아직 안정화가
# 안 된 상태에서 곧바로 SetEntityPose(burger)를 부르면 "Failed to reset pose for all
# discovered Gazebo robot candidates" 에러가 난다. world reset 직후에만 약간 더 기다린다.
export TB3_RL_POST_WORLD_RESET_SETTLE_SEC=0.30
export TB3_RL_RANDOM_OBSTACLES=1
export TB3_RL_RANDOM_OBSTACLE_COUNT_MIN=15
export TB3_RL_RANDOM_OBSTACLE_COUNT_MAX=30
export TB3_RL_RANDOM_OBSTACLE_ROBOT_CLEARANCE_M=0.85
export TB3_RL_RANDOM_OBSTACLE_PAIR_CLEARANCE_M=0.34
export TB3_RL_RANDOM_OBSTACLE_NOISE_SIGMA_M=0.35
# Coverage/confidence 부족은 종료 조건이 아니라 reward/diagnostic 신호로만 사용한다.
# v132: SLAM/confidence/coverage 업데이트가 같이 멈춘 episode tail은 penalty 없이 reset한다.

# ===== 체크포인트 경로 =====
# No-priority/map64 mode changes observation shape, so it must not load map32 checkpoints.
# v132+gSDE: gSDE changes the actor's log_std parameter shape, so it is NOT compatible with
# older non-SDE checkpoints (loading would hit a state_dict shape mismatch). Point at a fresh
# "_gsde" model/log dir instead of touching the existing non-SDE checkpoints in the old dir --
# the auto-checkpoint scan below will find nothing there and start from scratch automatically.
export MODEL_DIR="rl_models/pure_velocity_sac_map64_lidar60_h8_deltatcn_domain22_nopriority_gsde_v022_dt02_b128_obs63"
export LOG_DIR="rl_logs/pure_velocity_sac_map64_lidar60_h8_deltatcn_domain22_nopriority_gsde_v022_dt02_b128_obs63"
mkdir -p "${MODEL_DIR}" "${LOG_DIR}"

LOAD_MODEL="$(python3 - <<'PY'
import re
import zipfile
import os
import json
from pathlib import Path

model_dir = Path(os.environ.get("MODEL_DIR", "rl_models/pure_velocity_sac_map64_lidar60_h8_deltatcn_domain22_nopriority"))

def step(path: Path) -> int:
    # SB3 stores the authoritative counter inside every model archive.  Main
    # and emergency filenames do not contain a step, so comparing their mtime
    # (an epoch-sized integer) against checkpoint step numbers can select the
    # wrong model.  Rank every archive on the same num_timesteps scale.
    try:
        with zipfile.ZipFile(path) as archive:
            data = json.loads(archive.read("data"))
        return int(data.get("num_timesteps", -1))
    except Exception:
        pass
    match = re.search(r"_(\d+)_steps(?:\.zip)?$", path.stem)
    if match:
        return int(match.group(1))
    return -1

def valid(path: Path) -> bool:
    try:
        if not path.is_file() or path.stat().st_size <= 0:
            return False
        with zipfile.ZipFile(path) as archive:
            return archive.testzip() is None
    except Exception:
        return False

candidates = []
for pattern in (
    "sac_turtlebot3_burger_checkpoint_*_steps.zip",
    "sac_turtlebot3_burger.zip",
    "sac_turtlebot3_burger_emergency.zip",
):
    candidates.extend(model_dir.glob(pattern))
candidates = [path for path in candidates if valid(path)]
if candidates:
    print(max(candidates, key=lambda path: (step(path), path.stat().st_mtime)))
PY
)"

if [[ -n "${LOAD_MODEL}" ]]; then
    echo "[v132] Using valid checkpoint: ${LOAD_MODEL}"
else
    echo "[v132] No valid checkpoint found in ${MODEL_DIR}. Starting from scratch."
fi

LOAD_ARG=""
if [[ -n "${LOAD_MODEL}" ]]; then
    LOAD_ARG="--load-model ${LOAD_MODEL}"
fi

# ============================================================
# Gazebo 준비 확인
# ============================================================
echo ""
echo "================================================================"
echo " [v132] Gazebo 준비 대기 (/clock + /scan + /odom)"
echo "================================================================"
if [[ "${TB3_RL_GAZEBO_READY_CHECKED:-0}" == "1" ]]; then
    echo "  ✓ 상위 스크립트에서 이미 확인됨"
else
    tb3_rl_wait_for_gazebo_ready "${GAZEBO_READY_TIMEOUT:-120}"
    echo "  DDS 안정화 대기 (3초)..."
    sleep 3
fi

# ============================================================
# 학습 실행
# ============================================================
# Keep policy and executed /cmd_vel speeds SLAM-friendly so Cartographer has
# time to integrate scans without tearing the map.
RESET_CRITICS_ARG=()
if [[ "${TB3_RL_RESET_CRITICS_ON_LOAD:-auto}" == "1" ]]; then
    echo "  ✓ TB3_RL_RESET_CRITICS_ON_LOAD=1: critic reset 강제 활성화"
    RESET_CRITICS_ARG=(--sac-reset-critics)
fi
# In auto mode train_sac.py compares the semantics version stored inside the
# checkpoint.  Legacy checkpoints get exactly one actor-only migration; a
# successfully saved migrated checkpoint will not reset critics again even if
# it is copied to another directory.

REPLAY_RESUME_ARG=(--resume-replay-buffer)
if [[ "${TB3_RL_RESUME_REPLAY_BUFFER:-1}" == "0" ]]; then
    echo "  ✓ TB3_RL_RESUME_REPLAY_BUFFER=0: actor만 재사용하고 replay는 새로 시작"
    REPLAY_RESUME_ARG=(--no-resume-replay-buffer)
fi

# Fast confidence-visible mode with a fixed sim-time control contract:
# one RL step advances about 0.20s of simulation time, so simulation time runs
# at 5 policy/confidence updates per second (relaxed from the earlier 10Hz/0.10s
# floor to cut wall-clock cost per unit of simulated time -- gradient updates
# and other fixed per-step costs now amortize over 2x as much sim-time).
# Gazebo remains uncapped in wall time; if the machine cannot deliver
# 25 wall-steps/sec, throughput drops, but the policy still sees 5 steps per
# simulated second. Every *_steps flag below that represents a real-time
# duration has been HALVED versus the 0.10s version (steps * control_dt must
# stay constant, and control_dt doubled) so it still means the same
# wall/sim-clock duration.
python3 -m turtlebot3_rl_training.train_sac \
    --no-auto-start-gazebo \
    --timesteps 300000000 \
    --learning-starts 1500 \
    --buffer-size 200000 \
    --batch-size 128 \
    --sac-learning-rate 0.0001 \
    --sac-gamma 0.97 \
    --sac-use-sde \
    --sac-sde-sample-freq 4 \
    --sac-target-entropy -2.0 \
    --sac-reset-ent-coef 0.20 \
    --sac-min-ent-coef 0.03 \
    --sac-max-ent-coef 0.20 \
    --warmup-action-steps 15000 \
    --warmup-action-zero-linear-prob 0.05 \
    --warmup-action-random-prob 0.45 \
    --warmup-action-noise-prob 0.45 \
    --warmup-action-noise-std 0.45 \
    --reward-gamma 0.97 \
    --control-dt 0.20 \
    --physics-step-size 0.005 \
    --world-step-target-fraction 1.0 \
    --world-step-wait-timeout-sec 0.10 \
    --world-step-sensor-timeout-sec 0.05 \
    --no-world-step-auto-disable \
    --max-episode-steps 5000 \
    --entity-name burger \
    --set-pose-service /world/default/set_pose \
    --world-control-service /world/default/control \
    --cmd-vel-topic /cmd_vel \
    \
    --action-mode velocity \
    --num-lidar-bins 60 \
    --use-map-cnn \
    --map-obs-size 64 \
    --map-obs-size-m 6.0 \
    --use-temporal-cnn \
    --temporal-history-len 8 \
    --temporal-features-dim 64 \
    --cnn-features-dim 48 \
    --vector-features-dim 96 \
    --combined-features-dim 192 \
    \
    --max-linear-speed 0.22 \
    --max-angular-speed 0.70 \
    --velocity-command-linear-limit 0.22 \
    --velocity-command-angular-limit 0.70 \
    --linear-deadband 0.02 \
    --angular-deadband 0.04 \
    --velocity-forward-assist-mps 0.00 \
    --action-smoothing-alpha 0.65 \
    \
    --max-scan-age-sec 0.50 \
    \
    --velocity-safety-backup \
    --velocity-safety-backup-steps 4 \
    --velocity-safety-cooldown-steps 1 \
    --velocity-safety-penalty 2.50 \
    --no-velocity-safety-terminal \
    --velocity-safety-terminal-distance-m 0.00 \
    --velocity-safety-terminal-penalty 0.0 \
    --velocity-safety-terminal-forward-min 0.02 \
    --velocity-safety-trigger-distance-m 0.22 \
    --velocity-safety-stop-distance-m 0.22 \
    --velocity-safety-slow-distance-m 0.40 \
    --velocity-safety-slowdown \
    --velocity-safety-slow-min-scale 1.00 \
    --velocity-safety-slow-penalty 1.00 \
    --velocity-safety-slow-speed-power 1.20 \
    --velocity-safety-slow-danger-power 1.00 \
    \
    --front-fov-deg 60.0 \
    --front-angle-sigma-deg 20.0 \
    --confidence-max-range 2.0 \
    --seen-confidence-floor 70.0 \
    --confidence-decay-per-step 0.0 \
    --confidence-reward-weight 1.8 \
    --slam-map-update-reward \
    --slam-map-update-reward-weight 0.60 \
    --slam-map-update-reward-norm-cells 900.0 \
    --slam-map-update-reward-cap 0.60 \
    --coverage-stall-terminal \
    --coverage-stall-start-steps 150 \
    --coverage-stall-window-steps 90 \
    --coverage-stall-min-slam-new-cells 1 \
    --coverage-stall-min-confidence-updated-cells 8 \
    --coverage-stall-min-coverage-delta 0.0001 \
    --coverage-stall-required-consecutive-windows 2 \
    --coverage-stall-terminal-penalty 0.0 \
    --disable-wall-proximity-penalty \
    --disable-priority-map \
    --no-enable-corridor-priority-reward \
    --corridor-priority-reward-weight 0.0 \
    --priority-recompute-interval 2 \
    --priority-clear-fov-deg 60.0 \
    --priority-clear-max-range-m 2.0 \
    --priority-clear-min-weight 0.05 \
    --no-priority-stuck-restart \
    --priority-stuck-restart-steps 100 \
    \
    --lidar-empty-restart \
    --lidar-empty-timeout-sec 2.5 \
    --lidar-empty-grace-sec 1.0 \
    --lidar-empty-min-valid-beams 2 \
    \
    --collision-threshold 0.14 \
    --gap-occupied-threshold 50.0 \
    --no-velocity-spin-breaker \
    --velocity-spin-breaker-steps 5 \
    --velocity-spin-breaker-angular-ratio 0.45 \
    --velocity-spin-breaker-forward-mps 0.08 \
    --velocity-spin-breaker-angular-scale 0.25 \
    --velocity-spin-breaker-min-clearance-m 0.30 \
    --shake-restart \
    --shake-restart-steps 2 \
    --shake-tilt-threshold 0.18 \
    --shake-angular-xy-threshold 0.45 \
    --shake-linear-z-threshold 0.05 \
    --shake-ground-min-z -0.03 \
    --shake-ground-max-z 0.16 \
    --no-shake-yaw-wobble \
    --shake-spin-stall-restart-steps 6 \
    --shake-restart-penalty 20.0 \
    --restart-on-collision \
    --no-terminate-on-out-of-bounds \
    --safety-boundary-radius-m 8.0 \
    \
    --slam-backend cartographer \
    --slam-map-topic /map \
    --map-frame map \
    --pose-frame map \
    --safety-boundary-frame odom \
    --auto-start-slam \
    --wait-slam-map \
    --reset-slam-on-reset \
    --restart-slam-on-reset \
    --reset-slam-every-n-episodes 1 \
    --reset-tf-buffer-on-reset \
    --no-strict-slam-map-required \
    --slam-reset-timeout 20.0 \
    --strict-slam-map-wait-timeout-sec 10.0 \
    --strict-slam-map-min-known-cells 20 \
    --strict-slam-map-min-known-ratio 0.001 \
    --post-reset-ready-gate \
    --post-reset-ready-timeout-sec 5.0 \
    --post-reset-ready-min-known-ratio 0.001 \
    --post-reset-ready-min-known-cells 10 \
    --post-reset-ready-min-lidar-beams 20 \
    --no-post-reset-ready-require-priority \
    --post-reset-stabilize-sec "${TB3_RL_POST_RESET_STABILIZE_SEC:-0.5}" \
    \
    --reset-pose-mode list \
    --reset-pose-list="-5.30,1.25;-5.30,1.58;-5.30,1.92;-5.30,2.25;-4.47,1.82;-3.63,1.38;-2.80,0.95;-2.60,0.85;-2.40,0.75;-2.20,0.65;-1.67,0.85;-1.13,1.05;-0.60,1.25;-0.13,1.35;0.33,1.45;0.80,1.55;0.80,1.67;0.80,1.78;0.80,1.90;1.03,1.68;1.27,1.47;1.50,1.25;2.03,1.25;2.57,1.25;3.10,1.25;3.27,1.25;3.43,1.25;3.60,1.25;-5.30,-1.25;-5.30,-1.58;-5.30,-1.92;-5.30,-2.25;-4.47,-1.82;-3.63,-1.38;-2.80,-0.95;-2.60,-0.85;-2.40,-0.75;-2.20,-0.65;-1.67,-0.85;-1.13,-1.05;-0.60,-1.25;-0.13,-1.35;0.33,-1.45;0.80,-1.55;0.80,-1.67;0.80,-1.78;0.80,-1.90;1.03,-1.68;1.27,-1.47;1.50,-1.25;2.03,-1.25;2.57,-1.25;3.10,-1.25;3.27,-1.25;3.43,-1.25;3.60,-1.25" \
    --reset-pose-max-attempts 20 \
    --reset-pose-min-clearance-m 0.30 \
    \
    --rl-map-topic "" \
    --rl-confidence-topic "" \
    --rl-priority-topic "" \
    --rl-filtered-slam-topic "" \
    --waypoint-marker-topic "" \
    --map-publish-every-n "${TB3_RL_MAP_PUBLISH_EVERY_N:-1}" \
    --map-live-update-period-sec 0 \
    --map-keepalive-period-sec "${TB3_RL_MAP_KEEPALIVE_SEC:-1.0}" \
    \
    --model-dir "${MODEL_DIR}" \
    --log-dir "${LOG_DIR}" \
    ${LOAD_ARG} \
    "${RESET_CRITICS_ARG[@]}" \
    "${REPLAY_RESUME_ARG[@]}" \
    --save-replay-buffer \
    --train-freq-steps 4 \
    --gradient-steps 1 \
    --checkpoint-freq 50000 \
    \
    --show-training-progress \
    --progress-style block \
    --progress-print-freq 10 \
    --progress-window 10 \
    --progress-csv "${LOG_DIR}/metrics_compact_v132.csv" \
    --progress-csv-flush-every 10 \
    \
    --sac-verbose 0 \
    --debug-print-freq 0 \
    --no-check-env
