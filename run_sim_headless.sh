#!/usr/bin/env bash
# ============================================================
# run_sim_headless.sh
# TurtleBot3 Burger — Gazebo 헤드리스(GUI 없음) + RViz2
# ============================================================
#
# 사용법:
#   chmod +x run_sim_headless.sh
#   ./run_sim_headless.sh
#
# 옵션 (환경변수로 오버라이드):
#   SIM_GUI=true       ./run_sim_headless.sh  → Gazebo GUI 함께 실행
#   SIM_CARTO=true     ./run_sim_headless.sh  → Cartographer SLAM 포함 실행
#   SIM_RVIZ=false     ./run_sim_headless.sh  → RViz 없이 실행
#   SIM_WORLD=/path/to/world.sdf             → 학습 월드 교체
#
# 빌드 먼저 필요한 경우:
#   colcon build --packages-select turtlebot3_rl_training
# ============================================================

set -e

if [[ -n "${ZSH_VERSION:-}" ]]; then
    _TB3_RL_SCRIPT_PATH="${(%):-%x}"
else
    _TB3_RL_SCRIPT_PATH="${BASH_SOURCE[0]:-$0}"
fi
SCRIPT_DIR="$(cd "$(dirname "${_TB3_RL_SCRIPT_PATH}")" && pwd)"

# ── 공통 환경 설정 로드 ────────────────────────────────────
source "${SCRIPT_DIR}/setup_env.sh"

# ── 스폰 pose 목록 (train_sac --reset-pose-list와 동일) ─────
# pose 1: -2.80, 0.96
# pose 2:  5.00, 0.86
# SPAWN_POSE=rand(기본) / 1 / 2 로 선택
SPAWN_POSE="${SPAWN_POSE:-rand}"

if [[ "${SPAWN_POSE}" == "rand" ]]; then
    SPAWN_POSE=$(( (RANDOM % 2) + 1 ))
fi

case "${SPAWN_POSE}" in
    2)
        SIM_X_POSE="${SIM_X_POSE:-5.00}"
        SIM_Y_POSE="${SIM_Y_POSE:-0.86}"
        ;;
    *)  # 1 또는 그 외
        SIM_X_POSE="${SIM_X_POSE:--2.80}"
        SIM_Y_POSE="${SIM_Y_POSE:-0.96}"
        ;;
esac

# ── 런치 인수 (환경변수로 오버라이드 가능) ──────────────────
SIM_GUI="${SIM_GUI:-false}"
SIM_CARTO="${SIM_CARTO:-false}"
SIM_RVIZ="${SIM_RVIZ:-true}"
SIM_WORLD="${SIM_WORLD:-${SCRIPT_DIR}/src/turtlebot3_rl_training/world/random_maze_empty.sdf}"
SIM_LAUNCH="${SCRIPT_DIR}/src/turtlebot3_rl_training/launch/sim_headless_rviz.launch.py"
export TB3_RL_PREFER_SOURCE_LAUNCH="${TB3_RL_PREFER_SOURCE_LAUNCH:-1}"

echo "============================================================"
echo " TurtleBot3 Sim (headless Gazebo + RViz2)"
echo "------------------------------------------------------------"
echo "  TURTLEBOT3_MODEL : ${TURTLEBOT3_MODEL}"
echo "  ROS_DOMAIN_ID    : ${ROS_DOMAIN_ID}"
echo "  gui              : ${SIM_GUI}"
echo "  start_rviz       : ${SIM_RVIZ}"
echo "  start_cartographer: ${SIM_CARTO}"
echo "  world            : ${SIM_WORLD}"
echo "  spawn pose       : pose${SPAWN_POSE}  x=${SIM_X_POSE}, y=${SIM_Y_POSE}"
echo "============================================================"

ros2 launch "${SIM_LAUNCH}" \
    gui:="${SIM_GUI}" \
    start_rviz:="${SIM_RVIZ}" \
    start_cartographer:="${SIM_CARTO}" \
    world:="${SIM_WORLD}" \
    x_pose:="${SIM_X_POSE}" \
    y_pose:="${SIM_Y_POSE}"
