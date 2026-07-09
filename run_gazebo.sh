#!/usr/bin/env bash
# ============================================================
# run_gazebo.sh
# 터미널 1: Gazebo 시뮬레이션 시작
#
# 실행:
#   bash run_gazebo.sh
#
# 옵션 (환경변수로 오버라이드):
#   SPAWN_POSE=1|2|rand  스폰 위치 (기본=rand)
#   SIM_GUI=true         Gazebo GUI 함께 실행
#   SIM_RVIZ=false       RViz 없이 실행
#   SIM_CARTO=true       Gazebo 터미널에서 Cartographer까지 실행
# ============================================================

set -e

if [[ -n "${ZSH_VERSION:-}" ]]; then
    _TB3_RL_SCRIPT_PATH="${(%):-%x}"
else
    _TB3_RL_SCRIPT_PATH="${BASH_SOURCE[0]:-$0}"
fi
SCRIPT_DIR="$(cd "$(dirname "${_TB3_RL_SCRIPT_PATH}")" && pwd)"

# ── 환경 설정 로드 ──────────────────────────────────────────
source "${SCRIPT_DIR}/setup_env.sh"

cleanup_gazebo_stack() {
    pkill -SIGTERM -f "gz sim"                2>/dev/null || true
    pkill -SIGTERM -f "gz_sim"                2>/dev/null || true
    pkill -SIGTERM -f "rviz2"                 2>/dev/null || true
    pkill -SIGTERM -f "robot_state_publisher" 2>/dev/null || true
    pkill -SIGTERM -f "cartographer_node"     2>/dev/null || true
    pkill -SIGTERM -f "occupancy_grid_node"   2>/dev/null || true
    pkill -SIGTERM -f "parameter_bridge"      2>/dev/null || true
    pkill -SIGTERM -f "ros_gz_sim/create"     2>/dev/null || true
    sleep 2
    pkill -SIGKILL -f "gz sim"                2>/dev/null || true
    pkill -SIGKILL -f "gz_sim"                2>/dev/null || true
    pkill -SIGKILL -f "rviz2"                 2>/dev/null || true
    pkill -SIGKILL -f "robot_state_publisher" 2>/dev/null || true
    pkill -SIGKILL -f "cartographer_node"     2>/dev/null || true
}

on_exit() {
    echo ""
    echo "================================================================"
    echo " [2/2] 종료 및 정리"
    echo "================================================================"
    cleanup_gazebo_stack
    ros2 daemon stop 2>/dev/null || true
    echo "  Gazebo 종료됨."
}

trap on_exit EXIT
trap 'exit 130' INT TERM

# ── 0단계: 잔존 Gazebo/시각화 프로세스 정리 ─────────────────
echo ""
echo "================================================================"
echo " [0/2] 잔존 프로세스 정리"
echo "================================================================"

cleanup_gazebo_stack

# ROS2 데몬 재시작
ros2 daemon stop 2>/dev/null || true
sleep 1
ros2 daemon start 2>/dev/null || true

echo "  완료."
echo ""

# ── 1단계: Gazebo 시뮬레이션 시작 ──────────────────────────
echo "================================================================"
echo " [1/2] Gazebo 시뮬레이션 시작"
echo "================================================================"
echo ""

# run_sim_headless.sh를 백그라운드로 띄운 뒤 readiness를 확인한다.
if [[ -n "${ZSH_VERSION:-}" ]]; then
    zsh "${SCRIPT_DIR}/run_sim_headless.sh" &
else
    bash "${SCRIPT_DIR}/run_sim_headless.sh" &
fi
SIM_PID=$!

echo ""
echo "================================================================"
echo " Gazebo 준비 대기 (/clock + /scan + /odom)"
echo "================================================================"

tb3_rl_wait_for_gazebo_ready "${GAZEBO_READY_TIMEOUT:-120}"
echo "  ✓ Gazebo 준비 완료. 터미널 2에서 실행하세요:"
echo "    cd ${SCRIPT_DIR}"
echo "    bash run_train.sh"
echo ""
echo "  이 터미널은 닫지 마세요. Ctrl+C를 누르면 Gazebo를 정리하고 종료합니다."

wait "${SIM_PID}"
