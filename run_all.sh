#!/usr/bin/env bash
# ============================================================
# run_all.sh
# 호환용 래퍼: Gazebo는 새 터미널, 학습은 현재 터미널에서 실행
#
# 권장 실행 방식:
#   터미널 1: bash run_gazebo.sh
#   터미널 2: bash run_train.sh
#
# 이 스크립트는 위 두 터미널 흐름을 자동으로 열 수 있을 때만 사용한다.
# ============================================================

set -e

if [[ -n "${ZSH_VERSION:-}" ]]; then
    _TB3_RL_SCRIPT_PATH="${(%):-%x}"
else
    _TB3_RL_SCRIPT_PATH="${BASH_SOURCE[0]:-$0}"
fi
SCRIPT_DIR="$(cd "$(dirname "${_TB3_RL_SCRIPT_PATH}")" && pwd)"
TRAIN_SCRIPT="${TRAIN_SCRIPT:-run_train_v131_env_sanity.sh}"

source "${SCRIPT_DIR}/setup_env.sh"

echo ""
echo "================================================================"
echo " Two-terminal TurtleBot3 RL workflow"
echo "================================================================"
echo "  터미널 1: bash run_gazebo.sh"
echo "  터미널 2: bash run_train.sh"
echo ""

if ! command -v gnome-terminal >/dev/null 2>&1; then
    echo "[오류] gnome-terminal을 찾을 수 없습니다."
    echo "       아래처럼 직접 두 터미널에서 실행하세요:"
    echo "       1) cd ${SCRIPT_DIR} && bash run_gazebo.sh"
    echo "       2) cd ${SCRIPT_DIR} && bash run_train.sh"
    exit 1
fi

echo " [1/2] Gazebo 터미널 열기"
gnome-terminal \
    --title="TB3 Gazebo" \
    -- env \
        ROS_DOMAIN_ID="${ROS_DOMAIN_ID}" \
        TURTLEBOT3_MODEL="${TURTLEBOT3_MODEL}" \
        RMW_IMPLEMENTATION="${RMW_IMPLEMENTATION}" \
        TB3_RL_DISABLE_SHM_TRANSPORT="${TB3_RL_DISABLE_SHM_TRANSPORT}" \
        RMW_FASTRTPS_DEFAULT_PROFILES_FILE="${RMW_FASTRTPS_DEFAULT_PROFILES_FILE}" \
        FASTRTPS_DEFAULT_PROFILES_FILE="${FASTRTPS_DEFAULT_PROFILES_FILE}" \
        FASTDDS_DEFAULT_PROFILES_FILE="${FASTDDS_DEFAULT_PROFILES_FILE}" \
        SPAWN_POSE="${SPAWN_POSE:-rand}" \
        SIM_GUI="${SIM_GUI:-false}" \
        SIM_RVIZ="${SIM_RVIZ:-true}" \
        SIM_CARTO="${SIM_CARTO:-false}" \
    bash -c "cd '${SCRIPT_DIR}' && bash run_gazebo.sh"

echo ""
echo " [2/2] Gazebo 준비 후 학습 시작"
echo "       현재 터미널에서 ${TRAIN_SCRIPT} 실행"
echo ""

export TRAIN_SCRIPT
exec bash "${SCRIPT_DIR}/run_train.sh"
