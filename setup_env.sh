#!/usr/bin/env bash
# ============================================================
# setup_env.sh
# 공통 환경 변수 설정 (터미널 1, 2 모두에서 실행)
# ============================================================

if [[ -n "${ZSH_VERSION:-}" ]]; then
    _TB3_RL_SETUP_PATH="${(%):-%x}"
else
    _TB3_RL_SETUP_PATH="${BASH_SOURCE[0]:-$0}"
fi
SCRIPT_DIR="$(cd "$(dirname "${_TB3_RL_SETUP_PATH}")" && pwd)"

# ── Python venv ─────────────────────────────────────────────
if [[ -f "${HOME}/venv/ros2/bin/activate" ]]; then
    source "${HOME}/venv/ros2/bin/activate"
fi

# ── ROS2 소싱 ───────────────────────────────────────────────
if [[ -n "${ZSH_VERSION:-}" ]]; then
    source /opt/ros/jazzy/setup.zsh
    source "${SCRIPT_DIR}/install/setup.zsh"
else
    source /opt/ros/jazzy/setup.bash
    source "${SCRIPT_DIR}/install/setup.bash"
fi

# During active training/debugging, prefer the workspace source tree over the
# previously installed Python package. Otherwise script changes under src/ are
# silently ignored until the next colcon build.
export PYTHONPATH="${SCRIPT_DIR}/src/turtlebot3_rl_training:${PYTHONPATH:-}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/tb3_rl_matplotlib}"

# ── 공통 환경 변수 ──────────────────────────────────────────
export TURTLEBOT3_MODEL=burger
export TB3_RL_ENTITY_NAME=burger
# ROS_DOMAIN_ID는 여기서 건드리지 않는다 -- 사용자 ~/.zshrc에서 설정한 값을 그대로 쓴다.
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export TB3_RL_DISABLE_SHM_TRANSPORT=1

# ── FastDDS no-SHM 프로필 생성 ──────────────────────────────
FASTDDS_PROFILE=/tmp/tb3_rl_fastdds_no_shm.xml
cat > "${FASTDDS_PROFILE}" << 'XMLEOF'
<?xml version="1.0" encoding="UTF-8"?>
<profiles xmlns="http://www.eprosima.com/XMLSchemas/fastRTPS_Profiles">
  <transport_descriptors>
    <transport_descriptor>
      <transport_id>udpv4</transport_id>
      <type>UDPv4</type>
    </transport_descriptor>
  </transport_descriptors>
  <participant profile_name="default_profile" is_default_profile="true">
    <rtps>
      <userTransports>
        <transport_id>udpv4</transport_id>
      </userTransports>
      <useBuiltinTransports>false</useBuiltinTransports>
    </rtps>
  </participant>
</profiles>
XMLEOF
export RMW_FASTRTPS_DEFAULT_PROFILES_FILE="${FASTDDS_PROFILE}"
export FASTRTPS_DEFAULT_PROFILES_FILE="${FASTDDS_PROFILE}"
export FASTDDS_DEFAULT_PROFILES_FILE="${FASTDDS_PROFILE}"
echo "FastDDS no-SHM 프로필 생성: ${FASTDDS_PROFILE}"

# ── Gazebo readiness helpers ─────────────────────────────────
# These functions are intentionally kept in the shared environment file so both
# terminal-1 simulation scripts and terminal-2 training scripts use the same
# readiness contract.
tb3_rl_wait_for_topic() {
    local topic="$1"
    local max_wait="${2:-120}"
    local waited=0

    echo "  ${topic} 토픽 대기 중..."
    while ! ros2 topic list 2>/dev/null | grep -qx "${topic}"; do
        sleep 1
        waited=$((waited + 1))
        printf "\r  %ds 경과... (최대 %ds)" "${waited}" "${max_wait}"
        if [[ "${waited}" -ge "${max_wait}" ]]; then
            echo ""
            echo "[오류] ${topic} 토픽이 ${max_wait}초 내에 나타나지 않았습니다."
            echo "       터미널 1에서 먼저 실행하세요:"
            echo "       cd ${SCRIPT_DIR}"
            echo "       bash run_gazebo.sh"
            echo ""
            echo "       현재 보이는 ROS 토픽:"
            ros2 topic list 2>/dev/null | sed 's/^/         /' || true
            return 1
        fi
    done
    echo ""
    echo "  ✓ ${topic} OK"
}

tb3_rl_wait_for_gazebo_ready() {
    local max_wait="${1:-120}"
    tb3_rl_wait_for_topic "/clock" "${max_wait}"
    tb3_rl_wait_for_topic "/scan" "${max_wait}"
    tb3_rl_wait_for_topic "/odom" "${max_wait}"
    export TB3_RL_GAZEBO_READY_CHECKED=1
}
