#!/usr/bin/env bash
# ============================================================
# run_all.sh
# TurtleBot3 SAC 전체 실행 (시뮬 + 학습 한 번에)
#
# 실행:
#   ./run_all.sh
#
# 옵션 (환경변수로 오버라이드):
#   SPAWN_POSE=1|2|rand  스폰 위치 (기본=rand)
#   SIM_GUI=true         Gazebo GUI 함께 실행
#   TRAIN_SCRIPT=run_train.sh  사용할 학습 스크립트 (기본=run_train.sh)
# ============================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── 사용할 학습 스크립트 ────────────────────────────────────
TRAIN_SCRIPT="${TRAIN_SCRIPT:-run_train.sh}"

# ── ROS2 소싱 ───────────────────────────────────────────────
source /opt/ros/jazzy/setup.bash
source "${SCRIPT_DIR}/install/setup.bash"

# ── 공통 환경 변수 ──────────────────────────────────────────
export TURTLEBOT3_MODEL=burger
export ROS_DOMAIN_ID=22
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export TB3_RL_DISABLE_SHM_TRANSPORT=1

# ── FastDDS no-SHM 프로필 생성 (시뮬 + 학습 양쪽 모두 동일 transport 사용) ──
# train_sac 은 Python 코드에서 이 파일을 자동 생성하지만,
# 시뮬(parameter_bridge/gz sim)도 같은 프로필을 써야 transport 불일치가 없다.
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
echo "FastDDS no-SHM 프로필 생성: ${FASTDDS_PROFILE}"

# ── 0단계: 잔존 프로세스 전부 정리 ─────────────────────────
echo ""
echo "================================================================"
echo " [0/3] 잔존 프로세스 정리"
echo "================================================================"

pkill -SIGTERM -f "gz sim"                2>/dev/null || true
pkill -SIGTERM -f "gz_sim"                2>/dev/null || true
pkill -SIGTERM -f "rviz2"                 2>/dev/null || true
pkill -SIGTERM -f "robot_state_publisher" 2>/dev/null || true
pkill -SIGTERM -f "cartographer_node"     2>/dev/null || true
pkill -SIGTERM -f "occupancy_grid_node"   2>/dev/null || true
pkill -SIGTERM -f "parameter_bridge"      2>/dev/null || true
pkill -SIGTERM -f "ros_gz_sim/create"     2>/dev/null || true
pkill -SIGTERM -f "train_sac"             2>/dev/null || true

sleep 2

pkill -SIGKILL -f "gz sim"                2>/dev/null || true
pkill -SIGKILL -f "gz_sim"                2>/dev/null || true
pkill -SIGKILL -f "rviz2"                 2>/dev/null || true
pkill -SIGKILL -f "robot_state_publisher" 2>/dev/null || true
pkill -SIGKILL -f "cartographer_node"     2>/dev/null || true
pkill -SIGKILL -f "train_sac"             2>/dev/null || true

# ROS2 데몬 재시작
ros2 daemon stop 2>/dev/null || true
sleep 1
ros2 daemon start 2>/dev/null || true

echo "  완료."
echo ""

# ── 1단계: 시뮬(Gazebo + RViz)을 별도 gnome-terminal 창에서 실행 ──
echo "================================================================"
echo " [1/3] 시뮬레이션 시작 (별도 창)"
echo "================================================================"

# gnome-terminal은 GNOME 서버 프로세스로 창을 열므로 부모 환경변수가 상속 안 됨.
# RMW_FASTRTPS_DEFAULT_PROFILES_FILE 포함 모든 키를 명시적으로 전달.
gnome-terminal \
    --title="TB3 Sim (Gazebo + RViz)" \
    -- env \
        ROS_DOMAIN_ID=22 \
        TURTLEBOT3_MODEL=burger \
        RMW_IMPLEMENTATION=rmw_fastrtps_cpp \
        TB3_RL_DISABLE_SHM_TRANSPORT=1 \
        RMW_FASTRTPS_DEFAULT_PROFILES_FILE="${FASTDDS_PROFILE}" \
        SPAWN_POSE="${SPAWN_POSE:-rand}" \
        SIM_GUI="${SIM_GUI:-false}" \
    bash -c "
        cd '${SCRIPT_DIR}'
        bash run_sim_headless.sh
        echo ''
        echo '[시뮬 종료됨. 이 창은 닫아도 됩니다.]'
        read -r
    "

# ── 2단계: /scan + /odom 토픽 확인 ─────────────────────────
echo ""
echo "================================================================"
echo " [2/3] Gazebo 준비 대기 (/scan + /odom 토픽 확인)"
echo "================================================================"

MAX_WAIT=60
WAITED=0

while ! ros2 topic list 2>/dev/null | grep -q "^/scan$"; do
    sleep 1; WAITED=$((WAITED + 1))
    printf "\r  %ds 경과... /scan 대기 중 (최대 %ds)" "${WAITED}" "${MAX_WAIT}"
    if [[ "${WAITED}" -ge "${MAX_WAIT}" ]]; then
        echo ""; echo "[오류] /scan 토픽이 ${MAX_WAIT}초 내에 나타나지 않았습니다."; exit 1
    fi
done
echo ""; echo "  /scan OK"

WAITED=0
while ! ros2 topic list 2>/dev/null | grep -q "^/odom$"; do
    sleep 1; WAITED=$((WAITED + 1))
    printf "\r  %ds 경과... /odom 대기 중 (최대 %ds)" "${WAITED}" "${MAX_WAIT}"
    if [[ "${WAITED}" -ge "${MAX_WAIT}" ]]; then
        echo ""; echo "[오류] /odom 토픽이 ${MAX_WAIT}초 내에 나타나지 않았습니다."; exit 1
    fi
done
echo ""; echo "  /odom OK"

echo "  DDS 안정화 대기 (5초)..."
sleep 5
echo "  준비 완료 — 학습 시작합니다."
echo ""

# ── 3단계: 학습 실행 (현재 터미널) ─────────────────────────
echo "================================================================"
echo " [3/3] SAC 학습 시작 (${TRAIN_SCRIPT})"
echo "================================================================"
echo ""

exec bash "${SCRIPT_DIR}/${TRAIN_SCRIPT}"

# ── 0단계: 잔존 프로세스 전부 정리 ─────────────────────────
echo "================================================================"
echo " [0/3] 잔존 프로세스 정리"
echo "================================================================"

pkill -SIGTERM -f "gz sim"           2>/dev/null || true
pkill -SIGTERM -f "gz_sim"           2>/dev/null || true
pkill -SIGTERM -f "rviz2"            2>/dev/null || true
pkill -SIGTERM -f "robot_state_publisher" 2>/dev/null || true
pkill -SIGTERM -f "cartographer_node"     2>/dev/null || true
pkill -SIGTERM -f "occupancy_grid_node"   2>/dev/null || true
pkill -SIGTERM -f "parameter_bridge"      2>/dev/null || true
pkill -SIGTERM -f "ros_gz_sim/create"     2>/dev/null || true
pkill -SIGTERM -f "train_sac"             2>/dev/null || true

sleep 2

# SIGKILL로 확실히 종료
pkill -SIGKILL -f "gz sim"           2>/dev/null || true
pkill -SIGKILL -f "gz_sim"           2>/dev/null || true
pkill -SIGKILL -f "rviz2"            2>/dev/null || true
pkill -SIGKILL -f "robot_state_publisher" 2>/dev/null || true
pkill -SIGKILL -f "cartographer_node"     2>/dev/null || true
pkill -SIGKILL -f "train_sac"             2>/dev/null || true

# ROS2 데몬 재시작 (stale discovery 정리)
ros2 daemon stop 2>/dev/null || true
sleep 1
ros2 daemon start 2>/dev/null || true

echo "  완료."
echo ""

# ── 1단계: 시뮬(Gazebo + RViz)을 별도 gnome-terminal 창에서 실행 ──
echo "================================================================"
echo " [1/3] 시뮬레이션 시작 (별도 창)"
echo "================================================================"

# gnome-terminal은 GNOME 서버 프로세스로 창을 열기 때문에
# 부모 쉘 환경변수가 자동 상속되지 않는다.
# env KEY=VAL 로 명시적으로 전달한다.
gnome-terminal \
    --title="TB3 Sim (Gazebo + RViz)" \
    -- env \
        ROS_DOMAIN_ID=22 \
        TURTLEBOT3_MODEL=burger \
        RMW_IMPLEMENTATION=rmw_fastrtps_cpp \
        TB3_RL_DISABLE_SHM_TRANSPORT=1 \
        SPAWN_POSE="${SPAWN_POSE:-rand}" \
        SIM_GUI="${SIM_GUI:-false}" \
    bash -c "
        cd '${SCRIPT_DIR}'
        bash run_sim_headless.sh
        echo ''
        echo '[시뮬 종료됨. 이 창은 닫아도 됩니다.]'
        read -r
    "

# ── 2단계: /scan 토픽이 올라올 때까지 대기 (최대 30초) ─────
echo ""
echo "================================================================"
echo " [2/3] Gazebo 준비 대기 (/scan + /odom 토픽 확인)"
echo "================================================================"

MAX_WAIT=60
WAITED=0

# /scan 대기
while ! ros2 topic list 2>/dev/null | grep -q "^/scan$"; do
    sleep 1
    WAITED=$((WAITED + 1))
    printf "\r  %ds 경과... /scan 대기 중 (최대 %ds)" "${WAITED}" "${MAX_WAIT}"
    if [[ "${WAITED}" -ge "${MAX_WAIT}" ]]; then
        echo ""; echo "[오류] /scan 토픽이 ${MAX_WAIT}초 내에 나타나지 않았습니다."; exit 1
    fi
done
echo ""; echo "  /scan OK"

# /odom 대기
WAITED=0
while ! ros2 topic list 2>/dev/null | grep -q "^/odom$"; do
    sleep 1
    WAITED=$((WAITED + 1))
    printf "\r  %ds 경과... /odom 대기 중 (최대 %ds)" "${WAITED}" "${MAX_WAIT}"
    if [[ "${WAITED}" -ge "${MAX_WAIT}" ]]; then
        echo ""; echo "[오류] /odom 토픽이 ${MAX_WAIT}초 내에 나타나지 않았습니다."; exit 1
    fi
done
echo ""; echo "  /odom OK"

# parameter_bridge lazy connection + DDS discovery 안정화
echo "  브릿지/DDS 안정화 대기 (5초)..."
sleep 5
echo "  준비 완료 — 학습 시작합니다."
echo ""

# ── 3단계: 학습 실행 (현재 터미널) ─────────────────────────
echo "================================================================"
echo " [3/3] SAC 학습 시작 (v131)"
echo "================================================================"
echo ""

exec bash "${SCRIPT_DIR}/run_train_v131_env_sanity.sh"
