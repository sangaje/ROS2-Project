#!/usr/bin/env bash
set -u

main_domain="${1:-24}"
follower_domain="${2:-25}"

if [[ -f /opt/ros/jazzy/setup.bash ]]; then
  set +u
  # shellcheck disable=SC1091
  source /opt/ros/jazzy/setup.bash
  set -u
fi

if [[ -f install/local_setup.bash ]]; then
  set +u
  # shellcheck disable=SC1091
  source install/local_setup.bash
  set -u
elif [[ -f install/setup.bash ]]; then
  set +u
  # shellcheck disable=SC1091
  source install/setup.bash
  set -u
fi

missing=()
for name in ROS_DOMAIN_ID RMW_IMPLEMENTATION; do
  if [[ -z "${!name:-}" ]]; then
    missing+=("${name}")
  fi
done
if [[ "${RMW_IMPLEMENTATION:-}" == "rmw_cyclonedds_cpp" && -z "${CYCLONEDDS_URI:-}" ]]; then
  packaged_cyclone=""
  if command -v ros2 >/dev/null 2>&1; then
    share_dir="$(ros2 pkg prefix --share fleet_bringup 2>/dev/null || true)"
    if [[ -n "${share_dir}" && -f "${share_dir}/config/cyclonedds_fleet.xml" ]]; then
      packaged_cyclone="file://${share_dir}/config/cyclonedds_fleet.xml"
    fi
  fi
  if [[ -n "${packaged_cyclone}" ]]; then
    export CYCLONEDDS_URI="${packaged_cyclone}"
  else
    missing+=("CYCLONEDDS_URI")
  fi
fi
if (( ${#missing[@]} > 0 )); then
  printf 'Missing required shell environment variable(s): %s\n' "${missing[*]}" >&2
  printf 'Source ~/.bashrc before running this diagnostic script.\n' >&2
  exit 2
fi

run_domain() {
  local domain="$1"
  shift
  ROS_DOMAIN_ID="${domain}" "$@"
}

show_domain() {
  local domain="$1"
  local label="$2"
  shift 2

  # The ROS 2 CLI daemon is bound to the domain on which it was started.
  # Restart it before changing domains so domain 25 is not reported as 24.
  ros2 daemon stop >/dev/null 2>&1 || true

  echo
  echo "===== ${label} domain ${domain} ====="
  echo "-- nodes"
  run_domain "${domain}" timeout 4 ros2 node list 2>/dev/null | sort || true
  echo "-- topics"
  run_domain "${domain}" timeout 4 ros2 topic list -t 2>/dev/null | sort || true
  echo "-- critical topic publishers/subscribers"
  for topic in "$@"; do
    echo "[${topic}]"
    run_domain "${domain}" timeout 4 ros2 topic info "${topic}" -v 2>/dev/null || true
  done
}

show_tf() {
  local domain="$1"
  local target="$2"
  local source="$3"

  echo
  echo "===== TF domain ${domain}: ${target} -> ${source} ====="
  run_domain "${domain}" timeout 5 ros2 run tf2_ros tf2_echo "${target}" "${source}" 2>&1 | sed -n '1,12p' || true
}

show_lifecycle() {
  local domain="$1"
  local node="$2"

  echo
  echo "===== Lifecycle domain ${domain}: ${node} ====="
  run_domain "${domain}" timeout 4 ros2 lifecycle get "${node}" 2>&1 || true
}

echo "Fleet runtime diagnosis"
echo "main_domain=${main_domain} follower_domain=${follower_domain}"
echo "RMW_IMPLEMENTATION=${RMW_IMPLEMENTATION}"
echo "ROS_AUTOMATIC_DISCOVERY_RANGE=${ROS_AUTOMATIC_DISCOVERY_RANGE:-}"
echo "ROS_LOCALHOST_ONLY=${ROS_LOCALHOST_ONLY:-}"
echo "CYCLONEDDS_URI=${CYCLONEDDS_URI:-}"

echo
echo "===== local matching processes ====="
ps -eo pid,ppid,cmd | rg -i 'ros2 launch|domain_bridge|rviz2|cartographer|nav2|turtlebot3|robot_state_publisher|ld08|hls_lfcd|leader.launch|follower.launch' || true

show_domain "${main_domain}" "MAIN/LEADER" \
  /scan /scan_nav /odom /map /amcl_pose /tf /leader_pose /plan /burger_pose /burger_plan \
  /fleet/follow_enabled /fleet/follow_command /fleet/coordination_status \
  /fleet/robot_poses /fleet/collision_warning /fleet/hazard_pose /cmd_vel
show_domain "${follower_domain}" "FOLLOWER" \
  /scan /scan_nav /odom /map /shared_map_in /amcl_pose /tf /leader_pose /leader_plan \
  /burger_pose /plan /burger_scan_relay /fleet/follow_enabled \
  /fleet/follow_command /fleet/coordination_status /fleet/robot_poses \
  /fleet/collision_warning /fleet/hazard_pose /cmd_vel

show_lifecycle "${main_domain}" /amcl
show_lifecycle "${follower_domain}" /amcl

show_tf "${main_domain}" map base_footprint
show_tf "${follower_domain}" map base_footprint
show_tf "${main_domain}" map burger/base_footprint

echo
echo "Expected:"
echo "- Main domain must have /map and TF map -> base_footprint after leader.launch.py starts."
echo "- Follower domain must have /shared_map_in from leader /map, local /map from follower_map_relay, and TF map -> base_footprint after follower.launch.py starts."
echo "- Only follower.launch.py should start domain_bridge in normal real runs."
