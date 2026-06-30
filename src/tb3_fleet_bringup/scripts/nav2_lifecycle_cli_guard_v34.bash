#!/usr/bin/env bash
set +e

START_DELAY_SEC="${1:-24}"
ROUNDS="${2:-60}"
PERIOD_SEC="${3:-2}"

LOCALIZATION_NODES=(/map_server /amcl)
NAV_NODES=(/controller_server /planner_server /behavior_server /bt_navigator)
OPTIONAL_NODES=(/smoother_server /waypoint_follower /velocity_smoother /route_server /collision_monitor /docking_server)

state_of() { ros2 lifecycle get "$1" 2>/dev/null || true; }
activate_node() {
  local n="$1"
  local st
  st="$(state_of "$n")"
  if [ -z "$st" ]; then
    echo "V34_NAV2_LIFECYCLE_ABSENT | node=${n}"
    return
  fi
  echo "V34_NAV2_LIFECYCLE_STATE | node=${n} state=${st}"
  if echo "$st" | grep -q "unconfigured"; then
    echo "V34_NAV2_LIFECYCLE_CONFIGURE | node=${n}"
    ros2 lifecycle set "$n" configure || true
    sleep 0.35
    st="$(state_of "$n")"
  fi
  if echo "$st" | grep -q "inactive"; then
    echo "V34_NAV2_LIFECYCLE_ACTIVATE | node=${n}"
    ros2 lifecycle set "$n" activate || true
    sleep 0.35
  fi
}

wait_for_amcl_pose() {
  timeout 1.2 ros2 topic echo --once /amcl_pose >/tmp/v34_amcl_pose_${ROS_DOMAIN_ID}.log 2>/dev/null
  return $?
}

echo "V34_REAL_NAV2_LIFECYCLE_GUARD_READY | delay=${START_DELAY_SEC}s rounds=${ROUNDS} period=${PERIOD_SEC}s domain=${ROS_DOMAIN_ID}"
sleep "${START_DELAY_SEC}"

for i in $(seq 1 "${ROUNDS}"); do
  echo "V34_NAV2_LIFECYCLE_GUARD_ROUND | ${i}/${ROUNDS} domain=${ROS_DOMAIN_ID}"

  for n in "${LOCALIZATION_NODES[@]}"; do
    activate_node "$n"
  done

  if wait_for_amcl_pose; then
    echo "V34_AMCL_POSE_READY | domain=${ROS_DOMAIN_ID}"
    for n in "${NAV_NODES[@]}"; do
      activate_node "$n"
    done
  else
    echo "V34_WAIT_AMCL_POSE | localization not publishing /amcl_pose yet; not forcing planner activation this round"
  fi

  # Optional nodes may exist in Jazzy. They must not block /navigate_to_pose.
  for n in "${OPTIONAL_NODES[@]}"; do
    st="$(state_of "$n")"
    [ -z "$st" ] && continue
    if echo "$st" | grep -q "unconfigured\|inactive"; then
      echo "V34_NAV2_OPTIONAL_SKIP_NONBLOCKING | node=${n} state=${st}"
    fi
  done

  echo "V34_NAV2_SUMMARY_BEGIN | domain=${ROS_DOMAIN_ID}"
  for n in "${LOCALIZATION_NODES[@]}" "${NAV_NODES[@]}"; do
    echo "${n}: $(state_of "$n" || echo absent)"
  done
  if ros2 action info /navigate_to_pose 2>/dev/null | grep -q "/bt_navigator"; then
    echo "V34_REAL_NAV2_READY | /navigate_to_pose server is /bt_navigator | domain=${ROS_DOMAIN_ID}"
  elif ros2 action list 2>/dev/null | grep -q '^/navigate_to_pose$'; then
    echo "V34_NAV2_ACTION_PRESENT_BUT_SERVER_CHECK_FAILED | inspect: ros2 action info /navigate_to_pose"
  else
    echo "V34_NAV2_NOT_READY_YET | /navigate_to_pose missing | domain=${ROS_DOMAIN_ID}"
  fi
  echo "V34_NAV2_SUMMARY_END | domain=${ROS_DOMAIN_ID}"

  sleep "${PERIOD_SEC}"
done

echo "V34_REAL_NAV2_LIFECYCLE_GUARD_DONE | domain=${ROS_DOMAIN_ID}"
