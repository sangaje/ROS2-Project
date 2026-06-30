#!/usr/bin/env bash
set +e

START_DELAY_SEC="${1:-20}"
ROUNDS="${2:-90}"
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
    echo "V36_NAV2_LIFECYCLE_ABSENT | node=${n}"
    return
  fi
  echo "V36_NAV2_LIFECYCLE_STATE | node=${n} state=${st}"
  if echo "$st" | grep -q "unconfigured"; then
    echo "V36_NAV2_LIFECYCLE_CONFIGURE | node=${n}"
    ros2 lifecycle set "$n" configure || true
    sleep 0.60
    st="$(state_of "$n")"
    echo "V36_NAV2_LIFECYCLE_AFTER_CONFIGURE | node=${n} state=${st}"
  fi
  if echo "$st" | grep -q "inactive"; then
    echo "V36_NAV2_LIFECYCLE_ACTIVATE | node=${n}"
    ros2 lifecycle set "$n" activate || true
    sleep 0.60
    st="$(state_of "$n")"
    echo "V36_NAV2_LIFECYCLE_AFTER_ACTIVATE | node=${n} state=${st}"
  fi
}

has_topic_once() {
  local topic="$1"
  timeout 1.3 ros2 topic echo --once "$topic" >/tmp/v36_${ROS_DOMAIN_ID}_$(echo "$topic" | tr '/' '_').log 2>/dev/null
  return $?
}

has_tf() {
  local parent="$1"
  local child="$2"
  timeout 1.3 ros2 run tf2_ros tf2_echo "$parent" "$child" >/tmp/v36_${ROS_DOMAIN_ID}_tf_${parent}_${child}.log 2>/dev/null
  return $?
}

inputs_ready() {
  local ok=0
  if has_topic_once /odom_nav; then
    echo "V36_INPUT_READY | /odom_nav"
  else
    echo "V36_INPUT_WAIT | /odom_nav missing"
    ok=1
  fi
  if has_topic_once /scan_nav; then
    echo "V36_INPUT_READY | /scan_nav"
  else
    echo "V36_INPUT_WAIT | /scan_nav missing"
    ok=1
  fi
  if has_topic_once /amcl_pose; then
    echo "V36_INPUT_READY | /amcl_pose compatibility pose present"
  else
    echo "V36_INPUT_WAIT | /amcl_pose missing; continuing because v36 owns map->odom separately"
  fi
  if has_tf map odom; then
    echo "V36_TF_READY | map->odom"
  else
    echo "V36_TF_WAIT | map->odom missing"
    ok=1
  fi
  if has_tf odom base_footprint; then
    echo "V36_TF_READY | odom->base_footprint"
  else
    echo "V36_TF_WAIT | odom->base_footprint missing"
    ok=1
  fi
  return $ok
}

echo "V36_REAL_NAV2_LIFECYCLE_GUARD_READY | delay=${START_DELAY_SEC}s rounds=${ROUNDS} period=${PERIOD_SEC}s domain=${ROS_DOMAIN_ID} | amcl_pose_gate=disabled"
sleep "${START_DELAY_SEC}"

for i in $(seq 1 "${ROUNDS}"); do
  echo "V36_NAV2_LIFECYCLE_GUARD_ROUND | ${i}/${ROUNDS} domain=${ROS_DOMAIN_ID}"

  for n in "${LOCALIZATION_NODES[@]}"; do
    activate_node "$n"
  done

  if inputs_ready; then
    echo "V36_NAV_INPUTS_READY | configure/activate navigation stack now"
    for n in "${NAV_NODES[@]}"; do
      activate_node "$n"
    done
  else
    echo "V36_NAV_INPUTS_NOT_READY | not forcing navigation activation this round"
  fi

  for n in "${OPTIONAL_NODES[@]}"; do
    st="$(state_of "$n")"
    [ -z "$st" ] && continue
    echo "V36_NAV2_OPTIONAL_NONBLOCKING | node=${n} state=${st}"
  done

  echo "V36_NAV2_SUMMARY_BEGIN | domain=${ROS_DOMAIN_ID}"
  for n in "${LOCALIZATION_NODES[@]}" "${NAV_NODES[@]}"; do
    echo "${n}: $(state_of "$n" || echo absent)"
  done
  if ros2 action info /navigate_to_pose 2>/dev/null | grep -q "/bt_navigator"; then
    echo "V36_REAL_NAV2_READY | /navigate_to_pose server is /bt_navigator | domain=${ROS_DOMAIN_ID}"
  elif ros2 action list 2>/dev/null | grep -q '^/navigate_to_pose$'; then
    echo "V36_NAV2_ACTION_PRESENT_BUT_NOT_BT | inspect: ros2 action info /navigate_to_pose"
  else
    echo "V36_NAV2_NOT_READY_YET | /navigate_to_pose missing | domain=${ROS_DOMAIN_ID}"
  fi
  echo "V36_NAV2_SUMMARY_END | domain=${ROS_DOMAIN_ID}"

  sleep "${PERIOD_SEC}"
done

echo "V36_REAL_NAV2_LIFECYCLE_GUARD_DONE | domain=${ROS_DOMAIN_ID}"
