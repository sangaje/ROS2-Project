#!/usr/bin/env bash
set +e

START_DELAY_SEC="${1:-16}"
ROUNDS="${2:-40}"
PERIOD_SEC="${3:-2}"

# Required for /navigate_to_pose to appear and execute. Map/amcl are included
# so this same guard can recover both localization and navigation bringup.
NODES=(
  /map_server
  /amcl
  /controller_server
  /planner_server
  /behavior_server
  /bt_navigator
)

OPTIONAL_NODES=(
  /smoother_server
  /waypoint_follower
  /velocity_smoother
  /route_server
  /collision_monitor
  /docking_server
)

echo "V31_NAV2_LIFECYCLE_CLI_GUARD_READY | delay=${START_DELAY_SEC}s rounds=${ROUNDS} period=${PERIOD_SEC}s domain=${ROS_DOMAIN_ID}"
sleep "${START_DELAY_SEC}"

for i in $(seq 1 "${ROUNDS}"); do
  echo "V31_NAV2_LIFECYCLE_CLI_GUARD_ROUND | ${i}/${ROUNDS} domain=${ROS_DOMAIN_ID}"

  for n in "${NODES[@]}"; do
    state="$(ros2 lifecycle get "$n" 2>/dev/null || true)"
    if [ -z "$state" ]; then
      echo "V31_NAV2_LIFECYCLE_ABSENT | node=${n}"
      continue
    fi

    echo "V31_NAV2_LIFECYCLE_STATE | node=${n} state=${state}"

    if echo "$state" | grep -q "unconfigured"; then
      echo "V31_NAV2_LIFECYCLE_CONFIGURE | node=${n}"
      ros2 lifecycle set "$n" configure || true
      sleep 0.25
      echo "V31_NAV2_LIFECYCLE_ACTIVATE_AFTER_CONFIGURE | node=${n}"
      ros2 lifecycle set "$n" activate || true
      sleep 0.25
    elif echo "$state" | grep -q "inactive"; then
      echo "V31_NAV2_LIFECYCLE_ACTIVATE | node=${n}"
      ros2 lifecycle set "$n" activate || true
      sleep 0.25
    fi
  done

  # Optional Jazzy extras. These should not block /navigate_to_pose.
  for n in "${OPTIONAL_NODES[@]}"; do
    state="$(ros2 lifecycle get "$n" 2>/dev/null || true)"
    if [ -z "$state" ]; then
      continue
    fi
    if echo "$state" | grep -q "unconfigured"; then
      echo "V31_NAV2_LIFECYCLE_OPTIONAL_CONFIGURE | node=${n}"
      ros2 lifecycle set "$n" configure || true
      sleep 0.10
      ros2 lifecycle set "$n" activate || true
      sleep 0.10
    elif echo "$state" | grep -q "inactive"; then
      echo "V31_NAV2_LIFECYCLE_OPTIONAL_ACTIVATE | node=${n}"
      ros2 lifecycle set "$n" activate || true
      sleep 0.10
    fi
  done

  echo "V31_NAV2_LIFECYCLE_SUMMARY_BEGIN | domain=${ROS_DOMAIN_ID}"
  for n in "${NODES[@]}"; do
    echo "${n}: $(ros2 lifecycle get "$n" 2>/dev/null || echo absent)"
  done
  if ros2 action list 2>/dev/null | grep -q '^/navigate_to_pose$'; then
    echo "V31_NAV2_READY | /navigate_to_pose available | domain=${ROS_DOMAIN_ID}"
  else
    echo "V31_NAV2_NOT_READY_YET | /navigate_to_pose missing | domain=${ROS_DOMAIN_ID}"
  fi
  echo "V31_NAV2_LIFECYCLE_SUMMARY_END | domain=${ROS_DOMAIN_ID}"

  sleep "${PERIOD_SEC}"
done

echo "V31_NAV2_LIFECYCLE_CLI_GUARD_DONE | domain=${ROS_DOMAIN_ID}"
