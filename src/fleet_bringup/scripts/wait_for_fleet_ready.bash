#!/usr/bin/env bash
set -u

kind="${1:-tf}"
target="${2:-map}"
source_frame="${3:-base_footprint}"
timeout_sec="${4:-120}"

start_sec="$(date +%s)"

echo "[wait_for_fleet_ready] kind=${kind} target=${target} source=${source_frame} timeout=${timeout_sec}s domain=${ROS_DOMAIN_ID:-unset}"

while true; do
  now_sec="$(date +%s)"
  elapsed=$((now_sec - start_sec))
  if (( timeout_sec > 0 && elapsed >= timeout_sec )); then
    echo "[wait_for_fleet_ready] timeout after ${elapsed}s: ${kind} ${target} ${source_frame}" >&2
    exit 1
  fi

  case "${kind}" in
    scan)
      if timeout 4 ros2 topic echo /scan --qos-reliability best_effort --once >/dev/null 2>&1; then
        echo "[wait_for_fleet_ready] /scan is publishing"
        exit 0
      fi
      echo "[wait_for_fleet_ready] waiting for /scan (${elapsed}s)"
      ;;
    tf)
      if timeout 4 ros2 run tf2_ros tf2_echo "${target}" "${source_frame}" 2>&1 | grep -q 'Translation:'; then
        echo "[wait_for_fleet_ready] TF ready: ${target} -> ${source_frame}"
        exit 0
      fi
      echo "[wait_for_fleet_ready] waiting for TF ${target} -> ${source_frame} (${elapsed}s)"
      ;;
    *)
      echo "[wait_for_fleet_ready] unknown kind: ${kind}" >&2
      exit 2
      ;;
  esac

  sleep 2
done
