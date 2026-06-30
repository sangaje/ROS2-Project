#!/usr/bin/env bash
set -euo pipefail

# Gazebo GUI can crash if a shell inherited Snap's core20 libraries:
#   /snap/core20/.../libpthread.so.0: undefined symbol: __libc_pthread_init
# Keep ROS/Gazebo library paths, but remove Snap paths before exec.
clean_path_list() {
  local value="${1:-}"
  local out=""
  local part
  IFS=':' read -ra parts <<< "$value"
  for part in "${parts[@]}"; do
    [[ -z "$part" ]] && continue
    case "$part" in
      /snap/*|*/snap/*) continue ;;
    esac
    if [[ -z "$out" ]]; then
      out="$part"
    else
      out="$out:$part"
    fi
  done
  printf '%s' "$out"
}

export LD_LIBRARY_PATH="$(clean_path_list "${LD_LIBRARY_PATH:-}")"
export PATH="$(clean_path_list "${PATH:-}")"
unset SNAP SNAP_NAME SNAP_ARCH SNAP_REVISION SNAP_INSTANCE_NAME SNAP_REAL_HOME
unset GTK_PATH GIO_EXTRA_MODULES

mkdir -p "${HOME}/.gz" "${HOME}/.ros/log"

if [[ "${TB3_FLEET_SKIP_GZ_CLEANUP:-0}" != "1" ]]; then
  # Stale Gazebo servers keep publishing old /clock and odometry through
  # ros_gz_bridge, which makes Cartographer receive time going backwards.
  pkill -f "gz sim" >/dev/null 2>&1 || true
  pkill -f "ign gazebo" >/dev/null 2>&1 || true
  sleep 0.5
fi

exec gz sim "$@"
