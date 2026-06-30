#!/usr/bin/env bash
set -euo pipefail

# Snap-installed VSCode injects core20 library paths into child terminals.
# RViz then loads /snap/core20/.../libpthread.so.0 and crashes with:
#   undefined symbol: __libc_pthread_init, version GLIBC_PRIVATE
# Strip Snap/VSCode GUI paths before execing the real ROS RViz binary.
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
export XDG_DATA_DIRS="$(clean_path_list "${XDG_DATA_DIRS:-}")"

unset SNAP SNAP_NAME SNAP_ARCH SNAP_REVISION SNAP_INSTANCE_NAME SNAP_REAL_HOME SNAP_LIBRARY_PATH
unset GTK_PATH GTK_EXE_PREFIX GIO_MODULE_DIR GIO_EXTRA_MODULES
unset GDK_PIXBUF_MODULE_FILE GDK_PIXBUF_MODULEDIR
unset QT_PLUGIN_PATH QML2_IMPORT_PATH

exec /opt/ros/jazzy/lib/rviz2/rviz2 "$@"
