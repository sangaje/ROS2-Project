#!/usr/bin/env bash
set -euo pipefail

# Run ROS 2 RViz with a clean dynamic-linker path.  This avoids Snap/VSCode
# environments leaking /snap/core20 libraries into rviz2, which causes:
#   libpthread.so.0: undefined symbol: __libc_pthread_init, version GLIBC_PRIVATE

unset SNAP
unset SNAP_NAME
unset SNAP_ARCH
unset SNAP_INSTANCE_NAME
unset SNAP_REVISION
unset SNAP_COOKIE
unset SNAP_LIBRARY_PATH
unset SNAP_DATA
unset SNAP_COMMON
unset SNAP_USER_DATA
unset SNAP_USER_COMMON
unset LD_LIBRARY_PATH
unset LD_PRELOAD
unset LD_AUDIT
unset QT_PLUGIN_PATH
unset QML2_IMPORT_PATH
unset GIO_MODULE_DIR
unset GDK_PIXBUF_MODULE_FILE
unset GTK_PATH

export PATH="/opt/ros/jazzy/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"

exec /lib64/ld-linux-x86-64.so.2 \
  --library-path /opt/ros/jazzy/lib:/opt/ros/jazzy/opt/rviz_ogre_vendor/lib:/lib/x86_64-linux-gnu:/usr/lib/x86_64-linux-gnu \
  /opt/ros/jazzy/lib/rviz2/rviz2 "$@"
