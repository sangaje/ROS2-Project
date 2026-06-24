import glob
import os
import signal
import subprocess
import time
from typing import Iterable, Optional


def _emit(logger, level: str, msg: str) -> None:
    if logger is None:
        try:
            print(msg, flush=True)
        except Exception:
            pass
        return
    try:
        getattr(logger, level)(msg)
    except Exception:
        try:
            logger.info(msg)
        except Exception:
            pass


def terminate_process_tree(proc: Optional[subprocess.Popen], label: str = "process", logger=None,
                           term_timeout: float = 3.0, kill_timeout: float = 1.5) -> None:
    """Terminate a subprocess and its whole process group.

    ROS launch processes spawn children.  proc.terminate() only signals the
    shell/launch parent and can leave cartographer_node, occupancy_grid_node, or
    ros_gz_bridge alive.  This helper assumes child processes were started with
    start_new_session=True and kills the process group first, then falls back to
    the single process.
    """
    if proc is None:
        return
    try:
        if proc.poll() is not None:
            return
    except Exception:
        return

    _emit(logger, "info", f"PROCESS_GUARD_STOP | label={label} pid={getattr(proc, 'pid', None)}")

    try:
        pgid = os.getpgid(proc.pid)
    except Exception:
        pgid = None

    # Graceful stop first.
    try:
        if pgid is not None:
            os.killpg(pgid, signal.SIGTERM)
        else:
            proc.terminate()
    except Exception:
        try:
            proc.terminate()
        except Exception:
            pass

    try:
        proc.wait(timeout=max(float(term_timeout), 0.1))
        return
    except Exception:
        pass

    # Hard stop if the launch parent did not reap children.
    try:
        if pgid is not None:
            os.killpg(pgid, signal.SIGKILL)
        else:
            proc.kill()
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass

    try:
        proc.wait(timeout=max(float(kill_timeout), 0.1))
    except Exception:
        pass


def _run(cmd: list[str], timeout: float = 2.0) -> None:
    try:
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False, timeout=timeout)
    except Exception:
        pass


def pkill_patterns(patterns: Iterable[str], logger=None, label: str = "runtime", include_kill: bool = True) -> None:
    """Best-effort pkill by pattern with TERM and optional KILL.

    This is intentionally used only for cleanup paths and run scripts.  It does
    not change DDS/RMW settings.  Patterns are grouped here so the launch command
    can clean every stale Gazebo/Cartographer/bridge process before training.
    """
    pats = [str(p).strip() for p in patterns if str(p).strip()]
    if not pats:
        return
    _emit(logger, "warn", f"PROCESS_GUARD_PKILL_START | label={label} patterns={pats}")
    for pat in pats:
        _run(["pkill", "-TERM", "-f", pat], timeout=2.0)
    time.sleep(0.35)
    if include_kill:
        for pat in pats:
            _run(["pkill", "-KILL", "-f", pat], timeout=2.0)
    _emit(logger, "warn", f"PROCESS_GUARD_PKILL_DONE | label={label}")


def clean_fastdds_shm(logger=None) -> int:
    """Remove stale FastDDS/FastRTPS shared-memory lock files.

    Use only before launching ROS/Gazebo processes.  During active ROS execution,
    deleting these files can break live participants.
    """
    patterns = [
        "/dev/shm/fastrtps_*",
        "/dev/shm/sem.fastrtps_*",
        "/dev/shm/fastdds_*",
        "/dev/shm/sem.fastdds_*",
    ]
    count = 0
    for pat in patterns:
        for path in glob.glob(pat):
            try:
                os.remove(path)
                count += 1
            except FileNotFoundError:
                pass
            except Exception:
                pass
    if count:
        _emit(logger, "warn", f"PROCESS_GUARD_SHM_CLEAN | removed={count}")
    return count


def _shm_dir_usage_ratio(path: str = "/dev/shm") -> float:
    """Return used fraction (0..1) of the shared-memory filesystem, or 0.0 on error."""
    try:
        st = os.statvfs(path)
        total = float(st.f_blocks) * float(st.f_frsize)
        free = float(st.f_bavail) * float(st.f_frsize)
        if total <= 0.0:
            return 0.0
        return max(0.0, min(1.0, (total - free) / total))
    except Exception:
        return 0.0


def sweep_stale_fastdds_shm(logger=None, max_age_sec: float = 120.0) -> int:
    """Remove ONLY stale FastDDS/FastRTPS shared-memory segments while ROS is live.

    Unlike ``clean_fastdds_shm`` (which deletes everything and must run only before
    launch), this sweeper is safe to call between episodes during active training.
    It removes a lock/segment file only when BOTH hold:

      1. The file has not been modified for ``max_age_sec`` seconds (no live
         participant has touched it recently), AND
      2. No process currently holds the file open (verified via /proc fd scan when
         available; falls back to mtime-only when /proc is not inspectable).

    This is the targeted fix for the ``open_and_lock_file failed`` /
    ``init_port fastrtps_portNNNN`` crash that accumulates when Cartographer and
    bridges are repeatedly restarted across episodes: dead participants leave
    locked port files behind that are never reclaimed, eventually exhausting the
    shared-memory port space and killing the ROS process.
    """
    patterns = [
        "/dev/shm/fastrtps_*",
        "/dev/shm/sem.fastrtps_*",
        "/dev/shm/fastdds_*",
        "/dev/shm/sem.fastdds_*",
    ]
    now = time.time()
    max_age_sec = max(float(max_age_sec), 5.0)

    # Build a set of inodes that are currently held open by any process so we
    # never remove a segment belonging to a live participant.
    open_inodes: set[int] = set()
    proc_scan_ok = False
    try:
        for fd_dir in glob.glob("/proc/[0-9]*/fd/*"):
            try:
                target = os.readlink(fd_dir)
            except Exception:
                continue
            if "/dev/shm/" not in target:
                continue
            if not (("fastrtps" in target) or ("fastdds" in target)):
                continue
            try:
                open_inodes.add(os.stat(fd_dir).st_ino)
                proc_scan_ok = True
            except Exception:
                pass
    except Exception:
        proc_scan_ok = False

    removed = 0
    for pat in patterns:
        for path in glob.glob(pat):
            try:
                stat = os.stat(path)
            except FileNotFoundError:
                continue
            except Exception:
                continue

            age = now - stat.st_mtime
            if age < max_age_sec:
                continue
            # If we could scan /proc and this inode is open, it is live: skip.
            if proc_scan_ok and stat.st_ino in open_inodes:
                continue
            try:
                os.remove(path)
                removed += 1
            except FileNotFoundError:
                pass
            except Exception:
                pass

    if removed:
        _emit(logger, "warn",
              f"PROCESS_GUARD_SHM_SWEEP | removed_stale={removed} "
              f"proc_scan={'on' if proc_scan_ok else 'off'} max_age={max_age_sec:.0f}s")
    return removed


def ensure_non_shm_fastdds_profile(logger=None, prefer_udp: bool = True) -> bool:
    """Force FastDDS to avoid the shared-memory transport.

    The root cause of repeated ``open_and_lock_file failed`` /
    ``init_port fastrtps_portNNNN`` crashes is the FastDDS **shared-memory
    transport**.  Each participant grabs ``/dev/shm`` port lock files, and when
    SLAM/bridges are restarted every episode the freed ports are not always
    reclaimed, eventually exhausting the SHM port space and killing the next ROS
    process.  Port files are named by port number (not owner PID), so they cannot
    be safely swept while ROS is live.

    The robust fix is to stop using the SHM transport entirely and let FastDDS
    talk over UDP loopback, which has no ``/dev/shm`` footprint.  This writes a
    minimal FastDDS XML profile that disables SHM (UDPv4-only) and points
    ``FASTRTPS_DEFAULT_PROFILES_FILE`` / ``FASTDDS_DEFAULT_PROFILES_FILE`` at it,
    but only if the user has not already configured their own profile.

    Must be called BEFORE ``rclpy.init`` (i.e. before any participant is created).
    Returns True if a profile is now active.
    """
    # Respect an explicit user-provided profile; do not override it.
    for var in ("FASTRTPS_DEFAULT_PROFILES_FILE", "FASTDDS_DEFAULT_PROFILES_FILE"):
        existing = str(os.environ.get(var, "")).strip()
        if existing and os.path.exists(existing):
            _emit(logger, "info",
                  f"PROCESS_GUARD_FASTDDS_PROFILE | using existing {var}={existing}")
            return True

    # Only meaningful for the FastRTPS/FastDDS RMW.
    rmw = str(os.environ.get("RMW_IMPLEMENTATION", "")).strip().lower()
    if rmw and "fastrtps" not in rmw and "fastdds" not in rmw:
        _emit(logger, "info",
              f"PROCESS_GUARD_FASTDDS_PROFILE | RMW={rmw} is not FastDDS; skipping SHM-disable profile")
        return False

    # UDPv4-only profile: disable the auto SHM transport, use a UDP loopback
    # transport, and turn off the builtin transports so SHM is never added back.
    xml = """<?xml version="1.0" encoding="UTF-8"?>
<dds xmlns="http://www.eprosima.com/XMLSchemas/fastRTPS_Profiles">
  <profiles>
    <transport_descriptors>
      <transport_descriptor>
        <transport_id>udp_only_transport</transport_id>
        <type>UDPv4</type>
      </transport_descriptor>
    </transport_descriptors>
    <participant profile_name="udp_only_participant" is_default_profile="true">
      <rtps>
        <userTransports>
          <transport_id>udp_only_transport</transport_id>
        </userTransports>
        <useBuiltinTransports>false</useBuiltinTransports>
      </rtps>
    </participant>
  </profiles>
</dds>
"""
    try:
        path = "/tmp/tb3_rl_fastdds_no_shm.xml"
        with open(path, "w") as fh:
            fh.write(xml)
        os.environ["FASTRTPS_DEFAULT_PROFILES_FILE"] = path
        os.environ["FASTDDS_DEFAULT_PROFILES_FILE"] = path
        _emit(logger, "warn",
              f"PROCESS_GUARD_FASTDDS_PROFILE | SHM transport disabled (UDPv4-only) via {path} | "
              "prevents /dev/shm port exhaustion across SLAM restarts")
        return True
    except Exception as exc:
        _emit(logger, "warn",
              f"PROCESS_GUARD_FASTDDS_PROFILE_FAILED | could not write profile: {exc}")
        return False


def standard_patterns(include_gazebo: bool = False, include_nav2: bool = True) -> list[str]:
    pats = [
        "train_sac",
        "eval_policy",
        "cartographer_node",
        "occupancy_grid_node",
        "cartographer_ros",
        "turtlebot3_cartographer",
        "slam_toolbox",
        "async_slam_toolbox_node",
        "sync_slam_toolbox_node",
        "ros_gz_bridge",
        "parameter_bridge",
        "/model/burger/odometry@nav_msgs/msg/Odometry",
    ]
    if include_nav2:
        pats += [
            "nav2_bringup",
            "navigation_launch.py",
            "controller_server",
            "planner_server",
            "bt_navigator",
            "behavior_server",
            "waypoint_follower",
            "velocity_smoother",
            "lifecycle_manager",
        ]
    if include_gazebo:
        pats += [
            "turtlebot3_gazebo",
            "turtlebot3_house.launch.py",
            "gz sim",
            "gzserver",
            "gzclient",
            "ruby.*gz",
            "ign gazebo",
        ]
    return pats
