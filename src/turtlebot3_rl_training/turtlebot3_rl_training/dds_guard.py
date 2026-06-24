"""ROS 2 DDS transport guard for long-running TurtleBot3 RL jobs.

Target failure:
    [RTPS_TRANSPORT_SHM Error] Failed init_port fastrtps_port7000: open_and_lock_file failed

The training loop restarts Cartographer frequently.  Fast DDS shared-memory
transport can leave stale fastrtps_port* lock files under /dev/shm after repeated
kill/relaunch cycles.  This module avoids the failure without requiring CycloneDDS
installation:

Default policy in v121:
  1. Do NOT force rmw_cyclonedds_cpp.
  2. If the user's shell already forces rmw_cyclonedds_cpp but it is not installed,
     unset RMW_IMPLEMENTATION before rclpy.init so ROS falls back to the installed default.
  3. Install a Fast DDS UDP-only XML participant profile so rmw_fastrtps_cpp does
     not use shared memory.
  4. Remove stale /dev/shm FastDDS lock files using Python glob, not shell glob.
  5. Pass the same environment to internally spawned Cartographer/bridge/Nav2 processes.

Environment switches:
  TB3_RL_DDS_GUARD=0                 disable this module completely
  TB3_RL_FORCE_CYCLONEDDS=1          select CycloneDDS only when it is installed
  TB3_RL_FASTDDS_DISABLE_SHM=0       do not write/use the UDP-only FastDDS XML
  TB3_RL_CLEAN_FASTDDS_SHM=0         do not remove stale /dev/shm/fastrtps_* files
  TB3_RL_DDS_GUARD_VERBOSE=1         print verbose diagnostics
"""

from __future__ import annotations

import glob
import os
import subprocess
import time
from pathlib import Path
from typing import Mapping, Optional

_FASTDDS_UDP_ONLY_XML = """<?xml version=\"1.0\" encoding=\"UTF-8\" ?>
<profiles xmlns=\"http://www.eprosima.com/XMLSchemas/fastRTPS_Profiles\">
  <transport_descriptors>
    <transport_descriptor>
      <transport_id>tb3_rl_udp_transport</transport_id>
      <type>UDPv4</type>
    </transport_descriptor>
  </transport_descriptors>
  <participant profile_name=\"tb3_rl_no_shm_participant\" is_default_profile=\"true\">
    <rtps>
      <userTransports>
        <transport_id>tb3_rl_udp_transport</transport_id>
      </userTransports>
      <useBuiltinTransports>false</useBuiltinTransports>
    </rtps>
  </participant>
</profiles>
"""

_CONFIGURED = False
_LAST_INFO: dict[str, str] = {}


def _env_bool(name: str, default: bool = True, env: Optional[Mapping[str, str]] = None) -> bool:
    source = os.environ if env is None else env
    raw = str(source.get(name, "1" if default else "0")).strip().lower()
    return raw not in {"0", "false", "no", "off", "disable", "disabled"}


def _log(message: str, logger=None, level: str = "info") -> None:
    try:
        if logger is not None:
            fn = getattr(logger, level, None) or getattr(logger, "info", None)
            if fn is not None:
                fn(message)
                return
    except Exception:
        pass
    print(message, flush=True)


def _ros_pkg_exists(pkg: str) -> bool:
    """Check ROS package availability without inheriting a possibly broken RMW."""
    try:
        env = os.environ.copy()
        env.pop("RMW_IMPLEMENTATION", None)
        result = subprocess.run(
            ["ros2", "pkg", "prefix", pkg],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=2.5,
            env=env,
            check=False,
        )
        return result.returncode == 0
    except Exception:
        return False


def _write_fastdds_udp_only_profile() -> str:
    path = Path(os.environ.get("TB3_RL_FASTDDS_PROFILE_FILE", "/tmp/tb3_rl_fastdds_no_shm.xml"))
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        old = path.read_text() if path.exists() else ""
    except Exception:
        old = ""
    if old != _FASTDDS_UDP_ONLY_XML:
        path.write_text(_FASTDDS_UDP_ONLY_XML)
    return str(path)


def cleanup_fastdds_shm(logger=None, *, aggressive: bool = False) -> int:
    """Best-effort removal of stale Fast DDS shared-memory lock files."""
    if not _env_bool("TB3_RL_CLEAN_FASTDDS_SHM", True):
        return 0

    patterns = (
        "/dev/shm/fastrtps_*",
        "/dev/shm/sem.fastrtps_*",
        "/dev/shm/fastdds_*",
        "/dev/shm/sem.fastdds_*",
    )
    removed = 0
    for pattern in patterns:
        for item in glob.glob(pattern):
            try:
                p = Path(item)
                if p.is_dir():
                    continue
                if not aggressive:
                    try:
                        age = time.time() - p.stat().st_mtime
                        if age < 0.25:
                            continue
                    except Exception:
                        pass
                p.unlink(missing_ok=True)
                removed += 1
            except Exception:
                pass

    if removed:
        _log(f"DDS_GUARD | removed stale FastDDS SHM files count={removed}", logger, "warn")
    return removed


def configure_ros_transport_environment(process_label: str = "tb3_rl", logger=None) -> dict[str, str]:
    """Configure SHM-safe ROS transport before rclpy.init and subprocess launch."""
    global _CONFIGURED, _LAST_INFO
    if not _env_bool("TB3_RL_DDS_GUARD", True):
        _LAST_INFO = {"enabled": "0"}
        return dict(_LAST_INFO)

    if _CONFIGURED:
        return dict(_LAST_INFO)

    info: dict[str, str] = {"enabled": "1", "process": str(process_label)}

    rmw_before = str(os.environ.get("RMW_IMPLEMENTATION", "")).strip()
    cyclone_installed = _ros_pkg_exists("rmw_cyclonedds_cpp")

    # Critical v121 fix: never leave a non-installed Cyclone RMW in the environment.
    if rmw_before == "rmw_cyclonedds_cpp" and not cyclone_installed:
        os.environ.pop("RMW_IMPLEMENTATION", None)
        info["rmw_unset_missing"] = "rmw_cyclonedds_cpp"
        rmw_before = ""

    # Optional only.  Default is FastDDS no-SHM because it works without extra packages.
    if _env_bool("TB3_RL_FORCE_CYCLONEDDS", False):
        if cyclone_installed:
            os.environ["RMW_IMPLEMENTATION"] = "rmw_cyclonedds_cpp"
            info["rmw_selected"] = "rmw_cyclonedds_cpp"
        else:
            os.environ.pop("RMW_IMPLEMENTATION", None)
            info["rmw_selected"] = "system_default"
            info["cyclonedds_missing"] = "1"
    else:
        info["rmw_selected"] = os.environ.get("RMW_IMPLEMENTATION", "system_default") or "system_default"

    if _env_bool("TB3_RL_FASTDDS_DISABLE_SHM", True):
        try:
            xml = _write_fastdds_udp_only_profile()
            os.environ["FASTRTPS_DEFAULT_PROFILES_FILE"] = xml
            # Let rmw_fastrtps honor XML-defined participant transport settings.
            os.environ.setdefault("RMW_FASTRTPS_USE_QOS_FROM_XML", "1")
            info["fastdds_profile"] = xml
        except Exception as exc:
            info["fastdds_profile_error"] = str(exc)

    cleanup_fastdds_shm(logger=logger, aggressive=False)

    _CONFIGURED = True
    _LAST_INFO = info

    verbose = _env_bool("TB3_RL_DDS_GUARD_VERBOSE", False)
    if verbose or info.get("rmw_unset_missing") or info.get("cyclonedds_missing"):
        _log(
            "DDS_GUARD | "
            f"process={process_label} rmw={os.environ.get('RMW_IMPLEMENTATION', 'system_default')} "
            f"fastdds_profile={os.environ.get('FASTRTPS_DEFAULT_PROFILES_FILE', '')} "
            f"unset_missing={info.get('rmw_unset_missing', '0')} "
            f"cyclonedds_missing={info.get('cyclonedds_missing', '0')}",
            logger,
            "warn" if (info.get("rmw_unset_missing") or info.get("cyclonedds_missing")) else "info",
        )
    return dict(info)


def ros_subprocess_env(extra: Optional[Mapping[str, str]] = None) -> dict[str, str]:
    """Return DDS-safe environment for internally launched ROS subprocesses."""
    configure_ros_transport_environment(process_label="subprocess")
    env = os.environ.copy()
    if extra:
        env.update({str(k): str(v) for k, v in extra.items()})
    return env
