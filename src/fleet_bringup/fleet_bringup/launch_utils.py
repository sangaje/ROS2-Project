import os
from typing import Dict, List
from pathlib import Path
from urllib.parse import unquote, urlparse
from xml.etree import ElementTree as ET

from launch.actions import OpaqueFunction


REQUIRED_DDS_ENVIRONMENT = (
    'ROS_DOMAIN_ID',
    'RMW_IMPLEMENTATION',
)
CYCLONEDDS_REQUIRED_ENVIRONMENT = (
    'CYCLONEDDS_URI',
)
_CYCLONEDDS_CONFIG_NS = 'https://cdds.io/config'


def _perform_if_needed(value, context):
    if value is None:
        return None
    if hasattr(value, 'perform'):
        return value.perform(context)
    return str(value)


def _missing_required_environment() -> List[str]:
    missing = [name for name in REQUIRED_DDS_ENVIRONMENT if not os.environ.get(name, '').strip()]
    rmw = os.environ.get('RMW_IMPLEMENTATION', '').strip()
    if rmw == 'rmw_cyclonedds_cpp':
        missing.extend(
            name for name in CYCLONEDDS_REQUIRED_ENVIRONMENT
            if not os.environ.get(name, '').strip()
        )
    return missing


def _env_bool(name: str, default: bool = True) -> bool:
    raw = os.environ.get(name, '1' if default else '0').strip().lower()
    return raw not in ('0', 'false', 'no', 'off', 'disable', 'disabled')


def _cyclonedds_uri_to_path(uri: str) -> Path | None:
    uri = uri.strip()
    if not uri:
        return None
    parsed = urlparse(uri)
    if parsed.scheme == 'file':
        return Path(unquote(parsed.path))
    if parsed.scheme:
        return None
    return Path(uri)


def _bytes_from_cyclonedds_size(value: str) -> tuple[int | None, str | None]:
    raw = value.strip()
    if not raw:
        return None, None

    parts = raw.split()
    if len(parts) == 1:
        compact = parts[0]
        idx = 0
        while idx < len(compact) and (
            compact[idx].isdigit() or compact[idx] in ('.', '+', '-')
        ):
            idx += 1
        number = compact[:idx]
        unit = compact[idx:]
    elif len(parts) == 2:
        number, unit = parts
    else:
        return None, raw

    # CycloneDDS units are case-sensitive; "KB" is not accepted. Prefer
    # explicit bytes ("131072 B") in robot-local XML files.
    multipliers = {
        '': 1,
        'B': 1,
        'kB': 1000,
        'KiB': 1024,
        'MB': 1000 * 1000,
        'MiB': 1024 * 1024,
        'GB': 1000 * 1000 * 1000,
        'GiB': 1024 * 1024 * 1024,
    }
    if unit not in multipliers:
        return None, unit
    try:
        return int(float(number) * multipliers[unit]), None
    except ValueError:
        return None, raw


def _read_kernel_limit(name: str) -> int | None:
    try:
        return int(Path('/proc/sys/net/core', name).read_text().strip())
    except Exception:
        return None


def _validate_cyclonedds_socket_buffers() -> None:
    """Fail early for Cyclone configs that the current kernel cannot satisfy.

    This intentionally does not modify CYCLONEDDS_URI or sysctl values.  It only
    replaces the later rmw_create_node crash storm with one actionable message.
    """
    if os.environ.get('RMW_IMPLEMENTATION', '').strip() != 'rmw_cyclonedds_cpp':
        return
    if not _env_bool('FLEET_VALIDATE_CYCLONEDDS_BUFFERS', True):
        return

    path = _cyclonedds_uri_to_path(os.environ.get('CYCLONEDDS_URI', ''))
    if path is None or not path.exists():
        return

    try:
        root = ET.parse(path).getroot()
    except ET.ParseError:
        return
    ns = {'c': _CYCLONEDDS_CONFIG_NS}
    checks = (
        ('SocketReceiveBufferSize', 'rmem_max'),
        ('SocketSendBufferSize', 'wmem_max'),
    )
    problems = []
    invalid = []
    for tag, sysctl_name in checks:
        elem = root.find(f'.//c:{tag}', ns)
        if elem is None:
            continue
        raw_min = elem.get('min', '')
        requested, invalid_unit = _bytes_from_cyclonedds_size(raw_min)
        if invalid_unit is not None:
            invalid.append((tag, raw_min, invalid_unit))
            continue
        limit = _read_kernel_limit(sysctl_name)
        if requested is not None and limit is not None and requested > limit:
            problems.append((tag, raw_min, sysctl_name, limit))
    if not problems and not invalid:
        return

    if invalid:
        details = '; '.join(
            f'{tag} min={raw_min!r} has invalid unit {unit!r}'
            for tag, raw_min, unit in invalid
        )
        raise RuntimeError(
            'CycloneDDS socket buffer config has invalid units: '
            f'{details}. Code did not change your network settings. Use a '
            'CycloneDDS-supported size such as "131072 B", "128 KiB", or '
            '"128 kB" in the XML.'
        )

    details = '; '.join(
        f'{tag} min={requested} exceeds net.core.{sysctl_name}={limit}'
        for tag, requested, sysctl_name, limit in problems
    )
    raise RuntimeError(
        'CycloneDDS socket buffer config is too large for this machine: '
        f'{details}. Code did not change your network settings. Fix bashrc/'
        'CYCLONEDDS_URI or sysctl, e.g. lower the Socket*BufferSize min values '
        'in the XML or raise net.core.rmem_max/net.core.wmem_max.'
    )


def validate_shell_environment(expected_domain_id: str | None = None) -> None:
    """Fail fast when launch-time DDS values are missing or conflicting.

    Launch files in this workspace intentionally inherit DDS settings from the
    user's shell.  They should not patch, unset, or invent those values.
    """
    missing = _missing_required_environment()
    if missing:
        raise RuntimeError(
            'Missing required shell environment variable(s): '
            + ', '.join(missing)
            + '. Source your bashrc/setup before launching.'
        )
    _validate_cyclonedds_socket_buffers()

    actual_domain = os.environ.get('ROS_DOMAIN_ID', '').strip()
    if expected_domain_id is not None and str(expected_domain_id).strip() != actual_domain:
        raise RuntimeError(
            'Launch domain_id does not match shell ROS_DOMAIN_ID: '
            f'domain_id={expected_domain_id}, ROS_DOMAIN_ID={actual_domain}. '
            'Use the shell environment value or update your bashrc.'
        )


def clean_process_environment(domain_id: str) -> Dict[str, str]:
    """Return the current shell environment after validating it."""
    validate_shell_environment(str(domain_id))
    return os.environ.copy()


def dds_launch_environment(domain_id) -> List:
    """Launch actions that validate DDS settings inherited from the shell."""

    def _validate(context, *args, **kwargs):
        validate_shell_environment(_perform_if_needed(domain_id, context))
        return []

    return [OpaqueFunction(function=_validate)]


def launch_bool(value: str) -> bool:
    return value.strip().lower() in ('true', '1', 'yes', 'on')
