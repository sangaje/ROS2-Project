import os
from typing import Dict, List

from launch.actions import OpaqueFunction


REQUIRED_DDS_ENVIRONMENT = (
    'ROS_DOMAIN_ID',
    'RMW_IMPLEMENTATION',
)
CYCLONEDDS_REQUIRED_ENVIRONMENT = (
    'CYCLONEDDS_URI',
)


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


def validate_shell_environment(
    expected_domain_id: str | None = None,
    expected_ros_static_peers: str | None = None,
) -> None:
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

    actual_domain = os.environ.get('ROS_DOMAIN_ID', '').strip()
    if expected_domain_id is not None and str(expected_domain_id).strip() != actual_domain:
        raise RuntimeError(
            'Launch domain_id does not match shell ROS_DOMAIN_ID: '
            f'domain_id={expected_domain_id}, ROS_DOMAIN_ID={actual_domain}. '
            'Use the shell environment value or update your bashrc.'
        )

    expected_peers = (expected_ros_static_peers or '').strip()
    actual_peers = os.environ.get('ROS_STATIC_PEERS', '').strip()
    if expected_peers and expected_peers != actual_peers:
        raise RuntimeError(
            'Launch ros_static_peers does not match shell ROS_STATIC_PEERS: '
            f'ros_static_peers={expected_peers}, ROS_STATIC_PEERS={actual_peers}. '
            'Use the shell environment value or update your bashrc.'
        )


def clean_process_environment(
    domain_id: str, ros_static_peers: str = '',
) -> Dict[str, str]:
    """Return the current shell environment after validating it."""
    validate_shell_environment(str(domain_id), ros_static_peers)
    return os.environ.copy()


def dds_launch_environment(domain_id, ros_static_peers='') -> List:
    """Launch actions that validate DDS settings inherited from the shell."""

    def _validate(context, *args, **kwargs):
        validate_shell_environment(
            _perform_if_needed(domain_id, context),
            _perform_if_needed(ros_static_peers, context),
        )
        return []

    return [OpaqueFunction(function=_validate)]


def launch_bool(value: str) -> bool:
    return value.strip().lower() in ('true', '1', 'yes', 'on')
