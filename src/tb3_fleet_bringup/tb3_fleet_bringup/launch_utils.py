import os
from typing import Dict, List

from launch.actions import SetEnvironmentVariable, UnsetEnvironmentVariable


STALE_DDS_ENVIRONMENT = (
    'FASTRTPS_DEFAULT_PROFILES_FILE',
    'RMW_FASTRTPS_DEFAULT_PROFILES_FILE',
    'FASTDDS_DEFAULT_PROFILES_FILE',
    'ROS_DISCOVERY_SERVER',
)


def clean_process_environment(domain_id: str) -> Dict[str, str]:
    """Return a child-process environment using local subnet discovery."""
    environment = os.environ.copy()
    for name in STALE_DDS_ENVIRONMENT:
        environment.pop(name, None)
    environment.update({
        'ROS_DOMAIN_ID': str(domain_id),
        'ROS_AUTOMATIC_DISCOVERY_RANGE': 'SUBNET',
        'ROS_LOCALHOST_ONLY': '0',
        'RMW_IMPLEMENTATION': 'rmw_fastrtps_cpp',
    })
    return environment


def dds_launch_environment(domain_id) -> List:
    """Launch actions that apply the fleet DDS policy to included launches."""
    return [
        *(UnsetEnvironmentVariable(name) for name in STALE_DDS_ENVIRONMENT),
        UnsetEnvironmentVariable('ROS_LOCALHOST_ONLY'),
        SetEnvironmentVariable('ROS_DOMAIN_ID', domain_id),
        SetEnvironmentVariable('ROS_AUTOMATIC_DISCOVERY_RANGE', 'SUBNET'),
        SetEnvironmentVariable('ROS_LOCALHOST_ONLY', '0'),
        SetEnvironmentVariable('RMW_IMPLEMENTATION', 'rmw_fastrtps_cpp'),
    ]


def launch_bool(value: str) -> bool:
    return value.strip().lower() in ('true', '1', 'yes', 'on')
