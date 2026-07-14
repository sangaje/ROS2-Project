from glob import glob
import os
from setuptools import find_packages, setup

package_name = 'system_bringup'


def only_files(pattern):
    return [p for p in glob(pattern) if os.path.isfile(p)]


launch_files = only_files('launch/*.launch.py')
config_files = only_files('config/*.yaml') + only_files('config/*.json')
template_files = only_files('templates/*.html')
static_files = only_files('static/*')


setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml', 'README.md']),
        (os.path.join('share', package_name, 'launch'), launch_files),
        (os.path.join('share', package_name, 'config'), config_files),
        (os.path.join('share', package_name, 'rviz'), only_files('rviz/*.rviz')),
        (os.path.join('share', package_name, 'templates'), template_files),
        (os.path.join('share', package_name, 'static'), static_files),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='bomin',
    maintainer_email='qhals8380@gmail.com',
    description=(
        'Role-based orchestrator: turns on fleet bringup, the scout\'s '
        'Bayesian risk map and its RL policy (or the leader\'s fleet '
        'stack) from one launch file.'
    ),
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            'active_field_source_mux = system_bringup.active_field_source_mux:main',
            'leader_unified_dashboard = system_bringup.leader_unified_dashboard:main',
            'leader_shadow_follow = system_bringup.leader_shadow_follow:main',
            'scout_failover_coordinator = system_bringup.scout_failover_coordinator:main',
            'scout_rl_policy_worker = system_bringup.scout_rl_policy_worker:main',
            'system_readiness_monitor = system_bringup.system_readiness_monitor:main',
            'takeover_stack_manager = system_bringup.takeover_stack_manager:main',
            'unified_field_robot = system_bringup.unified_field_robot:main',
        ],
    },
)
