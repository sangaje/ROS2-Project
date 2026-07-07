from glob import glob
import os
from setuptools import find_packages, setup

package_name = 'system_bringup'


def only_files(pattern):
    return [p for p in glob(pattern) if os.path.isfile(p)]


launch_files = only_files('launch/*.launch.py')


setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml', 'README.md']),
        (os.path.join('share', package_name, 'launch'), launch_files),
        (os.path.join('share', package_name, 'rviz'), only_files('rviz/*.rviz')),
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
        'console_scripts': [],
    },
)
