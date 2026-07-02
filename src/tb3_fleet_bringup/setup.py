from glob import glob
import os
from setuptools import setup

package_name = 'tb3_fleet_bringup'


def only_files(pattern):
    return [p for p in glob(pattern) if os.path.isfile(p)]


launch_files = only_files('launch/*.launch.py')


setup(
    name=package_name,
    version='0.8.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), launch_files),
        (os.path.join('share', package_name, 'scripts'), only_files('scripts/*.py') + only_files('scripts/*.bash') + only_files('scripts/*.zsh')),
        (os.path.join('share', package_name, 'config'), only_files('config/*.yaml') + only_files('config/*.lua') + only_files('config/*.xml')),
        (os.path.join('share', package_name, 'rviz'), only_files('rviz/*.rviz')),
        (os.path.join('share', package_name, 'map'), only_files('map/*.yaml') + only_files('map/*.pgm')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='seil',
    maintainer_email='none@example.com',
    description='Real TurtleBot3 Burger dual-robot Nav2 fleet bringup.',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            'fleet_follow_signal = tb3_fleet_bringup.fleet_follow_signal:main',
        ],
    },
)
