from glob import glob
import os
from setuptools import setup

package_name = 'tb3_fleet_bringup'


def only_files(pattern):
    return [p for p in glob(pattern) if os.path.isfile(p)]


launch_files = only_files('launch/*.launch.py')


setup(
    name=package_name,
    version='0.10.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml', 'README.md']),
        (os.path.join('share', package_name, 'launch'), launch_files),
        (os.path.join('share', package_name, 'scripts'), only_files('scripts/*.bash') + only_files('scripts/*.zsh')),
        (os.path.join('share', package_name, 'config'), only_files('config/*.yaml') + only_files('config/*.lua') + only_files('config/*.xml')),
        (os.path.join('share', package_name, 'rviz'), only_files('rviz/*.rviz')),
        (os.path.join('share', package_name, 'map'), only_files('map/*.yaml') + only_files('map/*.pgm')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='seil',
    maintainer_email='none@example.com',
    description='Unified real and simulated TurtleBot3 Nav2 fleet bringup.',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            'fleet_debug_marker = tb3_fleet_bringup.fleet_debug_marker:main',
            'fleet_follow_signal = tb3_fleet_bringup.fleet_follow_signal:main',
            'fleet_follower = tb3_fleet_bringup.fleet_follower:main',
            'fleet_path_coordinator = tb3_fleet_bringup.fleet_path_coordinator:main',
            'global_localize_kickstart = tb3_fleet_bringup.global_localize_kickstart:main',
            'map_relay = tb3_fleet_bringup.map_relay:main',
            'pose_to_nav2 = tb3_fleet_bringup.pose_to_nav2:main',
            'pose_to_tf = tb3_fleet_bringup.pose_to_tf:main',
            'scan_frame_relay = tb3_fleet_bringup.scan_frame_relay:main',
            'sim_burger_scan_relay = tb3_fleet_bringup.sim_burger_scan_relay:main',
            'sim_burger_tf_relay = tb3_fleet_bringup.sim_burger_tf_relay:main',
            'tf_pose_publisher = tb3_fleet_bringup.tf_pose_publisher:main',
        ],
    },
)
