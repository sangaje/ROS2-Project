from setuptools import find_packages, setup
from glob import glob
import os

package_name = 'tb3_fleet_master'

setup(
    name=package_name,
    version='0.2.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='bomin',
    maintainer_email='qhals8380@gmail.com',
    description='Fleet master commander for multi-TurtleBot3 group goals.',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            'fleet_commander_node = tb3_fleet_master.fleet_commander_node:main',
            'send_group_goal = tb3_fleet_master.send_group_goal:main',
            'fleet_state_echo = tb3_fleet_master.fleet_state_echo:main',
        ],
    },
)
