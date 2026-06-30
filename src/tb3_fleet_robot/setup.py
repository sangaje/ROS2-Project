from setuptools import find_packages, setup
from glob import glob
import os

package_name = 'tb3_fleet_robot'

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
    description='Follower-side goal proxy and pose reporter for multi-TurtleBot3 fleet navigation.',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            'robot_goal_proxy = tb3_fleet_robot.robot_goal_proxy:main',
            'robot_pose_reporter = tb3_fleet_robot.robot_pose_reporter:main',
            'initial_pose_publisher = tb3_fleet_robot.initial_pose_publisher:main',
        ],
    },
)
