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
        (os.path.join('share', package_name, 'config'), only_files('config/*.yaml') + only_files('config/*.lua')),
        (os.path.join('share', package_name, 'rviz'), only_files('rviz/*.rviz')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='seil',
    maintainer_email='none@example.com',
    description='TurtleBot3 dual Nav2 fleet bringup with Gazebo, RViz, group goals, and formation goal proxies.',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            'send_nav_goal = tb3_fleet_bringup.send_nav_goal:main',
            'send_waffle_nav_goal = tb3_fleet_bringup.send_waffle_nav_goal:main',
            'check_dual_nav2_ready = tb3_fleet_bringup.check_dual_nav2_ready:main',
            'twist_stamped_to_twist = tb3_fleet_bringup.twist_stamped_to_twist:main',
            'single_twist_stamped_to_twist = tb3_fleet_bringup.single_twist_stamped_to_twist:main',
            'single_domain_nav2_frame_tools = tb3_fleet_bringup.single_domain_nav2_frame_tools:main',
            'leader_pose_publisher = tb3_fleet_bringup.leader_pose_publisher:main',
            'domain_bridge_nav2_follower = tb3_fleet_bringup.domain_bridge_nav2_follower:main',
            'waffle_burger_follower = tb3_fleet_bringup.waffle_burger_follower:main',
            'waffle_burger_nav2_follower = tb3_fleet_bringup.waffle_burger_nav2_follower:main',
            'nav2_frame_tools = tb3_fleet_bringup.nav2_frame_tools:main',
        ],
    },
)
