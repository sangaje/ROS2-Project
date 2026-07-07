from glob import glob
import os
from setuptools import find_packages, setup

package_name = 'region_mapper'

setup(
    name=package_name,
    version='0.3.2',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml', 'README.md']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'rviz'), glob('rviz/*.rviz')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
        (os.path.join('share', package_name, 'scripts'), glob('scripts/*.zsh')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='bomin',
    maintainer_email='qhals8380@gmail.com',
    description='Unified TurtleBot3 region graph + region-aware active SLAM auto-mapper.',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'slam_region_graph_node = region_mapper.slam_region_graph_node:main',
            'region_explorer_node = region_mapper.region_explorer_node:main',
            'region_auto_mapper_node = region_mapper.region_auto_mapper_node:main',
            'region_nav2_explorer_node = region_mapper.region_nav2_explorer_node:main',
        ],
    },
)
