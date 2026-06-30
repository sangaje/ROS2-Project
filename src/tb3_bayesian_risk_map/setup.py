from setuptools import setup
from glob import glob
import os

package_name = 'tb3_bayesian_risk_map'

setup(
    name=package_name,
    version='0.7.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
        (os.path.join('share', package_name, 'config'), glob('config/*.lua')),
        (os.path.join('share', package_name, 'rviz'), glob('rviz/*.rviz')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='bomin',
    maintainer_email='qhals8380@gmail.com',
    description='Persistent room-aware Bayesian risk map for TurtleBot3 using YOLO, Cartographer map, TF, and camera visibility.',
    license='MIT',
    entry_points={
        'console_scripts': [
            'bayesian_risk_map_node = tb3_bayesian_risk_map.bayesian_risk_map_node:main',
            'opencv_yolo_viewer_node = tb3_bayesian_risk_map.opencv_yolo_viewer_node:main',
        ],
    },
)
