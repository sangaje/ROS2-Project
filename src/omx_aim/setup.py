import os
from glob import glob

from setuptools import find_packages, setup

package_name = 'omx_aim'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
        (os.path.join('share', package_name, 'models'), glob('models/*.pt')),
    ],
    package_data={
        'omx.debug_stream': ['templates/*.html'],
    },
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='dladyddn133-creator',
    maintainer_email='dladyddn133@gmail.com',
    description='OpenManipulator-X 기반 자동 조준 + 정찰 시스템 (Burger/Waffle 협력)',
    license='TODO: License declaration',
    entry_points={
        'console_scripts': [
            'yolo_node = omx_aim.yolo_node:main',
            'waffle_node = omx_aim.waffle_node:main',
            'fire_node = omx_aim.fire_node:main',
            'map_relay = omx_aim.map_relay:main',
            'patrol_planner = omx_aim.patrol_planner:main',
            'auto_initialpose = omx_aim.auto_initialpose:main',
            'target_bridge = omx_aim.target_bridge:main',
            'scan_processor = omx_aim.scan_processor:main',
            'scan_diag = omx_aim.scan_diag:main',
            'scout_watchdog = omx_aim.scout_watchdog:main',
            'fake_risk_map = omx_aim.fake_risk_map:main',
            'fake_static_map = omx_aim.fake_static_map:main',
            'ik_teleop = omx_aim.ik_teleop:main',
            'unified_dashboard = omx_aim.unified_dashboard:main',
        ],
    },
)
