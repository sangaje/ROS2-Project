from setuptools import setup
from glob import glob
import os
package_name='scout_map_risk_bridge'
setup(
    name=package_name,
    version='0.4.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',['resource/'+package_name]),
        ('share/'+package_name,['package.xml']),
        (os.path.join('share',package_name,'launch'),glob('launch/*.launch.py')),
        (os.path.join('share',package_name,'rviz'),glob('rviz/*.rviz')),
        (os.path.join('share',package_name,'scripts'),glob('scripts/*.bash')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='bomin',
    maintainer_email='qhals8380@gmail.com',
    description='Bridge scout map and risk map across ROS 2 domains.',
    license='MIT',
    entry_points={'console_scripts':['scout_map_risk_bridge_node=scout_map_risk_bridge.scout_map_risk_bridge_node:main']},
)
