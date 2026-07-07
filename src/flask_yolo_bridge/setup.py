from glob import glob
import os

from setuptools import setup

package_name = 'flask_yolo_bridge'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    tests_require=['pytest'],
    zip_safe=True,
    maintainer='bomin',
    maintainer_email='qhals8380@gmail.com',
    description='Send ROS camera frames to a Flask YOLO server and publish compact detection JSON for risk mapping.',
    license='MIT',
    entry_points={
        'console_scripts': [
            'flask_yolo_server = flask_yolo_bridge.flask_yolo_server:main',
            'opencv_camera_to_flask_yolo = flask_yolo_bridge.opencv_camera_to_flask_yolo:main',
        ],
    },
)
