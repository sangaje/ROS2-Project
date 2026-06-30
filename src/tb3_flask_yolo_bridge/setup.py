from glob import glob
import os

from setuptools import setup

package_name = 'tb3_flask_yolo_bridge'

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
    zip_safe=True,
    maintainer='bomin',
    maintainer_email='qhals8380@gmail.com',
    description='Send ROS camera frames to a Flask YOLO server and publish compact detection JSON for risk mapping.',
    license='MIT',
    entry_points={
        'console_scripts': [
            'flask_yolo_server = tb3_flask_yolo_bridge.flask_yolo_server:main',
            'opencv_camera_publisher = tb3_flask_yolo_bridge.opencv_camera_publisher:main',
            'ros_image_to_flask_yolo = tb3_flask_yolo_bridge.ros_image_to_flask_yolo:main',
            'random_world_detection_test = tb3_flask_yolo_bridge.random_world_detection_test:main',
        ],
    },
)
