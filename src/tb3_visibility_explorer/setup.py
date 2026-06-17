from setuptools import find_packages, setup

package_name = 'tb3_visibility_explorer'

setup(
    name=package_name,
    version='31.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', [
            'launch/visibility_explorer.launch.py',
            'launch/slam_nav2_visibility.launch.py',
            'launch/real_robot_slam_nav2_visibility.launch.py',
            'launch/real_robot_cartographer_visibility.launch.py',
            'launch/real_robot_cartographer_nav2_visibility.launch.py',
        ]),
        ('share/' + package_name + '/config', [
            'config/real_robot_slam_toolbox.yaml',
        ]),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='bomin',
    maintainer_email='qhals8380@gmail.com',
    description='v31 Cartographer-backed visibility/NBV exploration for TurtleBot3.',
    license='MIT',
    entry_points={
        'console_scripts': [
            'visibility_explorer_node = tb3_visibility_explorer.visibility_explorer_node:main',
        ],
    },
)
