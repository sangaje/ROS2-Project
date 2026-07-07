from glob import glob
import os

from setuptools import setup

package_name = 'multi'


def collect_model_files():
    data_files = []
    for model_dir in glob('models/*'):
        if not os.path.isdir(model_dir):
            continue
        files = [f for f in glob(os.path.join(model_dir, '*')) if os.path.isfile(f)]
        if files:
            data_files.append((os.path.join('share', package_name, model_dir), files))
    # Keep legacy top-level sdf files installed too, so older launch variants still work.
    top_level_sdf = glob('models/*.sdf')
    if top_level_sdf:
        data_files.append((os.path.join('share', package_name, 'models'), top_level_sdf))
    return data_files


setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        (os.path.join('share', package_name), ['package.xml', 'README_MULTI_GZ.md']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'maps'), glob('maps/*')),
        (os.path.join('share', package_name, 'worlds'), glob('worlds/*.world')),
        (os.path.join('share', package_name, 'rviz'), glob('rviz/*.rviz')),
    ] + collect_model_files(),
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='user',
    maintainer_email='user@todo.todo',
    description=(
        'Gazebo/RViz multi TurtleBot3 rescue simulation with physical robot '
        'launch support'
    ),
    license='Apache License 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'static_map_publisher = multi.static_map_publisher:main',
            'goal_dispatcher = multi.goal_dispatcher:main',
            'simple_goal_controller = multi.simple_goal_controller:main',
            'auto_patrol_rescue = multi.auto_patrol_rescue:main',
            'robot_signal = multi.robot_signal:main',
            'region_nav2_goal = multi.region_nav2_goal:main',
        ],
    },
)
