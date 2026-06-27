import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    use_sim_time = LaunchConfiguration('use_sim_time')
    pkg_share = get_package_share_directory('tb3_region_mapper')
    rviz_config = os.path.join(pkg_share, 'rviz', 'region_graph.rviz')

    # This launch intentionally keeps a harmless process alive after RViz exits.
    # Closing the RViz GUI will not terminate the launch terminal. Use Ctrl-C to stop it.
    keepalive = ExecuteProcess(
        cmd=['python3', '-c', 'import time\nwhile True: time.sleep(3600)'],
        name='rviz_keepalive',
        output='log',
    )

    return LaunchDescription([
        DeclareLaunchArgument('use_sim_time', default_value='true'),
        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2_region_graph',
            output='screen',
            arguments=['-d', rviz_config],
            parameters=[{'use_sim_time': use_sim_time}],
        ),
        keepalive,
    ])
