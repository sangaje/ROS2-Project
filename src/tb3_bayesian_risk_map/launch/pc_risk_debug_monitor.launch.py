from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    pkg_share = get_package_share_directory('tb3_bayesian_risk_map')
    rviz_config = os.path.join(pkg_share, 'rviz', 'bayesian_risk_map.rviz')

    return LaunchDescription([
        DeclareLaunchArgument('start_rviz', default_value='true'),
        DeclareLaunchArgument('start_opencv_debug_view', default_value='true'),
        DeclareLaunchArgument('start_rqt_debug_view', default_value='false'),
        DeclareLaunchArgument('debug_image_topic', default_value='/risk/debug_yolo_image'),
        DeclareLaunchArgument('resize_width', default_value='960'),

        Node(
            condition=IfCondition(LaunchConfiguration('start_rviz')),
            package='rviz2',
            executable='rviz2',
            name='rviz2_risk_map',
            output='screen',
            arguments=['-d', rviz_config],
        ),

        Node(
            condition=IfCondition(LaunchConfiguration('start_opencv_debug_view')),
            package='tb3_bayesian_risk_map',
            executable='opencv_yolo_viewer_node',
            name='opencv_yolo_viewer_node',
            output='screen',
            parameters=[{
                'image_topic': LaunchConfiguration('debug_image_topic'),
                'resize_width': LaunchConfiguration('resize_width'),
            }],
        ),

        Node(
            condition=IfCondition(LaunchConfiguration('start_rqt_debug_view')),
            package='rqt_image_view',
            executable='rqt_image_view',
            name='rqt_yolo_debug_view',
            output='screen',
            arguments=[LaunchConfiguration('debug_image_topic')],
        ),
    ])
