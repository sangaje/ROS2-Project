from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, SetEnvironmentVariable
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    pkg_share = get_package_share_directory('tb3_bayesian_risk_map')
    rviz_config = os.path.join(pkg_share, 'rviz', 'slam_risk_live.rviz')

    return LaunchDescription([
        DeclareLaunchArgument('start_rviz', default_value='true'),
        DeclareLaunchArgument('start_opencv_debug_view', default_value='true'),
        DeclareLaunchArgument('start_rqt_debug_view', default_value='false'),
        DeclareLaunchArgument('domain_id', default_value='25'),
        DeclareLaunchArgument('rviz_config', default_value=rviz_config),
        DeclareLaunchArgument('debug_image_topic', default_value='/risk/debug_yolo_image/compressed'),
        DeclareLaunchArgument('image_type', default_value='auto'),
        DeclareLaunchArgument('resize_width', default_value='960'),
        DeclareLaunchArgument('grid_topics', default_value=''),

        SetEnvironmentVariable('ROS_DOMAIN_ID', LaunchConfiguration('domain_id')),
        SetEnvironmentVariable('RMW_IMPLEMENTATION', 'rmw_fastrtps_cpp'),
        SetEnvironmentVariable('FASTDDS_BUILTIN_TRANSPORTS', 'UDPv4'),

        Node(
            condition=IfCondition(LaunchConfiguration('start_rviz')),
            package='rviz2',
            executable='rviz2',
            name='rviz2_risk_map',
            output='screen',
            arguments=['-d', LaunchConfiguration('rviz_config')],
            parameters=[{'use_sim_time': False}],
        ),

        Node(
            condition=IfCondition(LaunchConfiguration('start_opencv_debug_view')),
            package='tb3_bayesian_risk_map',
            executable='opencv_yolo_viewer_node',
            name='opencv_yolo_viewer_node',
            output='screen',
            parameters=[{
                'image_topic': LaunchConfiguration('debug_image_topic'),
                'image_type': LaunchConfiguration('image_type'),
                'resize_width': LaunchConfiguration('resize_width'),
                'grid_topics': LaunchConfiguration('grid_topics'),
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
