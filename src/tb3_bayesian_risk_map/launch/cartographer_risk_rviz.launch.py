from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    pkg_share = get_package_share_directory('tb3_bayesian_risk_map')
    default_config = os.path.join(pkg_share, 'config', 'bayesian_risk_map.yaml')
    default_rviz = os.path.join(pkg_share, 'rviz', 'bayesian_risk_map.rviz')
    rviz_clean = PathJoinSubstitution([
        FindPackageShare('tb3_bayesian_risk_map'),
        'scripts',
        'rviz2_clean_env.bash',
    ])

    cartographer_config_dir = PathJoinSubstitution([
        FindPackageShare('turtlebot3_cartographer'),
        'config',
    ])

    return LaunchDescription([
        DeclareLaunchArgument('use_sim_time', default_value='false'),
        DeclareLaunchArgument('start_cartographer', default_value='true'),
        DeclareLaunchArgument('start_risk_map', default_value='true'),
        DeclareLaunchArgument('start_rviz', default_value='true'),
        DeclareLaunchArgument('start_yolo_view', default_value='false'),
        DeclareLaunchArgument('start_opencv_yolo_view', default_value='true'),

        DeclareLaunchArgument('cartographer_configuration_basename', default_value='turtlebot3_lds_2d.lua'),
        DeclareLaunchArgument('cartographer_resolution', default_value='0.05'),
        DeclareLaunchArgument('cartographer_publish_period_sec', default_value='1.0'),

        DeclareLaunchArgument('config_file', default_value=default_config),
        DeclareLaunchArgument('map_topic', default_value='/map'),
        DeclareLaunchArgument('image_topic', default_value='/camera/image_raw'),
        DeclareLaunchArgument('map_frame', default_value='map'),
        DeclareLaunchArgument('base_frame', default_value='base_link'),
        DeclareLaunchArgument('detection_source', default_value='local_yolo'),
        DeclareLaunchArgument('external_detection_topic', default_value='/risk/yolo_detections'),
        DeclareLaunchArgument('enable_yolo', default_value='true'),
        DeclareLaunchArgument('enable_fake_detection', default_value='false'),
        DeclareLaunchArgument('model_path', default_value='yolo11n.pt'),
        DeclareLaunchArgument('device', default_value='cpu'),
        DeclareLaunchArgument('update_rate_hz', default_value='2.0'),
        DeclareLaunchArgument('conf_threshold', default_value='0.20'),
        DeclareLaunchArgument('debug_show_opencv', default_value='false'),
        DeclareLaunchArgument('debug_save_images', default_value='false'),
        DeclareLaunchArgument('preserve_risk_on_map_resize', default_value='true'),
        DeclareLaunchArgument('source_halo_radius_m', default_value='0.75'),
        DeclareLaunchArgument('source_halo_sigma_m', default_value='0.35'),
        DeclareLaunchArgument('yolo_view_topic', default_value='/risk/debug_yolo_image'),
        DeclareLaunchArgument('opencv_view_resize_width', default_value='960'),

        DeclareLaunchArgument('rviz_config', default_value=default_rviz),

        Node(
            condition=IfCondition(LaunchConfiguration('start_cartographer')),
            package='cartographer_ros',
            executable='cartographer_node',
            name='cartographer_node',
            output='screen',
            parameters=[{'use_sim_time': LaunchConfiguration('use_sim_time')}],
            arguments=[
                '-configuration_directory', cartographer_config_dir,
                '-configuration_basename', LaunchConfiguration('cartographer_configuration_basename'),
            ],
        ),

        Node(
            condition=IfCondition(LaunchConfiguration('start_cartographer')),
            package='cartographer_ros',
            executable='cartographer_occupancy_grid_node',
            name='cartographer_occupancy_grid_node',
            output='screen',
            parameters=[{'use_sim_time': LaunchConfiguration('use_sim_time')}],
            arguments=[
                '-resolution', LaunchConfiguration('cartographer_resolution'),
                '-publish_period_sec', LaunchConfiguration('cartographer_publish_period_sec'),
            ],
        ),

        Node(
            condition=IfCondition(LaunchConfiguration('start_risk_map')),
            package='tb3_bayesian_risk_map',
            executable='bayesian_risk_map_node',
            name='bayesian_risk_map_node',
            output='screen',
            parameters=[
                LaunchConfiguration('config_file'),
                {
                    'use_sim_time': LaunchConfiguration('use_sim_time'),
                    'map_topic': LaunchConfiguration('map_topic'),
                    'image_topic': LaunchConfiguration('image_topic'),
                    'map_frame': LaunchConfiguration('map_frame'),
                    'base_frame': LaunchConfiguration('base_frame'),
                    'detection_source': LaunchConfiguration('detection_source'),
                    'external_detection_topic': LaunchConfiguration('external_detection_topic'),
                    'enable_yolo': LaunchConfiguration('enable_yolo'),
                    'enable_fake_detection': LaunchConfiguration('enable_fake_detection'),
                    'model_path': LaunchConfiguration('model_path'),
                    'device': LaunchConfiguration('device'),
                    'update_rate_hz': LaunchConfiguration('update_rate_hz'),
                    'conf_threshold': LaunchConfiguration('conf_threshold'),
                    'debug_show_opencv': LaunchConfiguration('debug_show_opencv'),
                    'debug_save_images': LaunchConfiguration('debug_save_images'),
                    'preserve_risk_on_map_resize': LaunchConfiguration('preserve_risk_on_map_resize'),
                    'source_halo_radius_m': LaunchConfiguration('source_halo_radius_m'),
                    'source_halo_sigma_m': LaunchConfiguration('source_halo_sigma_m'),
                },
            ],
        ),

        ExecuteProcess(
            condition=IfCondition(LaunchConfiguration('start_rviz')),
            cmd=[
                rviz_clean,
                '-d', LaunchConfiguration('rviz_config'),
                '--ros-args',
                '-r', '__node:=rviz2_risk_cartographer',
                '-p', ['use_sim_time:=', LaunchConfiguration('use_sim_time')],
            ],
            name='rviz2_risk_cartographer',
            output='screen',
        ),

        Node(
            condition=IfCondition(LaunchConfiguration('start_opencv_yolo_view')),
            package='tb3_bayesian_risk_map',
            executable='opencv_yolo_viewer_node',
            name='opencv_yolo_viewer_node',
            output='screen',
            parameters=[{
                'image_topic': LaunchConfiguration('yolo_view_topic'),
                'resize_width': LaunchConfiguration('opencv_view_resize_width'),
            }],
        ),

        Node(
            condition=IfCondition(LaunchConfiguration('start_yolo_view')),
            package='rqt_image_view',
            executable='rqt_image_view',
            name='rqt_yolo_debug_view',
            output='screen',
            arguments=[LaunchConfiguration('yolo_view_topic')],
        ),
    ])
