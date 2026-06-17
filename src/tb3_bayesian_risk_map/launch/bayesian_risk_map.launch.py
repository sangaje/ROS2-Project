from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    pkg_share = get_package_share_directory('tb3_bayesian_risk_map')
    default_config = os.path.join(pkg_share, 'config', 'bayesian_risk_map.yaml')

    return LaunchDescription([
        DeclareLaunchArgument('config_file', default_value=default_config),
        DeclareLaunchArgument('use_sim_time', default_value='false'),
        DeclareLaunchArgument('map_topic', default_value='/map'),
        DeclareLaunchArgument('image_topic', default_value='/camera/image_raw'),
        DeclareLaunchArgument('map_frame', default_value='map'),
        DeclareLaunchArgument('base_frame', default_value='base_link'),
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

        Node(
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
    ])
