from launch import LaunchDescription
from launch.actions import ExecuteProcess, LogInfo
from launch.substitutions import PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    config_path = PathJoinSubstitution([
        FindPackageShare('tb3_fleet_bridge'),
        'config',
        'leader_pose_25_to_26.yaml',
    ])
    return LaunchDescription([
        LogInfo(msg='V22_DOMAIN_BRIDGE | /leader_pose from ROS_DOMAIN_ID=25 to ROS_DOMAIN_ID=26'),
        ExecuteProcess(
            cmd=['ros2', 'run', 'domain_bridge', 'domain_bridge', config_path],
            output='screen',
        ),
    ])
