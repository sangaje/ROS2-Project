from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    robot_name = LaunchConfiguration('robot_name')
    map_frame = LaunchConfiguration('map_frame')
    base_frame = LaunchConfiguration('base_frame')
    navigate_action = LaunchConfiguration('navigate_action')
    init_x = LaunchConfiguration('init_x')
    init_y = LaunchConfiguration('init_y')
    init_yaw = LaunchConfiguration('init_yaw')
    initial_pose_repeat_count = LaunchConfiguration('initial_pose_repeat_count')
    use_sim_time = LaunchConfiguration('use_sim_time')

    goal_proxy = Node(
        package='tb3_fleet_robot',
        executable='robot_goal_proxy',
        name='robot_goal_proxy',
        output='screen',
        parameters=[{
            'robot_name': robot_name,
            'navigate_action': navigate_action,
            'use_sim_time': use_sim_time,
        }],
    )

    pose_reporter = Node(
        package='tb3_fleet_robot',
        executable='robot_pose_reporter',
        name='robot_pose_reporter',
        output='screen',
        parameters=[{
            'robot_name': robot_name,
            'map_frame': map_frame,
            'base_frame': base_frame,
            'use_sim_time': use_sim_time,
        }],
    )

    initial_pose = Node(
        package='tb3_fleet_robot',
        executable='initial_pose_publisher',
        name='initial_pose_publisher',
        output='screen',
        parameters=[{
            'frame_id': map_frame,
            'x': init_x,
            'y': init_y,
            'yaw': init_yaw,
            'repeat_count': initial_pose_repeat_count,
            'period_sec': 1.0,
            'use_sim_time': use_sim_time,
        }],
    )

    return LaunchDescription([
        DeclareLaunchArgument('robot_name', default_value='robot1'),
        DeclareLaunchArgument('map_frame', default_value='map'),
        DeclareLaunchArgument('base_frame', default_value='base_footprint'),
        DeclareLaunchArgument('navigate_action', default_value='/navigate_to_pose'),
        DeclareLaunchArgument('init_x', default_value='0.0'),
        DeclareLaunchArgument('init_y', default_value='0.0'),
        DeclareLaunchArgument('init_yaw', default_value='0.0'),
        DeclareLaunchArgument('initial_pose_repeat_count', default_value='20'),
        DeclareLaunchArgument('use_sim_time', default_value='true'),
        goal_proxy,
        pose_reporter,
        initial_pose,
    ])
