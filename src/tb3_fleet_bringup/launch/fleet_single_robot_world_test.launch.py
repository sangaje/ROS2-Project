from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, GroupAction, IncludeLaunchDescription, SetEnvironmentVariable, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    map_yaml = LaunchConfiguration('map')
    robot_name = LaunchConfiguration('robot_name')
    robot_domain = LaunchConfiguration('robot_domain')
    master_domain = LaunchConfiguration('master_domain')
    tb3_model = LaunchConfiguration('tb3_model')
    init_x = LaunchConfiguration('init_x')
    init_y = LaunchConfiguration('init_y')
    init_yaw = LaunchConfiguration('init_yaw')
    goal_x = LaunchConfiguration('goal_x')
    goal_y = LaunchConfiguration('goal_y')
    goal_yaw = LaunchConfiguration('goal_yaw')
    autostart_goal = LaunchConfiguration('autostart_goal')

    nav2_params = LaunchConfiguration('nav2_params')

    tb3_gazebo_launch = os.path.join(
        get_package_share_directory('turtlebot3_gazebo'),
        'launch',
        'turtlebot3_world.launch.py',
    )
    nav2_navigation_launch = os.path.join(
        get_package_share_directory('nav2_bringup'),
        'launch',
        'navigation_launch.py',
    )
    master_launch = os.path.join(
        get_package_share_directory('tb3_fleet_master'),
        'launch',
        'fleet_master_shared_map.launch.py',
    )
    bridge_launch = os.path.join(
        get_package_share_directory('tb3_fleet_bridge'),
        'launch',
        'fleet_bridges.launch.py',
    )
    follower_launch = os.path.join(
        get_package_share_directory('tb3_fleet_robot'),
        'launch',
        'robot_follower_shared_map.launch.py',
    )

    master_group = GroupAction(
        scoped=True,
        actions=[
            SetEnvironmentVariable('ROS_DOMAIN_ID', master_domain),
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(master_launch),
                launch_arguments={
                    'map': map_yaml,
                    'robot_names': robot_name,
                    'spacing': '0.60',
                    'formation_type': 'wedge',
                    'frame_id': 'map',
                    # Master does not receive Gazebo /clock in this test.
                    'use_sim_time': 'false',
                }.items(),
            ),
        ],
    )

    bridge_group = GroupAction(
        scoped=True,
        actions=[
            IncludeLaunchDescription(PythonLaunchDescriptionSource(bridge_launch)),
        ],
    )

    robot_gazebo_group = GroupAction(
        scoped=True,
        actions=[
            SetEnvironmentVariable('ROS_DOMAIN_ID', robot_domain),
            SetEnvironmentVariable('TURTLEBOT3_MODEL', tb3_model),
            IncludeLaunchDescription(PythonLaunchDescriptionSource(tb3_gazebo_launch)),
        ],
    )

    amcl_group = GroupAction(
        scoped=True,
        actions=[
            SetEnvironmentVariable('ROS_DOMAIN_ID', robot_domain),
            Node(
                package='nav2_amcl',
                executable='amcl',
                name='amcl',
                output='screen',
                parameters=[nav2_params, {
                    'use_sim_time': True,
                    'global_frame_id': 'map',
                    'odom_frame_id': 'odom',
                    'base_frame_id': 'base_footprint',
                    'scan_topic': 'scan',
                    'set_initial_pose': False,
                    'always_reset_initial_pose': False,
                }],
            ),
            Node(
                package='nav2_lifecycle_manager',
                executable='lifecycle_manager',
                name='lifecycle_manager_amcl',
                output='screen',
                parameters=[{
                    'use_sim_time': True,
                    'autostart': True,
                    'node_names': ['amcl'],
                }],
            ),
        ],
    )

    nav2_group = GroupAction(
        scoped=True,
        actions=[
            SetEnvironmentVariable('ROS_DOMAIN_ID', robot_domain),
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(nav2_navigation_launch),
                launch_arguments={
                    'use_sim_time': 'true',
                    'params_file': nav2_params,
                }.items(),
            ),
        ],
    )

    follower_group = GroupAction(
        scoped=True,
        actions=[
            SetEnvironmentVariable('ROS_DOMAIN_ID', robot_domain),
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(follower_launch),
                launch_arguments={
                    'robot_name': robot_name,
                    'map_frame': 'map',
                    'base_frame': 'base_footprint',
                    'navigate_action': '/navigate_to_pose',
                    'init_x': init_x,
                    'init_y': init_y,
                    'init_yaw': init_yaw,
                    'initial_pose_repeat_count': '30',
                    'use_sim_time': 'true',
                }.items(),
            ),
        ],
    )

    # Optional one-shot goal after the stack has time to become active.
    # Disabled by default because explicit ros2 run send_group_goal is easier to debug.
    goal_sender = GroupAction(
        scoped=True,
        actions=[
            SetEnvironmentVariable('ROS_DOMAIN_ID', master_domain),
            Node(
                package='tb3_fleet_master',
                executable='send_group_goal',
                name='send_group_goal_once',
                output='screen',
                arguments=[goal_x, goal_y, goal_yaw],
            ),
        ],
    )

    actions = [
        DeclareLaunchArgument('map', default_value='/opt/ros/jazzy/share/nav2_bringup/maps/tb3_sandbox.yaml'),
        DeclareLaunchArgument('robot_name', default_value='robot1'),
        DeclareLaunchArgument('master_domain', default_value='25'),
        DeclareLaunchArgument('robot_domain', default_value='26'),
        DeclareLaunchArgument('tb3_model', default_value='burger'),
        DeclareLaunchArgument('nav2_params', default_value='/opt/ros/jazzy/share/nav2_bringup/params/nav2_params.yaml'),
        DeclareLaunchArgument('init_x', default_value='0.0'),
        DeclareLaunchArgument('init_y', default_value='0.0'),
        DeclareLaunchArgument('init_yaw', default_value='0.0'),
        DeclareLaunchArgument('goal_x', default_value='2.0'),
        DeclareLaunchArgument('goal_y', default_value='-1.0'),
        DeclareLaunchArgument('goal_yaw', default_value='0.0'),
        DeclareLaunchArgument('autostart_goal', default_value='false'),
        master_group,
        TimerAction(period=1.0, actions=[bridge_group]),
        TimerAction(period=2.0, actions=[robot_gazebo_group]),
        TimerAction(period=8.0, actions=[amcl_group]),
        TimerAction(period=10.0, actions=[nav2_group]),
        TimerAction(period=14.0, actions=[follower_group]),
    ]

    # launch substitutions cannot be used in a normal Python if. Keep goal sender manual for reliability.
    return LaunchDescription(actions)
