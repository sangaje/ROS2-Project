#!/usr/bin/env zsh
set +e
pkill -INT -f "rviz2|region_auto_mapper|region_explorer|slam_region_graph|cartographer|slam_toolbox|gazebo|gz|robot_state_publisher|turtlebot3|teleop" || true
sleep 2
pkill -TERM -f "rviz2|region_auto_mapper|region_explorer|slam_region_graph|cartographer|slam_toolbox|gazebo|gz|robot_state_publisher|turtlebot3|teleop" || true
sleep 2
pkill -KILL -f "rviz2|region_auto_mapper|region_explorer|slam_region_graph|cartographer|slam_toolbox|gazebo|gz|robot_state_publisher|turtlebot3|teleop" || true
source /opt/ros/jazzy/setup.zsh
export ROS_DOMAIN_ID=22
ros2 daemon stop || true
sleep 1
ros2 daemon start || true
