#!/usr/bin/env bash
set -e
PROJECT_DIR="${1:-$HOME/Desktop/ROS2-Project}"
cd "$PROJECT_DIR"
source /opt/ros/jazzy/setup.bash
source "$HOME/turtlebot3_ws/install/setup.bash"
source install/local_setup.bash
export ROS_DOMAIN_ID=26
export TURTLEBOT3_MODEL=burger
ros2 launch tb3_fleet_bringup fleet_waffle_leader_burger_follower.launch.py
