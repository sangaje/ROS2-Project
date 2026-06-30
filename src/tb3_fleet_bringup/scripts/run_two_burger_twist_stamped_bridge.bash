#!/usr/bin/env bash
set -euo pipefail
WS="${1:-$HOME/Desktop/ROS2-Project}"
cd "$WS"
source /opt/ros/jazzy/setup.bash
source "$HOME/turtlebot3_ws/install/setup.bash"
source install/local_setup.bash
export ROS_DOMAIN_ID=26
export TURTLEBOT3_MODEL=burger
ros2 launch tb3_fleet_bringup fleet_two_burger_spawn_with_ros_bridge.launch.py
