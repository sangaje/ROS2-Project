#!/usr/bin/env zsh
set -e
cd ~/Desktop/ROS2-Project
source /opt/ros/jazzy/setup.zsh
source install/setup.zsh
export ROS_DOMAIN_ID=26
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export FASTDDS_BUILTIN_TRANSPORTS=UDPv4
export TURTLEBOT3_MODEL=burger
ros2 launch tb3_fleet_bringup fleet_domain26_burger_nav2_follower.launch.py \
  burger_x:=-3.20 burger_y:=-1.75 burger_yaw:=0.0
