#!/usr/bin/env zsh
set -e
cd ~/Desktop/ROS2-Project
source /opt/ros/jazzy/setup.zsh
source install/setup.zsh
export ROS_DOMAIN_ID=25
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export FASTDDS_BUILTIN_TRANSPORTS=UDPv4
export TURTLEBOT3_MODEL=waffle
ros2 launch tb3_fleet_bringup fleet_domain25_waffle_nav2_world.launch.py \
  burger_x:=-3.20 burger_y:=-1.75 burger_yaw:=0.0 \
  waffle_x:=-2.25 waffle_y:=-1.75 waffle_yaw:=0.0
