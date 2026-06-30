#!/usr/bin/env zsh
set -e
cd ${1:-$HOME/Desktop/ROS2-Project}
source /opt/ros/jazzy/setup.zsh
source ~/turtlebot3_ws/install/setup.zsh
source install/setup.zsh
export TURTLEBOT3_MODEL=burger
ros2 launch tb3_fleet_bringup fleet_two_burger_spawn_only.launch.py \
  robot1_name:=robot1 \
  robot2_name:=robot2 \
  robot1_x:=-2.0 \
  robot1_y:=-0.5 \
  robot1_yaw:=0.0 \
  robot2_x:=-2.0 \
  robot2_y:=0.5 \
  robot2_yaw:=0.0
