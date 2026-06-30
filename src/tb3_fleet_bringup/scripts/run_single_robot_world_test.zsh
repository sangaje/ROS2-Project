#!/usr/bin/env zsh
set -e
cd ~/Desktop/ROS2-Project
source /opt/ros/jazzy/setup.zsh
source ~/turtlebot3_ws/install/setup.zsh 2>/dev/null || true
source install/setup.zsh

if test -f "$(ros2 pkg prefix turtlebot3_navigation2)/share/turtlebot3_navigation2/map/map.yaml"; then
  FLEET_MAP="$(ros2 pkg prefix turtlebot3_navigation2)/share/turtlebot3_navigation2/map/map.yaml"
else
  FLEET_MAP="/opt/ros/jazzy/share/nav2_bringup/maps/tb3_sandbox.yaml"
fi

echo "FLEET_MAP=$FLEET_MAP"
ros2 launch tb3_fleet_bringup fleet_single_robot_world_test.launch.py \
  map:=$FLEET_MAP \
  robot_name:=robot1 \
  master_domain:=25 \
  robot_domain:=26 \
  tb3_model:=burger \
  init_x:=0.0 \
  init_y:=0.0 \
  init_yaw:=0.0
