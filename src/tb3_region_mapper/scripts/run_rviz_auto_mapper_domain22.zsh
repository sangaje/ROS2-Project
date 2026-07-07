#!/usr/bin/env zsh
set +e
unsetopt ERR_EXIT 2>/dev/null || true
unsetopt PIPE_FAIL 2>/dev/null || true
source /opt/ros/jazzy/setup.zsh
source install/setup.zsh
export ROS_DOMAIN_ID=22
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export TURTLEBOT3_MODEL=burger
rviz2 -d install/tb3_region_mapper/share/tb3_region_mapper/rviz/region_auto_mapper.rviz
