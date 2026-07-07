#!/usr/bin/env zsh
set +e
unsetopt ERR_EXIT 2>/dev/null || true
unsetopt PIPE_FAIL 2>/dev/null || true
source /opt/ros/jazzy/setup.zsh
source install/setup.zsh
export ROS_DOMAIN_ID=22
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export TURTLEBOT3_MODEL=burger
ros2 launch region_mapper sim_house_region_graph.launch.py use_sim_time:=true turtlebot3_model:=burger slam_backend:=cartographer robot_frame:=base_footprint global_frame:=map use_rviz:=false
