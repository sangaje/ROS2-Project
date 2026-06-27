#!/usr/bin/env zsh
set +e
unsetopt ERR_EXIT 2>/dev/null || true
unsetopt PIPE_FAIL 2>/dev/null || true
source /opt/ros/jazzy/setup.zsh
source install/setup.zsh
export ROS_DOMAIN_ID=22
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export TURTLEBOT3_MODEL=burger
ros2 launch tb3_region_mapper region_auto_mapper.launch.py use_sim_time:=true auto_start:=true cmd_vel_topic:=/cmd_vel max_linear_x:=0.20 max_angular_z:=0.85 front_stop_distance_m:=0.30 front_slow_distance_m:=0.68 side_stop_distance_m:=0.24 emergency_stop_distance_m:=0.18 view_fov_deg:=360.0 view_max_range_m:=3.2 region_coverage_threshold:=0.82 region_frontier_threshold:=10 candidate_grid_step_m:=0.25 candidate_min_clearance_m:=0.40 a_star_unknown_allowed:=false conservative_astar:=true path_min_clearance_m:=0.40 path_prefer_clearance_m:=0.95 path_wall_cost_weight:=28.0 path_unknown_cost:=100.0 search_motion_front_clearance_m:=0.48 select_next_region_policy:=nearest_uncovered
