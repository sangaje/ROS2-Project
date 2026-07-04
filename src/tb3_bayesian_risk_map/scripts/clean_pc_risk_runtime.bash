#!/usr/bin/env bash
set -u

# PC-only cleanup for real-robot risk mapping. This prevents stale Fleet/Gazebo
# publishers from injecting a second /map, /tf, or /cmd_vel into domains 24/25.
patterns=(
  '/opt/ros/jazzy/bin/[r]os2 launch tb3_fleet_bringup'
  '/opt/ros/jazzy/bin/[r]os2 launch tb3_bayesian_risk_map'
  '/home/seil/Desktop/ROS2-Project/install/tb3_[f]leet_bringup/'
  '[g]z sim'
  '/ros_gz_bridge/[p]arameter_bridge'
  '/domain_bridge/[d]omain_bridge'
  '[b]ayesian_risk_map_node'
  '[f]lask_yolo_server'
  '/rviz2/[r]viz2'
  '[c]artographer_occupancy_grid_node'
  '[c]artographer_node'
  '/nav2_map_server/[m]ap_server'
  '/nav2_lifecycle_manager/[l]ifecycle_manager'
  '/nav2_controller/[c]ontroller_server'
  '/nav2_planner/[p]lanner_server'
  '/nav2_behaviors/[b]ehavior_server'
  '/nav2_bt_navigator/[b]t_navigator'
)

for pattern in "${patterns[@]}"; do
  pkill -INT -f "${pattern}" 2>/dev/null || true
done

sleep 2

for pattern in "${patterns[@]}"; do
  pkill -TERM -f "${pattern}" 2>/dev/null || true
done

if command -v ros2 >/dev/null 2>&1; then
  ros2 daemon stop >/dev/null 2>&1 || true
fi

rm -rf /tmp/tb3_central_risk_domain_bridge \
       /tmp/tb3_fleet_bridge_dynamic

echo "PC Fleet/Gazebo/Risk runtime cleanup complete."
