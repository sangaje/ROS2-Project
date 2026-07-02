#!/usr/bin/env bash
set -euo pipefail

patterns=(
  'ros2 launch tb3_fleet_bringup fleet_real_two_burgers_pc.launch.py'
  'ros2 launch tb3_fleet_bringup fleet_real_domain25_burger_nav2.launch.py'
  'ros2 launch tb3_fleet_bringup fleet_real_domain25_waffle_nav2.launch.py'
  'ros2 launch tb3_fleet_bringup fleet_real_domain24_burger_nav2_follower.launch.py'
  'ros2 launch tb3_fleet_bringup fleet_real_domain25_rviz.launch.py'
  'ros2 launch tb3_fleet_bringup fleet_real_burger_robot.launch.py'
  '/cartographer_ros/cartographer_node'
  '/cartographer_ros/cartographer_occupancy_grid_node'
  '/domain_bridge/domain_bridge'
  '/rviz2/rviz2'
  '/nav2_controller/controller_server'
  '/nav2_planner/planner_server'
  '/nav2_behaviors/behavior_server'
  '/nav2_bt_navigator/bt_navigator'
  '/nav2_lifecycle_manager/lifecycle_manager'
  '/nav2_amcl/amcl'
  'domain_bridge_nav2_follower_direct_v40.py'
  'tf_pose_publisher_direct_v44.py'
  'pose_to_nav2_action_direct_v41.py'
  'fleet_debug_marker.py'
)

echo '[clean_real_fleet_runtime] stopping real fleet processes...'
for pattern in "${patterns[@]}"; do
  pkill -INT -f "${pattern}" 2>/dev/null || true
done

sleep 2

for pattern in "${patterns[@]}"; do
  pkill -TERM -f "${pattern}" 2>/dev/null || true
done

rm -f /tmp/fastdds_fleet_d24.xml \
      /tmp/fastdds_fleet_d25.xml \
      /tmp/fastdds_fleet_rviz_d25.xml \
      /tmp/fastdds_robot_d24.xml \
      /tmp/fastdds_robot_d25.xml

echo '[clean_real_fleet_runtime] remaining matching processes:'
pgrep -af 'fleet_real|cartographer_node|domain_bridge|rviz2_real_domain25_fleet|domain_bridge_nav2_follower|fleet_real_debug_marker' || true
