#!/usr/bin/env bash
set -euo pipefail

patterns=(
  'ros2 launch tb3_fleet_bringup real.launch.py'
  'ros2 launch tb3_fleet_bringup pc.launch.py'
  'ros2 launch tb3_fleet_bringup leader.launch.py'
  'ros2 launch tb3_fleet_bringup follower.launch.py'
  'ros2 launch tb3_fleet_bringup rviz.launch.py'
  'ros2 launch tb3_fleet_bringup robot.launch.py'
  'ros2 launch tb3_fleet_bringup sim.launch.py'
  'ros2 launch tb3_fleet_bringup sim_world.launch.py'
  'ros2 launch tb3_fleet_bringup sim_leader.launch.py'
  'ros2 launch tb3_fleet_bringup sim_follower.launch.py'
  'ros2 launch tb3_fleet_bridge bridges.launch.py'
  '/cartographer_ros/cartographer_node'
  '/cartographer_ros/cartographer_occupancy_grid_node'
  '/domain_bridge/domain_bridge'
  '/ros_gz_sim/create'
  '/ros_gz_bridge/parameter_bridge'
  '/gz sim'
  'gz sim -r -s'
  'gz sim -g'
  'ruby /opt/ros/jazzy/opt/gz_tools_vendor/bin/gz sim'
  'leader_gz_bridge'
  'follower_gz_bridge'
  'spawn_leader'
  'spawn_follower'
  '/rviz2/rviz2'
  'rviz2_fleet'
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
  'sim_burger_scan_relay.py'
  'scan_frame_relay.py'
  'real_burger_scan_frame_relay'
  'sim_burger_tf_relay.py'
  'sim_map_relay.py'
  'sim_burger_tf_forwarder.py'
  '/tf2_ros/static_transform_publisher'
  'tf2_echo'
)

ancestor_pids() {
  local pid="$$"
  while [[ -n "${pid}" && "${pid}" != "0" ]]; do
    echo "${pid}"
    pid="$(ps -o ppid= -p "${pid}" 2>/dev/null | tr -d ' ')"
  done
}

is_ancestor_pid() {
  local needle="$1"
  local ancestor
  for ancestor in "${ancestors[@]}"; do
    [[ "${needle}" == "${ancestor}" ]] && return 0
  done
  return 1
}

kill_matches() {
  local signal="$1"
  local pattern="$2"
  local pid
  while read -r pid; do
    [[ -z "${pid}" ]] && continue
    is_ancestor_pid "${pid}" && continue
    kill "-${signal}" "${pid}" 2>/dev/null || true
  done < <(pgrep -f "${pattern}" 2>/dev/null || true)
}

report_matches() {
  local pattern="$1"
  local pid
  while read -r pid; do
    [[ -z "${pid}" ]] && continue
    is_ancestor_pid "${pid}" && continue
    ps -o pid=,args= -p "${pid}" 2>/dev/null || true
  done < <(pgrep -f "${pattern}" 2>/dev/null || true)
}

echo '[clean_real_fleet_runtime] stopping real fleet processes...'
mapfile -t ancestors < <(ancestor_pids)
for pattern in "${patterns[@]}"; do
  kill_matches INT "${pattern}"
done

sleep 2

for pattern in "${patterns[@]}"; do
  kill_matches TERM "${pattern}"
done

sleep 1

for pattern in "${patterns[@]}"; do
  kill_matches KILL "${pattern}"
done

rm -f /tmp/turtlebot3_burger_follower.sdf \
      /tmp/turtlebot3_burger_follower_bridge.yaml \
      /tmp/sim_burger_amcl_initial_pose.yaml \
      /tmp/burger_amcl_initial_pose.yaml

rm -rf /tmp/tb3_sim_domain_bridge \
       /tmp/tb3_fleet_domain_bridge \
       /tmp/tb3_fleet_bridge_dynamic

echo '[clean_real_fleet_runtime] remaining matching processes:'
report_matches 'real.launch.py|pc.launch.py|leader.launch.py|follower.launch.py|robot.launch.py|sim.launch.py|sim_world.launch.py|sim_leader.launch.py|sim_follower.launch.py|bridges.launch.py|cartographer_node|domain_bridge|rviz2_fleet|domain_bridge_nav2_follower|fleet_debug_marker|sim_burger|sim_map|gz sim|parameter_bridge'
