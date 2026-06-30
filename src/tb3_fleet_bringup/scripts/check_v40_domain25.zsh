#!/usr/bin/env zsh
set +e
cd ~/Desktop/ROS2-Project
source /opt/ros/jazzy/setup.zsh
source install/setup.zsh
export ROS_DOMAIN_ID=25
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export FASTDDS_BUILTIN_TRANSPORTS=UDPv4

echo "===== DOMAIN $ROS_DOMAIN_ID NODES ====="
ros2 node list | sort

echo "===== DUPLICATES ====="
ros2 node list | sort | uniq -d || true

echo "===== TOPICS ====="
timeout 2 ros2 topic echo --once /clock >/tmp/v40_clock_${ROS_DOMAIN_ID}.log 2>&1 && echo OK:/clock || echo NO:/clock
timeout 2 ros2 topic echo --once /odom_nav >/tmp/v40_odom_${ROS_DOMAIN_ID}.log 2>&1 && echo OK:/odom_nav || echo NO:/odom_nav
timeout 2 ros2 topic echo --once /scan_nav >/tmp/v40_scan_${ROS_DOMAIN_ID}.log 2>&1 && echo OK:/scan_nav || echo NO:/scan_nav
timeout 2 ros2 topic echo --once /map >/tmp/v40_map_${ROS_DOMAIN_ID}.log 2>&1 && echo OK:/map || echo NO:/map
timeout 2 ros2 topic echo --once /leader_pose >/tmp/v40_leader_${ROS_DOMAIN_ID}.log 2>&1 && echo OK:/leader_pose || echo NO:/leader_pose

echo "===== TF ====="
timeout 5 ros2 run tf2_ros tf2_echo map odom >/tmp/v40_tf_map_odom_${ROS_DOMAIN_ID}.log 2>&1 && echo OK:map-odom || echo NO:map-odom
timeout 5 ros2 run tf2_ros tf2_echo odom base_footprint >/tmp/v40_tf_odom_base_${ROS_DOMAIN_ID}.log 2>&1 && echo OK:odom-base_footprint || echo NO:odom-base_footprint

echo "===== LIFECYCLE ====="
for n in /map_server /controller_server /planner_server /behavior_server /bt_navigator; do
  echo "----- $n -----"
  ros2 lifecycle get $n || true
done

echo "===== ACTION ====="
ros2 action list | grep navigate_to_pose || echo "NO /navigate_to_pose"
ros2 action info /navigate_to_pose || true
