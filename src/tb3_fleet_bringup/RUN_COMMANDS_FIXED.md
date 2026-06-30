# tb3_fleet fixed run commands

This fixed copy addresses the main non-moving issue in the uploaded zip:

- `fleet_waffle_nav2_group.launch.py` referenced `robot_footprint_map_free_space_filter_direct_v72.py`, but that script was missing.
- Without that script, `/map_raw` could exist while `/map` did not, so Nav2 could not plan/control and the robots would not move.
- The fixed package includes `robot_footprint_map_free_space_filter_direct_v72.py`.
- It also fixes two older launch references:
  - `fleet_domain26_burger_nav2_follower.launch.py`: `domain_burger_burger_*.yaml` -> `domain26_burger_*.yaml`
  - `fleet_waffle_slam_astar.launch.py`: missing dispatcher v63 -> existing dispatcher v72

## Install / build

```bash
cd ~/Desktop/ROS2-Project/src
rm -rf tb3_fleet_bringup tb3_fleet_bridge tb3_fleet_master tb3_fleet_robot
unzip ~/Downloads/tb3_fleet_fixed.zip -d .

cd ~/Desktop/ROS2-Project
rm -rf build/tb3_fleet_bringup install/tb3_fleet_bringup log
colcon build --symlink-install --packages-select tb3_fleet_bringup tb3_fleet_bridge tb3_fleet_master tb3_fleet_robot
source /opt/ros/jazzy/setup.bash
source install/setup.bash
```

## Recommended two-terminal run

### Terminal 1: leader Waffle + Gazebo + SLAM + Nav2

```bash
cd ~/Desktop/ROS2-Project
source /opt/ros/jazzy/setup.bash
source install/setup.bash
export ROS_DOMAIN_ID=25
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export FASTDDS_BUILTIN_TRANSPORTS=UDPv4
export TURTLEBOT3_MODEL=waffle

ros2 launch tb3_fleet_bringup fleet_waffle_nav2_group.launch.py \
  world_preset:=house \
  localization_mode:=slam \
  control_mode:=nav2 \
  domain_id:=25 \
  burger_x:=-3.20 burger_y:=-1.75 burger_yaw:=0.0 \
  waffle_x:=-2.25 waffle_y:=-1.75 waffle_yaw:=0.0
```

### Terminal 2: follower Burger, Domain 26

```bash
cd ~/Desktop/ROS2-Project
source /opt/ros/jazzy/setup.bash
source install/setup.bash
export ROS_DOMAIN_ID=26
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export FASTDDS_BUILTIN_TRANSPORTS=UDPv4
export TURTLEBOT3_MODEL=burger

ros2 launch tb3_fleet_bringup fleet_burger_nav2_group.launch.py \
  domain_id:=26 \
  leader_domain_id:=25 \
  control_mode:=astar_cmd \
  astar_target_mode:=leader \
  burger_x:=-3.20 burger_y:=-1.75 burger_yaw:=0.0 \
  map_origin_x:=-2.25 map_origin_y:=-1.75 map_origin_yaw:=0.0
```

## RViz / group goal

### Terminal 3: RViz in Domain 25

```bash
cd ~/Desktop/ROS2-Project
source /opt/ros/jazzy/setup.bash
source install/setup.bash
export ROS_DOMAIN_ID=25
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export FASTDDS_BUILTIN_TRANSPORTS=UDPv4

ros2 launch tb3_fleet_bringup fleet_domain25_fleet_debug_rviz.launch.py
```

In RViz, publish a goal to `/fleet_goal_pose` if the panel lets you choose the topic. If it only publishes `/goal_pose`, the v72 dispatcher also listens to `/goal_pose` as an alias.

## Health checks

### Domain 25: map must exist

```bash
export ROS_DOMAIN_ID=25
ros2 topic list | grep -E '^/map$|^/map_raw$|/cmd_vel|/gz_cmd_vel'
ros2 topic echo --once /map_raw
ros2 topic echo --once /map
```

If `/map_raw` exists but `/map` does not, the map-cleaner chain is still broken.

### Domain 25: force Waffle command path

```bash
export ROS_DOMAIN_ID=25
ros2 topic pub --once /cmd_vel geometry_msgs/msg/TwistStamped \
"{header: {frame_id: base_link}, twist: {linear: {x: 0.08}, angular: {z: 0.0}}}"

ros2 topic pub --once /cmd_vel geometry_msgs/msg/TwistStamped \
"{header: {frame_id: base_link}, twist: {linear: {x: 0.0}, angular: {z: 0.0}}}"
```

### Domain 26: force Burger command path

```bash
export ROS_DOMAIN_ID=26
ros2 topic pub --once /cmd_vel geometry_msgs/msg/TwistStamped \
"{header: {frame_id: base_link}, twist: {linear: {x: 0.08}, angular: {z: 0.0}}}"

ros2 topic pub --once /cmd_vel geometry_msgs/msg/TwistStamped \
"{header: {frame_id: base_link}, twist: {linear: {x: 0.0}, angular: {z: 0.0}}}"
```
