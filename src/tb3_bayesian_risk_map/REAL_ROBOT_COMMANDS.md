# TurtleBot3 Real Robot: Robot-Side YOLO + PC Debug Monitor

## Robot terminal 1: bringup

```bash
cd ~/ROS2-Project
source /opt/ros/jazzy/setup.bash
source install/setup.bash
export ROS_DOMAIN_ID=25
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export TURTLEBOT3_MODEL=burger
ros2 launch turtlebot3_bringup robot.launch.py
```

## Robot terminal 2: camera + YOLO inference + Cartographer + risk map

```bash
cd ~/ROS2-Project
source /opt/ros/jazzy/setup.bash
source install/setup.bash
export ROS_DOMAIN_ID=25
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export TURTLEBOT3_MODEL=burger
ros2 launch tb3_bayesian_risk_map robot_camera_yolo_inference.launch.py \
  use_sim_time:=false \
  model_path:=$PWD/yolo11n.pt \
  camera_device:=/dev/video0 \
  camera_width:=640 \
  camera_height:=480 \
  camera_fps:=15 \
  yolo_imgsz:=320 \
  yolo_max_rate_hz:=1.0 \
  conf_threshold:=0.25
```

## PC terminal 1: teleop

```bash
cd ~/ROS2-Project
source /opt/ros/jazzy/setup.bash
source install/setup.bash
export ROS_DOMAIN_ID=25
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export TURTLEBOT3_MODEL=burger
ros2 run turtlebot3_teleop teleop_keyboard
```

## PC terminal 2: debug monitor (RViz + YOLO debug image viewer)

```bash
cd ~/ROS2-Project
source /opt/ros/jazzy/setup.bash
source install/setup.bash
export ROS_DOMAIN_ID=25
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
ros2 launch tb3_bayesian_risk_map pc_risk_debug_monitor.launch.py \
  start_rviz:=true \
  start_opencv_debug_view:=true \
  debug_image_topic:=/risk/debug_yolo_image
```

## Quick checks

```bash
ros2 topic hz /risk/debug_yolo_image
ros2 topic hz /risk/risk_map
ros2 topic echo /risk/risk_map --once
```
