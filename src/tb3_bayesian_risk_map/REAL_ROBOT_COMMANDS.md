# TurtleBot3 Real Robot SLAM + Bayesian Risk Map

## Robot terminal 1: bringup

```zsh
export ROS_DOMAIN_ID=22
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export FASTDDS_BUILTIN_TRANSPORTS=UDPv4
export TURTLEBOT3_MODEL=burger
source /opt/ros/jazzy/setup.zsh
source ~/tb3_risk_ws/install/setup.zsh
ros2 launch turtlebot3_bringup robot.launch.py
```

## Robot terminal 2: Cartographer + camera + risk map

```zsh
export ROS_DOMAIN_ID=22
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export FASTDDS_BUILTIN_TRANSPORTS=UDPv4
export TURTLEBOT3_MODEL=burger
source /opt/ros/jazzy/setup.zsh
source ~/tb3_risk_ws/install/setup.zsh
ros2 launch tb3_bayesian_risk_map real_robot_risk_slam.launch.py \
  use_sim_time:=false \
  start_robot_bringup:=false \
  start_camera:=true \
  start_cartographer:=true \
  start_risk_map:=true \
  start_rviz:=false \
  teleop_mode:=true \
  risk_publish_rate_hz:=5.0 \
  region_update_period_sec:=1.5 \
  visibility_num_rays:=48 \
  enable_room_probability:=false \
  camera_device:=/dev/video0 \
  camera_pixel_format:=MJPG \
  camera_output_encoding:=rgb8 \
  detection_source:=local_yolo \
  enable_yolo:=true \
  model_path:=yolo11n.pt \
  device:=cpu \
  yolo_imgsz:=320 \
  yolo_max_rate_hz:=1.0 \
  conf_threshold:=0.25
```

## Teleop launch option

If you want the launch file to start teleop in the same terminal, add:

```zsh
start_teleop:=true
```

If keyboard input does not reach teleop in your environment, keep `start_teleop:=false` and use the separate teleop terminal below.

## Robot/PC terminal 3: teleop

```zsh
export ROS_DOMAIN_ID=22
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export FASTDDS_BUILTIN_TRANSPORTS=UDPv4
export TURTLEBOT3_MODEL=burger
source /opt/ros/jazzy/setup.zsh
ros2 run turtlebot3_teleop teleop_keyboard
```

## PC terminal: RViz

```zsh
export ROS_DOMAIN_ID=22
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export FASTDDS_BUILTIN_TRANSPORTS=UDPv4
source /opt/ros/jazzy/setup.zsh
source ~/tb3_risk_ws/install/setup.zsh
ros2 launch tb3_bayesian_risk_map rviz_risk_map.launch.py
```

## Direct OpenCV camera fallback

```zsh
ros2 launch tb3_bayesian_risk_map real_robot_risk_slam.launch.py \
  use_sim_time:=false \
  start_camera:=false \
  start_cartographer:=true \
  start_risk_map:=true \
  detection_source:=opencv_camera \
  opencv_camera_device:=/dev/video0 \
  opencv_camera_fourcc:=MJPG \
  enable_yolo:=true \
  model_path:=yolo11n.pt \
  device:=cpu \
  yolo_imgsz:=320 \
  yolo_max_rate_hz:=1.0 \
  conf_threshold:=0.25
```
