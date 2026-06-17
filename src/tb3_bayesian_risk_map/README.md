# tb3_bayesian_risk_map v4

## Fixes

- RViz config now includes interactive tools: Interact, MoveCamera, Select, FocusCamera, Measure.
- RViz LaserScan queue increased to reduce message filter queue-full spam.
- OpenCV YOLO viewer node added: `opencv_yolo_viewer_node`.
- All-in-one launch starts OpenCV YOLO viewer by default.
- TF buffer increased to 120 seconds.
- Risk node uses latest TF and falls back between `base_link` and `base_footprint`.

## Main launch

```zsh
ros2 launch tb3_bayesian_risk_map cartographer_risk_rviz.launch.py \
  use_sim_time:=false \
  start_opencv_yolo_view:=true
```

## OpenCV viewer only

```zsh
ros2 launch tb3_bayesian_risk_map opencv_yolo_view.launch.py \
  image_topic:=/risk/debug_yolo_image
```
