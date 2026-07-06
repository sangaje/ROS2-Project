# tb3_bayesian_risk_map

## Launch Files

- `robot_risk_source_stack.launch.py`: robot-side bringup, Cartographer map, camera sender, and pose publisher.
- `central_risk_map_bridge.launch.py`: central risk-map node plus domain bridges.
- `pc_robot_risk_stack.launch.py`: PC-side Flask server, central bridge/risk stack, and monitor.
- `pc_risk_debug_monitor.launch.py`: RViz/debug image monitor.
- `real_robot_risk_slam.launch.py`: lower-level real robot SLAM/risk launch used by the system bringup.

All DDS/domain/RMW settings are inherited from the shell environment.

```zsh
ros2 launch tb3_bayesian_risk_map robot_risk_source_stack.launch.py
```
