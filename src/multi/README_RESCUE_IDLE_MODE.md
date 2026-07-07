# Multi TurtleBot3 Rescue Idle Mode

This patch implements the intended demo behavior:

- `burger1` patrols during normal operation.
- `burger2` and `waffle1` remain idle/stopped.
- When `burger1` is reported as signal-lost, `burger1` stops and reports failure.
- `burger2` and `waffle1` switch to RESCUE mode and move to burger1's last known map/world pose.

Run:

```bash
ros2 launch multi pc_debug.launch.py auto_patrol:=true
```

Trigger burger1 failure:

```bash
ros2 topic pub --once /multi/fail_robot std_msgs/msg/String "{data: 'burger1'}"
```

Check status:

```bash
ros2 topic echo /multi/robot_status
ros2 topic echo /multi/mode
ros2 topic list | grep map_pose
```
