# Open Area Spawn Version

This version spawns all robots in the left open area of `turtlebot3_world`, away from the central pillar grid.

## Spawn poses

| Robot | Gazebo pose | RViz/map initial pose | Normal state |
|---|---:|---:|---|
| burger1 | x=-2.45, y=0.00 | x=-2.45, y=0.00 | Patrol |
| burger2 | x=-2.85, y=-0.55 | x=-2.85, y=-0.55 | Idle |
| waffle1 | x=-2.85, y=0.55 | x=-2.85, y=0.55 | Idle |

## Scenario

1. `burger1` patrols only in the left open area.
2. `burger2` and `waffle1` stay still.
3. When `burger1` receives `/tb3_multi/fail_robot`, it stops.
4. `burger2` and `waffle1` move to the last known `burger1` position.

## Test command

```bash
ros2 topic pub --once /tb3_multi/fail_robot std_msgs/msg/String "{data: 'burger1'}"
```
