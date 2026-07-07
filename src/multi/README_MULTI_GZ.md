# Multi TurtleBot3 Gazebo + RViz Click Goal / Auto Patrol / Rescue Test

This package launches three TurtleBot3 models in Gazebo:

- `burger1`
- `burger2`
- `waffle1`

RViz opens the same static map from `maps/turtlemap.yaml`.

There are two operation modes:

1. **Manual click mode**: RViz **Publish Point** publishes to `/clicked_point`; `goal_dispatcher` sends that clicked point to the currently selected robot.
2. **Auto patrol/rescue mode**: `auto_patrol_rescue` sends waypoint goals automatically. If a Burger loses signal, the remaining two robots move to the failed Burger's last known position.

## Launch

Manual mode:

```bash
ros2 launch multi pc_debug.launch.py
```

Auto patrol/rescue mode:

```bash
ros2 launch multi pc_debug.launch.py auto_patrol:=true
```

## Manual robot selection

Default target robot: `burger1`

Change target robot:

```bash
ros2 topic pub --once /target_robot std_msgs/msg/String "{data: 'burger1'}"
ros2 topic pub --once /target_robot std_msgs/msg/String "{data: 'burger2'}"
ros2 topic pub --once /target_robot std_msgs/msg/String "{data: 'waffle1'}"
```

Then click a point in RViz using **Publish Point**.

## Auto patrol/rescue test

Start with:

```bash
ros2 launch multi pc_debug.launch.py auto_patrol:=true
```

The three robots will loop through predefined waypoint lists.

Simulate Burger 1 signal loss:

```bash
ros2 topic pub --once /multi/fail_robot std_msgs/msg/String "{data: 'burger1'}"
```

Simulate Burger 2 signal loss:

```bash
ros2 topic pub --once /multi/fail_robot std_msgs/msg/String "{data: 'burger2'}"
```

Recover one robot:

```bash
ros2 topic pub --once /multi/recover_robot std_msgs/msg/String "{data: 'burger1'}"
```

Recover all robots and resume patrol:

```bash
ros2 topic pub --once /multi/recover_robot std_msgs/msg/String "{data: 'all'}"
```

You can also clear failures through `/multi/fail_robot`:

```bash
ros2 topic pub --once /multi/fail_robot std_msgs/msg/String "{data: 'clear'}"
```

## Important topics

| Purpose | Topic |
|---|---|
| RViz clicked point | `/clicked_point` |
| Selected robot command | `/target_robot` |
| Current selected robot | `/selected_robot` |
| Burger 1 goal | `/burger1/goal_point` |
| Burger 2 goal | `/burger2/goal_point` |
| Waffle 1 goal | `/waffle1/goal_point` |
| Burger 1 map pose | `/burger1/map_pose` |
| Burger 2 map pose | `/burger2/map_pose` |
| Waffle 1 map pose | `/waffle1/map_pose` |
| Burger 1 velocity | `/burger1/cmd_vel` |
| Burger 2 velocity | `/burger2/cmd_vel` |
| Waffle 1 velocity | `/waffle1/cmd_vel` |
| Auto patrol/rescue mode | `/multi/mode` |
| Rescue event message | `/multi/rescue_event` |
| Manual failure trigger | `/multi/fail_robot` |
| Manual recovery trigger | `/multi/recover_robot` |
| Map | `/map` |

## Signal-loss logic

`auto_patrol_rescue` supports two signal-loss paths:

1. Manual simulation through `/multi/fail_robot`.
2. Timeout detection when a Burger's `/map_pose` is not received for more than `signal_timeout_sec`.

When a Burger fails:

1. The failed Burger's last known `/map_pose` is saved.
2. The failed Burger is commanded to stop at that last known position.
3. The remaining active robots receive goals near that position.
4. RViz displays failure/rescue markers on `/multi/rescue_markers`.

By default, the two rescuing robots are separated by `rescue_offset_m=0.45` meters around the failed position to reduce overlap. Set `rescue_offset_m:=0.0` in the launch file if both should be sent to the exact same coordinate.

## Note

This is a Gazebo/RViz coordinate-following test. It does not use Nav2 path planning or obstacle avoidance. It is intended to confirm multi-robot goal dispatch, auto patrol behavior, and a simple signal-loss rescue scenario.


## Rescue and map-alignment notes

- `rescue_offset_m` controls whether the second active robot stops slightly away from the failed robot.
  - `rescue_offset_m:=0.0`: both remaining robots are commanded to the exact failed position.
  - `rescue_offset_m:=0.45`: the second robot stops 0.45 m away to avoid overlap.
- `maps/turtlemap.yaml` origin was aligned to the Gazebo `turtlebot3_world` model coordinate frame so RViz map coordinates match Gazebo world coordinates more closely.
