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

## Automatic Scout Failover

`multi` is a centralized Gazebo/RViz test package, not the real
`system_bringup` fleet stack. The active coordinator is
`auto_patrol_rescue`; robot motion is handled by each
`simple_goal_controller` through map-frame `PointStamped` goals on
`/<robot>/goal_point` and `/<robot>/rescue_goal`. There is no runtime
`ROS_DOMAIN_ID` role change and no subprocess `ros2 launch` takeover.

Current role structure:

| Feature | Scout (`burger1`) | Follower/successor (`waffle1`) |
| --- | --- | --- |
| robot base | Gazebo/physical bridge namespace | Gazebo/physical bridge namespace |
| pose publisher | publishes `/burger1/map_pose` | publishes `/waffle1/map_pose` |
| liveness topics | `/burger1/map_pose`, `/burger1/odom`; `/burger1/signal` when launched | same per namespace |
| follow/controller | waypoint patrol point goals | idle until rescue/failover goal |
| Nav2 | optional `region_nav2_goal`; rescue uses existing point controller | same |
| AMCL | not used in this package | not used in this package |
| Cartographer | not used in this package | not used in this package |
| patrol/exploration | waypoint patrol in `auto_patrol_rescue` | dormant until promotion |
| camera/risk | not modeled here | not modeled here |
| domain bridge | dynamic `/tmp` domain_bridge YAML | same |

Scout DOWN detection is aggregate-based. The coordinator keeps
`last_rx_time` for each required liveness source and only enters
`SCOUT_SUSPECTED_DOWN` when all required Scout sources are stale. After
`scout_down_grace_sec`, the last fresh Scout `/map_pose` is frozen as the
death pose. Odom is tracked as liveness only; it is not converted directly
into a recovery goal.

Failover state:

```text
FOLLOWING
  -> SCOUT_SUSPECTED_DOWN
  -> FAILOVER_COMMITTED
  -> NAVIGATING_TO_LAST_SCOUT_POSE
  -> PROMOTING_TO_SCOUT
  -> SCOUT_ACTIVE
```

The recovery goal is sent a bounded number of times to the designated
successor (`waffle1` by default), offset behind the last Scout yaw by
`failover_standoff_m` to avoid driving into the failed robot footprint.
Promotion happens only after the successor reaches the recovery target within
the existing `goal_tolerance`. Once promoted, automatic failback is disabled;
use `/multi/recover_robot` or `/multi/fail_robot` reset/clear to return to the
initial patrol assignment.

Map continuity: in this package the optional `static_map_publisher` publishes
a valid TRANSIENT_LOCAL static `/map` once. `auto_patrol_rescue` subscribes to
`/map` for diagnostics and ignores invalid empty maps (`width == 0`,
`height == 0`, invalid resolution, or mismatched data length) so its last valid
map diagnostic is not overwritten. Seamless Cartographer state takeover is not
implemented here because `multi` has no Cartographer, pbstream, or trajectory
resume mechanism.
