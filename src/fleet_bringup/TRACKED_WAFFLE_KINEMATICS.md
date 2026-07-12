# Tracked Waffle Kinematics

This repo now treats the real leader Waffle as a tracked differential-drive
chassis with one source-of-truth file:

`src/fleet_bringup/config/tracked_waffle_kinematics.yaml`

Initial effective values:

```yaml
effective_wheel_radius: 0.040
effective_track_separation: 0.447
```

The physical track spacing is still about `0.287 m`; the larger effective
separation accounts for track scrub during differential turns.

## Runtime Path Found In This Workspace

- `system_bringup/launch/system.launch.py` includes
  `fleet_bringup/launch/leader.launch.py` for `role:=leader`.
- `leader.launch.py` includes `fleet_bringup/launch/base.launch.py`.
- `base.launch.py` includes upstream
  `turtlebot3_bringup/launch/robot.launch.py`.
- Upstream `robot.launch.py` starts `turtlebot3_node/turtlebot3_ros` with
  `tb3_param_dir`.
- `turtlebot3_node` subscribes to `cmd_vel`; with
  `enable_stamped_cmd_vel: true`, that is `geometry_msgs/msg/TwistStamped`.
- `turtlebot3_node/src/turtlebot3.cpp` writes received `linear.x` and
  `angular.z` to the OpenCR control table fields `cmd_velocity_linear_x` and
  `cmd_velocity_angular_z`.
- `turtlebot3_node/src/odometry.cpp` publishes `/odom` from wheel joint
  states. Linear distance uses `wheels.radius`.
- With `odometry.use_imu: true`, `/odom` yaw follows IMU angle, not
  wheel-only `wheels.separation`. Use the calibration YAML only when testing
  wheel-only yaw.
- No OpenCR firmware source (`turtlebot3_core`) was present in this workspace.

## OpenCR Priority

Best fix:

1. Get the TurtleBot3 OpenCR firmware source (`turtlebot3_core`).
2. Change the firmware differential-drive constants to:

```text
wheel_radius = 0.040
wheel_separation = 0.447
```

3. Verify inverse kinematics:

```text
omega_R = (v + omega * L / 2) / r
omega_L = (v - omega * L / 2) / r
```

4. Flash OpenCR.
5. In `tracked_waffle_kinematics.yaml`, set:

```yaml
tracked_waffle_kinematics:
  ros__parameters:
    opencr_kinematics_corrected: true

tracked_cmd_vel_adapter:
  ros__parameters:
    enabled: false
    linear_gain: 1.0
    angular_gain: 1.0
```

Rollback: restore the previous firmware or set the YAML back to
`opencr_kinematics_corrected: false` and `tracked_cmd_vel_adapter.enabled:
true`.

## Fallback Cmd Path Before OpenCR Is Corrected

When `tracked_cmd_vel_adapter.enabled: true`:

```text
Nav2 controller / leader_shadow_follow / localization kickstart
  -> /cmd_vel_nav
  -> tracked_cmd_vel_adapter
  -> /cmd_vel
  -> turtlebot3_node/OpenCR
```

Only `tracked_cmd_vel_adapter` should publish final hardware `/cmd_vel`.

Initial fallback gains:

```yaml
linear_gain: 0.825
angular_gain: 1.286
```

These are disabled after OpenCR is corrected.

## Calibration

Straight distance:

```bash
ros2 run fleet_bringup tracked_waffle_calibration straight \
  --current-radius 0.040 \
  --sample 2.000:2.080 \
  --sample 2.000:2.060 \
  --sample 2.000:2.070
```

Rotation after radius calibration:

```bash
ros2 run fleet_bringup tracked_waffle_calibration rotate \
  --current-separation 0.447 \
  --sample 1800:1710 \
  --sample 1800:1720 \
  --sample 1800:1715
```

Use at least 2 m forward/backward distance tests and 360 degree or 5-turn
rotation tests. Use the median result.
