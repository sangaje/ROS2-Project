# scout_map_risk_bridge v3

Bridge scout `/map` and `/risk/risk_map` across ROS_DOMAIN_ID.

v3 changes:

- Stores latest `/map` and `/risk/risk_map`.
- Republishes cached latest messages every 1 second by default.
- This makes `ros2 topic echo --once /scout/map` work even if the original sample was already published.

Run:

```zsh
ros2 launch scout_map_risk_bridge scout_map_risk_bridge.launch.py \
  from_domain:=21 \
  to_domain:=20 \
  map_in:=/map \
  risk_in:=/risk/risk_map \
  map_out:=/scout/map \
  risk_out:=/scout/risk_map \
  target_frame:=scout_map
```
