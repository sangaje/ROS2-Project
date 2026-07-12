# Model Directory

Put runtime deep-learning models here.

- `target_v3.pt`: Ultralytics YOLO checkpoint used by OMX AIM (`yolo_node`) and the leader's Flask YOLO server for target detection/firing.
- `best.pt` / `best.engine`: risk-map obstacle/YOLO checkpoint used by `bayesian_risk_map` (separate pipeline from OMX targeting).
- `sac_turtlebot3_burger_emergency.zip`: ACTIVE_SCOUT RL policy used by `system_bringup`.
- `sac_turtlebot3_burger.zip`: older/default RL policy checkpoint kept for evaluation and fallback tooling.
