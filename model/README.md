# Model Directory

Put runtime deep-learning models here.

- `target_v3.engine`: TensorRT export used by default on Jetson -- OMX AIM (`yolo_node`), the leader's Flask YOLO server, and `bayesian_risk_map`. Target class is 0.
- `target_v3.pt`: Ultralytics YOLO checkpoint used as the source for `target_v3.engine`.
- `best.pt` / `best.engine`: older checkpoint/TensorRT export, no longer referenced by any default; kept only for legacy smoke tests.
- `sac_turtlebot3_burger_emergency.zip`: ACTIVE_SCOUT RL policy used by `system_bringup`.
- `sac_turtlebot3_burger.zip`: older/default RL policy checkpoint kept for evaluation and fallback tooling.
