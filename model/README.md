# Model Directory

Put runtime deep-learning models here.

- `target_v3.pt`: Ultralytics YOLO checkpoint used everywhere -- OMX AIM (`yolo_node`), the leader's Flask YOLO server, and `bayesian_risk_map`. Target class is 0.
- `best.pt` / `best.engine`: older checkpoint/TensorRT export, no longer referenced by any default; kept only for `tools/export_best_engine.py` and `tools/test_best_pt_camera.py`.
- `sac_turtlebot3_burger_emergency.zip`: ACTIVE_SCOUT RL policy used by `system_bringup`.
- `sac_turtlebot3_burger.zip`: older/default RL policy checkpoint kept for evaluation and fallback tooling.
