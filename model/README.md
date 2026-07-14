# Model Directory

Put runtime deep-learning models here.

- `target_v3.engine`: TensorRT export used by default on Jetson -- OMX AIM (`yolo_node`), the leader's Flask YOLO server, and `bayesian_risk_map`. Target class is 0.
- `target_v3.pt`: Ultralytics YOLO checkpoint used as the source for `target_v3.engine`.
- `best.pt` / `best.engine`: older YOLO checkpoint/TensorRT export, no longer referenced by any default; kept only for legacy smoke tests.

ACTIVE_SCOUT RL checkpoints live under `rl_models/`. The current runtime
policy is:

`rl_models/pure_velocity_sac_map64_lidar60_h8_deltatcn_domain22_nopriority_gsde_v022_dt02_b128_obs63/sac_turtlebot3_burger.zip`
