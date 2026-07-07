# tb3_region_mapper v27

Region-aware Cartographer + Nav2 exploration for TurtleBot3.

v27 changes:

- Coverage is updated by a dedicated high-rate timer while the robot is moving.
- Conservative coverage remains: front-FOV + LiDAR-clear + known-free only.
- Candidate selection now uses a rolling priority queue.
- The explorer keeps about 30 top candidates, re-scores them frequently, drops stale/low-score candidates, and streams the best goal to Nav2 immediately.
- Map-expansion, coverage-fill, next-region, and global-frontier goals can all live in the same queue.
- The node still does not publish `/cmd_vel`; Nav2 owns motion control.
- SLAM remains Cartographer only; no slam_toolbox.
