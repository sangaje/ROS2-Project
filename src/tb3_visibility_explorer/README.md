# tb3_visibility_explorer v17_align_then_follow_no_backtrack_reject

Hybrid visibility exploration for TurtleBot3.

v16 changes:
- Rejects movement targets behind the robot.
- Keeps unknown/unmapped regions as high-priority view targets, but does not backtrack into them.
- Rejects Nav2-computed paths whose initial segment points behind the robot.
- LiDAR probe remains unknown-first, but only within a forward/side-forward cone.
- Direct pure-pursuit stops and replans if the active path turns into a rear target.
