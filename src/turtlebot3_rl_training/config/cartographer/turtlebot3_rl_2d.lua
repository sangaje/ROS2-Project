-- Project-local tweak of the stock turtlebot3_cartographer/config/turtlebot3_lds_2d.lua.
-- Do not edit the system package copy; override cartographer_config_dir /
-- configuration_basename to point here instead.
--
-- Change vs stock: disable global pose graph optimization (loop closure).
-- Symptom this fixes: with SLAM restarted fresh every ~7-9s episode reset,
-- there is rarely enough time/revisit to benefit from loop closure, but the
-- periodic global optimization pass still runs and re-aligns/shifts already
---published submaps. Because /rl_confidence_map is locked onto the live SLAM
-- canvas (_try_lock_internal_grid_to_slam_canvas), each of those re-alignments
-- visibly "trembles" the map and shifts already-painted confidence cells to
-- new grid indices, making previously-explored area look unexplored again.

include "map_builder.lua"
include "trajectory_builder.lua"

options = {
  map_builder = MAP_BUILDER,
  trajectory_builder = TRAJECTORY_BUILDER,
  map_frame = "map",
  tracking_frame = "imu_link",
  published_frame = "odom",
  odom_frame = "odom",
  provide_odom_frame = false,
  publish_frame_projected_to_2d = true,
  use_odometry = true,
  use_nav_sat = false,
  use_landmarks = false,
  num_laser_scans = 1,
  num_multi_echo_laser_scans = 0,
  num_subdivisions_per_laser_scan = 1,
  num_point_clouds = 0,
  lookup_transform_timeout_sec = 0.2,
  submap_publish_period_sec = 0.3,
  pose_publish_period_sec = 5e-3,
  trajectory_publish_period_sec = 30e-3,
  rangefinder_sampling_ratio = 1.,
  odometry_sampling_ratio = 1.,
  fixed_frame_pose_sampling_ratio = 1.,
  imu_sampling_ratio = 1.,
  landmarks_sampling_ratio = 1.,
}

MAP_BUILDER.use_trajectory_builder_2d = true

TRAJECTORY_BUILDER_2D.min_range = 0.12
TRAJECTORY_BUILDER_2D.max_range = 3.5
TRAJECTORY_BUILDER_2D.missing_data_ray_length = 3.
TRAJECTORY_BUILDER_2D.use_imu_data = false
-- v133: tried disabling this (leaning on odometry + ceres local refinement
-- only) to reduce jitter, but it made scan alignment visibly worse (scans
-- landing at inconsistent poses instead of one coherent wall) -- odometry
-- alone isn't a reliable enough prior here. Reverted to stock TB3 behavior.
TRAJECTORY_BUILDER_2D.use_online_correlative_scan_matching = true
-- Stock TB3 lua set this to 0.1 degrees -- 10x tighter than cartographer's
-- own default of 1 degree. That makes the motion filter insert a new scan
-- into the map on almost any tiny rotation, including pure sensor/estimator
-- noise with no real motion. Each such insertion bakes in that scan's small
-- pose error, so over many turns a thin wall visibly thickens/smears and the
-- opposite-side free space grows outward -- which then reads as
-- "unexplored" and pulls the confidence-seeking policy back to it
-- repeatedly. Relax to cartographer's stock default so only scans from
-- actual, meaningful motion get inserted.
TRAJECTORY_BUILDER_2D.motion_filter.max_angle_radians = math.rad(1.)

-- Different lever than disabling correlative matching outright (that made
-- things worse -- reverted above): keep it on, but keep it close to the
-- odometry-seeded prior instead of letting it search far and settle on a
-- spurious "good-scoring" alignment. Stock cartographer defaults here are
-- linear_search_window=0.1m, angular_search_window=20deg, both delta cost
-- weights=1e-1. In this small (~6x3.5m), 30-obstacle room a 20 degree /
-- 0.1m search radius is huge relative to the scene and can lock onto the
-- wrong nearby wall/obstacle edge, especially right after an episode reset
-- when the map is still sparse. Narrow the search and penalize straying
-- from the prior more heavily.
TRAJECTORY_BUILDER_2D.real_time_correlative_scan_matcher.linear_search_window = 0.04
TRAJECTORY_BUILDER_2D.real_time_correlative_scan_matcher.angular_search_window = math.rad(8.)
TRAJECTORY_BUILDER_2D.real_time_correlative_scan_matcher.translation_delta_cost_weight = 1.0
TRAJECTORY_BUILDER_2D.real_time_correlative_scan_matcher.rotation_delta_cost_weight = 1.0

POSE_GRAPH.constraint_builder.min_score = 0.65
POSE_GRAPH.constraint_builder.global_localization_min_score = 0.7

-- Stock had this commented out (falls back to pose_graph.lua's default,
-- which periodically runs global optimization). Set explicitly to 0 to
-- disable it: local scan-matched pose stays put once published instead of
-- being retroactively nudged by loop-closure constraints.
POSE_GRAPH.optimize_every_n_nodes = 0

-- Match the Ceres build's actual thread cap (Raspberry Pi Ceres packages are
-- commonly built with 4 threads). Leaving the upstream default of 7 logs a
-- "Bounding to maximum number available" warning on every solve.
TRAJECTORY_BUILDER_2D.ceres_scan_matcher.ceres_solver_options.num_threads = 4
POSE_GRAPH.optimization_problem.ceres_solver_options.num_threads = 4

return options
