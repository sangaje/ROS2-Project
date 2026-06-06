import argparse
from pathlib import Path

import rclpy
from stable_baselines3 import SAC

from turtlebot3_rl_training.feature_extractor import MapVectorFeatureExtractor
from turtlebot3_rl_training.gazebo_nav_env import GazeboNavEnv
from turtlebot3_rl_training.ros_interface import TurtleBot3RosInterface


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        type=str,
        default="rl_models/sac_turtlebot3_burger.zip",
    )
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--control-dt", type=float, default=0.1)
    parser.add_argument("--physics-step-size", type=float, default=0.005)
    parser.add_argument("--max-episode-steps", type=int, default=300)

    parser.add_argument("--namespace", type=str, default="")
    parser.add_argument("--cmd-vel-topic", type=str, default="cmd_vel")

    parser.add_argument("--entity-name", type=str, default="turtlebot3_burger")
    parser.add_argument(
        "--set-pose-service",
        type=str,
        default="/world/default/set_pose",
    )
    parser.add_argument(
        "--world-control-service",
        type=str,
        default="/world/default/control",
    )

    parser.add_argument("--disable-pose-reset", action="store_true")
    parser.add_argument("--disable-world-step", action="store_true")
    parser.add_argument("--fixed-reset-yaw", action="store_true")
    parser.add_argument("--reset-z", type=float, default=0.05)

    parser.add_argument("--fallen-roll-threshold", type=float, default=0.7)
    parser.add_argument("--fallen-pitch-threshold", type=float, default=0.7)

    parser.add_argument("--max-linear-speed", type=float, default=0.35)
    parser.add_argument("--max-angular-speed", type=float, default=2.0)

    # SLAM 관련 옵션.
    parser.add_argument("--slam-map-topic", type=str, default="/map")
    parser.add_argument("--map-frame", type=str, default="map")
    parser.add_argument("--disable-slam-map", action="store_true")
    parser.add_argument("--auto-start-slam", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--wait-slam-map", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--reset-slam-on-reset", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--restart-slam-on-reset", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--slam-reset-service", type=str, default="/slam_toolbox/reset")
    parser.add_argument("--slam-reset-timeout", type=float, default=8.0)
    parser.add_argument("--slam-reset-warmup-steps", type=int, default=15)
    parser.add_argument("--rl-map-topic", type=str, default="/rl_task_map")
    parser.add_argument("--rl-confidence-topic", type=str, default="/rl_confidence_map")
    parser.add_argument("--rl-priority-topic", type=str, default="/rl_priority_map")
    parser.add_argument("--rl-path-topic", type=str, default="/rl_path")
    parser.add_argument("--rl-filtered-slam-topic", type=str, default="/rl_filtered_slam_map")
    parser.add_argument("--slam-map-accept-delay-sec", type=float, default=1.0)
    parser.add_argument("--slam-map-max-age-sec", type=float, default=3.0)
    parser.add_argument("--reset-x", type=float, default=0.0)
    parser.add_argument("--reset-y", type=float, default=0.0)

    # CNN map observation. Must match the model used during training.
    parser.add_argument("--use-map-cnn", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--map-obs-size", type=int, default=64)
    parser.add_argument("--map-obs-size-m", type=float, default=6.4)
    parser.add_argument("--use-temporal-cnn", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--temporal-history-len", type=int, default=8)
    parser.add_argument("--front-fov-deg", type=float, default=80.0)
    parser.add_argument("--front-angle-sigma-deg", type=float, default=20.0)
    parser.add_argument("--confidence-max-range", type=float, default=2.0)
    parser.add_argument("--seen-confidence-floor", type=float, default=80.0)
    parser.add_argument("--confidence-decay-per-step", type=float, default=0.0)
    parser.add_argument("--suppress-gap-confidence", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--gap-occupied-threshold", type=float, default=65.0)
    parser.add_argument("--gap-check-radius-m", type=float, default=1.20)
    parser.add_argument("--gap-min-width-m", type=float, default=0.20)
    parser.add_argument("--gap-max-width-m", type=float, default=2.00)
    parser.add_argument("--map-expand-chunk-cells", type=int, default=64)
    parser.add_argument("--priority-recompute-interval", type=int, default=8)
    parser.add_argument("--priority-visit-suppression-radius-m", type=float, default=0.55)
    parser.add_argument("--priority-visit-suppression-gain", type=float, default=0.35)
    parser.add_argument("--priority-visit-suppression-max", type=float, default=0.85)
    parser.add_argument("--priority-observed-suppression-gain", type=float, default=0.20)
    parser.add_argument("--priority-clear-fov-deg", type=float, default=90.0)
    parser.add_argument("--priority-clear-max-range-m", type=float, default=1.20)
    parser.add_argument("--priority-clear-robot-radius-m", type=float, default=0.45)
    parser.add_argument("--priority-clear-min-value", type=float, default=5.0)
    parser.add_argument("--priority-clear-sigma-m", type=float, default=0.35)
    parser.add_argument("--priority-clear-angle-sigma-deg", type=float, default=30.0)
    parser.add_argument("--priority-clear-min-weight", type=float, default=0.18)
    parser.add_argument("--priority-clear-visit-sigma-m", type=float, default=0.25)
    parser.add_argument("--wall-support-radius-m", type=float, default=0.70)
    parser.add_argument("--wall-support-density-threshold", type=float, default=0.025)
    parser.add_argument("--open-space-front-distance-m", type=float, default=1.80)
    parser.add_argument("--open-space-side-width-m", type=float, default=1.20)
    parser.add_argument("--open-space-forward-penalty", type=float, default=0.85)
    parser.add_argument("--map-publish-every-n", type=int, default=10)
    parser.add_argument("--map-keepalive-period-sec", type=float, default=1.0)

    # Action filtering / anti-jitter. Must match training-time semantics when possible.
    parser.add_argument("--action-smoothing-alpha", type=float, default=0.30)
    parser.add_argument("--max-linear-delta", type=float, default=0.08)
    parser.add_argument("--max-angular-delta", type=float, default=0.20)
    parser.add_argument("--linear-deadband", type=float, default=0.015)
    parser.add_argument("--angular-deadband", type=float, default=0.04)
    parser.add_argument("--enable-motion-mode-hysteresis", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--explored-stall-start-steps", type=int, default=8)
    parser.add_argument("--explored-stall-growth", type=float, default=0.008)
    parser.add_argument("--explored-stall-power", type=float, default=1.45)
    parser.add_argument("--explored-stall-max-penalty", type=float, default=1.20)

    parsed_args, _ = parser.parse_known_args()
    return parsed_args


def main(args=None):
    cli_args = parse_args()

    model_path = Path(cli_args.model)

    if not model_path.exists():
        print(f"Model file not found: {model_path}")
        return

    rclpy.init(args=args)

    use_slam_map = not cli_args.disable_slam_map
    map_topic = cli_args.slam_map_topic if use_slam_map else ""

    ros = TurtleBot3RosInterface(
        namespace=cli_args.namespace,
        cmd_vel_topic=cli_args.cmd_vel_topic,
        map_topic=map_topic,
        enable_tf=use_slam_map,
        auto_start_slam=use_slam_map and cli_args.auto_start_slam,
        slam_reset_service=cli_args.slam_reset_service,
    )

    if not ros.wait_for_sensor_ready(timeout_sec=10.0):
        ros.destroy_node()
        rclpy.shutdown()
        return

    if use_slam_map and cli_args.auto_start_slam:
        ros.ensure_slam_toolbox(timeout_sec=cli_args.slam_reset_timeout)

    if use_slam_map and cli_args.wait_slam_map:
        ros.wait_for_slam_map_ready(timeout_sec=max(10.0, cli_args.slam_reset_timeout))

    env = GazeboNavEnv(
        ros_interface=ros,
        entity_name=cli_args.entity_name,
        set_pose_service=cli_args.set_pose_service,
        enable_pose_reset=not cli_args.disable_pose_reset,
        random_reset_yaw=not cli_args.fixed_reset_yaw,
        reset_z=cli_args.reset_z,
        control_dt=cli_args.control_dt,
        physics_step_size=cli_args.physics_step_size,
        max_episode_steps=cli_args.max_episode_steps,
        fallen_roll_threshold=cli_args.fallen_roll_threshold,
        fallen_pitch_threshold=cli_args.fallen_pitch_threshold,
        world_control_service=cli_args.world_control_service,
        use_world_step=not cli_args.disable_world_step,
        max_linear_speed=cli_args.max_linear_speed,
        max_angular_speed=cli_args.max_angular_speed,
        use_slam_map=use_slam_map,
        map_frame=cli_args.map_frame,
        rl_map_topic=cli_args.rl_map_topic,
        rl_confidence_topic=cli_args.rl_confidence_topic,
        rl_priority_topic=cli_args.rl_priority_topic,
        rl_path_topic=cli_args.rl_path_topic,
        rl_filtered_slam_topic=cli_args.rl_filtered_slam_topic,
        slam_map_accept_delay_sec=cli_args.slam_map_accept_delay_sec,
        slam_map_max_age_sec=cli_args.slam_map_max_age_sec,
        reset_x=cli_args.reset_x,
        reset_y=cli_args.reset_y,
        reset_slam_on_reset=use_slam_map and cli_args.reset_slam_on_reset,
        restart_slam_on_reset=use_slam_map and cli_args.restart_slam_on_reset,
        slam_reset_timeout_sec=cli_args.slam_reset_timeout,
        slam_reset_warmup_steps=cli_args.slam_reset_warmup_steps,
        use_map_cnn=cli_args.use_map_cnn,
        map_obs_size=cli_args.map_obs_size,
        map_obs_size_m=cli_args.map_obs_size_m,
        use_temporal_cnn=cli_args.use_temporal_cnn,
        temporal_history_len=cli_args.temporal_history_len,
        front_fov_deg=cli_args.front_fov_deg,
        front_angle_sigma_deg=cli_args.front_angle_sigma_deg,
        confidence_max_range=cli_args.confidence_max_range,
        seen_confidence_floor=cli_args.seen_confidence_floor,
        confidence_decay_per_step=cli_args.confidence_decay_per_step,
        suppress_gap_confidence=cli_args.suppress_gap_confidence,
        gap_occupied_threshold=cli_args.gap_occupied_threshold,
        gap_check_radius_m=cli_args.gap_check_radius_m,
        gap_min_width_m=cli_args.gap_min_width_m,
        gap_max_width_m=cli_args.gap_max_width_m,
        map_expand_chunk_cells=cli_args.map_expand_chunk_cells,
        map_publish_every_n=cli_args.map_publish_every_n,
        priority_recompute_interval=cli_args.priority_recompute_interval,
        priority_visit_suppression_radius_m=cli_args.priority_visit_suppression_radius_m,
        priority_visit_suppression_gain=cli_args.priority_visit_suppression_gain,
        priority_visit_suppression_max=cli_args.priority_visit_suppression_max,
        priority_observed_suppression_gain=cli_args.priority_observed_suppression_gain,
        priority_clear_fov_deg=cli_args.priority_clear_fov_deg,
        priority_clear_max_range_m=cli_args.priority_clear_max_range_m,
        priority_clear_robot_radius_m=cli_args.priority_clear_robot_radius_m,
        priority_clear_min_value=cli_args.priority_clear_min_value,
        priority_clear_sigma_m=cli_args.priority_clear_sigma_m,
        priority_clear_angle_sigma_deg=cli_args.priority_clear_angle_sigma_deg,
        priority_clear_min_weight=cli_args.priority_clear_min_weight,
        priority_clear_visit_sigma_m=cli_args.priority_clear_visit_sigma_m,
        wall_support_radius_m=cli_args.wall_support_radius_m,
        wall_support_density_threshold=cli_args.wall_support_density_threshold,
        open_space_front_distance_m=cli_args.open_space_front_distance_m,
        open_space_side_width_m=cli_args.open_space_side_width_m,
        open_space_forward_penalty=cli_args.open_space_forward_penalty,
        map_keepalive_period_sec=cli_args.map_keepalive_period_sec,
        action_smoothing_alpha=cli_args.action_smoothing_alpha,
        max_linear_delta=cli_args.max_linear_delta,
        max_angular_delta=cli_args.max_angular_delta,
        linear_deadband=cli_args.linear_deadband,
        angular_deadband=cli_args.angular_deadband,
        enable_motion_mode_hysteresis=cli_args.enable_motion_mode_hysteresis,
        explored_stall_start_steps=cli_args.explored_stall_start_steps,
        explored_stall_growth=cli_args.explored_stall_growth,
        explored_stall_power=cli_args.explored_stall_power,
        explored_stall_max_penalty=cli_args.explored_stall_max_penalty,
    )

    # Importing MapVectorFeatureExtractor above ensures custom extractor class is available
    # when loading a model trained with MultiInputPolicy.
    _ = MapVectorFeatureExtractor
    model = SAC.load(str(model_path))

    try:
        for episode in range(cli_args.episodes):
            obs, info = env.reset()

            total_reward = 0.0
            done = False
            step_count = 0

            while not done:
                action, _ = model.predict(obs, deterministic=True)

                obs, reward, terminated, truncated, info = env.step(action)

                total_reward += reward
                step_count += 1

                done = terminated or truncated

            ros.get_logger().info(
                "Episode "
                f"{episode + 1}/{cli_args.episodes} | "
                f"reward={total_reward:.3f} | "
                f"steps={step_count} | "
                f"coverage={info.get('coverage_ratio', -1.0):.4f} | "
                f"new_cells={info.get('new_known_cells', -1)} | "
                f"mean_conf={info.get('mean_confidence', -1.0):.2f} | "
                f"conf_gain={info.get('confidence_gain', -1.0):.3f} | "
                f"stale={info.get('stale_known_cells', -1)} | "
                f"stale_refresh={info.get('stale_refresh_cells', -1)} | "
                f"low_conf={info.get('low_confidence_cells', -1)} | "
                f"priority={info.get('priority_score', -1.0):.2f} | "
                f"clear={info.get('priority_cleared_cells', -1)}:{info.get('priority_clear_gain', -1.0):.2f} | "
                f"frontiers={info.get('frontier_count', -1)} | "
                f"target={info.get('target_type', 'none')}:{info.get('target_priority', -1.0):.2f} | "
                f"slam={info.get('slam_map_available', False)} | "
                f"collision={info.get('collision', False)} | "
                f"fallen={info.get('fallen', False)} | "
                f"coverage_done={info.get('coverage_done', False)} | "
                f"sim_time={info.get('sim_time', -1.0):.3f}"
            )

    finally:
        env.close()
        ros.stop_robot()
        ros.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
