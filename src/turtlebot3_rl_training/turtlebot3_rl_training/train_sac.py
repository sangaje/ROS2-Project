import argparse
from pathlib import Path

import rclpy
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor

from turtlebot3_rl_training.feature_extractor import MapVectorFeatureExtractor
from turtlebot3_rl_training.gazebo_nav_env import GazeboNavEnv
from turtlebot3_rl_training.ros_interface import TurtleBot3RosInterface


class DebugCallback(BaseCallback):
    def __init__(self, print_freq: int = 100, verbose: int = 0):
        super().__init__(verbose)
        self.print_freq = int(print_freq)

    def _on_step(self) -> bool:
        if self.print_freq <= 0:
            return True

        if self.num_timesteps % self.print_freq == 0:
            infos = self.locals.get("infos", [])
            rewards = self.locals.get("rewards", [])
            dones = self.locals.get("dones", [])

            info = infos[0] if len(infos) > 0 else {}
            reward = rewards[0] if len(rewards) > 0 else 0.0
            done = dones[0] if len(dones) > 0 else False

            print(
                "[SAC DEBUG] "
                f"t={self.num_timesteps} | "
                f"reward={float(reward):.4f} | "
                f"done={bool(done)} | "
                f"coverage={info.get('coverage_ratio', -1.0):.4f} | "
                f"new_cells={info.get('new_known_cells', -1)} | "
                f"mean_conf={info.get('mean_confidence', -1.0):.2f} | "
                f"conf_gain={info.get('confidence_gain', -1.0):.3f} | "
                f"stale={info.get('stale_known_cells', -1)} | "
                f"stale_refresh={info.get('stale_refresh_cells', -1)} | "
                f"low_conf={info.get('low_confidence_cells', -1)} | "
                f"priority={info.get('priority_score', -1.0):.2f} | "
                f"clear={info.get('priority_cleared_cells', -1)}:{info.get('priority_clear_gain', -1.0):.2f} | "
                f"invalid={info.get('priority_invalidated_cells', -1)}:{info.get('priority_invalidated_gain', -1.0):.2f} | "
                f"wall={info.get('wall_support_score', -1.0):.2f} | "
                f"open={info.get('open_space_score', -1.0):.2f} | "
                f"stall={info.get('explored_stall_steps', -1)} | "
                f"frontiers={info.get('frontier_count', -1)} | "
                f"target={info.get('target_type', 'none')}:{info.get('target_priority', -1.0):.2f} | "
                f"angle={info.get('frontier_angle', 0.0):+.2f} | "
                f"path={info.get('target_reachable', False)}:"
                f"{info.get('path_distance', -1.0):.2f}:"
                f"{info.get('path_progress', 0.0):+.3f} | "
                f"perr={info.get('action_path_error', 0.0):+.2f}:"
                f"{info.get('action_path_alignment', 0.0):.2f}:"
                f"{info.get('action_path_signed', 0.0):+.2f} | "
                f"sw={info.get('target_switched', False)}:{info.get('target_lock_age', -1)} | "
                f"slam={info.get('slam_map_available', False)}:{info.get('slam_map_gate', 'n/a')} "
                f"age={info.get('slam_map_age_sec', -1.0):.2f} "
                f"delay={info.get('slam_map_delay_remaining_sec', 0.0):.2f} | "
                f"collision={info.get('collision', False)} | "
                f"fallen={info.get('fallen', False)} | "
                f"sim_time={info.get('sim_time', -1.0):.3f}",
                flush=True,
            )

        return True


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--timesteps", type=int, default=100_000)
    parser.add_argument("--learning-starts", type=int, default=1_000)
    parser.add_argument("--buffer-size", type=int, default=100_000)
    parser.add_argument("--batch-size", type=int, default=256)
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

    # CNN map observation.
    parser.add_argument("--use-map-cnn", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--map-obs-size", type=int, default=64)
    parser.add_argument("--map-obs-size-m", type=float, default=6.4)
    parser.add_argument("--cnn-features-dim", type=int, default=64)
    parser.add_argument("--vector-features-dim", type=int, default=160)
    parser.add_argument("--combined-features-dim", type=int, default=256)
    parser.add_argument("--policy-weight-decay", type=float, default=1e-5)
    parser.add_argument("--use-temporal-cnn", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--temporal-history-len", type=int, default=8)
    parser.add_argument("--temporal-features-dim", type=int, default=160)
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
    parser.add_argument("--map-publish-every-n", type=int, default=10)
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
    parser.add_argument("--map-keepalive-period-sec", type=float, default=1.0)

    # Training throughput. Off-policy SAC does not require a gradient step after every env step.
    parser.add_argument("--train-freq-steps", type=int, default=4)
    parser.add_argument("--gradient-steps", type=int, default=1)

    # Action filtering / anti-jitter.
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

    parser.add_argument("--no-check-env", action="store_true")

    parser.add_argument("--model-dir", type=str, default="rl_models")
    parser.add_argument("--log-dir", type=str, default="rl_logs")
    parser.add_argument("--debug-print-freq", type=int, default=100)
    parser.add_argument("--checkpoint-freq", type=int, default=10_000)
    parser.add_argument("--save-replay-buffer", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--save-vecnormalize", action=argparse.BooleanOptionalAction, default=False)

    parsed_args, _ = parser.parse_known_args()
    return parsed_args


def main(args=None):
    cli_args = parse_args()

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

    raw_env = GazeboNavEnv(
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

    if not cli_args.no_check_env:
        ros.get_logger().info("Running Stable-Baselines3 env checker...")
        check_env(raw_env, warn=True)
        ros.get_logger().info("Env checker passed.")

    env = Monitor(raw_env)

    model_dir = Path(cli_args.model_dir)
    log_dir = Path(cli_args.log_dir)

    model_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    policy_kwargs = dict(
        # Smaller actor/critic. The feature extractor already encodes map, LiDAR,
        # and temporal state; oversized SAC heads tend to overfit a fixed Gazebo
        # world and amplify reward exploits.
        net_arch=dict(
            pi=[128, 128],
            qf=[128, 128],
        ),
        optimizer_kwargs=dict(
            weight_decay=float(cli_args.policy_weight_decay),
        ),
    )

    policy_name = "MlpPolicy"

    if cli_args.use_map_cnn:
        policy_name = "MultiInputPolicy"
        policy_kwargs.update(
            features_extractor_class=MapVectorFeatureExtractor,
            features_extractor_kwargs=dict(
                map_features_dim=cli_args.cnn_features_dim,
                vector_features_dim=cli_args.vector_features_dim,
                temporal_features_dim=cli_args.temporal_features_dim,
                combined_features_dim=cli_args.combined_features_dim,
                use_temporal_cnn=cli_args.use_temporal_cnn,
            ),
        )

    ros.get_logger().info(
        f"SAC policy={policy_name}, use_map_cnn={cli_args.use_map_cnn}, "
        f"map_obs_shape=(5,{cli_args.map_obs_size},{cli_args.map_obs_size}), "
        f"use_temporal_cnn={cli_args.use_temporal_cnn}, "
        f"temporal_history_len={cli_args.temporal_history_len}, "
        f"front_fov_deg={cli_args.front_fov_deg}, "
        f"cnn_features_dim={cli_args.cnn_features_dim}, "
        f"vector_features_dim={cli_args.vector_features_dim}, "
        f"temporal_features_dim={cli_args.temporal_features_dim}, "
        f"map_encoder=onehot_slam_geometry_robot_centric_5ch, "
        f"priority_topic={cli_args.rl_priority_topic}, "
        f"path_topic={cli_args.rl_path_topic}, "
        f"filtered_slam_topic={cli_args.rl_filtered_slam_topic}, "
        f"slam_map_accept_delay={cli_args.slam_map_accept_delay_sec:.2f}s, "
        f"slam_map_max_age={cli_args.slam_map_max_age_sec:.2f}s, "
        f"gap_confidence_suppression=disabled, "
        f"priority_gap_width=[{cli_args.gap_min_width_m:.2f},{cli_args.gap_max_width_m:.2f}]m, "
        f"map_expand_chunk_cells={cli_args.map_expand_chunk_cells}, "
        f"map_publish_every_n={cli_args.map_publish_every_n}, "
        f"priority_recompute_interval={cli_args.priority_recompute_interval}, "
        f"wall_support_radius={cli_args.wall_support_radius_m:.2f}m, "
        f"wall_support_density_threshold={cli_args.wall_support_density_threshold:.3f}, "
        f"open_space_front_distance={cli_args.open_space_front_distance_m:.2f}m, "
        f"open_space_side_width={cli_args.open_space_side_width_m:.2f}m, "
        f"open_space_forward_penalty={cli_args.open_space_forward_penalty:.2f}, "
        f"priority_suppression_radius={cli_args.priority_visit_suppression_radius_m:.2f}m, "
        f"priority_suppression_gain={cli_args.priority_visit_suppression_gain:.2f}, "
        f"priority_observed_suppression_gain={cli_args.priority_observed_suppression_gain:.2f}, "
        f"priority_clear_fov={cli_args.priority_clear_fov_deg:.1f}deg, "
        f"priority_clear_range={cli_args.priority_clear_max_range_m:.2f}m, "
        f"priority_clear_sigma={cli_args.priority_clear_sigma_m:.2f}m, "
        f"priority_clear_min_weight={cli_args.priority_clear_min_weight:.2f}, "
        f"sac_net_arch=pi[128,128]/qf[128,128], "
        f"train_freq_steps={cli_args.train_freq_steps}, "
        f"gradient_steps={cli_args.gradient_steps}, "
        f"save_replay_buffer={cli_args.save_replay_buffer}, "
        f"weight_decay={cli_args.policy_weight_decay}"
    )

    model = SAC(
        policy=policy_name,
        env=env,
        learning_rate=3e-4,
        buffer_size=cli_args.buffer_size,
        learning_starts=cli_args.learning_starts,
        batch_size=cli_args.batch_size,
        tau=0.005,
        gamma=0.99,
        train_freq=(max(int(cli_args.train_freq_steps), 1), "step"),
        gradient_steps=max(int(cli_args.gradient_steps), 0),
        ent_coef="auto",
        verbose=1,
        tensorboard_log=str(log_dir),
        policy_kwargs=policy_kwargs,
    )

    debug_callback = DebugCallback(print_freq=cli_args.debug_print_freq)

    checkpoint_callback = CheckpointCallback(
        save_freq=cli_args.checkpoint_freq,
        save_path=str(model_dir),
        name_prefix="sac_turtlebot3_burger_checkpoint",
        save_replay_buffer=bool(cli_args.save_replay_buffer),
        save_vecnormalize=bool(cli_args.save_vecnormalize),
    )

    try:
        model.learn(
            total_timesteps=cli_args.timesteps,
            callback=[checkpoint_callback, debug_callback],
            log_interval=1,
        )

        save_path = model_dir / "sac_turtlebot3_burger"
        model.save(str(save_path))

        ros.get_logger().info(f"Saved model to {save_path}.zip")

    finally:
        ros.stop_robot()
        env.close()
        ros.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
