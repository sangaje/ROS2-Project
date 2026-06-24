#!/usr/bin/env bash
# TurtleBot3 SAC 학습 실행 스크립트 (안정화 패치 v2 반영)
#
# 사용법:
#   먼저 별도 터미널에서 Gazebo/TurtleBot3 시뮬레이션을 띄운 뒤,
#   이 스크립트를 실행하세요. (기존 워크플로우와 동일)
#
#   chmod +x run_train.sh
#   ./run_train.sh
#
# 변경점 요약:
#   - FastDDS SHM 전송 비활성화는 코드가 rclpy.init 전에 자동 적용하지만,
#     안전을 위해 아래에서 환경변수로도 명시합니다(이중 안전).
#   - buffer-size는 메모리 가드가 자동으로 안전 범위로 클램프합니다.
#     (map_seq 시계열 관찰 때문에 너무 큰 값은 수백 GiB까지 갈 수 있음)

set -e

cd ~/Desktop/ROS2_Project
source /opt/ros/jazzy/setup.zsh 2>/dev/null || source /opt/ros/jazzy/setup.bash
source install/setup.zsh 2>/dev/null || source install/setup.bash

# ===== ROS / DDS =====
export ROS_DOMAIN_ID=22
export TURTLEBOT3_MODEL=burger
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export TB3_RL_MODEL_ODOM_TOPIC=/model/burger/odometry

# ===== 안정화 패치 v2: FastDDS SHM 전송 비활성화 =====
# init_port fastrtps_portNNNN / open_and_lock_file failed 크래시의 근본 차단.
# 코드에서도 자동 적용되지만 여기서 명시해 둡니다. (끄려면 0)
export TB3_RL_DISABLE_SHM_TRANSPORT=1
# cartographer 재시작 정착 대기(초). 너무 짧으면 새 /map 지연 가능.
export TB3_RL_CARTOGRAPHER_RESTART_DELAY_SEC=0.8
# replay buffer 메모리 가드(자동 클램프). 1이면 비활성화.
export TB3_RL_DISABLE_BUFFER_GUARD=0
# 필요 시 메모리 예산을 직접 지정(GiB). 미지정 시 시스템 RAM의 ~35%.
# export TB3_RL_BUFFER_MEM_BUDGET_GIB=12

# ===== LiDAR =====
export TB3_RL_LIDAR_CANONICAL_FRONT_ZERO=1
export TB3_RL_LIDAR_FRONT_INDEX=0
export TB3_RL_LIDAR_ANGLE_OFFSET_DEG=0
export TB3_RL_LIDAR_FLIP_LR=0
export TB3_RL_LIDAR_UNIFORM_ANGLE_RESAMPLE=1
export TB3_RL_LIDAR_MEDIAN_KERNEL=3
export TB3_RL_LIDAR_LOWPASS_KERNEL=5
export TB3_RL_LIDAR_OBSTACLE_MARGIN_M=0.08

export TB3_RL_POLICY_SCAN_TOPIC=/rl_policy_scan
export TB3_RL_POLICY_SCAN_60_TOPIC=/rl_policy_scan_60
export TB3_RL_POLICY_SCAN_PUBLISH_EVERY_N=5
export TB3_RL_POLICY_SCAN_MARKER_TOPIC=/rl_policy_scan_60_points
export TB3_RL_RAW_SCAN_MARKER_TOPIC=/rl_raw_scan_points
export TB3_RL_RAW_SCAN_MARKER_UNCORRECTED=0

# ===== priority =====
export TB3_RL_FORCE_NO_PRIORITY=0
export TB3_RL_NO_PRIORITY_MODEL_INPUT=0
export TB3_RL_PRIORITY_CLUSTER_SPAWN_INTERVAL_STEPS=200
export TB3_RL_PRIORITY_MAX_SEED_POINTS=6
export TB3_RL_PRIORITY_SPAWN_MIN_RANGE_M=0.90
export TB3_RL_PRIORITY_SPAWN_MAX_RANGE_M=2.40
export TB3_RL_PRIORITY_SPAWN_IN_CURRENT_FOV=0
export TB3_RL_PRIORITY_BIRTH_DELTA=6.0

# ===== logging =====
export TB3_RL_QUIET_MAP_LOGS=1
export TB3_RL_QUIET_STARTUP_LOGS=1
# confidence 맵 origin을 RViz 로봇 위치(tf2 buffer, map->base)와 정확히 일치시키기
# tf2 우선 사용(RViz와 동일), TF 공백 시에만 manual 캐시 fallback
export TB3_RL_CONFIDENCE_PREFER_TF_BUFFER=1
export TB3_RL_USE_MANUAL_TF_CACHE_FOR_CONFIDENCE=1
export TB3_RL_CONFIDENCE_TF_BUFFER_FALLBACK=1
# CONFIDENCE_PUBLISH 로그 끄기 (0 = 완전 비활성화)
export TB3_RL_CONFIDENCE_PUBLISH_DEBUG_EVERY_N=0
# PRIORITY_BIRTH_THROTTLE / PRIORITY_RECOMPUTE 로그 끄기 (1 = 끔)
export TB3_RL_QUIET_PRIORITY_LOGS=1
# 매 episode reset/SLAM 재시작 반복 로그 끄기 (1 = 끔)
export TB3_RL_QUIET_RESET_LOGS=1
# RViz 디버그 오버레이(/rl_debug_overlay) 발행 주기 (1 = 매 step)
export TB3_RL_DEBUG_OVERLAY_EVERY_N=1

python3 -m turtlebot3_rl_training.train_sac \
    --timesteps 1000000000 \
    --learning-starts 1000 \
    --buffer-size 20000 \
    --batch-size 32 \
    --train-freq-steps 32 \
    --gradient-steps 1 \
    --sac-gamma 0.90 \
    --reward-gamma 0.90 \
    --control-dt 0.10 \
    --physics-step-size 0.005 \
    --max-episode-steps 10000 \
    --entity-name burger \
    --set-pose-service /world/default/set_pose \
    --world-control-service /world/default/control \
    --action-mode velocity \
    --cmd-vel-topic /cmd_vel \
    --max-linear-speed 0.60 \
    --max-angular-speed 0.60 \
    --velocity-command-linear-limit 0.60 \
    --velocity-command-angular-limit 0.60 \
    --velocity-safety-backup \
    --velocity-safety-trigger-distance-m 0.19 \
    --velocity-safety-stop-distance-m 0.24 \
    --velocity-safety-slow-distance-m 0.37 \
    --velocity-safety-backup-speed-mps 0.07 \
    --velocity-safety-turn-speed 0.30 \
    --velocity-safety-backup-steps 18 \
    --velocity-safety-cooldown-steps 10 \
    --velocity-safety-penalty 10.0 \
    --slam-backend cartographer \
    --slam-map-topic /map \
    --map-frame map \
    --pose-frame map \
    --safety-boundary-frame odom \
    --auto-start-slam \
    --wait-slam-map \
    --reset-slam-on-reset \
    --restart-slam-on-reset \
    --reset-slam-every-n-episodes 1 \
    --reset-tf-buffer-on-reset \
    --strict-slam-map-required \
    --strict-slam-map-wait-timeout-sec 60.0 \
    --strict-slam-map-min-known-cells 80 \
    --strict-slam-map-min-known-ratio 0.003 \
    --post-reset-ready-gate \
    --post-reset-ready-timeout-sec 12.0 \
    --post-reset-ready-min-known-ratio 0.003 \
    --post-reset-ready-min-known-cells 40 \
    --post-reset-ready-min-lidar-beams 30 \
    --no-post-reset-ready-require-priority \
    --reset-pose-mode list \
    --reset-pose-list="-2.80,0.96;5.00,0.86" \
    --reset-pose-min-clearance-m 0.13 \
    --rl-map-topic /rl_task_map \
    --rl-confidence-topic /rl_confidence_map \
    --rl-priority-topic /rl_priority_map \
    --rl-filtered-slam-topic /rl_filtered_slam_map \
    --waypoint-marker-topic /rl_debug_overlay \
    --map-publish-every-n 2 \
    --map-live-update-period-sec 0.01 \
    --map-keepalive-period-sec 0.50 \
    --use-map-cnn \
    --map-obs-size 32 \
    --map-obs-size-m 6.0 \
    --num-lidar-bins 60 \
    --use-temporal-cnn \
    --temporal-history-len 8 \
    --temporal-features-dim 64 \
    --cnn-features-dim 48 \
    --vector-features-dim 96 \
    --combined-features-dim 128 \
    --front-fov-deg 60.0 \
    --front-angle-sigma-deg 20.0 \
    --confidence-max-range 2.0 \
    --seen-confidence-floor 80.0 \
    --confidence-reward-weight 10.0 \
    --slam-map-update-reward \
    --slam-map-update-reward-weight 0.65 \
    --enable-corridor-priority-reward \
    --corridor-priority-reward-weight 2.75 \
    --priority-recompute-interval 4 \
    --priority-clear-fov-deg 60.0 \
    --priority-clear-max-range-m 2.0 \
    --priority-clear-min-weight 0.05 \
    --priority-stuck-restart \
    --priority-stuck-restart-steps 200 \
    --coverage-stall-terminal \
    --coverage-stall-start-steps 1000 \
    --coverage-stall-window-steps 500 \
    --coverage-stall-min-slam-new-cells 5 \
    --coverage-stall-min-confidence-updated-cells 30 \
    --coverage-stall-terminal-penalty -10.0 \
    --lidar-empty-restart \
    --lidar-empty-timeout-sec 2.5 \
    --lidar-empty-grace-sec 1.0 \
    --lidar-empty-min-valid-beams 2 \
    --collision-threshold 0.16 \
    --restart-on-collision \
    --no-terminate-on-out-of-bounds \
    --safety-boundary-radius-m 8.0 \
    --model-dir rl_models/pure_velocity_sac_map32_lidar60_h8_smallbuf_domain22 \
    --log-dir rl_logs/pure_velocity_sac_map32_lidar60_h8_smallbuf_domain22 \
    --checkpoint-freq 25000 \
    --show-training-progress \
    --progress-print-freq 2000 \
    --progress-window 20 \
    --debug-print-freq 0 \
    --no-check-env
