SHELL := /usr/bin/env bash

MODEL_DIR := model
LOG_DIR   := rl_logs/pure_velocity_sac_map64_lidar60_h8_deltatcn_domain22_nopriority_gsde_v022_dt02_b128

.PHONY: help gazebo train all clean clean-models clean-logs clean-shm

help:
	@echo "make gazebo        - Gazebo 시뮬레이션 시작 (터미널 1)"
	@echo "make train         - SAC 학습 시작 (터미널 2, Gazebo 준비된 후)"
	@echo "make clean-models  - $(MODEL_DIR) 삭제 (체크포인트 전부 삭제됨)"
	@echo "make clean-logs    - $(LOG_DIR) 삭제 (학습 로그/CSV 전부 삭제됨)"
	@echo "make clean         - clean-models + clean-logs"
	@echo "make clean-shm     - 남아있는 FastDDS SHM 파일 정리"

gazebo:
	bash run_gazebo.sh

train:
	bash run_train_v132_clean.sh

clean-models:
	@echo "삭제: $(MODEL_DIR)"
	rm -rf "$(MODEL_DIR)"

clean-logs:
	@echo "삭제: $(LOG_DIR)"
	rm -rf "$(LOG_DIR)"

clean: clean-models clean-logs

clean-shm:
	rm -f /dev/shm/fastrtps_* /dev/shm/sem.fastrtps_* /dev/shm/fastdds_* /dev/shm/sem.fastdds_* 2>/dev/null || true
