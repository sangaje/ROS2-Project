# 문 개방 동작 (VicPinky + OMX-AI)

## 시스템 구성

| 역할 | IP | 유저 | 설명 |
|---|---|---|---|
| Vic Pinky 베이스 | 10.10.14.118 | vic | ROS2 브링업, cmd_vel 수신 |
| OMX 팔 + robot_client | 10.10.14.95 | ububtu | LeRobot 추론 + 그리퍼 모니터링 |
| policy_server | 10.10.14.38 | asd | ACT 정책 추론 서버 |

- **ROS_DOMAIN_ID**: 30
- **lerobot 버전**: 0.4.4 (robot_client / policy_server 동일 버전 필수)
- **모델**: omx_act_policy_3 (ACT, 50000 steps)

## 동작 순서

1. OMX 팔이 LeRobot ACT 정책으로 문 손잡이에 접근 및 파지
2. 그리퍼 모터(ID 16) load값이 50 아래로 떨어지면 파지 성공 판단
3. Vic Pinky에 cmd_vel 발행 -> 2초 직선 후진
4. 후진 완료 후 그리퍼 오픈 -> 종료

## 실행 방법

### 1. Vic Pinky 브링업 (10.10.14.118)
```bash
export ROS_DOMAIN_ID=30
ros2 launch vicpinky_bringup bringup.launch.xml
```

### 2. policy_server (10.10.14.38)
```bash
~/venv/il_044/bin/python -m lerobot.async_inference.policy_server --host 0.0.0.0 --port 8080 --fps 30
```

### 3. robot_client + 오케스트레이터 (10.10.14.95)
터미널 1 - LeRobot 추론
```bash
~/venv/il/bin/python -m lerobot.async_inference.robot_client --robot.type omx_follower --robot.port /dev/ttyACM0 --robot.cameras '{"front": {"type": "opencv", "index_or_path": 0, "width": 640, "height": 480, "fps": 30}, "top": {"type": "opencv", "index_or_path": 2, "width": 640, "height": 480, "fps": 30}}' --task "Pick up Doll" --server_address 10.10.14.38:8080 --policy_type act --pretrained_name_or_path /home/ububtu/il_ws/src/lerobot/outputs/train/omx_act_policy_3/checkpoints/last/pretrained_model --policy_device cuda --actions_per_chunk 100 --chunk_size_threshold 0.7 --aggregate_fn_name weighted_average
```

터미널 2 - 그리퍼 모니터링 + Vic Pinky 후진 트리거
```bash
export ROS_DOMAIN_ID=30
python3 door_open_orchestrator.py
```

## OMX-AI 모터 구성 (OpenRB-150 경유)

| ID | 관절 |
|---|---|
| 11 | 베이스 |
| 12 | 숄더 |
| 13 | 엘보 |
| 14 | 리스트 피치 |
| 15 | 리스트 롤 |
| 16 | 그리퍼 |

## 그리퍼 파지 판단 기준

| 상태 | 모터 load값 |
|---|---|
| 열림 (open) | 57 ~ 59 |
| 파지 성공 | 40 후반대 |
| 판단 임계값 | 50 |

## 파일 구조
