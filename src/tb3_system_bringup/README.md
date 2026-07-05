# tb3_system_bringup

로봇의 역할(정찰/와플)에 맞춰 `tb3_fleet_bringup` + 베이지안 리스크맵 +
RL 정책까지 한 번에 켜는 오케스트레이터. 로봇마다 이 패키지의 런치 파일
하나만 실행하면 된다.

- `system.launch.py`: 역할별 전체 스택
- `viewer.launch.py`: PC에서 fleet 디버그 마커 + 리스크맵을 하나의 RViz로
  같이 보는 뷰어 (로봇을 켜지 않음)

## 실행

정찰봇:

```bash
export ROS_DOMAIN_ID=26
ros2 launch tb3_system_bringup system.launch.py \
  role:=scout main_domain_id:=24
```

와플 (베타, 아직 정찰 behavior 미통합 — 지금은 fleet bringup만 뜬다):

```bash
export ROS_DOMAIN_ID=24
ros2 launch tb3_system_bringup system.launch.py role:=waffle
```

PC 통합 뷰어 (fleet 디버그 마커 + 리스크맵을 같은 RViz 창에서):

```bash
export ROS_DOMAIN_ID=24
ros2 launch tb3_system_bringup viewer.launch.py
```

`system.launch.py`에 `start_rviz:=true`를 주면 스카우트 쪽에서 바로
같은 뷰어를 띄울 수도 있다 (내부적으로 `viewer.launch.py`를
`main_domain_id`로 include).

## `role`이 실제로 켜는 것

| role     | fleet_role 기본값 | 리스크맵 | RL 정책 |
|----------|------------------|----------|---------|
| `scout`  | `guard`          | 켜짐     | 켜짐    |
| `waffle` | `leader`         | 안 켜짐 (베타 placeholder) | 안 켜짐 |

`scout`는 리더를 따라가지도, 리더가 되지도 않는 `guard.launch.py` 위에서
RL 정책이 직접 `/cmd_vel`을 몰아 정찰하고, 코디네이터가 다른 로봇을
비켜줘야 할 때만 짧게 Nav2 목적지로 개입한다. `follower`처럼 리더를
계속 쫓아가게 하고 싶으면 `fleet_role:=follower`로 바꾸면 된다.

`waffle`은 아직 정찰봇 behavior와 합쳐지지 않은 베타 자리표시자다.
지금은 `tb3_fleet_bringup/leader.launch.py`만 실행한다.

## 주요 옵션

공통: `domain_id`, `main_domain_id`, `fleet_role`, `start_robot_bringup`,
`auto_localize` — 전부 `tb3_fleet_bringup`의 동명 인자와 그대로 대응된다.

리스크맵(scout, `start_risk_map:=true`가 기본): `start_camera`,
`risk_model_path`(YOLO 가중치), `start_cartographer`(기본 false),
`cartographer_configuration_basename`(기본
`turtlebot3_lds_2d_risk_safe.lua`).

### TF 소유권: AMCL vs 리스크맵 Cartographer

`guard.launch.py`의 AMCL이 이 스택에서 기본으로 `map->odom` TF를 갖는
쪽이다(`enable_amcl`, 기본 true). 리스크맵의 Cartographer가 대신 SLAM을
가지려면 반드시 `enable_amcl:=false`와 같이 써야 한다 — 그렇지 않으면
`system.launch.py`가 `start_cartographer:=true` + `enable_amcl:=true`
조합을 아예 에러로 막는다(둘 다 켜지면 `map->odom`을 동시에 방송하게
되므로).

```bash
# 리스크맵이 SLAM을 갖는 조합
ros2 launch tb3_system_bringup system.launch.py \
  role:=scout enable_amcl:=false start_cartographer:=true
```

**아직 안 풀린 부분(고치지 않고 남겨둠):**
- 이 스위치는 AMCL↔Cartographer 사이의 TF 소유권만 정리한 것이고, 로봇
  하드웨어 bringup(`turtlebot3_bringup/robot.launch.py`)이 기본 파라미터로
  여전히 `odom->base_footprint`를 직접 방송한다. Cartographer를 켜는
  쪽에서 이 TF까지 완전히 겹치지 않게 하려면 `tb3_bayesian_risk_map`의
  `turtlebot3_burger_no_odom_tf.yaml`(휠 오도메트리 TF 끔) 같은 하드웨어
  파라미터 교체가 별도로 필요한데, `guard.launch.py`/`system.launch.py`
  둘 다 아직 그 파라미터 파일을 바꿔 끼울 옵션이 없다.
- `guard.launch.py`의 `map_relay`(리더 맵을 브릿지받아 로컬 `/map`으로
  republish)는 `enable_amcl`과 무관하게 항상 켜져 있어서, Cartographer도
  켜면 같은 `/map` 토픽에 발행자가 둘이 된다. 이건 별도로 정리해야 한다.
- 정찰봇이 자기 SLAM으로 독자 맵을 가지면 `/guard_pose`가 리더의 공유
  맵 좌표계와 달라져서 `fleet_path_coordinator`의 회피 계산이 어긋날 수
  있다 — 아직 해결 방법을 논의 중.

RL 정책(scout, `start_rl_policy:=true`가 기본): `rl_model_path`(SB3
`.zip`), `rl_extra_args`(`eval_policy`에 그대로 넘길 추가 CLI 플래그
문자열), `rl_disable_slam_map`(기본 true).

### 알아둬야 할 통합 이슈: `eval_policy`와 SLAM 소유권

`turtlebot3_rl_training`의 `eval_policy --real-robot`은 **무조건**
자기 자신의 SLAM(Cartographer/slam_toolbox)과 `map->odom` TF를 새로
띄우려고 한다(`--no-auto-start-slam`을 줘도 무시하고 다시 켠다). 이
패키지는 fleet 스택(guard/follower의 AMCL, leader의 Cartographer)이
이미 그 TF를 갖고 있다고 가정하므로, 기본값으로 `--disable-slam-map`을
같이 넘겨서 `eval_policy`가 자체 SLAM을 켜지 않게 막아둔다.

다만 `--disable-slam-map`을 켜면 `eval_policy` 내부에서 맵 프레임 기반
기능(TF, safety boundary, priority map 정렬 등)도 함께 꺼진다. 학습 때
쓴 옵저베이션 구성에 따라 이게 정책 성능에 영향을 줄 수 있다 — 실제
로봇에서 처음 돌릴 때는 반드시 동작을 확인하고, 필요하면
`rl_disable_slam_map:=false`로 켜서 `eval_policy`가 자체 SLAM을 갖게
하되 그 경우 `fleet_role:=guard`(AMCL)와는 같이 쓰지 말 것 — 두
소스가 동시에 `map->odom`을 방송하면 TF가 깨진다.
