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

**follower.launch.py 로봇이 fleet에 없으면(예: 리더+정찰봇만 있는 구성)
`require_follower_pose:=false`를 반드시 같이 줘야 한다** — 기본값(true)
그대로면 코디네이터가 follower pose를 영원히 기다리다가 리더를 첫 위치에서
그대로 멈춰 세운다(`tb3_fleet_bringup` README의 "팔로워 없는 리더" 참고).

```bash
ros2 launch tb3_system_bringup system.launch.py \
  role:=waffle require_follower_pose:=false
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

| role     | fleet_role 기본값 | 리스크맵                   | RL 정책 |
|----------|-------------------|----------------------------|---------|
| `scout`  | `member`          | 켜짐                       | 켜짐    |
| `waffle` | `leader`          | 안 켜짐 (베타 placeholder) | 안 켜짐 |

`scout`는 리더를 따라가지도, 리더가 되지도 않는 `member.launch.py` 위에서
RL 정책이 직접 `/cmd_vel`을 몰아 정찰하고, 코디네이터가 다른 로봇을
비켜줘야 할 때만 짧게 Nav2 목적지로 개입한다. `follower`처럼 리더를
계속 쫓아가게 하고 싶으면 `fleet_role:=follower`로 바꾸면 된다.

`waffle`은 아직 정찰봇 behavior와 합쳐지지 않은 베타 자리표시자다.
지금은 `tb3_fleet_bringup/leader.launch.py`만 실행한다.

### follower는 결국 정찰봇이다

`role:=scout fleet_role:=follower`로 쓰면(정찰봇 하드웨어가 잠깐 리더를
그냥 쫓아만 다니는 모드) `system.launch.py`가 자동으로 **RL 정책만**
꺼준다 — `follower.launch.py`의 Nav2 추종 로직이 이미 `/cmd_vel`을 몰고
있는데 RL도 같이 몰면 둘이 로봇을 두고 싸우게 되기 때문. 리스크맵/카메라/YOLO는
follower 중에도 계속 켜져 있다 — 위험도만 수동적으로 계속 쌓을 뿐 이동
명령은 절대 안 내리므로 follower의 Nav2 추종과 부딪히지 않는다. 정찰
임무(RL 탐사)를 다시 켜려면 `fleet_role`을 다시 `member`(또는 비워서
기본값)로 바꾸면 된다.

## 주요 옵션

공통: `domain_id`, `main_domain_id`, `fleet_role`, `start_robot_bringup`,
`auto_localize` — 전부 `tb3_fleet_bringup`의 동명 인자와 그대로 대응된다.

리스크맵(scout, `start_risk_map:=true`가 기본): `start_camera`,
`risk_model_path`(YOLO 가중치), `start_cartographer`(기본 false),
`cartographer_configuration_basename`(기본
`turtlebot3_lds_2d_risk_safe.lua`).

### YOLO를 PC로 오프로드 (`start_camera_sender`)

`start_camera_sender:=true`를 주면 `system.launch.py`가 직접
`tb3_flask_yolo_bridge/opencv_camera_to_flask_yolo.launch.py`를 이
로봇에서 같이 띄운다 — 카메라를 잡아서 PC의 `flask_yolo_server`로 HTTP로
프레임을 보내고, 리스크맵은 로컬 YOLO 대신 그 결과(`external_detection_topic`)를
읽는다. 켜면 `start_camera:=false`, `enable_yolo:=false`,
`detection_source:=flask_topic`이 자동으로 강제되므로 따로 안 챙겨도 된다.
별도로 `opencv_camera_to_flask_yolo.launch.py`를 수동으로 또 띄울 필요 없음.

```bash
# 정찰봇: YOLO는 PC(flask_yolo_server)에서 돌리고 여기선 카메라만 전송
ros2 launch tb3_system_bringup system.launch.py \
  role:=scout start_camera_sender:=true

# PC: flask_yolo_server (YOLO 추론 서버)
ros2 launch tb3_flask_yolo_bridge flask_yolo_server.launch.py
```

- `camera_sender_device`(기본 `/dev/video1`): 이 로봇에서 잡을 카메라 장치.
- `flask_server_url`(기본 `http://seil:5005/detect`): PC의 flask_yolo_server
  주소. Tailscale 호스트명 `seil`을 기본값으로 쓴다(IP 대신).
- 리스크맵이 읽는 `external_detection_topic`과 센더의 `output_topic`이
  같은 값으로 자동으로 맞춰진다(둘 다 `external_detection_topic` 인자를
  공유).

### TF 소유권: AMCL vs 리스크맵 Cartographer

`member.launch.py`의 AMCL이 이 스택에서 기본으로 `map->odom` TF를 갖는
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

이 안전장치는 `fleet_role`이 `member`/`follower`(AMCL, `enable_amcl`로
끌 수 있음)나 `leader`(Cartographer, `enable_cartographer`로 끌 수 있음)일
때 적용된다. `fleet_role:=follower`도 `enable_amcl:=false`와 같이 쓰면
`start_cartographer:=true`를 그대로 쓸 수 있다 (2026-07-06 배선 완료 —
follower는 결국 정찰봇이므로 follower 중에도 리스크맵 Cartographer가
SLAM을 계속 소유해야 한다, 위 "follower는 결국 정찰봇이다" 참고).

`/map` 이중 발행 문제는 `map_relay`가 `count_publishers()`로 Cartographer
같은 다른 발행자가 있는지 감지해서 있으면 조용히 대기하도록 바뀌어서
해결됐다 (`tb3_fleet_bringup` README의 "`/map` 페일오버" 참고).

### 정찰봇이 SLAM을 갖고, 와플이 그 맵을 받아서 재발행

`role:=waffle`에 `enable_cartographer:=false`를 주면 와플(`leader.launch.py`)이
자체 Cartographer 대신, `/map_from_member`로 브릿지되어 들어오는 맵을
받아 AMCL로 로컬라이즈하고 그걸 자기 도메인의 `/map`으로 재발행한다.
정찰봇 쪽에서 `enable_amcl:=false start_cartographer:=true`로 SLAM을
갖고 있어야 짝이 맞는다.

```bash
# 와플: 정찰봇이 만든 맵을 받아서 AMCL + 재발행
ros2 launch tb3_system_bringup system.launch.py \
  role:=waffle enable_cartographer:=false

# 정찰봇: 직접 SLAM 소유
ros2 launch tb3_system_bringup system.launch.py \
  role:=scout enable_amcl:=false start_cartographer:=true start_rl_policy:=false
```

`/map_from_member` 브릿지는 정찰봇(member)의 `write_member_bridge_configs`가
매 실행마다 `main_domain_id`/`domain_id` 실행값으로 동적으로
`/tmp/tb3_fleet_domain_bridge/`에 다시 써서 만든다 — 와플 쪽 도메인이
바뀌어도 캐싱 없이 항상 최신 값으로 반영된다. 별도로 와플 쪽에 새
domain_bridge 프로세스를 띄울 필요는 없다(정찰봇이 양방향 다 실행).

### 하드웨어 odom TF 자동 교체 (해결됨, 2026-07-06)

`role:=scout enable_amcl:=false start_cartographer:=true` 조합을 쓰면
(정찰봇이 SLAM 소유, `fleet_role:=member`든 `fleet_role:=follower`든 둘 다
적용됨), `system.launch.py`가 자동으로:
- `member.launch.py`/`follower.launch.py`의 하드웨어 bringup에
  `tb3_bayesian_risk_map`의 `turtlebot3_burger_no_odom_tf.yaml`을
  `hardware_param_file`로 넘겨서 휠 오도메트리의 `odom->base_footprint`
  자체 방송을 끈다.
- 리스크맵의 `cartographer_configuration_basename`이 기본값
  (`turtlebot3_lds_2d_risk_safe.lua`, `map->base_footprint`만 직접 발행하고
  `odom`은 건너뜀)일 때만 `turtlebot3_lds_2d_risk_safe_no_odom.lua`(정상적인
  `map->odom->base_footprint` 전체 소유)로 자동으로 바꿔준다.

이 조합 없이 그냥 켜면 `odom`과 `base_footprint`를 두 곳(휠 오도메트리,
Cartographer)이 동시에 주장해서 TF 트리가 두 갈래로 쪼개지고
(`Tf has two or more unconnected trees`), Nav2가 `map`으로 좌표 변환을
못 해서 결과적으로 맵/코스트맵이 전혀 안 뜨는 것처럼 보인다 — 실기
3대 테스트 중 실제로 겪은 문제. `cartographer_configuration_basename`을
직접 다른 값으로 지정하면 이 자동 교체는 건너뛴다(사용자가 이미 알고
하는 것으로 간주).

- 정찰봇이 자기 SLAM으로 독자 맵을 가지면 `/member_pose`가 리더의 공유
  맵 좌표계와 달라져서 `fleet_path_coordinator`의 회피 계산이 어긋날 수
  있다 — 아직 해결 방법을 논의 중.
- `map_relay`의 `count_publishers()` 감지는 Cartographer가 막 뜨는 중이라
  아직 discovery에 안 잡힌 순간(수 초)에는 "발행자 없음"으로 오판해서
  잠깐 넘겨받을 수 있다. `start_cartographer:=true` 조합에서는
  `takeover_grace_sec` 기본값(2초)이 너무 짧을 수 있으니 실기 테스트 때
  Cartographer 기동 시간에 맞춰 늘리는 걸 권장.
- SLAM 소유권 자동 배선(`enable_amcl` 전달, `hardware_param_file` 자동
  주입)은 `fleet_role:=member`와 `fleet_role:=follower` 둘 다 지원한다
  (2026-07-06 배선 완료). `fleet_role:=leader`는 대신 `enable_cartographer`로
  같은 역할을 한다.

RL 정책(scout, `start_rl_policy:=true`가 기본): `rl_model_path`(SB3
`.zip`), `rl_extra_args`(`eval_policy`에 그대로 넘길 추가 CLI 플래그
문자열), `rl_disable_slam_map`(기본 true).

### 알아둬야 할 통합 이슈: `eval_policy`와 SLAM 소유권

`turtlebot3_rl_training`의 `eval_policy --real-robot`은 **무조건**
자기 자신의 SLAM(Cartographer/slam_toolbox)과 `map->odom` TF를 새로
띄우려고 한다(`--no-auto-start-slam`을 줘도 무시하고 다시 켠다). 이
패키지는 fleet 스택(member/follower의 AMCL, leader의 Cartographer)이
이미 그 TF를 갖고 있다고 가정하므로, 기본값으로 `--disable-slam-map`을
같이 넘겨서 `eval_policy`가 자체 SLAM을 켜지 않게 막아둔다.

다만 `--disable-slam-map`을 켜면 `eval_policy` 내부에서 맵 프레임 기반
기능(TF, safety boundary, priority map 정렬 등)도 함께 꺼진다. 학습 때
쓴 옵저베이션 구성에 따라 이게 정책 성능에 영향을 줄 수 있다 — 실제
로봇에서 처음 돌릴 때는 반드시 동작을 확인하고, 필요하면
`rl_disable_slam_map:=false`로 켜서 `eval_policy`가 자체 SLAM을 갖게
하되 그 경우 `fleet_role:=member`(AMCL)와는 같이 쓰지 말 것 — 두
소스가 동시에 `map->odom`을 방송하면 TF가 깨진다.
