# system_bringup

로봇의 역할(정찰/리더)에 맞춰 `fleet_bringup` + Scout Cartographer/risk
map + 경량 정찰 카메라 송신 + 리더 Jetson/OMX AIM을 한 번에 켜는
오케스트레이터. 기본 구조는 Scout Pi/Domain 22가 Cartographer SLAM과
authoritative `/map`을 소유하고, Leader Waffle/Jetson은 그 shared map을
받아 자기 AMCL/Nav2로 주행한다.
로봇마다 이 패키지의 런치 파일 하나만 실행하면 된다.

- `system.launch.py`: 역할별 전체 스택
- `pc.launch.py`: PC에서 디버그 RViz/뷰어 실행. YOLO 서버는 기본적으로
  리더 Jetson에서 실행한다.
- `viewer.launch.py`: PC에서 fleet 디버그 마커 + 리스크맵을 하나의 RViz로
  같이 보는 뷰어 (로봇을 켜지 않음)

## 실행

정찰봇:

```bash
ros2 launch system_bringup system.launch.py \
  role:=scout main_domain_id:=24
```

리더:

```bash
ros2 launch system_bringup system.launch.py \
  role:=leader risk_domain_id:=22 pc_domain_id:=28 \
  start_omx_aim:=true start_yolo_server:=true debug_stream:=true
```

**follower.launch.py 로봇이 fleet에 없으면(예: 리더+정찰봇만 있는 구성)
`require_follower_pose:=false`를 반드시 같이 줘야 한다** — 기본값(true)
그대로면 코디네이터가 follower pose를 영원히 기다리다가 리더를 첫 위치에서
그대로 멈춰 세운다(`fleet_bringup` README의 "팔로워 없는 리더" 참고).

```bash
ros2 launch system_bringup system.launch.py \
  role:=leader require_follower_pose:=false
```

PC 디버깅 실행 (fleet/risk RViz):

```bash
ros2 launch system_bringup pc.launch.py
```

PC에서 YOLO 서버는 기본으로 켜지지 않는다. 컴포넌트 디버깅용으로만
PC에서 직접 돌릴 때 `start_yolo_server:=true`를 준다.

## `role`이 실제로 켜는 것

| role     | fleet_role 기본값 | 리스크맵                   | RL 정책 | Jetson/OMX |
|----------|-------------------|----------------------------|---------|------------|
| `scout`  | `member`          | 켜짐                       | 기본 꺼짐 | 안 켜짐  |
| `leader` | `leader`          | 안 켜짐                    | 안 켜짐 | 켜짐       |

`scout`는 기본적으로 로봇 base, Cartographer, Bayesian risk map, camera
sender를 켠다. 카메라 프레임은 리더 Jetson의 `flask_yolo_server`로 보내고,
Scout-local YOLO와 risk-map 내부 camera capture는 자동으로 꺼진다.
Cartographer가 `/map`을 소유하는 동안 Scout 하위 Nav2는 기본으로 세우지
않아 Pi의 CPU와 `/cmd_vel` 경합을 줄인다. `follower`처럼 리더를 계속
쫓아가게 하고 싶으면 `fleet_role:=follower`로 바꾸면 된다.

`leader`는 `fleet_bringup/leader.launch.py`를 실행하고, 기본적으로 자체
Cartographer를 띄우지 않는다. `risk_domain_id`로 지정한 Scout/risk domain의
`/map`을 `/map_bridge`로 받아 leader domain의 `/map`으로 relay한 뒤, Leader
자기 LiDAR/odom으로 AMCL/Nav2를 실행한다. RViz 프로세스는 리더에서 실행하지 않는다.
리더 role은 `omx_aim/jetson.launch.py`를 하위 component로 include해서 Jetson
YOLO 서버, OMX AIM 파이프라인, debug stream, 통합 웹 대시보드를 함께 실행한다.

### 통합 대시보드의 Nav2 경로 표시

통합 대시보드는 map/risk/robot marker 위에 Nav2 `nav_msgs/Path`도 함께
그린다. 기본 구독 토픽은 다음과 같다.

| label | topic | 의미 |
|-------|-------|------|
| leader | `/plan` | Leader domain에서 Nav2 planner가 직접 발행하는 현재 경로 |
| leader_bridge | `/leader_plan` | bridge/PC 표시용 leader plan remap 경로 |
| follower | `/burger_plan` | follower domain의 `/plan`이 leader domain으로 들어온 경로 |
| member | `/member_plan` | member/scout Nav2 plan을 별도 bridge할 때 쓸 예약 경로 |

화면의 `Nav2 paths` 체크박스로 지도 위 경로 레이어를 켜고 끌 수 있고, 아래
`Nav2 Paths` 패널에는 각 topic의 frame, pose count, start/end 좌표, age가
표시된다.

### follower는 결국 정찰봇이다

`role:=scout fleet_role:=follower`로 쓰면(정찰봇 하드웨어가 잠깐 리더를
그냥 쫓아만 다니는 모드) `system.launch.py`가 자동으로 **RL 정책만**
꺼준다 — `follower.launch.py`의 Nav2 추종 로직이 이미 `/cmd_vel`을 몰고
있는데 RL도 같이 몰면 둘이 로봇을 두고 싸우게 되기 때문. Failover가 켜진
운용에서는 카메라 sender도 role-gated로 동작한다. 즉 follower 중에는 카메라
장치를 닫고 화면/YOLO 프레임을 보내지 않다가, 기존 active scout가 죽어서 이
로봇이 `ACTIVE_SCOUT`으로 takeover한 뒤에만 다시 카메라를 열고 송출한다.
정찰 임무(RL 탐사)를 처음부터 켜려면 `fleet_role`을 `member`(또는 비워서
기본값)로 둔다.

## 주요 옵션

공통: `domain_id`, `main_domain_id`, `fleet_role`, `start_robot_bringup`,
`auto_localize` — 전부 `fleet_bringup`의 동명 인자와 그대로 대응된다.

리스크맵(scout, `start_risk_map:=true`가 기본): `start_camera`,
`risk_model_path`(YOLO 가중치), `start_cartographer`(기본 true),
`cartographer_configuration_basename`(기본
`turtlebot3_lds_2d_risk_safe.lua`).

### YOLO를 리더 Jetson으로 오프로드 (`start_camera_sender`)

`start_camera_sender:=true`가 Scout 기본값이다. 정찰봇의 카메라 프레임만
리더 Jetson의 Flask YOLO 서버로 보내고, YOLO 결과(`/risk/yolo_detections`)를
정찰봇 domain에 발행한다. 로컬 risk map은 이 detection topic을 읽고,
Cartographer가 만든 authoritative `/map` 좌표계 위에 risk layer를 유지한다.

`start_camera_sender:=true`를 주면 `system.launch.py`가 직접
`flask_yolo_bridge/opencv_camera_to_flask_yolo.launch.py`를 이
로봇에서 같이 띄운다 — 카메라를 잡아서 리더 Jetson의 `flask_yolo_server`로 HTTP로
프레임을 보내고, 리스크맵은 로컬 YOLO 대신 그 결과(`external_detection_topic`)를
읽는다. 켜면 `start_camera:=false`, `enable_yolo:=false`,
`detection_source:=flask_topic`이 자동으로 강제되므로 따로 안 챙겨도 된다.
Failover가 켜진 경우 이 sender는 `/<robot_name>/role`을 보고 `ACTIVE_SCOUT`
상태에서만 카메라를 열어 프레임을 보낸다. 별도로
`opencv_camera_to_flask_yolo.launch.py`를 수동으로 또 띄울 필요 없음.

```bash
# 정찰봇: YOLO는 리더 Jetson(flask_yolo_server)에서 돌리고 여기선 카메라만 전송
ros2 launch system_bringup system.launch.py \
  role:=scout main_domain_id:=20 start_camera_sender:=true

# 리더 Jetson: system_bringup leader가 flask_yolo_server를 함께 실행
ros2 launch system_bringup system.launch.py \
  role:=leader start_yolo_server:=true
```

- `camera_sender_device`(기본 `/dev/video1`): 이 로봇에서 잡을 카메라 장치.
- `flask_server_url`(기본 `http://orin-jetson:5005/detect`): 리더 Jetson의
  flask_yolo_server 주소.
- 리스크맵이 읽는 `external_detection_topic`과 센더의 `output_topic`이
  같은 값으로 자동으로 맞춰진다(둘 다 `external_detection_topic` 인자를
  공유).

### TF 소유권: AMCL vs 리스크맵 Cartographer

system bringup의 scout 기본값은 리스크맵 Cartographer가 `map->odom` TF와
SLAM을 갖는 쪽이다(`start_cartographer:=true`, `enable_amcl:=false`).
AMCL 기반 member/follower로 운용하려면 `start_cartographer:=false
enable_amcl:=true`를 명시한다. 리스크맵의 Cartographer와 AMCL을 동시에
켜려 하면
`system.launch.py`가 `start_cartographer:=true` + `enable_amcl:=true`
조합을 아예 에러로 막는다(둘 다 켜지면 `map->odom`을 동시에 방송하게
되므로).

```bash
# 리스크맵이 SLAM을 갖는 조합
ros2 launch system_bringup system.launch.py \
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
해결됐다 (`fleet_bringup` README의 "`/map` 페일오버" 참고).

### 정찰봇이 SLAM을 갖고, 리더가 그 맵을 받아서 재발행

`role:=leader`는 기본적으로 자체 Cartographer를 띄우지 않는다
(`enable_cartographer:=false`). 대신 `risk_domain_id`로 지정한
risk/scout 도메인의 `/map`을 단방향 domain_bridge로 받아 `/map_bridge`에
넣고, 리더의 `map_relay`가 자기 도메인의 `/map`으로 재발행한다. 리더는 이
공유 맵으로 AMCL/Nav2를 실행하고, follower/member에는 leader→robot
브리지로 같은 `/map`을 fan-out한다.

```bash
# 리더: 정찰봇이 만든 맵을 받아서 AMCL + 재발행, PC로 debug 전달
ros2 launch system_bringup system.launch.py \
  role:=leader risk_domain_id:=22 pc_domain_id:=30 \
  member_domain_id:=22 follower_domain_id:=25

# 정찰봇: 직접 SLAM 소유
ros2 launch system_bringup system.launch.py \
  role:=scout domain_id:=22 main_domain_id:=24 start_rl_policy:=false
```

bridge YAML은 실행 시점에 `/tmp` 아래 unique 파일로 동적 생성된다. 맵
경로는 risk/scout→leader→follower/member/PC 모두 단방향이며, PC에서
leader로 되돌아가는 bridge는 만들지 않는다.

### 하드웨어 odom TF 자동 교체 (해결됨, 2026-07-06)

`role:=scout enable_amcl:=false start_cartographer:=true` 조합을 쓰면
(정찰봇이 SLAM 소유, `fleet_role:=member`든 `fleet_role:=follower`든 둘 다
적용됨), `system.launch.py`가 자동으로:
- `member.launch.py`/`follower.launch.py`의 하드웨어 bringup에
  `bayesian_risk_map`의 `turtlebot3_burger_no_odom_tf.yaml`을
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
