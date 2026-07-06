# tb3_fleet_bringup

리더와 팔로워의 실기·시뮬레이션 스택은 각각 하나의 런치 파일을 사용한다.

- `base.launch.py`: 모든 실물 로봇 역할의 공통 뼈대 — 하드웨어 bringup +
  Nav2 core(controller/planner/behavior/bt_navigator + lifecycle) +
  goal_pose→NavigateToPose 프록시만 있다. Cartographer, AMCL,
  domain_bridge는 일부러 안 들어있다 — 로컬라이제이션/브릿징 방식은
  역할마다 다르므로 이 파일을 include하는 쪽이 그 위에 얹는다.
- `member.launch.py`: `base.launch.py` + domain_bridge + AMCL. 리더도
  팔로워도 되지 않고, 코디네이터가 보내는 짧은 회피/복귀 목적지만
  실행하는 범용 fleet 멤버. `follower.launch.py`가 여기서 한 단계 더
  나가는 개념이고, `leader.launch.py`는 로컬라이제이션 방식이 완전히
  달라서(Cartographer, 브릿징 없음) `base.launch.py`만 직접 공유한다.
- `follower.launch.py`: domain_bridge, AMCL, Nav2, fleet follower
  (실물 모드는 내부적으로 `base.launch.py`를 사용, 시뮬레이션 모드는
  기존처럼 자체 구성)
- `leader.launch.py`: Cartographer, Nav2, fleet coordinator (실물 모드는
  내부적으로 `base.launch.py`를 사용, 시뮬레이션 모드는 기존처럼 자체 구성)
- `robot.launch.py`: 하드웨어 드라이버만 별도로 실행할 때 사용
- `rviz.launch.py`: PC 시각화
- `sim_world.launch.py`: Gazebo 월드와 두 로봇 모델만 실행

## 공통 환경

```bash
source ~/venv/ros/bin/activate
source /opt/ros/jazzy/setup.bash
source install/setup.bash

unset FASTRTPS_DEFAULT_PROFILES_FILE
unset RMW_FASTRTPS_DEFAULT_PROFILES_FILE
unset FASTDDS_DEFAULT_PROFILES_FILE
unset ROS_DISCOVERY_SERVER
```

## 실물

리더:

```bash
export ROS_DOMAIN_ID=24
ros2 launch tb3_fleet_bringup leader.launch.py
```

팔로워:

```bash
export ROS_DOMAIN_ID=25
ros2 launch tb3_fleet_bringup follower.launch.py main_domain_id:=24
```

멤버(선택, 3번째 이상의 로봇):

```bash
export ROS_DOMAIN_ID=26
ros2 launch tb3_fleet_bringup member.launch.py main_domain_id:=24
```

하드웨어 드라이버를 이미 별도로 실행했다면 세 명령 모두에 다음 옵션만 추가한다.

```bash
start_robot_bringup:=false
```

## 초기 위치 자동 탐색 (`auto_localize`)

`follower.launch.py`와 `member.launch.py`는 기본적으로 `auto_localize:=true`다.
매번 `follower_initial_x/y/yaw`(또는 `member_initial_x/y/yaw`)로 고정된
위치를 AMCL에 강제로 심는 대신, localization 스택이 뜨면
`global_localize_kickstart` 노드가 `/reinitialize_global_localization`을
호출해 맵 전체에 파티클을 고르게 뿌리고, 실물 로봇을 짧게(기본 8초) 제자리
회전시켜 스캔 매칭이 여러 시점을 확보하도록 돕는다.

고정 시드가 이미 정확히 맞는 상황이거나, 대칭적인 공간이라 자동 탐색이
불안정하면 `auto_localize:=false`로 끄고 기존처럼 `follower_initial_x/y/yaw`
(`member_initial_x/y/yaw`)를 실측값으로 넣는다.

## AMCL을 끄는 옵션 (`enable_amcl`)

`follower.launch.py`와 `member.launch.py` 둘 다 `enable_amcl`(기본 true)을
받는다. AMCL은 이 스택이 기본으로 갖는 `map->odom` TF 소스다. 다른 무언가
(예: 리스크맵 쪽 Cartographer)가 이 로봇의 SLAM/TF를 대신 갖게 하려면
`enable_amcl:=false`로 꺼야 한다 — 켠 채로 다른 SLAM까지 띄우면 같은 TF를
동시에 방송하게 된다. `enable_amcl:=false`면 AMCL, 그 lifecycle manager,
`global_localize_kickstart`(AMCL 전용 서비스라 AMCL 없이는 의미가 없다)가
전부 스킵된다.

`enable_amcl:=false`로 이 로봇이 Cartographer 등 자기 SLAM을 갖게 할
때는, `base.launch.py`/`member.launch.py`/`follower.launch.py` 모두가
받는 **`hardware_param_file`**도 같이 넘겨서 `turtlebot3_bringup`의
하드웨어 파라미터 파일을 바꿔야 한다(`tb3_bayesian_risk_map`의
`turtlebot3_burger_no_odom_tf.yaml` 등) — 안 그러면 휠 오도메트리가
여전히 `odom->base_footprint`를 자기가 따로 방송해서, Cartographer가
갖는 TF와 충돌해 트리가 두 갈래로 쪼개진다(`tb3_system_bringup`
README의 "하드웨어 odom TF 자동 교체" 참고 — `system.launch.py`의 SLAM
조합 launch(`fleet_role:=member`/`follower` 둘 다)에서는 이걸 자동으로
넣어준다).

## 리더가 SLAM을 포기하고 멤버의 맵을 받는 옵션 (`enable_cartographer`)

`leader.launch.py`도 (실물 모드에서) `enable_cartographer`(기본 true)를
받는다. `false`로 주면 자체 Cartographer 대신 `map_relay`(`input_topic`을
`/map_from_member`로) + AMCL 조합으로 바뀐다 — 정찰봇처럼 자기 SLAM을 가진
멤버(`enable_amcl:=false start_cartographer:=true`)가 있을 때, 그 멤버가
만든 맵을 리더가 받아서 AMCL로 로컬라이즈하고 자기 도메인의 `/map`으로
재발행하게 하는 옵션이다. `auto_localize`/`leader_initial_x/y/yaw`도
멤버 쪽과 동일한 패턴으로 지원한다.

```bash
ros2 launch tb3_fleet_bringup leader.launch.py enable_cartographer:=false
```

`/map_from_member` 브릿지는 멤버 쪽 `write_member_bridge_configs`가 매
실행마다 그 시점의 `domain_id`/`main_domain_id` 값으로 `/tmp/`에 새로
써서 만든다(캐싱 없음) — 와플의 도메인이 바뀌어도 다음 실행에 그대로
반영된다. 리더 쪽에 별도 domain_bridge 프로세스를 새로 띄울 필요는 없다
(멤버가 양방향 브릿지를 이미 다 실행한다).

## 시뮬레이션

리더 도메인의 터미널 1:

```bash
export ROS_DOMAIN_ID=24
ros2 launch tb3_fleet_bringup sim_world.launch.py
```

리더 도메인의 터미널 2:

```bash
export ROS_DOMAIN_ID=24
ros2 launch tb3_fleet_bringup leader.launch.py \
  use_sim_time:=true start_robot_bringup:=false
```

팔로워 도메인:

```bash
export ROS_DOMAIN_ID=25
ros2 launch tb3_fleet_bringup follower.launch.py \
  use_sim_time:=true start_robot_bringup:=false main_domain_id:=24
```

`start_robot_bringup`은 하드웨어 드라이버 실행 여부이고,
`use_sim_time`은 Gazebo 시간과 가상 센서 relay 사용 여부다. 따라서 실물
드라이버를 별도로 실행하는 경우에도 `use_sim_time`은 `false`로 유지한다.

## 팔로워 없는 리더 (`require_follower_pose`)

`leader.launch.py`의 코디네이터는 기본적으로(`require_follower_pose:=true`)
`/leader_pose`와 `/burger_pose`(follower)가 둘 다 있어야 리더를 움직이게
둔다 — follower가 아예 없는 fleet(예: 리더+멤버만 있는 구성)에서 이 기본값
그대로 쓰면 `/burger_pose`가 영원히 안 들어오기 때문에 **리더가 첫 pose를
받는 순간 그 자리에 정지 목적지를 박고 다시는 안 풀린다.** 실제로
`follower.launch.py`를 쓰는 로봇이 이 fleet에 없다면 반드시
`require_follower_pose:=false`로 꺼야 한다. 꺼도 멤버(`member.launch.py`)
관련 회피/양보 로직은 완전히 별개 상태 머신이라 그대로 동작한다.

## 상호 회피

중앙 코디네이터의 회피 판단은 Nav2 경로를 사용하지 않는다.
`/leader_pose`와 `/burger_pose`의 상대 위치와 시간 변화를 이용해 실제
속도와 미래 최근접 거리를 추정한다. 따라서 Nav2, 키보드 teleop, 직접
`/cmd_vel` 제어 모두 같은 방식으로 감지된다. Nav2 path의 끝점은 회피 후
원래 목적지를 복구하는 용도로만 사용한다.

- `/goal_pose`: Waffle의 사용자 목적지
- `/burger_user_goal`: Burger의 사용자 목적지
- `/leader/scan`: leader의 fleet용 LiDAR
- `/follower25/scan`: domain 25 follower의 fleet용 LiDAR
- `/fleet/leader_coord_goal`: 코디네이터가 Nav2에 전달하는 Waffle 목적지
- `/burger_goal_pose`: 코디네이터가 Nav2에 전달하는 Burger 목적지
- `/fleet/robot_poses`: `[leader, follower]` 순서의 실시간 위치
- `/fleet/collision_warning`: 현재 또는 예측 충돌 위험
- `/fleet/hazard_pose`: 예상 충돌 지점
- `/fleet/coordination_status`: 회피 상태와 통행 우선권

사용자 목적지는 코디네이터가 계속 보존한다. 우선권 로봇은 기존 Nav2
목적지를 중단하지 않고 그대로 진행하며, 양보 로봇만 짧은 회피 목적지로
이동했다가 상대가 통과하거나 멈추면 최신 사용자 목적지로 복귀한다.
Nav2 액션 서버가 아직 준비되지 않았거나 재시작 중이어도 최신 목적지는
폐기하지 않고 준비될 때까지 재시도한다.

이 안전 토픽은 leader domain에서 각 follower domain으로 브릿지된다.
Follower LiDAR 토픽의 숫자는 follower의 `domain_id`에서 자동 생성된다.
예를 들어 domain 31이면 `/follower31/scan`이다. 각 로봇 내부에서 Nav2가
사용하는 `/scan`은 변경하지 않는다.
한 대만 움직이면 움직이는 로봇이 우선이고, 두 대가 동시에 움직이면
Waffle이 우선이다. follow 모드에서도 Waffle이 우선이며 Burger만 비킨다.
두 로봇 중 하나의 위치가 1.5초 이상 갱신되지 않거나 안전거리가 확보되지
않은 채 회피 시간이 끝나면 원래 목적지를 재개하지 않고 안전 정지 상태를
유지한다.

## 멤버 (member)

멤버는 위 leader/follower 우선권 로직과는 완전히 분리된, 별도의 작은 상태
머신으로 동작한다. `/member_pose`로 자기 위치만 보고하고, 리더나 팔로워 중
하나가 가까워지거나(예측 충돌 포함) 접근하면 코디네이터가 `_yield_goal`과
동일한 방식으로 짧은 회피 지점을 계산해 `/member_goal_pose`로 보낸다. 멤버는
스스로 목적지를 만들지 않으므로, 위협이 사라지면 회피 직전에 서 있던
자리로 그대로 복귀한다. 멤버가 없거나 `/member_pose`가 들어오지 않으면 이
로직은 완전히 비활성 상태이며 기존 leader/follower 동작에는 아무 영향도
주지 않는다.

멤버가 `enable_amcl:=false`로 자기 SLAM(예: 리스크맵 Cartographer)을 갖는
경우, 그 로컬 `/map`은 `/map_from_member`로 리맵되어 main 도메인으로도
브릿지된다 — 리더 쪽에서 `enable_cartographer:=false`로 그 맵을 받아 쓸
수 있다(위 "리더가 SLAM을 포기하고 멤버의 맵을 받는 옵션" 참고).

## `/map` 페일오버 (`map_relay`)

`follower.launch.py`/`member.launch.py`의 `map_relay`는 더 이상 브릿지된
맵을 무조건 재발행하지 않는다. `count_publishers()`로 `/map`에 자기 말고
다른 발행자(예: 리스크맵 Cartographer, `enable_amcl:=false` 조합일 때)가
있는지 계속 확인해서, 있으면 조용히 대기하고 `takeover_grace_sec`(기본
2초) 이상 사라지면 그때만 넘겨받는다. 이어받을 때는 Cartographer가
마지막으로 냈던 맵을 우선 이어가고, 그런 적이 없으면 브릿지된 리더 맵으로
대체한다. 원래 발행자가 돌아오면 즉시 다시 조용해진다.
