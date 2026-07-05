# tb3_fleet_bringup

리더와 팔로워의 실기·시뮬레이션 스택은 각각 하나의 런치 파일을 사용한다.

- `leader.launch.py`: Cartographer, Nav2, fleet coordinator
- `follower.launch.py`: domain_bridge, AMCL, Nav2, fleet follower
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

하드웨어 드라이버를 이미 별도로 실행했다면 양쪽 명령에 다음 옵션만 추가한다.

```bash
start_robot_bringup:=false
```

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
