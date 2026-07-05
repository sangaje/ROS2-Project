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
