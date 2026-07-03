# USB stereo camera quick test

ROS 패키지로 묶기 전에 PC에서 USB 카메라 2대가 동시에 안정적으로 도는지 확인하는 순수 Python/OpenCV 테스트입니다.

## 1) 카메라 장치 확인

```bash
cd ~/Desktop/ROS2-Project
python3 stereo_usb_test/dual_camera_probe.py --list
```

또는:

```bash
v4l2-ctl --list-devices
```

## 2) 기본 실행

카메라 장치가 `/dev/video0`, `/dev/video2`라면:

```bash
cd ~/Desktop/ROS2-Project
python3 stereo_usb_test/dual_camera_probe.py \
  --left /dev/video0 \
  --right /dev/video2 \
  --width 320 \
  --height 240 \
  --fps 10 \
  --fourcc MJPG
```

키:

- `q` 또는 `ESC`: 종료
- `s`: 좌/우 현재 프레임 저장

## 3) 화면 없이 20초 FPS 로그만 보기

```bash
cd ~/Desktop/ROS2-Project
python3 stereo_usb_test/dual_camera_probe.py \
  --left /dev/video0 \
  --right /dev/video2 \
  --width 320 \
  --height 240 \
  --fps 10 \
  --fourcc MJPG \
  --duration 20 \
  --no-display
```

## 판단 기준

일단 성공 기준은 보수적으로 이렇게 보면 됩니다.

- 두 카메라 모두 평균 8 FPS 이상
- frame age가 보통 300ms 아래
- left/right skew가 너무 자주 200ms 이상 튀지 않음
- 캡처 실패가 계속 쌓이지 않음

이게 안정적이면 다음 단계로 YOLO bbox 매칭/간단 disparity 거리 추정을 붙일 수 있습니다.

## 4) 두 이미지/두 카메라로 거리 유도

두 카메라 렌즈 중심 사이 거리가 7.15cm이고, 카메라가 서로 평행하다고 가정합니다.

```bash
cd ~/Desktop/ROS2-Project

python3 stereo_usb_test/stereo_distance_probe.py \
  --left /dev/video0 \
  --right /dev/video2 \
  --width 320 \
  --height 240 \
  --fps 10 \
  --fourcc MJPG \
  --baseline-cm 7.15 \
  --hfov-deg 60
```

화각을 모르면 먼저 실제 거리 하나로 맞추는 게 제일 빠릅니다. 예를 들어 사람 프린트를 정확히 100cm 앞에 두고 좌/우 이미지에서 같은 점을 클릭한 뒤 `k`를 누릅니다.

```bash
python3 stereo_usb_test/stereo_distance_probe.py \
  --left /dev/video0 \
  --right /dev/video2 \
  --width 320 \
  --height 240 \
  --fps 10 \
  --fourcc MJPG \
  --baseline-cm 7.15 \
  --known-distance-cm 100
```

키:

- 좌/우 화면 클릭: 같은 물리 지점 선택
- `k`: `--known-distance-cm` 기준으로 focal pixel 자동 추정
- `[` / `]`: 화각 추정값 1도씩 조정
- `-` / `=`: focal pixel 2%씩 조정
- `r`: 클릭점 리셋
- `s`: 현재 이미지 저장
- `q` 또는 `ESC`: 종료

정지 이미지 파일 2장으로도 테스트할 수 있습니다.

```bash
python3 stereo_usb_test/stereo_distance_probe.py \
  --left-image stereo_usb_test/captures/left.jpg \
  --right-image stereo_usb_test/captures/right.jpg \
  --baseline-cm 7.15 \
  --known-distance-cm 100
```

YOLO가 설치되어 있고 모델 파일이 있으면 가장 큰 사람 bbox 중심으로 자동 거리도 볼 수 있습니다.

```bash
python3 stereo_usb_test/stereo_distance_probe.py \
  --left /dev/video0 \
  --right /dev/video2 \
  --baseline-cm 7.15 \
  --known-distance-cm 100 \
  --yolo-model yolo11n.pt \
  --auto-yolo
```
