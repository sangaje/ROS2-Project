"""Config 로딩 및 dataclass 변환."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class MotorConfig:
    port: str
    profile_velocity: int
    profile_acceleration: int
    elbow_p_gain: int
    elbow_i_gain: int
    elbow_d_gain: int


@dataclass
class CalibrationConfig:
    home: dict[str, int]
    sign: dict[str, int]


@dataclass
class SafetyConfig:
    angle_limits_deg: dict[str, tuple[float, float]]
    max_step_deg: float
    large_delta_threshold_tick: int

    angle_limits_rad: dict[str, tuple[float, float]] = field(init=False)
    max_step_rad: float = field(init=False)

    def __post_init__(self):
        self.angle_limits_rad = {
            m: (math.radians(lo), math.radians(hi))
            for m, (lo, hi) in self.angle_limits_deg.items()
        }
        self.max_step_rad = math.radians(self.max_step_deg)


@dataclass
class KeyboardConfig:
    arm_step: int
    gripper_step: int
    command_interval: float


@dataclass
class IbvsConfig:
    camera_index: int
    kp_yaw: float
    kp_pitch: float
    sign_vs_x: int
    sign_vs_y: int
    deadband_x: float         # ← deadband → deadband_x
    deadband_y: float
    control_hz: float
    camera_device: str = ''
    camera_backend: str = 'v4l2'
    camera_reconnect_period_sec: float = 1.0
    camera_required: bool = False
    camera_width: int = 1280
    camera_height: int = 720
    camera_fps: float = 15.0
    camera_fourcc: str = 'MJPG'
    camera_buffer_size: int = 1
    # ----- 움직이는 표적 추적 (Phase B) -----
    # 미분 게인. 0 이면 순수 P (기존 동작 유지).
    # 권장 시작값: kp * 0.1 ~ 0.3 (예: kp=0.02 -> kd=0.002~0.006)
    kd_yaw: float = 0.0
    kd_pitch: float = 0.0
    # de/dt EMA 필터 강도. 1.0 = 필터 없음(노이즈 그대로),
    # 0.3~0.5 = 노이즈 잘 잡힘, 0.1 = 매우 부드러움(반응 느림).
    derivative_ema_alpha: float = 0.4
    # IBVS 호출 간격이 이보다 크면 미분 상태 reset
    # (TRACKING 재진입, 표적 잃었다 다시 잡힘 등의 케이스)
    derivative_reset_gap_sec: float = 0.5


@dataclass
class YoloConfig:
    model_path: str
    target_class: int
    conf_threshold: float
    imgsz: int
    device: str = "0"
    half: bool = True


@dataclass
class FireConfig:
    hold_time_sec: float
    confirm_deadband_scale: float
    gripper_close_pos: int
    gripper_open_pos: int
    gripper_close_duration: float
    gripper_open_duration: float
    cooldown_sec: float
    lost_timeout_sec: float = 4.0
    aim_settle_sec: float = 0.7
    # 격발 펄스 동안 조준 유지 시간. fire_node 의 fire_duration_sec 와 일치 권장.
    # COOLDOWN 진입 후 이 시간이 지나야 home 명령이 발사됨.
    # cooldown_sec 보다 작아야 home 이 실행됨.
    fire_pulse_sec: float = 1.5
    # Legacy config compatibility only. Runtime ignores immediate fire and
    # always requires TRACKING -> CONFIRMING -> FIRING.
    immediate_on_detection: bool = False
    immediate_during_nav: bool = True
    immediate_cancel_nav: bool = True
    immediate_min_interval_sec: float = 1.0
    immediate_requires_armed: bool = True


@dataclass
class AutoTrackConfig:
    default_armed: bool
    duplicate_threshold_m: float


@dataclass
class PatrolConfig:
    """정찰 + 우선순위 큐 + LOS + 시각화."""
    scan_timeout_sec: float
    max_queue_size: int
    duplicate_threshold_m: float
    # LOS (단계 F)
    los_check_enabled: bool = True
    los_cost_threshold: int = 80
    costmap_topic: str = "/global_costmap/costmap"
    # 시각화 (단계 G)
    publish_queue_markers: bool = True
    marker_lifetime_sec: float = 2.0
    target_scan_timeout_sec: float = 5.0
    boundary_scan_timeout_sec: float = 1.0
    boundary_aim_settle_sec: float = 0.05
    scan_sweep_half_angle_deg: float = 60.0
    scan_sweep_period_sec: float = 5.0
    
@dataclass
class ViewPoseConfig:
    """CHECK_VIEW 판정 + VIEW_POSE v1 (H2)."""
    omx_yaw_limit_deg: float = 180.0
    min_distance_m: float = 0.3
    max_distance_m: float = 3.0
    stand_off_distance: float = 1.0
    candidate_count: int = 12 
    yaw_next_weight: float = 0.5
    footprint_radius_m: float = 0.24
    min_known_footprint_ratio: float = 0.45
    reject_unknown_footprint: bool = False
    require_clear_path_to_candidate: bool = False
    allow_unknown_target_los: bool = True
    frontier_bonus: float = 0.35
    obstacle_cost_weight: float = 1.5
    unknown_cost_weight: float = 0.8


@dataclass
class NavCrawlConfig:
    """VIEW_POSE 이동을 A* 짧은 구간(waypoint) 연속 이동으로 대체 (H5.1)."""
    enabled: bool = True
    astar_obstacle_cost_weight: float = 2.0
    astar_allow_unknown: bool = False
    astar_max_iterations: int = 20000
    astar_search_margin_m: float = 2.0
    # 거리 등분이 아니라 누적 회전량 기준으로 waypoint 를 나눈다: 회전이
    # 많은 구간(코너/갈지자)은 짧게, 직선 구간은 waypoint 없이 쭉 길게.
    # waypoint_min_spacing_m 은 A* 그리드 계단식 패턴 때문에 매 셀마다
    # waypoint 가 과하게 촘촘히 찍히지 않게 막는 최소 간격일 뿐, 목표
    # 간격이 아니다.
    waypoint_min_spacing_m: float = 0.25
    waypoint_turn_angle_rad: float = math.radians(35.0)
    waypoint_tolerance_m: float = 0.35
    refresh_period_sec: float = 0.5
    reachability_max_ratio: float = 1.6
    early_stop_on_los: bool = True


@dataclass
class BoundaryConfig:
    """BOUNDARY 자동 생성 (H4 예정)."""
    enable_during_target: bool = False
    enable_during_patrol: bool = True
    fan_half_angle_deg: float = 45.0
    angle_step_deg: float = 22.5
    distance_m: float = 1.5
    z: float = 0.3
    period_sec: float = 1.0
    max_queue_size: int = 10
    ttl_sec: float = 10.0

@dataclass
class WaffleConfig:
    """와플 Nav2 클라이언트 설정."""
    frame: str = "map"
    nav_action_name: str = "/navigate_to_pose"


@dataclass
class Config:
    motor: MotorConfig
    calibration: CalibrationConfig
    safety: SafetyConfig
    keyboard: KeyboardConfig
    ibvs: IbvsConfig
    yolo: YoloConfig | None = None
    fire: FireConfig | None = None
    autotrack: AutoTrackConfig | None = None
    patrol: PatrolConfig | None = None
    waffle: WaffleConfig | None = None
    view_pose: ViewPoseConfig | None = None
    nav_crawl: NavCrawlConfig | None = None
    boundary: BoundaryConfig | None = None

def find_config_path(path=None):
    if path is not None:
        p = Path(path)
        if not p.is_file():
            raise FileNotFoundError(f"Config 파일 없음: {p}")
        return p

    here = Path(__file__).resolve()
    candidates = []

    # colcon install 후: share/omx_aim/config/config.yaml
    try:
        from ament_index_python.packages import get_package_share_directory
        candidates.append(
            Path(get_package_share_directory("omx_aim")) / "config" / "config.yaml"
        )
    except Exception:
        pass

    candidates += [
        Path.cwd() / "config.yaml",
        # 소스 트리에서 직접 실행: src/omx_aim/config/config.yaml
        here.parent.parent / "config" / "config.yaml",
    ]
    for c in candidates:
        if c.is_file():
            return c

    raise FileNotFoundError(
        "config.yaml 을 찾을 수 없습니다:\n"
        + "\n".join(f"  - {c}" for c in candidates)
    )


def _tuple_pairs(d):
    out = {}
    for k, v in d.items():
        if len(v) != 2:
            raise ValueError(f"{k}: [lo, hi] 형태여야 하는데 {v}")
        out[k] = (float(v[0]), float(v[1]))
    return out


def load_config(path=None):
    config_path = find_config_path(path)
    with open(config_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    try:
        yolo_cfg = YoloConfig(**raw["yolo"]) if "yolo" in raw else None
        fire_cfg = FireConfig(**raw["fire"]) if "fire" in raw else None
        autotrack_cfg = AutoTrackConfig(**raw["autotrack"]) if "autotrack" in raw else None
        patrol_cfg = PatrolConfig(**raw["patrol"]) if "patrol" in raw else None
        waffle_cfg = WaffleConfig(**raw["waffle"]) if "waffle" in raw else None
        view_pose_cfg = ViewPoseConfig(**raw["view_pose"]) if "view_pose" in raw else None
        nav_crawl_cfg = NavCrawlConfig(**raw["nav_crawl"]) if "nav_crawl" in raw else NavCrawlConfig()
        boundary_cfg = BoundaryConfig(**raw["boundary"]) if "boundary" in raw else None
        cfg = Config(
            motor=MotorConfig(**raw["motor"]),
            calibration=CalibrationConfig(**raw["calibration"]),
            safety=SafetyConfig(
                angle_limits_deg=_tuple_pairs(raw["safety"]["angle_limits_deg"]),
                max_step_deg=raw["safety"]["max_step_deg"],
                large_delta_threshold_tick=raw["safety"]["large_delta_threshold_tick"],
            ),
            keyboard=KeyboardConfig(**raw["keyboard"]),
            ibvs=IbvsConfig(**raw["ibvs"]),
            yolo=yolo_cfg,
            fire=fire_cfg,
            autotrack=autotrack_cfg,
            patrol=patrol_cfg,
            waffle=waffle_cfg,  
            view_pose=view_pose_cfg,
            nav_crawl=nav_crawl_cfg,
            boundary=boundary_cfg,
        )
    except (KeyError, TypeError) as e:
        raise ValueError(
            f"Config 파일 형식 오류 ({config_path}): {e}"
        ) from e

    if yolo_cfg is not None and yolo_cfg.model_path:
        override_model_path = os.environ.get("OMX_YOLO_MODEL_PATH", "").strip()
        if override_model_path:
            yolo_cfg.model_path = override_model_path

        model_path = Path(str(yolo_cfg.model_path)).expanduser()
        if not model_path.is_absolute():
            # Installed config lives in share/omx_aim/config. Prefer package
            # data models, then the launch cwd used by the robot commands.
            candidates = [
                config_path.parent / model_path,
                config_path.parent.parent / "models" / model_path.name,
                Path.cwd() / model_path,
                Path.cwd() / model_path.name,
            ]
            for candidate in candidates:
                if candidate.is_file():
                    yolo_cfg.model_path = str(candidate.resolve())
                    break
            else:
                searched = "\n".join(f"  - {c}" for c in candidates)
                raise FileNotFoundError(
                    "YOLO TensorRT engine file not found. Set OMX_YOLO_MODEL_PATH "
                    f"to an absolute .engine/.plan path, or place {model_path.name} "
                    f"in one of:\n{searched}"
                )
        else:
            yolo_cfg.model_path = str(model_path)

        suffix = Path(str(yolo_cfg.model_path)).suffix.lower()
        if suffix == ".pt":
            raise ValueError(
                "PyTorch YOLO checkpoints are not allowed at runtime. "
                "Use model/target_v3.engine."
            )
        if suffix not in (".engine", ".plan"):
            raise ValueError(
                "YOLO runtime model must be a TensorRT .engine/.plan file, "
                f"got: {yolo_cfg.model_path}"
            )

    camera_device_override = os.environ.get("OMX_YOLO_CAMERA_DEVICE", "").strip()
    if not camera_device_override:
        camera_device_override = os.environ.get(
            "OMX_YOLO_LAUNCH_CAMERA_DEVICE", ""
        ).strip()
    if camera_device_override:
        cfg.ibvs.camera_device = camera_device_override

    camera_backend_override = os.environ.get("OMX_YOLO_CAMERA_BACKEND", "").strip()
    if camera_backend_override:
        cfg.ibvs.camera_backend = camera_backend_override
    reconnect_override = os.environ.get(
        "OMX_YOLO_CAMERA_RECONNECT_PERIOD_SEC", ""
    ).strip()
    if reconnect_override:
        try:
            cfg.ibvs.camera_reconnect_period_sec = float(reconnect_override)
        except ValueError as exc:
            raise ValueError(
                'OMX_YOLO_CAMERA_RECONNECT_PERIOD_SEC must be numeric, got '
                f'{reconnect_override!r}'
            ) from exc

    required_override = os.environ.get("OMX_YOLO_CAMERA_REQUIRED", "").strip()
    if required_override:
        if required_override.lower() not in ('true', 'false'):
            raise ValueError(
                'OMX_YOLO_CAMERA_REQUIRED must be true or false, got '
                f'{required_override!r}'
            )
        cfg.ibvs.camera_required = required_override.lower() == 'true'

    camera_index_override = os.environ.get("OMX_YOLO_CAMERA_INDEX", "").strip()
    if camera_index_override:
        try:
            cfg.ibvs.camera_index = int(camera_index_override)
        except ValueError as exc:
            raise ValueError(
                'OMX_YOLO_CAMERA_INDEX must be an integer camera index, got '
                f'{camera_index_override!r}'
            ) from exc

    return cfg
