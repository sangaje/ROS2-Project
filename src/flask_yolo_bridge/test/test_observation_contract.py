from flask_yolo_bridge.observation_contract import (
    PoseSample,
    build_observation_metadata,
    closest_pose_sample,
    echo_observation_metadata,
    parse_role_payload,
)


def test_closest_pose_uses_capture_time_not_latest_pose():
    samples = [
        PoseSample(10.0, 1.0, 2.0, 0.1),
        PoseSample(10.2, 3.0, 4.0, 0.2),
        PoseSample(13.0, 99.0, 99.0, 9.9),
    ]

    pose, error = closest_pose_sample(samples, 10.12, 0.15)

    assert pose == samples[1]
    assert round(error, 3) == 0.08


def test_closest_pose_rejects_missing_capture_pose():
    samples = [PoseSample(10.0, 1.0, 2.0, 0.1)]

    pose, error = closest_pose_sample(samples, 11.0, 0.2)

    assert pose is None
    assert round(error, 3) == 1.0


def test_observation_metadata_contains_atomic_capture_pose():
    meta = build_observation_metadata(
        robot_id='scout22',
        boot_id='boot-a',
        sequence=7,
        role='ACTIVE_SCOUT',
        role_epoch=3,
        frame_id='camera_link',
        camera_hfov_deg=62.0,
        capture_ros_sec=100.0,
        capture_wall_sec=200.0,
        capture_mono_sec=50.0,
        send_start_mono_sec=50.123,
        pose=PoseSample(99.98, 1.25, -0.5, 0.75),
        pose_time_error_sec=0.02,
        image_width=640,
        image_height=480,
        calibration_id='cal-v1',
    )

    assert meta['robot_id'] == 'scout22'
    assert meta['sequence'] == '7'
    assert meta['role_epoch'] == '3'
    assert meta['capture_pose_x'] == '1.250000'
    assert meta['capture_pose_y'] == '-0.500000'
    assert meta['capture_pose_yaw'] == '0.750000000'
    assert meta['pose_time_error_ms'] == '20.000'
    assert meta['capture_to_send_delay_ms'] == '123.000'
    assert meta['camera_calibration_id'] == 'cal-v1'


def test_server_echo_preserves_observation_metadata():
    echoed = echo_observation_metadata({
        'robot_id': 'follower21',
        'boot_id': 'boot-b',
        'sequence': '4',
        'capture_pose_x': '5.0',
        'capture_pose_y': '6.0',
        'capture_pose_yaw': '1.0',
        'ignored': 'x',
    })

    assert echoed == {
        'robot_id': 'follower21',
        'boot_id': 'boot-b',
        'sequence': '4',
        'capture_pose_x': '5.0',
        'capture_pose_y': '6.0',
        'capture_pose_yaw': '1.0',
    }


def test_role_payload_extracts_role_and_epoch():
    role, epoch = parse_role_payload(
        '{"role":"FOLLOWER","epoch":12}',
        'ACTIVE_SCOUT',
    )

    assert role == 'FOLLOWER'
    assert epoch == 12
