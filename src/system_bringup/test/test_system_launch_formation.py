from pathlib import Path


SYSTEM_LAUNCH = Path(__file__).parents[1] / 'launch' / 'system.launch.py'


def test_three_robot_initial_formation_defaults_are_scout_centered():
    text = SYSTEM_LAUNCH.read_text(encoding='utf-8')

    assert "'leader_initial_y'," in text
    assert "default_value='0.10'" in text
    assert "'scout_initial_y'," in text
    assert "default_value='0.0'" in text
    assert "'follower_initial_y'," in text
    assert "default_value='-0.10'" in text


def test_system_launch_passes_initial_pose_to_fleet_launches():
    text = SYSTEM_LAUNCH.read_text(encoding='utf-8')

    assert "fleet_launch_args['leader_initial_y']" in text
    assert "fleet_launch_args['member_initial_y']" in text
    assert "fleet_launch_args['follower_initial_y']" in text


def test_system_launch_uses_fixed_seed_localization_and_faster_nav2_leader_defaults():
    text = SYSTEM_LAUNCH.read_text(encoding='utf-8')

    assert "'auto_localize', default_value='false'" in text
    assert "'leader_auto_localize', default_value='false'" in text
    assert "'leader_shadow_max_linear_vel',\n            default_value='0.26'" in text
    assert "'leader_shadow_catchup_max_linear_vel',\n            default_value='0.26'" in text
    assert "'leader_shadow_max_angular_vel',\n            default_value='1.00'" in text
    assert "'leader_shadow_goal_update_period_sec',\n            default_value='1.0'" in text
    assert "'leader_shadow_goal_min_change_m',\n            default_value='0.35'" in text
    assert "'follow_goal_update_distance_m': 0.30" in text
    assert "'follow_startup_leader_motion_m': 0.30" in text


def test_scout_member_pose_is_stabilized_when_cartographer_corrects_while_stopped():
    text = (
        Path(__file__).parents[2]
        / 'fleet_bringup'
        / 'launch'
        / 'member.launch.py'
    ).read_text(encoding='utf-8')

    assert "'freeze_when_stationary': True" in text
    assert "'stationary_linear_threshold_m': 0.003" in text


def test_system_launch_uses_external_worker_without_in_process_runtime():
    text = SYSTEM_LAUNCH.read_text(encoding='utf-8')

    assert "default_value='external_worker'" in text
    assert "'start_rl_worker'" in text
    assert "'scout_rl_inference.launch.py'" in text
    assert "'initial_role_active': 'false'" in text
    assert "'rl_backend': rl_backend_value" in text
    assert "'require_video_ready': require_video_ready.perform(context)" in text
    assert "'video_ready_topic': video_ready_topic.perform(context)" in text


def test_system_launch_gates_motion_on_dashboard_video_ready():
    text = SYSTEM_LAUNCH.read_text(encoding='utf-8')

    assert "'require_video_ready'" in text
    assert "default_value='true'" in text
    assert "'video_ready_topic'" in text
    assert "default_value='/fleet/video_ready'" in text
    assert "'require_video_ready': launch_bool(" in text
    assert "'video_ready_topic': video_ready_topic.perform(context)" in text


def test_system_launch_starts_global_readiness_monitor_and_motion_gates():
    text = SYSTEM_LAUNCH.read_text(encoding='utf-8')

    assert "executable='system_readiness_monitor'" in text
    assert "'ready_topic': '/system/ready'" in text
    assert "'readiness_topic': '/system/readiness'" in text
    assert "'detail_topic': '/system/readiness_detail'" in text
    assert "'require_system_ready': True" in text
    assert "'system_ready_topic': '/system/ready'" in text
    assert "'require_system_ready': 'true'" in text


def test_leader_can_own_risk_map_from_scout_sources():
    text = SYSTEM_LAUNCH.read_text(encoding='utf-8')

    assert "'start_leader_risk_map'" in text
    assert 'include_risk_outputs=not launch_bool' in text
    assert "package='bayesian_risk_map'" in text
    assert "'pose_topic': scout_pose_topic.perform(context)" in text
    assert "'map_qos_durability': 'transient_local'" in text
    assert "'detection_source': 'flask_topic'" in text
    assert "'target_class': '-1'" in text
    assert "'positive_projection_mode': 'range_cone'" in text
    assert "'detection_timeout_sec': 6.0" in text
    assert "'detection_reuse_max_distance_m': 2.5" in text
    assert "'enable_visibility_tracking': True" in text
    assert "'leader_visible_risk_decay_per_sec': 3.5" in text


def test_scout_can_run_cartographer_without_local_risk_map():
    text = SYSTEM_LAUNCH.read_text(encoding='utf-8')

    assert 'or launch_bool(start_cartographer.perform(context))' in text
    assert "'start_risk_map': start_risk_map.perform(context)" in text


def test_follower_map_forwarding_is_explicit_for_takeover_commit():
    text = SYSTEM_LAUNCH.read_text(encoding='utf-8')
    leader_launch = (
        Path(__file__).parents[2] / 'fleet_bringup' / 'launch' / 'leader.launch.py'
    ).read_text(encoding='utf-8')

    assert "fleet_role_value == 'follower'" in text
    assert "fleet_launch_args['forward_map_to_main']" in text
    assert "forward_field_map_to_main.perform(context)" in text
    assert "f'/field/{follower_robot_name.perform(context)}/map'" in text
    assert "'active_scout_id_topic': active_scout_id_topic" in leader_launch
    assert "'follower_input_topic': follower_map_bridge_topic" in leader_launch
