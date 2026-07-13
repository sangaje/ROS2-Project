import ast
from pathlib import Path


LAUNCH_DIR = Path(__file__).parents[1] / 'launch'
CRITICAL_LAUNCHES = [
    LAUNCH_DIR / 'base.launch.py',
    LAUNCH_DIR / 'leader.launch.py',
    LAUNCH_DIR / 'member.launch.py',
    LAUNCH_DIR / 'follower.launch.py',
]


def _names_used(path):
    tree = ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
    return {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}


def test_critical_launches_do_not_gate_on_process_exit():
    forbidden = {
        'OnProcessExit',
        'RegisterEventHandler',
        'EmitEvent',
        'Shutdown',
    }

    for path in CRITICAL_LAUNCHES:
        assert forbidden.isdisjoint(_names_used(path)), path


def test_leader_localization_uses_local_scan_for_amcl_and_costmaps():
    text = (LAUNCH_DIR / 'leader.launch.py').read_text(encoding='utf-8')
    assert "'amcl_scan_topic'," in text
    assert "'costmap_scan_topic'," in text
    assert text.count("default_value='/scan'") >= 2
    assert "or '/scan'" in text
    assert "or '/scan_filtered'" not in text
    assert "'topic': scan_topic_value" not in text
    assert "'scan_topic': scan_topic_value" not in text


def test_leader_does_not_seed_from_scout_pose_by_default():
    text = (LAUNCH_DIR / 'leader.launch.py').read_text(encoding='utf-8')
    assert "'enable_scout_pose_seed': False" in text
    assert "'allow_blind_global_reinit': False" in text
    assert "'freeze_when_stationary': False" in text


def test_fixed_seed_defaults_do_not_global_localize_or_copy_other_robot_pose():
    leader = (LAUNCH_DIR / 'leader.launch.py').read_text(encoding='utf-8')
    follower = (LAUNCH_DIR / 'follower.launch.py').read_text(encoding='utf-8')
    member = (LAUNCH_DIR / 'member.launch.py').read_text(encoding='utf-8')

    assert "DeclareLaunchArgument('leader_initial_y', default_value='0.10')" in leader
    assert "'auto_localize',\n            default_value='false'" in leader
    assert "executable='amcl_fixed_seed_ready'" in leader

    assert "DeclareLaunchArgument('follower_initial_x', default_value='0.0')" in follower
    assert "DeclareLaunchArgument('follower_initial_y', default_value='-0.10')" in follower
    assert "'start_legacy_follower',\n            default_value='false'" in follower
    assert "'auto_localize',\n            default_value='false'" in follower
    assert "'enable_scout_pose_seed': False" in follower
    assert "executable='amcl_fixed_seed_ready'" in follower

    assert "'auto_localize',\n            default_value='false'" in member
    assert "'enable_scout_pose_seed': False" in member
    assert "executable='amcl_fixed_seed_ready'" in member
