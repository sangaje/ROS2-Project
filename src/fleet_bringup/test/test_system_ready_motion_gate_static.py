from pathlib import Path


def test_pose_to_nav2_drops_goals_before_system_ready():
    source = (
        Path(__file__).parents[1]
        / 'fleet_bringup'
        / 'pose_to_nav2.py'
    ).read_text(encoding='utf-8')

    assert "require_system_ready" in source
    assert "system_ready_topic" in source
    assert "NAV2_GOAL_DROPPED_FOR_SYSTEM_NOT_READY" in source
    assert "clear_pending=True" in source


def test_fleet_coordinator_drops_not_stashes_goals_before_system_ready():
    source = (
        Path(__file__).parents[1]
        / 'fleet_bringup'
        / 'fleet_path_coordinator.py'
    ).read_text(encoding='utf-8')

    assert "def _system_ready_blocks_goal" in source
    assert "FLEET_GOAL_DROPPED_FOR_SYSTEM_NOT_READY" in source
    assert "self._clear_all_goal_state()" in source
    assert "self.leader_user_goal = None" in source
    assert "self.follower_user_goal = None" in source
