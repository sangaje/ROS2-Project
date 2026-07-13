from pathlib import Path


def test_pose_to_nav2_retains_latest_goal_until_motion_gates_open():
    source = (
        Path(__file__).parents[1]
        / 'fleet_bringup'
        / 'pose_to_nav2.py'
    ).read_text(encoding='utf-8')

    assert "require_system_ready" in source
    assert "system_ready_topic" in source
    assert "NAV2_GOAL_HELD_FOR_SYSTEM_NOT_READY" in source
    assert "latest goal retained" in source
    assert "require_start_motion" in source
    assert "START_MOTION_FALSE_NAV2_CANCELLED" in source
    assert "NAV2_GOAL_HELD_FOR_START_MOTION_FALSE" in source


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
