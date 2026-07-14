import math
from types import SimpleNamespace

from geometry_msgs.msg import PoseStamped
import numpy as np

from bayesian_risk_map.bayesian_risk_map_node import (
    Detection2D,
    RoomAwareRiskMapNode,
)


def make_node():
    node = RoomAwareRiskMapNode.__new__(RoomAwareRiskMapNode)
    node.get_clock = lambda: SimpleNamespace(
        now=lambda: SimpleNamespace(nanoseconds=1_000_000_000)
    )
    node.occ_grid = np.zeros((120, 120), dtype=np.int16)
    node.map_resolution = 0.05
    node.map_origin_x = 0.0
    node.map_origin_y = 0.0
    node.map_origin_yaw = 0.0
    node.allow_unknown = False
    node.free_threshold = 30
    node.occupied_threshold = 65
    node.min_range_m = 0.2
    node.max_range_m = 5.0
    node.camera_hfov_deg = 62.0
    node.visibility_num_rays = 97
    node.source_min_value = 0.03
    node.bearing_sigma_deg = 8.0
    node.angular_sample_step_deg = 1.0
    node.range_sigma_m = 0.20
    node.use_bbox_range_prior = True
    node.evidence_distribution_radius_m = 0.45
    node.positive_projection_mode = 'bearing_consensus'
    node.positive_memory_alpha = 0.85
    node.positive_memory_map = np.zeros_like(node.occ_grid, dtype=np.float32)
    node.bearing_consensus_map = np.zeros_like(node.occ_grid, dtype=np.float32)
    node.detection_candidate_map = np.zeros_like(node.occ_grid, dtype=np.float32)
    node.evidence_points = []
    node.next_evidence_id = 1
    node.enable_visible_risk_decay = True
    node.visible_risk_decay_per_sec = 0.20
    node.visible_risk_decay_grace_sec = 1.0
    node.visible_evidence_clear_threshold = 0.5
    node.last_visible_risk_decay_ros_sec = None
    node.last_leader_miss_capture_sec = None
    node.leader_first_miss_dt_sec = 0.5
    node.leader_person_bayes_miss_log_odds_per_sec = 1.0
    node.leader_visible_risk_decay_per_sec = 1.0
    node.enable_empty_observation_map = True
    node.observed_empty_alpha = 1.0
    node.observed_empty_map = np.zeros_like(node.occ_grid, dtype=np.float32)
    node.risk_persist_in_unknown = True
    node.risk_dirty = False
    node.enable_person_probability_map = True
    node.person_bayes_prior_probability = 0.01
    node.person_bayes_hit_log_odds_gain = 8.0
    node.person_bayes_candidate_power = 0.5
    node.person_bayes_miss_log_odds_per_sec = 0.20
    node.person_bayes_decay_grace_sec = 1.0
    node.person_bayes_max_probability = 0.995
    node.person_bayes_max_update_dt_sec = 1.0
    node.person_log_odds_map = np.zeros_like(node.occ_grid, dtype=np.float32)
    node.person_probability_map = np.zeros_like(node.occ_grid, dtype=np.float32)
    node.person_location_estimate = None
    node.last_person_bayes_update_ros_sec = None
    node.last_person_detection_ros_sec = None

    node.bearing_consensus_sigma_deg = 2.0
    node.bearing_consensus_angle_step_deg = 0.5
    node.bearing_viewpoint_min_baseline_m = 0.20
    node.bearing_min_viewpoints = 2
    node.bearing_support_threshold = 0.12
    node.bearing_consensus_gain = 1.0
    node.bearing_single_view_gain = 0.28
    node.bearing_pair_min_vote = 0.02
    node.bearing_halo_seed_threshold = 0.03
    node.bearing_additional_view_bonus = 0.15
    node.bearing_use_bbox_range_prior = False
    node.bearing_range_sigma_m = 2.0
    node.bearing_observation_max_age_sec = 120.0
    node.bearing_max_viewpoints = 24
    node.bearing_max_observations_per_viewpoint = 8
    node.bearing_same_view_angle_merge_deg = 2.0

    node.bearing_observations = []
    node.next_bearing_observation_id = 1
    node.next_bearing_viewpoint_id = 1
    node.bearing_viewpoint_origins = {}
    node.bearing_consensus_peaks = []
    node.source_halo_seed_threshold = 0.12
    node.source_halo_seed_separation_m = 0.08
    node.source_halo_top_k = 80
    node.source_halo_radius_m = 0.75
    node.source_halo_sigma_m = 0.35
    return node


def detection_for(origin, target, confidence=0.9):
    bearing = math.atan2(target[1] - origin[1], target[0] - origin[0])
    return Detection2D(
        bbox=(0.0, 0.0, 1.0, 1.0),
        conf=confidence,
        bearing_rad=bearing,
        range_hat_m=5.0,
    )


def test_one_viewpoint_creates_low_gain_directional_corridor():
    node = make_node()
    origin = (1.0, 2.0)
    target = (3.0, 3.0)

    node.ingest_bearing_observations(
        (origin[0], origin[1], 0.0),
        [detection_for(origin, target)],
        1.0,
    )

    candidate = node.build_bearing_consensus_map()
    assert len(node.bearing_viewpoint_origins) == 1
    assert 0.15 < float(np.max(candidate)) < 0.35
    assert node.bearing_consensus_peaks == []
    assert node.update_positive_memory(candidate)
    assert float(np.max(node.positive_memory_map)) > 0.12


def test_bbox_bearing_matches_ros_left_right_convention():
    node = make_node()

    center = node.bbox_center_to_bearing((300.0, 0.0, 340.0, 40.0), 640)
    right = node.bbox_center_to_bearing((560.0, 0.0, 620.0, 40.0), 640)
    left = node.bbox_center_to_bearing((20.0, 0.0, 80.0, 40.0), 640)

    assert abs(center) < math.radians(1.0)
    assert right < 0.0
    assert left > 0.0


def test_detection_projection_puts_image_right_on_robot_right():
    node = make_node()
    node.map_origin_y = -3.0
    robot_pose = (1.0, 0.0, 0.0)

    center = Detection2D(
        bbox=(300.0, 0.0, 340.0, 100.0),
        conf=1.0,
        bearing_rad=node.bbox_center_to_bearing((300.0, 0.0, 340.0, 100.0), 640),
        range_hat_m=1.0,
    )
    right = Detection2D(
        bbox=(560.0, 0.0, 620.0, 100.0),
        conf=1.0,
        bearing_rad=node.bbox_center_to_bearing((560.0, 0.0, 620.0, 100.0), 640),
        range_hat_m=1.0,
    )
    left = Detection2D(
        bbox=(20.0, 0.0, 80.0, 100.0),
        conf=1.0,
        bearing_rad=node.bbox_center_to_bearing((20.0, 0.0, 80.0, 100.0), 640),
        range_hat_m=1.0,
    )

    center_map = node.build_detection_candidate_map(robot_pose, [center])
    right_map = node.build_detection_candidate_map(robot_pose, [right])
    left_map = node.build_detection_candidate_map(robot_pose, [left])

    center_y, center_x = np.unravel_index(int(np.argmax(center_map)), center_map.shape)
    right_y, right_x = np.unravel_index(int(np.argmax(right_map)), right_map.shape)
    left_y, left_x = np.unravel_index(int(np.argmax(left_map)), left_map.shape)
    center_world = node.grid_to_world(int(center_x), int(center_y))
    right_world = node.grid_to_world(int(right_x), int(right_y))
    left_world = node.grid_to_world(int(left_x), int(left_y))

    assert center_world[0] > robot_pose[0]
    assert abs(center_world[1] - robot_pose[1]) < 0.10
    assert right_world[0] > robot_pose[0]
    assert right_world[1] < robot_pose[1]
    assert left_world[0] > robot_pose[0]
    assert left_world[1] > robot_pose[1]


def test_distinct_viewpoints_localize_bearing_intersection():
    node = make_node()
    target = (3.0, 3.0)
    first_origin = (1.0, 2.0)
    second_origin = (1.0, 4.0)

    node.ingest_bearing_observations(
        (first_origin[0], first_origin[1], 0.0),
        [detection_for(first_origin, target)],
        1.0,
    )
    node.ingest_bearing_observations(
        (second_origin[0], second_origin[1], 0.0),
        [detection_for(second_origin, target)],
        2.0,
    )

    candidate = node.build_bearing_consensus_map()
    peak_y, peak_x = np.unravel_index(int(np.argmax(candidate)), candidate.shape)
    peak_world = node.grid_to_world(int(peak_x), int(peak_y))

    assert len(node.bearing_viewpoint_origins) == 2
    assert float(np.max(candidate)) > 0.5
    assert math.hypot(peak_world[0] - target[0], peak_world[1] - target[1]) < 0.20


def test_repeated_detection_at_same_position_is_not_independent_support():
    node = make_node()
    origin = (1.0, 2.0)
    target = (3.0, 3.0)
    detection = detection_for(origin, target)

    node.ingest_bearing_observations((origin[0], origin[1], 0.0), [detection], 1.0)
    node.ingest_bearing_observations((origin[0] + 0.05, origin[1], 0.0), [detection], 2.0)

    candidate = node.build_bearing_consensus_map()
    assert len(node.bearing_viewpoint_origins) == 1
    assert 0.15 < float(np.max(candidate)) < 0.35
    assert node.bearing_consensus_peaks == []


def test_repeated_consensus_map_does_not_saturate_memory():
    node = make_node()
    candidate = np.zeros_like(node.positive_memory_map)
    candidate[60, 60] = 0.8

    node.update_positive_memory(candidate)
    first_value = float(node.positive_memory_map[60, 60])
    node.update_positive_memory(candidate)
    second_value = float(node.positive_memory_map[60, 60])

    assert math.isclose(first_value, 0.68, rel_tol=1e-5)
    assert math.isclose(second_value, first_value, rel_tol=1e-6)


def test_low_confidence_single_view_still_creates_halo_seed():
    node = make_node()
    origin = (1.0, 2.0)
    target = (3.0, 3.0)
    low_confidence_detection = detection_for(origin, target, confidence=0.20)

    node.ingest_bearing_observations(
        (origin[0], origin[1], 0.0),
        [low_confidence_detection],
        1.0,
    )
    candidate = node.build_bearing_consensus_map()
    assert node.update_positive_memory(candidate)

    seeds = node.select_source_seeds(node.positive_memory_map)
    assert seeds
    assert max(seed[2] for seed in seeds) >= node.bearing_halo_seed_threshold
    risk = node.build_bounded_geodesic_halo(node.positive_memory_map)
    assert float(np.max(risk)) >= node.bearing_halo_seed_threshold


def test_bayesian_detection_builds_spatial_probability_memory():
    node = make_node()
    candidate = np.zeros_like(node.person_log_odds_map)
    candidate[60, 60] = 0.25
    candidate[60, 61] = 0.0625

    assert node.update_person_bayesian_memory(candidate, None, True, 1.0)

    assert float(node.person_probability_map[60, 60]) > 0.30
    assert 0.05 < float(node.person_probability_map[60, 61]) < float(
        node.person_probability_map[60, 60]
    )
    assert float(node.person_probability_map[10, 10]) == 0.0


def test_bayesian_absence_decays_only_visible_cells_after_grace():
    node = make_node()
    candidate = np.zeros_like(node.person_log_odds_map)
    candidate[60, 60] = 0.25  # visible later
    candidate[60, 70] = 0.25  # outside the later FOV
    visibility = np.zeros_like(candidate)
    visibility[60, 60] = 1.0

    node.update_person_bayesian_memory(candidate, visibility, True, 1.0)
    initial_visible = float(node.person_probability_map[60, 60])
    initial_hidden = float(node.person_probability_map[60, 70])

    # A detector flicker inside the grace period must not erase memory.
    assert not node.update_person_bayesian_memory(None, visibility, False, 1.2)
    assert math.isclose(
        float(node.person_probability_map[60, 60]), initial_visible, rel_tol=1e-6
    )

    # After the grace period, only the visible cell fades and it remains non-zero.
    assert node.update_person_bayesian_memory(None, visibility, False, 2.2)
    decayed_visible = float(node.person_probability_map[60, 60])
    assert 0.0 < decayed_visible < initial_visible
    assert math.isclose(
        float(node.person_probability_map[60, 70]), initial_hidden, rel_tol=1e-6
    )


def test_wall_and_unknown_cells_occlude_visibility_and_protect_memory():
    for occluder_value in (100, -1):
        node = make_node()
        # Make the distinction explicit: unknown space must still occlude camera
        # visibility even if another subsystem allows planning through unknown.
        node.allow_unknown = True
        node.occ_grid[:, 50] = occluder_value
        robot_pose = node.grid_to_world(20, 60) + (0.0,)
        visibility = node.compute_visibility_map(robot_pose)

        front_cell = (60, 40)
        occluded_cell = (60, 70)
        assert float(visibility[front_cell]) == 1.0
        assert float(visibility[60, 50]) == 0.0
        assert float(visibility[occluded_cell]) == 0.0

        candidate = np.zeros_like(node.person_log_odds_map)
        candidate[front_cell] = 0.25
        candidate[occluded_cell] = 0.25
        node.update_person_bayesian_memory(candidate, visibility, True, 1.0)
        initial_front = float(node.person_probability_map[front_cell])
        initial_occluded = float(node.person_probability_map[occluded_cell])

        node.update_person_bayesian_memory(None, visibility, False, 2.2)
        assert float(node.person_probability_map[front_cell]) < initial_front
        assert math.isclose(
            float(node.person_probability_map[occluded_cell]),
            initial_occluded,
            rel_tol=1e-6,
        )


def test_leader_valid_miss_is_consumed_once_and_only_decays_visible_cells():
    node = make_node()
    visible_cell = (60, 60)
    hidden_cell = (60, 70)
    visibility = np.zeros_like(node.occ_grid, dtype=np.float32)
    visibility[visible_cell] = 1.0

    candidate = np.zeros_like(node.person_log_odds_map)
    candidate[visible_cell] = 0.25
    candidate[hidden_cell] = 0.25
    node.update_person_bayesian_memory(candidate, None, True, 1.0)
    node.last_person_detection_ros_sec = None
    before_visible = float(node.person_probability_map[visible_cell])
    before_hidden = float(node.person_probability_map[hidden_cell])

    assert node.apply_leader_valid_no_detection(visibility, 10.0, 10.0)
    after_visible = float(node.person_probability_map[visible_cell])
    after_hidden = float(node.person_probability_map[hidden_cell])

    assert 0.0 < after_visible < before_visible
    assert math.isclose(after_hidden, before_hidden, rel_tol=1e-6)

    # Replaying the same bridge frame cannot apply the miss a second time.
    assert not node.apply_leader_valid_no_detection(visibility, 10.0, 10.1)
    assert math.isclose(
        float(node.person_probability_map[visible_cell]), after_visible, rel_tol=1e-6
    )


def test_leader_observation_uses_local_receipt_time_across_robot_clocks():
    class Clock:
        class Now:
            nanoseconds = 20_000_000_000

        def now(self):
            return self.Now()

    node = make_node()
    node.get_clock = lambda: Clock()
    node.leader_observation_max_age_sec = 1.0
    node.last_leader_observation_sequence = None
    node.leader_observation_wall = 20.0
    node.leader_observation = {
        'sequence': 7,
        'capture_stamp': 20.0,
        'camera_ready': True,
        'frame_valid': True,
        'inference_ran': True,
        'detected': False,
    }

    assert node.consume_leader_observation() == (False, 20.0)
    assert node.consume_leader_observation() is None

    node.leader_observation = {
        'sequence': 8,
        'capture_stamp': 18.0,
        'camera_ready': True,
        'frame_valid': True,
        'inference_ran': True,
        'detected': False,
    }
    # The leader and scout clocks may differ. A freshly received, sequenced
    # frame must still contribute a single visibility miss.
    node.leader_observation_wall = 20.0
    assert node.consume_leader_observation() == (False, 20.0)


def test_leader_pose_falls_back_to_base_heading_without_fresh_camera_yaw():
    class Clock:
        class Now:
            nanoseconds = 20_000_000_000

        def now(self):
            return self.Now()

    node = make_node()
    node.get_clock = lambda: Clock()
    node.leader_pose_max_age_sec = 2.0
    node.leader_camera_yaw = None
    node.leader_camera_yaw_wall = None
    pose = PoseStamped()
    pose.pose.position.x = 1.0
    pose.pose.position.y = 2.0
    pose.pose.orientation.w = 1.0
    node.leader_pose_msg = pose
    node.leader_pose_wall = 20.0

    assert node.get_leader_pose({}) == (1.0, 2.0, 0.0)
