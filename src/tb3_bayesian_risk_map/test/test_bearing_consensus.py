import math

import numpy as np

from tb3_bayesian_risk_map.bayesian_risk_map_node import (
    Detection2D,
    RoomAwareRiskMapNode,
)


def make_node():
    node = RoomAwareRiskMapNode.__new__(RoomAwareRiskMapNode)
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
    node.source_min_value = 0.03
    node.evidence_distribution_radius_m = 0.45
    node.positive_projection_mode = 'bearing_consensus'
    node.positive_memory_alpha = 0.85
    node.positive_memory_map = np.zeros_like(node.occ_grid, dtype=np.float32)
    node.risk_persist_in_unknown = True
    node.risk_dirty = False

    node.bearing_consensus_sigma_deg = 2.0
    node.bearing_consensus_angle_step_deg = 0.5
    node.bearing_viewpoint_min_baseline_m = 0.20
    node.bearing_min_viewpoints = 2
    node.bearing_support_threshold = 0.12
    node.bearing_consensus_gain = 1.0
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
    return node


def detection_for(origin, target, confidence=0.9):
    bearing = math.atan2(target[1] - origin[1], target[0] - origin[0])
    return Detection2D(
        bbox=(0.0, 0.0, 1.0, 1.0),
        conf=confidence,
        bearing_rad=bearing,
        range_hat_m=5.0,
    )


def test_one_viewpoint_does_not_create_range_guess():
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
    assert float(np.max(candidate)) == 0.0


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
    assert float(np.max(candidate)) == 0.0


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
