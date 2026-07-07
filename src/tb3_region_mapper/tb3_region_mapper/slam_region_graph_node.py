#!/usr/bin/env python3

import heapq
import json
import math
import time
from collections import deque
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Set, Tuple

import rclpy
from geometry_msgs.msg import Point, PoseStamped
from nav_msgs.msg import OccupancyGrid
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import String
from visualization_msgs.msg import Marker, MarkerArray

import tf2_ros


Cell = Tuple[int, int]


@dataclass
class GridInfo:
    width: int
    height: int
    resolution: float
    origin_x: float
    origin_y: float


@dataclass
class Pose2D:
    x: float
    y: float
    yaw: float


@dataclass
class RawRegion:
    cells: Set[Cell]
    centroid: Tuple[float, float]
    area_m2: float
    mean_clearance: float
    max_clearance: float
    elongation: float
    occupied_boundary_len: float
    unknown_boundary_len: float
    gateway_boundary_len: float
    closure_score: float
    unknown_boundary_ratio: float
    frontier_cell_count: int
    state: str = 'provisional'


@dataclass
class TrackedRegion(RawRegion):
    id: int = -1
    stable_count: int = 1
    created_time: float = 0.0
    updated_time: float = 0.0


@dataclass
class LockedRoom:
    id: int
    name: str
    pose: Tuple[float, float, float]
    source_centroid: Tuple[float, float]
    area_m2: float
    locked_time: float
    last_seen_time: float
    observations: int = 1
    state: str = 'STABLE_ROOM'
    visible: bool = True


@dataclass
class FrontierPortal:
    id: int
    parent_region_id: int
    cells: Set[Cell]
    centroid: Tuple[float, float]
    outward_yaw: float
    width: float


@dataclass
class GatewayCut:
    id: int
    cells: Set[Cell]
    centroid: Tuple[float, float]
    clearance: float
    confidence: float


class SlamRegionGraphNode(Node):
    """Extract provisional/open/stable regions from an incremental SLAM OccupancyGrid.

    This node is intentionally passive: it does not command the robot.  It is meant
    for teleop experiments in Gazebo/real robot SLAM.  The output is a region map
    plus RViz markers that show:
      - known-free provisional/open/stable regions,
      - latent unknown regions attached to frontiers,
      - approximate GVD/Voronoi cells,
      - accepted bottleneck/gateway cuts.
    """

    def __init__(self):
        super().__init__('slam_region_graph')

        # Frames / topics
        # 'use_sim_time' is a special ROS parameter and may already be declared
        # by rclpy when launch passes it as an override. Do not declare it here.
        self.declare_parameter('map_topic', '/map')
        self.declare_parameter('global_frame', 'map')
        self.declare_parameter('robot_frame', 'base_footprint')
        self.declare_parameter('region_map_topic', '/slam_region_graph/region_map')
        self.declare_parameter('marker_topic', '/slam_region_graph/markers')
        self.declare_parameter('summary_topic', '/slam_region_graph/summary')
        self.declare_parameter('selected_region_topic', '/slam_region_graph/selected_region_goal')
        self.declare_parameter('locked_rooms_topic', '/slam_region_graph/locked_rooms')

        # Occupancy interpretation
        # The raw Cartographer OccupancyGrid is noisy and not always binary.
        # v8 uses an adaptive occupancy filter so locally dark cells and
        # probability bands that look wall-like are treated as obstacles instead
        # of blindly accepting every value <= free_threshold as traversable.
        self.declare_parameter('free_threshold', 58)
        self.declare_parameter('occupied_threshold', 65)
        self.declare_parameter('enable_adaptive_occupancy_filter', True)
        self.declare_parameter('adaptive_occupied_min', 45)
        self.declare_parameter('adaptive_occupied_max', 68)
        self.declare_parameter('adaptive_occupied_margin', 2)
        self.declare_parameter('enable_local_dark_obstacle_filter', False)
        self.declare_parameter('dark_obstacle_min_value', 55)
        self.declare_parameter('dark_obstacle_local_contrast', 18)
        self.declare_parameter('dark_obstacle_neighbor_radius_m', 0.12)
        self.declare_parameter('wall_inflation_radius_m', 0.02)
        self.declare_parameter('obstacle_min_cluster_cells', 3)

        # v11: Cartographer maps are not binary.  Free/occupied cells may
        # appear as broad probability bands rather than clean 0/100 values.
        # Therefore wall extraction is done with an iterative high-pass/ridge
        # filter over known cells.  Absolute high values are still accepted as
        # obstacles, but mid-valued cells become obstacles only when they are
        # locally much darker than their neighbourhood and survive several
        # smoothing scales.
        self.declare_parameter('enable_iterative_highpass_wall_filter', True)
        self.declare_parameter('highpass_wall_iterations', 4)
        self.declare_parameter('highpass_smooth_radius_m', 0.18)
        self.declare_parameter('highpass_radius_growth_m', 0.04)
        self.declare_parameter('highpass_wall_min_value', 52)
        self.declare_parameter('highpass_wall_contrast', 12)
        self.declare_parameter('highpass_wall_votes_min', 3)
        self.declare_parameter('highpass_hysteresis_min_value', 45)
        self.declare_parameter('known_non_obstacle_is_free', True)
        self.declare_parameter('free_mask_denoise_iterations', 6)
        self.declare_parameter('free_fill_neighbor_min', 4)
        self.declare_parameter('free_keep_neighbor_min', 1)
        # v12: fill visually sparse maps.  Cartographer often leaves small
        # low-confidence/unknown holes inside already observed rooms.  These
        # should not break a region-id overlay, so small unknown islands
        # surrounded by free evidence are converted to free before room split.
        self.declare_parameter('fill_unknown_holes_as_free', True)
        self.declare_parameter('unknown_hole_fill_max_area_m2', 2.50)
        self.declare_parameter('unknown_hole_fill_min_free_boundary_ratio', 0.45)
        self.declare_parameter('unknown_hole_fill_max_occ_boundary_ratio', 0.28)
        self.declare_parameter('region_dense_fill_iterations', 30)
        self.declare_parameter('region_dense_fill_neighbor_min', 1)
        # v22: every known/reachable free cell must belong to one region.
        # Watershed separators are useful internally, but leaving them as -1 in
        # /slam_region_graph/region_map creates holes that confuse coverage-vs-region
        # planning.  Assign all free cells to the nearest region label before publishing.
        self.declare_parameter('force_assign_all_free_to_regions', True)
        self.declare_parameter('force_assign_source', 'reachable')  # reachable | all_free
        self.declare_parameter('force_assign_max_bridge_m', 999.0)
        self.declare_parameter('use_reachable_only', True)
        self.declare_parameter('fallback_to_all_free_without_tf', False)
        self.declare_parameter('fallback_to_all_free_when_robot_off_map', False)
        self.declare_parameter('robot_free_snap_radius_m', 0.35)

        # Update scheduling
        self.declare_parameter('timer_period', 0.25)
        self.declare_parameter('region_update_period', 1.0)
        self.declare_parameter('map_stable_time', 0.60)
        self.declare_parameter('force_update_without_map_delta', True)
        self.declare_parameter('min_changed_cells_for_update', 25)

        # Online SLAM maps are noisy while the robot is moving: Cartographer can
        # briefly change soft occupancy values near scan edges, which makes the
        # watershed/door split flicker.  For RViz and high-level exploration we
        # keep the last committed region map while the robot is moving and only
        # recompute the region graph when the robot is slow/stationary.
        self.declare_parameter('hold_region_graph_while_moving', True)
        self.declare_parameter('region_hold_linear_speed_mps', 0.055)
        self.declare_parameter('region_hold_angular_speed_rps', 0.22)
        self.declare_parameter('region_hold_republish_sec', 0.75)

        # Region extraction
        self.declare_parameter('min_region_area_m2', 0.10)
        self.declare_parameter('min_region_cells', 12)
        self.declare_parameter('region_connectivity_8', False)
        self.declare_parameter('max_regions', 80)
        self.declare_parameter('min_frontier_cluster_size', 4)
        self.declare_parameter('frontier_cluster_connectivity_8', True)

        # Approximate GVD / bottleneck cuts
        self.declare_parameter('enable_gvd', True)
        self.declare_parameter('enable_bottleneck_cuts', True)
        self.declare_parameter('gvd_min_clearance', 0.18)
        self.declare_parameter('door_clearance_max', 0.55)
        self.declare_parameter('door_clearance_min', 0.18)
        self.declare_parameter('max_gateway_cuts', 8)
        self.declare_parameter('cut_line_half_length', 0.42)
        self.declare_parameter('cut_line_width', 0.10)
        self.declare_parameter('cut_test_min_component_area_m2', 0.12)
        self.declare_parameter('cut_test_max_candidates', 32)

        # Extra aggressive doorway/entrance split.  This stage looks for
        # low-clearance medial cells and inserts cross-cuts at entrances.  It is
        # intentionally stricter than generic GVD visualization because room
        # segmentation must break at doors, not merely draw a skeleton.
        self.declare_parameter('enable_low_clearance_doorway_cuts', True)
        self.declare_parameter('doorway_cut_clearance_max', 0.46)
        self.declare_parameter('doorway_cut_clearance_min', 0.16)
        self.declare_parameter('doorway_cut_local_min_margin', 0.035)
        self.declare_parameter('doorway_cut_min_cluster_cells', 2)
        self.declare_parameter('doorway_cut_max_candidates', 40)
        self.declare_parameter('doorway_cut_duplicate_distance_m', 0.32)
        self.declare_parameter('doorway_cut_force_half_length_m', 0.68)
        self.declare_parameter('doorway_cut_force_width_m', 0.12)

        # Morphological room/zone split. This is the main room partition stage.
        # It uses wall-distance erosion to remove narrow doorways/bottlenecks,
        # builds seed components in the remaining open cores, then grows labels
        # back over the original reachable free-space.
        self.declare_parameter('enable_morphological_room_split', True)
        self.declare_parameter('room_seed_erosion_radius_m', 0.40)
        self.declare_parameter('room_seed_min_area_m2', 0.10)
        self.declare_parameter('room_seed_min_cells', 18)
        self.declare_parameter('room_split_connectivity_8', False)
        self.declare_parameter('region_separator_width_cells', 1)
        self.declare_parameter('region_separator_max_clearance_m', 0.55)
        self.declare_parameter('merge_tiny_split_regions', True)
        self.declare_parameter('tiny_split_region_area_m2', 0.08)
        self.declare_parameter('min_morphological_regions', 2)
        self.declare_parameter('use_clearance_priority_watershed', True)
        self.declare_parameter('watershed_conflict_clearance_max_m', 0.55)
        self.declare_parameter('watershed_separator_width_cells', 1)

        # Region state / tracking
        self.declare_parameter('region_match_max_distance', 0.85)
        self.declare_parameter('region_match_area_ratio_min', 0.25)
        self.declare_parameter('persist_region_ids', True)
        self.declare_parameter('region_match_overlap_ratio_min', 0.08)
        self.declare_parameter('region_reacquire_max_distance', 1.25)
        self.declare_parameter('stable_confirm_updates', 3)
        self.declare_parameter('open_unknown_boundary_ratio', 0.28)
        self.declare_parameter('stable_unknown_boundary_ratio', 0.16)
        self.declare_parameter('stable_closure_min', 0.50)
        self.declare_parameter('corridor_elongation_min', 4.5)
        self.declare_parameter('lock_stable_rooms', True)
        self.declare_parameter('lock_stable_room_pose', True)
        self.declare_parameter('lock_provisional_room_candidates', True)
        self.declare_parameter('locked_room_min_stable_count', 6)
        self.declare_parameter('locked_room_min_area_m2', 0.25)
        self.declare_parameter('locked_room_min_clearance_m', 0.30)
        self.declare_parameter('locked_room_claim_radius_m', 0.90)
        self.declare_parameter('lock_start_room_as_r1', True)
        self.declare_parameter('start_room_id', 1)
        self.declare_parameter('start_room_claim_radius_m', 1.25)
        self.declare_parameter('start_room_pose_from_robot', True)
        self.declare_parameter('publish_locked_room_markers', True)

        # Marker visualization. v9 defaults to a clean region-id-only view:
        # only colored region cells and compact R{id} labels are shown.  GVD,
        # gateway cuts, frontier arrows, and obstacle/debug encodings are still
        # computed internally and reported in summary, but hidden unless enabled.
        self.declare_parameter('region_map_id_only', True)
        self.declare_parameter('publish_gvd_markers', False)
        self.declare_parameter('publish_gateway_markers', False)
        self.declare_parameter('publish_region_text', True)
        self.declare_parameter('publish_unlocked_region_text', False)
        self.declare_parameter('region_text_id_only', True)
        self.declare_parameter('publish_latent_frontiers', False)
        self.declare_parameter('publish_region_outlines', False)
        self.declare_parameter('region_marker_alpha', 0.84)
        self.declare_parameter('region_outline_width_m', 0.035)
        self.declare_parameter('max_region_marker_cells', 100000)
        self.declare_parameter('marker_z', 0.035)
        self.declare_parameter('text_z', 0.35)
        self.declare_parameter('max_gvd_marker_points', 2500)

        self.map_topic = str(self.get_parameter('map_topic').value)
        self.global_frame = str(self.get_parameter('global_frame').value)
        self.robot_frame = str(self.get_parameter('robot_frame').value)
        self.region_map_topic = str(self.get_parameter('region_map_topic').value)
        self.marker_topic = str(self.get_parameter('marker_topic').value)
        self.summary_topic = str(self.get_parameter('summary_topic').value)
        self.selected_region_topic = str(self.get_parameter('selected_region_topic').value)
        self.locked_rooms_topic = str(self.get_parameter('locked_rooms_topic').value)

        self.free_threshold = int(self.get_parameter('free_threshold').value)
        self.occupied_threshold = int(self.get_parameter('occupied_threshold').value)
        self.enable_adaptive_occupancy_filter = bool(self.get_parameter('enable_adaptive_occupancy_filter').value)
        self.adaptive_occupied_min = int(self.get_parameter('adaptive_occupied_min').value)
        self.adaptive_occupied_max = int(self.get_parameter('adaptive_occupied_max').value)
        self.adaptive_occupied_margin = int(self.get_parameter('adaptive_occupied_margin').value)
        self.enable_local_dark_obstacle_filter = bool(self.get_parameter('enable_local_dark_obstacle_filter').value)
        self.dark_obstacle_min_value = int(self.get_parameter('dark_obstacle_min_value').value)
        self.dark_obstacle_local_contrast = int(self.get_parameter('dark_obstacle_local_contrast').value)
        self.dark_obstacle_neighbor_radius_m = float(self.get_parameter('dark_obstacle_neighbor_radius_m').value)
        self.wall_inflation_radius_m = float(self.get_parameter('wall_inflation_radius_m').value)
        self.obstacle_min_cluster_cells = int(self.get_parameter('obstacle_min_cluster_cells').value)
        self.enable_iterative_highpass_wall_filter = bool(self.get_parameter('enable_iterative_highpass_wall_filter').value)
        self.highpass_wall_iterations = int(self.get_parameter('highpass_wall_iterations').value)
        self.highpass_smooth_radius_m = float(self.get_parameter('highpass_smooth_radius_m').value)
        self.highpass_radius_growth_m = float(self.get_parameter('highpass_radius_growth_m').value)
        self.highpass_wall_min_value = int(self.get_parameter('highpass_wall_min_value').value)
        self.highpass_wall_contrast = int(self.get_parameter('highpass_wall_contrast').value)
        self.highpass_wall_votes_min = int(self.get_parameter('highpass_wall_votes_min').value)
        self.highpass_hysteresis_min_value = int(self.get_parameter('highpass_hysteresis_min_value').value)
        self.known_non_obstacle_is_free = bool(self.get_parameter('known_non_obstacle_is_free').value)
        self.free_mask_denoise_iterations = int(self.get_parameter('free_mask_denoise_iterations').value)
        self.free_fill_neighbor_min = int(self.get_parameter('free_fill_neighbor_min').value)
        self.free_keep_neighbor_min = int(self.get_parameter('free_keep_neighbor_min').value)
        self.fill_unknown_holes_as_free = bool(self.get_parameter('fill_unknown_holes_as_free').value)
        self.unknown_hole_fill_max_area_m2 = float(self.get_parameter('unknown_hole_fill_max_area_m2').value)
        self.unknown_hole_fill_min_free_boundary_ratio = float(self.get_parameter('unknown_hole_fill_min_free_boundary_ratio').value)
        self.unknown_hole_fill_max_occ_boundary_ratio = float(self.get_parameter('unknown_hole_fill_max_occ_boundary_ratio').value)
        self.region_dense_fill_iterations = int(self.get_parameter('region_dense_fill_iterations').value)
        self.region_dense_fill_neighbor_min = int(self.get_parameter('region_dense_fill_neighbor_min').value)
        self.force_assign_all_free_to_regions = bool(self.get_parameter('force_assign_all_free_to_regions').value)
        self.force_assign_source = str(self.get_parameter('force_assign_source').value)
        self.force_assign_max_bridge_m = float(self.get_parameter('force_assign_max_bridge_m').value)
        self.use_reachable_only = bool(self.get_parameter('use_reachable_only').value)
        self.fallback_to_all_free_without_tf = bool(self.get_parameter('fallback_to_all_free_without_tf').value)
        self.fallback_to_all_free_when_robot_off_map = bool(self.get_parameter('fallback_to_all_free_when_robot_off_map').value)
        self.robot_free_snap_radius_m = float(self.get_parameter('robot_free_snap_radius_m').value)

        self.timer_period = float(self.get_parameter('timer_period').value)
        self.region_update_period = float(self.get_parameter('region_update_period').value)
        self.map_stable_time = float(self.get_parameter('map_stable_time').value)
        self.force_update_without_map_delta = bool(self.get_parameter('force_update_without_map_delta').value)
        self.min_changed_cells_for_update = int(self.get_parameter('min_changed_cells_for_update').value)
        self.hold_region_graph_while_moving = bool(self.get_parameter('hold_region_graph_while_moving').value)
        self.region_hold_linear_speed_mps = float(self.get_parameter('region_hold_linear_speed_mps').value)
        self.region_hold_angular_speed_rps = float(self.get_parameter('region_hold_angular_speed_rps').value)
        self.region_hold_republish_sec = float(self.get_parameter('region_hold_republish_sec').value)

        self.min_region_area_m2 = float(self.get_parameter('min_region_area_m2').value)
        self.min_region_cells = int(self.get_parameter('min_region_cells').value)
        self.region_connectivity_8 = bool(self.get_parameter('region_connectivity_8').value)
        self.max_regions = int(self.get_parameter('max_regions').value)
        self.min_frontier_cluster_size = int(self.get_parameter('min_frontier_cluster_size').value)
        self.frontier_cluster_connectivity_8 = bool(self.get_parameter('frontier_cluster_connectivity_8').value)

        self.enable_gvd = bool(self.get_parameter('enable_gvd').value)
        self.enable_bottleneck_cuts = bool(self.get_parameter('enable_bottleneck_cuts').value)
        self.gvd_min_clearance = float(self.get_parameter('gvd_min_clearance').value)
        self.door_clearance_max = float(self.get_parameter('door_clearance_max').value)
        self.door_clearance_min = float(self.get_parameter('door_clearance_min').value)
        self.max_gateway_cuts = int(self.get_parameter('max_gateway_cuts').value)
        self.cut_line_half_length = float(self.get_parameter('cut_line_half_length').value)
        self.cut_line_width = float(self.get_parameter('cut_line_width').value)
        self.cut_test_min_component_area_m2 = float(self.get_parameter('cut_test_min_component_area_m2').value)
        self.cut_test_max_candidates = int(self.get_parameter('cut_test_max_candidates').value)
        self.enable_low_clearance_doorway_cuts = bool(self.get_parameter('enable_low_clearance_doorway_cuts').value)
        self.doorway_cut_clearance_max = float(self.get_parameter('doorway_cut_clearance_max').value)
        self.doorway_cut_clearance_min = float(self.get_parameter('doorway_cut_clearance_min').value)
        self.doorway_cut_local_min_margin = float(self.get_parameter('doorway_cut_local_min_margin').value)
        self.doorway_cut_min_cluster_cells = int(self.get_parameter('doorway_cut_min_cluster_cells').value)
        self.doorway_cut_max_candidates = int(self.get_parameter('doorway_cut_max_candidates').value)
        self.doorway_cut_duplicate_distance_m = float(self.get_parameter('doorway_cut_duplicate_distance_m').value)
        self.doorway_cut_force_half_length_m = float(self.get_parameter('doorway_cut_force_half_length_m').value)
        self.doorway_cut_force_width_m = float(self.get_parameter('doorway_cut_force_width_m').value)

        self.enable_morphological_room_split = bool(self.get_parameter('enable_morphological_room_split').value)
        self.room_seed_erosion_radius_m = float(self.get_parameter('room_seed_erosion_radius_m').value)
        self.room_seed_min_area_m2 = float(self.get_parameter('room_seed_min_area_m2').value)
        self.room_seed_min_cells = int(self.get_parameter('room_seed_min_cells').value)
        self.room_split_connectivity_8 = bool(self.get_parameter('room_split_connectivity_8').value)
        self.region_separator_width_cells = int(self.get_parameter('region_separator_width_cells').value)
        self.region_separator_max_clearance_m = float(self.get_parameter('region_separator_max_clearance_m').value)
        self.merge_tiny_split_regions = bool(self.get_parameter('merge_tiny_split_regions').value)
        self.tiny_split_region_area_m2 = float(self.get_parameter('tiny_split_region_area_m2').value)
        self.min_morphological_regions = int(self.get_parameter('min_morphological_regions').value)
        self.use_clearance_priority_watershed = bool(self.get_parameter('use_clearance_priority_watershed').value)
        self.watershed_conflict_clearance_max_m = float(self.get_parameter('watershed_conflict_clearance_max_m').value)
        self.watershed_separator_width_cells = int(self.get_parameter('watershed_separator_width_cells').value)

        self.region_match_max_distance = float(self.get_parameter('region_match_max_distance').value)
        self.region_match_area_ratio_min = float(self.get_parameter('region_match_area_ratio_min').value)
        self.persist_region_ids = bool(self.get_parameter('persist_region_ids').value)
        self.region_match_overlap_ratio_min = float(self.get_parameter('region_match_overlap_ratio_min').value)
        self.region_reacquire_max_distance = float(self.get_parameter('region_reacquire_max_distance').value)
        self.stable_confirm_updates = int(self.get_parameter('stable_confirm_updates').value)
        self.open_unknown_boundary_ratio = float(self.get_parameter('open_unknown_boundary_ratio').value)
        self.stable_unknown_boundary_ratio = float(self.get_parameter('stable_unknown_boundary_ratio').value)
        self.stable_closure_min = float(self.get_parameter('stable_closure_min').value)
        self.corridor_elongation_min = float(self.get_parameter('corridor_elongation_min').value)
        self.lock_stable_rooms = bool(self.get_parameter('lock_stable_rooms').value)
        self.lock_stable_room_pose = bool(self.get_parameter('lock_stable_room_pose').value)
        self.lock_provisional_room_candidates = bool(self.get_parameter('lock_provisional_room_candidates').value)
        self.locked_room_min_stable_count = int(self.get_parameter('locked_room_min_stable_count').value)
        self.locked_room_min_area_m2 = float(self.get_parameter('locked_room_min_area_m2').value)
        self.locked_room_min_clearance_m = float(self.get_parameter('locked_room_min_clearance_m').value)
        self.locked_room_claim_radius_m = float(self.get_parameter('locked_room_claim_radius_m').value)
        self.lock_start_room_as_r1 = bool(self.get_parameter('lock_start_room_as_r1').value)
        self.start_room_id = int(self.get_parameter('start_room_id').value)
        self.start_room_claim_radius_m = float(self.get_parameter('start_room_claim_radius_m').value)
        self.start_room_pose_from_robot = bool(self.get_parameter('start_room_pose_from_robot').value)
        self.publish_locked_room_markers = bool(self.get_parameter('publish_locked_room_markers').value)

        self.region_map_id_only = bool(self.get_parameter('region_map_id_only').value)
        self.publish_gvd_markers = bool(self.get_parameter('publish_gvd_markers').value)
        self.publish_gateway_markers = bool(self.get_parameter('publish_gateway_markers').value)
        self.publish_region_text = bool(self.get_parameter('publish_region_text').value)
        self.publish_unlocked_region_text = bool(self.get_parameter('publish_unlocked_region_text').value)
        self.region_text_id_only = bool(self.get_parameter('region_text_id_only').value)
        self.publish_latent_frontiers = bool(self.get_parameter('publish_latent_frontiers').value)
        self.publish_region_outlines = bool(self.get_parameter('publish_region_outlines').value)
        self.region_marker_alpha = float(self.get_parameter('region_marker_alpha').value)
        self.region_outline_width_m = float(self.get_parameter('region_outline_width_m').value)
        self.max_region_marker_cells = int(self.get_parameter('max_region_marker_cells').value)
        self.marker_z = float(self.get_parameter('marker_z').value)
        self.text_z = float(self.get_parameter('text_z').value)
        self.max_gvd_marker_points = int(self.get_parameter('max_gvd_marker_points').value)

        # Input /map QoS is intentionally permissive. Cartographer normally
        # offers RELIABLE + TRANSIENT_LOCAL, but other SLAM backends may use
        # VOLATILE or BEST_EFFORT. A permissive subscriber avoids false
        # WAIT_MAP/NO_MAP states when the backend changes.
        map_in_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
        )
        # Output map remains latched for RViz late-join behavior.
        map_out_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self.map_sub = self.create_subscription(OccupancyGrid, self.map_topic, self._on_map, map_in_qos)
        self.region_map_pub = self.create_publisher(OccupancyGrid, self.region_map_topic, map_out_qos)
        self.marker_pub = self.create_publisher(MarkerArray, self.marker_topic, 1)
        self.summary_pub = self.create_publisher(String, self.summary_topic, 10)
        self.selected_region_pub = self.create_publisher(PoseStamped, self.selected_region_topic, 1)
        self.locked_rooms_pub = self.create_publisher(String, self.locked_rooms_topic, 10)

        self.tf_buffer = tf2_ros.Buffer(cache_time=Duration(seconds=10.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.map_msg: Optional[OccupancyGrid] = None
        self.grid: Optional[GridInfo] = None
        self.prev_map_data: Optional[List[int]] = None
        self.last_map_change_time = 0.0
        self.last_map_geometry_change = 0.0
        self.last_region_update = 0.0
        self.changed_cells_since_update = 0

        self.regions: List[TrackedRegion] = []
        self.region_memory: Dict[int, TrackedRegion] = {}
        self.locked_rooms: Dict[int, LockedRoom] = {}
        self.frontier_portals: List[FrontierPortal] = []
        self.gateway_cuts: List[GatewayCut] = []
        self.gvd_cells: Set[Cell] = set()
        self._last_occupancy_debug: Dict[str, int] = {}
        self.last_region_map_msg: Optional[OccupancyGrid] = None
        self.last_motion_pose: Optional[Pose2D] = None
        self.last_motion_time: float = 0.0
        self.last_motion_linear_speed: float = 0.0
        self.last_motion_angular_speed: float = 0.0
        self.last_hold_republish_time: float = 0.0
        self.start_pose: Optional[Pose2D] = None
        self.start_pose_time: float = 0.0
        self.next_region_id = 1
        self.next_portal_id = 1
        self.next_cut_id = 1

        self.timer = self.create_timer(self.timer_period, self._on_timer)

        self.get_logger().info(
            'SlamRegionGraphNode started | '
            f'map={self.map_topic} frame={self.global_frame}->{self.robot_frame} '
            f'update={self.region_update_period:.2f}s stable={self.map_stable_time:.2f}s '
            f'gvd={self.enable_gvd} bottleneck_cuts={self.enable_bottleneck_cuts} '
            f'morph_split={self.enable_morphological_room_split} '
            f'erosion={self.room_seed_erosion_radius_m:.2f}m '
            f'vis=region_id_only:{self.region_map_id_only} '
            f'hold_motion={self.hold_region_graph_while_moving}'
        )

    # ------------------------------------------------------------------
    # ROS callbacks
    # ------------------------------------------------------------------

    def _on_map(self, msg: OccupancyGrid):
        new_grid = GridInfo(
            width=msg.info.width,
            height=msg.info.height,
            resolution=msg.info.resolution,
            origin_x=msg.info.origin.position.x,
            origin_y=msg.info.origin.position.y,
        )
        now = time.time()
        geometry_changed = self.grid is None or self._grid_changed(self.grid, new_grid)
        if geometry_changed:
            self.last_map_geometry_change = now
            self.changed_cells_since_update = 10**9
        elif self.prev_map_data is not None:
            changed = 0
            # Count a bounded number of changes cheaply enough for typical TB3 maps.
            for a, b in zip(self.prev_map_data, msg.data):
                if a != b:
                    changed += 1
            if changed > 0:
                self.last_map_change_time = now
                self.changed_cells_since_update += changed
        else:
            self.changed_cells_since_update = 10**9

        self.grid = new_grid
        self.map_msg = msg
        self.prev_map_data = list(msg.data)

    def _estimate_robot_motion(self, robot: Optional[Pose2D]) -> Tuple[float, float]:
        now = time.time()
        if robot is None:
            return 0.0, 0.0
        if self.last_motion_pose is None or self.last_motion_time <= 0.0:
            self.last_motion_pose = robot
            self.last_motion_time = now
            self.last_motion_linear_speed = 0.0
            self.last_motion_angular_speed = 0.0
            return 0.0, 0.0
        dt = max(1e-3, now - self.last_motion_time)
        dx = robot.x - self.last_motion_pose.x
        dy = robot.y - self.last_motion_pose.y
        dyaw = math.atan2(math.sin(robot.yaw - self.last_motion_pose.yaw), math.cos(robot.yaw - self.last_motion_pose.yaw))
        lin = math.hypot(dx, dy) / dt
        ang = abs(dyaw) / dt
        # Light low-pass to avoid one TF jitter sample unlocking/locking the graph.
        self.last_motion_linear_speed = 0.70 * self.last_motion_linear_speed + 0.30 * lin
        self.last_motion_angular_speed = 0.70 * self.last_motion_angular_speed + 0.30 * ang
        self.last_motion_pose = robot
        self.last_motion_time = now
        return self.last_motion_linear_speed, self.last_motion_angular_speed

    def _republish_held_region_graph(self, reason: str, linear_speed: float, angular_speed: float):
        now = time.time()
        if self.last_region_map_msg is not None and now - self.last_hold_republish_time >= self.region_hold_republish_sec:
            self.last_region_map_msg.header.stamp = self.get_clock().now().to_msg()
            self.region_map_pub.publish(self.last_region_map_msg)
            # Markers are kept from the last committed graph; do not regenerate
            # them from a transient moving map.
            self.last_hold_republish_time = now
        self._publish_summary({
            'state': 'HOLD_REGION_GRAPH_DURING_MOTION',
            'reason': reason,
            'regions': len(self.regions),
            'linear_speed_est_mps': round(linear_speed, 3),
            'angular_speed_est_rps': round(angular_speed, 3),
            'hold_linear_threshold_mps': round(self.region_hold_linear_speed_mps, 3),
            'hold_angular_threshold_rps': round(self.region_hold_angular_speed_rps, 3),
        })

    def _on_timer(self):
        if self.map_msg is None or self.grid is None:
            self._publish_summary({'state': 'WAIT_MAP'})
            return

        now = time.time()
        if now - self.last_region_update < self.region_update_period:
            return
        if now - self.last_map_geometry_change < self.map_stable_time:
            self._publish_summary({'state': 'MAP_GEOMETRY_STABILIZING'})
            return
        if (not self.force_update_without_map_delta and
                self.changed_cells_since_update < self.min_changed_cells_for_update):
            return

        robot = self._lookup_robot_pose()
        if robot is None and self.use_reachable_only and not self.fallback_to_all_free_without_tf:
            self._publish_summary({'state': 'WAIT_TF', 'regions': len(self.regions)})
            return

        linear_speed, angular_speed = self._estimate_robot_motion(robot)
        moving = (
            robot is not None
            and self.hold_region_graph_while_moving
            and self.regions
            and (
                linear_speed > self.region_hold_linear_speed_mps
                or angular_speed > self.region_hold_angular_speed_rps
            )
        )
        if moving:
            # Do not recompute watershed/door cuts from a transient moving SLAM
            # map.  Hold the last committed region partition until the robot is
            # slow enough.  This directly prevents the RViz region layer from
            # fragmenting/flickering while the robot drives.
            self.last_region_update = now
            self.changed_cells_since_update = 0
            self._republish_held_region_graph('robot_is_moving', linear_speed, angular_speed)
            return

        self.last_region_update = now
        self.changed_cells_since_update = 0
        try:
            self._update_region_graph(robot)
        except Exception as exc:
            self.get_logger().warn(f'REGION_UPDATE_FAILED | {exc}')
            self._publish_summary({'state': 'REGION_UPDATE_FAILED', 'error': str(exc)})

    # ------------------------------------------------------------------
    # Main graph extraction
    # ------------------------------------------------------------------

    def _update_region_graph(self, robot: Optional[Pose2D]):
        assert self.map_msg is not None and self.grid is not None
        start = time.time()

        free_cells, occupied_cells, unknown_cells = self._classify_cells()
        if not free_cells:
            # Still publish an empty diagnostic region map so RViz does not stay
            # in a misleading "No map received" state.  The summary carries
            # the occupancy histogram needed to tune thresholds.
            self._publish_region_map([])
            summary = {'state': 'NO_FREE_CELLS'}
            summary.update(self._map_value_stats())
            summary.update({
                'free_threshold': self.free_threshold,
                'occupied_threshold': self.occupied_threshold,
            })
            self._publish_summary(summary)
            return

        robot_cell = None
        if robot is not None:
            robot_cell = self._world_to_cell(robot.x, robot.y)
            self._capture_start_pose(robot, robot_cell, free_cells)

        if self.use_reachable_only:
            seed_cell = self._reachable_seed_cell(robot_cell, free_cells)
            if seed_cell is not None:
                reachable = self._flood_fill(seed_cell, free_cells, use8=True)
                source = 'reachable' if seed_cell == robot_cell else 'reachable_nearest_free'
            elif robot is None and self.fallback_to_all_free_without_tf:
                reachable = set(free_cells)
                source = 'all_free_fallback_no_tf'
            elif robot is not None and self.fallback_to_all_free_when_robot_off_map:
                reachable = set(free_cells)
                source = 'all_free_fallback_robot_off_map'
            else:
                self._publish_summary({
                    'state': 'WAIT_ROBOT_ON_FREE',
                    'regions': len(self.regions),
                    'robot_cell': None if robot_cell is None else {'x': robot_cell[0], 'y': robot_cell[1]},
                    'robot_pose': None if robot is None else {
                        'x': round(robot.x, 3),
                        'y': round(robot.y, 3),
                        'yaw': round(robot.yaw, 3),
                    },
                    'free_cells': len(free_cells),
                    'reason': 'robot_pose_not_inside_free_space',
                })
                return
        else:
            reachable = set(free_cells)
            source = 'all_free_configured'

        if not reachable:
            self._publish_summary({'state': 'NO_REACHABLE_FREE'})
            return

        clearance, nearest_label = self._distance_transform_with_labels(occupied_cells)

        self.gvd_cells = set()
        if self.enable_gvd and occupied_cells:
            self.gvd_cells = self._extract_approx_gvd(reachable, clearance, nearest_label)

        frontier_clusters = self._frontier_clusters(reachable, unknown_cells)

        self.gateway_cuts = []
        cut_cells: Set[Cell] = set()
        split_method = 'connected_components'
        seed_components = 0
        separator_cells = 0

        if self.enable_bottleneck_cuts and self.gvd_cells:
            self.gateway_cuts = self._detect_gateway_cuts(reachable, clearance)
            for cut in self.gateway_cuts:
                cut_cells |= cut.cells

        if self.enable_low_clearance_doorway_cuts:
            doorway_cuts = self._detect_low_clearance_doorway_cuts(reachable, clearance, cut_cells)
            self.gateway_cuts.extend(doorway_cuts)
            for cut in doorway_cuts:
                cut_cells |= cut.cells

        raw_regions: List[RawRegion]
        room_sets: List[Set[Cell]] = []
        room_separator_cells: Set[Cell] = set()
        if self.enable_morphological_room_split:
            room_sets, room_separator_cells, seed_components = self._morphological_room_split(reachable, clearance, cut_cells)

        if len(room_sets) >= self.min_morphological_regions:
            split_method = 'morphological_seeded_watershed'
            cut_cells |= room_separator_cells
            separator_cells = len(room_separator_cells)
            self._append_separator_cuts(room_separator_cells, clearance)
            raw_regions = self._cell_sets_to_regions(room_sets, free_cells, occupied_cells, unknown_cells, cut_cells, clearance)
        else:
            region_cells = set(reachable) - cut_cells
            raw_regions = self._components_to_regions(region_cells, free_cells, occupied_cells, unknown_cells, cut_cells, clearance)

        if self.force_assign_all_free_to_regions and raw_regions:
            assignable_cells = reachable if self.force_assign_source != 'all_free' else free_cells
            raw_regions = self._force_assign_free_cells_to_raw_regions(
                raw_regions,
                assignable_cells,
                free_cells,
                occupied_cells,
                unknown_cells,
                cut_cells,
                clearance,
            )

        raw_regions.sort(key=lambda r: r.area_m2, reverse=True)
        raw_regions = raw_regions[:self.max_regions]
        raw_regions = self._prioritize_start_region(raw_regions)

        tracked = self._track_regions(raw_regions)
        self._force_start_region_id(tracked)
        self.regions = tracked
        self.frontier_portals = self._attach_frontiers_to_regions(frontier_clusters, tracked, unknown_cells)
        self._ensure_start_room_locked(tracked, clearance)
        self._update_locked_rooms(tracked, clearance)

        self._publish_region_map(tracked)
        self._publish_markers(tracked, self.frontier_portals, self.gateway_cuts, self.gvd_cells)
        self._publish_best_region_goal(robot, tracked)
        self._publish_locked_rooms(tracked)

        elapsed_ms = 1000.0 * (time.time() - start)
        state_counts: Dict[str, int] = {}
        for r in tracked:
            state_counts[r.state] = state_counts.get(r.state, 0) + 1

        summary = {
            'state': 'OK',
            'source': source,
            'regions': len(tracked),
            'portals': len(self.frontier_portals),
            'gateway_cuts': len(self.gateway_cuts),
            'gvd_cells': len(self.gvd_cells),
            'split_method': split_method,
            'seed_components': seed_components,
            'separator_cells': separator_cells,
            'room_seed_erosion_radius_m': round(self.room_seed_erosion_radius_m, 3),
            'doorway_cut_clearance_max': round(self.doorway_cut_clearance_max, 3),
            'occupied_threshold_used': self._last_occupancy_debug.get('occupied_threshold_used', -1),
            'highpass_wall_cells': self._last_occupancy_debug.get('highpass_wall_cells', 0),
            'absolute_obstacle_cells': self._last_occupancy_debug.get('absolute_obstacle_cells', 0),
            'dark_obstacle_cells': self._last_occupancy_debug.get('dark_obstacle_cells', 0),
            'inflated_obstacle_cells': self._last_occupancy_debug.get('inflated_obstacle_cells', 0),
            'wall_filter_iterations': self._last_occupancy_debug.get('wall_filter_iterations', 0),
            'filled_unknown_hole_cells': self._last_occupancy_debug.get('filled_unknown_hole_cells', 0),
            'region_dense_fill_iterations': int(self.region_dense_fill_iterations),
            'force_assign_all_free_to_regions': bool(self.force_assign_all_free_to_regions),
            'force_assign_source': self.force_assign_source,
            'use_reachable_only': bool(self.use_reachable_only),
            'fallback_to_all_free_without_tf': bool(self.fallback_to_all_free_without_tf),
            'fallback_to_all_free_when_robot_off_map': bool(self.fallback_to_all_free_when_robot_off_map),
            'robot_free_snap_radius_m': round(self.robot_free_snap_radius_m, 3),
            'persist_region_ids': bool(self.persist_region_ids),
            'remembered_region_ids': sorted(self.region_memory.keys()),
            'lock_stable_rooms': bool(self.lock_stable_rooms),
            'lock_provisional_room_candidates': bool(self.lock_provisional_room_candidates),
            'locked_room_min_stable_count': int(self.locked_room_min_stable_count),
            'locked_room_min_clearance_m': round(self.locked_room_min_clearance_m, 3),
            'locked_room_ids': sorted(self.locked_rooms.keys()),
            'lock_start_room_as_r1': bool(self.lock_start_room_as_r1),
            'start_room_id': int(self.start_room_id),
            'start_pose': (
                None if self.start_pose is None else {
                    'x': round(self.start_pose.x, 3),
                    'y': round(self.start_pose.y, 3),
                    'yaw': round(self.start_pose.yaw, 3),
                }
            ),
            'watershed': bool(self.use_clearance_priority_watershed),
            'reachable': len(reachable),
            'free': len(free_cells),
            'unknown': len(unknown_cells),
            'occupied': len(occupied_cells),
            'states': state_counts,
            'visualization': 'region_id_only' if self.region_map_id_only else 'debug_region_map',
            'published_marker_namespaces': {
                'regions': True,
                'region_outlines': self.publish_region_outlines,
                'region_text': self.publish_region_text,
                'locked_room_anchors': self.publish_locked_room_markers,
                'locked_room_text': self.publish_locked_room_markers,
                'gateway_cuts': self.publish_gateway_markers,
                'approx_gvd': self.publish_gvd_markers,
                'latent_frontiers': self.publish_latent_frontiers,
            },
            'elapsed_ms': round(elapsed_ms, 1),
        }
        self._publish_summary(summary)
        self.get_logger().info(
            f'REGION_GRAPH | regions={len(tracked)} portals={len(self.frontier_portals)} '
            f'cuts={len(self.gateway_cuts)} gvd={len(self.gvd_cells)} source={source} '
            f'split={split_method} seeds={seed_components} sep={separator_cells} '
            f'states={state_counts} elapsed={elapsed_ms:.1f}ms'
        )

    # ------------------------------------------------------------------
    # Map classification / geometry
    # ------------------------------------------------------------------

    def _classify_cells(self) -> Tuple[Set[Cell], Set[Cell], Set[Cell]]:
        assert self.map_msg is not None and self.grid is not None
        vals = list(self.map_msg.data)
        w = self.grid.width

        known: Set[Cell] = set()
        unknown: Set[Cell] = set()
        absolute_obstacles: Set[Cell] = set()
        for idx, val in enumerate(vals):
            x = idx % w
            y = idx // w
            c = (x, y)
            if val < 0:
                unknown.add(c)
                continue
            known.add(c)
            if val >= self.occupied_threshold:
                absolute_obstacles.add(c)

        highpass_obstacles: Set[Cell] = set()
        if self.enable_iterative_highpass_wall_filter:
            highpass_obstacles = self._iterative_highpass_wall_cells(vals)

        dark_obstacles: Set[Cell] = set()
        if self.enable_local_dark_obstacle_filter:
            # Kept for compatibility, but v11 no longer relies on this single
            # threshold filter.  It is only an additional weak wall cue.
            dark_obstacles = self._local_dark_obstacle_cells(vals, self.occupied_threshold)

        occupied: Set[Cell] = (absolute_obstacles | highpass_obstacles | dark_obstacles) & known
        if self.obstacle_min_cluster_cells > 1 and occupied:
            occupied = self._prune_small_components(occupied, min_cells=self.obstacle_min_cluster_cells, use8=True)

        inflated_obstacles: Set[Cell] = set()
        if self.wall_inflation_radius_m > 1e-9 and occupied:
            radius_cells = max(1, int(math.ceil(self.wall_inflation_radius_m / max(1e-9, self.grid.resolution))))
            inflated_obstacles = self._inflate_cells(occupied, radius_cells) & known
            occupied |= inflated_obstacles

        if self.known_non_obstacle_is_free:
            # Cartographer known cells are often not clean 0/free and 100/wall.
            # Once wall ridges have been extracted, every known non-obstacle
            # cell is treated as traversable evidence.  This fixes the failure
            # mode where partially observed open space remains dark/ambiguous
            # and never enters the region graph.
            free: Set[Cell] = set(known) - set(occupied)
        else:
            free = {c for c in known if c not in occupied and vals[self._idx(c)] <= self.free_threshold}
            ambiguous = (set(known) - set(occupied)) - free
            unknown |= ambiguous

        if self.free_mask_denoise_iterations > 0:
            free = self._denoise_free_mask(free, occupied, known)

        filled_unknown_holes: Set[Cell] = set()
        if self.fill_unknown_holes_as_free and unknown:
            filled_unknown_holes = self._fill_unknown_holes_as_free(free, occupied, unknown)
            if filled_unknown_holes:
                free |= filled_unknown_holes
                unknown -= filled_unknown_holes

        unknown -= occupied
        unknown -= free

        self._last_occupancy_debug = {
            'free_threshold_config': int(self.free_threshold),
            'occupied_threshold_config': int(self.occupied_threshold),
            'occupied_threshold_used': int(self.occupied_threshold),
            'otsu_threshold': -1,
            'known_cells': int(len(known)),
            'absolute_obstacle_cells': int(len(absolute_obstacles)),
            'highpass_wall_cells': int(len(highpass_obstacles)),
            'dark_obstacle_cells': int(len(dark_obstacles)),
            'inflated_obstacle_cells': int(len(inflated_obstacles)),
            'wall_filter_iterations': int(self.highpass_wall_iterations),
            'wall_inflation_radius_milli': int(round(self.wall_inflation_radius_m * 1000.0)),
            'known_non_obstacle_is_free': bool(self.known_non_obstacle_is_free),
            'filled_unknown_hole_cells': int(len(filled_unknown_holes)),
        }
        return free, occupied, unknown

    def _iterative_highpass_wall_cells(self, vals: List[int]) -> Set[Cell]:
        assert self.grid is not None
        votes: Dict[Cell, int] = {}
        strong: Set[Cell] = set()
        iterations = max(1, int(self.highpass_wall_iterations))
        base_r = max(1e-9, self.highpass_smooth_radius_m)
        grow_r = max(0.0, self.highpass_radius_growth_m)

        for it in range(iterations):
            radius_m = base_r + it * grow_r
            radius_cells = max(1, int(math.ceil(radius_m / max(1e-9, self.grid.resolution))))
            smooth = self._box_mean_known(vals, radius_cells)
            for idx, val in enumerate(vals):
                if val < self.highpass_wall_min_value:
                    continue
                mean = smooth[idx]
                if mean is None:
                    continue
                residual = float(val) - float(mean)
                if residual >= self.highpass_wall_contrast:
                    c = (idx % self.grid.width, idx // self.grid.width)
                    votes[c] = votes.get(c, 0) + 1
                    if residual >= 1.6 * self.highpass_wall_contrast or val >= self.occupied_threshold:
                        strong.add(c)

        selected = {c for c, n in votes.items() if n >= self.highpass_wall_votes_min} | strong

        # Hysteresis expansion: weak but wall-like cells next to strong ridge
        # cells are also accepted.  This keeps continuous fuzzy Cartographer
        # wall bands from being broken into dotted fragments.
        if selected and self.highpass_hysteresis_min_value >= 0:
            extra: Set[Cell] = set()
            for c in selected:
                for n in self._neighbors8(c):
                    v = vals[self._idx(n)]
                    if v >= self.highpass_hysteresis_min_value and n not in selected:
                        extra.add(n)
            selected |= extra
        return selected

    def _box_mean_known(self, vals: List[int], radius_cells: int) -> List[Optional[float]]:
        assert self.grid is not None
        w = self.grid.width
        h = self.grid.height
        out: List[Optional[float]] = [None] * (w * h)
        r = max(1, int(radius_cells))
        for y in range(h):
            y0 = max(0, y - r)
            y1 = min(h - 1, y + r)
            for x in range(w):
                x0 = max(0, x - r)
                x1 = min(w - 1, x + r)
                s = 0
                n = 0
                for yy in range(y0, y1 + 1):
                    base = yy * w
                    for xx in range(x0, x1 + 1):
                        v = vals[base + xx]
                        if v >= 0:
                            s += int(v)
                            n += 1
                if n > 0:
                    out[y * w + x] = s / float(n)
        return out

    def _denoise_free_mask(self, free: Set[Cell], occupied: Set[Cell], known: Set[Cell]) -> Set[Cell]:
        # Iterative majority relaxation over known non-obstacle cells.  This is
        # the convergence-style filter the map needs: isolated speckles are
        # suppressed while locally supported observed space is recovered.
        cur = set(free)
        candidates = set(known) - set(occupied)
        for _ in range(max(0, self.free_mask_denoise_iterations)):
            nxt = set(cur)
            for c in candidates:
                fcnt = 0
                occnt = 0
                known_n = 0
                for n in self._neighbors8(c):
                    if n in known:
                        known_n += 1
                    if n in cur:
                        fcnt += 1
                    elif n in occupied:
                        occnt += 1
                if c not in cur:
                    if fcnt >= self.free_fill_neighbor_min:
                        nxt.add(c)
                else:
                    # Remove only extremely unsupported free speckles.  Do not
                    # erode valid thin corridors aggressively.
                    if fcnt < self.free_keep_neighbor_min and occnt >= 5 and known_n >= 5:
                        nxt.discard(c)
            if nxt == cur:
                break
            cur = nxt
        return cur

    def _fill_unknown_holes_as_free(
        self,
        free: Set[Cell],
        occupied: Set[Cell],
        unknown: Set[Cell],
    ) -> Set[Cell]:
        """Convert small enclosed unknown islands into free evidence.

        Cartographer does not produce a clean binary map while SLAM is still
        growing.  Indoor scans often leave small dark/unknown holes inside
        otherwise observed open space.  If those holes remain unknown, region
        labels look sparse and fragmented.  This method fills only small
        unknown components whose boundary is mostly free and only weakly
        occupied, so it does not intentionally cross walls or unexplored large
        areas.
        """
        if self.grid is None or not unknown:
            return set()

        max_cells = max(
            1,
            int(self.unknown_hole_fill_max_area_m2 / max(1e-9, self.grid.resolution ** 2)),
        )
        filled: Set[Cell] = set()
        visited: Set[Cell] = set()

        for start in list(unknown):
            if start in visited:
                continue

            q = deque([start])
            visited.add(start)
            comp: Set[Cell] = {start}
            touches_map_border = False
            free_boundary = 0
            occ_boundary = 0
            other_boundary = 0

            while q:
                c = q.popleft()
                x, y = c
                if x == 0 or y == 0 or x == self.grid.width - 1 or y == self.grid.height - 1:
                    touches_map_border = True

                for n in self._neighbors4(c):
                    if n in unknown:
                        if n not in visited:
                            visited.add(n)
                            comp.add(n)
                            if len(comp) <= max_cells + 1:
                                q.append(n)
                        continue
                    if n in free:
                        free_boundary += 1
                    elif n in occupied:
                        occ_boundary += 1
                    else:
                        other_boundary += 1

                # Very large unknown components are almost certainly unexplored
                # space, not a small interior hole.  Keep marking connected
                # cells visited only within the current queue frontier; then
                # abandon filling this component.
                if len(comp) > max_cells:
                    # Drain connected unknown component cheaply so the outer
                    # loop does not reprocess every cell.
                    while q:
                        c2 = q.popleft()
                        for n2 in self._neighbors4(c2):
                            if n2 in unknown and n2 not in visited:
                                visited.add(n2)
                                q.append(n2)
                    comp.clear()
                    break

            if not comp:
                continue
            if touches_map_border:
                continue

            boundary = free_boundary + occ_boundary + other_boundary
            if boundary <= 0:
                continue

            free_ratio = free_boundary / float(boundary)
            occ_ratio = occ_boundary / float(boundary)

            if (
                free_ratio >= self.unknown_hole_fill_min_free_boundary_ratio
                and occ_ratio <= self.unknown_hole_fill_max_occ_boundary_ratio
            ):
                filled |= comp

        return filled

    def _otsu_threshold(self, vals: List[int]) -> Optional[int]:
        hist = [0] * 101
        total = 0
        for v in vals:
            if 0 <= v <= 100:
                hist[int(v)] += 1
                total += 1
        if total < 32:
            return None

        sum_total = sum(i * hist[i] for i in range(101))
        sum_b = 0.0
        w_b = 0
        best_t = None
        best_var = -1.0
        for t in range(0, 100):
            w_b += hist[t]
            if w_b == 0:
                continue
            w_f = total - w_b
            if w_f == 0:
                break
            sum_b += t * hist[t]
            m_b = sum_b / w_b
            m_f = (sum_total - sum_b) / w_f
            between = float(w_b) * float(w_f) * (m_b - m_f) ** 2
            if between > best_var:
                best_var = between
                best_t = t
        return best_t

    def _local_dark_obstacle_cells(self, vals: List[int], occupied_threshold_used: int) -> Set[Cell]:
        assert self.grid is not None
        radius = max(1, int(math.ceil(self.dark_obstacle_neighbor_radius_m / max(1e-9, self.grid.resolution))))
        out: Set[Cell] = set()
        w = self.grid.width
        h = self.grid.height
        for idx, val in enumerate(vals):
            if val < self.dark_obstacle_min_value or val >= occupied_threshold_used:
                continue
            x = idx % w
            y = idx // w
            local: List[int] = []
            for yy in range(max(0, y - radius), min(h, y + radius + 1)):
                base = yy * w
                for xx in range(max(0, x - radius), min(w, x + radius + 1)):
                    if xx == x and yy == y:
                        continue
                    nv = vals[base + xx]
                    if nv >= 0:
                        local.append(int(nv))
            if len(local) < 4:
                continue
            local.sort()
            med = local[len(local) // 2]
            if int(val) - med >= self.dark_obstacle_local_contrast:
                out.add((x, y))
        return out

    def _inflate_cells(self, cells: Set[Cell], radius_cells: int) -> Set[Cell]:
        if radius_cells <= 0:
            return set(cells)
        inflated = set(cells)
        for x, y in cells:
            for dy in range(-radius_cells, radius_cells + 1):
                for dx in range(-radius_cells, radius_cells + 1):
                    if dx * dx + dy * dy > radius_cells * radius_cells:
                        continue
                    nx, ny = x + dx, y + dy
                    if self._in_bounds(nx, ny):
                        inflated.add((nx, ny))
        return inflated

    def _map_value_stats(self) -> Dict[str, int]:
        assert self.map_msg is not None
        vals = list(self.map_msg.data)
        if not vals:
            return {
                'map_total': 0,
                'map_min': 0,
                'map_max': 0,
                'map_unknown_lt0': 0,
                'map_zero': 0,
                'map_free_le_threshold': 0,
                'map_uncertain_between_thresholds': 0,
                'map_occupied_ge_threshold': 0,
            }
        unknown = sum(1 for v in vals if v < 0)
        occupied = sum(1 for v in vals if v >= self.occupied_threshold)
        free = sum(1 for v in vals if 0 <= v <= self.free_threshold)
        uncertain = sum(1 for v in vals if self.free_threshold < v < self.occupied_threshold)
        zero = sum(1 for v in vals if v == 0)
        stats = {
            'map_total': len(vals),
            'map_min': int(min(vals)),
            'map_max': int(max(vals)),
            'map_unknown_lt0': unknown,
            'map_zero': zero,
            'map_free_le_threshold': free,
            'map_uncertain_between_thresholds': uncertain,
            'map_occupied_ge_threshold': occupied,
        }
        stats.update(self._last_occupancy_debug)
        return stats

    def _grid_changed(self, a: GridInfo, b: GridInfo) -> bool:
        return (
            a.width != b.width or a.height != b.height or
            abs(a.resolution - b.resolution) > 1e-9 or
            abs(a.origin_x - b.origin_x) > 1e-6 or
            abs(a.origin_y - b.origin_y) > 1e-6
        )

    def _idx(self, c: Cell) -> int:
        assert self.grid is not None
        return c[1] * self.grid.width + c[0]

    def _in_bounds(self, x: int, y: int) -> bool:
        assert self.grid is not None
        return 0 <= x < self.grid.width and 0 <= y < self.grid.height

    def _cell_to_world(self, x: int, y: int) -> Tuple[float, float]:
        assert self.grid is not None
        return (
            self.grid.origin_x + (x + 0.5) * self.grid.resolution,
            self.grid.origin_y + (y + 0.5) * self.grid.resolution,
        )

    def _world_to_cell(self, x: float, y: float) -> Optional[Cell]:
        assert self.grid is not None
        cx = int(math.floor((x - self.grid.origin_x) / self.grid.resolution))
        cy = int(math.floor((y - self.grid.origin_y) / self.grid.resolution))
        if self._in_bounds(cx, cy):
            return (cx, cy)
        return None

    def _neighbors4(self, c: Cell) -> Iterable[Cell]:
        x, y = c
        for nx, ny in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
            if self._in_bounds(nx, ny):
                yield (nx, ny)

    def _neighbors8(self, c: Cell) -> Iterable[Cell]:
        x, y = c
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                nx, ny = x + dx, y + dy
                if self._in_bounds(nx, ny):
                    yield (nx, ny)

    # ------------------------------------------------------------------
    # TF
    # ------------------------------------------------------------------

    def _lookup_robot_pose(self) -> Optional[Pose2D]:
        try:
            tf = self.tf_buffer.lookup_transform(
                self.global_frame,
                self.robot_frame,
                rclpy.time.Time(),
                timeout=Duration(seconds=0.03),
            )
        except Exception:
            return None
        q = tf.transform.rotation
        yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                         1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        return Pose2D(tf.transform.translation.x, tf.transform.translation.y, yaw)

    # ------------------------------------------------------------------
    # Distance transform / approximate GVD
    # ------------------------------------------------------------------

    def _distance_transform_with_labels(self, occupied: Set[Cell]) -> Tuple[List[float], List[int]]:
        assert self.grid is not None
        n = self.grid.width * self.grid.height
        inf = float('inf')
        dist = [inf] * n
        label = [-1] * n
        heap: List[Tuple[float, int, int, int]] = []
        for source_id, c in enumerate(occupied):
            idx = self._idx(c)
            dist[idx] = 0.0
            label[idx] = source_id
            heapq.heappush(heap, (0.0, c[0], c[1], source_id))
        sqrt2 = math.sqrt(2.0)
        res = self.grid.resolution
        while heap:
            d, x, y, source_id = heapq.heappop(heap)
            idx = self._idx((x, y))
            if d > dist[idx] + 1e-9:
                continue
            for ny in (y - 1, y, y + 1):
                for nx in (x - 1, x, x + 1):
                    if nx == x and ny == y:
                        continue
                    if not self._in_bounds(nx, ny):
                        continue
                    step = res * (sqrt2 if nx != x and ny != y else 1.0)
                    nd = d + step
                    nidx = self._idx((nx, ny))
                    if nd + 1e-9 < dist[nidx]:
                        dist[nidx] = nd
                        label[nidx] = source_id
                        heapq.heappush(heap, (nd, nx, ny, source_id))
        return dist, label

    def _extract_approx_gvd(self, reachable: Set[Cell], clearance: List[float], nearest_label: List[int]) -> Set[Cell]:
        gvd: Set[Cell] = set()
        for c in reachable:
            idx = self._idx(c)
            if clearance[idx] < self.gvd_min_clearance or nearest_label[idx] < 0:
                continue
            labels = set()
            for n in self._neighbors8(c):
                lab = nearest_label[self._idx(n)]
                if lab >= 0:
                    labels.add(lab)
            if len(labels) >= 2:
                gvd.add(c)
        return self._prune_small_components(gvd, min_cells=4, use8=True)

    def _detect_gateway_cuts(self, reachable: Set[Cell], clearance: List[float]) -> List[GatewayCut]:
        candidates = {
            c for c in self.gvd_cells
            if self.door_clearance_min <= clearance[self._idx(c)] <= self.door_clearance_max
        }
        clusters = self._connected_components(candidates, use8=True)
        reps: List[Tuple[float, Cell, Set[Cell]]] = []
        for comp in clusters:
            if len(comp) < 2:
                continue
            rep = min(comp, key=lambda c: clearance[self._idx(c)])
            reps.append((clearance[self._idx(rep)], rep, comp))
        reps.sort(key=lambda x: x[0])
        reps = reps[:self.cut_test_max_candidates]

        accepted: List[GatewayCut] = []
        occupied_cut_cells: Set[Cell] = set()
        for clear, rep, comp in reps:
            if len(accepted) >= self.max_gateway_cuts:
                break
            line_cells = self._make_cut_line(rep, comp)
            if not line_cells:
                continue
            # Avoid near-duplicate cuts.
            too_close = False
            rx, ry = self._cell_to_world(*rep)
            for cut in accepted:
                if math.hypot(cut.centroid[0] - rx, cut.centroid[1] - ry) < 0.45:
                    too_close = True
                    break
            if too_close:
                continue
            trial_cut = set(line_cells) | occupied_cut_cells
            if self._cut_splits_reachable(reachable, trial_cut):
                occupied_cut_cells |= set(line_cells)
                accepted.append(GatewayCut(
                    id=self.next_cut_id,
                    cells=set(line_cells),
                    centroid=(rx, ry),
                    clearance=clear,
                    confidence=max(0.0, min(1.0, (self.door_clearance_max - clear) / max(1e-6, self.door_clearance_max))),
                ))
                self.next_cut_id += 1
        return accepted

    def _make_cut_line(self, rep: Cell, cluster: Set[Cell]) -> Set[Cell]:
        assert self.grid is not None
        pts = [self._cell_to_world(x, y) for x, y in cluster]
        if len(pts) >= 2:
            cx = sum(p[0] for p in pts) / len(pts)
            cy = sum(p[1] for p in pts) / len(pts)
            sxx = sum((p[0] - cx) ** 2 for p in pts) / len(pts)
            syy = sum((p[1] - cy) ** 2 for p in pts) / len(pts)
            sxy = sum((p[0] - cx) * (p[1] - cy) for p in pts) / len(pts)
            # Principal eigenvector of 2x2 covariance.
            if abs(sxy) > 1e-9 or abs(sxx - syy) > 1e-9:
                theta = 0.5 * math.atan2(2.0 * sxy, sxx - syy)
                tangent = (math.cos(theta), math.sin(theta))
            else:
                tangent = (1.0, 0.0)
        else:
            tangent = (1.0, 0.0)
        normal = (-tangent[1], tangent[0])
        wx, wy = self._cell_to_world(*rep)
        half_len = self.cut_line_half_length
        width = max(self.cut_line_width, self.grid.resolution * 1.25)
        minx = wx - abs(normal[0]) * half_len - width
        maxx = wx + abs(normal[0]) * half_len + width
        miny = wy - abs(normal[1]) * half_len - width
        maxy = wy + abs(normal[1]) * half_len + width
        cmin = self._world_to_cell(minx, miny)
        cmax = self._world_to_cell(maxx, maxy)
        if cmin is None or cmax is None:
            # Clamp via broad local box if line is near map boundary.
            rx, ry = rep
            radius = int(math.ceil((half_len + width) / self.grid.resolution))
            xs = range(max(0, rx - radius), min(self.grid.width, rx + radius + 1))
            ys = range(max(0, ry - radius), min(self.grid.height, ry + radius + 1))
        else:
            xs = range(max(0, min(cmin[0], cmax[0]) - 2), min(self.grid.width, max(cmin[0], cmax[0]) + 3))
            ys = range(max(0, min(cmin[1], cmax[1]) - 2), min(self.grid.height, max(cmin[1], cmax[1]) + 3))
        cells: Set[Cell] = set()
        nx, ny = normal
        for y in ys:
            for x in xs:
                px, py = self._cell_to_world(x, y)
                vx, vy = px - wx, py - wy
                along = vx * nx + vy * ny
                perp = abs(vx * (-ny) + vy * nx)
                if abs(along) <= half_len and perp <= width:
                    cells.add((x, y))
        return cells

    def _cut_splits_reachable(self, reachable: Set[Cell], cut_cells: Set[Cell]) -> bool:
        remaining = set(reachable) - cut_cells
        comps = self._connected_components(remaining, use8=False, early_stop_components=3)
        large = 0
        min_cells = max(1, int(self.cut_test_min_component_area_m2 / (self.grid.resolution ** 2))) if self.grid else 1
        for comp in comps:
            if len(comp) >= min_cells:
                large += 1
                if large >= 2:
                    return True
        return False

    def _detect_low_clearance_doorway_cuts(
        self,
        reachable: Set[Cell],
        clearance: List[float],
        existing_cut_cells: Set[Cell],
    ) -> List[GatewayCut]:
        if not reachable or self.grid is None:
            return []

        # Prefer medial-axis cells.  Cutting arbitrary wall-adjacent low-clearance
        # cells produces huge boundary bands; cutting low-clearance GVD cells
        # targets doorways and narrow entrances.
        if self.gvd_cells:
            search_domain = set(self.gvd_cells) & reachable
        else:
            search_domain = set(reachable)

        candidates = {
            c for c in search_domain
            if c not in existing_cut_cells and
            self.doorway_cut_clearance_min <= clearance[self._idx(c)] <= self.doorway_cut_clearance_max
        }
        if not candidates:
            return []

        local_minima = {c for c in candidates if self._is_clearance_local_min(c, clearance)}
        if len(local_minima) >= self.doorway_cut_min_cluster_cells:
            candidates = local_minima

        clusters = [
            comp for comp in self._connected_components(candidates, use8=True)
            if len(comp) >= self.doorway_cut_min_cluster_cells
        ]
        reps: List[Tuple[float, Cell, Set[Cell]]] = []
        for comp in clusters:
            rep = min(comp, key=lambda c: clearance[self._idx(c)])
            reps.append((clearance[self._idx(rep)], rep, comp))
        reps.sort(key=lambda x: x[0])
        reps = reps[:self.doorway_cut_max_candidates]

        accepted: List[GatewayCut] = []
        occupied_cut_cells = set(existing_cut_cells)
        old_half = self.cut_line_half_length
        old_width = self.cut_line_width
        self.cut_line_half_length = max(self.cut_line_half_length, self.doorway_cut_force_half_length_m)
        self.cut_line_width = max(self.cut_line_width, self.doorway_cut_force_width_m)
        try:
            for clear, rep, comp in reps:
                rx, ry = self._cell_to_world(*rep)
                duplicate = False
                for cut in accepted:
                    if math.hypot(cut.centroid[0] - rx, cut.centroid[1] - ry) < self.doorway_cut_duplicate_distance_m:
                        duplicate = True
                        break
                if duplicate:
                    continue

                line_cells = self._make_cut_line(rep, comp)
                if not line_cells:
                    continue
                trial_cut = occupied_cut_cells | set(line_cells)
                if self._cut_splits_reachable(reachable, trial_cut):
                    occupied_cut_cells |= set(line_cells)
                    accepted.append(GatewayCut(
                        id=self.next_cut_id,
                        cells=set(line_cells),
                        centroid=(rx, ry),
                        clearance=clear,
                        confidence=0.98,
                    ))
                    self.next_cut_id += 1
        finally:
            self.cut_line_half_length = old_half
            self.cut_line_width = old_width
        return accepted

    def _is_clearance_local_min(self, c: Cell, clearance: List[float]) -> bool:
        cur = clearance[self._idx(c)]
        if not math.isfinite(cur):
            return False
        higher = 0
        valid = 0
        for n in self._neighbors8(c):
            nv = clearance[self._idx(n)]
            if not math.isfinite(nv):
                continue
            valid += 1
            if nv >= cur + self.doorway_cut_local_min_margin:
                higher += 1
        return valid >= 3 and higher >= 2

    # ------------------------------------------------------------------
    # Morphological room split
    # ------------------------------------------------------------------

    def _morphological_room_split(
        self,
        reachable: Set[Cell],
        clearance: List[float],
        pre_cut_cells: Set[Cell],
    ) -> Tuple[List[Set[Cell]], Set[Cell], int]:
        """Partition reachable free-space with clearance-priority watershed.

        v11 change:
        - v7/v8 used plain BFS after erosion; that lets labels flood through
          doors again and creates arbitrary boundaries inside large rooms.
        - This version treats high-clearance open cores as seeds and expands
          labels in descending clearance order.  When two fronts collide at a
          low-clearance neck, the collision cells become separator cells.  This
          is closer to a watershed over the wall-distance field and is much more
          stable for Cartographer's non-binary occupancy maps.
        """
        if not reachable or self.grid is None:
            return [], set(), 0

        hard_barrier = set(pre_cut_cells)
        split_reachable = set(reachable) - hard_barrier
        if not split_reachable:
            return [], hard_barrier, 0

        seed_min_cells = max(
            self.room_seed_min_cells,
            int(self.room_seed_min_area_m2 / max(1e-9, self.grid.resolution ** 2)),
        )
        seed_cells: Set[Cell] = {
            c for c in split_reachable
            if math.isfinite(clearance[self._idx(c)]) and clearance[self._idx(c)] >= self.room_seed_erosion_radius_m
        }
        seed_components = [
            comp for comp in self._connected_components(seed_cells, use8=self.room_split_connectivity_8)
            if len(comp) >= seed_min_cells
        ]
        if len(seed_components) < self.min_morphological_regions:
            return [], hard_barrier, len(seed_components)

        seed_components.sort(key=len, reverse=True)
        seed_components = seed_components[:self.max_regions]

        labels: Dict[Cell, int] = {}
        separator: Set[Cell] = set(hard_barrier)
        neigh = self._neighbors8 if self.room_split_connectivity_8 else self._neighbors4

        if self.use_clearance_priority_watershed:
            heap: List[Tuple[float, int, int, int, int]] = []
            serial = 0
            for label, comp in enumerate(seed_components):
                for c in comp:
                    if c in labels:
                        continue
                    labels[c] = label
                    serial += 1
                    heapq.heappush(heap, (-clearance[self._idx(c)], serial, label, c[0], c[1]))

            while heap:
                _neg_clear, _serial, label, x, y = heapq.heappop(heap)
                c = (x, y)
                if labels.get(c) != label or c in separator:
                    continue
                for n in neigh(c):
                    if n not in split_reachable or n in separator:
                        continue
                    nlab = labels.get(n)
                    if nlab is None:
                        labels[n] = label
                        serial += 1
                        # Higher-clearance cells are expanded first.  This
                        # makes collisions naturally happen at narrow doors.
                        heappush_clear = clearance[self._idx(n)]
                        if not math.isfinite(heappush_clear):
                            heappush_clear = 0.0
                        heapq.heappush(heap, (-heappush_clear, serial, label, n[0], n[1]))
                    elif nlab != label:
                        if clearance[self._idx(n)] <= self.watershed_conflict_clearance_max_m:
                            separator.add(n)
                        if clearance[self._idx(c)] <= self.watershed_conflict_clearance_max_m:
                            separator.add(c)
        else:
            q = deque()
            for label, comp in enumerate(seed_components):
                for c in comp:
                    if c not in labels:
                        labels[c] = label
                        q.append(c)
            while q:
                c = q.popleft()
                label = labels[c]
                for n in neigh(c):
                    if n in split_reachable and n not in labels:
                        labels[n] = label
                        q.append(n)

        # Add geometric label boundaries and thicken them.  This also catches
        # high-clearance conflicts that were not marked by the low-clearance
        # watershed collision rule.
        separator |= self._label_boundary_cells(labels, split_reachable, clearance)
        if self.watershed_separator_width_cells > 0 and separator:
            extra = set(separator)
            frontier = set(separator)
            for _ in range(self.watershed_separator_width_cells):
                nxt: Set[Cell] = set()
                for c in frontier:
                    for n in self._neighbors8(c):
                        if n in split_reachable and clearance[self._idx(n)] <= self.region_separator_max_clearance_m:
                            nxt.add(n)
                nxt -= extra
                extra |= nxt
                frontier = nxt
                if not frontier:
                    break
            separator = extra | hard_barrier

        if self.region_dense_fill_iterations > 0:
            labels = self._densify_region_labels(labels, split_reachable, separator, neigh)

        label_sets: List[Set[Cell]] = [set() for _ in seed_components]
        for c, label in labels.items():
            if c in separator:
                continue
            if 0 <= label < len(label_sets):
                label_sets[label].add(c)

        if self.merge_tiny_split_regions:
            label_sets = self._merge_tiny_label_sets(label_sets, clearance)

        min_area_cells = max(
            self.min_region_cells,
            int(self.min_region_area_m2 / max(1e-9, self.grid.resolution ** 2)),
        )
        label_sets = [s for s in label_sets if len(s) >= min_area_cells]
        return label_sets, separator, len(seed_components)

    def _densify_region_labels(
        self,
        labels: Dict[Cell, int],
        reachable: Set[Cell],
        separator: Set[Cell],
        neigh,
    ) -> Dict[Cell, int]:
        """Fill unlabeled reachable cells by nearest local region label.

        Watershed splitting intentionally leaves separator/conflict cells empty,
        but the remaining non-separator reachable cells should still receive a
        region id for visualization and downstream graph use.  This performs a
        conservative iterative label relaxation: unlabeled cells are assigned to
        a neighboring label only when the local neighborhood has enough support.
        Separators remain hard barriers.
        """
        if not labels:
            return labels

        cur: Dict[Cell, int] = dict(labels)
        allowed = set(reachable) - set(separator)
        iterations = max(0, int(self.region_dense_fill_iterations))
        neighbor_min = max(1, int(self.region_dense_fill_neighbor_min))

        for _ in range(iterations):
            additions: Dict[Cell, int] = {}
            for c in allowed:
                if c in cur:
                    continue
                counts: Dict[int, int] = {}
                for n in neigh(c):
                    if n in separator:
                        continue
                    lab = cur.get(n)
                    if lab is not None:
                        counts[lab] = counts.get(lab, 0) + 1
                if not counts:
                    continue
                best_label, best_count = max(counts.items(), key=lambda kv: kv[1])
                if best_count >= neighbor_min:
                    additions[c] = best_label

            if not additions:
                break
            cur.update(additions)

        return cur

    def _label_boundary_cells(
        self,
        labels: Dict[Cell, int],
        reachable: Set[Cell],
        clearance: List[float],
    ) -> Set[Cell]:
        boundary: Set[Cell] = set()
        neigh = self._neighbors8 if self.room_split_connectivity_8 else self._neighbors4
        for c, lab in labels.items():
            if c not in reachable:
                continue
            touches_other = False
            for n in neigh(c):
                nlab = labels.get(n)
                if nlab is not None and nlab != lab:
                    touches_other = True
                    break
            if not touches_other:
                continue
            if clearance[self._idx(c)] <= self.region_separator_max_clearance_m:
                boundary.add(c)

        # Thicken the separator by a small cell radius for RViz legibility and
        # to prevent diagonally touching labels from visually merging.
        radius = max(0, self.region_separator_width_cells)
        if radius <= 0 or not boundary:
            return boundary
        thick = set(boundary)
        frontier = set(boundary)
        for _ in range(radius):
            nxt: Set[Cell] = set()
            for c in frontier:
                for n in self._neighbors8(c):
                    if n in reachable and clearance[self._idx(n)] <= self.region_separator_max_clearance_m:
                        nxt.add(n)
            nxt -= thick
            thick |= nxt
            frontier = nxt
            if not frontier:
                break
        return thick

    def _merge_tiny_label_sets(self, label_sets: List[Set[Cell]], clearance: List[float]) -> List[Set[Cell]]:
        if self.grid is None or not label_sets:
            return label_sets
        tiny_cells = int(self.tiny_split_region_area_m2 / max(1e-9, self.grid.resolution ** 2))
        if tiny_cells <= 0:
            return label_sets

        large = [set(s) for s in label_sets if len(s) >= tiny_cells]
        tiny = [set(s) for s in label_sets if len(s) < tiny_cells]
        if not tiny or not large:
            return label_sets

        cell_to_large: Dict[Cell, int] = {}
        for i, comp in enumerate(large):
            for c in comp:
                cell_to_large[c] = i

        for comp in tiny:
            votes: Dict[int, int] = {}
            for c in comp:
                for n in self._neighbors8(c):
                    idx = cell_to_large.get(n)
                    if idx is not None:
                        votes[idx] = votes.get(idx, 0) + 1
            if votes:
                target = max(votes.items(), key=lambda kv: kv[1])[0]
            else:
                cx, cy = self._centroid_world(comp)
                target = min(
                    range(len(large)),
                    key=lambda i: math.hypot(self._centroid_world(large[i])[0] - cx, self._centroid_world(large[i])[1] - cy),
                )
            large[target] |= comp
        return large

    def _append_separator_cuts(self, separator_cells: Set[Cell], clearance: List[float]):
        if not separator_cells:
            return
        comps = self._connected_components(separator_cells, use8=True)
        comps.sort(key=len, reverse=True)
        available = max(0, self.max_gateway_cuts - len(self.gateway_cuts))
        for comp in comps[:available]:
            if not comp:
                continue
            cx, cy = self._centroid_world(comp)
            cl_vals = [clearance[self._idx(c)] for c in comp if math.isfinite(clearance[self._idx(c)])]
            clear = min(cl_vals) if cl_vals else 0.0
            self.gateway_cuts.append(GatewayCut(
                id=self.next_cut_id,
                cells=set(comp),
                centroid=(cx, cy),
                clearance=clear,
                confidence=0.95,
            ))
            self.next_cut_id += 1

    # ------------------------------------------------------------------
    # Components / regions
    # ------------------------------------------------------------------

    def _flood_fill(self, start: Cell, allowed: Set[Cell], use8: bool) -> Set[Cell]:
        if start not in allowed:
            return set()
        q = deque([start])
        seen = {start}
        neigh = self._neighbors8 if use8 else self._neighbors4
        while q:
            c = q.popleft()
            for n in neigh(c):
                if n in allowed and n not in seen:
                    seen.add(n)
                    q.append(n)
        return seen

    def _connected_components(self, cells: Set[Cell], use8: bool, early_stop_components: Optional[int] = None) -> List[Set[Cell]]:
        unseen = set(cells)
        comps: List[Set[Cell]] = []
        neigh = self._neighbors8 if use8 else self._neighbors4
        while unseen:
            start = next(iter(unseen))
            q = deque([start])
            unseen.remove(start)
            comp = {start}
            while q:
                c = q.popleft()
                for n in neigh(c):
                    if n in unseen:
                        unseen.remove(n)
                        comp.add(n)
                        q.append(n)
            comps.append(comp)
            if early_stop_components is not None and len(comps) >= early_stop_components:
                # Still valid for cut test; no need to enumerate all components.
                break
        return comps

    def _prune_small_components(self, cells: Set[Cell], min_cells: int, use8: bool) -> Set[Cell]:
        out: Set[Cell] = set()
        for comp in self._connected_components(cells, use8=use8):
            if len(comp) >= min_cells:
                out |= comp
        return out

    def _force_assign_free_cells_to_raw_regions(
        self,
        raw_regions: List[RawRegion],
        assignable_cells: Set[Cell],
        free_cells: Set[Cell],
        occupied_cells: Set[Cell],
        unknown_cells: Set[Cell],
        cut_cells: Set[Cell],
        clearance: List[float],
    ) -> List[RawRegion]:
        """Assign every free cell in assignable_cells to the nearest region.

        The watershed split intentionally leaves separators and some ambiguous
        cells unlabelled.  That is fine for a pure graph view, but the Nav2
        region-coverage planner compares region_map against coverage_map.  Holes
        in region_map make valid free space look like "no region", so the robot
        repeatedly revisits already-labelled space instead of pushing through the
        unlabelled doorway/corridor.  This method performs a multi-source BFS
        over free cells from existing region cells and absorbs every reachable
        free cell into its nearest label.
        """
        if not raw_regions or not assignable_cells:
            return raw_regions

        # Use only actually-free cells.  Do not paint occupied cells as regions.
        allowed = set(assignable_cells) & set(free_cells)
        if not allowed:
            return raw_regions

        max_steps = None
        if self.force_assign_max_bridge_m > 0.0 and math.isfinite(self.force_assign_max_bridge_m):
            max_steps = max(1, int(round(self.force_assign_max_bridge_m / max(self.grid.resolution, 1e-9))))

        label_cells: List[Set[Cell]] = [set(r.cells) & allowed for r in raw_regions]
        owner: Dict[Cell, int] = {}
        q = deque()
        for label, cells in enumerate(label_cells):
            for c in cells:
                if c in owner:
                    continue
                owner[c] = label
                q.append((c, label, 0))

        if not owner:
            return raw_regions

        neigh = self._neighbors8 if self.region_connectivity_8 else self._neighbors4
        while q:
            c, label, dist_steps = q.popleft()
            if max_steps is not None and dist_steps >= max_steps:
                continue
            for n in neigh(c):
                if n not in allowed or n in owner:
                    continue
                owner[n] = label
                label_cells[label].add(n)
                q.append((n, label, dist_steps + 1))

        # If any disconnected free islands were not reached, attach them to the
        # nearest region centroid.  This is mostly for visualization consistency.
        missing = allowed - set(owner.keys())
        if missing:
            centroids = [self._centroid_world(cells) if cells else r.centroid for r, cells in zip(raw_regions, label_cells)]
            for c in missing:
                wx, wy = self._cell_to_world(*c)
                label = min(range(len(raw_regions)), key=lambda i: math.hypot(centroids[i][0] - wx, centroids[i][1] - wy))
                label_cells[label].add(c)

        return self._cell_sets_to_regions(label_cells, free_cells, occupied_cells, unknown_cells, cut_cells, clearance)

    def _cell_sets_to_regions(
        self,
        cell_sets: List[Set[Cell]],
        free_cells: Set[Cell],
        occupied_cells: Set[Cell],
        unknown_cells: Set[Cell],
        cut_cells: Set[Cell],
        clearance: List[float],
    ) -> List[RawRegion]:
        raw: List[RawRegion] = []
        min_area_cells = max(self.min_region_cells, int(self.min_region_area_m2 / (self.grid.resolution ** 2)))
        for comp in cell_sets:
            comp = set(comp)
            if len(comp) < min_area_cells:
                continue
            area = len(comp) * (self.grid.resolution ** 2)
            centroid = self._centroid_world(comp)
            cl_vals = [clearance[self._idx(c)] for c in comp if math.isfinite(clearance[self._idx(c)])]
            mean_clear = sum(cl_vals) / len(cl_vals) if cl_vals else 0.0
            max_clear = max(cl_vals) if cl_vals else 0.0
            elong = self._elongation(comp)
            occ_len, unk_len, gate_len, frontier_count = self._boundary_stats(comp, occupied_cells, unknown_cells, cut_cells)
            total = occ_len + unk_len + gate_len
            closure = occ_len / max(1e-9, total)
            unk_ratio = unk_len / max(1e-9, total)
            raw.append(RawRegion(
                cells=comp,
                centroid=centroid,
                area_m2=area,
                mean_clearance=mean_clear,
                max_clearance=max_clear,
                elongation=elong,
                occupied_boundary_len=occ_len,
                unknown_boundary_len=unk_len,
                gateway_boundary_len=gate_len,
                closure_score=closure,
                unknown_boundary_ratio=unk_ratio,
                frontier_cell_count=frontier_count,
            ))
        return raw

    def _components_to_regions(
        self,
        region_cells: Set[Cell],
        free_cells: Set[Cell],
        occupied_cells: Set[Cell],
        unknown_cells: Set[Cell],
        cut_cells: Set[Cell],
        clearance: List[float],
    ) -> List[RawRegion]:
        comps = self._connected_components(region_cells, use8=self.region_connectivity_8)
        raw: List[RawRegion] = []
        min_area_cells = max(self.min_region_cells, int(self.min_region_area_m2 / (self.grid.resolution ** 2)))
        for comp in comps:
            if len(comp) < min_area_cells:
                continue
            area = len(comp) * (self.grid.resolution ** 2)
            centroid = self._centroid_world(comp)
            cl_vals = [clearance[self._idx(c)] for c in comp if math.isfinite(clearance[self._idx(c)])]
            mean_clear = sum(cl_vals) / len(cl_vals) if cl_vals else 0.0
            max_clear = max(cl_vals) if cl_vals else 0.0
            elong = self._elongation(comp)
            occ_len, unk_len, gate_len, frontier_count = self._boundary_stats(comp, occupied_cells, unknown_cells, cut_cells)
            total = occ_len + unk_len + gate_len
            closure = occ_len / max(1e-9, total)
            unk_ratio = unk_len / max(1e-9, total)
            raw.append(RawRegion(
                cells=comp,
                centroid=centroid,
                area_m2=area,
                mean_clearance=mean_clear,
                max_clearance=max_clear,
                elongation=elong,
                occupied_boundary_len=occ_len,
                unknown_boundary_len=unk_len,
                gateway_boundary_len=gate_len,
                closure_score=closure,
                unknown_boundary_ratio=unk_ratio,
                frontier_cell_count=frontier_count,
            ))
        return raw

    def _boundary_stats(self, comp: Set[Cell], occupied: Set[Cell], unknown: Set[Cell], cuts: Set[Cell]) -> Tuple[float, float, float, int]:
        occ = 0.0
        unk = 0.0
        gate = 0.0
        frontier_count = 0
        res = self.grid.resolution if self.grid else 1.0
        for c in comp:
            touches_unknown = False
            for n in self._neighbors4(c):
                if n in occupied:
                    occ += res
                elif n in unknown:
                    unk += res
                    touches_unknown = True
                elif n in cuts:
                    gate += res
            if touches_unknown:
                frontier_count += 1
        return occ, unk, gate, frontier_count

    def _centroid_world(self, cells: Set[Cell]) -> Tuple[float, float]:
        if not cells:
            return (0.0, 0.0)
        sx = 0.0
        sy = 0.0
        for x, y in cells:
            wx, wy = self._cell_to_world(x, y)
            sx += wx
            sy += wy
        n = float(len(cells))
        return sx / n, sy / n

    def _elongation(self, cells: Set[Cell]) -> float:
        if len(cells) < 3:
            return 1.0
        pts = [self._cell_to_world(x, y) for x, y in cells]
        cx = sum(p[0] for p in pts) / len(pts)
        cy = sum(p[1] for p in pts) / len(pts)
        sxx = sum((p[0] - cx) ** 2 for p in pts) / len(pts)
        syy = sum((p[1] - cy) ** 2 for p in pts) / len(pts)
        sxy = sum((p[0] - cx) * (p[1] - cy) for p in pts) / len(pts)
        trace = sxx + syy
        det_term = math.sqrt(max(0.0, (sxx - syy) ** 2 + 4.0 * sxy * sxy))
        l1 = 0.5 * (trace + det_term)
        l2 = 0.5 * (trace - det_term)
        if l2 < 1e-9:
            return 999.0
        return max(1.0, l1 / l2)

    def _capture_start_pose(self, robot: Pose2D, robot_cell: Optional[Cell], free_cells: Set[Cell]) -> None:
        if not self.lock_start_room_as_r1 or self.start_pose is not None:
            return
        if robot_cell is None or robot_cell not in free_cells:
            return
        self.start_pose = Pose2D(robot.x, robot.y, robot.yaw)
        self.start_pose_time = time.time()
        self.next_region_id = max(self.next_region_id, self.start_room_id)
        self.get_logger().info(
            f'START_ROOM_CAPTURED | name=R{self.start_room_id} '
            f'pose=({robot.x:.2f}, {robot.y:.2f}, {robot.yaw:.2f})'
        )

    def _start_region_index(self, regions: List[RawRegion]) -> Optional[int]:
        if not self.lock_start_room_as_r1 or self.start_pose is None:
            return None
        start_cell = self._world_to_cell(self.start_pose.x, self.start_pose.y)
        if start_cell is not None:
            for i, r in enumerate(regions):
                if start_cell in r.cells:
                    return i

        best_i: Optional[int] = None
        best_dist = float('inf')
        for i, r in enumerate(regions):
            dist = math.hypot(r.centroid[0] - self.start_pose.x, r.centroid[1] - self.start_pose.y)
            if dist < best_dist:
                best_dist = dist
                best_i = i
        if best_i is not None and best_dist <= self.start_room_claim_radius_m:
            return best_i
        return None

    def _prioritize_start_region(self, regions: List[RawRegion]) -> List[RawRegion]:
        start_i = self._start_region_index(regions)
        if start_i is None or start_i == 0:
            return regions
        return [regions[start_i]] + regions[:start_i] + regions[start_i + 1:]

    def _force_start_region_id(self, regions: List[TrackedRegion]) -> None:
        if not self.lock_start_room_as_r1 or self.start_pose is None:
            return
        start_i = self._start_region_index(regions)
        if start_i is None:
            return

        start_region = regions[start_i]
        if start_region.id == self.start_room_id:
            return

        visible_ids = {r.id for i, r in enumerate(regions) if i != start_i}
        if self.start_room_id in visible_ids or self.start_room_id in self.locked_rooms:
            return

        old_id = start_region.id
        start_region.id = self.start_room_id
        self.region_memory.pop(old_id, None)
        if self.persist_region_ids:
            self.region_memory[self.start_room_id] = start_region
        self.next_region_id = max(self.next_region_id, self.start_room_id + 1)
        self.get_logger().info(f'START_ROOM_ID_ASSIGNED | old=R{old_id} new=R{self.start_room_id}')

    def _track_regions(self, raw_regions: List[RawRegion]) -> List[TrackedRegion]:
        now = time.time()
        memories: Dict[int, TrackedRegion] = {}
        for old in self.regions:
            memories[old.id] = old
        if self.persist_region_ids:
            memories.update(self.region_memory)

        current_ids = {old.id for old in self.regions}
        assignments: Dict[int, TrackedRegion] = {}
        used_ids: Set[int] = set()
        candidates: List[Tuple[float, int, int]] = []

        for raw_i, raw in enumerate(raw_regions):
            for room in self.locked_rooms.values():
                score = self._locked_room_match_score(raw, room)
                if score is not None:
                    candidates.append((score, raw_i, room.id))
            for old in memories.values():
                score = self._region_match_score(raw, old, old.id in current_ids)
                if score is None:
                    continue
                candidates.append((score, raw_i, old.id))

        candidates.sort(reverse=True)
        for _score, raw_i, rid in candidates:
            if raw_i in assignments or rid in used_ids:
                continue
            old = memories.get(rid)
            if old is None:
                room = self.locked_rooms.get(rid)
                if room is None:
                    continue
                x, y, _yaw = room.pose
                old = TrackedRegion(
                    cells=set(),
                    centroid=(x, y),
                    area_m2=room.area_m2,
                    mean_clearance=0.0,
                    max_clearance=0.0,
                    elongation=1.0,
                    occupied_boundary_len=0.0,
                    unknown_boundary_len=0.0,
                    gateway_boundary_len=0.0,
                    closure_score=1.0,
                    unknown_boundary_ratio=0.0,
                    frontier_cell_count=0,
                    id=room.id,
                    stable_count=max(1, room.observations),
                    created_time=room.locked_time,
                    updated_time=room.last_seen_time,
                    state=room.state,
                )
            assignments[raw_i] = old
            used_ids.add(rid)

        tracked: List[TrackedRegion] = []
        for raw_i, raw in enumerate(raw_regions):
            old = assignments.get(raw_i)
            if old is not None:
                rid = old.id
                stable_count = old.stable_count + 1
                created = old.created_time
            else:
                rid = self.next_region_id
                self.next_region_id += 1
                stable_count = 1
                created = now
            state = self._classify_region(raw, stable_count)

            # RawRegion already has a `state` field. Passing `**raw.__dict__`
            # and `state=...` together raises:
            #   TrackedRegion() got multiple values for keyword argument 'state'
            # Therefore copy all raw fields except the provisional raw state,
            # then inject the freshly classified tracked state.
            raw_fields = dict(raw.__dict__)
            raw_fields.pop('state', None)
            tracked.append(TrackedRegion(
                **raw_fields,
                id=rid,
                stable_count=stable_count,
                created_time=created,
                updated_time=now,
                state=state,
            ))
        if self.persist_region_ids:
            for region in tracked:
                self.region_memory[region.id] = region
        return tracked

    def _region_match_score(self, raw: RawRegion, old: TrackedRegion, currently_visible: bool) -> Optional[float]:
        dist = math.hypot(raw.centroid[0] - old.centroid[0], raw.centroid[1] - old.centroid[1])
        area_ratio = min(raw.area_m2, old.area_m2) / max(1e-9, max(raw.area_m2, old.area_m2))

        overlap = len(raw.cells & old.cells)
        if overlap > 0:
            raw_overlap = overlap / max(1, len(raw.cells))
            old_overlap = overlap / max(1, len(old.cells))
            if max(raw_overlap, old_overlap) < self.region_match_overlap_ratio_min and dist > self.region_match_max_distance:
                return None
            visibility_bonus = 0.75 if currently_visible else 0.0
            return 12.0 * raw_overlap + 8.0 * old_overlap + 2.0 * area_ratio - 0.35 * dist + visibility_bonus

        max_dist = self.region_match_max_distance if currently_visible else self.region_reacquire_max_distance
        if dist > max_dist or area_ratio < self.region_match_area_ratio_min:
            return None
        visibility_bonus = 0.5 if currently_visible else 0.0
        return 2.5 * area_ratio - 0.75 * dist + visibility_bonus

    def _locked_room_match_score(self, raw: RawRegion, room: LockedRoom) -> Optional[float]:
        if self.grid is None:
            return None
        x, y, _yaw = room.pose
        pose_cell = self._world_to_cell(x, y)
        if pose_cell is not None and pose_cell in raw.cells:
            return 1000.0 + min(50.0, math.sqrt(max(0.0, raw.area_m2)))

        dist = math.hypot(raw.centroid[0] - x, raw.centroid[1] - y)
        if dist > self.locked_room_claim_radius_m:
            return None
        area_ratio = min(raw.area_m2, room.area_m2) / max(1e-9, max(raw.area_m2, room.area_m2))
        return 800.0 + 5.0 * area_ratio - dist

    def _classify_region(self, r: RawRegion, stable_count: int) -> str:
        if r.unknown_boundary_ratio >= self.open_unknown_boundary_ratio:
            return 'OPEN_REGION'
        if stable_count >= self.stable_confirm_updates and r.unknown_boundary_ratio <= self.stable_unknown_boundary_ratio and r.closure_score >= self.stable_closure_min:
            if r.elongation >= self.corridor_elongation_min:
                return 'STABLE_CORRIDOR'
            return 'STABLE_ROOM'
        return 'PROVISIONAL_REGION'

    def _update_locked_rooms(self, regions: List[TrackedRegion], clearance: List[float]) -> None:
        if not self.lock_stable_rooms:
            return

        visible_ids = {r.id for r in regions}
        for room in self.locked_rooms.values():
            room.visible = room.id in visible_ids

        now = time.time()
        for r in regions:
            lock_reason = self._locked_room_reason(r)
            if lock_reason is None:
                continue
            pose = self._representative_pose_for_region(r, clearance)
            if pose is None:
                continue
            existing = self.locked_rooms.get(r.id)
            if existing is None:
                self.locked_rooms[r.id] = LockedRoom(
                    id=r.id,
                    name=f'R{r.id}',
                    pose=pose,
                    source_centroid=r.centroid,
                    area_m2=r.area_m2,
                    locked_time=now,
                    last_seen_time=now,
                    observations=1,
                    state=lock_reason,
                    visible=True,
                )
                self.get_logger().info(
                    f'LOCKED_ROOM | name=R{r.id} reason={lock_reason} pose=({pose[0]:.2f}, {pose[1]:.2f}, {pose[2]:.2f}) area={r.area_m2:.2f}m2'
                )
                continue

            existing.last_seen_time = now
            existing.observations += 1
            existing.state = lock_reason
            existing.visible = True
            existing.area_m2 = max(existing.area_m2, r.area_m2)
            existing.source_centroid = r.centroid
            if not self.lock_stable_room_pose:
                existing.pose = pose

    def _ensure_start_room_locked(self, regions: List[TrackedRegion], clearance: List[float]) -> None:
        if not self.lock_start_room_as_r1 or self.start_pose is None:
            return
        if self.start_room_id in self.locked_rooms:
            return

        start_i = self._start_region_index(regions)
        if start_i is None:
            return
        region = regions[start_i]
        if region.id != self.start_room_id:
            return

        pose: Optional[Tuple[float, float, float]] = None
        if self.start_room_pose_from_robot:
            start_cell = self._world_to_cell(self.start_pose.x, self.start_pose.y)
            if start_cell is not None and start_cell in region.cells:
                pose = (self.start_pose.x, self.start_pose.y, self.start_pose.yaw)
        if pose is None:
            pose = self._representative_pose_for_region(region, clearance)
        if pose is None:
            return

        now = time.time()
        self.locked_rooms[self.start_room_id] = LockedRoom(
            id=self.start_room_id,
            name=f'R{self.start_room_id}',
            pose=pose,
            source_centroid=region.centroid,
            area_m2=region.area_m2,
            locked_time=now,
            last_seen_time=now,
            observations=max(1, region.stable_count),
            state='START_ROOM',
            visible=True,
        )
        self.get_logger().info(
            f'START_ROOM_LOCKED | name=R{self.start_room_id} '
            f'pose=({pose[0]:.2f}, {pose[1]:.2f}, {pose[2]:.2f}) area={region.area_m2:.2f}m2'
        )

    def _reachable_seed_cell(self, robot_cell: Optional[Cell], free_cells: Set[Cell]) -> Optional[Cell]:
        if robot_cell is None:
            return None
        if robot_cell in free_cells:
            return robot_cell
        if self.grid is None or self.robot_free_snap_radius_m <= 0.0:
            return None

        max_radius = max(1, int(math.ceil(self.robot_free_snap_radius_m / max(1e-9, self.grid.resolution))))
        best: Optional[Cell] = None
        best_d2: Optional[int] = None
        rx, ry = robot_cell
        for dy in range(-max_radius, max_radius + 1):
            for dx in range(-max_radius, max_radius + 1):
                c = (rx + dx, ry + dy)
                if c not in free_cells:
                    continue
                d2 = dx * dx + dy * dy
                if best_d2 is None or d2 < best_d2:
                    best = c
                    best_d2 = d2
        return best

    def _locked_room_reason(self, r: TrackedRegion) -> Optional[str]:
        if r.area_m2 < self.locked_room_min_area_m2:
            return None
        if r.max_clearance < self.locked_room_min_clearance_m:
            return None
        if r.state == 'STABLE_ROOM':
            return 'STABLE_ROOM'
        if not self.lock_provisional_room_candidates:
            return None
        if r.state == 'OPEN_REGION' or r.elongation >= self.corridor_elongation_min:
            return None
        if r.stable_count < max(self.stable_confirm_updates, self.locked_room_min_stable_count):
            return None
        if r.unknown_boundary_ratio >= self.open_unknown_boundary_ratio:
            return None
        return 'LOCKED_ROOM_CANDIDATE'

    def _representative_pose_for_region(self, r: TrackedRegion, clearance: List[float]) -> Optional[Tuple[float, float, float]]:
        if not r.cells:
            return None
        cx, cy = r.centroid
        safe_cells = [
            c for c in r.cells
            if math.isfinite(clearance[self._idx(c)]) and clearance[self._idx(c)] >= self.locked_room_min_clearance_m
        ]
        if not safe_cells:
            return None

        # Prefer cells far from walls, while still staying near the region center.
        best = max(
            safe_cells,
            key=lambda c: (
                clearance[self._idx(c)]
                - 0.20 * math.hypot(self._cell_to_world(*c)[0] - cx, self._cell_to_world(*c)[1] - cy)
            ),
        )
        x, y = self._cell_to_world(*best)
        return (x, y, 0.0)

    # ------------------------------------------------------------------
    # Frontiers / latent regions
    # ------------------------------------------------------------------

    def _frontier_clusters(self, reachable: Set[Cell], unknown: Set[Cell]) -> List[Set[Cell]]:
        frontiers: Set[Cell] = set()
        for c in reachable:
            if any(n in unknown for n in self._neighbors4(c)):
                frontiers.add(c)
        comps = self._connected_components(frontiers, use8=self.frontier_cluster_connectivity_8)
        return [c for c in comps if len(c) >= self.min_frontier_cluster_size]

    def _attach_frontiers_to_regions(self, frontier_clusters: List[Set[Cell]], regions: List[TrackedRegion], unknown: Set[Cell]) -> List[FrontierPortal]:
        cell_to_region: Dict[Cell, int] = {}
        for r in regions:
            for c in r.cells:
                cell_to_region[c] = r.id
        portals: List[FrontierPortal] = []
        for fc in frontier_clusters:
            counts: Dict[int, int] = {}
            for c in fc:
                rid = cell_to_region.get(c)
                if rid is not None:
                    counts[rid] = counts.get(rid, 0) + 1
            if not counts:
                continue
            parent = max(counts.items(), key=lambda kv: kv[1])[0]
            centroid = self._centroid_world(fc)
            yaw = self._frontier_outward_yaw(fc, unknown)
            width = len(fc) * (self.grid.resolution if self.grid else 0.05)
            portals.append(FrontierPortal(
                id=self.next_portal_id,
                parent_region_id=parent,
                cells=fc,
                centroid=centroid,
                outward_yaw=yaw,
                width=width,
            ))
            self.next_portal_id += 1
        return portals

    def _frontier_outward_yaw(self, frontier: Set[Cell], unknown: Set[Cell]) -> float:
        vx = 0.0
        vy = 0.0
        for c in frontier:
            wx, wy = self._cell_to_world(*c)
            for n in self._neighbors8(c):
                if n in unknown:
                    nx, ny = self._cell_to_world(*n)
                    vx += nx - wx
                    vy += ny - wy
        if abs(vx) < 1e-9 and abs(vy) < 1e-9:
            return 0.0
        return math.atan2(vy, vx)

    # ------------------------------------------------------------------
    # Publications
    # ------------------------------------------------------------------

    def _publish_region_map(self, regions: List[TrackedRegion]):
        assert self.map_msg is not None and self.grid is not None
        msg = OccupancyGrid()
        msg.header = self.map_msg.header
        msg.header.frame_id = self.global_frame
        msg.info = self.map_msg.info

        if self.region_map_id_only:
            # v9: publish only region identity.  Do not encode raw obstacles,
            # inflated obstacles, gateway cuts, GVD, or free/unknown background
            # into this map.  This keeps RViz focused on the partition result.
            # Non-region cells stay unknown so they visually recede behind the
            # original /map layer.
            data = [-1] * (self.grid.width * self.grid.height)
            for r in regions:
                # OccupancyGrid data is int8-like and bounded to [-1, 100].
                # Use a spaced categorical value for visualization; the true id
                # is always shown by the R{id} text marker and summary.
                v = 1 + ((r.id * 37) % 98)
                for c in r.cells:
                    data[self._idx(c)] = v
            msg.data = data
            self.last_region_map_msg = msg
            self.region_map_pub.publish(msg)
            return

        data = [0] * (self.grid.width * self.grid.height)
        for idx, val in enumerate(self.map_msg.data):
            if val < 0:
                data[idx] = -1
            elif val >= self.occupied_threshold:
                data[idx] = 100
            else:
                data[idx] = 0
        # Encode regions as separated occupancy values. RViz OccupancyGrid
        # is not categorical, but spaced values make adjacent region ids more
        # distinguishable than 1,2,3,... under the standard color schemes.
        for r in regions:
            v = 12 + ((r.id * 17) % 82)
            for c in r.cells:
                data[self._idx(c)] = v
        msg.data = data
        self.last_region_map_msg = msg
        self.region_map_pub.publish(msg)

    def _publish_markers(
        self,
        regions: List[TrackedRegion],
        portals: List[FrontierPortal],
        cuts: List[GatewayCut],
        gvd_cells: Set[Cell],
    ):
        ma = MarkerArray()
        now = self.get_clock().now().to_msg()

        # Clear old markers.
        clear = Marker()
        clear.action = Marker.DELETEALL
        ma.markers.append(clear)

        marker_id = 1
        for r in regions:
            color = self._color_for_region(r.id, r.state)
            cube = Marker()
            cube.header.frame_id = self.global_frame
            cube.header.stamp = now
            cube.ns = 'regions'
            cube.id = marker_id
            marker_id += 1
            cube.type = Marker.POINTS
            cube.action = Marker.ADD
            cube.scale.x = self.grid.resolution * 1.25
            cube.scale.y = self.grid.resolution * 1.25
            cube.scale.z = 0.0
            cube.color.r, cube.color.g, cube.color.b, cube.color.a = color
            cube.pose.orientation.w = 1.0
            # Draw cells as flat point sprites instead of 3D cubes; CUBE_LIST
            # produces visible dark seams between neighboring map cells in RViz.
            # Region-id-only overlay: draw dense region cells by default.
            # Keep a safety stride only for abnormally large maps.
            stride = max(1, len(r.cells) // max(1, self.max_region_marker_cells))
            for k, c in enumerate(r.cells):
                if k % stride != 0:
                    continue
                wx, wy = self._cell_to_world(*c)
                p = Point(x=wx, y=wy, z=self.marker_z)
                cube.points.append(p)
            ma.markers.append(cube)

            if self.publish_region_outlines:
                outline = Marker()
                outline.header.frame_id = self.global_frame
                outline.header.stamp = now
                outline.ns = 'region_outlines'
                outline.id = marker_id
                marker_id += 1
                outline.type = Marker.LINE_LIST
                outline.action = Marker.ADD
                outline.scale.x = max(0.005, self.region_outline_width_m)
                outline.color.r, outline.color.g, outline.color.b, _alpha = color
                outline.color.a = 0.95
                outline.pose.orientation.w = 1.0
                outline.points = self._region_outline_points(r.cells)
                ma.markers.append(outline)

            if self.publish_region_text and (self.publish_unlocked_region_text or r.id in self.locked_rooms):
                text = Marker()
                text.header.frame_id = self.global_frame
                text.header.stamp = now
                text.ns = 'region_text'
                text.id = marker_id
                marker_id += 1
                text.type = Marker.TEXT_VIEW_FACING
                text.action = Marker.ADD
                text.pose.position.x = r.centroid[0]
                text.pose.position.y = r.centroid[1]
                text.pose.position.z = self.text_z
                text.pose.orientation.w = 1.0
                text.scale.z = 0.18
                text.color.r = 1.0
                text.color.g = 1.0
                text.color.b = 1.0
                text.color.a = 0.95
                if self.region_text_id_only:
                    room = self.locked_rooms.get(r.id)
                    text.text = room.name if room is not None else f'R{r.id}'
                    text.scale.z = 0.24
                else:
                    text.text = (
                        f'R{r.id} {r.state}\n'
                        f'A={r.area_m2:.1f}m² unk={r.unknown_boundary_ratio:.2f}\n'
                        f'close={r.closure_score:.2f} age={r.stable_count}'
                    )
                ma.markers.append(text)

        if self.publish_latent_frontiers:
            for ptl in portals:
                sph = Marker()
                sph.header.frame_id = self.global_frame
                sph.header.stamp = now
                sph.ns = 'latent_frontiers'
                sph.id = marker_id
                marker_id += 1
                sph.type = Marker.SPHERE
                sph.action = Marker.ADD
                sph.pose.position.x = ptl.centroid[0]
                sph.pose.position.y = ptl.centroid[1]
                sph.pose.position.z = self.text_z * 0.55
                sph.pose.orientation.w = 1.0
                sph.scale.x = 0.18
                sph.scale.y = 0.18
                sph.scale.z = 0.18
                sph.color.r = 1.0
                sph.color.g = 0.6
                sph.color.b = 0.05
                sph.color.a = 0.95
                ma.markers.append(sph)

                arr = Marker()
                arr.header.frame_id = self.global_frame
                arr.header.stamp = now
                arr.ns = 'frontier_arrows'
                arr.id = marker_id
                marker_id += 1
                arr.type = Marker.ARROW
                arr.action = Marker.ADD
                arr.pose.orientation.w = 1.0
                arr.scale.x = 0.035
                arr.scale.y = 0.07
                arr.scale.z = 0.12
                arr.color.r = 1.0
                arr.color.g = 0.65
                arr.color.b = 0.05
                arr.color.a = 0.9
                x0, y0 = ptl.centroid
                x1 = x0 + 0.35 * math.cos(ptl.outward_yaw)
                y1 = y0 + 0.35 * math.sin(ptl.outward_yaw)
                arr.points.append(Point(x=x0, y=y0, z=self.text_z * 0.35))
                arr.points.append(Point(x=x1, y=y1, z=self.text_z * 0.35))
                ma.markers.append(arr)

        if self.publish_gateway_markers:
            for cut in cuts:
                mk = Marker()
                mk.header.frame_id = self.global_frame
                mk.header.stamp = now
                mk.ns = 'gateway_cuts'
                mk.id = marker_id
                marker_id += 1
                mk.type = Marker.CUBE_LIST
                mk.action = Marker.ADD
                mk.scale.x = self.grid.resolution * 1.2
                mk.scale.y = self.grid.resolution * 1.2
                mk.scale.z = 0.06
                mk.pose.orientation.w = 1.0
                mk.color.r = 1.0
                mk.color.g = 0.05
                mk.color.b = 0.05
                mk.color.a = 0.95
                for c in cut.cells:
                    wx, wy = self._cell_to_world(*c)
                    mk.points.append(Point(x=wx, y=wy, z=self.marker_z + 0.045))
                ma.markers.append(mk)

        if self.publish_gvd_markers and gvd_cells:
            mk = Marker()
            mk.header.frame_id = self.global_frame
            mk.header.stamp = now
            mk.ns = 'approx_gvd'
            mk.id = marker_id
            marker_id += 1
            mk.type = Marker.POINTS
            mk.action = Marker.ADD
            mk.scale.x = 0.035
            mk.scale.y = 0.035
            mk.pose.orientation.w = 1.0
            mk.color.r = 0.1
            mk.color.g = 0.75
            mk.color.b = 1.0
            mk.color.a = 0.8
            stride = max(1, len(gvd_cells) // max(1, self.max_gvd_marker_points))
            for k, c in enumerate(gvd_cells):
                if k % stride != 0:
                    continue
                wx, wy = self._cell_to_world(*c)
                mk.points.append(Point(x=wx, y=wy, z=self.marker_z + 0.08))
            ma.markers.append(mk)

        if self.publish_locked_room_markers:
            for room in sorted(self.locked_rooms.values(), key=lambda r: r.id):
                x, y, _yaw = room.pose
                anchor = Marker()
                anchor.header.frame_id = self.global_frame
                anchor.header.stamp = now
                anchor.ns = 'locked_room_anchors'
                anchor.id = marker_id
                marker_id += 1
                anchor.type = Marker.SPHERE
                anchor.action = Marker.ADD
                anchor.pose.position.x = x
                anchor.pose.position.y = y
                anchor.pose.position.z = self.text_z * 0.28
                anchor.pose.orientation.w = 1.0
                anchor.scale.x = 0.16
                anchor.scale.y = 0.16
                anchor.scale.z = 0.16
                anchor.color.r = 1.0
                anchor.color.g = 0.92
                anchor.color.b = 0.15
                anchor.color.a = 0.95 if room.visible else 0.45
                ma.markers.append(anchor)

                label = Marker()
                label.header.frame_id = self.global_frame
                label.header.stamp = now
                label.ns = 'locked_room_text'
                label.id = marker_id
                marker_id += 1
                label.type = Marker.TEXT_VIEW_FACING
                label.action = Marker.ADD
                label.pose.position.x = x
                label.pose.position.y = y
                label.pose.position.z = self.text_z * 0.72
                label.pose.orientation.w = 1.0
                label.scale.z = 0.15
                label.color.r = 1.0
                label.color.g = 0.95
                label.color.b = 0.25
                label.color.a = 0.95 if room.visible else 0.55
                label.text = f'{room.name} lock'
                ma.markers.append(label)

        self.marker_pub.publish(ma)

    def _region_outline_points(self, cells: Set[Cell]) -> List[Point]:
        """Return map-cell boundary segments for a filled region.

        The region interior is already published as a CUBE_LIST.  This method
        only traces the exposed cell edges, so RViz shows the actual SLAM-map
        silhouette instead of a rectangular bounding box.
        """
        if self.grid is None or not cells:
            return []

        points: List[Point] = []

        def corner_point(x: int, y: int) -> Point:
            return Point(
                x=self.grid.origin_x + x * self.grid.resolution,
                y=self.grid.origin_y + y * self.grid.resolution,
                z=self.marker_z + 0.025,
            )

        def add_edge(x1: int, y1: int, x2: int, y2: int) -> None:
            points.append(corner_point(x1, y1))
            points.append(corner_point(x2, y2))

        for x, y in cells:
            edge_checks = (
                ((x, y - 1), (x, y), (x + 1, y)),
                ((x + 1, y), (x + 1, y), (x + 1, y + 1)),
                ((x, y + 1), (x, y + 1), (x + 1, y + 1)),
                ((x - 1, y), (x, y), (x, y + 1)),
            )
            for neighbor, start, end in edge_checks:
                nx, ny = neighbor
                if (
                    nx < 0
                    or ny < 0
                    or nx >= self.grid.width
                    or ny >= self.grid.height
                    or neighbor not in cells
                ):
                    add_edge(start[0], start[1], end[0], end[1])

        return points

    def _publish_summary(self, d: Dict):
        msg = String()
        msg.data = json.dumps(d, ensure_ascii=False)
        self.summary_pub.publish(msg)

    def _publish_locked_rooms(self, regions: List[TrackedRegion]) -> None:
        current = {r.id: r for r in regions}
        rooms = []
        for room in sorted(self.locked_rooms.values(), key=lambda r: r.id):
            live = current.get(room.id)
            x, y, yaw = room.pose
            item = {
                'id': room.id,
                'name': room.name,
                'pose': {'x': round(x, 4), 'y': round(y, 4), 'yaw': round(yaw, 4)},
                'area_m2': round(room.area_m2, 3),
                'locked_time': round(room.locked_time, 3),
                'last_seen_time': round(room.last_seen_time, 3),
                'observations': room.observations,
                'state': room.state,
                'visible': room.visible,
            }
            if live is not None:
                item['current'] = {
                    'centroid': {'x': round(live.centroid[0], 4), 'y': round(live.centroid[1], 4)},
                    'area_m2': round(live.area_m2, 3),
                    'state': live.state,
                    'stable_count': live.stable_count,
                }
            rooms.append(item)

        msg = String()
        msg.data = json.dumps({
            'frame_id': self.global_frame,
            'count': len(rooms),
            'rooms': rooms,
        }, ensure_ascii=False)
        self.locked_rooms_pub.publish(msg)

    def _publish_best_region_goal(self, robot: Optional[Pose2D], regions: List[TrackedRegion]):
        if robot is None or not regions:
            return
        def priority(r: TrackedRegion) -> float:
            dist = math.hypot(r.centroid[0] - robot.x, r.centroid[1] - robot.y)
            state_bonus = {
                'OPEN_REGION': 4.0,
                'PROVISIONAL_REGION': 2.0,
                'STABLE_ROOM': 1.0,
                'STABLE_CORRIDOR': 0.8,
            }.get(r.state, 0.0)
            return state_bonus + 2.0 * r.unknown_boundary_ratio + 0.1 * math.sqrt(max(0.0, r.area_m2)) - 0.25 * dist
        best = max(regions, key=priority)
        ps = PoseStamped()
        ps.header.frame_id = self.global_frame
        ps.header.stamp = self.get_clock().now().to_msg()
        ps.pose.position.x = best.centroid[0]
        ps.pose.position.y = best.centroid[1]
        ps.pose.position.z = 0.0
        ps.pose.orientation.w = 1.0
        self.selected_region_pub.publish(ps)

    def _color_for_region(self, rid: int, state: str) -> Tuple[float, float, float, float]:
        # v11: fixed region-id palette, deliberately avoiding red because red
        # was previously confused with obstacle/gateway debug markers.  Color
        # now means only region id; state is shown only in summary/text when
        # enabled.
        palette = [
            (0.10, 0.75, 1.00),  # cyan
            (0.15, 0.95, 0.45),  # green
            (0.45, 0.50, 1.00),  # blue-violet
            (0.95, 0.75, 0.20),  # amber
            (0.75, 0.35, 1.00),  # purple
            (0.25, 0.95, 0.85),  # teal
            (0.95, 0.55, 0.25),  # orange
            (0.55, 0.85, 0.25),  # yellow-green
            (0.25, 0.65, 0.95),  # sky-blue
            (0.90, 0.45, 0.85),  # magenta
        ]
        r, g, b = palette[(max(1, rid) - 1) % len(palette)]
        alpha = max(0.05, min(1.0, self.region_marker_alpha))
        return (r, g, b, alpha)

    def _hsv_to_rgb(self, h: float, s: float, v: float) -> Tuple[float, float, float]:
        i = int(h * 6.0)
        f = h * 6.0 - i
        p = v * (1.0 - s)
        q = v * (1.0 - f * s)
        t = v * (1.0 - (1.0 - f) * s)
        i %= 6
        if i == 0:
            return v, t, p
        if i == 1:
            return q, v, p
        if i == 2:
            return p, v, t
        if i == 3:
            return p, q, v
        if i == 4:
            return t, p, v
        return v, p, q


def main(args=None):
    rclpy.init(args=args)
    node = SlamRegionGraphNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.destroy_node()
        except Exception:
            pass
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
