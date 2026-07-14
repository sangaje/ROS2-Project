# Fleet Runtime Graph and Simplification Notes

This document tracks the default real-robot launch path after the runtime
readiness/motion-gate cleanup.

## Default Runtime Graph

| Area | Process | Domain | Runs when | Owns | Key inputs | Key outputs | Duplicate removed |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Leader | `leader_unified_dashboard` | 20 | `role:=leader`, dashboard enabled | browser render manifest, video freshness, final motion permission | `/system/ready`, map/risk/pose/status/video state | `/fleet/start_motion`, `/fleet/readiness_detail`, `/fleet/video_ready` | Replaces separate bridged dashboard UI/backend readiness outputs |
| Leader | `system_readiness_monitor` | 20 | `role:=leader` | infrastructure readiness only | map, leader/scout/follower poses, failover state | `/system/ready`, `/system/readiness_detail` | No longer bridged to field motion nodes |
| Leader | `leader_shadow_follow` | 20 | leader shadow enabled | leader Nav2 goal generation | active scout pose, leader pose, `/fleet/start_motion` | `/fleet/leader_coord_goal`, `/fleet/leader_nav_cancel` | Does not create `/cmd_vel` publisher unless direct-shadow mode is explicitly selected |
| Field | `unified_field_robot` | 21/22 | field robot stack | common role runtime and single hardware command authority | role, poses, localization, `/fleet/start_motion`, `/fleet/active_scout_rl_cmd` | `/cmd_vel`, status, role heartbeat, Nav2 goals | Receives external RL commands and owns hardware `/cmd_vel` publication |
| Scout | `scout_rl_policy_worker` | 22 | ACTIVE_SCOUT external worker | RL inference only | role/failover/localization, `/fleet/start_motion` | `/fleet/active_scout_rl_cmd` | No longer publishes hardware `/cmd_vel` directly in default launch |
| OMX | `waffle_node` | 20 | OMX stack enabled | OMX Nav2 action client | `/omx/nav_goal`, `/fleet/start_motion` | `/waffle/*` status/result/ack | Renamed motion gate from video-ready compatibility naming to start-motion |
| Bridge | `domain_bridge` configs | 20<->21/22 | registry/field bridge enabled | selected cross-domain topics | generated YAML | one process per direction/pair | Cross-domain readiness reduced to `/fleet/start_motion` and `/fleet/readiness_detail` |
| Camera | `opencv_camera_to_flask_yolo` | field | camera sender enabled | role-budgeted JPEG upload | camera frames | HTTP JPEG upload, bitrate logs | Existing role budgets retained |
| Map | `map_relay` | leader/field | map gateway enabled | latest-map-only gateway | `/map` | map bridge topic | Existing rate-limit/duplicate suppression retained |

## State Topics

Default cross-domain motion readiness is now:

- `/fleet/start_motion`: final latched `Bool` permission for motion.
- `/fleet/readiness_detail`: latched JSON detail for diagnosis.

Leader-local infrastructure readiness remains available as `/system/ready`, but
default field motion nodes no longer subscribe to it as an independent motion
gate. Dashboard UI/backend detail topics are no longer generated or bridged in
the default route.

## Static Metrics

Measured against `HEAD` before this cleanup and the current working tree:

| Metric | Before | After |
| --- | ---: | ---: |
| Selected runtime/code lines | 10095 | 10264 |
| Selected launch lines | 4565 | 4597 |
| Selected launch arguments | 202 | 204 |
| Python function count | 249 | 255 |
| Average function length | 48.4 | 48.0 |
| Longest function length | 1954 | 1959 |

Line count did not decrease in this step because the previously requested
`/fleet/start_motion` barrier and characterization tests were added in the same
working tree. The execution path was simplified by removing bridged dashboard
readiness topics, removing the default direct `/system/ready` motion gate, and
making hardware `/cmd_vel` publication single-owner for the default Scout RL
path.

## Characterization Tests

The following static invariants were added:

- Cross-domain readiness bridge contains only `/fleet/start_motion` and
  `/fleet/readiness_detail`.
- `leader_unified_dashboard` is the only `/fleet/start_motion` publisher.
- Default Scout RL commands flow through `/fleet/active_scout_rl_cmd` into
  `unified_field_robot`.
- `leader_shadow_follow` does not create a hardware `/cmd_vel` publisher unless
  direct-shadow mode is enabled.
