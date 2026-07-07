# Stable aligned spawn patch

- Gazebo robots spawn in the upper open area away from pillars.
- RViz markers use the exact same coordinates as Gazebo spawn poses.
- Gazebo /tf bridge and static map-to-odom TF publishers are disabled to avoid RViz flickering.
- /map is published once with TRANSIENT_LOCAL QoS instead of being re-published every second.
- Normal demo: burger1 patrols; burger2 and waffle1 stay idle. On burger1 failure, burger2 and waffle1 go to burger1 last pose.
