# Safe Inside Map Spawn

This variant spawns all robots inside the free area of `turtlemap.pgm`, away from the 3x3 pillar grid.

- burger1: (-1.90, 0.00), patrol robot
- burger2: (-1.80, -0.55), idle until rescue
- waffle1: (-1.80, 0.55), idle until rescue

Only burger1 patrols. When burger1 is failed with `/tb3_multi/fail_robot`, burger2 and waffle1 go to burger1's last position.
