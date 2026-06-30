#!/usr/bin/env python3

from __future__ import annotations

import time
from typing import List, Optional

import rclpy
from rclpy.node import Node
from lifecycle_msgs.srv import GetState, ChangeState
from lifecycle_msgs.msg import Transition

# Keep this list conservative. These are the nodes that must be active for /navigate_to_pose
# to accept and execute goals. Optional Jazzy extras are monitored separately.
REQUIRED_NODES = [
    'map_server',
    'amcl',
    'controller_server',
    'smoother_server',
    'planner_server',
    'behavior_server',
    'bt_navigator',
    'waypoint_follower',
    'velocity_smoother',
]

OPTIONAL_NODES = [
    'route_server',
    'collision_monitor',
    'docking_server',
]

STATE_NAMES = {
    0: 'unknown',
    1: 'unconfigured',
    2: 'inactive',
    3: 'active',
    4: 'finalized',
}


class PersistentNav2LifecycleGuard(Node):
    """Keep Nav2 lifecycle nodes active in the single-domain no-namespace tests.

    Earlier test versions occasionally left planner_server inactive while bt_navigator was
    active, which made action discovery appear partially valid but planning unreliable.
    This guard does not exit after the first success. It keeps monitoring and re-activating
    inactive/unconfigured required nodes.
    """

    def __init__(self) -> None:
        super().__init__('force_nav2_lifecycle_v29')
        self.declare_parameter('nodes', REQUIRED_NODES)
        self.declare_parameter('optional_nodes', OPTIONAL_NODES)
        self.declare_parameter('start_delay_sec', 8.0)
        self.declare_parameter('period_sec', 3.0)
        self.declare_parameter('service_timeout_sec', 1.5)
        self.declare_parameter('transition_timeout_sec', 8.0)
        self.declare_parameter('log_every_n_ok', 10)

        self.nodes: List[str] = [str(x) for x in self.get_parameter('nodes').value]
        self.optional_nodes: List[str] = [str(x) for x in self.get_parameter('optional_nodes').value]
        self.start_delay_sec = float(self.get_parameter('start_delay_sec').value)
        self.period_sec = float(self.get_parameter('period_sec').value)
        self.service_timeout_sec = float(self.get_parameter('service_timeout_sec').value)
        self.transition_timeout_sec = float(self.get_parameter('transition_timeout_sec').value)
        self.log_every_n_ok = max(1, int(self.get_parameter('log_every_n_ok').value))
        self.tick_count = 0
        self.started = False

        self.get_logger().info(
            'V29_PERSISTENT_NAV2_LIFECYCLE_GUARD_READY | '
            f'start_delay={self.start_delay_sec}s period={self.period_sec}s nodes={self.nodes}'
        )
        self.start_time = time.monotonic()
        self.timer = self.create_timer(self.period_sec, self._tick)

    @staticmethod
    def _norm(name: str) -> str:
        return name if name.startswith('/') else f'/{name}'

    def _wait_service(self, client, timeout: float) -> bool:
        end = time.monotonic() + timeout
        while time.monotonic() < end and rclpy.ok():
            if client.wait_for_service(timeout_sec=0.2):
                return True
        return False

    def _get_state(self, node_name: str) -> Optional[int]:
        n = self._norm(node_name)
        cli = self.create_client(GetState, f'{n}/get_state')
        if not self._wait_service(cli, self.service_timeout_sec):
            return None
        fut = cli.call_async(GetState.Request())
        rclpy.spin_until_future_complete(self, fut, timeout_sec=self.service_timeout_sec)
        if not fut.done() or fut.result() is None:
            return None
        return int(fut.result().current_state.id)

    def _change_state(self, node_name: str, transition_id: int) -> bool:
        n = self._norm(node_name)
        cli = self.create_client(ChangeState, f'{n}/change_state')
        if not self._wait_service(cli, self.service_timeout_sec):
            self.get_logger().warn(f'V29_LIFECYCLE_NO_CHANGE_SERVICE | node={n}')
            return False
        req = ChangeState.Request()
        req.transition.id = transition_id
        fut = cli.call_async(req)
        rclpy.spin_until_future_complete(self, fut, timeout_sec=self.transition_timeout_sec)
        if not fut.done() or fut.result() is None:
            self.get_logger().warn(f'V29_LIFECYCLE_TRANSITION_TIMEOUT | node={n} transition={transition_id}')
            return False
        ok = bool(fut.result().success)
        self.get_logger().info(f'V29_LIFECYCLE_TRANSITION | node={n} transition={transition_id} success={ok}')
        return ok

    def _ensure_active_once(self, node_name: str, optional: bool = False) -> bool:
        n = self._norm(node_name)
        state = self._get_state(n)
        if state is None:
            if optional:
                self.get_logger().debug(f'V29_LIFECYCLE_OPTIONAL_ABSENT | node={n}')
                return True
            self.get_logger().warn(f'V29_LIFECYCLE_REQUIRED_ABSENT | node={n}')
            return False

        if state == 3:
            return True

        self.get_logger().warn(f'V29_LIFECYCLE_FIX_NEEDED | node={n} state={STATE_NAMES.get(state, str(state))}[{state}]')

        # Normal lifecycle path: unconfigured -> inactive -> active.
        if state == 1:
            self._change_state(n, Transition.TRANSITION_CONFIGURE)
            state = self._get_state(n)
            self.get_logger().info(f'V29_LIFECYCLE_AFTER_CONFIGURE | node={n} state={STATE_NAMES.get(state, str(state))}[{state}]')

        if state == 2:
            self._change_state(n, Transition.TRANSITION_ACTIVATE)
            state = self._get_state(n)
            self.get_logger().info(f'V29_LIFECYCLE_AFTER_ACTIVATE | node={n} state={STATE_NAMES.get(state, str(state))}[{state}]')

        # Some nodes can land in finalized after a failed configure. Do not try to recover
        # from finalized here; the launch log will contain the actual plugin error.
        ok = (state == 3)
        if not ok and not optional:
            self.get_logger().error(f'V29_LIFECYCLE_STILL_NOT_ACTIVE | node={n} state={STATE_NAMES.get(state, str(state))}[{state}]')
        return ok or optional

    def _tick(self) -> None:
        if not self.started:
            if time.monotonic() - self.start_time < self.start_delay_sec:
                return
            self.started = True
            self.get_logger().info('V29_LIFECYCLE_MONITOR_START')

        self.tick_count += 1
        ok_all = True
        states = []
        for node_name in self.nodes:
            ok = self._ensure_active_once(node_name, optional=False)
            ok_all = ok_all and ok
            st = self._get_state(node_name)
            states.append(f'{self._norm(node_name)}={STATE_NAMES.get(st, str(st))}[{st}]')

        for node_name in self.optional_nodes:
            self._ensure_active_once(node_name, optional=True)

        if ok_all:
            if self.tick_count % self.log_every_n_ok == 0 or self.tick_count <= 2:
                self.get_logger().info('V29_LIFECYCLE_ALL_REQUIRED_ACTIVE | /navigate_to_pose should be available | ' + ' '.join(states))
        else:
            self.get_logger().warn('V29_LIFECYCLE_NOT_READY | ' + ' '.join(states))


def main() -> None:
    rclpy.init()
    node = PersistentNav2LifecycleGuard()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
