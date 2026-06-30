#!/usr/bin/env python3
import time
import rclpy
from rclpy.node import Node

class CheckDualNav2Ready(Node):
    def __init__(self):
        super().__init__('check_dual_nav2_ready')
        self.declare_parameter('timeout_sec', 60.0)
        self.expected = ['/burger/navigate_to_pose', '/waffle/navigate_to_pose']

    def run(self):
        timeout = float(self.get_parameter('timeout_sec').value)
        start = time.time()
        while rclpy.ok() and time.time() - start < timeout:
            names_and_types = self.get_action_names_and_types()
            names = sorted([n for n, _ in names_and_types])
            missing = [x for x in self.expected if x not in names]
            self.get_logger().info('NAV2_ACTION_CHECK | actions=' + ','.join(names) + ' | missing=' + ','.join(missing))
            if not missing:
                self.get_logger().info('DUAL_NAV2_READY | /burger/navigate_to_pose and /waffle/navigate_to_pose are available')
                return 0
            time.sleep(2.0)
            rclpy.spin_once(self, timeout_sec=0.1)
        self.get_logger().error('DUAL_NAV2_NOT_READY_TIMEOUT | run ros2 node list and inspect launch logs')
        return 1

def main():
    rclpy.init()
    node = CheckDualNav2Ready()
    try:
        raise SystemExit(node.run())
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()
