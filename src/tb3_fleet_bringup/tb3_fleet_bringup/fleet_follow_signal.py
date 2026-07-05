#!/usr/bin/env python3

import argparse
import os
import sys
import time


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Send a short Burger follow-state command on the selected fleet domain.'
    )
    parser.add_argument(
        'command',
        choices=['follow', 'resume', 'pause', 'stop', 'toggle'],
    )
    parser.add_argument('--domain', default=os.environ.get('ROS_DOMAIN_ID', '24'))
    args = parser.parse_args(sys.argv[1:])

    os.environ['ROS_DOMAIN_ID'] = str(args.domain)
    os.environ.pop('ROS_DISCOVERY_SERVER', None)
    os.environ.pop('FASTRTPS_DEFAULT_PROFILES_FILE', None)
    os.environ.pop('FASTDDS_DEFAULT_PROFILES_FILE', None)
    os.environ['ROS_LOCALHOST_ONLY'] = '0'
    os.environ['ROS_AUTOMATIC_DISCOVERY_RANGE'] = 'SUBNET'
    os.environ['RMW_IMPLEMENTATION'] = 'rmw_fastrtps_cpp'

    import rclpy
    from rclpy.node import Node
    from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
    from std_msgs.msg import String

    rclpy.init()
    node = Node('fleet_follow_signal')
    qos = QoSProfile(
        depth=1,
        reliability=ReliabilityPolicy.RELIABLE,
        durability=DurabilityPolicy.TRANSIENT_LOCAL,
    )
    pub = node.create_publisher(String, '/fleet/follow_command', qos)

    deadline = time.monotonic() + 3.0
    while pub.get_subscription_count() == 0 and time.monotonic() < deadline:
        rclpy.spin_once(node, timeout_sec=0.1)

    msg = String()
    msg.data = args.command.upper()
    for _ in range(3):
        pub.publish(msg)
        rclpy.spin_once(node, timeout_sec=0.15)

    matched = pub.get_subscription_count()
    print(
        f'follow_command={msg.data} domain={args.domain} '
        f'matched_subscriptions={matched}'
    )
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
