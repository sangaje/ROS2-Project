
import argparse
import signal
import sys
import time
from dataclasses import dataclass
from typing import Optional

import rclpy
from rclpy.context import Context
from rclpy.executors import SingleThreadedExecutor
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from nav_msgs.msg import OccupancyGrid


def source_qos(depth: int = 10) -> QoSProfile:
    # Source side should match ordinary live publishers robustly.
    # Cartographer /map and risk map are expected to publish periodically.
    return QoSProfile(
        reliability=ReliabilityPolicy.RELIABLE,
        durability=DurabilityPolicy.VOLATILE,
        history=HistoryPolicy.KEEP_LAST,
        depth=depth,
    )


def target_qos(depth: int = 1) -> QoSProfile:
    # Target side is latched-like so RViz can receive the latest map/risk.
    return QoSProfile(
        reliability=ReliabilityPolicy.RELIABLE,
        durability=DurabilityPolicy.TRANSIENT_LOCAL,
        history=HistoryPolicy.KEEP_LAST,
        depth=depth,
    )


@dataclass
class BridgeState:
    name: str
    in_topic: str
    out_topic: str
    count_in: int = 0
    count_out: int = 0
    last_in_time: float = 0.0
    last_out_time: float = 0.0
    last_frame: str = ''
    last_size: str = ''
    last_msg: Optional[OccupancyGrid] = None


class ScoutMapRiskBridge:
    """
    Bridges exactly two scout products:
      /map -> /scout/map
      /risk/risk_map -> /scout/risk_map

    v3:
      - stores latest map/risk
      - republishes cached latest messages periodically
      - makes `ros2 topic echo --once` easier to debug
    """

    def __init__(self, args):
        self.args = args

        self.ctx_from = Context()
        self.ctx_to = Context()

        rclpy.init(args=None, context=self.ctx_from, domain_id=args.from_domain)
        rclpy.init(args=None, context=self.ctx_to, domain_id=args.to_domain)

        self.from_node = rclpy.create_node(
            f'scout_map_risk_bridge_from_{args.from_domain}',
            context=self.ctx_from,
        )
        self.to_node = rclpy.create_node(
            f'scout_map_risk_bridge_to_{args.to_domain}',
            context=self.ctx_to,
        )

        self.src_qos = source_qos(depth=args.source_qos_depth)
        self.dst_qos = target_qos(depth=args.target_qos_depth)

        self.map_state = BridgeState('map', args.map_in, args.map_out)
        self.risk_state = BridgeState('risk', args.risk_in, args.risk_out)

        self.map_pub = self.to_node.create_publisher(OccupancyGrid, args.map_out, self.dst_qos)
        self.risk_pub = self.to_node.create_publisher(OccupancyGrid, args.risk_out, self.dst_qos)

        self.map_sub = self.from_node.create_subscription(
            OccupancyGrid,
            args.map_in,
            lambda msg: self.on_grid(msg, self.map_pub, self.map_state),
            self.src_qos,
        )
        self.risk_sub = self.from_node.create_subscription(
            OccupancyGrid,
            args.risk_in,
            lambda msg: self.on_grid(msg, self.risk_pub, self.risk_state),
            self.src_qos,
        )

        self.from_node.create_timer(args.log_period_sec, self.log_status)
        if args.republish_period_sec > 0.0:
            self.from_node.create_timer(args.republish_period_sec, self.republish_cached)

        self.from_node.get_logger().info(
            f'ScoutMapRiskBridge v3 started | domain {args.from_domain} -> {args.to_domain}'
        )
        self.from_node.get_logger().info(f'MAP  : {args.map_in} -> {args.map_out}')
        self.from_node.get_logger().info(f'RISK : {args.risk_in} -> {args.risk_out}')
        self.from_node.get_logger().info(
            f'cached republish period: {args.republish_period_sec:.2f}s'
        )
        if args.rewrite_frame:
            self.from_node.get_logger().info(
                f'frame rewrite enabled: header.frame_id -> {args.target_frame}'
            )

    def rewrite_and_copy(self, msg: OccupancyGrid) -> OccupancyGrid:
        out = OccupancyGrid()
        out.header = msg.header
        out.info = msg.info
        out.data = msg.data
        if self.args.rewrite_frame:
            out.header.frame_id = self.args.target_frame
        return out

    def on_grid(self, msg: OccupancyGrid, pub, state: BridgeState):
        out = self.rewrite_and_copy(msg)
        state.last_msg = out
        state.count_in += 1
        state.last_in_time = time.time()
        state.last_frame = out.header.frame_id
        state.last_size = f'{out.info.width}x{out.info.height}@{out.info.resolution:.3f}'

        pub.publish(out)
        state.count_out += 1
        state.last_out_time = time.time()

    def republish_one(self, pub, state: BridgeState):
        if state.last_msg is None:
            return
        pub.publish(state.last_msg)
        state.count_out += 1
        state.last_out_time = time.time()

    def republish_cached(self):
        self.republish_one(self.map_pub, self.map_state)
        self.republish_one(self.risk_pub, self.risk_state)

    def log_status(self):
        now = time.time()
        parts = []
        for s in (self.map_state, self.risk_state):
            age_in = -1.0 if s.last_in_time <= 0.0 else now - s.last_in_time
            age_out = -1.0 if s.last_out_time <= 0.0 else now - s.last_out_time
            parts.append(
                f'{s.name}:in={s.count_in},out={s.count_out},age_in={age_in:.1f}s,'
                f'age_out={age_out:.1f}s,frame={s.last_frame},size={s.last_size}'
            )
        self.from_node.get_logger().info('SCOUT_BRIDGE_STATUS | ' + ' | '.join(parts))

    def spin(self):
        executor = SingleThreadedExecutor(context=self.ctx_from)
        executor.add_node(self.from_node)

        stop = {'value': False}

        def on_signal(signum, frame):
            stop['value'] = True

        signal.signal(signal.SIGINT, on_signal)
        signal.signal(signal.SIGTERM, on_signal)

        try:
            while rclpy.ok(context=self.ctx_from) and not stop['value']:
                executor.spin_once(timeout_sec=0.1)
        finally:
            executor.remove_node(self.from_node)
            self.from_node.destroy_node()
            self.to_node.destroy_node()
            rclpy.shutdown(context=self.ctx_from)
            rclpy.shutdown(context=self.ctx_to)


def parse_args(argv):
    p = argparse.ArgumentParser(
        description='Bridge scout /map and /risk/risk_map between ROS 2 domains.',
        allow_abbrev=False,
    )

    p.add_argument('--from-domain', type=int, required=True)
    p.add_argument('--to-domain', type=int, required=True)

    p.add_argument('--map-in', default='/map')
    p.add_argument('--risk-in', default='/risk/risk_map')

    p.add_argument('--map-out', default='/scout/map')
    p.add_argument('--risk-out', default='/scout/risk_map')

    p.add_argument('--rewrite-frame', action='store_true', default=True)
    p.add_argument('--no-rewrite-frame', dest='rewrite_frame', action='store_false')
    p.add_argument('--target-frame', default='scout_map')

    p.add_argument('--source-qos-depth', type=int, default=10)
    p.add_argument('--target-qos-depth', type=int, default=1)
    p.add_argument('--republish-period-sec', type=float, default=1.0)
    p.add_argument('--log-period-sec', type=float, default=2.0)

    cleaned = []
    skip_ros_args = False
    i = 0
    while i < len(argv):
        token = argv[i]
        if token == '--ros-args':
            skip_ros_args = True
            i += 1
            continue
        if skip_ros_args:
            i += 1
            continue
        cleaned.append(token)
        i += 1

    args, unknown = p.parse_known_args(cleaned)
    if unknown:
        print(f'[scout_map_risk_bridge] ignoring unknown args: {unknown}', file=sys.stderr)
    return args


def main(argv=None):
    args = parse_args(sys.argv[1:] if argv is None else argv)
    bridge = ScoutMapRiskBridge(args)
    bridge.spin()


if __name__ == '__main__':
    main()
