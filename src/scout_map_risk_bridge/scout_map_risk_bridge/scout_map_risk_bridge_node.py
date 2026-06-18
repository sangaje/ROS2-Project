
import argparse, signal, sys, time
from dataclasses import dataclass
from typing import Optional

import rclpy
from rclpy.context import Context
from rclpy.executors import SingleThreadedExecutor
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from nav_msgs.msg import OccupancyGrid
from std_msgs.msg import String


def qos(rel, dur, depth):
    return QoSProfile(reliability=rel, durability=dur, history=HistoryPolicy.KEEP_LAST, depth=depth)


def source_qos_profiles(depth):
    return [
        qos(ReliabilityPolicy.RELIABLE, DurabilityPolicy.VOLATILE, depth),
        qos(ReliabilityPolicy.RELIABLE, DurabilityPolicy.TRANSIENT_LOCAL, depth),
        qos(ReliabilityPolicy.BEST_EFFORT, DurabilityPolicy.VOLATILE, depth),
        qos(ReliabilityPolicy.BEST_EFFORT, DurabilityPolicy.TRANSIENT_LOCAL, depth),
    ]


def target_qos_profiles(depth):
    return [
        qos(ReliabilityPolicy.RELIABLE, DurabilityPolicy.TRANSIENT_LOCAL, depth),
        qos(ReliabilityPolicy.RELIABLE, DurabilityPolicy.VOLATILE, depth),
    ]


@dataclass
class State:
    name: str
    in_topic: str
    out_topic: str
    in_count: int = 0
    out_count: int = 0
    dup_count: int = 0
    last_in: float = 0.0
    last_out: float = 0.0
    frame: str = ""
    size: str = ""
    key: str = ""
    msg: Optional[OccupancyGrid] = None


class Bridge:
    def __init__(self, args):
        self.args = args
        self.ctx_from = Context()
        self.ctx_to = Context()
        rclpy.init(args=None, context=self.ctx_from, domain_id=args.from_domain)
        rclpy.init(args=None, context=self.ctx_to, domain_id=args.to_domain)

        self.n_from = rclpy.create_node(f"scout_bridge_from_{args.from_domain}", context=self.ctx_from)
        self.n_to = rclpy.create_node(f"scout_bridge_to_{args.to_domain}", context=self.ctx_to)

        self.map_state = State("map", args.map_in, args.map_out)
        self.risk_state = State("risk", args.risk_in, args.risk_out)

        tq = target_qos_profiles(args.target_qos_depth)
        self.map_pubs = [self.n_to.create_publisher(OccupancyGrid, args.map_out, q) for q in tq]
        self.risk_pubs = [self.n_to.create_publisher(OccupancyGrid, args.risk_out, q) for q in tq]
        self.status_pubs = [self.n_to.create_publisher(String, args.status_out, q) for q in tq]

        self.subs = []
        for q in source_qos_profiles(args.source_qos_depth):
            self.subs.append(self.n_from.create_subscription(
                OccupancyGrid, args.map_in,
                lambda msg, qq=q: self.on_msg(msg, self.map_pubs, self.map_state),
                q))
            self.subs.append(self.n_from.create_subscription(
                OccupancyGrid, args.risk_in,
                lambda msg, qq=q: self.on_msg(msg, self.risk_pubs, self.risk_state),
                q))

        self.n_from.create_timer(args.log_period_sec, self.log_status)
        self.n_from.create_timer(args.republish_period_sec, self.republish_cached)

        self.n_from.get_logger().info(f"ScoutMapRiskBridge v4 started | domain {args.from_domain} -> {args.to_domain}")
        self.n_from.get_logger().info(f"MAP  : {args.map_in} -> {args.map_out}")
        self.n_from.get_logger().info(f"RISK : {args.risk_in} -> {args.risk_out}")
        self.n_from.get_logger().info(f"STATUS: {args.status_out}")
        self.n_from.get_logger().info("QoS: source=4 profiles/topic, target=2 publishers/topic")

    def make_key(self, msg):
        return f"{msg.header.stamp.sec}.{msg.header.stamp.nanosec}:{msg.info.width}x{msg.info.height}:{len(msg.data)}"

    def copy_msg(self, msg):
        out = OccupancyGrid()
        out.header = msg.header
        out.info = msg.info
        out.data = msg.data
        if self.args.rewrite_frame:
            out.header.frame_id = self.args.target_frame
        return out

    def publish_all(self, pubs, msg):
        for p in pubs:
            p.publish(msg)

    def on_msg(self, msg, pubs, st):
        key = self.make_key(msg)
        if key == st.key:
            st.dup_count += 1
            return
        out = self.copy_msg(msg)
        st.key = key
        st.msg = out
        st.in_count += 1
        st.last_in = time.time()
        st.frame = out.header.frame_id
        st.size = f"{out.info.width}x{out.info.height}@{out.info.resolution:.3f}"
        self.publish_all(pubs, out)
        st.out_count += len(pubs)
        st.last_out = time.time()

    def republish_cached(self):
        for pubs, st in [(self.map_pubs, self.map_state), (self.risk_pubs, self.risk_state)]:
            if st.msg is not None:
                self.publish_all(pubs, st.msg)
                st.out_count += len(pubs)
                st.last_out = time.time()
        self.publish_status()

    def status(self):
        now = time.time()
        parts = [f"from_domain={self.args.from_domain}", f"to_domain={self.args.to_domain}"]
        for st in [self.map_state, self.risk_state]:
            age_in = -1.0 if st.last_in <= 0 else now - st.last_in
            age_out = -1.0 if st.last_out <= 0 else now - st.last_out
            parts.append(f"{st.name}:in={st.in_count},out={st.out_count},dup={st.dup_count},age_in={age_in:.1f},age_out={age_out:.1f},frame={st.frame},size={st.size},in_topic={st.in_topic},out_topic={st.out_topic}")
        return " | ".join(parts)

    def publish_status(self):
        m = String()
        m.data = self.status()
        for p in self.status_pubs:
            p.publish(m)

    def log_status(self):
        self.n_from.get_logger().info("SCOUT_BRIDGE_STATUS | " + self.status())
        self.publish_status()

    def spin(self):
        ex = SingleThreadedExecutor(context=self.ctx_from)
        ex.add_node(self.n_from)
        stop = {"v": False}
        def on_sig(*_): stop["v"] = True
        signal.signal(signal.SIGINT, on_sig)
        signal.signal(signal.SIGTERM, on_sig)
        try:
            while rclpy.ok(context=self.ctx_from) and not stop["v"]:
                ex.spin_once(timeout_sec=0.1)
        finally:
            ex.remove_node(self.n_from)
            self.n_from.destroy_node()
            self.n_to.destroy_node()
            rclpy.shutdown(context=self.ctx_from)
            rclpy.shutdown(context=self.ctx_to)


def parse_args(argv):
    p = argparse.ArgumentParser(allow_abbrev=False)
    p.add_argument("--from-domain", type=int, required=True)
    p.add_argument("--to-domain", type=int, required=True)
    p.add_argument("--map-in", default="/map")
    p.add_argument("--risk-in", default="/risk/risk_map")
    p.add_argument("--map-out", default="/scout/map")
    p.add_argument("--risk-out", default="/scout/risk_map")
    p.add_argument("--status-out", default="/scout_bridge/status")
    p.add_argument("--rewrite-frame", action="store_true", default=True)
    p.add_argument("--no-rewrite-frame", dest="rewrite_frame", action="store_false")
    p.add_argument("--target-frame", default="scout_map")
    p.add_argument("--source-qos-depth", type=int, default=10)
    p.add_argument("--target-qos-depth", type=int, default=1)
    p.add_argument("--republish-period-sec", type=float, default=1.0)
    p.add_argument("--log-period-sec", type=float, default=2.0)

    cleaned = []
    skip = False
    for tok in argv:
        if tok == "--ros-args":
            skip = True
            continue
        if skip:
            continue
        cleaned.append(tok)
    args, unknown = p.parse_known_args(cleaned)
    if unknown:
        print(f"[scout_map_risk_bridge] ignoring unknown args: {unknown}", file=sys.stderr)
    return args


def main(argv=None):
    args = parse_args(sys.argv[1:] if argv is None else argv)
    Bridge(args).spin()


if __name__ == "__main__":
    main()
