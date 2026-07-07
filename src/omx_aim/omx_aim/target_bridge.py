#!/usr/bin/env python3
"""Target Bridge - 표적 좌표 forward 노드 (단계 D 이후).

yolo_node 가 직접 TF 변환을 하므로, bridge 는 단순 forward 만.
    
역할:
    1. /target_in_map 또는 /clicked_point  -> /omx/target_in_map  (HIGH)
    2. /patrol_in_map                       -> /omx/patrol_in_map  (NORMAL)

토픽:
    입력:
        /target_in_map    PointStamped  외부 긴급 표적
        /patrol_in_map    PointStamped  외부 정찰 좌표
        /clicked_point    PointStamped  RViz 'Publish Point'
    출력:
        /omx/target_in_map  PointStamped  yolo_node (HIGH)
        /omx/patrol_in_map  PointStamped  yolo_node (NORMAL)
        /bridge/status      String        디버그

RViz 사용:
    P 키 + 맵 클릭 → /clicked_point → /omx/target_in_map (긴급)
    
    RViz 클릭은 z=0. default_target_z 파라미터로 override 가능.

파라미터:
    default_target_z (m): RViz 클릭 시 z override (0이면 그대로)
    map_frame:            map frame 이름
"""

from __future__ import annotations

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import PointStamped
from std_msgs.msg import String


class TargetBridge(Node):

    def __init__(self):
        super().__init__('target_bridge')

        self.declare_parameter('default_target_z', 0.0)
        self.declare_parameter('map_frame', 'map')
        
        self.default_target_z = self.get_parameter('default_target_z').value
        self.map_frame = self.get_parameter('map_frame').value

        self.get_logger().info("=" * 50)
        self.get_logger().info("Target Bridge (forward only)")
        self.get_logger().info("=" * 50)
        self.get_logger().info(f"Map frame: {self.map_frame}")
        if self.default_target_z != 0.0:
            self.get_logger().info(
                f"RViz click z override: {self.default_target_z:+.3f} m")

        # Subscribers
        self.create_subscription(
            PointStamped, '/target_in_map',
            self.on_target_in_map, 10)
        self.create_subscription(
            PointStamped, '/patrol_in_map',
            self.on_patrol_in_map, 10)
        self.create_subscription(
            PointStamped, '/clicked_point',
            self.on_clicked_point, 10)

        # Publishers
        self.pub_target = self.create_publisher(
            PointStamped, '/omx/target_in_map', 10)
        self.pub_patrol = self.create_publisher(
            PointStamped, '/omx/patrol_in_map', 10)
        self.pub_status = self.create_publisher(
            String, '/bridge/status', 10)

        self.target_count = 0
        self.patrol_count = 0
        self.click_count = 0

        self.create_timer(1.0, self.publish_status_periodic)

        self.get_logger().info("입력:")
        self.get_logger().info("  /target_in_map  PointStamped  긴급")
        self.get_logger().info("  /patrol_in_map  PointStamped  정찰")
        self.get_logger().info("  /clicked_point  PointStamped  RViz")
        self.get_logger().info("출력:")
        self.get_logger().info("  /omx/target_in_map  (HIGH)")
        self.get_logger().info("  /omx/patrol_in_map  (NORMAL)")
        self.get_logger().info("=" * 50)
        self.get_logger().info("=== Bridge ready ===")

    def _fix_header(self, msg: PointStamped):
        if not msg.header.frame_id:
            msg.header.frame_id = self.map_frame
        elif msg.header.frame_id != self.map_frame:
            self.get_logger().warn(
                f"표적 frame='{msg.header.frame_id}', "
                f"'{self.map_frame}' 으로 가정")
            msg.header.frame_id = self.map_frame

    def on_target_in_map(self, msg: PointStamped):
        self.target_count += 1
        self._fix_header(msg)
        self.get_logger().info(
            f"[#T{self.target_count}] 긴급 표적 (map): "
            f"({msg.point.x:+.3f}, {msg.point.y:+.3f}, {msg.point.z:+.3f})")
        self.pub_target.publish(msg)

    def on_patrol_in_map(self, msg: PointStamped):
        self.patrol_count += 1
        self._fix_header(msg)
        self.get_logger().info(
            f"[#P{self.patrol_count}] 정찰 좌표 (map): "
            f"({msg.point.x:+.3f}, {msg.point.y:+.3f}, {msg.point.z:+.3f})")
        self.pub_patrol.publish(msg)

    def on_clicked_point(self, msg: PointStamped):
        self.click_count += 1
        self._fix_header(msg)
        
        if self.default_target_z != 0.0:
            self.get_logger().info(
                f"RViz z={msg.point.z:.3f} -> override "
                f"{self.default_target_z:.3f}")
            msg.point.z = self.default_target_z
        
        self.get_logger().info(
            f"[#C{self.click_count}] RViz 클릭 -> 긴급 표적: "
            f"({msg.point.x:+.3f}, {msg.point.y:+.3f}, {msg.point.z:+.3f})")
        
        self.target_count += 1
        self.pub_target.publish(msg)

    def publish_status_periodic(self):
        msg = String()
        msg.data = (f"forward: target={self.target_count}, "
                    f"patrol={self.patrol_count}, "
                    f"clicks={self.click_count}")
        self.pub_status.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = TargetBridge()
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("\n중단됨.")
    except Exception as e:
        print(f"노드 에러: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()