#!/usr/bin/env python3
"""Route RViz /clicked_point goals to the currently selected robot."""

from typing import Dict, List

from geometry_msgs.msg import PointStamped
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from visualization_msgs.msg import Marker


class GoalDispatcher(Node):
    """Forward RViz clicked goals to the selected robot namespace."""

    def __init__(self) -> None:
        super().__init__('goal_dispatcher')
        self.declare_parameter('robots', ['burger1', 'waffle1'])
        self.declare_parameter('default_robot', 'burger1')
        robots_param = self.get_parameter('robots').value
        self.robots: List[str] = list(robots_param)
        self.selected_robot = str(self.get_parameter('default_robot').value)
        if self.selected_robot not in self.robots and self.robots:
            self.selected_robot = self.robots[0]

        self.goal_pubs: Dict[str, object] = {
            robot: self.create_publisher(PointStamped, f'/{robot}/goal_point', 10)
            for robot in self.robots
        }
        self.selected_pub = self.create_publisher(String, '/selected_robot', 10)
        self.marker_pub = self.create_publisher(Marker, '/multi/goal_markers', 10)

        self.create_subscription(String, '/target_robot', self.on_target_robot, 10)
        self.create_subscription(PointStamped, '/clicked_point', self.on_clicked_point, 10)
        self.create_timer(1.0, self.publish_selected_robot)

        self.get_logger().info(
            'GoalDispatcher started. Valid robot names: '
            + ', '.join(self.robots)
            + f'. Current target: {self.selected_robot}'
        )

    def on_target_robot(self, msg: String) -> None:
        name = msg.data.strip()
        if name not in self.robots:
            self.get_logger().warn(
                f'Invalid target_robot "{name}". Use one of: {", ".join(self.robots)}'
            )
            return
        self.selected_robot = name
        self.publish_selected_robot()
        self.get_logger().info(
            f'RViz clicked goals will now be sent to: {self.selected_robot}'
        )

    def on_clicked_point(self, msg: PointStamped) -> None:
        if self.selected_robot not in self.goal_pubs:
            self.get_logger().warn(f'No publisher for selected robot {self.selected_robot}')
            return
        goal = PointStamped()
        goal.header.stamp = self.get_clock().now().to_msg()
        goal.header.frame_id = msg.header.frame_id or 'map'
        goal.point.x = msg.point.x
        goal.point.y = msg.point.y
        goal.point.z = 0.0
        self.goal_pubs[self.selected_robot].publish(goal)
        self.publish_goal_marker(self.selected_robot, goal)
        self.get_logger().info(
            f'Goal sent to {self.selected_robot}: x={goal.point.x:.2f}, y={goal.point.y:.2f}'
        )

    def publish_selected_robot(self) -> None:
        msg = String()
        msg.data = self.selected_robot
        self.selected_pub.publish(msg)

    def publish_goal_marker(self, robot: str, goal: PointStamped) -> None:
        marker = Marker()
        marker.header.frame_id = 'map'
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = 'multi_goals'
        marker.id = self.robots.index(robot) if robot in self.robots else 99
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose.position.x = goal.point.x
        marker.pose.position.y = goal.point.y
        marker.pose.position.z = 0.05
        marker.pose.orientation.w = 1.0
        marker.scale.x = 0.22
        marker.scale.y = 0.22
        marker.scale.z = 0.22
        marker.color.a = 1.0
        if robot == 'burger1':
            marker.color.r = 1.0
            marker.color.g = 0.1
            marker.color.b = 0.1
        else:
            marker.color.r = 0.1
            marker.color.g = 0.3
            marker.color.b = 1.0
        self.marker_pub.publish(marker)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = GoalDispatcher()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
