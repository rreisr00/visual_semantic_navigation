#!/usr/bin/env python3
"""Optional odometry-driven trigger for the existing capture action."""
from __future__ import annotations

import time
from math import atan2, pi

import rclpy
from nav_msgs.msg import Odometry
from rclpy.action import ActionClient
from rclpy.node import Node

from semantic_interfaces.action import CaptureWaypoint
from semantic_navigation_core.topology import NodeCreationPolicy


class TopologyMapperNode(Node):
    """Apply a configurable motion policy and request multiview captures."""

    def __init__(self) -> None:
        super().__init__("topology_mapper")
        self.declare_parameter("odom_topic", "/odom")
        self.declare_parameter("capture_action_name", "capture_waypoint")
        self.declare_parameter("scene_id", "aws_small_house")
        self.declare_parameter("minimum_translation_m", 1.5)
        self.declare_parameter("minimum_rotation_deg", 45.0)
        self.declare_parameter("minimum_time_s", 2.0)
        self.declare_parameter("duplicate_distance_m", 0.5)
        self.declare_parameter("maximum_edge_distance_m", 4.0)
        self.declare_parameter("capture_views_deg", [0.0, 90.0, 180.0, 270.0])
        self.declare_parameter("rotate_robot", True)
        self.declare_parameter("retry_period_s", 5.0)

        self._policy = NodeCreationPolicy(
            minimum_translation_m=float(
                self.get_parameter("minimum_translation_m").value
            ),
            minimum_rotation_rad=float(
                self.get_parameter("minimum_rotation_deg").value
            ) * pi / 180.0,
            minimum_time_s=float(self.get_parameter("minimum_time_s").value),
            duplicate_distance_m=float(
                self.get_parameter("duplicate_distance_m").value
            ),
            maximum_edge_distance_m=float(
                self.get_parameter("maximum_edge_distance_m").value
            ),
        )
        self._client = ActionClient(
            self,
            CaptureWaypoint,
            self.get_parameter("capture_action_name").value,
        )
        self._latest = None
        self._previous = None
        self._pending = False
        self._last_attempt = 0.0
        self.create_subscription(
            Odometry,
            self.get_parameter("odom_topic").value,
            self._on_odometry,
            10,
        )

    def _on_odometry(self, message: Odometry) -> None:
        pose = message.pose.pose
        sample = (
            (float(pose.position.x), float(pose.position.y)),
            _yaw(pose.orientation),
            _stamp_seconds(message.header.stamp),
        )
        self._latest = sample
        if self._pending:
            return
        if time.monotonic() - self._last_attempt < float(
            self.get_parameter("retry_period_s").value
        ):
            return
        previous_position = self._previous[0] if self._previous else None
        previous_yaw = self._previous[1] if self._previous else None
        previous_stamp = self._previous[2] if self._previous else None
        if not self._policy.should_create(
            previous_position,
            previous_yaw,
            previous_stamp,
            sample[0],
            sample[1],
            sample[2],
        ):
            return
        self._dispatch_capture()

    def _dispatch_capture(self) -> None:
        self._last_attempt = time.monotonic()
        if not self._client.server_is_ready():
            self.get_logger().warn(
                "Capture action is not ready; automatic mapping will retry",
                throttle_duration_sec=5.0,
            )
            return
        goal = CaptureWaypoint.Goal()
        goal.scene_id = self.get_parameter("scene_id").value
        goal.relative_view_yaws_deg = [
            float(value)
            for value in self.get_parameter("capture_views_deg").value
        ]
        goal.rotate_robot = bool(self.get_parameter("rotate_robot").value)
        self._pending = True
        future = self._client.send_goal_async(goal)
        future.add_done_callback(self._on_goal_response)

    def _on_goal_response(self, future) -> None:
        try:
            handle = future.result()
        except Exception as exc:  # noqa: BLE001
            self.get_logger().error(f"Automatic capture request failed: {exc}")
            self._pending = False
            return
        if not handle.accepted:
            self.get_logger().warn("Automatic capture goal was rejected")
            self._pending = False
            return
        handle.get_result_async().add_done_callback(self._on_capture_result)

    def _on_capture_result(self, future) -> None:
        try:
            result = future.result().result
        except Exception as exc:  # noqa: BLE001
            self.get_logger().error(f"Automatic capture result failed: {exc}")
            self._pending = False
            return
        if result.success and self._latest is not None:
            self._previous = self._latest
            self.get_logger().info(
                f"Automatic waypoint '{result.node_id}' captured with "
                f"{result.captured_views} view(s)"
            )
        else:
            self.get_logger().warn(f"Automatic capture failed: {result.message}")
        self._pending = False


def _stamp_seconds(stamp) -> float:
    value = float(stamp.sec) + float(stamp.nanosec) * 1e-9
    return value if value > 0.0 else time.monotonic()


def _yaw(quaternion) -> float:
    return atan2(
        2.0 * (quaternion.w * quaternion.z + quaternion.x * quaternion.y),
        1.0 - 2.0 * (quaternion.y * quaternion.y + quaternion.z * quaternion.z),
    )


def main(args=None) -> None:
    rclpy.init(args=args)
    node = TopologyMapperNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
