#!/usr/bin/env python3
"""Manual Waypoint Capturer Node.

Listens on the ``/trigger_capture`` topic (std_msgs/Empty). When a message is
received the node:

1. Reads the latest frame from ``/camera/image_raw``.
2. Looks up the robot pose ``map → base_link`` via tf2.
3. Calls the ``get_embedding`` service (SigLIP) to embed the image.
4. Persists a new node in the Knowledge Graph with the pose, embedding, and a
   unique UUID.
"""

import uuid

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration

from std_msgs.msg import Empty
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseStamped

from tf2_ros import Buffer, TransformListener, LookupException, ConnectivityException, ExtrapolationException

from semantic_map_manager_interfaces.srv import GetEmbedding
from knowledge_graph.knowledge_graph_client import KnowledgeGraphClient


class WaypointCaptureNode(Node):
    """ROS 2 node that captures semantic waypoints on demand."""

    def __init__(self) -> None:
        super().__init__("waypoint_capturer")

        # --- tf2 ---
        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)

        # --- latest camera frame (kept in memory) ---
        self._latest_image: Image | None = None
        self._image_sub = self.create_subscription(
            Image, "/camera/image_raw", self._image_callback, 10
        )

        # --- trigger subscriber ---
        self._trigger_sub = self.create_subscription(
            Empty, "/trigger_capture", self._trigger_callback, 10
        )

        # --- SigLIP service client ---
        self._embedding_client = self.create_client(GetEmbedding, "get_embedding")

        # --- Knowledge Graph client ---
        self._kg_client = KnowledgeGraphClient(self)

        self.get_logger().info("Waypoint capturer node ready. Listening on /trigger_capture.")

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------

    def _image_callback(self, msg: Image) -> None:
        self._latest_image = msg

    def _trigger_callback(self, _: Empty) -> None:
        self.get_logger().info("Capture triggered.")

        # 1. Check we have an image
        if self._latest_image is None:
            self.get_logger().warn("No image received yet – skipping capture.")
            return

        # 2. Get robot pose
        pose = self._get_robot_pose()
        if pose is None:
            self.get_logger().warn("Could not obtain robot pose – skipping capture.")
            return

        # 3. Get SigLIP embedding
        embedding = self._get_image_embedding(self._latest_image)
        if embedding is None:
            self.get_logger().warn("Embedding service failed – skipping capture.")
            return

        # 4. Persist node in the Knowledge Graph
        node_id = str(uuid.uuid4())
        attributes = {
            "id": node_id,
            "pose_x": str(pose.pose.position.x),
            "pose_y": str(pose.pose.position.y),
            "pose_z": str(pose.pose.position.z),
            "orient_x": str(pose.pose.orientation.x),
            "orient_y": str(pose.pose.orientation.y),
            "orient_z": str(pose.pose.orientation.z),
            "orient_w": str(pose.pose.orientation.w),
            "embedding": ",".join(f"{v:.6f}" for v in embedding),
        }

        try:
            self._kg_client.add_node(node_id=node_id, class_id="waypoint", attributes=attributes)
            self.get_logger().info(f"Saved waypoint {node_id} at ({pose.pose.position.x:.2f}, {pose.pose.position.y:.2f}).")
        except Exception as exc:  # noqa: BLE001
            self.get_logger().error(f"Failed to save waypoint to knowledge graph: {exc}")

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_robot_pose(self) -> PoseStamped | None:
        """Look up the map → base_link transform and return it as a PoseStamped."""
        try:
            transform = self._tf_buffer.lookup_transform(
                "map", "base_link", rclpy.time.Time(), timeout=Duration(seconds=1.0)
            )
            pose = PoseStamped()
            pose.header = transform.header
            pose.pose.position.x = transform.transform.translation.x
            pose.pose.position.y = transform.transform.translation.y
            pose.pose.position.z = transform.transform.translation.z
            pose.pose.orientation = transform.transform.rotation
            return pose
        except (LookupException, ConnectivityException, ExtrapolationException) as exc:
            self.get_logger().error(f"TF lookup failed: {exc}")
            return None

    def _get_image_embedding(self, image: Image) -> list[float] | None:
        """Synchronously call the get_embedding service and return the vector."""
        if not self._embedding_client.wait_for_service(timeout_sec=5.0):
            self.get_logger().error("get_embedding service not available.")
            return None

        request = GetEmbedding.Request()
        request.image = image
        request.use_image = True

        future = self._embedding_client.call_async(request)
        rclpy.spin_until_future_complete(self, future, timeout_sec=10.0)

        if future.result() is None:
            self.get_logger().error("get_embedding service call timed out.")
            return None

        result: GetEmbedding.Response = future.result()
        if not result.success:
            self.get_logger().error(f"Embedding service error: {result.message}")
            return None

        return list(result.embedding)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = WaypointCaptureNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
