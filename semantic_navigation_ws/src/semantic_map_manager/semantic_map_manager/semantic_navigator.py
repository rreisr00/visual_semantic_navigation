#!/usr/bin/env python3
"""Semantic Navigator Node.

Accepts a text query on the ``/navigate_to_semantic_target`` topic
(std_msgs/String, e.g. "Find the sofa") and:

1. Embeds the query text via the ``get_embedding`` service (SigLIP 2).
2. Retrieves all waypoint nodes from the Knowledge Graph.
3. Computes cosine similarity between the query embedding and each stored
   image embedding.
4. Sends the robot to the best-matching waypoint using Nav2's
   ``nav2_simple_commander`` API.
"""

import numpy as np

import rclpy
from rclpy.node import Node

from std_msgs.msg import String
from geometry_msgs.msg import PoseStamped, Quaternion

from nav2_simple_commander.robot_navigator import BasicNavigator, TaskResult

from semantic_map_manager_interfaces.srv import GetEmbedding
from knowledge_graph.knowledge_graph_client import KnowledgeGraphClient


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Return the cosine similarity between two 1-D arrays."""
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-8
    return float(np.dot(a, b) / denom)


class SemanticNavigatorNode(Node):
    """ROS 2 node that navigates to the most semantically similar waypoint."""

    def __init__(self) -> None:
        super().__init__("semantic_navigator")

        # --- Nav2 simple commander ---
        self._navigator = BasicNavigator()

        # --- SigLIP client ---
        self._embedding_client = self.create_client(GetEmbedding, "get_embedding")

        # --- Knowledge Graph client ---
        self._kg_client = KnowledgeGraphClient(self)

        # --- Text query subscriber ---
        self._query_sub = self.create_subscription(
            String, "/navigate_to_semantic_target", self._query_callback, 10
        )

        self.get_logger().info(
            "Semantic navigator ready. Publish a query to /navigate_to_semantic_target."
        )

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------

    def _query_callback(self, msg: String) -> None:
        query = msg.data.strip()
        if not query:
            return
        self.get_logger().info(f"Received navigation query: '{query}'")
        self._navigate_to_query(query)

    # ------------------------------------------------------------------
    # Core logic
    # ------------------------------------------------------------------

    def _navigate_to_query(self, query: str) -> None:
        # 1. Embed the query text
        query_embedding = self._get_text_embedding(query)
        if query_embedding is None:
            self.get_logger().error("Could not embed query – aborting navigation.")
            return

        # 2. Retrieve all waypoint nodes from the knowledge graph
        try:
            nodes = self._kg_client.get_nodes(class_id="waypoint")
        except Exception as exc:  # noqa: BLE001
            self.get_logger().error(f"Failed to retrieve nodes from knowledge graph: {exc}")
            return

        if not nodes:
            self.get_logger().warn("Knowledge graph is empty – no waypoints available.")
            return

        # 3. Rank nodes by cosine similarity
        best_node = None
        best_score = float("-inf")  # cosine similarity range is [-1, 1]

        for node in nodes:
            attrs = node.get("attributes", {})
            raw_embedding = attrs.get("embedding", "")
            if not raw_embedding:
                continue
            try:
                node_embedding = np.array([float(v) for v in raw_embedding.split(",")], dtype=np.float32)
            except ValueError:
                continue

            score = _cosine_similarity(query_embedding, node_embedding)
            if score > best_score:
                best_score = score
                best_node = attrs

        if best_node is None:
            self.get_logger().warn("No valid waypoints found in the knowledge graph.")
            return

        self.get_logger().info(
            f"Best match: id={best_node.get('id')} score={best_score:.4f}"
        )

        # 4. Build PoseStamped and navigate
        goal_pose = self._build_pose(best_node)
        self._send_nav_goal(goal_pose)

    def _build_pose(self, attrs: dict) -> PoseStamped:
        """Reconstruct a PoseStamped from the node attribute dictionary."""
        pose = PoseStamped()
        pose.header.frame_id = "map"
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.pose.position.x = float(attrs.get("pose_x", 0.0))
        pose.pose.position.y = float(attrs.get("pose_y", 0.0))
        pose.pose.position.z = float(attrs.get("pose_z", 0.0))
        pose.pose.orientation = Quaternion(
            x=float(attrs.get("orient_x", 0.0)),
            y=float(attrs.get("orient_y", 0.0)),
            z=float(attrs.get("orient_z", 0.0)),
            w=float(attrs.get("orient_w", 1.0)),
        )
        return pose

    def _send_nav_goal(self, goal_pose: PoseStamped) -> None:
        """Send a navigation goal to Nav2 and wait for completion."""
        self.get_logger().info(
            f"Navigating to ({goal_pose.pose.position.x:.2f}, {goal_pose.pose.position.y:.2f}) …"
        )
        self._navigator.goToPose(goal_pose)

        while not self._navigator.isTaskComplete():
            feedback = self._navigator.getFeedback()
            if feedback:
                self.get_logger().info(
                    f"Distance remaining: {feedback.distance_remaining:.2f} m",
                    throttle_duration_sec=2.0,
                )

        result = self._navigator.getResult()
        if result == TaskResult.SUCCEEDED:
            self.get_logger().info("Navigation succeeded.")
        elif result == TaskResult.CANCELED:
            self.get_logger().warn("Navigation was canceled.")
        else:
            self.get_logger().error(f"Navigation failed with result: {result}")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_text_embedding(self, text: str) -> np.ndarray | None:
        """Call the get_embedding service for a text query."""
        if not self._embedding_client.wait_for_service(timeout_sec=5.0):
            self.get_logger().error("get_embedding service not available.")
            return None

        request = GetEmbedding.Request()
        request.text = text
        request.use_image = False

        future = self._embedding_client.call_async(request)
        rclpy.spin_until_future_complete(self, future, timeout_sec=10.0)

        if future.result() is None:
            self.get_logger().error("get_embedding service call timed out.")
            return None

        result: GetEmbedding.Response = future.result()
        if not result.success:
            self.get_logger().error(f"Embedding service error: {result.message}")
            return None

        return np.array(result.embedding, dtype=np.float32)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = SemanticNavigatorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
