#!/usr/bin/env python3
"""Semantic Orchestrator Node.

Coordinates the full retrieval pipeline:
  1. Receives a navigation goal as text (/nav_goal_text) or image (/nav_goal_image).
  2. Obtains a query embedding (and optional object list) from the encoder.
  3. Queries the knowledge graph for all stored waypoints.
  4. Ranks waypoints using cosine similarity (siglip_pure) or a hybrid
     cosine + Jaccard score (siglip_yolo).
  5. Sends the robot to the best matching waypoint via Nav2.
  6. Publishes total retrieval latency to /retrieval_latency and the winning
     node ID to /retrieval_result.

Parameters
----------
retrieval_mode        : str   - "siglip_pure" or "siglip_yolo"
hybrid_embedding_weight: float - cosine similarity weight (siglip_yolo, default 0.7)
hybrid_object_weight  : float - Jaccard similarity weight (siglip_yolo, default 0.3)
"""

import time

import numpy as np
import rclpy
from rclpy.node import Node

from std_msgs.msg import String, Float64
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseStamped, Quaternion

from nav2_simple_commander.robot_navigator import BasicNavigator, TaskResult

from semantic_interfaces.srv import GetEmbedding, GetVisualFeatures
from semantic_map_manager.knowledge_graph_client import KnowledgeGraphClient
from semantic_map_manager.utils import cosine_similarity


class SemanticOrchestratorNode(Node):
    """ROS 2 node that orchestrates semantic retrieval and navigation."""

    def __init__(self) -> None:
        super().__init__("semantic_orchestrator")

        self.declare_parameter("retrieval_mode", "siglip_yolo")
        self.declare_parameter("hybrid_embedding_weight", 0.7)
        self.declare_parameter("hybrid_object_weight", 0.3)

        self._mode = self.get_parameter("retrieval_mode").get_parameter_value().string_value
        self._embed_w = (
            self.get_parameter("hybrid_embedding_weight").get_parameter_value().double_value
        )
        self._obj_w = (
            self.get_parameter("hybrid_object_weight").get_parameter_value().double_value
        )

        # Nav2
        self._navigator = BasicNavigator()

        # Service clients
        self._text_embed_client = self.create_client(GetEmbedding, "get_embedding")
        self._visual_feat_client = self.create_client(GetVisualFeatures, "get_visual_features")

        # Knowledge Graph
        self._kg_client = KnowledgeGraphClient(self)

        # Input subscribers
        self._text_sub = self.create_subscription(
            String, "/nav_goal_text", self._text_goal_callback, 10
        )
        self._image_sub = self.create_subscription(
            Image, "/nav_goal_image", self._image_goal_callback, 10
        )

        # Output publishers
        self._latency_pub = self.create_publisher(Float64, "/retrieval_latency", 10)
        self._result_pub = self.create_publisher(String, "/retrieval_result", 10)

        self.get_logger().info(f"Semantic orchestrator ready (mode={self._mode}).")

    # ------------------------------------------------------------------
    # Goal callbacks
    # ------------------------------------------------------------------

    def _text_goal_callback(self, msg: String) -> None:
        query = msg.data.strip()
        if not query:
            return
        self.get_logger().info(f"Text goal: '{query}'")
        t0 = time.perf_counter()

        embedding = self._get_text_embedding(query)
        if embedding is None:
            self.get_logger().error("Text embedding failed.")
            return

        self._execute_retrieval(embedding, query_objects=[], t_start=t0)

    def _image_goal_callback(self, msg: Image) -> None:
        self.get_logger().info("Image goal received.")
        t0 = time.perf_counter()

        features = self._get_visual_features(msg)
        if features is None:
            self.get_logger().error("Visual feature extraction failed.")
            return

        embedding = np.array(features["embedding"], dtype=np.float32)
        self._execute_retrieval(embedding, query_objects=features["objects"], t_start=t0)

    # ------------------------------------------------------------------
    # Core retrieval pipeline
    # ------------------------------------------------------------------

    def _execute_retrieval(
        self,
        query_embedding: np.ndarray,
        query_objects: list[str],
        t_start: float,
    ) -> None:
        try:
            waypoints = self._kg_client.get_nodes(class_id="waypoint")
        except Exception as exc:  # noqa: BLE001
            self.get_logger().error(f"KG query failed: {exc}")
            return

        if not waypoints:
            self.get_logger().warn("No waypoints in knowledge graph.")
            return

        best_node_attrs, best_node_id = self._rank_waypoints(
            query_embedding, query_objects, waypoints
        )

        if best_node_attrs is None:
            self.get_logger().warn("No valid waypoints found.")
            return

        total_latency = time.perf_counter() - t_start
        self.get_logger().info(f"Best waypoint: {best_node_id} (latency={total_latency:.3f}s)")

        lat_msg = Float64()
        lat_msg.data = total_latency
        self._latency_pub.publish(lat_msg)

        result_msg = String()
        result_msg.data = best_node_id
        self._result_pub.publish(result_msg)

        goal_pose = self._build_pose(best_node_attrs)
        self._send_nav_goal(goal_pose)

    def _rank_waypoints(
        self,
        query_embedding: np.ndarray,
        query_objects: list[str],
        waypoints: list[dict],
    ) -> tuple[dict | None, str]:
        best_attrs = None
        best_id = ""
        best_score = float("-inf")

        query_obj_set = set(query_objects)

        for wp in waypoints:
            attrs = wp.get("attributes", {})
            raw_emb = attrs.get("embedding", "")
            if not raw_emb:
                continue
            try:
                wp_embedding = np.array(
                    [float(v) for v in raw_emb.split(",")], dtype=np.float32
                )
            except ValueError:
                continue

            cos_score = cosine_similarity(query_embedding, wp_embedding)

            if self._mode == "siglip_yolo" and query_obj_set:
                jaccard = self._jaccard(query_obj_set, attrs.get("detected_objects", ""))
                score = self._embed_w * cos_score + self._obj_w * jaccard
            else:
                score = cos_score

            if score > best_score:
                best_score = score
                best_attrs = attrs
                best_id = wp.get("node_id", attrs.get("id", ""))

        return best_attrs, best_id

    # ------------------------------------------------------------------
    # Navigation
    # ------------------------------------------------------------------

    def _build_pose(self, attrs: dict) -> PoseStamped:
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
        self.get_logger().info(
            f"Navigating to ({goal_pose.pose.position.x:.2f}, "
            f"{goal_pose.pose.position.y:.2f}) ..."
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
            self.get_logger().error(f"Navigation failed: {result}")

    # ------------------------------------------------------------------
    # Service call helpers
    # ------------------------------------------------------------------

    def _get_text_embedding(self, text: str) -> np.ndarray | None:
        if not self._text_embed_client.wait_for_service(timeout_sec=5.0):
            self.get_logger().error("get_embedding service not available.")
            return None
        req = GetEmbedding.Request()
        req.text = text
        req.use_image = False
        future = self._text_embed_client.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=10.0)
        if future.result() is None or not future.result().success:
            return None
        return np.array(future.result().embedding, dtype=np.float32)

    def _get_visual_features(self, image: Image) -> dict | None:
        if not self._visual_feat_client.wait_for_service(timeout_sec=5.0):
            self.get_logger().error("get_visual_features service not available.")
            return None
        req = GetVisualFeatures.Request()
        req.image = image
        future = self._visual_feat_client.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=15.0)
        if future.result() is None or not future.result().success:
            return None
        res = future.result()
        return {"embedding": list(res.visual_embedding), "objects": list(res.detected_objects)}

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _jaccard(set_a: set[str], stored_objects_csv: str) -> float:
        if not stored_objects_csv:
            return 0.0
        set_b = set(stored_objects_csv.split(","))
        intersection = len(set_a & set_b)
        union = len(set_a | set_b)
        return intersection / union if union > 0 else 0.0


def main(args=None) -> None:
    rclpy.init(args=args)
    node = SemanticOrchestratorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
