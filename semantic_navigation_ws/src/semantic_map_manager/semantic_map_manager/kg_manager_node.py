#!/usr/bin/env python3
"""Knowledge Graph Manager Node.

Exposes a /store_waypoint service (StoreWaypoint.srv) that persists a
waypoint and its visual features into the knowledge graph.

Modes (set via 'retrieval_mode' parameter):
  siglip_pure  – stores a single Waypoint node with its visual_embedding.
  siglip_yolo  – additionally stores each detected object as an Object node
                 and creates a CONTAINS edge from the Waypoint to each Object.

The node also subscribes to /trigger_capture to support the original manual
capture workflow: on trigger it calls /get_visual_features + reads the robot
pose from TF2, then calls its own /store_waypoint handler internally.

Parameters
----------
retrieval_mode : str - "siglip_pure" or "siglip_yolo"
"""

import uuid

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration

from std_msgs.msg import Empty
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseStamped

from tf2_ros import Buffer, TransformListener, LookupException, ConnectivityException, ExtrapolationException

from semantic_interfaces.srv import GetVisualFeatures, StoreWaypoint
from semantic_map_manager.knowledge_graph_client import KnowledgeGraphClient
from semantic_map_manager.utils import create_graph_manager_instance


class KGManagerNode(Node):
    """ROS 2 node that bridges visual features into the knowledge graph."""

    def __init__(self) -> None:
        super().__init__("kg_manager")

        self.declare_parameter("retrieval_mode", "siglip_yolo")
        self._mode = self.get_parameter("retrieval_mode").get_parameter_value().string_value

        # Knowledge Graph access
        self._kg_client = KnowledgeGraphClient(self)
        self._graph = create_graph_manager_instance()

        # TF2 for manual capture trigger
        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)

        # Latest image buffer (used on manual trigger)
        self._latest_image: Image | None = None
        self._image_sub = self.create_subscription(
            Image, "/camera/image_raw", self._image_callback, 10
        )

        # Manual trigger subscriber (backward-compatible with waypoint_capture)
        self._trigger_sub = self.create_subscription(
            Empty, "/trigger_capture", self._trigger_callback, 10
        )

        # Visual encoder client
        self._encoder_client = self.create_client(GetVisualFeatures, "get_visual_features")

        # StoreWaypoint service (also used by orchestrator if needed)
        self._store_srv = self.create_service(
            StoreWaypoint, "store_waypoint", self._handle_store_waypoint
        )

        self.get_logger().info(f"KG manager ready (mode={self._mode}).")

    # ------------------------------------------------------------------
    # Image buffer
    # ------------------------------------------------------------------

    def _image_callback(self, msg: Image) -> None:
        self._latest_image = msg

    # ------------------------------------------------------------------
    # Manual capture trigger
    # ------------------------------------------------------------------

    def _trigger_callback(self, _: Empty) -> None:
        self.get_logger().info("Capture triggered.")

        if self._latest_image is None:
            self.get_logger().warn("No image yet – skipping capture.")
            return

        pose = self._get_robot_pose()
        if pose is None:
            self.get_logger().warn("TF2 lookup failed – skipping capture.")
            return

        features = self._call_encoder(self._latest_image)
        if features is None:
            self.get_logger().warn("Visual encoder failed – skipping capture.")
            return

        node_id = str(uuid.uuid4())
        req = StoreWaypoint.Request()
        req.node_id = node_id
        req.pose = pose
        req.visual_embedding = features["embedding"]
        req.detected_objects = features["objects"]

        resp = StoreWaypoint.Response()
        self._handle_store_waypoint(req, resp)
        if resp.success:
            self.get_logger().info(
                f"Waypoint {node_id} stored at "
                f"({pose.pose.position.x:.2f}, {pose.pose.position.y:.2f})."
            )
        else:
            self.get_logger().error(f"Store failed: {resp.message}")

    # ------------------------------------------------------------------
    # StoreWaypoint service handler
    # ------------------------------------------------------------------

    def _handle_store_waypoint(
        self,
        request: StoreWaypoint.Request,
        response: StoreWaypoint.Response,
    ) -> StoreWaypoint.Response:
        try:
            pose = request.pose
            attrs = {
                "id": request.node_id,
                "pose_x": str(pose.pose.position.x),
                "pose_y": str(pose.pose.position.y),
                "pose_z": str(pose.pose.position.z),
                "orient_x": str(pose.pose.orientation.x),
                "orient_y": str(pose.pose.orientation.y),
                "orient_z": str(pose.pose.orientation.z),
                "orient_w": str(pose.pose.orientation.w),
                "embedding": ",".join(f"{v:.6f}" for v in request.visual_embedding),
            }

            self._kg_client.add_node(
                node_id=request.node_id, class_id="waypoint", attributes=attrs
            )

            if self._mode == "siglip_yolo":
                self._store_objects_and_edges(request.node_id, request.detected_objects)

            response.success = True
            response.message = "OK"
        except Exception as exc:  # noqa: BLE001
            self.get_logger().error(f"StoreWaypoint error: {exc}")
            response.success = False
            response.message = str(exc)

        return response

    def _store_objects_and_edges(self, waypoint_id: str, detected_objects: list[str]) -> None:
        for label in detected_objects:
            obj_id = f"{waypoint_id}_{label}"
            self._kg_client.add_node(
                node_id=obj_id,
                class_id="object",
                attributes={"label": label, "source_waypoint": waypoint_id},
            )
            try:
                edge = self._graph.create_edge("CONTAINS", waypoint_id, obj_id)
                self._graph.update_edge(edge)
            except Exception as exc:  # noqa: BLE001
                self.get_logger().warn(f"Could not create CONTAINS edge for {obj_id}: {exc}")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_robot_pose(self) -> PoseStamped | None:
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

    def _call_encoder(self, image: Image) -> dict | None:
        if not self._encoder_client.wait_for_service(timeout_sec=5.0):
            self.get_logger().error("get_visual_features service not available.")
            return None

        req = GetVisualFeatures.Request()
        req.image = image
        future = self._encoder_client.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=15.0)

        if future.result() is None:
            self.get_logger().error("get_visual_features timed out.")
            return None

        result: GetVisualFeatures.Response = future.result()
        if not result.success:
            self.get_logger().error(f"Encoder error: {result.message}")
            return None

        return {"embedding": list(result.visual_embedding), "objects": list(result.detected_objects)}


def main(args=None) -> None:
    rclpy.init(args=args)
    node = KGManagerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
