"""ROS 2 Lifecycle visual encoder node.

Wraps ``semantic_vision_core.SemanticVisionPipeline`` as a stateless service
server.  The node never subscribes to camera topics; callers pass a
``sensor_msgs/Image`` directly in the ``GetVisualFeatures`` service request.

Lifecycle transitions
---------------------
configure   Read parameters → instantiate SemanticVisionPipeline (loads ML models).
activate    Create ``/get_visual_features`` service + latency publisher.
deactivate  Destroy service + publisher.
cleanup     Release pipeline and cv_bridge.
shutdown    No-op.
"""
from __future__ import annotations

import time
from typing import Optional

import rclpy
from rclpy.lifecycle import LifecycleNode, TransitionCallbackReturn
from std_msgs.msg import Float64

from semantic_interfaces.srv import GetVisualFeatures

try:
    from cv_bridge import CvBridge

    _CV_BRIDGE_OK = True
except ImportError:
    _CV_BRIDGE_OK = False


class VisualEncoderNode(LifecycleNode):
    """Stateless lifecycle node: SigLIP embeddings + optional YOLO detections."""

    def __init__(self) -> None:
        super().__init__("visual_encoder")

        # Declare parameters once at construction so they are available to
        # external tooling (ros2 param list) before configure is called.
        self.declare_parameter("retrieval_mode", "siglip_pure")
        self.declare_parameter("siglip_model_id", "google/siglip-base-patch16-224")
        self.declare_parameter("yolo_model_path", "yolov8n.pt")
        self.declare_parameter("yolo_confidence_threshold", 0.4)

        self._pipeline = None
        self._bridge: Optional[CvBridge] = None
        self._srv = None
        self._latency_pub = None

    # ── Lifecycle callbacks ─────────────────────────────────────────────────── #

    def on_configure(self, state) -> TransitionCallbackReturn:
        mode = self.get_parameter("retrieval_mode").value
        model_id = self.get_parameter("siglip_model_id").value
        yolo_path = self.get_parameter("yolo_model_path").value
        yolo_conf = self.get_parameter("yolo_confidence_threshold").value

        if not _CV_BRIDGE_OK:
            self.get_logger().error("cv_bridge is not available.")
            return TransitionCallbackReturn.FAILURE

        self._bridge = CvBridge()

        self.get_logger().info(f"Loading vision pipeline (mode={mode!r}) …")
        try:
            from semantic_vision_core.vision_pipeline import SemanticVisionPipeline

            self._pipeline = SemanticVisionPipeline(
                retrieval_mode=mode,
                siglip_model_id=model_id,
                yolo_model_path=yolo_path,
                yolo_confidence_threshold=yolo_conf,
            )
        except (ImportError, ValueError, RuntimeError) as exc:
            self.get_logger().error(f"Pipeline initialisation failed: {exc}")
            return TransitionCallbackReturn.FAILURE

        self.get_logger().info("Visual encoder configured.")
        return TransitionCallbackReturn.SUCCESS

    def on_activate(self, state) -> TransitionCallbackReturn:
        self._latency_pub = self.create_publisher(
            Float64, "/feature_extraction_latency", 10
        )
        self._srv = self.create_service(
            GetVisualFeatures, "get_visual_features", self._handle_request
        )
        self.get_logger().info("Visual encoder active — /get_visual_features ready.")
        return super().on_activate(state)

    def on_deactivate(self, state) -> TransitionCallbackReturn:
        if self._srv is not None:
            self.destroy_service(self._srv)
            self._srv = None
        if self._latency_pub is not None:
            self.destroy_publisher(self._latency_pub)
            self._latency_pub = None
        self.get_logger().info("Visual encoder deactivated.")
        return super().on_deactivate(state)

    def on_cleanup(self, state) -> TransitionCallbackReturn:
        self._pipeline = None
        self._bridge = None
        self.get_logger().info("Visual encoder cleaned up.")
        return TransitionCallbackReturn.SUCCESS

    def on_shutdown(self, state) -> TransitionCallbackReturn:
        self._pipeline = None
        return TransitionCallbackReturn.SUCCESS

    # ── Service handler ─────────────────────────────────────────────────────── #

    def _handle_request(
        self,
        request: GetVisualFeatures.Request,
        response: GetVisualFeatures.Response,
    ) -> GetVisualFeatures.Response:
        t0 = time.perf_counter()

        # Convert ROS Image → numpy RGB
        try:
            image_rgb = self._bridge.imgmsg_to_cv2(
                request.image, desired_encoding="rgb8"
            )
        except Exception as exc:  # noqa: BLE001
            self.get_logger().error(f"cv_bridge conversion failed: {exc}")
            return self._error_response(response, f"Image conversion error: {exc}")

        # Run pipeline
        try:
            embedding, objects = self._pipeline.process_image(image_rgb)
        except Exception as exc:  # noqa: BLE001
            self.get_logger().error(f"Pipeline inference failed: {exc}")
            return self._error_response(response, f"Pipeline error: {exc}")

        # Build response
        response.visual_embedding = embedding.tolist()
        response.detected_objects = objects
        response.success = True
        response.message = "OK"

        # Publish latency
        latency = time.perf_counter() - t0
        msg = Float64()
        msg.data = latency
        self._latency_pub.publish(msg)
        self.get_logger().debug(f"Request served in {latency * 1e3:.1f} ms.")

        return response

    # ── Helpers ─────────────────────────────────────────────────────────────── #

    @staticmethod
    def _error_response(
        response: GetVisualFeatures.Response, message: str
    ) -> GetVisualFeatures.Response:
        response.success = False
        response.message = message
        response.visual_embedding = []
        response.detected_objects = []
        return response


def main(args=None) -> None:
    rclpy.init(args=args)
    node = VisualEncoderNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
