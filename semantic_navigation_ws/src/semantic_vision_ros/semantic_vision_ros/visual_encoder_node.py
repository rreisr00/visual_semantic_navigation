"""ROS 2 Lifecycle visual encoder node.

Wraps ``semantic_vision_core.SemanticVisionPipeline`` as a stateless service
server. The node never subscribes to camera topics; callers pass a
``sensor_msgs/Image`` directly in the service request.

Services (created on activate)
------------------------------
get_visual_features  SigLIP embedding + optional YOLO detections from an image.
get_embedding        SigLIP embedding from an image (use_image) or text.

Both GPU services share one ``MutuallyExclusiveCallbackGroup`` so inference is
serialised — two SigLIP passes never fight over VRAM concurrently.

Fault tolerance
---------------
On CUDA OOM a handler empties the cache and retries once; on persistent OOM it
returns an error and self-recovers (deactivate → cleanup → reload on CPU) so the
lifecycle manager can re-configure/activate it. ``on_error`` releases the model.

Lifecycle transitions
---------------------
configure   Read params → instantiate SemanticVisionPipeline (loads ML models).
activate    Create services + latency publisher.
deactivate  Destroy services + publisher.
cleanup     Release pipeline and cv_bridge.
error       Release pipeline.
shutdown    No-op.
"""
from __future__ import annotations

import os
import time
from typing import Optional

import numpy as np
import rclpy
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.lifecycle import LifecycleNode, TransitionCallbackReturn
from std_msgs.msg import Float64

from diagnostic_updater import Updater, DiagnosticStatusWrapper

from semantic_interfaces.msg import ObjectDetection, SemanticRelation
from semantic_interfaces.srv import GetEmbedding, GetVisualFeatures
from semantic_navigation_core.relations import infer_relations
from semantic_navigation_core.geometry import infer_3d_relations, project_box_center
from semantic_navigation_core.types import ObjectObservation

try:
    from cv_bridge import CvBridge

    _CV_BRIDGE_OK = True
except ImportError:
    _CV_BRIDGE_OK = False

try:
    import torch

    _TORCH_OK = True
except ImportError:
    _TORCH_OK = False


def _is_oom(exc: Exception) -> bool:
    """True if the exception looks like a CUDA out-of-memory error."""
    if _TORCH_OK and isinstance(exc, torch.cuda.OutOfMemoryError):
        return True
    return "out of memory" in str(exc).lower()


def _empty_cuda_cache() -> None:
    if _TORCH_OK and torch.cuda.is_available():
        torch.cuda.empty_cache()


class VisualEncoderNode(LifecycleNode):
    """Stateless lifecycle node: SigLIP embeddings + optional YOLO detections."""

    def __init__(self) -> None:
        super().__init__("visual_encoder")

        self.declare_parameter("retrieval_mode", "siglip_pure")
        self.declare_parameter("siglip_model_id", "google/siglip-base-patch16-224")
        self.declare_parameter(
            "yolo_model_path", "~/.ros/semantic_models/yolov8n.pt"
        )
        self.declare_parameter("yolo_confidence_threshold", 0.4)
        self.declare_parameter("include_crop_embeddings", True)
        self.declare_parameter("max_crop_embeddings", 16)
        self.declare_parameter("local_files_only", True)
        self.declare_parameter("minimum_depth_m", 0.1)
        self.declare_parameter("maximum_depth_m", 10.0)
        # When True the next configure loads SigLIP on CPU (OOM fallback).
        self.declare_parameter("force_cpu", False)

        self._pipeline = None
        self._bridge: Optional[CvBridge] = None
        self._srv = None
        self._embed_srv = None
        self._latency_pub = None
        self._gpu_cbg = MutuallyExclusiveCallbackGroup()
        self._force_cpu = False
        self._recovering = False
        self._recovery_timer = None

        # Diagnostics run regardless of lifecycle state.
        self._diag = Updater(self)
        self._diag.setHardwareID("visual_encoder")
        self._diag.add("pipeline", self._pipeline_diagnostic)
        self.create_timer(1.0, self._diag.update)

    # ── Lifecycle callbacks ─────────────────────────────────────────────────── #

    def on_configure(self, state) -> TransitionCallbackReturn:
        mode = self.get_parameter("retrieval_mode").value
        model_id = self.get_parameter("siglip_model_id").value
        # expanduser so the default works regardless of the launch CWD
        # (a bare "yolov8n.pt" resolves against CWD and silently re-downloads).
        yolo_path = os.path.expanduser(self.get_parameter("yolo_model_path").value)
        yolo_conf = self.get_parameter("yolo_confidence_threshold").value
        force_cpu = self._force_cpu or bool(self.get_parameter("force_cpu").value)

        if not _CV_BRIDGE_OK:
            self.get_logger().error("cv_bridge is not available.")
            return TransitionCallbackReturn.FAILURE

        self._bridge = CvBridge()
        device = "cpu" if force_cpu else None

        self.get_logger().info(
            f"Loading vision pipeline (mode={mode!r}, device={device or 'auto'}) …"
        )
        try:
            from semantic_vision_core.vision_pipeline import SemanticVisionPipeline

            self._pipeline = SemanticVisionPipeline(
                retrieval_mode=mode,
                siglip_model_id=model_id,
                yolo_model_path=yolo_path,
                yolo_confidence_threshold=yolo_conf,
                device=device,
                local_files_only=bool(self.get_parameter("local_files_only").value),
            )
        except (ImportError, ValueError, RuntimeError) as exc:
            self.get_logger().error(f"Pipeline initialisation failed: {exc}")
            return TransitionCallbackReturn.FAILURE

        self._recovering = False
        self.get_logger().info(
            f"Visual encoder configured (device={self._pipeline.device})."
        )
        return TransitionCallbackReturn.SUCCESS

    def on_activate(self, state) -> TransitionCallbackReturn:
        self._latency_pub = self.create_publisher(
            Float64, "/feature_extraction_latency", 10
        )
        self._srv = self.create_service(
            GetVisualFeatures, "get_visual_features", self._handle_features,
            callback_group=self._gpu_cbg,
        )
        self._embed_srv = self.create_service(
            GetEmbedding, "get_embedding", self._handle_embedding,
            callback_group=self._gpu_cbg,
        )
        self.get_logger().info(
            "Visual encoder active — /get_visual_features and /get_embedding ready."
        )
        return super().on_activate(state)

    def on_deactivate(self, state) -> TransitionCallbackReturn:
        for srv in (self._srv, self._embed_srv):
            if srv is not None:
                self.destroy_service(srv)
        self._srv = None
        self._embed_srv = None
        if self._latency_pub is not None:
            self.destroy_publisher(self._latency_pub)
            self._latency_pub = None
        self.get_logger().info("Visual encoder deactivated.")
        return super().on_deactivate(state)

    def on_cleanup(self, state) -> TransitionCallbackReturn:
        self._release_pipeline()
        self._bridge = None
        self.get_logger().info("Visual encoder cleaned up.")
        return TransitionCallbackReturn.SUCCESS

    def on_error(self, state) -> TransitionCallbackReturn:
        self.get_logger().error("Visual encoder entered error processing.")
        self._release_pipeline()
        return TransitionCallbackReturn.SUCCESS

    def on_shutdown(self, state) -> TransitionCallbackReturn:
        self._release_pipeline()
        return TransitionCallbackReturn.SUCCESS

    def _release_pipeline(self) -> None:
        self._pipeline = None
        _empty_cuda_cache()

    # ── Service handlers ────────────────────────────────────────────────────── #

    def _handle_features(
        self,
        request: GetVisualFeatures.Request,
        response: GetVisualFeatures.Response,
    ) -> GetVisualFeatures.Response:
        t0 = time.perf_counter()
        try:
            image_rgb = self._bridge.imgmsg_to_cv2(
                request.image, desired_encoding="rgb8"
            )
        except Exception as exc:  # noqa: BLE001
            self.get_logger().error(f"cv_bridge conversion failed: {exc}")
            return self._features_error(response, f"Image conversion error: {exc}")

        try:
            embedding = self._run_gpu(lambda: self._pipeline.embed_image(image_rgb))
            detections = []
            crop_embeddings = []
            if self._pipeline.mode == "siglip_yolo":
                detections = self._run_gpu(lambda: self._pipeline.detect(image_rgb))
                if bool(self.get_parameter("include_crop_embeddings").value):
                    limit = max(0, int(self.get_parameter("max_crop_embeddings").value))
                    crop_embeddings = self._run_gpu(
                        lambda: self._pipeline.embed_crops(
                            image_rgb, [d.box for d in detections[:limit]]
                        )
                    )
        except Exception as exc:  # noqa: BLE001
            return self._features_error(response, f"Pipeline error: {exc}")

        response.visual_embedding = embedding.tolist()
        response.detected_objects = list(dict.fromkeys(d.label for d in detections))
        depth_image = None
        intrinsics = None
        depth_scale = 1.0
        depth_frame = (
            request.depth_image.header.frame_id
            or request.camera_info.header.frame_id
        )
        if request.use_depth and request.depth_image.data:
            try:
                depth_image = np.asarray(self._bridge.imgmsg_to_cv2(
                    request.depth_image, desired_encoding="passthrough"
                ))
                depth_scale = 0.001 if depth_image.dtype == np.uint16 else 1.0
                matrix = request.camera_info.k
                intrinsics = (
                    float(matrix[0]), float(matrix[4]),
                    float(matrix[2]), float(matrix[5]),
                )
            except Exception as exc:  # noqa: BLE001
                self.get_logger().warn(
                    f"Depth conversion failed; keeping 2D hypotheses: {exc}"
                )
                depth_image = None
                intrinsics = None
        object_observations = []
        for index, detection in enumerate(detections):
            msg = ObjectDetection()
            msg.object_id = f"detection_{index:04d}"
            msg.class_name = detection.label
            msg.confidence = float(detection.confidence)
            msg.bounding_box = [float(value) for value in detection.box]
            msg.position_2d = [
                float((detection.box[0] + detection.box[2]) * 0.5),
                float((detection.box[1] + detection.box[3]) * 0.5),
            ]
            if index < len(crop_embeddings):
                msg.crop_embedding = np.asarray(
                    crop_embeddings[index], dtype=np.float32
                ).tolist()
            response.detections.append(msg)
            position_3d = None
            if depth_image is not None and intrinsics is not None:
                position_3d = project_box_center(
                    tuple(float(value) for value in detection.box),
                    depth_image,
                    intrinsics,
                    depth_scale=depth_scale,
                    minimum_depth_m=float(
                        self.get_parameter("minimum_depth_m").value
                    ),
                    maximum_depth_m=float(
                        self.get_parameter("maximum_depth_m").value
                    ),
                )
            if position_3d is not None:
                msg.position_3d = [float(value) for value in position_3d]
                msg.position_3d_valid = True
                msg.position_3d_frame = depth_frame
            object_observations.append(ObjectObservation(
                label=detection.label,
                confidence=float(detection.confidence),
                box=tuple(float(value) for value in detection.box),
                embedding=(
                    np.asarray(crop_embeddings[index], dtype=np.float32)
                    if index < len(crop_embeddings) else None
                ),
                object_id=msg.object_id,
                position_2d=tuple(msg.position_2d),
                position_3d=position_3d,
            ))
        for relation in infer_relations(object_observations):
            response.relations.append(_relation_message(
                relation,
                request.image.header.frame_id,
                request.image.header.stamp,
            ))
        if depth_image is not None:
            for relation in infer_3d_relations(
                object_observations, reference_frame=depth_frame
            ):
                response.relations.append(_relation_message(
                    relation,
                    depth_frame,
                    request.depth_image.header.stamp,
                ))
        response.success = True
        response.message = "OK"
        self._publish_latency(t0)
        return response

    def _handle_embedding(
        self,
        request: GetEmbedding.Request,
        response: GetEmbedding.Response,
    ) -> GetEmbedding.Response:
        t0 = time.perf_counter()
        try:
            if request.use_image:
                image_rgb = self._bridge.imgmsg_to_cv2(
                    request.image, desired_encoding="rgb8"
                )
                embedding = self._run_gpu(lambda: self._pipeline.embed_image(image_rgb))
            else:
                text = request.text
                embedding = self._run_gpu(lambda: self._pipeline.embed_text(text))
        except Exception as exc:  # noqa: BLE001
            response.success = False
            response.message = f"Embedding error: {exc}"
            response.embedding = []
            return response

        response.embedding = np.asarray(embedding, dtype=np.float32).tolist()
        response.success = True
        response.message = "OK"
        self._publish_latency(t0)
        return response

    # ── GPU execution with OOM recovery ─────────────────────────────────────── #

    def _run_gpu(self, fn):
        """Run a GPU inference closure, recovering once from CUDA OOM.

        Raises on any non-recoverable failure (caller maps it to an error
        response). On persistent OOM it kicks off self-recovery and raises.
        """
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001
            if not _is_oom(exc):
                self.get_logger().error(f"Inference failed: {exc}")
                raise
            self.get_logger().warn("CUDA OOM — emptying cache and retrying once.")
            _empty_cuda_cache()
            try:
                return fn()
            except Exception as exc2:  # noqa: BLE001
                self.get_logger().error(f"CUDA OOM persisted: {exc2}")
                self._trigger_self_recovery()
                raise

    def _trigger_self_recovery(self) -> None:
        """Deactivate + cleanup with CPU fallback so the manager re-configures."""
        if self._recovering:
            return
        self._recovering = True
        self._force_cpu = True
        self.get_logger().error(
            "Persistent CUDA OOM — recovering (deactivate→cleanup, CPU fallback)."
        )
        # Run the transitions off the service thread (one-shot timer).
        self._recovery_timer = self.create_timer(
            0.1, self._do_self_recovery, callback_group=self._gpu_cbg
        )

    def _do_self_recovery(self) -> None:
        if self._recovery_timer is not None:
            self._recovery_timer.cancel()
            self._recovery_timer = None
        try:
            self.trigger_deactivate()
            self.trigger_cleanup()
        except Exception as exc:  # noqa: BLE001
            self.get_logger().error(f"Self-recovery transition failed: {exc}")

    # ── Helpers ─────────────────────────────────────────────────────────────── #

    def _publish_latency(self, t0: float) -> None:
        if self._latency_pub is None:
            return
        msg = Float64()
        msg.data = time.perf_counter() - t0
        self._latency_pub.publish(msg)

    @staticmethod
    def _features_error(
        response: GetVisualFeatures.Response, message: str
    ) -> GetVisualFeatures.Response:
        response.success = False
        response.message = message
        response.visual_embedding = []
        response.detected_objects = []
        response.detections = []
        response.relations = []
        return response

    def _pipeline_diagnostic(
        self, stat: DiagnosticStatusWrapper
    ) -> DiagnosticStatusWrapper:
        if self._pipeline is None:
            stat.summary(DiagnosticStatusWrapper.WARN, "Pipeline not loaded")
        else:
            stat.summary(DiagnosticStatusWrapper.OK, "Pipeline ready")
            stat.add("device", self._pipeline.device)
            stat.add("mode", self._pipeline.mode)
        stat.add("recovering", str(self._recovering))
        return stat


def _relation_message(relation, reference_frame, timestamp) -> SemanticRelation:
    message = SemanticRelation()
    message.subject_id = relation.subject_id or relation.subject
    message.predicate = relation.predicate
    message.object_id = relation.object_id or relation.obj
    message.confidence = float(relation.confidence)
    message.reference_frame = reference_frame or relation.reference_frame
    message.relation_type = relation.relation_type
    message.timestamp = timestamp
    return message


def main(args=None) -> None:
    rclpy.init(args=args)
    node = VisualEncoderNode()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        try:
            executor.shutdown(timeout_sec=2.0)
            node.destroy_node()
        except KeyboardInterrupt:
            pass
        finally:
            rclpy.try_shutdown()


if __name__ == "__main__":
    main()
