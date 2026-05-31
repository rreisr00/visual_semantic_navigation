#!/usr/bin/env python3
"""Visual Encoder Node.

Loads google/siglip-base-patch16-224 and (optionally) yolov8n.pt at startup
and exposes a GetVisualFeatures service that returns:
  - visual_embedding: L2-normalised SigLIP image embedding
  - detected_objects: list of YOLO class labels (empty in siglip_pure mode)

Publishes per-call feature-extraction latency to /feature_extraction_latency.

Parameters
----------
retrieval_mode          : str  - "siglip_pure" or "siglip_yolo"
yolo_model_path         : str  - path to yolov8n.pt (used only in siglip_yolo)
yolo_confidence_threshold: float - YOLO detection threshold (default 0.4)
siglip_model_id         : str  - HuggingFace model ID for SigLIP
"""

import time

import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64

from semantic_interfaces.srv import GetVisualFeatures

try:
    import cv2
    from cv_bridge import CvBridge
    from PIL import Image as PILImage
    import torch
    from transformers import AutoProcessor, AutoModel

    _BASE_DEPS = True
except ImportError as _e:
    _BASE_DEPS = False
    _BASE_ERR = str(_e)

try:
    from ultralytics import YOLO as _YOLO

    _YOLO_DEP = True
except ImportError:
    _YOLO_DEP = False


class VisualEncoderNode(Node):
    """ROS 2 node serving SigLIP embeddings and optional YOLO detections."""

    def __init__(self) -> None:
        super().__init__("visual_encoder")

        self.declare_parameter("retrieval_mode", "siglip_yolo")
        self.declare_parameter("yolo_model_path", "yolov8n.pt")
        self.declare_parameter("yolo_confidence_threshold", 0.4)
        self.declare_parameter("siglip_model_id", "google/siglip-base-patch16-224")

        self._mode = self.get_parameter("retrieval_mode").get_parameter_value().string_value
        self._yolo_path = self.get_parameter("yolo_model_path").get_parameter_value().string_value
        self._yolo_conf = (
            self.get_parameter("yolo_confidence_threshold").get_parameter_value().double_value
        )
        self._model_id = self.get_parameter("siglip_model_id").get_parameter_value().string_value

        self._siglip_model = None
        self._siglip_processor = None
        self._yolo_model = None
        self._bridge = None

        if not _BASE_DEPS:
            self.get_logger().error(f"Missing base dependencies: {_BASE_ERR}")
        else:
            self._bridge = CvBridge()
            self._load_siglip()
            if self._mode == "siglip_yolo":
                self._load_yolo()

        self._latency_pub = self.create_publisher(Float64, "/feature_extraction_latency", 10)
        self._srv = self.create_service(
            GetVisualFeatures, "get_visual_features", self._handle_get_visual_features
        )
        self.get_logger().info(f"Visual encoder ready (mode={self._mode}).")

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _load_siglip(self) -> None:
        self.get_logger().info(f"Loading SigLIP model '{self._model_id}' ...")
        try:
            self._siglip_processor = AutoProcessor.from_pretrained(self._model_id)
            self._siglip_model = AutoModel.from_pretrained(self._model_id)
            self._siglip_model.eval()
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self._siglip_model.to(device)
            self.get_logger().info(f"SigLIP loaded on {device}.")
        except Exception as exc:  # noqa: BLE001
            self.get_logger().error(f"SigLIP load failed: {exc}")

    def _load_yolo(self) -> None:
        if not _YOLO_DEP:
            self.get_logger().error("ultralytics not installed – YOLO unavailable.")
            return
        self.get_logger().info(f"Loading YOLO model '{self._yolo_path}' ...")
        try:
            self._yolo_model = _YOLO(self._yolo_path)
            self.get_logger().info("YOLO loaded.")
        except Exception as exc:  # noqa: BLE001
            self.get_logger().error(f"YOLO load failed: {exc}")

    # ------------------------------------------------------------------
    # Service handler
    # ------------------------------------------------------------------

    def _handle_get_visual_features(
        self,
        request: GetVisualFeatures.Request,
        response: GetVisualFeatures.Response,
    ) -> GetVisualFeatures.Response:
        t0 = time.perf_counter()

        if self._siglip_model is None or self._siglip_processor is None:
            response.success = False
            response.message = "SigLIP model not loaded."
            response.visual_embedding = []
            response.detected_objects = []
            return response

        embedding = self._embed_image(request.image)
        if embedding is None:
            response.success = False
            response.message = "SigLIP embedding failed."
            response.visual_embedding = []
            response.detected_objects = []
            return response

        detected: list[str] = []
        if self._mode == "siglip_yolo" and self._yolo_model is not None:
            detected = self._detect_objects(request.image)

        response.visual_embedding = embedding.tolist()
        response.detected_objects = detected
        response.success = True
        response.message = "OK"

        latency_msg = Float64()
        latency_msg.data = time.perf_counter() - t0
        self._latency_pub.publish(latency_msg)

        return response

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _embed_image(self, ros_image_msg) -> np.ndarray | None:
        try:
            cv_img = self._bridge.imgmsg_to_cv2(ros_image_msg, desired_encoding="rgb8")
            pil_img = PILImage.fromarray(cv_img)
            inputs = self._siglip_processor(images=pil_img, return_tensors="pt")
            device = next(self._siglip_model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                features = self._siglip_model.get_image_features(**inputs)
            vec = features.squeeze(0).cpu().numpy()
            return vec / (np.linalg.norm(vec) + 1e-8)
        except Exception as exc:  # noqa: BLE001
            self.get_logger().error(f"Image embedding error: {exc}")
            return None

    def _detect_objects(self, ros_image_msg) -> list[str]:
        try:
            cv_img = self._bridge.imgmsg_to_cv2(ros_image_msg, desired_encoding="bgr8")
            results = self._yolo_model(cv_img, conf=self._yolo_conf, verbose=False)
            labels: list[str] = []
            for result in results:
                for cls_id in result.boxes.cls.cpu().numpy().astype(int):
                    label = result.names[cls_id]
                    if label not in labels:
                        labels.append(label)
            return labels
        except Exception as exc:  # noqa: BLE001
            self.get_logger().error(f"YOLO detection error: {exc}")
            return []


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
