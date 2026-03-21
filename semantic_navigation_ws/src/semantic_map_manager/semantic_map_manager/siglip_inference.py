#!/usr/bin/env python3
"""SigLIP 2 AI Inference Node.

Loads the google/siglip-base-patch16-224 model once at startup and exposes a
``GetEmbedding`` service (semantic_map_manager_interfaces/srv/GetEmbedding)
that returns a float32 embedding vector for a given image or text.

Service API
-----------
Request
    image    (sensor_msgs/Image) -- ROS image to embed (when use_image=True)
    text     (string)            -- free-form text to embed (when use_image=False)
    use_image (bool)             -- selects the modality

Response
    embedding (float32[])        -- L2-normalised SigLIP embedding
    success   (bool)
    message   (string)
"""

import numpy as np
import rclpy
from rclpy.node import Node

from semantic_map_manager_interfaces.srv import GetEmbedding

# Optional heavy imports – handled gracefully so the node can still start even
# when running in a minimal CI environment without GPU/internet access.
try:
    import cv2
    from cv_bridge import CvBridge
    from PIL import Image as PILImage
    import torch
    from transformers import AutoProcessor, AutoModel

    _DEPS_AVAILABLE = True
except ImportError as _import_err:  # pragma: no cover
    _DEPS_AVAILABLE = False
    _IMPORT_ERR_MSG = str(_import_err)

_MODEL_ID = "google/siglip-base-patch16-224"


class SigLIPInferenceNode(Node):
    """ROS 2 node that serves SigLIP 2 embeddings via a service."""

    def __init__(self) -> None:
        super().__init__("siglip_inference")

        self._model = None
        self._processor = None
        self._bridge = None

        if not _DEPS_AVAILABLE:
            self.get_logger().error(
                f"Required Python packages are missing – inference will be unavailable. "
                f"Error: {_IMPORT_ERR_MSG}"
            )
        else:
            self._load_model()
            self._bridge = CvBridge()

        self._srv = self.create_service(
            GetEmbedding, "get_embedding", self._handle_get_embedding
        )
        self.get_logger().info("SigLIP inference node ready.")

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_model(self) -> None:
        """Download / load the SigLIP 2 model and processor."""
        self.get_logger().info(f"Loading model '{_MODEL_ID}' …")
        try:
            self._processor = AutoProcessor.from_pretrained(_MODEL_ID)
            self._model = AutoModel.from_pretrained(_MODEL_ID)
            self._model.eval()
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self._model.to(device)
            self.get_logger().info(f"Model loaded on {device}.")
        except Exception as exc:  # noqa: BLE001
            self.get_logger().error(f"Failed to load model: {exc}")
            self._model = None
            self._processor = None

    def _embed_image(self, ros_image_msg) -> np.ndarray | None:
        """Convert a ROS Image message to a SigLIP embedding."""
        try:
            cv_img = self._bridge.imgmsg_to_cv2(ros_image_msg, desired_encoding="rgb8")
            pil_img = PILImage.fromarray(cv_img)
            inputs = self._processor(images=pil_img, return_tensors="pt")
            device = next(self._model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                features = self._model.get_image_features(**inputs)
            embedding = features.squeeze(0).cpu().numpy()
            return embedding / (np.linalg.norm(embedding) + 1e-8)
        except Exception as exc:  # noqa: BLE001
            self.get_logger().error(f"Image embedding failed: {exc}")
            return None

    def _embed_text(self, text: str) -> np.ndarray | None:
        """Compute a SigLIP text embedding for the given string."""
        try:
            inputs = self._processor(text=[text], return_tensors="pt", padding=True)
            device = next(self._model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                features = self._model.get_text_features(**inputs)
            embedding = features.squeeze(0).cpu().numpy()
            return embedding / (np.linalg.norm(embedding) + 1e-8)
        except Exception as exc:  # noqa: BLE001
            self.get_logger().error(f"Text embedding failed: {exc}")
            return None

    # ------------------------------------------------------------------
    # Service handler
    # ------------------------------------------------------------------

    def _handle_get_embedding(
        self, request: GetEmbedding.Request, response: GetEmbedding.Response
    ) -> GetEmbedding.Response:
        if self._model is None or self._processor is None:
            response.success = False
            response.message = "Model is not loaded."
            response.embedding = []
            return response

        if request.use_image:
            embedding = self._embed_image(request.image)
            modality = "image"
        else:
            embedding = self._embed_text(request.text)
            modality = f"text='{request.text}'"

        if embedding is None:
            response.success = False
            response.message = f"Failed to compute {modality} embedding."
            response.embedding = []
        else:
            response.success = True
            response.message = "OK"
            response.embedding = embedding.tolist()

        return response


def main(args=None) -> None:
    rclpy.init(args=args)
    node = SigLIPInferenceNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
