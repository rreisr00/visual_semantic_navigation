"""Pure-Python semantic vision pipeline — no ROS 2 imports.

Supported modes
---------------
siglip_pure  SigLIP image embedding only.
siglip_yolo  SigLIP image embedding + YOLOv8 object detection.
"""
from __future__ import annotations

import numpy as np

# Heavy ML deps are optional at import time so the module can be imported
# even before the venv is fully available (e.g. during colcon introspection).
try:
    import torch
    from PIL import Image as PILImage
    from transformers import AutoModel, AutoProcessor

    _SIGLIP_DEPS_OK = True
except ImportError as _e:
    _SIGLIP_DEPS_OK = False
    _SIGLIP_DEPS_ERR = str(_e)


class SemanticVisionPipeline:
    """SigLIP (always) + optional YOLOv8 inference pipeline.

    Args:
        retrieval_mode: ``"siglip_pure"`` or ``"siglip_yolo"``.
        siglip_model_id: HuggingFace model ID or local path.
        yolo_model_path: Path to YOLOv8 weights (ignored in siglip_pure).
        yolo_confidence_threshold: Minimum YOLO detection confidence.

    Raises:
        ValueError: Unknown retrieval_mode.
        ImportError: Required ML dependency missing.
        RuntimeError: Model loading failed.
    """

    SUPPORTED_MODES: tuple[str, ...] = ("siglip_pure", "siglip_yolo")

    def __init__(
        self,
        retrieval_mode: str = "siglip_pure",
        siglip_model_id: str = "google/siglip-base-patch16-224",
        yolo_model_path: str = "yolov8n.pt",
        yolo_confidence_threshold: float = 0.4,
    ) -> None:
        if retrieval_mode not in self.SUPPORTED_MODES:
            raise ValueError(
                f"retrieval_mode must be one of {self.SUPPORTED_MODES}, "
                f"got {retrieval_mode!r}"
            )
        if not _SIGLIP_DEPS_OK:
            raise ImportError(
                f"Missing SigLIP dependencies: {_SIGLIP_DEPS_ERR}. "
                "Install via: pip install torch pillow transformers"
            )

        self._mode = retrieval_mode
        self._yolo_conf = float(yolo_confidence_threshold)
        self._yolo_model = None

        self._load_siglip(siglip_model_id)
        if retrieval_mode == "siglip_yolo":
            self._load_yolo(yolo_model_path)

    # ── public ─────────────────────────────────────────────────────────────── #

    @property
    def mode(self) -> str:
        """Active retrieval mode."""
        return self._mode

    def process_image(
        self, image_rgb: np.ndarray
    ) -> tuple[np.ndarray, list[str]]:
        """Run the full pipeline on one frame.

        Args:
            image_rgb: uint8 array (H, W, 3) in **RGB** order.

        Returns:
            embedding: L2-normalised float32 vector.
            objects: Unique class-name strings (empty in ``siglip_pure`` mode).

        Raises:
            RuntimeError: On inference failure.
        """
        embedding = self._embed(image_rgb)
        objects: list[str] = []
        if self._mode == "siglip_yolo" and self._yolo_model is not None:
            objects = self._detect(image_rgb)
        return embedding, objects

    # ── private ────────────────────────────────────────────────────────────── #

    def _load_siglip(self, model_id: str) -> None:
        self._processor = AutoProcessor.from_pretrained(model_id)
        self._model = AutoModel.from_pretrained(model_id)
        self._model.eval()
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model.to(self._device)

    def _load_yolo(self, model_path: str) -> None:
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise ImportError(
                "ultralytics not installed. Install via: pip install ultralytics"
            ) from exc
        self._yolo_model = YOLO(model_path)

    def _embed(self, image_rgb: np.ndarray) -> np.ndarray:
        pil = PILImage.fromarray(image_rgb)
        inputs = self._processor(images=pil, return_tensors="pt")
        inputs = {k: v.to(self._device) for k, v in inputs.items()}
        with torch.no_grad():
            features = self._model.get_image_features(**inputs)
        vec = features.squeeze(0).cpu().numpy().astype(np.float32)
        return vec / (np.linalg.norm(vec) + 1e-8)

    def _detect(self, image_rgb: np.ndarray) -> list[str]:
        import cv2

        bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        results = self._yolo_model(bgr, conf=self._yolo_conf, verbose=False)
        seen: set[str] = set()
        labels: list[str] = []
        for result in results:
            for cls_id in result.boxes.cls.cpu().numpy().astype(int):
                name = result.names[cls_id]
                if name not in seen:
                    seen.add(name)
                    labels.append(name)
        return labels
