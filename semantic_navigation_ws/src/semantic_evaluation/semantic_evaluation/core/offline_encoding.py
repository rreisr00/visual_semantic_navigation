"""Batch encoding of offline datasets through the real vision pipeline.

Fills ``SemanticNode`` observations (embeddings, detections, crop embeddings,
relation hypotheses) using the same ``semantic_vision_core`` pipeline the ROS
``visual_encoder`` node wraps, with every heavy result cached via
:class:`~semantic_evaluation.core.embedding_cache.EmbeddingCache`.

Import note: ``semantic_vision_core`` is imported lazily so this module stays
importable (e.g. for tests of the pure parts) without torch installed.
"""
from __future__ import annotations

import os
from typing import TYPE_CHECKING, Iterable, Sequence

import numpy as np

from semantic_evaluation.core.embedding_cache import EmbeddingCache
from semantic_navigation_core.relations import infer_relations
from semantic_navigation_core.types import ObjectObservation, SemanticNode

if TYPE_CHECKING:  # pragma: no cover - typing only
    from semantic_vision_core.vision_pipeline import SemanticVisionPipeline


def _load_image_rgb(path: str) -> np.ndarray:
    from PIL import Image

    with Image.open(os.path.expanduser(path)) as img:
        return np.asarray(img.convert("RGB"))


def encode_observations(
    nodes: Iterable[SemanticNode],
    pipeline: "SemanticVisionPipeline",
    cache: EmbeddingCache,
    model_id: str,
) -> int:
    """Fill missing observation embeddings from their images (cached).

    Observations that already carry an embedding (e.g. loaded from the real
    knowledge graph) are left untouched, preserving parity with the online
    system. Returns the number of embeddings computed or loaded from cache.
    """
    filled = 0
    for node in nodes:
        for obs in node.observations:
            if obs.embedding is not None or not obs.image_path:
                continue
            key = cache.image_key(model_id, obs.image_path)
            embedding = cache.get_array(key)
            if embedding is None:
                embedding = pipeline.embed_image(_load_image_rgb(obs.image_path))
                cache.put_array(key, embedding)
            obs.embedding = embedding
            filled += 1
    return filled


def embed_texts(
    texts: Sequence[str],
    pipeline: "SemanticVisionPipeline",
    cache: EmbeddingCache,
    model_id: str,
) -> list[np.ndarray]:
    """Text embeddings through the same encoder as the ROS ``/get_embedding``."""
    embeddings: list[np.ndarray] = []
    for text in texts:
        key = cache.text_key(model_id, text)
        embedding = cache.get_array(key)
        if embedding is None:
            embedding = pipeline.embed_text(text)
            cache.put_array(key, embedding)
        embeddings.append(embedding)
    return embeddings


def detect_objects(
    nodes: Iterable[SemanticNode],
    pipeline: "SemanticVisionPipeline",
    cache: EmbeddingCache,
    yolo_model_path: str,
    confidence: float,
    with_crop_embeddings: bool = False,
    siglip_model_id: str = "",
) -> int:
    """Run YOLO on every observation image, optionally embedding the crops.

    Existing label-only objects (reconstructed from the graph, box=None) are
    replaced by the full detections when the image is available — same
    detector and threshold, now with geometry. Returns processed image count.
    """
    processed = 0
    for node in nodes:
        for obs in node.observations:
            if not obs.image_path or not os.path.isfile(obs.image_path):
                continue
            det_key = cache.detections_key(
                yolo_model_path, obs.image_path, confidence
            )
            raw = cache.get_json(det_key)
            if raw is None:
                image = _load_image_rgb(obs.image_path)
                detections = pipeline.detect(image)
                raw = [
                    {"label": d.label, "confidence": d.confidence, "box": d.box}
                    for d in detections
                ]
                cache.put_json(det_key, raw)
            objects = [
                ObjectObservation(
                    label=str(d["label"]),
                    confidence=float(d["confidence"]),
                    box=tuple(float(v) for v in d["box"]),
                )
                for d in raw
            ]
            if with_crop_embeddings and objects:
                crops_key = cache.crops_key(
                    siglip_model_id, obs.image_path, det_key
                )
                crop_embs = cache.get_arrays(crops_key)
                if crop_embs is None:
                    image = _load_image_rgb(obs.image_path)
                    crop_embs = pipeline.embed_crops(
                        image, [o.box for o in objects]
                    )
                    cache.put_arrays(crops_key, crop_embs)
                for obj, emb in zip(objects, crop_embs):
                    obj.embedding = emb
            obs.objects = objects
            processed += 1
    return processed


def infer_dataset_relations(nodes: Iterable[SemanticNode]) -> int:
    """Fill 2D relation hypotheses per observation; returns relation count."""
    total = 0
    for node in nodes:
        for obs in node.observations:
            obs.relations = infer_relations(obs.objects)
            total += len(obs.relations)
    return total
