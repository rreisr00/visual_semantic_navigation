"""Association of repeated object detections across node observations."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from semantic_navigation_core.types import ObjectObservation


@dataclass(frozen=True)
class AssociationConfig:
    """Thresholds for conservative object-instance association."""

    maximum_3d_distance_m: float = 0.75
    minimum_crop_similarity: float = 0.75
    allow_single_class_fallback: bool = True


@dataclass(frozen=True)
class AssociationMatch:
    """Selected persistent object and traceable association evidence."""

    object_id: str
    score: float
    evidence: str


def cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    """Return cosine similarity, or -1 when either vector is unusable."""
    a = np.asarray(left, dtype=np.float32).reshape(-1)
    b = np.asarray(right, dtype=np.float32).reshape(-1)
    if not a.size or a.shape != b.shape:
        return -1.0
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denominator) if denominator > 0.0 else -1.0


def match_object(
    detection: ObjectObservation,
    candidates: Sequence[ObjectObservation],
    config: AssociationConfig = AssociationConfig(),
    excluded_ids: set[str] | None = None,
) -> AssociationMatch | None:
    """Match a detection to one existing instance of the same category.

    Metric 3D evidence has priority. Crop similarity is used when positions
    are unavailable. A class-only fallback is allowed only when exactly one
    candidate remains, avoiding arbitrary swaps between repeated instances.
    """
    excluded = excluded_ids or set()
    compatible = [
        item for item in candidates
        if item.label == detection.label
        and item.object_id
        and item.object_id not in excluded
    ]
    if not compatible:
        return None

    if detection.position_3d is not None:
        metric = [item for item in compatible if item.position_3d is not None]
        if metric:
            position = np.asarray(detection.position_3d, dtype=np.float32)
            best = min(
                metric,
                key=lambda item: float(np.linalg.norm(
                    position - np.asarray(item.position_3d, dtype=np.float32)
                )),
            )
            distance = float(np.linalg.norm(
                position - np.asarray(best.position_3d, dtype=np.float32)
            ))
            if distance <= config.maximum_3d_distance_m:
                score = 1.0 - distance / max(config.maximum_3d_distance_m, 1e-9)
                return AssociationMatch(best.object_id, score, "position_3d")

    if detection.embedding is not None:
        scored = [
            (cosine_similarity(detection.embedding, item.embedding), item)
            for item in compatible if item.embedding is not None
        ]
        if scored:
            score, best = max(scored, key=lambda pair: pair[0])
            if score >= config.minimum_crop_similarity:
                return AssociationMatch(best.object_id, score, "crop_embedding")
            return None

    if config.allow_single_class_fallback and len(compatible) == 1:
        return AssociationMatch(compatible[0].object_id, 0.5, "single_class")
    return None
