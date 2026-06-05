"""Pure-Python waypoint ranking — cosine similarity + optional Jaccard.

No ROS imports. Operates on :class:`~semantic_navigation_core.types.Waypoint`
objects whose embeddings are already typed numpy arrays, so there is no CSV
parsing anywhere in the retrieval path.
"""
from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np

from semantic_navigation_core.types import RankedWaypoint, Waypoint

SIGLIP_PURE = "siglip_pure"
SIGLIP_YOLO = "siglip_yolo"
SUPPORTED_MODES: tuple[str, ...] = (SIGLIP_PURE, SIGLIP_YOLO)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity in [-1, 1]; 0.0 if either vector is degenerate."""
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    if a.size == 0 or b.size == 0 or a.shape != b.shape:
        return 0.0
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        return 0.0
    return float(np.dot(a, b) / denom)


def jaccard(set_a: Iterable[str], set_b: Iterable[str]) -> float:
    """Jaccard index between two collections of object labels."""
    a, b = set(set_a), set(set_b)
    if not a and not b:
        return 0.0
    union = len(a | b)
    return len(a & b) / union if union else 0.0


def score_waypoint(
    query_embedding: np.ndarray,
    query_objects: set[str],
    waypoint: Waypoint,
    mode: str,
    embed_weight: float,
    object_weight: float,
) -> float:
    """Hybrid score for a single waypoint.

    In ``siglip_pure`` (or when there are no query objects) the score is plain
    cosine similarity; in ``siglip_yolo`` it is a weighted cosine + Jaccard blend.
    """
    cos = cosine_similarity(query_embedding, waypoint.embedding)
    if mode == SIGLIP_YOLO and query_objects:
        jac = jaccard(query_objects, waypoint.objects)
        return embed_weight * cos + object_weight * jac
    return cos


def rank_waypoints(
    query_embedding: np.ndarray,
    query_objects: Sequence[str],
    waypoints: Sequence[Waypoint],
    mode: str = SIGLIP_PURE,
    embed_weight: float = 0.7,
    object_weight: float = 0.3,
) -> list[RankedWaypoint]:
    """Rank waypoints by descending score.

    Args:
        query_embedding: Query visual/text embedding.
        query_objects: Object labels detected in the query (image goals).
        waypoints: Candidate waypoints (embeddings already typed).
        mode: ``"siglip_pure"`` or ``"siglip_yolo"``.
        embed_weight: Cosine weight in hybrid mode.
        object_weight: Jaccard weight in hybrid mode.

    Returns:
        ``RankedWaypoint`` list sorted high→low. Waypoints with an empty
        embedding are skipped. The caller takes ``[0]`` for the best match.

    Raises:
        ValueError: Unknown ``mode``.
    """
    if mode not in SUPPORTED_MODES:
        raise ValueError(f"mode must be one of {SUPPORTED_MODES}, got {mode!r}")

    q_objs = set(query_objects)
    ranked: list[RankedWaypoint] = []
    for wp in waypoints:
        if wp.embedding is None or np.asarray(wp.embedding).size == 0:
            continue
        score = score_waypoint(
            query_embedding, q_objs, wp, mode, embed_weight, object_weight
        )
        ranked.append(RankedWaypoint(waypoint=wp, score=score))

    ranked.sort(key=lambda r: r.score, reverse=True)
    return ranked
