"""Multi-view score/embedding aggregation — pure numpy, no ROS imports.

A :class:`~semantic_navigation_core.types.SemanticNode` may hold several
observations. This module turns the per-view cosine similarities (computed
with the same :func:`~semantic_navigation_core.ranking.cosine_similarity`
used online) into one node-level score, or the per-view embeddings into one
aggregated embedding compatible with the single-embedding ROS path.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from semantic_navigation_core.ranking import cosine_similarity
from semantic_navigation_core.types import SemanticNode

AGG_SINGLE = "single"          # first view only (parity with the online system)
AGG_MEAN = "mean"              # mean over all views
AGG_MAX = "max"                # best view
AGG_TOPK_MEAN = "topk_mean"    # mean of the K best views
AGG_MAX_TOPK = "max_topk"      # max_weight*max + topk_weight*mean(top-K)
SUPPORTED_AGGREGATIONS: tuple[str, ...] = (
    AGG_SINGLE, AGG_MEAN, AGG_MAX, AGG_TOPK_MEAN, AGG_MAX_TOPK,
)


@dataclass
class MultiviewConfig:
    """Configuration for multi-view aggregation (all weights tunable)."""

    method: str = AGG_MAX_TOPK
    top_k: int = 3
    max_weight: float = 0.7
    topk_weight: float = 0.3

    def __post_init__(self) -> None:
        if self.method not in SUPPORTED_AGGREGATIONS:
            raise ValueError(
                f"method must be one of {SUPPORTED_AGGREGATIONS}, "
                f"got {self.method!r}"
            )
        if self.top_k < 1:
            raise ValueError(f"top_k must be >= 1, got {self.top_k}")


def aggregate_view_scores(
    view_scores: Sequence[float], config: MultiviewConfig
) -> float:
    """Collapse per-view similarity scores into one node score.

    Returns 0.0 for an empty score list (node without usable views).
    """
    scores = np.asarray(view_scores, dtype=np.float32)
    if scores.size == 0:
        return 0.0
    if config.method == AGG_SINGLE:
        return float(scores[0])
    if config.method == AGG_MEAN:
        return float(scores.mean())
    if config.method == AGG_MAX:
        return float(scores.max())

    top_k = np.sort(scores)[::-1][: config.top_k]
    if config.method == AGG_TOPK_MEAN:
        return float(top_k.mean())
    # AGG_MAX_TOPK
    return float(
        config.max_weight * scores.max() + config.topk_weight * top_k.mean()
    )


def score_node_views(
    query_embedding: np.ndarray,
    node: SemanticNode,
    config: MultiviewConfig,
) -> float:
    """Query↔node score: per-view cosine similarities, then aggregation."""
    view_scores = [
        cosine_similarity(query_embedding, emb) for emb in node.embeddings()
    ]
    return aggregate_view_scores(view_scores, config)


def mean_embedding(embeddings: Sequence[np.ndarray]) -> np.ndarray:
    """L2-normalised mean of L2-normalised embeddings.

    Useful to collapse a multi-view node into one embedding that can be
    stored through the existing single-embedding ROS pipeline. Returns an
    empty array when there are no embeddings.
    """
    if not embeddings:
        return np.array([], dtype=np.float32)
    stacked = np.stack([np.asarray(e, dtype=np.float32) for e in embeddings])
    mean = stacked.mean(axis=0)
    norm = float(np.linalg.norm(mean))
    if norm == 0.0:
        return mean
    return mean / norm
