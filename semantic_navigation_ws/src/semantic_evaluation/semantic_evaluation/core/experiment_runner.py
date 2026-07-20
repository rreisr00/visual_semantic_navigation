"""Offline experiment runner: queries × scene × method → case results.

Thin orchestration shared by the notebooks so they stay declarative: prepares
ground truth per scene, converts :class:`ExperimentQuery` to the core
:class:`SemanticQuery`, times each ranking call and annotates rank/rejection.
No rclpy imports.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

from semantic_evaluation.core.offline_dataset import (
    ExperimentQuery,
    SceneDataset,
    resolve_valid_nodes,
)
from semantic_evaluation.core.retrieval_metrics import (
    RetrievalCaseResult,
    annotate_rank,
    apply_rejection,
)
from semantic_navigation_core.retrieval import (
    RetrievalConfig,
    SemanticQuery,
    rank_nodes,
)


@dataclass
class PreparedQuery:
    """A query with its scene-resolved ground truth and text embedding."""

    query: ExperimentQuery
    valid_node_ids: list[str]
    embedding: np.ndarray | None = None


def prepare_queries(
    queries: Iterable[ExperimentQuery], dataset: SceneDataset
) -> list[PreparedQuery]:
    """Queries applicable to this scene, with resolved ground truth."""
    return [
        PreparedQuery(query=q, valid_node_ids=resolve_valid_nodes(q, dataset))
        for q in queries
        if (q.scene_id or q.dataset_id) in (None, "", dataset.scene_id)
    ]


def to_semantic_query(
    prepared: PreparedQuery,
    position: tuple[float, float] | None = None,
) -> SemanticQuery:
    """Map the annotated experiment query to the core retrieval query."""
    q = prepared.query
    return SemanticQuery(
        text=q.text,
        embedding=prepared.embedding,
        objects=list(q.expected_objects),
        relations=list(q.expected_relations),
        room=q.expected_room,
        position=position,
    )


def run_method(
    method_label: str,
    prepared: Sequence[PreparedQuery],
    dataset: SceneDataset,
    config: RetrievalConfig,
    rejection_threshold: float,
    query_position: tuple[float, float] | None = None,
) -> list[RetrievalCaseResult]:
    """Run one retrieval method over every prepared query of a scene.

    Args:
        method_label: Name recorded in the results (may differ from
            ``config.method`` to distinguish e.g. multiview aggregation variants).
        prepared: Output of :func:`prepare_queries` with embeddings filled.
        dataset: The scene under evaluation.
        config: Core retrieval configuration (method + weights + multiview).
        rejection_threshold: Best-score threshold for negative-query rejection.
        query_position: (x, y) origin used by ``baseline_nearest``.

    Returns:
        One annotated :class:`RetrievalCaseResult` per query.
    """
    results: list[RetrievalCaseResult] = []
    for p in prepared:
        semantic_query = to_semantic_query(p, position=query_position)
        t0 = time.perf_counter()
        ranked = rank_nodes(semantic_query, dataset.nodes, config)
        latency = time.perf_counter() - t0
        result = RetrievalCaseResult(
            query_id=p.query.query_id,
            method=method_label,
            dataset_id=dataset.scene_id if p.query.dataset_id else "",
            scene_id=dataset.scene_id if p.query.scene_id else "",
            query_type=p.query.query_type,
            language=p.query.language,
            is_negative=p.query.is_negative,
            valid_node_ids=list(p.valid_node_ids),
            ranked_ids=[r.node.node_id for r in ranked],
            scores=[float(r.score) for r in ranked],
            score_components=(dict(ranked[0].components) if ranked else {}),
            latency_s=latency,
        )
        annotate_rank(result)
        apply_rejection(result, rejection_threshold)
        results.append(result)
    return results
