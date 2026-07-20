"""Unit tests for multiview aggregation."""
import numpy as np
import pytest

from semantic_navigation_core.multiview import (
    AGG_MAX,
    AGG_MAX_TOPK,
    AGG_MEAN,
    AGG_SINGLE,
    AGG_TOPK_MEAN,
    MultiviewConfig,
    aggregate_view_scores,
    mean_embedding,
    score_node_views,
)
from semantic_navigation_core.ranking import cosine_similarity
from semantic_navigation_core.types import Observation, SemanticNode

SCORES = [0.2, 0.8, 0.5]


def test_aggregations():
    assert aggregate_view_scores(SCORES, MultiviewConfig(method=AGG_SINGLE)) == pytest.approx(0.2)
    assert aggregate_view_scores(SCORES, MultiviewConfig(method=AGG_MEAN)) == pytest.approx(0.5)
    assert aggregate_view_scores(SCORES, MultiviewConfig(method=AGG_MAX)) == pytest.approx(0.8)
    assert aggregate_view_scores(
        SCORES, MultiviewConfig(method=AGG_TOPK_MEAN, top_k=2)
    ) == pytest.approx(0.65)
    # 0.7 * max + 0.3 * mean(top-2) = 0.7*0.8 + 0.3*0.65
    assert aggregate_view_scores(
        SCORES, MultiviewConfig(method=AGG_MAX_TOPK, top_k=2)
    ) == pytest.approx(0.7 * 0.8 + 0.3 * 0.65)


def test_empty_scores_yield_zero():
    assert aggregate_view_scores([], MultiviewConfig()) == 0.0


def test_invalid_config_rejected():
    with pytest.raises(ValueError):
        MultiviewConfig(method="nope")
    with pytest.raises(ValueError):
        MultiviewConfig(top_k=0)


def test_score_node_views_matches_cosine():
    query = np.array([1.0, 0.0], dtype=np.float32)
    views = [np.array([1.0, 0.0]), np.array([0.0, 1.0])]
    node = SemanticNode(
        node_id="n1",
        observations=[
            Observation(observation_id=f"v{i}", embedding=e)
            for i, e in enumerate(views)
        ],
    )
    expected = max(cosine_similarity(query, v) for v in views)
    assert score_node_views(query, node, MultiviewConfig(method=AGG_MAX)) == pytest.approx(expected)


def test_mean_embedding_is_normalised():
    result = mean_embedding([np.array([1.0, 0.0]), np.array([0.0, 1.0])])
    assert np.linalg.norm(result) == pytest.approx(1.0)
    assert mean_embedding([]).size == 0
