"""Unit tests for the retrieval methods (baselines, SigLIP, hybrids)."""
import numpy as np
import pytest

from semantic_navigation_core.ranking import cosine_similarity
from semantic_navigation_core.retrieval import (
    METHOD_BASELINE_NEAREST,
    METHOD_BASELINE_RANDOM,
    METHOD_BASELINE_ROOM,
    METHOD_SIGLIP_SINGLE,
    METHOD_SIGLIP_MULTIVIEW,
    METHOD_SIGLIP_OBJECTS,
    METHOD_HYBRID_FULL,
    HybridWeights,
    RetrievalConfig,
    SemanticQuery,
    default_weights,
    object_score,
    rank_nodes,
)
from semantic_navigation_core.multiview import AGG_MAX, MultiviewConfig
from semantic_navigation_core.types import (
    Observation,
    ObjectObservation,
    SemanticNode,
    SpatialRelation,
)


def node(node_id, embeddings, position=(0.0, 0.0, 0.0), room=None, objects=()):
    observations = [
        Observation(
            observation_id=f"{node_id}_v{i}",
            embedding=np.asarray(e, dtype=np.float32),
            objects=list(objects) if i == 0 else [],
        )
        for i, e in enumerate(embeddings)
    ]
    return SemanticNode(
        node_id=node_id, position=position, observations=observations, room_id=room
    )


QUERY_EMB = np.array([1.0, 0.0], dtype=np.float32)
NODES = [
    node("a", [[1.0, 0.0]], position=(0.0, 0.0, 0.0), room="cocina"),
    node("b", [[0.0, 1.0], [0.9, 0.1]], position=(5.0, 0.0, 0.0), room="salon"),
]


def test_single_view_matches_online_cosine():
    """siglip_single_view must equal the orchestrator's siglip_pure path: plain cosine on view 0."""
    config = RetrievalConfig(method=METHOD_SIGLIP_SINGLE)
    ranked = rank_nodes(SemanticQuery(embedding=QUERY_EMB), NODES, config)
    assert ranked[0].node.node_id == "a"
    for r in ranked:
        expected = cosine_similarity(QUERY_EMB, r.node.observations[0].embedding)
        assert r.score == pytest.approx(expected)


def test_multiview_max_aggregation_uses_best_view():
    config = RetrievalConfig(
        method=METHOD_SIGLIP_MULTIVIEW, multiview=MultiviewConfig(method=AGG_MAX)
    )
    ranked = rank_nodes(SemanticQuery(embedding=QUERY_EMB), NODES, config)
    b = next(r for r in ranked if r.node.node_id == "b")
    assert b.score == pytest.approx(
        cosine_similarity(QUERY_EMB, np.array([0.9, 0.1], dtype=np.float32))
    )


def test_baseline_random_is_deterministic_with_seed():
    config = RetrievalConfig(method=METHOD_BASELINE_RANDOM, seed=7)
    first = [r.node.node_id for r in rank_nodes(SemanticQuery(), NODES, config)]
    second = [r.node.node_id for r in rank_nodes(SemanticQuery(), NODES, config)]
    assert first == second


def test_baseline_nearest_orders_by_distance():
    config = RetrievalConfig(method=METHOD_BASELINE_NEAREST)
    ranked = rank_nodes(SemanticQuery(position=(4.0, 0.0)), NODES, config)
    assert ranked[0].node.node_id == "b"
    with pytest.raises(ValueError):
        rank_nodes(SemanticQuery(), NODES, config)


def test_baseline_room_prefers_matching_room():
    config = RetrievalConfig(method=METHOD_BASELINE_ROOM)
    ranked = rank_nodes(SemanticQuery(room="cocina"), NODES, config)
    assert ranked[0].node.node_id == "a"
    assert ranked[0].score == 1.0


def test_object_score_confidence_weighted():
    n = node(
        "c", [[1.0, 0.0]],
        objects=[ObjectObservation(label="cup", confidence=0.8)],
    )
    score = object_score(["cup"], n)
    # coverage = 0.8, jaccard = 1.0 → 0.5*0.8 + 0.5*1.0
    assert score == pytest.approx(0.9)
    assert object_score([], n) == 0.0
    assert object_score(["cup"], node("d", [[1.0, 0.0]])) == 0.0


def test_objects_method_combines_components():
    n_obj = node(
        "obj", [[0.8, 0.6]],
        objects=[ObjectObservation(label="cup", confidence=1.0)],
    )
    n_plain = node("plain", [[0.8, 0.6]])
    config = RetrievalConfig(
        method=METHOD_SIGLIP_OBJECTS, weights=HybridWeights(alpha=0.7, beta=0.3)
    )
    ranked = rank_nodes(
        SemanticQuery(embedding=QUERY_EMB, objects=["cup"]),
        [n_obj, n_plain],
        config,
    )
    assert ranked[0].node.node_id == "obj"
    assert ranked[0].components["object_match_score"] > 0.0


def test_hybrid_full_room_and_relations_break_ties():
    rel = SpatialRelation("cup", "NEAR", "tv", confidence=1.0)
    n_full = node(
        "full", [[1.0, 0.0]], room="cocina",
        objects=[ObjectObservation(label="cup"), ObjectObservation(label="tv")],
    )
    n_full.observations[0].relations = [rel]
    n_bare = node("bare", [[1.0, 0.0]], room="salon")
    config = RetrievalConfig(
        method=METHOD_HYBRID_FULL, weights=default_weights(METHOD_HYBRID_FULL)
    )
    query = SemanticQuery(
        embedding=QUERY_EMB, objects=["cup"], relations=[rel], room="cocina"
    )
    ranked = rank_nodes(query, [n_bare, n_full], config)
    assert ranked[0].node.node_id == "full"
    assert ranked[0].components["room_match_score"] == 1.0
    assert ranked[0].components["relation_match_score"] == 1.0


def test_nodes_without_embeddings_are_skipped():
    empty = SemanticNode(node_id="empty")
    config = RetrievalConfig(method=METHOD_SIGLIP_SINGLE)
    ranked = rank_nodes(SemanticQuery(embedding=QUERY_EMB), [empty, NODES[0]], config)
    assert [r.node.node_id for r in ranked] == ["a"]


def test_unknown_method_rejected():
    with pytest.raises(ValueError):
        RetrievalConfig(method="inexistente")
