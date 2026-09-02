"""Unit tests for the offline experiment runner."""
import numpy as np

from semantic_evaluation.core.experiment_runner import (
    prepare_queries,
    run_method,
    to_semantic_query,
)
from semantic_evaluation.core.offline_dataset import ExperimentQuery, SceneDataset
from semantic_navigation_core.retrieval import METHOD_SIGLIP_SINGLE, RetrievalConfig
from semantic_navigation_core.types import Observation, SemanticNode


def make_dataset():
    def node(node_id, vec, room):
        return SemanticNode(
            node_id=node_id,
            room_id=room,
            observations=[Observation(
                observation_id=f"{node_id}_v0",
                embedding=np.asarray(vec, dtype=np.float32),
            )],
        )
    return SceneDataset(
        scene_id="s1",
        nodes=[node("a", [1.0, 0.0], "cocina"), node("b", [0.0, 1.0], "salon")],
    )


QUERIES = [
    ExperimentQuery(query_id="q1", text="cocina", scene_id="s1",
                    expected_room="cocina", query_type="room"),
    ExperimentQuery(query_id="q2", text="otra escena", scene_id="s2"),
    ExperimentQuery(query_id="q3", text="piscina", scene_id="s1",
                    is_negative=True, query_type="negative"),
]


def test_prepare_filters_by_scene_and_resolves_gt():
    prepared = prepare_queries(QUERIES, make_dataset())
    assert [p.query.query_id for p in prepared] == ["q1", "q3"]
    assert prepared[0].valid_node_ids == ["a"]      # via expected_room
    assert prepared[1].valid_node_ids == []          # negative


def test_to_semantic_query_maps_fields():
    prepared = prepare_queries(QUERIES, make_dataset())[0]
    prepared.embedding = np.array([1.0, 0.0], dtype=np.float32)
    sq = to_semantic_query(prepared, position=(1.0, 2.0))
    assert sq.room == "cocina" and sq.position == (1.0, 2.0)
    assert sq.embedding is prepared.embedding


def test_run_method_annotates_rank_and_rejection():
    dataset = make_dataset()
    prepared = prepare_queries(QUERIES, dataset)
    for p in prepared:
        p.embedding = np.array([1.0, 0.0], dtype=np.float32)  # ≈ nodo "a"
    results = run_method(
        "siglip_single_view", prepared, dataset,
        RetrievalConfig(method=METHOD_SIGLIP_SINGLE),
        rejection_threshold=0.5,
    )
    by_id = {r.query_id: r for r in results}
    assert by_id["q1"].rank == 1 and by_id["q1"].method == "siglip_single_view"
    assert by_id["q1"].latency_s >= 0.0
    # La negativa apunta al mismo embedding → puntuación alta → NO rechazada.
    assert by_id["q3"].is_negative and not by_id["q3"].rejected
