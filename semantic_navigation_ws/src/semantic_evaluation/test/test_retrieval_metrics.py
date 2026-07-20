"""Unit tests for the offline retrieval metrics."""
import math

from semantic_evaluation.core.retrieval_metrics import (
    RetrievalCaseResult,
    annotate_rank,
    apply_rejection,
    mean_rank,
    mean_reciprocal_rank,
    metric_distance,
    nearby_success_rate,
    negative_rejection_rate,
    rank_of_first_valid,
    recall_at_k,
    results_to_rows,
    summarize,
    topological_distance,
)


def case(query_id, valid, ranked, scores=None, negative=False, method="m1"):
    result = RetrievalCaseResult(
        query_id=query_id,
        method=method,
        is_negative=negative,
        valid_node_ids=valid,
        ranked_ids=ranked,
        scores=scores or [1.0] * len(ranked),
    )
    return annotate_rank(result)


def test_rank_of_first_valid():
    assert rank_of_first_valid(["a", "b", "c"], ["b"]) == 2
    assert rank_of_first_valid(["a"], ["z"]) is None
    assert rank_of_first_valid([], ["z"]) is None


def test_recall_and_mrr():
    results = [
        case("q1", ["a"], ["a", "b"]),          # rank 1
        case("q2", ["b"], ["a", "b"]),          # rank 2
        case("q3", ["z"], ["a", "b"]),          # not found
    ]
    assert recall_at_k(results, 1) == 1 / 3
    assert recall_at_k(results, 3) == 2 / 3
    assert mean_reciprocal_rank(results) == (1.0 + 0.5 + 0.0) / 3
    assert mean_rank(results) == 1.5


def test_negative_rejection():
    neg_low = apply_rejection(
        case("n1", [], ["a"], scores=[0.02], negative=True), threshold=0.1
    )
    neg_high = apply_rejection(
        case("n2", [], ["a"], scores=[0.5], negative=True), threshold=0.1
    )
    assert neg_low.rejected and not neg_high.rejected
    assert negative_rejection_rate([neg_low, neg_high]) == 0.5
    # Negative queries never pollute the positive metrics.
    assert math.isnan(recall_at_k([neg_low, neg_high], 1))


def test_nearby_success_uses_rooms():
    results = [case("q1", ["a"], ["b", "a"])]  # top-1 wrong node, same room
    rooms = {"a": "cocina", "b": "cocina"}
    assert nearby_success_rate(results, rooms) == 1.0
    assert nearby_success_rate(results, {"a": "cocina", "b": "salon"}) == 0.0
    assert nearby_success_rate(results, {"a": "cocina", "b": None}) == 0.0


def test_distances():
    positions = {"a": (0.0, 0.0), "b": (3.0, 4.0)}
    assert metric_distance("b", ["a"], positions) == 5.0
    assert math.isnan(metric_distance("z", ["a"], positions))
    edges = [("a", "b"), ("b", "c")]
    assert topological_distance(edges, "a", "c") == 2
    assert topological_distance(edges, "a", "a") == 0
    assert topological_distance(edges, "a", "island") is None


def test_summarize_groups_by_method():
    results = [
        case("q1", ["a"], ["a"], method="m1"),
        case("q2", ["a"], ["b", "a"], method="m2"),
    ]
    rows = summarize(results, group_keys=("method",))
    by_method = {row["method"]: row for row in rows}
    assert by_method["m1"]["recall_at_1"] == 1.0
    assert by_method["m2"]["recall_at_1"] == 0.0
    assert by_method["m2"]["recall_at_3"] == 1.0
    detail = results_to_rows(results)
    assert detail[0]["predicted_node_id"] == "a"
    assert detail[1]["rank_first_valid"] == 2
