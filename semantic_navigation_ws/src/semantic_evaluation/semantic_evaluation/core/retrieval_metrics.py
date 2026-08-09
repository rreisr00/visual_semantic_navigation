"""Offline retrieval metrics (Recall@K, MRR, distances) — no rclpy imports.

Complements ``evaluation_logic`` (top-1 / room accuracy for the online
campaign) with ranking-quality metrics for the offline notebooks. All inputs
are plain Python/NumPy so the module works identically inside ROS later.
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Iterable, Mapping, Sequence

NAN = float("nan")


@dataclass
class RetrievalCaseResult:
    """Outcome of one (query, method) retrieval run over a scene.

    ``rank`` is the 1-indexed position of the first valid node in
    ``ranked_ids`` (None when no valid node was retrieved). For negative
    queries ``valid_node_ids`` is empty and ``rejected`` says whether the
    system correctly refused (best score below the rejection threshold).
    """

    query_id: str
    method: str
    dataset_id: str = ""
    scene_id: str = ""
    query_type: str = ""
    language: str = ""
    is_negative: bool = False
    target_visible: bool = True
    valid_node_ids: list[str] = field(default_factory=list)
    ranked_ids: list[str] = field(default_factory=list)
    scores: list[float] = field(default_factory=list)
    score_components: dict[str, float] = field(default_factory=dict)
    rank: int | None = None
    rejected: bool = False
    latency_s: float = NAN

    @property
    def top1_id(self) -> str:
        return self.ranked_ids[0] if self.ranked_ids else ""


def rank_of_first_valid(
    ranked_ids: Sequence[str], valid_ids: Iterable[str]
) -> int | None:
    """1-indexed rank of the first valid node, or None if absent."""
    valid = set(valid_ids)
    for idx, node_id in enumerate(ranked_ids, start=1):
        if node_id in valid:
            return idx
    return None


def annotate_rank(result: RetrievalCaseResult) -> RetrievalCaseResult:
    """Fill ``rank`` in place (no-op for negative queries) and return it."""
    if not result.is_negative and result.target_visible:
        result.rank = rank_of_first_valid(result.ranked_ids, result.valid_node_ids)
    return result


def apply_rejection(
    result: RetrievalCaseResult, threshold: float
) -> RetrievalCaseResult:
    """Mark the case as rejected when its best score is below ``threshold``."""
    best = result.scores[0] if result.scores else NAN
    result.rejected = math.isnan(best) or best < threshold
    return result


# ── Aggregations (positive queries use `rank`; negatives use `rejected`) ─── #


def _positives(results: Iterable[RetrievalCaseResult]) -> list[RetrievalCaseResult]:
    return [r for r in results if not r.is_negative and r.target_visible]


def recall_at_k(results: Iterable[RetrievalCaseResult], k: int) -> float:
    """Fraction of positive queries whose first valid node is in the top K."""
    pos = _positives(results)
    if not pos:
        return NAN
    return sum(1 for r in pos if r.rank is not None and r.rank <= k) / len(pos)


def mean_reciprocal_rank(results: Iterable[RetrievalCaseResult]) -> float:
    """MRR over positive queries (missing valid node contributes 0)."""
    pos = _positives(results)
    if not pos:
        return NAN
    return sum(1.0 / r.rank if r.rank else 0.0 for r in pos) / len(pos)


def mean_rank(results: Iterable[RetrievalCaseResult]) -> float:
    """Mean rank of the first valid node over queries where it was found."""
    ranks = [r.rank for r in _positives(results) if r.rank is not None]
    if not ranks:
        return NAN
    return sum(ranks) / len(ranks)


def negative_rejection_rate(results: Iterable[RetrievalCaseResult]) -> float:
    """Fraction of negative queries correctly rejected (NaN if none)."""
    neg = [r for r in results if r.is_negative]
    if not neg:
        return NAN
    return sum(1 for r in neg if r.rejected) / len(neg)


def room_false_positive_rate(results: Iterable[RetrievalCaseResult]) -> float:
    """Fraction of invisible targets for which a room/node was still accepted."""
    invisible = [result for result in results if not result.target_visible]
    if not invisible:
        return NAN
    return sum(1 for result in invisible if not result.rejected) / len(invisible)


def exact_success_rate(results: Iterable[RetrievalCaseResult]) -> float:
    """Fraction of positive queries answered with a valid node at rank 1."""
    return recall_at_k(results, 1)


def nearby_success_rate(
    results: Iterable[RetrievalCaseResult],
    room_by_node: Mapping[str, str | None],
) -> float:
    """Fraction of positive queries whose top-1 shares a room with a valid node.

    "Nearby semantic success": the exact waypoint was missed but the predicted
    node lies in the same room as some ground-truth node — still a useful
    navigation outcome. Nodes without a room never match.
    """
    pos = _positives(results)
    if not pos:
        return NAN
    hits = 0
    for r in pos:
        pred_room = room_by_node.get(r.top1_id)
        if pred_room is None:
            continue
        valid_rooms = {
            room_by_node.get(v) for v in r.valid_node_ids
        } - {None}
        if pred_room in valid_rooms:
            hits += 1
    return hits / len(pos)


def mean_latency_s(results: Iterable[RetrievalCaseResult]) -> float:
    """NaN-aware mean of per-query latencies."""
    values = [r.latency_s for r in results if not math.isnan(r.latency_s)]
    if not values:
        return NAN
    return sum(values) / len(values)


# ── Distances ────────────────────────────────────────────────────────────── #


def metric_distance(
    predicted_id: str,
    valid_ids: Iterable[str],
    position_by_node: Mapping[str, tuple[float, float]],
) -> float:
    """Min Euclidean (x, y) distance from the prediction to any valid node."""
    pred = position_by_node.get(predicted_id)
    if pred is None:
        return NAN
    distances = [
        math.hypot(pred[0] - p[0], pred[1] - p[1])
        for v in valid_ids
        if (p := position_by_node.get(v)) is not None
    ]
    return min(distances) if distances else NAN


def topological_distance(
    edges: Iterable[tuple[str, str]], source: str, target: str
) -> int | None:
    """Hop count between two nodes over undirected edges (BFS); None if apart."""
    if source == target:
        return 0
    adjacency: dict[str, set[str]] = {}
    for a, b in edges:
        adjacency.setdefault(a, set()).add(b)
        adjacency.setdefault(b, set()).add(a)
    seen = {source}
    queue: deque[tuple[str, int]] = deque([(source, 0)])
    while queue:
        current, dist = queue.popleft()
        for neighbour in adjacency.get(current, ()):
            if neighbour == target:
                return dist + 1
            if neighbour not in seen:
                seen.add(neighbour)
                queue.append((neighbour, dist + 1))
    return None


# ── Summary rows for pandas ──────────────────────────────────────────────── #


def summarize(
    results: Sequence[RetrievalCaseResult],
    group_keys: tuple[str, ...] = ("method",),
    room_by_node: Mapping[str, str | None] | None = None,
) -> list[dict]:
    """One summary dict per group (rows for a ``pandas.DataFrame``).

    Groups by the given ``RetrievalCaseResult`` attributes (e.g.
    ``("method",)`` or ``("method", "query_type")``) and reports the offline
    metric suite per group.
    """
    groups: dict[tuple, list[RetrievalCaseResult]] = {}
    for r in results:
        key = tuple(getattr(r, k) for k in group_keys)
        groups.setdefault(key, []).append(r)

    rows: list[dict] = []
    for key in sorted(groups, key=str):
        items = groups[key]
        row: dict = dict(zip(group_keys, key))
        row.update({
            "n_queries": len(items),
            "n_negative": sum(1 for r in items if r.is_negative),
            "recall_at_1": recall_at_k(items, 1),
            "recall_at_3": recall_at_k(items, 3),
            "recall_at_5": recall_at_k(items, 5),
            "mean_reciprocal_rank": mean_reciprocal_rank(items),
            "mean_rank_first_valid": mean_rank(items),
            "negative_rejection_rate": negative_rejection_rate(items),
            "room_false_positive_rate": room_false_positive_rate(items),
            "mean_retrieval_latency_ms": mean_latency_s(items) * 1000.0,
        })
        if room_by_node is not None:
            row["nearby_success"] = nearby_success_rate(items, room_by_node)
        rows.append(row)
    return rows


def results_to_rows(results: Sequence[RetrievalCaseResult]) -> list[dict]:
    """Flat per-case dicts (detailed CSV/Parquet export)."""
    return [
        {
            "query_id": r.query_id,
            "method": r.method,
            "dataset_id": r.dataset_id,
            "scene_id": r.scene_id or None,
            "query_type": r.query_type,
            "language": r.language,
            "is_negative": r.is_negative,
            "target_visible": r.target_visible,
            "room_false_positive": (
                not r.rejected if not r.target_visible else NAN
            ),
            "valid_node_ids": "|".join(r.valid_node_ids),
            "predicted_node_id": r.top1_id,
            "rank_first_valid": r.rank,
            "recall_at_1": (
                bool(r.rank == 1) if not r.is_negative and r.target_visible else NAN
            ),
            "recall_at_3": (
                bool(r.rank and r.rank <= 3)
                if not r.is_negative and r.target_visible else NAN
            ),
            "recall_at_5": (
                bool(r.rank and r.rank <= 5)
                if not r.is_negative and r.target_visible else NAN
            ),
            "reciprocal_rank": (
                (1.0 / r.rank if r.rank else 0.0)
                if not r.is_negative and r.target_visible else NAN
            ),
            "semantic_success": (
                bool(r.rank == 1)
                if not r.is_negative and r.target_visible else r.rejected
            ),
            "rejected": r.rejected,
            "retrieval_latency_ms": r.latency_s * 1000.0,
            "global_similarity": r.score_components.get("global_similarity", NAN),
            "object_match_score": r.score_components.get("object_match_score", NAN),
            "crop_similarity": r.score_components.get("crop_similarity", NAN),
            "relation_match_score": r.score_components.get("relation_match_score", NAN),
            "room_match_score": r.score_components.get("room_match_score", NAN),
            "hybrid_score": r.scores[0] if r.scores else NAN,
            "top5_ids": "|".join(r.ranked_ids[:5]),
        }
        for r in results
    ]
