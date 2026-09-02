"""Pure accuracy / aggregation logic — no rclpy / ROS message imports.

Room-level accuracy has two sources:
- **graph**: the knowledge graph links each waypoint to its parent room via a
  ``CONTAINS`` room→waypoint edge; ``build_room_map`` extracts that mapping
  from a snapshot's edge list and ``room_of`` resolves a waypoint's room.
- **label** (fallback / legacy): waypoint ids follow ``<room>[<sep><instance>]``
  and the room portion is derived from the id (``room_key``). Waypoints without
  a room edge fall back to this heuristic automatically.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Mapping

from semantic_evaluation.core.metrics import TestCaseResult

# Room-key strategies.
STRATEGY_STRIP_LAST = "strip_last"   # cocina_01 -> cocina, sala_estar_02 -> sala_estar
STRATEGY_FIRST_TOKEN = "first_token"  # cocina_01 -> cocina, sala_estar_02 -> sala
SUPPORTED_STRATEGIES = (STRATEGY_STRIP_LAST, STRATEGY_FIRST_TOKEN)

# Room sources for room-level accuracy.
ROOM_SOURCE_GRAPH = "graph"
ROOM_SOURCE_LABEL = "label"
SUPPORTED_ROOM_SOURCES = (ROOM_SOURCE_GRAPH, ROOM_SOURCE_LABEL)

# Edge type linking a room (source) to a waypoint (target) in the graph.
ROOM_EDGE_TYPE = "CONTAINS"


def build_room_map(
    edges: Iterable[tuple[str, str, str]],
    waypoint_ids: set[str],
) -> dict[str, str]:
    """Extract waypoint→room from snapshot edges ``(type, source, target)``.

    A CONTAINS edge whose *target* is a waypoint must come from a room —
    waypoint→object edges have the waypoint as *source*, so requiring the
    source to not be a waypoint filters them out without needing node types.
    First room wins if a waypoint somehow has several.
    """
    room_map: dict[str, str] = {}
    for edge_type, source, target in edges:
        if edge_type != ROOM_EDGE_TYPE:
            continue
        if target in waypoint_ids and source not in waypoint_ids:
            room_map.setdefault(target, source)
    return room_map


def room_of(
    node_id: str,
    room_map: Mapping[str, str] | None,
    separator: str = "_",
    strategy: str = STRATEGY_STRIP_LAST,
) -> str:
    """Room of a waypoint: graph parent if mapped, else the label heuristic."""
    if room_map is not None and node_id in room_map:
        return room_map[node_id]
    return room_key(node_id, separator, strategy)


def room_key(
    node_id: str,
    separator: str = "_",
    strategy: str = STRATEGY_STRIP_LAST,
) -> str:
    """Derive the room identifier from a waypoint id.

    ``strip_last`` drops the trailing instance token, preserving multi-token room
    names: ``cocina_01`` -> ``cocina``, ``sala_estar_02`` -> ``sala_estar``.
    ``first_token`` keeps only the first token. Ids without a separator (or
    empty) are returned unchanged.
    """
    if not node_id:
        return node_id
    parts = node_id.split(separator)
    if len(parts) <= 1:
        return node_id
    if strategy == STRATEGY_STRIP_LAST:
        return separator.join(parts[:-1])
    if strategy == STRATEGY_FIRST_TOKEN:
        return parts[0]
    raise ValueError(
        f"Unknown room-key strategy '{strategy}'. "
        f"Supported: {SUPPORTED_STRATEGIES}"
    )


def is_top1_correct(predicted_node_id: str, expected_node_id: str) -> bool:
    """Exact waypoint match (both non-empty)."""
    return bool(predicted_node_id) and predicted_node_id == expected_node_id


def is_room_level_correct(
    predicted_node_id: str,
    expected_node_id: str,
    separator: str = "_",
    strategy: str = STRATEGY_STRIP_LAST,
    room_map: Mapping[str, str] | None = None,
) -> bool:
    """Same-room match: graph parent rooms when mapped, label keys otherwise."""
    if not predicted_node_id or not expected_node_id:
        return False
    return room_of(predicted_node_id, room_map, separator, strategy) == room_of(
        expected_node_id, room_map, separator, strategy
    )


def annotate_accuracy(
    result: TestCaseResult,
    separator: str = "_",
    strategy: str = STRATEGY_STRIP_LAST,
    room_map: Mapping[str, str] | None = None,
) -> TestCaseResult:
    """Fill ``top1_correct`` and ``room_correct`` in place and return the result."""
    result.top1_correct = is_top1_correct(
        result.predicted_node_id, result.expected_node_id
    )
    result.room_correct = is_room_level_correct(
        result.predicted_node_id, result.expected_node_id,
        separator, strategy, room_map,
    )
    return result


@dataclass
class AggregateResult:
    """Campaign-level aggregates. Rates are 0..1, the rest are NaN-aware means."""

    n_cases: int = 0
    top1_rate: float = 0.0
    room_rate: float = 0.0
    success_rate: float = 0.0
    room_false_positive_rate: float = float("nan")
    mean_visual_extraction_s: float = float("nan")
    mean_retrieval_s: float = float("nan")
    mean_navigation_s: float = float("nan")
    mean_end_to_end_s: float = float("nan")
    mean_cpu_percent: float = float("nan")
    mean_ram_used_mb: float = float("nan")
    mean_total_nodes: float = float("nan")
    mean_total_edges: float = float("nan")
    mean_score: float = float("nan")


def _nan_mean(values: Iterable[float]) -> float:
    """Mean ignoring NaN; NaN when there is no valid sample."""
    present = [v for v in values if v is not None and not math.isnan(float(v))]
    if not present:
        return float("nan")
    return sum(present) / len(present)


def aggregate(
    results: list[TestCaseResult],
    separator: str = "_",
    strategy: str = STRATEGY_STRIP_LAST,
    room_map: Mapping[str, str] | None = None,
) -> AggregateResult:
    """Aggregate a list of (already annotated or not) case results.

    Accuracy is recomputed defensively so callers need not pre-annotate —
    pass ``room_map`` here too or the graph-based room accuracy would be
    silently overwritten by the label heuristic. Rates are fractions of all
    cases; latency / hardware / graph figures are NaN-aware means so
    decision-only navigation (NaN) does not skew the totals.
    """
    n = len(results)
    if n == 0:
        return AggregateResult()

    annotated = [annotate_accuracy(r, separator, strategy, room_map) for r in results]

    invisible = [result for result in annotated if not result.target_visible]
    return AggregateResult(
        n_cases=n,
        top1_rate=sum(1 for r in annotated if r.top1_correct) / n,
        room_rate=sum(1 for r in annotated if r.room_correct) / n,
        success_rate=sum(1 for r in annotated if r.success) / n,
        room_false_positive_rate=(
            sum(1 for result in invisible if result.accepted) / len(invisible)
            if invisible else float("nan")
        ),
        mean_visual_extraction_s=_nan_mean(
            r.latency.visual_extraction_s for r in annotated
        ),
        mean_retrieval_s=_nan_mean(r.latency.retrieval_s for r in annotated),
        mean_navigation_s=_nan_mean(r.latency.navigation_s for r in annotated),
        mean_end_to_end_s=_nan_mean(r.latency.end_to_end_s for r in annotated),
        mean_cpu_percent=_nan_mean(r.hardware.cpu_percent for r in annotated),
        mean_ram_used_mb=_nan_mean(r.hardware.ram_used_mb for r in annotated),
        mean_total_nodes=_nan_mean(r.graph.total_nodes for r in annotated),
        mean_total_edges=_nan_mean(r.graph.total_edges for r in annotated),
        mean_score=_nan_mean(r.score for r in annotated),
    )
