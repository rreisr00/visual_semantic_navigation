"""Pure accuracy / aggregation logic — no rclpy / ROS message imports.

Room-level accuracy treats waypoint ids as ``<room>[<sep><instance>]`` and
compares the room portion. The split strategy is configurable so the same code
serves ``cocina_01`` (strip_last) and other conventions without edits.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

from semantic_evaluation.core.metrics import TestCaseResult

# Room-key strategies.
STRATEGY_STRIP_LAST = "strip_last"   # cocina_01 -> cocina, sala_estar_02 -> sala_estar
STRATEGY_FIRST_TOKEN = "first_token"  # cocina_01 -> cocina, sala_estar_02 -> sala
SUPPORTED_STRATEGIES = (STRATEGY_STRIP_LAST, STRATEGY_FIRST_TOKEN)


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
) -> bool:
    """Same-room match: the predicted and expected ids share a room key."""
    if not predicted_node_id or not expected_node_id:
        return False
    return room_key(predicted_node_id, separator, strategy) == room_key(
        expected_node_id, separator, strategy
    )


def annotate_accuracy(
    result: TestCaseResult,
    separator: str = "_",
    strategy: str = STRATEGY_STRIP_LAST,
) -> TestCaseResult:
    """Fill ``top1_correct`` and ``room_correct`` in place and return the result."""
    result.top1_correct = is_top1_correct(
        result.predicted_node_id, result.expected_node_id
    )
    result.room_correct = is_room_level_correct(
        result.predicted_node_id, result.expected_node_id, separator, strategy
    )
    return result


@dataclass
class AggregateResult:
    """Campaign-level aggregates. Rates are 0..1, the rest are NaN-aware means."""

    n_cases: int = 0
    top1_rate: float = 0.0
    room_rate: float = 0.0
    success_rate: float = 0.0
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
) -> AggregateResult:
    """Aggregate a list of (already annotated or not) case results.

    Accuracy is recomputed defensively so callers need not pre-annotate. Rates
    are fractions of all cases; latency / hardware / graph figures are NaN-aware
    means so decision-only navigation (NaN) does not skew the totals.
    """
    n = len(results)
    if n == 0:
        return AggregateResult()

    annotated = [annotate_accuracy(r, separator, strategy) for r in results]

    return AggregateResult(
        n_cases=n,
        top1_rate=sum(1 for r in annotated if r.top1_correct) / n,
        room_rate=sum(1 for r in annotated if r.room_correct) / n,
        success_rate=sum(1 for r in annotated if r.success) / n,
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
