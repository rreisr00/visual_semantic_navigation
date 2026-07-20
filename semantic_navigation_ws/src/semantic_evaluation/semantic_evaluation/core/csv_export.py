"""CSV export of evaluation results — pure Python, no rclpy.

Schema is fixed (``CSV_COLUMNS``): one row per case followed by a single
``__AGGREGATE_MEAN__`` row. On the aggregate row the accuracy columns hold rates
in ``0..1`` while the latency / hardware / graph columns hold NaN-aware means.
"""
from __future__ import annotations

import csv
import math
from typing import Any, Mapping

from semantic_evaluation.core.evaluation_logic import (
    STRATEGY_STRIP_LAST,
    AggregateResult,
    aggregate,
)
from semantic_evaluation.core.metrics import TestCaseResult

AGGREGATE_ROW_ID = "__AGGREGATE_MEAN__"

CSV_COLUMNS: list[str] = [
    "case_id",
    "query",
    "query_kind",
    "expected_node_id",
    "predicted_node_id",
    "top1_correct",
    "room_correct",
    "success",
    "score",
    "visual_extraction_s",
    "retrieval_s",
    "navigation_s",
    "end_to_end_s",
    "cpu_percent",
    "ram_used_mb",
    "total_nodes",
    "total_edges",
]

_FLOAT_FMT = "{:.6f}"


def _fmt(value: Any) -> str:
    """Format a float for CSV: empty string for NaN, fixed precision otherwise."""
    if value is None:
        return ""
    if isinstance(value, float):
        if math.isnan(value):
            return ""
        return _FLOAT_FMT.format(value)
    return str(value)


def case_to_row(result: TestCaseResult) -> dict[str, str]:
    """One CSV row for a single resolved case (booleans as 0/1)."""
    return {
        "case_id": result.case_id,
        "query": result.query,
        "query_kind": result.query_kind,
        "expected_node_id": result.expected_node_id,
        "predicted_node_id": result.predicted_node_id,
        "top1_correct": "1" if result.top1_correct else "0",
        "room_correct": "1" if result.room_correct else "0",
        "success": "1" if result.success else "0",
        "score": _fmt(result.score),
        "visual_extraction_s": _fmt(result.latency.visual_extraction_s),
        "retrieval_s": _fmt(result.latency.retrieval_s),
        "navigation_s": _fmt(result.latency.navigation_s),
        "end_to_end_s": _fmt(result.latency.end_to_end_s),
        "cpu_percent": _fmt(result.hardware.cpu_percent),
        "ram_used_mb": _fmt(result.hardware.ram_used_mb),
        "total_nodes": str(result.graph.total_nodes),
        "total_edges": str(result.graph.total_edges),
    }


def aggregate_to_row(agg: AggregateResult) -> dict[str, str]:
    """The trailing ``__AGGREGATE_MEAN__`` row: rates for accuracy, means else."""
    return {
        "case_id": AGGREGATE_ROW_ID,
        "query": "",
        "query_kind": "",
        "expected_node_id": "",
        "predicted_node_id": "",
        "top1_correct": _fmt(agg.top1_rate),
        "room_correct": _fmt(agg.room_rate),
        "success": _fmt(agg.success_rate),
        "score": _fmt(agg.mean_score),
        "visual_extraction_s": _fmt(agg.mean_visual_extraction_s),
        "retrieval_s": _fmt(agg.mean_retrieval_s),
        "navigation_s": _fmt(agg.mean_navigation_s),
        "end_to_end_s": _fmt(agg.mean_end_to_end_s),
        "cpu_percent": _fmt(agg.mean_cpu_percent),
        "ram_used_mb": _fmt(agg.mean_ram_used_mb),
        "total_nodes": _fmt(agg.mean_total_nodes),
        "total_edges": _fmt(agg.mean_total_edges),
    }


def build_rows(
    results: list[TestCaseResult],
    separator: str = "_",
    strategy: str = STRATEGY_STRIP_LAST,
    room_map: Mapping[str, str] | None = None,
) -> list[dict[str, str]]:
    """Per-case rows + the final aggregate row, all annotated for accuracy.

    ``room_map`` must be forwarded here: ``aggregate`` re-annotates each
    result defensively, so omitting it would overwrite graph-based
    ``room_correct`` values with the label heuristic.
    """
    agg = aggregate(results, separator, strategy, room_map)  # also annotates
    rows = [case_to_row(r) for r in results]
    rows.append(aggregate_to_row(agg))
    return rows


def write_csv(
    path: str,
    results: list[TestCaseResult],
    separator: str = "_",
    strategy: str = STRATEGY_STRIP_LAST,
    room_map: Mapping[str, str] | None = None,
) -> str:
    """Write the fixed-schema CSV to ``path`` and return that path."""
    rows = build_rows(results, separator, strategy, room_map)
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return path
