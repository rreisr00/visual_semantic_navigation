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
    "query_id",
    "query_type",
    "language",
    "exact_valid_nodes",
    "nearby_valid_nodes",
    "rank_first_valid",
    "is_negative",
    "accepted",
    "semantic_success",
    "nearby_semantic_success",
    "navigation_success",
    "end_to_end_success",
    "retrieval_latency_ms",
    "navigation_time_s",
    "failure_type",
    "nav2_error_code",
    "nav2_error_message",
    "top_k_candidates",
    "campaign_id",
    "run_id",
    "scene_id",
    "method",
    "start_pose_id",
    "frozen_config_hash",
    "number_of_recoveries",
    "path_length_m",
    "optimal_path_length_m",
    "spl",
    "final_distance_m",
    "adjustment_distance_m",
    "goal_validation_status",
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
        "query_id": result.query_id or result.case_id,
        "query_type": result.query_type or result.query_kind,
        "language": result.language,
        "exact_valid_nodes": "|".join(result.exact_valid_nodes),
        "nearby_valid_nodes": "|".join(result.nearby_valid_nodes),
        "rank_first_valid": _fmt(result.rank_first_valid),
        "is_negative": "1" if result.is_negative else "0",
        "accepted": "1" if result.accepted else "0",
        "semantic_success": "1" if result.semantic_success else "0",
        "nearby_semantic_success": (
            "1" if result.nearby_semantic_success else "0"
        ),
        "navigation_success": (
            "" if result.navigation_success is None
            else "1" if result.navigation_success else "0"
        ),
        "end_to_end_success": (
            "" if result.end_to_end_success is None
            else "1" if result.end_to_end_success else "0"
        ),
        "retrieval_latency_ms": _fmt(result.retrieval_latency_ms),
        "navigation_time_s": _fmt(result.navigation_time_s),
        "failure_type": result.failure_type,
        "nav2_error_code": str(result.nav2_error_code),
        "nav2_error_message": result.nav2_error_message,
        "top_k_candidates": "|".join(result.top_k_candidates),
        "campaign_id": result.campaign_id,
        "run_id": result.run_id,
        "scene_id": result.scene_id,
        "method": result.method,
        "start_pose_id": result.start_pose_id,
        "frozen_config_hash": result.frozen_config_hash,
        "number_of_recoveries": str(result.number_of_recoveries),
        "path_length_m": _fmt(result.path_length_m),
        "optimal_path_length_m": _fmt(result.optimal_path_length_m),
        "spl": _fmt(result.spl),
        "final_distance_m": _fmt(result.final_distance_m),
        "adjustment_distance_m": _fmt(result.adjustment_distance_m),
        "goal_validation_status": result.goal_validation_status,
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
        "query_id": "",
        "query_type": "",
        "language": "",
        "exact_valid_nodes": "",
        "nearby_valid_nodes": "",
        "rank_first_valid": "",
        "is_negative": "",
        "accepted": "",
        "semantic_success": "",
        "nearby_semantic_success": "",
        "navigation_success": "",
        "end_to_end_success": "",
        "retrieval_latency_ms": "",
        "navigation_time_s": "",
        "failure_type": "",
        "nav2_error_code": "",
        "nav2_error_message": "",
        "top_k_candidates": "",
        "campaign_id": "",
        "run_id": "",
        "scene_id": "",
        "method": "",
        "start_pose_id": "",
        "frozen_config_hash": "",
        "number_of_recoveries": "",
        "path_length_m": "",
        "optimal_path_length_m": "",
        "spl": "",
        "final_distance_m": "",
        "adjustment_distance_m": "",
        "goal_validation_status": "",
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
