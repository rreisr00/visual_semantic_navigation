"""Pure data types for evaluation metrics — no rclpy / ROS message imports.

These dataclasses are the only contract between the ROS wrapper nodes and the
pure evaluation logic, so the whole of ``core`` stays unit-testable with plain
``pytest`` and no ROS environment.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

NAN = float("nan")


def _is_present(value: float | None) -> bool:
    """True when ``value`` is a real (non-None, non-NaN) number."""
    return value is not None and not math.isnan(value)


@dataclass
class LatencyBreakdown:
    """Per-phase latencies (seconds) returned by the orchestrator.

    ``navigation_s`` is NaN in decision-only mode (no Nav2 drive). Any phase that
    was never reached is left as NaN so it can be ignored by NaN-aware means.
    """

    visual_extraction_s: float = NAN
    retrieval_s: float = NAN
    navigation_s: float = NAN

    @property
    def end_to_end_s(self) -> float:
        """NaN-aware sum of the three phases.

        Missing phases (NaN) are skipped; the result is NaN only when every
        phase is missing. This keeps decision-only runs (navigation = NaN)
        meaningful instead of poisoning the total.
        """
        parts = [self.visual_extraction_s, self.retrieval_s, self.navigation_s]
        present = [p for p in parts if _is_present(p)]
        if not present:
            return NAN
        return float(sum(present))


@dataclass
class HardwareSample:
    """A single hardware utilisation sample. NaN when psutil is unavailable."""

    cpu_percent: float = NAN
    ram_used_mb: float = NAN


@dataclass
class GraphContext:
    """Knowledge-graph size at the moment a test case was resolved."""

    total_nodes: int = 0
    total_edges: int = 0


@dataclass
class TestCaseResult:
    """The full outcome of a single evaluation case.

    Accuracy flags (``top1_correct`` / ``room_correct``) are filled in by
    ``evaluation_logic.annotate_accuracy`` after the case is resolved.
    """

    __test__ = False  # not a pytest test class despite the "Test" prefix

    case_id: str
    query: str                       # query_text, or image_path for image cases
    query_kind: str                  # "text" | "image"
    expected_node_id: str
    predicted_node_id: str
    success: bool
    latency: LatencyBreakdown = field(default_factory=LatencyBreakdown)
    hardware: HardwareSample = field(default_factory=HardwareSample)
    graph: GraphContext = field(default_factory=GraphContext)
    score: float = NAN
    top1_correct: bool = False
    room_correct: bool = False
    query_id: str = ""
    query_type: str = ""
    language: str = ""
    exact_valid_nodes: list[str] = field(default_factory=list)
    nearby_valid_nodes: list[str] = field(default_factory=list)
    rank_first_valid: int | None = None
    is_negative: bool = False
    accepted: bool = False
    semantic_success: bool = False
    nearby_semantic_success: bool = False
    navigation_success: bool | None = None
    end_to_end_success: bool | None = None
    retrieval_latency_ms: float = NAN
    navigation_time_s: float = NAN
    failure_type: str = ""
    nav2_error_code: int = 0
    nav2_error_message: str = ""
    top_k_candidates: list[str] = field(default_factory=list)
    campaign_id: str = ""
    run_id: str = ""
    scene_id: str = ""
    method: str = ""
    start_pose_id: str = ""
    frozen_config_hash: str = ""
    number_of_recoveries: int = 0
    path_length_m: float = NAN
    optimal_path_length_m: float = NAN
    spl: float = NAN
    final_distance_m: float = NAN
    adjustment_distance_m: float = NAN
    goal_validation_status: str = ""
