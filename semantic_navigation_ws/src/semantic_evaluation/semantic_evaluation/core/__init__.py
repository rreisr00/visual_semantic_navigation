"""Pure evaluation core — no rclpy / ROS message imports.

Everything in this subpackage is unit-testable with plain ``pytest`` and no ROS
environment. The wrapper nodes orchestrate I/O and delegate all logic here.
"""
from semantic_evaluation.core.csv_export import (
    AGGREGATE_ROW_ID,
    CSV_COLUMNS,
    build_rows,
    write_csv,
)
from semantic_evaluation.core.evaluation_logic import (
    AggregateResult,
    aggregate,
    annotate_accuracy,
    is_room_level_correct,
    is_top1_correct,
    room_key,
    STRATEGY_FIRST_TOKEN,
    STRATEGY_STRIP_LAST,
    SUPPORTED_STRATEGIES,
)
from semantic_evaluation.core.hardware import HardwareSampler
from semantic_evaluation.core.metrics import (
    GraphContext,
    HardwareSample,
    LatencyBreakdown,
    TestCaseResult,
)

__all__ = [
    "AGGREGATE_ROW_ID",
    "CSV_COLUMNS",
    "build_rows",
    "write_csv",
    "AggregateResult",
    "aggregate",
    "annotate_accuracy",
    "is_room_level_correct",
    "is_top1_correct",
    "room_key",
    "STRATEGY_FIRST_TOKEN",
    "STRATEGY_STRIP_LAST",
    "SUPPORTED_STRATEGIES",
    "HardwareSampler",
    "GraphContext",
    "HardwareSample",
    "LatencyBreakdown",
    "TestCaseResult",
]
