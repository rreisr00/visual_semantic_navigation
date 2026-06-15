"""Pure-core unit tests — run with plain pytest, no ROS environment.

Covers room_key, top-1 / room accuracy, NaN-aware end-to-end latency,
aggregation and the fixed-schema CSV export (including the aggregate row).
"""
import math

import pytest

from semantic_evaluation.core.csv_export import (
    AGGREGATE_ROW_ID,
    CSV_COLUMNS,
    build_rows,
    write_csv,
)
from semantic_evaluation.core.evaluation_logic import (
    STRATEGY_FIRST_TOKEN,
    aggregate,
    annotate_accuracy,
    is_room_level_correct,
    is_top1_correct,
    room_key,
)
from semantic_evaluation.core.metrics import (
    GraphContext,
    HardwareSample,
    LatencyBreakdown,
    TestCaseResult,
)


# ── room_key ──────────────────────────────────────────────────────────────── #

def test_room_key_strip_last_single_token_instance():
    assert room_key("cocina_01") == "cocina"


def test_room_key_strip_last_multi_token_room():
    assert room_key("sala_estar_02") == "sala_estar"


def test_room_key_first_token_strategy():
    assert room_key("sala_estar_02", strategy=STRATEGY_FIRST_TOKEN) == "sala"


def test_room_key_custom_separator():
    assert room_key("cocina-01", separator="-") == "cocina"


def test_room_key_no_separator_returns_as_is():
    assert room_key("cocina") == "cocina"


def test_room_key_empty():
    assert room_key("") == ""


def test_room_key_unknown_strategy_raises():
    with pytest.raises(ValueError):
        room_key("cocina_01", strategy="nope")


# ── top-1 / room accuracy ─────────────────────────────────────────────────── #

def test_is_top1_correct():
    assert is_top1_correct("cocina_01", "cocina_01")
    assert not is_top1_correct("cocina_01", "cocina_02")
    assert not is_top1_correct("", "")


def test_is_room_level_correct():
    assert is_room_level_correct("cocina_01", "cocina_02")
    assert not is_room_level_correct("cocina_01", "sala_estar_02")
    assert not is_room_level_correct("", "cocina_01")


def test_annotate_accuracy_sets_flags():
    r = _result("c", predicted="cocina_02", expected="cocina_01", success=True)
    annotate_accuracy(r)
    assert r.top1_correct is False
    assert r.room_correct is True


# ── NaN-aware end-to-end latency ──────────────────────────────────────────── #

def test_end_to_end_sums_three_phases():
    lb = LatencyBreakdown(0.1, 0.2, 0.3)
    assert lb.end_to_end_s == pytest.approx(0.6)


def test_end_to_end_ignores_nan_navigation_decision_only():
    lb = LatencyBreakdown(0.1, 0.2, float("nan"))
    assert lb.end_to_end_s == pytest.approx(0.3)


def test_end_to_end_all_nan_is_nan():
    lb = LatencyBreakdown(float("nan"), float("nan"), float("nan"))
    assert math.isnan(lb.end_to_end_s)


# ── aggregation ───────────────────────────────────────────────────────────── #

def test_aggregate_rates_and_nan_aware_means():
    results = [
        _result(
            "c1", predicted="cocina_01", expected="cocina_01", success=True,
            latency=LatencyBreakdown(0.1, 0.2, 0.3),
            hardware=HardwareSample(10.0, 1000.0),
            graph=GraphContext(4, 2),
        ),
        _result(
            "c2", predicted="cocina_09", expected="cocina_01", success=False,
            latency=LatencyBreakdown(0.3, 0.4, float("nan")),  # decision-only
            hardware=HardwareSample(30.0, 2000.0),
            graph=GraphContext(6, 4),
        ),
    ]
    agg = aggregate(results)

    assert agg.n_cases == 2
    assert agg.top1_rate == pytest.approx(0.5)   # only c1 exact
    assert agg.room_rate == pytest.approx(1.0)   # both cocina
    assert agg.success_rate == pytest.approx(0.5)
    assert agg.mean_visual_extraction_s == pytest.approx(0.2)
    assert agg.mean_navigation_s == pytest.approx(0.3)   # NaN of c2 ignored
    assert agg.mean_end_to_end_s == pytest.approx((0.6 + 0.7) / 2)
    assert agg.mean_cpu_percent == pytest.approx(20.0)
    assert agg.mean_total_nodes == pytest.approx(5.0)


def test_aggregate_empty():
    agg = aggregate([])
    assert agg.n_cases == 0
    assert agg.top1_rate == 0.0


# ── CSV export ────────────────────────────────────────────────────────────── #

def test_build_rows_schema_and_aggregate_row():
    results = [
        _result(
            "c1", predicted="cocina_01", expected="cocina_01", success=True,
            latency=LatencyBreakdown(0.1, 0.2, 0.3),
        ),
        _result(
            "c2", predicted="sala_estar_01", expected="cocina_01", success=False,
            latency=LatencyBreakdown(0.1, 0.2, float("nan")),
        ),
    ]
    rows = build_rows(results)

    # one row per case + aggregate row
    assert len(rows) == 3
    assert all(set(r.keys()) == set(CSV_COLUMNS) for r in rows)
    assert rows[-1]["case_id"] == AGGREGATE_ROW_ID

    # per-case booleans serialised as 0/1
    assert rows[0]["top1_correct"] == "1"
    assert rows[1]["room_correct"] == "0"

    # NaN navigation rendered as empty string
    assert rows[1]["navigation_s"] == ""

    # aggregate accuracy columns are rates in 0..1
    assert float(rows[-1]["top1_correct"]) == pytest.approx(0.5)
    assert float(rows[-1]["success"]) == pytest.approx(0.5)


def test_write_csv_roundtrip(tmp_path):
    import csv

    results = [
        _result(
            "c1", predicted="cocina_01", expected="cocina_01", success=True,
            latency=LatencyBreakdown(0.1, 0.2, 0.3),
        ),
    ]
    out = tmp_path / "out.csv"
    write_csv(str(out), results)

    with out.open() as fh:
        reader = csv.DictReader(fh)
        assert reader.fieldnames == CSV_COLUMNS
        rows = list(reader)

    assert len(rows) == 2  # 1 case + aggregate
    assert rows[-1]["case_id"] == AGGREGATE_ROW_ID


# ── helpers ───────────────────────────────────────────────────────────────── #

def _result(
    case_id,
    predicted,
    expected,
    success,
    latency=None,
    hardware=None,
    graph=None,
):
    return TestCaseResult(
        case_id=case_id,
        query="q",
        query_kind="text",
        expected_node_id=expected,
        predicted_node_id=predicted,
        success=success,
        latency=latency or LatencyBreakdown(),
        hardware=hardware or HardwareSample(),
        graph=graph or GraphContext(),
    )
