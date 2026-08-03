#!/usr/bin/env python3
"""Validate ROS 2 campaigns and export TFM-ready metric tables."""
from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = next(
    candidate
    for candidate in (Path(__file__).resolve(), *Path(__file__).resolve().parents)
    if (candidate / "semantic_navigation_ws" / "src").is_dir()
)
sys.path.insert(0, str(REPO_ROOT / "experiments" / "shared"))

from notebook_bootstrap import bootstrap_simulation  # noqa: E402
from semantic_evaluation.core.campaign_analysis import (  # noqa: E402
    aggregate_campaign_levels,
    discover_runs,
    export_campaign_analysis,
    load_campaign_cases,
)
from semantic_evaluation.core.config_validation import expand_path  # noqa: E402


def main() -> int:
    ctx = bootstrap_simulation()
    campaigns_root, missing = expand_path(
        ctx["config"]["paths"]["campaigns_root"], ctx["repo_root"]
    )
    if missing or campaigns_root is None:
        raise RuntimeError(f"unresolved campaigns path variables: {missing}")
    runs = discover_runs(campaigns_root)
    if not runs:
        print(f"No campaigns found below {campaigns_root}", file=sys.stderr)
        return 2

    cases, issues = load_campaign_cases(runs, ctx["frozen_config_hash"])
    if cases.empty:
        print("No compatible campaign cases were found.", file=sys.stderr)
        for issue in issues:
            print(f"- {issue}", file=sys.stderr)
        return 3

    analysis = ctx["config"]["campaign_analysis"]
    summaries = aggregate_campaign_levels(
        cases,
        confidence_level=float(analysis["confidence_level"]),
        bootstrap_samples=int(analysis["bootstrap_samples"]),
        seed=42,
    )
    results_root, missing = expand_path(
        ctx["config"]["paths"]["results_root"], ctx["repo_root"]
    )
    if missing or results_root is None:
        raise RuntimeError(f"unresolved results path variables: {missing}")
    results_root.mkdir(parents=True, exist_ok=True)

    # CSV tables are convenient for LibreOffice, LaTeX and plotting tools.
    csv_root = results_root / "csv"
    csv_root.mkdir(parents=True, exist_ok=True)
    cases.to_csv(csv_root / "all_cases.csv", index=False)
    for level, frame in summaries.items():
        frame.to_csv(csv_root / f"{level}_summary.csv", index=False)

    metric_columns = [
        "recall_at_1",
        "recall_at_3",
        "recall_at_5",
        "reciprocal_rank",
        "semantic_success",
        "nearby_semantic_success",
        "navigation_success",
        "end_to_end_success",
        "retrieval_latency_ms",
        "navigation_time_s",
        "spl",
        "path_length_m",
        "optimal_path_length_m",
        "final_distance_m",
        "number_of_recoveries",
    ]
    available_metrics = [column for column in metric_columns if column in cases]
    breakdown = (
        cases.groupby(
            ["scene_id", "method", "query_type", "language"], dropna=False
        )[available_metrics]
        .mean(numeric_only=True)
        .reset_index()
    )
    breakdown.to_csv(csv_root / "metrics_by_query_type.csv", index=False)
    failures = (
        cases.groupby(["scene_id", "method", "failure_type"], dropna=False)
        .size()
        .rename("n_cases")
        .reset_index()
    )
    failures.to_csv(csv_root / "failure_taxonomy.csv", index=False)

    # Parquet keeps types intact for the campaign notebook.  CSV exports above
    # remain available if the optional parquet engine is not installed.
    try:
        export_campaign_analysis(results_root, cases, summaries, issues)
    except ImportError as exc:
        print(f"Parquet export skipped: {exc}", file=sys.stderr)

    print(f"Frozen configuration hash: {ctx['frozen_config_hash']}")
    print(f"Runs discovered: {len(runs)}; valid cases: {len(cases)}")
    print(f"CSV results: {csv_root}")
    if issues:
        print("Validation issues:")
        for issue in issues:
            print(f"- {issue}")
    print(summaries["scenes"].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
