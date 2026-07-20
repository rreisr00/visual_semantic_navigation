import pytest

pd = pytest.importorskip("pandas", reason="campaign analysis requires python3-pandas")

from semantic_evaluation.core.campaign_analysis import (
    aggregate_campaign_levels,
    classify_failure,
    discover_runs,
    load_campaign_cases,
    validate_campaign_frame,
)
from semantic_evaluation.core.evaluation_statistics import (
    bootstrap_interval,
    normalize_predicate,
    relation_metrics,
)
from semantic_evaluation.core.experimental_schemas import CampaignSpec
from semantic_navigation_core.types import SpatialRelation


def _campaign(run_id="run_001"):
    return CampaignSpec.from_mapping({
        "campaign_id": "campaign_a", "scene_id": "scene_01", "run_id": run_id,
        "seed": 7, "method": "single_view_siglip", "start_pose_id": "start_a",
        "query_suite_id": "suite_a", "frozen_config_hash": "hash",
        "git_commit": "commit", "timestamp": "2026-01-01T00:00:00Z",
        "status": "complete", "success_semantics": "navigation_success",
    })


def test_relation_normalization_and_per_predicate_metrics():
    assert normalize_predicate("next to") == "NEAR"
    rows = relation_metrics(
        [SpatialRelation("chair", "NEAR", "table")],
        [SpatialRelation("table", "next to", "chair"),
         SpatialRelation("cup", "ABOVE", "table")],
    )
    by_predicate = {row["predicate"]: row for row in rows}
    assert by_predicate["NEAR"]["f1"] == 1.0
    assert by_predicate["ABOVE"]["recall"] == 0.0


def test_bootstrap_interval_and_single_sample_uncertainty():
    mean, low, high = bootstrap_interval([True, False, True], samples=200, seed=3)
    assert mean == pytest.approx(2 / 3)
    assert low <= mean <= high
    _, low, high = bootstrap_interval([True])
    assert pd.isna(low) and pd.isna(high)


def test_campaign_csv_uses_names_not_column_order_and_separates_success():
    frame = pd.DataFrame([{
        "success": 1, "predicted_node_id": "node_a", "case_id": "query_a",
        "expected_node_id": "node_a", "retrieval_s": 0.02, "navigation_s": 3.0,
    }])[["navigation_s", "case_id", "success", "expected_node_id",
         "retrieval_s", "predicted_node_id"]]
    canonical, issues = validate_campaign_frame(frame, _campaign(), "evaluation.csv")
    assert not issues
    assert canonical.loc[0, "semantic_success"]
    assert canonical.loc[0, "navigation_success"]
    assert canonical.loc[0, "end_to_end_success"]
    assert canonical.loc[0, "retrieval_latency_ms"] == pytest.approx(20.0)


def test_legacy_success_without_semantics_is_not_assumed():
    campaign = _campaign()
    campaign = CampaignSpec(**{**campaign.__dict__, "success_semantics": None})
    frame = pd.DataFrame([{"case_id": "q", "predicted_node_id": "a",
                           "expected_node_id": "a", "success": 1}])
    canonical, issues = validate_campaign_frame(frame, campaign, "evaluation.csv")
    assert canonical.loc[0, "navigation_success"] is None
    assert any("not interpreted" in issue for issue in issues)


def test_failure_taxonomy_and_case_run_campaign_scene_aggregation():
    assert classify_failure({"semantic_success": False, "navigation_success": True}) == \
        "incorrect_node_navigation_completed"
    frames = []
    for index, ok in enumerate((True, False), start=1):
        frame = pd.DataFrame([{"case_id": f"q{index}", "predicted_node_id": "a",
                               "expected_node_id": "a", "success": ok}])
        canonical, _ = validate_campaign_frame(frame, _campaign(f"run_{index:03d}"), "evaluation.csv")
        frames.append(canonical)
    cases = pd.concat(frames, ignore_index=True)
    summaries = aggregate_campaign_levels(cases, bootstrap_samples=100)
    assert len(summaries["runs"]) == 2
    assert summaries["campaigns"].loc[0, "n_runs"] == 2
    assert len(summaries["scenes"]) == 1


def test_campaign_loader_requires_matching_manifest_and_frozen_hash(tmp_path):
    run_root = tmp_path / "scene_01" / "run_001"
    run_root.mkdir(parents=True)
    campaign = _campaign()
    import json
    import yaml

    (run_root / "campaign.yaml").write_text(
        yaml.safe_dump(campaign.__dict__, sort_keys=True), encoding="utf-8"
    )
    manifest = {
        key: getattr(campaign, key)
        for key in (
            "campaign_id", "scene_id", "run_id", "method",
            "query_suite_id", "frozen_config_hash",
        )
    }
    (run_root / "manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    pd.DataFrame([{
        "case_id": "query_a", "predicted_node_id": "node_a",
        "expected_node_id": "node_a", "success": 1,
    }]).to_csv(run_root / "evaluation.csv", index=False)

    runs = discover_runs(tmp_path)
    cases, issues = load_campaign_cases(runs, expected_config_hash="hash")
    assert len(cases) == 1
    assert not issues

    cases, issues = load_campaign_cases(runs, expected_config_hash="other")
    assert cases.empty
    assert any("does not match expected" in issue for issue in issues)
