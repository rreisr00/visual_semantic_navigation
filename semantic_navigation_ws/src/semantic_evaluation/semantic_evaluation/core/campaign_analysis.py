"""Load, validate and aggregate ROS 2 navigation campaigns without rclpy."""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd
import yaml

from semantic_evaluation.core.csv_export import AGGREGATE_ROW_ID
from semantic_evaluation.core.evaluation_statistics import bootstrap_interval
from semantic_evaluation.core.experimental_schemas import CampaignSpec, RunSpec

CANONICAL_CASE_COLUMNS = [
    "method", "dataset_id", "scene_id", "campaign_id", "run_id", "query_id",
    "query_type", "language", "predicted_node_id", "valid_node_ids",
    "rank_first_valid", "recall_at_1", "recall_at_3", "recall_at_5",
    "reciprocal_rank", "semantic_success", "nearby_semantic_success",
    "navigation_success", "end_to_end_success", "retrieval_latency_ms",
    "navigation_time_s", "target_visible", "room_false_positive", "failure_type",
]

_BOOLEAN_COLUMNS = {
    "recall_at_1", "recall_at_3", "recall_at_5", "semantic_success",
    "nearby_semantic_success", "navigation_success", "end_to_end_success",
    "is_negative", "rejected", "target_visible", "room_false_positive",
}
_NUMERIC_COLUMNS = {
    "rank_first_valid", "reciprocal_rank", "retrieval_latency_ms",
    "navigation_time_s", "path_length_m", "optimal_path_length_m",
    "final_distance_m", "topological_distance", "metric_distance",
}


def _load_campaign_spec(path: Path) -> CampaignSpec:
    with path.open(encoding="utf-8") as handle:
        return CampaignSpec.from_mapping(yaml.safe_load(handle) or {}, str(path))


def discover_runs(campaigns_root: str | Path) -> list[RunSpec]:
    root = Path(campaigns_root)
    if not root.is_dir():
        return []
    runs: list[RunSpec] = []
    for campaign_file in sorted(root.rglob("campaign.yaml")):
        run_root = campaign_file.parent
        evaluation_file = run_root / "evaluation.csv"
        manifest_file = run_root / "manifest.json"
        campaign = _load_campaign_spec(campaign_file)
        runs.append(RunSpec(campaign, run_root, evaluation_file, manifest_file))
    return runs


def _bool_value(value: Any) -> bool | None:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    if isinstance(value, bool):
        return value
    # La fila __AGGREGATE_MEAN__ obliga a pandas a tipar estas columnas como
    # float64, asi que tras descartarla los valores siguen siendo 0.0/1.0.
    # Solo 0 y 1 exactos son booleanos: cualquier otro numero indica que una
    # fila agregada se ha colado y debe fallar en vez de redondearse.
    if isinstance(value, (int, float)):
        if value == 0:
            return False
        if value == 1:
            return True
        raise ValueError(f"cannot convert {value!r} to boolean")
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes"}:
        return True
    if normalized in {"0", "false", "no"}:
        return False
    raise ValueError(f"cannot convert {value!r} to boolean")


def _valid_ids(value: Any) -> list[str]:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return []
    if isinstance(value, list):
        return [str(item) for item in value]
    text = str(value).strip()
    if not text:
        return []
    if text.startswith("["):
        parsed = json.loads(text)
        return [str(item) for item in parsed]
    return [item for item in text.replace("|", ",").split(",") if item]


def _rank(predicted: str, valid: list[str], existing: Any) -> int | None:
    if existing is not None and not pd.isna(existing):
        return int(existing)
    return 1 if predicted and predicted in set(valid) else None


def classify_failure(row: Mapping[str, Any], run_status: str = "complete") -> str:
    if run_status not in {"complete", "completed"}:
        return "combined_failure"
    if bool(row.get("data_error", False)):
        return "data_logging_failure"
    if bool(row.get("timeout", False)):
        return "timeout"
    explicit = str(row.get("failure_type") or "")
    if explicit:
        return explicit
    semantic = row.get("semantic_success")
    navigation = row.get("navigation_success")
    stage = str(row.get("navigation_failure_stage") or "").lower()
    if semantic is False and navigation is True:
        return "semantic_mismatch"
    if semantic is False and navigation is False:
        return "combined_failure"
    if semantic is False:
        return "semantic_mismatch"
    if semantic is True and navigation is False and stage == "planning":
        return "no_path"
    if semantic is True and navigation is False and stage == "control":
        return "controller_failure"
    if semantic is True and navigation is False:
        return "planner_failure"
    return "none"


def validate_campaign_frame(
    frame: pd.DataFrame, campaign: CampaignSpec, source: str
) -> tuple[pd.DataFrame, list[str]]:
    """Canonicalize a campaign CSV by column name, never by column order."""
    issues: list[str] = []
    data = frame.copy()
    if "case_id" in data:
        data = data.loc[data["case_id"].astype(str) != AGGREGATE_ROW_ID].copy()
    if "query_id" not in data and "case_id" in data:
        data["query_id"] = data["case_id"].astype(str)
    required = {"query_id", "predicted_node_id"}
    missing = sorted(required - set(data.columns))
    if missing:
        raise ValueError(f"{source}: missing required columns {missing}")
    if "valid_node_ids" not in data:
        if "expected_node_id" in data:
            data["valid_node_ids"] = data["expected_node_id"].fillna("").astype(str)
        else:
            raise ValueError(
                f"{source}: missing valid_node_ids (or legacy expected_node_id)"
            )
    if data["query_id"].duplicated().any():
        duplicates = sorted(
            data.loc[data["query_id"].duplicated(), "query_id"].astype(str).unique()
        )
        issues.append(f"{source}: duplicate query ids {duplicates}")
    data["valid_node_ids"] = data["valid_node_ids"].map(_valid_ids)
    data["predicted_node_id"] = data["predicted_node_id"].fillna("").astype(str)
    existing_rank = data["rank_first_valid"] if "rank_first_valid" in data else [None] * len(data)
    data["rank_first_valid"] = [
        _rank(predicted, valid, rank)
        for predicted, valid, rank in zip(
            data["predicted_node_id"], data["valid_node_ids"], existing_rank
        )
    ]
    data["recall_at_1"] = data["rank_first_valid"].map(lambda value: value == 1)
    data["recall_at_3"] = data["rank_first_valid"].map(
        lambda value: value is not None and value <= 3
    )
    data["recall_at_5"] = data["rank_first_valid"].map(
        lambda value: value is not None and value <= 5
    )
    data["reciprocal_rank"] = data["rank_first_valid"].map(
        lambda value: 1.0 / value if value else 0.0
    )
    if "semantic_success" not in data:
        if "top1_correct" in data:
            data["semantic_success"] = data["top1_correct"].map(_bool_value)
        else:
            data["semantic_success"] = data["recall_at_1"]
    if "navigation_success" not in data:
        if "success" in data and campaign.success_semantics == "navigation_success":
            data["navigation_success"] = data["success"].map(_bool_value)
        else:
            data["navigation_success"] = None
            if "success" in data:
                issues.append(
                    f"{source}: legacy 'success' not interpreted; set "
                    "success_semantics: navigation_success in campaign.yaml"
                )
    if "retrieval_latency_ms" not in data:
        data["retrieval_latency_ms"] = (
            pd.to_numeric(data["retrieval_s"], errors="coerce") * 1000.0
            if "retrieval_s" in data else np.nan
        )
    if "navigation_time_s" not in data:
        data["navigation_time_s"] = (
            pd.to_numeric(data["navigation_s"], errors="coerce")
            if "navigation_s" in data else np.nan
        )
    if "target_visible" not in data:
        data["target_visible"] = (
            data["is_negative"].map(lambda value: not bool(_bool_value(value)))
            if "is_negative" in data else True
        )
    for column in _BOOLEAN_COLUMNS & set(data.columns):
        data[column] = data[column].map(_bool_value)
    if "room_false_positive" not in data:
        accepted = data["accepted"] if "accepted" in data else [False] * len(data)
        data["room_false_positive"] = [
            bool(_bool_value(value)) if visible is False else None
            for visible, value in zip(data["target_visible"], accepted)
        ]
    for column in _NUMERIC_COLUMNS & set(data.columns):
        data[column] = pd.to_numeric(data[column], errors="coerce")
    data["end_to_end_success"] = [
        bool(semantic and navigation)
        if semantic is not None and navigation is not None else None
        for semantic, navigation in zip(data["semantic_success"], data["navigation_success"])
    ]
    metadata = {
        "method": campaign.method,
        "dataset_id": None,
        "scene_id": campaign.scene_id,
        "campaign_id": campaign.campaign_id,
        "run_id": campaign.run_id,
    }
    for column, value in metadata.items():
        data[column] = value
    for column in ("query_type", "language"):
        if column not in data:
            data[column] = "unknown"
    if "nearby_semantic_success" not in data:
        data["nearby_semantic_success"] = None
    data["failure_type"] = [
        classify_failure(row, campaign.status) for row in data.to_dict("records")
    ]
    ordered = [column for column in CANONICAL_CASE_COLUMNS if column in data]
    extras = [column for column in data.columns if column not in ordered]
    return data[ordered + extras], issues


def load_campaign_cases(
    runs: Iterable[RunSpec], expected_config_hash: str | None
) -> tuple[pd.DataFrame, list[str]]:
    frames: list[pd.DataFrame] = []
    issues: list[str] = []
    for run in runs:
        if expected_config_hash and run.campaign.frozen_config_hash != expected_config_hash:
            issues.append(
                f"{run.root}: frozen_config_hash {run.campaign.frozen_config_hash} "
                f"does not match expected {expected_config_hash}"
            )
            continue
        if not run.manifest_file.is_file():
            issues.append(f"{run.manifest_file}: missing manifest.json")
            continue
        try:
            manifest = json.loads(run.manifest_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            issues.append(f"{run.manifest_file}: invalid manifest: {exc}")
            continue
        mismatches = []
        for field_name in (
            "campaign_id", "scene_id", "run_id", "method",
            "query_suite_id", "frozen_config_hash",
        ):
            expected = getattr(run.campaign, field_name)
            if str(manifest.get(field_name, "")) != str(expected):
                mismatches.append(field_name)
        if mismatches:
            issues.append(
                f"{run.manifest_file}: metadata mismatch for {sorted(mismatches)}"
            )
            continue
        if not run.evaluation_file.is_file():
            issues.append(f"{run.evaluation_file}: missing evaluation.csv")
            continue
        frame, frame_issues = validate_campaign_frame(
            pd.read_csv(run.evaluation_file), run.campaign, str(run.evaluation_file)
        )
        frames.append(frame)
        issues.extend(frame_issues)
    if not frames:
        return pd.DataFrame(columns=CANONICAL_CASE_COLUMNS), issues
    return pd.concat(frames, ignore_index=True, sort=False), issues


def _summary(
    frame: pd.DataFrame,
    group_columns: list[str],
    confidence_level: float,
    bootstrap_samples: int,
    seed: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    metrics = [
        "semantic_success", "navigation_success", "end_to_end_success",
        "room_false_positive",
    ]
    for keys, group in frame.groupby(group_columns, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_columns, keys))
        row["n_queries"] = len(group)
        row["n_runs"] = group["run_id"].nunique() if "run_id" in group else 1
        for metric in metrics:
            values = [value for value in group[metric] if value is not None and not pd.isna(value)]
            mean, low, high = bootstrap_interval(
                values, confidence_level, bootstrap_samples, seed
            )
            row[metric] = mean
            row[f"{metric}_ci_low"] = low
            row[f"{metric}_ci_high"] = high
        row["retrieval_latency_ms"] = pd.to_numeric(
            group["retrieval_latency_ms"], errors="coerce"
        ).mean()
        row["navigation_time_s"] = pd.to_numeric(
            group["navigation_time_s"], errors="coerce"
        ).mean()
        rows.append(row)
    return pd.DataFrame(rows)


def aggregate_campaign_levels(
    cases: pd.DataFrame,
    confidence_level: float = 0.95,
    bootstrap_samples: int = 2000,
    seed: int = 42,
) -> dict[str, pd.DataFrame]:
    if cases.empty:
        empty = pd.DataFrame()
        return {"runs": empty, "campaigns": empty, "scenes": empty}
    run_summary = _summary(
        cases, ["scene_id", "campaign_id", "run_id", "method"],
        confidence_level, bootstrap_samples, seed,
    )
    campaign_summary = _summary(
        cases, ["scene_id", "campaign_id", "method"],
        confidence_level, bootstrap_samples, seed,
    )
    scene_summary = _summary(
        cases, ["scene_id", "method"], confidence_level, bootstrap_samples, seed
    )
    return {"runs": run_summary, "campaigns": campaign_summary, "scenes": scene_summary}


def export_campaign_analysis(
    output_root: str | Path,
    cases: pd.DataFrame,
    summaries: Mapping[str, pd.DataFrame],
    issues: Iterable[str],
) -> None:
    root = Path(output_root)
    targets = {
        "cases": root / "cases" / "all_cases.parquet",
        "runs": root / "runs" / "run_summary.parquet",
        "campaigns": root / "campaigns" / "campaign_summary.parquet",
        "scenes": root / "scenes" / "scene_summary.parquet",
        "failures": root / "failures" / "failure_cases.parquet",
    }
    for path in targets.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    cases.to_parquet(targets["cases"], index=False)
    summaries.get("runs", pd.DataFrame()).to_parquet(targets["runs"], index=False)
    summaries.get("campaigns", pd.DataFrame()).to_parquet(targets["campaigns"], index=False)
    summaries.get("scenes", pd.DataFrame()).to_parquet(targets["scenes"], index=False)
    cases.loc[cases["failure_type"] != "none"].to_parquet(targets["failures"], index=False)
    (root / "figures").mkdir(parents=True, exist_ok=True)
    (root / "manifest.json").write_text(
        json.dumps({"issues": list(issues), "n_cases": len(cases)}, indent=2),
        encoding="utf-8",
    )
